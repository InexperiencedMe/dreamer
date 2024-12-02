import torch
import torch.nn as nn
import torch.functional as F
from utils import *
from neuralNets import *
import os
import copy

class Dreamer:
    def __init__(self):
        self.worldModelBatchSize = 8
        self.actorCriticBatchSize = 32
        self.representationLength = 16
        self.representationClasses = 16
        self.representationSize = self.representationLength * self.representationClasses
        self.actionSize = 3
        self.recurrentStateSize = 512
        self.compressedObservationsSize = 512
        self.obsShape = (3, 96, 96)
        self.imaginationHorizon = 16
        self.betaPrior = 1
        self.betaPosterior = 0.1
        self.betaReconstruction = 20
        self.betaReward = 1
        self.betaKL = 1
        self.entropyScale = 0.0003
        self.tau = 0.05
        self.gamma = 0.997
        self.lambda_ = 0.95
        
        self.convEncoder     = ConvEncoder(self.obsShape, self.compressedObservationsSize).to(device)
        self.convDecoder     = ConvDecoder(self.representationSize + self.recurrentStateSize, self.obsShape).to(device)
        self.sequenceModel   = SequenceModel(self.representationSize, self.actionSize, self.recurrentStateSize).to(device)
        self.priorNet        = PriorNet(self.recurrentStateSize, self.representationLength, self.representationClasses).to(device)
        self.posteriorNet    = PosteriorNet(self.recurrentStateSize + self.compressedObservationsSize, self.representationLength, self.representationClasses).to(device)
        self.rewardPredictor = RewardPredictor(self.recurrentStateSize + self.representationSize).to(device)
        self.actor           = Actor(self.recurrentStateSize + self.representationSize, self.actionSize, actionHigh=[1, 1, 1], actionLow=[-1, 0, 0])
        self.critic          = Critic(self.recurrentStateSize + self.representationSize).to(device)
        self.targetCritic    = copy.deepcopy(self.critic)

        self.recurrentState  = self.sequenceModel.initializeRecurrentState()
        self.valueMoments    = Moments()
        self.totalUpdates    = 0
        self.freeNats        = 1

        self.worldModelOptimizer = optim.AdamW(
            list(self.convEncoder.parameters()) + list(self.convDecoder.parameters()) + list(self.sequenceModel.parameters()) +
            list(self.priorNet.parameters()) + list(self.posteriorNet.parameters()) + list(self.rewardPredictor.parameters()), 
            lr=3e-4)
        
        self.criticOptimizer = optim.AdamW(self.critic.parameters(), lr=3e-4)
        self.actorOptimizer = optim.AdamW(self.actor.parameters(), lr=3e-4)

    @torch.no_grad()
    def act(self, observation, reset=False):
        if reset:
            self.recurrentState = self.sequenceModel.initializeRecurrentState()

        encodedObservation = self.convEncoder(observation)
        latentState, _, = self.posteriorNet(torch.cat((self.recurrentState, encodedObservation), -1))
        fullState = torch.cat((self.recurrentState, latentState), -1)
        return self.actor(fullState, training=False)


    def trainWorldModel(self, observations, actions, rewards):
        # print(f"wc input obs: {observations.shape}")
        # print(f"wc input actions: {actions.shape}")
        # print(f"wc input rewards: {rewards.shape}")
        sequenceLength = actions.shape[1]

        encodedObservations = self.convEncoder(observations.view(self.worldModelBatchSize*(sequenceLength + 1), *self.obsShape))
        encodedObservations = encodedObservations.view(self.worldModelBatchSize, sequenceLength + 1, -1)
        initialRecurrentState = self.sequenceModel.initializeRecurrentState(self.worldModelBatchSize)
        # print(f"wc init encodedObs: {encodedObservations.shape}")
        # print(f"wc init recurrentState: {initialRecurrentState.shape}")

        posteriorNetOutputs = []
        recurrentStates = [initialRecurrentState]
        priorNetLogits = []
        posteriorNetLogits = []

        for timestep in range(sequenceLength):
            posteriorNetOutput, posteriorNetCurrentLogits = self.posteriorNet(torch.cat((recurrentStates[timestep], encodedObservations[:, timestep]), -1))
            posteriorNetOutputs.append(posteriorNetOutput)
            posteriorNetLogits.append(posteriorNetCurrentLogits)

            # if timestep == 0:
                # print(f"will be passing to seqeunce model posterior {posteriorNetOutputs[timestep].shape}, actions {actions[:, timestep].shape} and recurrent {recurrentStates[timestep].shape}")
    
            recurrentState = self.sequenceModel(posteriorNetOutputs[timestep].detach(), actions[:, timestep], recurrentStates[timestep])
            recurrentStates.append(recurrentState)

            _, priorNetCurrentLogits = self.priorNet(recurrentStates[timestep + 1])
            priorNetLogits.append(priorNetCurrentLogits)

        recurrentStates = torch.stack(recurrentStates, dim=1)                       # [batchSize, sequenceLength + 1, recurrentStateSize]
        posteriorNetOutputs = torch.stack(posteriorNetOutputs, dim=1)               # [batchSize, sequenceLength    , representationSize]
        posteriorNetLogits = torch.stack(posteriorNetLogits, dim=1)                 # [batchSize, sequenceLength    , representationLength, representationClasses]
        priorNetLogits = torch.stack(priorNetLogits, dim=1)                         # [batchSize, sequenceLength    , representationLength, representationClasses]
        fullStates = torch.cat((recurrentStates[:, 1:], posteriorNetOutputs), -1)   # [batchSize, sequenceLength    , recurrentSize + representationSize]
        # print(f"wm recurrentStates: {recurrentStates.shape}")
        # print(f"wm posteriorNetLogits: {posteriorNetLogits.shape}")
        # print(f"wm posteriorNetOutputs: {posteriorNetOutputs.shape}")
        # print(f"wm priorNetLogits: {priorNetLogits.shape}")
        # print(f"wm fullStates: {fullStates.shape}")

        reconstructedObservations = self.convDecoder(fullStates.view(self.worldModelBatchSize*sequenceLength, -1)) # [batchSize, sequenceLength, *obsShape]
        reconstructedObservations = reconstructedObservations.view(self.worldModelBatchSize, sequenceLength, *self.obsShape)
        predictedRewards = self.rewardPredictor(fullStates) # [batchSize, sequenceLength] To match the rewards replay
        # print(f"wm reconstructedObservations: {reconstructedObservations.shape}")
        # print(f"wm predictedRewards: {predictedRewards.shape}")


        reconstructionLoss = F.mse_loss(reconstructedObservations, observations[:, 1:], reduction="none").mean(dim=[-3, -2, -1]).mean()
        rewardPredictorLoss = F.mse_loss(predictedRewards, symlog(rewards))

        priorDistribution       = torch.distributions.Categorical(logits=priorNetLogits)
        priorDistributionSG     = torch.distributions.Categorical(logits=priorNetLogits.detach())
        posteriorDistribution   = torch.distributions.Categorical(logits=posteriorNetLogits)
        posteriorDistributionSG = torch.distributions.Categorical(logits=posteriorNetLogits.detach())

        priorLoss = torch.distributions.kl_divergence(posteriorDistributionSG, priorDistribution)
        posteriorLoss = torch.distributions.kl_divergence(posteriorDistribution, priorDistributionSG)
        freeNats = torch.full_like(priorLoss, self.freeNats)
        klLoss = (self.betaPrior*torch.maximum(priorLoss, freeNats) + self.betaPosterior*torch.maximum(posteriorLoss, freeNats)).mean()

        worldModelLoss =  self.betaReconstruction*reconstructionLoss + self.betaReward*rewardPredictorLoss + self.betaKL*klLoss

        self.worldModelOptimizer.zero_grad()
        worldModelLoss.backward()
        self.worldModelOptimizer.step()

        sampledFullStates = fullStates.view(self.worldModelBatchSize*sequenceLength, -1)[torch.randperm(self.worldModelBatchSize*sequenceLength)[:self.actorCriticBatchSize]]

        return sampledFullStates.detach(), worldModelLoss.item(), reconstructionLoss.item(), rewardPredictorLoss.item(), klLoss.item()
    
    def trainActorCritic(self, initialFullState):
        with torch.no_grad():
            fullState = initialFullState.detach()
            recurrentState, latentState = torch.split(fullState, [self.recurrentStateSize, self.representationSize], -1)
            action = self.actor(fullState, training=False)
            # print(f"ac init fullState: {fullState.shape}")
            # print(f"ac init recurrentState, latentState: {recurrentState.shape}, {latentState.shape}")
            # print(f"ac init action: {action.shape}")
        
        fullStates, actionLogProbabilities, entropies = [], [], []
        for _ in range(self.imaginationHorizon):
            nextRecurrentState = self.sequenceModel(latentState, action, recurrentState)
            nextLatentState, _ = self.priorNet(nextRecurrentState)
            nextFullState = torch.cat((nextRecurrentState, nextLatentState), -1)
            action, logProbabilities, entropy = self.actor(nextFullState)
            # print(f"ac nextRecurrentState: {nextRecurrentState.shape}")
            # print(f"ac nextLatentState: {nextLatentState.shape}")
            # print(f"ac nextFullState: {nextFullState.shape}")

            fullStates.append(nextFullState)
            actionLogProbabilities.append(logProbabilities)
            entropies.append(entropy)

            recurrentState = nextRecurrentState
            latentState    = nextLatentState
            fullState      = nextFullState

        fullStates = torch.stack(fullStates, dim=1)
        actionLogProbabilities = torch.stack(actionLogProbabilities[:-1], dim=1)
        entropies = torch.stack(entropies[:-1], dim=1)
        predictedRewards = self.rewardPredictor(fullStates[:, :-1], useSymexp=True)
        # print(f"ac stacked fullStates: {fullStates.shape}")
        # print(f"ac stacked actionLogProbabilities: {actionLogProbabilities.shape}")
        # print(f"ac stacked entropies: {entropies.shape}")
        # print(f"ac stacked predictedRewards: {predictedRewards.shape}")

        valueEstimates = self.targetCritic(fullStates)
        # print(f"ac valueEstimates: {valueEstimates.shape}")
        with torch.no_grad():
            lambdaValues = self.lambdaValues(predictedRewards, valueEstimates, gamma=self.gamma, lambda_=self.lambda_)
            # print(f"ac lambdaValues: {lambdaValues.shape}")
            _, inverseScale = self.valueMoments(lambdaValues) # Very slow. Might as well divide by EMA of range, no quantiles
            advantages = (lambdaValues - valueEstimates[:, :-1])/inverseScale

        # Actor Update
        # NOTE: Actor has to use .sample() when using advantages, .rsample() when using -lambdaValues
        actorLoss = -torch.mean(advantages.detach()*actionLogProbabilities + self.entropyScale*entropies) # TODO: Try weighted average - we trust early lambda values more
        # actorLoss = -torch.mean(lambdaValues) # DreamerV1 style loss
        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100)
        self.actorOptimizer.step()

        # Critic Update
        valuesForCriticUpdate = self.critic(fullStates[:, :-1].detach())
        criticLoss = F.mse_loss(valuesForCriticUpdate, lambdaValues.detach())
        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100)
        self.criticOptimizer.step()

        for param, targetParam in zip(self.critic.parameters(), self.targetCritic.parameters()):
            targetParam.data.copy_(self.tau * param.data + (1 - self.tau) * targetParam.data)

        return criticLoss.detach().cpu().item(), actorLoss.detach().cpu().item(), valueEstimates.detach().mean().cpu().item()

    def lambdaValues(self, rewards, values, gamma=0.997, lambda_=0.95):
        returns = torch.zeros_like(rewards)
        bootstrap = values[:, -1]
        for i in reversed(range(rewards.shape[-1])):
            returns[:, i] = rewards[:, i] + gamma * ((1 - lambda_)*values[:, i] + lambda_*bootstrap)
            bootstrap = returns[:, i]
        return returns

    def reconstructObservations(self, observations, actions):
        encodedObservations = self.convEncoder(observations)
        initialRecurrentState = self.sequenceModel.initializeRecurrentState().to(device)
        sequenceLength = len(actions)

        posteriorNetOutputs = []
        recurrentStates = [initialRecurrentState]

        for timestep in range(sequenceLength):
            posteriorNetOutput, _ = self.posteriorNet(torch.cat((recurrentStates[timestep], encodedObservations[timestep].unsqueeze(0)), -1))
            posteriorNetOutputs.append(posteriorNetOutput)

            recurrentState = self.sequenceModel(posteriorNetOutputs[timestep].detach(), torch.atleast_2d(actions[timestep]), recurrentStates[timestep])
            recurrentStates.append(recurrentState)

        posteriorNetOutputs = torch.stack(posteriorNetOutputs)
        recurrentStates = torch.stack(recurrentStates)
        fullStates = torch.cat((recurrentStates[1:], posteriorNetOutputs), -1)
        reconstructedObservations = self.convDecoder(fullStates)

        return reconstructedObservations.detach()
    
    @torch.no_grad()
    def rolloutInitialize(self, initialObservation):
        encodedObservation = self.convEncoder(initialObservation)
        initialRecurrentState = self.sequenceModel.initializeRecurrentState()
        posteriorNetOutput, _ = self.posteriorNet(torch.cat((initialRecurrentState, encodedObservation), -1))
        return initialRecurrentState, posteriorNetOutput

    @torch.no_grad()
    def rolloutStep(self, recurrentState, latentState, action):
        newRecurrentState = self.sequenceModel(latentState, torch.atleast_2d(action), recurrentState)
        newLatentState, _ = self.priorNet(newRecurrentState)

        fullState = torch.cat((newRecurrentState, newLatentState), -1)
        reconstructedObservation = self.convDecoder(torch.atleast_2d(fullState))
        predictedReward = symexp(self.rewardPredictor(fullState))
        return newRecurrentState, newLatentState, reconstructedObservation, predictedReward

    def saveCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'

        checkpoint = {
            'convEncoder'           : self.convEncoder.state_dict(),
            'convDecoder'           : self.convDecoder.state_dict(),
            'sequenceModel'         : self.sequenceModel.state_dict(),
            'priorNet'              : self.priorNet.state_dict(),
            'posteriorNet'          : self.posteriorNet.state_dict(),
            'rewardPredictor'       : self.rewardPredictor.state_dict(),
            'actor'                 : self.actor.state_dict(),
            'critic'                : self.critic.state_dict(),
            'targetCritic'          : self.critic.state_dict(),
            'worldModelOptimizer'   : self.worldModelOptimizer.state_dict(),
            'criticOptimizer'       : self.criticOptimizer.state_dict(),
            'actorOptimizer'        : self.actorOptimizer.state_dict(),
            'valueMoments'          : self.valueMoments.state_dict(),
            'recurrentState'        : self.recurrentState,
            'totalUpdates'          : self.totalUpdates
        }
        torch.save(checkpoint, checkpointPath)

    def loadCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'
        if not os.path.exists(checkpointPath):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpointPath}")
        
        checkpoint = torch.load(checkpointPath)
        self.convEncoder.load_state_dict(         checkpoint['convEncoder'])
        self.convDecoder.load_state_dict(         checkpoint['convDecoder'])
        self.sequenceModel.load_state_dict(       checkpoint['sequenceModel'])
        self.priorNet.load_state_dict(            checkpoint['priorNet'])
        self.posteriorNet.load_state_dict(        checkpoint['posteriorNet'])
        self.rewardPredictor.load_state_dict(     checkpoint['rewardPredictor'])
        self.actor.load_state_dict(               checkpoint['actor'])
        self.critic.load_state_dict(              checkpoint['critic'])
        self.targetCritic.load_state_dict(        checkpoint['targetCritic'])
        self.worldModelOptimizer.load_state_dict( checkpoint['worldModelOptimizer'])
        self.criticOptimizer.load_state_dict(     checkpoint['criticOptimizer'])
        self.actorOptimizer.load_state_dict(      checkpoint['actorOptimizer'])
        self.valueMoments.load_state_dict(        checkpoint['valueMoments'])
        self.recurrentState =                     checkpoint['recurrentState']
        self.totalUpdates =                       checkpoint['totalUpdates']
        # print(f"Loaded checkpoint from: {checkpointPath}")
