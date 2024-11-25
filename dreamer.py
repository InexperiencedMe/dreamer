import torch
import torch.nn as nn
import torch.functional as F
from utils import *
from neuralNets import *
import os
import copy

class Dreamer:
    def __init__(self):
        self.representationLength = 16
        self.representationClasses = 16
        self.representationSize = self.representationLength * self.representationClasses
        self.actionSize = 3
        self.recurrentStateSize = 512
        self.compressedObservationsSize = 512
        self.obsShape = (3, 96, 96)
        self.imaginationHorizon = 16
        self.betaPrior = 10
        self.betaPosterior = 1
        self.betaReconstruction = 20
        self.betaReward = 1
        self.betaKL = 1
        self.entropyScale = 0.001
        self.tau = 0.05
        self.gamma = 0.997
        self.lambda_ = 0.95
        
        self.convEncoder     = ConvEncoder(self.obsShape, self.compressedObservationsSize).to(device)
        self.convDecoder     = ConvDecoder(self.representationSize + self.recurrentStateSize, self.obsShape).to(device)
        self.sequenceModel   = SequenceModel(self.representationSize, self.actionSize, self.recurrentStateSize).to(device)
        self.priorNet        = PriorNet(self.recurrentStateSize, self.representationLength, self.representationClasses).to(device)
        self.posteriorNet    = PosteriorNet(self.recurrentStateSize + self.compressedObservationsSize, self.representationLength, self.representationClasses).to(device)
        self.rewardPredictor = RewardPredictor(self.recurrentStateSize + self.representationSize).to(device)
        self.actor           = Actor(self.recurrentStateSize + self.representationSize, self.actionSize)
        # self.actor           = ActorCleanRLStyle(self.recurrentStateSize + self.representationSize, self.actionSize, actionHigh=[1, 1, 1], actionLow=[-1, 0, 0])
        self.critic          = Critic(self.recurrentStateSize + self.representationSize).to(device)
        self.targetCritic    = copy.deepcopy(self.critic)

        self.recurrentState  = self.sequenceModel.initializeRecurrentState()
        self.valueMoments    = Moments()
        self.totalUpdates    = 0

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

        encodedObservation = self.convEncoder(observation).view(-1)
        latentState, _, = self.posteriorNet(torch.cat((self.recurrentState, encodedObservation), -1))
        fullState = torch.cat((self.recurrentState, latentState), -1)
        return self.actor(fullState, training=False)


    def trainWorldModel(self, observations, actions, rewards):
        encodedObservations = self.convEncoder(observations)
        # TODO: Exchange this local version to the internal recurrent state assigned to the object
        initialRecurrentState = self.sequenceModel.initializeRecurrentState()
        episodeLength = len(actions)

        posteriorNetOutputs = []
        recurrentStates = [initialRecurrentState]
        priorNetLogits = []
        posteriorNetLogits = []

        for timestep in range(episodeLength):
            posteriorNetOutput, posteriorNetCurrentLogits = self.posteriorNet(torch.cat((recurrentStates[timestep], encodedObservations[timestep]), -1))
            posteriorNetOutputs.append(posteriorNetOutput)
            posteriorNetLogits.append(posteriorNetCurrentLogits)

            recurrentState = self.sequenceModel(posteriorNetOutputs[timestep].detach(), actions[timestep], recurrentStates[timestep])
            recurrentStates.append(recurrentState)

            _, priorNetCurrentLogits = self.priorNet(recurrentStates[timestep])
            priorNetLogits.append(priorNetCurrentLogits)

        recurrentStates = torch.stack(recurrentStates)              # [episodeLength + 1, recurrentStateSize]
        posteriorNetOutputs = torch.stack(posteriorNetOutputs)      # [episodeLength    , representationSize]
        posteriorNetLogits = torch.stack(posteriorNetLogits)        # [episodeLength    , representationSize]
        priorNetLogits = torch.stack(priorNetLogits)                # [episodeLength    , representationSize]
        fullStates = torch.cat((recurrentStates[1:], posteriorNetOutputs), -1)

        reconstructedObservations = self.convDecoder(fullStates)
        predictedRewards = self.rewardPredictor(fullStates)

        reconstructionLoss = F.mse_loss(reconstructedObservations, observations[1:], reduction="none").mean(dim=[-3, -2, -1]).mean()
        rewardPredictorLoss = F.mse_loss(predictedRewards, symlog(rewards))

        priorDistribution       = torch.distributions.Categorical(logits=priorNetLogits)
        posteriorDistribution   = torch.distributions.Categorical(logits=posteriorNetLogits)
        priorDistributionSG     = torch.distributions.Categorical(logits=priorNetLogits.detach())
        posteriorDistributionSG = torch.distributions.Categorical(logits=posteriorNetLogits.detach())

        priorLoss = torch.distributions.kl_divergence(posteriorDistributionSG, priorDistribution).mean()
        posteriorLoss = torch.distributions.kl_divergence(posteriorDistribution, priorDistributionSG).mean()
        klLoss = self.betaPrior*priorLoss + self.betaPosterior*posteriorLoss # We add it because they have the same value, no need to distinguish it

        worldModelLoss =  self.betaReconstruction*reconstructionLoss + self.betaReward*rewardPredictorLoss + self.betaKL*klLoss

        self.worldModelOptimizer.zero_grad()
        worldModelLoss.backward()
        self.worldModelOptimizer.step()

        sampledFullState = fullStates[torch.randperm(fullStates.shape[0])[:1]].detach() # One initial state for imagination rollout

        return sampledFullState, worldModelLoss.item(), reconstructionLoss.item(), rewardPredictorLoss.item(), klLoss.item()
    
    def trainActorCritic(self, initialFullState):
        with torch.no_grad():
            fullState = initialFullState.detach()
            recurrentState, latentState = torch.split(fullState, [self.recurrentStateSize, self.representationSize], -1)
        
        fullStates, actionLogProbabilities, entropies = [], [], []
        for _ in range(self.imaginationHorizon):
            action, logProbabilities, entropy = self.actor(fullState)
            nextRecurrentState = self.sequenceModel(latentState, action, recurrentState)
            nextLatentState, _ = self.priorNet(recurrentState)
            nextFullState = torch.cat((nextRecurrentState, nextLatentState), -1)

            fullStates.append(nextFullState)
            actionLogProbabilities.append(logProbabilities)
            entropies.append(entropy)

            recurrentState          = nextRecurrentState
            latentState             = nextLatentState
            fullState               = nextFullState

        fullStates = torch.stack(fullStates)
        actionLogProbabilities = torch.stack(actionLogProbabilities)
        entropies = torch.stack(entropies)
        predictedRewards = self.rewardPredictor(fullStates[:-1], useSymexp=True).detach()

        valueEstimates = self.targetCritic(fullStates)
        with torch.no_grad():
            lambdaValues = self.lambdaValues(predictedRewards, valueEstimates, gamma=self.gamma, lambda_=self.lambda_)

            offset, inverseScale = self.valueMoments(lambdaValues)
            normalizedLambdaValues = (lambdaValues - offset)/inverseScale
            normalizedValueEstimates = (valueEstimates - offset)/inverseScale
            advantages = normalizedLambdaValues - normalizedValueEstimates

        # Actor Update
        actorLoss = -torch.mean(advantages.detach() * actionLogProbabilities + self.entropyScale * entropies)
        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100)
        self.actorOptimizer.step()

        # Critic Update
        valuesForCriticUpdate = self.critic(fullStates.detach())
        criticLoss = F.mse_loss(valuesForCriticUpdate, lambdaValues.detach())
        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100)
        self.criticOptimizer.step()

        for param, targetParam in zip(self.critic.parameters(), self.targetCritic.parameters()):
            targetParam.data.copy_(self.tau * param.data + (1 - self.tau) * targetParam.data)

        return criticLoss.detach().cpu().item(), actorLoss.detach().cpu().item(), valueEstimates.detach().mean().cpu().item()

    def lambdaValues(self, rewards, values, gamma=0.997, lambda_=0.95):
        returns = torch.zeros_like(values)
        bootstrap = values[-1]
        returns[-1] = bootstrap
        for i in reversed(range(len(rewards))):
            returns[i] = rewards[i] + gamma * ((1 - lambda_)*values[i] + lambda_*bootstrap)
            bootstrap = returns[i]
        return returns

    def reconstructObservations(self, observations, actions):
        encodedObservations = self.convEncoder(observations)
        initialRecurrentState = self.sequenceModel.initializeRecurrentState().to(device)
        episodeLength = len(actions)

        posteriorNetOutputs = []
        recurrentStates = [initialRecurrentState]

        for timestep in range(episodeLength):
            posteriorNetOutput, _ = self.posteriorNet(torch.cat((recurrentStates[timestep], encodedObservations[timestep]), -1))
            posteriorNetOutputs.append(posteriorNetOutput)

            recurrentState = self.sequenceModel(posteriorNetOutputs[timestep].detach(), actions[timestep], recurrentStates[timestep])
            recurrentStates.append(recurrentState)

        posteriorNetOutputs = torch.stack(posteriorNetOutputs)      # [episodeLength    , representationSize]
        recurrentStates = torch.stack(recurrentStates)              # [episodeLength + 1, recurrentStateSize]
        fullStates = torch.cat((recurrentStates[1:], posteriorNetOutputs), -1)
        reconstructedObservations = self.convDecoder(fullStates)

        return reconstructedObservations.detach()
    
    @torch.no_grad()
    def rolloutInitialize(self, initialObservation):
        encodedObservation = self.convEncoder(initialObservation)
        initialRecurrentState = self.sequenceModel.initializeRecurrentState()
        posteriorNetOutput, _ = self.posteriorNet(torch.cat((initialRecurrentState, encodedObservation.view(-1)), -1))
        return initialRecurrentState, posteriorNetOutput

    @torch.no_grad()
    def rolloutStep(self, recurrentState, latentState, action):
        newRecurrentState = self.sequenceModel(latentState, action, recurrentState)
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
            'worldModelOptimizer'   : self.worldModelOptimizer.state_dict(),
            'criticOptimizer'       : self.criticOptimizer.state_dict(),
            'actorOptimizer'        : self.actorOptimizer.state_dict(),
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
        self.worldModelOptimizer.load_state_dict( checkpoint['worldModelOptimizer'])
        self.criticOptimizer.load_state_dict(     checkpoint['criticOptimizer'])
        self.actorOptimizer.load_state_dict(      checkpoint['actorOptimizer'])
        self.recurrentState =                     checkpoint['recurrentState']
        self.totalUpdates =                       checkpoint['totalUpdates']
        print(f"Loaded checkpoint from: {checkpointPath}")
