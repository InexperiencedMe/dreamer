import torch
import torch.nn as nn
import torch.functional as F
from utils import *
from neuralNets import *
import os
import copy

class Dreamer:
    def __init__(self):
        self.worldModelBatchSize        = 4
        self.actorCriticBatchSize       = 16
        self.representationLength       = 16
        self.representationClasses      = 16
        self.representationSize         = self.representationLength*self.representationClasses
        self.actionSize                 = 3             # This should be taken at initialization from gym
        self.recurrentStateSize         = 512
        self.compressedObservationSize  = 512
        self.obsShape                   = (3, 96, 96)   # This should be taken at initialization from gym
        self.imaginationHorizon         = 16
        self.betaPrior                  = 1
        self.betaPosterior              = 0.1
        self.betaReconstruction         = 20            # The loss seemed so low, that I boosted its importance. Maybe I shouldnt?
        self.betaReward                 = 1
        self.betaKL                     = 1
        self.entropyScale               = 0.0003
        self.tau                        = 0.05
        self.gamma                      = 0.997
        self.lambda_                    = 0.95
        
        self.convEncoder     = ConvEncoder(self.obsShape, self.compressedObservationSize).to(device)
        self.convDecoder     = ConvDecoder(self.representationSize + self.recurrentStateSize, self.obsShape).to(device)
        self.sequenceModel   = SequenceModel(self.representationSize, self.actionSize, self.recurrentStateSize).to(device)
        self.priorNet        = PriorNet(self.recurrentStateSize, self.representationLength, self.representationClasses).to(device)
        self.posteriorNet    = PosteriorNet(self.recurrentStateSize + self.compressedObservationSize, self.representationLength, self.representationClasses).to(device)
        self.rewardPredictor = RewardPredictor(self.recurrentStateSize + self.representationSize).to(device)
        self.actor           = Actor(self.recurrentStateSize + self.representationSize, self.actionSize, actionHigh=[1, 1, 1], actionLow=[-1, 0, 0])
        self.critic          = Critic(self.recurrentStateSize + self.representationSize).to(device)
        self.targetCritic    = copy.deepcopy(self.critic)

        self.recurrentState  = self.sequenceModel.initializeRecurrentState()
        self.valueMoments    = Moments()
        self.totalUpdates    = 0
        self.freeNats        = 1
        self.clipGradients   = False # Potentially useful, but slow and I prefer speeeeed

        self.worldModelParams    = (list(self.convEncoder.parameters()) + list(self.convDecoder.parameters()) + list(self.sequenceModel.parameters()) +
                                    list(self.priorNet.parameters()) + list(self.posteriorNet.parameters()) + list(self.rewardPredictor.parameters()))
        self.worldModelOptimizer = optim.AdamW(self.worldModelParams, lr=3e-4)
        
        self.criticOptimizer = optim.AdamW(self.critic.parameters(), lr=3e-4)
        self.actorOptimizer  = optim.AdamW(self.actor.parameters(), lr=3e-4)


    @torch.no_grad()
    def act(self, observation, reset=False):
        if reset:
            self.recurrentState = self.sequenceModel.initializeRecurrentState()

        encodedObservation = self.convEncoder(observation)
        latentState, _, = self.posteriorNet(torch.cat((self.recurrentState, encodedObservation), -1))
        fullState = torch.cat((self.recurrentState, latentState), -1)
        return self.actor(fullState, training=False)


    def trainWorldModel(self, observations, actions, rewards):  # actions synced with obs, rewards for all obs except first
        sequenceLength = observations.shape[1]                  # equal to stepCountLimit from main

        encodedObservations = self.convEncoder(observations.view(self.worldModelBatchSize*sequenceLength, *self.obsShape))
        encodedObservations = encodedObservations.view(self.worldModelBatchSize, sequenceLength, -1) # [batchSize, sequenceLength, compressedObservationSize]

        posteriors, recurrentStates, priorLogits, posteriorLogits = [], [], [], []
        for timestep in range(sequenceLength - 1):
            if timestep == 0:
                with torch.no_grad(): # This context seems to improve it, but not completely fix it
                    recurrentState = self.sequenceModel.initializeRecurrentState(self.worldModelBatchSize)              # initialize the "past"
                    posterior, _ = self.posteriorNet(torch.cat((recurrentState, encodedObservations[:, timestep]), 1))  # calculate state representation based on no past and current obs
                    action = actions[:, timestep]                                                                       # action we took during the initial state
                    recurrentStates.append(recurrentState) # appending to sync index with timestep, even though not used later
                    posteriors.append(posterior)           # appending to sync index with timestep, even though not used later
            else:
                recurrentState = recurrentStates[timestep]
                posterior = posteriors[timestep]
                action = actions[:, timestep]
            
            nextRecurrentState = self.sequenceModel(posterior, action, recurrentState) # recurrent state for timestep + 1
            _, nextPriorLogits = self.priorNet(nextRecurrentState)
            nextPosterior, nextPosteriorLogits = self.posteriorNet(torch.cat((nextRecurrentState, encodedObservations[:, timestep + 1]), -1))

            recurrentStates.append(nextRecurrentState)
            priorLogits.append(nextPriorLogits)
            posteriors.append(nextPosterior)
            posteriorLogits.append(nextPosteriorLogits)

                       
        recurrentStates = torch.stack(recurrentStates[1:], dim=1)       # [batchSize, sequenceLength-1, recurrentStateSize] # resyncing so now tensors are synced
        posteriors      = torch.stack(posteriors[1:], dim=1)            # [batchSize, sequenceLength-1, representationSize] # resyncing so now tensors are synced
        posteriorLogits = torch.stack(posteriorLogits, dim=1)           # [batchSize, sequenceLength-1, representationLength, representationClasses]
        priorLogits     = torch.stack(priorLogits, dim=1)               # [batchSize, sequenceLength-1, representationLength, representationClasses]
        fullStates      = torch.cat((recurrentStates, posteriors), -1)  # [batchSize, sequenceLength-1, recurrentSize + representationSize]

        reconstructedObservations = self.convDecoder(fullStates.view(self.worldModelBatchSize*(sequenceLength - 1), -1))
        reconstructedObservations = reconstructedObservations.view(self.worldModelBatchSize, sequenceLength - 1, *self.obsShape)
        predictedRewards = self.rewardPredictor(fullStates) # [batchSize, sequenceLength-1]

        reconstructionLoss = F.mse_loss(reconstructedObservations, observations[:, 1:], reduction="none").mean(dim=[-3, -2, -1]).mean()
        rewardPredictorLoss = F.mse_loss(predictedRewards, symlog(rewards))

        priorDistribution       = torch.distributions.Categorical(logits=priorLogits)
        priorDistributionSG     = torch.distributions.Categorical(logits=priorLogits.detach())
        posteriorDistribution   = torch.distributions.Categorical(logits=posteriorLogits)
        posteriorDistributionSG = torch.distributions.Categorical(logits=posteriorLogits.detach())

        priorLoss       = torch.distributions.kl_divergence(posteriorDistributionSG, priorDistribution)
        posteriorLoss   = torch.distributions.kl_divergence(posteriorDistribution  , priorDistributionSG)
        freeNats        = torch.full_like(priorLoss, self.freeNats)
        klLoss          = (self.betaPrior*torch.maximum(priorLoss, freeNats) + self.betaPosterior*torch.maximum(posteriorLoss, freeNats)).mean()

        worldModelLoss  =  self.betaReconstruction*reconstructionLoss + self.betaReward*rewardPredictorLoss + self.betaKL*klLoss

        self.worldModelOptimizer.zero_grad()
        worldModelLoss.backward()
        if self.clipGradients:
            torch.nn.utils.clip_grad_norm_(self.worldModelParams, self.gradientClipValue)
        self.worldModelOptimizer.step()

        sampledFullStates = fullStates.view(self.worldModelBatchSize*(sequenceLength - 1), -1)[torch.randperm(self.worldModelBatchSize*(sequenceLength - 1))[:self.actorCriticBatchSize]]
        return sampledFullStates.detach(), worldModelLoss.item(), reconstructionLoss.item(), rewardPredictorLoss.item(), klLoss.item()
    

    def trainActorCritic(self, initialFullState):
        with torch.no_grad():
            fullState = initialFullState.detach()                       # [actorCriticBatchSize, recurrentSize + representationSize]
            action = self.actor(fullState, training=False)              # [actorCriticBatchSize, actionSize]
            recurrentState, latentState = torch.split(fullState, [self.recurrentStateSize, self.representationSize], -1)
        
        fullStates, actionLogProbabilities, entropies = [], [], []
        for _ in range(self.imaginationHorizon):
            recurrentState = self.sequenceModel(latentState, action, recurrentState)
            latentState, _ = self.priorNet(recurrentState)
            fullState = torch.cat((recurrentState, latentState), -1)
            action, logProbabilities, entropy = self.actor(fullState.detach())

            fullStates.append(fullState)
            actionLogProbabilities.append(logProbabilities)
            entropies.append(entropy)

        fullStates              = torch.stack(fullStates, dim=1)                            # [batchSize, horizon, recurrentSize + representationSize]
        actionLogProbabilities  = torch.stack(actionLogProbabilities[:-1], dim=1)           # [batchSize, horizon-1]
        entropies               = torch.stack(entropies[:-1], dim=1)                        # [batchSize, horizon-1]

        with torch.no_grad():
            predictedRewards        = self.rewardPredictor(fullStates[:, :-1], useSymexp=True)                          # [batchSize, horizon-1]
            valueEstimates = self.targetCritic(fullStates)                                                              # [batchSize, horizon]
            lambdaValues = self.lambdaValues(predictedRewards, valueEstimates, gamma=self.gamma, lambda_=self.lambda_)  # [batchSize, horizon-1]
            _, inverseScale = self.valueMoments(lambdaValues)
            advantages = (lambdaValues - valueEstimates[:, :-1])/inverseScale

        # Actor Update
        # NOTE: Actor has to use .sample() when using advantages, .rsample() when using -lambdaValues
        actorLoss = -torch.mean(advantages.detach()*actionLogProbabilities + self.entropyScale*entropies)
        # actorLoss = -torch.mean(lambdaValues) # DreamerV1 style loss

        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        if self.clipGradients:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradientClipValue)
        self.actorOptimizer.step()

        # Critic Update
        valuesForCriticUpdate = self.critic(fullStates[:, :-1].detach())
        criticLoss = F.mse_loss(valuesForCriticUpdate, lambdaValues.detach())

        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        if self.clipGradients:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradientClipValue)
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

        posteriors = []
        recurrentStates = [initialRecurrentState]

        for timestep in range(sequenceLength):
            posterior, _ = self.posteriorNet(torch.cat((recurrentStates[timestep], encodedObservations[timestep].unsqueeze(0)), -1))
            posteriors.append(posterior)

            recurrentState = self.sequenceModel(posteriors[timestep].detach(), torch.atleast_2d(actions[timestep]), recurrentStates[timestep])
            recurrentStates.append(recurrentState)

        posteriors = torch.stack(posteriors)
        recurrentStates = torch.stack(recurrentStates)
        fullStates = torch.cat((recurrentStates[1:], posteriors), -1)
        reconstructedObservations = self.convDecoder(fullStates)

        return reconstructedObservations.detach()


    @torch.no_grad()
    def rolloutInitialize(self, initialObservation):
        encodedObservation = self.convEncoder(initialObservation)
        initialRecurrentState = self.sequenceModel.initializeRecurrentState()
        posterior, _ = self.posteriorNet(torch.cat((initialRecurrentState, encodedObservation), -1))
        return initialRecurrentState, posterior


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
