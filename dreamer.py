import torch
import torch.nn as nn
import torch.functional as F
from utils import *
from neuralNets import *
import os
import csv

class Dreamer:
    def __init__(self):
        self.representationLength = 32
        self.representationClasses = 32
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
        # self.tau = 0.05

        self.convEncoder     = ConvEncoder(self.obsShape, self.compressedObservationsSize).to(device)
        self.convDecoder     = ConvDecoder(self.representationSize + self.recurrentStateSize, self.obsShape).to(device)
        self.sequenceModel   = SequenceModel(self.representationSize, self.actionSize, self.recurrentStateSize).to(device)
        self.priorNet        = PriorNet(self.recurrentStateSize, self.representationLength, self.representationClasses).to(device)
        self.posteriorNet    = PosteriorNet(self.recurrentStateSize + self.compressedObservationsSize, self.representationLength, self.representationClasses).to(device)
        self.rewardPredictor = RewardPredictor(self.recurrentStateSize + self.representationSize).to(device)
        self.actor           = Actor(self.recurrentStateSize + self.representationSize, self.actionSize)
        self.critic          = Critic(self.recurrentStateSize + self.representationSize)

        self.recurrentState = self.sequenceModel.initializeRecurrentState()
        self.totalUpdates = 0
        self.valueMoments = Moments()

        self.worldModelOptimizer = optim.AdamW(
            list(self.convEncoder.parameters()) + list(self.convDecoder.parameters()) + list(self.sequenceModel.parameters()) +
            list(self.priorNet.parameters()) + list(self.posteriorNet.parameters()) + list(self.rewardPredictor.parameters()), 
            lr=1e-3)
        
        self.criticOptimizer = optim.AdamW(self.critic.parameters(), lr=3e-4)
        self.actorOptimizer = optim.AdamW(self.actor.parameters(), lr=3e-4)

    @torch.no_grad()
    def act(self, observation, reset=False):
        if reset:
            self.recurrentState = self.sequenceModel.initializeRecurrentState()

        encodedObservation = self.convEncoder(observation).view(-1)
        latentState, _, = self.posteriorNet(torch.cat((self.recurrentState, encodedObservation), -1))
        fullStateRepresentation = torch.cat((self.recurrentState, latentState), -1)
        return self.actor(fullStateRepresentation, training=False)


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
        fullStateRepresentations = torch.cat((recurrentStates[1:], posteriorNetOutputs), -1)

        reconstructedObservations = self.convDecoder(fullStateRepresentations)
        predictedRewards = self.rewardPredictor(fullStateRepresentations)

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

        return worldModelLoss.item(), reconstructionLoss.item(), rewardPredictorLoss.item(), klLoss.item()
    
    def trainActorCritic(self, observations):
        currentObservation = observations[0].unsqueeze(0)
        encodedObservation = self.convEncoder(currentObservation)
        recurrentState = self.sequenceModel.initializeRecurrentState()
        latentState, _ = self.posteriorNet(torch.cat((recurrentState, encodedObservation.view(-1)), -1))
        fullStateRepresentation = torch.cat((recurrentState, latentState), -1)

        fullStateRepresentations = [fullStateRepresentation]
        actionLogProbabilities = []
        predictedRewards = []
        for _ in range(self.imaginationHorizon):
            action, logProbabilities, _ = self.actor(fullStateRepresentation)

            nextRecurrentState = self.sequenceModel(latentState, action, recurrentState)
            nextLatentState, _ = self.priorNet(recurrentState)

            nextFullStateRepresentation = torch.cat((nextRecurrentState, nextLatentState), -1)
            reward = self.rewardPredictor(nextFullStateRepresentation)

            fullStateRepresentations.append(nextFullStateRepresentation)
            actionLogProbabilities.append(logProbabilities)
            predictedRewards.append(reward)

            recurrentState          = nextRecurrentState
            latentState             = nextLatentState
            fullStateRepresentation = nextFullStateRepresentation

        fullStateRepresentations = torch.stack(fullStateRepresentations)
        actionLogProbabilities = torch.stack(actionLogProbabilities).sum(-1)
        predictedRewards = torch.stack(predictedRewards)

        # Critic Update
        valueEstimates = self.critic(fullStateRepresentations.detach())
        lambdaValues = self.lambdaValues(predictedRewards, valueEstimates)

        criticLoss = F.mse_loss(valueEstimates[:-1], lambdaValues.detach()) # 1 more value as bootstrap
        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100)
        self.criticOptimizer.step()

        valueEstimatesForActor = self.critic(fullStateRepresentations[:-1]) 
        offset, inverseScale = self.valueMoments(lambdaValues.detach()) 
        normalizedLambdaValues = (lambdaValues - offset) / inverseScale
        normalizedValueEstimates = (valueEstimatesForActor - offset) / inverseScale
        advantage = normalizedLambdaValues.detach() - normalizedValueEstimates
        discounts = torch.cumprod(torch.full((len(advantage),), 0.99, device=device), dim=0) / 0.99
        _, _, entropy = self.actor(fullStateRepresentations[:-1])
        actorLoss = -torch.mean(discounts * (advantage*actionLogProbabilities + self.entropyScale*entropy))
        # print(f"Actor loss is {actorLoss} because we -mean the discounted (advantages {advantage} plus entropy {self.entropyScale*entropy}")

        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100)
        self.actorOptimizer.step()

        return criticLoss.item(), actorLoss.item(), valueEstimates.detach().mean().cpu().item()

    def lambdaValues(self, rewards, values, gamma=0.99, lambda_=0.95):
        n = len(rewards)
        returns = torch.zeros(n, device=device)
        bootstrap = values[-1]
        for t in reversed(range(n)):
            returns[t] = rewards[t] + gamma*((1 - lambda_)*values[t + 1] + lambda_*bootstrap)
            bootstrap = returns[t]
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
        fullStateRepresentations = torch.cat((recurrentStates[1:], posteriorNetOutputs), -1)
        reconstructedObservations = self.convDecoder(fullStateRepresentations)

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

        fullStateRepresentation = torch.cat((newRecurrentState, newLatentState), -1)
        reconstructedObservation = self.convDecoder(torch.atleast_2d(fullStateRepresentation))
        predictedReward = symexp(self.rewardPredictor(fullStateRepresentation))
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
