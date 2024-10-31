import torch
import torch.nn as nn
import torch.functional as F
from utils import *
from neuralNets import *

class Dreamer:
    def __init__(self):
        self.representationClasses = 32
        self.representationSize = self.representationClasses ** 2
        self.actionSize = 3
        self.recurrentStateSize = 256
        self.compressedObservationsSize = 256
        self.obsShape = (3, 96, 96)

        self.convEncoder     = ConvEncoder(self.obsShape, self.compressedObservationsSize).to(device)
        self.convDecoder     = ConvDecoder(self.representationSize + self.recurrentStateSize, self.obsShape).to(device)
        self.sequenceModel   = SequenceModel(self.representationSize, self.actionSize, self.recurrentStateSize).to(device)
        self.priorNet        = PriorNet(self.recurrentStateSize, self.representationClasses).to(device)
        self.posteriorNet    = PosteriorNet(self.recurrentStateSize + self.compressedObservationsSize, self.representationClasses).to(device)
        self.rewardPredictor = RewardPredictor(self.recurrentStateSize + self.representationSize).to(device)

        self.worldModelOptimizer = optim.AdamW(
            list(self.convEncoder.parameters()) + list(self.convDecoder.parameters()) + 
            list(self.sequenceModel.parameters()) + list(self.priorNet.parameters()) + 
            list(self.posteriorNet.parameters()) + list(self.rewardPredictor.parameters()), 
            lr=3e-4
)

    def train(self, observations, actions, rewards):
        encodedObservations = self.convEncoder(observations)
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

        posteriorNetOutputs = torch.stack(posteriorNetOutputs)      # [episodeLength    , representationSize]
        recurrentStates = torch.stack(recurrentStates)              # [episodeLength + 1, recurrentStateSize]
        priorNetLogits = torch.stack(priorNetLogits)                # [episodeLength    , representationSize]
        posteriorNetLogits = torch.stack(posteriorNetLogits)        # [episodeLength    , representationSize]
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
        klLoss = priorLoss + posteriorLoss # We add it because they have the same value, no need to distinguish it

        worldModelLoss = reconstructionLoss + rewardPredictorLoss + klLoss

        self.worldModelOptimizer.zero_grad()
        worldModelLoss.backward()
        self.worldModelOptimizer.step()

        return worldModelLoss.item(), reconstructionLoss.item(), rewardPredictorLoss.item(), klLoss.item()
    
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