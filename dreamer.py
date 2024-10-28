import torch
import torch.nn as nn
import torch.functional as F
from utils import *
from neuralNets import *

class Dreamer:
    def __init__(self):
        self.representationClasses = 16
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
            lr=1e-3
)

    def train(self, observations, actions, rewards):
        encodedObservations = self.convEncoder(observations)
        initialRecurrentState = self.sequenceModel.initializeRecurrentState().to(device)
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

        reconstructionLoss = F.mse_loss(reconstructedObservations, observations[1:], reduction="none").mean(dim=[-1, -2, -3]).mean()
        priorNetLoss = F.mse_loss(priorNetLogits, posteriorNetLogits.detach())
        rewardPredictorLoss = 0.05*F.mse_loss(predictedRewards, rewards)

        worldModelLoss = reconstructionLoss + priorNetLoss + rewardPredictorLoss

        self.worldModelOptimizer.zero_grad()
        worldModelLoss.backward()
        self.worldModelOptimizer.step()

        return worldModelLoss.item(), reconstructionLoss.item(), priorNetLoss.item(), rewardPredictorLoss.item()
    
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