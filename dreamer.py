import torch
import torch.nn as nn
import torch.functional as F
from utils import *
from neuralNets import *
from neuralNetsSD import *
import os
import copy

class Dreamer:
    def __init__(self):
        self.worldModelBatchSize        = 4
        self.actorCriticBatchSize       = 8
        self.representationLength       = 8
        self.representationClasses      = 8
        self.representationSize         = self.representationLength*self.representationClasses
        self.actionSize                 = 3             # This should be taken at initialization from gym
        self.recurrentStateSize         = 256          # 4096 in the final version, but decrease for faster development
        self.compressedObservationSize  = 256
        self.obsShape                   = (3, 32, 32)   # This should be taken at initialization from gym
        self.imaginationHorizon         = 15
        self.betaPrior                  = 1
        self.betaPosterior              = 0.1
        self.betaReconstruction         = 10
        self.betaReward                 = 1
        self.betaKL                     = 1
        self.entropyScale               = 0.0003
        self.tau                        = 0.02
        self.gamma                      = 0.997
        self.lambda_                    = 0.95
        self.worldModelLR               = 1e-4
        self.criticLR                   = 1e-4
        self.actorLR                    = 1e-4

        self.actionLow       = [-1, 0, 0]
        self.actionHigh      = [1, 1, 1]
                
        self.convEncoder     = ConvEncoderSD(self.obsShape, self.compressedObservationSize).to(device)
        self.convDecoder     = ConvDecoderSD(self.representationSize + self.recurrentStateSize, self.obsShape).to(device)
        self.sequenceModel   = SequenceModelSD(self.representationSize, self.actionSize, self.recurrentStateSize).to(device)
        self.priorNet        = PriorNetSD(self.recurrentStateSize, self.representationLength, self.representationClasses).to(device)
        self.posteriorNet    = PosteriorNetSD(self.recurrentStateSize + self.compressedObservationSize, self.representationLength, self.representationClasses).to(device)
        self.rewardPredictor = RewardPredictorSD(self.recurrentStateSize + self.representationSize).to(device)
        # self.actor           = Actor(self.recurrentStateSize + self.representationSize, self.actionSize, actionLow=self.actionLow, actionHigh=self.actionHigh)
        self.actor           = ActorSD(self.recurrentStateSize + self.representationSize, self.actionSize)
        self.critic          = CriticSD(self.recurrentStateSize + self.representationSize).to(device)

        self.recurrentState  = self.sequenceModel.initializeRecurrentState()
        self.valueMoments    = Moments()
        self.totalUpdates    = 0
        self.freeNats        = 1
        self.clipGradients   = False  # Potentially useful, but slow and I prefer speeeeed

        self.worldModelParams    = (list(self.convEncoder.parameters()) + list(self.convDecoder.parameters()) + list(self.sequenceModel.parameters()) +
                                    list(self.priorNet.parameters()) + list(self.posteriorNet.parameters()) + list(self.rewardPredictor.parameters()))
        self.worldModelOptimizer = optim.AdamW(self.worldModelParams, lr=self.worldModelLR)
        
        self.criticOptimizer = optim.AdamW(self.critic.parameters(), lr=self.criticLR)
        self.actorOptimizer  = optim.AdamW(self.actor.parameters(), lr=self.actorLR)


    @torch.no_grad()
    def act(self, observation, reset=False):
        if reset:
            self.recurrentState = self.sequenceModel.initializeRecurrentState()

        encodedObservation = self.convEncoder(observation)
        latentState, _, = self.posteriorNet(torch.cat((self.recurrentState, encodedObservation), -1))
        fullState = torch.cat((self.recurrentState, latentState), -1)
        return self.actor(fullState, training=False)

    def trainWorldModel(self, observations, actions, rewards):
        sequenceLength = observations.shape[1] 
        recurrentState = torch.zeros(self.worldModelBatchSize, self.recurrentStateSize).to(device)
        prior = torch.zeros(self.worldModelBatchSize, self.representationSize).to(device)

        encodedObservations = self.convEncoder(observations.view(self.worldModelBatchSize*sequenceLength, *self.obsShape))
        encodedObservations = encodedObservations.view(self.worldModelBatchSize, sequenceLength, -1) # [batchSize, sequenceLength, compressedObservationSize]

        recurrentStates, priors, priorsDistMean, priorsDistStd, posteriors, posteriorsDistMean, posteriorsDistStd = [], [], [], [], [], [], []
        for timestep in range(1, sequenceLength):
            recurrentState = self.sequenceModel(prior, actions[:, timestep - 1], recurrentState)
            prior, priorDistribution = self.priorNet(recurrentState)
            posterior, posteriorDistribution = self.posteriorNet(torch.cat((recurrentState, encodedObservations[:, timestep]), -1))

            recurrentStates.append(recurrentState)
            priors.append(prior)
            priorsDistMean.append(priorDistribution.mean)
            priorsDistStd.append(priorDistribution.scale)
            posteriors.append(posterior)
            posteriorsDistMean.append(posteriorDistribution.mean)
            posteriorsDistStd.append(posteriorDistribution.scale)

            prior = posterior

        recurrentStates     = torch.stack(recurrentStates, dim=1)
        priors              = torch.stack(priors, dim=1)
        priorsDistMean      = torch.stack(priorsDistMean, dim=1)
        priorsDistStd       = torch.stack(priorsDistStd, dim=1)
        posteriors          = torch.stack(posteriors, dim=1)
        posteriorsDistMean  = torch.stack(posteriorsDistMean, dim=1)
        posteriorsDistStd   = torch.stack(posteriorsDistStd, dim=1)
        fullStates          = torch.cat((recurrentStates, posteriors), -1)

        reconstructedObservations = self.convDecoder(fullStates.view(self.worldModelBatchSize*(sequenceLength - 1), -1))
        reconstructedObservations = reconstructedObservations.view(self.worldModelBatchSize, sequenceLength - 1, *self.obsShape)
        predictedRewards = self.rewardPredictor(fullStates) # [batchSize, sequenceLength-1]

        reconstructionLoss = self.betaReconstruction*F.mse_loss(reconstructedObservations, observations[:, 1:], reduction="none").mean(dim=[-3, -2, -1]).mean()
        rewardPredictorLoss = self.betaReward*F.mse_loss(predictedRewards, symlog(rewards))

        prior_dist = create_normal_dist(priorsDistMean, priorsDistStd, event_shape=1)
        posterior_dist = create_normal_dist(posteriorsDistMean, posteriorsDistStd, event_shape=1)
        
        klLoss = torch.mean(torch.max(torch.distributions.kl_divergence(posterior_dist, prior_dist), torch.tensor(self.freeNats, device=device)))

        worldModelLoss  =  reconstructionLoss + rewardPredictorLoss + klLoss

        self.worldModelOptimizer.zero_grad()
        worldModelLoss.backward()
        if self.clipGradients:
            torch.nn.utils.clip_grad_norm_(self.worldModelParams, self.gradientClipValue)
        self.worldModelOptimizer.step()

        sampledFullStates = fullStates.view(self.worldModelBatchSize*(sequenceLength - 1), -1)[torch.randperm(self.worldModelBatchSize*(sequenceLength - 1))[:self.actorCriticBatchSize]]
        metrics = {
            "worldModelLoss"        : worldModelLoss.item() - self.freeNats,
            "reconstructionLoss"    : reconstructionLoss.item(),
            "rewardPredictorLoss"   : rewardPredictorLoss.item(),
            "averageWMreward"       : predictedRewards.mean().item(),
            "klLoss"                : klLoss.item() - self.freeNats}
        return sampledFullStates.detach(), metrics
    

    def trainActorCritic(self, initialFullState):
        fullState = initialFullState.detach()   # [actorCriticBatchSize, recurrentSize + representationSize]
        recurrentState, latentState = torch.split(fullState, [self.recurrentStateSize, self.representationSize], -1)

        fullStates = []
        for _ in range(self.imaginationHorizon):
            action = self.actor(fullState.detach())
            recurrentState = self.sequenceModel(latentState, action, recurrentState)
            latentState, _ = self.priorNet(recurrentState)
            fullStates.append(torch.cat((recurrentState, latentState), -1))

        fullStates = torch.stack(fullStates, dim=1)

        predictedRewards = self.rewardPredictor(fullStates, useSymexp=True)
        values = self.critic(fullStates)
        continues = self.gamma*torch.ones_like(values)

        lambdaValues = self.compute_lambda_values(
            predictedRewards,
            values,
            continues,
            self.imaginationHorizon
        )

        actorLoss = -torch.mean(lambdaValues)

        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        self.actorOptimizer.step()

        valuesForCriticUpdate = self.critic(fullStates[:, :-1].detach())
        criticLoss = F.mse_loss(valuesForCriticUpdate, lambdaValues.detach())

        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        self.criticOptimizer.step()

        metrics = {
            "actorLoss"         : actorLoss.item(),
            # "logprobs"          : actionLogProbabilities.mean().item(),
            # "advantages"        : advantages.mean().item(),
            # "entropy"           : entropies.mean().item(),
            "averageACreward"   : predictedRewards.mean().item(),
            "criticLoss"        : criticLoss.item(),
            # "targetCriticValue" : targetCriticValues.mean().item(),
            "criticValue"       : values.mean().item()}
        return metrics


    def compute_lambda_values(self, rewards, values, continues, horizon_length, lambda_=0.95):
        """
        rewards : (batch_size, time_step, hidden_size)
        values : (batch_size, time_step, hidden_size)
        continue flag will be added
        """
        rewards = rewards[:, :-1]
        continues = continues[:, :-1]
        next_values = values[:, 1:]
        last = next_values[:, -1]
        inputs = rewards + continues * next_values * (1 - lambda_)

        outputs = []
        # single step
        for index in reversed(range(horizon_length - 1)):
            last = inputs[:, index] + continues[:, index] * lambda_ * last
            outputs.append(last)
        returns = torch.stack(list(reversed(outputs)), dim=1).to(device)
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
            # 'targetCritic'          : self.critic.state_dict(),
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
        # self.targetCritic.load_state_dict(        checkpoint['targetCritic'])
        self.worldModelOptimizer.load_state_dict( checkpoint['worldModelOptimizer'])
        self.criticOptimizer.load_state_dict(     checkpoint['criticOptimizer'])
        self.actorOptimizer.load_state_dict(      checkpoint['actorOptimizer'])
        self.valueMoments.load_state_dict(        checkpoint['valueMoments'])
        self.recurrentState =                     checkpoint['recurrentState']
        self.totalUpdates =                       checkpoint['totalUpdates']
        # print(f"Loaded checkpoint from: {checkpointPath}")
