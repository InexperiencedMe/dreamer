import torch
import torch.nn as nn
import torch.functional as F
from utils import *
from neuralNets import *

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

        self.convEncoder     = ConvEncoder(self.obsShape, self.compressedObservationsSize).to(device)
        self.convDecoder     = ConvDecoder(self.representationSize + self.recurrentStateSize, self.obsShape).to(device)
        self.sequenceModel   = SequenceModel(self.representationSize, self.actionSize, self.recurrentStateSize).to(device)
        self.priorNet        = PriorNet(self.recurrentStateSize, self.representationLength, self.representationClasses).to(device)
        self.posteriorNet    = PosteriorNet(self.recurrentStateSize + self.compressedObservationsSize, self.representationLength, self.representationClasses).to(device)
        self.rewardPredictor = RewardPredictor(self.recurrentStateSize + self.representationSize).to(device)
        self.actor           = Actor()
        self.critic          = Critic()

        self.worldModelOptimizer = optim.AdamW(
            list(self.convEncoder.parameters()) + list(self.convDecoder.parameters()) + 
            list(self.sequenceModel.parameters()) + list(self.priorNet.parameters()) + 
            list(self.posteriorNet.parameters()) + list(self.rewardPredictor.parameters()), 
            lr=3e-4
)

    def trainWorldModel(self, observations, actions, rewards):
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
        klLoss = priorLoss + posteriorLoss # We add it because they have the same value, no need to distinguish it

        worldModelLoss = reconstructionLoss + rewardPredictorLoss + klLoss

        self.worldModelOptimizer.zero_grad()
        worldModelLoss.backward()
        self.worldModelOptimizer.step()

        return worldModelLoss.item(), reconstructionLoss.item(), rewardPredictorLoss.item(), klLoss.item()
    
    def trainAgent(self, observations, actions):
        currentObservation = observations[0]
        encodedObservation = self.convEncoder(currentObservation)
        recurrentState = self.sequenceModel.initializeRecurrentState()
        latentState, _ = self.posteriorNet(torch.cat((recurrentState, encodedObservation), -1))

        fullStateRepresentations = []
        actionLogProbabilities = []
        predictedRewards = []
        for _ in range(self.imaginationHorizon):
            action, logProbabilities = self.actor(torch.cat((recurrentState, latentState), -1))
            actionLogProbabilities.append(logProbabilities)

            nextRecurrentState = self.sequenceModel(latentState, action, recurrentState)
            nextLatentState, _ = self.priorNet(recurrentState)

            fullStateRepresentation = torch.cat((nextRecurrentState, nextLatentState), -1)
            reward = self.rewardPredictor(fullStateRepresentation)

            fullStateRepresentations.append(fullStateRepresentation)
            actionLogProbabilities.append(actionLogProbabilities)
            predictedRewards.append(reward)

            recurrentState = nextRecurrentState
            latentState = nextLatentState

        fullStateRepresentations = torch.stack(fullStateRepresentation)
        actionLogProbabilities = torch.stack(actionLogProbabilities)
        predictedRewards = torch.stack(predictedRewards)

        valueEstimates = self.critic(fullStateRepresentations)
        lastValue = valueEstimates[-1]  # Bootstrap from last predicted value



        lambda_returns = self.lambda_return(
            reward=predicted_rewards,
            value=value_predictions,
            pcont=predicted_discounts,
            bootstrap=lastValue,
            lambda_=self.config.lambda_,
        )
        self.returns = lambda_returns[0, :].detach().cpu().numpy()  # Save for logging

        # Update value network (critic)
        value_targets = lambda_returns.detach()
        value_loss = F.mse_loss(value_predictions, value_targets)

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config.max_grad_norm
        )
        self.critic_optimizer.step()

        # Update policy network (actor)
        value_predictions_for_policy = self.critic(flattened_hidden_states).view(
            self.config.imagination_horizon, self.config.batch_size
        )
        advantages = lambda_returns.detach() - value_predictions_for_policy
        policy_loss = -(advantages * action_log_probabilities).mean()

        # Add entropy regularization to encourage exploration
        current_action_distribution = self.actor(flattened_hidden_states)
        if self.is_discrete:
            action_probs = current_action_distribution.view(
                self.config.imagination_horizon, self.config.batch_size, -1
            )
            log_probs = torch.log(action_probs + 1e-8)
            entropy = -torch.sum(action_probs * log_probs, dim=-1).mean()
        else:
            action_mean, action_std = current_action_distribution
            normal_dist = torch.distributions.Normal(action_mean, action_std)
            entropy = normal_dist.entropy().mean()
        policy_loss += -self.config.entropy_scale * entropy

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.max_grad_norm
        )
        self.actor_optimizer.step()

        # Update target value network
        self._soft_update(self.target_critic, self.critic)

        return policy_loss.item(), value_loss.item()

    def lambdaReturns(rewards, values, gamma=0.99, lambda_=0.95):
        n = len(rewards)
        returns = torch.zeros(n)
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