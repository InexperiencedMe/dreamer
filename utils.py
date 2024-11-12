import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import csv
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sequentialModel1D(inputSize, hiddenSizes, outputSize, finishWithActivation = False, activationFunction = nn.Tanh):
    layers = []
    currentInputSize = inputSize

    for hiddenSize in hiddenSizes:
        layers.append(nn.Linear(currentInputSize, hiddenSize))
        layers.append(activationFunction())
        currentInputSize = hiddenSize
    
    layers.append(nn.Linear(currentInputSize, outputSize))
    if finishWithActivation:
        layers.append(activationFunction())

    return nn.Sequential(*layers).to(device)

def sequentialModel3D(inputChannels, hiddenChannels, activationFunction = nn.Tanh):
        layers = []
        currentInputSize = inputChannels
        kernelSize = 7
        
        for hiddenSize in hiddenChannels:
            stride = kernelSize // 2
            layers.append(nn.Conv2d(currentInputSize, hiddenSize, kernelSize, stride))
            layers.append(activationFunction())
            currentInputSize = hiddenSize
            kernelSize = max(3, kernelSize - 2)

        layers.append(nn.Flatten())
        return nn.Sequential(*layers).to(device)

def calculateConvNetOutputSize(net, inputShape):
    return torch.numel(net(torch.ones(inputShape)))

class UnityInterface():
    def __init__(self, envName, seed=None):
        self.channelEnvironment = EnvironmentParametersChannel()
        self.channelEngine = EngineConfigurationChannel()
        self.env = UnityEnvironment(file_name=envName, seed=seed, side_channels=[self.channelEnvironment, self.channelEngine])
        self.env.reset()
        self.behaviorNames = list(self.env.behavior_specs)
        self.specs = self.prepareSpecs()
    
    def prepareSpecs(self):
        specs = {}
        for behavior in self.behaviorNames:
            behaviorSpecs = {}
            obsSpecs = self.env.behavior_specs[behavior].observation_specs
            behaviorSpecs["AgentsCount"] = self.countAgents(behavior)
            behaviorSpecs["Observations"] = [obsSpecs[i].shape for i in range(len(obsSpecs))]
            behaviorSpecs["ContinuousActions"] = self.env.behavior_specs[behavior].action_spec.continuous_size
            behaviorSpecs["DiscreteActions"] = self.env.behavior_specs[behavior].action_spec.discrete_branches
            specs[behavior] = behaviorSpecs
        return specs
        
    def getSpecs(self, behaviorName=None):
        if behaviorName == None:
            return self.specs
        else:
            return self.specs[behaviorName]
    
    def getBehaviorNames(self):
        return self.behaviorNames
    
    def getSteps(self, behaviorName):
        decisionSteps, terminalSteps = self.env.get_steps(behaviorName)
        return decisionSteps, terminalSteps

    def setActions(self, behaviorName, continuousActions = None, discreteActions = None):
        self.env.set_actions(behaviorName, ActionTuple(continuous=continuousActions, discrete=discreteActions))

    def step(self):
        self.env.step()

    def reset(self):
        self.env.reset()

    def close(self):
        self.env.close()

    def countAgents(self, behaviorName=None):
        if behaviorName is None:
            agentsCount = 0
            for behavior in self.behaviorNames:
                decisionSteps, terminalSteps = self.getSteps(behaviorName)
                agentsCount += len(set(decisionSteps).union(set(terminalSteps)))
        else:
            decisionSteps, terminalSteps = self.getSteps(behaviorName)
            agentsCount = len(set(decisionSteps).union(set(terminalSteps)))
        return agentsCount
    
def displayImage(imageNdarray):
    import matplotlib.pyplot as plt
    plt.imshow(imageNdarray)
    plt.axis('off')
    plt.show()

def saveImage(imageNdarray, filename):
    from PIL import Image
    image = Image.fromarray(imageNdarray)
    image.save(filename)

def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

def saveLossesToCSV(filename, metrics):
        with open(filename + ".csv", mode='a', newline='') as file:
            csv.writer(file).writerow(metrics)

def plotMetrics(filename, start_epoch=0, end_epoch=None):
    data = pd.read_csv(filename, header=None)
    data.columns = ["epoch", "worldModelLoss", "reconstructionLoss", "rewardPredictionLoss", 
                    "klLoss", "criticLoss", "actorLoss", "valueEstimate"]

    if end_epoch is None:
        end_epoch = data["epoch"].max()
    data = data[(data["epoch"] >= start_epoch) & (data["epoch"] <= end_epoch)]

    plt.figure(figsize=(16, 9))
    for column in data.columns[1:]:
        plt.plot(data["epoch"], data[column], label=column)
    
    plt.legend()
    plt.title("Losses Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.show()

class EpisodeBuffer:
    def __init__(self, size=20):
        self.size = size
        self.observations = deque(maxlen=size)
        self.actions = deque(maxlen=size)
        self.rewards = deque(maxlen=size)

    def addEpisode(self, observations, actions, rewards):
        self.observations.append(observations)
        self.actions.append(actions)
        self.rewards.append(rewards)

    def sampleEpisode(self):
        episodeIndex = random.randint(0, len(self) - 1)
        return self.observations[episodeIndex], self.actions[episodeIndex], self.rewards[episodeIndex]
    
    def __len__(self):
        return len(self.observations)
    