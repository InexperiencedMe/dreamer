import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
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