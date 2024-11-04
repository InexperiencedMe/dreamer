import torch
import torch.nn as nn
import torch.functional as F
from utils import *
import torch.distributions as distributions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SequenceModel(nn.Module):
    def __init__(self, representationSize, actionSize, recurrentStateSize):
        super().__init__()
        self.representationSize = representationSize
        self.actionSize = actionSize
        self.recurrentStateSize = recurrentStateSize
        self.recurrent = nn.GRUCell(representationSize + actionSize, recurrentStateSize)

    def forward(self, representation, action, recurrentState):
        return self.recurrent(torch.cat((representation, action), -1), recurrentState)
    
    def initializeRecurrentState(self):
        return torch.zeros(self.recurrentStateSize).to(device)

class PriorNet(nn.Module):
    def __init__(self, inputSize, representationLength=16, representationClasses=16):
        super().__init__()
        self.representationLength = representationLength
        self.representationClasses = representationClasses
        self.mlp = sequentialModel1D(inputSize, [512, 256], representationLength*representationClasses)
    
    def forward(self, x):
        logits = self.mlp(x)
        sample = F.gumbel_softmax(logits.view(self.representationLength, self.representationClasses))
        return sample.view(-1), logits
    
class PosteriorNet(nn.Module):
    def __init__(self, inputSize, representationLength=16, representationClasses=16):
        super().__init__()
        self.representationLength = representationLength
        self.representationClasses = representationClasses
        self.mlp = sequentialModel1D(inputSize, [512, 256], representationLength*representationClasses)
    
    def forward(self, x):
        logits = self.mlp(x)
        sample = F.gumbel_softmax(logits.view(self.representationLength, self.representationClasses))
        return sample.view(-1), logits


class ConvEncoder(nn.Module):
    def __init__(self, inputShape, outputSize):
        super(ConvEncoder, self).__init__()
        c, h, w = inputShape
        self.convolutionalNet = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, h/2, w/2)
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, h/4, w/4)
            nn.Tanh(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, h/8, w/8)
            nn.Tanh(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: (256, h/16, w/16)
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(256 * (h // 16) * (w // 16), outputSize),
            nn.Tanh(),
        )

    def forward(self, obs):
        return self.convolutionalNet(obs.float())

class ConvDecoder(nn.Module):
    def __init__(self, inputSize, outputShape):
        super(ConvDecoder, self).__init__()
        self.outputShape = outputShape
        c, h, w = outputShape
        self.fc = nn.Sequential(
            nn.Linear(inputSize, 256 * (h // 16) * (w // 16)),
            nn.Tanh(),
        )
        self.deconvolutionalNet = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, h/8, w/8)
            nn.Tanh(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, h/4, w/4)
            nn.Tanh(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, h/2, w/2)
            nn.Tanh(),
            nn.ConvTranspose2d(32, c, kernel_size=4, stride=2, padding=1),  # Output: (c, h, w)
            nn.Sigmoid(),  # Output pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.fc(x)
        batchSize = x.size(0)
        c, h, w = 256, self.outputShape[1] // 16, self.outputShape[2] // 16
        x = x.view(batchSize, c, h, w)
        return (self.deconvolutionalNet(x))

class RewardPredictor(nn.Module):
    def __init__(self, inputSize):
        super(RewardPredictor, self).__init__()
        self.mlp = sequentialModel1D(inputSize, [256, 256], 1)

    def forward(self, x):
        return self.mlp(x)
    
class Actor(nn.Module):
    def __init__(self, inputSize, actionSize):
        super(Actor, self).__init__()
        self.mean = sequentialModel1D(inputSize, [256], actionSize)
        self.logStd = sequentialModel1D(inputSize, [256], actionSize)

    def forward(self, x):
        if self.is_discrete:
            mean = self.mean(x)
            std = torch.exp(self.logStd(x))
            distribution = distributions.Normal(mean, std)
            action = distribution.sample()
            return action, distribution.log_prob(action)

class Critic(nn.Module):
    def __init__(self, inputSize):
        super(Critic, self).__init__()
        self.mlp = sequentialModel1D(inputSize, [256], 1)

    def forward(self, x):
        return self.mlp(x).squeeze(-1)