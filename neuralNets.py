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
            nn.Conv2d(c, 16, kernel_size=4, stride=2, padding=1),  # Output: (16, h/2, w/2)
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, h/4, w/4)
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, h/8, w/8)
            nn.Tanh(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, h/16, w/16)
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(128 * (h // 16) * (w // 16), outputSize),
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
            nn.Linear(inputSize, 128 * (h // 16) * (w // 16)),
            nn.Tanh(),
        )
        self.deconvolutionalNet = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, h/8, w/8)
            nn.Tanh(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, h/4, w/4)
            nn.Tanh(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Output: (16, h/2, w/2)
            nn.Tanh(),
            nn.ConvTranspose2d(16, c, kernel_size=4, stride=2, padding=1),  # Output: (c, h, w)
            nn.Sigmoid(),  # Output pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.fc(x)
        batchSize = x.size(0)
        c, h, w = 128, self.outputShape[1] // 16, self.outputShape[2] // 16
        x = x.view(batchSize, c, h, w)
        return (self.deconvolutionalNet(x))

class RewardPredictor(nn.Module):
    def __init__(self, inputSize):
        super(RewardPredictor, self).__init__()
        self.mlp = sequentialModel1D(inputSize, [256, 256], 1)

    def forward(self, x):
        return self.mlp(x)

LOG_STD_MAX = 2
LOG_STD_MIN = -5
class Actor(nn.Module):
    def __init__(self, inputSize, actionSize, actionLow=[-1], actionHigh=[1]):
        super(Actor, self).__init__()
        self.preprocess = sequentialModel1D(inputSize, [256, 256], 256)
        self.mean = sequentialModel1D(256, [256], actionSize)
        self.logStd = sequentialModel1D(256, [256], actionSize)
        self.register_buffer("actionScale", (torch.tensor(actionHigh, device=device) + torch.tensor(actionLow, device=device) / 2.0))
        self.register_buffer("actionBias", (torch.tensor(actionHigh, device=device) - torch.tensor(actionLow, device=device) / 2.0))

    def forward(self, x, training=True):
        x = self.preprocess(x)
        mean = self.mean(x)
        logStd = torch.tanh(self.logStd(x))
        logStd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (logStd + 1)
        std = torch.exp(logStd)
        distribution = distributions.Normal(mean, std)
        sample = distribution.rsample()
        sampleTanh = torch.tanh(sample)
        action = sampleTanh*self.actionScale + self.actionBias

        if training:
            logProbabilities = distribution.log_prob(sample)
            logProbabilities -= torch.log(self.actionScale * (1 - sampleTanh.pow(2)) + 1e-6)
            logProbabilities = logProbabilities.sum(-1, keepdim=True)
            return action, logProbabilities, distribution.entropy().mean()
        else:
            return action


class Critic(nn.Module):
    def __init__(self, inputSize):
        super(Critic, self).__init__()
        self.mlp = sequentialModel1D(inputSize, [256, 256], 1)

    def forward(self, x):
        return self.mlp(x).squeeze(-1)