import torch
import torch.nn as nn
import torch.functional as F
from utils import *
import torch.distributions as distributions
from torch.distributions import TanhTransform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorSD(nn.Module):
    def __init__(self, inputSize, actionSize):
        super().__init__()
        self.network = sequentialModel1D(inputSize, [256, 256], 2*actionSize)

    def forward(self, x, training=True):
        x = self.network(x)
        dist = create_normal_dist(
            x,
            mean_scale=5,
            init_std=5.,
            min_std=0.0001,
            activation=torch.tanh,
        )
        dist = torch.distributions.TransformedDistribution(dist, TanhTransform())
        action = torch.distributions.Independent(dist, 1).rsample()
        return action
    

class CriticSD(nn.Module):
    def __init__(self, inputSize):
        super(CriticSD, self).__init__()
        self.mlp = sequentialModel1D(inputSize, [256, 256], 1)

    def forward(self, x):
        return self.mlp(x).squeeze(-1)
    
class ConvEncoderSD(nn.Module):
    def __init__(self, inputShape, outputSize):
        super(ConvEncoderSD, self).__init__()
        c, h, w = inputShape
        self.convolutionalNet = nn.Sequential(
            nn.Conv2d(c, 8, kernel_size=4, stride=2, padding=1),    # Output: (8, h/2, w/2)
            nn.Tanh(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),    # Output: (16, h/4, w/4)
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),   # Output: (32, h/8, w/8)
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),   # Output: (64, h/16, w/16)
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(64 * (h // 16) * (w // 16), outputSize),
            nn.Tanh(),
        )

    def forward(self, obs):
        return self.convolutionalNet(obs.float())


class ConvDecoderSD(nn.Module):
    def __init__(self, inputSize, outputShape):
        super(ConvDecoderSD, self).__init__()
        self.outputShape = outputShape
        c, h, w = outputShape
        self.fc = nn.Sequential(
            nn.Linear(inputSize, 64 * (h // 16) * (w // 16)),
            nn.Tanh(),
        )
        self.deconvolutionalNet = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # Output: (32, h/8, w/8)
            nn.Tanh(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # Output: (16, h/4, w/4)
            nn.Tanh(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),    # Output: (8, h/2, w/2)
            nn.Tanh(),
            nn.ConvTranspose2d(8, c, kernel_size=4, stride=2, padding=1),     # Output: (c, h, w)
            nn.Sigmoid(),  # Output pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.fc(x)
        batchSize = x.size(0)
        c, h, w = 64, self.outputShape[1] // 16, self.outputShape[2] // 16
        x = x.view(batchSize, c, h, w)
        return (self.deconvolutionalNet(x))


class SequenceModelSD(nn.Module):
    def __init__(self, representationSize, actionSize, recurrentStateSize):
        super().__init__()
        self.representationSize = representationSize
        self.actionSize = actionSize
        self.recurrentStateSize = recurrentStateSize
        self.preprocess = sequentialModel1D(representationSize + actionSize, [], 256, finishWithActivation=True)
        self.recurrent = nn.GRUCell(256, recurrentStateSize)

    def forward(self, representation, action, recurrentState):
        return self.recurrent(self.preprocess(torch.cat((representation, action), -1)), recurrentState)
    
    def initializeRecurrentState(self, size=1):
        return torch.zeros((size, self.recurrentStateSize)).to(device)


class PriorNetSD(nn.Module):
    def __init__(self, inputSize, representationLength=16, representationClasses=16):
        super().__init__()
        self.representationLength = representationLength
        self.representationClasses = representationClasses
        self.representationSize = representationLength*representationClasses
        self.mlp = sequentialModel1D(inputSize, [512, 256], self.representationSize*2)

    def forward(self, x):
        x = self.mlp(x)
        outputDist = create_normal_dist(x, min_std=0.0001)
        output = outputDist.rsample()
        return output, outputDist


class PosteriorNetSD(nn.Module):
    def __init__(self, inputSize, representationLength=16, representationClasses=16):
        super().__init__()
        self.representationLength = representationLength
        self.representationClasses = representationClasses
        self.representationSize = representationLength*representationClasses
        self.mlp = sequentialModel1D(inputSize, [512, 256], self.representationSize*2)

    def forward(self, x):
        x = self.mlp(x)
        outputDist = create_normal_dist(x, min_std=0.0001)
        output = outputDist.rsample()
        return output, outputDist


class RewardPredictorSD(nn.Module):
    def __init__(self, inputSize):
        super(RewardPredictorSD, self).__init__()
        self.mlp = sequentialModel1D(inputSize, [256, 256], 1)

    def forward(self, x, useSymexp=False):
        out = self.mlp(x).squeeze(-1)
        if useSymexp:
            return symexp(out)
        else:
            return out