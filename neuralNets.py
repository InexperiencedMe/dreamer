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
        sample = F.gumbel_softmax(logits.view(self.representationLength, self.representationClasses), hard=True)
        return sample.view(-1), logits
    
class PosteriorNet(nn.Module):
    def __init__(self, inputSize, representationLength=16, representationClasses=16):
        super().__init__()
        self.representationLength = representationLength
        self.representationClasses = representationClasses
        self.mlp = sequentialModel1D(inputSize, [512, 256], representationLength*representationClasses)
    
    def forward(self, x):
        logits = self.mlp(x)
        sample = F.gumbel_softmax(logits.view(self.representationLength, self.representationClasses), hard=True)
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
        self.setLastLayerToZeros(self.mlp)

    def forward(self, x, useSymexp=False):
        out = self.mlp(x).squeeze(-1)
        if useSymexp:
            return symexp(out)
        else:
            return out

    def setLastLayerToZeros(self, network):
        nn.init.zeros_(network[-1].weight)
        nn.init.zeros_(network[-1].bias)

class Actor(nn.Module):
    def __init__(self, inputSize, actionSize):
        super(Actor, self).__init__()
        self.mean = sequentialModel1D(inputSize, [256], actionSize)
        self.logStd = sequentialModel1D(inputSize, [256], actionSize)

    def forward(self, x, training=True):
        mean = self.mean(x)
        std = torch.exp(self.logStd(x))
        # print(f"actor raw output:\nmean {mean} of shape {mean.shape}\std {std} of shape {std.shape}")
        distribution = distributions.Normal(mean, std)
        action = distribution.rsample() # Rsample when we use DreamerV1 update, normal sample when we use advantages??
        if training:
            # print(f"Will be returning entropy of shape: {distribution.entropy().sum(-1).shape} instead of {distribution.entropy().mean().shape} like before")
            return action, distribution.log_prob(action).sum(-1), distribution.entropy().sum(-1)
        else:
            return action

LOG_STD_MAX = 2
LOG_STD_MIN = -5
class ActorCleanRLStyle(nn.Module):
    def __init__(self, inputSize, actionSize, actionLow=[-1], actionHigh=[1]):
        super(ActorCleanRLStyle, self).__init__()
        self.mean = sequentialModel1D(inputSize, [256, 256], actionSize)
        self.logStd = sequentialModel1D(inputSize, [256, 256], actionSize)
        self.register_buffer("actionScale", ((torch.tensor(actionHigh, device=device) - torch.tensor(actionLow, device=device)) / 2.0))
        self.register_buffer("actionBias", ((torch.tensor(actionHigh, device=device) + torch.tensor(actionLow, device=device)) / 2.0))


    def forward(self, x, training=True):
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
            return action, logProbabilities.sum(-1), distribution.entropy().sum(-1)
        else:
            return action

class Critic(nn.Module):
    def __init__(self, inputSize):
        super(Critic, self).__init__()
        self.mlp = sequentialModel1D(inputSize, [256, 256], 1)
        self.setLastLayerToZeros(self.mlp)

    def forward(self, x):
        return self.mlp(x).squeeze(-1)

    def setLastLayerToZeros(self, network):
        nn.init.zeros_(network[-1].weight)
        nn.init.zeros_(network[-1].bias)

class ActorGPT(nn.Module):
    def __init__(self, inputSize, actionSize, actionBounds):
        """
        Args:
            inputSize (int): Dimension of the input features.
            actionSize (int): Number of actions.
            actionBounds (list of tuples): List of (min, max) for each action.
        """
        super(ActorGPT, self).__init__()
        self.mean = sequentialModel1D(inputSize, [256, 256], actionSize)
        self.logStd = sequentialModel1D(inputSize, [256, 256], actionSize)
        
        # Store action bounds
        self.actionBounds = torch.tensor(actionBounds, device=device)

    def forward(self, x, training=True):
        mean = self.mean(x)
        std = torch.exp(self.logStd(x))
        
        # Create a Normal distribution
        distribution = distributions.Normal(mean, std)
        action = distribution.rsample()  # Reparameterized sampling
        
        # Map action to bounded range using tanh and rescaling
        boundedAction = self._applyBounds(torch.tanh(action))
        
        if training:
            logProb = distribution.log_prob(action).sum(-1)
            entropy = distribution.entropy().sum(-1)
            return boundedAction, logProb, entropy
        else:
            return boundedAction

    def _applyBounds(self, action):
        """
        Rescale action to the specified bounds.
        Args:
            action (torch.Tensor): Action values in the range (-1, 1).
        Returns:
            torch.Tensor: Scaled action values.
        """
        minBounds = self.actionBounds[:, 0]
        maxBounds = self.actionBounds[:, 1]
        return (action + 1) / 2 * (maxBounds - minBounds) + minBounds