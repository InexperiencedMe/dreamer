import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import csv
from collections import deque, namedtuple
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as pgo
import plotly.colors as pc
import imageio.v2 as imageio
import gymnasium as gym
import cv2 as cv
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
    
def displayImage(imageNdarray):
    import matplotlib.pyplot as plt
    plt.imshow(imageNdarray)
    plt.axis('off')
    plt.show()

def saveImage(imageNdarray, filename):
    from PIL import Image
    image = Image.fromarray(imageNdarray)
    image.save(filename)

@torch.no_grad()
def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))

@torch.no_grad()
def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

def saveLossesToCSV(filename, metrics):
    fileAlreadyExists = os.path.isfile(filename + ".csv")
    with open(filename + ".csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        if not fileAlreadyExists:
            writer.writerow(metrics.keys())
        writer.writerow(metrics.values())

def plotMetrics(filename, title="", show=True, save=False, savePath="metricsPlot", window=10):
    if not filename.endswith(".csv"):
        filename += ".csv"

    data = pd.read_csv(filename)
    fig = pgo.Figure()

    colors = pc.DEFAULT_PLOTLY_COLORS
    num_colors = len(colors)

    for idx, column in enumerate(data.columns[1:]):
        fig.add_trace(pgo.Scatter(
            x=data["i"], y=data[column], mode='lines',
            name=f"{column} (original)",
            line=dict(color='gray', width=1, dash='dot'),
            opacity=0.5, visible='legendonly'))

        smoothed_data = data[column].rolling(window=window, min_periods=1).mean()
        fig.add_trace(pgo.Scatter(
            x=data["i"], y=smoothed_data, mode='lines',
            name=f"{column} (smoothed)",
            line=dict(color=colors[idx % num_colors], width=2)))  # Cycle through colors

    fig.update_layout(
        title=f"{title}",
        title_x=0.5,
        xaxis_title="Iterations (i)",
        yaxis_title="Value",
        template="plotly_dark",
        height=1080,
        width=1920,
        legend=dict(
            x=0.04,
            y=0.04,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="White",
            borderwidth=2))

    # Save and/or show the plot
    if save:
        if not savePath.endswith(".html"):
            savePath += ".html"
        fig.write_html(savePath)
        print(f"Saved html plot to {savePath}")

    if show:
        fig.show()

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
    
    def sampleEpisodes(self, numEpisodes):
        if numEpisodes > len(self):
            raise ValueError("Requested more samples than available episodes.")
        
        episodeIndices = random.sample(range(len(self)), numEpisodes)
        observationsList = [self.observations[i] for i in episodeIndices]
        actionsList = [self.actions[i] for i in episodeIndices]
        rewardsList = [self.rewards[i] for i in episodeIndices]

        observationsStacked = torch.stack(observationsList)
        actionsStacked = torch.stack(actionsList)
        rewardsStacked = torch.stack(rewardsList)

        return observationsStacked, actionsStacked, rewardsStacked

    def getNewestEpisode(self):
        episodeIndex = len(self) - 1
        return self.observations[episodeIndex], self.actions[episodeIndex], self.rewards[episodeIndex]
    
    def __len__(self):
        return len(self.observations)
    
def saveVideoFrom4DTensor(observations, filename, fps=30):
    if not filename.lower().endswith(".mp4"):
        filename += ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for observation in observations:
            frame = (observation.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
            video.append_data(frame)


def saveVideoFromGymEnv(actor, envName, filename, frameLimit=512, fps=30, macroBlockSize=16):
    env = gym.make(envName, render_mode="rgb_array")
    observation, _ = env.reset()
    observation = torch.from_numpy(np.transpose(observation, (2, 0, 1))).unsqueeze(0).to(device) / 255.0
    done, frameCount, totalReward = False, 0, 0
    frames = []

    while not done and frameCount < frameLimit:
        action = actor.act(observation, reset=(frameCount == 0)).view(-1)
        observation, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        observation = torch.from_numpy(np.transpose(observation, (2, 0, 1))).unsqueeze(0).to(device) / 255.0
        done = terminated or truncated
        totalReward += reward
        frame = env.render()
        frameCount += 1

        # Reisizng to get rid of the warning with macroBlockSize
        targetHeight = (frame.shape[0] + macroBlockSize - 1) // macroBlockSize * macroBlockSize
        targetWidth = (frame.shape[1] + macroBlockSize - 1) // macroBlockSize * macroBlockSize
        resizedFrame = cv.resize(frame, (targetWidth, targetHeight), interpolation=cv.INTER_LINEAR)
        frames.append(resizedFrame)

    env.close()
    finalFilename = f"{filename}_reward_{int(totalReward)}.mp4"
    
    with imageio.get_writer(finalFilename, fps=fps) as video:
        for frame in frames:
            video.append_data(frame)
    print(f"Saved video to {finalFilename}")

class Moments(nn.Module):
    def __init__( self, decay = 0.99, min_=1, percentileLow = 0.05, percentileHigh = 0.95):
        super().__init__()
        self._decay = decay
        self._min = torch.tensor(min_)
        self._percentileLow = percentileLow
        self._percentileHigh = percentileHigh
        self.register_buffer("low", torch.zeros((), dtype=torch.float32, device=device))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.detach()
        low = torch.quantile(x, self._percentileLow)
        high = torch.quantile(x, self._percentileHigh)
        self.low = self._decay*self.low + (1 - self._decay)*low
        self.high = self._decay*self.high + (1 - self._decay)*high
        inverseScale = torch.max(self._min, self.high - self.low)
        return self.low.detach(), inverseScale.detach()
    