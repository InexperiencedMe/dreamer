import gymnasium as gym
import numpy as np
import torch
from utils import *
from dreamer import *
import random
torch.set_printoptions(threshold=2000, linewidth=200, sci_mode=False)
np.set_printoptions(threshold=2000, linewidth=200)

environmentName         = "CarRacing-v3"
renderMode              = None
numUpdates              = 10000
episodesBeforeStart     = 20
playInterval            = 10
stepCountLimit          = 256
bufferSize              = 30
resume                  = False
saveMetrics             = True
saveCheckpoints         = True
runName                 = f"{environmentName}__AAAAAA_FULLSTATE_DETACHED"
checkpointToLoad        = f"checkpoints/{runName}_42000"
metricsFilename         = f"metrics/{runName}"
plotFilename            = f"plots/{runName}"
videoFilename           = f"videos/{runName}"
saveMetricsInterval     = 10
checkpointInterval      = 500
numNewEpisodePlay       = 1
seed                    = 2

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make(environmentName, render_mode=renderMode)
observationShape = torch.tensor(env.observation_space.shape)
actionSize = torch.tensor(env.action_space.shape) if hasattr(env.action_space, 'shape') else np.array([env.action_space.n])
print(f"Env {environmentName} with observations {observationShape} and actions {actionSize}\n###\n")
dreamer = Dreamer()

episodeBuffer = EpisodeBuffer(size=bufferSize)

if resume:
    dreamer.loadCheckpoint(checkpointToLoad)
    start = dreamer.totalUpdates
else:
    start = 0

for i in range(start - episodesBeforeStart, start + numUpdates + 1):
    for _ in range(numNewEpisodePlay):
        if i % playInterval == 0 or i < start:
            observation, info = env.reset(seed=seed + abs(i))
            observation = torch.from_numpy(np.transpose(observation, (2, 0, 1))).unsqueeze(0).to(device).float()/255.0
            observations, actions, rewards, dones = [], [], [], []
            stepCount, totalReward, done = 0, 0, False
            while not done:
                action = dreamer.act(observation, reset=(stepCount == 0)).view(-1)
                newObservation, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                newObservation = torch.from_numpy(np.transpose(newObservation, (2, 0, 1))).unsqueeze(0).to(device).float()/255.0
                stepCount += 1
                done = terminated or truncated or stepCount >= stepCountLimit
                totalReward += reward
                
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                # dones.append(done)
                observation = newObservation

            if len(observations) == stepCountLimit: # preventing very rare cases where episode terminates early. I could also rebuild the buffer so a sequence is stitched from multiple episodes
                episodeBuffer.addEpisode(torch.stack(observations).squeeze(1), # observation includes initial
                                        torch.stack(actions).to(device),       # action synced with observation
                                        torch.tensor(rewards[:-1]).to(device))      # reward only for next step (no reward for initial observation), 1 fewer reward than obs and actions

    if i > start:
        selectedEpisodeObservations, selectedEpisodeActions, selectedEpisodeRewards = episodeBuffer.sampleEpisodes(dreamer.worldModelBatchSize)
        sampledFullStates, worldModelLoss, reconstructionLoss, rewardPredictionLoss, klLoss = dreamer.trainWorldModel(selectedEpisodeObservations, selectedEpisodeActions, selectedEpisodeRewards)
        criticLoss, actorLoss, valueEstimate = dreamer.trainActorCritic(sampledFullStates)

    if i % saveMetricsInterval == 0 and i > start and saveMetrics:
        saveLossesToCSV(metricsFilename, {
            "i": i,
            "worldModelLoss": worldModelLoss,
            "reconstructionLoss": reconstructionLoss,
            "rewardPredictionLoss": rewardPredictionLoss,
            "klLoss": klLoss,
            "criticLoss": criticLoss,
            "actorLoss": actorLoss,
            "valueEstimate": valueEstimate,
            "totalReward": totalReward})
        
        # print(f"\nnewest actions:\n{episodeBuffer.getNewestEpisode()[1][:5]}")

    if i % checkpointInterval == 0 and i > start and saveCheckpoints:
        print(f"i {i:6}: worldModelLoss, criticLoss, actorLoss, reward = {worldModelLoss:8.4f}, {criticLoss:8.4f}, {actorLoss:8.4f}, {totalReward:.2f}")
        dreamer.totalUpdates = i
        dreamer.saveCheckpoint(f"checkpoints/{runName}_{i}")
        plotMetrics(metricsFilename, show=False, save=True, savePath=f"{plotFilename}_{i}") # TODO: plot should replace the unnecessary previous file
        saveVideoFromGymEnv(dreamer, environmentName, f"{videoFilename}_{i}", frameLimit=stepCountLimit)

env.close()