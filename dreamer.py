import torch
import torch.nn as nn
import torch.functional as F
from utils import *

class Dreamer:
    def __init__(self):
        pass

    def train():
        pass
        # Encode all obs with encoder that uses symlog and outputs recurrentStateSize-sized output

        # Init recurrent state
        # Going through the whole episode:
            # pass recurrent state and transformed obs to posterior net
            # append posterior net output to a list

            # _, recurrentState = rnn(posteriorNetOutput + action)
            # append recurrentState to a list of recurrentStates

            # pass recurrentState to a priorNet to get priorNetOutput
            # append priorNetOutput to a list

        # stack tensors

        # kl loss computation

        # calculate fullStateRepresentation by concatenating recurrentState and posteriorNetOutput
        # pass through a decoder fullStateRepresentation to get back obs

        # in the future will be passing fullStateRepresentation to get also rewards and continuation

        # calculate all losses and step the optimizer