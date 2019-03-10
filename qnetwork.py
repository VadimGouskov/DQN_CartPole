import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
from cartpole.utils import showImage, saveImage, showTensor, saveTensor

# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor # FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


import numpy as np

class Qnetwork(nn.Module):
    def __init__(self, actionSpaceSize , alpha):
        super(Qnetwork, self).__init__()
        self.alpha = alpha
        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 2)

        self.optimizer = optimizer.RMSprop(self.parameters(), lr=self.alpha)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input):

        prop = input
        prop = F.relu(self.fc1(prop))
        prop = F.relu(self.fc2(prop))

        out = self.fc3(prop)

        return out


    def printSize(self, observation, message = "<message>"):
        print(message + " - input size of is: " + str(len(observation)) + " x " + str(len(observation[0])))


