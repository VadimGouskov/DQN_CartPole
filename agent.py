import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from cartpole.qnetwork import Qnetwork
import cartpole.utils as u
from torch.autograd.variable import Variable


class Agent(object):
    def __init__(self, gamma, epsilon, epsilonDecay, epsilonMin, alpha, maxMemSize, actionSpace, targetReplaceCount):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.epsilonMin = epsilonMin
        self.alpha = alpha
        self.maxMemSize = maxMemSize
        self.actionSpace = actionSpace
        self.targetReplaceCount = targetReplaceCount

        self.steps = 0
        self.memory = []
        self.memCounter = 0

        self.QPolicy = Qnetwork(2, alpha) # TODO make actionspace length dynamic (here 2)
        self.QTarget = Qnetwork(2, alpha)

    def storeTransition(self, state, action, reward, state_):
        if self.memCounter < self.maxMemSize:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.memCounter % self.maxMemSize] = [state, action, reward, state_]
        self.memCounter += 1

    def chooseAction(self, observation):
        isRandom = 0
        randomActionChance = np.random.random()

        if randomActionChance < self.epsilon:
            action = self.actionSpace.sample()
            isRandom = 1
        else:
            act = torch.argmax(self.QPolicy.forward(torch.Tensor(observation)))
            action = act.data.numpy() #act.data.numpy()
            print(action, ' ',  end="", flush=True)

        return action, isRandom

    def learn(self, batchSize):

        # BATCHIN
        if(batchSize > 1):
            if self.memCounter < batchSize:
                minibatch = self.memory[0:self.memCounter-1]
            else:
                batchStart = np.random.randint(0, self.memCounter - batchSize)
                minibatch = self.memory[batchStart: batchStart + batchSize]
        else:
            sampleIndex = np.random.randint(0, self.memCounter)
            minibatch = self.memory[sampleIndex: sampleIndex+1]

        if(len(minibatch) == 0):
            return
        replayMemory = np.array(minibatch)

        #PREDICTION
        try:
            policy = self.QPolicy.forward(u.toTensor(replayMemory[:,0])).to(self.QPolicy.device)
            next = self.QTarget.forward(u.toTensor(replayMemory[:,3])).to(self.QTarget.device)
            maxAction = torch.argmax(next).to(self.QPolicy.device)
            rewards = torch.Tensor(replayMemory[:,2].astype(np.float))
        except:
            print("failed at step:", self.steps)
            exit()

        test = rewards + self.gamma * torch.max(next)
        target = rewards + self.gamma * torch.max(next) #policy[maxAction]

        #LOSS
        self.QPolicy.optimizer.zero_grad()
        loss = self.QPolicy.loss(policy, target).to(self.QPolicy.device)
        loss.backward()
        self.QPolicy.optimizer.step()
        # self.Qevaluation.optimizer.zero_grad()


        # TODO every C steps set PTarget to Policy
        if self.steps % self.targetReplaceCount == 0:
            stateDict = self.QPolicy.state_dict()
            self.QTarget.load_state_dict(self.QPolicy.state_dict())
            #target = policy

        # EPSILON DECAY
        if self.steps > 100:
            if self.epsilon > self.epsilonMin:
                self.epsilon *= self.epsilonDecay
            else:
                self.epsilon = self.epsilonMin

        self.steps += 1

