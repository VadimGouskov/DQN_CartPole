import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from cartpole.qnetwork import Qnetwork
import cartpole.utils as u
from torch.autograd.variable import Variable


class Agent(object):
    def __init__(self, gamma, epsilon, epsilonDecay, epsilonMin, alpha, maxMemSize, actionSpace, targetReplaceCount, episodeEnd):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.epsilonMin = epsilonMin
        self.alpha = alpha
        self.maxMemSize = maxMemSize
        self.actionSpace = actionSpace
        self.episodeEnd = episodeEnd
        self.targetReplaceCount = targetReplaceCount

        self.steps = 0
        self.stepCounter = 0
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
        randomActionChance = np.random.random()

        if randomActionChance < self.epsilon:
            action = self.actionSpace.sample()
        else:
            # TODO LEFTOFF
            act = torch.argmax(self.QPolicy.forward(torch.Tensor(observation)))
            action = act.data.numpy()

        self.steps += 1
        return action

    def learn(self, batchSize):
        self.QPolicy.optimizer.zero_grad()
        # if self.targetReplaceCount is not None and self.stepCounter%self.targetReplaceCount == 0:
        #     #TODO remove ? not used ?
        #     self.Qprediction.load_state_dict(self.Qevaluation.state_dict())

        # get minibatch from memory
        minibatch = []

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

        # memory : []


        # train model w/ minibatch
        # test = replayMemory[:, 0]
        #TODO instead of [0][0] get all the [:][0]
        # test = replayMemory[:, 0]
        # best = u.toTensorArray(test)
        try:
            policy = self.QPolicy.forward(u.toTensor(replayMemory[:,0])).to(self.QPolicy.device)

            target = self.QTarget.forward(u.toTensor(replayMemory[:,3])).to(self.QTarget.device)
            maxAction = torch.argmax(target).to(self.QPolicy.device)
            rewards = torch.Tensor(replayMemory[:,2].astype(np.float))
        except:
            print("step:", self.stepCounter)
            top = self.stepCounter
            exit()


        # BELLMAN
        # targetCalc = rewards + self.gamma * torch.max(prediction[1])
        # maxPrediction= torch.max(prediction[1])
        # target[:, maxAction] = rewards + self.gamma * torch.max(target[:,1])

        # self.QTarget[:, maxAction] = rewards + self.gamma * torch.max(target)
        test = rewards + self.gamma * torch.max(target)
        policy[maxAction] = rewards + self.gamma * torch.max(target)

        #LOSS Function
        loss = self.QPolicy.loss(target, policy).to(self.QPolicy.device)
        loss.backward()
        self.QPolicy.optimizer.step()
        # self.Qevaluation.optimizer.zero_grad()
        self.stepCounter += 1

        if self.stepCounter % self.targetReplaceCount == 0:
            target = policy

        # EPSILON DECAY
        if self.steps > 20:
            if self.epsilon > self.epsilonMin:
                self.epsilon *= self.epsilonDecay
            else:
                self.epsilon = self.epsilonMin



