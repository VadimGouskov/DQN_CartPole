import gym
from cartpole.qnetwork import Qnetwork
from cartpole.agent import Agent
import numpy as np
import time
from cartpole.utils import saveImage, showImage, standardPreprocess, rgb2gray
import torch


np.set_printoptions(threshold=np.nan)
RENDER = False
scores = []
epsilonHistory = []
numberOfGames = 1000
batchSize = 1



env = gym.make('CartPole-v0')
env.reset()
actionSpace = env.action_space
observation_, reward, done, info   = env.step(actionSpace.sample())
action = actionSpace.sample()

print("cuda available: ", torch.cuda.is_available())


agent = Agent(  gamma=0.95,
                epsilon = 1.0,
                epsilonDecay=0.99,
                epsilonMin=0.1,
                alpha = 0.003,
                maxMemSize=150,
                targetReplaceCount=1, # always replace target immediately (after every batch)
                actionSpace=actionSpace,
                episodeEnd=0.05
              )
# INITIALIZE AGENT MEMORY TODO: is this still necessary & isn't this already handled in agent.storeTransition?
while agent.memCounter < agent.maxMemSize:
    print(agent.memCounter)
    observation = env.reset()
    observation = observation[:].astype(np.float32)
    done = False
    initFrameCounter = 0

    while not done:
        #choose random action
        # step and convert image to greyscale
        observation_, reward, done, info = env.step(actionSpace.sample())
        observation_ = observation_[:].astype(np.float32)
        agent.storeTransition(observation, action, reward, observation_)

        observation = observation_
print("memory initialized")





for i in range(numberOfGames):
    print("starting game ", str(i), " w/ epsilon ", agent.epsilon )
    epsilonHistory.append(agent.epsilon)
    done = False
    frameCounter = 0
    observation = env.reset()

    observation, reward, done, info = env.step(actionSpace.sample())
    observation = observation.astype(np.float32)

    while not done:

        # Use the chosen action as an index to choose the real action according to observation
        action = agent.chooseAction(observation)

        #step, convert to gray and add score
        observation_, reward, done, info = env.step(action)
        observation_ = observation_[:].astype(np.float32)
        scores.append(reward)

        #TODO implement custom reward system

        # store transition and learn
        agent.storeTransition(observation, action, reward, observation_)

        agent.learn(batchSize)

        #
        observation = observation_

        if RENDER:
            env.render()

        #for now now maximum render 150 before next episode frames
        frameCounter += 1
        if(frameCounter > 150):
            done = True
            frameCounter = 0

    totalScore = sum(scores)
    scores = []
    print("totalscore = ", totalScore)

env.close()

# for i in range(0, 50 ):
#     observation_, reward, done, info = env.step([0.0, 1.0, 0.0])
#     env.render()
#
# saveImage(observation_[0:80, 0:95])


# image = np.array(standardPreprocess(observation_)).astype(np.uint8)

#TODO TRY TO PRINT GREYSCALE ARRAY (FOR SHOW)

# grayImage = []
# for m in range(len(image)):
#     grayImage.append()
#     for n in range(len(image[m])):
#         grayImage[m][n] = 5
#
# print(grayImage[0,0])





# agent = Agent

'''TESTING NETWORK'''
# for i in range(0, 15):
#     observation_, reward, done, info = env.step([0.0, 1.0, 0.0])
#     env.render()
#
# qnet = Qnetwork(len(discreteActions), 0.01)
#
#
# print(observation_)
# qnet.forward(observation_[0:80, 0:96])

''' some things'''
#  96 x 96
# print("input image is: " + str(len(observation_)) + " x " + str(len(observation_[0])))

# for _ in range(1000):
#     time.sleep(10)
#     observation_, reward, done, info = env.step(act)
#     print

    #env.step(env.action_space.sample()) # take a random action
