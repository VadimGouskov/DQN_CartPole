import gym
from cartpole.qnetwork import Qnetwork
from cartpole.agent import Agent
import numpy as np
import time
from cartpole.utils import saveImage, showImage, standardPreprocess, rgb2gray
import cartpole.utils as u
import torch
import matplotlib.pyplot as plot


np.set_printoptions(threshold=np.nan)
RENDER = False

scores = []
avarageScores = []
score = 0

randomActionHistory = []
randomActions = 0

epsilonHistory = []
numberOfGames = 500
batchSize = 1


env = gym.make('CartPole-v0')
env.reset()
actionSpace = env.action_space
observation_, reward, done, info   = env.step(actionSpace.sample())
action = actionSpace.sample()

print("cuda available: ", torch.cuda.is_available())




agent = Agent(  gamma=0.95,
                epsilon = 1.0,
                epsilonDecay=0.995,
                epsilonMin=0.1,
                alpha = 0.001,
                maxMemSize=900,
                targetReplaceCount=50,
                actionSpace=actionSpace,
              )

def processFrame(observation, action):

    observation_, reward, done, info = env.step(action)

    reward = u.negativeRewardOnTerminal(done, reward)

    observation_ = observation_[:].astype(np.float32)
    agent.storeTransition(observation, action, reward, observation_)

    return observation_, reward, done

# INITIALIZE AGENT MEMORY TODO: is this still necessary & isn't this already handled in agent.storeTransition?
while agent.memCounter < agent.maxMemSize:
    # print(agent.memCounter)
    observation = env.reset()
    observation = observation[:].astype(np.float32)
    done = False
    initFrameCounter = 0

    while not done:
        #choose random action
        # step and convert image to greyscale
        # TODO LEFTOFF save rewards in buffer after the episode and apply future reward discount! (also do this during training)
        #
        action = actionSpace.sample()
        observation_, reward, done = processFrame(observation, action)

        observation = observation_
print("memory initialized")




for i in range(numberOfGames):
    print("starting game ", str(i), " w/ epsilon ", agent.epsilon )
    done = False
    frameCounter = 0
    observation = env.reset()

    observation, reward, done, info = env.step(actionSpace.sample())
    observation = observation.astype(np.float32)

    while not done:

        # Use the chosen action as an index to choose the real action according to observation
        action, isRandom = agent.chooseAction(observation)
        randomActions += isRandom

        observation_, reward, done = processFrame(observation, action)

        agent.learn(batchSize)

        observation = observation_

        score += reward

        if RENDER:
            env.render()

        #for now now maximum render 200 before next episode frames
        # TODO cartpole resets after 100?
        frameCounter += 1
        if(frameCounter > 200):
            done = True

    # GENERATE EVALUATION DATA
    scores.append(score)
    avarageScores.append( sum(scores) / len(scores))
    print("episode score = ", score)
    randomActionRate = randomActions/frameCounter * 100
    randomActionHistory.append(randomActionRate)
    epsilonHistory.append(agent.epsilon)


    randomActions = 0
    frameCounter = 0
    score = 0

# env.close()
# plot.plot(scores, 'b', epsilonHistory, 'r')
# plot.show()

env.close()

t = np.arange(0, numberOfGames, 1)

fig, ax1 = plot.subplots()

color = 'tab:blue'
ax1.set_xlabel('episode')
ax1.set_ylabel('scores', color=color)
ax1.plot(t, scores, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# color = 'tab:red'
# ax2.set_ylabel('Random Action Rate', color=color)  # we already handled the x-label with ax1
# ax2.plot(t, randomActionHistory, color=color)
# ax2.tick_params(axis='y', labelcolor=color)

ax3 = ax1.twinx()
color = 'tab:red'
ax3.plot(t, avarageScores, color=color)
ax3.set_ylabel('Avarage score', color=color)  # we already handled the x-label with ax1
ax3.tick_params(axis='y', labelcolor=color)

# ax2.tick_params(axis='y', labelcolor=color)



fig.tight_layout()  # otherwise the right y-label is slightly clipped
plot.show()

exit()
