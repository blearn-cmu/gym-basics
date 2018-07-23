import gym
import scipy
import numpy as np

env = gym.make("CartPole-v0")

for episode in range(10):  
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    print("Reward for Episode {}: {}".format(episode, totalreward))

env.close()
