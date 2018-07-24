import pickle
import gym
import numpy as np

env = gym.make("Taxi-v2").env

# Start by loading in a learned Q-Table from training
q_table = pickle.load(open("q_table.p", "rb"))

def run(num_episodes):
    for i in range(num_episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
        
        done = False
        
        while not done:
            env.render()  # NOTE: Rendering this environment in a Windows console usualy doesn't print nicely
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)
    
            if reward == -10:
                penalties += 1
    
            epochs += 1
            if epochs >= 5000: # Cap an episode at 5k timesteps
                break
run(1) # run one episode
