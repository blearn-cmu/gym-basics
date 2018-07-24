import sys
import gym
import numpy as np
import random
import matplotlib.pyplot as plt

# Initialize the Taxi Environment
env = gym.make("Taxi-v2").env

# Initialize the Q-Table
q_table = np.zeros([env.observation_space.n, env.action_space.n]) # (500, 6) matrix

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# Training function using Q-Learning
'''
Arguments:
    num_episdoes - Number of episodes to train for
    alpha - alpha value used
    gamma - gamma value used
    epsilon - epsilon value used
'''
def train(num_episodes, alpha, gamma, epsilon, verbose=False):
    if verbose:
        print("Training for {} episodes".format(num_episodes))

    for i in range(1, num_episodes+1):
        state = env.reset()

        epochs, penalties, reward = 0, 0, 0
        done = False

        while not done:  # Run until episode is solved

            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values
    
            next_state, reward, done, info = env.step(action) 
            
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value
    
            if reward == -10:
                penalties += 1
    
            state = next_state
            epochs += 1

        if i % 100 == 0:
            sys.stdout.write("\rEpisode: {}".format(i))
            sys.stdout.flush()

    if verbose:
        print("\nTraining finished\n")

# Evaluation function
'''
Arguments:
    num_episodes - Number of episodes to run the simulation
Returns:
    all_epochs - A list of the number of timesteps it took to solve one episode, one value for each episode
    all_penalties - A list of the total penalties received per episode, one value for each episode
'''
def evaluate(num_episodes, verbose=False):
    all_epochs = []
    all_penalties = []

    if verbose:
        print("Evaluating on {} episodes".format(num_episodes))
    for i in range(num_episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
        
        done = False
        
        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)
    
            if reward == -10:
                penalties += 1
    
            epochs += 1
            if epochs >= 5000: # Cap an episode at 5k timesteps
                break

        sys.stdout.write("\rEpisode: {}".format(i))
        sys.stdout.flush()
   
        all_epochs.append(epochs)
        all_penalties.append(penalties)

    if verbose:
        print("\nEvaluating finished\n")
        print("Mean timesteps: {}".format(np.array(all_epochs).mean()))
        print("Mean penalties: {}".format(np.array(all_penalties).mean()))
    return all_epochs, all_penalties


def plot_timesteps_histogram(epochs_list):
    fig = plt.figure()
    plt.hist(epochs_list)
    fig.suptitle('Histogram of Timesteps for Random Search', fontsize=20)
    plt.xlabel('Episodes required to solve', fontsize=18)
    plt.ylabel('Frequency', fontsize=16)
    plt.show()

def plot_penalties_histogram(penalties_list):
    fig = plt.figure()
    plt.hist(penalties_list)
    fig.suptitle('Histogram of Penalties for Random Search', fontsize=20)
    plt.xlabel('Episodes required to solve', fontsize=18)
    plt.ylabel('Frequency', fontsize=16)
    plt.show()

train(100000, alpha, gamma, epsilon, True)
times, penalties = evaluate(100, True)
plot_timesteps_histogram(times)
plot_penalties_histogram(penalties)
