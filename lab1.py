# Task 1: Environment Setup

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

print("Gymnasium Version:", gym.__version__)


# Task 2: Create and Initialize an RL Environment

env = gym.make("CartPole-v1")

observation, info = env.reset()

print("\nInitial Observation:")
print(observation)

print("\nEnvironment Information:")
print(info)


# Task 3: Explore Observation and Action Spaces

print("\nObservation Space:", env.observation_space)
print("Action Space:", env.action_space)
print("Type of Observation Space:", type(env.observation_space))
print("Number of Possible Actions:", env.action_space.n)


# Task 4: Execute a Random Agent

observation, info = env.reset()

done = False
step = 0
total_reward = 0

while not done:
    action = env.action_space.sample()      # Random action

    observation, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    step += 1
    total_reward += reward

    print(f"\nStep {step}")
    print("Action:", action)
    print("Observation:", observation)
    print("Reward:", reward)
    print("Episode Finished:", done)

print("\nEpisode Completed")
print("Total Steps:", step)
print("Cumulative Reward:", total_reward)

env.close()