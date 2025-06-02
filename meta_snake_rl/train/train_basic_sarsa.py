import random
import numpy as np
from meta_snake_rl.env.snake_env import SnakeEnv, select_action, get_state
from meta_snake_rl.utils.constants import ACTIONS, OPPOSITE
import matplotlib.pyplot as plt
import time

# Q Table 
Q = {}

# Basic MC Algorithm Parameters
alpha = 0.1
gamma = 0.95
epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.9985
episodes = 5000
episode_rewards = []

# Create environment
env = SnakeEnv(grid_size=15, variant="classic")

# Create Episode Loop
for episode in range(episodes):
    total_reward = 0
    env.reset()
    state = get_state(env.snake, env.apple, env.direction, env.grid_size)
    direction = env.direction
    done = False
    action = select_action(Q, state, epsilon, direction)

    # Game Loop
    while not done: 
        _, reward, done, _ = env.step(action)

        if episode % 100 == 0: 
            env.render()
            time.sleep(0.01)

        next_state = get_state(env.snake, env.apple, env.direction, env.grid_size)
        next_direction = env.direction
        next_action = select_action(Q, next_state, epsilon, next_direction)

        if state not in Q:
            Q[state] = {a: 0 for a in ACTIONS}
        if next_state not in Q:
            Q[next_state] = {a: 0 for a in ACTIONS}
        
        old_q = Q[state][action]
        new_q = Q[next_state][next_action]
    
        # Update rule of SARSA
        Q[state][action] += alpha * (reward + gamma*new_q - old_q)

        # Move forward
        state = next_state
        action = next_action
        direction = next_direction
        total_reward += reward

    # Do this for plotting 
    episode_rewards.append(total_reward)

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon* decay_rate)

    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")
    

plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("SARSA Training Progress")
plt.ioff()
plt.show()



# Game Loop