import sys
import os

import random
import numpy as np
import matplotlib.pyplot as plt
from meta_snake_rl.env.snake_env import SnakeEnv, select_action, get_state
from meta_snake_rl.utils.constants import ACTIONS, OPPOSITE


# Add META-RL-SNAKE (project root) to path
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)



# Q-learning Training Loop
def q_learning_loop(env, alpha, gamma, epsilon, min_epsilon, decay_rate, episodes):
    Q = {}
    episode_rewards = []

    for episode in range(episodes):
        total_reward = 0
        env.reset()
        state = get_state(env.snake, env.apple, env.direction, env.grid_size, env.variant)
        direction = env.direction
        done = False

        while not done:
            action = select_action(Q, state, epsilon, direction)
            _, reward, done, _ = env.step(action)
            next_state = get_state(env.snake, env.apple, env.direction, env.grid_size, env.variant)
            next_direction = env.direction

            if state not in Q:
                Q[state] = {a: 0 for a in ACTIONS}
            if next_state not in Q:
                Q[next_state] = {a: 0 for a in ACTIONS}

            old_q = Q[state][action]
            max_next_q = max(Q[next_state].values())

            Q[state][action] += alpha * (reward + gamma * max_next_q - old_q)

            state = next_state
            direction = next_direction
            total_reward += reward

        episode_rewards.append(total_reward)
        epsilon = max(min_epsilon, epsilon * decay_rate)

    return Q, episode_rewards

# SARSA Training Loop
def sarsa_loop(env, alpha, gamma, epsilon, min_epsilon, decay_rate, episodes):
    Q = {}
    episode_rewards = []

    for episode in range(episodes):
        total_reward = 0
        env.reset()
        state = get_state(env.snake, env.apple, env.direction, env.grid_size, env.variant)
        direction = env.direction
        done = False
        action = select_action(Q, state, epsilon, direction)

        while not done:
            _, reward, done, _ = env.step(action)

            next_state = get_state(env.snake, env.apple, env.direction, env.grid_size, env.variant)
            next_direction = env.direction
            next_action = select_action(Q, next_state, epsilon, next_direction)

            if state not in Q:
                Q[state] = {a: 0 for a in ACTIONS}
            if next_state not in Q:
                Q[next_state] = {a: 0 for a in ACTIONS}

            old_q = Q[state][action]
            new_q = Q[next_state][next_action]
            Q[state][action] += alpha * (reward + gamma * new_q - old_q)

            state = next_state
            action = next_action
            direction = next_direction
            total_reward += reward

        episode_rewards.append(total_reward)
        epsilon = max(min_epsilon, epsilon * decay_rate)

    return Q, episode_rewards


def meta_sarsa_loop(variants, alpha, gamma, epsilon, min_epsilon, decay_rate, episodes):
    Q = {}
    episode_rewards = []

    for ep in range(episodes):
        variant = random.choice(variants)
        env = SnakeEnv(grid_size=15, variant=variant)
        env.reset()

        state = get_state(env.snake, env.apple, env.direction, env.grid_size, env.variant)
        direction = env.direction 
        action = select_action(Q, state, epsilon, direction)
        done = False
        total_reward = 0

        while not done:
            _, reward, done, _ = env.step(action)
            
            next_state = get_state(env.snake, env.apple, env.direction, env.grid_size, env.variant)
            next_direction = env.direction
            next_action = select_action(Q, next_state, epsilon, next_direction)

            if state not in Q:
                Q[state] = {a: 0 for a in ACTIONS}
            if next_state not in Q:
                Q[next_state] = {a: 0 for a in ACTIONS}

            old_q = Q[state][action]
            new_q = Q[next_state][next_action]
            Q[state][action] += alpha * (reward + gamma * new_q - old_q)

            state = next_state
            action = next_action
            direction = next_direction
            total_reward += reward

        episode_rewards.append(total_reward)
        epsilon = max(min_epsilon, epsilon * decay_rate)

    return Q, episode_rewards
        
def meta_q_loop(variants, alpha, gamma, epsilon, min_epsilon, decay_rate, episodes):
    Q = {}
    episode_rewards = []

    for ep in range(episodes):
        variant = random.choice(variants)
        env = SnakeEnv(grid_size=15, variant=variant)
        env.reset()

        total_reward = 0
        state = get_state(env.snake, env.apple, env.direction, env.grid_size, env.variant)
        direction = env.direction
        done = False

        while not done:
            action = select_action(Q, state, epsilon, direction)
            _, reward, done, _ = env.step(action)
            next_state = get_state(env.snake, env.apple, env.direction, env.grid_size, env.variant)
            next_direction = env.direction

            if state not in Q:
                Q[state] = {a: 0 for a in ACTIONS}
            if next_state not in Q:
                Q[next_state] = {a: 0 for a in ACTIONS}

            old_q = Q[state][action]
            max_next_q = max(Q[next_state].values())

            Q[state][action] += alpha * (reward + gamma * max_next_q - old_q)

            state = next_state
            direction = next_direction
            total_reward += reward

        episode_rewards.append(total_reward)
        epsilon = max(min_epsilon, epsilon * decay_rate)

    return Q, episode_rewards