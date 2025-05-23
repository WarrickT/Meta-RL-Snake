from env.snake_env import SnakeEnv
import numpy as np
import time

env = SnakeEnv(grid_size=10, variant="classic")
state = env.reset()

done = False
total_reward = 0

for step in range(100):
    action = np.random.choice(4)
    state, reward, done, info = env.step(action)
    env.render()
    total_reward += reward

    if done:
        print(f"Game Over at step {step}, Total Reward: {total_reward}")
        break

time.sleep(2)
