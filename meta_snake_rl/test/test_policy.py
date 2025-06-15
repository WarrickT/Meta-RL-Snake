import numpy as np
from meta_snake_rl.env.snake_env import SnakeEnv, select_action, get_state

# One problem is looping infinitely. 

def test_policy(Q, variant, episodes=100):
    rewards = []

    for ep in range(episodes):
        print("Starting episode # ", (ep + 1), " for ", variant)
        env = SnakeEnv(grid_size=15, variant=variant)
        env.reset()
        state = get_state(env.snake, env.apple, env.direction, env.grid_size, env.variant)

        steps = 0

        direction = env.direction
        done = False 
        total_reward = 0

        while not done: 
            action = select_action(Q, state, epsilon=0.0, direction=direction)
            _, reward, done, _ = env.step(action)
            steps += 1

            state = get_state(env.snake, env.apple, env.direction, env.grid_size, env.variant)
            direction = env.direction
            total_reward += reward

            if steps == 1000: 
                print("Okay why are are we doing this?")
                print("Infinite loop, breaking!")
                total_reward = 0
                break


        rewards.append(total_reward)
        print("Done episode # ", (ep + 1), " with reward ", total_reward, " for ", variant)

    print(f"Avg reward on {variant}: {np.mean(rewards):.2f}")
