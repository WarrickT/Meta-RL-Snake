import matplotlib.pyplot as plt

def plot_rewards_progress(episode_rewards):
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.grid(True)
    plt.legend()
    plt.show()