import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

# Reset environment to start a new episode
observation, info = env.reset()

print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

while not episode_over:
  action = env.action_space.sample()

  observation, reward, terminated, truncated, info = env.step(action)

  # reward: +1 for each step the pole stays upright
  # terminated: True if pole falls too far (agent failed)
  # truncated: True if we hit the time limit (500 steps)

  total_reward += reward
  episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()
