import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from model import PolicyNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainingConfig:
  learning_rate: float = 0.001
  gamma: float = 0.99
  num_episodes: int = 500
  max_steps: int = 500
  checkpoint_dir: str = "checkpoints"
  checkpoint_interval: int = 100
  log_interval: int = 50


def make_env():
  return gym.make("CartPole-v1")


def make_model():
  return PolicyNetwork().to(DEVICE)


def make_optimizer(model, config):
  return optim.Adam(model.parameters(), lr=config.learning_rate)


def train(env, model, optimizer, config):
  os.makedirs(config.checkpoint_dir, exist_ok=True)

  episode_rewards = []

  for episode in range(config.num_episodes):
    observation, info = env.reset()
    log_probs = []
    rewards = []

    for step in range(config.max_steps):
      action, log_prob = model.select_action(observation)
      observation, reward, terminated, truncated, info = env.step(action)

      log_probs.append(log_prob)
      rewards.append(reward)

      if terminated or truncated:
        break

    # Compute discounted returns
    returns = []
    cumsum = 0
    for r in reversed(rewards):
      cumsum = r + config.gamma * cumsum
      returns.insert(0, cumsum)

    returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Policy gradient update
    loss = torch.tensor(0.0, device=DEVICE)
    for log_prob, ret in zip(log_probs, returns):
      loss = loss - log_prob * ret

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    episode_reward = sum(rewards)
    episode_rewards.append(episode_reward)

    if (episode + 1) % config.log_interval == 0:
      avg_reward = np.mean(episode_rewards[-config.log_interval:])
      print(
          f"Episode {episode + 1}/{config.num_episodes}, "
          f"Avg Reward (last {config.log_interval}): {avg_reward:.2f}, "
          f"Last Episode Reward: {episode_reward}"
      )

    if (episode + 1) % config.checkpoint_interval == 0:
      checkpoint_path = (
          f"{config.checkpoint_dir}/model_ep{episode + 1}.pt"
      )
      torch.save(model.state_dict(), checkpoint_path)
      print(f"Saved checkpoint to {checkpoint_path}")

  # Save final model
  final_path = f"{config.checkpoint_dir}/model_final.pt"
  torch.save(model.state_dict(), final_path)
  print(f"\nTraining complete! Final model saved to {final_path}")

  env.close()


def main():
  config = TrainingConfig()
  env = make_env()
  model = make_model()
  optimizer = make_optimizer(model, config)
  train(env, model, optimizer, config)


if __name__ == "__main__":
  main()
