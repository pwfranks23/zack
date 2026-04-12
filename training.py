"""Training loop with experiment tracking."""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mlflow
import numpy as np
import torch
import torch.optim as optim

import model
import util

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainingConfig:
  learning_rate: float = 0.001
  gamma: float = 0.99
  num_episodes: int = 500
  max_steps: int = 500
  seed: int = 0
  checkpoint_dir: str = "checkpoints"
  checkpoint_interval: int = 100
  log_interval: int = 50
  eval_interval: int = 50
  eval_episodes: int = 10
  experiment_name: str = "cartpole-local"


def make_env() -> gym.Env:
  """Create gym environment"""
  return gym.make("CartPole-v1")


def make_model() -> model.PolicyNetwork:
  """Create policy network on configured device."""
  return model.PolicyNetwork().to(DEVICE)


def make_optimizer(
    model: torch.nn.Module,
    config: TrainingConfig,
) -> optim.Adam:
  """Create optimizer for model"""
  return optim.Adam(model.parameters(), lr=config.learning_rate)


def evaluate_policy(
    model: torch.nn.Module,
    config: TrainingConfig,
) -> dict[str, float]:
  """Evaluate policy with greedy action selection.

  Args:
    model: Policy network to evaluate.
    config: Configuration with eval parameters.

  Returns:
    Dictionary with eval_mean_reward, eval_std_reward, etc.
  """
  eval_env = make_env()
  eval_env.action_space.seed(config.seed + 10_000)
  eval_rewards = []
  model_was_training = model.training
  model.eval()

  for episode in range(config.eval_episodes):
    observation, info = eval_env.reset(seed=config.seed + 10_000 + episode)
    total_reward = 0.0

    for step in range(config.max_steps):
      action = model.select_greedy_action(observation)
      observation, reward, terminated, truncated, info = eval_env.step(action)
      total_reward += reward

      if terminated or truncated:
        break

    eval_rewards.append(total_reward)

  eval_env.close()
  if model_was_training:
    model.train()

  rewards_array = np.array(eval_rewards, dtype=np.float32)
  return {
      "eval_mean_reward": float(rewards_array.mean()),
      "eval_std_reward": float(rewards_array.std()),
      "eval_min_reward": float(rewards_array.min()),
      "eval_max_reward": float(rewards_array.max()),
  }


def log_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
  """Save model checkpoint and log to MLflow.

  Args:
    model: Model to save.
    checkpoint_path: Path to save checkpoint file.
  """
  torch.save(model.state_dict(), checkpoint_path)
  mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
  print(f"Saved checkpoint to {checkpoint_path}")


def train(
    env: gym.Env,
    model: torch.nn.Module,
    optimizer: optim.Adam,
    config: TrainingConfig,
) -> None:
  """Train policy with REINFORCE and log metrics to MLflow.

  Args:
    env: Training environment.
    model: Policy network to train.
    optimizer: Optimizer for model parameters.
    config: Training configuration.
  """
  tracking_dir = Path("mlruns").resolve()
  tracking_dir.mkdir(parents=True, exist_ok=True)
  mlflow.set_tracking_uri(tracking_dir.as_uri())
  mlflow.set_experiment(config.experiment_name)
  os.makedirs(config.checkpoint_dir, exist_ok=True)

  git_commit = util.get_git_commit()
  run_name = (
      f"lr={config.learning_rate}_gamma={config.gamma}_seed={config.seed}"
  )

  episode_rewards = []
  best_eval_mean_reward = float("-inf")
  best_running_avg_reward = float("-inf")
  start_time = time.time()

  with mlflow.start_run(run_name=run_name):
    mlflow.log_params({
        "learning_rate": config.learning_rate,
        "gamma": config.gamma,
        "num_episodes": config.num_episodes,
        "max_steps": config.max_steps,
        "seed": config.seed,
        "checkpoint_interval": config.checkpoint_interval,
        "log_interval": config.log_interval,
        "eval_interval": config.eval_interval,
        "eval_episodes": config.eval_episodes,
        "device": str(DEVICE),
    })
    if git_commit is not None:
      mlflow.set_tag("git_commit", git_commit)

    run_id = mlflow.active_run().info.run_id
    run_checkpoint_dir = Path(config.checkpoint_dir) / run_id
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for episode in range(config.num_episodes):
      observation, info = env.reset(seed=config.seed + episode)
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
      returns = (returns - returns.mean()) / (
          returns.std(unbiased=False) + 1e-8
      )

      # Policy gradient update
      loss = torch.tensor(0.0, device=DEVICE)
      for log_prob, ret in zip(log_probs, returns):
        loss = loss - log_prob * ret

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      episode_reward = float(sum(rewards))
      episode_rewards.append(episode_reward)
      running_avg_reward = float(
          np.mean(episode_rewards[-config.log_interval:])
      )
      best_running_avg_reward = max(
          best_running_avg_reward, running_avg_reward)

      mlflow.log_metrics(
          {
              "train_reward": episode_reward,
              "running_avg_reward": running_avg_reward,
              "loss": float(loss.item()),
              "episode_length": len(rewards),
              "wall_time_sec": float(time.time() - start_time),
          },
          step=episode + 1,
      )

      if (episode + 1) % config.log_interval == 0:
        print(
            f"Episode {episode + 1}/{config.num_episodes}, "
            f"Avg Reward (last {config.log_interval}): "
            f"{running_avg_reward:.2f}, "
            f"Last Episode Reward: {episode_reward}"
        )

      if config.eval_interval > 0 and (
              episode + 1) % config.eval_interval == 0:
        eval_metrics = evaluate_policy(model, config)
        mlflow.log_metrics(eval_metrics, step=episode + 1)
        best_eval_mean_reward = max(
            best_eval_mean_reward, eval_metrics["eval_mean_reward"]
        )
        print(
            f"Eval after episode {episode + 1}: "
            f"mean={eval_metrics['eval_mean_reward']:.2f}, "
            f"std={eval_metrics['eval_std_reward']:.2f}"
        )

      if (episode + 1) % config.checkpoint_interval == 0:
        checkpoint_path = run_checkpoint_dir / f"model_ep{episode + 1}.pt"
        log_checkpoint(model, str(checkpoint_path))

    # Save final model
    final_path = run_checkpoint_dir / "model_final.pt"
    log_checkpoint(model, str(final_path))
    summary = {
      "best_train_reward": float(max(episode_rewards)),
      "final_train_reward": float(episode_rewards[-1]),
      "best_running_avg_reward": float(best_running_avg_reward),
      "best_eval_mean_reward": (
        best_eval_mean_reward
        if best_eval_mean_reward != float("-inf")
        else None
      ),
      "tracking_uri": tracking_dir.as_uri(),
      "run_id": run_id,
    }
    mlflow.log_metrics({
      key: value for key, value in summary.items()
      if isinstance(value, (int, float))
    })
    summary_path = run_checkpoint_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
      json.dump(summary, summary_file, indent=2)
    mlflow.log_artifact(str(summary_path), artifact_path="reports")
    print(f"\nTraining complete! Final model saved to {final_path}")

  env.close()


def main() -> None:
  """Run training with default configuration."""
  config = TrainingConfig()
  env = make_env()
  util.set_seed(config.seed, env)
  model = make_model()
  optimizer = make_optimizer(model, config)
  train(env, model, optimizer, config)


if __name__ == "__main__":
  main()
