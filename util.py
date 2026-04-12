"""Utility functions for training."""

import random
import subprocess
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, env: Any = None) -> None:
  """Set random seeds for reproducibility.

  Args:
    seed: Random seed value.
    env: Optional environment to seed.
  """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  if env is not None:
    env.action_space.seed(seed)


def get_git_commit() -> str | None:
  """Get the current git commit hash.

  Returns:
    Commit hash string, or None if not in a git repository.
  """
  try:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
  except (FileNotFoundError, subprocess.CalledProcessError):
    return None
  return result.stdout.strip()
