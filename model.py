"""Policy network for CartPole control."""

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
  """Simple configurable policy network for discrete-action control."""

  def __init__(
      self,
      input_size: int = 4,
      hidden_size: int = 128,
      output_size: int = 2,
  ) -> None:
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )

  def forward(self, state: torch.Tensor) -> torch.Tensor:
    """Compute policy logits from state.

    Args:
      state: Tensor of shape (batch_size, input_size) or (input_size,).

    Returns:
      Logits tensor of shape (batch_size, output_size) or (output_size,).
    """
    return self.net(state)

  def select_action(self, state) -> tuple:
    """Sample an action from the policy.

    Args:
      state: Observation array of shape (input_size,).

    Returns:
      Tuple of (action, log_prob) where action is int in [0, output_size-1]
      and log_prob is a scalar tensor.
    """
    device = next(self.parameters()).device
    state_tensor = torch.as_tensor(
        state, dtype=torch.float32, device=device
    ).unsqueeze(0)
    logits = self.forward(state_tensor)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob

  def select_greedy_action(self, state) -> int:
    """Select the highest-probability action for evaluation.

    Args:
      state: Observation array of shape (input_size,).

    Returns:
      Action with highest probability.
    """
    device = next(self.parameters()).device
    state_tensor = torch.as_tensor(
        state, dtype=torch.float32, device=device
    ).unsqueeze(0)
    with torch.no_grad():
      logits = self.forward(state_tensor)
      action = torch.argmax(logits, dim=-1)
    return action.item()
