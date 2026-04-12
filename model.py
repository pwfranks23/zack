import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
  """Simple configurable policy network for discrete-action control."""

  def __init__(self, input_size=4, hidden_size=128, output_size=2):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )

  def forward(self, state):
    """
    Args:
      state: torch tensor of shape (batch_size, input_size) or (input_size,)
    Returns:
      logits: torch tensor of shape (batch_size, output_size) or (output_size,)
    """
    return self.net(state)

  def select_action(self, state):
    """
    Select an action given a state.
    Args:
      state: numpy array of shape (input_size,)
    Returns:
      action: int in [0, output_size - 1]
        log_prob: torch scalar
    """
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    logits = self.forward(state_tensor)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob
