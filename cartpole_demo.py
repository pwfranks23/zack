"""Interactive CartPole demo with click-based disturbances."""

import gymnasium as gym
import numpy as np
import pygame
import torch

import model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/model_final.pt"
MAX_ANGLE_KICK = 0.03
MAX_ANGULAR_VEL_KICK = 1.25
MAX_CART_VEL_KICK = 0.15


def load_model(model_path: str) -> model.PolicyNetwork:
  """Load a trained policy network.

  Args:
    model_path: Path to the saved model checkpoint.

  Returns:
    Loaded policy network in eval mode.
  """
  policy = model.PolicyNetwork().to(DEVICE)
  policy.load_state_dict(torch.load(model_path, map_location=DEVICE))
  policy.eval()
  return policy


def choose_action(
    policy: model.PolicyNetwork,
    observation: np.ndarray,
) -> int:
  """Choose greedy action from the policy.

  Args:
    policy: Policy network.
    observation: Current environment observation.

  Returns:
    Greedy discrete action.
  """
  state_tensor = torch.as_tensor(
      observation,
      dtype=torch.float32,
      device=DEVICE,
  ).unsqueeze(0)
  with torch.no_grad():
    logits = policy(state_tensor)
    probs = torch.softmax(logits, dim=-1)
    action = probs.argmax(dim=-1).item()
  return action


def apply_disturbance(
    environment: gym.Env,
    angle_kick: float,
    angular_vel_kick: float,
    cart_vel_kick: float,
) -> tuple[float, float, float]:
  """Apply a disturbance by editing the simulator state.

  Args:
    environment: CartPole environment.
    angle_kick: Additive kick for pole angle.
    angular_vel_kick: Additive kick for pole angular velocity.
    cart_vel_kick: Additive kick for cart velocity.

  Returns:
    Applied disturbance tuple.
  """
  cart_pos, cart_vel, pole_angle, pole_ang_vel = environment.unwrapped.state
  environment.unwrapped.state = np.array(
      [
          cart_pos,
          cart_vel + cart_vel_kick,
          pole_angle + angle_kick,
          pole_ang_vel + angular_vel_kick,
      ],
      dtype=np.float32,
  )
  return angle_kick, angular_vel_kick, cart_vel_kick


def apply_click_disturbance(
    environment: gym.Env,
    click_x: int,
) -> tuple[float, float, float]:
  """Apply directional disturbance based on click x-position.

  Args:
    environment: CartPole environment.
    click_x: Horizontal pixel coordinate of click.

  Returns:
    Disturbance tuple that was applied.
  """
  screen_width = getattr(environment.unwrapped, "screen_width", 600)
  center_x = screen_width / 2.0
  normalized = (click_x - center_x) / center_x
  normalized = float(np.clip(normalized, -1.0, 1.0))

  angle_kick = 0.0  # Only kick velocity
  angular_vel_kick = normalized * MAX_ANGULAR_VEL_KICK
  cart_vel_kick = 0.0  # Only kick the pole, not the cart
  return apply_disturbance(
      environment,
      angle_kick=angle_kick,
      angular_vel_kick=angular_vel_kick,
      cart_vel_kick=cart_vel_kick,
  )


def handle_mouse_interactions(environment: gym.Env) -> bool:
  """Handle pygame input and apply click disturbance.

  Args:
    environment: CartPole environment.

  Returns:
    True if user requested quit, else False.
  """
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      return True
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
      angle_kick, angular_vel_kick, cart_vel_kick = apply_click_disturbance(
          environment,
          click_x=event.pos[0],
      )
      print(
          "Click disturbance: "
          f"d_theta={angle_kick:+.4f}, "
          f"d_theta_dot={angular_vel_kick:+.4f}, "
          f"d_x_dot={cart_vel_kick:+.4f}"
      )
  return False


def describe_termination_cause(environment: gym.Env) -> str:
  """Describe why the CartPole episode terminated.

  Args:
    environment: CartPole environment.

  Returns:
    Human-readable description of threshold violation.
  """
  cart_pos, cart_vel, pole_angle, pole_ang_vel = environment.unwrapped.state
  x_threshold = environment.unwrapped.x_threshold
  theta_threshold = environment.unwrapped.theta_threshold_radians

  cart_out_of_bounds = abs(cart_pos) > x_threshold
  pole_out_of_bounds = abs(pole_angle) > theta_threshold

  if cart_out_of_bounds and pole_out_of_bounds:
    return (
        "cart and pole thresholds exceeded "
        f"(|x|={abs(cart_pos):.3f}>{x_threshold:.3f}, "
        f"|theta|={abs(pole_angle):.3f}>{theta_threshold:.3f})"
    )
  if cart_out_of_bounds:
    return (
        "cart threshold exceeded "
        f"(|x|={abs(cart_pos):.3f}>{x_threshold:.3f})"
    )
  if pole_out_of_bounds:
    return (
        "pole angle threshold exceeded "
        f"(|theta|={abs(pole_angle):.3f}>{theta_threshold:.3f})"
    )
  return (
      "terminated without direct threshold match "
      f"(x={cart_pos:.3f}, x_dot={cart_vel:.3f}, "
      f"theta={pole_angle:.3f}, theta_dot={pole_ang_vel:.3f})"
  )


def main() -> None:
  """Run interactive CartPole demo."""
  environment = gym.make(
      "CartPole-v1",
      render_mode="human",
      max_episode_steps=10000)
  policy = load_model(MODEL_PATH)

  observation, info = environment.reset()
  print(f"Model loaded from {MODEL_PATH}")
  print(f"Starting observation: {observation}")
  print("Left-click in the render window to inject a disturbance.")

  user_quit = False
  episode_idx = 1
  total_reward = 0.0

  while not user_quit:
    action = choose_action(policy, observation)
    observation, reward, terminated, truncated, info = environment.step(action)

    user_quit = handle_mouse_interactions(environment)

    total_reward += reward
    if terminated:
      cause = describe_termination_cause(environment)
      print(f"Episode {episode_idx} terminated: {cause}")
    if terminated or truncated:
      end_reason = "terminated" if terminated else "truncated"
      print(
          f"Episode {episode_idx} finished ({end_reason}). "
          f"Reward: {total_reward:.1f}"
      )
      episode_idx += 1
      total_reward = 0.0
      observation, info = environment.reset()

  print("Demo ended by user quit.")
  environment.close()


if __name__ == "__main__":
  main()
