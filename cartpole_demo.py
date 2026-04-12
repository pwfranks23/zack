# python3 -m cartpole_demo
import gymnasium as gym
import numpy as np
import torch

from model import PolicyNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/model_final.pt"

env = gym.make("CartPole-v1", render_mode="human")

DISTURBANCE_EVERY_STEPS = 50
MAX_ANGLE_KICK = 0.03
MAX_ANGULAR_VEL_KICK = 0.25
MAX_CART_VEL_KICK = 0.15


def load_model():
  model = PolicyNetwork().to(DEVICE)
  model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
  model.eval()
  return model


def choose_action(model, observation):
  state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(DEVICE)
  with torch.no_grad():
    logits = model(state_tensor)
    probs = torch.softmax(logits, dim=-1)
    action = probs.argmax(dim=-1).item()
  return action


def apply_disturbance_if_needed(environment, step_number):
  if step_number == 0 or step_number % DISTURBANCE_EVERY_STEPS != 0:
    return None

  cart_pos, cart_vel, pole_angle, pole_ang_vel = environment.unwrapped.state

  angle_kick = np.random.uniform(-MAX_ANGLE_KICK, MAX_ANGLE_KICK)
  angular_vel_kick = np.random.uniform(-MAX_ANGULAR_VEL_KICK,
                                       MAX_ANGULAR_VEL_KICK)
  cart_vel_kick = np.random.uniform(-MAX_CART_VEL_KICK, MAX_CART_VEL_KICK)

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


# Reset environment to start a new episode
model = load_model()
observation, info = env.reset()
print(f'Model loaded from {MODEL_PATH}')

print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0
step = 0

while not episode_over:
  action = choose_action(model, observation)

  observation, reward, terminated, truncated, info = env.step(action)

  disturbance = apply_disturbance_if_needed(env, step)
  if disturbance is not None:
    angle_kick, angular_vel_kick, cart_vel_kick = disturbance
    print(
      "Disturbance applied at "
      f"step {step}: "
      f"d_theta={angle_kick:+.4f}, "
      f"d_theta_dot={angular_vel_kick:+.4f}, "
      f"d_x_dot={cart_vel_kick:+.4f}"
    )

  # reward: +1 for each step the pole stays upright
  # terminated: True if pole falls too far (agent failed)
  # truncated: True if we hit the time limit (500 steps)

  total_reward += reward
  episode_over = terminated or truncated
  step += 1

print(f"Episode finished! Total reward: {total_reward}")
env.close()
