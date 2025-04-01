import argparse
from stable_baselines3 import PPO
from custom_walker2d import CustomEnvWrapper
import gymnasium as gym

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default=None, help="Path to the saved model (.zip)")
parser.add_argument("--vel", action="store_true", help="Enable bumping") # For 1-2
parser.add_argument("--bump", action="store_true", help="Enable bumping") # For Extra

args = parser.parse_args()

env = CustomEnvWrapper(render_mode="human", bumps=args.bump)
model = PPO.load(args.model) if args.model is not None else None
obs, _ = env.reset()

while True:
    if model is not None:
        action, _ = model.predict(obs, deterministic=True)
    else:
        action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()