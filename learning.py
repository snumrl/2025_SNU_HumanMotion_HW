from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from custom_walker2d import CustomEnvWrapper
from stable_baselines3.common.callbacks import CheckpointCallback

N_ENVS = 4

def make_env(bumps = False):
    def _init():
        env = CustomEnvWrapper(render_mode=None, bumps=bumps)
        return env
    return _init

policy_kwargs = dict(
    net_arch=[dict(pi=[128, 64, 64], vf=[128, 64, 64])],
    log_std_init=-1.0 
)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,  
    save_path='./checkpoints/',
    name_prefix='walker_model'
)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--vel", action="store_true", help="Enable bumping") # For 1-2
parser.add_argument("--bump", action="store_true", help="Enable bumping") # For Extra

args = parser.parse_args()
if __name__ == "__main__":
    num_cpu = N_ENVS
    env = SubprocVecEnv([make_env(bumps = args.bump) for _ in range(num_cpu)])
    env = VecMonitor(env)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/", policy_kwargs=policy_kwargs, device="cpu", learning_rate=0.0001)
    model.learn(total_timesteps=10000000000, callback=checkpoint_callback)
    model.save("ppo_custom_walker2d_parallel")