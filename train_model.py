import gymnasium
import os
import numpy
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.atari_wrappers import WarpFrame

env_str = "CarRacing-v3"
log_dir = "./logs/{}".format(env_str)
os.makedirs(log_dir, exist_ok=True)
gray_scale = True

# If gray_scale True, convert obs to gray scale 84 x 84 image
wrapper_class = WarpFrame if gray_scale else None

# Create Training CarRacing environment
env = make_vec_env(env_str, n_envs=4, wrapper_class=wrapper_class)
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

# Create Evaluation CarRacing environment
env_val = make_vec_env(env_str, n_envs=4, wrapper_class=wrapper_class)
env_val = VecFrameStack(env_val, n_stack=4)
env_val = VecTransposeImage(env_val)

# Create Evaluation Callback
eval_callback = EvalCallback(
    env_val,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=25_000,
    render=False,
    n_eval_episodes=20
)

# Initialize SAC
model = SAC('CnnPolicy', env, verbose=0, buffer_size=50_000, device='cuda')

# Train the model
model.learn(
    total_timesteps=750_000,
    progress_bar=True,
    callback=[eval_callback]
)

# Save the model
model.save(os.path.join(log_dir, "sac_car_racing"))

env.close()
env_val.close()
