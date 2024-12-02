import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder, VecTransposeImage
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.evaluation import evaluate_policy

env_str = "CarRacing-v3"
log_dir = "./logs/{}".format(env_str)
video_dir = "./videos/"
os.makedirs(video_dir, exist_ok=True)
gray_scale = True

# If gray_scale True, convert obs to gray scale 84 x 84 image
wrapper_class = WarpFrame if gray_scale else None

# Create Evaluation CarRacing environment
env = make_vec_env(env_str, n_envs=1, seed=0, wrapper_class=wrapper_class)
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

# Load the best model
best_model_path = os.path.join(log_dir, "best_model.zip")
best_model = PPO.load(best_model_path, env=env)

# Evaluate the model
mean_reward, std_reward = evaluate_policy(best_model, env, n_eval_episodes=20)
print(f"Best Model - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Record video of the best model playing CarRacing
env = VecVideoRecorder(
    env, video_dir,
    video_length=10000,
    record_video_trigger=lambda x: x == 0,
    name_prefix="best_model_car_racing_ppo"
)

obs = env.reset()
for _ in range(10000):
    action, _states = best_model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        break

env.close()

# Load evaluation results
data = np.load(os.path.join(log_dir, "evaluations.npz"))

# Extract the relevant data
timesteps = data['timesteps']
results = data['results']

# Calculate the mean and standard deviation of the results
mean_results = np.mean(results, axis=1)
std_results = np.std(results, axis=1)

# Plot the results
plt.figure()
plt.plot(timesteps, mean_results)
plt.fill_between(
    timesteps,
    mean_results - std_results,
    mean_results + std_results,
    alpha=0.3
)

plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.title(f"PPO Performance on {env_str}")
plt.show()
