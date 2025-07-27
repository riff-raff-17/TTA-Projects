import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor

# Configuration
ENV_ID = "LunarLander-v3"
TOTAL_TIMESTEPS = 200_000
VIDEO_FREQ = 50000
VIDEO_LENGTH = 1000
LOG_DIR = "./logs"
VIDEO_DIR = os.path.join(LOG_DIR, "videos")
MODEL_PATH = os.path.join(LOG_DIR, "ppo_lunarlander")

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Create environment
def make_env():
    env = gym.make(ENV_ID, render_mode="rgb_array")
    return Monitor(env)

env = DummyVecEnv([make_env])
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join(LOG_DIR, "tb"))

# Training loop with video
for step in range(0, TOTAL_TIMESTEPS, VIDEO_FREQ):
    print(f"Training from step {step} to {step + VIDEO_FREQ}")
    
    # Wrap a fresh env with VecVideoRecorder for this segment
    video_env = VecVideoRecorder(
        DummyVecEnv([make_env]),
        video_folder=VIDEO_DIR,
        record_video_trigger=lambda x: x == 0,
        video_length=VIDEO_LENGTH,
        name_prefix=f"rl-video-{step}"
    )

    # Train using the video environment
    model.set_env(video_env)
    model.learn(total_timesteps=VIDEO_FREQ, reset_num_timesteps=False)
    video_env.close()

# Save final model
model.save(MODEL_PATH)
env.close()
