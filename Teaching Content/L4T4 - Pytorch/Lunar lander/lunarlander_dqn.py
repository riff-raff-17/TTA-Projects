import gymnasium as gym
import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor

# Configuration
ENV_ID = "LunarLander-v3"
TOTAL_TIMESTEPS = 200_001
VIDEO_FREQ = 50000
VIDEO_LENGTH = 1000
LOG_DIR = "./logs"
VIDEO_DIR = os.path.join(LOG_DIR, "videos")
MODEL_PATH = os.path.join(LOG_DIR, "dqn_lunarlander")

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Create environment
def make_env():
    env = gym.make(ENV_ID, render_mode="rgb_array")
    return Monitor(env)

# Base VecEnv
base_env = DummyVecEnv([make_env])

# Instantiate DQN
model = DQN(
    policy="MlpPolicy",
    env=base_env,
    learning_rate=6.3e-4,
    buffer_size=50_000,
    learning_starts=0,
    batch_size=128,
    gamma=0.99,
    train_freq=4,
    gradient_steps=-1,
    target_update_interval=250,
    exploration_fraction=0.12,
    exploration_final_eps=0.1,
    policy_kwargs=dict(net_arch=[256, 256]), 
    tensorboard_log=os.path.join(LOG_DIR, "tb_dqn"),
    verbose=1,
)

# Training loop with video at intervals
for step in range(0, TOTAL_TIMESTEPS, VIDEO_FREQ):
    print(f"Training from step {step} to {step + VIDEO_FREQ}")

# Wrap a fresh env with VecVideoRecorder for this segment
    video_env = VecVideoRecorder(
        DummyVecEnv([make_env]),
        video_folder=VIDEO_DIR,
        record_video_trigger=lambda x: x == 0,
        video_length=VIDEO_LENGTH,
        name_prefix=f"dqn-video-{step}"
    )

    # Train using the video environment
    model.set_env(video_env)
    model.learn(total_timesteps=VIDEO_FREQ, reset_num_timesteps=False)
    video_env.close()

# Save final model
model.save(MODEL_PATH)
base_env.close()
