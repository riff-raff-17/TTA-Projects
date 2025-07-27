import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from torchviz import make_dot

# 1. Create the environment
def make_env():
    env = gym.make("LunarLander-v3")
    return Monitor(env)

env = DummyVecEnv([make_env])

# 2. Load or create the PPO model
model = PPO("MlpPolicy", env, verbose=0)

# 3. Get one observation and convert it to a tensor
obs = env.reset()[0]  # Only take the obs (not info)
obs_tensor = torch.tensor([obs], dtype=torch.float32)

# 4. Forward pass through policy network
# This returns a `Distribution` object with logits inside
dist = model.policy.get_distribution(obs_tensor)

# 5. Create and save computation graph
dot = make_dot(dist.distribution.logits,
               params=dict(model.policy.named_parameters()))
dot.format = "png"
dot.render("ppo_policy")

print("Saved policy visualization to ppo_policy.png")
