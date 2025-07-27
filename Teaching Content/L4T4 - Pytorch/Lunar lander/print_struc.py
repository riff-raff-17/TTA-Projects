import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


# 1. Create the environment
def make_env():
    env = gym.make("LunarLander-v3")
    return Monitor(env)

env = DummyVecEnv([make_env])

# 2. Load or create the PPO model
model = PPO("MlpPolicy", env, verbose=0)

# 3. Print full architecture
print("========== PPO Policy Architecture ==========")
print(model.policy)

print("\n========== MLP Extractor (Shared Layers) ==========")
print(model.policy.mlp_extractor)

print("\n========== Actor (Policy Network) ==========")
print(model.policy.action_net)

print("\n========== Critic (Value Network) ==========")
print(model.policy.value_net)

print("\n Model structure printed successfully.")
