'''This JUST trains the model. 
If you want the saved videoes, go to cartpole_vids.'''

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# 1. Define the Policy Network
env = gym.make('CartPole-v1')

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

# Hyperparameters
state_dim = env.observation_space.shape[0]   # 4
hidden_dim = 128
action_dim = env.action_space.n             # 2
learning_rate = 1e-2
gamma = 0.99

policy_net = PolicyNetwork(state_dim, hidden_dim, action_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# 2. Function to select action based on policy
def select_action(state):
    state = torch.from_numpy(state).float()
    probs = policy_net(state)
    dist = Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)

# 3. Function to compute discounted returns
def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    # Normalize for stability
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    return returns

# 4. Training loop
def train_policy_gradient(num_episodes=500, print_interval=50):
    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        # Unpack reset()’s tuple into state and info
        state, _ = env.reset()  
        log_probs = []
        rewards = []
        done = False

        while not done:
            action, log_prob = select_action(state)
            # Unpack step()’s tuple into next_state, reward, terminated, truncated, info
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        # Compute returns and policy loss
        returns = compute_returns(rewards, gamma)
        loss = []
        for log_prob, R in zip(log_probs, returns):
            loss.append(-log_prob * R)
        loss = torch.stack(loss).sum()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        if episode % print_interval == 0:
            avg_reward = sum(episode_rewards[-print_interval:]) / print_interval
            print(f"Episode {episode}\tAverage Reward: {avg_reward:.2f}")

    return episode_rewards

# 5. Run training
episode_rewards = train_policy_gradient(num_episodes=500)

# 6. Plot rewards over episodes
def plot_rewards(rewards):
    plt.figure(figsize=(10,5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Reward over Time (REINFORCE)')
    plt.show()

plot_rewards(episode_rewards)
