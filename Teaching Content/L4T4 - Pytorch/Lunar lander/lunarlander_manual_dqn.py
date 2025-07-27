import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from gymnasium.wrappers import RecordVideo

#  DQN Network Definition 
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

#  Replay Buffer 
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

#  Training Loop with Gymnasium RecordVideo 
def train_dqn_with_video(env_id="LunarLander-v3",
                         episodes=500,
                         batch_size=64,
                         gamma=0.99,
                         lr=1e-3,
                         buffer_capacity=100000,
                         min_buffer_size=1000,
                         target_update_freq=1000,
                         epsilon_start=1.0,
                         epsilon_end=0.01,
                         epsilon_decay=0.995,
                         video_folder="videos"):
    # Create environment and wrap for video
    env = gym.make(env_id, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda ep: ep % 50 == 0,
        name_prefix="dqn_lander"
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize networks and optimizer
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    epsilon = epsilon_start
    total_steps = 0

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            total_steps += 1
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = policy_net(state_tensor).argmax().item()

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            # Store transition
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            # Train if buffer has enough samples
            if len(replay_buffer) >= min_buffer_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states_tensor = torch.FloatTensor(states)
                actions_tensor = torch.LongTensor(actions).unsqueeze(1)
                rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
                next_states_tensor = torch.FloatTensor(next_states)
                dones_tensor = torch.FloatTensor(dones).unsqueeze(1)

                # Compute current Q values
                q_values = policy_net(states_tensor).gather(1, actions_tensor)

                # Compute target Q values
                with torch.no_grad():
                    next_q_values = target_net(next_states_tensor).max(1)[0].unsqueeze(1)
                    target_q = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)

                # Optimize
                loss = nn.MSELoss()(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update target network
                if total_steps % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        # Decay epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_end)
        print(f"Episode {ep+1}/{episodes} | Reward: {ep_reward:.2f} | Epsilon: {epsilon:.3f}")

    env.close()

#  Run Training
if __name__ == "__main__":
    train_dqn_with_video()
