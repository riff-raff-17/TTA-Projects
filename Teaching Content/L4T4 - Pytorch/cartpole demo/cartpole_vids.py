import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym


# 1. Constants and directories

ENV_NAME        = "CartPole-v1"
NUM_EPISODES    = 500            # Total episodes to train
SAVE_INTERVAL   = 50             # Save a checkpoint every 50 episodes
CKPT_DIR        = "checkpoints"  # Where checkpoints (.pth files) will go
VIDEO_POST_DIR  = "videos_post"  # Where post-training mp4s will go

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(VIDEO_POST_DIR, exist_ok=True)

# 2. Policy net

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1     = nn.Linear(state_dim, hidden_dim)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

# 3. Action selection and return computation

def select_action(state, policy_net):
    state_tensor = torch.from_numpy(state).float()
    probs        = policy_net(state_tensor)
    dist         = Categorical(probs)
    action       = dist.sample()
    return action.item(), dist.log_prob(action)


def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    # Normalize for stability:
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    return returns


# 4. Training loop

def train_and_save():
    env = gym.make(ENV_NAME)

    state_dim  = env.observation_space.shape[0]  
    action_dim = env.action_space.n             
    hidden_dim = 128
    learning_rate = 1e-2
    gamma = 0.99

    policy_net = PolicyNetwork(state_dim, hidden_dim, action_dim)
    optimizer  = optim.Adam(policy_net.parameters(), lr=learning_rate)

    episode_rewards = []

    for episode in range(1, NUM_EPISODES + 1):
        # Gym ≥ 0.26: reset() returns (obs, info)
        state, _ = env.reset()
        log_probs = []
        rewards   = []
        done      = False

        while not done:
            action, log_prob = select_action(state, policy_net)
            # Gym ≥ 0.26: step() returns (next_state, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        returns = compute_returns(rewards, gamma)
        loss    = torch.stack([-lp * R for lp, R in zip(log_probs, returns)]).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        # Save checkpoint every SAVE_INTERVAL episodes
        if episode % SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(CKPT_DIR, f"policy_ep{episode:04d}.pth")
            torch.save(policy_net.state_dict(), ckpt_path)
            avg_reward = sum(episode_rewards[-SAVE_INTERVAL:]) / SAVE_INTERVAL
            print(f"[Episode {episode:04d}] AvgReward (last {SAVE_INTERVAL} eps) = {avg_reward:.2f} → Saved: {ckpt_path}")

    env.close()
    return episode_rewards


# 5. Post training
# Loads checkpoints and records a video for each

def load_policy(path, state_dim, hidden_dim, action_dim):
    model = PolicyNetwork(state_dim, hidden_dim, action_dim)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def record_from_checkpoints():
    from gym.wrappers import RecordVideo

    ckpt_files = sorted(f for f in os.listdir(CKPT_DIR) if f.endswith(".pth"))
    if not ckpt_files:
        print("No checkpoints found in:", CKPT_DIR)
        return

    dummy_env = gym.make(ENV_NAME)
    state_dim  = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n
    hidden_dim = 128  # must match what was used during training
    dummy_env.close()

    for filename in ckpt_files:
        ep_num    = filename.split("policy_ep")[1].split(".pth")[0]
        ckpt_path = os.path.join(CKPT_DIR, filename)
        policy_net = load_policy(ckpt_path, state_dim, hidden_dim, action_dim)
        

        # Create a fresh environment with render_mode='rgb_array'
        # This just runs the checkpoint in a new env
        # lol
        env_rec = RecordVideo(
            gym.make(ENV_NAME, render_mode="rgb_array"),
            video_folder=VIDEO_POST_DIR,
            episode_trigger=lambda ep: True,
            name_prefix=f"post_ep{ep_num}"
        )

        state, _ = env_rec.reset()
        done     = False

        while not done:
            state_tensor = torch.from_numpy(state).float()
            with torch.no_grad():
                probs = policy_net(state_tensor)
            action = torch.argmax(probs).item()

            next_state, reward, terminated, truncated, _ = env_rec.step(action)
            done  = terminated or truncated
            state = next_state

        env_rec.close()
        print(f"Recorded video for checkpoint ep{ep_num} → saved in: {VIDEO_POST_DIR}")

# 6. Running everything

if __name__ == "__main__":
    print("=== TRAINING PHASE ===")
    episode_rewards = train_and_save()
    print("\nTraining complete. Checkpoints are in:", CKPT_DIR)

    print("\n=== POST-TRAINING RECORDING ===")
    record_from_checkpoints()
    print("Post-training videos saved in:", VIDEO_POST_DIR)