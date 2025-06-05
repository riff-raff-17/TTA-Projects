import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from gym.wrappers import RecordVideo

# 1. Hyperparameters and Directories

ENV_NAME      = "CartPole-v1"
NUM_EPISODES  = 500
SAVE_INTERVAL = 50
HIDDEN_DIM    = 128
LEARNING_RATE = 1e-2
GAMMA         = 0.99

CKPT_DIR      = "checkpoints"
VIDEO_DIR     = "videos_post"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)


# 2. Policy Network (with state_dim and action_dim saved as attributes)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # save dims so we can re‐instantiate the same shape later:
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


# 3. Utility Functions

def select_action(state, policy_net):
    state = torch.from_numpy(state).float()
    probs = policy_net(state)
    dist  = Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)


def compute_returns(rewards):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + GAMMA * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    return (returns - returns.mean()) / (returns.std() + 1e-9)


def save_checkpoint(model, episode):
    filename = os.path.join(CKPT_DIR, f"policy_ep{episode:04d}.pth")
    torch.save({
        'state_dict': model.state_dict(),
        'state_dim':  model.state_dim,
        'action_dim': model.action_dim
    }, filename)
    return filename


def load_checkpoint(path):
    ckpt = torch.load(path)
    model = PolicyNetwork(ckpt['state_dim'], ckpt['action_dim'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model

# 4. Training Loop

def train_and_save():
    env = gym.make(ENV_NAME)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model     = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    episode_rewards = []

    for ep in range(1, NUM_EPISODES + 1):
        state, _    = env.reset()
        log_probs   = []
        rewards     = []
        done        = False

        while not done:
            action, log_prob = select_action(state, model)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        returns = compute_returns(rewards)
        loss    = sum(-lp * R for lp, R in zip(log_probs, returns))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(sum(rewards))

        if ep % SAVE_INTERVAL == 0:
            ckpt_path = save_checkpoint(model, ep)
            avg_reward = sum(episode_rewards[-SAVE_INTERVAL:]) / SAVE_INTERVAL
            print(f"[Episode {ep:04d}] AvgReward (last {SAVE_INTERVAL}) = {avg_reward:.2f} → Saved {ckpt_path}")

    env.close()
    return episode_rewards


# 5. Recording Loop

def record_from_checkpoints():
    ckpt_files = sorted(f for f in os.listdir(CKPT_DIR) if f.endswith(".pth"))
    if not ckpt_files:
        print("No checkpoints in", CKPT_DIR)
        return

    for filename in ckpt_files:
        ep_num   = filename.split("policy_ep")[1].split(".pth")[0]
        ckpt_path = os.path.join(CKPT_DIR, filename)
        policy_net = load_checkpoint(ckpt_path)

        # Only record one episode per checkpoint
        env = RecordVideo(
            gym.make(ENV_NAME, render_mode="rgb_array"),
            video_folder=VIDEO_DIR,
            episode_trigger=lambda x: True,
            name_prefix=f"post_ep{ep_num}"
        )
        state, _ = env.reset()
        done     = False

        while not done:
            with torch.no_grad():
                probs = policy_net(torch.from_numpy(state).float())
            action = torch.argmax(probs).item()
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state

        env.close()
        print(f"Saved video for checkpoint ep{ep_num}: {VIDEO_DIR}")


# 6. Main

if __name__ == "__main__":
    print("=== TRAINING PHASE ===")
    rewards = train_and_save()
    print("\n=== RECORDING PHASE ===")
    record_from_checkpoints()
    print("All done. Checkpoints:", CKPT_DIR, "| Videos:", VIDEO_DIR)
