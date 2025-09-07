'''Uses gymansium instead of gym'''

import os
import pickle  # for catching UnpicklingError explicitly
import numpy as np
from numpy.core.multiarray import scalar as np_scalar  # allow-list target
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


# =========================
# 1) Hyperparameters & Dirs
# =========================

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


# =========================
# 2) Policy Network
# =========================

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # keep dims so we can re-instantiate later
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.net = nn.Sequential(
            nn.Linear(self.state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, self.action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# 3) Utilities
# =========================

def select_action(state, policy_net):
    state = torch.from_numpy(state).float()
    probs = policy_net(state)
    dist  = Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)


def compute_returns(rewards):
    R = 0.0
    returns = []
    for r in reversed(rewards):
        R = r + GAMMA * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    return (returns - returns.mean()) / (returns.std() + 1e-9)


def save_checkpoint(model, episode):
    """
    Save a strictly 'weights-only friendly' checkpoint.
    (Dims are forced to Python ints; payload is a plain dict.)
    """
    filename = os.path.join(CKPT_DIR, f"policy_ep{episode:04d}.pth")
    payload = {
        "state_dict": model.state_dict(),
        "state_dim": int(model.state_dim),
        "action_dim": int(model.action_dim),
    }
    torch.save(payload, filename)
    return filename


def _infer_dims_from_state_dict(state_dict):
    """
    Best-effort inference for this exact architecture:
    Sequential: [Linear(0), ReLU(1), Linear(2), Softmax(3)]
    We read shapes of the first and last Linear layer weights.
    """
    # find "net.<idx>.weight" linear layers
    linear_weight_keys = [
        k for k in state_dict.keys()
        if k.startswith("net.") and k.endswith(".weight")
    ]
    if not linear_weight_keys:
        raise ValueError("Could not find linear layer weights to infer dims.")

    # sort by the integer after "net."
    def layer_index(k):
        try:
            return int(k.split(".")[1])
        except Exception:
            return 1_000_000

    linear_weight_keys.sort(key=layer_index)

    first_w = state_dict[linear_weight_keys[0]]  # shape [hidden, state_dim]
    last_w  = state_dict[linear_weight_keys[-1]] # shape [action_dim, hidden]

    if first_w.ndim != 2 or last_w.ndim != 2:
        raise ValueError("Unexpected weight tensor shape when inferring dims.")

    state_dim  = int(first_w.shape[1])
    action_dim = int(last_w.shape[0])
    return state_dim, action_dim


def load_checkpoint(path, map_location="cpu"):
    """
    Safe-first loader for PyTorch 2.6+.

    1) Try weights_only=True with a minimal allow-list for NumPy scalar
       (fixes the error you saw).
    2) If that fails with UnpicklingError and you absolutely trust the file,
       you may temporarily enable the commented fallback to weights_only=False
       to convert and re-save a safe checkpoint.

    Supports:
      - {'state_dict', 'state_dim', 'action_dim'} payloads (your current format)
      - raw state_dict (dims inferred)
    """
    try:
        # Strict, safe load with a tiny allow-list for your error case
        with torch.serialization.safe_globals([np_scalar]):
            ckpt = torch.load(path, weights_only=True, map_location=map_location)
    except pickle.UnpicklingError as e:
        # ---- If you absolutely trust the source, you may uncomment the next two lines:
        # ckpt = torch.load(path, weights_only=False, map_location=map_location)
        # print("Loaded with weights_only=False (unsafe for untrusted sources).")
        raise RuntimeError(
            "Secure load failed. If you trust this checkpoint, temporarily allow "
            "weights_only=False in load_checkpoint, then re-save a safe weights file."
        ) from e

    # ckpt can be:
    #  (A) {'state_dict', 'state_dim', 'action_dim'}
    #  (B) raw state_dict (mapping of parameter_name -> tensor)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        state_dim  = int(ckpt["state_dim"])
        action_dim = int(ckpt["action_dim"])
    elif isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        # raw state_dict; infer dims
        state_dict = ckpt
        state_dim, action_dim = _infer_dims_from_state_dict(state_dict)
    else:
        raise ValueError(
            f"Unsupported checkpoint structure at {path!r}. "
            "Expected a dict with 'state_dict' or a raw state_dict."
        )

    model = PolicyNetwork(state_dim, action_dim)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# =========================
# 4) Training Loop
# =========================

def train_and_save():
    env = gym.make(ENV_NAME)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model     = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    episode_rewards = []

    for ep in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        log_probs = []
        rewards   = []
        done      = False

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


# =========================
# 5) Recording Loop
# =========================

def record_from_checkpoints():
    ckpt_files = sorted(f for f in os.listdir(CKPT_DIR) if f.endswith(".pth"))
    if not ckpt_files:
        print("No checkpoints in", CKPT_DIR)
        return

    for filename in ckpt_files:
        ep_num    = filename.split("policy_ep")[1].split(".pth")[0]
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
        done = False

        while not done:
            with torch.no_grad():
                probs = policy_net(torch.from_numpy(state).float())
            action = torch.argmax(probs).item()
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state

        env.close()
        print(f"Saved video for checkpoint ep{ep_num}: {VIDEO_DIR}")


# =========================
# 6) Main
# =========================

if __name__ == "__main__":
    print("=== TRAINING PHASE ===")
    rewards = train_and_save()
    print("\n=== RECORDING PHASE ===")
    record_from_checkpoints()
    print("All done. Checkpoints:", CKPT_DIR, "| Videos:", VIDEO_DIR)
