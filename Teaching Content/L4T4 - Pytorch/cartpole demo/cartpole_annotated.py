'''Teaching Version
This file is organized to introduce ideas in the same order students experience them:
env -> policy -> sampling -> episode loop -> returns -> loss -> training -> plotting
Inline "TEACHER NOTE" comments are brief descriptions and short "Show & Tell" blocks
can be quickly uncommented to show off different aspects of the program.'''

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# 1) The Playground (Gymnasium)
# -----------------------------
# TEACHER NOTE:
# The environment provides:
#   - state (what the agent sees)  : shape (4,) for CartPole
#   - action space (what it can do): 2 discrete actions (left/right)
#   - reward each step (how well it's doing)
env = gym.make('CartPole-v1')

# (Optional but nice for reproducibility in class demos)
# torch.manual_seed(0)
# env.reset(seed=0)

# 2) The Policy Network (Decision-maker)
# --------------------------------------
# TEACHER NOTE:
# Given a state (4 numbers), output a probability for each action (2 numbers),
# via a small MLP ending with softmax so outputs sum to 1.

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # TEACHER NOTE:
        # x is a single state tensor (shape [4]). No batch dimension here keeps it simple.
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

# Hyperparameters
state_dim = env.observation_space.shape[0]   # 4
hidden_dim = 128
action_dim = env.action_space.n              # 2
learning_rate = 1e-2
gamma = 0.99

policy_net = PolicyNetwork(state_dim, hidden_dim, action_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# --- Show & Tell #1: Peek at policy output for one state -----------------------
# TEACHER NOTE:
# Right after introducing the policy, show the class that the network really outputs
# a distribution that sums to 1.
# s, _ = env.reset(seed=0)
# with torch.no_grad():
#     p = policy_net(torch.from_numpy(s).float())
# print("Action probabilities (sum=1):", p.numpy(), "sum:", float(p.sum()))

# 3) Select an action by sampling from the policy
# -----------------------------------------------
# TEACHER NOTE:
# We SAMPLE (not argmax) to encourage exploration—trying actions with some randomness
# helps discover better strategies.

def select_action(state):
    state = torch.from_numpy(state).float()
    probs = policy_net(state)        # probabilities over actions
    dist = Categorical(probs)        # a categorical distribution parameterized by probs
    action = dist.sample()           # sample an action (0 or 1)
    return action.item(), dist.log_prob(action)

# --- Show & Tell #2: Tiny trajectory to show sampling --------------------------
# TEACHER NOTE:
# After explaining sampling, demo 2–3 steps so students see the randomness in action choices.
# s, _ = env.reset(seed=1)
# for t in range(3):
#     a, lp = select_action(s)
#     s, r, term, trunc, _ = env.step(a)
#     print(f"step {t}: action={a}, reward={r}")
#     if term or trunc:
#         print("episode ended early")
#         break

# 4) Compute discounted returns (credit assignment over time)
# -----------------------------------------------------------
# TEACHER NOTE:
# Returns push credit back to earlier actions: R_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
# We normalize for training stability (variance reduction).

def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R          # accumulate from the end
        returns.insert(0, R)       # prepend so returns align with time steps
    returns = torch.tensor(returns)
    # Normalize for stability (helps gradients behave)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    return returns

# --- Show & Tell #3: Toy returns calculation -----------------------------------
# TEACHER NOTE:
# After explaining discounted returns, demo with a short reward list.
# This shows raw discounted sums vs. normalized values.
rs = [1, 1, 1, 1]  # pretend 4-step episode, reward=1 each time

# Compute raw discounted returns (no normalization)
# gamma = 0.99
# raw_returns = []
# R = 0
# for r in reversed(rs):
#     R = r + gamma * R
#     raw_returns.insert(0, R)

# # Compute normalized returns (like in training)
# returns_tensor = torch.tensor(raw_returns, dtype=torch.float32)
# normalized_returns = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-9)

# print("Toy rewards:                ", rs)
# print("Raw discounted returns:     ", [round(x, 2) for x in raw_returns])
# print("Normalized discounted returns:", [round(x.item(), 2) for x in normalized_returns])

# 5) Training loop (REINFORCE)
# ----------------------------
# TEACHER NOTE:
# One episode = roll out until done, collect (log_probs, rewards),
# then compute returns and form the policy gradient loss: sum(-log_pi(a_t|s_t) * R_t).
# Intuition: if R_t is high, increase log probability of that action; if low, decrease it.

def train_policy_gradient(num_episodes=500, print_interval=50):
    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        # Gymnasium API: reset() returns (obs, info)
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            action, log_prob = select_action(state)
            # Gymnasium API: step() returns (obs, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        # Turn rewards into returns (credit over time)
        returns = compute_returns(rewards, gamma)

        # REINFORCE loss: encourage actions that led to higher return
        loss_terms = []
        for log_prob, R in zip(log_probs, returns):
            loss_terms.append(-log_prob * R)
        loss = torch.stack(loss_terms).sum()

        # Standard PyTorch update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        # Periodic progress printouts keep learners engaged
        if episode % print_interval == 0:
            avg_reward = sum(episode_rewards[-print_interval:]) / print_interval
            print(f"Episode {episode}\tAverage Reward: {avg_reward:.2f}")

    return episode_rewards

# 6) Run training
# ---------------
# TEACHER NOTE:
# This may take a moment depending on hardware. Early rewards are noisy—look for the trend.
episode_rewards = train_policy_gradient(num_episodes=500)

# --- Show & Tell #4: Peek at early rewards -------------------------------------
# TEACHER NOTE:
# Ground the learning curve with the first few numbers before plotting.
# print("First 10 episode rewards:", episode_rewards[:10])

# 7) Plot rewards over episodes
# -----------------------------
# TEACHER NOTE:
# The curve should trend upward as the agent learns to keep the pole balanced longer.

def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Reward over Time (REINFORCE)')
    plt.show()

plot_rewards(episode_rewards)


# 8) (Optional) Discussion prompts
# --------------------------------
# TEACHER NOTE (talking points, not code):
# - What changed when we increased hidden_dim? (capacity)
# - What if we used argmax instead of sampling? (no exploration)
# - Why normalize returns? (stability/variance)
# - What if gamma is smaller/larger? (short vs long-term credit)
# - How could we add a baseline or entropy bonus to improve stability/exploration?