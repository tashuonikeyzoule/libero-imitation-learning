import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import minari

N_EPISODES = 500  # number of expert episodes used for training

dataset = minari.load_dataset("mujoco/pusher/expert-v0")

obs_list, act_list, expert_returns = [], [], []
for ep in dataset.iterate_episodes(episode_indices=range(N_EPISODES)):
    n = len(ep.actions)
    obs_list.append(ep.observations[:n])   # pair each action with the observation before it
    act_list.append(ep.actions)
    expert_returns.append(float(np.sum(ep.rewards)))

X = np.concatenate(obs_list).astype(np.float32)
Y = np.concatenate(act_list).astype(np.float32)
print("training pairs:", X.shape, Y.shape)
print(f"Expert average return (dataset, {N_EPISODES} episodes): {np.mean(expert_returns):.1f}")

# Normalise observations
obs_mean, obs_std = X.mean(0), X.std(0) + 1e-6
Xn = torch.tensor((X - obs_mean) / obs_std)
Yt = torch.tensor(Y)

# Train: regression with MSE loss
net = nn.Sequential(nn.Linear(23, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 7))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(30):
    perm = torch.randperm(len(Xn))
    total = 0.0
    for i in range(0, len(Xn), 256):
        idx = perm[i:i + 256]
        loss = loss_fn(net(Xn[idx]), Yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    print(f"epoch {epoch + 1}: loss = {total / len(Xn):.4f}")

# Evaluate in the simulator
env = gym.make("Pusher-v5")

def bc_policy(obs):
    with torch.no_grad():
        o = torch.tensor((obs - obs_mean) / obs_std, dtype=torch.float32)
        return np.clip(net(o).numpy(), -2.0, 2.0)

def random_policy(obs):
    return env.action_space.sample()

def evaluate(policy, episodes=20, seed0=1000):
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done, ret = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy(obs))
            ret += r
            done = term or trunc
        returns.append(ret)
    return np.mean(returns)

print(f"Random policy average return: {evaluate(random_policy):.1f}")
print(f"BC policy average return:     {evaluate(bc_policy):.1f}")