import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

env = gym.make("CartPole-v1")


def expert(obs):
    # Scripted expert: push in the direction the pole is falling.
    # obs = [cart position, cart velocity, pole angle, pole angular velocity]
    return int(obs[2] + obs[3] > 0)


def rollout(policy, episodes, seed0=0):
    """Let `policy` (a function: observation -> action) play `episodes` episodes.

    Returns all (observation, action) pairs and the return of each episode.
    """
    states, actions, returns = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done, ret = False, 0.0
        while not done:
            a = policy(obs)
            states.append(obs)
            actions.append(a)
            obs, r, term, trunc, _ = env.step(a)
            ret += r
            done = term or trunc
        returns.append(ret)
    return np.array(states, dtype=np.float32), np.array(actions), returns


# 1) Collect expert demonstrations
S, A, rets = rollout(expert, episodes=50)
print(f"Expert average return: {np.mean(rets):.1f}, samples: {len(S)}")

# 2) Train the behavior cloning network (plain supervised learning)
net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
X = torch.tensor(S)
Y = torch.tensor(A, dtype=torch.long)

for epoch in range(30):
    perm = torch.randperm(len(X))
    total = 0.0
    for i in range(0, len(X), 64):
        idx = perm[i:i + 64]
        loss = loss_fn(net(X[idx]), Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    print(f"epoch {epoch + 1}: loss = {total / len(X):.4f}")


# 3) Evaluate the learned policy on seeds not used for data collection
def bc_policy(obs):
    with torch.no_grad():
        return int(net(torch.tensor(obs, dtype=torch.float32)).argmax())


_, _, bc_rets = rollout(bc_policy, episodes=20, seed0=1000)
print(f"BC average return: {np.mean(bc_rets):.1f} (max 500)")


# 4) Effect of the number of demonstration episodes (single training seed per setting)
def train_and_eval(n_demo_episodes, epochs=30, seed=0):
    torch.manual_seed(seed)
    S, A, _ = rollout(expert, episodes=n_demo_episodes)
    net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    X = torch.tensor(S)
    Y = torch.tensor(A, dtype=torch.long)
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            loss = loss_fn(net(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    def policy(obs):
        with torch.no_grad():
            return int(net(torch.tensor(obs, dtype=torch.float32)).argmax())

    _, _, rets = rollout(policy, episodes=20, seed0=1000)
    return np.mean(rets)


for n in [2, 5, 10, 20, 50]:
    print(f"Demonstration episodes {n}: BC average return {train_and_eval(n):.1f}")
