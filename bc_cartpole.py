import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

env = gym.make("CartPole-v1")

def expert(obs):
    # 简单的规则专家:杆往哪边倒,就往哪边推
    return int(obs[2] + obs[3] > 0)

def rollout(policy, episodes, seed0=0):
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

# 1) 收集专家示范
S, A, rets = rollout(expert, episodes=50)
print(f"专家平均回报: {np.mean(rets):.1f}, 数据量: {len(S)}")

# 2) 训练行为克隆网络(普通监督学习)
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

# 3) 评估学出来的策略
def bc_policy(obs):
    with torch.no_grad():
        return int(net(torch.tensor(obs, dtype=torch.float32)).argmax())

_, _, bc_rets = rollout(bc_policy, episodes=20, seed0=1000)
print(f"BC 平均回报: {np.mean(bc_rets):.1f}(满分 500)")