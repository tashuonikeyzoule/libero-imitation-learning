"""
Stage 4: behavior cloning on Pusher-v5 with an auxiliary "future fingertip motion" loss.

Idea (a tiny, low-dimensional version of the idea in ELAN4D, CoRL 2026):
the policy is trained to predict the expert action AND, through a second
output head, how the arm's fingertip will move over the next H steps.
The second head is used only during training and is ignored at test time.

Question: does this extra prediction target change the policy's
performance, especially when there are only a few demonstrations?

Comparison:
  - baseline: action loss only (aux weight 0.0)
  - aux:      action loss + AUX_WEIGHT * future-fingertip loss
for several numbers of demonstration episodes and several training seeds.
"""
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import minari

H = 5                       # predict the fingertip displacement for the next 1..H steps
AUX_WEIGHT = 0.1            # weight of the auxiliary loss (same value as ELAN4D's lambda_track)
EPISODE_COUNTS = [20, 50, 500]
TRAIN_SEEDS = [0, 1, 2]
EPOCHS = 30
EVAL_EPISODES = 20
TIP = slice(14, 17)         # fingertip (x, y, z) inside the 23-number Pusher observation
OBJ = slice(17, 20)         # object position
GOAL = slice(20, 23)        # goal position

dataset = minari.load_dataset("mujoco/pusher/expert-v0")


def load_data(n_episodes):
    """Return observations X, expert actions Y, and future fingertip displacements Z."""
    xs, ys, zs = [], [], []
    for ep in dataset.iterate_episodes(episode_indices=range(n_episodes)):
        obs = ep.observations            # n + 1 observations
        n = len(ep.actions)              # n actions
        tip = obs[:, TIP]
        for t in range(n):
            future = [tip[min(t + k, n)] - tip[t] for k in range(1, H + 1)]
            zs.append(np.concatenate(future))  # 3 * H numbers
        xs.append(obs[:n])
        ys.append(ep.actions)
    X = np.concatenate(xs).astype(np.float32)
    Y = np.concatenate(ys).astype(np.float32)
    Z = np.array(zs, dtype=np.float32)
    return X, Y, Z


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(23, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.action_head = nn.Linear(256, 7)        # used at test time
        self.future_head = nn.Linear(256, 3 * H)    # used only during training

    def forward(self, x):
        h = self.trunk(x)
        return self.action_head(h), self.future_head(h)


def train(X, Y, Z, aux_weight, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs_mean, obs_std = X.mean(0), X.std(0) + 1e-6
    z_std = Z.std(0) + 1e-6                          # scale the aux target to roughly unit size
    Xn = torch.tensor((X - obs_mean) / obs_std)
    Yt = torch.tensor(Y)
    Zn = torch.tensor(Z / z_std)

    net = Policy()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(Xn))
        for i in range(0, len(Xn), 256):
            idx = perm[i:i + 256]
            pred_a, pred_z = net(Xn[idx])
            loss = mse(pred_a, Yt[idx]) + aux_weight * mse(pred_z, Zn[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net, obs_mean, obs_std


def evaluate(net, obs_mean, obs_std, env):
    """Average return, and average final object-to-goal distance (smaller is better)."""
    returns, final_dists = [], []
    for ep in range(EVAL_EPISODES):
        obs, _ = env.reset(seed=1000 + ep)
        done, ret = False, 0.0
        while not done:
            with torch.no_grad():
                o = torch.tensor((obs - obs_mean) / obs_std, dtype=torch.float32)
                action = np.clip(net(o)[0].numpy(), -2.0, 2.0)
            obs, r, term, trunc, _ = env.step(action)
            ret += r
            done = term or trunc
        returns.append(ret)
        final_dists.append(float(np.linalg.norm(obs[OBJ] - obs[GOAL])))
    return np.mean(returns), np.mean(final_dists)


env = gym.make("Pusher-v5")
print(f"H = {H}, aux weight = {AUX_WEIGHT}, epochs = {EPOCHS}, "
      f"training seeds = {TRAIN_SEEDS}, eval episodes = {EVAL_EPISODES} (seeds 1000-{999 + EVAL_EPISODES})")
print("episodes | method   | return mean +- std (over training seeds) | final obj-goal distance mean +- std")

for n_ep in EPISODE_COUNTS:
    X, Y, Z = load_data(n_ep)
    for name, w in [("baseline", 0.0), ("aux", AUX_WEIGHT)]:
        rets, dists = [], []
        for seed in TRAIN_SEEDS:
            net, m, s = train(X, Y, Z, w, seed)
            r, d = evaluate(net, m, s, env)
            rets.append(r)
            dists.append(d)
            print(f"   (episodes {n_ep}, {name}, seed {seed}: return {r:.1f}, final distance {d:.3f})")
        print(f"{n_ep:8d} | {name:8s} | {np.mean(rets):7.1f} +- {np.std(rets):5.1f}"
              f"                     | {np.mean(dists):.3f} +- {np.std(dists):.3f}")