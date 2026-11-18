import gymnasium as gym
import minari

# 1) Check that the Pusher-v5 environment works
env = gym.make("Pusher-v5")
obs, _ = env.reset(seed=0)
print("env obs shape:", obs.shape, "| action space:", env.action_space)

# 2) Download and inspect the expert dataset
minari.download_dataset("mujoco/pusher/expert-v0")
ds = minari.load_dataset("mujoco/pusher/expert-v0")
print("episodes:", ds.total_episodes, "| steps:", ds.total_steps)
print("obs space:", ds.observation_space)
print("action space:", ds.action_space)