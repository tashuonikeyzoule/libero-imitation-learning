Imitation Learning for Robotics: Behavior Cloning Baselines

A learning project in which I build imitation-learning baselines in PyTorch, starting from the simplest setting and working toward robot manipulation. The long-term goal is to understand how policies are learned from demonstrations, as a stepping stone toward vision-language-action (VLA) models such as OpenVLA.

Status: work in progress. The first baseline (CartPole) is complete. The experiments below marked "in progress" or "planned" are not done yet, and this README is updated as they are.

Stage 1: Behavior cloning on CartPole (done)

bc_cartpole.py implements the full imitation-learning pipeline end to end:

Collect demonstrations. A hand-written expert rule (push in the direction the pole is falling) plays 50 episodes of CartPole-v1. Every step is recorded as an (observation, action) pair, giving 24,834 training samples.
Train a policy by supervised learning. An MLP (4 -> 64 -> 64 -> 2, ReLU) predicts the expert's action from the 4-dimensional observation. Cross-entropy loss, Adam (lr 1e-3), batch size 64, 30 epochs.
Evaluate the learned policy. The trained network plays 20 episodes on random seeds not used for data collection (seeds 1000-1019).
Results
Policy	Average return (max 500)	Episodes
Expert (hand-written rule)	496.7	50 (demonstration data)
Behavior cloning (MLP)	462.1	20 (held-out seeds)

Training loss fell from 0.381 (epoch 1) to 0.025 (epoch 30). The full console output is in results.txt.

The cloned policy gets close to the expert but not all the way. This is consistent with the usual weakness of behavior cloning: the demonstrations rarely contain states where the pole is close to falling, so the learned policy has little experience to draw on there, and small mistakes can accumulate.

Note that this is a single run with a single random seed, so the exact numbers will vary between runs.

Stage 2: Effect of the amount of demonstration data (in progress)

How many expert episodes does behavior cloning need? Results will be filled in here once the experiment is run.

Demonstration episodes	BC average return
2	TBD
5	TBD
10	TBD
20	TBD
50	462.1 (from Stage 1)
Roadmap (planned, not done)
Behavior cloning with continuous actions (regression instead of classification), the step needed before robot arms.
Behavior cloning on a simulated robot manipulation task (robomimic or LIBERO, depending on what installs and runs on my machine).
Further reading toward VLA models (OpenVLA).
How to run
bash
pip install -r requirements.txt
python bc_cartpole.py

Tested with Python 3.10.8 on Windows (CPU only). Package versions are in requirements.txt. The training here is small enough that no GPU is needed.

Notes
The expert is a simple scripted rule, not a trained agent.
This is a self-study project. The code was written while learning the topic, with AI assistance for explanations and debugging.
