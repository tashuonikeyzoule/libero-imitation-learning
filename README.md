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

Stage 2: Effect of the amount of demonstration data (done, single seed)

How many expert episodes does behavior cloning need? I retrained the same network on different numbers of expert episodes and evaluated each policy on 20 held-out episodes (seeds 1000-1019). Raw output: results_stage2.txt.

Demonstration episodes	BC average return (max 500)
2	381.5
5	482.6
10	475.0
20	462.6
50	462.1

Observations:

With only 2 episodes the policy is clearly worse (381.5).
From 5 episodes onward the return is roughly flat at about 460-480, still below the expert's 496.7.
This is one training run per setting, so the ordering among 5, 10, 20 and 50 episodes is not meaningful. For reference, re-running the Stage 1 script gave 462.8 instead of 462.1, so differences of this size are run-to-run noise.
Next step: repeat each setting with several training seeds and report mean and standard deviation, to check whether the plateau below the expert is real.
Stage 3: Behavior cloning on a simulated robot arm (done, single run)

bc_pusher.py repeats the same pipeline on a simulated robot arm, with continuous actions. inspect_pusher_dataset.py downloads and inspects the data.

Environment: Pusher-v5 (MuJoCo, via Gymnasium). Observation: 23 numbers. Action: 7 continuous numbers in [-2, 2].
Demonstrations: the Minari dataset mujoco/pusher/expert-v0 (5,000 episodes, 500,000 steps). According to the dataset page, the expert is a reinforcement-learning agent (SAC). I used the first 500 episodes (50,000 observation-action pairs).
Method: observations standardised with the training mean and standard deviation; MLP 23 -> 256 -> 256 -> 7; mean-squared-error loss (regression instead of the classification loss used for CartPole); Adam (lr 1e-3), batch size 256, 30 epochs.
Evaluation: 20 episodes in Pusher-v5, seeds 1000-1019. Returns are negative; closer to zero is better. Raw output: results_stage3.txt.
Policy	Average return
Random actions	-147.7
Behavior cloning (500 expert episodes)	-33.9
Expert (average over the 500 training episodes in the dataset)	-24.8

The cloned policy closes roughly 93% of the gap between random actions and the expert, measured on average return.

Caveats:

One training run and 20 evaluation episodes, so small differences are not reliable.
The expert's average comes from the dataset's own episodes, not from re-running the expert on my evaluation seeds.
Only average return is measured. I have not computed a task success rate.
The training loss falls quickly in the first epoch and only slowly afterwards (0.0384 to 0.0328). I have not investigated why.
Roadmap (planned, not done)
Effect of the number of demonstration episodes and of several training seeds on the Pusher task (as done for CartPole).
A task-specific success metric instead of average return only.
Image observations and language-conditioned policies, as a stepping stone towards VLA models (OpenVLA).
How to run
bash
pip install -r requirements.txt
python bc_cartpole.py

Tested with Python 3.10.8 on Windows (CPU only). Package versions are in requirements.txt. The training here is small enough that no GPU is needed.

Stages 1 and 2 (CartPole) ran in Python 3.10.8 with gymnasium 0.29.1. Stage 3 (Pusher) needs a newer Gymnasium and ran in a separate Python 3.11 virtual environment with gymnasium 1.4.0, minari 0.5.4 and mujoco 4.0.1 (plus PyTorch, h5py and huggingface_hub; exact versions of those were not recorded):

bash
pip install "gymnasium[mujoco]" "minari[hf,hdf5]" torch
python inspect_pusher_dataset.py   # downloads the dataset (about 195 MB)
python bc_pusher.py
Notes
The expert is a simple scripted rule, not a trained agent.
This is a self-study project. The code was written while learning the topic, with AI assistance for explanations and debugging.

Notes
The expert is a simple scripted rule, not a trained agent.
This is a self-study project. The code was written while learning the topic, with AI assistance for explanations and debugging.
