import torch
import argparse
import json
import os
import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import numpy as np
import torch
from torch import nn

from pfrl.agents.iqn import IQN, ImplicitQuantileQFunction, CosineBasisLinear
from pfrl.explorers import LinearDecayEpsilonGreedy
from pfrl.replay_buffers import ReplayBuffer

n_actions = 6

q_func = ImplicitQuantileQFunction(
    psi=nn.Sequential(
        nn.Conv2d(4, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
    ),
    phi=nn.Sequential(
        CosineBasisLinear(64, 3136),
        nn.ReLU(),
    ),
    f=nn.Sequential(
        nn.Linear(3136, 512),
        nn.ReLU(),
        nn.Linear(512, n_actions),
    ),
)

# Use the same hyper parameters as https://arxiv.org/abs/1710.10044
opt = torch.optim.Adam(q_func.parameters())

rbuf = ReplayBuffer(10**6)

explorer = LinearDecayEpsilonGreedy(
    0,
    0,
    10_000,
    lambda: np.random.randint(n_actions),
)

def phi(x):
    # Feature extractor
    return np.asarray(x, dtype=np.float32) / 255

agent = IQN(
    q_func,
    opt,
    rbuf,
    gpu=-1,
    gamma=0.99,
    explorer=explorer,
    phi=phi,
)

agent.load("best")

def make_env():
    env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode="human")
    env = AtariPreprocessing(env)
    env = FrameStack(env, 4)
    return env


env = make_env()
path_of_video_with_name = "iqn_videotest2.mp4"
state = env.reset()
video_recorder = None
video_recorder = VideoRecorder(env, path_of_video_with_name, enabled=True)
done = False
obs = env.reset()
state = np.asarray(obs)
while not done:
    video_recorder.capture_frame()
    print(state.shape)
    action = agent.act(state)
    obs, rewards, done, info = env.step(action)
    state = np.asarray(obs)

print("Saved video.")

video_recorder.close()
video_recorder.enabled = False
env.close()