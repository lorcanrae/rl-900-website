import streamlit as st
import numpy as np

import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack

import torch
from torch import nn

import time

try:
    from pfrl.agents.iqn import IQN, ImplicitQuantileQFunction, CosineBasisLinear
except:
    from pfrl.agents.iqn import IQN, ImplicitQuantileQFunction, CosineBasisLinear
from pfrl.explorers import LinearDecayEpsilonGreedy
from pfrl.replay_buffers import ReplayBuffer

### Helper functions

def instantiate_environmnent():
    env = gym.make("SpaceInvadersNoFrameskip-v4")
    env = AtariPreprocessing(env, grayscale_newaxis=False, frame_skip=5)
    env = FrameStack(env, 4)

    return env

### Best Model

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

model = agent.load('best')

### End Helper Funtions

# Instantiate

env = instantiate_environmnent()

### Main

col1, col2 = st.columns([1, 1])

with col1:
    st.title('BIGCHAMP-900')
    st.markdown("""### Time to save the world!""")
    episodes = st.expander('N Games').slider('N Games', 1, 10, 5)
    start_model = st.button("BLAST OFF!")
    stop_model = st.button("Ease up Champ!")

    st.markdown('''
                Team:
                - Alexander Gribius
                - Dan Hawkins
                - Alberto Lopez Rueda
                - Lorcan Rae
                With thanks to Oliver Giles
                ''')

DISPLAY_WIDTH = 420

with col2:

    if start_model:

        with st.empty():

            for episode in range(1, episodes+1):
                state = np.asarray(env.reset())
                done = False
                score = 0

                while not done:
                    action = agent.act(state)
                    state, reward, done, info = env.step(action)
                    state = np.asarray(state)
                    score += reward
                    st.image(env.render(mode='rgb_array'), width=DISPLAY_WIDTH)

                    time.sleep(0.035)

                    if stop_model:
                        break

    else:
        with st.empty():
            state = np.asarray(env.reset()).reshape(84, 84, 4)
            st.image(env.render(mode='rgb_array'), width=DISPLAY_WIDTH)
