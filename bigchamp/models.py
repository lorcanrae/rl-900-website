import tensorflow as tf
from tensorflow import keras

import os

# IQN Teacher Model Imports
import numpy as np
import torch

from pfrl.agents.iqn import IQN, ImplicitQuantileQFunction, CosineBasisLinear
from pfrl.explorers import LinearDecayEpsilonGreedy
from pfrl.replay_buffers import ReplayBuffer

# DQN Model

class DQN(keras.Model):
    '''DQN Model with three Convoluting layers'''
    def __init__(self, num_actions):
        super(DQN,self).__init__()
        self.conv1 = keras.layers.Conv2D(
            filters=32,
            kernel_size=8,
            activation='relu',
            input_shape=(84, 84, 4)
            )
        self.pool1 = keras.layers.MaxPool2D(pool_size=(4, 4))
        self.conv2 = keras.layers.Conv2D(
            filters=64,
            kernel_size=4,
            activation='relu'
            )
        self.pool2 = keras.layers.MaxPool2D(pool_size=(2, 2))
        self.conv3 = keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation='relu'
            )
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(units=512, activation='relu')
        self.action = keras.layers.Dense(units=num_actions, activation='linear')

    @tf.function
    def call(self, states):
        x = self.conv1(states)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.action(x)


# DuellingDQN Model
# Adapted from model by Markel Sanz Ausin from
# https://markelsanz14.medium.com/introduction-to-reinforcement-learning-part-4-double-dqn-and-dueling-dqn-b349c9a61ea1

class DuelingDQN(keras.Model):
    def __init__(self, num_actions):
        '''Dueling DQN Model with three Convoluting layers'''
        super(DuelingDQN, self).__init__()

        self.conv1 = keras.layers.Conv2D(
            filters=32,
            kernel_size=8,
            strides=4,
            activation="relu",
            kernel_initializer=keras.initializers.VarianceScaling(2.0),
            data_format="channels_last",
            input_shape=(84,84,4)
        )
        self.conv2 = keras.layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=2,
            activation="relu",
            kernel_initializer=keras.initializers.VarianceScaling(2.0),
        )
        self.conv3 = keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            activation="relu",
            kernel_initializer=keras.initializers.VarianceScaling(2.0),
        )
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(
            units=512,
            activation="relu",
            kernel_initializer=keras.initializers.VarianceScaling(2.0),
        )
        self.V = keras.layers.Dense(1)
        self.A = keras.layers.Dense(num_actions)

    @tf.function
    def call(self, states):
        states = tf.cast(states, tf.float32)
        x = self.conv1(states)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        V = self.V(x)
        A = self.A(x)
        Q = V + tf.subtract(A, tf.reduce_mean(A, axis=1, keepdims=True))
        return Q


# IQN Teacher Model
# Model from: TODO

def phi(x):
        return np.asarray(x, dtype=np.float32) / 255

def torch_iqn_teacher(num_actions=6, rel_path_to_best='../teacher_model/best'):
    q_func = ImplicitQuantileQFunction(
        psi=torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        ),
        phi=torch.nn.Sequential(
            CosineBasisLinear(64, 3136),
            torch.nn.ReLU(),
        ),
        f=torch.nn.Sequential(
            torch.nn.Linear(3136, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_actions),
        ),
    )

    # Use the same hyper parameters as https://arxiv.org/abs/1710.10044
    opt = torch.optim.Adam(q_func.parameters())

    rbuf = ReplayBuffer(10**6)

    explorer = LinearDecayEpsilonGreedy(
        0,
        0,
        10_000,
        lambda: np.random.randint(num_actions),
    )

    agent = IQN(
        q_func,
        opt,
        rbuf,
        gpu=-1,
        gamma=0.99,
        explorer=explorer,
        phi=phi,
    )

    abs_cwd_dir = os.path.dirname(os.path.abspath(__file__))
    # print(abs_cwd_dir)
    target_dir = os.path.join(abs_cwd_dir, rel_path_to_best)
    # print(target_dir)
    agent.load(target_dir)
    return agent


if __name__ == '__main__':
    agent = torch_iqn_teacher()
