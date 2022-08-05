# Sourced from
# https://markelsanz14.medium.com/introduction-to-reinforcement-learning-part-4-double-dqn-and-dueling-dqn-b349c9a61ea1

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DuelingDQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(DuelingDQN, self).__init__()
        self.scale = tf.keras.layers.Lambda(
            lambda x: x/255
        )
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=8,
            strides=4,
            activation="relu",
            kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
            bias_initializer=tf.keras.initializers.Zeros(),
            data_format="channels_last",
            input_shape=(84,84,4)
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=2,
            activation="relu",
            kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
            bias_initializer=tf.keras.initializers.Zeros(),
            data_format="channels_last",
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            activation="relu",
            kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
            bias_initializer=tf.keras.initializers.Zeros(),
            data_format="channels_last",
        )
        self.flatten =tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            units=512,
            activation="relu",
            kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
            bias_initializer=tf.keras.initializers.Zeros(),
        )
        self.V = tf.keras.layers.Dense(1)
        self.A = tf.keras.layers.Dense(num_actions)

    @tf.function
    def call(self, states):
        x = self.scale(states)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        V = self.V(x)
        A = self.A(x)
        Q = V + tf.subtract(A, tf.reduce_mean(A, axis=1, keepdims=True))
        return Q
