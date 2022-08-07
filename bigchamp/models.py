import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
