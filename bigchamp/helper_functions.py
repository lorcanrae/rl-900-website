import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack


# Create environmnent and preprocess
def instantiate_environmnent():
    env = gym.make("SpaceInvadersNoFrameskip-v4")
    env = AtariPreprocessing(env, grayscale_newaxis=False, frame_skip=5)
    env = FrameStack(env, 4)

    return env

# Reward function - no time to play :(
def reward_function(reward):
    return reward
