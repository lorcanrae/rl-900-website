import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack


# Create environmnent and preprocess
def instantiate_environmnent():
    env = gym.make("SpaceInvadersNoFrameskip-v4")
    env = AtariPreprocessing(env, grayscale_newaxis=False, frame_skip=5)
    env = FrameStack(env, 4)
    return env

# Reward function - not really enough time to get these working
def reward_function(reward, _, lives, rewards_history):
    # staying alive award
    # reward += 10
    # if _['episode_frame_number'] > 2000:
    #     reward += 100000

    # if lives > _['lives']:
    #     reward = 0
    #     lives -= 1
    #
    #     deduction = 0
    #     for i in range(1, 10):
    #         rewards_history[-i] = deduction
    #         deduction += 1

        # punishment = _['episode_frame_number'] + 100
    # if _['episode_frame_number'] < punishment:
    #     reward = 0
    #     punishment -= 1

    return reward, lives, rewards_history
