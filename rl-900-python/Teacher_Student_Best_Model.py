import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack


from pfrl.agents.iqn import IQN, ImplicitQuantileQFunction, CosineBasisLinear
from pfrl.explorers import LinearDecayEpsilonGreedy
from pfrl.replay_buffers import ReplayBuffer

import numpy as np
import torch
from torch import nn

# Create environmnent and preprocess


def instantiate_environmnent(game="SpaceInvaders-v4"):
    env = gym.make('SpaceInvadersNoFrameskip-v4', render_mode='human')
    env = AtariPreprocessing(env)
    env = FrameStack(env, 4)
    return env

def phi(x):
    # Feature extractor
    return np.asarray(x, dtype=np.float32) / 255

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(84, 84, 4))

    # Convolutions on the frames on the screen
    layer = layers.Conv2D(32, 8, activation="relu")(inputs)
    layer = layers.MaxPool2D(pool_size=(4, 4))(layer)
    layer = layers.Conv2D(64, 4, activation="relu")(layer)
    layer = layers.MaxPool2D(pool_size=(2, 2))(layer)
    layer = layers.Conv2D(64, 3, activation="relu")(layer)

    layer = layers.Flatten()(layer)

    layer = layers.Dense(512, activation="relu")(layer)
    action = layers.Dense(num_actions, activation="linear")(layer)

    return keras.Model(inputs=inputs, outputs=action)


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


if __name__ == "__main__":
    # Get Best Model

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

    num_actions = 6

    model_name = 'Student_Teacher_BestModel'

    # Instantiate environment
    env = instantiate_environmnent()

    # The first model makes the predictions for Q-values which are used to make a action
    model = create_q_model()
    # Build a target model for the prediction of future rewards
    # The weights of a target model get updated every 10000 steps thus when the loss between the Q-values is calculated the target Q-value is stable
    model_target = create_q_model()

    # Configuration paramaters for the whole setup
    gamma = 0.99  # Discount factor for past rewards
    epsilon = 1.0  # Epsilon greedy parameter
    epsilon_min = 0.1  # Minimum epsilon greedy parameter
    epsilon_max = 1.0  # Maximum epsilon greedy parameter
    epsilon_interval = epsilon_max - epsilon_min  # Rate at which to reduce chance of random action being taken
    batch_size = 32  # Size of batch taken from replay buffer
    max_steps_per_episode = 10000

    # Optimizer improves training time
    optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

    # Replay buffers
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    score_history = []

    # Information variables
    running_reward = 0
    episode_count = 0
    frame_count = 0
    explored = 0
    exploited = 0
    lives = 3

    # Number of frames to take random action and observe output and greediness factor
    epsilon_random_frames = 1_000_000  # 50000 # Should change depending on training time
    epsilon_greedy_frames = 10_000_000  # 1000000 # Should change depending on training time

    # Maximum replay length
    max_memory_length = 100000

    # Train the model after 4 actions
    update_after_actions = 4

    # How often to update the target network
    update_target_network = 10000

    # Using huber loss for stability
    loss_function = keras.losses.Huber()

    while True:  # Run until solved
        state = np.asarray(env.reset()).reshape(84, 84, 4)

        # Episode information
        frames_this_episode = 0
        episode_reward = 0
        score = 0

        for timestep in range(1, max_steps_per_episode):
            frame_count += 1
            frames_this_episode += 1

            # Use epsilon-greedy for exploration
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                # action = np.random.choice(num_actions)
                # Take Non random choice
                action = agent.act(state.reshape(4, 84, 84))
                # print(state.shape)
                # print(action)
            else:
                # Predict action Q-values from environment state
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)

                # Take best action
                action = tf.argmax(action_probs[0]).numpy()

            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # Apply the sampled action in our environment
            state_next, reward, done, _ = env.step(action)
            state_next = np.asarray(state_next).reshape(84, 84, 4)
            episode_frame_number = _["episode_frame_number"]
            score += reward

            # Reward modifier (This will also affect score)
            reward, lives, rewards_history = reward_function(reward, _, lives, rewards_history)

            episode_reward += reward

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            # Update every fourth frame and once batch size is over 32
            if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target.predict(state_next_sample)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, num_actions)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if frame_count % update_target_network == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))
                model.save(f"{model_name}.h5")

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                print("----------")
                print("----------")
                print(f"Game Number: {episode_count + 1}")
                if frame_count < epsilon_random_frames:
                    print("EXPLORATION")
                    explored += 1
                else:
                    print("EXPLOITATION")
                    exploited += 1
                print(f"Score: {score}")
                print(f"Reward: {episode_reward}")
                print(f"Timesteps: {frames_this_episode}")
                print(f"Game Frames Survived: {episode_frame_number}")
                print(f"Epsilon: {epsilon}")
                break

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        score_history.append(score)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history[-10:])
        running_score = np.mean(score_history[-10:])

        print(f"Running Score (last 10 games): {running_score}")
        print(f"Running Reward (last 10 games): {running_reward}")
        print(f"Explored: {explored}, Exploited: {exploited}")

        # Save model checkpoint
        if episode_count % 250 == 0:
           model.save(f"{model_name}-ep{episode_count}.h5")

        episode_count += 1

        # if frame_count >= 30_000:  # Condition to consider the task solved
        #     print("Solved at episode {}!".format(episode_count))
        #     break

    model.save(f"{model_name}.h5")
