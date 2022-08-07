import numpy as np
import tensorflow as tf
from tensorflow import keras

from models import DQN, DuelingDQN
from helper_functions import instantiate_environmnent, reward_function


def train_bigchamp(n_actions=6,
                   rand_frames=1_000_000,
                   greedy_frames=5_000_000,
                   model_type=DuelingDQN,
                   ):
    '''Train Agent'''

    # Instantiate environment and models
    env = instantiate_environmnent()

    model = model_type(n_actions)
    model_target = model_type(n_actions)

    # Configuration paramaters for the whole setup
    gamma = 0.99 # Discount factor for past rewards
    epsilon = 1.0 # Epsilon greedy parameter
    epsilon_min = 0.1 # Minimum epsilon greedy parameter
    epsilon_max = 1.0 # Maximum epsilon greedy parameter
    epsilon_interval = epsilon_max - epsilon_min # Chance of random action
    batch_size = 32 # Size of batch taken from replay buffer
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

    # Number of frames to take random action and observe output and greediness factor
    epsilon_random_frames = rand_frames # Should change depending on training time
    epsilon_greedy_frames = greedy_frames # Should change depending on training time

    # Maximum replay length
    max_memory_length = 100_000

    # Train the model after 4 actions
    update_after_actions = 4

    # How often to update the target network
    update_target_network = 10_000

    # Using huber loss for stability
    loss_function = keras.losses.Huber()

    # Main Training bloc - run until solved
    while True:
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
                action = np.random.choice(n_actions)
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
            reward = reward_function(reward)

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
                masks = tf.one_hot(action_sample, n_actions)

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
                print("xxxxxxxxxx")
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))

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
        if episode_count % 100 == 0:
            model.save("model3")

        episode_count += 1

        if running_score > 1000:  # Condition to consider the task solved
            print("xxxxxxxxxx")
            print("Solved at episode {}!".format(episode_count))
            break
