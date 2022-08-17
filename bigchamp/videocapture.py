import numpy as np
import tensorflow as tf
from tensorflow import keras

import os

from gym.wrappers.monitoring.video_recorder import VideoRecorder

from bigchamp.helper_functions import instantiate_environmnent


def create_video(model_name, video_name, n_episodes=10):
    '''Video capture for saved TensorFlow models'''
    env = instantiate_environmnent()

    abs_cwd = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(abs_cwd, '..', 'saved_models', model_name)

    model = keras.models.load_model(model_dir, compile=False)
    video = VideoRecorder(env, f'{video_name}.mp4', enabled=True)

    for episode in range(1, n_episodes+1):
        state = np.asarray(env.reset()).reshape(84, 84, 4)
        done = False
        score = 0

        while not done:
            batch_state = tf.expand_dims(state, 0)
            action = np.argmax(model.predict(batch_state)[0])
            state, reward, done, _ = env.step(action)
            state = np.asarray(state).reshape(84, 84, 4)
            video.capture_frame()
            score += reward

        print(f'Episode:{episode} Score:{score}')

    video.close()
    env.close()


if __name__ == '__main__':
    create_video(n_episodes=10,
                 model_name='model-duelingdqn-teacher-e1000',
                 video_name='model-e1000.mp4')
