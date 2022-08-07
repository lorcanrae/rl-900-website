import numpy as np
import tensorflow as tf
from tensorflow import keras

from gym.wrappers.monitoring.video_recorder import VideoRecorder

from helper_functions import instantiate_environmnent


def create_video(model_path, video_name, n_episodes=10):
    env = instantiate_environmnent()

    model = keras.models.load_model(model_path)
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
    create_video(n_episodes=10, model_path='model2', video_name='model.mp4')
