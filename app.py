import streamlit as st

from tensorflow import keras
import numpy as np

import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack

LOCAL_MODEL = r'saved_models/Student_Teacher_BestModel-ep3250.h5'

### Helper functions

def instantiate_environmnent():
    env = gym.make("SpaceInvadersNoFrameskip-v4")
    env = AtariPreprocessing(env, grayscale_newaxis=True, frame_skip=5)
    env = FrameStack(env, 4)
    return env

def load_model(model_h5=LOCAL_MODEL):
    model = keras.models.load_model(model_h5)
    return model

# Instantiate environment and model

env = instantiate_environmnent()
model = load_model()

# Streamlit

st.set_page_config(layout='wide')

### Sidebar

# TODO: add sidebar to have a separate page with more detailed about

### Main

col1, col2, col3 = st.columns([1, 1, 1])

DISPLAY_WIDTH = 420

with col1:
    st.title('BIGCHAMP-900')
    st.markdown('''## Watch Bigchamp play in real-time!''')
    # st.markdown("""## Time to save the world!""")
    episodes = st.slider('N Games', 1, 10, 10)
    start_model = st.button("BLAST OFF!")
    stop_model = st.button("Ease up Champ!")

with col2:
    if start_model:
        with st.empty():
            for episode in range(1, episodes+1):
                state = np.asarray(env.reset()).reshape(84, 84, 4)
                done = False
                score = 0

                while not done:
                    batch_state = np.expand_dims(state, 0)
                    action = np.argmax(model.predict(batch_state)[0])
                    state, reward, done, info = env.step(action)
                    state = np.asarray(state).reshape(84, 84, 4)
                    score += reward
                    st.image(env.render(mode='rgb_array'), width=DISPLAY_WIDTH)

                    if stop_model:
                        break

    else:
        with st.empty():
            state = np.asarray(env.reset()).reshape(84, 84, 4)
            st.image(env.render(mode='rgb_array'), width=DISPLAY_WIDTH)

with col3:
    st.markdown('''## About''')
    st.markdown('''Bigchamp was trained on a Dueling Double DQN for 3250 games using the IQN model (trained on over 300,000 games) for teaching during exploration.''')
    st.markdown('''
                Team:
                - Alexander Gribius
                - Dan Hawkins
                - Alberto Lopez Rueda
                - Lorcan Rae
                With thanks to Oliver Giles
                ''')
