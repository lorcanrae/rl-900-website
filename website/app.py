import streamlit as st
# import tensorflow as tf
from tensorflow import keras
import numpy as np

import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack


LOCAL_MODEL = r'../modelling/saved-models/Student_Teacher_BestModel-ep500.h5'

### Helper functions

def instantiate_environmnent():
    env = gym.make("SpaceInvadersNoFrameskip-v4")
    env = AtariPreprocessing(env, grayscale_newaxis=True, frame_skip=5)
    env = FrameStack(env, 4)

    return env

def load_model(model_h5=LOCAL_MODEL):
    model = keras.models.load_model(model_h5)
    return model

### End Helper Funtions

# Instantiate

env = instantiate_environmnent()
model = load_model()

st.set_page_config(layout='wide')


### Sidebar

# st.sidebar.title('BIGCHAMP-900')

# st.sidebar.markdown("""## Time to save the world!""")
# episodes = st.sidebar.expander('N games').slider('', 1, 10, 5)
# start_model = st.sidebar.button("BLAST OFF!")
# stop_model = st.sidebar.button("Stand down Champ!")

st.sidebar.markdown('''
                Team:
                - Alex Gribius
                - Dan Hawkins
                - Alberto Lopez Rueda
                - Lorcan Rae

                With thanks to Oliver Giles
                ''')


### Main

col1, col2 = st.columns([1, 1])

with col2:
    st.title('BIGCHAMP-900')
    st.markdown("""## Time to save the world!""")
    episodes = st.slider('N Games', 1, 10, 5)
    start_model = st.button("BLAST OFF!")
    stop_model = st.button("Stand down Champ!")

DISPLAY_WIDTH = 420

with col1:

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
