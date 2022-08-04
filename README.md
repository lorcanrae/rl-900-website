# Reinforcement Learning - BIGCHAMP900

Training an AI Agent to play Atari Space Invaders.

An agent was trained using a Duelling Double DQN (Duelling DDQN) model, using an established
model (IQN) as a Teacher during exploration. The agents where trained using Google Cloud Platforms (GCP)
VertexAI and the final model was embedded into a website to play in real time while I presented the
project to my cohort.

A project by Lorcan Rae, Alexander Gribius, Daniel Hawkins, Alberto Lopez Rueda for Le Wagon Batch #900 in London.

<p float='left'>
  <img src='https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg' width='60'>
  <img src='https://raw.githubusercontent.com/devicons/devicon/1119b9f84c0290e0f0b38982099a2bd027a48bf1/icons/numpy/numpy-original.svg' width='60'>
  <img src='https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg' width='60'>
  <img src='https://raw.githubusercontent.com/devicons/devicon/1119b9f84c0290e0f0b38982099a2bd027a48bf1/icons/pytorch/pytorch-original.svg' width='60'>
  <img src='https://www.vectorlogo.zone/logos/google_cloud/google_cloud-icon.svg' width='60'>
  <img src='https://streamlit.io/images/brand/streamlit-mark-color.png' width='60'>

## Model

TODO

### Local Minima

All agents fell into local minima - resulting in performing continuous strategies. We believe this is primarily due to
inadequate training.

## Training

TODO

## Known Issues

There where known issues that could have been handled differently:
- Time - the project was done in 8 days, reinforcement models take a huge amount of time (and/or processing power)
to train.
- Content - Reinforcement Learning was outside of the scope of Le Wagon's teaching syllabus, we were required to
learn the subject matter and apply it.
- Frame Stacking - there was an error in frame shape and stacking as input into the model. As part of preprocessing
frames are scaled and greyscaled to a single colour channel resulting in a frame shape of (84, 84, 1).
To give the model a senses of temporality, four frames are stacked together, which should have resulted in an frame
shape of (4, 84, 84, 1) then taking an element wise maximum of the four stacked frames as input to the model.
We could not get this working with our existing framework and didn't have time to adequately diagnose and remedy.
- Dependency conflicts - the package used to simulate Atari, [Gym](https://www.gymlibrary.ml/), has conflicts with
a number of other packages. If any capture is required, Gym needs to access a renderer or graphics card,
painful to do if using a WSL development environment.
- TensorFlow and PyTorch - Duelling DDQN models where written using TensorFlow, the IQN teacher model was written
in PyTorch. It would have been ideal to keep it simple and use only a single neural net package, TensorFlow _or_ Pytorch.

## Tools

This project was written in Python using Numpy, TensorFlow, PyTorch. The models were trained
on GCP VertexAI inside jupyter notebooks and the front end was created and deployed with Streamlit.

Python: Numpy, TensorFlow, PyTorch, Streamlit
Google Cloud Platform: VertexAI
