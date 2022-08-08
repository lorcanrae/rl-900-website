# Reinforcement Learning - BIGCHAMP900

Training an AI Agent to play Atari Space Invaders.

An agent was trained using a Duelling DQN model, using an established model (IQN) as
a Teacher during exploration. The agents where trained using Google Cloud Platforms (GCP)
VertexAI and the final model was embedded into a website to play in real time while I presented the
project to my cohort.

The Agents highest score achieved was 665, see below for our agent playing Space Invaders.

<p align="center">
  <img width="200" height="300" src="https://github.com/lorcanrae/rl-900-website/blob/master/saved_media/weekendmodel-model-e2-s665-30f.gif?raw=true">
</p>

A project by Lorcan Rae, Alexander Gribius, Daniel Hawkins, Alberto Lopez Rueda for Le Wagon Batch #900 in London.

## Models

TODO

### Local Minima

All agents fell into local minima - resulting in Agents performing the same strategies repeatedly.
I believe this is primarily due to not training for long enough.

## Training

Training was done on GCP VertexAI Virtual Machine in a Jupyter Notebook. I personally don't think this was great, but I
can see why our Le Wagon teachers pushed us heavily in this direction to save time.

## Known Issues

There where known issues that could have been handled differently:
- Time - the project was done in 8 days, reinforcement models take a huge amount of time (and/or processing power)
to train. All of our trained Agents appeared to fall into local minima.
- Content - Reinforcement Learning was outside of the scope of Le Wagon's syllabus, we learnt the subject matter and
applied it in less than two weeks.
- Frame Stacking - there was an error in frame shape and stacking. As part of preprocessing
frames are scaled and greyscaled to a single colour channel resulting in a frame shape of (84, 84, 1).
To give the model a senses of temporality, four frames are stacked together, which should have resulted in an frame
shape of (4, 84, 84, 1) then taking an element wise maximum of the four stacked frames as input to the model.
We could not get this working with our existing framework and didn't have time to adequately diagnose and remedy.
- Dependency conflicts - the package used to simulate Atari, [Gym](https://www.gymlibrary.ml/), has conflicts with
a number of other packages. If any frame or video capture is required, Gym needs a renderer (like a graphics card),
painful to do if using a WSL development environment.
- TensorFlow and PyTorch - the Duelling DDQN models where written using TensorFlow, the IQN teacher model was written
in PyTorch. Ideally they would have used the same package.
- Incentive Structures - The only incentive structure agents where trained on was the default scoring system,
i.e. 5 points for a first row alien, 10 points for a second row alien, up to 30 points for the sixth row aliens and 200
points for killing the mother-ship. We noticed that there was a number of agents that tried to kill the mother-ship to
their own detriment. Time permitting, it would have been interesting to experiment with custom incentive structures, for
example, prioritizing killing lower level aliens to buy more time to clear the screen.

## Tools

This project was written in Python using Numpy, TensorFlow, PyTorch. Models were trained
on GCP VertexAI inside jupyter notebooks and the front end was created and deployed with Streamlit.

Python: Numpy, TensorFlow, PyTorch, Streamlit
Google Cloud Platform: VertexAI

<p align='center', float='left'>
  <img src='https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg' width='50'>
  <img src='https://raw.githubusercontent.com/devicons/devicon/1119b9f84c0290e0f0b38982099a2bd027a48bf1/icons/numpy/numpy-original.svg' width='50'>
  <img src='https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg' width='50'>
  <img src='https://raw.githubusercontent.com/devicons/devicon/1119b9f84c0290e0f0b38982099a2bd027a48bf1/icons/pytorch/pytorch-original.svg' width='50'>
  <img src='https://www.vectorlogo.zone/logos/google_cloud/google_cloud-icon.svg' width='50'>
  <img src='https://streamlit.io/images/brand/streamlit-mark-color.png' width='50'>
</p>
