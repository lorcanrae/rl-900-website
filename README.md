# Reinforcement Learning - BIGCHAMP900

Training an AI Agent to play Atari Space Invaders.

An agent was trained using a Duelling DQN model, using an established model (IQN) as
a Teacher during exploration. The agents where trained using Google Cloud Platforms (GCP)
VertexAI and the final model was embedded into a website (currently unavailable) to play in real time while I presented the
project to my cohort.

The Agents highest score achieved was 665, see below for a video of our agent playing Space Invaders.

<p align="center">
  <img width="300" height="450" src="https://github.com/lorcanrae/rl-900-website/blob/master/saved_media/weekendmodel-model-e2-s665-30f.gif?raw=true">
</p>

A project by Lorcan Rae, Daniel Hawkins, Alberto Lopez Rueda and Alexander Gribius for Le Wagon Batch #900 in London.

<p align='center', float='left'>
  <img src='https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg' width='50'>
  <img src='https://raw.githubusercontent.com/devicons/devicon/1119b9f84c0290e0f0b38982099a2bd027a48bf1/icons/numpy/numpy-original.svg' width='50'>
  <img src='https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg' width='50'>
  <img src='https://raw.githubusercontent.com/devicons/devicon/1119b9f84c0290e0f0b38982099a2bd027a48bf1/icons/pytorch/pytorch-original.svg' width='50'>
  <img src='https://www.vectorlogo.zone/logos/google_cloud/google_cloud-icon.svg' width='50'>
  <img src='https://streamlit.io/images/brand/streamlit-mark-color.png' width='50'>
</p>

## Models

Two model architecture's are used to train agents, Deep Q Networks (DQN) and Dueling Deep Q Networks (Dueling DQN).
Q-Learning algoriths are based on state (**s**), action (**a**) and reward (**r**). With the goal to learn how much long term reward
it will receive for each state (**s**), action (**a**) pair, and act to either maximise or minimise that reward.

Two DQN and Dueling DQN models are leveraged to stabilise the learning process. The *main neural network* is used to estimate
the Q-values for the current state **s** and action **a**. The *second neural network* has the exact same architecture as the
main network, but it is used to estimate the Q-values of the next state and action, **s'** and **a'** respectively, based on the
current state.
All learning is done in the *main network*, the *target network* is frozen and the weights are transferred from *main* to *target*
at fixed intervals, usually 10,000 frames.

The Dueling DQN model differs in architecture from the DQN model by splitting the output, **Q**, into two separate parts, the
value function **V(s)** and the advantage function **A(s, a)**. The value function function tells us how much reward we will collect
from a given state **s**, and the advantage function tells us how much better one action is compared to other actions. In splitting
these layers out, states containing more *importance* to long term reward can be weighted more heavily.

<p align="center">
  <img width="800" height="600" src="https://github.com/lorcanrae/rl-900-website/blob/master/saved_media/dqn-dueldqn-model-arch.png?raw=true">
</p>
Top: DQN Model Architecture, Bottom: Dueling DQN Model Architecture

Image extract from Wang, Ziyu, et al. “Dueling network architectures for deep reinforcement learning.” arXiv preprint arXiv:1511.06581 (2015)

### Local Minima

All agents fell into local minima - resulting in Agents performing the same strategies repeatedly.
I believe this is primarily due to not training for long enough.

## Training

Training was done on GCP VertexAI Virtual Machine in a Jupyter Notebook. I personally don't think this was great, but I
can see why our Le Wagon teachers pushed us heavily in this direction to save time.

## Known Issues

- This package requires using gym ver 0.21.x, a requirement of the PyTorch teacher model. A change to gyms source
code is required in `venv/versions/lib/python_version/site-packages/gym/wrappers/atari_preprocessing.py` because of changes
to numpy.

```python
self.env.unwrapped.np_random.randint(1, self.noop_max + 1)
# Needs to be changed to:
self.env.unwrapped.np_random.integers(1, self.noop_max + 1)
```

- Time - the project was done in 8 days, reinforcement models take a non-trivial amount of time (and/or processing power)
to train. All of our trained Agents appeared to fall into local minima.
- Content - Reinforcement Learning was outside of the scope of Le Wagon's syllabus, we learnt the subject matter and
applied it in less than two weeks.
- Frame Stacking - there was an error in frame shape and stacking. As part of preprocessing
frames are scaled and greyscaled to a single colour channel resulting in a frame shape of (84, 84, 1).
To give the model a senses of temporality, four frames are stacked together, which should have resulted in an frame
shape of (4, 84, 84, 1) then taking an element wise maximum of the four stacked frames as input to the model with shape (84, 84, 1).
We could not get this working with our existing framework and didn't have time to adequately diagnose and remedy.
- Dependency conflicts - the package used to simulate Atari, [Gym](https://www.gymlibrary.ml/), has conflicts with
a number of other packages, primarily it can be touchy with TensorFlow and PyTorch. If any frame or video capture is required, Gym needs a renderer (like a graphics card), not conducive in a WSL development environment.
- TensorFlow and PyTorch - the DQN and Dueling DQN models where written using TensorFlow, the IQN teacher model was written
in PyTorch. Ideally all models would have been developed using the same package.
- Incentive Structures - The only incentive structure agents where trained on was the default scoring system,
i.e. 5 points for a first row alien, 10 points for a second row alien, up to 30 points for the sixth row aliens and 200
points for killing the mother-ship. We noticed that there was a number of agents that tried to kill the mother-ship to
their own detriment. It would have been interesting to experiment with custom incentive structures to see the
impact on the agents behaviour.


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
