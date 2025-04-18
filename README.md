Project code for training agents to speedrun in various scenarios in ViZDoom.

### Installation

To get started, I advise using Anaconda/Conda environment as there are a lot of dependencies. Since this code primarily uses Python 3.7 which is no longer maintained and cannot be used to pip install ViZDoom, it must be built from source.

Refer to this guide for more details: https://vizdoom.farama.org/introduction/building/

A3C and DQN implementations are located under the algorithms directory.

To run each agent, in A3C you run ``deadly_corridor.py`` and in DQN you run ``main.py``, each with their respectice config files you can use to adjust the scenario and parameters.

To monitor the progress, do ``tensorboard --logdir <dir_name>`` where ``<dir_name>`` is the directory specified for logging to in the config file. 

#### Overview

There are also gifs that are created to watch the agent play.

#### Dependencies

* Python <= 3.7.* 
* ViZDoom >= 1.1.8
* Tensorflow == 1.15.0 
* Tensorboard == 1.15.0
* Keras == 2.3.1
* Seaborn == 0.12.0
* Scipy == 1.73.0
* Moviepy == 1.0.3
* Matplotlib == 3.5.3




