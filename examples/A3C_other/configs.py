# coding: utf-8

img_dim = 80
a_size = 4
ACTION_DIM = 9
RNN_DIM = 256
new_img_dim = (108, 60)

# maze_1 -> 

filepath="/maze_1"

# hyperparameters we control

max_episodes = 100
num_workers= 8
living_reward = -0.001
gamma=0.9
lr=1e-5
frameskip=4

