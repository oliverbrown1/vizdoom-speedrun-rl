# coding: utf-8

img_dim = 80
a_size = 4
ACTION_DIM = 9
RNN_DIM = 256
new_img_dim = (108, 60)

# files1 gamma = 0.8
# files2 show difference between all actions and limited actions
# files3 show difference between 32 steps and max steps
# files4 

filepath="/files1"

# hyperparameters we control

max_episodes = 1000
num_workers= 8
living_reward = -0.001
gamma=0.9
lr=1e-5
frameskip=4

