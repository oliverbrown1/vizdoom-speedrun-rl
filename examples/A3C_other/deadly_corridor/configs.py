# coding: utf-8

img_dim = 80
a_size = 4
ACTION_DIM = 9
RNN_DIM = 256
new_img_dim = (108, 60)

# files1 gamma = 0.95
# files2 gamma = 0.9

filepath="/files2"

# hyperparameters we control

max_episodes = 1000
num_workers= 8
living_reward = -0.001
gamma=0.9
lr=1e-5
frameskip=4

