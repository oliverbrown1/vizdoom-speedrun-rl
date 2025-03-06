# coding: utf-8

img_dim = 80
a_size = 4
ACTION_DIM = 9
RNN_DIM = 256
new_img_dim = (108, 60)


filepath="maze_demo_4"

# hyperparameters we control

max_episodes = 1100
num_workers= 8
living_reward = -0.001
gamma=0.99
lr=1e-5
frameskip=4
gradient_clip_val = 20
episode_timeout_steps = 2000

