# coding: utf-8

img_dim = 80
a_size = 4
ACTION_DIM = 9
RNN_DIM = 256
new_img_dim = (108, 60)

# maze_1 -> hyperparameters of basic_scenario
# maze_2 -> gamma = 0.99, lr=1e-6, gradient_clip_val = 20
# maze_3 -> gamma = 0.95, lr=1e-4, living_reward = -0.0001
# maze_4 -> lr=1e-6, gamma = 0.99, living_reward = -0.001, network_update_interval = 64 steps
# maze_5 -> lr=1e-5, gradient_clip_val=10
# maze_6 -> change episode timeout to 2000 steps
# maze_7 -> 1000 steps
# maze_8 -> 4000 steps
# maze_9 -> 5000 steps
# maze_10 -> 4000 steps, lr=1e-5, gradient_clip_val = 20
# maze_11 -> lr = 1e-4
# maze_12 -> lr=1e-5, gamma = 0.95
# maze_13 -> maze_10 (gamma = 0.99) and modified reward system (armour bonus +1 -> +0.1)

filepath="maze_13"

# hyperparameters we control

max_episodes = 1000
num_workers= 8
living_reward = -0.001
gamma=0.99
lr=1e-5
frameskip=4
gradient_clip_val = 20
episode_timeout_steps = 4000

