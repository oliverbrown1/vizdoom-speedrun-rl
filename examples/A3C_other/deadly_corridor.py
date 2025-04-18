#!/usr/bin/env python3
# coding: utf-8
import time
import sys
import string
import threading

import os
import tensorflow as tf

from vizdoom import *

import agent
import network
import configs as cfg

tf.config.threading.set_intra_op_parallelism_threads(40)
tf.config.threading.set_inter_op_parallelism_threads(40)

max_episode_length = 1500
gamma = cfg.gamma  # discount rate for advantage estimation and reward discounting
s_size = 6400 # 80 * 80 * 1
a_size = 3  # Agent can move Left, Right, or Fire
num_workers = cfg.num_workers
load_model = False

model_path = f'./{cfg.filepath}/check_point'
frames_path = f'./{cfg.filepath}/frames'

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(frames_path):
    os.makedirs(frames_path)

def main_train(tf_configs=None):
    s_t = time.time()

    tf.reset_default_graph()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    with tf.device("/cpu:0"): 
        # before learning rate was 1e-6
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=cfg.lr)
        master_network = network.ACNetwork('global', optimizer, shape=cfg.new_img_dim)  # Generate global network
        agents = []
        # Create worker classes
        for i in range(num_workers):
            agents.append(agent.Agent(DoomGame(), i, s_size, a_size, optimizer, model_path, global_episodes))
    saver = tf.train.Saver(max_to_keep=100)

    with tf.Session(config=tf_configs) as sess:
        coord = tf.train.Coordinator()
        if load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.
        worker_threads = []
        for ag in agents:
            agent_train = lambda: ag.train_a3c(max_episode_length, gamma, sess, coord, saver)
            t = threading.Thread(target=(agent_train))
            t.start()
            time.sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)
    print("training ends, costs{}".format(time.time() - s_t))


def main_play(tf_configs=None):
    tf.reset_default_graph()

    with tf.Session(config=tf_configs) as sess:

        ag = agent.Agent(DoomGame(), 0, s_size, a_size, play=True)

        print('Loading Model...')
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, os.path.join(model_path, 'model-800.ckpt'))
        print('Successfully loaded!')

        ag.play_game(sess, 10)


if __name__ == '__main__':

    train = True
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if train:
        main_train(tf_configs=config)
    else:
        main_play(tf_configs=config)

