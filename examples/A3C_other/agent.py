# coding: utf-8
# implement of agent
import time
import random
import pickle
import itertools as iter

import numpy as np
import tensorflow as tf

from utils import *


import matplotlib
gui_env = [i for i in matplotlib.rcsetup.interactive_bk]
for gui in gui_env:
    print("testing", gui)
    try:
        matplotlib.use(gui, warn=False, force=True)
        from matplotlib import pyplot as plt
        print("Using ..... ", matplotlib.get_backend())
    except:
        print("    ", gui, "Not found")

from vizdoom import *

import utils
import network
import configs as cfg


class Agent(object):
    """
    Agent
    """
    def __init__(self, game, name, s_size, a_size, optimizer=None, model_path=None, global_episodes=None, play=False):

        self.summary_step = 3

        self.name = "worker_" + str(name)
        self.number = name

        self.global_step_count = 0
        self.episode_reward = []
        self.episode_step_counts = []
        self.episode_episode_total_pickes = []
        # self.episode_lengths = []
        # self.episode_mean_values = []
        # self.episode_health = []
        # self.episode_kills = []

        # Create the local copy of the network and the tensorflow op to
        # copy global parameters to local network
        if not play:
            self.model_path = model_path
            self.trainer = optimizer
            self.global_episodes = global_episodes
            self.increment = self.global_episodes.assign_add(1)
            self.local_AC_network = network.ACNetwork(self.name, optimizer, play=play, shape=cfg.new_img_dim)
            self.summary_writer = tf.compat.v1.summary.FileWriter(f"./{cfg.filepath}/summaries/agent_%s" % str(self.number))
            self.update_local_ops = tf.group(*utils.update_target_graph('global', self.name))
        else:
            self.local_AC_network = network.ACNetwork(self.name, optimizer, play=play, shape=cfg.new_img_dim)
        if not isinstance(game, DoomGame):
            raise TypeError("Type Error")

        # The Below code is related to setting up the Doom environment
        game = DoomGame()
        # game.set_doom_scenario_path('../scenarios/deadly_corridor.cfg')
        # game.load_config("./maps/basic.cfg")
        import os
        current_working_directory = os.getcwd()
        print(current_working_directory)
        game.set_doom_scenario_path("../../maps/my_way_home.wad")
        # game.set_doom_map("map01")
        # needed for custom maze scenarios
        # game.set_doom_map("map02")
        game.set_screen_resolution(ScreenResolution.RES_400X225)
        game.set_screen_format(ScreenFormat.RGB24)
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        # Enables labeling of the in game objects.
        game.set_labels_buffer_enabled(True)
        # game.add_available_button(Button.MOVE_LEFT)
        # game.add_available_button(Button.MOVE_RIGHT)
        game.add_available_button(Button.MOVE_FORWARD)
        # game.add_available_button(Button.MOVE_BACKWARD)
        game.add_available_button(Button.SPEED)
        game.add_available_button(Button.TURN_LEFT)
        game.add_available_button(Button.TURN_RIGHT)
        # game.add_available_button(Button.ATTACK)
        # game.add_available_game_variable(GameVariable.USER1)
        game.set_episode_timeout(cfg.episode_timeout_steps)
        game.set_episode_start_time(10)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_living_reward(cfg.living_reward)
        game.set_mode(Mode.PLAYER)
        if play:
            # game.add_game_args("+viz_render_all 1")
            game.set_render_hud(False)
            game.set_ticrate(35)
        game.init()
        self.env = game
        self.actions = self.button_combinations()

    def infer(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = utils.discount(rewards_plus, gamma)[:-1]
        value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * value_plus[1:] - value_plus[:-1]
        advantages = utils.discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {
            self.local_AC_network.target_v: discounted_rewards,
            self.local_AC_network.inputs: np.stack(observations),
            self.local_AC_network.actions: actions,
            self.local_AC_network.advantages: advantages
        }
        l, v_l, p_l, e_l, g_n, v_n, _ = sess.run([
                                            self.local_AC_network.loss,
                                            self.local_AC_network.value_loss,
                                            self.local_AC_network.policy_loss,
                                            self.local_AC_network.entropy,
                                            self.local_AC_network.grad_norms,
                                            self.local_AC_network.var_norms,
                                            self.local_AC_network.apply_grads],
                                            feed_dict=feed_dict)
        return l / len(rollout), v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def train_a3c(self, max_episode_length, gamma, sess, coord, saver):
        if not isinstance(saver, tf.train.Saver):
            raise TypeError('saver should be tf.train.Saver')

        episode_count = sess.run(self.global_episodes)
        start_t = time.time()
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)  # update local ops in every episode
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False

                # last_total_health = 100
                # last_total_ammo2 = 52  # total is 52
                # last_total_kills = 0
                self.env.new_episode()
                episode_st = time.time()

                while not self.env.is_episode_finished():

                    s = self.env.get_state().screen_buffer
                    episode_frames.append(s)
                    s = utils.process_frame(s, cfg.new_img_dim)
                    # Take an action using probabilities from policy network output.
                    a_dist, v = sess.run([self.local_AC_network.policy, self.local_AC_network.value],
                                         feed_dict={self.local_AC_network.inputs: [s]})
                    # get a action_index from a_dist in self.local_AC.policy
                    a_index = self.choose_action_index(a_dist[0], deterministic=False)
                    # make an action
                    move_reward = self.env.make_action(self.actions[a_index], cfg.frameskip)
                    self.global_step_count += 1

                    # ammo2_delta = self.env.get_game_variable(GameVariable.AMMO2) - last_total_ammo2
                    # last_total_ammo2 = self.env.get_game_variable(GameVariable.AMMO2)

                    # health_delta = self.env.get_game_variable(GameVariable.HEALTH) - last_total_health
                    # last_total_health = self.env.get_game_variable(GameVariable.HEALTH)

                    # health_reward = self.health_reward_function(health_delta)
                    # ammo2_reward = self.ammo2_reward_function(ammo2_delta)

                    # kill_reward, last_total_kills = self.kills_reward_function(last_total_kills)

                    # reward = move_reward + health_reward + ammo2_reward + kill_reward
                    episode_reward += move_reward

                    d = self.env.is_episode_finished()
                    if d:
                        s1 = s
                    else:  # game is not finished
                        s1 = self.env.get_state().screen_buffer
                        episode_frames.append(s1)
                        s1 = utils.process_frame(s1, cfg.new_img_dim)

                    episode_buffer.append([s, a_index, move_reward, s1, d, v[0, 0]])
                    episode_values.append(v[0, 0])
                    # summaries information
                    s = s1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    # if len(episode_buffer) == 64 and d is False and episode_step_count != max_episode_length - 1:
                    # A3C and n-step q-learning -> instead of updating the network every n steps, we update it at the end of episode
                    # because reward can only be earned at the end of the episode for basic_scenario.wad
                    if d:
                        # Since we don't know what the true final return is,
                        # we "bootstrap" from our current value estimation.
                        v1 = sess.run(self.local_AC_network.value,
                                      feed_dict={self.local_AC_network.inputs: [s]})[0, 0]
                        l, v_l, p_l, e_l, g_n, v_n = self.infer(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d is True:
                        # self.episode_health.append(self.env.get_game_variable(GameVariable.HEALTH))
                        print('{}, episode #{}, reward: {}, steps:{}, time costs:{}'.format(
                            self.name, episode_count,
                            episode_reward, episode_step_count, time.time()-episode_st))
                        break

                # summaries
                self.episode_reward.append(episode_reward)
                self.episode_step_counts.append(episode_step_count)
                # self.episode_episode_total_pickes.append(last_total_health)
                # self.episode_kills.append(last_total_kills)
                # self.episode_lengths.append(episode_step_count)
                # self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    l, v_l, p_l, e_l, g_n, v_n = self.infer(episode_buffer, sess, gamma, 0.0)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and self.name == 'worker_0':
                    if episode_count % 1000 == 0:
                        saver.save(sess, self.model_path+'/model-'+str(episode_count)+'.ckpt')
                        print("Episode count {}, saved Model, time costs {}".format(episode_count, time.time()-start_t))
                        start_t = time.time()
                        time_per_step = 0.005
                        # time_per_step = 0.05
                        images = np.array(episode_frames)
                        # print("hi")
                        make_gif(images,f'./{cfg.filepath}/frames/image'+str(episode_count)+'.gif',
                            duration=len(images)*time_per_step,true_image=True,salience=False)

                    # mean_picked = np.mean(self.episode_episode_total_pickes[-5:])
                    mean_reward = np.mean(self.episode_reward[-5:])
                    mean_steps = np.mean(self.episode_step_counts[-5:])
                    # mean_health = np.mean(self.episode_health[-5:])
                    # mean_length = np.mean(self.episode_lengths[-5:])
                    # mean_value = np.mean(self.episode_mean_values[-5:])
                    # mean_kills = np.mean(self.episode_kills[-5:])
                    summary = tf.compat.v1.Summary()
                    summary.value.add(tag='Steps per episode', simple_value=mean_steps)
                    summary.value.add(tag='Mean Reward Against Episodes', simple_value=mean_reward)
                    # summary.value.add(tag='Performance/Health', simple_value=mean_health)
                    # summary.value.add(tag='Performance/Kills', simple_value=mean_kills)
                    # summary.value.add(tag='Performance/Steps', simple_value=mean_length)
                    # summary.value.add(tag='Performance/Mean Value', simple_value=mean_value)
                    # summary.value.add(tag='Losses/Total Loss', simple_value=l)
                    # summary.value.add(tag='Losses/Value Loss', simple_value=v_l)
                    # summary.value.add(tag='Losses/Policy Loss', simple_value=p_l)
                    # summary.value.add(tag='Losses/Entropy', simple_value=e_l)
                    # summary.value.add(tag='Losses/Grad Norm', simple_value=g_n)
                    # summary.value.add(tag='Losses/Var Norm', simple_value=v_n)
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
                if episode_count == cfg.max_episodes:  # thread to stop
                    print("Stop training name:{}".format(self.name))
                    coord.request_stop()

    import time
    def play_game(self, sess, episode_num):
        if not isinstance(sess, tf.Session):
            raise TypeError('saver should be tf.train.Saver')

        for i in range(episode_num):


            self.env.new_episode()
            state = self.env.get_state()
            s = utils.process_frame(state.screen_buffer, cfg.new_img_dim)
            episode_rewards = 0
            last_total_shaping_reward = 0
            step = 0
            s_t = time.time()

            while not self.env.is_episode_finished():
                state = self.env.get_state()
                s = utils.process_frame(state.screen_buffer, cfg.new_img_dim)
                a_dist, v = sess.run([self.local_AC_network.policy, self.local_AC_network.value],
                                     feed_dict={self.local_AC_network.inputs: [s]})
                # get a action_index from a_dist in self.local_AC.policy
                a_index = self.choose_action_index(a_dist[0], deterministic=True)
                # make an action
                reward = self.env.make_action(self.actions[a_index])

                step += 1

                # shaping_reward = doom_fixed_to_double(self.env.get_game_variable(GameVariable.USER1)) / 100.
                # r = (shaping_reward - last_total_shaping_reward)
                # last_total_shaping_reward += r

                episode_rewards += reward

                print('Current step: #{}'.format(step))
                print('Current action: ', self.actions[a_index])
                # print('Current health: ', self.env.get_game_variable(GameVariable.HEALTH))
                # print('Current kills: {0}'.format(self.env.get_game_variable(GameVariable.KILLCOUNT)))
                print('Current reward: {0}'.format(reward))
                # state = self.env.get_state()
                # s = utils.process_frame(state.screen_buffer, cfg.new_img_dim)
                time.sleep(0.1)
            print('End episode: {}, Total Reward: {}, {}'.format(i, episode_rewards, last_total_shaping_reward))
            print('time costs: {}'.format(time.time() - s_t))
            time.sleep(5)

    @staticmethod
    def choose_action_index(policy, deterministic=False):
        if deterministic:
            return np.argmax(policy)

        r = random.random()
        cumulative_reward = 0
        for i, p in enumerate(policy):
            cumulative_reward += p
            if r <= cumulative_reward:
                return i

        return len(policy) - 1

    def health_reward_function(self, health_delta):
        health, reward = self.env.get_game_variable(GameVariable.HEALTH), 0
        if health_delta == 0:
            reward = 0
        elif health_delta < 0:
            reward = -5.
        return reward

    @staticmethod
    def ammo2_reward_function(ammo2_delta):
        if ammo2_delta == 0:
            return 0
        elif ammo2_delta > 0:
            return ammo2_delta * 0.5
        else:
            return -ammo2_delta * 0.5

    def kills_reward_function(self, last_total_kills):
        kill_count = self.env.get_game_variable(GameVariable.KILLCOUNT)
        kill_delta = kill_count - last_total_kills
        reward = 0
        if kill_delta > 0:
            reward = kill_delta * 2.5
        return reward, kill_count

    @staticmethod
    def button_combinations():
        actions = []
        # m_left_right = [[True, False], [False, True], [False, False]]  # move left and move right
        # attack = [[True], [False]]
        # m_forward_backward_speed = [[True, False, True], [False, True, True], [False, False, False], [True, False, False], [False, True, False]]  # move forward and backward
        move_with_speed = [[True, False], [True, True], [False, False]]
        t_left_right = [[True, False], [False, True], [False, False]]  # turn left and turn right

    # for i in m_left_right:
        for j in move_with_speed:
        # for k in m_forward_backward_speed:
            for l in t_left_right:
                actions.append(j+l)
                        
        return actions
