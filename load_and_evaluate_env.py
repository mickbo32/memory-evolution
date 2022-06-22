import dill  # pickle extension
import logging
import os
import pickle
from pprint import pprint
import random  # neat uses random  # todo: allow seeding in neat
import sys
import time
import typing
from typing import Literal, Optional

import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import neat
import numpy as np
from numpy.random import SeedSequence
import pandas as pd

from gym.utils.env_checker import check_env  # from stable_baselines.common.env_checker import check_env

import memory_evolution
from memory_evolution.agents import RandomActionAgent, RnnNeatAgent, CtrnnNeatAgent
from memory_evolution.envs import BaseForagingEnv, MazeForagingEnv, TMaze, RadialArmMaze
from memory_evolution.evaluate import evaluate_agent
from memory_evolution.logging import set_main_logger

from memory_evolution.load import load_env, load_agent, get_checkpoint_number, AVAILABLE_LOADING_METHODS


if __name__ == '__main__':

    # matplotlib settings:
    isRunningInPyCharm = "PYCHARM_HOSTED" in os.environ
    if isRunningInPyCharm:
        mpl.use('Qt5Agg')  # Change matplotlib backend to show correctly in PyCharm.

    # ----- Settings -----
    RENDER = True  # False  # render or just save gif files
    # ---
    # LOAD_AGENT = '8539704_2022-05-19_163834.593420+0000'
    # LOAD_AGENT = '8541080_2022-05-20_214011.159858+0000'
    # LOAD_AGENT = '8547986_2022-05-27_131437.320058+0000'
    # LOAD_AGENT_DIR = "logs/saved_logs/outputs-link/no-date0/logs/"
    # LOAD_AGENT = '8565200-8_2022-06-11_103533.997939+0000'  # first_arm_accuracy, target_reached_rate, fitness: 0.625; 1; -83.55;
    # LOAD_AGENT = '8564839-1_2022-06-10_151155.886805+0000'  # first_arm_accuracy, target_reached_rate, fitness: 0.995; 0.995; -82.25;
    # LOAD_AGENT = '8564839-18_2022-06-10_161004.988468+0000'  # first_arm_accuracy, target_reached_rate, fitness: 1; 1; -76.25;
    # LOAD_AGENT_DIR = "logs/saved_logs/outputs-link/2022-06-13_training_allocentric_90/logs/"
    # LOAD_AGENT = '8565845-1_2022-06-12_093058.453153+0000'  # first_arm_accuracy, target_reached_rate, fitness: 0.99; 0.995; -71.8;
    LOAD_AGENT = '8566609-28_2022-06-13_115413.598940+0000'  # first_arm_accuracy, target_reached_rate, fitness: 1; 1; -68.55;
    LOAD_AGENT_DIR = "logs/saved_logs/outputs-link/2022-06-13_training_egocentric_90/logs/"
    # LOAD_FROM: AVAILABLE_LOADING_METHODS = 'checkpoint'
    LOAD_FROM: AVAILABLE_LOADING_METHODS = 'pickle'
    N_EPISODES = 12  # 0  # 5
    LOGGING_DIR = 'logs'
    # ---
    # CHECKPOINT_NUMBER = None  # if None, load the last checkpoint
    CHECKPOINT_NUMBER = None
    # ---
    # override variables if provided as program arguments
    if len(sys.argv) == 1:
        pass
    elif 2 <= len(sys.argv) <= 5:
        LOAD_AGENT = sys.argv[1]
        if len(sys.argv) >= 3:
            LOAD_AGENT_DIR = sys.argv[2]
        elif len(sys.argv) >= 4:
            LOAD_FROM = sys.argv[3]
        elif len(sys.argv) >= 5:
            N_EPISODES = sys.argv[4]
    else:
        raise RuntimeError(sys.argv)

    # logging settings:
    LOGGING_DIR, UTCNOW = set_main_logger(file_handler_all=None,
                                          logging_dir=LOGGING_DIR,
                                          stdout_handler=logging.INFO - 2,
                                          file_handler_now_filename_fmt=LOAD_AGENT + "_LOADED__now_{utcnow}.log")
    logging.info(__file__)
    LOADED_UTCNOW = LOAD_AGENT
    if LOAD_FROM == 'checkpoint':
        CHECKPOINT_NUMBER = get_checkpoint_number(LOAD_AGENT, LOAD_AGENT_DIR, LOAD_FROM, CHECKPOINT_NUMBER)
        LOADED_UTCNOW += f'_checkpoint-{CHECKPOINT_NUMBER}'
    LOADED_UTCNOW += '_LOADED_AGENT___now_' + UTCNOW
    logging.info(LOADED_UTCNOW)

    # neat random seeding:
    # random.seed(42)
    logging.debug(random.getstate())
    # Use random.setstate(state) to set an old state, where 'state' have been obtained from a previous call to getstate().

    # ----- ENVIRONMENT -----

    # env = RadialArmMaze(corridor_width=.2,
    #                     window_size=200, seed=4242, agent_size=.075, food_size=.05, n_food_items=1,
    #                     # init_agent_position=(.5, .1), init_food_positions=((.9, .5),),
    #                     init_agent_position=(.9, .5), init_food_positions=((.5, .9),),
    #                     vision_depth=.2, vision_field_angle=135, max_steps=400, vision_resolution=8)

    env = load_env(LOAD_AGENT, LOAD_AGENT_DIR)
    # test trained agents in env without landmarks: TODO
    '''
    from logs:
    allo:
    base_foraging  :  env: memory_evolution.envs.radial_arm_maze.RadialArmMaze(arms=4, corridor_width=0.2, window_size=200, env_size=1.0, kwargs={'agent_size': 0.075, 'food_size': 0.05, 'n_food_items': 1, 'max_steps': 400, 'vision_depth': 0.2, 'vision_field_angle': 135, 'vision_resolution': 3, 'vision_channels': 3, 'vision_point_radius': 0.04, 'agent_color': array([  0, 255, 255], dtype=uint8), 'background_color': array([0, 0, 0], dtype=uint8), 'outside_color': array([255,   0,   0], dtype=uint8), 'food_color': array([  0, 200,  55], dtype=uint8), 'food_visible': False, 'random_init_agent_position': ((0.5, 0.1), (0.5, 0.9), (0.1, 0.5)), 'init_food_positions': ((0.9, 0.5),), 'landmark_size': 0.25, 'init_landmarks_positions': ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)), 'landmarks_colors': (array([255,   0, 255], dtype=uint8), array([255, 255,   0], dtype=uint8), array([255, 127, 127], dtype=uint8), array([255, 255, 255], dtype=uint8)), 'borders': None, 'pairing_init_food_positions': None, 'rotation_step': 15.0, 'forward_step': 0.01, 'observation_noise': None, 'init_agent_position': None, 'inverted_color_rendering': True, 'fps': None, 'seed': None, 'platform': None})
    ego:
    base_foraging  :  env: memory_evolution.envs.radial_arm_maze.RadialArmMaze(arms=4, corridor_width=0.2, window_size=200, env_size=1.0, kwargs={'agent_size': 0.075, 'food_size': 0.05, 'n_food_items': 1, 'max_steps': 400, 'vision_depth': 0.2, 'vision_field_angle': 135, 'vision_resolution': 3, 'vision_channels': 3, 'vision_point_radius': 0.04, 'agent_color': array([  0, 255, 255], dtype=uint8), 'background_color': array([0, 0, 0], dtype=uint8), 'outside_color': array([255,   0,   0], dtype=uint8), 'food_color': array([  0, 200,  55], dtype=uint8), 'food_visible': False, 'random_init_agent_position': ((0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)), 'pairing_init_food_positions': (((0.9, 0.5),), ((0.1, 0.5),), ((0.5, 0.1),), ((0.5, 0.9),)), 'landmark_size': 0.25, 'init_landmarks_positions': ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)), 'landmarks_colors': (array([255,   0, 255], dtype=uint8), array([255, 255,   0], dtype=uint8), array([255, 127, 127], dtype=uint8), array([255, 255, 255], dtype=uint8)), 'borders': None, 'rotation_step': 15.0, 'forward_step': 0.01, 'observation_noise': None, 'init_agent_position': None, 'init_food_positions': None, 'inverted_color_rendering': True, 'fps': None, 'seed': None, 'platform': None})
    '''

    # ----- AGENT -----

    agent, other_loads = load_agent(LOAD_AGENT, LOAD_AGENT_DIR, LOAD_FROM, CHECKPOINT_NUMBER)
    agent.set_env(env)
    config = other_loads['config']

    # ----- MAIN LOOP -----
    # Evolve, interact, repeat.

    print('Evaluating agent ...\n')

    # to do: tests of evaluate_agent: episodes=1 | episodes=2 ;

    RANDOM_AGENT_UTCNOW = 'RandomActionAgent_' + UTCNOW
    # evaluate_agent(RandomActionAgent(env), env, episodes=2, render=True,
    #                save_gif=True,
    #                save_gif_dir=os.path.join(LOGGING_DIR, 'frames_' + RANDOM_AGENT_UTCNOW),
    #                save_gif_name=RANDOM_AGENT_UTCNOW + '.gif')

    if N_EPISODES > 0:
        pass
        # evaluate_agent(agent, env, episodes=2, render=True,
        #                save_gif=False)
        # env.close(); sys.exit()
        # evaluate_agent(agent, env, episodes=N_EPISODES, render=RENDER,
        #                save_gif=True,
        #                save_gif_dir=os.path.join(LOGGING_DIR, 'frames_' + LOADED_UTCNOW),
        #                save_gif_name=LOADED_UTCNOW + '.gif')
        # env.close(); sys.exit()
        evaluate_agent(agent, env, episodes=N_EPISODES, render=RENDER,
                       save_gif=True,
                       save_gif_name=os.path.join(LOGGING_DIR, LOADED_UTCNOW + '_frames.gif'))
        # Note: if you run twice evaluate_agent with the same name it will overwrite the previous gif
        #   (but if save_gif_dir is provided it will raise an error because the directory already exists).
    env.close(); sys.exit()

    # ----- CLOSING AND REPORTING -----

    # testing the agent accuracy (first arm accuracy):
    if RENDER:
        accuracy = memory_evolution.evaluate.test_agent_first_arm_accuracy(
            agent, env, episodes=10,
            render=True)
        print(f"test_agent_first_arm_accuracy: {accuracy}")
    accuracy = memory_evolution.evaluate.test_agent_first_arm_accuracy(
        agent, env, episodes=100,
        render=False)
    print(f"test_agent_first_arm_accuracy: {accuracy}")
    # env.close(); sys.exit()

    # test general target-reached rate (to discriminate bad v.s. border-follower v.s. allocentric/egocentric successful agents):
    if RENDER:
        target_reached_rate = memory_evolution.evaluate.test_agent_target_reached_rate(
            agent, env, episodes=10,
            render=True)
        print(f"test_agent_target_reached_rate: {target_reached_rate}")
    target_reached_rate = memory_evolution.evaluate.test_agent_target_reached_rate(
        agent, env, episodes=100,
        render=False)
    print(f"test_agent_target_reached_rate: {target_reached_rate}")

    # fitness:
    print(f"Fitness: {agent.genome.fitness}")

    env.close()

