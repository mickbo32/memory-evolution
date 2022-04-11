import json
import logging
import multiprocessing
import os
import pickle
from pprint import pprint
import random  # neat uses random  # todo: allow seeding in neat
import shutil
import sys
import time
from typing import Optional

import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import neat
import numpy as np
from numpy.random import SeedSequence
import pandas as pd

from gym.utils.env_checker import check_env  # from stable_baselines.common.env_checker import check_env

from memory_evolution.agents import RandomActionAgent, RnnNeatAgent, CtrnnNeatAgent
from memory_evolution.envs import BaseForagingEnv, MazeForagingEnv, TMaze
from memory_evolution.evaluate import evaluate_agent
from memory_evolution.utils import set_main_logger

# matplotlib settings:
isRunningInPyCharm = "PYCHARM_HOSTED" in os.environ
if isRunningInPyCharm:
    mpl.use('Qt5Agg')  # Change matplotlib backend to show correctly in PyCharm.


if __name__ == '__main__':

    # ----- Settings -----

    # logging settings:
    logging_dir, UTCNOW = set_main_logger(file_handler_all=None, stdout_handler=logging.INFO)
    logging.info(__file__)
    
    # get some stats:
    version_msg = f"Python version\n{sys.version}\nVersion info\n{sys.version_info}\n"
    logging.info(version_msg)
    cpu_count = multiprocessing.cpu_count()
    logging.info(f"CPU count: {cpu_count}\n")
    cwd = os.getcwd()
    logging.info(f"Current working directory: {cwd!r}\n")
    
    # neat random seeding:
    random.seed(42)
    logging.debug(random.getstate())
    # Use random.setstate(state) to set an old state, where 'state' have been obtained from a previous call to getstate().

    # ----- ENVIRONMENT -----

    # env = gym.make('CartPole-v0')
    # env = BaseForagingEnv(640, (1.5, 1.), fps=None, seed=42)
    # env = BaseForagingEnv(640, (1.5, 1.), agent_size=.5, food_size=.3, fps=None, seed=42)
    # env = TMaze(env_size=(1.5, 1.), fps=None, seed=42, n_food_items=0)
    # env = TMaze(env_size=(1.5, 1.), fps=None, seed=42, n_food_items=20)
    # env = TMaze(env_size=(1.5, 1.), fps=None, seed=42, n_food_items=50)
    # env = TMaze(.1001, env_size=(1.5, 1.), fps=None, seed=42)
    # env = TMaze(seed=42)
    # env = TMaze(env_size=(1.5, 1.), seed=42, agent_size=.15, n_food_items=10, max_steps=500, vision_resolution=7)  # todo: use in tests
    # env = BaseForagingEnv(env_size=(1.5, 1.), seed=42, agent_size=.15, n_food_items=10, max_steps=500, vision_resolution=7) # todo: use in tests
    # env = TMaze(seed=42, agent_size=.15, n_food_items=10, max_steps=500, vision_resolution=7, observation_noise=('normal', 0.0, 0.5))
    # env = TMaze(env_size=(1.5, 1.), seed=42, agent_size=.15, n_food_items=10, max_steps=500, vision_resolution=7)
    env = BaseForagingEnv(window_size=200, env_size=(1.5, 1.), seed=42, agent_size=.15, n_food_items=10, max_steps=400, vision_resolution=7)
    # env = TMaze(seed=42, agent_size=.10, n_food_items=10, max_steps=500, vision_resolution=7)
    logging.debug(env._seed)  # todo: use a variable seed (e.g.: seed=42; env=TMaze(seed=seed); logging.debug(seed)) for assignation of seed, don't access the internal variable
    print('observation_space:',
          env.observation_space.shape,
          np.asarray(env.observation_space.shape).prod())
    # picKle env:
    with open(os.path.join(logging_dir, UTCNOW + '_' + 'env.pickle'), "wb") as f:
        pickle.dump(env, f)
    # check pickle env:  # todo: move in tests
    with open(os.path.join(logging_dir, UTCNOW + '_' + 'env.pickle'), "rb") as f:
        _loaded_env = pickle.load(f)
        assert type(_loaded_env) is type(env)
        assert _loaded_env._init_params == env._init_params
    # check env:
    check_env(env)  # todo: move in tests
    print('Env checked.')

    # print(env.action_space)  # Discrete(4)
    # print(env.observation_space)  # Box([[[0] ... [255]]], (5, 5, 1), uint8)
    # print(env.observation_space.low)  # [[[0] ... [0]]]
    # print(env.observation_space.high)  # [[[255] ... [255]]]
    # print(env.observation_space.shape)  # (5, 5, 1)
    # print(env.observation_space.sample())  # [[[102] ... [203]]] / [[[243] ... [64]]] / each time different

    # ----- AGENT -----

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-rnn')
    # logging: save current config file for later use:
    shutil.copyfile(config_path, os.path.join(logging_dir, UTCNOW + '_' + 'config'))

    agent = RnnNeatAgent(config_path)

    # ----- MAIN LOOP -----
    # Evolve, interact, repeat.

    # Rendering settings:
    # render, parallel, render_best = 0, 1, 1    # local execution, show also stuff
    render, parallel, render_best = 0, 1, 0    # remote execution, just save gifs

    # evaluate_agent(RandomActionAgent(env), env, episodes=2, render=True)

    checkpointer = neat.Checkpointer(generation_interval=100,
                                     time_interval_seconds=300,
                                     filename_prefix=os.path.join(
                                         logging_dir,
                                         UTCNOW + '_' + 'neat-checkpoint-'))

    agent.set_env(env)
    winner = agent.evolve(500, render=render, checkpointer=checkpointer, parallel=parallel,
                          filename_tag=UTCNOW + '_', path_dir=logging_dir, image_format='png',
                          render_best=False)
    # fixme: todo: parallel=True use the same seed for the environment in each process
    #     (but for the agent is correctly using a different seed it seems)

    # render the best agent:
    evaluate_agent(agent, env, episodes=2, render=render_best,
                   save_gif=True,
                   save_gif_name=os.path.join(logging_dir, 'frames_' + UTCNOW + '.gif'))

    # ----- CLOSING AND REPORTING -----

    env.close()

