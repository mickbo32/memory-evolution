import logging
import os
import pickle
from pprint import pprint
import random  # neat uses random  # todo: allow seeding in neat
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

from memory_evolution.agents import BaseAgent, RnnNeatAgent, CtrnnNeatAgent
from memory_evolution.envs import BaseForagingEnv, MazeForagingEnv, TMaze
from memory_evolution.evaluate import evaluate_agent
from memory_evolution.utils import set_main_logger

# matplotlib settings:
mpl.use('Qt5Agg')  # Change matplotlib backend to show correctly in PyCharm.


if __name__ == '__main__':

    # ----- Settings -----
    LOAD_AGENT = '2022-03-13_183513.643804+0000'
    LOAD_AGENT_DIR = "logs/saved logs/"

    # logging settings:
    logging_dir, UTCNOW = set_main_logger(file_handler_all=None, stdout_handler=logging.INFO)
    logging.debug(__file__)
    LOADED_UTCNOW = 'loaded_agent_' + LOAD_AGENT + '__now_' + UTCNOW

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
    env = BaseForagingEnv(window_size=200, env_size=(1.5, 1.), seed=42, agent_size=.15, n_food_items=10, rotation_step=20., max_steps=500, vision_resolution=7)
    logging.debug(env._seed)  # todo: use a variable seed (e.g.: seed=42; env=TMaze(seed=seed); logging.debug(seed)) for assignation of seed, don't access the internal variable
    print('observation_space:',
          env.observation_space.shape,
          np.asarray(env.observation_space.shape).prod())

    # ----- AGENT -----

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-rnn')

    # load from pickle:
    with open(os.path.join(LOAD_AGENT_DIR, LOAD_AGENT + "_genome.pickle"), "rb") as f:
        genome = pickle.load(f)
    agent = RnnNeatAgent(config_path, genome=genome)

    # load from checkpoint:
    p = neat.Checkpointer.restore_checkpoint(
        'logs/saved logs/2022-03-23_185500.926591+0000_neat-checkpoint-166')
    # pprint(p.population)
    pop = sorted([genome for id, genome in p.population.items() if genome.fitness is not None],
                 key=lambda x: -x.fitness)
    pprint([(genome.key, genome, genome.fitness) for genome in pop])
    assert pop
    best_genome = pop[1]
    agent = RnnNeatAgent(config_path, genome=best_genome)
    print()

    # ----- MAIN LOOP -----
    # Evolve, interact, repeat.

    print('Evaluating agent ...\n')

    from main import RandomAgent
    RANDOM_AGENT_UTCNOW = 'RandomAgent_' + UTCNOW
    # evaluate_agent(RandomAgent(env), env, episodes=2, render=True,
    #                save_gif=True,
    #                save_gif_dir=os.path.join(logging_dir, 'frames_' + RANDOM_AGENT_UTCNOW),
    #                save_gif_name=RANDOM_AGENT_UTCNOW + '.gif')

    agent.set_env(env)
    # evaluate_agent(agent, env, episodes=2, render=True,
    #                save_gif=False)
    # evaluate_agent(agent, env, episodes=2, render=True,
    #                save_gif=True,
    #                save_gif_dir=os.path.join(logging_dir, 'frames_' + LOADED_UTCNOW),
    #                save_gif_name=LOADED_UTCNOW + '.gif')
    evaluate_agent(agent, env, episodes=1, render=True,
                   save_gif=True,
                   save_gif_name=os.path.join(logging_dir, 'frames_' + LOADED_UTCNOW + '.gif'))
    # Note: if you run twice evaluate_agent with the same name it will overwrite the previous gif
    #   (but if save_gif_dir is provided it will raise an error because the directory already exists).

    # ----- CLOSING AND REPORTING -----

    env.close()

