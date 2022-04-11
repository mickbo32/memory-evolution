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

from memory_evolution.agents import RandomActionAgent, RnnNeatAgent, CtrnnNeatAgent
from memory_evolution.envs import BaseForagingEnv, MazeForagingEnv, TMaze
from memory_evolution.evaluate import evaluate_agent
from memory_evolution.utils import set_main_logger

# matplotlib settings:
isRunningInPyCharm = "PYCHARM_HOSTED" in os.environ
if isRunningInPyCharm:
    mpl.use('Qt5Agg')  # Change matplotlib backend to show correctly in PyCharm.

# other consts:
AVAILABLE_LOADING_METHODS = Literal['pickle', 'checkpoint']


if __name__ == '__main__':

    # ----- Settings -----
    LOAD_AGENT = '2022-04-09_203055.967693+0000'
    LOAD_AGENT_DIR = "logs/saved_logs/no-date/logs/"
    N_EPISODES = 3
    LOAD_FROM: AVAILABLE_LOADING_METHODS = 'checkpoint'

    assert LOAD_FROM in typing.get_args(AVAILABLE_LOADING_METHODS), LOAD_FROM

    CHECKPOINT_NUMBER = None  # if None, load the last checkpoint

    # compute runtime consts:
    LOAD_ENV = os.path.join(LOAD_AGENT_DIR, LOAD_AGENT + '_env.pickle')
    if LOAD_FROM == 'pickle':
        CONFIG_PATH = os.path.join(LOAD_AGENT_DIR, LOAD_AGENT + '_config')
        LOAD_AGENT_PATH = os.path.join(LOAD_AGENT_DIR, LOAD_AGENT + '_genome.pickle')
        assert os.path.isfile(LOAD_AGENT_PATH), LOAD_AGENT_PATH
    elif LOAD_FROM == 'checkpoint':
        _LOAD_FROM_CHECKPOINT_TAG = '_neat-checkpoint-'
        if CHECKPOINT_NUMBER is None:
            prefix = LOAD_AGENT + _LOAD_FROM_CHECKPOINT_TAG
            checkpoint_numbers = sorted([int(f.removeprefix(prefix))
                                         for f in os.listdir(LOAD_AGENT_DIR)
                                         if f.startswith(prefix)])
            print(f"Checkpoints found (generation_number) for agent {LOAD_AGENT}:", checkpoint_numbers)
            CHECKPOINT_NUMBER = checkpoint_numbers[-1]  # max(checkpoint_numbers)
            print('CHECKPOINT_NUMBER:', CHECKPOINT_NUMBER)
        LOAD_AGENT_PATH = os.path.join(LOAD_AGENT_DIR, LOAD_AGENT + _LOAD_FROM_CHECKPOINT_TAG + str(CHECKPOINT_NUMBER))
        assert os.path.isfile(LOAD_AGENT_PATH), LOAD_AGENT_PATH
    else:
        raise AssertionError

    # logging settings:
    logging_dir, UTCNOW = set_main_logger(file_handler_all=None, stdout_handler=logging.INFO)
    logging.info(__file__)
    LOADED_UTCNOW = 'loaded_agent_' + LOAD_AGENT + '__now_' + UTCNOW

    # neat random seeding:
    random.seed(42)
    logging.debug(random.getstate())
    # Use random.setstate(state) to set an old state, where 'state' have been obtained from a previous call to getstate().

    # ----- ENVIRONMENT -----

    with open(LOAD_ENV, "rb") as f:
        env = pickle.load(f)
    print(env.__str__init_params__)
    logging.debug(env._seed)  # todo: use a variable seed (e.g.: seed=42; env=TMaze(seed=seed); logging.debug(seed)) for assignation of seed, don't access the internal variable
    print('observation_space:',
          env.observation_space.shape,
          np.asarray(env.observation_space.shape).prod())

    # ----- AGENT -----

    # load from pickle:
    if LOAD_FROM == 'pickle':
        with open(LOAD_AGENT_PATH, "rb") as f:
            genome = pickle.load(f)
        agent = RnnNeatAgent(CONFIG_PATH, genome=genome)

    # load from checkpoint:
    elif LOAD_FROM == 'checkpoint':
        p = neat.Checkpointer.restore_checkpoint(LOAD_AGENT_PATH)
        config = p.config
        # pprint(p.population)
        pop = sorted([genome for _id, genome in p.population.items() if genome.fitness is not None],
                     key=lambda x: -x.fitness)
        pprint([(genome.key, genome, genome.fitness) for genome in pop])
        assert pop
        best_genome = pop[1]
        agent = RnnNeatAgent(config, genome=best_genome)
        print()

    else:
        raise AssertionError

    # ----- MAIN LOOP -----
    # Evolve, interact, repeat.

    print('Evaluating agent ...\n')

    # to do: tests of evaluate_agent: episodes=1 | episodes=2 ;

    RANDOM_AGENT_UTCNOW = 'RandomActionAgent_' + UTCNOW
    # evaluate_agent(RandomActionAgent(env), env, episodes=2, render=True,
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
    evaluate_agent(agent, env, episodes=N_EPISODES, render=True,
                   save_gif=True,
                   save_gif_name=os.path.join(logging_dir, 'frames_' + LOADED_UTCNOW + '.gif'))
    # Note: if you run twice evaluate_agent with the same name it will overwrite the previous gif
    #   (but if save_gif_dir is provided it will raise an error because the directory already exists).

    # ----- CLOSING AND REPORTING -----

    env.close()

