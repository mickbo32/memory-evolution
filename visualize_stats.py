from functools import reduce
from operator import mul
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
from memory_evolution.utils import get_color_str, get_utcnow_str, denormalize_observation

# matplotlib settings:
isRunningInPyCharm = "PYCHARM_HOSTED" in os.environ
if isRunningInPyCharm:
    mpl.use('Qt5Agg')  # Change matplotlib backend to show correctly in PyCharm.

# other consts:
AVAILABLE_LOADING_METHODS = Literal['pickle', 'checkpoint']


if __name__ == '__main__':

    # ----- Settings -----
    RENDER = False  # render or just save gif files
    # ---
    LOAD_AGENT = '8499798_2022-04-25_163845.730764+0000'
    # LOAD_AGENT = '8505569_2022-05-01_140544.462333+0000'
    LOAD_AGENT_DIR = "logs/saved_logs/no-date/logs/"
    N_EPISODES = 2
    LOAD_FROM: AVAILABLE_LOADING_METHODS = 'pickle'
    LOGGING_DIR = 'logs'
    # ---
    # override variables if provided as program arguments
    if len(sys.argv) == 1:
        pass
    elif 2 <= len(sys.argv) <= 5:
        LOAD_AGENT = sys.argv[1]
        if len(sys.argv) >= 3:
            LOAD_AGENT_DIR = sys.argv[2]
        elif len(sys.argv) >= 4:
            N_EPISODES = sys.argv[3]
        elif len(sys.argv) >= 5:
            LOAD_FROM = sys.argv[4]
    else:
        raise RuntimeError(sys.argv)

    assert LOAD_FROM in typing.get_args(AVAILABLE_LOADING_METHODS), LOAD_FROM

    CHECKPOINT_NUMBER = None  # if None, load the last checkpoint

    # compute runtime consts:
    LOAD_ENV = os.path.join(LOAD_AGENT_DIR, LOAD_AGENT + '_env.pickle')
    LOAD_PHENOTYPE = os.path.join(LOAD_AGENT_DIR, LOAD_AGENT + '_phenotype.pickle')
    LOAD_STATS = os.path.join(LOAD_AGENT_DIR, LOAD_AGENT + '_stats.pickle')
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
    UTCNOW = get_utcnow_str()
    LOADED_UTCNOW = LOAD_AGENT + '_LOADED_AGENT___now_' + UTCNOW

    # ----- ENVIRONMENT -----

    # env = RadialArmMaze(corridor_width=.2,
    #                     window_size=200, seed=4242, agent_size=.075, food_size=.05, n_food_items=1,
    #                     # init_agent_position=(.5, .1), init_food_positions=((.9, .5),),
    #                     init_agent_position=(.9, .5), init_food_positions=((.5, .9),),
    #                     vision_depth=.2, vision_field_angle=135, max_steps=400, vision_resolution=8)

    # todo: json, so you can change stuffs
    with open(LOAD_ENV, "rb") as f:
        env = pickle.load(f)

    print(env.__str__init_params__())
    print('observation_space:',
          env.observation_space.shape,
          np.asarray(env.observation_space.shape).prod())

    # ----- AGENT -----


    with open(LOAD_PHENOTYPE, "rb") as f:
        Phenotype = pickle.load(f)

    # load from pickle:
    if LOAD_FROM == 'pickle':
        with open(LOAD_AGENT_PATH, "rb") as f:
            genome = pickle.load(f)
        agent = Phenotype(CONFIG_PATH, genome=genome)
        config = agent.config

    # load from checkpoint:
    elif LOAD_FROM == 'checkpoint':

        input(
            "Loading population can be memory intensive, do you really want to open it?"
            " (be sure you have a lot of free memory (e.g. close Chrome before going on))"
            " [Press ENTER to continue]")
        print('Loading population and agent...')

        p = neat.Checkpointer.restore_checkpoint(LOAD_AGENT_PATH)
        config = p.config
        # pprint(p.population)
        pop = sorted([genome for _id, genome in p.population.items() if genome.fitness is not None],
                     key=lambda x: -x.fitness)
        pprint([(genome.key, genome, genome.fitness) for genome in pop])
        assert pop
        best_genome = pop[1]
        agent = Phenotype(config, genome=best_genome)
        print()

    else:
        raise AssertionError

    # with open(LOAD_STATS, "rb") as f:
    #     input(
    #         "Stats file can be big, do you really want to open it?"
    #         " (be sure you have a lot of free memory (e.g. close Chrome before going on))"
    #         " [Press ENTER to continue]")
    #     print('Loading stats...')
    #     stats = pickle.load(f)
    # print(stats)
    #
    # agent.visualize_evolution(stats, stats_ylog=True, view=True,
    #                           filename_stats=os.path.join(LOGGING_DIR, LOADED_UTCNOW + "_fitness.png"),
    #                           filename_speciation=os.path.join(LOGGING_DIR, LOADED_UTCNOW + "_speciation.png"))

    obs_shape = env.observation_space.shape
    obs_size = reduce(mul, env.observation_space.shape, 1)
    input_nodes = config.genome_config.input_keys
    assert obs_size == len(input_nodes) == len(agent.phenotype.input_nodes)
    assert input_nodes == agent.phenotype.input_nodes
    agent.set_env(env)
    agent.visualize_genome(agent.genome, view=True, name='Genome',
                           default_input_node_color='palette',
                           filename=os.path.join(LOGGING_DIR, LOADED_UTCNOW + "_genome.gv"),
                           format='svg',
                           show_palette=True)

