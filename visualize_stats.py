import dill  # pickle extension
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
from memory_evolution.logging import get_utcnow_str
from memory_evolution.utils import get_color_str, denormalize_observation

from memory_evolution.load import load_env, load_agent, get_checkpoint_number, AVAILABLE_LOADING_METHODS


if __name__ == '__main__':

    # matplotlib settings:
    isRunningInPyCharm = "PYCHARM_HOSTED" in os.environ
    if isRunningInPyCharm:
        mpl.use('Qt5Agg')  # Change matplotlib backend to show correctly in PyCharm.

    # ----- Settings -----
    RENDER = False  # render or just save gif files
    # ---
    # LOAD_AGENT = '8499798_2022-04-25_163845.730764+0000'  # best genome
    # LOAD_AGENT = '8508254_2022-05-04_021025.108389+0000'
    # LOAD_AGENT = '8525497_2022-05-08_144349.993182+0000'  # vision_channels = 1
    # LOAD_AGENT = '8527358_2022-05-09_104749.699383+0000'  # vision_channels = 3
    LOAD_AGENT = '8536464_2022-05-17_082404.342015+0000'
    LOAD_AGENT = '8539704_2022-05-19_163834.593420+0000'
    LOAD_AGENT = '8541080_2022-05-20_214011.159858+0000'
    LOAD_AGENT = '8547986_2022-05-27_131437.320058+0000'
    LOAD_AGENT = '8552035_2022-05-31_094211.507686+0000'
    LOAD_AGENT_DIR = "logs/saved_logs/outputs-link/no-date0/logs/"
    N_EPISODES = 2
    LOAD_FROM: AVAILABLE_LOADING_METHODS = 'pickle'
    # LOAD_FROM: AVAILABLE_LOADING_METHODS = 'checkpoint'
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

    CHECKPOINT_NUMBER = None  # if None, load the last checkpoint

    # compute runtime consts:
    LOAD_STATS = os.path.join(LOAD_AGENT_DIR, LOAD_AGENT + '_stats.pickle')

    # logging settings:
    UTCNOW = get_utcnow_str()
    LOADED_UTCNOW = LOAD_AGENT + '_LOADED_AGENT___now_' + UTCNOW

    # ----- ENVIRONMENT -----

    env = load_env(LOAD_AGENT, LOAD_AGENT_DIR)

    # ----- AGENT -----

    agent, other_loads = load_agent(LOAD_AGENT, LOAD_AGENT_DIR, LOAD_FROM, CHECKPOINT_NUMBER)
    agent.set_env(env)
    config = other_loads['config']

    # --- stats and genome visualization ---

    with open(LOAD_STATS, "rb") as f:
        # input(
        #     "Stats file can be big, do you really want to open it?"
        #     " (be sure you have a lot of free memory (e.g. close Chrome before going on))"
        #     " [Press ENTER to continue]")
        print('Loading stats...')
        stats = pickle.load(f)
    print(stats)
    assert len(stats.generation_statistics) == len(stats.most_fit_genomes)
    assert stats.best_genome() is max(stats.most_fit_genomes, key=lambda x: x.fitness)
    assert all([(max(_genome_fitness
                     for _specie in _species.values()
                     for _genome_fitness in _specie.values()) == _best_genome.fitness)
                for _species, _best_genome in zip(stats.generation_statistics, stats.most_fit_genomes)])

    agent.visualize_evolution(stats, stats_ylog=False, view=True,
                              filename_stats=os.path.join(LOGGING_DIR, LOADED_UTCNOW + "_fitness.png"),
                              filename_speciation=os.path.join(LOGGING_DIR, LOADED_UTCNOW + "_speciation.png"))

    obs_shape = env.observation_space.shape
    obs_size = reduce(mul, obs_shape, 1)
    input_nodes = config.genome_config.input_keys
    assert obs_size == len(input_nodes) == len(agent.phenotype.input_nodes), (
        obs_shape, len(input_nodes), len(agent.phenotype.input_nodes))
    assert input_nodes == agent.phenotype.input_nodes
    agent.set_env(env)
    agent.visualize_genome(agent.genome, view=True, name='Genome',
                           default_input_node_color='palette',
                           filename=os.path.join(LOGGING_DIR, LOADED_UTCNOW + "_genome.gv"),
                           format='svg',
                           show_palette=True)

