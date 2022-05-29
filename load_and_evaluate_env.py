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
    # LOAD_AGENT = '8492571_2022-04-20_212136.999729+0000'
    LOAD_AGENT = '8525497_2022-05-08_144349.993182+0000'
    LOAD_AGENT = '8527358_2022-05-09_104749.699383+0000'
    LOAD_AGENT = '8535242_2022-05-16_101934.963202+0000'
    # LOAD_AGENT = '8536569_2022-05-17_091348.920601+0000'
    # LOAD_AGENT = '8536464_2022-05-17_082404.342015+0000'
    # LOAD_AGENT = '8537163_2022-05-17_135612.558440+0000'
    # LOAD_AGENT = '8537772_2022-05-18_112406.667878+0000'
    LOAD_AGENT = '8539704_2022-05-19_163834.593420+0000'
    LOAD_AGENT = '8541080_2022-05-20_214011.159858+0000'
    LOAD_AGENT = '8547986_2022-05-27_131437.320058+0000'
    LOAD_AGENT_DIR = "logs/saved_logs/no-date/logs/"
    # LOAD_FROM: AVAILABLE_LOADING_METHODS = 'checkpoint'
    LOAD_FROM: AVAILABLE_LOADING_METHODS = 'pickle'
    N_EPISODES = 5
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

    assert LOAD_FROM in typing.get_args(AVAILABLE_LOADING_METHODS), LOAD_FROM

    # compute runtime consts:
    LOAD_ENV = os.path.join(LOAD_AGENT_DIR, LOAD_AGENT + '_env.pickle')
    LOAD_PHENOTYPE = os.path.join(LOAD_AGENT_DIR, LOAD_AGENT + '_phenotype.pkl')
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
    logging_dir, UTCNOW = set_main_logger(file_handler_all=None,
                                          logging_dir=LOGGING_DIR,
                                          stdout_handler=logging.INFO - 2,
                                          file_handler_now_filename_fmt=LOAD_AGENT + "_LOADED__now_{utcnow}.log")
    del LOGGING_DIR  # from now on use 'logging_dir' instead.
    logging.info(__file__)
    LOADED_UTCNOW = LOAD_AGENT
    if LOAD_FROM == 'checkpoint':
        LOADED_UTCNOW += f'_checkpoint-{CHECKPOINT_NUMBER}'
    LOADED_UTCNOW += '_LOADED_AGENT___now_' + UTCNOW

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

    # todo: json, so you can change stuffs
    with open(LOAD_ENV, "rb") as f:
        env = pickle.load(f)

    print(env.__str__init_params__())
    logging.debug(env._seed)  # todo: use a variable seed (e.g.: seed=42; env=TMaze(seed=seed); logging.debug(seed)) for assignation of seed, don't access the internal variable
    print('observation_space:',
          env.observation_space.shape,
          np.asarray(env.observation_space.shape).prod())

    # ----- AGENT -----

    with open(LOAD_PHENOTYPE, "rb") as f:
        Phenotype, _Phenotype_attrs = dill.load(f)
        for name, value in _Phenotype_attrs.items():
            setattr(Phenotype, name, value)

    # load from pickle:
    if LOAD_FROM == 'pickle':
        with open(LOAD_AGENT_PATH, "rb") as f:
            genome = pickle.load(f)
        agent = Phenotype(CONFIG_PATH, genome=genome)

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
        agent = Phenotype(config, genome=best_genome)
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

    if N_EPISODES > 0:
        # evaluate_agent(agent, env, episodes=2, render=True,
        #                save_gif=False)
        # evaluate_agent(agent, env, episodes=2, render=True,
        #                save_gif=True,
        #                save_gif_dir=os.path.join(logging_dir, 'frames_' + LOADED_UTCNOW),
        #                save_gif_name=LOADED_UTCNOW + '.gif')
        evaluate_agent(agent, env, episodes=N_EPISODES, render=RENDER,
                       save_gif=True,
                       save_gif_name=os.path.join(logging_dir, LOADED_UTCNOW + '_frames.gif'))
        # Note: if you run twice evaluate_agent with the same name it will overwrite the previous gif
        #   (but if save_gif_dir is provided it will raise an error because the directory already exists).

    # ----- CLOSING AND REPORTING -----

    # testing the agent accuracy:
    if RENDER:
        accuracy = memory_evolution.evaluate.test_agent_first_arm_accuracy(
            agent, env, episodes=10,
            render=True)
        print(f"test_agent_first_arm_accuracy: {accuracy}")
    accuracy = memory_evolution.evaluate.test_agent_first_arm_accuracy(
        agent, env, episodes=100,
        render=False)
    print(f"test_agent_first_arm_accuracy: {accuracy}")

    env.close()

