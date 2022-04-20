import json
import logging
import multiprocessing
import os
import pickle
from pprint import pprint
import random  # neat uses random  # todo: allow seeding in neat
import re
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
from memory_evolution.envs import BaseForagingEnv, MazeForagingEnv, TMaze, RadialArmMaze
from memory_evolution.evaluate import evaluate_agent
from memory_evolution.utils import set_main_logger

# matplotlib settings:
isRunningInPyCharm = "PYCHARM_HOSTED" in os.environ
if isRunningInPyCharm:
    mpl.use('Qt5Agg')  # Change matplotlib backend to show correctly in PyCharm.


if __name__ == '__main__':

    # parse command-line arguments passed to the program:
    JOB_ID = ''  # type: str
    if len(sys.argv) == 1:  # local execution
        pass
    elif len(sys.argv) == 2:  # remote execution

        # remote execution,
        # JOB_ID should be passed as argument to the program when running it on the remote cluster server.
        JOB_ID = str(sys.argv[1])
        match = re.match(r"^([0-9]+).hpc-head-n1.unitn.it$", JOB_ID)
        if match:
            JOB_ID = match.group(1)  # type: str
        assert isinstance(JOB_ID, str), type(JOB_ID)

        # remote execution, no input devices.
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_MOUSEDRIVER'] = 'dummy'
        """
        # Alternatively,
        # Environment variables can be set in the script which calls this program:
        #...
        echo "Starting (PBS_JOBID=${PBS_JOBID}) ..."
        source ~/miniconda3/bin/activate evo
        python --version
        export SDL_AUDIODRIVER='dummy'
        export SDL_VIDEODRIVER='dummy'
        export SDL_MOUSEDRIVER='dummy'
        python memory-evolution/main.py "${PBS_JOBID}"
        """
    else:
        raise RuntimeError(sys.argv)

    # ----- Settings -----

    # logging settings:
    logging_dir, UTCNOW = set_main_logger(file_handler_all=None,
                                          stdout_handler=logging.INFO,
                                          file_handler_now_filename_fmt="log_" + JOB_ID + "_{utcnow}.log")
    logging.info(__file__)

    # if job_id is passed to the program, use it in the log tag:
    if JOB_ID:
        LOG_TAG = JOB_ID + '_' + UTCNOW
    else:
        LOG_TAG = UTCNOW
    logging.info('TAG: ' + LOG_TAG)
    
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

    env = BaseForagingEnv(window_size=200, env_size=(1.5, 1.), seed=42, agent_size=.15, n_food_items=10, vision_depth=.25, vision_field_angle=135, max_steps=1000, vision_resolution=7)

    # env = RadialArmMaze(3, 1., window_size=200, env_size=2., seed=42, agent_size=.15, n_food_items=10, vision_depth=.25, vision_field_angle=135, max_steps=400, vision_resolution=7)
    # env = RadialArmMaze(9, window_size=200, env_size=2., seed=42, agent_size=.15, n_food_items=10, vision_depth=.25, vision_field_angle=135, max_steps=400, vision_resolution=7)
    # env = RadialArmMaze(5, 1., window_size=200, env_size=2., seed=42, agent_size=.15, n_food_items=10, vision_depth=.25, vision_field_angle=135, max_steps=400, vision_resolution=7)
    # env = RadialArmMaze(2, window_size=200, env_size=2., seed=42, agent_size=.15, n_food_items=10, vision_depth=.25, vision_field_angle=135, max_steps=400, vision_resolution=7)
    # env = RadialArmMaze(4, window_size=200, env_size=2., seed=42, agent_size=.15, n_food_items=10, vision_depth=.25, vision_field_angle=135, max_steps=400, vision_resolution=7)
    # env = RadialArmMaze(window_size=200, seed=42, agent_size=.15, n_food_items=10, vision_depth=.25, vision_field_angle=135, max_steps=400, vision_resolution=7)

    # env = TMaze(seed=42, agent_size=.10, n_food_items=10, max_steps=500, vision_resolution=7)
    logging.debug(env._seed)  # todo: use a variable seed (e.g.: seed=42; env=TMaze(seed=seed); logging.debug(seed)) for assignation of seed, don't access the internal variable
    print('observation_space:',
          env.observation_space.shape,
          np.asarray(env.observation_space.shape).prod())
    # picKle env:
    with open(os.path.join(logging_dir, LOG_TAG + '_env.pickle'), "wb") as f:
        pickle.dump(env, f)
    # check pickle env:  # todo: move in tests
    with open(os.path.join(logging_dir, LOG_TAG + '_env.pickle'), "rb") as f:
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
    shutil.copyfile(config_path, os.path.join(logging_dir, LOG_TAG + '_config'))

    Phenotype = RnnNeatAgent
    with open(os.path.join(logging_dir, LOG_TAG + '_phenotype.pickle'), "wb") as f:
        pickle.dump(Phenotype, f)
    agent = Phenotype(config_path)

    # ----- MAIN LOOP -----
    # Evolve, interact, repeat.

    # Rendering settings:
    if not JOB_ID:  # local execution
        # note: if you render all will be slow, but good for debugging
        # note2: if you render all and if you minimize the window or you put it in a part of the screen not visible
        #        the algorithm will go way faster, so you can make it faster and debugging
        #        at your choice by knowing this.
        render, parallel, render_best = True, False, True      # local execution, render all
        # render, parallel, render_best = False, True, True     # local execution, show best
    else:  # remote execution
        render, parallel, render_best = False, True, False    # remote execution, just save gifs

    # evaluate_agent(RandomActionAgent(env), env, episodes=2, render=True)

    checkpointer = neat.Checkpointer(generation_interval=100,
                                     time_interval_seconds=300,
                                     filename_prefix=os.path.join(
                                         logging_dir,
                                         LOG_TAG + '_neat-checkpoint-'))

    agent.set_env(env)
    winner = agent.evolve(1000, render=render, checkpointer=checkpointer, parallel=parallel,
                          filename_tag=LOG_TAG + '_', path_dir=logging_dir, image_format='png',
                          render_best=False)
    # fixme: todo: parallel=True use the same seed for the environment in each process
    #     (but for the agent is correctly using a different seed it seems)

    # render the best agent:
    evaluate_agent(agent, env, episodes=2, render=render_best,
                   save_gif=True,
                   save_gif_name=os.path.join(logging_dir, LOG_TAG + '_frames.gif'))

    # ----- CLOSING AND REPORTING -----

    env.close()

