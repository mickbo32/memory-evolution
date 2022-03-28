import logging
import os
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

from memory_evolution.agents import RandomActionAgent, RnnNeatAgent, CtrnnNeatAgent
from memory_evolution.envs import BaseForagingEnv, MazeForagingEnv, TMaze
from memory_evolution.evaluate import evaluate_agent
from memory_evolution.utils import set_main_logger

# matplotlib settings:
mpl.use('Qt5Agg')  # Change matplotlib backend to show correctly in PyCharm.


if __name__ == '__main__':

    # ----- Settings -----

    # logging settings:
    logging_dir, UTCNOW = set_main_logger(file_handler_all=None, stdout_handler=logging.INFO)
    logging.debug(__file__)

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
    env = BaseForagingEnv(window_size=200, env_size=(1.5, 1.), seed=42, agent_size=.15, n_food_items=10, max_steps=1000, vision_resolution=7)
    # env = TMaze(seed=42, agent_size=.10, n_food_items=10, max_steps=500, vision_resolution=7)
    logging.debug(env._seed)  # todo: use a variable seed (e.g.: seed=42; env=TMaze(seed=seed); logging.debug(seed)) for assignation of seed, don't access the internal variable
    print('observation_space:',
          env.observation_space.shape,
          np.asarray(env.observation_space.shape).prod())
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
    agent = RnnNeatAgent(config_path)

    # ----- MAIN LOOP -----
    # Evolve, interact, repeat.

    # evaluate_agent(RandomActionAgent(env), env, episodes=2, render=True)

    checkpointer = neat.Checkpointer(generation_interval=100,
                                     time_interval_seconds=300,
                                     filename_prefix=os.path.join(
                                         logging_dir,
                                         UTCNOW + '_' + 'neat-checkpoint-'))

    agent.set_env(env)
    winner = agent.evolve(500, render=0, checkpointer=checkpointer, parallel=1,
                          filename_tag=UTCNOW + '_', path_dir=logging_dir, image_format='png')
    # fixme: todo: parallel=True use the same seed for the environment in each process
    #     (but for the agent is correct and different it seems)
    evaluate_agent(agent, env, episodes=2, render=True,
                   save_gif=True,
                   save_gif_name=os.path.join(logging_dir, 'frames_' + UTCNOW + '.gif'))
    # run(env, episodes=2)

    # ----- CLOSING AND REPORTING -----

    env.close()


'''
Better efficiency:
env = TMaze(env_size=(1.5, 1.), seed=42, agent_size=.15, n_food_items=10, max_steps=500, vision_resolution=7)
(parallel = False)

* b163541 (HEAD -> main, origin/main, origin/HEAD) Merge branch 'continuous' into main Continuous environment and agents with evolution

render = False
Episode finished after 500 timesteps, for a total of 500 simulated seconds (in 7.13121553 actual seconds).

render = True
Episode finished after 500 timesteps, for a total of 500 simulated seconds (in 19.826593934 actual seconds).


* ce3470d (continuous) big refactoring, environment efficiency improved, env_img, pixel space mask for valid positions, geometry, pygame sprites for items

render = False
Episode finished after 500 timesteps, for a total of 500 simulated seconds (in 1.843639818 actual seconds).

render = True
Episode finished after 500 timesteps, for a total of 500 simulated seconds (in 15.735012616 actual seconds).


Note: on both tests here above with rendering True, the rendering was slow because of the external screen used,
by using only the integrated screen of the pc the rendering time goes down to 19->14 and 15->3.5 respectively.


* cd9570b

render = False
Episode finished after 500 timesteps, for a total of 500 simulated seconds (in 1.6946048 actual seconds).
render = False;parallel=True
Episode finished after 500 timesteps, for a total of 500 simulated seconds (in 3.641211265 actual seconds).

render = True
Episode finished after 500 timesteps, for a total of 500 simulated seconds (in 2.768054353 actual seconds).

'''

