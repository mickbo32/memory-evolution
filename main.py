from pprint import pprint

import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import SeedSequence
import pandas as pd

from gym.utils.env_checker import check_env  # from stable_baselines.common.env_checker import check_env

from memory_evolution.envs import BaseForagingEnv, MazeForagingEnv, TMaze

mpl.use('Qt5Agg')  # Change matplotlib backend to show correctly in PyCharm.


if __name__ == '__main__':

    # env = gym.make('CartPole-v0')
    # env = BaseForagingEnv(640, (1.5, 1.), fps=None, seed=42)
    # env = BaseForagingEnv(640, (1.5, 1.), agent_size=.5, food_size=.3, fps=None, seed=42)
    # env = TMaze(env_size=(1.5, 1.), fps=None, seed=42, n_food_items=0)
    # env = TMaze(env_size=(1.5, 1.), fps=None, seed=42, n_food_items=20)
    # env = TMaze(env_size=(1.5, 1.), fps=None, seed=42, n_food_items=50)
    # env = TMaze(.1001, env_size=(1.5, 1.), fps=None, seed=42)
    env = TMaze(seed=42)
    check_env(env)  # todo: move in tests

    # print(env.action_space)  # Discrete(4)
    # print(env.observation_space)  # Box([[[0] ... [255]]], (5, 5, 1), uint8)
    # print(env.observation_space.low)  # [[[0] ... [0]]]
    # print(env.observation_space.high)  # [[[255] ... [255]]]
    # print(env.observation_space.shape)  # (5, 5, 1)
    # print(env.observation_space.sample())  # [[[102] ... [203]]] / [[[243] ... [64]]] / each time different

    print('Main loop')
    for i_episode in range(2):#20):
        observation = env.reset()
        t = 0
        for t in range(100):
            assert env.step_count == t, (env.step_count, t)
            env.render()
            # print(observation)
            action = env.action_space.sample()
            assert env.step_count == t, (env.step_count, t)
            observation, reward, done, info = env.step(action)
            assert env.step_count == t + 1, (env.step_count, t)
            print(info['state']['agent'])
            print(len(info['state']['food_items']), info['state']['food_items'])
            # pprint(info)
            if done:
                env.render()
                print("Episode finished after {} timesteps".format(t + 1))
                break
        else:
            assert False, "Episode has not finished after {} timesteps".format(t + 1)
    env.close()

