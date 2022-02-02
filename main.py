
import numpy as np
from numpy.random import SeedSequence
import pandas as pd
import gym

from gym.utils.env_checker import check_env  # from stable_baselines.common.env_checker import check_env

from memory_evolution.envs import MemoForagingEnv


if __name__ == '__main__':

    env = MemoForagingEnv(3, 5, seed=2002)  # env = gym.make('CartPole-v0')
    check_env(env)  # todo: move in tests

    # print(env.action_space)  # Discrete(4)
    # print(env.observation_space)  # Box([[[0] ... [255]]], (5, 5, 1), uint8)
    # print(env.observation_space.low)  # [[[0] ... [0]]]
    # print(env.observation_space.high)  # [[[255] ... [255]]]
    # print(env.observation_space.shape)  # (5, 5, 1)
    # print(env.observation_space.sample())  # [[[102] ... [203]]] / [[[243] ... [64]]] / each time different

    for i_episode in range(3):#20):
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
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
        else:
            assert False, "Episode has not finished after {} timesteps".format(t + 1)
    env.close()

