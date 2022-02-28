from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from collections.abc import Iterable, Sequence
import math
import multiprocessing
from numbers import Number, Real
from pprint import pprint
from typing import Optional, Union, Any, Literal
from warnings import warn
import sys
import time

import gym
from gym import spaces
import matplotlib as mpl
import matplotlib.pyplot as plt
import neat
import numpy as np
from numpy.random import SeedSequence, default_rng
import pandas as pd
import pygame
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint, MultiPolygon
from shapely.ops import unary_union, triangulate

import memory_evolution


def evaluate_agent(agent, env: gym.Env,
                   episodes: int = 1,
                   max_actual_time_per_episode: Optional[Union[int, float]] = None,
                   episodes_fitness_aggr_func: Literal['mean', 'max', 'min'] = 'min',
                   render: bool = False,
                   ) -> float:
    """Evaluate agent in a gym environment and return the fitness of the agent.

    Args:
        agent: the agent, an object with an ``action()`` method which takes the
            observation as argument and returns a valid action and with a
            ``reset()`` method which reset the agent to an initial
            state ``t==0``.
        env: a valid gym environment object (it can be an object of a
            subclass of gym.Evn).
        episodes: number of independent episodes that will be run.
        max_actual_time_per_episode: if ``None`` it will just run until the
            environment is done, otherwise after ``max_actual_time_per_episode``
            seconds the environment will be closed and an error is raised.
        episodes_fitness_aggr_func: function to use to aggregate all the fitness
            values collected for each single independent episode (default is
            'min': The genome's fitness is its worst performance across all runs).
        render: if True, render the environment while evaluating the agent.

    Returns:
        The fitness value of the agent (a score that tells how good is the agent in solving the task).

    Raises:
        RuntimeError: if ``max_iters_per_episode`` is not ``None`` and
            episode has not finished after ``max_iters_per_episode`` timesteps.
    """
    time_step = env.time_step  # todo: non serve, basta fare env.t - prev_env_t; togli time_step da env e chiama env._dt
    fitnesses = []
    for i_episode in range(episodes):
        if render:
            print(f'Starting episode #{i_episode} ...')
        start_time_episode = time.perf_counter_ns()
        # Reset env and agent:
        observation = env.reset()
        agent.reset()
        assert env.t == 0., env.t
        fitness = 0.0
        if render:
            # print(observation)
            env.render()
        i = 0
        done = False
        # while not done and (max_iters_per_episode is None or t < max_iters_per_episode):
        # while not done and (max_env_t_per_episode is None or env.t < max_env_t_per_episode):
        while not done and (max_actual_time_per_episode is None
                            or (time.perf_counter_ns() - start_time_episode) < max_actual_time_per_episode * 10 ** 9):
            # Agent performs an action based on the current observation (and
            # its internal state, i.e. memory):
            if isinstance(env, memory_evolution.envs.BaseForagingEnv):
                assert env.step_count == i, (env.step_count, i)
            action = agent.action(observation)
            if isinstance(env, memory_evolution.envs.BaseForagingEnv):
                assert env.step_count == i, (env.step_count, i)
            observation, reward, done, info = env.step(action)
            if isinstance(env, memory_evolution.envs.BaseForagingEnv):
                assert env.step_count == i + 1, (env.step_count, i)
            if render:
                # print("Observation:", observation, sep='\n')
                print("Action:", action, sep=' ')
                # print(info['state']['agent'])
                # print(len(info['state']['food_items']), info['state']['food_items'])
                # pprint(info)
                env.render()
                print()
            i += 1
        end_t = env.t
        end_time_episode = time.perf_counter_ns()
        fitnesses.append(fitness)
        if done:
            if render:
                print("{0} fitness {1}".format(agent, fitness))
                print(f"Episode finished after {i} timesteps"
                      f", for a total of {end_t} simulated seconds"
                      f" (in {(end_time_episode - start_time_episode) / 10 ** 9} actual seconds).")
        else:
            raise RuntimeError(
                f"Episode has not finished after {i} timesteps"
                f" and {end_t} simulated seconds"
                f" (in {(end_time_episode - start_time_episode) / 10 ** 9} actual seconds).")
    final_fitness = getattr(np, episodes_fitness_aggr_func)(fitnesses)
    # env.close()  # use it only in main, otherwise it will be closed and
    # opened again each time it is evaluated and it slows down everything.
    return final_fitness

