import os.path
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from collections.abc import Callable, Iterable, Sequence
import functools
import logging
import math
import multiprocessing
from numbers import Number, Real
from pprint import pprint
from typing import Optional, Union, Any, Literal
from warnings import warn
import sys
import tempfile
import time

import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import SeedSequence, default_rng
import pandas as pd

import memory_evolution


def fitness_func_null(*, reward, steps, done, env, **kwargs) -> float:
    # reward, steps, done, env = kwargs['reward'], kwargs['steps'], kwargs['done'], kwargs['env']
    fitness = 0.0
    return fitness


def fitness_func_total_reward(*, reward, steps, done, env, **kwargs) -> float:
    """return total reward"""
    return reward


def fitness_func_time_inverse(*, reward, steps, done, env, **kwargs) -> float:
    """1/T, where T are the timesteps [0,+inf)
    note: if T == 0: return +inf
    attrs: min = 0.0;
    """
    if steps == 0:
        return float('inf')
    return 1. / steps
fitness_func_time_inverse.min = 0.


def fitness_func_time_exp(*, reward, steps, done, env, **kwargs) -> float:
    """exp(-T), where T are the timesteps [0,+inf)
    attrs: min = 0.0; max = 1.0;
    """
    return math.exp(-steps)
fitness_func_time_exp.min = 0.
fitness_func_time_exp.max = 1.


def minimize_inverse(f):
    """decorator to transform a function to minimize in a function to maximize (the returned function),
    using inverse: returns h := 1/f
    note: if f(x) == 0: return +inf
    note: the domain of the new function accepts only non-negative numbers (it raises an error otherwise)
    """
    # @functools.wraps(f)
    def f_maximize(*args, **kwargs):
        y = f(*args, **kwargs)
        if y < 0:
            raise ValueError("the domain of the new function accepts only non-negative numbers")
        if y == 0:
            return float('inf')
        return 1 / y
    return f_maximize


def minimize_exp(f):
    """decorator to transform a function to minimize in a function to maximize (the returned function),
    using exp: returns h := exp(-f)
    """
    # @functools.wraps(f)
    def f_maximize(*args, **kwargs):
        return math.exp(-f(*args, **kwargs))
    return f_maximize


class BaseFitness(Callable):
    """Use this class to create your custom fitness subclass."""

    @abstractmethod
    def __call__(self, **kwargs) -> float:
        raise NotImplementedError


class FitnessRewardAndSteps(BaseFitness):
    """Multi-objective_optimization with [normalized] linear scalarization.
    Examples:
        >>> fitness_func = FitnessRewardAndSteps()
    """

    def __init__(self, reward_weight=1., steps_weight=1., normalize_weights=True):
        if normalize_weights:
            tot = reward_weight + steps_weight
            reward_weight /= tot
            steps_weight /= tot
        self._normalize_weights = normalize_weights
        self.reward_weight = reward_weight
        self.steps_weight = steps_weight

        self.min = 0  # assuming non-negative reward
        assert self.reward_weight > 0 and self.steps_weight > 0, 'assuming normalization of reward and steps and positive weights'
        self.max = self.reward_weight + self.steps_weight  # assuming normalization of reward and steps and positive weights
        assert self.min <= self.max

    def __call__(self, *, reward, steps, done, env, **kwargs) -> float:
        assert reward >= 0, (reward, 'assuming non-negative reward')
        #assert isinstance(env, memory_evolution.envs.BaseForagingEnv), type(env)  # todo: use weights per normalizzare, usa env in FitnessRewardAndStepsBaseForagingEnv
        reward /= env.maximum_reward  # normalize total_reward
        assert 0 <= reward <= 1, reward
        steps /= env.max_steps  # normalize timesteps
        # note: the agent could take the last food item in the last timestep
        #   thus, 'timesteps_normalized' could be 1
        # note2: more timesteps is bad, less timestep is good:
        #        thus fitness: 1 - 'timesteps_normalized'
        #        or fitness: 1 / 'timesteps_normalized'
        assert 0 <= steps <= 1, steps
        fitness = self.reward_weight * reward + self.steps_weight * (1 - steps)
        # todo:  (1 - steps) v.s. 1/steps
        assert self.min <= fitness <= self.max, (fitness, (self.min, self.max))
        return fitness

    def __repr__(self):
        return f"{type(self).__qualname__}({self.reward_weight}, {self.steps_weight}, {self._normalize_weights})"


    # # fitness: (total_reward, agent_distance_from_start):
    # end_agent_pos = info['state']['agent'].pos
    # agent_distance_from_start = memory_evolution.geometry.euclidean_distance(first_agent_pos, end_agent_pos)
    # # print(f"agent_distance_from_start: {agent_distance_from_start}")
    # logging.debug(f"agent_distance_from_start: {agent_distance_from_start}")
    # # todo: neat should be able to take tuples as fitness  # però poi come fa a scegliere per il mating?
    # # fitness = (fitness, agent_distance_from_start)
    # fitness += agent_distance_from_start / max(env.env_size) * .99

    # # fitness: (total_reward, 1 - timesteps used to get all food items if all food items collected):
    # if env.food_items_collected == env.n_food_items:
    #     timesteps_normalized = step / env.max_steps
    #     # note: the agent could take the last food item in the last timestep
    #     #   thus, 'timesteps_normalized' could be 1
    #     # note2: more timesteps is bad, less timestep is good, thus fitness: 1 - 'timesteps_normalized'
    #     logging.debug(f"timesteps_normalized: {timesteps_normalized};"
    #                   f" 1-timesteps_normalized: {1 - timesteps_normalized}")
    #     fitness += 1 - timesteps_normalized
    # else:
    #     assert step == env.step_count == env.max_steps
    #     assert env.food_items_collected < env.n_food_items



