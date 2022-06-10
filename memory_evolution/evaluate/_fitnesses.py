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
from memory_evolution.geometry import Pos, euclidean_distance

# todo: function for Multi-objective_optimization given fitness_func and multi_objective_optimization_algorithm
# todo: use agent to create a fitness_func Multi-objective_optimization with complexity of agent network


def fitness_func_null(*, reward, steps, done, env, agent, **kwargs) -> float:
    """Null fitness func, it returns always 0.0

    Args:
        reward: total reward of the episode
        steps: total number of steps taken by the episode
        done: if the episode is done (or aborted)
        env: environment
        agent: agent (the neural network phenotype and genotype,
            the agent who choose the actions given the observations;
            note, if agent actions needs to be evaluated or environment
            related properties of the agent needs to be evaluated,
            use env.agent instead).
        kwargs: other custom arguments
    Returns:
        The fitness value
    """
    # reward, steps, done, env = kwargs['reward'], kwargs['steps'], kwargs['done'], kwargs['env']
    fitness = 0.0
    return fitness


def fitness_func_total_reward(*, reward, steps, done, env, agent, **kwargs) -> float:
    """return total reward"""
    return reward


def fitness_func_time_minimize(*, reward, steps, done, env, agent, **kwargs) -> float:
    """-T (-inf,0], where T are the timesteps [0,+inf)
    attrs: max = 0.0;
    """
    return -steps
fitness_func_time_minimize.max = 0.


def fitness_func_time_inverse(*, reward, steps, done, env, agent, **kwargs) -> float:
    """1/T, where T are the timesteps [0,+inf)
    note: if T == 0: return +inf
    attrs: min = 0.0;
    """
    if steps == 0:
        return float('inf')
    return 1. / steps
fitness_func_time_inverse.min = 0.


def fitness_func_time_exp(*, reward, steps, done, env, agent, **kwargs) -> float:
    """exp(-T), where T are the timesteps [0,+inf)
    attrs: min = 0.0; max = 1.0;
    """
    return math.exp(-steps)
fitness_func_time_exp.min = 0.
fitness_func_time_exp.max = 1.


def minimize(f):
    """decorator to transform a function to minimize in a function to maximize (the returned function),
    using -1 multiplication (just putting a minus in front of f): returns h := -f
    """
    @functools.wraps(f)
    def f_maximize(*args, **kwargs):
        return -f(*args, **kwargs)
    return f_maximize


def minimize_inverse(f):
    """decorator to transform a function to minimize in a function to maximize (the returned function),
    using inverse: returns h := 1/f
    note: if f(x) == 0: return +inf
    note: the domain of the new function accepts only non-negative numbers (it raises an error otherwise)
    """
    @functools.wraps(f)
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
    @functools.wraps(f)
    def f_maximize(*args, **kwargs):
        return math.exp(-f(*args, **kwargs))
    return f_maximize


class BaseFitness(Callable):
    """Use this class to create your custom fitness subclass."""

    def __init__(self):
        # def instance minmax:
        # this is for pickling (pickle doesn't pickle class attributes but only instance attributes)
        # todo: should be tested, moreover nested pickling should be tested
        if hasattr(type(self), 'min'):
            self.min = type(self).min
        if hasattr(type(self), 'max'):
            self.max = type(self).max

    @abstractmethod
    # def __call__(self, **kwargs) -> float:
    def __call__(self, *, reward, steps, done, env, agent, **kwargs) -> float:
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

    def __call__(self, *, reward, steps, done, env, agent, **kwargs) -> float:
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
# # todo: neat should be able to take tuples as fitness  # però poi come fa a scegliere per il mating?  # Multi-objective_optimization, ma in questo caso dovresti anche scegnliere l'algoritmo di Multi-objective_optimization
# # fitness = (fitness, agent_distance_from_start)
# fitness += agent_distance_from_start / max(env.env_size) * .99


class FitnessDistanceInverse(BaseFitness):
    """Fitness function get distance between agent.pos and a given position,
    and it returns the inverse of it.

    f = 1 / dist_agent_to_pos ; if dist_agent_to_pos==0: f = +inf
    """

    min = 0.0

    def __init__(self, pos: Iterable):
        super().__init__()
        if not isinstance(pos, Pos):
            if not isinstance(pos, Iterable):
                raise TypeError(type(pos))
            pos = Pos(*pos)
        self._pos = pos

    @property
    def pos(self):
        return self._pos

    def __call__(self, *, reward, steps, done, env, agent, **kwargs) -> float:
        # if env.n_food_items != 1 or env.max_steps is None:
        #     RuntimeError("this fitness func can evaluate only envs with n_food_items == 1 and env.max_steps limit")
        if not done:
            RuntimeError("cannot evaluate a not done environment with this fitness func")
        # info = kwargs['info']
        # init_food_positions = info['env_info']['init_food_positions']
        # assert init_food_positions is not None
        # assert len(init_food_positions) == 1, init_food_positions
        # target_pos = init_food_positions[0]
        # assert isinstance(target_pos, Pos), target_pos
        target_pos = self.pos
        last_agent_pos = env.agent.pos
        # assert isinstance(last_agent_pos, Pos), (type(last_agent_pos), last_agent_pos)
        assert isinstance(last_agent_pos, (Pos, np.ndarray, Sequence)), (type(last_agent_pos), last_agent_pos)
        assert len(target_pos) == len(last_agent_pos), (len(target_pos), len(last_agent_pos))
        dist = euclidean_distance(last_agent_pos, target_pos)
        logging.log(logging.DEBUG + 5, f"agent distance from target: {dist}")
        assert dist >= 0
        if dist == 0:
            fitness = float('inf')
        else:
            fitness = 1 / dist
        return fitness


class FitnessDistanceMinimize(BaseFitness):
    """Fitness function get distance between agent.pos and a given position,
    and it returns the inverted of it (minus the distance).

    f = -dist_agent_to_pos
    """

    max = 0.0

    def __init__(self, pos: Iterable):
        super().__init__()
        if not isinstance(pos, Pos):
            if not isinstance(pos, Iterable):
                raise TypeError(type(pos))
            pos = Pos(*pos)
        self._pos = pos

    @property
    def pos(self):
        return self._pos

    def __call__(self, *, reward, steps, done, env, agent, **kwargs) -> float:
        # if env.n_food_items != 1 or env.max_steps is None:
        #     RuntimeError("this fitness func can evaluate only envs with n_food_items == 1 and env.max_steps limit")
        if not done:
            RuntimeError("cannot evaluate a not done environment with this fitness func")
        # info = kwargs['info']
        # init_food_positions = info['env_info']['init_food_positions']
        # assert init_food_positions is not None
        # assert len(init_food_positions) == 1, init_food_positions
        # target_pos = init_food_positions[0]
        # assert isinstance(target_pos, Pos), target_pos
        target_pos = self.pos
        last_agent_pos = env.agent.pos
        # assert isinstance(last_agent_pos, Pos), (type(last_agent_pos), last_agent_pos)
        assert isinstance(last_agent_pos, (Pos, np.ndarray, Sequence)), (type(last_agent_pos), last_agent_pos)
        assert len(target_pos) == len(last_agent_pos), (len(target_pos), len(last_agent_pos))
        dist = euclidean_distance(last_agent_pos, target_pos)
        logging.log(logging.DEBUG + 4, f"agent distance from target: {dist}")
        assert dist >= 0
        fitness = -dist
        return fitness


