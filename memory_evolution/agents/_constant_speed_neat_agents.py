from abc import ABC, abstractmethod
from collections import defaultdict, Counter, deque
from collections.abc import Iterable, Sequence
from functools import reduce
import inspect
import json
import math
import multiprocessing
from numbers import Number, Real
from operator import mul
import os
import pickle
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
import pygame

from memory_evolution.agents import BaseNeatAgent, RnnNeatAgent
from memory_evolution.utils import normalize_observation
from memory_evolution.utils import MustOverride, override


class ConstantSpeedRnnNeatAgent(RnnNeatAgent):

    def __init__(self, config, genome=None):
        super().__init__(config, genome=genome)
        self.node_names = {0: 'rotation'}

    def set_env(self, env: gym.Env) -> None:
        """Extends base class method with the same name."""
        obs_size = reduce(mul, env.observation_space.shape, 1)
        if self.config.genome_config.num_inputs != obs_size:
            raise ValueError(
                f"Network input ({self.config.genome_config.num_inputs}) "
                f"doesn't fits 'env.observation_space' "
                f"({env.observation_space.shape} -> size: {obs_size}), "
                "change config.genome_config.num_inputs in your config file "
                "or change environment; "
                "config.genome_config.num_inputs should be equal to the total "
                "size of 'env.observation_space.shape' (total size, "
                "i.e. the sum of its dimensions)."
            )
        act_size = reduce(mul, env.action_space.shape, 1)
        if len(env.action_space.shape) != 1:
            raise ValueError("'len(env.action_space.shape)' should be 1;")
        assert 2 == act_size, act_size
        if self.config.genome_config.num_outputs != act_size - 1:
            raise ValueError(
                f"Network output ({self.config.genome_config.num_outputs}) "
                f"doesn't fits 'env.action_space'"
                f"({env.action_space.shape} -> size: {act_size}), "
                "change config.genome_config.num_outputs in your config file "
                "or change environment."
            )
        super(BaseNeatAgent, self).set_env(env)

    def action(self, observation: np.ndarray) -> np.ndarray:
        """Extends the base method."""
        super().action(observation)
        net = self.phenotype
        assert (2,) == self._env.action_space.shape, self._env.action_space.shape
        net_output = net.activate(self._normalize_observation(observation))
        assert 1 == len(net_output), len(net_output)
        rotation_output = net_output[0]
        forward_output = self._env.action_space.high[1]
        action = (rotation_output, forward_output)
        action = np.asarray(action, dtype=self._env.action_space.dtype)
        assert self._env.action_space.shape == action.shape, (self._env.action_space.shape, action.shape)
        assert self._env.action_space.contains(action), (self._env.action_space, action)
        return action

    def reset(self) -> None:
        """Extends the base method."""
        super().reset()

