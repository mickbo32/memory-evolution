from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from collections.abc import Iterable, Sequence
import math
import multiprocessing
from numbers import Number, Real
import os
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

from memory_evolution.evaluate import evaluate_agent
from memory_evolution.utils import normalize_observation
from memory_evolution.utils import MustOverride, override
from .exceptions import EnvironmentNotSetError


class BaseAgent(ABC):
    """The agent, is an object with an ``action()`` method which takes the
    observation from the environment as argument and returns a valid action,
    and it has a ``reset()`` method which reset the agent to an initial
    state ``t==0``.
    """

    def __init__(self):
        self._env = None

    def get_env(self) -> gym.Env:
        if self._env is None:
            raise EnvironmentNotSetError
        return self._env

    def set_env(self, env: gym.Env) -> None:
        # if not np.array_equiv(env.observation_space.low, 0) or not np.array_equiv(env.observation_space.high, 1):
        #     raise ValueError("env.observation_space should be in the range [0,1]")
        self._env = env

    @staticmethod
    def _normalize_observation(observation: np.ndarray) -> np.ndarray:
        return normalize_observation(observation)
        # returned dtype is dtype('float64'), should it be cast to np.float32 ?
        # casting is not necessary for neat (I think neat main graph it
        # doesn't use numpy).

    @abstractmethod
    def action(self, observation: np.ndarray) -> np.ndarray:
        """Takes an observation from the environment as argument and returns
        a valid action.

        Args:
            observation: observation from the environment.

        Returns:
            An action.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the agent to an initial state ``t==0``."""
        pass

