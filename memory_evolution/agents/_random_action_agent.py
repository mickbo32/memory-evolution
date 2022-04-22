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
from .exceptions import EnvironmentNotSetError
from memory_evolution.agents import BaseAgent


class RandomActionAgent(BaseAgent):
    """This agent performs random action chosen uniformly from
    the action space of the environment."""

    def __init__(self, env):
        super().__init__()
        self.set_env(env)

    def action(self, obs):
        """Overrides base method."""
        return self.get_env().action_space.sample()

    def reset(self):
        """Overrides base method."""
        pass

    @staticmethod
    def eval_genome(genome, config) -> float:
        fitness = 0.0
        return fitness

    def evolve(self, *args, **kwargs):
        return self


