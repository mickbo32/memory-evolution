from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from collections.abc import Iterable, Sequence
import math
import multiprocessing
from numbers import Number, Real
from typing import Optional, Union, Any, Literal
from warnings import warn
import sys

import gym
from gym import spaces
import matplotlib as mpl
import matplotlib.pyplot as plt
import neat
import numpy as np
from numpy.random import SeedSequence, default_rng
import pygame

from memory_evolution.agents import BaseNeatAgent
from memory_evolution.utils import normalize_observation
from memory_evolution.utils import MustOverride, override


class RnnNeatAgent(BaseNeatAgent):

    phenotype_class = neat.nn.RecurrentNetwork

    def __init__(self, config, genome=None):
        super().__init__(config, genome=genome)

    def action(self, observation: np.ndarray) -> np.ndarray:
        """Extends the base method."""
        super().action(observation)
        net = self.phenotype
        action = net.activate(self._normalize_observation(observation))
        return np.asarray(action, dtype=self._env.action_space.dtype)

    def reset(self) -> None:
        """Extends the base method."""
        super().reset()

