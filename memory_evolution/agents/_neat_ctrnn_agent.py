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
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint, MultiPolygon
from shapely.ops import unary_union, triangulate

from memory_evolution.agents import BaseNeatAgent
from memory_evolution.utils import MustOverride, override


class CtrnnNeatAgent(BaseNeatAgent):

    phenotype_class = neat.ctrnn.CTRNN

    def __init__(self, config, genome=None):
        super().__init__(config, genome=genome)

    def action(self, observation: np.ndarray) -> np.ndarray:
        """Extends the base method."""
        super().action(observation)
        net = self.phenotype
        action = net.activate(observation.reshape(-1))
        return np.asarray(action, dtype=self._env.action_space.dtype)

    def reset(self) -> None:
        """Extends the base method."""
        super().reset()
        #self._t = 0

