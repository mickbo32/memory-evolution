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
from memory_evolution.utils import evaluate_agent
from memory_evolution.utils import MustOverride, override


class RnnNeatAgent(BaseNeatAgent):

    phenotype_class = neat.nn.RecurrentNetwork

    def __init__(self, config, genome=None):
        super().__init__(config, genome=genome)

    def action(self, observation: np.ndarray) -> np.ndarray:
        """Extends the base method."""
        super().action(observation)
        net = self.phenotype
        action = net.activate(self.normalize_observation(observation))
        return np.asarray(action, dtype=self._env.action_space.dtype)

    def reset(self) -> None:
        """Extends the base method."""
        super().reset()

    def visualize_genome(self, genome, name='Genome',
                         view=False, filename=None,
                         show_disabled=True, prune_unused=False):
        """Extends the base method."""
        super().visualize_genome(genome,
                                 name=name,
                                 view=view,
                                 filename=filename,
                                 show_disabled=show_disabled,
                                 prune_unused=prune_unused)

