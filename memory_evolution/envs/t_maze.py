from collections import defaultdict, Counter
from collections.abc import Sequence
import math
from numbers import Number, Real
from typing import Optional, Union, Any
from warnings import warn
import sys

import gym
from gym import spaces
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import SeedSequence, default_rng
import pygame as pg
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPolygon
from shapely.ops import unary_union, triangulate

from memory_evolution.geometry import is_simple_polygon, Pos
from memory_evolution.utils import MustOverride, override

from .maze_foraging import MazeForagingEnv, Agent, FoodItem


class TMaze(MazeForagingEnv):

    def __init__(self,
                 corridor_width: Real = .2,
                 window_size: Union[int, Sequence[int]] = 640,  # (640, 480),
                 env_size: Union[float, Sequence[float]] = 1.,
                 *args,
                 **kwargs
                 ) -> None:

        n_channels = 3
        self._corridor_width = corridor_width

        up_right = np.asarray(self._get_env_size(env_size))
        down_left = np.asarray((0., 0.))
        up_left = np.asarray((0., up_right[1]))
        down_right = np.asarray((up_right[0], 0.))
        center = up_right / 2

        if corridor_width >= up_right.min(axis=None):
            raise ValueError("'corridor_width' too big")

        borders = [
            [down_left,
             (center[0] - corridor_width / 2, down_left[1]),
             (center[0] - corridor_width / 2, up_right[1] - corridor_width),
             (down_left[0], up_right[1] - corridor_width)],
            [down_right,
             (center[0] + corridor_width / 2, down_right[1]),
             (center[0] + corridor_width / 2, up_right[1] - corridor_width),
             (down_right[0], up_right[1] - corridor_width)],
        ]

        borders = [Polygon(plg) for plg in borders]

        super().__init__(
            borders,
            window_size,
            env_size,
            *args,
            **kwargs
        )

        assert tuple(up_right) == self._env_size
        assert n_channels == self._n_channels, self._n_channels

