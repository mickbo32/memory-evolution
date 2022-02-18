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
import pygame
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPolygon
from shapely.ops import unary_union, triangulate

from memory_evolution.utils import COLORS, is_color, is_simple_polygon, Pos, convert_image_to_pygame
from memory_evolution.utils import MustOverride, override

from .maze_foraging import MazeForagingEnv, Agent, FoodItem


class TMaze(MazeForagingEnv):

    def __init__(self,
                 corridor_width: Real = .2,
                 window_size: Union[int, Sequence[int]] = 640,  # (640, 480),
                 env_size: Union[float, Sequence[float]] = 1.,
                 n_food_items: int = 3,
                 rotation_step: float = 20,
                 forward_step: float = .01,
                 agent_size: float = .05,
                 food_size: float = .05,
                 vision_depth: float = .2,
                 vision_field_angle: float = 180.,
                 vision_resolution: int = 10,
                 fps: Optional[int] = None,
                 seed=None,
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
            borders=borders,
            window_size=window_size,
            env_size=env_size,
            n_food_items=n_food_items,
            rotation_step=rotation_step,
            forward_step=forward_step,
            agent_size=agent_size,
            food_size=food_size,
            vision_depth=vision_depth,
            vision_field_angle=vision_field_angle,
            vision_resolution=vision_resolution,
            fps=fps,
            seed=seed,
        )

        assert tuple(up_right) == self._env_size
        assert n_channels == self._n_channels, self._n_channels

