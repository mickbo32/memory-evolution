from collections import defaultdict, Counter
from collections.abc import Sequence
import math
from numbers import Number, Real
from typing import Any, Literal, Optional, Union
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
from memory_evolution.utils import convert_image_to_pygame
from memory_evolution.utils import MustOverride, override

from .base_foraging import BaseForagingEnv, Agent, FoodItem, get_valid_item_positions_mask


class MazeForagingEnv(BaseForagingEnv):

    def __init__(self,
                 borders: list[Polygon],
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
                 max_steps: Optional[int] = None,
                 fps: Optional[int] = None,
                 seed=None,
                 ) -> None:
        super().__init__(
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
            max_steps=max_steps,
            fps=fps,
            seed=seed,
        )

        if not all(isinstance(plg, Polygon) for plg in borders):
            raise TypeError("'borders' should be a list of 'Polygon' objects")
        if not all(plg.is_valid for plg in borders):
            raise ValueError("'borders' contain a non-valid polygon object")
        if not all(isinstance(plg.boundary, LineString) and not list(plg.interiors)
                   for plg in borders):
            # not MultiLineString boundary
            # empty interiors, no holes
            raise ValueError("'borders' contain a complex polygon object, "
                             "only simple polygons are allowed")

        self.border_color = self.outside_color
        self._borders = borders
        self.__borders_coords_on_screen = [[self.get_point_env2win(pt) for pt in plg.boundary.coords]
                                           for plg in self._borders]
        self.__borders_union = unary_union(self._borders)
        for plg in self.__borders_coords_on_screen:
            plg = Polygon(plg)
            assert is_simple_polygon(plg), plg.wkt
        # update self._platform:
        self._platform = self._platform.difference(self.__borders_union)
        if not is_simple_polygon(self._platform):
            raise ValueError(f"'borders' create an unreachable part in the maze "
                             f"(or there is a problem with self._platform: "
                             f"{(self._platform.wkt, self._platform.boundary)})")

        # update background np.ndarray and background_img pygame image
        background = self._soil.copy()
        borders_mask = get_valid_item_positions_mask(self._platform, 0., self.window_size, self.env_size)
        background = convert_image_to_pygame(background)
        assert self._soil.shape[1::-1] == background.get_size()
        self._background_img = borders_mask.to_surface(
            background,
            setsurface=background,
            unsetcolor=self.border_color
        )

        # update valid positions:
        # valid positions:
        self._compute_and_set_valid_positions(self._platform)

    def _draw_env(self, screen) -> None:
        # draw stuff:
        # pass

        # draw agent and food items later, so you can see them above borders
        # if for some reasons they are plotted outside the maze, but mainly you can see
        # vision points above borders
        super()._draw_env(screen)

    def _init_state(self) -> None:
        super()._init_state()

    def _update_state(self, action) -> Real:
        return super()._update_state(action)

    def _get_observation(self) -> np.ndarray:
        return super()._get_observation()

    def _get_point_color(self, point):
        # base method it reads the env_img, thus if you update that in the init it is enough.
        return super()._get_point_color(point)

    def _is_done(self) -> bool:
        return super()._is_done()

    def _get_info(self) -> dict:
        info = super()._get_info()
        info['env_info']['borders'] = self._borders
        return info

    def is_valid_position(self, pos, item: Literal['agent', 'food'], is_env_pos: bool = True) -> bool:
        return super().is_valid_position(pos, item, is_env_pos)

