from collections import defaultdict, Counter
from collections.abc import Iterable, Sequence
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

from memory_evolution.geometry import is_simple_polygon, Pos, get_random_non_overlapping_positions_with_triangulation
from memory_evolution.utils import convert_image_to_pygame
from memory_evolution.utils import MustOverride, override

from .base_foraging import BaseForagingEnv, Agent, FoodItem, get_valid_item_positions_mask


class MazeForagingEnv(BaseForagingEnv):

    def __init__(self,
                 *,
                 borders: Optional[Iterable[Polygon]] = None,
                 platform: Optional[Polygon] = None,
                 **kwargs
                 ) -> None:
        super().__init__(
            **kwargs
        )

        if 1 != sum((borders is None,  platform is None)):
            raise ValueError("only one among 'borders' or 'platform' should be provided.")

        if borders is not None:
            assert platform is None
            platform = self.get_platform_from_borders(borders)
        else:
            if not isinstance(platform, Polygon):
                raise TypeError(f"'platform' should be a Polygon, instead got platform of type {type(platform)}")
            if not is_simple_polygon(self._platform):
                raise ValueError(f"'platform' should be a simple polygon")

        # override self._platform:
        assert isinstance(platform, Polygon), type(platform)
        assert is_simple_polygon(platform), platform.wkt
        self._platform = platform

        # update background np.ndarray and background_img pygame image
        border_color = self.outside_color
        background = self._soil.copy()
        borders_mask = get_valid_item_positions_mask(self._platform, 0., self.window_size, self.env_size)
        background = convert_image_to_pygame(background)
        assert self._soil.shape[1::-1] == background.get_size()
        self._background_img = borders_mask.to_surface(
            background,
            setsurface=background,
            unsetcolor=border_color
        )

        # update valid positions:
        # valid positions:
        self._compute_and_set_valid_positions(self._platform)

    def get_platform_from_borders(self, borders: Iterable):

        if not isinstance(borders, Iterable):
            raise TypeError(f"'borders' should be iterable, instead borders type is not ({type(borders)})")
        if not isinstance(borders, list):
            borders = list(borders)

        if not all(isinstance(plg, Polygon) for plg in borders):
            raise TypeError("'borders' should be an iterable of 'Polygon' objects")
        if not all(plg.is_valid for plg in borders):
            raise ValueError("'borders' contain a non-valid polygon object")
        # empty interiors, no holes:
        if not all(isinstance(plg.boundary, LineString) and not list(plg.interiors)
                   for plg in borders):
            # not MultiLineString boundary
            # empty interiors, no holes
            raise ValueError("'borders' contain a complex polygon object, "
                             "only simple polygons are allowed")
        __borders_coords_on_screen = [[self.get_point_env2win(pt) for pt in plg.boundary.coords]
                                      for plg in borders]
        __borders_union = unary_union(borders)
        for plg in __borders_coords_on_screen:
            plg = Polygon(plg)
            assert is_simple_polygon(plg), plg.wkt
        # get new platform from base self._platform:
        base_platform = Polygon((Point(0, 0), Point(0, self._env_size[1]),
                                 Point(*self._env_size), Point(self._env_size[0], 0)))
        platform = base_platform.difference(__borders_union)
        if not is_simple_polygon(platform):
            raise ValueError(f"'borders' create an unreachable part in the maze "
                             f"(or there is a problem with the new platform: "
                             f"{(platform.wkt, platform.boundary)})")

        return platform

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
        info['env_info']['platform'] = self._platform
        return info

    def is_valid_position(self, pos, item: Literal['agent', 'food'], is_env_pos: bool = True) -> bool:
        return super().is_valid_position(pos, item, is_env_pos)

    # def _get_random_non_overlapping_positions(self,
    #                                           n,
    #                                           radius: Union[list, int],
    #                                           items=None,
    #                                           ) -> list[Pos]:
    #     return get_random_non_overlapping_positions_with_triangulation(
    #         n, radius, self._platform, self._env_size, self.env_space.np_random)

