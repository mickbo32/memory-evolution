from collections import defaultdict, Counter
from collections.abc import Iterable, Sequence
import logging
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
from shapely.ops import unary_union

import memory_evolution
from memory_evolution.geometry import is_simple_polygon, Pos
from memory_evolution.utils import convert_image_to_pygame
from memory_evolution.utils import MustOverride, override
from memory_evolution.utils import EmptyDefaultValueError, get_default_value

from .base_foraging import BaseForagingEnv, Agent, FoodItem, get_valid_item_positions_mask


# todo: move in tests:
def __test():
    plg = Polygon(((0,0), (1,1), (0,1)))
    try:
        plg.boundary.coords[0] = (2, 2)
    except TypeError as err:
        assert err.args == ("'CoordinateSequence' object does not support item assignment",)
    else:
        raise AssertionError("Polygon should be read-only (immutable), "
                             "otherwise code below is not good for the @property valid_platform,"
                             "which could be changed by the user, instead this should NOT be allowed.")
__test()


class MazeForagingEnv(BaseForagingEnv):

    def __init__(self,
                 *,
                 borders: Optional[Iterable[Polygon]] = None,
                 platform: Optional[Polygon] = None,
                 random_init_agent_position: Optional[Sequence[Pos]] = None,
                 **kwargs
                 ) -> None:
        """Build a Maze environment.

        The maze is build using ``borders`` or ``platform``, only one of them should be provided.

        Args:
            borders: list of borders, if ``borders`` are provided they are used to build the maze.
            platform: the platform of the maze, a single connected valid Polygon on which the agent can move;
                if ``platform`` is provided it is used to build the maze.
            random_init_agent_position: list of positions from which the initial agent position will be chosen.
        """
        if kwargs.get('init_agent_position', None) is not None and random_init_agent_position is not None:
            raise ValueError("'init_agent_position' and 'random_init_agent_position' cannot be provided together.")
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
        border_color = self._outside_color
        background = self._soil.copy()
        borders_mask = get_valid_item_positions_mask(self._platform, 0., self.window_size, self.env_size)
        background = convert_image_to_pygame(background)
        assert self._soil.shape[1::-1] == background.get_size()
        self._background_img = borders_mask.to_surface(
            background,
            setsurface=background,
            setcolor=self._background_color,
            unsetcolor=border_color
        )

        # update valid positions:
        # valid positions:
        self._compute_and_set_valid_positions(self._platform)

        # random_init_agent_position used for choosing an initial agent position:
        if kwargs.get('init_agent_position', None) is not None and random_init_agent_position is not None:
            raise ValueError("'init_agent_position' and 'random_init_agent_position' cannot be provided together.")
        if random_init_agent_position is not None:
            assert self._init_agent_position is None
            if not isinstance(random_init_agent_position, Sequence):
                raise TypeError(
                    "'random_init_agent_position' must be a Sequence of positions among which choosing from")
            for pos in random_init_agent_position:
                if not isinstance(pos, Iterable):
                    raise TypeError(
                        "position in 'random_init_agent_position' should be something from which a point can be generated (an Iterable)")
            random_init_agent_position = [(pos if isinstance(pos, Pos) else Pos(*pos))
                                          for pos in random_init_agent_position]
            for pos in random_init_agent_position:
                if len(pos) != 2:
                    raise ValueError("position in 'random_init_agent_position' should be 2D (and without channels)")
        self._random_init_agent_position = random_init_agent_position
        # TODO: check random_init_agent_position are all valid agent positions

    @property
    @override
    def maximum_reward(self):
        return super().maximum_reward

    @property
    def platform(self):
        """Valid platform in which the agent can move.
        Platform is a simple polygon, where each point in it is reachable from any other point in it."""
        return self._platform

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

    def _update_env_img(self) -> None:
        super()._update_env_img()

    def _render_env(self, screen) -> None:
        super()._render_env(screen)

    def _init_state(self) -> None:
        if self._random_init_agent_position is not None:
            assert self._init_agent_position is None
            # don't use self.np_random.choice() otherwise it converts Pos object in np.ndarray array
            idx = self.np_random.integers(0, len(self._random_init_agent_position))
            __init_agent_position = self._random_init_agent_position[idx]
            self._init_agent_position = __init_agent_position
            self.debug_info['_init_state'][f'_{type(self).__name__}__init_agent_position'] = __init_agent_position
            logging.debug(f"_init_state: random_init_agent_position chosen: {self._init_agent_position}")
        super()._init_state()
        if self._random_init_agent_position is not None:
            self._init_agent_position = None

    def _update_state(self, action) -> Real:
        return super()._update_state(action)

    def _get_observation(self) -> np.ndarray:
        return super()._get_observation()

    def _get_point_color(self, point,
                         use_env_space=get_default_value(BaseForagingEnv._get_point_color, 'use_env_space'),
                         use_neighbours=get_default_value(BaseForagingEnv._get_point_color, 'use_neighbours'),
                         aggregation_func=get_default_value(BaseForagingEnv._get_point_color, 'aggregation_func'),
                         ):
        # base method it reads the env_img, thus if you update that in the init it is enough.
        return super()._get_point_color(point,
                                        use_env_space=use_env_space,
                                        use_neighbours=use_neighbours,
                                        aggregation_func=aggregation_func)

    def _is_done(self) -> bool:
        return super()._is_done()

    def _get_info(self) -> dict:
        info = super()._get_info()

        # platform:
        info['env_info']['platform'] = self._platform

        # self._random_init_agent_position:
        assert self._init_agent_position is None
        assert 'random_init_agent_position' not in info['env_info']
        info['env_info']['random_init_agent_position'] = self._random_init_agent_position
        assert 'init_agent_position' in info['env_info']
        assert info['env_info']['init_agent_position'] is None
        if self._random_init_agent_position is not None:
            info['env_info']['init_agent_position'] = self.debug_info.get(
                '_init_state', {}).get(f'_{type(self).__name__}__init_agent_position', None)

        return info

    def is_valid_position(self, pos, item: Literal['agent', 'food'], is_env_pos: bool = True) -> bool:
        return super().is_valid_position(pos, item, is_env_pos)

    def _get_random_non_overlapping_positions(self,
                                              n,
                                              radius: Union[list, int],
                                              items=None,
                                              ) -> list[Pos]:
        return memory_evolution.geometry.get_random_non_overlapping_positions_with_lasvegas(
            n, radius, self._platform, self._env_size, self.env_space.np_random,
            self, items,
            optimization_with_platform_triangulation=True,
        )

