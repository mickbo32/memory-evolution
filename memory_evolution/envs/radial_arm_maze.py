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

from memory_evolution.geometry import is_simple_polygon, Pos, transform
from memory_evolution.utils import MustOverride, override

from .maze_foraging import MazeForagingEnv, Agent, FoodItem

# # For debugging:
# import geopandas as gpd


class RadialArmMaze(MazeForagingEnv):

    def __init__(self,
                 arms: int = 4,
                 corridor_width: Optional[float] = None,  # default: max_corridor_width / 3
                 window_size: Union[int] = 320,
                 env_size: Union[float] = 1.,
                 *args,
                 **kwargs
                 ) -> None:

        n_channels = 3
        if not isinstance(window_size, Real):
            raise TypeError("Allowed square window only, 'window_size' must be a number, the side length.")
        if not isinstance(env_size, Real):
            raise TypeError("Allowed square env only, 'env_size' must be a number, the side length.")
        self._radius = env_size / 2
        if arms < 2:
            raise ValueError("'arms' number must be greater or equal to 2")
        # note: using radians
        max_corridor_angle = 2 * math.pi / arms
        max_corridor_width = 2 * self._radius * math.sin(max_corridor_angle / 2)
        # print('max_corridor_width;', max_corridor_width)
        # print('max_corridor_angle:', math.degrees(max_corridor_angle))
        if corridor_width is None:
            corridor_width = max_corridor_width / 3
        if corridor_width >= max_corridor_width:
            raise ValueError("'corridor_width' too big")
        self._corridor_width = corridor_width
        self._corridor_angle = 2 * math.asin(self._corridor_width / (2 * self._radius))  # in radians
        self._intra_arms_angle = 2 * math.pi / arms - self._corridor_angle  # in radians
        self._inner_angle = self._corridor_angle + self._intra_arms_angle  # in radians
        assert math.isclose(max_corridor_angle, self._inner_angle)
        self._inner_radius = self._corridor_width / 2 / math.sin(self._inner_angle / 2)
        # print([math.degrees(alpha) for alpha in (self._corridor_angle, self._intra_arms_angle, self._inner_angle)])

        up_right = np.asarray(self._get_env_size(env_size))
        down_left = np.asarray((0., 0.))
        up_left = np.asarray((0., up_right[1]))
        down_right = np.asarray((up_right[0], 0.))
        center = up_right / 2

        # Start a point in the middle of the bottom side of env:
        # and an inner point 'self._inner_radius' below the center of env:
        corridor_end_point = np.asarray((center[0], down_left[1]))
        inner_point = np.asarray((center[0], center[1] - self._inner_radius))

        # Rotate the point half self._corridor_angle counterclockwise with the center of the env as origin of rotation:
        # and rotate the inner one of half self._inner_angle:
        corridor_end_point = transform.rotate(corridor_end_point, self._corridor_angle / 2, center, use_radians=True)
        inner_point = transform.rotate(inner_point, self._inner_angle / 2, center, use_radians=True)
        _first_corridor_end_point = corridor_end_point
        _first_inner_point = inner_point

        # Rotate the inner point self._inner_angle counterclockwise -> get next inner point,
        # Rotate the point self._intra_arms_angle counterclockwise -> get next corridor start,
        # Rotate the point self._corridor_angle counterclockwise -> get next corridor end,
        # Iterate rotation for each arm:
        points = []
        for a in range(arms):
            points.append(corridor_end_point)
            points.append(inner_point)
            corridor_start_point = transform.rotate(corridor_end_point, self._intra_arms_angle, center, use_radians=True)
            points.append(corridor_start_point)
            inner_point = transform.rotate(inner_point, self._inner_angle, center, use_radians=True)
            corridor_end_point = transform.rotate(corridor_start_point, self._corridor_angle, center, use_radians=True)
        np.testing.assert_allclose(_first_corridor_end_point, corridor_end_point)
        np.testing.assert_allclose(_first_inner_point, inner_point)
        assert 3 * arms == len(points), len(points)

        # outer_points = (down_left, down_right, up_right, up_left)
        # maze_border = Polygon(outer_points, holes=[points])
        # assert 1 == len(maze.interiors), len(maze.interiors)
        # assert 3 * arms == len(points) == len(maze_border.interiors[0].coords) - 1, (
        #     len(points), len(maze_border.interiors[0].coords))
        maze = Polygon(points)
        assert 3 * arms == len(points) == len(maze.boundary.coords) - 1, (
            len(points), len(maze.boundary.coords))

        # # plotting to DEBUG:
        # fig, ax = plt.subplots()
        # gpd.GeoSeries([Point(p) for p in points]).plot(ax=ax, color='r')
        # gpd.GeoSeries(Point(center)).plot(ax=ax, color='b')
        # gpd.GeoSeries(Point(_first_corridor_end_point)).plot(ax=ax, color='g')
        # gpd.GeoSeries(Point(_first_inner_point)).plot(ax=ax, color='g')
        # gpd.GeoSeries(Point(corridor_end_point)).plot(ax=ax, color='purple')
        # gpd.GeoSeries(Point(inner_point)).plot(ax=ax, color='purple')
        # gpd.GeoSeries(LineString(points)).plot(ax=ax, color='gray')
        # plt.show()
        # gpd.GeoSeries(maze.buffer(0)).plot()
        # plt.show()

        # maze_border: RadialArmMaze(3, 1., env_size=2.)  # not valid polygon
        # maze_border: RadialArmMaze(2), RadialArmMaze(4), RadialArmMaze(3, 1.5, env_size=2.)  # valid polygons with holes
        plg = maze
        assert isinstance(plg, Polygon), plg.wkt
        assert plg.is_valid, plg.wkt
        assert isinstance(plg.boundary, LineString) and not list(plg.interiors), plg.wkt

        super().__init__(
            platform=maze,
            window_size=window_size,
            env_size=env_size,
            *args,
            **kwargs
        )
        self._update_init_params(['platform', 'window_size', 'env_size'])

        assert tuple(up_right) == self._env_size
        assert n_channels == self._n_channels, self._n_channels

    @property
    @override
    def maximum_reward(self):
        return super().maximum_reward

