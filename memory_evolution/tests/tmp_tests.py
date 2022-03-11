from collections import defaultdict, Counter
from collections.abc import Iterable, Sequence
import logging
import math
from numbers import Number, Real
from typing import Any, Literal, Optional, Union
from warnings import warn
import sys
import sys
import time
from time import perf_counter_ns
from timeit import timeit
import unittest

import gym
from gym import spaces
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import SeedSequence, default_rng
from shapely import affinity, ops
# from shapely.affinity import rotate, scale, translate
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint, MultiPolygon, GeometryCollection
# from shapely.ops import unary_union, triangulate

import memory_evolution
from memory_evolution.geometry import transform


class TestTransform(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        seed = None
        cls.sq = SeedSequence(seed)
        logging.debug(cls.sq)

    def setUp(self) -> None:
        # seed = None
        # sq = SeedSequence(seed)
        seedq = self.sq.spawn(1)[0]
        self.rng = default_rng(seedq)
        # [rng.random(3) for rng in [default_rng(s) for s in sq.spawn(3)]] v.s. [rng.random(3) for rng in [default_rng(s.entropy) for s in sq.spawn(3)]]
        # seeds = sq.generate_state(3)  # Return the requested number of words for PRNG seeding.  # np.ndarray
        # seeds = [int(s) for s in seeds]  # if int is needed.
        type(self).sq = seedq.spawn(1)[0]  # update sq for new seeds next time you ask a seed.

        self.space_size = (-10, 10)
        self.n = 100
        self.n_none = 2
        self.points_in_polygon = 7
        assert self.n > self.n_none

        self.start_time = perf_counter_ns()

    def tearDown(self) -> None:
        end_time = perf_counter_ns()
        tot_time = (end_time - self.start_time) / 10 ** 9
        print(f"Test {self.id()} it took {tot_time} seconds to be completed.")

    def generate_points(self, low, high, n, n_none=0):
        if n <= n_none:  # at least one not None
            raise ValueError(n, n_none)
        return [None] * n_none + list(self.rng.uniform(low, high, (n - n_none, 2)))

    def generate_polygons(self, low, high, n, points_in_polygon):
        polygons = []
        while len(polygons) < n:
            plg = Polygon(self.rng.uniform(low, high, (points_in_polygon, 2)))
            assert points_in_polygon + 1 == len(plg.boundary.coords), plg.wkt
            if memory_evolution.geometry.is_simple_polygon(plg):
                polygons.append(plg)
        return polygons

    def test_translate(self):
        xoffs = self.rng.random(self.n) * (self.space_size[1] - self.space_size[0]) + self.space_size[0]
        yoffs = self.rng.random(self.n) * (self.space_size[1] - self.space_size[0]) + self.space_size[0]
        points = self.generate_points(*self.space_size, self.n)
        for pt, xoff, yoff in zip(points, xoffs, yoffs):
            pt = Point(pt)
            np.testing.assert_array_equal(
                affinity.translate(pt, xoff, yoff).coords[0],  # with Point actually it works also without .coords[0]
                transform.translate(pt.coords[0], xoff, yoff))
        polygons = self.generate_polygons(*self.space_size, self.n, self.points_in_polygon)
        for plg, xoff, yoff in zip(polygons, xoffs, yoffs):
            np.testing.assert_array_equal(
                affinity.translate(plg, xoff, yoff).boundary.coords,
                transform.translate(plg.boundary.coords, xoff, yoff))
        with self.assertRaises(Exception) as cm:
            transform.translate(3., 0, 0)
        # print(f"{type(cm.exception).__qualname__}: {cm.exception}")
        with self.assertRaises(Exception) as cm:
            transform.translate([3], 0, 0)
        # print(f"{type(cm.exception).__qualname__}: {cm.exception}")
        # self.assertEqual([], transform.translate([]))
        return
        # print(plg.wkt)
        setup_str = """
from memory_evolution.geometry import transform
from shapely import affinity
from shapely import wkt
from shapely.geometry import Point, Polygon
plg = wkt.loads('POLYGON ((-2.779406512537104 -0.8383971427772874, 8.071900296391455 1.4861084803746, 3.354380654774552 7.283105127608085, -5.348559715301375 -1.783960288817031, -9.086621576879988 3.488231279648939, -3.08776005118649 -5.964994792802402, 4.588044993964361 -6.902095880902199, -2.779406512537104 -0.8383971427772874))')
xoff = 1
yoff = 2
"""
        print(f'timeit {self.id()}:', timeit("affinity.translate(plg, xoff, yoff)", setup_str, number=100000))
        print(f'timeit {self.id()}:', timeit("transform.translate(plg.boundary.coords, xoff, yoff)", setup_str, number=100000))

    def test_scale(self):
        xfacts = self.rng.random(self.n) * (self.space_size[1] - self.space_size[0]) + self.space_size[0]
        yfacts = self.rng.random(self.n) * (self.space_size[1] - self.space_size[0]) + self.space_size[0]
        xoffs = self.rng.random(self.n) * (self.space_size[1] - self.space_size[0]) + self.space_size[0]
        yoffs = self.rng.random(self.n) * (self.space_size[1] - self.space_size[0]) + self.space_size[0]
        points = self.generate_points(*self.space_size, self.n)
        for pt, xfact, yfact, xoff, yoff in zip(points, xfacts, yfacts, xoffs, yoffs):
            pt = Point(pt)
            np.testing.assert_allclose(
                affinity.scale(pt, xfact, yfact, origin=(xoff, yoff)).coords[0],  # with Point actually it works also without .coords[0]
                transform.scale(pt.coords[0], xfact, yfact, (xoff, yoff)))
        polygons = self.generate_polygons(*self.space_size, self.n, self.points_in_polygon)
        for plg, xfact, yfact, xoff, yoff in zip(polygons, xfacts, yfacts, xoffs, yoffs):
            np.testing.assert_allclose(
                affinity.scale(plg, xfact, yfact, origin=(xoff, yoff)).boundary.coords,
                transform.scale(plg.boundary.coords, xfact, yfact, (xoff, yoff)))
        with self.assertRaises(Exception) as cm:
            transform.scale(3., 0, 0)
        # print(f"{type(cm.exception).__qualname__}: {cm.exception}")
        with self.assertRaises(Exception) as cm:
            transform.scale([3], 0, 0)
        # print(f"{type(cm.exception).__qualname__}: {cm.exception}")
        # self.assertEqual([], transform.translate([]))
        return
        # print(plg.wkt)
        setup_str = """
from memory_evolution.geometry import transform
from shapely import affinity
from shapely import wkt
from shapely.geometry import Point, Polygon
plg = wkt.loads('POLYGON ((-2.779406512537104 -0.8383971427772874, 8.071900296391455 1.4861084803746, 3.354380654774552 7.283105127608085, -5.348559715301375 -1.783960288817031, -9.086621576879988 3.488231279648939, -3.08776005118649 -5.964994792802402, 4.588044993964361 -6.902095880902199, -2.779406512537104 -0.8383971427772874))')
xfact = -2
yfact = 3
xoff = 1
yoff = 2
"""
        print(f'timeit {self.id()}:', timeit("affinity.scale(plg, xfact, yfact, origin=(xoff, yoff))", setup_str, number=100000))
        print(f'timeit {self.id()}:', timeit("transform.scale(plg.boundary.coords, xfact, yfact, (xoff, yoff))", setup_str, number=100000))

    def test_rotate(self):
        origins = self.generate_points(*self.space_size, self.n, self.n_none)
        angles = self.rng.random(self.n) * 1000
        use_radians = [bool(x) for x in self.rng.integers(2, size=self.n)]
        points = self.generate_points(*self.space_size, self.n)
        for pt, an, orig, ur in zip(points, angles, origins, use_radians):
            pt = Point(pt)
            if orig is None:
                np.testing.assert_allclose(
                    affinity.rotate(pt, an, (0, 0), use_radians=ur),  # .coords[0],
                    transform.rotate(pt.coords[0], an, (0, 0), ur))
            else:
                np.testing.assert_allclose(
                    affinity.rotate(pt, an, Point(orig), ur),  # .coords[0],
                    transform.rotate(pt.coords[0], an, orig, ur))
        polygons = self.generate_polygons(*self.space_size, self.n, self.points_in_polygon)
        for plg, an, orig, ur in zip(polygons, angles, origins, use_radians):
            if orig is None:
                np.testing.assert_allclose(
                    affinity.rotate(plg, an, (0, 0), use_radians=ur).boundary.coords,
                    transform.rotate(plg.boundary.coords, an, (0, 0), ur))
            else:
                np.testing.assert_allclose(
                    affinity.rotate(plg, an, Point(orig), ur).boundary.coords,
                    transform.rotate(plg.boundary.coords, an, orig, ur))
        # self.assertEqual([], transform.rotate([]))
        return
        # print(plg.wkt)
        setup_str = """
from memory_evolution.geometry import transform
from shapely import affinity
from shapely import wkt
from shapely.geometry import Point, Polygon
plg = wkt.loads('POLYGON ((-2.779406512537104 -0.8383971427772874, 8.071900296391455 1.4861084803746, 3.354380654774552 7.283105127608085, -5.348559715301375 -1.783960288817031, -9.086621576879988 3.488231279648939, -3.08776005118649 -5.964994792802402, 4.588044993964361 -6.902095880902199, -2.779406512537104 -0.8383971427772874))')
an = 250
orig = (1, -1)
ur = False
"""
        print(f'timeit {self.id()}:', timeit("affinity.rotate(plg, an, Point(orig), ur)", setup_str, number=100000))
        print(f'timeit {self.id()}:', timeit("transform.rotate(plg.boundary.coords, an, orig, ur)", setup_str, number=100000))


if __name__ == '__main__':
    unittest.main()
