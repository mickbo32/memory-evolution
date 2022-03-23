from collections import defaultdict, Counter
from collections.abc import Sequence
import logging
import math
from numbers import Number, Real
from typing import Any, Literal, Optional, Union
from warnings import warn
import sys
import sys

import gym
from gym import spaces
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import SeedSequence, default_rng
# from shapely.affinity import rotate, scale, translate
# from shapely.affinity import affine_transform
# from shapely.ops import transform
# from scipy.spatial.transform import Rotation as R
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint, MultiPolygon, GeometryCollection
from shapely.ops import unary_union, triangulate

# DEBUG:
# shapely visualization on plt: (import also: import matplotlib.pyplot as plt)
import geopandas as gpd


class Pos:
    """Coordinates of a Position.
    `coords` coordinates should be all float
    """

    def __init__(self, *coords):
        assert isinstance(coords, tuple)
        if not all(isinstance(c, float) for c in coords):
            raise TypeError('`coords` coordinates should be all float')
        self._coords = tuple(coords)

    def __len__(self):
        return len(self._coords)

    def __iter__(self):
        return iter(self._coords)

    def __getitem__(self, item):
        return self._coords[item]

    def __eq__(self, other):
        if isinstance(other, Pos):
            return self._coords == other._coords
        return False

    def __hash__(self):
        return hash(self._coords)

    @property
    def x(self):
        if len(self) > 3 or len(self) < 1:
            raise AttributeError('When `len(self)` is higher than 3 or less than 1, `x` attribute is not supported')
        return self._coords[0]

    @property
    def y(self):
        if len(self) > 3 or len(self) < 2:
            raise AttributeError('When `len(self)` is higher than 3 or less than 2, `y` attribute is not supported')
        return self._coords[1]

    @property
    def z(self):
        if len(self) > 3 or len(self) < 3:
            raise AttributeError('When `len(self)` is higher than 3 or less than 3, `z` attribute is not supported')
        return self._coords[2]

    def __repr__(self):
        return (f"{type(self).__name__}("
                + ", ".join([str(c) for c in self._coords]) + ")")
        # return (f"{__name__ if __name__ != '__main__' else ''}.{type(self).__qualname__}("
        #         + ", ".join([str(c) for c in self._coords]) + ")")


# todo: move in tests
np.testing.assert_array_equal(np.asarray((2., 3.)), np.asarray(Pos(2., 3.)))


def euclidean_distance(a, b):
    assert len(a) == len(b), (a, b)
    return math.sqrt(sum((bx - ax) ** 2 for ax, bx in zip(a, b)))


def _pixels_on_line(x0, y0, x1, y1):
    """Bresenham's line algorithm"""
    dx = x1 - x0
    assert dx >= 0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy
    m2 = 2 * dy
    m_err = m2 - dx
    y = y0
    for x in range(x0, x1 + 1):
        # print(dx, dy, m2, m_err)
        yield x, y
        if m_err > 0:
            y = y + yi
            m_err -= 2 * dx
        m_err += m2


def pixels_on_line(x0, y0, x1, y1):
    """Bresenham's line algorithm, it returns the indexes (i.e. pixels)
    of the line in the window image space;
    this version preserve the order of points (first point is always first
    point yielded, last point is always last point yielded) and yields
    pixels of the line in order from first to the last.

    Note: x0, y0, x1, y1 are indexes in the window image.
    Note2: It yieilds the pixels,
    """
    dx = x1 - x0
    dy = y1 - y0
    if abs(dy) <= abs(dx):
        if x0 <= x1:
            for x, y in _pixels_on_line(x0, y0, x1, y1):
                yield x, y
        else:
            for x, y in _pixels_on_line(-x0, y0, -x1, y1):
                yield -x, y
    else:
        if y0 <= y1:
            for y, x in _pixels_on_line(y0, x0, y1, x1):
                yield x, y
        else:
            for y, x in _pixels_on_line(-y0, x0, -y1, x1):
                yield x, -y


# test pixels_on_line  # todo: use asserts and move in tests
'''
# test pixels_on_line  # todo: use asserts and move in tests
from memory_evolution.utils._envs import pixels_on_line
import numpy as np
p = [
    (0,1,6,4), (1,0,5,0), (0,1,0,5), (0,1,1,6), (1,0,6,1),
    (0,0,6,6), (0,6,6,0), (0,6,6,1), (0,6,6,5), 
    (0,4,6,1), 
    (6,4,0,1), (5,0,1,0), (0,5,0,1), (1,6,0,1),
    (6,0,0,6), (6,1,0,6), (6,5,0,6),
]
for x0, y0, x1, y1 in p:
    z=np.zeros((7,7))
    pxls=list(pixels_on_line(x0, y0, x1, y1))
    print('line:', (x0, y0, x1, y1), 'pixels:', pxls)
    assert pxls, pxls  # pxls should contain at least one point
    z[tuple(zip(*pxls))] = 1
    print(z.swapaxes(0, 1))
# order is preserved
for x0, y0, x1, y1 in p:
    assert (x0, y0) == next(pixels_on_line(x0, y0, x1, y1))
for x0, y0, x1, y1 in p:
    pxs=list(pixels_on_line(x0, y0, x1, y1))
    assert (x0, y0) == pxs[0]
    assert (x1, y1) == pxs[-1]
####################
line: (0, 1, 6, 4) pixels: [(0, 1), (1, 1), (2, 2), (3, 2), (4, 3), (5, 3), (6, 4)]
[[0. 0. 0. 0. 0. 0. 0.]
 [1. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 1. 0. 0. 0.]
 [0. 0. 0. 0. 1. 1. 0.]
 [0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]]
line: (1, 0, 5, 0) pixels: [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
[[0. 1. 1. 1. 1. 1. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]]
line: (0, 1, 0, 5) pixels: [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
[[0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]]
line: (0, 1, 1, 6) pixels: [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (1, 6)]
[[0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]]
line: (1, 0, 6, 1) pixels: [(1, 0), (2, 0), (3, 0), (4, 1), (5, 1), (6, 1)]
[[0. 1. 1. 1. 0. 0. 0.]
 [0. 0. 0. 0. 1. 1. 1.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]]
line: (0, 0, 6, 6) pixels: [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
[[1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 0. 1.]]
line: (0, 6, 6, 0) pixels: [(0, 6), (1, 5), (2, 4), (3, 3), (4, 2), (5, 1), (6, 0)]
[[0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]]
line: (0, 6, 6, 1) pixels: [(0, 6), (1, 5), (2, 4), (3, 4), (4, 3), (5, 2), (6, 1)]
[[0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 1. 1. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]]
line: (0, 6, 6, 5) pixels: [(0, 6), (1, 6), (2, 6), (3, 6), (4, 5), (5, 5), (6, 5)]
[[0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 1. 1.]
 [1. 1. 1. 1. 0. 0. 0.]]
line: (0, 4, 6, 1) pixels: [(0, 4), (1, 4), (2, 3), (3, 3), (4, 2), (5, 2), (6, 1)]
[[0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 1. 1. 0.]
 [0. 0. 1. 1. 0. 0. 0.]
 [1. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]]
line: (6, 4, 0, 1) pixels: [(6, 4), (5, 4), (4, 3), (3, 3), (2, 2), (1, 2), (0, 1)]
[[0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 1. 0. 0.]
 [0. 0. 0. 0. 0. 1. 1.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]]
line: (5, 0, 1, 0) pixels: [(5, 0), (4, 0), (3, 0), (2, 0), (1, 0)]
[[0. 1. 1. 1. 1. 1. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]]
line: (0, 5, 0, 1) pixels: [(0, 5), (0, 4), (0, 3), (0, 2), (0, 1)]
[[0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]]
line: (1, 6, 0, 1) pixels: [(1, 6), (1, 5), (1, 4), (0, 3), (0, 2), (0, 1)]
[[0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]]
line: (6, 0, 0, 6) pixels: [(6, 0), (5, 1), (4, 2), (3, 3), (2, 4), (1, 5), (0, 6)]
[[0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]]
line: (6, 1, 0, 6) pixels: [(6, 1), (5, 2), (4, 3), (3, 3), (2, 4), (1, 5), (0, 6)]
[[0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 1. 1. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0.]]
line: (6, 5, 0, 6) pixels: [(6, 5), (5, 5), (4, 5), (3, 5), (2, 6), (1, 6), (0, 6)]
[[0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 1. 1. 1.]
 [1. 1. 1. 0. 0. 0. 0.]]
'''



def is_simple_polygon(polygon: Polygon):
    return (
        isinstance(polygon, Polygon)
        and polygon.is_valid
        and isinstance(polygon.boundary, LineString)  # not MultiLineString boundary
        and not list(polygon.interiors)  # empty interiors, no holes
    )


def is_triangle(polygon: Polygon):
    res = is_simple_polygon(polygon)
    if res:
        coords = polygon.boundary.coords
        res = len(coords) == 4
        assert coords[0] == coords[-1]
    return res


assert not is_triangle(Polygon((Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0))))
assert is_triangle(Polygon((Point(0, 0), Point(0, 1), Point(1, 1))))


# fixme: doesn't work with complex polygons: it misses some triangles inside the figure
def triangulate_nonconvex_polygon(polygon: Union[Polygon, MultiPolygon]):
    raw_triangles = triangulate(polygon)
    triangles = []
    for tr in raw_triangles:
        if polygon.contains(tr):
            # print('contains')
            triangles.append(tr)
        elif polygon.overlaps(tr):
            # print('overlaps', end=' ')
            inx = polygon.intersection(tr)
            # print(inx.wkt, end=' ')
            assert isinstance(inx, (GeometryCollection, MultiPolygon)), inx.wkt
            if isinstance(inx, GeometryCollection):
                # print('[' + ', '.join(g.wkt for g in inx) + ']', end=' ')
                # # overlaps [POINT (2 4), POLYGON ((0 0, 0 3, 1.5 3, 0 0))] 2
                # # overlaps [LINESTRING (5 3, 2 4), POLYGON ((1.5 3, 5 3, 0 0, 1.5 3))] 2
                inx = GeometryCollection([g for g in inx if isinstance(g, Polygon)])
                assert len(inx) == 1, inx.wkt
                inx = inx[0]
                assert 4 <= len(inx.boundary.coords) <= 5, inx.wkt
            elif isinstance(inx, MultiPolygon):
                for i in inx:
                    assert 4 <= len(i.boundary.coords) <= 5, inx.wkt
            trs = triangulate(inx)
            # print(len(trs))
            trs_cov = []
            for t in trs:
                if polygon.contains(t):
                    trs_cov.append(t)
                else:
                    # assert polygon.touches(t), (polygon.overlaps(t),
                    #                             polygon.covers(t),
                    #                             polygon.disjoint(tr),
                    #                             polygon.intersection(t).area / inx.area,
                    #                             inx.area,
                    #                             GeometryCollection((t, inx, tr, polygon)).wkt)
                    # # assert polygon_.covers(polygon_.intersection(tr))  # ERROR: this doesn't make sense...
                    # # assert tr.covers(polygon_.intersection(tr))  # ERROR: this doesn't make sense...
                    # # ERROR, since it is an intersection it should be covered; is it a floating point precision error??
                    # fixme
                    ass = polygon.touches(t)
                    if not ass:
                        logging.warning('Some triangles are weird')
                    assert ass or not polygon.covers(polygon.intersection(tr)), (
                        polygon.overlaps(t),
                        polygon.covers(t),
                        polygon.disjoint(tr),
                        polygon.intersection(t).area / inx.area,
                        inx.area,
                        GeometryCollection((t, inx, tr, polygon)).wkt)
            triangles.extend(trs_cov)
        else:
            # print('else(touches)')
            assert polygon.touches(tr), (polygon.overlaps(tr), polygon.covers(tr), polygon.disjoint(tr))
    return triangles


def get_random_point_in_triangle(triangle, random_generator) -> Point:

    epsilon = np.finfo(np.float32).resolution * (3 * 10)

    if not isinstance(random_generator, np.random.Generator):
        raise TypeError("'random_state' is not a np.random.Generator object"
                        f", the object provided is of type {type(random_generator)} instead.")
    if not isinstance(triangle, Polygon):
        raise TypeError("'triangle' is not a Polygon object")
    if not is_triangle(triangle):
        raise ValueError("'triangle' is not a triangle")
    if triangle.has_z:
        raise ValueError("only 2D is implemented")

    # translate triangle to origin, get a and b vectors or the space:
    coords = np.asarray(triangle.boundary.coords)
    assert 4 == len(coords), coords
    assert np.array_equal(coords[0], coords[-1]), coords
    orig = coords[0].copy()
    coords -= orig
    assert np.array_equal(coords[0], (0, 0)), coords
    a = coords[1]
    b = coords[2]

    # generate random point in the space parallelogram:
    # uniform [0, 1)  # [0,1] would be better, but anyway changed for floating point precision errors, so no problem
    u1, u2 = random_generator.random(2)  # * (1 - epsilon * 2) + epsilon

    # if point outside triangle (2nd half of parallelogram) map in the triangle (1st half of parallelogram):
    if u1 + u2 > 1:
        u1 = 1 - u1
        u2 = 1 - u2

    # linear combination of a and b:
    pnt = u1 * a + u2 * b

    # translate back to original position:
    pnt += orig

    pnt = Point(pnt)
    # Note: take in account floating point precision errors -> epsilon
    assert triangle.intersects(pnt), (pnt.wkt, triangle.boundary.coords[:])
    # print(pnt)
    return pnt


def get_random_non_overlapping_positions_with_triangulation(
        n,
        radius: Union[list, int],
        platform,
        env_size,
        random_generator,
) -> list[Pos]:
    def _is_valid_polygon_position(platform, pos: Union[Point, Polygon]):
        if not isinstance(pos, (Point, Polygon)):
            raise TypeError("'pos' should be an instance of Point or Polygon")
        if pos.has_z:
            raise ValueError("'pos' should be 2D (and without channels)")
        return platform.covers(pos)

    if not isinstance(random_generator, np.random.Generator):
        raise TypeError("'random_state' is not a np.random.Generator object"
                        f", the object provided is of type {type(random_generator)} instead.")
    if isinstance(radius, int):
        radius = [radius] * n
    if n != len(radius):
        raise ValueError(f"'radius' should be int or a list of 'n' integers, "
                         f"instead has {len(radius)} elements.")
    assert 2 == len(env_size), env_size

    # more efficient and always ending version:  # todo: test it with polygons with holes

    epsilon = max(np.finfo(np.float32).resolution * (3 * 10), .0001)
    init_platform = platform
    rng = random_generator

    poses = []
    # chosen = []  # polygons already positioned
    chosen = unary_union([])  # polygons already positioned
    for i, r in enumerate(radius):

        # select the platform and take only the available parts:
        platform = init_platform.difference(chosen)  # unary_union(chosen))

        # reduce the platform by the radius of the new object (it could be Polygon or MultiPolygon):
        platform = platform.buffer(-r)  # - epsilon * r)
        assert isinstance(platform, (Polygon, MultiPolygon)), type(platform)
        # print(platform.boundary.wkt)
        # print(f" platform_area_remained={platform.area}")
        if platform.is_empty:
            raise RuntimeError("There is not enough space to fit all the figures in the environment.")

        # divide the platform in triangles:
        triangles = triangulate_nonconvex_polygon(platform)
        # print('Triangles:\n\t' + '\n\t'.join(tr.wkt for tr in triangles))

        # choose randomly a triangle proportionally to its area:
        probs = np.asarray([tr.area for tr in triangles])
        probs /= probs.sum(None)
        tr = rng.choice(triangles, p=probs)

        # pick a random point in this triangle:
        pt = get_random_point_in_triangle(tr, rng)

        # create object, update poses and chosen:
        pt_buff = pt.buffer(r)  # + epsilon * r)
        poses.append(pt)
        chosen = chosen.union(pt_buff)  # chosen.append(pt_buff)

    assert n == len(poses) == len(radius)
    # # plotting during debug:
    # fig, ax = plt.subplots()
    # gpd.GeoSeries(init_platform.boundary).plot(ax=ax, color='k')
    # gpd.GeoSeries(platform.buffer(0)).plot(ax=ax, color='b')
    # gpd.GeoSeries([tr.boundary for tr in triangles]).plot(ax=ax, color='gray')
    # gpd.GeoSeries([Point(p) for p in poses]).plot(ax=ax, color='r')
    # plt.show()
    assert all(_is_valid_polygon_position(init_platform, pos.buffer(r)) for pos, r in zip(poses, radius)), (
        [(p.wkt, r) for p, r in zip(poses, radius) if not _is_valid_polygon_position(init_platform, p.buffer(r))])
    return [Pos(p.x, p.y) for p in poses]


def is_point_in_circle(point, radius, origin=(0, 0)) -> bool:
    """Returns True if ``point`` is inside or on the border of a circle
    of radius ``radius`` and origin ``origin``."""
    if len(point) != 2 or len(origin) != 2:
        raise NotImplementedError("Only 2D geometry is implemented.")
    x0, y0 = origin
    x, y = point
    return (x - x0) ** 2 + (y - y0) ** 2 <= radius ** 2


def get_random_non_overlapping_positions_with_lasvegas(
        n,
        radius: Union[list, int],
        platform,
        env_size,
        random_generator,
        # epsilon=0,  # max(np.finfo(np.float32).resolution * (3 * 10), .0001),
        _env=None,
        _env_items=None,
) -> list[Pos]:
    assert _env is None and _env_items is None or _env is not None and _env_items is not None, (_env, _env_items)
    assert _env_items is not None and len(_env_items) == n, (_env_items, n)

    def _is_valid_polygon_position(platform, pos: Union[Point, Polygon]):
        if not isinstance(pos, (Point, Polygon)):
            raise TypeError("'pos' should be an instance of Point or Polygon")
        if pos.has_z:
            raise ValueError("'pos' should be 2D (and without channels)")
        return platform.covers(pos)

    if not isinstance(random_generator, np.random.Generator):
        raise TypeError("'random_state' is not a np.random.Generator object"
                        f", the object provided is of type {type(random_generator)} instead.")
    if isinstance(radius, int):
        radius = [radius] * n
    if n != len(radius):
        raise ValueError(f"'radius' should be int or a list of 'n' integers, "
                         f"instead has {len(radius)} elements.")
    if n <= 0:
        raise ValueError(f"'n' should be greater or equal to 1 (n={n}).")
    assert 2 == len(env_size), env_size

    def get_reduced_platform(init_platform, chosen, r):
        # select the platform and take only the available parts:
        platform = init_platform.difference(chosen)  # unary_union(chosen))

        # reduce the platform by the radius of the new object (it could be Polygon or MultiPolygon):
        platform = platform.buffer(-r)  # - epsilon * r)
        assert isinstance(platform, (Polygon, MultiPolygon)), type(platform)
        # print(platform.boundary.wkt)
        # print(f" platform_area_remained={platform.area}")
        if platform.is_empty:
            raise RuntimeError("There is not enough space to fit all the figures in the environment.")

        return platform

    init_platform = platform
    poses = []
    chosen = unary_union([])  # polygons already positioned
    i = 0
    update_platform = True
    while i < len(radius):

        # compute available parts of platform:
        if update_platform:
            r = radius[i]
            platform = get_reduced_platform(init_platform, chosen, r)

        # pick a random point inside env_size:
        # note: rng.random() choose in the range [0,1), thus will never pick 1, but this is not a problem
        x = random_generator.random() * env_size[0]
        y = random_generator.random() * env_size[1]
        pt = Point(x, y)

        # check if point is valid: if valid, update poses and chosen:
        if (platform.covers(pt)
                and (_env is None or _env.is_valid_position((pt.x, pt.y), _env_items[i], is_env_pos=True))):
            # the check with env is important to compensate for pixel approximations.

            # update poses and chosen:
            pt_buff = pt.buffer(r)  # + epsilon * r)
            poses.append(pt)
            chosen = chosen.union(pt_buff)

            # update counter (and update_platform flag):
            update_platform = True
            i += 1

    assert n == len(poses) == len(radius)
    assert all(_is_valid_polygon_position(init_platform, pos.buffer(r)) for pos, r in zip(poses, radius)), (
        [(p.wkt, r) for p, r in zip(poses, radius) if not _is_valid_polygon_position(init_platform, p.buffer(r))])
    return [Pos(p.x, p.y) for p in poses]


"""
# wrong:
def get_random_non_overlapping_positions_with_lasvegas(
        env: memory_evolution.envs.BaseForagingEnv,
        n,
        items_type: Union[list, str],
        items_radius: dict,
) -> list[Pos]:
    env_size = env.env_size

    if isinstance(items_type, str):
        items_type = [items_type] * n
    if n != len(items_type):
        raise ValueError(f"'radius' should be int or a list of 'n' integers, "
                         f"instead has {len(items_type)} elements.")
    assert 2 == len(env_size), env_size

    rng = env.env_space.np_random

    poses = []  # to preserve the order
    # chosen = set()  # to check fastly if new pos was already taken
    i = 0
    while i < len(items_type):
        ity = items_type[i]

        # pick a random point inside env_size:
        # Note: rng.random() choose in the range [0,1), thus will never pick 1,
        # but this is problem that can be ignored safely.
        x = rng.random() * env_size[0]
        y = rng.random() * env_size[1]
        pt = Pos(x, y)

        # check if point is valid and new: if valid and new, update poses and chosen:
        if env.is_valid_position(pt, ity, is_env_pos=True):
            # todo: improve this
            for j, p in enumerate(poses):
                if is_point_in_circle(pt, items_radius[])
            else:
                i += 1
                poses.append(pt)
                # chosen.add(pt)

    assert n == len(poses) == len(items_type)
    return poses
"""

