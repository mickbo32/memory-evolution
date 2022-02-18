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
import matplotlib.colors as mcolors  # https://matplotlib.org/stable/gallery/color/named_colors.html
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import SeedSequence, default_rng
import pygame
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPolygon, GeometryCollection
from shapely.ops import unary_union, triangulate


# names of colors here: https://matplotlib.org/3.5.1/gallery/color/named_colors.html
COLORS = {key: (np.asarray(col) * 255).astype(np.uint8)
          for colors in (
              mcolors.BASE_COLORS,
              {k: mcolors.hex2color(col) for k, col in mcolors.TABLEAU_COLORS.items()},
              {k: mcolors.hex2color(col) for k, col in mcolors.CSS4_COLORS.items()},
          )
          for key, col in colors.items()}
def is_color(col):
    return (isinstance(col, np.ndarray)
            and col.dtype == np.uint8
            and col.ndim == 1
            and col.shape[0] == 3
            and all((0 <= c <= 255) for c in col))
assert all(is_color(col)
           for col in COLORS.values()), COLORS
assert any(any((c == 255) for c in col)
           for col in COLORS.values()), COLORS
# print(COLORS)


def black_n_white(img: np.ndarray):
    return (img.sum(-1) / img.shape[-1]).round().astype(img.dtype)[..., None]
    # np.sum: dtype: if a is unsigned then an unsigned integer
    #         of the same precision as the platform integer is used.


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


IMAGE_FORMAT = Literal["P", "RGB", "BGR", "RGBX", "RGBA", "ARGB"]


def convert_image_to_pygame(image, format_: IMAGE_FORMAT = "RGB"):
    if isinstance(image, np.ndarray) and image.dtype == np.uint8:
        return pygame.image.frombuffer(image.tobytes(), image.shape[1::-1], format_)
    else:
        raise NotImplementedError(f"{type(image)!r}"
                                  + (f", dtype={image.dtype!r}"
                                     if isinstance(image, np.ndarray)
                                     else ''))


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
                        warn('Some triangles are weird')
                        print('Some triangles are weird')
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

