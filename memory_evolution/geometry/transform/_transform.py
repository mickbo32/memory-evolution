from collections import defaultdict, Counter
from collections.abc import Iterable, Sequence
import logging
import math
from math import cos, sin
from math import degrees, pi as PI, radians
from numbers import Number, Real
from typing import Any, Literal, Optional, Union
from warnings import warn
import sys
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def translate(point, xoff=0., yoff=0.):
    """translate a point or a list of points,
     shifting it by the corresponding offset for each axis"""
    point = np.asarray(point)
    if not (point.ndim == 1 and (len(point) == 2 or len(point) == 0)
            or point.ndim == 2 and point.shape[1] == 2):
        raise ValueError(point)
    assert point.dtype != object, point
    return point + np.asarray((xoff, yoff))
    ''' # version 1
    """translate a point or an iterable of points,
     shifting it by the corresponding offset for each axis"""
    if isinstance(point, Iterable) and isinstance(next(iter(point)), Iterable):
        return [
            translate(pt, xoff, yoff)
            for pt in point
        ]
    else:
        assert len(point) == 2, point
        x, y = point
        return x + xoff, y + yoff
    '''


def get_rotation_matrix_2d(angle, use_radians, xoff=0., yoff=0.):
    if not use_radians:
        angle = radians(angle)
    # return np.asarray((
    #     (cos(angle), -sin(angle)),
    #     (sin(angle),  cos(angle)),
    # ))
    return np.asarray((
        (cos(angle), -sin(angle), xoff),
        (sin(angle),  cos(angle), yoff),
        (         0,           0,    1),
    ))


def rotate(point, angle, origin, use_radians=False):
    """rotate a point or a list of points"""
    if not use_radians:
        use_radians = True
        angle = radians(angle)
    point = np.asarray(point)
    assert point.ndim == 1 and (len(point) == 2 or len(point) == 0) or point.ndim == 2 and point.shape[1] == 2, point
    assert point.dtype != object, point
    point = np.hstack((point, np.ones((point.shape[0], 1) if point.ndim == 2 else (1,))))
    xoff = origin[0] - origin[0] * cos(angle) + origin[1] * sin(angle)
    yoff = origin[1] - origin[0] * sin(angle) - origin[1] * cos(angle)
    point = get_rotation_matrix_2d(angle, use_radians, xoff, yoff) @ point.T
    return (point[:-1, :].T if point.ndim == 2 else point[:-1].T).copy()
    ''' # verison 2
    if not use_radians:
        use_radians = True
        angle = radians(angle)
    if isinstance(point, Iterable) and isinstance(next(iter(point)), Iterable):
        return [
            rotate(pt, angle, origin, use_radians)
            for pt in point
        ]
    else:
        assert len(point) == 2, point
        xoff = origin[0] - origin[0] * cos(angle) + origin[1] * sin(angle)
        yoff = origin[1] - origin[0] * sin(angle) - origin[1] * cos(angle)
        point = get_rotation_matrix_2d(angle, use_radians, xoff, yoff) @ np.asarray((*point, 1))[:, None]
        return tuple(point.reshape(-1)[:-1])
    '''
    '''  # version 1
    if isinstance(point, Iterable) and isinstance(next(iter(point)), Iterable):
        return [
            rotate(pt, angle, origin, use_radians)
            for pt in point
        ]
    else:
        assert len(point) == 2, point
        point = np.asarray(translate(point, -origin[0], -origin[1]))
        point = get_rotation_matrix_2d(angle, use_radians) @ point[:, None]
        point = translate(point.reshape(-1), origin[0], origin[1])
        return tuple(point)
    '''


def scale(point, xfact=1., yfact=1., origin=np.asarray((0, 0))):
    """scale a point or a list of points,
     by the corresponding offset for each axis"""
    point = np.asarray(point)
    if not (point.ndim == 1 and (len(point) == 2 or len(point) == 0)
            or point.ndim == 2 and point.shape[1] == 2):
        raise ValueError(point)
    assert point.dtype != object, point
    origin = np.asarray(origin)
    return (point - origin) * np.asarray((xfact, yfact)) + origin

