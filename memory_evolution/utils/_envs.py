from collections import defaultdict, Counter
import math
from numbers import Number, Real
from typing import Optional, Union, Any
from warnings import warn

import gym
from gym import spaces
import matplotlib as mpl
import matplotlib.colors as mcolors  # https://matplotlib.org/stable/gallery/color/named_colors.html
import matplotlib.pyplot as plt
import numpy as np


# names of colors here: https://matplotlib.org/3.5.1/gallery/color/named_colors.html
COLORS = {key: (np.asarray(col) * 255).astype(np.uint8)
          for colors in (
              mcolors.BASE_COLORS,
              {k: mcolors.hex2color(col) for k, col in mcolors.TABLEAU_COLORS.items()},
              {k: mcolors.hex2color(col) for k, col in mcolors.CSS4_COLORS.items()},
          )
          for key, col in colors.items()}
assert all((isinstance(col, np.ndarray)
           and col.dtype == np.uint8
           and col.ndim == 1
           and col.shape[0] == 3)
           and all((0 <= c <= 255) for c in col)
           for col in COLORS.values()), COLORS
assert any(any((c == 255) for c in col)
           for col in COLORS.values()), COLORS
# print(COLORS)


class Pos:
    """Coordinates of a Position.
    `coords` coordinates should be all float
    """

    def __init__(self, *coords):
        # if ndim <= 0:
        #     raise ValueError('`ndim` must be positive.')
        # if ndim != len(coords):
        #     raise TypeError(f'`ndim`={ndim} coords are expected, {len(coords)} coords was given (as arguments)')
        assert isinstance(coords, tuple)
        if not all(isinstance(c, float) for c in coords):
            raise TypeError('`coords` coordinates should be all float')
        self._coords = tuple(coords)

    @property
    def ndim(self):
        return len(self._coords)

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
        if self.ndim > 3 or self.ndim < 1:
            raise AttributeError('When `ndim` is higher than 3 or less than 1, `x` attribute is not supported')
        return self._coords[0]

    @property
    def y(self):
        if self.ndim > 3 or self.ndim < 2:
            raise AttributeError('When `ndim` is higher than 3 or less than 2, `y` attribute is not supported')
        return self._coords[1]

    @property
    def z(self):
        if self.ndim > 3 or self.ndim < 3:
            raise AttributeError('When `ndim` is higher than 3 or less than 3, `z` attribute is not supported')
        return self._coords[2]

    def __repr__(self):
        return (f"{type(self).__name__}("
                + ", ".join([str(c) for c in self._coords]) + ")")
        # return (f"{__name__ if __name__ != '__main__' else ''}.{type(self).__qualname__}("
        #         + ", ".join([str(c) for c in self._coords]) + ")")


