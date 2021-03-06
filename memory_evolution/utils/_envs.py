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
import pygame as pg


# note: evaluate difference between matplotlib.colors and pg.Color('name of the color')

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


def get_color_str(col: np.ndarray) -> str:
    if not is_color(col):
        raise ValueError(f"'col' is not a color")
    return '#' + ''.join(f"{c:02x}" for c in col)


def black_n_white(img: np.ndarray):
    return (img.sum(-1) / img.shape[-1]).round().astype(img.dtype)[..., None]
    # np.sum: dtype: if a is unsigned then an unsigned integer
    #         of the same precision as the platform integer is used.


_NORMALIZE__MAX_PIXEL_VALUE = np.asarray(255, dtype=np.float32)  # float32 so the output of division is a float32


def normalize_observation(observation: np.ndarray) -> np.ndarray:
    """Normalize and ravel."""
    return observation.reshape(-1, order='C') / _NORMALIZE__MAX_PIXEL_VALUE


def denormalize_observation(observation: np.ndarray, obs_shape) -> np.ndarray:
    return observation.reshape(obs_shape, order='C') * _NORMALIZE__MAX_PIXEL_VALUE


def invert_colors_inplace(surface: pg.Surface):
    pixels = pg.surfarray.pixels2d(surface)  # use a reference to pixels
    pixels ^= 2 ** 32 - 1


IMAGE_FORMAT = Literal["P", "RGB", "BGR", "RGBX", "RGBA", "ARGB"]


def convert_image_to_pygame(image, format_: IMAGE_FORMAT = "RGB"):
    if isinstance(image, np.ndarray) and image.dtype == np.uint8:
        return pg.image.frombuffer(image.tobytes(), image.shape[1::-1], format_)
    else:
        raise NotImplementedError(f"{type(image)!r}"
                                  + (f", dtype={image.dtype!r} ('np.uint8' is supported instead)"
                                     if isinstance(image, np.ndarray)
                                     else ''))


def convert_pg_surface_to_array(surface):
    """note: returns a np.ndarray, channels are 3, it converts just colors (not alpha)"""
    if isinstance(surface, pg.Surface):
        # you could use pygame.image.tostring(Surface, format, flipped=False) -> string
        arr = pg.surfarray.array3d(surface).swapaxes(0, 1)
        # use pygame.surfarray.pixels3d if you don't want a copy but you want a reference
        # note: if you reference the surface, it will remain locked for the lifetime of the array,
        #       since the array generated by this function shares memory with the surface. See
        #       the pygame.Surface.lock() lock the Surface memory for pixel access - lock the
        #       Surface memory for pixel access method.
        assert arr.dtype == np.uint8, arr.dtype
        assert arr.ndim == 3, arr.ndim  # rows, columns, channels
        assert arr.shape[2] == 3, arr.shape  # 3 channels
        assert arr.shape[1::-1] == surface.get_size(), arr.shape  # width, height -> columns, rows -> rows, columns
        return arr
    else:
        raise ValueError(f"{type(surface)!r} isn't a pygame.Surface")


def convert_pg_mask_to_array(mask):
    """note: returns a np.ndarray, channels are 1, it converts just colors (not alpha)"""
    if isinstance(mask, pg.mask.Mask):
        arr = pg.surfarray.array_alpha(mask.to_surface(unsetcolor=(0, 0, 0, 0))).swapaxes(0, 1)
        # use pygame.surfarray.pixel_alpha if you don't want a copy but you want a reference
        # note: if you reference the surface, it will remain locked, see above
        assert arr.dtype == np.uint8, arr.dtype  # here arr values should be only 0 or 255
        arr = arr.astype(bool)
        assert arr.dtype == np.bool_, arr.dtype
        assert arr.ndim == 2, arr.ndim  # rows, columns
        assert arr.shape[1::-1] == mask.get_size(), arr.shape  # width, height -> columns, rows -> rows, columns
        return arr
    else:
        raise ValueError(f"{type(mask)!r} isn't a pygame.mask.Mask")


class PickableClock:
    """Pickable wrapper for pg.time.Clock

    Note: this object is pickable only if no attribute have been accessed yet.
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._clock = None

    def __getattr__(self, name):
        """Create an instance of pg.time.Clock only after the first
        attribute access, return pg.time.Clock attribute values.
        """
        import traceback
        if name.startswith('_'):
            if name not in self.__dict__:
                raise AttributeError(name, self)
            return getattr(self, name)
        else:
            if self._clock is None:
                clk = pg.time.Clock(*self._args, **self._kwargs)
                assert not hasattr(clk, '_args')
                assert not hasattr(clk, '_kwargs')
                assert not hasattr(clk, '_clock')
                self._clock = clk
            return getattr(self._clock, name)

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            clk = self._clock
            setattr(clk, name, value)

    def __dir__(self):
        if self._clock is None:
            return dir(pg.time.Clock)
        else:
            return dir(self._clock)


