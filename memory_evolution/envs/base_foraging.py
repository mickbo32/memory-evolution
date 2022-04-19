from copy import deepcopy
import os
from collections import defaultdict, Counter
from collections.abc import Iterable, Sequence
from functools import reduce
import inspect
import logging
import math
from numbers import Number, Real
from operator import mul
import re
from typing import Literal, Optional, Union, Any
from warnings import warn
import sys

import gym
from gym import spaces
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import SeedSequence, default_rng
import pandas as pd
import pygame as pg
# from shapely.affinity import rotate, scale, translate
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint, MultiPolygon
from shapely.ops import unary_union, triangulate

import memory_evolution
from memory_evolution.geometry import *  # while developing I keep '*', then it will be imported only the stuff used
from memory_evolution.utils import *  # while developing I keep '*', then it will be imported only the stuff used
from memory_evolution.utils import MustOverride, override


def check_scaling_factor_across_axes(window_size, env_size) -> Real:
    """It raises an error if scaling factor is wrong across axes."""
    msg_2d_error = 'Only 2D points implemented'
    if len(window_size) != len(env_size):
        raise "'window_size' and 'env_size' have different dimensions."
    if len(window_size) != 2:
        raise NotImplementedError(msg_2d_error)
    base_sf = window_size[0] / env_size[0]
    for ax in range(1, len(window_size)):
        sf = window_size[ax] / env_size[ax]
        if window_size[ax] != math.ceil(base_sf * env_size[ax]):
            # height should be ceiled
            raise ValueError(
                f"'window_size' and 'env_size' should have the same"
                f" scaling factor for width and height"
                f" (taking in account that height is ceiled);\n"
                f"\t\tinstead they have different scaling factors:"
                f" {base_sf} and {sf} (former w.r.t. axis=0, latter w.r.t. axis={ax});\n"
                f"\t\t'window_size': {window_size}\n"
                f"\t\t'env_size': {env_size};\n"
            )


def get_env2win_scaling_factor(window_size, env_size, axis=0) -> Real:
    """Scaling factor to obtain a window distance starting from an env distance.

    Note: with axis=1 there could be small approximation (depending on the environment used),
        use it only to compute pixel's stuff for visualization,
        in any other case better using axis=0 which should the True main scaling factor.
    
    Examples:
        Here an examples of how to obtain a window distance starting from a
        env distance and vice versa.

        >>> window_distance = env_distance * get_env2win_scaling_factor(window_size, env_size)

        >>> env_distance = window_distance / get_env2win_scaling_factor(window_size, env_size)

    """
    # check of correctness of scaling factors along axes are not done here but in env.__init__ for efficiency purposes.
    return window_size[axis] / env_size[axis]


def get_point_env2win(point, window_size, env_size
                      ) -> tuple[int, ...]:
    """Take a point in the environment coordinate system
    and transform it in the window coordinate system (i.e. a pixel).
    """
    msg_2d_error = 'Only 2D points implemented'
    msg_point_outside_env_error = (
        'point not in the environment (outside env_size);'
        ' point: {point}; env_size: {env_size}')
    if len(point) != 2:
        raise NotImplementedError(msg_2d_error)
    if not (0 <= point[0] <= env_size[0] and 0 <= point[1] <= env_size[1]):
        raise ValueError(msg_point_outside_env_error.format(point=point, env_size=env_size))
    point = (point[0] * get_env2win_scaling_factor(window_size, env_size, axis=0),
             (env_size[1] - point[1]) * get_env2win_scaling_factor(window_size, env_size, axis=1))
    # map the borders of env in the correct window pixel:
    if point[0] == window_size[0]:
        point = (point[0] - 1, point[1])
    if point[1] == window_size[1]:
        point = (point[0], point[1] - 1)
    return int(point[0]), int(point[1])  # tuple(int(x) for x in point)


def get_point_win2env(point, window_size, env_size
                      ) -> Pos:
    """Take a point in the window coordinate system (i.e. a pixel)
    and transform it in the environment coordinate system.
    """
    msg_2d_error = 'Only 2D points implemented'
    msg_point_outside_env_error = 'point outside environment window'
    if len(point) != 2:
        raise NotImplementedError(msg_2d_error)
    for x in point:
        if not isinstance(x, int):
            TypeError(f"window coordinates should be integers, "
                      f"instead point has a coordinate {type(x)}")
    if not (0 <= point[0] < window_size[0] and 0 <= point[1] < window_size[1]):
        raise ValueError(msg_point_outside_env_error)
    # get the centroid of the pixel:
    point = [x + .5 for x in point]
    assert 0 <= point[0] < window_size[0] and 0 <= point[1] < window_size[1], (point, window_size)
    point = [point[0] / get_env2win_scaling_factor(window_size, env_size, axis=0),
             (window_size[1] - point[1]) / get_env2win_scaling_factor(window_size, env_size, axis=1)]
    point = Pos(*point)
    return point


def get_valid_item_positions_mask(platform: Polygon, item_radius: Real,
                                  window_size: tuple, env_size: tuple
                                  ) -> pg.mask.Mask:
    """
    ``platform`` and ``item_radius`` is in the environment space.

    Note:
    Expensive method, use this only for init purposes;
    e.g.:
    in __init__():
    self._valid_agent_positions = get_valid_item_positions_mask(
        platform, self._agent.radius, self._window_size, self._env_size);
    """
    if not isinstance(platform, Polygon):
        raise TypeError(type(platform))
    if not is_simple_polygon(platform):
        raise ValueError("'platform' is not a simple polygon")
    valid_positions = pg.mask.Mask(window_size)
    assert valid_positions.get_at((0, 0)) == 0
    platform = platform.buffer(-item_radius)
    assert isinstance(platform, (Polygon, MultiPolygon)), type(platform)
    for j in range(window_size[0]):  # win_x
        for i in range(window_size[1]):  # win_y
            x, y = get_point_win2env((j, i), window_size, env_size)
            valid_positions.set_at((j, i), not platform.disjoint(Point(x, y)))
    return valid_positions


class Texture:
    """Positive static pattern applied to the soil and maze.
    - static uniform noise
    - lines
    - curves
    - Gaussian random field
    Applied positively to the floor and negatively to the borders or high-contrast objects.
    """
    pass


class CircleItem(pg.sprite.Sprite):
    """A circular object in the window and in the environment.

    When subclassed, after calling the __init__() method, append item's
    minimal characterizing properties to self._self_repr_properties and
    simpler characterizing properties to self._self_str_properties to have a
    correct __str__() and __repr__() of the object.
    """

    def __init__(self, pos, size: Real, color,
                 env):
        """

        Args:
            pos: a point representing the position of the item.
            size: the item diameter.
            env: environment which own the object.
            color: color of the item.
        """
        # Call the parent class (Sprite) constructor
        super().__init__()

        self._env = env
        self._size = size
        self._radius = size / 2
        self.color = color

        # Create an image of the agent, and fill it with a color.
        # This could also be an image loaded from the disk.
        self.image = pg.Surface((self.size_on_screen, self.size_on_screen), pg.SRCALPHA)
        pg.draw.circle(self.image, self.color, (self.radius_on_screen, self.radius_on_screen), self.radius_on_screen)

        # Fetch the rectangle object that has the dimensions of the image
        # Update the position of this object by setting the values of rect.x and rect.y
        self.rect = self.image.get_rect()  # center=self.win_pos)

        # Create mask for correct collision detection
        self.mask = pg.mask.from_surface(self.image)
        # print(size, (self.size_on_screen, self.size_on_screen),
        #       (self.radius_on_screen, self.radius_on_screen), self.radius_on_screen)
        # print(convert_pg_mask_to_array(self.mask))
        assert self.mask.get_at((0, 0)) == 0
        assert self.mask.get_at((self.radius_on_screen, self.radius_on_screen)) == 1

        self.pos = pos  # this updates also self.win_pos and self.rect.center
        self._self_str_properties = [self.pos, self._size]
        self._self_repr_properties = [self.pos, self._size, self.color, self._env]

    # Since here items don't move, there is no need for overriding the update method.
    # def update(self, *args: Any, **kwargs: Any) -> None:
    #     """Override the base method."""
    #     '''
    #     update()
    #     method to control sprite behavior
    #     update(*args, **kwargs) -> None
    #     The default implementation of this method does nothing; it's just a convenient "hook" that you can override. This method is called by Group.update() with whatever arguments you give it.
    #
    #     There is no need to use this method if not using the convenience method by the same name in the Group class.
    #     '''
    #     pass

    @property
    def size(self):
        return self._size

    @property
    def radius(self):
        return self._radius

    @property
    def size_on_screen(self):
        env2win_scaling_factor = get_env2win_scaling_factor(
            window_size=self._env.window_size, env_size=self._env.env_size)
        return math.ceil(self._size * env2win_scaling_factor)

    @property
    def radius_on_screen(self):
        return self.size_on_screen // 2

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value
        self._win_pos = get_point_env2win(self._pos,
                                          window_size=self._env.window_size,
                                          env_size=self._env.env_size)
        self.rect.center = self._win_pos

    @property
    def win_pos(self):
        return self._win_pos

    def __str__(self):
        return f"{type(self).__name__}({', '.join(map(str,self._self_str_properties))})"

    def __repr__(self):
        return (f"{__name__ if __name__ != '__main__' else ''}"
                f".{type(self).__qualname__}({', '.join(map(repr,self._self_repr_properties))})")


class Agent(CircleItem):
    """Agent"""

    def __init__(self, pos, size: Real, head_direction: Union[int, float],
                 color,
                 env):
        """

        Args:
            pos: a point representing the position of the agent.
            size: the agent diameter.
            head_direction: head direction in degrees.
            env: environment which own the agent.
            color: color of the agent.
        """
        super().__init__(pos, size, color=color, env=env)
        self._self_str_properties.append(head_direction)
        self._self_repr_properties.append(head_direction)

        self.head_direction = head_direction
        if head_direction and len(self.pos) != 2:
            raise NotADirectoryError("`head_direction` not implemented yet for spaces different from 2D space")

    @property
    def head_direction(self):
        return self._head_direction

    @head_direction.setter
    def head_direction(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("`head_direction` must be int or float")
        if not (0 <= value < 360):
            raise ValueError("`head_direction` must be in the [0,360) range.")
        self._head_direction = value


class FoodItem(CircleItem):
    """Rewarding (maybe, not actual reward, but increase in agent life span) food items."""

    def __init__(self, pos, size: Real, color, env):
        """

        Args:
            pos: a point representing the position of the food item.
            size: the food item diameter.
        """
        super().__init__(pos, size, color=color, env=env)
        # self._self_str_properties.append(NOTHING)
        # self._self_repr_properties.append(NOTHING)


def _constructor(cls, args, kwargs):
    print(cls, args, kwargs)
    return cls(*args, **kwargs)


# todo: rename simply ForagingEnv
class BaseForagingEnv(gym.Env, MustOverride):
    """Custom Environment that follows gym interface,
    it develops an agent moving in an environment in search for foods items.

    Food item is collected when the agent step upon (or touches) the center of the food item.

    If this class is subclassed, in the __init__() method of the subclass it
    should be created a correct ``self._background_img`` for the environment
    and it should be called ``self._compute_and_set_valid_positions(platform)`` to compute
    the correct valid positions for the new class.
    """
    metadata = {'render.modes': ['human', 'human+save[save_dir]', 'save[save_dir]']}

    def __new__(cls, *args, **kwargs):
        """Constructor used to save init values used during construction and initialization of this object."""
        obj = super().__new__(cls)
        try:
            ba = inspect.signature(obj.__init__).bind(*args, **kwargs)
            ba.apply_defaults()
            obj._init_params = ba
        except TypeError as err:
            logging.warning(str(err) + ", an error should be raised when running the __init__() method")
            obj._init_params = None
        else:  # add subclass default parameters as kwargs if not already present in kwargs
            # signature_params = defaultdict(list)
            # for _param in obj._init_params.signature.parameters.values():
            #     signature_params[_param.kind].append(_param.name)
            # assert len(signature_params) <= 5, signature_params
            positional_only_names = [
                p.name
                for p in obj._init_params.signature.parameters.values()
                if p.kind == p.POSITIONAL_ONLY]
            positional_or_keyword_names = [
                p.name
                for p in obj._init_params.signature.parameters.values()
                if p.kind == p.POSITIONAL_OR_KEYWORD]
            var_positional = [
                p.name
                for p in obj._init_params.signature.parameters.values()
                if p.kind == p.VAR_POSITIONAL]
            assert len(var_positional) <= 1
            var_keyword = [
                p.name
                for p in obj._init_params.signature.parameters.values()
                if p.kind == p.VAR_KEYWORD]
            assert len(var_keyword) <= 1
            assert len(positional_only_names) + len(positional_or_keyword_names) == len(obj._init_params.args) - (
                len(obj._init_params.arguments[var_positional[0]]) if var_positional else 0), (
                positional_only_names, positional_or_keyword_names, obj._init_params.args, obj._init_params.arguments)
            if var_keyword:  # todo: actually I should enforce that subclasses of env must have a **kwargs parameter and use it when calling the init of superclass
                mro_it = iter(cls.__mro__)
                _cls = next(mro_it)  # discard itself class
                while _cls != BaseForagingEnv:
                    try:
                        _cls = next(mro_it)
                    except StopIteration:
                        raise AssertionError(f"{BaseForagingEnv.__qualname__} not found in {cls.__qualname__}.__mro__")
                    params = inspect.signature(_cls.__init__).parameters.values()
                    default_params = {p.name: p.default for p in params if p.default is not p.empty}
                    # add kwargs if not already present
                    for key, val in default_params.items():
                        if (key not in obj._init_params.arguments[var_keyword[0]]
                                and key not in positional_or_keyword_names):
                            obj._init_params.arguments[var_keyword[0]][key] = val
            assert len(positional_only_names) + len(positional_or_keyword_names) == len(obj._init_params.args) - (
                len(obj._init_params.arguments[var_positional[0]]) if var_positional else 0)
        logging.log(logging.DEBUG + 7,
                    obj.__str__init_params__())
        return obj

    def _update_init_params(self, kwargs_keys):
        """Call this method in the subclass when calling super().__init__(names..., *args, **kwargs)
        with more values (names...),
        (calling it after or before doesn't matter)
        otherwise you will have both the default value
        and the computed value during initialization and your pickable object will
        raise an error when loaded.

        What is actually doing is removing, if present, the kwargs_keys from kwargs of self._init_params,
        this because they are computed by the __init__ you have just called and so you should not store
        the default value for the same variable because it is not used, but it will be passed by the __init__.
        """
        # fixme: if in a long chain of __init__ call I use a kw and then pass another
        #  value with the same name in a later call,
        #  this will delete both and give error when loading the pickle object.
        var_keyword = [
            p.name
            for p in self._init_params.signature.parameters.values()
            if p.kind == p.VAR_KEYWORD]
        assert len(var_keyword) <= 1
        if var_keyword:
            for key in kwargs_keys:
                if key in self._init_params.arguments[var_keyword[0]]:
                    del self._init_params.arguments[var_keyword[0]][key]

    def __str__init_params__(self):
        return (f"env: {type(self).__module__}.{type(self).__qualname__}("
                + ', '.join(f"{k}={v}" for k, v in self._init_params.arguments.items())
                + ")")

    def __reduce__(self):  # for pickle
        # todo: use only kwargs? (it is safer and more consistent across time and versions,
        #  but you need to remove *args in all possible subclass: you can do it for the class here,
        #  but the user can always do in another way, thus: leave args here, take it out form subclasses you define)
        args = self._init_params.args
        kwargs = self._init_params.kwargs
        return _constructor, (type(self), args, kwargs)

    def __init__(self,
                 window_size: Union[int, Sequence[int]] = 320,  # (640, 480),
                 env_size: Union[float, Sequence[float]] = 1.,
                 n_food_items: int = 3,
                 rotation_step: float = 15.,
                 forward_step: float = .01,
                 agent_size: float = .10,
                 food_size: float = .05,
                 vision_depth: float = .2,
                 vision_field_angle: float = 180.,
                 vision_resolution: int = 10,
                 observation_noise: Sequence[str, Real, Real] = None,
                 init_agent_position: Optional[Pos] = None,
                 init_food_positions: Optional[list] = None,
                 max_steps: Optional[int] = None,
                 fps: Optional[int] = None,
                 seed=None,  # todo: int or SeedSequence
                 ) -> None:
        """Inits environment

        The environment is done when all food items are collected or ``max_steps`` is reached.

        Args:
            window_size: if it is a Sequence of ints: (width, height);
                         if it is an int it is window_width, window_height is adjusted accordingly.
            env_size: if it is a Sequence of floats: (width, height);
                      if it is a float: (env_size, env_size).
            n_food_items: number of food items.
            rotation_step: amount of maximum rotation per tick.
            forward_step: amount of maximum forward motion per tick.
            agent_size: agent diameter.
            food_size: food item diameter.
            vision_depth: depth of the sight of the agent, each straight line of vision
                    will be extended for a ``vision_depth`` length (from agent center).
            vision_field_angle: angle of the full field of view of the agent.
            vision_resolution: how many sample to take for each straight line of vision,
                    i.e. the number of observation points will be
                    ``vision_resolution * self.vision_field_n_angles``,
                    the observation shape can be accessed by ``self.observation_space.shape``.
            init_food_positions: if provided should be a list of ``n_food_items`` valid
                    food item positions (in the environment space). If ``None``,
                    ``n_food_items`` food items are generated in random positions.
            observation_noise: If None, observation is returned as it is. Otherwise, add noise
                    to the observation, in this case ``observation_noise``
                    describes which method to use for noise generation, it should be a Sequence
                    which has as first element a string that refer to which method to use for
                    random number generation, the following elements are the arguments
                    to that random number generating method chosen; the noise is then added
                    to the observation, and clipped to the observation space low and high values,
                    just before being returned by the step() and reset() methods.
                    Valid random number generation methods are: {'normal', 'uniform'}.
                    Examples:
                        observation_noise=('normal', 0.0, 1.0)  # normal distribution mean=0.0, std=1.0
                        observation_noise=('uniform', 0.0, 1.0)  # uniform distribution [low=0.0, high=1.0)
            max_steps: after this number of steps the environment is done, even if no all
                    food items are collected(the ``max_steps``_th step it will return
                    done True); if ``None`` continue forever (done always False).
            fps: frames per second, if it is None it does the rendering as fast as possible.
            seed: seed for random generators.
        """
        super().__init__()

        if not (isinstance(env_size, float)
                or (isinstance(env_size, Sequence)
                    and len(env_size) == 2
                    and all(isinstance(x, float) for x in env_size))
                ):
            raise TypeError(env_size)
        if not (isinstance(window_size, int)
                or (isinstance(window_size, Sequence)
                    and len(window_size) == 2
                    and all(isinstance(x, int) for x in window_size))
                ):
            raise TypeError(window_size)

        self._n_channels = 3
        self._env_size = self._get_env_size(env_size)
        self._window_size = (tuple(window_size)
                             if isinstance(window_size, Sequence)
                             else (window_size, math.ceil(window_size * self._env_size[1] / self._env_size[0])))
        assert 2 == len(self._env_size) == len(self._window_size)
        check_scaling_factor_across_axes(self._window_size, self._env_size)
        self._env2win_resize_factor = get_env2win_scaling_factor(self._window_size, self._env_size)
        self._env_img_shape = (self._window_size[1], self._window_size[0], self._n_channels)  # array shape
        self._env_img_size = self._window_size  # pygame surface size
        self._n_food_items = n_food_items
        self._rotation_step = rotation_step
        if forward_step >= agent_size / 4:
            raise ValueError("'forward_step' should be way smaller of agent radius "  # otherwise it could jump borders
                             "(i.e. agent radius 'agent_size/2' -> 'forward_step < agent_size/4')")
        self._forward_step = forward_step
        self._agent_size = agent_size
        self._food_size = food_size
        self._vision_depth = vision_depth
        self._vision_field_angle = vision_field_angle
        self._vision_resolution = vision_resolution
        self._vision_start = self._agent_size / 2 - self._food_size / 2
        self._vision_start = min(self._vision_depth, max(0., self._vision_start))
        vision_step = (self._vision_depth - self._vision_start) / self._vision_resolution
        reference_depth = .66
        distance_points_at_same_depth = vision_step
        gamma = math.degrees(2 * math.asin((distance_points_at_same_depth / 2)
                                           / (self._vision_depth * reference_depth)))
        self.vision_field_n_angles = int(self._vision_field_angle / gamma)
        # print('vision:', self._vision_depth, self._vision_field_angle, self._vision_resolution,
        #       vision_step, gamma, self.vision_field_n_angles)
        self._max_steps = max_steps
        self._fps = 0 if fps is None else fps
        self._seed = seed
        self._seedsequence = SeedSequence(self._seed)

        self.agent_color = COLORS['red']
        self.food_color = COLORS['black']
        self.background_color = COLORS['white']  # todo: do it with Texture
        self.outside_color = COLORS['black']

        self._vision_point_win_radius = max(self._env2win_resize_factor * min(self._env_size) * .005, 1)
        self._vision_point_transparency = self.__get_vision_point_transparency(
            self._vision_point_win_radius, self._env2win_resize_factor * vision_step)
        # self._vision_points_group = pg.sprite.Group()  # all points are the same, I don't care which one goes where, thus the group is sufficient, I don't need an extra list.

        self._platform = Polygon((Point(0, 0), Point(0, self._env_size[1]),
                                  Point(*self._env_size), Point(self._env_size[0], 0)))
        assert is_simple_polygon(self._platform), (self._platform.wkt, self._platform.boundary)
        self._main_border = self._platform.boundary  # self._main_border_line v.s. self._main_border->.buffer(.05 * self._env_size[0], single_sided=True)
        self._fpsClock = pg.time.Clock()
        self.debug_info = defaultdict(dict)
        self.time_step = 1
        self.t = None

        self._step_count = None
        self._agent = None  # todo: probably this is not needed
        self._agent_group = pg.sprite.GroupSingle()  # todo: call this _agent
        self._food_items = []  # todo: probably this is not needed
        self._food_items_group = pg.sprite.Group()  # todo: call this _food_items
        self._env_img = None  # todo: make state (env_state) an object
        self.__get_observation_points_cache = {}
        self._food_items_collected = None

        # sq = SeedSequence(self._seed)
        # seed = sq.spawn(1)[0]
        # rng = default_rng(seed)
        # [rng.random(3) for rng in [default_rng(s) for s in sq.spawn(3)]] v.s. [rng.random(3) for rng in [default_rng(s.entropy) for s in sq.spawn(3)]]
        seeds = self._seedsequence.generate_state(3)  # default_rng() can be created with _seedsequence directly, spaces not.
        seeds = [int(s) for s in seeds]  # spaces want int.
        self._seedsequence = self._seedsequence.spawn(1)[0]  # update _seedsequence for new seeds next time you ask a seed.
        self.action_space = spaces.Box(low=0., high=1.,
                                       shape=(2,),
                                       dtype=np.float32,
                                       seed=seeds[0])  # [rotation, forward motion]
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self._vision_resolution,
                                                   self.vision_field_n_angles,
                                                   1),
                                            dtype=np.uint8,
                                            seed=seeds[1])
        self.env_space = spaces.Box(low=0, high=255,
                                    shape=self._env_img_shape,
                                    dtype=np.uint8,
                                    seed=seeds[2])

        # parse observation noise generator tuple:
        if observation_noise is not None:
            obs_noise_err_msg = (
                f"'observation_noise' not correct, see documentation of "
                f"{__name__ if __name__ != '__main__' else ''}.{type(self).__qualname__}.__init__"
            )  # note: the optimizer remove doc-strings, so I cannot print 'self.__init__.__doc__' here
            valid_random_generation_methods = {'normal', 'uniform'}
            if not isinstance(observation_noise, Sequence):
                raise TypeError(obs_noise_err_msg)
            if len(observation_noise) != 3 or observation_noise[0] not in valid_random_generation_methods:
                raise ValueError(obs_noise_err_msg)
            random_func = getattr(self.observation_space.np_random, observation_noise[0], None)
            assert random_func is not None, random_func
            observation_noise_args = observation_noise[1:]
            def get_observation_noise():
                return random_func(*observation_noise_args,
                                   size=self.observation_space.shape).astype(self.observation_space.dtype)
            observation_noise = get_observation_noise
        self._observation_noise = observation_noise

        bgd_col = self.background_color
        assert is_color(bgd_col), bgd_col
        self._soil = np.ones(self.env_space.shape, dtype=self.env_space.dtype) * bgd_col
        assert self.env_space.contains(self._soil)  # note: this check also dtype

        # print(type(self), type(self) is BaseForagingEnv)
        if type(self) is BaseForagingEnv:
            # do it here only if you are using it and not subclassing,
            # otherwise do it in the __init__ of the subclass.

            # background img:
            background = self._soil.copy()
            self._background_img = convert_image_to_pygame(background)

            # valid positions:
            self._compute_and_set_valid_positions(self._platform)

        # self._observation = None  # todo

        if init_agent_position is not None:
            pos = init_agent_position
            if not self.is_valid_position(pos, 'agent', is_env_pos=True):
                raise ValueError(f"agent position in 'init_agent_position'"
                                 f" is not valid: {pos}")
        self._init_agent_position = init_agent_position
        if init_food_positions is not None:
            if len(init_food_positions) != self._n_food_items:
                raise ValueError(f"{self._n_food_items} food item valid positions expected,"
                                 f" but 'init_food_positions' contains "
                                 f"{len(init_food_positions)} positions.")
            for pos in init_food_positions:
                if not self.is_valid_position(pos, 'food', is_env_pos=True):
                    raise ValueError(f"food item positions in 'init_food_positions'"
                                     f" is not valid: {pos}")
        self._init_food_positions = init_food_positions

        # Rendering:
        self._rendering = False  # rendering never used
        self._rendering_reset_request = True  # ask the rendering engine to reset the screen
        # init pygame module:
        pg_init_ret = pg.init()
        logging.debug(f"pg.init() returned: {pg_init_ret}")
        self._screen = None
        # self._env_surface = pg.Surface(self._env_size)

        # Other control variables:
        self.__has_been_ever_reset = False

    def _compute_and_set_valid_positions(self, platform):
        """Call this method in self.__init__()

        note: this function has side effects (it sets
        self._valid_agent_positions and self._valid_food_item_positions).
        """
        # self._valid_positions = (background == bgd_col)
        # assert self._valid_positions.dtype == bool

        self._valid_agent_positions = get_valid_item_positions_mask(
            platform, self._agent_size / 2, self._window_size, self._env_size)
        self._valid_food_item_positions = get_valid_item_positions_mask(
            platform, self._food_size / 2, self._window_size, self._env_size)
        # print(convert_pg_mask_to_array(self._valid_agent_positions).astype(int))
        # plt.matshow(convert_pg_mask_to_array(self._valid_agent_positions).astype(int))
        # plt.show()
        # plt.matshow(convert_pg_mask_to_array(self._valid_food_item_positions).astype(int))
        # plt.show()

    @property
    def env_size(self):
        return self._env_size

    @property
    def window_size(self):
        return self._window_size

    @staticmethod
    def _get_env_size(env_size):
        return tuple(env_size) if isinstance(env_size, Sequence) else (env_size, env_size)

    @property
    def n_food_items(self):
        return self._n_food_items

    @property
    def rotation_step(self):
        return self._rotation_step

    @property
    def forward_step(self):
        return self._forward_step

    @property
    def agent_size(self):
        if self._agent is None:
            return self._agent_size
        elif isinstance(self._agent, Agent):
            sz = self._agent.size
            assert self._agent_size == sz
            return sz
        else:
            raise AssertionError(self._agent)

    @property
    def food_size(self):
        return self._food_size

    @property
    def is_observation_noise_present(self):
        return self._observation_noise is not None

    @property
    def step_count(self):
        return self._step_count

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def food_items_collected(self):
        return self._food_items_collected

    @staticmethod
    def __get_vision_point_transparency(point_win_radius, vision_win_step):
        transparency = .8  # base
        threshold = .5
        assert 0 < threshold < 1
        area_covered_by_exterior_points = math.pi * point_win_radius ** 2 / vision_win_step ** 2
        # note: area could be more than 1 (if many point circles cover the same area)
        if area_covered_by_exterior_points >= threshold:
            transparency *= threshold / area_covered_by_exterior_points
        assert 0 <= transparency <= 1, transparency
        return int(255 * transparency)

    def step(self, action) -> tuple[np.ndarray, Real, bool, dict]:
        # logging.debug('Step')
        self.debug_info['step']['running'] = True
        self.debug_info['step']['count'] = self._step_count
        if not self.__has_been_ever_reset:
            self.reset()
        if not self.action_space.contains(action):
            raise ValueError(f"'action' is not in action_space; action={action}, action_space={self.action_space}")  # if the values are correct, the error is probably due to different dtype
        self.t += self.time_step

        # update environment state:
        # compute reward:
        reward = self._update_state(action)

        # get the observation from the environment state:
        observation = self._get_observation()
        if self._observation_noise is not None:
            # add noise
            observation = observation + self._observation_noise()
            observation.clip(self.observation_space.low, self.observation_space.high)

        # Is it done?:
        done = self._is_done()

        # debugging info:
        info = self._get_info()

        self._step_count += 1
        self.debug_info['step']['count'] = self._step_count
        self.debug_info['step']['running'] = False
        return observation, reward, done, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[gym.core.ObsType, tuple[gym.core.ObsType, dict]]:
        self.debug_info['reset']['running'] = True

        # reset init mandatory first due diligence:
        logging.debug('Reset')
        super().reset(seed=seed,
                      return_info=return_info,
                      options=options)
        self.__has_been_ever_reset = True
        self._step_count = 0
        self.t = 0
        self._food_items_collected = 0

        msg = (
            "Custom subclass should create a 'self._background_img'"
            " in the __init__() method."
        )
        assert hasattr(self, '_background_img'), msg
        msg = (
            "Custom subclass should call 'self._compute_and_set_valid_positions(platform)'"
            " in the __init__() method."
            " (to compute for valid positions of items in the new env class)"
        )
        assert hasattr(self, '_valid_agent_positions'), msg
        assert hasattr(self, '_valid_food_item_positions'), msg

        # init environment state:
        self._init_state()

        # get the observation from the environment state:
        observation = self._get_observation()
        if self._observation_noise is not None:
            # add noise
            observation = observation + self._observation_noise()
            observation.clip(self.observation_space.low, self.observation_space.high)
        assert self.observation_space.contains(observation), observation.dtype

        # ask the rendering engine to reset the screen:
        self._rendering_reset_request = True

        self.debug_info['reset']['running'] = False
        if not return_info:
            return observation
        else:
            return observation, self._get_info()

    def render(self, mode='human'):
        """Overrides the base method.

        Args:
            mode: a string containing 'human', or 'save[save_dir]'
                where save_dir is a path, or both separated by a '+';
                in 'save[save_dir]' mode, save_dir will be created if
                it doesn't exist, if it does exist an exception will
                be thrown.
        """
        logging.debug('Rendering')
        self.debug_info['render']['running'] = True
        if self._rendering is False:
            self._rendering = True
            # Since I'm using pygame stuffs also for computing the environment,
            # better to do pg.init() it in the __init__() method regardless of rendering or not;
            # note: after doing pg.quit(), you need to do pg.init() again if you want to use again pg,
            #   fixme: multiple independent environment objects concurrently running could close the each others screens
            #       ; fix it or allow only maximum one object to be alive at any moment.
            # # init pygame module:
            # pg.init()  # it is done only if self.render() is called at least once (see self.render() method).

        # reset screen if asked:
        if self._rendering_reset_request:
            self._rendering_reset_request = False
            # init rendering engine:
            # init main screen (init window if game in human mode):
            if mode == 'human' or re.search(r'^(?:.*\+)?human(?:\+.*)?$', mode):
                # logo = pg.image.load("logo32x32.png")
                # pg.display.set_icon(logo)
                pg.display.set_caption(f'{type(self).__qualname__}')
                # init screen:
                self._screen = pg.display.set_mode(self._window_size)
            else:
                self._screen = pg.Surface(self._window_size, pg.SRCALPHA)
            self._screen.blit(self._background_img, (0, 0))


            # if 'save' mode, init save_dir:
            match = re.search(r'^(?:.*\+)?save\[(.*)](?:\+.*)?$', mode)
            if match:
                os.makedirs(match.group(1), exist_ok=False)

        self._screen.blit(self._background_img, (0, 0))  # todo: update instead of rewriting each time
        self._draw_env(self._screen)

        # flip/update the screen:
        if mode == 'human' or re.search(r'^(?:.*\+)?human(?:\+.*)?$', mode):
            pg.display.flip()  # pg.display.update()

        # if in 'save' mode, save frames of the main screen.
        if mode != 'human':
            # you could also use mode.split('+') and then check separately the modes provided.
            match = re.search(r'^(?:.*\+)?save\[(.*)](?:\+.*)?$', mode)
            if match:
                pg.image.save(self._screen,
                              os.path.join(match.group(1), f"frame_{self._step_count}.jpg"))
            elif re.search(r'^(?:.*\+)?human(?:\+.*)?$', mode):
                pass
            else:
                raise ValueError(f"mode={mode!r} is not a valid rendering mode.")

        # tick() stops the program, do you want to see everything slowly or just some samples?
        # if you want to slow down and see everything use tick(), otherwise use set_timer()
        # and check events (or just use a timer and a variable tracking the last frame time)
        frame_dt = self._fpsClock.tick(self._fps)
        logging.debug(f'frame_dt: {frame_dt}')
        # pygame Clock tick is built for efficiency, not for precision,
        # thus take in account some dt error in the assertion.
        assert self._fps == 0 or frame_dt >= 1000 / self._fps * .99 - 1, (
            frame_dt, 1000 / self._fps * .99 - 1, self._fps)

        # todo: when using rendering (self._rendering is True) start a thread to check this; _check_quit_and_quit();
        for event in pg.event.get():
            if event.type == pg.QUIT:
                warn("Program manually closed. Quitting...")
                self.close()
                sys.exit()

        self.debug_info['render']['running'] = False

    def _check_quit_and_quit(self) -> None:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                warn("Program manually closed. Quitting...")
                self.close()
                sys.exit()

    def close(self):
        # todo: as for files:
        #  A closed file cannot be read or written any more. Any operation, which requires that the file be opened
        #  will raise a ValueError after the file has been closed. Calling close() more than once is allowed.
        self.debug_info['close']['running'] = True
        self.__has_been_ever_reset = False
        pg.quit()
        self.debug_info['close']['running'] = False

    def __get_sizes(self, window_size=None, env_size=None):  # todo: useless, remove it.
        """Returns window_size and env_size of the environment.
        If window_size or env_size is not provided (or None) it uses the one
        of the self object, otherwise it uses the one provided.
        In the second case, if self._window_size or self._env_size or is not set raises an error."""
        if window_size is None:
            if getattr(self, "_window_size", None) is None:
                raise ValueError(
                    "If window_size is not provided (or None) it uses the one "
                    "of the self object, in this case self._window_size should "
                    "be set, otherwise it is raised an error.")
            window_size = self._window_size
        if env_size is None:
            if getattr(self, "_env_size", None) is None:
                raise ValueError(
                    "If env_size is not provided (or None) it uses the one "
                    "of the self object, in this case self._env_size should "
                    "be set, otherwise it is raised an error.")
            env_size = self._env_size
        return window_size, env_size

    def get_point_env2win(self, point) -> tuple:
        """Take a point in the environment coordinate system
        and transform it in the window coordinate system (i.e. a pixel).
        """
        return get_point_env2win(point, self._window_size, self._env_size)

    def get_point_win2env(self, point) -> Pos:
        """Take a point in the window coordinate system (i.e. a pixel)
        and transform it in the environment coordinate system.
        """
        return get_point_win2env(point, self._window_size, self._env_size)

    @override
    def _draw_env(self, screen) -> None:
        """Draw the environment in the screen.

        Note: to override the background update the attribute
        ``self._background_img`` (pygame surface) in self.__init__()
        """

        # draw food items:
        self._food_items_group.draw(screen)

        # draw agent:
        # agent is drawn later than food items, so that it is shown on top.
        self._agent_group.draw(screen)

        # draw field of view of the agent:
        r = self._vision_point_win_radius
        for pt in self._get_observation_points().reshape((-1, 2)):
            if 0 <= pt[0] <= self._env_size[0] and 0 <= pt[1] <= self._env_size[1]:
                pt = self.get_point_env2win(pt)
                # pg.draw.circle(screen, COLORS['blue'], pt.coords[0], r)
                circle = pg.Surface((r * 2, r * 2), pg.SRCALPHA)
                pg.draw.circle(circle, (*COLORS['blue'], self._vision_point_transparency), (r, r), r)
                screen.blit(circle, (pt[0] - r, pt[1] - r))
        # # todo: put them all in a group and draw only at the end: is it faster?
        # points = self._get_observation_points().reshape((-1, 2))
        # assert len(self._vision_points_group) == len(points), (len(self._vision_points_group), len(points))
        # for vpt, pt in zip(self._vision_points_group.sprites(), points):
        #     if 0 <= pt[0] <= self._env_size[0] and 0 <= pt[1] <= self._env_size[1]:
        #         vpt.pos = pt  # not all points are updated; fixme (if it is worth it, but it can get slower, thus not worth it if that is the case)

    @override
    def _init_state(self) -> None:
        """Create and return a new environment state (used for initialization or reset)"""

        # init environment space:
        self._env_img = self._background_img.copy()    # .convert()  # create a copy in the fastest format for blitting
        assert self._env_img.get_size() == self._env_img_size, (
            self._env_img.get_size(), self._env_img_size)

        # get random init positions:
        n_random_positions = 0
        positions = []
        if self._init_agent_position is None:
            n_random_positions += 1
        else:
            positions.append(self._init_agent_position)
        if self._init_food_positions is None:
            n_random_positions += self._n_food_items
        if n_random_positions:
            positions.extend(self._get_random_non_overlapping_positions(
                n_random_positions,
                [self._food_size / 2] * self._n_food_items + [self._agent_size / 2],
                ['food'] * self._n_food_items + ['agent']))
        if self._init_food_positions is not None:
            positions.extend(self._init_food_positions)

        # init agent in a random position:
        pos = positions.pop()
        assert self.is_valid_position(pos, 'agent'), (pos, self._agent_size / 2)
        hd = self.env_space.np_random.rand() * 360
        if self._agent is not None:
            self._agent.kill()
        self._agent = Agent(pos, self._agent_size, head_direction=hd, color=self.agent_color, env=self)
        # print(self._agent_size, self._agent.radius, str(self._agent.pos))
        self._agent_group.empty()  # remove all previous food item sprites from the group
        self._agent_group.add(self._agent)

        # init food items:
        self._food_items = []
        while positions:
            pos = positions.pop()
            assert self.is_valid_position(pos, 'food'), (pos, self._food_size / 2)
            self._food_items.append(FoodItem(pos, self._food_size, self.food_color, env=self))
        assert isinstance(self._food_items[0], pg.sprite.Sprite)
        self._food_items_group.empty()  # remove all previous food item sprites from the group
        self._food_items_group.add(*self._food_items)
        assert len(self._food_items) == len(self._food_items_group)

        # init field of view of the agent:
        # # init vision points
        # # self._vision_points_group.empty()  # remove all previous food item sprites from the group
        # # r = self._vision_point_win_radius
        # # vision_points = [CircleItem(self._agent.pos,  # initialize them all in a valid position: the agent position.
        # #                             r*2,
        # #                             (*COLORS['blue'], self._vision_point_transparency),
        # #                             env=self)
        # #                  for _ in range(reduce(mul, self.observation_space.shape, 1))]
        # # self._vision_points_group.add(*vision_points)

        # self.debug_info['_init_state']

    @override
    def _update_state(self, action) -> Real:
        """Update environment state. Compute and return reward.

        Rotate agent and then move it forward.
        """
        dtype = action.dtype
        assert dtype.kind == 'f', dtype
        epsilon = np.finfo(dtype).resolution * 3

        # agent negligible movement (percentage in relation to its size)
        agent_negligible_movement_pct = .01

        # update agent position:
        # >> rotate:
        rotation_step = self._rotation_step * (action[0] * 2 - 1.)
        assert -1. * self._rotation_step <= rotation_step <= 1. * self._rotation_step, rotation_step
        # add initial rotation inertia
        if abs(rotation_step) < self._rotation_step * .02:
            logging.debug('rotation inertia hit')
            rotation_step = 0.
        self._agent.head_direction = (self._agent.head_direction
                                      + rotation_step) % 360
        # >> go forward:
        forward_step = self._forward_step * action[1]
        assert 0. <= forward_step, forward_step
        assert 0. <= forward_step <= 1. * self._forward_step, forward_step
        # Try first if pos is it valid,
        # only if is not valid (there is collision) use other algorithm to find a correct pos.
        # Note: this is for efficiency reason, actually a correct check would
        #       check also any other position in between the starting
        #       position and the final position of the forward movement.
        pos = transform.rotate(
            transform.translate(self._agent.pos, forward_step, 0),
            self._agent.head_direction,
            origin=self._agent.pos,
        )
        win_pos = self.get_point_env2win(pos)
        valid_pos = pos
        if not self.is_valid_position(win_pos, 'agent', is_env_pos=False):
            valid_pos = self._agent.pos  # the agent initial position is valid, start from here.
            win_agent_pos = self.get_point_env2win(self._agent.pos)
            win_mov_line_iter = pixels_on_line(*win_agent_pos,
                                               *win_pos)
            # the agent initial position is of course valid, thus remove it from the iterator:
            win_pos = next(win_mov_line_iter)
            # first_point = win_pos
            # print((first_point, win_agent_pos))
            # np.testing.assert_array_equal(win_agent_pos, first_point,
            #                               err_msg=f"{(first_point, win_agent_pos)}")
            # second_point = None
            win_valid_pos = None
            for win_pos in win_mov_line_iter:
                # if second_point is None:
                #     second_point = win_pos
                #     print((second_point, win_agent_pos))
                #     assert not np.array_equal(win_agent_pos, second_point), (
                #         f"{(second_point, win_agent_pos)}")
                #     np.testing.assert_allclose(win_agent_pos, second_point,
                #                                atol=1,
                #                                err_msg=f"{(second_point, win_agent_pos)}")
                if self.is_valid_position(win_pos, 'agent', is_env_pos=False):
                    win_valid_pos = win_pos
                else:
                    break  # first occurrence not valid stops the forward movement (i.e. collision)
            if win_valid_pos is not None:
                valid_pos = self.get_point_win2env(win_valid_pos)
        assert self.is_valid_position(valid_pos, 'agent'), (valid_pos, win_pos, self._agent.radius)
        self._agent.pos = valid_pos

        # food collected?
        # note: it could be that more than one food item is collected at the same time
        food_collected = 0
        remove_food = set()
        for idx, food in enumerate(self._food_items):  # O(N), but foods are few
            if self._is_food_item_collected(self._agent, food):
                remove_food.add(idx)
                food_collected += 1
                self._food_items_collected += 1
                logging.log(logging.DEBUG + 3,
                            f'Food collected.    [Total food items collected until now: {self._food_items_collected}]')
        for j in remove_food:
            food = self._food_items[j]
            food.kill()  # remove from all groups
        self._food_items = [food for i, food in enumerate(self._food_items) if i not in remove_food]  # O(len(self._food_items)) if isinstance(remove_food, set) else O(len(remove_food) * len(self._food_items))
        assert len(self._food_items) == len(self._food_items_group) == self._n_food_items - self._food_items_collected
        assert food_collected >= 0, food_collected

        # update _env_img: -> pos: & rotate:
        # todo

        # compute reward:
        reward = food_collected

        # self.debug_info['_update_state']
        return reward

    # todo: you can make it O(1) improve it by having a mask with food collected positions calculated in the init,
    #       when collecting a food, remove positions from mask or leave them and when in a collected position do
    #       a further check (this second option probably is simpler to be coded, and probably more efficient for few
    #       food items on the map).
    #       Note: for now the main use-case of Env will be with few foods (or even one)
    # todo: improve efficiency (and test if efficiency is improved)
    @staticmethod
    def _is_food_item_collected(agent: Agent, food_item: FoodItem) -> bool:
        """Food item is collected when the agent step upon (or touches) the center of the food item."""
        # Is the food center inside (or on the border of) the agent circle?
        return is_point_in_circle(food_item.pos, agent.radius, agent.pos)

    @override
    def _get_observation(self) -> np.ndarray:
        """Get the observation (without noise) from the environment state.

        Note: the noise will be added by the step() and reset() methods if an
            ``observation_noise`` argument was provided when the environment
            was constructed.
        """
        points = self._get_observation_points()
        obs = np.empty((*self.observation_space.shape[:-1], self._n_channels), dtype=self.observation_space.dtype)
        for i in range(self.observation_space.shape[0]):
            for j in range(self.observation_space.shape[1]):
                obs[i, j] = self._get_point_color(points[i][j])
        obs = black_n_white(obs)
        self.debug_info['_get_observation']['obs'] = obs
        return obs

    def _get_observation_points(self) -> np.ndarray:

        if self.__get_observation_points_cache.get('last_step_count', None) != self._step_count:
            self.__get_observation_points_cache['last_step_count'] = self._step_count

            # Create a segment of length self._vision_depth from the agent center (offset self._vision_start)
            #   with self._vision_resolution points,
            #   the starting segment is parallel to x-axis (same direction of x-axis)
            line_endpoints = (
                (self._agent.pos[0] + self._vision_start, self._agent.pos[1]),
                (self._agent.pos[0] + self._vision_depth, self._agent.pos[1])
            )
            span_x = np.linspace(line_endpoints[0][0], line_endpoints[1][0], num=self._vision_resolution, endpoint=True)
            span_y = np.linspace(line_endpoints[0][1], line_endpoints[1][1], num=self._vision_resolution, endpoint=True)
            assert span_x[-1] == line_endpoints[1][0], f'{span_x[-1]:.30f}'
            assert span_y[-1] == line_endpoints[1][1], f'{span_y[-1]:.30f}'
            line = np.column_stack((span_x, span_y))
            assert len(line) == self._vision_resolution

            # Rotate the segment for self.vision_field_n_angles times between
            #   "self._agent.head_direction + self._vision_field_angle / 2" and
            #   "self._agent.head_direction - self._vision_field_angle / 2",
            #   with agent center as rotation origin

            # Compute absolute angles for rotation:
            angles = np.linspace(self._agent.head_direction + self._vision_field_angle / 2,
                                 self._agent.head_direction - self._vision_field_angle / 2,
                                 num=self.vision_field_n_angles, endpoint=True)
            assert len(angles) == self.vision_field_n_angles

            # Rotate segments and put them in the observation matrix:
            points = np.empty((self._vision_resolution, self.vision_field_n_angles, 2), dtype=line.dtype)
            for j, alpha in enumerate(angles):
                points[:, j] = transform.rotate(line, alpha, self._agent.pos)

            self.__get_observation_points_cache['points'] = points
        # else:
        #     print('cache hit')

        return self.__get_observation_points_cache['points']

    @override
    def _get_point_color(self, point):
        """Returns the color in the env_img in correspondence of ``point``
        (``point`` in the env space)."""
        col = self.outside_color
        if 0 <= point[0] <= self._env_size[0] and 0 <= point[1] <= self._env_size[1]:
            col = self._env_img.get_at(
                get_point_env2win(point, self._window_size, self._env_size)
            )[:-1]  # discard the alpha value
            assert len(col) == 3, col
        return col

    @override
    def _is_done(self) -> bool:
        done = False
        if self._max_steps is not None:
            done = done or self._step_count >= self._max_steps - 1

        assert self._food_items_collected <= self.n_food_items, (self._food_items_collected, self.n_food_items)
        done = done or self._food_items_collected >= self.n_food_items

        self.debug_info['_is_done'] = {'done': done,
                                       'food_items_collected': self._food_items_collected,
                                       'n_food_items': self.n_food_items,
                                       'step_count': self._step_count,
                                       'step_count_note': 'this is the self._step_count before self.step() is finished, thus starting form 0',
                                       'max_steps': self._max_steps}
        return done

    @override
    def _get_info(self) -> dict:
        """Get debugging info (the environment state, plus some extra useful information).

        Do not change values in the returned info-dict (undefined behaviour),
        it is for read-only purpose.
        """
        info = {
            'state': {
                'agent': self._agent,
                'food_items': self._food_items,
            },
            'env_info': {
                'window_size': self._window_size,
                'env_size': self._env_size,
                'n_food_items': self._n_food_items,
                'rotation_step': self._rotation_step,
                'forward_step': self._forward_step,
                'agent_size': self._agent_size,
                'food_size': self._food_size,
                'vision_depth': self._vision_depth,
                'vision_field_angle': self._vision_field_angle,
                'vision_resolution': self._vision_resolution,
                'max_steps': self._max_steps,
                'fps': self._fps,
                'seed': self._seed,
                'time_step': self.time_step,
            },
            'env_img': self._env_img,  # memory_evolution.utils.convert_pg_surface_to_array(self._env_img), # converting is too expensive, don't do it
            'current_step': self._step_count,
            't': self.t,
            'debug_info': self.debug_info,
        }
        return info

    @override
    def is_valid_position(self, pos, item: Literal['agent', 'food'], is_env_pos: bool = True) -> bool:
        """Returns True if pos is in a valid position.

        Args:
            pos: a point in the environment.
            item: for which item it is checked if the position is valid.
            is_env_pos: True if ``pos`` is in the environment coordinate
                system, False if it is in the window coordinate system (i.e.
                a pixel)

        Returns:
            True if pos is in a valid position.
        """
        if len(pos) != 2:
            raise ValueError("'pos' should be 2D (and without channels)")
        if is_env_pos:
            res = (0 <= pos[0] <= self._env_size[0] and 0 <= pos[1] <= self._env_size[1])
        else:
            res = (0 <= pos[0] < self._window_size[0] and 0 <= pos[1] < self._window_size[1])
        if res:
            if is_env_pos:
                # get win_pos:
                pos = self.get_point_env2win(pos)
            if item == 'agent':
                res = self._valid_agent_positions.get_at(pos)
            elif item == 'food':
                res = self._valid_food_item_positions.get_at(pos)
            else:
                raise ValueError(f"item: Literal['agent', 'food'], got {item!r} instead.")
        return bool(res)

    def _get_random_non_overlapping_positions(self,
                                              n,
                                              radius: Union[list, int],
                                              items=None,
                                              ) -> list[Pos]:
        # todo: try both methods, cache the time they take, and use the more efficient
        #  (or chose by calculating free area)
        # return get_random_non_overlapping_positions_with_triangulation(
        #     n, radius, self._platform, self._env_size, self.env_space.np_random)
        return get_random_non_overlapping_positions_with_lasvegas(
            n, radius, self._platform, self._env_size, self.env_space.np_random,
            self, items,
            optimization_with_platform_triangulation=False,
        )

