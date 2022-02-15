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
import pygame
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union

from memory_evolution.utils import COLORS, is_color, Pos, convert_image_to_pygame
from memory_evolution.utils import MustOverride, override


class Texture:
    """Positive static pattern applied to the soil and maze.
    - static uniform noise
    - lines
    - curves
    - Gaussian random field
    Applied positively to the floor and negatively to the borders or high-contrast objects.
    """
    pass


class Agent:
    """Agent"""

    def __init__(self, pos: Point, size: Real, head_direction: Union[int, float]):
        """

        Args:
            pos: a point representing the position of the agent.
            size: the agent diameter.
            head_direction: head direction in degrees.
        """
        self.pos = pos
        self._size = size
        self._radius = size / 2
        self.head_direction = head_direction
        if head_direction and self.pos.has_z:  # self.pos.ndim != 2:
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

    @property
    def size(self):
        return self._size

    @property
    def radius(self):
        return self._radius

    def get_polygon(self):
        polygon = self.pos.buffer(self._radius)
        assert type(polygon) is Polygon
        return polygon

    def __str__(self):
        return f"{type(self).__name__}({self.pos.wkt}, {self._size}, {self.head_direction})"

    def __repr__(self):
        return (f"{__name__ if __name__ != '__main__' else ''}"
                f".{type(self).__qualname__}({self.pos.wkt}, "
                f"{self._size!r}, {self.head_direction!r})")


class FoodItem:
    """Rewarding (maybe, not actual reward, but increase in agent life span) food items."""

    def __init__(self, pos: Point, size: Real):
        """

        Args:
            pos: a point representing the position of the food item.
            size: the food item diameter.
        """
        self.pos = pos
        self._size = size
        self._radius = size / 2

    @property
    def size(self):
        return self._size

    @property
    def radius(self):
        return self._radius

    def get_polygon(self):
        polygon = self.pos.buffer(self._radius)
        assert type(polygon) is Polygon
        return polygon

    def __str__(self):
        return f"{type(self).__name__}({self.pos.wkt}, {self._size})"

    def __repr__(self):
        return (f"{__name__ if __name__ != '__main__' else ''}.{type(self).__qualname__}("
                f"{self.pos.wkt}, {self._size!r})")


class BaseForagingEnv(gym.Env, MustOverride):
    """Custom Environment that follows gym interface,
    it develops an agent moving in an environment in search for foods items.

    Food item is collected when: "food collected if the agent intersects the central point of the food"
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 window_size: Union[int, Sequence[int]] = 640,  # (640, 480),
                 env_size: Union[float, Sequence[float]] = 1.,
                 n_food_items: int = 3,
                 rotation_step: float = 90,
                 forward_step: float = .1,
                 agent_size: float = .1,
                 food_size: float = .1,
                 min_border_width: float = .05,
                 fps: int = 60,  # 60 or 30  # todo
                 seed=None,
                 ) -> None:
        """Inits environment

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
            min_border_width: positive number, minimum width of a border.
            fps: frames per second.
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
        if not isinstance(min_border_width, float):
            raise TypeError(min_border_width)

        self._n_channels = 3
        self._env_size = tuple(env_size) if isinstance(env_size, Sequence) else (env_size, env_size)
        self._window_size = (tuple(window_size)
                             if isinstance(window_size, Sequence)
                             else (window_size, math.ceil(window_size * self._env_size[1] / self._env_size[0])))
        assert 2 == len(self._env_size) == len(self._window_size)
        self._env2win_resize_factor = self._window_size[0] / self._env_size[0]
        self._env_img_shape = (self._window_size[1], self._window_size[0], self._n_channels)
        self._n_food_items = n_food_items
        self._rotation_step = rotation_step
        self._forward_step = forward_step
        self._agent_size = agent_size
        self._food_size = food_size
        self._min_border_width = min_border_width
        self._fps = fps
        self._seed = seed

        self.agent_color = COLORS['red']
        self.food_color = COLORS['black']
        self.background_color = COLORS['white']  # todo: do it with Texture

        self._platform = Polygon((Point(0, 0), Point(0, self._env_size[1]),
                                  Point(*self._env_size), Point(self._env_size[0], 0)))
        # self._borders = []
        self._main_border = self._platform.boundary
        self._fpsClock = pygame.time.Clock()
        self.debug_info = defaultdict(dict)

        self.step_count = None
        self._agent = None
        self._food_items = None
        # self.__food_items_polygons = list()  # set()
        self.__food_items_union = None
        self._env_img = None  # todo: make state (env_state) an object

        self.action_space = spaces.Box(low=np.asarray([-1., 0.], dtype=np.float32),
                                       high=np.asarray([1., 1.], dtype=np.float32),
                                       dtype=np.float32,
                                       seed=self._seed)  # [rotation, forward motion]
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(10, 10, 1),
                                            dtype=np.uint8,
                                            seed=self._seed)
        self.env_space = spaces.Box(low=0, high=255,
                                    shape=self._env_img_shape,
                                    dtype=np.uint8,
                                    seed=self._seed)

        bgd_col = self.background_color
        assert is_color(bgd_col), bgd_col
        self._soil = np.ones(self.env_space.shape, dtype=self.env_space.dtype) * bgd_col
        assert self.env_space.contains(self._soil)  # note: this check also dtype
        self._background = self._soil.copy()
        assert self.env_space.contains(self._background)
        self._background_img = convert_image_to_pygame(self._background)

        # self.__observation = None  # todo

        # Rendering:
        self._rendering = False  # rendering never used
        self._rendering_reset_request = True  # ask the rendering engine to reset the screen
        # init pygame module:  # it is done only is self.render() is called at least once (see self.render() method).
        # pygame.init()  # it is done only is self.render() is called at least once (see self.render() method).
        self._screen = None
        # self._env_surface = pygame.Surface(self._env_size)

        # Other control variables:
        self.__has_been_ever_reset = False

    def step(self, action) -> tuple[np.ndarray, Real, bool, dict]:
        print('Step')
        if not self.__has_been_ever_reset:
            # warn('Calling step() method before reset() method. Forcing reset() method...')
            self.reset()
        if not self.action_space.contains(action):
            raise ValueError(f"'action' is not in action_space; action={action}, action_space={self.action_space}")

        # update environment state:
        # compute reward:
        reward = self._update_state(action)

        # create an observation from the environment state:
        observation = self._get_observation()

        # Is it done?:
        done = self._is_done()

        # debugging info:
        info = self._get_info()

        self.step_count += 1
        return observation, reward, done, info

    def reset(self):
        print('Reset')
        self.__has_been_ever_reset = True
        self.step_count = 0

        # init environment state:
        self._init_state()

        # create an observation from the environment state:
        observation = self._get_observation()
        assert self.observation_space.contains(observation)

        # ask the rendering engine to reset the screen:
        self._rendering_reset_request = True

        return observation

    def render(self, mode='human'):
        print('Rendering')
        if self._rendering is False:
            self._rendering = True

            # init pygame module:
            pygame.init()

        self._rendering_reset_request = True  # todo: update instead of rewriting each time
        if self._rendering_reset_request:
            self._rendering_reset_request = False
            # init rendering engine:
            # init window:
            # logo = pygame.image.load("logo32x32.png")
            # pygame.display.set_icon(logo)
            pygame.display.set_caption(f'{type(self).__qualname__}')
            # init screen:
            self._screen = pygame.display.set_mode(self._window_size)
            self._screen.blit(self._background_img, (0, 0))

        # draw agent:
        # todo: use Sprites and update the position without rewriting all the screen again
        pygame.draw.circle(self._screen,
                           self.agent_color,
                           np.asarray(self._agent.pos.coords[0]) * self._env2win_resize_factor,
                           self._agent.radius * self._env2win_resize_factor)

        # draw food items:
        for food in self._food_items:
            pygame.draw.circle(self._screen,
                               self.food_color,
                               np.asarray(food.pos.coords[0]) * self._env2win_resize_factor,
                               food.radius * self._env2win_resize_factor)

        # flip/update the screen:
        pygame.display.flip()  # pygame.display.update()

        # tick() stops the program, do you want to see everything slowly or just some samples?
        # if you want to slow down and see everything use tick(), otherwise use set_timer()
        # and check events (or just use a timer and a variable tracking the last frame time)
        dt = self._fpsClock.tick(self._fps)
        print(dt)
        # pygame Clock tick is built for efficiency, not for precision,
        # thus take in account some dt error in the assertion.
        assert dt >= 1000 / self._fps * .99 - 1, (dt, 1000 / self._fps * .99 - 1, self._fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                warn("Program manually closed. Quitting...")
                self.close()
                sys.exit()

    def close(self):
        # pygame.display.quit()
        pygame.quit()

    def _check_quit_and_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                warn("Program manually closed. Quitting...")
                self.close()
                sys.exit()

    @override
    def _init_state(self) -> None:
        """Create and return a new environment state (used for initialization or reset)"""

        # init environment space:
        # pass
        # self._env_img = self._soil.copy()  # todo: togli
        # assert self.env_space.contains(self._env_img)  # todo: togli

        positions = self._get_random_non_overlapping_positions(
            1 + self._n_food_items,
            [self._food_size / 2] * self._n_food_items + [self._agent_size / 2])

        # init agent in a random position:
        pos = positions.pop()
        hd = self.env_space.np_random.rand() * 360
        self._agent = Agent(pos, self._agent_size, head_direction=hd)
        # print(self._agent_size, self._agent.radius, str(self._agent.pos))

        # init food items:
        self._food_items = []
        while positions:
            self._food_items.append(FoodItem(positions.pop(), self._food_size))
        # self.__food_items_polygons = self.__get_food_items_polygons(self._food_items)
        # assert self._n_food_items == len(self._food_items) == len(self.__food_items_polygons)
        self.__food_items_union = self.__get_food_items_union(self._food_items)

        # self.debug_info['_init_state']

    # @staticmethod
    # def __get_food_items_polygons(food_items):
    #     """Use ``self.__food_items_polygons`` for fast access instead
    #     (this method is only for updating ``self.__food_items_polygons``)
    #
    #     It preserves the order of food_items."""
    #     return [food.get_polygon() for food in food_items]

    @staticmethod
    def __get_food_items_union(food_items):
        """Use ``self.__food_items_union`` for fast access instead
        (this method is only for updating ``self.__food_items_union``)"""
        return unary_union([food.get_polygon() for food in food_items])

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

        # # update agent position:
        # # rotate
        # self._agent.head_direction = (self._agent.head_direction
        #                               + self._rotation_step * action[0]) % 360
        # forward_step = self._forward_step * action[1]
        # pos = rotate(
        #     translate(self._agent.pos, forward_step, 0),
        #     self._agent.head_direction,
        #     origin=self._agent.pos,
        # )
        # pos_buff = pos.buffer(self._agent.radius)
        # # fixme: prendi in considerazione anche il rettangolo di spostamneto per la collision
        # if not self.is_valid_position(pos_buff):
        #     collision = pos_buff.intersection(self._platform.exterior)
        #     agent_pos_dist = collision.distance(self._agent.pos)  # fixme: questa non è la vera distanza di collisione
        #     # fixme: distanza di collisione dovrebbe essere parallela alla direzione dello spostamento
        #     #        (puoi fare questo con una trasformazione, e poi il calcolo della distanza, ma è complesso,
        #     #        c'è un metodo migliore?)
        #     # fixme: inoltre prendi in considerazione anche il rettangolo di spostamneto per la collision
        #     direction0 = LineString((self._agent.pos, pos))  # small segment
        #     # scaling is to get a segment bigger than the diagonal of env_size
        #     # (diagonal = math.sqrt(env_size[0]**2 + env_size[1]**2);
        #     #  diagonal < env_size[0]**2 + env_size[1]**2 < (env_size[0] + env_size[1])**2;
        #     #  faster, but overflows or float precision errors??)
        #     # (we would actually like to have a line, not a segment).
        #     # diagonal_factor = math.sqrt(self._env_size[0]**2 + self._env_size[1]**2)
        #     diagonal_factor = (self._env_size[0] + self._env_size[1]) ** 2  # faster? float precision errors?
        #     scaling = diagonal_factor / direction0.length
        #     direction1 = scale(direction0,
        #                        scaling, scaling, scaling,
        #                        origin=self._agent.pos)
        #     direction2 = scale(direction0,
        #                        -scaling, -scaling, -scaling,
        #                        origin=self._agent.pos)
        #     direction = unary_union((direction1, direction2))
        #     itx = direction.intersection(self._main_border)
        #     assert hasattr(itx, 'geoms'), (
        #         direction0.coords[:], direction.wkt, itx.wkt)
        #     itx = itx.geoms
        #     assert len(itx) == 2 and all((type(x) is Point) for x in itx), (
        #         direction0.coords[:], direction.wkt, [x.wkt for x in itx])
        #     direction_dist = collision.distance(direction)
        #     extra_step = math.sqrt(self._agent.radius ** 2 - direction_dist ** 2)
        #     forward_step = math.sqrt(agent_pos_dist ** 2 - direction_dist ** 2) - extra_step
        #     forward_step *= 1 - epsilon  # preventing floating precision errors, stay safe
        #     pos = rotate(
        #         translate(self._agent.pos, forward_step, 0),
        #         self._agent.head_direction,
        #         origin=self._agent.pos,
        #     )
        #     pos_buff = pos.buffer(self._agent.radius)
        # assert self.is_valid_position(pos_buff), (pos.wkt, self._agent.radius)
        # self._agent.pos = pos

        # update agent position:
        # rotate
        self._agent.head_direction = (self._agent.head_direction
                                      + self._rotation_step * action[0]) % 360
        agent_buff = self._agent.get_polygon()
        forward_step = self._forward_step * action[1]
        prev_step = forward_step
        # Try first if pos is it valid,
        # only if is not valid (there is collision) use bisection algorithm to find a correct pos.
        valid = False
        i = 0
        while not valid:
            # print(forward_step, self._env_size)
            pos = rotate(
                translate(self._agent.pos, forward_step, 0),
                self._agent.head_direction,
                origin=self._agent.pos,
            )
            mov_line = LineString((self._agent.pos, pos))
            movement_buff = unary_union((agent_buff,
                                         mov_line.buffer(self._agent.radius),
                                         mov_line.buffer(-self._agent.radius),
                                         pos.buffer(self._agent.radius)))
            if self.is_valid_position(movement_buff):
                assert prev_step > 0 and prev_step >= forward_step
                if prev_step == forward_step:
                    valid = True
                else:
                    dist = pos.buffer(self._agent.radius).distance(self._platform.exterior)
                    # print(f'valid, using bisection (i={i})', forward_step, dist)
                    if dist <= agent_negligible_movement_pct * self._agent.size:
                        valid = True
                    else:
                        forward_step = (prev_step + forward_step) / 2
                        # Do not update prev_step here
            else:
                # print(f'not valid (i={i})', self._agent.pos, pos)
                prev_step = forward_step
                forward_step /= 2
            # If the program get stuck in this while loop you should still be able to quit:
            # don't check at any iteration (waste of resources),
            # don't check in at the first iteration (it will exit normally from the loop with high probability);
            if i >= 20 - 1:
                if i % 100 == 99 and self._rendering:
                    print(f"In the while loop (i={i}), quit event check.")
                    self._check_quit_and_quit()
                if i == 20 - 1 and not valid:
                    warn(f'Bisection method for collision correction is taking more than {i+1} iterations...')
                if i == 30 - 1 and not valid:
                    warn(f'Bisection method for collision correction is taking more than {i+1} iterations...')
                if i == 40 - 1 and not valid:
                    warn(f'Bisection method for collision correction is taking more than {i+1} iterations...')
            i += 1
        if prev_step != 0:
            mov_line = LineString((self._agent.pos, pos))
            movement_buff = unary_union((agent_buff,
                                         mov_line.buffer(self._agent.radius),
                                         mov_line.buffer(-self._agent.radius),
                                         pos.buffer(self._agent.radius)))
            assert self.is_valid_position(movement_buff), (pos.wkt, self._agent.radius)
        self._agent.pos = pos

        # food collected?
        # "food collected if the agent intersects the central point of the food"
        if self.__food_items_union.intersects(agent_buff):
            # find which food (it could be more than one, if food items are allowed to be close to each other):
            # # for idx, food_buff in enumerate(self.__food_items_polygons):
            # #     if food_buff.intersects(agent_buff):
            # #         intersection = self.food_buff.intersection(agent_buff)
            # #         intersection_pct = intersection.area / food_buff.area
            remove_food = []
            for idx, food in enumerate(self._food_items):
                if agent_buff.intersects(food.pos):
                    remove_food.append(idx)
                    print('Food collected')
            for j, idx in enumerate(remove_food):
                # if self._food_items is small this is efficient, otherwise not
                # (this is O(R*N), N foods and R removed, you could do it O([N+]R)
                # by swapping the last and the one you want removing and then pop the last)
                self._food_items.pop(idx - j)
            # self.__food_items_polygons = self.__get_food_items_polygons(self._food_items)
            self.__food_items_union = self.__get_food_items_union(self._food_items)

        # update _env_img: -> pos: & rotate:
        # todo

        # compute reward:
        reward = 0

        # self.debug_info['_update_state']
        return reward

    @override
    def _get_observation(self):  # todo: very costly; can be improved
        """create an observation from the environment state"""
        obs = self.observation_space.sample()
        self.debug_info['_get_observation']['obs'] = obs
        return obs

    @override
    def _is_done(self) -> bool:
        self.debug_info['_is_done'] = {}
        return self.step_count >= 40 - 1

    @override
    def _get_info(self) -> dict:
        """Get debugging info (the environment state, plus some extra useful information).

        Do not change values in the returned info-dict (undefined behaviour),
        it is for read-only purpose.
        """
        info = {
            'state': {
                'env_shape': self.env_space.shape,
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
                'min_border_width': self._min_border_width,
                'fps': self._fps,
                'seed': self._seed,
            },
            'env_state_img': self._env_img,
            'current_step': self.step_count,
            'debug_info': self.debug_info,
        }
        return info

    @override
    def is_valid_position(self, pos: Union[Point, Polygon]) -> bool:
        """Returns True if pos is in a valid position.

        Note: Under the hood it uses ``contains()`` which it means if a point
        is on the border (e.g. Point(0, 0)) is considered invalid, but a
        polygon lying on the border is valid (e.g. Polygon((Point(0, 0),
        Point(0, 0.01), Point(0.01, 0.01), Point(0.01,0)))).

        Args:
            pos: a position or a polygon (Point or circle or Polygon (circle/poit.buffer is a Polygon)).

        Returns:
            True if pos is in a valid position.
        """
        assert type(Point(0, 0).buffer(1)) is Polygon, type(Point(0, 0).buffer(1))  # todo: move in tests
        # convert pos to index if it is not already:
        if not isinstance(pos, (Point, Polygon)):
            raise TypeError('`pos` should be an instance of Point or Polygon')
        if pos.has_z:
            raise ValueError('`pos` should be 2D (and without channels)')
        return self._platform.contains(pos)  # and unary_union(self._borders).disjoint(pos)

    def _get_random_non_overlapping_positions(self,
                                              n,
                                              radius: Union[list, int],
                                              overlapping: list = None,  # set = None
                                              ) -> list:
        if overlapping is None:
            overlapping = list()  # set()
        if isinstance(radius, int):
            radius = [radius] * n
        if n != len(radius):
            raise ValueError(f"`radius` should be int or a list of `n` integers, "
                             f"instead has {len(radius)} elements.")
        poses = []
        while len(poses) < n:
            assert 2 == len(self._env_size), self._env_size
            pos = Point(self.env_space.np_random.random_sample(len(self._env_size)) * self._env_size)
            pos_buff = pos.buffer(radius[len(poses)])
            if unary_union(overlapping).disjoint(pos_buff) and self.is_valid_position(pos_buff):
                overlapping.append(pos_buff)
                poses.append(pos)
        assert n == len(poses) == len(radius)
        return poses

