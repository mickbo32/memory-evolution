from collections import defaultdict, Counter
import math
from numbers import Number, Real
from typing import Optional, Union, Any
from warnings import warn

import gym
from gym import spaces
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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


class Agent:
    """Agent"""

    def __init__(self, pos: Pos, head_direction: bool = False):
        self.pos = pos
        self._has_head_direction = head_direction
        self._head_direction = None
        if head_direction and self.pos.ndim != 2:
            raise NotADirectoryError("`head_direction` not implemented yet for spaces different from 2D space")

    @property
    def head_direction(self):
        if not self._has_head_direction:
            raise AttributeError(f"Agent object at {hex(id(self))} doesn't have head direction.")
        return self._head_direction

    @head_direction.setter
    def head_direction(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("`head_direction` must be int or float")
        if not (0 <= value < 360):
            raise ValueError("`head_direction` must be in the [0,360) range.")
        self._head_direction = value

    @property
    def has_head_direction(self):
        return self._has_head_direction

    def __str__(self):
        return f"{type(self).__name__}({self.pos})"

    def __repr__(self):
        return f"{__name__ if __name__ != '__main__' else ''}.{type(self).__qualname__}({self.pos!r})"


class FoodItem:
    """Rewarding (maybe, not actual reward, but increase in agent life span) food items."""

    def __init__(self, pos: Pos):
        self.pos = pos

    def __str__(self):
        return f"{type(self).__name__}({self.pos})"

    def __repr__(self):
        return f"{__name__ if __name__ != '__main__' else ''}.{type(self).__qualname__}({self.pos!r})"


class BaseForagingEnv(gym.Env, MustOverride):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 height: int = 100,
                 width: Optional[int] = None,  # if None => square maze
                 n_food_items: int = 3,
                 *,
                 n_channels: int = 1,
                 # color_reversed: bool = False,  # if False => white=255, black=0; if True => white=0, black=255;
                 seed=None,
                 head_direction: bool = True,
                 ) -> None:
        super().__init__()

        if head_direction:
            self._num_actions = 4  # rotate left, go straight, rotate right, none
        else:
            self._num_actions = 5  # left, up, right, down, none
        self._height = height
        self._width = height if width is None else width
        self._n_channels = n_channels
        self._n_food_items = n_food_items
        self._seed = seed
        self.agent_color = np.asarray([0])
        self.food_color = np.asarray([0])
        # todo: background color(, borders color)
        self._figure, self._ax = plt.subplots()
        self.rendering_pause_interval = .1  # 1e-20
        self.debug_info = defaultdict(dict)

        # head direction
        self._head_direction_mode = head_direction
        if head_direction:
            self._head_rotation_angle = 90

        self.step_count = None
        self._agent = None
        self._food_items = None
        self.__food_items_poses = set()
        self._state_img = None  # todo: make state (env_state) an object

        self.action_space = spaces.Discrete(self._num_actions, seed=self._seed)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self._height, self._width, self._n_channels),
                                            dtype=np.uint8,
                                            seed=self._seed)
        self.env_space = spaces.Box(low=0, high=255,
                                    shape=(self._height, self._width, self._n_channels),
                                    dtype=np.uint8,
                                    seed=self._seed)

        self._soil = np.ones(self.env_space.shape, dtype=self.env_space.dtype) * 255
        assert self.env_space.contains(self._soil)
        # self.__observation = None  # todo

        self.__has_been_ever_reset = False

    def step(self, action) -> tuple[np.ndarray, Real, bool, dict]:

        if not self.__has_been_ever_reset:
            # warn('Calling step() method before reset() method. Forcing reset() method...')
            self.reset()

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
        return observation

    def render(self, mode='human'):
        # print(self._state_img)
        self._ax.cla()
        self._ax.matshow(self._state_img, cmap='gray')
        plt.pause(self.rendering_pause_interval)

    def close(self):
        self._ax.cla()
        self._figure.clf()
        plt.close(fig=self._figure)

    @override
    def _init_state(self):
        """Create and return a new environment state (used for initialization or reset)"""

        # init environment space:
        self._state_img = self._soil.copy()
        assert self.env_space.contains(self._state_img)

        # init agent in a random position:
        idx = self.__get_random_non_overlapping_idx_position()
        pos = self.idx_to_coord(idx)
        self._agent = Agent(pos, head_direction=self._head_direction_mode)
        self._state_img[idx] = self.agent_color
        if self._head_direction_mode:
            self._agent.head_direction = 90

        # init food items:
        self._food_items = []
        for idx in self.__get_random_non_overlapping_idx_position(
                    self._n_food_items,
                    {self.coord_to_idx(self._agent.pos)}
                ):
            self._food_items.append(FoodItem(self.idx_to_coord(idx)))
            self._state_img[idx] = self.food_color
        self.__food_items_poses = {food.pos for food in self._food_items}
        assert len(self._food_items) == len(self.__food_items_poses)
        assert set(food.pos for food in self._food_items) == self.__food_items_poses
        assert self._agent.pos not in self.__food_items_poses

        # self.debug_info['_init_state']

    def _update_state(self, action):
        """Update environment state. Compute and return reward."""
        assert self.action_space.contains(action)

        # update agent position:
        action_on_map = None
        if self._head_direction_mode:
            assert self._agent.has_head_direction
            if action == 0:  # rotate left
                self._agent.head_direction = (self._agent.head_direction + self._head_rotation_angle) % 360
            elif action == 1:  # go straight
                action_on_map = {
                    0: 2,
                    90: 1,
                    180: 0,
                    270: 3,
                }[self._agent.head_direction]
            elif action == 2:  # rotate right
                self._agent.head_direction = (self._agent.head_direction - self._head_rotation_angle) % 360
            elif action == 3:  # none
                pass
        else:
            action_on_map = action

        if action_on_map is not None:
            pos = self._agent.pos
            if action_on_map == 0 and 0 <= pos.x - 1 < self.env_space.shape[1]:  # left
                self._agent.pos = Pos(pos.x - 1, pos.y)
            elif action_on_map == 1 and 0 <= pos.y + 1 < self.env_space.shape[0]:  # up
                self._agent.pos = Pos(pos.x, pos.y + 1)
            elif action_on_map == 2 and 0 <= pos.x + 1 < self.env_space.shape[1]:  # right
                self._agent.pos = Pos(pos.x + 1, pos.y)
            elif action_on_map == 3 and 0 <= pos.y - 1 < self.env_space.shape[0]:  # down
                self._agent.pos = Pos(pos.x, pos.y - 1)
            elif action_on_map == 4:  # none
                pass
            assert self.is_valid_position(self._agent.pos), (self._agent.pos, pos, self.env_space.shape)

            # update _state_img: pos:
            pre_idx = self.coord_to_idx(pos)
            new_idx = self.coord_to_idx(self._agent.pos)
            self._state_img[pre_idx] = self._soil[pre_idx]
            self._state_img[new_idx] = self.agent_color

            # food collected?
            if self._agent.pos in self.__food_items_poses:
                self.__food_items_poses.remove(self._agent.pos)
                for i, food in enumerate(self._food_items):
                    if food.pos == self._agent.pos:
                        break
                else:
                    raise AssertionError('self._agent.pos in self.__food_items_poses but not in self._food_items')
                self._food_items.pop(i)
                assert len(self._food_items) == len(self.__food_items_poses)
                print('Food collected')

        # update _state_img: rotate:
        # todo

        # compute reward:
        reward = 0

        # self.debug_info['_update_state']
        return reward

    def _get_observation(self):  # todo: very costly; can be improved
        """create an observation from the environment state"""
        obs = self._soil.copy()
        # put the agent in the maze:
        obs[self.coord_to_idx(self._agent.pos)] = 0
        # put the food items in the maze:
        for food in self._food_items:
            obs[self.coord_to_idx(food.pos)] = 0
        self.debug_info['_get_observation']['obs'] = obs
        return obs

    def _is_done(self):
        self.debug_info['_is_done'] = {}
        return self.step_count >= 40 - 1

    def _get_info(self):
        """Get debugging info (the environment state, plus some extra useful information)."""
        info = {
            'state': {
                'env_shape': self.env_space.shape,
                'agent_position': self._agent,
                'food_items': self._food_items,
            },
            'current_step': self.step_count,
            'debug_info': self.debug_info,
        }
        if self._head_direction_mode:
            info['state']['agent_head_direction'] = self._agent.head_direction
        return info

    @staticmethod
    def coord_to_idx(pos: Pos) -> tuple[int]:  # Collection[float] / Union[np.ndarray, tuple[float], list[float]]
        """`pos` should be all float (coordinate), index should be all int (index)"""
        if not isinstance(pos, Pos):
            raise TypeError('`pos` should be a Pos object')
        if pos.ndim != 2:
            raise ValueError('Position `pos` should be a 2D vector of coordinates in a 2D space.')
        return tuple(map(round, (pos[1], pos[0])))

    @staticmethod
    def idx_to_coord(idx: tuple[int]) -> Pos:
        """`pos` should be all float (coordinate), idx should be all int (index)"""
        if 2 <= len(idx) <= 3:
            if not all(isinstance(i, int) for i in idx):
                raise TypeError('`idx` type must be tuple[int]')
            # channels are ignored
            return Pos(*map(float, (idx[1], idx[0])))
        else:
            raise ValueError('The index `idx` should refer to 2D space with N channels, '
                             'thus 2D or 3D vector (channels are ignored).')

    def is_valid_position(self, pos: Union[Pos, tuple[int]]) -> bool:
        """Returns True if pos is a valid position.

        Args:
            pos: a position (Pos) or an index (tuple[int]).

        Returns:
            True if pos is a valid position.
        """
        # convert pos to index if it is not already:
        if isinstance(pos, Pos):
            pos = self.coord_to_idx(pos)
        elif not (isinstance(pos, tuple) and all(isinstance(i, int) for i in pos)):
            raise TypeError('`pos` should be an instance of Pos or a tuple of int (an index)')
        if len(pos) != 2:  # n.b.: do not consider channels for position
            raise ValueError('`pos` should be 2D (and without channels)')
        idx = pos
        return all(0 <= idx[d] < self.env_space.shape[d] for d in range(len(self.env_space.shape) - 1))

    def __get_random_non_overlapping_idx_position(self,
                                                  n: int = 1,
                                                  overlapping: set = None
                                                  ) -> Union[int, list]:
        if overlapping is None:
            overlapping = set()
        idxes = []
        while len(idxes) < n:
            idx = []
            for d in range(len(self.env_space.shape) - 1):  # do not consider channels for position
                idx.append(self.env_space.np_random.randint(self.env_space.shape[d]))
            idx = tuple(idx)
            assert self.is_valid_position(idx), idx
            if idx not in overlapping:
                overlapping.add(idx)
                idxes.append(idx)
        return idxes[0] if len(idxes) == 1 else idxes

