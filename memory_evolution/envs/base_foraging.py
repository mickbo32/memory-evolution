from abc import ABCMeta, abstractmethod
from collections import defaultdict, Counter
import math
from numbers import Number, Real
from typing import Optional, Union, Any
from warnings import warn

import numpy as np
import gym
from gym import spaces


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
    # todo: make it immutable

    def __init__(self, *coords):
        # if ndim <= 0:
        #     raise ValueError('`ndim` must be positive.')
        # if ndim != len(coords):
        #     raise TypeError(f'`ndim`={ndim} coords are expected, {len(coords)} coords was given (as arguments)')
        assert isinstance(coords, tuple)
        if not all(isinstance(c, float) for c in coords):
            raise TypeError('`coords` coordinates should be all float')
        self._coords = coords

    @property
    def ndim(self):
        return len(self._coords)

    def __len__(self):
        return len(self._coords)

    def __iter__(self):
        return iter(self._coords)

    def __getitem__(self, item):
        return self._coords[item]

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

    def __str__(self):
        return (f"{type(self).__name__}("
                + ", ".join([str(c) for c in self._coords]) + ")")

    def __repr__(self):
        return (f"{__name__ if __name__ != '__main__' else ''}.{type(self).__qualname__}("
                + ", ".join([str(c) for c in self._coords]) + ")")


class Agent:
    """Agent"""

    def __init__(self, pos: Pos):
        self.pos = pos

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


class BaseForagingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 height: int = 100,
                 width: Optional[int] = None,  # if None => square maze
                 n_channels: int = 1,
                 n_food_items: int = 3,
                 *,
                 # color_reversed: bool = False,  # if False => white=255, black=0; if True => white=0, black=255;
                 seed=None,
                 ) -> None:
        super().__init__()

        self._num_actions = 4  # left, straight, right, none
        self._height = height
        self._width = height if width is None else width
        self._n_channels = n_channels
        self._n_food_items = n_food_items
        self._seed = seed
        self.debug_info = defaultdict(dict)

        self.step_count = None
        self._agent = None
        self._food_items = None

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
        pass  # todo

    def close(self):
        pass

    # todo: @override
    def _init_state(self):
        """Create and return a new environment state (used for initialization or reset)"""

        # init agent in a random position:
        idx = self.__get_random_non_overlapping_idx_position()
        pos = self.idx_to_coord(idx)
        self._agent = Agent(pos)

        # init food items:
        self._food_items = [
            FoodItem(self.idx_to_coord(idx))
            for idx in self.__get_random_non_overlapping_idx_position(
                self._n_food_items,
                {self.coord_to_idx(self._agent.pos)}
            )
        ]

        # self.debug_info['_init_state']

    def _update_state(self, action):
        """Update environment state. Compute and return reward."""
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
        return False

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

