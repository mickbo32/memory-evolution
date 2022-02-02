import math
from typing import Optional

import numpy as np
import gym
from gym import spaces


class MemoForagingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 height: int = 100,
                 width: Optional[int] = None,  # if None => square maze
                 n_channels: int = 1,
                 *,
                 # color_reversed: bool = False,  # if False => white=255, black=0; if True => white=0, black=255;
                 seed=None,
                 ) -> None:
        super().__init__()

        self._num_actions = 4  # left, straight, right, none
        self._height = height
        self._width = height if width is None else width
        self._n_channels = n_channels
        self._seed = seed
        # self._corridor_width = 10

        self.agent_position = None
        self.maze = None
        self.step_count = None

        self.action_space = spaces.Discrete(self._num_actions, seed=self._seed)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self._height, self._width, self._n_channels),
                                            dtype=np.uint8,
                                            seed=self._seed)
        self.env_space = spaces.Box(low=0, high=255,
                                    shape=(self._height, self._width, self._n_channels),
                                    dtype=np.uint8,
                                    seed=self._seed)

        self.reset()

    def step(self, action):
        # update environment state:
        pass

        # create an observation from the environment state:
        observation = self._get_observation()

        # compute reward:
        reward = 0  # 1 if action == 3 else -1

        # Is it done?:
        done = True

        # debugging info:
        info = {
            'maze': self.maze,
            'agent_position': self.agent_position,
            # 'current_step': self.step_count,
        }

        self.step_count += 1
        return observation, reward, done, info

    def reset(self):
        print('Reset')
        self.step_count = 0

        # init environment state:
        self._init_state()

        # create an observation from the environment state:
        observation = self._get_observation()

        assert self.observation_space.contains(observation)
        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def _get_observation(self):  # very costly; can be improved
        """create an observation from the environment state"""
        obs = self.maze.copy()
        # put the agent in the maze:
        obs[self.coord_to_idx(self.agent_position)] = 0
        return obs

    @staticmethod
    def coord_to_idx(pos) -> tuple[int]:
        if len(pos) != 2:
            raise ValueError('Position `pos` should be a 2D vector of coordinates in a 2D space.')
        return tuple(map(math.floor, (pos[1], pos[0])))

    @staticmethod
    def idx_to_coord(idx) -> np.ndarray:
        if 2 <= len(idx) <= 3:
            assert all(isinstance(i, int) for i in idx)
            # channels are ignored
            return np.asarray((idx[1], idx[0]), dtype=float) + .5  # place in the center of the cell
        else:
            raise ValueError('The index `idx` should refer to 2D space with N channels, '
                             'thus 2D or 3D vector (channels are ignored).')

    @classmethod
    def is_valid_agent_position(cls, maze, pos):
        # convert pos to index if it is not already:
        pos = pos if not isinstance(pos, np.ndarray) else cls.coord_to_idx(pos)
        assert isinstance(pos, tuple)
        try:
            print(pos)
            print(maze)

            print(cls.coord_to_idx(pos))
            print(maze[pos])
        except IndexError:
            return False
        return not np.array_equal(np.asarray([0]), maze[pos])

    def _get_new_maze(self):
        maze = np.ones(self.env_space.shape, dtype=self.env_space.dtype) * 255

        # bottom_left = (0, 0)
        # top_right = np.asarray((self.env_space.shape[1], 0))
        # bottom_right = np.asarray((self.env_space.shape[1], self.env_space.shape[0]))
        # top_left = np.asarray((0, self.env_space.shape[0]))
        # center = top_right // 2
        # print(top_right, bottom_right, top_left, center)
        #
        # walls = [  # sets of points representing polygons (points here are float, the pixel computation is done later)
        #     [bottom_left + self._corridor_width / np.sqrt(2),
        #      center - (self._corridor_width / 2),
        #      np.asarray((center[0] - (self._corridor_width / 2), 0))],
        # ]
        #
        # for wall in walls:
        #     pass  # todo

        assert self.env_space.contains(maze)
        return maze

    @staticmethod
    def valid_agent_idxes(maze, get_list=True, clear_cache=False, *, _cache={}):
        """Returns a list or a mask (np.ndarray) representing the valid agent
        positions (as indexes in the discrete space of the maze).
        Do not use _cache directly.
        It works only for static mazes (because it uses a cache to store the
        maze valid positions),
        if the maze changes use clear_cache=True each time."""
        if clear_cache:
            _cache.clear()
        maze_id = id(maze)
        if maze_id not in _cache:
            # print('cache miss')
            mask = (maze != np.asarray([0]))
            valid_idxes = [(i, j)
                           for i, row in enumerate(mask)
                           for j, val in enumerate(row)
                           if val]
            _cache[maze_id] = (mask, valid_idxes)
        # else:
        #     print('cache hit')
        return _cache[maze_id][1 if get_list else 0]

    def _init_state(self):
        """Create, update and return a new environment state (used for initialization or reset)"""

        # create maze:
        self.maze = self._get_new_maze()

        # init random agent position:
        j = self.env_space.np_random.randint(self.env_space.shape[1])
        i = self.env_space.np_random.randint(self.env_space.shape[0])
        idx = (i, j)
        assert self.is_valid_agent_position(self.maze, idx), idx
        self.agent_position = self.idx_to_coord(idx)
        print(self.agent_position)
        # self.idx_to_coord()? self.env_space.np_random.choice(self.valid_agent_idxes(self.maze, get_list=True))

        # put the agent in the maze:
        # you are doing it in _get_observation() virtually;

        # your state is self.maze + self.agent_position
