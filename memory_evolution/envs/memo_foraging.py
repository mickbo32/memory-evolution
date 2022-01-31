from typing import Optional

import numpy as np
import gym
from gym import spaces


class MemoForagingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 num_actions: int = 4,  # left, straight, right, none
                 height: int = 5,
                 width: Optional[int] = None,  # if None => square maze
                 n_channels: int = 1,
                 # *,
                 # color_reversed: bool = False,  # if False => white=255, black=0; if True => white=0, black=255;
                 ) -> None:
        super().__init__()

        self.num_actions = num_actions
        self.height = height
        self.width = height if width is None else width
        self.n_channels = n_channels

        self.state = None  # internal state of the environment
        self.step_count = None

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, self.n_channels),
                                            dtype=np.uint8)

        self.reset()

    def step(self, action):
        # update environment state:
        self.state

        # create an observation from the environment state:
        observation = self.state

        # compute reward:
        reward = 0  # 1 if action == 3 else -1

        # Is it done?:
        done = True

        # debugging info:
        info = {}

        self.step_count += 1
        return observation, reward, done, info

    def reset(self):
        self.step_count = 0

        # init environment state:
        # print(self.observation_space.shape)
        # print(self.observation_space.sample())
        # print(np.ones(self.observation_space.shape, dtype=self.observation_space.dtype) * 255)
        self.state = np.ones(self.observation_space.shape, dtype=self.observation_space.dtype) * 255

        # create an observation from the environment state:
        observation = self.state

        assert self.observation_space.contains(observation)
        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        pass
