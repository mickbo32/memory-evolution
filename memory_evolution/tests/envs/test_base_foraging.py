from collections import Counter
from collections.abc import Collection
import unittest

import numpy as np
from numpy.random import SeedSequence
from numpy.testing import assert_array_equal
import gym
from gym import spaces

from memory_evolution.envs import *

"""
TODO:
    * Use different seeds and test log the seed.
    * assert type(Point(0, 0).buffer(1)) is Polygon, type(Point(0, 0).buffer(1))
"""


class TestMemoForagingEnv(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = BaseForagingEnv()
        cls.env2 = BaseForagingEnv(5, 3, seed=2002)

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.env, cls.env2

    def test_class_constructor(self):
        env = BaseForagingEnv()
        env = BaseForagingEnv(3, 5, seed=2002)
        env = BaseForagingEnv(50, seed=2002)
        env = BaseForagingEnv(50, 100, seed=2002)
        env = BaseForagingEnv(n_channels=3, seed=2002)

    def test_valid_agent_idxes(self):
        maze = BaseForagingEnv(3, 5, seed=2002)._get_new_maze()
        maze[0, 0, 0] = 0
        maze[0, 1, 0] = 200

        for options in [{'clear_cache': False},
                        {'clear_cache': True},
                        {'clear_cache': True},
                        {'clear_cache': False},
                        {'clear_cache': False}, ]:
            mask = BaseForagingEnv.valid_agent_idxes(maze, get_list=False, clear_cache=options['clear_cache'])
            true_mask = np.asarray([[0, 1, 1, 1, 1], [1]*5, [1]*5], dtype=bool)[:, :, None]
            self.assertEqual(true_mask.dtype, mask.dtype)
            assert_array_equal(true_mask, mask)

            valid_idxes = BaseForagingEnv.valid_agent_idxes(maze, get_list=True, clear_cache=options['clear_cache'])
            true_valid_idxes = [(i, j) for i in range(maze.shape[0]) for j in range(maze.shape[1])]
            true_valid_idxes.remove((0, 0))
            # for pos in valid_positions:
            #     self.assertIsInstance(idx, np.ndarray)
            #     self.assertEqual(float, valid_idxes.dtype)
            for idx in valid_idxes:
                self.assertIsInstance(idx, tuple)
                self.assertEqual(2, len(idx))
            # valid_idxes = [tuple(idx) for idx in valid_idxes]
            self.assertCountEqual(true_valid_idxes, valid_idxes)  # order can be different (lists can have duplicates)
            c = Counter(valid_idxes)
            tc = Counter(true_valid_idxes)
            self.assertEqual(tc, c, msg=(f'lists have different elements\n'
                                         f'\t-> elements not present that should be: {tc - c}\n'
                                         f'\t-> extra elements that shouldn\'t be present: {c-tc}'))
            d = {k: v for k, v in c.items() if v != 1}
            self.assertFalse(d, msg=f'list contains duplicates: {d}')
            self.assertEqual(len(set(valid_idxes)), len(valid_idxes), msg=f'list contains duplicates')


if __name__ == '__main__':
    unittest.main()
