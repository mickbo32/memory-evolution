from collections import Counter
from collections.abc import Collection
import unittest

import numpy as np
from numpy.random import SeedSequence
from numpy.testing import assert_array_equal
import gym
from gym import spaces

from memory_evolution.envs import *
from memory_evolution.envs.base_foraging import *

"""
TODO:
    * Use different seeds and test log the seed.
    * assert type(Point(0, 0).buffer(1)) is Polygon, type(Point(0, 0).buffer(1))
"""


class TestEnvConversions(unittest.TestCase):

    def test_point_conversion_env2win(self):
        try:
            env = BaseForagingEnv(window_size=200, env_size=(1., 1.111), agent_size=.05)
            # print(env.window_size)  # (200, 223)
            assert (200, 223) == env.window_size, env.window_size
            env = BaseForagingEnv(window_size=(200, 223), env_size=(1., 1.111), agent_size=.05)
            # print(env.window_size)  # (200, 223)
        except ValueError as err:
            if str(err).startswith("'window_size' and 'env_size'"):
                self.fail(f"'win_size_x = 200' and 'env_size = (1., 1.111)' shouldn't raise the scaling exception."
                          f" (scaling factors should be fine: 200.0 and 200.7200720072007)")
            else:
                raise
        try:
            env = BaseForagingEnv(window_size=200, env_size=(1.111, 1.), agent_size=.05)
            # print(env.window_size)  # (200, 181)
            assert (200, 181) == env.window_size, env.window_size
            env = BaseForagingEnv(window_size=(200, 181), env_size=(1.111, 1.), agent_size=.05)
            # print(env.window_size)  # (200, 181)
            # print([get_env2win_scaling_factor(env.window_size, env.env_size, axis=i)
            #        for i in range(len(env.window_size))])
        except ValueError as err:
            if str(err).startswith("'window_size' and 'env_size'"):
                self.fail(f"'win_size_x = 200' and 'env_size = (1.111, 1.)' shouldn't raise the scaling exception."
                          f" (scaling factors should be fine: 180.01800180018003 and 181.0)")
            else:
                raise
        try:
            env = BaseForagingEnv(window_size=200, env_size=(1.111, 1.111), agent_size=.05)
            # print(env.window_size)  # (200, 200)
            assert (200, 200) == env.window_size, env.window_size
            env = BaseForagingEnv(window_size=(200, 200), env_size=(1.111, 1.111), agent_size=.05)
            # print(env.window_size)  # (200, 200)
            # print([get_env2win_scaling_factor(env.window_size, env.env_size, axis=i)
            #        for i in range(len(env.window_size))])
        except ValueError as err:
            if str(err).startswith("'window_size' and 'env_size'"):
                self.fail(f"'win_size_x = 200' and 'env_size = (1.111, 1.111)' shouldn't raise the scaling exception."
                          f" (scaling factors should be fine: 180.01800180018003 and 180.01800180018003)")
            else:
                raise
        with self.assertRaises(ValueError) as cm:
            env = BaseForagingEnv(window_size=(200, 222), env_size=(1., 1.111), agent_size=.05)
        err = cm.exception
        self.assertTrue(str(err).startswith("'window_size' and 'env_size'"))
        with self.assertRaises(ValueError) as cm:
            env = BaseForagingEnv(window_size=(200, 180), env_size=(1., 1.111), agent_size=.05)
        err = cm.exception
        self.assertTrue(str(err).startswith("'window_size' and 'env_size'"))
        with self.assertRaises(ValueError) as cm:
            env = BaseForagingEnv(window_size=(200, 199), env_size=(1., 1.111), agent_size=.05)
        err = cm.exception
        self.assertTrue(str(err).startswith("'window_size' and 'env_size'"))

        env = BaseForagingEnv(window_size=320, env_size=(1.5, 1.), agent_size=.05)
        self._test_point_conversion_env2win(env)
        env = BaseForagingEnv(window_size=200, env_size=(1.111, 1.111), agent_size=.05)
        self._test_point_conversion_env2win(env)
        env = BaseForagingEnv(window_size=127, env_size=(1.5, 1.), agent_size=.05)
        self._test_point_conversion_env2win(env)
        env = BaseForagingEnv(window_size=10, env_size=(1.5, 1.), agent_size=.05)
        self._test_point_conversion_env2win(env)
        env = BaseForagingEnv(window_size=9, env_size=(1.5, 1.), agent_size=.05)
        self._test_point_conversion_env2win(env)

    def _test_point_conversion_env2win(self, env):
        agent_size = env._agent_size
        agent_radius = agent_size / 2
        env_size = env.env_size
        window_size = env.window_size
        print("window_size, env_size:\t", window_size, env_size)
        dtype = env.action_space.dtype
        assert dtype.kind == 'f', dtype
        epsilon = np.finfo(dtype).resolution * 3
        assert 1. + epsilon != 1., f'{epsilon:.30f}, {1. + epsilon:.30f}, {dtype}'
        # # >> show sequence of conversions >>
        # env_pos = (env_size[0] / 2, env_size[1] / 2)
        # _n = 200
        # for i in range(_n):
        #     print("i:", i, "env_pos:", env_pos, sep='\t')
        #     window_pos = get_point_env2win(env_pos, window_size, env_size)
        #     print("i:", i, "window_pos:", window_pos, sep='\t')
        #     env_pos = get_point_win2env(window_pos, window_size, env_size)
        # print("i:", _n, "env_pos:", env_pos, sep='\t')
        # # << show sequence of conversions <<
        def test_raises(env_pos, msg=''):
            assert not env.is_valid_position(env_pos, 'agent'), (env_pos, agent_radius, msg)
            with self.assertRaises(ValueError, msg=(env_pos, agent_radius, msg)) as cm:
                window_pos = get_point_env2win(env_pos, window_size, env_size)
        def test_not_valid(env_pos, msg=''):
            assert not env.is_valid_position(env_pos, 'agent'), (
                env_pos, agent_radius, msg,
                get_point_env2win(env_pos, window_size, env_size),
            )
            window_pos = get_point_env2win(env_pos, window_size, env_size)
            assert not env.is_valid_position(window_pos, 'agent', is_env_pos=False), (
                env_pos, window_pos, agent_radius,
                agent_radius * get_env2win_scaling_factor(window_size, env_size), msg)
            env_pos = get_point_win2env(window_pos, window_size, env_size)
            assert not env.is_valid_position(env_pos, 'agent'), (env_pos, window_pos, agent_radius, msg)
            self.assertEqual(window_pos, get_point_env2win(env_pos, window_size, env_size), msg=msg)
        def test_valid(env_pos, msg=''):
            assert env.is_valid_position(env_pos, 'agent'), (
                env_pos, agent_radius, msg,
                get_point_env2win(env_pos, window_size, env_size),
                env._valid_agent_positions.get_at(get_point_env2win(env_pos, window_size, env_size)),
                # print('\n' + '\n'.join(
                #     ''.join(f"{_x:d}" for _x in row)
                #     for row in memory_evolution.utils.convert_pg_mask_to_array(env._valid_agent_positions)),
                #     file=sys.stderr)
            )
            window_pos = get_point_env2win(env_pos, window_size, env_size)
            assert env.is_valid_position(window_pos, 'agent', is_env_pos=False), (
                env_pos, window_pos, agent_radius,
                agent_radius * get_env2win_scaling_factor(window_size, env_size), msg)
            env_pos = get_point_win2env(window_pos, window_size, env_size)
            assert env.is_valid_position(env_pos, 'agent'), (env_pos, window_pos, agent_radius, msg)
            self.assertEqual(window_pos, get_point_env2win(env_pos, window_size, env_size), msg=msg)
        fct_x = window_size[0] / env_size[0]
        fct_y = window_size[1] / env_size[1]
        check_scaling_factor_across_axes(window_size, env_size)
        print("scaling factors: ", fct_x, fct_y)
        outside_boundaries = (math.ceil(agent_radius * fct_x - .5) / fct_x,
                              math.ceil(agent_radius * fct_y - .5) / fct_y,
                              math.floor((env_size[0] - agent_radius) * fct_x + .5) / fct_x,
                              math.floor((env_size[1] - agent_radius) * fct_y + .5) / fct_y)
        print("agent radius (axis=0): ", agent_radius, agent_radius * fct_x,
              outside_boundaries[0], outside_boundaries[2])
        print("agent radius (axis=1): ", agent_radius, agent_radius * fct_y,
              outside_boundaries[1], outside_boundaries[3])
        for x in np.linspace(0., env_size[0], window_size[0] * 5, endpoint=True):
            for y in np.linspace(0., env_size[1], window_size[1] * 5, endpoint=True):
                env_pos = (x, y)
                # valid if centroid of pixel is valid
                outside = (env_pos[0] < outside_boundaries[0],
                           env_pos[1] < outside_boundaries[1],
                           env_pos[0] > outside_boundaries[2],
                           env_pos[1] > outside_boundaries[3])
                # if it is exactly on the env border it is okay
                borders = (env_pos[0] == agent_radius,
                           env_pos[1] == agent_radius,
                           env_pos[0] == env_size[0] - agent_radius,
                           env_pos[1] == env_size[1] - agent_radius)
                # if it is exactly on the env border it is okay
                if not all(not b for o, b in zip(outside, borders) if o):
                    raise RuntimeError('test has a bug')
                if any(outside):
                    test_not_valid(env_pos, msg=''.join(f"{x:d}" for x in outside))
                else:
                    test_valid(env_pos, msg=''.join(f"{x:d}" for x in outside))
        # extra_test_valid_positions = [
        #     (agent_radius, agent_radius),
        #     (env_size[0] - agent_radius, env_size[1] - agent_radius),
        #     (agent_radius, env_size[1] - agent_radius),
        #     (env_size[0] - agent_radius, agent_radius),
        # ]
        # for env_pos in extra_test_valid_positions:
        #     test_valid(env_pos)
        if 1 / window_size[0] < agent_radius / env_size[0] and 1 / window_size[1] < agent_radius / env_size[1]:
            extra_test_not_valid_positions = [
                (0, 0), env_size, (0, env_size[1]), (env_size[0], 0),
                # (env_size[0] - agent_radius, (env_size[1] - agent_radius) * (1. + epsilon)),
                # ((env_size[0] - agent_radius) * (1. + epsilon), env_size[1] - agent_radius),
            ]
            for env_pos in extra_test_not_valid_positions:
                test_not_valid(env_pos)
        else:
            print('window_size too small')
            warn('window_size too small')
        extra_test_raises = [
            (-1, 0), (0, -1),
            (env_size[0] * (1. + epsilon), env_size[1]),
            (env_size[0], env_size[1] * (1. + epsilon)),
        ]
        for env_pos in extra_test_raises:
            test_raises(env_pos)
        print()


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
