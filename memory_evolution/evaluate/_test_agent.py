import os.path
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from collections.abc import Callable, Iterable, Sequence
import logging
import math
import multiprocessing
from numbers import Number, Real
from pprint import pprint
from typing import Optional, Union, Any, Literal
from warnings import warn
import sys
import tempfile
import time

import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import SeedSequence, default_rng
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPolygon
from shapely.ops import unary_union

import memory_evolution
from memory_evolution.envs import RadialArmMaze
from memory_evolution.geometry import is_simple_polygon
from . import evaluate_agent

# # For debugging:
# import geopandas as gpd


def _get_init_and_taget_arms_platform(env: RadialArmMaze):
    """Return a subset of the platform which contain only the initial and target arms.

    NOTE: env.agent must be in its initial position
    """
    # plgi.buffer(0).boundary.coords[:]
    assert is_simple_polygon(env.platform), (env.platform, type(env.platform))
    assert isinstance(env.platform.boundary, LineString)
    assert len(env.platform.interiors) == 0, env.platform.wkt
    boundary = list(env.platform.boundary.coords)
    assert boundary == list(env.platform.exterior.coords), (
        env.platform.boundary.coords, env.platform.exterior.coords)
    center = np.asarray(env.env_size) / 2
    center_point = Point(center)
    assert env.arms * 3 + 1 == len(boundary), (env.arms, len(boundary))
    assert boundary[0] == boundary[-1], boundary
    boundary = boundary[:-1]
    boundary = [Point(p) for p in boundary]  # otherwise they are tuples

    assert len(boundary) % 3 == 0, len(boundary)
    assert len(boundary) >= 6, len(boundary)
    corridor_end_point = boundary[0]
    inner_point = boundary[1]
    corridor_start_point = boundary[2]
    assert math.isclose(center_point.distance(corridor_end_point), center_point.distance(corridor_start_point))
    assert not math.isclose(center_point.distance(corridor_end_point), center_point.distance(inner_point))
    assert center_point.distance(corridor_end_point) > center_point.distance(inner_point)
    inner_dist = center_point.distance(inner_point)
    outer_dist = center_point.distance(corridor_end_point)

    # get all arms:
    inner_boundary = [boundary[arm * 3 + 1] for arm in range(env.arms)]
    assert len(inner_boundary) == env.arms
    assert all(math.isclose(inner_dist, center_point.distance(p)) for p in inner_boundary)
    arms = []
    for arm in range(env.arms):
        bdr = (
            boundary[arm * 3],
            *inner_boundary[arm:],
            *inner_boundary[:arm],
            boundary[(arm * 3 - 1) % len(boundary)],
        )
        assert len(bdr) == 2 + env.arms, len(bdr)
        plg = Polygon(bdr)
        arms.append(plg)

    # select initial and target arms
    agent_pos = Point(env.agent.pos)
    assert len(env.food_items) == 1, env.food_items
    food_positions = [Point(food.pos) for food in env.food_items]
    assert len(food_positions) == 1, (env.food_items, food_positions)
    target_pos = food_positions[0]
    _agent_arm_count = 0
    _target_arm_count = 0
    agent_arm = -1
    target_arm = -1
    for idx, arm in enumerate(arms):
        if arm.covers(agent_pos):
            _agent_arm_count += 1
            agent_arm = idx
        if arm.covers(target_pos):
            _target_arm_count += 1
            target_arm = idx
    # # plotting to DEBUG:
    # fig, ax = plt.subplots()
    # gpd.GeoSeries(env.platform.buffer(0)).plot(ax=ax, color='gray')
    # # gpd.GeoSeries([arm.boundary for arm in arms]).plot(ax=ax, color='r')
    # for _idx, _arm in enumerate(arms):
    #     gpd.GeoSeries(_arm.boundary).plot(ax=ax, color=(0, _idx / (len(arms) - 1), 1))
    # gpd.GeoSeries(inner_boundary).plot(ax=ax, color='g')
    # gpd.GeoSeries(center_point).plot(ax=ax, color='purple')
    # plt.show()
    if _agent_arm_count != 1:
        raise RuntimeError(f"agent on multiple arms, {_agent_arm_count} "
                           f"(agent could be in the center, not an arm initial position).")
    assert 0 <= agent_arm < env.arms, agent_arm
    if _target_arm_count != 1:
        raise RuntimeError(f"target on multiple arms, {_target_arm_count} "
                           f"(target must be one and positioned at the end of an arm).")
    assert 0 <= target_arm < env.arms, target_arm
    if agent_arm == target_arm:
        raise RuntimeError(f"agent and target should be on different arms,"
                           f" instead they are both in the #{agent_arm} arm.")
    # merge agent and target arms:
    reduced_platform = unary_union((arms[agent_arm], arms[target_arm]))
    assert is_simple_polygon(reduced_platform)

    # # plotting to DEBUG:
    # fig, ax = plt.subplots()
    # gpd.GeoSeries(env.platform.buffer(0)).plot(ax=ax, color='gray')
    # gpd.GeoSeries(reduced_platform.buffer(0)).plot(ax=ax, color='b')
    # # gpd.GeoSeries([arm.boundary for arm in arms]).plot(ax=ax, color='r')
    # for _idx, _arm in enumerate(arms):
    #     gpd.GeoSeries(_arm.boundary).plot(ax=ax, color=(0, _idx / (len(arms) - 1), 1))
    # gpd.GeoSeries(center_point).plot(ax=ax, color='purple')
    # plt.show()

    return reduced_platform


def test_agent_first_arm_accuracy(
        agent,
        env: RadialArmMaze,
        episodes: int = 100,
        max_actual_time_per_episode: Optional[Union[int, float]] = None,
        render: bool = False,
):
    """Evaluate the agent for ``episodes`` episodes in a RadialArmMaze
    and return the accuracy of exploring the correct arm at first.

    test_agent_first_arm_accuracy success is when
    the agent succeeded without visiting any wrong arm
    (the only arm visited is the target arm
    and the initial arm)

    A failure is when the agent explore any of the wrong arms or the time
    finish before the agent can succeed (if env.max_step is not None).
    A success is when the agent succeed without visiting any wrong arms
    (if env.max_step is not None, the agent succeed if it can
    reach the target before the time is finished).

    Note: env must be a RadialArmMaze with a single target in one arm.
    """
    logging.info("test_agent_first_arm_accuracy...")
    logging.log(logging.INFO - 1,
                "test_agent_first_arm_accuracy success is when"
                " the agent succeeded without visiting any wrong arm"
                " (the only arm visited is the target arm"
                " and the initial arm)")
    if not isinstance(env, RadialArmMaze):
        raise TypeError(f"'env' should be a RadialArmMaze, instead got {type(env)}")
    if env.n_food_items != 1:
        raise ValueError(f"'env.n_food_items' must be 1, instead is {env.n_food_items}")

    rendering_mode = 'observation+human'
    successes = 0
    for i_episode in range(episodes):
        msg = f'testing - Starting episode #{i_episode} ...'
        logging.debug(msg)
        start_time_episode_including_reset = time.perf_counter_ns()
        # Reset env and agent:
        observation, info = env.reset(return_info=True)
        agent.reset()
        logging.log(logging.DEBUG + 8, f"env.agent.pos: {env.agent.pos}; "
                                       f"env.agent.head_direction: {env.agent.head_direction};")
        assert env.t == 0., env.t
        agent_positions = []
        agent_positions.append(env.agent.pos)
        valid_positions_platform = _get_init_and_taget_arms_platform(env)
        start_time_episode = time.perf_counter_ns()
        reset_actual_time = (start_time_episode - start_time_episode_including_reset) / 10 ** 9
        msg = f"testing - Episode reset took {reset_actual_time} actual seconds."
        logging.log(logging.DEBUG + 3, '\n\t' + msg)
        total_reward = 0.0
        if render:
            env.render(mode=rendering_mode)
        step = 0
        done = False
        while not done and (max_actual_time_per_episode is None
                            or (time.perf_counter_ns() - start_time_episode) < max_actual_time_per_episode * 10 ** 9):
            # Agent performs an action based on the current observation (and
            # its internal state, i.e. memory):
            assert env.step_count == step, (env.step_count, step)
            action = agent.action(observation)
            assert env.step_count == step, (env.step_count, step)
            observation, reward, done, info = env.step(action)
            assert env.step_count == step + 1, (env.step_count, step)
            agent_positions.append(env.agent.pos)
            total_reward += reward
            if render:
                logging.debug(f"testing - Observation hash: {hash(observation.tobytes())}")
                logging.debug(f"testing - Action hash: {hash(action.tobytes())}")
                env.render(mode=rendering_mode)
            step += 1
        end_t = env.t
        end_time_episode = time.perf_counter_ns()
        assert env.food_items_collected == total_reward, (env.food_items_collected, total_reward)
        assert total_reward == int(total_reward), (total_reward, int(total_reward))  # total_reward should be an integer
        if env.food_items_collected == env.n_food_items:
            assert total_reward == env.food_items_collected == env.n_food_items == env.maximum_reward
            # note: the agent could take the last food item in the last timestep
            assert 0 <= step == env.step_count <= env.max_steps
        else:
            assert step == env.step_count == env.max_steps
            assert 0 <= env.food_items_collected < env.n_food_items
        # testing asserts:
        assert len(agent_positions) == step + 1, (len(agent_positions), step)  # todo: optimize: if visit another arm stop
        # was the target reached?:
        if env.food_items_collected == env.n_food_items:
            assert total_reward == env.maximum_reward, (total_reward, env.maximum_reward)
            # was the target reached without visiting any wrong arms?:
            wrong_arm_visited = False
            for pos in agent_positions:
                p = Point(pos)
                assert env.platform.covers(p)
                if valid_positions_platform.disjoint(p):
                    wrong_arm_visited = True
                    break
            if not wrong_arm_visited:
                successes += 1
                logging.log(logging.INFO - 1, "Success +1")
            else:
                logging.log(logging.INFO - 1, "Fail")
        else:
            assert total_reward < env.maximum_reward, (total_reward, env.maximum_reward)
            logging.log(logging.INFO - 1, "Fail")

        if done:
            if isinstance(env, memory_evolution.envs.BaseForagingEnv):
                assert step == env.step_count == info['debug_info']['_is_done']['step_count'], (
                    step, env.step_count, info['debug_info']['_is_done']['step_count'])
                assert info['debug_info']['_is_done']['done'] is True, info['debug_info']['_is_done']
                assert (info['debug_info']['_is_done']['food_items_collected'] == info['debug_info']['_is_done']['n_food_items']
                        or env.step_count == info['debug_info']['_is_done']['max_steps']), info['debug_info']['_is_done']

            actual_time = (end_time_episode - start_time_episode) / 10 ** 9
            msg = (
                f"testing - "
                # f"{agent} fitness {fitness}\n"
                f"{agent}\n"
                f"Episode loop finished after {step} timesteps"
                f", for a total of {end_t} simulated seconds"
                f" (in {actual_time} actual seconds)."
            )
            logging.log(logging.DEBUG + 4, '\n\t' + '\n\t'.join(msg.split('\n')))
        else:
            if isinstance(env, memory_evolution.envs.BaseForagingEnv):
                assert step == env.step_count == info['debug_info']['_is_done']['step_count'], (
                    step, env.step_count, info['debug_info']['_is_done']['step_count'])
                assert info['debug_info']['_is_done']['done'] is False, info['debug_info']['_is_done']
                assert not (info['debug_info']['_is_done']['food_items_collected'] == info['debug_info']['_is_done']['n_food_items']
                            or env.step_count == info['debug_info']['_is_done']['max_steps']), info['debug_info']['_is_done']
            raise RuntimeError(
                f"Episode has not finished after {step} timesteps"
                f" and {end_t} simulated seconds"
                f" (in {(end_time_episode - start_time_episode) / 10 ** 9} actual seconds).")
    accuracy = successes / episodes
    assert 0. <= accuracy <= 1., (accuracy, successes, episodes)
    logging.info(f"test_agent_first_arm_accuracy (episodes={episodes}): {accuracy}")
    return accuracy


def fitness_func_target_reached(*, reward, steps, done, env, agent, **kwargs) -> float:
    """Returns 1.0 if target reached, 0.0 if not."""
    assert done
    return float(env.food_items_collected == env.n_food_items)
fitness_func_target_reached.min = 0.
fitness_func_target_reached.max = 1.


def test_agent_target_reached_rate(
        agent,
        env: RadialArmMaze,
        episodes: int = 100,
        max_actual_time_per_episode: Optional[Union[int, float]] = None,
        render: bool = False,
):
    """Evaluate the agent for ``episodes`` episodes in a RadialArmMaze
    and return the rate of successful agent trails reaching eventually
    the target in time.
    
    # together with test_agent_first_arm_accuracy, can be used to discriminate
    # bad v.s. border-follower v.s. allocentric/egocentric successful agents

    Note: env must be a RadialArmMaze with a single target in one arm.
    """
    logging.info("test_agent_target_reached_rate...")
    if not isinstance(env, RadialArmMaze):
        raise TypeError(f"'env' should be a RadialArmMaze, instead got {type(env)}")
    if env.n_food_items != 1:
        raise ValueError(f"'env.n_food_items' must be 1, instead is {env.n_food_items}")

    successes = evaluate_agent(
        agent=agent,
        env=env,
        episodes=episodes,
        episodes_aggr_func=None,
        fitness_func=fitness_func_target_reached,
        max_actual_time_per_episode=max_actual_time_per_episode,
        render=render,
        save_gif=False,
    )
    assert len(successes) == episodes, (len(successes), episodes)
    assert ((successes == 0.) | (successes == 1.)).all(None), successes
    n_successes = successes.sum(None)
    assert 0 <= n_successes <= episodes, (n_successes, episodes)

    success_rate = n_successes / episodes
    assert 0. <= success_rate <= 1., (success_rate, successes, episodes)
    logging.info(f"test_agent_target_reached_rate (episodes={episodes}): {success_rate}")
    return success_rate

