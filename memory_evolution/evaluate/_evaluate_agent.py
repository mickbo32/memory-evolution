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

import memory_evolution
from ._fitnesses import fitness_func_total_reward


def evaluate_agent(agent,
                   env: gym.Env,
                   episodes: int = 1,
                   episodes_aggr_func: Literal['min', 'max', 'mean', 'median'] = 'min',
                   fitness_func: Callable[..., float] = fitness_func_total_reward,
                   max_actual_time_per_episode: Optional[Union[int, float]] = None,
                   render: bool = False,
                   save_gif: bool = False,
                   save_gif_dir: str = None,
                   save_gif_name: str = "frames.gif",
                   ) -> float:
    """Evaluate agent in a gym environment and return the fitness of the agent.

    Args:
        agent: the agent, an object with an ``action()`` method which takes the
            observation as argument and returns a valid action and with a
            ``reset()`` method which reset the agent to an initial
            state ``t==0``.
        env: a valid gym environment object (it can be an object of a
            subclass of gym.Evn).
        episodes: number of independent episodes that will be run.
        episodes_aggr_func: function to use to aggregate all the fitness
            values collected for each single independent episode (default is
            'min': The genome's fitness is its worst performance across all runs).
        fitness_func: callable function to compute the fitness of a single episode,
            this function is called at the end of each episode, it takes **kwargs
            as arguments, and it should return a float, the fitness of the episode;
            by default kwargs contains:
            reward (total reward), steps (timesteps), done, env.
        max_actual_time_per_episode: if ``None`` it will just run until the
            environment is done, otherwise after ``max_actual_time_per_episode``
            seconds the environment will be closed and an error is raised.
            This time do not include the episode reset time, but only the
            episode loop (i.e. steps) time.
        render: if True, render the environment while evaluating the agent.
        save_gif: if True, save a gif for each episode showing the agent
            in action; if True also the ``save_gif_dir`` argument should
            be provided, ``save_gif_name`` is optional. Note: it is not safe
            to use this when performing computation in parallel.  # (to prevent this base_foraging raise an error if the frame directory already exists).
        save_gif_dir: if ``save_gif`` is True, this is the path where frames
            and the gif will be saved; the directory should not exist and
            it will be created; if ``save_gif_dir`` is provided, ``save_gif_name``
            will be the name of the gif file which is generated inside this
            folder; if ``save_gif_dir`` is not provided, a temporary directory
            will be used for storing the frames while generating the gif file
            and ``save_gif_name`` is the path to the gif file relative to the
            working directory or it is a full path to the gif file.
            If ``save_gif`` is False this argument is ignored.
        save_gif_name: if ``save_gif`` is True, this is the name of the
            gif. If ``save_gif`` is False this argument is ignored.

    Returns:
        The fitness value of the agent (a score that tells how good is the agent in solving the task).

    Raises:
        RuntimeError: if ``max_iters_per_episode`` is not ``None`` and
            episode has not finished after ``max_iters_per_episode`` timesteps.
    """
    if not isinstance(episodes_aggr_func, str):
        raise TypeError(f"'episodes_aggr_func' should be str, instead got {type(episodes_aggr_func)}")
    if not isinstance(fitness_func, Callable):
        raise TypeError(f"'fitness_func' should be Callable, instead got {type(fitness_func)}")

    keep_frames = bool(save_gif_dir)
    if save_gif and save_gif_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix='frames-')
        # temp_dir will be automatically deleted when the program is closed
        # or is garbage-collected or temp_dir.cleanup() is explicitly called.
        # anyway it will be created in /tmp/, which is cleaned by the os when
        # restarting.
        save_gif_dir = os.path.join(temp_dir.name, 'frames')
        logging.debug(f"temp_dir: {temp_dir!r}")
        logging.debug(f"save_gif_dir: {save_gif_dir!r}")
    episode_str = ''
    rendering_mode = ''
    if render:
        rendering_mode += 'human'
    fitnesses = []
    for i_episode in range(episodes):
        msg = f'Starting episode #{i_episode} ...'
        logging.debug(msg)
        # if render:
        #     print(msg)
        if save_gif:
            if rendering_mode:
                rendering_mode += '+'
            rendering_mode += 'save[' + save_gif_dir
            if episodes > 1:
                episode_str = f"__episode_{i_episode}"
                if keep_frames:
                    rendering_mode += episode_str
            rendering_mode += ']'
        start_time_episode_including_reset = time.perf_counter_ns()
        # Reset env and agent:
        observation, info = env.reset(return_info=True)
        agent.reset()
        if isinstance(env, memory_evolution.envs.BaseForagingEnv) and info['env_info']['init_agent_position'] is None:
            env_agent = info['state']['agent']
            first_agent_pos = env_agent.pos
            # check that agent position change from one episode to the other:
            # note: there is a chance that could be the same, but should be very low.
            if 'prev_episode_agent_pos' in locals():
                assert env_agent.pos != prev_episode_agent_pos
            else:
                assert i_episode == 0, i_episode
            prev_episode_agent_pos = env_agent.pos
        assert env.t == 0., env.t
        start_time_episode = time.perf_counter_ns()
        reset_actual_time = (start_time_episode - start_time_episode_including_reset) / 10 ** 9
        msg = (
            f"\n"
            f"Episode reset took {reset_actual_time} actual seconds."
        )
        logging.log(logging.DEBUG + 5, '\n\t' + '\n\t'.join(msg.split('\n')))
        # if render:
        #     print(msg, end='\n\n')
        total_reward = 0.0
        if render or save_gif:
            # # print(observation)
            env.render(mode=rendering_mode)
        step = 0
        done = False
        # while not done and (max_iters_per_episode is None or t < max_iters_per_episode):
        # while not done and (max_env_t_per_episode is None or env.t < max_env_t_per_episode):
        while not done and (max_actual_time_per_episode is None
                            or (time.perf_counter_ns() - start_time_episode) < max_actual_time_per_episode * 10 ** 9):
            # Agent performs an action based on the current observation (and
            # its internal state, i.e. memory):
            if isinstance(env, memory_evolution.envs.BaseForagingEnv):
                assert env.step_count == step, (env.step_count, step)
            action = agent.action(observation)
            if isinstance(env, memory_evolution.envs.BaseForagingEnv):
                assert env.step_count == step, (env.step_count, step)
            observation, reward, done, info = env.step(action)
            if isinstance(env, memory_evolution.envs.BaseForagingEnv):
                assert env.step_count == step + 1, (env.step_count, step)
            total_reward += reward
            if render or save_gif:
                logging.debug(f"Observation hash: {hash(observation.tobytes())}")
                logging.debug(f"Action hash: {hash(action.tobytes())}")
                # # print("Observation:", observation, sep='\n')
                # print("Action:", action, sep=' ')
                # # print(info['state']['agent'])
                # # print(len(info['state']['food_items']), info['state']['food_items'])
                # # pprint(info)
                env.render(mode=rendering_mode)
                # print()
            step += 1
        end_t = env.t
        end_time_episode = time.perf_counter_ns()
        if isinstance(env, memory_evolution.envs.BaseForagingEnv):
            assert env.food_items_collected == total_reward, (env.food_items_collected, total_reward)
            # total_reward should be an integer
            assert total_reward == int(total_reward), (total_reward, int(total_reward))

            if env.food_items_collected == env.n_food_items:
                assert total_reward == env.food_items_collected == env.n_food_items == env.maximum_reward
                # note: the agent could take the last food item in the last timestep
                assert 0 <= step == env.step_count <= env.max_steps
            else:
                assert step == env.step_count == env.max_steps
                assert 0 <= env.food_items_collected < env.n_food_items

        fitness = fitness_func(reward=total_reward, steps=step, done=done, env=env)
        msg = f"fitness: {fitness}"
        # print(msg)
        logging.debug(msg)
        if not isinstance(fitness, Real):
            raise TypeError(f"return value of 'fitness_func()' should be float, instead got {type(fitness)}")
        fitnesses.append(fitness)
        if done:
            if isinstance(env, memory_evolution.envs.BaseForagingEnv):
                assert step == env.step_count == info['debug_info']['_is_done']['step_count'] + 1, (
                    step, env.step_count, info['debug_info']['_is_done']['step_count'])
                assert info['debug_info']['_is_done']['done'] is True, info['debug_info']['_is_done']
                assert (info['debug_info']['_is_done']['food_items_collected'] == info['debug_info']['_is_done']['n_food_items']
                        or env.step_count == info['debug_info']['_is_done']['max_steps']), info['debug_info']['_is_done']

            actual_time = (end_time_episode - start_time_episode) / 10 ** 9
            msg = (
                f"{agent} fitness {fitness}\n"
                f"Episode loop finished after {step} timesteps"
                f", for a total of {end_t} simulated seconds"
                f" (in {actual_time} actual seconds)."
            )
            logging.log(logging.DEBUG + 5, '\n\t' + '\n\t'.join(msg.split('\n')))
            # if render:
            #     print(msg, end='\n\n')
            if save_gif:
                gifname = save_gif_name
                if episode_str:
                    root, ext = os.path.splitext(save_gif_name)
                    gifname = root + episode_str + ext
                frames_dir = save_gif_dir + (episode_str if keep_frames else '')
                logging.debug(f"gifname: {gifname};  frames_dir: {frames_dir};")
                gif_duration = actual_time / (step + 1)  # all steps plus the 0 step.
                # gif duration needs to be at least 0.01, if it is less
                # (and it will be kept only until the second decimal digit, thus round it to it)
                # (instead of rounding all the same, you could use different duration of the first
                # image and few others to encode the actual total duration (but then be careful when
                # calculating FPS to convert the gif to video))
                gif_duration = max(0.01, round(gif_duration, 2))
                if gif_duration < 0.01:
                    raise RuntimeError(f"gif_duration: {gif_duration}")
                logging.info(f"Gif generated: gif_total_duration={gif_duration * (step + 1)}, "
                             f"gif_duration={gif_duration}, but actual time was actual_time={actual_time}")
                memory_evolution.utils.generate_gif_from_path(frames_dir,
                                                              gif_name=gifname,
                                                              remove_frames=(not keep_frames),
                                                              save_gif_in_frames_dir=keep_frames,
                                                              duration=gif_duration)
                # todo: sarebbe bene fare anche un video .mp4, però accertati che FPS in uscita siano e.g. 50 o 60,
                #       e non 100+ come nella gif che semplicemente aggrega tutti i frames.
        else:
            if isinstance(env, memory_evolution.envs.BaseForagingEnv):
                assert step == env.step_count == info['debug_info']['_is_done']['step_count'] + 1, (
                    step, env.step_count, info['debug_info']['_is_done']['step_count'])
                assert info['debug_info']['_is_done']['done'] is False, info['debug_info']['_is_done']
                assert not (info['debug_info']['_is_done']['food_items_collected'] == info['debug_info']['_is_done']['n_food_items']
                            or env.step_count == info['debug_info']['_is_done']['max_steps']), info['debug_info']['_is_done']
            raise RuntimeError(
                f"Episode has not finished after {step} timesteps"
                f" and {end_t} simulated seconds"
                f" (in {(end_time_episode - start_time_episode) / 10 ** 9} actual seconds).")
    if save_gif and save_gif_dir is None:
        temp_dir.cleanup()
    final_fitness = getattr(np, episodes_aggr_func)(fitnesses)
    # env.close()  # use it only in main, otherwise it will be closed and
    # opened again each time it is evaluated.
    return final_fitness

