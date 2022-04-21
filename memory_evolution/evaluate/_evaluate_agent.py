import os.path
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from collections.abc import Iterable, Sequence
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


# def compute_fitness(total_reward, other_metric, other_metric2) -> float:
#     fitness = 0.0
#     return fitness


def evaluate_agent(agent,
                   env: gym.Env,
                   episodes: int = 1,
                   max_actual_time_per_episode: Optional[Union[int, float]] = None,
                   episodes_fitness_aggr_func: Literal['min', 'max', 'mean', 'median'] = 'min',
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
        max_actual_time_per_episode: if ``None`` it will just run until the
            environment is done, otherwise after ``max_actual_time_per_episode``
            seconds the environment will be closed and an error is raised.
            This time do not include the episode reset time, but only the
            episode loop (i.e. steps) time.
        episodes_fitness_aggr_func: function to use to aggregate all the fitness
            values collected for each single independent episode (default is
            'min': The genome's fitness is its worst performance across all runs).
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
    time_step = env.time_step  # todo: non serve, basta fare env.t - prev_env_t; togli time_step da env e chiama env.dt
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
        fitness = 0.0  # food collected
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
            fitness += reward
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
            assert env.food_items_collected == fitness, (env.food_items_collected, fitness)
            # todo: env should have only basic attributes, don't add unnecessary attributes
            #  (don't_use unnecessary attributes .time_step, .food_items_collected, etc...,
            #  use them only for asserts).
        if isinstance(env, memory_evolution.envs.BaseForagingEnv):
            assert fitness == int(fitness), (fitness, int(fitness))  # fitness should be an integer at this point

            # todo: neat should be able to take tuples as fitness

            # # fitness: (total_reward, agent_distance_from_start):
            # end_agent_pos = info['state']['agent'].pos
            # agent_distance_from_start = memory_evolution.geometry.euclidean_distance(first_agent_pos, end_agent_pos)
            # # print(f"agent_distance_from_start: {agent_distance_from_start}")
            # logging.debug(f"agent_distance_from_start: {agent_distance_from_start}")
            # # todo: neat should be able to take tuples as fitness
            # # fitness = (fitness, agent_distance_from_start)
            # fitness += agent_distance_from_start / max(env.env_size) * .99

            # # fitness: (total_reward, 1 - timesteps used to get all food items if all food items collected):
            # if env.food_items_collected == env.n_food_items:
            #     timesteps_normalized = step / env.max_steps
            #     # note: the agent could take the last food item in the last timestep
            #     #   thus, 'timesteps_normalized' could be 1
            #     # note2: more timesteps is bad, less timestep is good, thus fitness: 1 - 'timesteps_normalized'
            #     logging.debug(f"timesteps_normalized: {timesteps_normalized};"
            #                   f" 1-timesteps_normalized: {1 - timesteps_normalized}")
            #     fitness += 1 - timesteps_normalized
            # else:
            #     assert step == env.step_count == env.max_steps
            #     assert env.food_items_collected < env.n_food_items

            # todo: do a function to calculate fitness given reward, timesteps, done, env (as **kwargs)
            # Multi-objective_optimization with normalized linear scalarization
            total_reward = fitness
            total_reward_normalized = fitness / env.maximum_reward  # normalize total_reward
            assert 0 <= total_reward_normalized <= 1, total_reward_normalized
            timesteps_normalized = step / env.max_steps  # normalize timesteps
            # note: the agent could take the last food item in the last timestep
            #   thus, 'timesteps_normalized' could be 1
            # note2: more timesteps is bad, less timestep is good, thus fitness: 1 - 'timesteps_normalized'
            logging.debug(f"timesteps_normalized: {timesteps_normalized};"
                          f" 1-timesteps_normalized: {1 - timesteps_normalized}")
            assert 0 <= timesteps_normalized <= 1, timesteps_normalized
            fitness = (total_reward_normalized + (1 - timesteps_normalized)) / 2
            assert 0 <= fitness <= 1, fitness
            if env.food_items_collected == env.n_food_items:
                assert total_reward == env.food_items_collected == env.n_food_items == env.maximum_reward
                # note: the agent could take the last food item in the last timestep
                #   thus, 'timesteps_normalized' could be 1
                assert 0 <= step == env.step_count <= env.max_steps
            else:
                assert step == env.step_count == env.max_steps
                assert 0 <= env.food_items_collected < env.n_food_items

            msg = f"fitness: {fitness}"
            # print(msg)
            logging.debug(msg)
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
    final_fitness = getattr(np, episodes_fitness_aggr_func)(fitnesses)
    # env.close()  # use it only in main, otherwise it will be closed and
    # opened again each time it is evaluated.
    return final_fitness

