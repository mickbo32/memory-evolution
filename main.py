from pprint import pprint

import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import neat
import numpy as np
from numpy.random import SeedSequence
import os
import pandas as pd
import sys
import time

from gym.utils.env_checker import check_env  # from stable_baselines.common.env_checker import check_env

from memory_evolution.agents import RnnNeatAgent, CtrnnNeatAgent
from memory_evolution.envs import BaseForagingEnv, MazeForagingEnv, TMaze
from memory_evolution.utils import evaluate_agent

mpl.use('Qt5Agg')  # Change matplotlib backend to show correctly in PyCharm.


class NoneAgent:

    def __init__(self, env):
        self.env = env

    def action(self, obs):
        return self.env.action_space.sample()

    def reset(self):
        pass

    @staticmethod
    def eval_genome(genome, config) -> float:
        fitness = 0.0
        return fitness

    @classmethod
    def eval_genomes(cls, genomes, config) -> None:
        for genome_id, genome in genomes:
            genome.fitness = cls.eval_genome(genome, config)

    def get_init_population(self, *args, **kwargs):
        return None

    def evolve(self, *args, **kwargs):
        return self

    def run(self, *args, **kwargs):
        # Evolution
        # None

        # Run until a solution is found. (Run for up to n generations.)
        start_time = time.time()
        # start_time_monotonic = time.monotonic_ns()
        # start_time_perf_counter = time.perf_counter_ns()
        # start_time_process_time = time.process_time_ns()
        # start_time_thread_time = time.thread_time_ns()
        start_time_perf_counter = time.perf_counter_ns()
        start_time_process_time = time.process_time_ns()
        start_time_thread_time = time.thread_time_ns()
        print(f"Evolution started at", pd.Timestamp.utcnow().isoformat(' '))
        winner = None
        end_time = time.time()
        end_time_perf_counter = time.perf_counter_ns()
        end_time_process_time = time.process_time_ns()
        end_time_thread_time = time.thread_time_ns()
        tot_time_perf_counter = end_time_perf_counter - start_time_perf_counter
        tot_time_process_time = end_time_process_time - start_time_process_time
        tot_time_thread_time = end_time_thread_time - start_time_thread_time
        print(f"Evolution took {tot_time_perf_counter / 10**9} seconds"
              f" (for a total of {tot_time_process_time / 10**9} seconds of process time and"
              f" {tot_time_thread_time / 10**9} seconds of thread time).")
        print(f"equal to {pd.Timedelta(nanoseconds=tot_time_perf_counter)!s}"
              f" (for a total of {pd.Timedelta(nanoseconds=tot_time_process_time)!s} and"
              f" {pd.Timedelta(nanoseconds=tot_time_thread_time)!s}).")

        # Display stats on the evolution performed.
        pass

        # Display the winning genome.
        print('\nNone genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        winner_net = self.phenotype.create(winner, self.config)
        # todo: (input, action) + render
        # for xi, xo in zip(xor_inputs, xor_outputs):
        #     output = winner_net.activate(xi)
        #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))


def run(env: gym.Env, agent=None, episodes=1) -> None:
    """Runs the main loop: evolve, interact, repeat.

    For ``episodes``

    Args:
        env: a gym environment object (the gym environment constructor should
            be a subclass of gym.Evn)
        agent: the agent, an object with an ``action()`` method which takes the
            observation as argument and returns a valid action and with a
            ``reset()`` method which reset the agent to an initial
            state ``t==0``;
            if agent is ``None``, it performs random actions.
        episodes: number of independent episodes that will be run.
    """
    print('Main loop')

    if agent is None:
        agent = NoneAgent()

    for i_episode in range(episodes):
        observation = env.reset()
        t = 0
        for t in range(100):
            assert env.step_count == t, (env.step_count, t)
            env.render()
            # print(observation)
            action = env.action_space.sample()
            assert env.step_count == t, (env.step_count, t)
            observation, reward, done, info = env.step(action)
            assert env.step_count == t + 1, (env.step_count, t)
            # print(observation)
            # print(info['state']['agent'])
            # print(len(info['state']['food_items']), info['state']['food_items'])
            # pprint(info)
            if done:
                env.render()
                print("Episode finished after {} timesteps".format(t + 1))
                break
        else:
            assert False, "Episode has not finished after {} timesteps".format(t + 1)
    env.close()


if __name__ == '__main__':

    # ----- ENVIRONMENT -----

    # env = gym.make('CartPole-v0')
    # env = BaseForagingEnv(640, (1.5, 1.), fps=None, seed=42)
    # env = BaseForagingEnv(640, (1.5, 1.), agent_size=.5, food_size=.3, fps=None, seed=42)
    # env = TMaze(env_size=(1.5, 1.), fps=None, seed=42, n_food_items=0)
    # env = TMaze(env_size=(1.5, 1.), fps=None, seed=42, n_food_items=20)
    # env = TMaze(env_size=(1.5, 1.), fps=None, seed=42, n_food_items=50)
    # env = TMaze(.1001, env_size=(1.5, 1.), fps=None, seed=42)
    env = TMaze(seed=42)
    print('observation_space:',
          env.observation_space.shape,
          np.asarray(env.observation_space.shape).prod())
    check_env(env)  # todo: move in tests

    # print(env.action_space)  # Discrete(4)
    # print(env.observation_space)  # Box([[[0] ... [255]]], (5, 5, 1), uint8)
    # print(env.observation_space.low)  # [[[0] ... [0]]]
    # print(env.observation_space.high)  # [[[255] ... [255]]]
    # print(env.observation_space.shape)  # (5, 5, 1)
    # print(env.observation_space.sample())  # [[[102] ... [203]]] / [[[243] ... [64]]] / each time different

    # ----- AGENT -----

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)

    config_path = os.path.join(local_dir, 'config-rnn')
    agent = RnnNeatAgent(config_path)

    # ----- MAIN LOOP -----
    # Evolve, interact, repeat.

    # evaluate_agent(NoneAgent(env), env, episodes=2, render=True)

    checkpointer = neat.Checkpointer(generation_interval=100,
                                     time_interval_seconds=300,
                                     filename_prefix='neat-checkpoint-')
    agent.set_env(env)
    winner = agent.evolve(render=True, checkpointer=checkpointer)
    print(type(winner))
    evaluate_agent(agent, env, episodes=2, render=True)
    # run(env, episodes=2)

