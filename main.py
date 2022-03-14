import logging
import os
from pprint import pprint
import random  # neat uses random  # todo: allow seeding in neat
import sys
import time
from typing import Optional

import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import neat
import numpy as np
from numpy.random import SeedSequence
import pandas as pd

from gym.utils.env_checker import check_env  # from stable_baselines.common.env_checker import check_env

from memory_evolution.agents import BaseAgent, RnnNeatAgent, CtrnnNeatAgent
from memory_evolution.envs import BaseForagingEnv, MazeForagingEnv, TMaze
from memory_evolution.utils import evaluate_agent, set_main_logger

# matplotlib settings:
mpl.use('Qt5Agg')  # Change matplotlib backend to show correctly in PyCharm.


class RandomAgent(BaseAgent):

    def __init__(self, env):
        super().__init__()
        self.set_env(env)

    def action(self, obs):
        return self.get_env().action_space.sample()

    def reset(self):
        pass

    @staticmethod
    def eval_genome(genome, config) -> float:
        fitness = 0.0
        return fitness

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
        agent = RandomAgent(env)

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

    # ----- Settings -----

    # logging settings:
    logging_dir, UTCNOW = set_main_logger(file_handler_all=None, stdout_handler=logging.INFO)
    logging.debug(__file__)

    # neat random seeding:
    random.seed(42)
    logging.debug(random.getstate())
    # Use random.setstate(state) to set an old state, where 'state' have been obtained from a previous call to getstate().

    # ----- ENVIRONMENT -----

    # env = gym.make('CartPole-v0')
    # env = BaseForagingEnv(640, (1.5, 1.), fps=None, seed=42)
    # env = BaseForagingEnv(640, (1.5, 1.), agent_size=.5, food_size=.3, fps=None, seed=42)
    # env = TMaze(env_size=(1.5, 1.), fps=None, seed=42, n_food_items=0)
    # env = TMaze(env_size=(1.5, 1.), fps=None, seed=42, n_food_items=20)
    # env = TMaze(env_size=(1.5, 1.), fps=None, seed=42, n_food_items=50)
    # env = TMaze(.1001, env_size=(1.5, 1.), fps=None, seed=42)
    # env = TMaze(seed=42)
    # env = TMaze(env_size=(1.5, 1.), seed=42, agent_size=.15, n_food_items=10, max_steps=500, vision_resolution=7)  # todo: use in tests
    # env = BaseForagingEnv(env_size=(1.5, 1.), seed=42, agent_size=.15, n_food_items=10, max_steps=500, vision_resolution=7) # todo: use in tests
    # env = TMaze(env_size=(1.5, 1.), seed=42, agent_size=.15, n_food_items=10, max_steps=500, vision_resolution=7)
    # env = BaseForagingEnv(env_size=(1.5, 1.), seed=42, agent_size=.15, n_food_items=10, max_steps=500, vision_resolution=7)
    env = TMaze(seed=42, agent_size=.15, n_food_items=10, max_steps=500, vision_resolution=7)
    logging.debug(env._seed)  # todo: use a variable seed (e.g.: seed=42; env=TMaze(seed=seed); logging.debug(seed)) for assignation of seed, don't access the internal variable
    print('observation_space:',
          env.observation_space.shape,
          np.asarray(env.observation_space.shape).prod())
    check_env(env)  # todo: move in tests
    print('Env checked.')

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

    # evaluate_agent(RandomAgent(env), env, episodes=2, render=True)

    checkpointer = neat.Checkpointer(generation_interval=100,
                                     time_interval_seconds=300,
                                     filename_prefix=os.path.join(
                                         logging_dir,
                                         UTCNOW + '_' + 'neat-checkpoint-'))

    agent.set_env(env)
    winner = agent.evolve(render=0, checkpointer=checkpointer, parallel=1,
                          filename_tag=UTCNOW + '_', path_dir=logging_dir, file_ext='.png')
    # fixme: todo: parallel=True use the same seed for the environment in each process
    #     (but for the agent is correct and different it seems)
    print(type(winner))
    print(list(map(type, winner)))
    evaluate_agent(agent, env, episodes=2, render=True,
                   save_gif=True,
                   save_gif_dir=os.path.join(logging_dir, 'frames_' + UTCNOW),
                   save_gif_name=UTCNOW + '.gif')
    # run(env, episodes=2)

    # ----- CLOSING AND REPORTING -----

    env.close()


'''
Better efficiency:
env = TMaze(env_size=(1.5, 1.), seed=42, agent_size=.15, n_food_items=10, max_steps=500, vision_resolution=7)
(parallel = False)

* b163541 (HEAD -> main, origin/main, origin/HEAD) Merge branch 'continuous' into main Continuous environment and agents with evolution

render = False
Episode finished after 500 timesteps, for a total of 500 simulated seconds (in 7.13121553 actual seconds).

render = True
Episode finished after 500 timesteps, for a total of 500 simulated seconds (in 19.826593934 actual seconds).


* ce3470d (continuous) big refactoring, environment efficiency improved, env_img, pixel space mask for valid positions, geometry, pygame sprites for items

render = False
Episode finished after 500 timesteps, for a total of 500 simulated seconds (in 1.843639818 actual seconds).

render = True
Episode finished after 500 timesteps, for a total of 500 simulated seconds (in 15.735012616 actual seconds).


Note: on both tests here above with rendering True, the rendering was slow because of the external screen used,
by using only the integrated screen of the pc the rendering time goes down to 19->14 and 15->3.5 respectively.


* cd9570b

render = False
Episode finished after 500 timesteps, for a total of 500 simulated seconds (in 1.6946048 actual seconds).
render = False;parallel=True
Episode finished after 500 timesteps, for a total of 500 simulated seconds (in 3.641211265 actual seconds).

render = True
Episode finished after 500 timesteps, for a total of 500 simulated seconds (in 2.768054353 actual seconds).

'''

