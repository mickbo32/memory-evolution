import dill  # pickle extension
from functools import reduce
import json
import logging
from operator import mul
import os
import pickle
from pprint import pprint
import random  # neat uses random  # todo: allow seeding in neat
import sys
import time
import typing
from typing import Literal, Optional
import warnings

import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import neat
import numpy as np
from numpy.random import SeedSequence
import pandas as pd
from tqdm import tqdm

import memory_evolution
from memory_evolution.agents import RandomActionAgent, RnnNeatAgent, CtrnnNeatAgent
from memory_evolution.envs import BaseForagingEnv, MazeForagingEnv, TMaze, RadialArmMaze
from memory_evolution.evaluate import evaluate_agent
from memory_evolution.logging import set_main_logger

from memory_evolution.load import load_env, load_agent, get_checkpoint_number, AVAILABLE_LOADING_METHODS


def plot_bars(df: pd.DataFrame, ax, ylabel=None, title=None, ylim=None,
              std=True, box=False,
              color=None, ecolor='black', mincolor='gray', maxcolor='gray',
              alpha=0.5,
              ):
    """Plot metrics along axis=0"""
    if sum((std, box)) > 1:
        raise ValueError("select no more than one among 'std' and 'box'")
    x = np.arange(len(df.columns))
    std_bars = {}
    if std:
        std_bars = {'yerr': df.std(axis=0),
                    'align': 'center', 'alpha': alpha, 'ecolor': ecolor, 'capsize': 10}
    ax.bar(x, df.mean(axis=0), color=color,
           **std_bars)
    if std:
        ax.plot(x, df.max(axis=0), 'x', color=maxcolor)  # you can also use plt.scatter()
        ax.plot(x, df.min(axis=0), 'x', color=mincolor)  # you can also use plt.scatter()
    if box:
        cmap = plt.get_cmap("tab10")
        df.boxplot(ax=ax, positions=x, widths=.5, capprops={'color': cmap(0)})  # , showfliers=False, showmeans=True, sym='x', flierprops={'color': 'gray'}

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(df.columns)
    ax.yaxis.grid(True)
    if title is not None:
        ax.set_title(title)
    return x


def plot_accuracy_results(all_results: pd.DataFrame, view=True, filename=None):
    np.testing.assert_array_equal(all_results.columns,
                                  ['test_agent_first_arm_accuracy', 'test_agent_target_reached_rate', 'Fitness'])
    accuracy_labels = ['test_agent_first_arm_accuracy', 'test_agent_target_reached_rate']
    accuracy_plot_labels = ['first arm accuracy', 'target reached rate']
    fitness_labels = ['Fitness']
    fitness_plot_labels = ['Best fitness']
    accuracy_results = all_results[accuracy_labels].copy(deep=True)
    fitness_results = all_results[fitness_labels].copy(deep=True)

    accuracy_results.columns = accuracy_plot_labels
    # print(accuracy_results)
    fitness_results.columns = fitness_plot_labels
    # print(fitness_results)

    fig, axes = plt.subplots(1, 2, figsize=(6.4 * 1.6, 4.8))
    x = np.arange(len(all_results.columns))
    plot_bars(accuracy_results, axes[0], ylabel="Accuracy (%)",
              title=f"Accuracy results (averaged across {len(all_results.index)} evolution processes)")
    plot_bars(fitness_results, axes[1], ylabel="Fitness",
              title=f"Best fitness (averaged across {len(all_results.index)} evolution processes)")

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    if view:
        plt.show(block=True)

    plt.close()


def plot_avg_df(df: pd.DataFrame, ax=None, avg_label="average", avg_color='b', show_maxminstd_label=True):
    """Plot metrics along axis=0"""
    if ax is None:
        fig, ax = plt.subplots(1)
    avg_df = df.mean(axis=0)
    # ax.plot(df.columns, avg_df, 'b-', label=avg_label)
    ax.plot(df.columns, avg_df, '-', color=avg_color, label=avg_label)
    kwargs = {'label': "max"} if show_maxminstd_label else {}
    ax.plot(df.columns, df.max(axis=0), '--', color='gray', **kwargs)
    kwargs = {'label': "min"} if show_maxminstd_label else {}
    ax.plot(df.columns, df.min(axis=0), '--', color='gray', **kwargs)
    kwargs = {'label': '\u00B11 std'} if show_maxminstd_label else {}
    ax.fill_between(df.columns,
                    avg_df - df.std(axis=0), avg_df + df.std(axis=0),
                    facecolor='yellow', alpha=0.5, **kwargs)
    return ax


def get_best_fitness(all_stats, filename=None) -> pd.DataFrame:
    """ Get the populations' best fitness over generations.

    If ``filename`` is None, don't save.
    """
    assert len(all_stats) >= 1, "'all_stats' should have at least one evolution process"
    if not isinstance(next(iter(all_stats.values())), neat.statistics.StatisticsReporter):
        raise TypeError(next(iter(all_stats.values())))

    generation = None
    best_fitness = []
    for statistics in all_stats.values():
        if generation is None:
            generation = range(len(statistics.most_fit_genomes))
        assert len(statistics.most_fit_genomes) == len(generation), (len(statistics.most_fit_genomes), len(generation))
        _best_fitness = [c.fitness for c in statistics.most_fit_genomes]
        # _avg_fitness = np.array(statistics.get_fitness_mean())
        # _stdev_fitness = np.array(statistics.get_fitness_stdev())
        best_fitness.append(_best_fitness)
    best_fitness = pd.DataFrame(best_fitness, columns=generation)
    assert generation is not None
    assert len(best_fitness.index) == len(all_stats) and len(best_fitness.columns) == len(generation)
    if filename is not None:
        best_fitness.to_csv(filename + '.csv')

    return best_fitness


def plot_best_fitness(best_fitness: pd.DataFrame, ylog=False, view=True, filename=None, ylim=None, save_csv=True):
    """ Plots the populations' best fitness over generations.

    If ``filename`` is None, don't save.
    """

    # fig, ax = plt.subplots(1)
    # avg_best_fitness = best_fitness.mean(axis=0)
    # ax.plot(generation, avg_best_fitness, 'b-', label="average best fitness")
    # ax.plot(generation, best_fitness.max(axis=0), '--', color='gray', label="max")
    # ax.plot(generation, best_fitness.min(axis=0), '--', color='gray', label="min")
    # ax.fill_between(generation,
    #                 avg_best_fitness - best_fitness.std(axis=0), avg_best_fitness + best_fitness.std(axis=0),
    #                 facecolor='yellow', alpha=0.5, label='\u00B11 std')
    ax = plot_avg_df(best_fitness, avg_label="average best fitness")

    plt.title(f"Population's best fitness (averaged across {len(best_fitness)} evolution processes)")
    ax.set_xlabel("Generations")
    ax.set_ylabel("Fitness")
    ax.grid()
    ax.legend(loc="best")
    if ylog:
        base = 10
        # ax.set_yscale('symlog')
        ax.set_yscale(mpl.scale.SymmetricalLogScale(ax, base=base, linthresh=1, subs=[2.5, 5, 7.5]))
        ax.grid(True, which='minor', color='gainsboro', linestyle=':', linewidth=.5)
        if ylim is not None:
            pass  # todo: yticks
    if ylim is not None:
        ax.set_ylim(ylim)  # ax.set_ylim([ymin, ymax])
        # ax.xlim(right=xmax)  # xmax is your value
        # ax.xlim(left=xmin)  # xmin is your value
        # ax.ylim(top=ymax)  # ymax is your value
        # ax.ylim(bottom=ymin)  # ymin is your value

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    if view:
        plt.show(block=True)

    plt.close()


def get_genome_metrics(all_stats, filename=None) -> (pd.DataFrame, pd.DataFrame):
    """ Get the populations' best genome metrics over generations.

    If ``filename`` is None, don't save.
    """
    assert len(all_stats) >= 1, "'all_stats' should have at least one evolution process"
    if not isinstance(next(iter(all_stats.values())), neat.statistics.StatisticsReporter):
        raise TypeError(next(iter(all_stats.values())))

    generation = None
    nodes = []
    connections = []
    for statistics in all_stats.values():
        if generation is None:
            generation = range(len(statistics.most_fit_genomes))
        assert len(statistics.most_fit_genomes) == len(generation), (len(statistics.most_fit_genomes), len(generation))
        nodes.append([len(g.nodes) for g in statistics.most_fit_genomes])
        connections.append([len(g.connections) for g in statistics.most_fit_genomes])
        # todo: do also for pruned network
    nodes = pd.DataFrame(nodes, columns=generation)
    connections = pd.DataFrame(connections, columns=generation)
    assert generation is not None
    assert len(nodes.index) == len(all_stats) and len(nodes.columns) == len(generation)
    assert len(connections.index) == len(all_stats) and len(connections.columns) == len(generation)
    if filename is not None:
        nodes.to_csv(filename + '_nodes.csv')
        connections.to_csv(filename + '_connections.csv')

    return nodes, connections


def plot_genome_metrics(nodes: pd.DataFrame, connections: pd.DataFrame,
                        # ylog=False,
                        view=True, filename=None, ylim=None, save_csv=True):
    """ Plots the populations' best genome metrics over generations.

    If ``filename`` is None, don't save.
    """
    assert len(nodes) == len(connections)
    assert len(nodes.index) == len(connections.index)
    assert len(nodes.columns) == len(connections.columns)

    fig, ax = plt.subplots(1)
    plot_avg_df(nodes, ax=ax, avg_label="average nodes (without inputs)", avg_color='g', show_maxminstd_label=False)
    plot_avg_df(connections, ax=ax, avg_label="average connections", avg_color='b')

    plt.title(f"Population's best genome (averaged across {len(nodes)} evolution processes)")
    ax.set_xlabel("Generations")
    ax.set_ylabel("Number of")
    ax.grid()
    ax.legend(loc="best")
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    if view:
        plt.show(block=True)

    plt.close()


if __name__ == '__main__':

    # matplotlib and pd settings:
    # pd.set_option('precision', 8)
    isRunningInPyCharm = "PYCHARM_HOSTED" in os.environ
    if isRunningInPyCharm:
        mpl.use('Qt5Agg')  # Change matplotlib backend to show correctly in PyCharm.

        pd.set_option("display.max_columns", 20, "expand_frame_repr", False)
        # pd.set_option("display.max_columns", 100, "display.width", 120, "expand_frame_repr", True)
        # with pd.option_context('display.max_columns', 20, 'expand_frame_repr', False, 'precision', 8):
        # pd.set_option('display.min_rows', 20)
        # pd.set_option('display.max_rows', 80)

    # ----- Settings -----
    RENDER = False  # True  # False  # render or just save gif files
    # ---
    # LOAD_DIR_TAG = '2022-06-08T172246_training_allocentric_100'
    LOAD_DIR_TAG = '2022-06-10_small_allo_and_ego_test'
    # LOAD_DIR_TAG = '2022-06-13_training_allocentric_90'
    # LOAD_DIR_TAG = '2022-06-13_training_egocentric_90'
    # LOAD_DIR_TAG = '2022-06-16T173323_allo30_same-prob_600epochs'
    # LOAD_DIR_TAG = '2022-06-23T073841_ego30_same-prob_600epochs'
    # LOAD_DIR_TAG = '2022-06-20T085938_allo30_unconnected'
    # LOAD_DIR_TAG = '2022-06-22T205534_ego30_unconnected'
    LOAD_AGENT_DIR = f"logs/saved_logs/outputs-link/{LOAD_DIR_TAG}/logs/"
    LOAD_FROM: AVAILABLE_LOADING_METHODS = 'pickle'
    N_EPISODES = 0  # 3  # 0  # 5
    LOGGING_DIR = 'logs'

    # logging settings:
    assert LOAD_FROM != 'checkpoint'
    LOGGING_DIR, UTCNOW = set_main_logger(file_handler_all=None,
                                          logging_dir=LOGGING_DIR,
                                          stdout_handler=logging.INFO,# - 2,
                                          file_handler_now_filename_fmt=LOAD_DIR_TAG + "_LOADED_DIR___now_{utcnow}.log")
    logging.info(__file__)
    LOADED_DIR_TAG_UTCNOW = LOAD_DIR_TAG
    LOADED_DIR_TAG_UTCNOW += '_LOADED_DIR___now_' + UTCNOW

    # neat random seeding:
    # random.seed(42)
    logging.debug(random.getstate())
    # Use random.setstate(state) to set an old state, where 'state' have been obtained from a previous call to getstate().

    # ----- LOAD -----

    agents_tags = sorted([os.path.basename(f.removesuffix('_genome.pickle'))
                          for f in os.listdir(LOAD_AGENT_DIR)
                          if f.endswith('_genome.pickle')])

    # NOTE: assuming all of them are sharing the same environment
    #env = load_env(agents_tags[0], LOAD_AGENT_DIR)

    # test in env without landmarks:
    warnings.warn("The environment is NOT LOADED, a different one is created for testing...")
    # # starting position up, target East:
    env = RadialArmMaze(arms=4, corridor_width=0.2, window_size=200, env_size=1.0, **{
        'agent_size': 0.075, 'food_size': 0.05, 'n_food_items': 1, 'max_steps': 400,
        'vision_depth': 0.2, 'vision_field_angle': 135, 'vision_resolution': 3, 'vision_channels': 3,
        'vision_point_radius': 0.04,
        'agent_color': np.asarray([0, 255, 255], dtype=np.uint8),
        'background_color': np.asarray([0, 0, 0], dtype=np.uint8),
        'outside_color': np.asarray([255, 0, 0], dtype=np.uint8),
        'food_color': np.asarray([0, 200, 55], dtype=np.uint8),
        'food_visible': False, 'random_init_agent_position': None,
        'init_agent_position': (.5, .9),  # North arm
        'init_food_positions': ((0.9, 0.5),),  # East arm
        # 'init_food_positions': ((0.1, 0.5),),  # West arm
        # 'init_food_positions': ((0.5, 0.1),),  # South arm
        'landmark_size': 0.25, 'init_landmarks_positions': None, 'landmarks_colors': None,
        'borders': None, 'pairing_init_food_positions': None, 'rotation_step': 15.0, 'forward_step': 0.01,
        'observation_noise': None, 'inverted_color_rendering': True, 'fps': None, 'seed': None})
    '''
    from logs:
    allo:
    base_foraging  :  env: memory_evolution.envs.radial_arm_maze.RadialArmMaze(arms=4, corridor_width=0.2, window_size=200, env_size=1.0, kwargs={'agent_size': 0.075, 'food_size': 0.05, 'n_food_items': 1, 'max_steps': 400, 'vision_depth': 0.2, 'vision_field_angle': 135, 'vision_resolution': 3, 'vision_channels': 3, 'vision_point_radius': 0.04, 'agent_color': array([  0, 255, 255], dtype=uint8), 'background_color': array([0, 0, 0], dtype=uint8), 'outside_color': array([255,   0,   0], dtype=uint8), 'food_color': array([  0, 200,  55], dtype=uint8), 'food_visible': False, 'random_init_agent_position': ((0.5, 0.1), (0.5, 0.9), (0.1, 0.5)), 'init_food_positions': ((0.9, 0.5),), 'landmark_size': 0.25, 'init_landmarks_positions': ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)), 'landmarks_colors': (array([255,   0, 255], dtype=uint8), array([255, 255,   0], dtype=uint8), array([255, 127, 127], dtype=uint8), array([255, 255, 255], dtype=uint8)), 'borders': None, 'pairing_init_food_positions': None, 'rotation_step': 15.0, 'forward_step': 0.01, 'observation_noise': None, 'init_agent_position': None, 'inverted_color_rendering': True, 'fps': None, 'seed': None, 'platform': None})
    ego:
    base_foraging  :  env: memory_evolution.envs.radial_arm_maze.RadialArmMaze(arms=4, corridor_width=0.2, window_size=200, env_size=1.0, kwargs={'agent_size': 0.075, 'food_size': 0.05, 'n_food_items': 1, 'max_steps': 400, 'vision_depth': 0.2, 'vision_field_angle': 135, 'vision_resolution': 3, 'vision_channels': 3, 'vision_point_radius': 0.04, 'agent_color': array([  0, 255, 255], dtype=uint8), 'background_color': array([0, 0, 0], dtype=uint8), 'outside_color': array([255,   0,   0], dtype=uint8), 'food_color': array([  0, 200,  55], dtype=uint8), 'food_visible': False, 'random_init_agent_position': ((0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)), 'pairing_init_food_positions': (((0.9, 0.5),), ((0.1, 0.5),), ((0.5, 0.1),), ((0.5, 0.9),)), 'landmark_size': 0.25, 'init_landmarks_positions': ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)), 'landmarks_colors': (array([255,   0, 255], dtype=uint8), array([255, 255,   0], dtype=uint8), array([255, 127, 127], dtype=uint8), array([255, 255, 255], dtype=uint8)), 'borders': None, 'rotation_step': 15.0, 'forward_step': 0.01, 'observation_noise': None, 'init_agent_position': None, 'init_food_positions': None, 'inverted_color_rendering': True, 'fps': None, 'seed': None, 'platform': None})
    '''

    agents = {}
    configs = {}
    for LOAD_AGENT in agents_tags:
        agent, other_loads = load_agent(LOAD_AGENT, LOAD_AGENT_DIR, LOAD_FROM)
        agent.set_env(env)
        agents[LOAD_AGENT] = agent
        configs[LOAD_AGENT] = other_loads['config']
    del LOAD_AGENT, agent, other_loads

    # ---
    # FIXME: see below, this is a temporary patch
    warnings.warn("Temporary patch to workaround the fact dill don't save the global scope. "
                  "Pay attention at what fitness function was used an which one you are loading. "
                  "If they are different the code will be broke.")
    ff_time = memory_evolution.evaluate.fitness_func_time_minimize
    min_ff_time = ff_time(reward=None, steps=env.max_steps, done=None, env=None, agent=None)
    '''
    ### FIXME: on loaded agent, fitness_func (pickled with dill) reference to main which is not anymore available:
    In [12]: type(agent).fitness_func(reward=None, steps=34, done=True, env=None, agent=None)
    Traceback (most recent call last):
      File "/home/michele/anaconda3/envs/evo/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3397, in run_code
        exec(code_obj, self.user_global_ns, self.user_ns)
      File "<ipython-input-12-d92a02ff8cc5>", line 1, in <cell line: 1>
        type(agent).fitness_func(reward=None, steps=34, done=True, env=None, agent=None)
      File "/home/michele.baldo/memory-evolution/training_egocentric.py", line 244, in fitness_func
        ...
    NameError: name 'ff_time' is not defined
    '''
    # test me:
    _prev_ff = None
    for LOAD_AGENT in agents_tags:
        print(LOAD_AGENT)
        _Phenotype = type(agents[LOAD_AGENT])

        # TODO: tmp, just to be sure the right phenotype was loaded in the experiments
        assert _Phenotype is memory_evolution.agents.ConstantSpeedRnnNeatAgent, _Phenotype
        assert _Phenotype.fitness_func is not memory_evolution.agents.BaseNeatAgent.fitness_func, 'tmp, it is okay if it fails when you change stuffs'
        assert _Phenotype.eval_num_episodes is not memory_evolution.agents.BaseNeatAgent.eval_num_episodes, 'tmp, it is okay if it fails when you change stuffs'
        assert _Phenotype.fitness_func is not memory_evolution.agents.BaseNeatAgent.fitness_func, 'tmp, it is okay if it fails when you change stuffs'
        assert _Phenotype.eval_num_episodes == 20, 'tmp, it is okay if it fails when you change stuffs'

        # FIXME: super shortcut because the previous patch don't work anymore (NameError: name 'ff_time' is not defined)
        ff_time = memory_evolution.evaluate.fitness_func_time_minimize
        min_ff_time = ff_time(reward=None, steps=env.max_steps, done=None, env=None, agent=None)
        # user defined fitness_func (outside any module, just in main) so it can be pickled with dill.
        def fitness_func(*, reward, steps, done, env, agent, **kwargs) -> float:
            ft = ff_time(reward=reward, steps=steps, done=done, env=env, agent=agent, **kwargs)
            assert min_ff_time <= ft <= ff_time.max
            fitness = ft
            return fitness
        fitness_func.min = min_ff_time
        fitness_func.max = ff_time.max
        _Phenotype.fitness_func = fitness_func
    # ---

    # ----- ANALYSE -----

    # --- debugging ---
    # # Select just one agent to see stuffs (for DEBUGGING):
    # LOAD_AGENT, agent = next(iter(agents.items()))  # agents['LOAD_AGENT']  # select which agent to inspect
    # agent.genome.connections
    # agent.genome.nodes
    # LOAD_STATS = os.path.join(LOAD_AGENT_DIR, LOAD_AGENT + '_stats.pickle')
    # with open(LOAD_STATS, "rb") as f:
    #     stats = pickle.load(f)
    # sys.exit()

    # --- evaluate and render ---
    SHOW_JUST_ONE_AGENT = True
    RENDER_ALL_AGAIN = False
    # Show just one agent:
    if SHOW_JUST_ONE_AGENT and N_EPISODES > 0:
        LOAD_AGENT, agent = next(iter(agents.items()))  # agents['LOAD_AGENT']  # select which agent to evaluate
        print(f'Evaluating agent {LOAD_AGENT} ...\n')
        evaluate_agent(agent, env, episodes=N_EPISODES, render=True, save_gif=False)
        # evaluate_agent(agent, env, episodes=N_EPISODES, render=RENDER,
        #                save_gif=True,
        #                save_gif_name=os.path.join(LOGGING_DIR, LOADED_DIR_TAG_UTCNOW + '__LOAD_AGENT_' + LOAD_AGENT + '_frames.gif'))
        # # Note: if you run twice evaluate_agent with the same name it will overwrite the previous gif
        # #   (but if save_gif_dir is provided it will raise an error because the directory already exists).
    if RENDER_ALL_AGAIN and N_EPISODES > 0:
        print('Rendering agents ...\n')
        for LOAD_AGENT, agent in agents.items():
            evaluate_agent(agent, env, episodes=N_EPISODES, render=RENDER,
                           save_gif=True,
                           save_gif_name=os.path.join(LOGGING_DIR, LOAD_AGENT + '_LOADED_AGENT__' + LOADED_DIR_TAG_UTCNOW + '_frames.gif'))
            # Note: if you run twice evaluate_agent with the same name it will overwrite the previous gif
            #   (but if save_gif_dir is provided it will raise an error because the directory already exists).

    # --- loading results (json) ---
    LOAD_ALL_RESULTS = os.path.join(LOAD_AGENT_DIR, LOAD_DIR_TAG + '_all_results.csv')
    ALL_RESULTS_DESCRIPTION = os.path.join(LOAD_AGENT_DIR, LOAD_DIR_TAG + '_all_results_description.csv')
    ALL_RESULTS_PLOT = os.path.join(LOAD_AGENT_DIR, LOAD_DIR_TAG + '_bar_plot_with_error_bars.png')
    reloaded = False
    if os.path.exists(LOAD_ALL_RESULTS):
        assert os.path.exists(ALL_RESULTS_DESCRIPTION), ALL_RESULTS_DESCRIPTION
        all_results = pd.read_csv(LOAD_ALL_RESULTS, index_col=0)
    else:
        reloaded = True
        all_results = []
        all_results_index = []
        for LOAD_AGENT in agents_tags:
            with open(os.path.join(LOAD_AGENT_DIR, LOAD_AGENT + '_results.json'), 'r') as f:
                results = json.load(f)
            pprint(results, sort_dicts=False)
            # results["BestGenome"]["test_agent_first_arm_accuracy"]
            # results["BestGenome"]["test_agent_target_reached_rate"]
            # results["BestGenome"]["Fitness"]
            all_results.append(results["BestGenome"])
            all_results_index.append(LOAD_AGENT)
        all_results = pd.DataFrame(all_results, index=all_results_index)
        all_results.to_csv(LOAD_ALL_RESULTS)
        all_results.describe().to_csv(ALL_RESULTS_DESCRIPTION)
    if os.path.exists(ALL_RESULTS_PLOT):
        assert not reloaded, "data are reloaded, but old plot is still there"
    else:
        print("Plotting...")
        plot_accuracy_results(all_results, view=False, filename=ALL_RESULTS_PLOT)
    print(all_results)
    # print(all_results.info())
    print(all_results.describe())
    print()

    # --- stats and genome stats on evolution visualization ---
    BEST_FITNESS_PLOT = os.path.join(LOAD_AGENT_DIR, LOAD_DIR_TAG + '_best_fitness.png')
    BEST_GENOME_METRICS = os.path.join(LOAD_AGENT_DIR, LOAD_DIR_TAG + '_genome_metrics.png')
    reloaded = False
    if os.path.exists(BEST_FITNESS_PLOT + '.csv'):
        assert os.path.exists(BEST_GENOME_METRICS + '_nodes.csv'), BEST_GENOME_METRICS
        assert os.path.exists(BEST_GENOME_METRICS + '_connections.csv'), BEST_GENOME_METRICS
        best_fitness = pd.read_csv(BEST_FITNESS_PLOT + '.csv', index_col=0)
        nodes = pd.read_csv(BEST_GENOME_METRICS + '_nodes.csv', index_col=0)
        connections = pd.read_csv(BEST_GENOME_METRICS + '_connections.csv', index_col=0)
        best_fitness.columns = best_fitness.columns.map(int)
        nodes.columns = nodes.columns.map(int)
        connections.columns = connections.columns.map(int)
    else:
        reloaded = True
        assert not os.path.exists(BEST_GENOME_METRICS + '_nodes.csv'), BEST_GENOME_METRICS
        assert not os.path.exists(BEST_GENOME_METRICS + '_connections.csv'), BEST_GENOME_METRICS
        # LOAD_ALL_STATS = os.path.join(LOAD_AGENT_DIR, LOAD_DIR_TAG + '_all_stats.pickle')
        # NOTE: all stats are too heavy to be loaded and pickled all together (but okay to be loaded only)
        print("Loading and working with stats...")
        all_stats = {}
        for LOAD_AGENT in tqdm(agents_tags):
            LOAD_STATS = os.path.join(LOAD_AGENT_DIR, LOAD_AGENT + '_stats.pickle')

            with open(LOAD_STATS, "rb") as f:
                # print('Loading stats...')
                stats = pickle.load(f)
            # print(stats)
            assert len(stats.generation_statistics) == len(stats.most_fit_genomes)
            assert stats.best_genome() is max(stats.most_fit_genomes, key=lambda x: x.fitness)
            assert all([(max(_genome_fitness
                             for _specie in _species.values()
                             for _genome_fitness in _specie.values()) == _best_genome.fitness)
                        for _species, _best_genome in zip(stats.generation_statistics, stats.most_fit_genomes)])
            all_stats[LOAD_AGENT] = stats

            # agent.visualize_evolution(stats, stats_ylog=False, view=True,
            #                           filename_stats=os.path.join(LOGGING_DIR, LOADED_DIR_TAG_UTCNOW + '__LOAD_AGENT_' + LOAD_AGENT + "_fitness.png"),
            #                           filename_speciation=os.path.join(LOGGING_DIR, LOADED_DIR_TAG_UTCNOW + '__LOAD_AGENT_' + LOAD_AGENT + "_speciation.png"))

        # print(all_stats)
        best_fitness = get_best_fitness(all_stats, filename=BEST_FITNESS_PLOT)
        nodes, connections = get_genome_metrics(all_stats, filename=BEST_GENOME_METRICS)
    if os.path.exists(BEST_FITNESS_PLOT):
        assert not reloaded, "data are reloaded, but old plot is still there"
    else:
        agent = next(iter(agents.values()))
        ylim = (type(agent).fitness_func.min, type(agent).fitness_func.max)
        # ylim = (-400, 0)
        assert len(ylim) == 2, ylim
        assert ylim[0] <= ylim[1], ylim
        # ylim_range = ylim[1] - ylim[0]
        # offset = ylim_range * .03
        # ylim = (ylim[0] - offset, ylim[1] + offset)
        plot_best_fitness(best_fitness,
                          view=False,
                          # filename=None,
                          filename=BEST_FITNESS_PLOT,
                          ylim=ylim)
    if os.path.exists(BEST_GENOME_METRICS):
        assert not reloaded, "data are reloaded, but old plot is still there"
    else:
        plot_genome_metrics(nodes, connections,
                            view=False,  # ylim=(0,1),
                            # filename=None,
                            filename=BEST_GENOME_METRICS)

    # --- genome visualization ---
    for LOAD_AGENT in agents_tags:
        agent = agents[LOAD_AGENT]
        config = configs[LOAD_AGENT]

        obs_shape = env.observation_space.shape
        obs_size = reduce(mul, obs_shape, 1)
        input_nodes = config.genome_config.input_keys
        assert obs_size == len(input_nodes) == len(agent.phenotype.input_nodes), (
            obs_shape, len(input_nodes), len(agent.phenotype.input_nodes))
        assert input_nodes == agent.phenotype.input_nodes
        # agent.visualize_genome(agent.genome, view=True, name='Genome',
        #                        default_input_node_color='palette',
        #                        filename=os.path.join(LOGGING_DIR, LOADED_DIR_TAG_UTCNOW + '__LOAD_AGENT_' + LOAD_AGENT + "_genome.gv"),
        #                        format='svg',
        #                        show_palette=True)

        # --- do stuff with stats and genomes ---
        pass
    # env.close(); sys.exit()

    # --- test again all agents ---
    ACCURACY_TRIALS = 200
    NEW_NUMBER = 0
    NEW_RESULTS = os.path.join(LOGGING_DIR, LOAD_DIR_TAG + f'_new{NEW_NUMBER}_results.csv')
    NEW_RESULTS_DESCRIPTION = os.path.join(LOGGING_DIR, LOAD_DIR_TAG + f'_new{NEW_NUMBER}_results_description.csv')
    NEW_RESULTS_PLOT = os.path.join(LOGGING_DIR, LOAD_DIR_TAG + f'_new{NEW_NUMBER}_results_bar_plot.png')
    if os.path.exists(NEW_RESULTS):
        assert os.path.exists(NEW_RESULTS_DESCRIPTION), NEW_RESULTS_DESCRIPTION
        new_results = pd.read_csv(NEW_RESULTS, index_col=0)
    else:
        new_results = []
        print('Test again agents ...\n')
        new_fitnesses = {}
        new_first_arm_accuracies = {}
        new_target_reached_rates = {}
        for LOAD_AGENT, agent in tqdm(agents.items()):

            # note, fitness is measured across few episodes, but you have a lot of agents.
            new_fitnesses[LOAD_AGENT] = agent.eval_genome(agent.genome, agent.config)
            print(f"New fitness: {new_fitnesses[LOAD_AGENT]}")

            # testing the agent first arm accuracy:
            accuracy = memory_evolution.evaluate.test_agent_first_arm_accuracy(
                agent, env, episodes=ACCURACY_TRIALS,
                render=False)
            print(f"test_agent_first_arm_accuracy (n={ACCURACY_TRIALS}): {accuracy}")
            new_first_arm_accuracies[LOAD_AGENT] = accuracy

            # test general target-reached rate (to discriminate bad v.s. border-follower v.s. allocentric/egocentric successful agents):
            target_reached_rate = memory_evolution.evaluate.test_agent_target_reached_rate(
                agent, env, episodes=ACCURACY_TRIALS,
                render=False)
            print(f"test_agent_target_reached_rate (n={ACCURACY_TRIALS}): {target_reached_rate}")
            new_target_reached_rates[LOAD_AGENT] = target_reached_rate

        print(new_first_arm_accuracies)
        print(new_target_reached_rates)
        print(new_fitnesses)
        new_results = {
            'test_agent_first_arm_accuracy': new_first_arm_accuracies,
            'test_agent_target_reached_rate': new_target_reached_rates,
            'Fitness': new_fitnesses,
        }
        new_results = pd.DataFrame(new_results)
        # print(new_results)
        new_results.to_csv(NEW_RESULTS)
        new_results.describe().to_csv(NEW_RESULTS_DESCRIPTION)
    if not os.path.exists(NEW_RESULTS_PLOT):
        print("Plotting...")
        plot_accuracy_results(new_results, view=True, filename=NEW_RESULTS_PLOT)
    print(new_results)
    # print(new_results.info())
    print(new_results.describe())
    print()

    # --- closing ---
    print('\n')
    env.close()

