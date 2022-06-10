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


def plot_avg_df(df, ax=None, avg_label="average", avg_color='b', show_maxminstd_label=True):
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


def plot_best_fitness(all_stats, ylog=False, view=False, filename=None, ylim=None):
    """ Plots the populations' best fitness over generations.

    If ``filename`` is None, don't save.
    """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

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

    # fig, ax = plt.subplots(1)
    # avg_best_fitness = best_fitness.mean(axis=0)
    # ax.plot(generation, avg_best_fitness, 'b-', label="average best fitness")
    # ax.plot(generation, best_fitness.max(axis=0), '--', color='gray', label="max")
    # ax.plot(generation, best_fitness.min(axis=0), '--', color='gray', label="min")
    # ax.fill_between(generation,
    #                 avg_best_fitness - best_fitness.std(axis=0), avg_best_fitness + best_fitness.std(axis=0),
    #                 facecolor='yellow', alpha=0.5, label='\u00B11 std')
    ax = plot_avg_df(best_fitness, avg_label="average best fitness")

    plt.title(f"Population's best fitness (averaged across {len(all_stats)} evolution processes)")
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
        plt.show()

    plt.close()


def plot_genome_metrics(all_stats, ylog=False, view=False, filename=None, ylim=None):
    """ Plots the populations' best fitness over generations.

    If ``filename`` is None, don't save.
    """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

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

    fig, ax = plt.subplots(1)
    plot_avg_df(nodes, ax=ax, avg_label="average nodes (without inputs)", avg_color='g', show_maxminstd_label=False)
    plot_avg_df(connections, ax=ax, avg_label="average connections", avg_color='b')

    plt.title(f"Population's best genome (averaged across {len(all_stats)} evolution processes)")
    ax.set_xlabel("Generations")
    ax.set_ylabel("Number of")
    ax.grid()
    ax.legend(loc="best")

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    if view:
        plt.show()

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
    RENDER = True  # False  # render or just save gif files
    # ---
    LOAD_DIR_TAG = '2022-06-08T172246_training_allocentric_100'
    LOAD_AGENT_DIR = f"logs/saved_logs/outputs-link/{LOAD_DIR_TAG}/logs/"
    LOAD_FROM: AVAILABLE_LOADING_METHODS = 'pickle'
    N_EPISODES = 0  # 5
    LOGGING_DIR = 'logs'

    # logging settings:
    assert LOAD_FROM != 'checkpoint'
    LOGGING_DIR, UTCNOW = set_main_logger(file_handler_all=None,
                                          logging_dir=LOGGING_DIR,
                                          stdout_handler=logging.INFO - 2,
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
    env = load_env(agents_tags[0], LOAD_AGENT_DIR)

    agents = {}
    configs = {}
    for LOAD_AGENT in agents_tags:
        agent, other_loads = load_agent(LOAD_AGENT, LOAD_AGENT_DIR, LOAD_FROM)
        agent.set_env(env)
        agents[LOAD_AGENT] = agent
        configs[LOAD_AGENT] = other_loads['config']
    del LOAD_AGENT, agent, other_loads

    # ----- MAIN -----

    # # Select just one agent to see stuffs (for DEBUGGING):
    # LOAD_AGENT, agent = next(iter(agents.items()))  # agents['LOAD_AGENT']  # select which agent to inspect
    # agent.genome.connections
    # agent.genome.nodes
    # LOAD_STATS = os.path.join(LOAD_AGENT_DIR, LOAD_AGENT + '_stats.pickle')
    # with open(LOAD_STATS, "rb") as f:
    #     stats = pickle.load(f)
    # sys.exit()

    if N_EPISODES > 0:
        LOAD_AGENT, agent = next(iter(agents.items()))  # agents['LOAD_AGENT']  # select which agent to evaluate
        print('Evaluating agent ...\n')
        evaluate_agent(agent, env, episodes=N_EPISODES, render=RENDER,
                       save_gif=True,
                       save_gif_name=os.path.join(LOGGING_DIR, LOADED_DIR_TAG_UTCNOW + '__LOAD_AGENT_' + LOAD_AGENT + '_frames.gif'))
        # Note: if you run twice evaluate_agent with the same name it will overwrite the previous gif
        #   (but if save_gif_dir is provided it will raise an error because the directory already exists).

    # loading results:
    LOAD_ALL_RESULTS = os.path.join(LOAD_AGENT_DIR, LOAD_DIR_TAG + '_all_results.csv')
    LOAD_ALL_RESULTS_DESCRIBTION = os.path.join(LOAD_AGENT_DIR, LOAD_DIR_TAG + '_all_results_description.csv')
    if os.path.exists(LOAD_ALL_RESULTS):
        assert os.path.exists(LOAD_ALL_RESULTS_DESCRIBTION)
        all_results = pd.read_csv(LOAD_ALL_RESULTS, index_col=0)
    else:
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
        all_results.describe().to_csv(LOAD_ALL_RESULTS_DESCRIBTION)
    print(all_results)
    # print(all_results.info())
    print(all_results.describe())
    print()

    # --- stats and genome visualization ---
    BEST_FITNESS_PLOT = os.path.join(LOAD_AGENT_DIR, LOAD_DIR_TAG + '_best_fitness.png')
    BEST_GENOME_METRICS = os.path.join(LOAD_AGENT_DIR, LOAD_DIR_TAG + '_genome_metrics.png')
    if not os.path.exists(BEST_FITNESS_PLOT):
        assert not os.path.exists(BEST_GENOME_METRICS)
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
        agent = next(iter(agents.values()))
        try:
            ylim = (type(agent).fitness_func.min, type(agent).fitness_func.max)
        except AttributeError as err:
            warnings.warn(f"{type(err).__qualname__}: {err.args}")
            ylim = (-400, 0)
        assert len(ylim) == 2, ylim
        assert ylim[0] <= ylim[1], ylim
        # ylim_range = ylim[1] - ylim[0]
        # offset = ylim_range * .03
        # ylim = (ylim[0] - offset, ylim[1] + offset)
        plot_best_fitness(all_stats, view=True,
                          # filename=None,
                          filename=BEST_FITNESS_PLOT,
                          ylim=ylim)

        plot_genome_metrics(all_stats, view=True,
                            # filename=None,
                            filename=BEST_GENOME_METRICS,
                            ylim=ylim)

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

    print('\n')
    env.close()

