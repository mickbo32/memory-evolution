from collections.abc import Sequence
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

from analyse_results import plot_bars, plot_avg_df


def plot_accuracy_results(dfs_results: Sequence[pd.DataFrame],
                          x_lables,
                          view=True, filename=None, ylim=None, color=None):
    assert len(dfs_results) >= 1
    index = dfs_results[0].index
    columns = dfs_results[0].columns
    features = len(columns)
    assert len(x_lables) == len(dfs_results), len(x_lables)

    if ylim is None:
        ylim = [None] * features
    elif len(ylim) != features:
        raise ValueError("'ylim' should have a lim for each column")

    assert all(len(index) == len(df.index) for df in dfs_results[1:])
    # assert all(not np.array_equal(index, df.index) for df in dfs_results[1:]), [df.index for df in dfs_results]
    experiments = np.arange(len(index))

    for df in dfs_results[1:]:
        np.testing.assert_array_equal(columns, df.columns)

    prev_labels = ['test_agent_first_arm_accuracy', 'test_agent_target_reached_rate', 'Fitness']
    new_labels = ['first arm accuracy', 'target reached rate', 'Best fitness']
    for all_results in dfs_results:
        np.testing.assert_array_equal(all_results.columns, prev_labels)
        # all_results.columns = new_labels
        # all_results.reset_index(drop=True, inplace=True)

    fig, axes = plt.subplots(1, features, figsize=(6.4 * 1.6, 4.8))
    x = np.arange(len(columns))
    for j, col, y_label in zip(range(features), columns, new_labels):
        results = pd.DataFrame({x_label: df[col].reset_index(drop=True)
                                for x_label, df in zip(x_lables, dfs_results)})
        results.name = y_label
        print(results.name)
        print(results)
        print()

        plot_bars(results, axes[j], ylabel=results.name, color=color, ylim=ylim[j], std=False, box=True)
    fig.suptitle(f"Results (averaged across {len(experiments)} evolution processes)")

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    if view:
        plt.show(block=True)

    plt.close()


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
    DIR_TAG, ALLO_DIR_TAG, EGO_DIR_TAG = 'exp90', '2022-06-13_training_allocentric_90', '2022-06-13_training_egocentric_90'
    # DIR_TAG, ALLO_DIR_TAG, EGO_DIR_TAG = 'exp30_same-prob', '2022-06-16T173323_allo30_same-prob_600epochs', '2022-06-23T073841_ego30_same-prob_600epochs'
    # DIR_TAG, ALLO_DIR_TAG, EGO_DIR_TAG = 'exp30_unconnected', '2022-06-20T085938_allo30_unconnected', '2022-06-22T205534_ego30_unconnected'
    ALLO_AGENT_DIR = f"logs/saved_logs/outputs-link/{ALLO_DIR_TAG}/logs/"
    EGO_AGENT_DIR = f"logs/saved_logs/outputs-link/{EGO_DIR_TAG}/logs/"
    LOGGING_DIR = 'logs'

    # logging settings:
    LOGGING_DIR, UTCNOW = set_main_logger(file_handler_all=None,
                                          logging_dir=LOGGING_DIR,
                                          stdout_handler=logging.INFO,# - 2,
                                          file_handler_now_filename_fmt="LOADED_DIR___now_{utcnow}.log")
    logging.info(__file__)
    LOADED_DIR_TAG_UTCNOW = 'LOADED_DIR__now_' + UTCNOW

    logging.info(repr((ALLO_DIR_TAG, EGO_DIR_TAG)))

    # neat random seeding:
    # random.seed(42)
    logging.debug(random.getstate())
    # Use random.setstate(state) to set an old state, where 'state' have been obtained from a previous call to getstate().

    # ----- LOAD -----
    pass

    # ----- ANALYSE -----

    # --- loading results (json) ---
    ALLO_LOAD_ALL_RESULTS = os.path.join(ALLO_AGENT_DIR, ALLO_DIR_TAG + '_all_results.csv')
    allo_all_results = pd.read_csv(ALLO_LOAD_ALL_RESULTS, index_col=0)
    EGO_LOAD_ALL_RESULTS = os.path.join(EGO_AGENT_DIR, EGO_DIR_TAG + '_all_results.csv')
    ego_all_results = pd.read_csv(EGO_LOAD_ALL_RESULTS, index_col=0)
    print(allo_all_results)
    print(ego_all_results)
    print(allo_all_results.info())
    print(ego_all_results.info())
    print(allo_all_results.describe())
    print(ego_all_results.describe())

    ALLO_RESULTS_PLOT = os.path.join(LOGGING_DIR, LOADED_DIR_TAG_UTCNOW + '__' + ALLO_DIR_TAG + '_bar_plot.png')
    EGO_RESULTS_PLOT = os.path.join(LOGGING_DIR, LOADED_DIR_TAG_UTCNOW + '__' + EGO_DIR_TAG + '_bar_plot.png')
    # plot_accuracy_results(allo_all_results, view=True, filename=ALLO_RESULTS_PLOT)
    # plot_accuracy_results(ego_all_results, view=True, filename=EGO_RESULTS_PLOT)
    plot_accuracy_results((allo_all_results, ego_all_results),
                          x_lables=('Allocentric task', 'Egocentric task'),
                          view=True,
                          filename=os.path.join(LOGGING_DIR, LOADED_DIR_TAG_UTCNOW + '_' + DIR_TAG + '_results.png'),
                          ylim=((-.05, 1.05), (-.05, 1.05), (-400, -0)),
                          color=('#999999', 'lightblue'))
    print('\n')
    sys.exit()

    # --- stats and genome stats on evolution visualization ---
    ALLO_BEST_FITNESS_PLOT = os.path.join(ALLO_AGENT_DIR, ALLO_DIR_TAG + '_best_fitness.png')
    ALLO_BEST_GENOME_METRICS = os.path.join(ALLO_AGENT_DIR, ALLO_DIR_TAG + '_genome_metrics.png')
    EGO_BEST_FITNESS_PLOT = os.path.join(EGO_AGENT_DIR, EGO_DIR_TAG + '_best_fitness.png')
    EGO_BEST_GENOME_METRICS = os.path.join(EGO_AGENT_DIR, EGO_DIR_TAG + '_genome_metrics.png')
    allo_best_fitness = pd.read_csv(ALLO_BEST_FITNESS_PLOT + '.csv', index_col=0)
    allo_best_fitness.index.astype(int)
    allo_best_fitness.columns.astype(int)
    allo_nodes = pd.read_csv(ALLO_BEST_GENOME_METRICS + '_nodes.csv', index_col=0)
    allo_connections = pd.read_csv(ALLO_BEST_GENOME_METRICS + '_connections.csv', index_col=0)
    ego_best_fitness = pd.read_csv(EGO_BEST_FITNESS_PLOT + '.csv', index_col=0)
    ego_nodes = pd.read_csv(EGO_BEST_GENOME_METRICS + '_nodes.csv', index_col=0)
    ego_connections = pd.read_csv(EGO_BEST_GENOME_METRICS + '_connections.csv', index_col=0)
    allo_best_fitness.columns = allo_best_fitness.columns.map(int)
    allo_nodes.columns = allo_nodes.columns.map(int)
    allo_connections.columns = allo_connections.columns.map(int)
    ego_best_fitness.columns = ego_best_fitness.columns.map(int)
    ego_nodes.columns = ego_nodes.columns.map(int)
    ego_connections.columns = ego_connections.columns.map(int)
    print(allo_best_fitness)
    print(allo_nodes)
    print(allo_connections)
    print(allo_best_fitness.info())
    print(allo_nodes.info())
    print(allo_connections.info())
    # print(allo_best_fitness.describe())
    # print(allo_nodes.describe())
    # print(allo_connections.describe())
    print('\n')

    ylim = (-400, 0)
    assert len(ylim) == 2, ylim
    assert ylim[0] <= ylim[1], ylim
    # ylim_range = ylim[1] - ylim[0]
    # offset = ylim_range * .03
    # ylim = (ylim[0] - offset, ylim[1] + offset)
    plot_best_fitness(allo_best_fitness,
                      view=True,
                      # filename=None,
                      filename=os.path.join(LOGGING_DIR, LOADED_DIR_TAG_UTCNOW + '__' + ALLO_DIR_TAG + '_best_fitness.png'),
                      ylim=ylim)
    plot_best_fitness(ego_best_fitness,
                      view=True,
                      # filename=None,
                      filename=os.path.join(LOGGING_DIR, LOADED_DIR_TAG_UTCNOW + '__' + EGO_DIR_TAG + '_best_fitness.png'),
                      ylim=ylim)

    plot_genome_metrics(allo_nodes, allo_connections,
                        view=True,  # ylim=(0,1),
                        # filename=None,
                        filename=os.path.join(LOGGING_DIR, LOADED_DIR_TAG_UTCNOW + '__' + ALLO_DIR_TAG + '_genome_metrics.png'))
    plot_genome_metrics(ego_nodes, ego_connections,
                        view=True,  # ylim=(0,1),
                        # filename=None,
                        filename=os.path.join(LOGGING_DIR, LOADED_DIR_TAG_UTCNOW + '__' + EGO_DIR_TAG + '_genome_metrics.png'))
    print('\n')

    # --- closing ---
    print('\n')

