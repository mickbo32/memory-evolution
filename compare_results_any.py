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
from compare_results import plot_accuracy_results


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
    LOGGING_DIR = 'logs'
    TAG = 'allo90_without_landmarks'
    CSVS = [
        'logs/saved_logs/imgs-link/allo90/without_landmarks/2022-06-13_training_allocentric_90_new0_results.csv',  # 'allo90_without_landmarks_east'
        'logs/saved_logs/imgs-link/allo90/without_landmarks/2022-06-13_training_allocentric_90_new1_results.csv',  #  'allo90_without_landmarks_west'
        'logs/saved_logs/imgs-link/allo90/without_landmarks/2022-06-13_training_allocentric_90_new2_results.csv',  # 'allo90_without_landmarks_south'
    ]
    x_labels = ('Target\nEast', 'Target\nWest', 'Target\nSouth')
    # TAG = 'ego90_without_landmarks'
    # CSVS = ['logs/saved_logs/imgs-link/ego90/without_landmarks/2022-06-13_training_egocentric_90_new0_results.csv']  # 'ego90_without_landmarks_east'
    # x_labels = ('Target\nEast',)

    # logging settings:
    LOGGING_DIR, UTCNOW = set_main_logger(file_handler_all=None,
                                          logging_dir=LOGGING_DIR,
                                          stdout_handler=logging.INFO,# - 2,
                                          file_handler_now_filename_fmt="LOADED_DIR___now_{utcnow}.log")
    logging.info(__file__)
    LOADED_DIR_TAG_UTCNOW = 'LOADED_DIR__now_' + UTCNOW

    logging.info(TAG)
    logging.info(CSVS)

    # neat random seeding:
    # random.seed(42)
    logging.debug(random.getstate())
    # Use random.setstate(state) to set an old state, where 'state' have been obtained from a previous call to getstate().

    # ----- LOAD -----
    pass

    # ----- ANALYSE -----

    # --- loading results (json) ---
    dfs_results = []
    for csv in CSVS:
        results = pd.read_csv(csv, index_col=0)
        dfs_results.append(results)
        print(results)
        print(results.info())
        print(results.describe())

    plot_accuracy_results(dfs_results,
                          x_lables=x_labels,
                          view=True,
                          filename=os.path.join(LOGGING_DIR, LOADED_DIR_TAG_UTCNOW + '_' + TAG + '_results.png'),
                          ylim=((-.05, 1.05), (-.05, 1.05), (-405, -0)),  # ((0, 1.05), (0, 1.05), (-400, -0)),  # (None, None, (-100, -50)),,  # (None, None, (-400, -0)),
                          color=('#999999', 'lightblue', 'lightgreen'))
    print('\n')


    # --- closing ---
    print('\n')

