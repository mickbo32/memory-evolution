import dill  # pickle extension
import json
import logging
import multiprocessing
import os
import pickle
from pprint import pprint
import random  # neat uses random  # todo: allow seeding in neat
import re
import shutil
import sys
import time
from typing import Optional

import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import neat
import numpy as np
import pygame as pg
from numpy.random import SeedSequence
import pandas as pd

from gym.utils.env_checker import check_env  # from stable_baselines.common.env_checker import check_env

import memory_evolution
from memory_evolution.agents import RandomActionAgent, RnnNeatAgent, CtrnnNeatAgent, ConstantSpeedRnnNeatAgent
from memory_evolution.envs import BaseForagingEnv, MazeForagingEnv, TMaze, RadialArmMaze
from memory_evolution.evaluate import evaluate_agent
from memory_evolution.logging import set_main_logger
from memory_evolution.utils import COLORS

# matplotlib settings:
isRunningInPyCharm = "PYCHARM_HOSTED" in os.environ
if isRunningInPyCharm:
    mpl.use('Qt5Agg')  # Change matplotlib backend to show correctly in PyCharm.


if __name__ == '__main__':

    EPOCHS = 300

    # parse command-line arguments passed to the program:
    JOB_ID = ''  # type: str
    if len(sys.argv) == 1:  # local execution
        pass
    elif len(sys.argv) == 2:  # remote execution

        # remote execution,
        # JOB_ID should be passed as argument to the program when running it on the remote cluster server.
        JOB_ID = str(sys.argv[1])
        # match = re.match(r"^([0-9]+).hpc-head-n1.unitn.it$", JOB_ID)
        match = re.match(
            r"^(?P<JOB_ID>[0-9]+)(?P<_PBS_ARRAY_INDEX>(?:\[(?P<PBS_ARRAY_INDEX>[0-9]+)])?).hpc-head-n1.unitn.it(?P=_PBS_ARRAY_INDEX)$",
            JOB_ID)
        assert 'tag' not in locals()
        assert 'PBS_ARRAY_INDEX' not in locals()
        if match:
            # JOB_ID = match.group(1)  # type: str
            _job_id = match['JOB_ID']
            _pbs_array_index = match['PBS_ARRAY_INDEX']
            JOB_ID = _job_id
            if _pbs_array_index is not None:
                JOB_ID += '-' + _pbs_array_index
        else:
            raise AssertionError(JOB_ID)
        assert isinstance(JOB_ID, str), type(JOB_ID)

        # remote execution, no input devices.
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_MOUSEDRIVER'] = 'dummy'
        """
        # Alternatively,
        # Environment variables can be set in the script which calls this program:
        #...
        echo "Starting (PBS_JOBID=${PBS_JOBID}) ..."
        source ~/miniconda3/bin/activate evo
        python --version
        export SDL_AUDIODRIVER='dummy'
        export SDL_VIDEODRIVER='dummy'
        export SDL_MOUSEDRIVER='dummy'
        python memory-evolution/main.py "${PBS_JOBID}"
        """
    else:
        raise RuntimeError(sys.argv)

    # ----- Settings -----

    # logging settings:
    logging_dir, UTCNOW = set_main_logger(file_handler_all=None,
                                          stdout_handler=logging.INFO - 1,  # logging.INFO,
                                          file_handler_now=logging.DEBUG + 5,  # +5 otherwise the log file is too heavy.
                                          file_handler_now_filename_fmt=JOB_ID + '_' * bool(JOB_ID) + "{utcnow}.log")
    logging.info(__file__)

    # if job_id is passed to the program, use it in the log tag:
    if JOB_ID:
        LOG_TAG = JOB_ID + '_' + UTCNOW
    else:
        LOG_TAG = UTCNOW
    logging.info('TAG: ' + LOG_TAG)
    
    # get some stats:
    version_msg = f"Python version\n{sys.version}\nVersion info\n{sys.version_info}\n"
    logging.info(version_msg)
    cpu_count = multiprocessing.cpu_count()
    logging.info(f"CPU count: {cpu_count}\n")
    cwd = os.getcwd()
    logging.info(f"Current working directory: {cwd!r}\n")
    
    # neat random seeding:
    logging.debug(random.getstate())
    # Use random.setstate(state) to set an old state, where 'state' have been obtained from a previous call to getstate().

    # ----- ENVIRONMENT -----

    # env = RadialArmMaze(corridor_width=.2,
    #                     window_size=200, seed=42, agent_size=.075, food_size=.05, n_food_items=1, max_steps=400,
    #                     init_agent_position=(.5, .1), init_food_positions=((.9, .5),),
    #                     vision_depth=.2, vision_field_angle=135, vision_resolution=8)

    max_steps = 400  # max_steps=100
    corridor_width = .2
    landmark_size = .25  # .15
    lm_dist = 1. / 2  # corridor_width + landmark_size * 1.10
    lm_bord = 1. / 4  # landmark_size / 2 + .1
    env = RadialArmMaze(corridor_width=corridor_width,
                        window_size=200, agent_size=.075, food_size=.05, n_food_items=1, max_steps=max_steps,
                        # vision_depth=.2, vision_field_angle=135, vision_resolution=7,
                        # vision_depth=.2, vision_field_angle=135, vision_resolution=4,
                        # vision_channels=3, vision_point_radius=.025,
                        # vision_depth=.25, vision_field_angle=135, vision_resolution=3,
                        # vision_channels=3, vision_point_radius=.05,
                        vision_depth=.2, vision_field_angle=135, vision_resolution=3,
                        vision_channels=3, vision_point_radius=.04,
                        # vision_depth=.2, vision_field_angle=135, vision_resolution=3,
                        # vision_channels=3, vision_point_radius=.035,
                        agent_color=COLORS['cyan'],
                        background_color=np.asarray((0, 0, 0), dtype=np.uint8),
                        outside_color=np.asarray((255, 0, 0), dtype=np.uint8),
                        food_color=np.asarray((0, 200, 55), dtype=np.uint8), food_visible=False,
                        random_init_agent_position=((.5, .1), (.5, .9), (.1, .5), (.9, .5),),
                        pairing_init_food_positions=(((.9, .5),), ((.1, .5),), ((.5, .1), ), ((.5, .9),),),
                        landmark_size=landmark_size,
                        init_landmarks_positions=((.5 - lm_dist / 2, lm_bord), (.5 + lm_dist / 2, lm_bord),
                                                  (.5 - lm_dist / 2, 1. - lm_bord), (.5 + lm_dist / 2, 1. - lm_bord),),
                        landmarks_colors=(
                            np.asarray((255, 0, 255), dtype=np.uint8), np.asarray((255, 255, 0), dtype=np.uint8),
                            np.asarray((255, 127, 127), dtype=np.uint8), np.asarray((255, 255, 255), dtype=np.uint8),
                        ),
                        )
    # env = TMaze(
    #                     corridor_width=corridor_width,
    #                     window_size=200, agent_size=.075, food_size=.05, n_food_items=1, max_steps=400,
    #                     vision_depth=.2, vision_field_angle=135, vision_resolution=3,
    #                     vision_channels=1, vision_point_radius=.04,
    #                     agent_color=COLORS['cyan'],
    #                     background_color=np.asarray((0, 0, 0), dtype=np.uint8),
    #                     outside_color=np.asarray((255, 0, 0), dtype=np.uint8),
    #                     food_color=np.asarray((0, 200, 55), dtype=np.uint8), food_visible=False,
    #                     random_init_agent_position=((.1, .9), (.5, .1),),
    #                     init_food_positions=((target_pos := (.9, .9)),),
    #                     )

    logging.info(f"Env: {type(env).__qualname__}")
    logging.info(f"observation_space: "
                 f"{env.observation_space.shape} "
                 f"{np.asarray(env.observation_space.shape).prod()}")
    # picKle env:
    with open(os.path.join(logging_dir, LOG_TAG + '_env.pickle'), "wb") as f:
        pickle.dump(env, f)
    # check pickle env:  # todo: move in tests
    def assert_init_params_equal(_init_params_1, _init_params_2):
        if not isinstance(_init_params_1, dict):
            _init_params_1 = _init_params_1.arguments
        if not isinstance(_init_params_2, dict):
            _init_params_2 = _init_params_2.arguments
        assert _init_params_1.keys() == _init_params_2.keys()
        for k, v in _init_params_1.items():
            if isinstance(v, (np.ndarray, list, tuple)):
                np.array_equal(v, _init_params_2[k])
            elif isinstance(v, dict):
                assert_init_params_equal(v, _init_params_2[k])
            else:
                assert v == _init_params_2[k]
    with open(os.path.join(logging_dir, LOG_TAG + '_env.pickle'), "rb") as f:
        _loaded_env = pickle.load(f)
        assert type(_loaded_env) is type(env)
        # assert _loaded_env._init_params == env._init_params
        assert_init_params_equal(env._init_params, _loaded_env._init_params)
    # check env:
    #check_env(env)  # todo: move in tests
    random.seed()  # reseed, because check_env(env) sets always the same random.seed
    logging.debug(random.getstate())
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
    config_path = os.path.join(local_dir, 'config-cs-rnn')
    # logging: save current config file for later use:
    shutil.copyfile(config_path, os.path.join(logging_dir, LOG_TAG + '_config'))

    # select Phenotype:
    Phenotype = ConstantSpeedRnnNeatAgent

    # set Phenotype attributes (overwrite default values, e.g. fitness and evaluate_agent params):
    # Phenotype.fitness_func = memory_evolution.evaluate.FitnessRewardAndSteps(5., 5., normalize_weights=False)
    # Phenotype.eval_num_episodes = 2
    # Phenotype.eval_episodes_aggr_func = 'min'
    # Phenotype.fitness_func = memory_evolution.evaluate.FitnessRewardAndSteps(4., 6., normalize_weights=False)
    # Phenotype.eval_num_episodes = 5
    # Phenotype.eval_episodes_aggr_func = 'median'
    # # allocentric RadialArmMaze:
    assert env.n_food_items == 1 and env.max_steps is not None
    # Phenotype.fitness_func = memory_evolution.evaluate.fitness_func_time_inverse
    # Phenotype.fitness_func = memory_evolution.evaluate.FitnessDistanceInverse(target_pos)
    # Phenotype.eval_num_episodes = 2
    # Phenotype.eval_num_episodes = 3
    # Phenotype.eval_num_episodes = 5
    # Phenotype.eval_num_episodes = 10
    # Phenotype.eval_num_episodes = 30
    # Phenotype.eval_num_episodes = 50
    # Phenotype.eval_episodes_aggr_func = 'min'
    # Phenotype.eval_episodes_aggr_func = 'median'
    # Phenotype.eval_episodes_aggr_func = 'mean'
    # #
    ff_time = memory_evolution.evaluate.fitness_func_time_minimize
    min_ff_time = ff_time(reward=None, steps=max_steps, done=None, env=None, agent=None)
    # user defined fitness_func (outside any module, just in main) so it can be pickled with dill.
    def fitness_func(*, reward, steps, done, env, agent, **kwargs) -> float:
        ft = ff_time(reward=reward, steps=steps, done=done, env=env, agent=agent, **kwargs)
        assert min_ff_time <= ft <= ff_time.max
        fitness = ft
        return fitness
    fitness_func.min = min_ff_time
    fitness_func.max = ff_time.max
    print(f"Fitness bounds: min={fitness_func.min} max={fitness_func.max}")
    Phenotype.fitness_func = fitness_func
    #
    Phenotype.eval_num_episodes = 20
    Phenotype.eval_episodes_aggr_func = 'mean'

    # dump Phenotype for later use:
    with open(os.path.join(logging_dir, LOG_TAG + '_phenotype.pkl'), "wb") as f:
        dill.dump((Phenotype,
                   {'fitness_func': Phenotype.fitness_func,  # use dill to pickle this
                    'eval_num_episodes': Phenotype.eval_num_episodes,
                    'eval_episodes_aggr_func': Phenotype.eval_episodes_aggr_func,
                    }),
                  f)
        # pickle do not save classes, but just reference them, so the class attribute will not be saved
        # thus save the class stuff explicitly
        # dill module (pickle extension) dumps classes, but only if defined in __main__ (not in modules or other files)
    # construct agent:
    agent = Phenotype(config_path)

    # check phenotype.pickle: (todo move in tests (but also leave here as example))
    # dill module (pickle extension) dumps classes, but only if defined in __main__ (not in modules or other files)
    # assert A is pickle.loads(pickle.dumps(A))  # wrong
    # assert A is not dill.loads(dill.dumps(A))  # good
    # assert A is not pickle.loads(dill.dumps(A))  # also fine  # here you can use also pickle to load (but dill must be used to dump the class)
    with open(os.path.join(logging_dir, LOG_TAG + '_phenotype.pkl'), "rb") as f:
        Phenotype_, _Phenotype_attrs = dill.load(f)
        # Phenotype_.__dict__.update(_Phenotype_dict)  # not possible with classes, because classes __dict__ is not a dict
        for name, value in _Phenotype_attrs.items():
            setattr(Phenotype_, name, value)
    assert Phenotype is Phenotype_  # both pickle (classes) and dill (module classes)
    # assert Phenotype.fitness_func is Phenotype_.fitness_func is fitness_func  # pickle
    assert Phenotype.fitness_func is Phenotype_.fitness_func is not fitness_func  # dill
    assert memory_evolution.evaluate.fitness_func_total_reward in memory_evolution.evaluate.__dict__.values()
    # assert fitness_func not in memory_evolution.evaluate.__dict__.values()
    assert Phenotype_.fitness_func not in memory_evolution.evaluate.__dict__.values()  # dill

    logging.info(f"Phenotype: {Phenotype.__qualname__}")
    logging.info(f"Phenotype.fitness_func: {Phenotype.fitness_func}")
    logging.info(f"Phenotype.eval_num_episodes: {Phenotype.eval_num_episodes}")
    logging.info(f"Phenotype.eval_episodes_aggr_func: {Phenotype.eval_episodes_aggr_func}")
    print()

    # ----- MAIN LOOP -----
    # Evolve, interact, repeat.

    # Rendering settings:
    if not JOB_ID:  # local execution
        # note: if you render all will be slow, but good for debugging
        # note2: if you render all and if you minimize the window or you put it in a part of the screen not visible
        #        the algorithm will go way faster, so you can make it faster and debugging
        #        at your choice by knowing this.
        render, parallel, render_best = True, False, True      # local execution, render all
        # render, parallel, render_best = False, True, True     # local execution, show best
    else:  # remote execution
        render, parallel, render_best = False, True, False    # remote execution, just save gifs

    # evaluate_agent(RandomActionAgent(env), env, episodes=2, render=True)

    # checkpointer = neat.Checkpointer(generation_interval=200,
    #                                  time_interval_seconds=600,
    #                                  filename_prefix=os.path.join(
    #                                      logging_dir,
    #                                      LOG_TAG + '_neat-checkpoint-'))

    agent.set_env(env)
    logging.info("Evolving...")
    winner, stats = agent.evolve(EPOCHS, render=render,
                                 # checkpointer=checkpointer,
                                 parallel=parallel,
                                 filename_tag=LOG_TAG + '_', path_dir=logging_dir, image_format='png',
                                 view_best=False,
                                 stats_ylog=False)
    # fixme: todo: parallel=True use the same seed for the environment in each process
    #     (but for the agent is correctly using a different seed it seems)
    logging.info("Evolution finished.")

    # render the best agent:
    evaluate_agent(agent, env, episodes=3, render=render_best,
                   save_gif=True,
                   save_gif_name=os.path.join(logging_dir, LOG_TAG + '_frames.gif'))
    # note: if you evolve the agent with parallel execution, agent.evolve(parallel=True),
    #   all the program variables will be copied and will have an independent life for each process,
    #   thus the environment will be used in each process with a different cache, and when each process
    #   is ended, the env here in the main will still be the env newly created which has never been reset before,
    #   thus if you evolve the agent with parallel execution, agent.evolve(parallel=True),
    #   evaluate_agent() here will reset the environment for the first time ever for the main process
    #   and cache will be empty because the env was never used before in the main process.

    # ----- CLOSING AND REPORTING -----

    ACCURACY_TRIALS = 200

    # testing the agent first arm accuracy:
    accuracy = memory_evolution.evaluate.test_agent_first_arm_accuracy(
        agent, env, episodes=ACCURACY_TRIALS,
        render=False)
    print(f"test_agent_first_arm_accuracy (n={ACCURACY_TRIALS}): {accuracy}")

    # test general target-reached rate (to discriminate bad v.s. border-follower v.s. allocentric/egocentric successful agents):
    target_reached_rate = memory_evolution.evaluate.test_agent_target_reached_rate(
        agent, env, episodes=ACCURACY_TRIALS,
        render=False)
    print(f"test_agent_target_reached_rate (n={ACCURACY_TRIALS}): {target_reached_rate}")

    # fitness:
    print(f"BestGenomeFitness: {winner.fitness}")

    with open(os.path.join(logging_dir, LOG_TAG + '_results.json'), 'w') as f:
        json.dump({
            'BestGenome': {
                'test_agent_first_arm_accuracy': accuracy,
                'test_agent_target_reached_rate': target_reached_rate,
                'Fitness': winner.fitness,
            }
        }, f)

    env.close()

