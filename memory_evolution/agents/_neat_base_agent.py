from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from collections.abc import Iterable, Sequence
from functools import reduce
import math
import multiprocessing
from numbers import Number, Real
from operator import mul
import os
import pickle
from typing import Optional, Union, Any, Literal
from warnings import warn
import sys
import time

import gym
from gym import spaces
import matplotlib as mpl
import matplotlib.pyplot as plt
import neat
import numpy as np
from numpy.random import SeedSequence, default_rng
import pandas as pd
import pygame
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint, MultiPolygon
from shapely.ops import unary_union, triangulate

from memory_evolution.agents import BaseAgent
from memory_evolution.agents.exceptions import EnvironmentNotSetError
from memory_evolution.evaluate import evaluate_agent
from memory_evolution.utils import MustOverride, override
from .exceptions import NotEvolvedError


def __low_level_func(*args, **kwargs):
    raise RuntimeError("'__low_level_func()' used before assignment.")


def __top_level_func(*args, **kwargs):
    return __low_level_func(*args, **kwargs)


def _top_level_wrapper(low_level_func):
    """'multiprocessing.Pool' needs the eval_genome function to be in module
    scope, because needs to be pickable and only functions defined at the top
    level of a module are pickable. Here a wrapper that wraps the low level
    function and returns a top level function."""
    global __low_level_func
    assert '__low_level_func' in globals(), "'__low_level_func()' was not found in globals() of this file"
    __low_level_func = low_level_func
    return __top_level_func


class BaseNeatAgent(BaseAgent, ABC):

    @classmethod
    @property
    @abstractmethod
    def phenotype_class(cls):
        raise NotImplementedError

    def __init__(self, config: Union[str, bytes, os.PathLike, int, neat.config.Config], genome=None):
        """

        Args:
            config: config of config_file path (path should be string, bytes, os.PathLike or integer(?)).
            genome: the genome, if None the agent needs to be evolved to
                generate a genome from evolution (using evolution parameters
                in 'config').
        """
        super().__init__()

        # Load configuration.
        if isinstance(config, neat.config.Config):
            self.config = config
        else:
            # config is config_file
            self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                      config)
        self._genome = None
        self._phenotype = None
        if genome is not None:
            self.genome = genome
        self.node_names = {0: 'rotation', 1: 'forward'}  # {-1: 'A', -2: 'B', 0: 'A XOR B'}
        self._render = False

    def set_env(self, env: gym.Env) -> None:
        """Extends base class method with the same name."""
        config = 0
        obs_size = reduce(mul, env.observation_space.shape, 1)
        if self.config.genome_config.num_inputs != obs_size:
            raise ValueError(
                f"Network input ({self.config.genome_config.num_inputs}) "
                f"doesn't fits 'env.observation_space' "
                f"({env.observation_space.shape} -> size: {obs_size}), "
                "change config.genome_config.num_inputs in your config file "
                "or change environment; "
                "config.genome_config.num_inputs should be equal to the total "
                "size of 'env.observation_space.shape' (total size, "
                "i.e. the sum of its dimensions)."
            )
        act_size = reduce(mul, env.action_space.shape, 1)
        if len(env.action_space.shape) != 1:
            raise ValueError("'len(env.action_space.shape)' should be 1;")
        if self.config.genome_config.num_outputs != act_size:
            raise ValueError(
                f"Network output ({self.config.genome_config.num_outputs}) "
                f"doesn't fits 'env.action_space'"
                f"({env.action_space.shape} -> size: {act_size}), "
                "change config.genome_config.num_outputs in your config file "
                "or change environment."
            )
        super().set_env(env)

    @property
    def genome(self):
        if self._genome is None:
            assert self._phenotype is None, (self._genome, self._phenotype)
            raise NotEvolvedError(
                "Agent has never been evolved before, "
                "evolve the agent before asking for the genome."
            )
        assert self._phenotype is not None, (self._genome, self._phenotype)
        return self._genome

    @genome.setter
    def genome(self, genome):
        self._genome = genome
        net = self.phenotype_class.create(self._genome, self.config)
        self._phenotype = net

    @property
    def phenotype(self):
        if self._genome is None:
            assert self._phenotype is None, (self._genome, self._phenotype)
            raise NotEvolvedError(
                "Agent has never been evolved before, "
                "evolve the agent before asking for the phenotype."
            )
        assert self._phenotype is not None, (self._genome, self._phenotype)
        return self._phenotype

    @abstractmethod
    def action(self, observation: np.ndarray) -> np.ndarray:
        """Overrides 'action()' method from base class.
        Takes an observation from the environment as argument and returns
        a valid action.

        Args:
            observation: observation from the environment.

        Returns:
            An action.

        Raises:
            NotEvolvedError: Raised when asking an agent which has never been
                evolved before to perform an action.
        """
        if self._genome is None:
            assert self._phenotype is None, (self._genome, self._phenotype)
            raise NotEvolvedError(
                "Agent has never been evolved before, "
                "evolve the agent before asking for an action."
            )
        assert self._phenotype is not None, (self._genome, self._phenotype)
        return NotImplemented

    # def get_eval_genome_func()
    # @classmethod
    def eval_genome(self, genome, config) -> float:
        """Use the Agent network phenotype and the discrete actuator force function."""
        assert genome is not None, self
        assert self._env is not None, self
        agent = type(self)(config, genome=genome)
        agent.set_env(self.get_env())  # fixme: parallel execution should make a copy of env for each pool.
        #                              # todo: is it doing this?
        assert agent._genome is not None, agent
        assert agent._phenotype is not None, agent
        fitness = evaluate_agent(agent, self.get_env(),
                                 episodes=2, render=self._render)
        return fitness

    @abstractmethod
    def reset(self) -> None:
        """Extends 'action()' method from base class.
        Reset the agent to an initial state ``t==0``."""
        super().reset()

    def eval_genomes(self, genomes, config) -> None:
        for genome_id, genome in genomes:
            genome.fitness = self.eval_genome(genome, config)

    def get_init_population(self, reporters: Optional[Iterable] = None):
        p = neat.Population(self.config)
        if reporters is not None:
            p = self._add_reporters(p, reporters)
        return p

    @staticmethod
    def _add_reporters(population: neat.Population,
                       reporters: Iterable,
                       ) -> neat.Population:
        """Add a stdout reporter to show progress in the terminal."""
        for rep in reporters:
            population.add_reporter(rep)
        return population

    def evolve_population(self,
                          population: neat.Population,
                          n=None,
                          *,
                          parallel=False,
                          parallel_num_workers=None,  # default(None): multiprocessing.cpu_count()
                          parallel_timeout=None,
                          ):
        """Evolve a population of agents of the type of self and then return
        the best genome.

        Note, it doesn't save the best genome in self.
        """
        if parallel:
            if parallel_num_workers is None:
                parallel_num_workers = multiprocessing.cpu_count()
                assert parallel_num_workers > 0, parallel_num_workers
            # 'multiprocessing.Pool' needs the eval_genome function to be in module scope
            # (defined at the top level of a module, i.e. top level of the file)
            eval_genome = _top_level_wrapper(self.eval_genome)
            pe = neat.ParallelEvaluator(parallel_num_workers, eval_genome, parallel_timeout)
            winner = population.run(pe.evaluate, n)
        else:
            winner = population.run(self.eval_genomes, n)
        return winner

    def visualize_evolution(self, stats, stats_ylog=False, view=False,
                            filename_stats="fitness.svg",
                            filename_speciation="speciation.svg"):
        neat.visualize.plot_stats(stats, ylog=stats_ylog, view=view, filename=filename_stats)
        neat.visualize.plot_species(stats, view=view, filename=filename_speciation)

    def visualize_genome(self, genome, name='Genome',
                         view=False, filename=None,
                         show_disabled=True, prune_unused=False,
                         node_colors=None, format='sgv'):  # 'format' abbreviation: fmt
        """Display the genome."""
        print('\n{!s}:\n{!s}'.format(name, genome))

        node_names = self.node_names
        neat.visualize.draw_net(
            self.config, genome, view=view,
            filename=filename,
            node_names=node_names,
            show_disabled=show_disabled,
            prune_unused=prune_unused,
            node_colors=node_colors,
            format=format
        )

    def evolve(self,
               n=None,
               parallel=False,
               checkpointer: Union[None, int, float, neat.Checkpointer] = None,
               render: bool = False,
               filename_tag: str = '',
               path_dir: str = '',
               image_format: str = 'svg',
               ) -> tuple[neat.genome.DefaultGenome, neat.statistics.StatisticsReporter]:
        """Evolve a population of agents of the type of self and then return
        the best genome. The best genome is also saved in self. It returns also
        some stats about the evolution.

        The initial ``population`` is created using the method
        ``self.get_init_population(reporters)`` with some default reporters,
        at the end of the evolution it returns a tuple with the winner
        and a neat.StatisticsReporter reporter with stats about the evolution.
        """
        prev_rendering_option = self._render
        self._render = render
        if render and parallel:
            raise ValueError("Parallel evolution cannot be rendered. "
                             "('render' and 'parallel' cannot be True at the same time)")

        # Load configuration.
        # Create the population, which is the top-level object for a NEAT run.
        # Add a stdout reporter to show progress in the terminal.
        stats = neat.StatisticsReporter()
        reporters = [
            neat.StdOutReporter(True),
            stats,
        ]
        if checkpointer is not None:
            if not isinstance(checkpointer, neat.Checkpointer):
                raise TypeError("'checkerpointer' should be None or neat.Checkpointer")
            # if isinstance(checkpointer, int):
            #     checkpointer = neat.Checkpointer(generation_interval=checkpointer, time_interval_seconds=300, filename_prefix='neat-checkpoint-')
            # if isinstance(checkpointer, float):
            #     checkpointer = neat.Checkpointer(generation_interval=100, time_interval_seconds=checkpointer, filename_prefix='neat-checkpoint-')
            reporters += [checkpointer]
        p = self.get_init_population(reporters)

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
        winner = self.evolve_population(p, n, parallel=parallel)
        self.genome = winner
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

        if image_format.startswith('.'):
            image_format = image_format[1:]
        def make_filename(filename):
            return os.path.join(path_dir, filename_tag + filename)

        # Pickle winner.
        with open(make_filename("genome.pickle"), "wb") as f:
            pickle.dump(winner, f)

        # Display stats on the evolution performed.
        self.visualize_evolution(stats, stats_ylog=True, view=True,
                                 filename_stats=make_filename("fitness." + image_format),
                                 filename_speciation=make_filename("speciation." + image_format))

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))
        self.visualize_genome(winner, view=True, name='Best Genome', filename=make_filename("winner-genome.gv"),
                              format=image_format)

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        print('rendering one episode with the best agent...')
        evaluate_agent(self, self.get_env(), episodes=1, render=True)

        # Try to reload population from a saved checkpointer and run evolution for few generations.
        if checkpointer is not None and checkpointer.last_generation_checkpoint != -1:
            last_cp_gen = checkpointer.last_generation_checkpoint
            last_cp_time = checkpointer.last_time_checkpoint
            last_cp_time_from_start = last_cp_time - start_time
            print(f"Last checkpoint saved at generation #{last_cp_gen}"
                  f" after {last_cp_time_from_start} seconds from start.")
            print("Restoring checkpoint and running up to 10 generations:")
            p = neat.Checkpointer.restore_checkpoint(
                make_filename(f'neat-checkpoint-{last_cp_gen}'))
            p.run(self.eval_genomes, min(10, max(1, last_cp_gen // 4)))

        self._render = prev_rendering_option
        return winner, stats

