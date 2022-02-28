from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from collections.abc import Iterable, Sequence
import math
import multiprocessing
from numbers import Number, Real
import os
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

from memory_evolution.utils import (
    black_n_white, COLORS, convert_image_to_pygame, is_color,
    is_simple_polygon, is_triangle,
    Pos, triangulate_nonconvex_polygon,
)
from memory_evolution.utils import evaluate_agent
from memory_evolution.utils import MustOverride, override


class NotEvolvedError(Exception):
    """Raised when asking the genome of an agent which has never been
    evolved before to perform an action."""

    default_msg = (
        "Agent has never been evolved before, "
        "evolve the agent before asking for "
        "any property of it."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        if not self.args:
            self.args = (self.default_msg,)


class EnvironmentNotSetError(Exception):
    """Raised when an environment is needed
    but an environment has not been set yet."""

    default_msg = (
        "An environment is needed but an environment"
        " has not been set yet."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        if not self.args:
            self.args = (self.default_msg,)


class BaseNeatAgent(ABC):

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
        self._env = None
        self.node_names = {0: 'rotation', 1: 'forward'}  # {-1: 'A', -2: 'B', 0: 'A XOR B'}

    def get_env(self) -> gym.Env:
        if self._env is None:
            raise EnvironmentNotSetError
        return self._env

    def set_env(self, env: gym.Env) -> None:
        self._env = env

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
        return self._genome

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

    # def get_eval_genome_func()
    # @classmethod
    # todo: controlla sia giusto; fai che config può essere anche un ogetto neat
    def eval_genome(self, genome, config) -> float:
        """Use the Agent network phenotype and the discrete actuator force function."""
        assert genome is not None, self
        assert self._env is not None, self
        agent = type(self)(config, genome=genome)
        agent.set_env(self.get_env())
        assert agent._genome is not None, agent
        assert agent._phenotype is not None, agent
        fitness = evaluate_agent(agent, self.get_env(), render=True)
        return fitness

    @abstractmethod
    def action(self, observation: np.ndarray) -> np.ndarray:
        """Takes an observation from the environment as argument and returns
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

    @abstractmethod
    def reset(self) -> None:
        """Reset the agent to an initial state ``t==0``."""
        pass

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
            pe = neat.ParallelEvaluator(parallel_num_workers, self.eval_genome, parallel_timeout)
            winner = population.run(pe.evaluate, n)
        else:
            winner = population.run(self.eval_genomes, n)
        return winner

    def visualize_evolution(self, stats, stats_ylog=False, view=False,
                            filename_stats="fitness.svg",
                            filename_speciation="speciation.svg"):
        neat.visualize.plot_stats(stats, ylog=stats_ylog, view=view, filename=filename_stats)
        neat.visualize.plot_species(stats, view=view, filename=filename_speciation)

    @abstractmethod
    def visualize_genome(self, genome, name='Genome',
                         view=False, filename=None,
                         show_disabled=True, prune_unused=False):
        """Display the genome."""
        print('\n{!s}:\n{!s}'.format(name, genome))

        node_names = self.node_names
        neat.visualize.draw_net(
            self.config, genome, view=view,
            filename=filename,
            node_names=node_names,
            show_disabled=show_disabled,
            prune_unused=prune_unused
        )

    def evolve(self,
               n=None,
               parallel=False,
               checkpointer: Union[None, int, float, neat.Checkpointer] = None,
               render: bool = False,  # todo
               ):
        """Evolve a population of agents of the type of self and then return
        the best genome. The best genome is also saved in self. It returns also
        some stats about the evolution.

        The initial ``population`` is created using the method
        ``self.get_init_population(reporters)`` with some default reporters,
        at the end of the evolution it returns a tuple with the winner
        and a neat.StatisticsReporter reporter with stats about the evolution.
        """
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
        self.visualize_evolution(stats, stats_ylog=True, view=True)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))
        self.visualize_genome("winner-genome.gv")

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        print('rendering one episode with the best agent...')
        evaluate_agent(self, self.get_env(), render=True)

        if checkpointer is not None:
            last_cp_gen = checkpointer.last_generation_checkpoint
            last_cp_time = checkpointer.last_time_checkpoint
            last_cp_time_from_start = last_cp_time - start_time
            print(f"Last checkpoint saved at generation #{last_cp_gen}"
                  f" after {last_cp_time_from_start} seconds from start.")
            print("Restoring checkpoint and running up to 10 generations:")
            p = neat.Checkpointer.restore_checkpoint(
                f'neat-checkpoint-{last_cp_gen}')
            p.run(self.eval_genomes, 10)

        self.genome = winner
        return winner, stats

