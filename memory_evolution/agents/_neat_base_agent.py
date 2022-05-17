from abc import ABC, abstractmethod
from collections import defaultdict, Counter, deque
from collections.abc import Iterable, Sequence
from functools import reduce
import inspect
import json
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
# from neat import visualize
import numpy as np
from numpy.random import SeedSequence, default_rng
import pandas as pd
import pygame
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint, MultiPolygon
from shapely.ops import unary_union, triangulate

import memory_evolution
from memory_evolution import visualize
from memory_evolution.agents import BaseAgent
from memory_evolution.agents.exceptions import EnvironmentNotSetError
from memory_evolution.evaluate import evaluate_agent, FitnessRewardAndSteps
from memory_evolution.utils import get_color_str, normalize_observation
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

    # fitness_func is class attribute, because all agents of the same
    # type (or in same population) should be evaluated in the same way
    # (so that the fitnesses of two agents can be compared).
    fitness_func: inspect.signature(evaluate_agent).parameters['fitness_func'].annotation = FitnessRewardAndSteps(4., 6., normalize_weights=False)
    eval_num_episodes: inspect.signature(evaluate_agent).parameters['episodes'].annotation = 5
    eval_episodes_aggr_func: inspect.signature(evaluate_agent).parameters['episodes_aggr_func'].annotation = 'median'
    # You can access the dict of annotations with: print(type(self).__annotations__)

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
                                 episodes=type(self).eval_num_episodes,
                                 episodes_aggr_func=type(self).eval_episodes_aggr_func,
                                 fitness_func=type(self).fitness_func,  # get the class function, otherwise it will apply self to the future call because it believes it is a method.
                                 render=self._render)
        return fitness

    @abstractmethod
    def reset(self) -> None:
        """Extends 'action()' method from base class.
        Reset the agent to an initial state ``t==0``."""
        super().reset()
        self.phenotype.reset()

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
        ylim = None
        if hasattr(self.fitness_func, 'min') and hasattr(self.fitness_func, 'max'):
            if not stats_ylog:
                y_range = self.fitness_func.max - self.fitness_func.min
                offset = y_range * .03
                ylim = (self.fitness_func.min - offset, self.fitness_func.max + offset)
            else:
                ylim = (self.fitness_func.min, self.fitness_func.max)
        visualize.plot_stats(stats, ylog=stats_ylog, view=view, filename=filename_stats, ylim=ylim)
        visualize.plot_species(stats, view=view, filename=filename_speciation)

    def visualize_genome(self, genome, name='Genome',
                         view=False, filename=None,
                         show_disabled=True, prune_unused=False,
                         node_colors=None, format='sgv',
                         default_input_node_color: Literal['palette', 'default'] = 'default',  # default: 'lightgray'
                         show_palette=True,
                         ):  # 'format' abbreviation: fmt
        """Display the genome."""
        print('\n{!s}:\n{!s}'.format(name, genome))

        rankdir = 'TB'
        node_positions = None
        node_attributes = None
        if default_input_node_color == 'palette':
            rankdir = 'LR'
            env = self.get_env()
            vision_shape = env.vision_shape
            vision_channels = env.vision_channels
            assert len(vision_shape) == 3, vision_shape
            assert vision_channels == vision_shape[2], (vision_channels, vision_shape)
            obs_shape = env.observation_space.shape
            assert obs_shape == vision_shape, (obs_shape, vision_shape)  # this actually could be false, if it is false you need to change the code below
            obs_channels = obs_shape[2]
            assert obs_channels == 1 or obs_channels == 3, obs_shape
            obs_size = reduce(mul, obs_shape, 1)
            input_nodes = self.config.genome_config.input_keys
            assert obs_size == len(input_nodes) == len(self.phenotype.input_nodes)
            assert input_nodes == self.phenotype.input_nodes
            n_rows = obs_shape[0]
            n_cols = obs_shape[1]
            palette = np.empty((n_rows, n_cols, 3), dtype=np.uint8)
            indexes = np.empty((n_rows, n_cols), dtype=int)
            _input_node_colors = {}
            # _input_node_positions = {}
            k = 0
            for i in range(n_rows):
                for j in range(n_cols):
                    indexes[i, j] = k
                    norm_i = i / (n_rows - 1)
                    norm_j = j / (n_cols - 1)
                    norm_diag = (norm_i + norm_j) / 2
                    palette[i, j] = np.asarray((255 * norm_i, 255 * norm_j, 255 * (1. - norm_diag)), dtype=np.uint8)
                    _input_node_colors[input_nodes[k]] = get_color_str(palette[i, j])
                    if obs_channels == 3:
                        k += 1
                        _input_node_colors[input_nodes[k]] = get_color_str(palette[i, j])
                        k += 1
                        _input_node_colors[input_nodes[k]] = get_color_str(palette[i, j])
                    # _input_node_positions[input_nodes[k]] = (j * .5, - i * .5)
                    k += 1
            assert k == len(input_nodes)
            assert obs_size == len(input_nodes) == len(_input_node_colors)
            if show_palette:
                plt.matshow(palette)
                inputs_palette_filename = ('' if filename is None else filename) + "_inputs_palette.png"
                plt.savefig(inputs_palette_filename)
                if view:
                    plt.show()
            # if node_colors are provided the default _input_node_colors value is overwritten if present:
            if node_colors is not None:
                node_colors = _input_node_colors | node_colors
            else:
                node_colors = _input_node_colors
            # node_positions = _input_node_positions

            # compute node ranks:
            output_nodes = self.config.genome_config.output_keys
            _output_nodes_set = set(output_nodes)
            hidden_nodes = [n.key for n in self.genome.nodes.values() if n.key not in _output_nodes_set]
            graph = {k: [] for k in (*input_nodes, *hidden_nodes, *output_nodes)}
            for cg in self.genome.connections.values():
                in_node, out_node = cg.key
                graph[in_node].append(out_node)
            def bfs(graph, nodes):
                assert list(graph.keys())[:len(input_nodes)] == input_nodes
                node_rank = {}
                q = deque()
                for node in nodes:
                    assert node not in node_rank
                    node_rank[node] = 0
                    q.append(node)
                    while q:
                        u = q.popleft()
                        level = node_rank[u]
                        for v in graph[u]:
                            if v not in node_rank or node_rank[v] > level + 1:
                                node_rank[v] = level + 1
                                q.append(v)
                return node_rank
            starting_nodes = set(graph.keys())
            for cg in self.genome.connections.values():
                in_node, out_node = cg.key
                if out_node in starting_nodes:
                    starting_nodes.remove(out_node)
            node_rank = bfs(graph, starting_nodes)
            max_rank = max(node_rank.values())
            assert min(node_rank.values()) == 0
            rank_hidden = defaultdict(list)
            if hidden_nodes:
                for h in hidden_nodes:
                    rank_hidden[node_rank[h]].append(h)
                max_hidden_rank = max(rank_hidden)
            else:
                max_hidden_rank = 0
            # # outputs are placed all in the max rank (they are also in the correct order):
            # for o in output_nodes:
            #     rank_hidden[max_rank].append(o)

        order_inputs = False
        order_outputs = False
        if default_input_node_color == 'palette':
            if obs_channels == 1:
                order_inputs = False
                order_outputs = False
            elif obs_channels == 3:
                order_inputs = True
                order_outputs = True

        node_names = self.node_names
        dot = visualize.draw_net(
            self.config, genome, view=view,
            filename=filename,
            node_names=node_names,
            show_disabled=show_disabled,
            prune_unused=prune_unused,
            node_colors=node_colors,
            node_positions=node_positions,
            node_attributes=node_attributes,
            rankdir=rankdir,
            format=format,
            order_inputs=order_inputs,
            order_outputs=order_outputs,
            render=False,
        )

        # ordering of nodes:
        if default_input_node_color == 'palette' and obs_channels == 1:
            dot.graph_attr['compound'] = 'true'

            # order by adding invisible edges:
            # inputs:
            for k, k2 in zip(self.config.genome_config.input_keys[:-1], self.config.genome_config.input_keys[1:]):
                name1 = node_names.get(k, str(k))
                name2 = node_names.get(k2, str(k2))
                dot.edge(name1, name2, _attributes={'style': 'invis'})
            # outputs:
            for k, k2 in zip(self.config.genome_config.output_keys[:-1], self.config.genome_config.output_keys[1:]):
                name1 = node_names.get(k, str(k))
                name2 = node_names.get(k2, str(k2))
                dot.edge(name1, name2, _attributes={'style': 'invis'})

            # put the inputs in column clusters and row groups:
            with dot.subgraph(name='cluster_inputs') as dot_inputs:
                dot_inputs.attr(style='dotted')
                for col in indexes.T:
                    with dot_inputs.subgraph() as dot_sub_inputs:
                        dot_sub_inputs.attr(rank='same')
                        for i, k in enumerate(col):
                            k = input_nodes[k]
                            name = node_names.get(k, str(k))
                            dot_sub_inputs.node(name, _attributes={'group': 'input_group_' + str(i)})

            # put hidden and output in clusters:
            with dot.subgraph(name='cluster_outputs') as dot_outputs:
                dot_outputs.attr(rank='same', style='invis')
                for k in output_nodes:
                    name = node_names.get(k, str(k))
                    dot_outputs.node(name)
            with dot.subgraph(name='cluster_hidden') as dot_hidden:
                dot_hidden.attr(style='invis')
                # note: you should do subgraph for rank if you really want to enforce it also for all hidden nodes
                for k in self.genome.nodes:
                    if k not in output_nodes:
                        # this will show also pruned nodes (minor issue because I don't prune any node)
                        name = node_names.get(k, str(k))
                        dot_hidden.node(name)

            # use the ranks computed before to create invisible links:
            _rank_style = 'invis'  # 'dotted'  # 'invis'  # 'dotted'
            dot.node('rank_i', _attributes={'style': _rank_style, 'group': 'rank'})
            if hidden_nodes:
                dot.node('rank_0', _attributes={'style': _rank_style, 'group': 'rank'})
                dot.edge('rank_i', 'rank_0', _attributes={'style': _rank_style})
                for rank in range(1, max_hidden_rank + 1):
                    dot.node('rank_' + str(rank), _attributes={'style': _rank_style, 'group': 'rank'})
                    dot.edge('rank_' + str(rank - 1), 'rank_' + str(rank), _attributes={'style': _rank_style})
                dot.node('rank_o', _attributes={'style': _rank_style, 'group': 'rank'})
                assert rank == max_hidden_rank, (rank, max_hidden_rank)
                dot.edge('rank_' + str(rank), 'rank_o', _attributes={'style': _rank_style})
            else:
                dot.node('rank_o', _attributes={'style': _rank_style, 'group': 'rank'})
                dot.edge('rank_i', 'rank_o', _attributes={'style': _rank_style})
            # inputs:
            k = input_nodes[-1]
            name = node_names.get(k, str(k))
            dot.edge('rank_i', name, _attributes={'style': _rank_style, 'lhead': 'cluster_inputs'})
            # hidden:
            if 0 in rank_hidden:
                rank = 0
                k = rank_hidden[rank][0]
                name = node_names.get(k, str(k))
                dot.edge('rank_' + str(rank), name, _attributes={'style': _rank_style, 'lhead': 'cluster_hidden'})
            for rank in range(1, max_hidden_rank + 1):
                k = rank_hidden[rank][0]
                name = node_names.get(k, str(k))
                dot.edge('rank_' + str(rank), name, _attributes={'style': _rank_style, 'lhead': 'cluster_hidden'})
            # outputs:
            k = output_nodes[-1]
            name = node_names.get(k, str(k))
            dot.edge('rank_o', name, _attributes={'style': _rank_style, 'lhead': 'cluster_outputs'})

        dot.render(filename, view=view)
        print(dot.source)

    def evolve(self,
               n=None,
               parallel=False,
               checkpointer: Union[None, int, float, neat.Checkpointer] = None,
               render: bool = False,
               filename_tag: str = '',
               path_dir: str = '',
               image_format: str = 'svg',
               view_best: bool = False,  # render the best agent and show stats and genome by opening plots
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

        # Pickle stats.
        with open(make_filename("stats.pickle"), "wb") as f:
            pickle.dump(stats, f)

        # Display stats on the evolution performed.
        self.visualize_evolution(stats, stats_ylog=True, view=view_best,
                                 filename_stats=make_filename("fitness." + image_format),
                                 filename_speciation=make_filename("speciation." + image_format))

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))
        self.visualize_genome(winner, view=view_best, name='Best Genome', filename=make_filename("winner-genome.gv"),
                              default_input_node_color='palette',
                              format=image_format)

        # Show output of the most fit genome against training data.
        if view_best:
            print('\nOutput:')
            print('rendering one episode with the best agent...')
            evaluate_agent(self, self.get_env(), episodes=1, render=True, save_gif=False)

        # # Try to reload population from a saved checkpointer and run evolution for few generations.
        # if checkpointer is not None and checkpointer.last_generation_checkpoint != -1:
        #     last_cp_gen = checkpointer.last_generation_checkpoint
        #     last_cp_time = checkpointer.last_time_checkpoint
        #     last_cp_time_from_start = last_cp_time - start_time
        #     print(f"Last checkpoint saved at generation #{last_cp_gen}"
        #           f" after {last_cp_time_from_start} seconds from start.")
        #     print("Restoring checkpoint and running up to 10 generations:")
        #     p = neat.Checkpointer.restore_checkpoint(
        #         make_filename(f'neat-checkpoint-{last_cp_gen}'))
        #     p.run(self.eval_genomes, min(10, max(1, last_cp_gen // 4)))

        self._render = prev_rendering_option
        return winner, stats

