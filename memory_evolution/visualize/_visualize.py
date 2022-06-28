from __future__ import print_function

import copy
import math
from typing import Literal, Optional
from collections.abc import Sequence
import warnings

import neat

try:
    import graphviz
except ModuleNotFoundError as err:
    graphviz = None
    warnings.warn(__package__ + ': Missing optional dependency to display nets (graphviz)'
                  " (to install it try 'conda install -n <name-of-your-conda-environment> python-graphviz pydot' or use pip, then restart the terminal)")
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ModuleNotFoundError as err:
    plt = None
    warnings.warn(__package__ + ': Missing optional dependency to display plots (matplotlib)')
import numpy as np


def plot_stats(statistics, ylog=False, view=False, filename='fitness.svg', ylim=None):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    #plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        base = 10
        ax = plt.gca()
        # ax.set_yscale('symlog')
        ax.set_yscale(mpl.scale.SymmetricalLogScale(ax, base=base, linthresh=1, subs=[2.5, 5, 7.5]))
        ax.grid(True, which='minor', color='gainsboro', linestyle=':', linewidth=.5)
        if ylim is not None:
            pass  # todo: yticks
    if ylim is not None:
        plt.gca().set_ylim(ylim)  # ax.set_ylim([ymin, ymax])
        # plt.xlim(right=xmax)  # xmax is your value
        # plt.xlim(left=xmin)  # xmin is your value
        # plt.ylim(top=ymax)  # ymax is your value
        # plt.ylim(bottom=ymin)  # ymin is your value

    plt.tight_layout()
    plt.savefig(filename)
    if view:
        plt.show(block=True)

    plt.close()


def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show(block=True)

    plt.close()


'''  # first version
def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, format='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}
    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}
    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=format, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}
        dot.node(name, _attributes=node_attrs)

    if not (inputs | outputs >= node_names.keys()):
        raise ValueError("'node_names' can contain only inputs and outputs nodes")

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add(cg.key)

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())
        used_nodes.update(inputs)  # add also inputs, genome.nodes.keys() don't contain inputs.

    for n in used_nodes:
        if n not in inputs and n not in outputs:
            attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}
            dot.node(node_names.get(n, str(n)), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input, output = cg.key
            if cg.key[0] in used_nodes and cg.key[1] in used_nodes:
                a = node_names.get(input, str(input))
                b = node_names.get(output, str(output))
                style = 'solid' if cg.enabled else 'dotted'
                color = 'green' if cg.weight > 0 else 'red'
                width = str(0.1 + abs(cg.weight / 5.0))
                dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot
'''

# version 2
def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, node_positions=None, node_attributes=None,
             default_node_attrs=None, rankdir: Literal['TB', 'LR', 'BT', 'RL'] = 'TB',
             order_inputs=True, order_outputs=True, render=True,
             format='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}
    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}
    assert type(node_colors) is dict

    if node_positions is None:
        node_positions = {}
    else:
        for n in node_positions:
            pos = node_positions[n]
            if not isinstance(pos, Sequence):
                raise TypeError(f"'pos' is not a Sequence (it is '{type(pos).__qualname__}' instead)")
            if len(pos) != 2:
                raise ValueError(f"'pos' is not 2D ({pos})")
            node_positions[n] = ','.join(str(x) for x in pos) + '!'
    assert type(node_positions) is dict

    if node_attributes is None:
        node_attributes = {}
    assert type(node_attributes) is dict
    assert all(type(v) is dict for v in node_attributes.values())

    if default_node_attrs is None:
        default_node_attrs = {}
    assert type(default_node_attrs) is dict
    _default_node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'
    }
    default_node_attrs = _default_node_attrs | default_node_attrs

    graph_kwargs = {}
    if node_positions:
        graph_kwargs['engine'] = 'neato'
    dot = graphviz.Digraph(format=format, node_attr=default_node_attrs,
                           **graph_kwargs)
    dot.graph_attr['rankdir'] = rankdir
    # dot.graph_attr['style'] = 'dotted'  # apply this attr to all subgraph which as a name starting with 'cluster_'

    # with dot.subgraph(name='inputs') as dot_inputs:
    # dot_inputs.attr(rank='min')
    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        if k in node_positions:
            input_attrs['pos'] = node_positions[k]
        if k in node_attributes:
            input_attrs.update(node_attributes[k])
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        output_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}
        if k in node_positions:
            output_attrs['pos'] = node_positions[k]
        if k in node_attributes:
            output_attrs.update(node_attributes[k])
        dot.node(name, _attributes=output_attrs)

    if not (inputs | outputs >= node_names.keys()):
        raise ValueError("'node_names' can contain only inputs and outputs nodes")

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add(cg.key)

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())
        used_nodes.update(inputs)  # add also inputs, genome.nodes.keys() don't contain inputs.

    for n in used_nodes:
        if n not in inputs and n not in outputs:
            attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}
            if n in node_positions:
                attrs['pos'] = node_positions[n]
            if n in node_attributes:
                attrs.update(node_attributes[n])
            dot.node(node_names.get(n, str(n)), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input, output = cg.key
            if cg.key[0] in used_nodes and cg.key[1] in used_nodes:
                a = node_names.get(input, str(input))
                b = node_names.get(output, str(output))
                style = 'solid' if cg.enabled else 'dotted'
                color = 'green' if cg.weight > 0 else 'red'
                width = str(0.1 + abs(cg.weight / 5.0))
                dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    if order_inputs:
        # order by adding invisible edges:
        for k, k2 in zip(config.genome_config.input_keys[:-1], config.genome_config.input_keys[1:]):
            name1 = node_names.get(k, str(k))
            name2 = node_names.get(k2, str(k2))
            dot.edge(name1, name2, _attributes={'style': 'invis'})

        # put inputs at start:
        with dot.subgraph(name='inputs') as dot_inputs:
            dot_inputs.attr(rank='min')
            for k in config.genome_config.input_keys:
                name = node_names.get(k, str(k))
                dot_inputs.node(name)

    if order_outputs:
        # I've commented this otherwise outputs are shown too close to each other.
        # # order by adding invisible edges:
        # for k, k2 in zip(config.genome_config.output_keys[:-1], config.genome_config.output_keys[1:]):
        #     name1 = node_names.get(k, str(k))
        #     name2 = node_names.get(k2, str(k2))
        #     dot.edge(name1, name2, _attributes={'style': 'invis'})

        # put outputs as last:
        with dot.subgraph(name='outputs') as dot_outputs:
            dot_outputs.attr(rank='max')
            for k in config.genome_config.output_keys:
                name = node_names.get(k, str(k))
                dot_outputs.node(name)

    if order_inputs or order_outputs:
        with dot.subgraph(name='hidden') as dot_hidden:
            # dot_hidden.attr(style='dotted')  # label='hidden'
            for k in genome.nodes:
                if k not in outputs:
                    if not prune_unused or k in used_nodes:
                        name = node_names.get(k, str(k))
                        dot_hidden.node(name)

    if render:
        dot.render(filename, view=view)

    return dot
