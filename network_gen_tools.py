"""
NETWORK GENERATION TOOLS

Module that provides functions to generate single and double layer
networks.

Author: Paulo Cesar Ventura da Silva.
"""

import networkx as nx
import random as rnd
import numpy as np


def remove_selfloops(g):
    """Function to remove every self loop on graph G.
    Changes are directly applied to object G.

    :param g:(nx graph, digraph, etc) the graph.
    """
    nodes = g.nodes_with_selfloops()
    ebunch = [(ni, ni) for ni in nodes]
    g.remove_edges_from(ebunch)


def make_connex(g, max_steps=None):
    """ Modifies the edges of a graph to make it fully connex, without
    changing the degrees of the nodes. Changes are performed in place.

    Inputs
    ----------
    g : nx.Graph

    max_steps : int
        Maximum rewiring trials until connex. If not informed, the
        number of trials is unlimited.
    """

    # Defines a counter increment function.
    # If max_steps is "infinity", the counter is not incremented.
    if max_steps is None:
        def increment(n):
            return n
        max_steps = 10  # Dummy value to avoid error comparing with i.
    else:
        def increment(n):
            return n+1

    # Extracts graph components and sorts by size
    components = sorted(nx.connected_components(g), key=len,
                        reverse=True)

    # Main loop. Terminates if the graph has a single component, or if
    # the maximum number of iteration is reached.
    i = 0
    while len(components) != 1 and i < max_steps:
        # Gets giant component, and another random component
        giant = components[0]
        small = rnd.sample(components[1:], 1)[0]

        # Gets two connected nodes in each component
        n1_giant = rnd.sample(giant, 1)[0]
        n2_giant = rnd.sample(list(g.neighbors(n1_giant)), 1)[0]
        n1_small = rnd.sample(small, 1)[0]
        n2_small = rnd.sample(list(g.neighbors(n1_small)), 1)[0]
        # n1_giant = rnd.sample(giant, 1)[0]
        # n2_giant = rnd.sample(g.neighbors(n1_giant), 1)[0]
        # n1_small = rnd.sample(small, 1)[0]
        # n2_small = rnd.sample(g.neighbors(n1_small), 1)[0]

        # Rewire links in attempt to connect the two components
        g.remove_edges_from([(n1_giant, n2_giant),
                             (n1_small, n2_small)])
        g.add_edges_from([(n1_giant, n1_small),
                          (n2_giant, n2_small)])

        # Recalculate the graph components
        components = sorted(nx.connected_components(g), key=len,
                            reverse=True)

        # Increments the counter (or not, if max_steps is None).
        i = increment(i)

    # If the maximum number of steps was reached, raises an error
    if i == max_steps and len(components) != 1:
        raise RuntimeError("Hey, failed to make the graph connex after"
                           "{} iterations in 'make_connex(g)'.\n "
                           "Number of components: {}"
                           .format(i, len(components)))


def my_powerlaw_sequence(size, gamma, k_min=1, k_max=None):
    """ Generates a sequence of integers with powerlaw probability
    distribution, with adjustable minimum and maximum values.
    Alternative to networkx.utils.powerlaw_sequence

    P(k) = C k**(-gamma), k_min <= k <= k_max

    Parameters
    ----------
    size : int
        Size of the sample (number of elements in the sequence).
    gamma : float
        Exponent of the powerlaw distribution.
    k_min : int
        Minimum value of the sequence elements.
    k_max : int
        Maximum value (included) of the sequence elements.
        If not informed, is set to sqrt(size).
    """

    # If k_max is not informed, it is set to the closest integer of
    # sqrt(size)
    if k_max is None:
        k_max = int(np.sqrt(size) + 0.5)

    # Checks if k_min is not less than k_max
    if k_min >= k_max:
        raise ValueError("Hey, k_min >= k_max in powerlaw distribution"
                         "\nk_min = {}\nk_max = {}".format(k_min,
                                                           k_max))

    # Generates the sample of values of k (range from k_min to k_max).
    x_sample = np.arange(k_min, k_max+1, dtype=int)
    # Applies the powerlaw function, without normalization.
    p_sample = x_sample**(-float(gamma))

    # Normalization
    norm = p_sample.sum()
    p_sample = p_sample / norm

    # Random sequence generation
    return np.random.choice(x_sample, size, p=p_sample)


def my_poisson_sequence(size, nu, kmin=0):
    # This prevents an (almost) infinite loop in the "while" bellow.
    if nu < kmin:
        raise ValueError("Hey, my_poisson_sequence can't have parameter nu = {} "
                         "smaller than kmin = {}.".format(nu, kmin))

    seq = np.random.poisson(nu, size)

    # If kmin is greater than 0, the invalid values are redrawn from Poisson.
    # This rescales the whole distribution.
    if kmin > 0:
        for i, a in enumerate(seq):
            while seq[i] < kmin:
                # Tries to replace by another valid Poisson number
                seq[i] = np.random.poisson(nu)

    return seq


def make_sequence_even(deg_seq):
    """ Checks if the sum of a sequence of integers is even or odd.
    If it is odd, adds one unit to a random element of the sequence.

    Parameters
    ----------
    deg_seq : numpy.array
        Sequence of numbers. Changes are made in place.
    """
    if sum(deg_seq) % 2 == 1:
        i = rnd.choice(range(len(deg_seq)))
        deg_seq[i] += 1


def make_sequence_3multiple(deg_seq):
    """Checks if the sum of a sequence is a multiple of 3 and, if not,
    adds or subtracts one unit of a random element."""
    # Chooses the number to add.
    add_number = [0, -1, +1][sum(deg_seq) % 3]
    i = rnd.choice(range(len(deg_seq)))
    deg_seq[i] += add_number


# Variable that stores all the possible network types and their
# respective keywords.
layer_keyword_dict = dict()
layer_keyword_dict['ER'] = ['p', 'seed']
layer_keyword_dict['BA'] = ['m', 'seed']
layer_keyword_dict['SF-CM'] = ['gamma', 'k_min', 'seed']
layer_keyword_dict['FILE'] = ['path']


def generate_layer(net_type, size, **kwargs):
    """ Generates a layer for the double-layer network.

    Parameters
    ----------
    net_type : str
        Network type ('ER', 'SF-CM', ...). See 'Suported network types'
        bellow.
    size : int
        Number of nodes.

    As keyword arguments, inform other layer parameters,
    according to layer_keyword_dict.

    Returns
    ----------
    g : nx.Graph
        Generated layer.

    Supported network types
    ----------
    'ER': Erdös-Renyi random graph G(n,p).
        prefix+'_p' = probability for each edge.

    'BA': Barabási-Albert model for scale free graph.
        prefix+'_m' = m parameter. Number of edges to create for
            each new node.

    'SF-CM': Scale-Free network, using configuration
        model.
        prefix+'_gamma' = Exponent of the power law distribution
        prefix+'_k_min' = Minimum degree of each node.

    'clustered-poisson': clustered network with poisson dergee distributions,
        using the extended configuration model proposed by Newman 2009 and
        Miller 2009 (independently). Check nx.random_clustered for more info.
        prefix+''

    'FILE': Network read from a file, saved as 'edgelist' standard (see
        networkx.write_edgelist function).
        prefix+'_meank_i' = Independent average degree - that of regular config. model.
        prefix+'_meank_t' = Triangles average degree - that of triadic config. model.
        prefix+'_k_min' = minimum *independent* degree. Triangle min degree is set to zero.
    """
    # Gets the size, the network type and the seed from input dict
    n = size

    # Elif ladder for the network types

    # Type: Barabasi-Albert network
    if net_type == 'BA':
        m = int(kwargs["m"])
        try:
            seed = int(kwargs["seed"])
            g = nx.barabasi_albert_graph(n, m, seed)
            g.name += ', seed = {}'.format(seed)
        # If no seed is informed, does not use
        except KeyError:
            g = nx.barabasi_albert_graph(n, m)

    # Type: Erdös-Rényi network (G(n,p))
    elif net_type == 'ER':

        # Gets the probability or the average degree.
        try:
            p = float(kwargs['p'])
        except KeyError:
            p = float(kwargs['kmean'])/size

        try:
            seed = int(kwargs["seed"])
            g = nx.erdos_renyi_graph(n, p, seed)
            # Updates the graph name to include the seed
            g.name += ', seed = {}'.format(seed)
        # If no seed is informed, does not use
        except KeyError:
            g = nx.erdos_renyi_graph(n, p)

        # Tries to make the network connex
        make_connex(g, n)

    # Type: Scale-free with configuration model.
    elif net_type == 'SF-CM':
        gamma = float(kwargs['gamma'])
        k_min = int(kwargs['k_min'])

        # Sets the seed.
        try:
            seed = int(kwargs['seed'])
            rnd.seed(seed)
        except KeyError:
            seed = None

        # Generates a valid (even) powerlaw sequence of node degrees.
        deg_seq = my_powerlaw_sequence(n, gamma, k_min)
        make_sequence_even(deg_seq)

        # Generates the graph from the power law sequence.
        g = nx.Graph(nx.configuration_model(deg_seq))
        remove_selfloops(g)

        # Tries to make the graph connex. May fail depending on the
        # graph itself. In this case, try other seeds.
        make_connex(g, n)

        # Sets the name of the graph.
        g.name = "Scale-free_configuration-model - n={:d}, " \
                 "gamma={:0.3f}, " \
                 "k_min={:d}, " \
                 "seed={}".format(n, gamma, k_min, seed)

    # Type: Miller/Newman (2009) model for clustered networks.
    elif net_type == "clustered-poisson":

        # Average independent and triangle degrees
        meank_i = float(kwargs["meank_i"])
        meank_t = float(kwargs["meank_t"])

        # Sets the seed.
        try:
            seed = int(kwargs['seed'])
            rnd.seed(seed)
        except KeyError:
            seed = None

        # Minimum independent degree
        # The triangle degree has no min, i.e., it is zero.
        try:
            kmin_i = int(kwargs["k_min"])
        except KeyError:
            kmin_i = 0

        # Generates the joint degree sequence
        ki_seq = my_poisson_sequence(n, meank_i, kmin_i)
        kt_seq = my_poisson_sequence(n, meank_t)
        make_sequence_even(ki_seq)
        make_sequence_3multiple(kt_seq)
        joint_deg_seq = [(ki, kt) for ki, kt in zip(ki_seq, kt_seq)]

        g = nx.random_clustered_graph(joint_deg_seq)
        remove_selfloops(g)
        make_connex(g, n)

    # Type: Read from file (edgelist type).
    elif net_type == 'FILE':
        fname = kwargs['path']

        try:
            g = nx.read_edgelist(fname, nodetype=int)
        except FileNotFoundError:
            raise FileNotFoundError("Hey, network file was not found"
                                    ". File name: {}".format(fname))
        g.name = "FILE: {}".format(fname)

    # Unrecognized type
    else:
        raise KeyError("Network type '{}' not recognized!".
                       format(net_type))

    rnd.seed()  # Returns the random seed to its standard.
    remove_selfloops(g)  # Removes self edges from network.
    return g


def generate_layer_from_dict(input_dict, prefix):
    """Uses the function 'generate_layer' to create a layer, but gathers
    information from an input dictionary.

    The parameters to be used must start with 'prefix' and be separated
    from the keyword by an underscore '_'.
    """
    size = int(input_dict["network_size"])
    net_type = input_dict[prefix + "_type"]
    prefix_len = len(prefix)

    layer_dict = dict()

    # Imports the valid entries from input dictionary
    for key, value in input_dict.items():
        try:
            # If the key starts with the informed prefix
            if key[:prefix_len] == prefix:
                layer_dict[key[prefix_len+1:]] = value  # Excludes first character after
        # If the keyword is smaller than prefix, it is not considered
        except IndexError:
            continue

    return generate_layer(net_type, size, **layer_dict)


def average_degree(g):
    """Calculates the average degree of an undirected unweighted graph k."""
    return sum(k for k in g.degree().values()) / len(g)
