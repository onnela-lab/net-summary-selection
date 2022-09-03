# -*- coding: utf-8 -*-
"""
Functions related to the generation of the data (reference tables) employed
in our paper.

The models used are either four settings of the Barabási-Albert model,
or the Duplication Mutation Complementation / Duplication with Random Mutation models.
"""

import networkx as nx
import pandas as pd
import random
import scipy.stats as ss
from cost_based_selection import summaries


def BA_ref_table(num_sim_model, num_nodes):
    """ Generation of a reference table under the four Barabási-Albert (BA) models.

    Function to generate a reference table with num_sim_model simulated networks per
    model, where each model is the Barabási-Albert BA(n_1, n_2) model, with n_1
    the number of nodes num_nodes and n_2 equal to 1, 2, 3 or 4.
    The summary statistics computed are the ones defined in summaries.py.

    Args:
        num_sim_model (int):
            the number of simulated networks per BA model.
        num_nodes (int):
            the number of nodes in a simulated network.

    Returns:
        dfModIndex (pandas.core.frame.DataFrame):
            a pandas DataFrame with one column labeled `modIndex` containing the
            different model indexes, 1, 2, 3 or 4 corresponding to the
            parameter value n_2 used.
        dfSummaries (pandas.core.frame.DataFrame):
            a pandas DataFrame containing, for each simulated data (in rows), the
            summary statistic values (in columns).
        dfIsDisc (pandas.core.frame.DataFrame):
            a pandas DataFrame containing in each column the nature of the
            corresponding summary statistic: True for a discrete summary,
            False for a continuous one.
        dfTimes (pandas.core.frame.DataFrame):
            a pandas DataFrame containing, for each simulated data (in rows),
            the time in seconds to compute each summary statistic values
            (in columns).

    """
    num_attachments = [1, 2, 3, 4]

    # Populate the dataset.
    results = {}
    is_discrete_dict = None
    for i in range(num_sim_model):
        for m in num_attachments:
            simulated_graph = nx.barabasi_albert_graph(num_nodes, m)
            summary_dict, time_dict, is_discrete_dict = summaries.compute_summaries(simulated_graph)
            result = {
                'summary': summary_dict,
                'time': time_dict,
                'model_index': m,
            }
            for key, value in result.items():
                results.setdefault(key, []).append(value)

    # Convert to dataframes.
    dfModIndex = pd.DataFrame(results.pop('model_index'), columns=["modIndex"])
    dfIsDisc = pd.DataFrame([is_discrete_dict])
    results = {key: pd.DataFrame(value) for key, value in results.items()}

    return dfModIndex, results['summary'], dfIsDisc, results['time']


def DMC(seed_network, num_nodes, q_mod, q_con):
    """ Simulate one network according to the DMC model.

    Simulate one network according to the Duplication Mutation Complementation
    model (Vázquez et al. 2003).

    A. Vázquez, A. Flammini, A. Maritan, and A. Vespignani.
    Modeling of protein interaction networks. Complexus, 1:38–44, 2003.

    Args:
        seed_network (networkx.classes.graph.Graph):
            a networkx graph which is the seed network used to simulate the
            network.
        num_nodes (int):
            the final number of nodes in the simulated network.
        q_mod (float):
            the probability (between 0 and 1) of edge removal during the
            mutation step.
        q_con (float):
            the probability (between 0 and 1) of connecting the new node and
            the duplicated node during the complementation step.

    Returns:
        sim_network (networkx.classes.graph.Graph):
            the simulated DMC network.

    """

    G = seed_network.copy()
    seed_num_nodes = G.number_of_nodes()
    for v in range(seed_num_nodes, num_nodes):
        # Select a random node u in the graph for duplication
        u = random.choice(list(G.nodes()))
        # Add a new node to the graph
        G.add_node(v)
        # and duplicate the relationships of u to v
        G.add_edges_from([(v, w) for w in G.neighbors(u)])
        # For each neighbors of u
        for w in list(G.neighbors(u)):
            # We generate a Bernoulli random variable with parameter q_mod
            # if it's 1 we remove at random the relationship (v,w) or (u,w)
            if ss.bernoulli.rvs(q_mod):
                edge = random.choice([(v, w), (u, w)])
                G.remove_edge(*edge)  # * to unpack the tuple edge
        # Finally, with probability q_con, add an edge between
        # the duplicated and duplicate nodes
        if ss.bernoulli.rvs(q_con):
            G.add_edge(u, v)
    return G


def DMR(seed_network, num_nodes, q_del, q_new):
    """ Simulate one network according to the DMR model.

    Simulate one network according to the Duplication with Random Mutation
    model (Solé et al. 2002).

    R. V. Solé,  R. Pastor-Satorras, E. Smith, and T. B. Kepler.
    A model of large-scale proteome evolution. Advances in Complex Systems,
    5(1):43--54, 2002.

    Args:
        seed_network (networkx.classes.graph.Graph):
            a networkx graph which is the seed network used to simulate the
            network.
        num_nodes (int):
            the final number of nodes in the simulated network.
        q_del (float):
            the probability (between 0 and 1) of edge removal during the
            mutation step.
        q_new (float):
            the probability (between 0 and 1) of connecting the new node and
            the duplicated node is q_new divided by the initial number of nodes
            at the start of the network construction step.

    Returns:
        sim_network (networkx.classes.graph.Graph):
            the simulated DMC network.

    """

    G = seed_network.copy()
    seed_num_nodes = G.number_of_nodes()
    for v in range(seed_num_nodes, num_nodes):
        node_list = list(G.nodes())
        # Select a random node u in the graph for duplication
        u = random.choice(node_list)
        # Add a new node to the graph
        G.add_node(v)
        # and duplicate the relationships of u to v
        G.add_edges_from([(v, w) for w in G.neighbors(u)])
        # For each neighbor of v
        for w in list(G.neighbors(v)):
            # with probability q_del we remove the link (v,w)
            if ss.bernoulli.rvs(q_del):
                G.remove_edge(v, w)
        # And for all other node, we establish a link with v with
        # proba q_new/number_of_nodes_before_duplication
        nodes_to_link = random.sample(node_list, ss.binom.rvs(v, q_new / v))
        G.add_edges_from([(v, x) for x in nodes_to_link])

    return G


def DMC_DMR_ref_table(seed_network, num_sim_model, num_nodes):
    """ Generation of a reference table under the DMC and DMR models.

    Function to generate a reference table with num_sim_model simulated data per
    model, where the models are the Duplication Mutation Complementation model
    and the Duplication with Random Mutation model. The DMC model carries the
    index 1, while DMR carries the index 2. The priors on parameters are
    Uniform [0.25, 0.75] for all model parameters.
    The summary statistics computed are the ones defined in summaries.py.

    Args:
        seed_network (networkx.classes.graph.Graph):
            a networkx graph which is the seed network used to simulate each
            network.
        num_sim_model (int):
            the number of simulated networks per model.
        num_nodes (int):
            the number of nodes in a simulated network.

    Returns:
        dfModIndex (pandas.core.frame.DataFrame):
            a pandas DataFrame with one column labeled `modIndex` containing the
            different model indexes, 1 for DMC, 2 for DMR.
        dfSummaries (pandas.core.frame.DataFrame):
            a pandas DataFrame containing, for each simulated data (in rows), the
            summary statistic values (in columns).
        dfIsDisc (pandas.core.frame.DataFrame):
            a pandas DataFrame containing in each column the nature of the
            corresponding summary statistic: True for a discrete summary,
            False for a continuous one.
        dfTimes (pandas.core.frame.DataFrame):
            a pandas DataFrame containing, for each simulated data (in rows),
            the time in seconds to compute each summary statistic values
            (in columns).

    """

    resList1 = []
    resList2 = []

    for i in range(num_sim_model):

        # Parameter generation
        q_mod_sim = ss.uniform.rvs(0.25, 0.5, size=1)[0]
        q_con_sim = ss.uniform.rvs(0.25, 0.5, size=1)[0]

        q_del_sim = ss.uniform.rvs(0.25, 0.5, size=1)[0]
        q_new_sim = ss.uniform.rvs(0.25, 0.5, size=1)[0]

        simuGraphDMC = DMC(seed_network=seed_network, num_nodes=num_nodes,
                           q_mod=q_mod_sim, q_con=q_con_sim)
        simuGraphDMR = DMR(seed_network=seed_network, num_nodes=num_nodes,
                           q_del=q_del_sim, q_new=q_new_sim)

        dictSums1, dictTimes1, dictIsDisc1 = summaries.compute_summaries(simuGraphDMC)
        dictSums2, dictTimes2, dictIsDisc2 = summaries.compute_summaries(simuGraphDMR)

        resList1.append([1, dictSums1, dictTimes1, dictIsDisc1])
        resList2.append([2, dictSums2, dictTimes2, dictIsDisc2])

    modIndex = [1]*num_sim_model + [2]*num_sim_model

    listOfSummaries = []
    listOfIsDisc = []
    listOfTimes = []

    # Model DMC (indexed 1)
    for simIdx in range(num_sim_model):
        listOfSummaries += [resList1[simIdx][1]]
        listOfTimes += [resList1[simIdx][2]]

    # Model DMR (indexed 2)
    for simIdx in range(num_sim_model):
        listOfSummaries += [resList2[simIdx][1]]
        listOfTimes += [resList2[simIdx][2]]

    listOfIsDisc = [resList1[0][3]]

    dfModIndex = pd.DataFrame(modIndex, columns=["modIndex"])
    dfSummaries = pd.DataFrame(listOfSummaries)
    dfIsDisc = pd.DataFrame(listOfIsDisc)
    dfTimes = pd.DataFrame(listOfTimes)

    return dfModIndex, dfSummaries, dfIsDisc, dfTimes
