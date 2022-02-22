# -*- coding: utf-8 -*-
""" Summary statistics related functions.

Codes to compute 54 summary statistics on an undirected graph,
(plus 4 noise variables), as well as evaluation times in seconds
and whether or not the summary statistic is discrete.
"""

import networkx as nx
import numpy as np
import scipy.stats as ss
import time
from networkx.algorithms import approximation


def _compute_summaries_v1(G):
    """
    Compute network features, computational times and their nature.

    Evaluate 54 summary statistics of a network G, plus 4 noise variables,
    store the computational time to evaluate each summary statistic, and keep
    track of their nature (discrete or not).

    Args:
        G (networkx.classes.graph.Graph):
            an undirected networkx graph.

    Returns:
        resDicts (tuple):
            a tuple containing the elements:
            - dictSums (dict): a dictionary with the name of the summaries
            as keys and the summary statistic values as values;
            - dictTimes (dict): a dictionary with the name of the summaries
            as keys and the time to compute each one as values;
            - dictIsDist (dict): a dictionary indicating if the summary is
            discrete (True) or not (False).
    """

    dictSums = dict()   # Will store the summary statistic values
    dictTimes = dict()  # Will store the evaluation times
    dictIsDisc = dict() # Will store the summary statistic nature

    # Extract the largest connected component
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G_lcc = G.subgraph(Gcc[0])

    # Number of edges
    start = time.time()
    dictSums["num_edges"] = G.number_of_edges()
    dictTimes["num_edges"] = time.time()-start
    dictIsDisc["num_edges"] = True

    # Number of connected components
    start = time.time()
    dictSums["num_of_CC"] = nx.number_connected_components(G)
    dictTimes["num_of_CC"] = time.time() - start
    dictIsDisc["num_of_CC"] = True

    # Number of nodes in the largest connected component
    start = time.time()
    dictSums["num_nodes_LCC"] = nx.number_of_nodes(G_lcc)
    dictTimes["num_nodes_LCC"] = time.time() - start
    dictIsDisc["num_nodes_LCC"] = True

    # Number of edges in the largest connected component
    start = time.time()
    dictSums["num_edges_LCC"] = G_lcc.number_of_edges()
    dictTimes["num_edges_LCC"] = time.time() - start
    dictIsDisc["num_edges_LCC"] = True

    # Diameter of the largest connected component
    start = time.time()
    dictSums["diameter_LCC"] = nx.diameter(G_lcc)
    dictTimes["diameter_LCC"] = time.time() - start
    dictIsDisc["diameter_LCC"] = True

    # Average geodesic distance (shortest path length in the LCC)
    start = time.time()
    dictSums["avg_geodesic_dist_LCC"] = nx.average_shortest_path_length(G_lcc)
    dictTimes["avg_geodesic_dist_LCC"] = time.time() - start
    dictIsDisc["avg_geodesic_dist_LCC"] = False

    # Average degree of the neighborhood of each node
    start = time.time()
    dictSums["avg_deg_connectivity"] = np.mean(list(nx.average_degree_connectivity(G).values()))
    dictTimes["avg_deg_connectivity"] = time.time() - start
    dictIsDisc["avg_deg_connectivity"] = False

    # Average degree of the neighbors of each node in the LCC
    start = time.time()
    dictSums["avg_deg_connectivity_LCC"] = np.mean(list(nx.average_degree_connectivity(G_lcc).values()))
    dictTimes["avg_deg_connectivity_LCC"] = time.time() - start
    dictIsDisc["avg_deg_connectivity_LCC"] = False

    # Recover the degree distribution
    start_degree_extract = time.time()
    degree_vals = list(dict(G.degree()).values())
    degree_extract_time = time.time() - start_degree_extract

    # Entropy of the degree distribution
    start = time.time()
    dictSums["degree_entropy"] = ss.entropy(degree_vals)
    dictTimes["degree_entropy"] = time.time() - start + degree_extract_time
    dictIsDisc["degree_entropy"] = False

    # Maximum degree
    start = time.time()
    dictSums["degree_max"] = max(degree_vals)
    dictTimes["degree_max"] = time.time() - start + degree_extract_time
    dictIsDisc["degree_max"] = True

    # Average degree
    start = time.time()
    dictSums["degree_mean"] = np.mean(degree_vals)
    dictTimes["degree_mean"] = time.time() - start + degree_extract_time
    dictIsDisc["degree_mean"] = False

    # Median degree
    start = time.time()
    dictSums["degree_median"] = np.median(degree_vals)
    dictTimes["degree_median"] = time.time() - start + degree_extract_time
    dictIsDisc["degree_median"] = False

    # Standard deviation of the degree distribution
    start = time.time()
    dictSums["degree_std"] = np.std(degree_vals)
    dictTimes["degree_std"] = time.time() - start + degree_extract_time
    dictIsDisc["degree_std"] = False

    # Quantile 25%
    start = time.time()
    dictSums["degree_q025"] = np.quantile(degree_vals, 0.25)
    dictTimes["degree_q025"] = time.time() - start + degree_extract_time
    dictIsDisc["degree_q025"] = False

    # Quantile 75%
    start = time.time()
    dictSums["degree_q075"] = np.quantile(degree_vals, 0.75)
    dictTimes["degree_q075"] = time.time() - start + degree_extract_time
    dictIsDisc["degree_q075"] = False

    # This is already computed above as "avg_geodesic_dist_LCC".
    # Average geodesic distance
    start = time.time()
    dictSums["avg_shortest_path_length_LCC"] = nx.average_shortest_path_length(G_lcc)
    dictTimes["avg_shortest_path_length_LCC"] = time.time() - start
    dictIsDisc["avg_shortest_path_length_LCC"] = False

    # Average global efficiency:
    # The efficiency of a pair of nodes in a graph is the multiplicative
    # inverse of the shortest path distance between the nodes.
    # The average global efficiency of a graph is the average efficiency of
    # all pairs of nodes.
    start = time.time()
    dictSums["avg_global_efficiency"] = nx.global_efficiency(G)
    dictTimes["avg_global_efficiency"] = time.time() - start
    dictIsDisc["avg_global_efficiency"] = False

    # Harmonic mean which is 1/avg_global_efficiency
    # This seems like it's missing the 1 / x, so I'm dropping it.
    start = time.time()
    dictSums["harmonic_mean"] = nx.global_efficiency(G)
    dictTimes["harmonic_mean"] = time.time() - start
    dictIsDisc["harmonic_mean"] = False

    # Average local efficiency
    # The local efficiency of a node in the graph is the average global
    # efficiency of the subgraph induced by the neighbors of the node.
    # The average local efficiency is the average of the
    # local efficiencies of each node.
    start = time.time()
    dictSums["avg_local_efficiency_LCC"] = nx.local_efficiency(G_lcc)
    dictTimes["avg_local_efficiency_LCC"] = time.time() - start
    dictIsDisc["avg_local_efficiency_LCC"] = False

    # Node connectivity
    # The node connectivity is equal to the minimum number of nodes that
    # must be removed to disconnect G or render it trivial.
    # Only on the largest connected component here.
    start = time.time()
    dictSums["node_connectivity_LCC"] = nx.node_connectivity(G_lcc)
    dictTimes["node_connectivity_LCC"] = time.time() - start
    dictIsDisc["node_connectivity_LCC"] = True

    # Edge connectivity
    # The edge connectivity is equal to the minimum number of edges that
    # must be removed to disconnect G or render it trivial.
    # Only on the largest connected component here.
    start = time.time()
    dictSums["edge_connectivity_LCC"] = nx.edge_connectivity(G_lcc)
    dictTimes["edge_connectivity_LCC"] = time.time() - start
    dictIsDisc["edge_connectivity_LCC"] = True

    # Graph transitivity
    # 3*times the number of triangles divided by the number of triades
    start = time.time()
    dictSums["transitivity"] = nx.transitivity(G)
    dictTimes["transitivity"] = time.time() - start
    dictIsDisc["transitivity"] = False

    # Number of triangles
    start = time.time()
    dictSums["num_triangles"] = np.sum( list( nx.triangles(G).values() ) )/3
    dictTimes["num_triangles"] = time.time() - start
    dictIsDisc["num_triangles"] = True

    # Estimate of the average clustering coefficient of G:
    # Average local clustering coefficient, with local clustering coefficient
    # defined as C_i = (nbr of pairs of neighbors of i that are connected)/(nbr of pairs of neighbors of i)
    start = time.time()
    dictSums["avg_clustering_coef"] = nx.average_clustering(G)
    dictTimes["avg_clustering_coef"] = time.time() - start
    dictIsDisc["avg_clustering_coef"] = False

    # Square clustering (averaged over nodes):
    # the fraction of possible squares that exist at the node.

    # We average it over nodes
    start = time.time()
    dictSums["square_clustering_mean"] = np.mean( list( nx.square_clustering(G).values() ) )
    dictTimes["square_clustering_mean"] = time.time() - start
    dictIsDisc["square_clustering_mean"] = False

    # We compute the median
    start = time.time()
    dictSums["square_clustering_median"] = np.median( list( nx.square_clustering(G).values() ) )
    dictTimes["square_clustering_median"] = time.time() - start
    dictIsDisc["square_clustering_median"] = False

    # We compute the standard deviation
    start = time.time()
    dictSums["square_clustering_std"] = np.std( list( nx.square_clustering(G).values() ) )
    dictTimes["square_clustering_std"] = time.time() - start
    dictIsDisc["square_clustering_std"] = False

    # Number of 2-cores
    start = time.time()
    dictSums["num_2cores"] = len(nx.k_core(G, k=2))
    dictTimes["num_2cores"] = time.time() - start
    dictIsDisc["num_2cores"] = True

    # Number of 3-cores
    start = time.time()
    dictSums["num_3cores"] = len(nx.k_core(G, k=3))
    dictTimes["num_3cores"] = time.time() - start
    dictIsDisc["num_3cores"] = True

    # Number of 4-cores
    start = time.time()
    dictSums["num_4cores"] = len(nx.k_core(G, k=4))
    dictTimes["num_4cores"] = time.time() - start
    dictIsDisc["num_4cores"] = True

    # Number of 5-cores
    start = time.time()
    dictSums["num_5cores"] = len(nx.k_core(G, k=5))
    dictTimes["num_5cores"] = time.time() - start
    dictIsDisc["num_5cores"] = True

    # Number of 6-cores
    start = time.time()
    dictSums["num_6cores"] = len(nx.k_core(G, k=6))
    dictTimes["num_6cores"] = time.time() - start
    dictIsDisc["num_6cores"] = True

    # Number of k-shells
    # The k-shell is the subgraph induced by nodes with core number k.
    # That is, nodes in the k-core that are not in the k+1-core

    # Number of 2-shells
    start = time.time()
    dictSums["num_2shells"] = len( nx.k_shell(G, 2) )
    dictTimes["num_2shells"] = time.time() - start
    dictIsDisc["num_2shells"] = True

    # Number of 3-shells
    start = time.time()
    dictSums["num_3shells"] = len( nx.k_shell(G, 3) )
    dictTimes["num_3shells"] = time.time() - start
    dictIsDisc["num_3shells"] = True

    # Number of 4-shells
    start = time.time()
    dictSums["num_4shells"] = len( nx.k_shell(G, 4) )
    dictTimes["num_4shells"] = time.time() - start
    dictIsDisc["num_4shells"] = True

    # Number of 5-shells
    start = time.time()
    dictSums["num_5shells"] = len( nx.k_shell(G, 5) )
    dictTimes["num_5shells"] = time.time() - start
    dictIsDisc["num_5shells"] = True

    # Number of 6-shells
    start = time.time()
    dictSums["num_6shells"] = len( nx.k_shell(G, 6) )
    dictTimes["num_6shells"] = time.time() - start
    dictIsDisc["num_6shells"] = True


    start = time.time()
    listOfCliques = list(nx.enumerate_all_cliques(G))
    enum_all_cliques_time = time.time() - start

    # Number of 4-cliques
    start = time.time()
    n4Clique = 0
    for li in listOfCliques:
        if len(li)==4:
            n4Clique += 1
    dictSums["num_4cliques"] = n4Clique
    dictTimes["num_4cliques"] = time.time() - start + enum_all_cliques_time
    dictIsDisc["num_4cliques"] = True

    # Number of 5-cliques
    start = time.time()
    n5Clique = 0
    for li in listOfCliques:
        if len(li)==5:
            n5Clique += 1
    dictSums["num_5cliques"] = n5Clique
    dictTimes["num_5cliques"] = time.time() - start + enum_all_cliques_time
    dictIsDisc["num_5cliques"] = True

    # Maximal size of a clique in the graph
    start = time.time()
    dictSums["max_clique_size"] = len(approximation.clique.max_clique(G))
    dictTimes["max_clique_size"] = time.time() - start
    dictIsDisc["max_clique_size"] = True

    # Approximated size of a large clique in the graph
    start = time.time()
    dictSums["large_clique_size"] = approximation.large_clique_size(G)
    dictTimes["large_clique_size"] = time.time() - start
    dictIsDisc["large_clique_size"] = True

    # Number of shortest path of size k
    start = time.time()
    listOfPLength = list(nx.shortest_path_length(G))
    path_length_time = time.time() - start

    # when k = 3
    start = time.time()
    n3Paths = 0
    for node in G.nodes():
        tmp = list( listOfPLength[node][1].values() )
        n3Paths += tmp.count(3)
    dictSums["num_shortest_3paths"] = n3Paths/2
    dictTimes["num_shortest_3paths"] = time.time() - start + path_length_time
    dictIsDisc["num_shortest_3paths"] = True

    # when k = 4
    start = time.time()
    n4Paths = 0
    for node in G.nodes():
        tmp = list( listOfPLength[node][1].values() )
        n4Paths += tmp.count(4)
    dictSums["num_shortest_4paths"] = n4Paths/2
    dictTimes["num_shortest_4paths"] = time.time() - start + path_length_time
    dictIsDisc["num_shortest_4paths"] = True

    # when k = 5
    start = time.time()
    n5Paths = 0
    for node in G.nodes():
        tmp = list( listOfPLength[node][1].values() )
        n5Paths += tmp.count(5)
    dictSums["num_shortest_5paths"] = n5Paths/2
    dictTimes["num_shortest_5paths"] = time.time() - start + path_length_time
    dictIsDisc["num_shortest_5paths"] = True

    # when k = 6
    start = time.time()
    n6Paths = 0
    for node in G.nodes():
        tmp = list( listOfPLength[node][1].values() )
        n6Paths += tmp.count(6)
    dictSums["num_shortest_6paths"] = n6Paths/2
    dictTimes["num_shortest_6paths"] = time.time() - start + path_length_time
    dictIsDisc["num_shortest_6paths"] = True


    # Size of the minimum (weight) node dominating set:
    # A subset of nodes where each node not in the subset has for direct
    # neighbor a node of the dominating set.
    start = time.time()
    T = approximation.min_weighted_dominating_set(G)
    dictSums["size_min_node_dom_set"] = len(T)
    dictTimes["size_min_node_dom_set"] = time.time() - start
    dictIsDisc["size_min_node_dom_set"] = True

    # Idem but with the edge dominating set
    start = time.time()
    T = approximation.min_edge_dominating_set(G)
    dictSums["size_min_edge_dom_set"] = 2*len(T) # times 2 to have a number of nodes
    dictTimes["size_min_edge_dom_set"] = time.time() - start
    dictIsDisc["size_min_edge_dom_set"] = True

    # The Wiener index of a graph is the sum of the shortest-path distances
    # between each pair of reachable nodes. For pairs of nodes in undirected graphs,
    # only one orientation of the pair is counted.
    # (On LCC otherwise inf)
    start = time.time()
    dictSums["wiener_index_LCC"] = nx.wiener_index(G_lcc)
    dictTimes["wiener_index_LCC"] = time.time() - start
    dictIsDisc["wiener_index_LCC"] = True


    # Betweenness node centrality (averaged over nodes):
    # at node u it is defined as B_u = sum_i,j sigma(i,u,j)/sigma(i,j)
    # where sigma is the number of shortest path between i and j going through u or not

    start = time.time()
    betweenness = list(nx.betweenness_centrality(G).values())
    time_betweenness = time.time() - start

    # Averaged across nodes
    start = time.time()
    dictSums["betweenness_centrality_mean"] = np.mean( betweenness )
    dictTimes["betweenness_centrality_mean"] = time.time() - start + time_betweenness
    dictIsDisc["betweenness_centrality_mean"] = False

    # Maximum across nodes
    start = time.time()
    dictSums["betweenness_centrality_max"] = max( betweenness )
    dictTimes["betweenness_centrality_max"] = time.time() - start + time_betweenness
    dictIsDisc["betweenness_centrality_max"] = False


    # Central point dominance
    # CPD = sum_u(B_max - B_u)/(N-1)
    start = time.time()
    dictSums["central_point_dominance"] = sum( max( betweenness ) - np.array(betweenness) )/(len(betweenness)-1)
    dictTimes["central_point_dominance"] = time.time() - start + time_betweenness
    dictIsDisc["central_point_dominance"] = False

    # Estrata index : sum_i^n exp(lambda_i)
    # with n the number of nodes, lamda_i the i-th eigen value of the adjacency matrix of G
    start = time.time()
    dictSums["Estrata_index"] = nx.estrada_index(G)
    dictTimes["Estrata_index"] = time.time() - start
    dictIsDisc["Estrata_index"] = False

    # Eigenvector centrality
    # For each node, it is the average eigenvalue centrality of its neighbors,
    # where centrality of node i is taken as the i-th coordinate of x
    # such that Ax = lambda*x (for the maximal eigen value)

    # Averaged
    start = time.time()
    dictSums["avg_eigenvec_centrality"] = np.mean( list( nx.eigenvector_centrality_numpy(G).values() ) )
    dictTimes["avg_eigenvec_centrality"] = time.time() - start
    dictIsDisc["avg_eigenvec_centrality"] = False

    # Maximum
    start = time.time()
    dictSums["max_eigenvec_centrality"] = max( list( nx.eigenvector_centrality_numpy(G).values() ) )
    dictTimes["max_eigenvec_centrality"] = time.time() - start
    dictIsDisc["max_eigenvec_centrality"] = False


    ### Noise generation ###

    # Noise simulated from a Normal(0,1) distribution
    start = time.time()
    dictSums["noise_Gauss"] = ss.norm.rvs(0,1)
    dictTimes["noise_Gauss"] = time.time() - start
    dictIsDisc["noise_Gauss"] = False

    # Noise simulated from a Uniform distribution [0-50]
    start = time.time()
    dictSums["noise_Unif"] = ss.uniform.rvs(0,50)
    dictTimes["noise_Unif"] = time.time() - start
    dictIsDisc["noise_Unif"] = False

    # Noise simulated from a Bernoulli B(0.5) distribution
    start = time.time()
    dictSums["noise_Bern"] = ss.bernoulli.rvs(0.5)
    dictTimes["noise_Bern"] = time.time() - start
    dictIsDisc["noise_Bern"] = True

    # Noise simulated from a discrete uniform distribution [0,50[
    start = time.time()
    dictSums["noise_disc_Unif"] = ss.randint.rvs(0, 50)
    dictTimes["noise_disc_Unif"] = time.time() - start
    dictIsDisc["noise_disc_Unif"] = True

    resDicts = (dictSums, dictTimes, dictIsDisc)

    return resDicts


class SummaryResults:
    def __init__(self) -> None:
        self.results = {}
        self._current = None
        self._current_key = None

    def __call__(self, key, is_discrete, depends_on=None):
        if key in self.results:
            raise RuntimeError
        if self._current_key:
            raise RuntimeError
        self._current_key = key
        self._current = {
            'is_discrete': is_discrete,
            'depends_on': depends_on or [],
        }
        return self

    def __enter__(self):
        self._current['start'] = time.time()

    def __exit__(self, *args):
        end = time.time()
        if self._current_key is None:
            raise RuntimeError
        self._current['end'] = end
        self._current['duration'] = end - self._current['start']
        self._current['time'] = self._current['duration'] \
            + sum(self.results[other]['time'] for other in self._current['depends_on'])
        self.results[self._current_key] = self._current
        self._current = self._current_key = None

    @property
    def value(self):
        raise NotImplementedError

    @value.setter
    def value(self, value):
        self._current['value'] = value


def dod2array(dod: dict, num_nodes: int, fill_value=np.nan):
    array = fill_value * np.ones((num_nodes, num_nodes))
    for i, other in dod.items():
        for j, value in other.items():
            array[i, j] = value
    return array


def _compute_summaries_v2(graph: nx.Graph, return_full_results=False):
    assert nx.is_connected(graph), "the graph is not connected"
    results = SummaryResults()

    with results('num_edges', True):
        results.value = graph.number_of_edges()

    with results('connected_components', None):
        connected_components = list(nx.connected_components(graph))

    with results('num_of_CC', True, depends_on=['connected_components']):
        results.value = len(connected_components)

    with results('avg_deg_connectivity', False):
        results.value = np.mean(list(nx.average_degree_connectivity(graph).values()))

    with results('degrees', None):
        degrees = np.asarray([d for _, d in graph.degree()])

    degree_statistics = {
        'entropy': (False, ss.entropy),
        'max': (True, np.max),
        'median': (False, np.median),
        'mean': (False, np.mean),
        'std': (False, np.std),
        'q025': (False, lambda x: np.quantile(x, 0.25)),
        'q075': (False, lambda x: np.quantile(x, 0.75)),
    }

    for name, (disc, func) in degree_statistics.items():
        with results(f'degree_{name}', disc, depends_on=['degrees']):
            results.value = func(degrees)

    with results('num_triangles', True):
        triangles = np.sum(list(nx.triangles(graph).values())) / 3
        results.value = triangles

    with results('transitivity', False):
        results.value = nx.transitivity(graph)

    with results('avg_clustering_coef', False):
        results.value = nx.average_clustering(graph)

    # Betweenness centrality measures.
    with results('betweenness_centrality', None):
        betweenness = np.asarray(list(nx.betweenness_centrality(graph).values()))

    with results('betweenness_centrality_mean', False, depends_on=['betweenness_centrality']):
        results.value = betweenness.mean()

    with results('betweenness_centrality_max', False, depends_on=['betweenness_centrality']):
        results.value = betweenness.max()

    with results('central_point_dominance', False, depends_on=['betweenness_centrality']):
        results.value = sum(max(betweenness) - betweenness) / (len(betweenness) - 1)

    with results('Estrata_index', False):  # keeping typo for consistency.
        results.value = nx.estrada_index(graph)

    # Eigenvector centrality measures.
    with results('eigenvector_centrality', None):
        eigenvector_centrality = np.asarray(list(nx.eigenvector_centrality_numpy(graph).values()))

    with results('avg_eigenvec_centrality', False):
        results.value = eigenvector_centrality.mean()

    with results('max_eigenvec_centrality', False):
        results.value = eigenvector_centrality.max()

    # Clique information.
    with results('cliques', None):
        cliques = list(nx.enumerate_all_cliques(graph))
        clique_sizes = np.asarray([len(c) for c in cliques])

    for clique_size in [4, 5]:
        with results(f'num_{clique_size}cliques', True, depends_on=['cliques']):
            results.value = np.sum(clique_sizes == clique_size)

    with results('max_clique_size', True, depends_on=['cliques']):
        results.value = np.max(clique_sizes)

    # Square clustering information.
    with results('square_clustering', None):
        square_clustering = np.asarray(list(nx.square_clustering(graph).values()))

    square_clustering_stats = {
        'mean': np.mean,
        'median': np.median,
        'std': np.std,
    }
    for key, func in square_clustering_stats.items():
        with results(f'square_clustering_{key}', False, depends_on=['square_clustering']):
            results.value = func(square_clustering)

    # Core information.
    with results('core_number', None):
        core_number = nx.core_number(graph)

    cores = [2, 3, 4, 5, 6]
    for core in cores:
        with results(f'num_{core}cores', True, depends_on=['core_number']):
            results.value = len(nx.k_core(graph, core, core_number))

    # Shell information.
    shells = [2, 3, 4, 5, 6]
    for shell in shells:
        with results(f'num_{shell}shells', True, depends_on=['core_number']):
            results.value = len(nx.k_shell(graph, shell, core_number))

    # Shortest path lengths.
    with results('shortest_paths', None):
        shortest_paths = dict(nx.shortest_path_length(graph))
        shortest_paths = dod2array(shortest_paths, graph.number_of_nodes(), np.inf)

    for length in [3, 4, 5, 6]:
        with results(f'num_shortest_{length}paths', True, depends_on=['shortest_paths']):
            results.value = np.sum(shortest_paths == length) / 2

    with results('size_min_node_dom_set', True):
        results.value = len(approximation.min_weighted_dominating_set(graph))

    with results('size_min_edge_dom_set', True):
        results.value = len(approximation.min_edge_dominating_set(graph)) * 2

    with results('avg_global_efficiency', False, depends_on=['shortest_paths']):
        results.value = np.mean(1 / shortest_paths[shortest_paths > 0])

    # Statistics just on the largest connected component.
    with results('largest_connected_component', None, depends_on=['connected_components']):
        nodes = max(connected_components, key=len)
        # We `copy` to avoid having the inefficient SubGraphView.
        largest_connected_component: nx.Graph = graph.subgraph(nodes).copy()

    with results('num_nodes_LCC', True, depends_on=['largest_connected_component']):
        results.value = largest_connected_component.number_of_nodes()

    with results('num_edges_LCC', True, depends_on=['largest_connected_component']):
        results.value = largest_connected_component.number_of_edges()

    with results('lcc_shortest_paths', None, depends_on=['largest_connected_component']):
        lcc_shortest_paths = dict(nx.shortest_path_length(largest_connected_component))

    with results('lcc_eccentricities', None):
        eccentricities = nx.eccentricity(largest_connected_component, sp=lcc_shortest_paths)

    with results('diameter_LCC', True, depends_on=['lcc_eccentricities']):
        results.value = nx.diameter(None, eccentricities)

    with results('avg_geodesic_dist_LCC', False, depends_on=['lcc_shortest_paths']):
        results.value = np.mean([
            length for source, lengths in lcc_shortest_paths.items()
            for target, length in lengths.items() if source != target
        ])

    with results('avg_deg_connectivity_LCC', False, depends_on=['largest_connected_component']):
        connectivity = nx.average_degree_connectivity(largest_connected_component)
        results.value = np.mean(list(connectivity.values()))

    with results('avg_local_efficiency_LCC', False, depends_on=['largest_connected_component']):
        results.value = nx.local_efficiency(largest_connected_component)

    with results('node_connectivity_LCC', True, depends_on=['largest_connected_component']):
        results.value = nx.node_connectivity(largest_connected_component)

    with results('edge_connectivity_LCC', True, depends_on=['largest_connected_component']):
        results.value = nx.edge_connectivity(largest_connected_component)

    # Generate random noise.
    rvs = {
        'Gauss': (False, lambda: np.random.normal(0, 1)),
        'Unif': (False, lambda: np.random.uniform(0, 50)),
        'Bern': (True, lambda: int(np.random.uniform(0, 1) < 0.5)),
        'disc_Unif': (True, lambda: np.random.randint(0, 50)),
    }
    for key, (disc, func) in rvs.items():
        with results(f'noise_{key}', disc):
            results.value = func()

    if return_full_results:
        return results

    # Transpose the dictionaries.
    dicts = {}
    for key, result in results.results.items():
        if 'value' not in result:
            continue
        for attr in ['value', 'time', 'is_discrete']:
            dicts.setdefault(attr, {})[key] = result[attr]
    return dicts['value'], dicts['time'], dicts['is_discrete']


compute_summaries = _compute_summaries_v1
