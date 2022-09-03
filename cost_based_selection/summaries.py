import time
import numpy as np
import networkit as nk
import networkx as nx
from scipy import stats
import typing


def global_efficiency(graph, distances: np.ndarray = None):
    if distances is None:
        apsp = nk.distance.APSP(graph)
        apsp.run()
        distances = np.asarray(apsp.getDistances())
    distances = distances[(distances > 0) & (distances < graph.numberOfNodes())]
    if distances.size == 0:
        return 0
    denom = (graph.numberOfNodes() - 1) * graph.numberOfNodes()
    return np.sum(1 / distances) / denom


def get_compacted_subgraph(graph, nodes):
    subgraph = nk.graphtools.subgraphFromNodes(graph, nodes)
    node_ids = nk.graphtools.getContinuousNodeIds(subgraph)
    return nk.graphtools.getCompactedGraph(subgraph, node_ids)


def local_efficiency(graph):
    efficiencies = []
    for node in graph.iterNodes():
        subgraph = get_compacted_subgraph(graph, graph.iterNeighbors(node))
        efficiencies.append(global_efficiency(subgraph))
    return np.mean(efficiencies)


class SummaryResults:
    """
    Container for storing summary statistics, including timings. Supports amortizing compute cost
    across different summary statistics at runtime while keeping track of the time had the
    statistics been evaluated independently.
    """
    def __init__(self) -> None:
        self.results = {}
        self._current = None
        self._current_key = None

    def __call__(self, key: str, is_discrete: bool, depends_on: typing.Iterable[str] = None) \
            -> 'SummaryResults':
        """
        Start a context for evaluating a statistic.

        Args:
            key: Name of the summary statistic.
            is_discrete: Whether the statistic is discrete or continuous.
            depends_on: Optional list of statistic names this statistic depends on. Compute times
                of the dependencies will be added.
        """
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
        # Demand a value if we have set discrete to False or True. Demand no value if None.
        if self._current['is_discrete'] is None:
            assert 'value' not in self._current
        else:
            assert 'value' in self._current
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


def dod2array(dod: dict, num_nodes: int, fill_value: float = np.nan) -> np.ndarray:
    """
    Convert a dictionary of dictionaries to a numpy array.

    dod: Dictionary of dictionaries of values.
    num_nodes: Number of nodes in the network.
    fill_value: Default value if a value is missing in the dictionary of dictionaries.
    """
    array = fill_value * np.ones((num_nodes, num_nodes))
    for i, other in dod.items():
        for j, value in other.items():
            array[i, j] = value
    return array


def compute_summaries(graph: nx.Graph, return_full_results: bool = False,
                      include_connectivity: bool = False):
    """
    Compute summary statistics.

    Args:
        graph: Graph to compute summary statistics for.
        return_full_results: Return the `SummaryResults` object instead of collapsing into a
            dictionary.
        include_connectivity: Whether to include edge and node connectivity, two very expensive
            summary statistics.
    """
    # Convert to networkit graph for faster performance.
    nkgraph: nk.Graph = nk.nxadapter.nx2nk(graph)

    results = SummaryResults()

    with results('avg_deg_connectivity', False):
        results.value = np.mean(list(nx.average_degree_connectivity(graph).values()))

    with results('degrees', None):
        degrees = np.asarray([d for _, d in graph.degree()])

    degree_statistics = {
        'entropy': (False, stats.entropy),
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
        betweenness = nk.centrality.Betweenness(nkgraph, normalized=True)
        betweenness.run()
        betweenness = np.asarray(betweenness.scores())

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
        lscc = nk.centrality.LocalSquareClusteringCoefficient(nkgraph)
        square_clustering = np.asarray(lscc.run().scores())

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

    with results('shortest_path_lengths', None):
        # shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
        # shortest_path_lengths_np = dod2array(shortest_path_lengths, graph.number_of_nodes(),
        #                                      np.inf)
        apsp = nk.distance.APSP(nkgraph).run()
        shortest_path_lengths_np = np.asarray(apsp.getDistances())

    for length in [3, 4, 5, 6]:
        with results(f'num_shortest_{length}paths', True, depends_on=['shortest_path_lengths']):
            results.value = np.sum(shortest_path_lengths_np == length) / 2

    with results('avg_global_efficiency', False, depends_on=['shortest_path_lengths']):
        results.value = np.mean(1 / shortest_path_lengths_np[shortest_path_lengths_np > 0])

    with results('size_min_node_dom_set', True):
        results.value = len(nx.approximation.min_weighted_dominating_set(graph))

    with results('size_min_edge_dom_set', True):
        results.value = len(nx.approximation.min_edge_dominating_set(graph)) * 2

    # Components.

    with results('connected_components', None):
        components = list(nx.connected_components(graph))

    with results('num_of_CC', True, depends_on=['connected_components']):
        results.value = len(components)

    with results('largest_connected_component', None, depends_on=['connected_components']):
        nodes = max(components, key=len)
        largest_connected_component = graph.subgraph(nodes).copy()
    nklcc = nk.nxadapter.nx2nk(largest_connected_component)

    with results('num_edges_LCC', True, depends_on=['largest_connected_component']):
        results.value = largest_connected_component.number_of_edges()

    with results('num_nodes_LCC', True, depends_on=['largest_connected_component']):
        results.value = largest_connected_component.number_of_nodes()

    with results('avg_deg_connectivity_LCC', False, depends_on=['largest_connected_component']):
        results.value = \
            np.mean(list(nx.average_degree_connectivity(largest_connected_component).values()))

    # Shortest path lengths in the largest connected component.
    with results('shortest_path_lengths_LCC', None, depends_on=['largest_connected_component']):
        shortest_path_lengths_LCC = \
            dict(nx.all_pairs_shortest_path_length(largest_connected_component))

    with results('eccentricities_LCC', None, depends_on=['shortest_path_lengths_LCC']):
        eccentricities = nx.eccentricity(largest_connected_component, sp=shortest_path_lengths_LCC)

    with results('diameter_LCC', True, depends_on=['eccentricities_LCC']):
        results.value = nx.diameter(None, eccentricities)

    with results('avg_geodesic_dist_LCC', False, depends_on=['shortest_path_lengths_LCC']):
        results.value = np.mean([
            length for source, lengths in shortest_path_lengths_LCC.items()
            for target, length in lengths.items() if source != target
        ])

    with results('avg_local_efficiency_LCC', False, depends_on=['largest_connected_component']):
        # results.value = nx.local_efficiency(largest_connected_component)
        results.value = local_efficiency(nklcc)

    if include_connectivity:
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
