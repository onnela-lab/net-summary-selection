import time
import numpy as np
import networkx as nx
from scipy import stats
import typing


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

    def __call__(self, key: str, is_discrete: bool, depends_on: typing[str] = None) \
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


def compute_summaries(graph: nx.Graph, return_full_results=False):
    """
    Compute summary statistics.

    Args:
        graph: Graph to compute summary statistics for.
        return_full_results: Return the `SummaryResults` object instead of collapsing into a
            dictionary.
    """
    assert nx.is_connected(graph), "the graph is not connected"
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
        shortest_paths = dict(nx.all_pairs_shortest_path(graph))
        shortest_path_lengths = {u: {v: len(p) - 1 for v, p in x.items()}
                                 for u, x in shortest_paths.items()}
        shortest_path_lengths_np = dod2array(shortest_path_lengths, graph.number_of_nodes(), np.inf)

    for length in [3, 4, 5, 6]:
        with results(f'num_shortest_{length}paths', True, depends_on=['shortest_paths']):
            results.value = np.sum(shortest_path_lengths_np == length) / 2

    with results('size_min_node_dom_set', True):
        results.value = len(nx.approximation.min_weighted_dominating_set(graph))

    with results('size_min_edge_dom_set', True):
        results.value = len(nx.approximation.min_edge_dominating_set(graph)) * 2

    with results('avg_global_efficiency', False, depends_on=['shortest_paths']):
        results.value = np.mean(1 / shortest_path_lengths_np[shortest_path_lengths_np > 0])

    with results('eccentricities', None, depends_on=['shortest_paths']):
        eccentricities = nx.eccentricity(graph, sp=shortest_path_lengths)

    with results('diameter', True, depends_on=['eccentricities']):
        results.value = nx.diameter(None, eccentricities)

    with results('avg_geodesic_dist', False, depends_on=['shortest_paths']):
        results.value = np.mean([
            length for source, lengths in shortest_path_lengths.items()
            for target, length in lengths.items() if source != target
        ])

    with results('avg_local_efficiency', False):
        results.value = nx.local_efficiency(graph)

    with results('node_connectivity', True):
        results.value = nx.node_connectivity(graph)

    with results('edge_connectivity_LCC', True):
        results.value = nx.edge_connectivity(graph)

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
