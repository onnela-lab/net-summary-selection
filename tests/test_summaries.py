from cost_based_selection.summaries import compute_summaries as compute_summaries_v1
from cost_based_selection.summaries2 import compute_summaries as compute_summaries_v2
import networkx as nx
import numpy as np
import warnings


def test_equivalence_v1_v2():
    graph1 = nx.barabasi_albert_graph(100, 3)
    graph2 = nx.erdos_renyi_graph(100, .1)
    graph2 = nx.relabel_nodes(graph2, {u: u + 99 for u in graph2})
    graph = nx.compose(graph1, graph2)

    values1, times1, discrete1 = compute_summaries_v1(graph)
    values2, times2, discrete2 = compute_summaries_v2(graph)

    unique_values = {}
    for key, value in values1.items():
        unique_values.setdefault(value, []).append(key)
    for value, keys in unique_values.items():
        if len(keys) > 1:
            warnings.warn(f"{keys} have the same value {value}")

    common = set(values1) & set(values2)
    for key in common:
        # We cannot compare noise directly. v1 used a heuristic, we use the exact clique size in v2.
        if key.startswith('noise') or key == 'max_clique_size':
            continue
        try:
            np.testing.assert_allclose(values1[key], values2[key])
        except AssertionError as ex:
            raise AssertionError(key) from ex
        assert discrete1[key] == discrete2[key], key

    missing = set(values1) - set(values2) - {
        # We already enumerate all the cliques so using an approximation is not necessary.
        'large_clique_size',
        # There is an implementation error so this is just a redundant feature. Even if there
        # wasn't, it's just an inverse transform.
        'harmonic_mean',
        # Dropping because it's already implemented as avg_geodesic_dist_LCC.
        'avg_shortest_path_length_LCC',
        # Wiener index is just a linear rescaling of the average path length.
        'wiener_index_LCC',
        # We're dropping the number of components because we assume the graph is connected.
        'num_of_CC',
    }
    extra = set(values2) - set(values1)

    # Numerically verify most of the claims above.
    np.testing.assert_allclose(values1['avg_global_efficiency'], values1['harmonic_mean'])
    np.testing.assert_allclose(values1['avg_geodesic_dist_LCC'],
                               values1['avg_shortest_path_length_LCC'])
    np.testing.assert_allclose(values1['wiener_index_LCC'], values1['avg_geodesic_dist_LCC'] *
                               values1['num_nodes_LCC'] * (values1['num_nodes_LCC'] - 1) / 2)

    if missing or extra:
        raise AssertionError(f"{len(missing)} missing statistics ({', '.join(sorted(missing))}) "
                             f"and {len(extra)} extra statistics ({', '.join(sorted(extra))})")
