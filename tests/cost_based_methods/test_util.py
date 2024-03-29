from cost_based_selection.old import cost_based_methods as cost_based_methods_old
from cost_based_selection.cost_based_methods import util
import numpy as np
import pytest
from scipy import spatial, stats


@pytest.mark.parametrize('adjusted', [False, True], ids=["unadjusted", "adjusted"])
@pytest.mark.parametrize('first_discrete', [False, True],
                         ids=["first_continuous", "first_discrete"])
@pytest.mark.parametrize('marginal', [True, False], ids=["marginal", "conditional"])
def test_marginal_conditional_mi_regression(adjusted: bool, first_discrete: bool, marginal: bool):
    np.random.seed(3)
    n = 100
    is_disc = np.asarray([first_discrete, False, True])
    y = np.random.randint(3, size=n)
    X = np.transpose([np.random.randint(3, size=n) if d else np.random.normal(0, 1, size=n) for d
                      in is_disc])

    _, marginal_old, conditional_old = cost_based_methods_old.JMI(X, y, is_disc)

    # We can only test exactly if we are dealing with discrete-discrete features without adjustment.
    # Let's build a mask. For the rest, let's just hope they're correlated.
    offdiag = ~np.eye(is_disc.size).astype(bool)
    mask = is_disc[:, None] & is_disc & (not adjusted)

    items = []

    if marginal:
        marginal_new = util.evaluate_pairwise_mutual_information(
            X, is_disc, adjusted=adjusted)
        items.append(("marginal", marginal_new, marginal_old))
    else:
        conditional_new = util.evaluate_conditional_mutual_information(
            X, is_disc, y, adjusted=adjusted)
        assert set(conditional_old) == set(conditional_new)
        items.extend((key, value, conditional_old[key]) for key, value in conditional_new.items())

    for key, new, old in items:
        np.testing.assert_allclose(old[mask & offdiag], new[mask & offdiag], rtol=1e-6)
        a = old[~mask & offdiag]
        b = new[~mask & offdiag]
        if not np.allclose(a, b):
            corr, _ = stats.pearsonr(a, b)
            assert corr > 0.5, f"MI not correlated for {key}"


@pytest.mark.parametrize("p", [1, 2])
def test_nearest_neighbor(p):
    X = np.random.normal(0, 1, (100, 5))
    distance = spatial.distance.cdist(X, X, p=p, metric="minkowski")
    nn_minkowski = util.NearestNeighbors(X, p)
    nn_precomputed = util.NearestNeighbors(X, distance)
    # Compare nearest neighbors for a few samples.
    for i in np.random.randint(X.shape[0], size=10):
        distance_minkowski, neighbors_minkowski = nn_minkowski.query(X[i], 5)
        distance_precomputed, neighbors_precomputed = nn_precomputed.query(i, 5)
        np.testing.assert_allclose(distance_minkowski, distance_precomputed)
        np.testing.assert_allclose(neighbors_minkowski, neighbors_precomputed)
        assert distance_minkowski[0] == 0
        assert distance_precomputed[0] == 0
        assert neighbors_minkowski[0] == i
        assert neighbors_precomputed[0] == i
