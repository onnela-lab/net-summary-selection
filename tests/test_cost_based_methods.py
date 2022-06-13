from cost_based_selection import cost_based_methods
from cost_based_selection.old import cost_based_methods as cost_based_methods_old
from cost_based_selection import data_generation
from cost_based_selection import preprocessing_utils
import functools as ft
import numpy as np
import pytest
from scipy import special, stats
import typing


@pytest.fixture(scope='session')
def network_data_dict():
    num_sims_per_model = 10
    num_nodes = 20

    # Generate data.
    model_indices, summaries, is_discrete, times = \
        data_generation.BA_ref_table(num_sims_per_model, num_nodes)

    # Preprocess the data.
    model_indices, summaries, is_discrete, times = \
        preprocessing_utils.drop_redundant_features(model_indices, summaries, is_discrete, times)
    average_costs = preprocessing_utils.compute_avg_cost(times)

    # Train test split for validation.
    X = np.array(summaries)
    y = model_indices.modIndex.tolist()
    is_discrete = is_discrete.iloc[0, :].tolist()
    return {
        'X': X,
        'y': y,
        'is_disc': is_discrete,
        'cost_vec': average_costs,
    }


@pytest.fixture(params=[
    'JMI', 'JMIM', 'mRMR', 'reliefF-distance', 'reliefF-rf prox', 'pen_rf_importance-impurity',
    'pen_rf_importance-permutation', 'weighted_rf_importance-impurity',
    'weighted_rf_importance-permutation', 'random',
])
def method_name(request):
    return request.param


@pytest.fixture
def methods(method_name: str):
    results = []
    for module in [cost_based_methods, cost_based_methods_old]:
        name, *args = method_name.split('-')
        try:
            func = getattr(module, name)
            if func is module.reliefF:
                func = ft.partial(func, proximity=args[0])
                if module is cost_based_methods:
                    func = ft.partial(func, debug=True)
            elif func in (module.pen_rf_importance, module.weighted_rf_importance):
                func = ft.partial(func, imp_type=args[0])
        except AttributeError:
            func = None
        results.append(func)
    return results


@pytest.fixture(params=[0., 1e-3])
def penalty(request):
    return request.param


@pytest.fixture
def synthetic_data_dict():
    # We generate data for a simple two-options model choice problem with fixed seed. This allows us
    # to both sanity check methods (they should rank informative features highly) and to pick up any
    # regressions.
    np.random.seed(0)
    X = np.random.normal(0, 1, (50, 5))
    coefs = np.arange(X.shape[1])
    logits = X @ coefs
    y: np.ndarray = np.random.uniform(0, 1, X.shape[0]) < special.expit(logits)
    cost_vec = np.random.uniform(0, 1, X.shape[1])
    is_disc = np.zeros(X.shape[1], bool)
    discrete_indices = [2]
    for idx in discrete_indices:
        X[:, idx] = X[:, idx].astype(int)
        is_disc[idx] = True
    return {
        'X': X,
        'y': y.astype(int),
        'cost_vec': cost_vec,
        'is_disc': is_disc,
    }


EXPECTED_RANKINGS = {
    'JMI-0.0': [4, 3, 2, 1, 0],
    'JMI-0.001': [4, 3, 1, 2, 0],
    'JMIM-0.0': [4, 3, 2, 0, 1],
    'JMIM-0.001': [4, 3, 2, 0, 1],
    'mRMR-0.0': [4, 3, 1, 2, 0],
    'mRMR-0.001': [4, 3, 1, 2, 0],
    'reliefF-distance-0.0': [2, 1, 0, 4, 3],
    'reliefF-distance-0.001': [1, 2, 0, 4, 3],
    'reliefF-rf prox-0.0': [4, 2, 1, 0, 3],
    'reliefF-rf prox-0.001': [4, 2, 1, 0, 3],
}


def is_deterministic(name: str) -> bool:
    return not any(x in name for x in ['pen_rf_importance', 'weighted_rf_importance', 'random'])


def test_method(method_name: str, network_data_dict: dict,
                methods: typing.Iterable[typing.Callable], penalty: float):
    deterministic = is_deterministic(method_name)
    rankings = []
    for method in methods:
        if method is None:
            continue
        # Evaluate the ranking for this method and add it to the list for comparison.
        ranking, *_ = method(**network_data_dict, cost_param=penalty)
        np.testing.assert_array_equal(np.unique(ranking), np.arange(len(ranking)))
        rankings.append(ranking)
        # Run again to check for determinism.
        if deterministic:
            ranking2, *_ = method(**network_data_dict, cost_param=penalty)
            np.testing.assert_array_equal(ranking, ranking2)
    # Mutual information methods have changed substantially because we adjust for chance.
    if method_name in {'JMI'}:
        return
    # These methods have changed ever so slightly due to different distance evaluations. But the
    # rankings remain highly correlated.
    if method_name.startswith('reliefF'):
        corr, pval = stats.spearmanr(*rankings)
        assert corr > 0.5
        assert pval < 1e-3
    elif deterministic:
        # Can only compare old and new implementation for deterministic tests.
        np.testing.assert_array_equal(*rankings)


def test_regression(method_name: str, methods: typing.Iterable[typing.Callable],
                    penalty: float, synthetic_data_dict: dict):
    assert synthetic_data_dict['is_disc'].sum() > 0

    for method in methods:
        if method is None:
            continue
        ranking, *_ = method(**synthetic_data_dict, cost_param=penalty)

        key = f'{method_name}-{penalty}'

        if not is_deterministic(method_name):
            message = f"{key} is not deterministic"
            assert key not in EXPECTED_RANKINGS, message
            pytest.skip(message)

        try:
            np.testing.assert_array_equal(ranking, EXPECTED_RANKINGS[key])
        except KeyError:
            raise KeyError(f"'{key}': {list(ranking)},")
        except AssertionError as ex:
            raise AssertionError(f"{key}\n{ex}") from ex


@pytest.mark.parametrize('adjusted', [False, True])
@pytest.mark.parametrize('first_discrete', [False, True])
def test_marginal_conditional_mi_regression(adjusted: bool, first_discrete: bool):
    n = 100
    is_disc = np.asarray([first_discrete, False, True])
    y = np.random.randint(3, size=n)
    X = np.transpose([np.random.randint(3, size=n) if d else np.random.normal(0, 1, size=n) for d
                      in is_disc])

    _, marginal_old, conditional_old = cost_based_methods_old.JMI(X, y, is_disc)

    marginal_new = cost_based_methods.evaluate_pairwise_mutual_information(
        X, is_disc, adjusted=adjusted)
    conditional_new = cost_based_methods.evaluate_conditional_mutual_information(
        X, is_disc, y, adjusted=adjusted)

    # We can only test exactly for (a) the first column because added random noise may differ or (b)
    # if we are dealing with discrete-discrete features without adjustment. Let's build a mask.
    # For the rest, let's just hope they're correlated.
    mask = (is_disc[:, None] & is_disc & (not adjusted)) | (np.arange(is_disc.size) == 0)
    np.testing.assert_allclose(marginal_old[mask], marginal_new[mask], rtol=1e-6)
    assert set(conditional_old) == set(conditional_new)
    for key, old in conditional_old.items():
        np.testing.assert_allclose(old[mask], conditional_new[key][mask], rtol=1e-6)
        corr, _ = stats.pearsonr(old[~mask], conditional_new[key][~mask])
        assert corr > 0.5

    # For the rest, let's just hope they're correlated.
