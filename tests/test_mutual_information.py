import numpy as np
import pytest
import sklearn
import sklearn.metrics
import sklearn.feature_selection


def test_mutual_info_classif_mutual_info_score():
    # Check equivalence of mutual_info_classif and mutual_info_score.
    x, y = np.random.randint(10, size=(2, 100))
    a = sklearn.metrics.mutual_info_score(x, y)
    b = sklearn.feature_selection.mutual_info_classif(x[:, None], y, discrete_features=[True])
    np.testing.assert_allclose(a, b)
    # Given that these are the same, we can adjust using `adjusted_mutual_info_score`.


def test_mutual_info_unadjusted():
    k = 3
    x, y = np.random.randint(1_000_000_000, size=(2, k))
    score = sklearn.metrics.mutual_info_score(x, y)
    # All classes are distinct so we expect the mutual information to be equal to the entropy.
    np.testing.assert_allclose(score, np.log(k))
    np.testing.assert_array_less(sklearn.metrics.adjusted_mutual_info_score(x, y), score)


@pytest.mark.xfail(reason="https://github.com/scikit-learn/scikit-learn/issues/23720")
def test_classif_regression_symmetry():
    np.random.seed(0)
    n = 100
    d = np.random.randint(10, size=n)
    c = np.random.normal(0, 1, size=n)

    # Use continuous feature to classify discrete target.
    score1 = sklearn.feature_selection.mutual_info_classif(
        c[:, None], d, discrete_features=[False], random_state=123)

    # Use discrete feature to regress continuous target.
    score2 = sklearn.feature_selection.mutual_info_regression(
        d[:, None], c, discrete_features=[True], random_state=123)

    np.testing.assert_allclose(score1, score2)
