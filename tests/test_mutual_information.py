import numpy as np
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


def test_classif_regression_symmetry():
    n = 100
    x = np.random.randint(10, size=n)
    y = np.random.normal(0, 1, size=n)

    score1 = sklearn.feature_selection.mutual_info_classif(y[:, None], x, discrete_features=[False])
    score2 = sklearn.feature_selection.mutual_info_regression(x[:, None], y,
                                                              discrete_features=[True])
    np.testing.assert_allclose(score1, score2)
