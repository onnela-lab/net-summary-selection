import numpy as np
from .util import evaluate_mi, evaluate_pairwise_mutual_information


def mRMR(X, y, is_disc, cost_vec=None, cost_param=0, num_features_to_select=None, random_seed=123,
         MI_matrix=None, MI_conditional=None, adjusted: bool = False):
    """
    Cost-based feature ranking with maximum relevance minimum redundancy.

    Cost-based adaptation of the filter feature selection algorithm Maximal-
    Relevance-Minimal-Redundancy (mRMR, Peng et al. (2005)).

    H. Peng, F. Long, and C. Ding.  Feature Selection Based on Mutual
    Information: Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy.
    IEEE Transactions on pattern analysis and machine intelligence,
    27:1226--1238, 2005.

    Args:
        X (numpy.ndarray):
            the numerical features to use as training data, where
            each row represents an individual, and each column a feature.
        y (list):
            a list of integers representing the training data labels.
        is_disc (list):
            a list of booleans indicating with True if the feature is discrete
            and False if continuous.
        cost_vec (numpy.ndarray):
            the vector of costs represented by a numpy.ndarray with shape
            (1, X.shape[1]). If None, the cost is set to zero for each feature.
        cost_param (float):
            the positive cost penalization parameter. 0 by default.
        random_seed (int):
            the random seed to use with the mutual_information function
            (when computing the Mutual Information (MI) involving one or more
            continuous features).
        MI_matrix (numpy.ndarray):
            the matrix of precomputed pairwise MI between pairs of features to
            save times when wanting to use multiple cost values.
            By default this matrix is computed in the function.

    Returns:
        ranking (list):
            list containing the indices of the ranked features as specified in
            X, in decreasing order of importance.
        matrix_MI (numpy.ndarray):
            the matrix of precomputed MI between pairs of features.

    """
    num_features = X.shape[1]

    if cost_vec is None:
        # If no cost is specified, then all costs are set as equal to zero
        cost_vec = np.zeros(num_features)

    assert num_features_to_select is None  # We'll just return the ranking without discarding info.

    # unRanked contains the feature indices unranked
    unRanked = list(range(num_features))

    # Computing all the MIs I(X_j; y)
    # initial_scores = mutual_info_classif(X, y, discrete_features=is_disc,
    #                                      random_state=random_seed)
    initial_scores = np.asarray([
        evaluate_mi(x, y, x_discrete, True, adjusted and x_discrete, random_state=random_seed) for
        x, x_discrete in zip(X.T, is_disc)
    ])
    # The cost based will substract lambda*cost for each item of initial_scores
    initial_scores_mcost = initial_scores - cost_param * cost_vec

    if MI_matrix is None:
        matrix_MI = evaluate_pairwise_mutual_information(X, is_disc, random_seed)
    else:
        matrix_MI = MI_matrix

    # ranking contains the indices of the final ranking in decreasing order of importance
    ranking = []

    # The first selected feature is the one with the maximal penalized I(X_j, Y) value
    selected = np.argmax(initial_scores_mcost)
    ranking.append(selected)
    unRanked.pop(selected)

    # Until we have the desired number of selected_features, we apply the selection criterion
    for _ in range(1, num_features):
        featureRel = []
        # Compute the criterion to maximize for each unranked covariate
        for idx in unRanked:
            featureRel.append(initial_scores_mcost[idx] - np.mean(matrix_MI[ranking, idx]))

        tmp_idx = np.argmax(featureRel)
        ranking.append(unRanked[tmp_idx])
        unRanked.pop(tmp_idx)

    return ranking, matrix_MI
