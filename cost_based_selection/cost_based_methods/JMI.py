import numpy as np
from .util import evaluate_conditional_mutual_information, evaluate_mi, \
    evaluate_pairwise_mutual_information


def JMI(X, y, is_disc, cost_vec=None, cost_param=0, num_features_to_select=None, random_seed=123,
        MI_matrix=None, MI_conditional=None, adjusted: bool = False):
    """
    Cost-based feature ranking based on Joint Mutual Information.

    Cost-based adaptation of the filter feature selection algorithm based on
    Joint Mutual Information (Yang and Moody (1999)).

    H. H. Yang and J. Moody. Feature selection based on joint mutual information.
    In Advances in intelligent data analysis, proceedings of international
    ICSC symposium, pages 22--25, 1999.

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
        MI_conditional (dict):
            a dictionary that contains the precomputed numpy.ndarray of conditional
            pairwise MI between features, conditioned to the response values.
            Each key is a response modality, and each value is a conditional
            MI matrix between features I(X_i, X_j | y=key). Useful to save
            computational times when wanting to use multiple cost values, but
            by default it is computed in the function.

    Returns:
        ranking (list):
            list containing the indices of the ranked features as specified in
            X, in decreasing order of importance.
        matrix_MI_Xk_Xj (numpy.ndarray):
            the matrix of precomputed MI between pairs of features.
        MI_condY (dict):
            a dictionary that contains the precomputed numpy.ndarray of conditional
            pairwise MI between features, conditioned to the response values.
            Each key is a response modality, and each value is a conditional
            MI matrix between features I(X_i, X_j | y=key).
    """

    num_features = X.shape[1]

    if cost_vec is None:
        # If no cost is specified, then all costs are set as equal to zero
        cost_vec = np.zeros(num_features)

    assert num_features_to_select is None  # We'll just return the ranking without discarding info.

    # unRanked contains the feature indices unranked
    unRanked = list(range(num_features))

    # Computing all the MIs I(X_j; y)
    initial_scores = np.asarray([
        evaluate_mi(x, y, x_discrete, y_discrete=True, adjusted=adjusted and x_discrete,
                    random_state=random_seed) for x, x_discrete in zip(X.T, is_disc)
    ])

    # The cost based will substract lambda*cost for each item of initial_scores
    initial_scores_mcost = initial_scores - cost_param * cost_vec

    if MI_matrix is None:
        matrix_MI_Xk_Xj = evaluate_pairwise_mutual_information(X, is_disc, random_seed=random_seed,
                                                               adjusted=adjusted)
    else:
        matrix_MI_Xk_Xj = MI_matrix

    # For the Joint mutual information, we also need to compute the matrices
    # I(Xk, Xj | Y=y) for y in Y

    # Create a dictionary that will contains the corresponding MI matrices given the different
    # unique values of y.
    if MI_conditional is None:
        MI_condY = evaluate_conditional_mutual_information(X, is_disc, y, random_seed=random_seed,
                                                           adjusted=adjusted)
    else:
        MI_condY = MI_conditional

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
            vecSummed = np.zeros(len(ranking))
            for valY in np.unique(y):
                # Compute I(Xk; Xj | Y)
                vecSummed += MI_condY[valY][ranking, idx]

            criterionVal = initial_scores_mcost[idx] - np.mean(matrix_MI_Xk_Xj[ranking, idx]) \
                + np.mean(vecSummed)

            featureRel.append(criterionVal)

        tmp_idx = np.argmax(featureRel)
        ranking.append(unRanked[tmp_idx])
        unRanked.pop(tmp_idx)

    return ranking, matrix_MI_Xk_Xj, MI_condY
