# -*- coding: utf-8 -*-
""" Implementation of cost-based feature selection/ranking algorithms.

Implementation of the cost-based version of the filter feature selection method
based on Maximal-Relevance-Minimal-Redundancy (mRMR), Joint Mutual Information
(JMI), Joint Mutual Information Maximization (JMIM), a version of
ReliefF that can compute nearest neighbors either with random forests, or with
an L1 distance. A cost-based ranking is also available by penalization of the
random forest feature importance, or by using the feature importance of
a random forest where the sampling of features at each internal node
is proportional to the inverse of their cost.
Moreover, to analyze the rankings for different penalization parameter values,
we also implement corresponding functions that return the different rankings
for each penalization value.

"""

import collections
import copy
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_mutual_info_score
from scipy import spatial
from tqdm import tqdm
import itertools as it
from .old.cost_based_methods import _private_proximity_matrix
from ._util import evaluate_proximity_matrix

# To use the R package ranger for RF importance computation
import rpy2.robjects


def random_ranking(X, y, is_disc, cost_vec=None, cost_param=0):
    """
    Return a random feature ranking.
    """
    # Select features sequentially proportional to their inverse selection probability.
    if cost_vec is None:
        proba = np.ones(X.shape[1])
    else:
        assert cost_vec.shape == (X.shape[1],)
        proba = cost_vec ** -cost_param

    # "Rank" the features by sequentially selecting them proportional to the given probability. We
    # first convert to lists because they are easier to deal with for sequential sampling.
    candidates = list(np.arange(proba.size))
    proba = list(proba)
    ranking = []
    while candidates:
        # We need to renormalize the probabilities each time.
        idx = np.random.choice(len(proba), p=np.asarray(proba) / np.sum(proba))
        ranking.append(candidates.pop(idx))
        proba.pop(idx)
    return ranking,


def evaluate_mi(x: np.ndarray, y: np.ndarray, x_discrete: bool, y_discrete: bool,
                adjusted: bool = False, n_neighbors: int = 3, random_state: int = 123):
    """
    Evaluate mutual information, possibly adjusting for chance agreement.
    """
    # Give up if either of the variables are discrete and have only one unique value or all unique
    # values.
    for z, disc in [(x, x_discrete), (y, y_discrete)]:
        if not disc:
            continue
        nunique = np.unique(z).size
        if nunique == 1 or nunique == z.size:
            return 0
    # Use the same method as sklearn.feature_selection.mutual_info_* if not adjusted.
    # if not adjusted:
    #     return _mutual_info._compute_mi(x, y, x_discrete, y_discrete, n_neighbors=n_neighbors)
    if not adjusted:
        if y_discrete:
            return mutual_info_classif(x[:, None], y, discrete_features=[x_discrete],
                                       random_state=random_state).squeeze()
        else:
            return mutual_info_regression(x[:, None], y, discrete_features=[x_discrete],
                                          random_state=random_state).squeeze()
    # Evaluate adjusted mutual info if desired and both features are discrete.
    if not (x_discrete and y_discrete):
        raise ValueError("can only adjust mutual information for discrete variables")
    return adjusted_mutual_info_score(x, y)


def evaluate_pairwise_mutual_information(
        X: np.ndarray, is_disc: np.ndarray, random_seed: int = 123, adjusted=True,
        progress: bool = False) -> np.ndarray:
    """
    Compute all pairwise mutual information scores.
    """
    _, num_features = X.shape
    matrix_MI = np.zeros((num_features, num_features), dtype=float)

    pairs = it.combinations_with_replacement(range(num_features), 2)
    npairs = num_features * (num_features + 1) // 2
    for i, j in tqdm(pairs, desc="pairwise", total=npairs) if progress else pairs:
        adjusted_ = adjusted and is_disc[i] and is_disc[j]
        score = evaluate_mi(X[:, i], X[:, j], is_disc[i], is_disc[j], adjusted_,
                            random_state=random_seed)
        matrix_MI[i, j] = matrix_MI[j, i] = score
    return matrix_MI


def evaluate_conditional_mutual_information(
        X: np.ndarray, is_disc: np.ndarray, y: np.ndarray, random_seed: int = 123, adjusted=True,
        progress: bool = False) -> np.ndarray:
    """
    Compute pairwise mutual information conditional on the class of `y`.
    """
    _, num_features = X.shape
    # Create a dictionary that will contains the corresponding MI matrices
    # conditionally on the different unique values of y
    MI_condY = dict()
    # For each modality of y
    for valY in np.unique(y):

        # Initialize a new matrix
        matTmp = np.zeros((num_features, num_features), dtype=float)
        # Extract the rows of X with this modality of Y
        subX = X[y == valY]

        # proportion of this modality
        proValY = np.mean(y == valY)

        pairs = it.combinations_with_replacement(range(num_features), 2)
        npairs = num_features * (num_features + 1) // 2
        for i, j in tqdm(pairs, desc=f"conditional: {valY}", total=npairs) if progress else pairs:
            adjusted_ = adjusted and is_disc[i] and is_disc[j]
            score = evaluate_mi(subX[:, i], subX[:, j], is_disc[i], is_disc[j], adjusted_,
                                random_state=random_seed)
            matTmp[i, j] = matTmp[j, i] = proValY * score

        MI_condY[valY] = matTmp
    return MI_condY


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


def JMI(X, y, is_disc, cost_vec=None, cost_param=0, num_features_to_select=None, random_seed=123,
        MI_matrix=None, MI_conditional=None, adjusted: bool = False):
    """
    Cost-based feature ranking based on Joint Mutual Information.

    Cost-based adaptation of the filter feature selection algorithm based on
    Joint Mutual Information (Yang and Moody (1999)).

    H. H. Yang and J. Moody. Feature selection based on joint mutual information.
    In Advances in intelligent data analysis, proceedings of international
    ICSC symposium, pages 22—-25, 1999.

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
    # initial_scores = mutual_info_classif(X, y, discrete_features=is_disc,
    #                                      random_state=random_seed)
    initial_scores = np.asarray([
        evaluate_mi(x, y, x_discrete, True, adjusted and x_discrete, random_state=random_seed) for
        x, x_discrete in zip(X.T, is_disc)
    ])

    # The cost based will substract lambda*cost for each item of initial_scores
    initial_scores_mcost = initial_scores - cost_param * cost_vec

    if MI_matrix is None:
        matrix_MI_Xk_Xj = evaluate_pairwise_mutual_information(X, is_disc, random_seed)
    else:
        matrix_MI_Xk_Xj = MI_matrix

    # For the Joint mutual information, we also need to compute the matrices
    # I(Xk, Xj | Y=y) for y in Y

    # Create a dictionary that will contains the corresponding MI matrices given the different
    # unique values of y.
    if MI_conditional is None:
        MI_condY = evaluate_conditional_mutual_information(X, is_disc, y, random_seed)
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


def JMIM(X, y, is_disc, cost_vec=None, cost_param=0, num_features_to_select=None, random_seed=123,
         MI_matrix=None, MI_conditional=None, adjusted: bool = False):
    """ Cost-based feature ranking based on Joint Mutual Information Maximization.

    Cost-based adaptation of the filter feature selection algorithm based on
    Joint Mutual Information Maximization (Bennasar et al. (2015)).

    M. Bennasar, Y. Hicks, and R. Setchi. Feature selection using Joint Mutual
    Information Maximisation. Expert Systems With Applications, 42:8520--8532,
    2015.

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

    for featIdx in range(num_features):
        if is_disc[featIdx] and len(np.unique(X[:, featIdx])) == X.shape[0]:
            is_disc[featIdx] = False

    # Computing all the MIs I(X_j; y)
    # initial_scores = mutual_info_classif(X, y, discrete_features=is_disc,
    #                                      random_state=random_seed)
    initial_scores = np.asarray([
        evaluate_mi(x, y, x_discrete, True, adjusted and x_discrete, random_state=random_seed) for
        x, x_discrete in zip(X.T, is_disc)
    ])
    initial_scores_mcost = initial_scores - cost_param*cost_vec

    if MI_matrix is None:
        matrix_MI_Xk_Xj = evaluate_pairwise_mutual_information(X, is_disc, random_seed)
    else:
        matrix_MI_Xk_Xj = MI_matrix

    # For the Joint mutual information, we also need to compute the matrices
    # I(Xk, Xj | Y=y) for y in Y

    # Create a dictionary that will contains the corresponding MI matrices given the different
    # unique values of y.
    if MI_conditional is None:
        MI_condY = evaluate_conditional_mutual_information(X, is_disc, y, random_seed)
    else:
        MI_condY = MI_conditional

    # ranking contains the indices of the final ranking in decreasing order of importance
    ranking = []

    # The first selected feature is the one with the maximal penalized I(X_j, Y) value
    selected = np.argmax(initial_scores_mcost)
    ranking.append(selected)
    unRanked.pop(selected)

    # Until we have the desired number of selected_features, we apply the selection criterion
    for k in range(1, num_features):

        featureRel = []
        # Compute the criterion to maximize for each unranked covariate
        for idx in unRanked:
            vecSummed = np.zeros(len(ranking))
            for valY in np.unique(y):
                vecSummed += MI_condY[valY][ranking, idx]

            criterionVal = np.min(initial_scores[ranking] - matrix_MI_Xk_Xj[ranking, idx]
                                  + vecSummed) + initial_scores_mcost[idx]
            # J(Xk) = min_j [ I(Xj;Y) - I(Xk;Xj) + I(Xk;Xj|Y) ] + (I(Xk;Y) - lambda * costk)

            featureRel.append(criterionVal)

        tmp_idx = np.argmax(featureRel)
        ranking.append(unRanked[tmp_idx])
        unRanked.pop(tmp_idx)

    return ranking, matrix_MI_Xk_Xj, MI_condY


def reliefF(X, y, cost_vec=None, cost_param=0, num_neighbors=10, num_features_to_select=None,
            proximity="distance", min_samples_leaf=100, n_estimators=500, sim_matrix=None,
            is_disc=None, debug=False, norm: str = "range"):
    """ Cost-based feature ranking adaptation of the ReliefF algorithm.

    Cost-based adaptation of the ReliefF algorithm, where the nearest neighbors
    of each data can be identified either using a classic L1 distance, or a
    random forest proximity matrix.

    I. Kononenko. Estimating attributes: Analysis and extensions of relief.
    In F. Bergadano and L. De Raedt, editors, Machine Learning: ECML-94,
    pages 171--182, Berlin, Heidelberg, 1994. Springer Berlin Heidelberg.

    Args:
        X (numpy.ndarray):
            the numerical features to use as training data, where
            each row represents an individual, and each column a feature.
        y (list):
            a list of integers representing the training data labels.
        cost_vec (numpy.ndarray):
            the vector of costs represented by a numpy.ndarray with shape
            (1, X.shape[1]). If None, the cost is set to zero for each feature.
        cost_param (float):
            the positive cost penalization parameter. 0 by default.
        num_neighbors (int):
            the number of nearest neighbors. 10 by default.
        proximity (str):
            a string that is either "distance" to use the classic version of
            reliefF, or "rf prox" to use the random forest proximity between
            data to deduce the neighbors. "distance" by default.
        min_samples_leaf (int):
            when using proximity = "rf prox", the minimum number of samples
            required to split an internal node. 100 by default.
        n_estimators (int):
            the number of trees in the random forest. Only relevant when
            proximity = "rf prox". 500 by default.
        sim_matrix (numpy.ndarray):
            the precomputed matrix of pairwise similarity between data,
            either distance or random forest proximity. This argument is
            returned to speed up the analysis when working with multiple
            cost_param values.
        norm: Whether to normalize by `range` or `standardize` the features before further
            processing.

    Returns:
        ranking (list):
            a list containing the indices of the ranked features as
            specified in X, in decreasing order of importance.
        weights (dict):
            a dictionary with as keys the covariate index, and as values the
            corresponding scores used to obtain the ranking.
        sim_matrix (numpy.ndarray):
            the pairwise distance/proximity matrix used.

    """

    y = np.array(y)
    nTrain = X.shape[0]
    nCov = X.shape[1]

    if proximity not in ['distance', 'rf prox']:
        raise ValueError(
            f"The argument proximity must be either 'distance' or 'rf prox', not '{proximity}'.")

    if (cost_vec is None):
        # If no cost is specified, then all costs are set as equal to zero
        cost_vec = np.zeros(nCov)

    assert num_features_to_select is None  # We'll just return the ranking without discarding info.

    # Data standardization
    X_std = copy.deepcopy(X)
    cov_means = np.mean(X, axis=0)
    cov_std = np.std(X, axis=0)
    for i in range(X.shape[1]):
        X_std[:, i] = (X[:, i] - cov_means[i])/cov_std[i]

    # Determine the number/proportion of classes in y
    classes = np.unique(y)
    nClasses = len(classes)
    pClasses = collections.Counter(y)
    nbrData = np.sum(list(pClasses.values()))
    for cLab in pClasses:
        pClasses[cLab] = pClasses[cLab]/nbrData

    # Compute for each covariate the max and min values. Useful for L1 dist.
    if norm == "range":
        maxXVal = np.max(X_std, axis=0)
        minXVal = np.min(X_std, axis=0)
        X_norm = X_std / (maxXVal - minXVal)
    elif norm == "standardize":
        X_norm = X_std
    else:
        raise ValueError(norm)

    # If we use the classic (Manhattan) distance:
    if proximity == "distance":
        if sim_matrix is None:
            distMat = spatial.distance.squareform(spatial.distance.pdist(X_norm, 'cityblock'))
        else:
            distMat = sim_matrix

    # If we use the RF proximity matrix instead of classic distance:
    if proximity == "rf prox":
        if sim_matrix is None:
            # Train a random forest and deduce the proximity matrix
            model = RandomForestClassifier(n_estimators=n_estimators,
                                           min_samples_leaf=min_samples_leaf)
            model.fit(X_std, y)
            proxMatRF = evaluate_proximity_matrix(model.apply(X_std))
            if debug:
                proxMatRF_old = _private_proximity_matrix(model, X_std, normalize=True)
                np.testing.assert_allclose(proxMatRF, proxMatRF_old)
            distMat = -proxMatRF
        else:
            distMat = -sim_matrix

    # For each training data R_i:
    # Search for k nearest hits
    # Search, for each class different than R_i's, the k nearest misses

    # To store the indices of the nearest hits
    kNearHits = np.zeros(num_neighbors, dtype=int)
    # To store the indices of the misses for all class different than R_i
    kNearMisses = np.zeros((nClasses-1, num_neighbors), dtype=int)

    # Initialize the weights to zero
    weightsDic = dict()
    for cov in range(nCov):
        weightsDic[cov] = 0

    m = nTrain  # Here we compute the score using all the training data
    for i in range(m):
        # For the same class that R_i, keep the indices achieving the k lower distances
        argSorted = np.argsort(distMat[i, y == y[i]])  # We withdraw the i-th element
        kNearHits = argSorted[argSorted != i][0:num_neighbors]
        classDifRi = classes[classes != y[i]]
        for c in range(len(classDifRi)):
            tmp = classDifRi[c]
            kNearMisses[c, :] = np.argsort(distMat[i, y == tmp])[0:num_neighbors]

        # Compute the elements diff(A, R_i, H_j) for j in 1:k, per feature A
        for cov in range(nCov):
            compDistRiFromHits = np.abs(X_norm[i, cov] - X_norm[kNearHits, cov])
            if debug:
                compDistRiFromHits_old = [
                    np.abs(X_std[i, cov] - X_std[hit, cov])/(maxXVal[cov] - minXVal[cov])
                    for hit in kNearHits
                ]
                np.testing.assert_allclose(compDistRiFromHits, compDistRiFromHits_old, atol=1e-9)
            weightsDic[cov] -= np.mean(compDistRiFromHits)/m

            # For each class different from the one of R_i, do the same with
            # weight by prior proba ratio
            for c in range(len(classDifRi)):
                compDistRiFromMisses = np.abs(X_norm[i, cov] - X_norm[kNearMisses[c], cov])
                if debug:
                    compDistRiFromMisses_old = [
                        np.abs(X_std[i, cov] - X_std[miss, cov])/(maxXVal[cov] - minXVal[cov])
                        for miss in kNearMisses[c]
                    ]
                    np.testing.assert_allclose(compDistRiFromMisses, compDistRiFromMisses_old,
                                               atol=1e-9)

                # Reminder: pClasses is a dictionary
                tmp = classDifRi[c]
                weightsDic[cov] += (pClasses[tmp] / (1-pClasses[y[i]])) \
                    * np.mean(compDistRiFromMisses) / m

            # Finally also update with the penalization (cost)
            # I do not use the /(m*k) term but only /m to be more consistent
            # with the other criteria of this module.
            weightsDic[cov] -= cost_param*cost_vec[cov]/(m)

    # Return the number of feature requested, in decreasing order, plus weights
    ranking = np.argsort(-np.array(list(weightsDic.values())))
    ranking = ranking.tolist()

    return ranking, weightsDic, distMat


def pen_rf_importance(X, y, cost_vec=None, cost_param=0, num_features_to_select=None,
                      imp_type="impurity", min_samples_leaf=1,
                      n_estimators=500, rf_importance_vec=None, is_disc=None):
    """ Cost-based feature ranking with penalized random forest importance.

    The cost-based ranking of the features are deduced by penalizing the
    random forest importance by the feature costs.

    Args:
        X (numpy.ndarray):
            the numerical features to use as training data, where
            each row represents an individual, and each column a feature.
        y (list):
            a list of integers representing the training data labels.
        cost_vec (numpy.ndarray):
            the vector of costs represented by a numpy.ndarray with shape
            (1, X.shape[1]). If None, the cost is set to zero for each feature.
        cost_param (float):
            the positive cost penalization parameter. 0 by default.
        imp_type (str):
            a string, either "impurity" or "permutation", to use the
            random forest importance based on the decrease of the impurity
            measure (MDI), or based on the decrease of accuracy due to random permutation
            of the covariate values (MDA). "impurity" by default.
        min_samples_leaf (int):
            the minimum number of samples required to split an internal node.
            1 by default.
        n_estimators (int):
            the number of trees in the random forest. 500 by default.
        rf_importance_vec (numpy.ndarray):
            an array that contains the precomputed unpenalized random forest
            importance. Useful when analyzing the rankings for different
            cost_parameter value, to reduce the computational time.

    Returns:
        ranking (list):
            a list containing the indices of the ranked features as
            specified in X, in decreasing order of importance.
        unpenalized_rf_importance (numpy.ndarray):
            an array that contains the computed UNPENALIZED random forest
            importance. This might be used to reduce the computational time
            when implementing a version with multiple cost_parameter values.

    """
    nCov = X.shape[1]

    # Coerce to integers if we've got strings.
    y = np.asarray(y)
    if y.dtype.kind in 'US':
        _, y = np.unique(y, return_inverse=True)

    if imp_type not in ['impurity', 'permutation']:
        raise ValueError("The argument imp_type must be either 'impurity' or 'permutation'.")

    if (cost_vec is None):
        # If no cost is specified, then all costs are set as equal to zero
        cost_vec = np.zeros(nCov)

    assert num_features_to_select is None  # We'll just return the ranking without discarding info.

    if(rf_importance_vec is None):
        # For format compatibility between python and R (rpy2)
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()

        rpy2.robjects.globalenv["X_train"] = X
        rpy2.robjects.globalenv["y_train"] = y
        rpy2.robjects.globalenv["imp_type"] = imp_type
        rpy2.robjects.globalenv["min_samples_leaf"] = min_samples_leaf
        rpy2.robjects.globalenv["n_estimators"] = n_estimators

        unpenalized_rf_importance = rpy2.robjects.r('''
        # Check if ranger is installed
        packages = c("ranger")
        package.check <- lapply(
                packages,
                FUN = function(x) {
                        if (!require(x, character.only = TRUE)) {
                                install.packages(x, dependencies = TRUE)
                                library(x, character.only = TRUE)
                        }
                      })
        # Determine the importance
        library(ranger)
        trainedRF <- ranger(x=as.data.frame(X_train), y = as.numeric(y_train),
                            classification = TRUE, importance = imp_type,
                            num.trees = n_estimators, min.node.size = min_samples_leaf,
                            num.threads = 1)
        trainedRF$variable.importance
        ''')

        numpy2ri.deactivate()

    else:
        unpenalized_rf_importance = copy.deepcopy(rf_importance_vec)

    rf_importance_copy = copy.deepcopy(unpenalized_rf_importance)

    # To facilitate the comparison between different types of importance,
    # we set values between 0 and 1, and to sum to 1.
    rf_importance_copy = (np.array(rf_importance_copy)-np.min(rf_importance_copy)) \
        / (np.max(rf_importance_copy) - np.min(rf_importance_copy))
    rf_importance_copy = rf_importance_copy/np.sum(rf_importance_copy)

    for cov in range(nCov):
        rf_importance_copy[cov] -= cost_param * cost_vec[cov]

    ranking = np.argsort(-rf_importance_copy)
    ranking = ranking.tolist()

    return ranking, unpenalized_rf_importance


def weighted_rf_importance(X, y: np.ndarray, cost_vec=None, cost_param=0,
                           num_features_to_select=None,
                           imp_type="impurity", min_samples_leaf=1,
                           n_estimators=500, is_disc=None):
    """ Cost-based feature ranking using weighted random forest importance.

    The cost-based ranking of the features are deduced using the feature
    importance of a weighted random forest, where the probability of sampling
    a covariate at a given node is proportional to 1/(cost)^cost_param.

    Args:
        X (numpy.ndarray):
            the numerical features to use as training data, where
            each row represents an individual, and each column a feature.
        y (list):
            a list of integers representing the training data labels.
        cost_vec (numpy.ndarray):
            the vector of costs represented by a numpy.ndarray with shape
            (1, X.shape[1]). If None, the cost is set to zero for each feature.
        cost_param (float):
            the positive cost penalization parameter. 0 by default.
        imp_type (str):
            a string, either "impurity" or "permutation", to use the
            random forest importance based on the decrease of the impurity
            measure (MDI), or based on the decrease of accuracy due to random permutation
            of the covariate values (MDA). "impurity" by default.
        min_samples_leaf (int):
            the minimum number of samples required to split an internal node.
            1 by default.
        n_estimators (int):
            the number of trees in the random forest. 500 by default.

    Returns:
        ranking (list):
            a list containing the indices of the ranked features as
            specified in X, in decreasing order of importance.
    """

    nCov = X.shape[1]

    # Coerce to integers if we've got strings.
    y = np.asarray(y)
    if y.dtype.kind in 'US':
        _, y = np.unique(y, return_inverse=True)

    if imp_type not in ['impurity', 'permutation']:
        raise ValueError("The argument imp_type must be either 'impurity' or 'permutation'.")

    if (cost_vec is None):
        # If no cost is specified, then all costs are set as equal to zero
        cost_vec = np.zeros(nCov)

    assert num_features_to_select is None  # We'll just return the ranking without discarding info.

    # Compute the rf weights for sampling the covariates
    # Note, a base importance of 0.01 is added to all features to avoid num. errors
    sampling_weights = (1/(cost_vec+0.01)**cost_param) / (np.sum(1/(cost_vec+0.01)**cost_param))

    # For format compatibility between python and R (rpy2)
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()

    rpy2.robjects.globalenv["X_train"] = X
    rpy2.robjects.globalenv["y_train"] = y
    rpy2.robjects.globalenv["imp_type"] = imp_type
    rpy2.robjects.globalenv["min_samples_leaf"] = min_samples_leaf
    rpy2.robjects.globalenv["n_estimators"] = n_estimators
    rpy2.robjects.globalenv["sampling_weights"] = sampling_weights

    weighted_rf_importance = rpy2.robjects.r('''
    # Check if ranger is installed
    packages = c("ranger")
    package.check <- lapply(
            packages,
            FUN = function(x) {
                    if (!require(x, character.only = TRUE)) {
                            install.packages(x, dependencies = TRUE)
                            library(x, character.only = TRUE)}
            }
           )
    # Determine the importance
    library(ranger)
    trainedWeightedRF <- ranger(x=as.data.frame(X_train), y = as.numeric(y_train),
                        classification = TRUE, importance = imp_type,
                        num.trees = n_estimators, min.node.size = min_samples_leaf,
                        num.threads = 1, split.select.weights = as.numeric(sampling_weights))
    trainedWeightedRF$variable.importance
    ''')

    numpy2ri.deactivate()

    ranking = np.argsort(-weighted_rf_importance)
    ranking = ranking.tolist()

    return (ranking,)
