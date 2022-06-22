import collections
import numpy as np
from scipy import spatial
from sklearn.ensemble import RandomForestClassifier
from ..old.cost_based_methods import _private_proximity_matrix
from ._util import evaluate_proximity_matrix


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

    y = np.asarray(y)
    nTrain, nCov = X.shape

    if proximity not in ['distance', 'rf prox']:
        raise ValueError(
            f"The argument proximity must be either 'distance' or 'rf prox', not '{proximity}'.")

    if (cost_vec is None):
        # If no cost is specified, then all costs are set as equal to zero
        cost_vec = np.zeros(nCov)

    assert num_features_to_select is None  # We'll just return the ranking without discarding info.

    # Data standardization
    X_std = X.copy()
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
    elif proximity == "rf prox":
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
    else:
        raise ValueError(proximity)

    # For each training data R_i:
    # Search for k nearest hits
    # Search, for each class different than R_i's, the k nearest misses

    # To store the indices of the nearest hits
    kNearHits = np.zeros(num_neighbors, dtype=int)
    # To store the indices of the misses for all class different than R_i
    kNearMisses = np.zeros((nClasses - 1, num_neighbors), dtype=int)

    # Initialize the weights to zero
    weightsDic = {i: 0 for i in range(nCov)}

    m = nTrain  # Here we compute the score using all the training data
    for i in range(m):
        # For the same class that R_i, keep the indices achieving the k lower distances
        argSorted = np.argsort(distMat[i, y == y[i]])  # We withdraw the i-th element
        kNearHits = argSorted[argSorted != i][:num_neighbors]
        classDifRi = classes[classes != y[i]]
        for c in range(len(classDifRi)):
            tmp = classDifRi[c]
            kNearMisses[c, :] = np.argsort(distMat[i, y == tmp])[:num_neighbors]

        # Compute the elements diff(A, R_i, H_j) for j in 1:k, per feature A
        for cov in range(nCov):
            compDistRiFromHits = np.abs(X_norm[i, cov] - X_norm[kNearHits, cov])
            if debug:
                compDistRiFromHits_old = [
                    np.abs(X_std[i, cov] - X_std[hit, cov])/(maxXVal[cov] - minXVal[cov])
                    for hit in kNearHits
                ]
                np.testing.assert_allclose(compDistRiFromHits, compDistRiFromHits_old, atol=1e-9)
            weightsDic[cov] -= np.mean(compDistRiFromHits) / m

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
                weightsDic[cov] += (pClasses[tmp] / (1 - pClasses[y[i]])) \
                    * np.mean(compDistRiFromMisses) / m

            # Finally also update with the penalization (cost)
            # I do not use the /(m*k) term but only /m to be more consistent
            # with the other criteria of this module.
            weightsDic[cov] -= cost_param*cost_vec[cov] / m

    # Return the number of feature requested, in decreasing order, plus weights
    ranking = np.argsort(-np.array(list(weightsDic.values())))
    ranking = ranking.tolist()

    return ranking, weightsDic, distMat
