import numpy as np
import rpy2.robjects
from scipy.special import softmax


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
    log_prob = - cost_param * np.log(np.maximum(cost_vec, 1e-9))
    sampling_weights = np.maximum(softmax(log_prob), 1e-9)
    sampling_weights /= sampling_weights.sum()
    assert np.all(np.isfinite(sampling_weights))

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
