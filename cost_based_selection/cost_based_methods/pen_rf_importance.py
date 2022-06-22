import copy
import numpy as np
import rpy2


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
