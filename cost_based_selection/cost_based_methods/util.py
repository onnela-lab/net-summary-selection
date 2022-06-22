import itertools as it
import numbers
import numpy as np
from scipy.spatial import KDTree
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import adjusted_mutual_info_score
from tqdm import tqdm
import typing


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
    x = np.asarray(x)
    y = np.asarray(y)
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
    matrix_MI = np.nan * np.empty((num_features, num_features), dtype=float)

    pairs = it.combinations(range(num_features), 2)
    npairs = num_features * (num_features - 1) // 2
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
        matTmp = np.nan * np.empty((num_features, num_features), dtype=float)
        # Extract the rows of X with this modality of Y
        subX = X[y == valY]

        # proportion of this modality
        proValY = np.mean(y == valY)

        pairs = it.combinations(range(num_features), 2)
        npairs = num_features * (num_features - 1) // 2
        for i, j in tqdm(pairs, desc=f"conditional: {valY}", total=npairs) if progress else pairs:
            adjusted_ = adjusted and is_disc[i] and is_disc[j]
            score = evaluate_mi(subX[:, i], subX[:, j], is_disc[i], is_disc[j], adjusted=adjusted_,
                                random_state=random_seed)
            matTmp[i, j] = matTmp[j, i] = proValY * score

        MI_condY[valY] = matTmp
    return MI_condY


class NearestNeighbors:
    """
    Nearest neighbor search, using k-d trees for Minkowski metrics and brute force for precomputed
    distances.

    Args:
        X: Coordinates of points with shape `(num_reference, num_features)`, where `num_reference`
            is the number of elements that can be queried.
        distance: L-p norm if a float; precomputed distance if an array. The precomputed array
            has shape `(num_reference, num_query)` and need not be square.
    """
    def __init__(self, X: np.ndarray, distance: float):
        self.X = X
        self.num_reference, self.num_features = X.shape
        self.distance = distance

        if precomputed := isinstance(distance, np.ndarray):
            num_reference, self.num_query = distance.shape
            assert num_reference == self.num_reference
            self.tree = None
        else:
            self.num_query = None
            self.tree = KDTree(self.X)
        self.precomputed = precomputed

    def query(self, x: typing.Union[np.ndarray, int], num_neighbors: int,
              return_distance: bool = True):
        """
        Query for nearest neighbors.

        Args:
            x: Vector of coordinates if `distance` is a float; index in [0, num_query) if
                precomputed.

        Returns:
            neighbors:
        """
        if self.precomputed:
            assert isinstance(x, numbers.Integral) and 0 <= x < self.num_query
            distance = self.distance[:, x]
            neighbors = np.argsort(distance)[:num_neighbors]
            distance = distance[neighbors]
        else:
            distance, neighbors = self.tree.query(x, k=num_neighbors, p=self.distance)

        if return_distance:
            return distance, neighbors
        return neighbors
