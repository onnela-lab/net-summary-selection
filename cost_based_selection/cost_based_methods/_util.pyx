# cython: boundscheck = False
import numpy as np


def evaluate_proximity_matrix(long[:, :] leaves):
    cdef:
        double[:, :] proximity
        long num_items = leaves.shape[0]
        long num_trees = leaves.shape[1]
        double value = 1.0 / num_trees
        long i, j, k

    # Initialize the proximity matrix.
    proximity = np.zeros((num_items, num_items))
    # Count the number of times each pair of items belongs to the same leaf. We only consider one
    # off-diagonal for efficiency.
    for i in range(num_trees):
        for j in range(num_items):
            for k in range(j + 1, num_items):
                if leaves[j, i] == leaves[k, i]:
                    proximity[j, k] += value

    # Fill the diagonal with 0.5 so it adds to 1.0 after transpositions and addition.
    for j in range(num_items):
        proximity[j, j] = 0.5

    proximity_array = np.asarray(proximity)
    return proximity_array + proximity_array.T
