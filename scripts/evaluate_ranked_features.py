import argparse
from cost_based_selection import preprocessing_utils
import glob
import numpy as np
import os
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_rankings(directory):
    """
    Load rankings from a directory.

    Returns:
        penalties: q-length vector of penalty values.
        features: p-length vector of feature names.
        rankings: q by p matrix of ranked feature indices.
        durations: q-length vector of the time it took to rank features.
    """
    filenames = glob.glob(os.path.join(directory, '*.pkl'))

    penalties = []
    rankings = []
    durations = []
    features = None
    for filename in filenames:
        with open(filename, 'rb') as fp:
            result = pickle.load(fp)
        rankings.append(result['ranking'])
        penalties.append(result['args']['penalty'])
        if features is None:
            features = result['features']
        np.testing.assert_array_equal(features, result['features'])
        durations.append(result['end'] - result['start'])

    rankings = np.asarray(rankings)
    penalties = np.asarray(penalties)
    durations = np.asarray(durations)

    # Sort by penalty values in case the filenames are in a weird order.
    idx = np.argsort(penalties)
    penalties = penalties[idx]
    rankings = rankings[idx]
    durations = durations[idx]

    return {
        'penalties': penalties,
        'rankings': rankings,
        'features': features,
        'durations': durations,
    }


def evaluate_test_statistics(data, rankings, max_features, model_cls, folds):
    num_feature_range = np.arange(max_features) + 1
    accuracies = []
    costs = []

    for ranking in rankings:
        accuracy_row = []
        for num_features in num_feature_range:
            # Evaluate the cross-validated scores.
            pipeline = make_pipeline(StandardScaler(), model_cls())
            ranked_X = data['X'][:, ranking][:, :num_features]
            scores = cross_val_score(pipeline, ranked_X, data['y'], cv=folds)
            accuracy_row.append(scores)
        accuracies.append(accuracy_row)
        costs.append(data['costs'][ranking[:max_features]])

    cumulative_costs = np.cumsum(costs, axis=1)
    return {
        'accuracies': np.asarray(accuracies),
        'cumulative_costs': cumulative_costs,
        'normalized_cumulative_costs': cumulative_costs / data['costs'].sum()
    }


def __main__(args: list[str] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='sklearn model to use for classification', default='SVC')
    parser.add_argument('--folds', help='number of cross validation folds', type=int, default=10)
    parser.add_argument('ranking_path', help='directory containing rankings')
    parser.add_argument('test_data_path', help='directory containing test data')
    parser.add_argument('output', help='output path')
    args = parser.parse_args(args)

    if args.model == 'SVC':
        model_cls = SVC
    else:
        raise NotImplementedError(args.model)

    # Load and collate the rankings.
    rankings = load_rankings(args.ranking_path)
    data = preprocessing_utils.load_simulations(os.path.join(args.test_data_path, '*.pkl'))
    # Evaluate the statistics.
    result = rankings | evaluate_test_statistics(data, rankings['rankings'], 15, model_cls,
                                                 args.folds)
    # Save the results.
    with open(args.output, 'wb') as fp:
        pickle.dump(result, fp)


if __name__ == '__main__':
    __main__()
