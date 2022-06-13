import argparse
from cost_based_selection import cost_based_methods, preprocessing_utils
import functools as ft
import logging
import numpy as np
import os
import pickle
import random
import time


# Set up callables for all the different methods.
METHODS = {key: getattr(cost_based_methods, key) for key
           in ['JMI', 'JMIM', 'mRMR', 'random_ranking']}
for key in ['pen_rf_importance', 'weighted_rf_importance']:
    for implementation in ['impurity', 'permutation']:
        METHODS[f'{key}_{implementation}'] = ft.partial(
            getattr(cost_based_methods, key), imp_type=implementation)

for proximity in ['rf prox', 'distance']:
    METHODS[f"reliefF_{proximity.replace(' ', '_')}"] = ft.partial(
        cost_based_methods.reliefF, proximity=proximity)


def __main__():
    # Set up basic logging.
    logging.basicConfig(level='INFO')
    logger = logging.getLogger('rank_features')

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', help='random number generator seed', type=int)
    parser.add_argument('--penalty', help='penalty factor for costly features', type=float,
                        default=0)
    parser.add_argument('--mi', help='precomputed mutual information')
    parser.add_argument('--adjusted', help='whether to adjust mutual information',
                        action='store_true')
    parser.add_argument('method', help='method to use for ranking features', choices=METHODS)
    parser.add_argument('output', help='output path for the reference table')
    parser.add_argument('filenames', help='simulation files to load', nargs='+')
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Load and prepare data for ranking.
    data = preprocessing_utils.load_simulations(args.filenames)
    kwargs = {
        'X': data['X'],
        'y': data['y'],
        'is_disc': data['is_discrete'],
        'cost_vec': data['costs'],
        'cost_param': args.penalty,
    }

    # Get the method and add precomputed mutual information.
    method = METHODS[args.method]
    if args.method in ['JMI', 'JMIM', 'mRMR']:
        with open(args.mi, 'rb') as fp:
            mutual_info = pickle.load(fp)
        assert mutual_info['args']['adjusted'] == args.adjusted, "use MI adjustment consistently"
        method = ft.partial(method, MI_matrix=mutual_info['marginal'],
                            MI_conditional=mutual_info['conditional'], adjusted=args.adjusted)
    else:
        assert not args.mi, f"cannot use mutual information with {args.method}"

    # Run the analysis.
    result = {
        'args': vars(args),
        'start': time.time(),
        'features': data['features'],
    }
    ranking, *_ = method(**kwargs)
    result['end'] = time.time()
    result['ranking'] = ranking

    logger.info('ranked %d features in %.1f seconds using %s', len(ranking),
                result['end'] - result['start'], args.method)

    # Store the results.
    directory = os.path.dirname(args.output)
    os.makedirs(directory, exist_ok=True)
    with open(args.output, 'wb') as fp:
        pickle.dump(result, fp)


if __name__ == '__main__':
    __main__()
