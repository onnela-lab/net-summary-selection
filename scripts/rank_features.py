import argparse
from cost_based_selection import cost_based_methods, preprocessing_utils
import functools as ft
import logging
import os
import pickle
import time


logging.basicConfig(level='INFO')


METHODS = {key: getattr(cost_based_methods, key) for key in ['JMI', 'JMIM', 'mRMR', 'random']}
for key in ['pen_rf_importance', 'weighted_rf_importance']:
    for implementation in ['impurity', 'permutation']:
        METHODS[f'{key}_{implementation}'] = ft.partial(
            getattr(cost_based_methods, key), imp_type=implementation)

for proximity in ['rf prox', 'distance']:
    METHODS[f"reliefF_{proximity.replace(' ', '_')}"] = ft.partial(
        cost_based_methods.reliefF, proximity=proximity)


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--penalty', help='penalty factor for costly features', type=float,
                        default=0)
    parser.add_argument('method', help='method to use for ranking features', choices=METHODS)
    parser.add_argument('output', help='output path for the reference table')
    parser.add_argument('filenames', help='simulation files to load', nargs='+')
    args = parser.parse_args()

    logger = logging.getLogger('rank_features')

    data = preprocessing_utils.load_simulations(args.filenames)

    kwargs = {
        'X': data['X'],
        'y': data['y'],
        'is_disc': data['is_discrete'],
        'cost_vec': data['costs'],
        'cost_param': args.penalty,
    }

    # Run the analysis.
    result = {
        'args': vars(args),
        'start': time.time(),
        'features': data['features'],
    }
    ranking, *_ = METHODS[args.method](**kwargs)

    logger.info('ranked %d features', len(ranking))

    result['end'] = time.time()
    result['ranking'] = ranking

    # Store the results.
    directory = os.path.dirname(args.output)
    os.makedirs(directory, exist_ok=True)
    with open(args.output, 'wb') as fp:
        pickle.dump(result, fp)


if __name__ == '__main__':
    __main__()
