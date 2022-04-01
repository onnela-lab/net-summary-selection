import argparse
from cost_based_selection import preprocessing_utils
from cost_based_selection import cost_based_methods
import logging
import pickle


def __main__():
    # Set up basic logging.
    logging.basicConfig(level='INFO')
    logger = logging.getLogger('rank_features')

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='output path for the reference table')
    parser.add_argument('filenames', help='simulation files to load', nargs='+')
    args = parser.parse_args()

    # Load and prepare data for ranking.
    data = preprocessing_utils.load_simulations(args.filenames)

    # Evaluate the marginal and conditional mutual information and save.
    result = {
        'args': vars(args),
        'marginal': cost_based_methods.evaluate_pairwise_mutual_information(
            data['X'], data['is_discrete']
        ),
        'conditional': cost_based_methods.evaluate_conditional_mutual_information(
            data['X'], data['is_discrete'], data['y'],
        )
    }
    with open(args.output, 'wb') as fp:
        pickle.dump(result, fp)
    logger.info('saved results to %s', args.output)


if __name__ == '__main__':
    __main__()
