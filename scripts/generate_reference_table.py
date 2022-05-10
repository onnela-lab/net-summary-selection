import argparse
import contextlib
import logging
import networkx as nx
import numpy as np
import os
import pickle
import random
import time
from tqdm import tqdm
from cost_based_selection import data_generation
from cost_based_selection.summaries import compute_summaries


logging.basicConfig(level='INFO')


@contextlib.contextmanager
def log_time(logger: logging.Logger, message: str, **kwargs):
    start = time.time()
    yield
    kwargs['time'] = time.time() - start
    logger.info(message, kwargs)


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', help='random number generator seed', type=int)
    parser.add_argument('--show_progress', '-p', help='show a progress bar', action='store_true')
    parser.add_argument('model', help='model to simulate from', choices=['ba', 'dmX'])
    parser.add_argument('num_nodes', type=int, help='number of nodes in the graph')
    parser.add_argument('num_samples', type=int, help='number of samples to generate')
    parser.add_argument('output', help='output path for the reference table')
    args = parser.parse_args()

    logger = logging.getLogger('generate_reference_table')

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    result = {
        'args': vars(args),
        'start': time.time(),
    }

    steps = range(args.num_samples)
    if args.show_progress:
        steps = tqdm(steps)
    for step in steps:
        # Generate a graph.
        with log_time(logger, 'generated %(model)s graph in %(time).3fs', model=args.model):
            if args.model == 'ba':
                model_label = 1 + np.random.randint(4)
                graph = nx.barabasi_albert_graph(args.num_nodes, model_label)
            elif args.model == 'dmX':
                seed_network = nx.Graph()
                seed_network.add_edge(0, 1)
                qs = np.random.uniform(0.25, 0.75, 2)
                model_label = np.random.choice(['dmc', 'dmr'])
                if model_label == 'dmc':
                    graph = data_generation.DMC(seed_network, args.num_nodes, *qs)
                elif model_label == 'dmr':
                    graph = data_generation.DMR(seed_network, args.num_nodes, *qs)
                else:
                    raise ValueError(model_label)
            else:
                raise ValueError(args.model)
            result.setdefault('model_labels', []).append(model_label)

        # Evaluate the summary statistics.
        with log_time(logger, 'computed summaries for %(model)s graph in %(time).3fs',
                      model=model_label):
            result.setdefault('summaries', []).append(compute_summaries(graph, True).results)
        logger.info('computed summary statistics for %d of %d %s graphs (%.1f%%)', step + 1,
                    args.num_samples, args.model, 100 * (step + 1) / args.num_samples)

    result['end'] = time.time()
    result['duration'] = result['end'] - result['start']
    logger.info('computed summaries for %d samples in %.3fs', args.num_samples, result['duration'])
    directory = os.path.dirname(args.output)
    os.makedirs(directory, exist_ok=True)
    with open(args.output, 'wb') as fp:
        pickle.dump(result, fp)


if __name__ == '__main__':
    __main__()
