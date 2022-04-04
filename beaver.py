import beaver_build as bb
import itertools as it
import pathlib

# Limit within-process concurrency because it breaks multiprocessing by setting global environment
# variables for all subprocess and shell commands.
bb.Subprocess.ENV.update({
    'NUMEXPR_NUM_THREADS': 1,
    'OPENBLAS_NUM_THREADS': 1,
})
ROOT = pathlib.Path('workspace')


# Generate reference tables in batches.
SIMULATION_ROOT = ROOT / 'simulations'
CONFIG_BY_SPLIT = {
    'train': (1000, 1),
    'test': (1000, 2),
    'small': (100, 3),
    'medium': (500, 4),
}
MODELS = ['ba', 'dmX']
BATCH_SIZE = 100
NUM_BATCHES = 50
SIMULATIONS = {}

script = bb.File('scripts/generate_reference_table.py')
for model in MODELS:
    for split, (num_nodes, seed) in CONFIG_BY_SPLIT.items():
        key = (model, split)
        with bb.group_artifacts(SIMULATION_ROOT, model, split):
            for batch in range(NUM_BATCHES):
                args = ['$!', '$<', f'--seed={seed}', model, num_nodes, BATCH_SIZE, '$@']
                SIMULATIONS.setdefault(key, []).extend(bb.Subprocess(f'{batch}.pkl', script, args))


# Generate rankings.
METHODS = [
    'JMI', 'JMIM', 'mRMR', 'reliefF_distance', 'reliefF_rf_prox', 'pen_rf_importance_impurity',
    'random_ranking', 'pen_rf_importance_permutation', 'weighted_rf_importance_impurity',
    'weighted_rf_importance_permutation'
]
RANKING_SPLITS = ['train', 'small', 'medium']
PENALTIES = [0, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2]
RANKING_ROOT = ROOT / 'rankings'

# Iterate over all combinations of models and splits for which to rank features.
mi_script = bb.File('scripts/evaluate_mutual_information.py')
rank_script = bb.File('scripts/rank_features.py')
for model, split in it.product(MODELS, RANKING_SPLITS):
    with bb.group_artifacts(RANKING_ROOT, model, split):
        # Get the simulations we're going to need.
        sims = SIMULATIONS[(model, split)]

        # Precompute the mutual information.
        args = ['$!', mi_script, '$@', *sims]
        mutual_info, = bb.Subprocess(['mutual_information.pkl'], sims, args)

        # Run all the methods, adding the precomputed mutual information to some methods.
        for method, penalty in it.product(METHODS, PENALTIES):
            with bb.group_artifacts(method):
                inputs = list(sims)
                args = ['$!', rank_script, f'--penalty={penalty}', method, '$@']
                if method in ['JMI', 'JMIM', 'mRMR']:
                    args.append(f'--mi={mutual_info}')
                    inputs.append(mutual_info)
                args.extend(sims)
                bb.Subprocess(f'{penalty}.pkl', inputs, args)
