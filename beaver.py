import beaver_build as bb
import itertools as it
import pathlib

# Limit within-process concurrency because it breaks multiprocessing by setting global environment
# variables for all subprocess and shell commands.
bb.Subprocess.ENV.update({
    'NUMEXPR_NUM_THREADS': 1,
    'OPENBLAS_NUM_THREADS': 1,
    'OMP_NUM_THREADS': 1,
})
ROOT = pathlib.Path('workspace')


# Generate reference tables in batches.
SIMULATION_ROOT = ROOT / 'simulations'
CONFIG_BY_SPLIT = {
    'train': (1000, 1000),
    'test': (1000, 2000),
    'small': (100, 3000),
    'medium': (500, 4000),
}
MODELS = ['ba', 'dmX']
BATCH_SIZE = 100
NUM_BATCHES = 50
SIMULATIONS = {}

script = bb.File('scripts/generate_reference_table.py')
for model in MODELS:
    for split, (num_nodes, seed_prefix) in CONFIG_BY_SPLIT.items():
        key = (model, split)
        with bb.group_artifacts(SIMULATION_ROOT, model, split):
            for batch in range(NUM_BATCHES):
                seed = seed_prefix + batch
                args = ['$!', '$<', f'--seed={seed}', model, num_nodes, BATCH_SIZE, '$@']
                SIMULATIONS.setdefault(key, []).extend(bb.Subprocess(f'{batch}.pkl', script, args))


# Generate rankings.
METHODS = [
    'JMI', 'JMIM', 'mRMR', 'reliefF_distance', 'reliefF_rf_prox', 'random_ranking',
    'pen_rf_importance_impurity', 'pen_rf_importance_permutation',
    'weighted_rf_importance_impurity', 'weighted_rf_importance_permutation',
]
RANKING_SPLITS = ['train', 'small', 'medium']
PENALTIES = [0, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 8, 9, 10, 11, 12, 12.8, 25.6, 51.2]
RANKING_ROOT = ROOT / 'rankings'

# Iterate over all combinations of models and splits for which to rank features.
mi_script = bb.File('scripts/evaluate_mutual_information.py')
rank_script = bb.File('scripts/rank_features.py')
eval_script = bb.File('scripts/evaluate_ranked_features.py')
for model, split in it.product(MODELS, RANKING_SPLITS):
    test_data = bb.Group(SIMULATION_ROOT / model / 'test')
    with bb.group_artifacts(RANKING_ROOT, model, split):
        # Get the simulations we're going to need.
        sims = SIMULATIONS[(model, split)]

        # Precompute the mutual information.
        args = ['$!', mi_script, '--adjusted', '$@', *sims]
        mutual_info, = bb.Subprocess(['mutual_information.pkl'], sims, args)

        # Run all the methods, adding the precomputed mutual information to some methods.
        for method in METHODS:
            with bb.group_artifacts(method):
                rankings = []
                for penalty in PENALTIES:
                    inputs = list(sims)
                    args = ['$!', rank_script, '--adjusted', f'--penalty={penalty}', method, '$@']
                    if method in ['JMI', 'JMIM', 'mRMR']:
                        args.append(f'--mi={mutual_info}')
                        inputs.append(mutual_info)
                    args.extend(sims)
                    rankings.extend(bb.Subprocess(f'{penalty}.pkl', inputs, args))

            # Run the evaluation.
            ranking_path = bb.Group(method)
            args = ['$!', eval_script, ranking_path.name, test_data.name, '$@']
            bb.Subprocess(f'{method}_eval.pkl', rankings + SIMULATIONS[(model, "test")], args)


# Run non-deterministc methods with different seeds to test how variable the outputs are.
model = "dmX"
split = "train"
penalty = 0.5
with bb.group_artifacts(RANKING_ROOT, model, split):
    sims = SIMULATIONS[(model, split)]
    mutual_info = bb.File('mutual_information.pkl')
    for method in METHODS:
        with bb.group_artifacts("repeats", method):
            for seed in range(50):
                args = ['$!', rank_script, f'--seed={seed}', f'--penalty={penalty}', method, '$@',
                        *sims]
                if method in ['JMI', 'JMIM', 'mRMR']:
                    args.append(f'--mi={mutual_info}')
                    inputs.append(mutual_info)
                bb.Subprocess(f'seed-{seed}.pkl', list(sims), args)
