from doit.action import CmdAction
from doit.tools import create_folder
import functools as ft
import itertools as it
import numpy as np
import os
import pathlib
import sys


# Standard environment variables to avoid interaction between different processes.
ENV = {
    "NUMEXPR_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
}
cmd_action = ft.partial(CmdAction, shell=False, env=os.environ | ENV)
default_task = {
    "io": {
        "capture": False,
    }
}

# Interpreter for launching subtasks.
PYTHON = sys.executable
# Arguments for starting a subprocess that halts on exceptions.
DEBUG_ARGS = [PYTHON, "-m", "pdb", "-c", "continue"]
# Root directory for generating results.
ROOT = pathlib.Path("workspace")
# The different network models we consider.
MODELS = ["ba", "dmX"]
# Batch size and number of batches. The size of the reference table is (number of batches) *
# (batch size).
BATCH_SIZE = 100
NUM_BATCHES = 50
# Configurations for different simulations. seed_offset guarantees that there are no shared seeds
# amongst different simulations.
CONFIGS = {
    "train": {
        "num_nodes": 1_000,
        "seed_offset": 0,
    },
    "test": {
        "num_nodes": 1_000,
        "seed_offset": 999,
    },
}

# Generate `num_seeds` different versions of the pilot rankings.
num_seeds = 5
for num_nodes, seed in it.product((1 + np.arange(10)) * 100, range(num_seeds)):
    CONFIGS[f"pilot/num_nodes-{num_nodes}/seed-{seed}"] = {
        "num_nodes": num_nodes,
        "seed_offset": num_nodes * 31 + 7919 * seed,
    }

CONFIGS.update({
    "yeast/yu2008": {
        "num_nodes": 1278,
        "seed_offset": 9717,
    },
    "yeast/ito2001": {
        "num_nodes": 813,
        "seed_offset": 9789,
    },
    "yeast/uetz2000": {
        "num_nodes": 437,
        "seed_offset": 9990,
    }
})

# Different methods to generate rankings from.
METHODS = [
    "JMI", "JMIM", "mRMR", "reliefF_l1", "reliefF_rf", "random_ranking",
    "pen_rf_importance_impurity", "pen_rf_importance_permutation",
    "weighted_rf_importance_impurity", "weighted_rf_importance_permutation",
]
# Cost regularization values.
PENALTIES = [
    0, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1, 1.6, 2.3, 3.2, 4.5, 6.4, 8, 9, 10,
    11, 12, 12.8, 25.6, 51.2, 102.4, 204.8,
]


# Generate all tasks within one function. This is somewhat unconventional but avoids having repeated
# loops in different tasks.
def task_main():
    simulation_script = "scripts/generate_reference_table.py"
    ranking_script = "scripts/rank_features.py"
    mi_script = "scripts/evaluate_mutual_information.py"
    eval_script = "scripts/evaluate_ranked_features.py"

    for model in MODELS:
        configs_by_seed = {}
        for key, config in CONFIGS.items():
            # Don't run Barabasi-Albert simulations for yeast models.
            if key.startswith("yeast") and model != "dmX":
                continue
            simulation_basename = os.path.join(model, "simulations", key)
            # Generate realisations of different models with varying number of nodes.
            simulation_targets = []
            for batch in range(NUM_BATCHES):
                seed = config["seed_offset"] + batch
                configs_by_seed.setdefault(seed, []).append(f"{key}/batch-{batch}")
                name = f"batch-{batch}"
                target = ROOT / simulation_basename / (name + ".pkl")
                action = cmd_action([PYTHON, simulation_script, f"--seed={seed}", model,
                                    str(config["num_nodes"]), str(BATCH_SIZE), target])
                yield default_task | {
                    "basename": simulation_basename,
                    "name": name,
                    "actions": [action],
                    "file_dep": [simulation_script],
                    "targets": [target],
                    "doc": f"simulation for {model} models with {config['num_nodes']} (batch "
                           f"{batch})",
                }
                simulation_targets.append(target)

            # No need to do any ranking or evaluation for the yeast data.
            if key.startswith("yeast"):
                continue

            # No need to rank the test set.
            if key == "test":
                continue

            # Precompute the (conditional) mutual information between features.
            mi_basename = os.path.join(model, "rankings", key)
            name = "mutual_information"
            mi_target = ROOT / mi_basename / (name + ".pkl")
            action = cmd_action([PYTHON, mi_script, "--adjusted", mi_target, *simulation_targets])
            yield default_task | {
                "basename": mi_basename,
                "name": name,
                "actions": [action],
                "file_dep": simulation_targets,
                "targets": [mi_target],
            }

            # Rank features using different methods for different configurations. We only use
            # unpenalized methods for comparing between features selected based on varying number of
            # nodes.
            penalties = [0.] if key.startswith("pilot") else PENALTIES
            for method in METHODS:
                ranking_basename = os.path.join(model, "rankings", key, method)
                ranking_targets = []
                for penalty in penalties:
                    # Building the trees for the weighted RF penalization takes AGES if the penalty
                    # is large. So we just skip large penalties here.
                    if method.startswith("weighted_rf_importance") and penalty > 100:
                        continue

                    name = f"penalty-{penalty}"
                    target = ROOT / ranking_basename / (name + ".pkl")

                    # Construct arguments for the ranking of features.
                    file_dep = list(simulation_targets)
                    args = [PYTHON, ranking_script, '--adjusted', f'--penalty={penalty}', method,
                            target]
                    if method in ['JMI', 'JMIM', 'mRMR']:
                        args.append(f'--mi={mi_target}')
                        file_dep.append(mi_target)
                    args = args + simulation_targets

                    yield default_task | {
                        "basename": ranking_basename,
                        "actions": [cmd_action(args)],
                        "targets": [target],
                        "name": name,
                        "file_dep": file_dep,
                    }
                    ranking_targets.append(target)

                # Evaluate ranked features on the test set.
                evaluation_basename = os.path.join(model, "evaluation", key)
                target = ROOT / evaluation_basename / f"{method}.pkl"
                action = cmd_action([PYTHON, eval_script, "--model=KNN", ROOT / ranking_basename,
                                    ROOT / model / "simulations" / "test", target])
                # Construct dependencies which comprises rankings and the simulations of the test
                # set.
                file_dep = list(ranking_targets)
                file_dep.extend([ROOT / model / "simulations" / "test" / f"batch-{batch}.pkl" for
                                 batch in range(NUM_BATCHES)])

                yield default_task | {
                    "basename": evaluation_basename,
                    "name": method,
                    "targets": [target],
                    "actions": [action],
                    "file_dep": file_dep,
                }

        for seed, configs in configs_by_seed.items():
            if len(configs) > 1:
                raise ValueError(f"configs {configs} have the same seed {seed}")


def task_yeast_interactome():
    urls = {
        "yu2008": "http://interactome.dfci.harvard.edu/S_cerevisiae/download/CCSB-Y2H.txt",
        "ito2001": "http://interactome.dfci.harvard.edu/S_cerevisiae/download/Ito_core.txt",
        "uetz2000": "http://interactome.dfci.harvard.edu/S_cerevisiae/download/Uetz_screen.txt",
    }
    for key, url in urls.items():
        dataset_filename = ROOT / f"dmX/data/{key}.txt"
        yield {
            "basename": "yeast_interactome/data",
            "name": key,
            "actions": [
                (create_folder, [dataset_filename.parent]),
                ["curl", "-o", str(dataset_filename), "-L", url]
            ],
            "uptodate": [True],
            "targets": [dataset_filename],
        }
