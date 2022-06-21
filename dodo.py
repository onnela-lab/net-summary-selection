from doit.action import CmdAction
import functools as ft
import itertools as it
import numpy as np
import os
import pathlib
import re
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
for num_nodes in (1 + np.arange(10)) * 100:
    CONFIGS[f"n{num_nodes}"] = {
        "num_nodes": num_nodes,
        "seed_offset": num_nodes * 31,
    }

# Different methods to generate rankings from.
METHODS = [
    "JMI", "JMIM", "mRMR", "reliefF_distance", "reliefF_rf_prox", "random_ranking",
    "pen_rf_importance_impurity", "pen_rf_importance_permutation",
    "weighted_rf_importance_impurity", "weighted_rf_importance_permutation",
]
# Cost regularization values.
PENALTIES = [
    0, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.1, 1.6, 2.3, 3.2, 4.5, 6.4, 8, 9, 10, 11, 12,
    12.8, 25.6, 51.2,
]


def task_main():
    """
    Generate all tasks within one function. This is somewhat unconventional but avoids having
    repeated loops in different tasks.
    """
    simulation_script = "scripts/generate_reference_table.py"
    ranking_script = "scripts/rank_features.py"
    mi_script = "scripts/evaluate_mutual_information.py"
    eval_script = "scripts/evaluate_ranked_features.py"

    for (key, config), model in it.product(CONFIGS.items(), MODELS):
        # Generate realisations of different models with varying number of nodes.
        simulation_targets = []
        for batch in range(NUM_BATCHES):
            seed = config["seed_offset"] + batch
            basename = os.path.join(model, "simulations", key)
            name = f"batch-{batch}"
            target = ROOT / basename / (name + ".pkl")
            action = cmd_action([PYTHON, simulation_script, f"--seed={seed}", model,
                                str(config["num_nodes"]), str(BATCH_SIZE), target])
            yield default_task | {
                "basename": basename,
                "name": name,
                "actions": [action],
                "file_dep": [simulation_script],
                "targets": [target],
            }
            simulation_targets.append(target)

        # No need to rank the test set.
        if key == "test":
            continue

        # Precompute the (conditional) mutual information between features.
        basename = os.path.join(model, "rankings", key)
        name = "mutual_information"
        mi_target = ROOT / basename / (name + ".pkl")
        action = cmd_action([PYTHON, mi_script, "--adjusted", mi_target, *simulation_targets])
        yield default_task | {
            "basename": basename,
            "name": name,
            "actions": [action],
            "file_dep": simulation_targets,
            "targets": [mi_target],
        }

        # Rank features using different methods for different configurations. We only use
        # unpenalized methods for comparing between features selected based on varying number of
        # nodes.
        penalties = [0.] if re.match(r"n\d+", key) else PENALTIES
        for method in METHODS:
            basename = os.path.join(model, "rankings", key, method)
            ranking_targets = []
            for penalty in penalties:
                name = f"penalty-{penalty}"
                target = ROOT / basename / (name + ".pkl")

                # Construct arguments for the ranking of features.
                file_dep = list(simulation_targets)
                args = [PYTHON, ranking_script, '--adjusted', f'--penalty={penalty}', method,
                        target]
                if method in ['JMI', 'JMIM', 'mRMR']:
                    args.append(f'--mi={mi_target}')
                    file_dep.append(mi_target)
                args = args + simulation_targets

                yield default_task | {
                    "basename": basename,
                    "actions": [cmd_action(args)],
                    "targets": [target],
                    "name": name,
                    "file_dep": file_dep,
                }
                ranking_targets.append(target)

            # Evaluate ranked features on the test set.
            name = "evaluation"
            target = ROOT / basename / (name + ".pkl")
            action = cmd_action([PYTHON, eval_script, "--model=KNN", ROOT / basename,
                                 ROOT / model / "simulations" / "test", target])
            # Construct dependencies which comprises rankings and the simulations of the test set.
            file_dep = list(ranking_targets)
            file_dep.extend([ROOT / model / "simulations" / "test" / f"batch-{batch}.pkl" for batch
                             in range(NUM_BATCHES)])

            yield default_task | {
                "basename": basename,
                "name": name,
                "targets": [target],
                "actions": [action],
                "file_dep": file_dep,
            }
