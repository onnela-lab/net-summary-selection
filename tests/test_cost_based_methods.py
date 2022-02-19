from cost_based_selection import cost_based_methods
from cost_based_selection import data_generation
from cost_based_selection import preprocessing_utils
import numpy as np
import pytest


@pytest.fixture(scope='session')
def data_dict():
    num_sims_per_model = 50
    num_nodes = 20

    # Generate data.
    model_indices, summaries, is_discrete, times = \
        data_generation.BA_ref_table(num_sims_per_model, num_nodes)

    # Preprocess the data.
    model_indices, summaries, is_discrete, times = \
        preprocessing_utils.drop_redundant_features(model_indices, summaries, is_discrete, times)
    average_costs = preprocessing_utils.compute_avg_cost(times)

    # Train test split for validation.
    X = np.array(summaries)
    y = model_indices.modIndex.tolist()
    is_discrete = is_discrete.iloc[0, :].tolist()
    return {
        'X': X,
        'y': y,
        'is_disc': is_discrete,
        'cost_vec': average_costs,
    }


@pytest.mark.parametrize('method', [
    cost_based_methods.JMI,
    cost_based_methods.JMIM,
    cost_based_methods.mRMR,
    cost_based_methods.reliefF,
    cost_based_methods.pen_rf_importance,
    cost_based_methods.weighted_rf_importance,
])
def test_method(data_dict: dict, method):
    # Remove `is_disc` for methods that don't accept the argument.
    if method in {cost_based_methods.reliefF, cost_based_methods.pen_rf_importance,
                  cost_based_methods.weighted_rf_importance}:
        data_dict = data_dict.copy()
        del data_dict['is_disc']
    method(**data_dict, cost_param=1)
