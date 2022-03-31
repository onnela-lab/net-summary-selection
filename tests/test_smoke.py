from cost_based_selection import cost_based_analysis
from cost_based_selection import cost_based_methods
from cost_based_selection import data_generation
from cost_based_selection import preprocessing_utils
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def test_smoke():
    """
    This test reproduces the README code.
    """
    num_sims_per_model = 10
    num_nodes = 20

    # Generate data.
    model_indices, summaries, is_discrete, times = \
        data_generation.BA_ref_table(num_sims_per_model, num_nodes)

    # Preprocess the data.
    model_indices, summaries, is_discrete, times = \
        preprocessing_utils.drop_redundant_features(model_indices, summaries, is_discrete, times)
    noise_indices = preprocessing_utils.noise_position(summaries)
    average_costs = preprocessing_utils.compute_avg_cost(times)

    # Train test split for validation.
    X = np.array(summaries)
    y = model_indices.modIndex.tolist()
    is_discrete = is_discrete.iloc[0, :].tolist()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, stratify=y)

    # Apply the JMI method over a grid.
    result = pd.DataFrame({"cost_param": [0, 1, 2]})
    rankings = [cost_based_methods.JMI(
            X=X_train, y=y_train, is_disc=is_discrete, cost_vec=average_costs, cost_param=cost_param
        )[0] for cost_param in result.cost_param]
    result = pd.concat([result, pd.DataFrame(rankings, columns=1 + np.arange(X.shape[1]))], axis=1)

    # Run cross validation for model classification.
    num_statistics = 15
    classifier = KNeighborsClassifier
    kwargs = {'n_neighbors': 10}

    avg_accuracy, std_accuracy, total_cost, prop_noise = \
        cost_based_analysis.accuracy_classifier_plot(
            dfPen_Ranking=result, X_val=X_val, y_val=y_val, cost_vec=average_costs,
            noise_idx=noise_indices, subset_size=num_statistics, classifier_func=classifier,
            args_classifier=kwargs, num_fold=3, save_name=None,
        )


if __name__ == '__main__':
    test_smoke()
