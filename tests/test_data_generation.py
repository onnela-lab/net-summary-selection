from cost_based_selection import data_generation as dg
import networkx as nx


def test_BA_ref_table():
    num_nodes = 20
    num_sims_per_model = 10
    num_sims = 4 * num_sims_per_model
    model_index, summaries, is_discrete, times = dg.BA_ref_table(num_sims_per_model, num_nodes)
    assert model_index.shape == (num_sims, 1)
    assert summaries.shape == (num_sims, 51)
    assert is_discrete.shape == (1, 51)
    assert times.shape == (num_sims, 51)


def test_DMC_DMR_ref_table():
    seed_network = nx.Graph()
    seed_network.add_node(0)
    model_index, summaries, is_discrete, times = dg.DMC_DMR_ref_table(seed_network, 10, 20)
