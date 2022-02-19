from cost_based_selection import data_generation as dg
import networkx as nx


def test_BA_ref_table():
    model_index, summaries, is_discrete, times = dg.BA_ref_table(10, 20)


def test():
    seed_network = nx.Graph()
    seed_network.add_node(0)
    model_index, summaries, is_discrete, times = dg.DMC_DMR_ref_table(seed_network, 10, 20)
