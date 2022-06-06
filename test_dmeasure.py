import networkx as nx
import numpy as np

from DMeasure import DMeasure as _DMeasure


def DMeasure(G1, G2):
    dist = _DMeasure(G1, G2)
    assert 0 <= dist < 1
    return dist


def test_same_graph():
    G = nx.karate_club_graph()
    assert DMeasure(G, G) == 0


def test_different_graphs():
    G1 = nx.fast_gnp_random_graph(100, 0.3)
    G2 = nx.barabasi_albert_graph(100, 5)
    assert DMeasure(G1, G2) > 0


def test_symmetric():
    G1 = nx.barabasi_albert_graph(100, 4)
    G2 = nx.fast_gnp_random_graph(100, 0.3)
    assert DMeasure(G1, G2) == DMeasure(G2, G1)


def test_isomorphic_graphs():
    G1 = nx.fast_gnp_random_graph(150, 0.10)

    N = G1.order()
    new_nodes = [(i + 5) % N for i in G1.nodes]

    # create G1 by permuting the adjacency matrix
    new_adj_mat = nx.to_numpy_array(G1, nodelist=new_nodes)
    G2 = nx.from_numpy_array(new_adj_mat)

    assert nx.is_isomorphic(G1, G2)
    assert np.isclose(DMeasure(G1, G2), 0.0, atol=1e-3)
