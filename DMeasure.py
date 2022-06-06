from collections import Counter

import numpy as np
import networkx as nx
from scipy.stats import entropy


def node_distance(G: nx.Graph) -> np.ndarray:
    """
    Return an NxN matrix that consists of histograms of shortest path
    lengths between nodes i and j.
    """

    N = G.number_of_nodes()
    a = np.zeros((N, N))

    dists = nx.shortest_path_length(G)
    for idx, row in enumerate(dists):
        counts = Counter(row[1].values())
        a[idx] = [counts[l] for l in range(1, N + 1)]

    return a / (N - 1)


def NND(G: nx.Graph) -> (np.ndarray, np.ndarray):
    """
    This function calculates the network node dispersion of a graph G. This
    function also returns the average of the each node-distance distribution.
    """

    N = G.number_of_nodes()
    nd = node_distance(G)
    pdfm = np.mean(nd, axis=0)

    norm = np.log(nx.diameter(G) + 1)

    ndf = nd.flatten()
    # calculate the entropy, with the convention that 0/0 = 0
    entr = -1 * sum(ndf * np.log(ndf, out=np.zeros_like(ndf), where=(ndf != 0)))

    nnd = max([0, entropy(pdfm) - entr / N]) / norm

    return nnd, pdfm


def alpha_centrality_prob(G: nx.Graph, niter: int):
    """
    Returns a probability distribution over alpha centralities for the network.
    Parameters
    ----------
    G (nx.Graph): the graph in question.
    niter (int): the number of iterations needed to converge properly.
    Returns:
    alpha_prob (np.ndarray): a vector of probabilities for each node in G.
    """

    # calculate the alpha centrality for each node
    N = G.number_of_nodes()
    alpha = 1 / N

    A = nx.to_numpy_array(G)

    s = A.sum(axis=1)
    cr = s.copy()

    for _ in range(niter):
        cr = s + alpha * A.dot(cr)

    # turn the alpha centralities into a probability distribution
    cr = cr / (N - 1)
    r = sorted(cr / (N ** 2))
    alpha_prob = list(r) + [max([0, 1 - sum(r)])]

    return np.array(alpha_prob)


def js_divergence(P: np.ndarray, Q: np.ndarray):
    """Jensen-Shannon divergence between `P` and `Q`.
    Parameters
    ----------
    P, Q (np.ndarray)
        Two discrete distributions represented as 1D arrays. They are
        assumed to have the same support
    Returns
    -------
    float
        The Jensen-Shannon divergence between `P` and `Q`.
    """
    M = 0.5 * (P + Q)
    jsd = 0.5 * (entropy(P, M, base=2) + entropy(Q, M, base=2))

    # If the input distributions are identical, floating-point error in the
    # construction of the mixture matrix can result in negative values that are
    # very close to zero. If one wants to compute the root-JSD metric, these
    # negative values lead to undesirable nans.
    if np.isclose(jsd, 0.0):
        return 0
    else:
        return jsd


def DMeasure(G1: nx.Graph, G2: nx.Graph, w1=0.45, w2=0.45, w3=0.10, niter=50):
    """The D-Measure is a comparison of structural dissimilarities between graphs.
    Parameters
    ----------
    G1 (nx.Graph):
        the first graph to be compared.
    G2 (nx.Graph):
        the second graph to be compared.
    w1 (float):
        weight of the first term in the calculation;
        with w2 and w3, must sum to 1.0.
    w2 (float):
        weight of the second term in the calculation;
        with w1 and w3, must sum to 1.0.
    w3 (float):
        weight of the third term in the calculation;
        with w1 d w2, must sum to 1.0.
    niter (int):
        the alpha centralities are calculated using power iteration, with
        this many iterations
    Returns
    -------
    dist (float):
        between 0 and 1, the D-measure distance between G1 and G2
    """

    if sum([w1, w2, w3]) != 1:
        raise ValueError("Weights must sum to one.")

    first_term = 0
    second_term = 0
    third_term = 0

    if w1 + w2 > 0:
        g1_nnd, g1_pdfs = NND(G1)
        g2_nnd, g2_pdfs = NND(G2)

        first_term = np.sqrt(js_divergence(g1_pdfs, g2_pdfs))
        second_term = np.abs(np.sqrt(g1_nnd) - np.sqrt(g2_nnd))

    if w3 > 0:
        def alpha_jsd(G1, G2):
            """
            Compute the Jensen-Shannon divergence between the
            alpha-centrality probability distributions of two graphs.
            """
            p1 = alpha_centrality_prob(G1, niter=niter)
            p2 = alpha_centrality_prob(G2, niter=niter)

            m = max([len(p1), len(p2)])

            P1 = np.zeros(m)
            P2 = np.zeros(m)

            P1[(m - len(p1)): m] = p1
            P2[(m - len(p2)): m] = p2

            return js_divergence(P1, P2)

        G1c = nx.complement(G1)
        G2c = nx.complement(G2)

        first_jsd = alpha_jsd(G1, G2)
        second_jsd = alpha_jsd(G1c, G2c)
        third_term = 0.5 * (np.sqrt(first_jsd) + np.sqrt(second_jsd))

    dist = w1 * first_term + w2 * second_term + w3 * third_term

    return dist
