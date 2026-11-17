"""Utility functions for CASTLE baseline: random DAG generation and nonlinear data generation."""

import numpy as np
import networkx as nx
import pandas as pd


def random_dag(num_nodes, num_edges):
    """Generate a random directed acyclic graph (DAG).

    Args:
        num_nodes: Number of nodes in the graph.
        num_edges: Number of directed edges.

    Returns:
        networkx.DiGraph: A random DAG.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))

    # Build a topological ordering to guarantee DAG property
    order = list(range(num_nodes))
    np.random.shuffle(order)

    max_edges = num_nodes * (num_nodes - 1) // 2
    num_edges = min(num_edges, max_edges)

    possible_edges = [(order[i], order[j]) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    chosen = np.random.choice(len(possible_edges), size=num_edges, replace=False)

    for idx in chosen:
        u, v = possible_edges[idx]
        G.add_edge(u, v)

    return G


def gen_data_nonlinear(G, SIZE=1000, var=1.0):
    """Generate nonlinear data from a DAG using a structural equation model.

    Each node is computed as a nonlinear function of its parents plus noise.

    Args:
        G: networkx.DiGraph representing the causal structure.
        SIZE: Number of samples to generate.
        var: Noise variance.

    Returns:
        pd.DataFrame: Generated data with columns named by node indices.
    """
    num_nodes = G.number_of_nodes()
    order = list(nx.topological_sort(G))

    data = np.zeros((SIZE, num_nodes))
    noise = np.random.normal(0, np.sqrt(var), (SIZE, num_nodes))

    for node in order:
        parents = list(G.predecessors(node))
        parent_data = data[:, parents] if parents else np.zeros((SIZE, 1))

        if len(parents) == 0:
            data[:, node] = noise[:, node]
        elif len(parents) == 1:
            data[:, node] = np.tanh(parent_data[:, 0]) + noise[:, node]
        else:
            # Nonlinear combination of multiple parents
            data[:, node] = np.tanh(np.sum(parent_data, axis=1)) + noise[:, node]

    # Scale to reasonable range
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

    return pd.DataFrame(data, columns=[str(i) for i in range(num_nodes)])
