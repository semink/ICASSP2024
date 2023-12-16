import networkx as nx
import numpy as np
import scipy


def block_circular_layout(G, N, outer_scale=1, inner_scale=1):
    pos = nx.circular_layout(G, scale=outer_scale)

    # Reposition nodes in a block to be evenly spaced sqaure grid that centers at a given position
    def reposition(pos, center):
        # pos: dictionary of node positions
        # center: center position of the block
        new_pos = {}
        size = len(pos)
        n = int(np.sqrt(size))
        m = int(np.ceil(size / n))
        keymap = {i: p for i, p in zip(range(size), pos.keys())}
        for i in range(size):
            new_pos[keymap[i]] = (
                center[0] + inner_scale * (i % n - n / 2) * 0.5,
                center[1] + inner_scale * (i // n - m / 2) * 0.5,
            )
        return new_pos

    # Reposition nodes in each block
    new_pos = {}
    for i, j in zip(np.cumsum([0] + N)[:-1], np.cumsum(N)):
        new_pos.update(reposition({k: pos[k] for k in range(i, j)}, pos[i]))
    return new_pos


def balanced_M_block_cyclic_graph(M, N, sparcity=0.3):
    if N % M != 0:
        raise ValueError("N/M must be an integer.")
    # if type of sparcity is not float, raise error
    if type(sparcity) != float:
        if len(sparcity) != M:
            raise ValueError(
                "sparcity must be a float or a list of floats that has same length of M."
            )
    else:
        sparcity = [sparcity] * M

    node_per_block = N // M

    block_matrix = []
    for m, s in zip(range(M), sparcity):
        A_position = (m - 1 + M) % M
        row = [np.zeros((node_per_block, node_per_block))] * M
        sub_A = np.random.rand(node_per_block, node_per_block)
        # set elements of sub_A to zero with probability 1 - sparcity
        sub_A = sub_A * (np.random.rand(node_per_block, node_per_block) > s).astype(
            float
        )
        row[A_position] = sub_A
        block_matrix.append(row)
    A = np.block(block_matrix)
    # A[A != 0] = 1  # set non-zero elements to 1
    return A


def polar_decomposition_based_on_M_block_cyclic_graph(A, M, N):
    U, S, vh = SVD_of_M_block_cyclic_graph(A, M, N)
    Q, F, P = polar_decomposition_based_on_SVD(U, S, vh)
    return Q, F, P


def SVD_of_M_block_cyclic_graph(A, M, N):
    node_per_block = int(N / M)
    U = []
    block_Vh = []
    S = []
    for m in range(M):
        block_position = (m - 1 + M) % M
        sub_A = A[
            np.array(m * node_per_block + np.arange(node_per_block))[:, np.newaxis],
            np.array(block_position * node_per_block + np.arange(node_per_block)),
        ]
        u, s, vh = np.linalg.svd(sub_A)
        U.append(u)
        S.append(np.diag(s))
        row = [np.zeros((node_per_block, node_per_block))] * M
        row[block_position] = vh
        block_Vh.append(row)
    U = scipy.linalg.block_diag(*U)
    S = scipy.linalg.block_diag(*S)
    vh = np.block(block_Vh)
    return U, S, vh


def polar_decomposition_based_on_SVD(U, S, vh):
    F = vh.T @ S @ vh
    P = U @ S @ U.T
    Q = U @ vh
    return Q, F, P


def cum_psd(freq):
    cpsd = (freq.abs() ** 2).cumsum(axis=1)
    return cpsd.T / cpsd.max(axis=1).values
