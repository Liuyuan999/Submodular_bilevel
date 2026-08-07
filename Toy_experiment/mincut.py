"""Shared submodular function minimization (SFM) oracle.

Every SFM call needed by ``pen:alg`` in both toy applications is a pairwise
binary energy of the form

    E(y) = sum_k unary_k(y_k) + sum_{(i,j)} theta_ij(y_i, y_j),

with each pairwise table theta_ij submodular (theta_ij(0,1) + theta_ij(1,0)
>= theta_ij(0,0) + theta_ij(1,1)) -- exactly the Ising/graph-cut recipe of
``app:lem:ising``. This module reduces such an energy to a min s-t cut via
the standard Kolmogorov-Zabih construction and solves it with PyMaxflow
(Boykov-Kolmogorov), the off-the-shelf max-flow/min-cut solver used
throughout the graph-cuts/Ising/MRF literature.

Convention: y_k = 0 <-> node k on the SOURCE side, y_k = 1 <-> SINK side.
For a pairwise table (E00, E01, E10, E11) with submodularity slack
    d_ij := E01 + E10 - E00 - E11 >= 0,
the decomposition

    theta_ij(y_i, y_j) = E01 + (E11 - E01) y_i + (E00 - E01) (1 - y_j)
                          + d_ij * y_i * (1 - y_j)

adds (E11 - E01) to node i's y=1 unary cost, (E00 - E01) to node j's y=0
unary cost, a constant E01, and a directed edge j -> i of capacity d_ij
(cut exactly when y_j = 0 and y_i = 1).
"""

import itertools
import random

import maxflow
import numpy as np


def solve_energy(n, unary, pairwise):
    """Minimize a pairwise submodular binary energy via min s-t cut.

    unary: (n, 2) array, unary[k] = (cost if y_k=0, cost if y_k=1).
    pairwise: list of (i, j, E00, E01, E10, E11), i != j, each table
        submodular. Multiple entries for the same (i, j) are summed.

    Returns (y, energy) with y an (n,) int array in {0,1} and energy the
    minimum value of E(y).
    """
    unary = np.asarray(unary, dtype=float)
    D = unary.copy()
    const = 0.0
    dir_edges = []
    for i, j, E00, E01, E10, E11 in pairwise:
        d = E01 + E10 - E00 - E11
        if d < -1e-7:
            raise ValueError(
                f"pairwise term ({i},{j}) is not submodular: slack {d:.6g} < 0"
            )
        d = max(d, 0.0)
        const += E01
        D[i, 1] += E11 - E01
        D[j, 0] += E00 - E01
        if d > 0.0:
            dir_edges.append((j, i, d))

    g = maxflow.Graph[float](n, len(dir_edges))
    nodes = g.add_nodes(n)
    for k in range(n):
        g.add_tedge(nodes[k], D[k, 1], D[k, 0])
    for j, i, d in dir_edges:
        g.add_edge(nodes[j], nodes[i], d, 0.0)
    flow = g.maxflow()
    y = np.array([g.get_segment(nodes[k]) for k in range(n)], dtype=int)
    return y, flow + const


def _brute_force(n, unary, pairwise):
    best_cost, best_y = None, None
    for y in itertools.product([0, 1], repeat=n):
        cost = sum(unary[k][y[k]] for k in range(n))
        for i, j, E00, E01, E10, E11 in pairwise:
            table = {(0, 0): E00, (0, 1): E01, (1, 0): E10, (1, 1): E11}
            cost += table[(y[i], y[j])]
        if best_cost is None or cost < best_cost - 1e-9:
            best_cost, best_y = cost, y
    return np.array(best_y), best_cost


def self_check(num_trials=200, seed=0, max_nodes=6):
    """Cross-check solve_energy against brute force on random small instances.

    Raises AssertionError on any mismatch. Called once at import time by
    both application scripts before the outer pen:alg loop runs, so a
    broken graph reduction is caught immediately rather than silently
    corrupting the algorithm's gradients downstream.
    """
    rng = random.Random(seed)
    for trial in range(num_trials):
        n = rng.randint(2, max_nodes)
        unary = [
            (round(rng.uniform(-3, 3), 3), round(rng.uniform(-3, 3), 3))
            for _ in range(n)
        ]
        pairs = list(itertools.combinations(range(n), 2))
        rng.shuffle(pairs)
        pairwise = []
        for i, j in pairs[: rng.randint(1, len(pairs))]:
            E00 = round(rng.uniform(0, 3), 3)
            E11 = round(rng.uniform(0, 3), 3)
            E01 = round(rng.uniform(0, 3), 3)
            slack = round(rng.uniform(0, 3), 3)
            E10 = E00 + E11 - E01 + slack
            pairwise.append((i, j, E00, E01, E10, E11))
        y_cut, cost_cut = solve_energy(n, unary, pairwise)
        _, cost_bf = _brute_force(n, unary, pairwise)
        if abs(cost_cut - cost_bf) > 1e-5:
            raise AssertionError(
                f"mincut.solve_energy self-check failed on trial {trial}: "
                f"got {cost_cut}, brute force {cost_bf}"
            )
    return True


if __name__ == "__main__":
    self_check()
    print("mincut.solve_energy self-check passed.")
