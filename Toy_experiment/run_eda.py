"""Toy instance of Example app:ex:eda (EDA circuit re-partitioning under
continuous placement), run through Algorithm pen:alg.

Dataset: the real ISPD98 `ibm01` netlist (TILOS-AI-Institute/
HypergraphPartitioning, benchmark/ISPD_benchmark/ibm01.hgr), hMETIS format:
header "#hyperedges #vertices", then one hyperedge per line as space-
separated 1-indexed cell ids. The full netlist (14111 hyperedges, 12752
cells) is far bigger than a toy, so we BFS out from one seed cell to a
small connected sub-netlist and clique-expand its surviving hyperedges
(dropping any hyperedge with more than MAX_HEDGE_SIZE pins -- global
nets like clock/reset that would otherwise dominate connectivity) into
the plain pairwise graph app:ex:eda is defined over.

x = (z_i)_{i in V} in R^{2n}, a 2D placement.
w_ij(x) = w0_ij * exp(-||z_i-z_j||^2/sigma^2)   (netlist-connection strength)
g(x,y) = sum_{(i,j) in E} w_ij(x) * 1{y_i=1,y_j=0}          (directed cut cost)
f(x,y) = sum_{(i,j) in E} a_ij(x) * 1{y_i=y_j},  a_ij(x):=rho*w_ij(x)
beta = sup 2*a_ij(x)/w_ij(x) = 2*rho exactly (same sigma cancels).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mincut
import penalty_alg

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

SEED_CELL = 50
SUBSET_SIZE = 50
MAX_HEDGE_SIZE = 8

SIGMA = 1.0
RHO = 0.15
BETA = 2.0 * RHO
GAMMA = 1.2 * BETA
BOX_MARGIN = 1.5
# g's unary c_i^1(x):=KAPPA*z_i_x, c_i^0(x):=-KAPPA*z_i_x (app:ex:eda's generic
# unary terms, instantiated as a physical die-half assignment cost tied to
# each cell's x-placement). Without it, minimizing g's pure cut cost alone
# (no unary) is degenerate: putting every cell on one side is always a
# global optimum (cost 0), independent of x, which would make y2 carry no
# information. Unary terms never affect submodularity of the pairwise part
# (app:lem:ising: Delta^2 only sees the pairwise table), so this is free.
KAPPA = 0.3

NUM_ITERS = 150
STEP0 = 0.03


def parse_hgr(path):
    with open(path) as f:
        lines = f.readlines()
    n_hedges, n_verts = (int(t) for t in lines[0].split()[:2])
    hedges = [[int(t) - 1 for t in line.split()] for line in lines[1 : 1 + n_hedges]]
    return hedges, n_verts


def build_sub_netlist(hedges, seed, target_size, max_hedge_size):
    cell_to_hedges = {}
    for hidx, e in enumerate(hedges):
        if len(e) > max_hedge_size:
            continue
        for c in e:
            cell_to_hedges.setdefault(c, []).append(hidx)

    visited = {seed}
    order = [seed]
    qi = 0
    while len(visited) < target_size and qi < len(order):
        c = order[qi]
        qi += 1
        for h in sorted(cell_to_hedges.get(c, []), key=lambda h: len(hedges[h])):
            for nb in hedges[h]:
                if nb not in visited:
                    visited.add(nb)
                    order.append(nb)
                    if len(visited) >= target_size:
                        break
            if len(visited) >= target_size:
                break

    V = sorted(visited)
    relabel = {c: i for i, c in enumerate(V)}
    surviving = [e for e in hedges if len(e) <= max_hedge_size and all(c in visited for c in e)]

    w0 = {}
    for e in surviving:
        k = len(e)
        for a in range(k):
            for b in range(a + 1, k):
                i, j = relabel[e[a]], relabel[e[b]]
                if i > j:
                    i, j = j, i
                w0[(i, j)] = w0.get((i, j), 0.0) + 1.0 / (k - 1)

    edges = [(i, j, w) for (i, j), w in sorted(w0.items())]
    return len(V), edges


def spring_layout(n, edges, iters=400, lr=0.03, seed=0):
    rng = np.random.RandomState(seed)
    z = rng.uniform(-1.0, 1.0, size=(n, 2))
    for _ in range(iters):
        grad = np.zeros_like(z)
        for i, j, w in edges:
            d = z[i] - z[j]
            dist = np.linalg.norm(d) + 1e-9
            diff = dist - 1.0
            gij = (2.0 * diff / dist) * d * min(w, 1.0)
            grad[i] += gij
            grad[j] -= gij
        z -= lr * grad
    return z


class EDAApp:
    def __init__(self, n, edges, z0):
        self.n = n
        self.edges = edges  # list of (i, j, w0)
        self.x0 = z0.reshape(-1).copy()
        self.R = np.abs(z0).max() + BOX_MARGIN

    def _split(self, x):
        return x.reshape(self.n, 2)

    def project(self, x):
        return np.clip(x, -self.R, self.R)

    def _weighted_edges(self, x):
        z = self._split(x)
        out = []
        for i, j, w0 in self.edges:
            d = z[i] - z[j]
            dist2 = float(np.dot(d, d))
            w = w0 * np.exp(-dist2 / SIGMA ** 2)
            out.append((i, j, w, d))
        return out

    def _g_unary(self, x):
        z = self._split(x)
        unary = np.zeros((self.n, 2))
        unary[:, 1] = KAPPA * z[:, 0]
        unary[:, 0] = -KAPPA * z[:, 0]
        return unary

    def solve_g(self, x):
        info = self._weighted_edges(x)
        unary = self._g_unary(x)
        pairwise = [(i, j, 0.0, 0.0, w, 0.0) for (i, j, w, d) in info]
        y, g_val = mincut.solve_energy(self.n, unary, pairwise)
        return y, g_val

    def solve_f_plus_gamma_g(self, x, gamma):
        info = self._weighted_edges(x)
        unary = gamma * self._g_unary(x)
        pairwise = []
        for i, j, w, d in info:
            a = RHO * w
            pairwise.append((i, j, a, 0.0, gamma * w, a))
        y, energy = mincut.solve_energy(self.n, unary, pairwise)
        f_val = sum(RHO * w for (i, j, w, d) in info if y[i] == y[j])
        g_val = (energy - f_val) / gamma
        return y, f_val, g_val

    def grad_x_g(self, x, y):
        info = self._weighted_edges(x)
        grad = np.zeros((self.n, 2))
        for i, j, w, d in info:
            if y[i] == 1 and y[j] == 0:
                gi = w * (-2.0 / SIGMA ** 2) * d
                grad[i] += gi
                grad[j] -= gi
        grad[:, 0] += KAPPA * (2 * y - 1)
        return grad.reshape(-1)

    def grad_x_f(self, x, y):
        info = self._weighted_edges(x)
        grad = np.zeros((self.n, 2))
        for i, j, w, d in info:
            if y[i] == y[j]:
                gi = RHO * w * (-2.0 / SIGMA ** 2) * d
                grad[i] += gi
                grad[j] -= gi
        return grad.reshape(-1)

    def f(self, x, y):
        info = self._weighted_edges(x)
        return sum(RHO * w for (i, j, w, d) in info if y[i] == y[j])

    def g(self, x, y):
        info = self._weighted_edges(x)
        cut_cost = sum(w for (i, j, w, d) in info if y[i] == 1 and y[j] == 0)
        z = self._split(x)
        unary_cost = np.sum(KAPPA * z[:, 0] * (2 * y - 1))
        return cut_cost + unary_cost


if __name__ == "__main__":
    print("Running mincut self-check...")
    mincut.self_check()
    print("Self-check passed.\n")

    hedges, n_verts_total = parse_hgr(os.path.join(DATA, "ibm01.hgr"))
    print(f"Parsed ibm01.hgr: {len(hedges)} hyperedges, {n_verts_total} cells")

    n, edges = build_sub_netlist(hedges, SEED_CELL, SUBSET_SIZE, MAX_HEDGE_SIZE)
    print(f"Sub-netlist: {n} cells, {len(edges)} clique-expanded pairwise nets")

    z0 = spring_layout(n, edges)
    app = EDAApp(n, edges, z0)

    print(f"beta = {BETA:.4f}, gamma = {GAMMA:.4f}, box radius = {app.R:.3f}")
    step_sizes = STEP0 / (1.0 + 0.05 * np.arange(NUM_ITERS))
    log = penalty_alg.run(app, GAMMA, NUM_ITERS, step_sizes)

    y2_final = log["y2_hist"][-1]
    print(f"Final partition sizes: side1={y2_final.sum()}, side0={n - y2_final.sum()}")

    def plot_layout(ax, z, y, title):
        for i, j, w0 in edges:
            cut = y[i] != y[j]
            ax.plot(
                [z[i, 0], z[j, 0]],
                [z[i, 1], z[j, 1]],
                color="red" if cut else "lightgray",
                linewidth=1.5 if cut else 0.8,
                zorder=1,
            )
        colors = np.where(y == 1, "tab:blue", "tab:orange")
        ax.scatter(z[:, 0], z[:, 1], c=colors, s=40, zorder=2, edgecolors="black", linewidths=0.5)
        ax.set_title(title)
        ax.set_aspect("equal")

    z_init = z0
    z_final = app._split(log["x_hist"][-1])
    y_init = log["y2_hist"][0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    plot_layout(axes[0], z_init, y_init, "Initial placement (t=0), side by y2")
    plot_layout(axes[1], z_final, y2_final, f"Final placement (t={NUM_ITERS}), side by y2")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "eda_layout.png"), dpi=150)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].plot(log["p_gamma"])
    axes[0].set_title(r"$p_\gamma(x_t)$")
    axes[0].set_xlabel("iteration")
    axes[1].plot(log["exact_gap"])
    axes[1].set_title(r"exactness gap $g(x_t,y_1)-g(x_t,y_2)$")
    axes[1].set_xlabel("iteration")
    axes[2].plot(log["phi_est"])
    axes[2].set_title(r"$\phi$ estimate $f(x_t,y_2)$")
    axes[2].set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "eda_curves.png"), dpi=150)

    print(f"\nSaved plots to {OUT}/eda_layout.png and {OUT}/eda_curves.png")
