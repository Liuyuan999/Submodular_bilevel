"""Toy instance of Example app:ex:seg (binary image segmentation with
continuous data terms), run through Algorithm pen:alg.

Dataset: a real photo + hand-scribbled trimap from OpenCV's opencv_extra
test data (testdata/cv/grabcut/image1652.ppm, mask1652.ppm). The scribble
mask marks small definite-foreground (1) and definite-background (0)
regions (the rest, 2, is unmarked) -- it is used only to (a) initialize the
two color models and (b) as a qualitative sanity check at the end; it never
enters the pen:alg loop itself.

x = (mu_fg, mu_bg) in [0,1]^3 x [0,1]^3 (RGB color means, normalized).
g(x,y) = sum_i ||I_i-mu_fg||^2 y_i + ||I_i-mu_bg||^2 (1-y_i)
         + sum_{(i,j) in N} lambda * 1{y_i != y_j}                (Potts)
f(x,y) = sum_{(i,j) in N} mu_ij(x) * 1{y_i = y_j},
         mu_ij(x) := c / (1 + ||mu_fg - mu_bg||^2)                (agreement reward)
beta = sup mu_ij(x)/lambda = c/lambda, attained at mu_fg = mu_bg.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

import mincut
import penalty_alg

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

GRID = 45          # downsample to GRID x GRID pixels
LAMBDA = 0.10       # Potts weight in g
C_AGREE = 0.05      # numerator of mu_ij(x) in f
BETA = C_AGREE / LAMBDA
GAMMA = 1.2 * BETA
NUM_ITERS = 150
STEP0 = 1e-4  # grad_x_g sums (not averages) over ~n/2 pixels, so ||u|| ~ O(n);
# STEP0 is calibrated so STEP0*||u_0|| is a small fraction of the [0,1]^3 box.


def load_data():
    img = Image.open(os.path.join(DATA, "image1652.ppm")).convert("RGB")
    img = img.resize((GRID, GRID), Image.BILINEAR)
    I = np.asarray(img, dtype=float).reshape(-1, 3) / 255.0  # (n,3)

    mask = Image.open(os.path.join(DATA, "mask1652.ppm"))
    mask = mask.resize((GRID, GRID), Image.NEAREST)
    scribble = np.asarray(mask, dtype=int).reshape(-1)  # values in {0,1,2}
    return I, scribble


def build_adjacency(grid):
    edges = []
    for r in range(grid):
        for c in range(grid):
            i = r * grid + c
            if c + 1 < grid:
                edges.append((i, i + 1))
            if r + 1 < grid:
                edges.append((i, i + grid))
    return edges


class SegmentationApp:
    def __init__(self, I, edges, scribble):
        self.I = I               # (n,3) pixel colors in [0,1]
        self.n = I.shape[0]
        self.edges = edges       # list of (i,j)

        fg_seed = I[scribble == 1]
        bg_seed = I[scribble == 0]
        if len(fg_seed) == 0 or len(bg_seed) == 0:
            km = KMeans(n_clusters=2, n_init=4, random_state=0).fit(I)
            mu_fg0, mu_bg0 = km.cluster_centers_[0], km.cluster_centers_[1]
        else:
            mu_fg0, mu_bg0 = fg_seed.mean(axis=0), bg_seed.mean(axis=0)
        self.x0 = np.concatenate([mu_fg0, mu_bg0])

    def _split(self, x):
        return x[:3], x[3:]

    def project(self, x):
        return np.clip(x, 0.0, 1.0)

    def _pairwise_potts(self, weight):
        return [(i, j, 0.0, weight, weight, 0.0) for (i, j) in self.edges]

    def solve_g(self, x):
        mu_fg, mu_bg = self._split(x)
        unary = np.stack(
            [np.sum((self.I - mu_bg) ** 2, axis=1), np.sum((self.I - mu_fg) ** 2, axis=1)],
            axis=1,
        )
        pairwise = self._pairwise_potts(LAMBDA)
        y, g_val = mincut.solve_energy(self.n, unary, pairwise)
        return y, g_val

    def solve_f_plus_gamma_g(self, x, gamma):
        mu_fg, mu_bg = self._split(x)
        r2 = np.sum((mu_fg - mu_bg) ** 2)
        mu_ij = C_AGREE / (1.0 + r2)
        unary = gamma * np.stack(
            [np.sum((self.I - mu_bg) ** 2, axis=1), np.sum((self.I - mu_fg) ** 2, axis=1)],
            axis=1,
        )
        pairwise = [
            (i, j, mu_ij, gamma * LAMBDA, gamma * LAMBDA, mu_ij) for (i, j) in self.edges
        ]
        y, energy = mincut.solve_energy(self.n, unary, pairwise)
        f_val = mu_ij * sum(1 for (i, j) in self.edges if y[i] == y[j])
        g_val = (energy - f_val) / gamma
        return y, f_val, g_val

    def grad_x_f(self, x, y):
        mu_fg, mu_bg = self._split(x)
        r = mu_fg - mu_bg
        r2 = np.sum(r ** 2)
        S = sum(1 for (i, j) in self.edges if y[i] == y[j])
        coeff = -2.0 * C_AGREE * S / (1.0 + r2) ** 2
        return np.concatenate([coeff * r, -coeff * r])

    def grad_x_g(self, x, y):
        mu_fg, mu_bg = self._split(x)
        fg_mask = y == 1
        bg_mask = y == 0
        g_fg = 2.0 * np.sum(mu_fg - self.I[fg_mask], axis=0) if fg_mask.any() else np.zeros(3)
        g_bg = 2.0 * np.sum(mu_bg - self.I[bg_mask], axis=0) if bg_mask.any() else np.zeros(3)
        return np.concatenate([g_fg, g_bg])

    def f(self, x, y):
        mu_fg, mu_bg = self._split(x)
        r2 = np.sum((mu_fg - mu_bg) ** 2)
        mu_ij = C_AGREE / (1.0 + r2)
        return mu_ij * sum(1 for (i, j) in self.edges if y[i] == y[j])

    def g(self, x, y):
        mu_fg, mu_bg = self._split(x)
        unary = np.where(y == 1, np.sum((self.I - mu_fg) ** 2, axis=1), np.sum((self.I - mu_bg) ** 2, axis=1))
        pairwise = LAMBDA * sum(1 for (i, j) in self.edges if y[i] != y[j])
        return unary.sum() + pairwise


if __name__ == "__main__":
    print("Running mincut self-check...")
    mincut.self_check()
    print("Self-check passed.\n")

    I, scribble_global = load_data()
    edges = build_adjacency(GRID)
    app = SegmentationApp(I, edges, scribble_global)

    print(f"beta = {BETA:.4f}, gamma = {GAMMA:.4f}")
    step_sizes = STEP0 / (1.0 + 0.05 * np.arange(NUM_ITERS))
    log = penalty_alg.run(app, GAMMA, NUM_ITERS, step_sizes)

    y2_final = log["y2_hist"][-1]
    fg_seed_mask = scribble_global == 1
    bg_seed_mask = scribble_global == 0
    consistency = np.mean(
        np.concatenate([y2_final[fg_seed_mask] == 1, y2_final[bg_seed_mask] == 0])
    )
    print(f"Final scribble-consistency (fraction of seeded pixels correctly labeled): {consistency:.3f}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(I.reshape(GRID, GRID, 3))
    axes[0].set_title("Input (downsampled)")
    axes[1].imshow(scribble_global.reshape(GRID, GRID), cmap="viridis")
    axes[1].set_title("Scribble seeds (0=bg,1=fg,2=unmarked)")
    axes[2].imshow(log["y2_hist"][0].reshape(GRID, GRID), cmap="gray")
    axes[2].set_title("y2 at t=0")
    axes[3].imshow(y2_final.reshape(GRID, GRID), cmap="gray")
    axes[3].set_title(f"y2 at t={NUM_ITERS} (final)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "segmentation_masks.png"), dpi=150)

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
    fig.savefig(os.path.join(OUT, "segmentation_curves.png"), dpi=150)

    print(f"\nSaved plots to {OUT}/segmentation_masks.png and {OUT}/segmentation_curves.png")
