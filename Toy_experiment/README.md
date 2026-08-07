# Toy experiments: `pen:alg` on real open data

Two small, runnable instances of Algorithm `pen:alg` (the penalty subgradient
method of `Discrete_Bilevel_under_Submodular_Condition/bilevel_discret.tex`,
Sec. "Algorithm and convergence") on real open datasets, following the two
worked examples in the same paper: binary image segmentation (`app:ex:seg`)
and EDA circuit re-partitioning (`app:ex:eda`).

Both applications share the same generic loop (`penalty_alg.py`) and the
same off-the-shelf submodular-minimization oracle (`mincut.py`, a
Kolmogorov–Zabih reduction to min *s*-*t* cut solved with
[PyMaxflow](https://github.com/pmneila/PyMaxflow)'s Boykov–Kolmogorov
implementation) — exactly what the paper means by "both SFM calls are
minimum *s*-*t* cuts" (`app:lem:ising`).

## What's demonstrated

At every iteration `t`, `pen:alg`:
1. solves `y1 = SFM(f(x_t,·) + γ·g(x_t,·))` (the joint solve),
2. solves `y2 = SFM(g(x_t,·))` (the pure lower-level solve),
3. takes the oracle step `x_{t+1} = Π_X(x_t − η_t·u_t)` with
   `u_t = ∇_x f(x_t,y1) + γ(∇_x g(x_t,y1) − ∇_x g(x_t,y2))` (`pen:eq:oracle`).

Each script logs and plots, per iteration:
- `p_γ(x_t) = f(x_t,y1) + γ(g(x_t,y1) − g(x_t,y2))` (since `p_γ = γ(v_γ−v)`),
- the **exactness/tie gap** `g(x_t,y1) − g(x_t,y2)`, which
  `pen:lem:basic`/`pen:cor:awayties` predict should trend to `0` away from
  tie sets — the main *executable* check that the toy exercises the paper's
  theory, not just generic gradient descent,
- `φ(x_t) := f(x_t,y2)`, the running bilevel-objective estimate.

Both runs show the gap collapsing to ~0, with (in the EDA run) a clean
re-spike and re-collapse around the point where the trajectory crosses a
tie set — visually confirming `pen:cor:awayties`.

## Datasets

- **Segmentation**: `image1652.ppm` + `mask1652.ppm` from OpenCV's
  [`opencv_extra`](https://github.com/opencv/opencv_extra) test data
  (`testdata/cv/grabcut/`). `mask1652.ppm` is a sparse scribble/trimap
  (`0`=definite background, `1`=definite foreground, `2`=unmarked), used only
  to (a) initialize the two color models and (b) as a qualitative
  scribble-consistency check at the end — it never enters the `pen:alg` loop.
  This particular photo happens to be stored as grayscale (R=G=B at every
  pixel), which limits how well a global-mean-color model can separate the
  object from the background; the algorithm's mechanics (SFM oracle,
  exactness-gap convergence) are unaffected by this.
- **EDA**: `ibm01.hgr`, the real ISPD98 `ibm01` netlist from
  [`TILOS-AI-Institute/HypergraphPartitioning`](https://github.com/TILOS-AI-Institute/HypergraphPartitioning)
  (`benchmark/ISPD_benchmark/ibm01.hgr`), hMETIS format (14,111 hyperedges,
  12,752 cells). The full netlist is far bigger than a toy, so `run_eda.py`
  BFS-samples a small connected sub-netlist (~50 cells, dropping hyperedges
  with more than 8 pins to avoid one global net swallowing the whole
  neighborhood) and clique-expands its surviving hyperedges into the plain
  pairwise graph `app:ex:eda` is defined over.

Both files are cached under `data/`; re-download with:
```
curl -sf -o data/image1652.ppm https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/cv/grabcut/image1652.ppm
curl -sf -o data/mask1652.ppm  https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/cv/grabcut/mask1652.ppm
curl -sf -o data/ibm01.hgr     https://raw.githubusercontent.com/TILOS-AI-Institute/HypergraphPartitioning/main/benchmark/ISPD_benchmark/ibm01.hgr
```

## Files

- `mincut.py` — shared SFM oracle (Kolmogorov–Zabih reduction + PyMaxflow
  min-cut), with a brute-force self-check run automatically before either
  application's outer loop.
- `penalty_alg.py` — generic implementation of `pen:alg`.
- `run_segmentation.py` — `app:ex:seg` instance: `x=(μ_fg,μ_bg)` color
  models, Potts smoothness in `g`, a color-model-agreement reward in `f`.
- `run_eda.py` — `app:ex:eda` instance: `x=(z_i)` a 2D placement, Gaussian-
  of-distance netlist cut cost in `g`, a same-side reward in `f`.
- `data/` — cached datasets (see above).
- `output/` — generated plots (masks/layout + convergence curves).

## Running

```
pip install PyMaxflow
python3 run_segmentation.py   # -> output/segmentation_masks.png, segmentation_curves.png
python3 run_eda.py            # -> output/eda_layout.png, eda_curves.png
```

Both scripts run `mincut.self_check()` first (asserts the min-cut reduction
matches brute-force enumeration on random instances) before starting the
`pen:alg` loop.
