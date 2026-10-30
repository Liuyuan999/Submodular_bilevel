# DREAMPlace MMS adaptec1

The prepared instance has 210,967 movable nodes, 63 heuristic macros, 607,328
graph edges, and 8,005 connectivity-based implication constraints. Coordinates
come from official DREAMPlace global placement, not the bundled GraphPlanner
placement. The upper variable is a three-dimensional spatial bias in
[-4,4] x [-4,4] x [-1,1]. The lower objective combines softplus unaries and
nonnegative cut costs with y_i <= y_anchor(i). The upper objective is Hamming
loss to the exact constrained reference at x_ref=(4,4,0).

## Final protocols

- BiGAP: gamma=0.5, K_g=K_F=3, eta=5e-5, rho=0.05, dual steps=0.1, M=1.
  Independent g/F multipliers and graph states persist. Each Lagrangian cut
  is exact; three dual updates do not certify constrained follower optimality.
- Baselines: Weighted-PD, CMA-ES (initial sigma=0.4), zeroth-order, and random
  search, with baseline penalty gamma=0.01.
- 50-second confirmation: held-out seeds 8--15, matched CPU/node, Intel Xeon
  Gold 6330. BiGAP mean score is 63.1 versus CMA-ES 161.8 at 50 seconds.
- 300-second exploratory study: seeds 0--7. The gamma sweep used heterogeneous
  CPUs, so this is not a matched-hardware speed comparison. BiGAP gamma=0.5
  mean/median scores are 0.9066/0.2668; CMA-ES attains zero.
- Gamma ablation: {0.01,0.05,0.1,0.5,1,5,25}. The original sweep explicitly
  disables F-oracle normalization; reproduction preserves that setting.

The common score is f(y) + max(g(x,y)-v(x),0) + sum_i max(c_i(y),0), where
v(x) comes from an independent exact constrained cut. BiGAP reports the paired
F response at the smallest observed g-response Hamming loss (first tie kept).
Baselines use the best audited available g or F response, not a g-only policy.
Only candidates completed within budget and inside the declared box are
eligible; no clipping or score reevaluation repairs invalid candidates.
Auditing occurs after training.

The construction establishes A1--A3 for this model. A4 is not globally verified.
This is a macro-anchored assignment surrogate, not a physical halo or a hard
area-balancing model.

## Plot archived results

From the package root:

```bash
python reproduce.py figures --benchmark dreamplace
```

Four 50-second and eight 300-second comparison/ablation PDFs are regenerated
in `dreamplace/figures/`, together with JSON summaries. Mean plots show
mean +/- std (lower shading clipped at zero); median plots show median/IQR.
Linear and symlog plots are separate.

## Rerun

Keep new runs separate from archived results:

```bash
export EDA_RESULTS_DIR="$PWD/generated/dreamplace"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python dreamplace/run_mms_study.py prepare
python dreamplace/run_mms_confirm_50s.py prepare
for method in bigap weighted cmaes zo random; do
  for seed in 8 9 10 11 12 13 14 15; do
    python dreamplace/run_mms_confirm_50s.py train --method "$method" --seed "$seed"
    python dreamplace/run_mms_confirm_50s.py audit --method "$method" --seed "$seed"
  done
done
python dreamplace/run_mms_confirm_50s.py summarize
```

For 300 seconds, use `run_mms_study.py train/audit --method METHOD --seed SEED`
for seeds 0--7, then `run_mms_gamma_yf.py train/audit --gamma GAMMA --seed SEED`
for {0.05,0.1,0.5,1,5,25}. Finally run `dreamplace/plot_mms_300s_paper.py`.
All use the initialization rule `default_rng(1000+seed)`. Unset
`EDA_RESULTS_DIR` to plot the archive again. Full runs can produce gigabytes
of raw labels; `generated/` is ignored by Git. Compact exports contain audit
metrics and coordinates, not labels, so re-auditing requires fresh training.

## Placement preprocessing

Optimization requires no DREAMPlace installation. To recreate the placement,
obtain the official source and MMS data in `../DATA_SOURCES.md`, build
DREAMPlace following its instructions, and run:

For CUDA 12.5, apply `cuda12_compat.patch` to the pinned source checkout before
building, using `git -C /path/to/DREAMPlace apply /path/to/supp_eda/dreamplace/cuda12_compat.patch`.
This is the CUB include compatibility change used for the recorded GPU placement.

```bash
python dreamplace/dreamplace_placement.py --source /path/to/DREAMPlace \
  --install /path/to/install --aux /path/to/mms/adaptec1/adaptec1.aux \
  --output-dir generated/placement --threads 16 --gpu
python dreamplace/mms_data.py --aux /path/to/mms/adaptec1/adaptec1.aux \
  --placement generated/placement/adaptec1/adaptec1.gp.pl \
  --output generated/adaptec1.json.gz
```

The official MMS configuration has legalization and detailed placement disabled.
GPU coordinates may not reproduce bitwise; the bundled prepared data fixes
the exact numerical input for the bilevel experiments.
