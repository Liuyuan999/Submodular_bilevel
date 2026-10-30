# OpenROAD GCD placement refinement and RC sizing

The prepared instance uses the official OpenROAD GCD/Nangate45 regression
design at revision `7678d51077daa1ae82fbe799e096282f9d3dcaad`.
All 295 eligible combinational X1/X2 pairs are included: 590 continuous
coordinates, 851 RC arcs, and 277 nonzero binary pair terms. DEF connections
are authoritative; synthesis Verilog is cross-checked. Source hashes,
extraction decisions, units, and fit diagnostics are in `data/gcd_instance.json`.

Selected cells move within a one-micrometer coordinate box intersected with
die containment. Binary variables choose small/large variants. The upper
objective includes smooth wirelength, displacement, RC timing, area, and
leakage; the lower objective includes timing and resource costs. Both density
weights are zero. This is local continuous refinement with an RC surrogate,
not legalization or a full static-timing flow.

The follower is unconstrained. Nonpositive pair coefficients give submodularity;
A1--A3 hold structurally (A3 is vacuous). Unconstrained Algorithm 1 uses separate
persistent exact cuts for g and g+f/gamma, with no dual multipliers or dual K.
The outer direction retains its original scale and the direct x gradient of f.
A4 is not required by this unconstrained specialization.

## Results and metric

Eight seeds per configuration, 300 seconds each, Intel Xeon Gold 6330:

- BiGAP gamma={0.1,1,10,50,100}, eta=10, rho=0.05, M=1.
- Sequential placement followed by sizing, alternating updates, zeroth-order,
  nested CMA-ES, and random search.
- BiGAP reports F; all baselines report their exact g response.
- Plot f(x,y) + gamma*(g(x,y)-v(x)), best-so-far using the same candidate for
  both terms. Each panel's gamma applies to all methods. Baseline LL gaps are
  zero because their g cuts are exact.
- Training checkpoints and response solves are timed. Deferred audits are
  excluded. The uniform-iterate Algorithm 1 output is stored separately from
  best-so-far diagnostics.

`results_bundle.tar.gz` includes all 80 audit trajectories. Stored legacy
`score` fields are unweighted diagnostics; `comparison.penalty_values`
recomputes the actual gamma-weighted criterion from f, g, and v. Do not plot
the stored unweighted incumbent as the penalty.

```bash
python reproduce.py figures --benchmark openroad
```

This regenerates mean/std and median/IQR comparison PDFs for all five gammas.

## Reproduce

From the package root, using the bundled prepared instance:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m openroad.comparison train --task 72 \
  --output generated/openroad --time-budget 300
python -m openroad.comparison audit --task 72 --output generated/openroad
```

Task 72 is BiGAP gamma=100, seed=0. Tasks 0--31 cover BiGAP gamma=0.1,1,10,50
in eight-seed groups; 32--71 cover sequential, alternating, zeroth-order,
CMA-ES, and random search; 72--79 cover gamma=100. After running all tasks:

```bash
python -m openroad.comparison summarize --output generated/openroad --include-gamma100
python -m openroad.plot --study generated/openroad --output generated/openroad_figures
```

For a short smoke check, train task 72 with `--iterations 2 --time-budget 5`
and a fresh output directory, then audit it. Train and audit with unchanged
source files; hashes reject mixed implementations.

To rebuild and independently verify the benchmark from pinned official data:

```bash
python -m openroad.fetch_data
python -m openroad.benchmark --output generated/gcd_instance.json
python -m openroad.verify --instance generated/gcd_instance.json \
  --output generated/gcd_verification.json
```

The fetcher checks four upstream hashes. Downloaded inputs are ignored by Git.
Verification compares cuts to a binary MILP and checks analytic gradients;
the original verification record is in `data/gcd_verification.json`.
