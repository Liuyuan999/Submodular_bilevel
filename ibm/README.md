# ISPD98 ibm01 constrained partitioning

Run commands from the package root. `ibm/constrained/scripts/benchmark.py` is
the final runner; `ibm/constrained/scripts/constrained_core.py` supplies the
model and baseline routines. The shared module is
`ibm/unconstrained/scripts/unconstrained_benchmark.py`.

The final configuration uses gamma=0.01, K_lambda=K_mu=3, eta=5e-5,
rho=0.01, theta_F=theta_g=5e-4, one query per update, independent persistent
inner states, eight shared starts, and 300 seconds per method/seed.

```bash
mkdir -p generated/ibm01
BENCH_THREADS=1 TIME_BUDGET=300 START_SEEDS="0 1 2 3 4 5 6 7" \
KDUAL=3 KEVAL=10 V3_K_START=3 V3_K_END=3 NWORKERS=8 \
V3_WARMSTART_DUAL=1 V3_WARMSTART_RESPONSE=1 V3_CERTIFIED_AUDIT=1 \
V3_EXCLUDE_AUDIT_TIME=1 V3_DEFER_AUDIT=1 V3_RECORD_INTERVAL_SECONDS=1 \
V3_REPORT_RESPONSE=g THETA_F=5e-4 THETA_G=5e-4 F_THETA_RULE=constant \
GAMMA=0.01 GAMMA_START=0.01 GAMMA_END=0.01 GAMMA_SCHEDULE=constant \
ETA=5e-5 ETA_SCALING=constant UPDATE=fixed RHO=0.01 METHODS=bigap_pd \
OUT=generated/ibm01/bigap_K3.json python ibm/constrained/scripts/benchmark.py
```

Use `NWORKERS=1` for sequential seeds. Original baselines use
`METHODS=cmaes,weighted_pd,zo,random` and KDUAL=KEVAL=10. Additional
literature-inspired comparators are in `scripts/additional_baselines.py`;
their nested MILP and binary/discretized restrictions are identified there.

Retained ablations in `ibm/results/ablations_bundle.tar.gz` cover K={1,3,5,10,20,30},
gamma={0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,20}, and
rho={0,0.001,0.01,0.1,0.15,0.2,0.3,0.5}. Both responses are retained where
recorded; the main comparison uses only the g response.

The plotted score is f(y_g) plus positive certified LL gap and constraint
violation. The LL bound is computed in a deferred audit excluded from training
time. The cut-budget constraint has no structural exact-support guarantee;
the certificate is not treated as an exact binary follower optimum.

`python reproduce.py figures --benchmark ibm01` regenerates four comparison
PDFs in `generated/ibm01_figures/`. The corresponding TSV tables are in
`ibm/figure_summaries.tar.gz`. For custom ablations, set `RESULTS`
(space-separated files), `LABELS` (pipe-separated), `PLOT_RESPONSES=g,F`,
`CENTER_STAT=mean` or `median`, and `PLOT_STEM=generated/ablation`, then run
`ibm/constrained/scripts/plot_certified_suites.py`. Mean shading is mean +/- std;
median shading is the IQR.
