# EDA experiments

Self-contained experimental code, prepared benchmark instances, audited results,
and PDF figures for three bilevel EDA studies. No manuscript, scheduler files,
machine-specific setup, or external checkout is needed to inspect the results.

## Contents

```text
supp_eda/
├── constrained/scripts/       ISPD98 ibm01 runner, model, baselines, plotting
├── unconstrained/scripts/     Shared ibm01 graph model and data loader
├── data/                      ibm01 hypergraph, coordinates, reference labels
├── results/
│   ├── final/                 ibm01 comparison results
│   └── ablations/             ibm01 K, gamma, and rho studies
├── figures/                   ibm01 PDF figures and tabular summaries
├── dreamplace/
│   ├── *.py                   MMS preprocessing, BiGAP, baselines, reporting
│   ├── data/                  Prepared adaptec1 instance and provenance
│   ├── results/
│   │   ├── dreamplace_adaptec1_v1/             300 s original comparison
│   │   ├── dreamplace_adaptec1_gamma_yf_v1/     300 s gamma sweep
│   │   ├── dreamplace_adaptec1_confirm_50s_v1/  50 s held-out confirmation
│   │   └── dreamplace_adaptec1_300s_feasible/   Filtered comparison summary
│   └── figures/               50 s and 300 s comparisons, gamma ablations
├── openroad/
│   ├── *.py                   RC model, exact cuts, BiGAP, baselines, reporting
│   ├── data/                  GCD instance and independent verification
│   ├── results/comparison_300s/ 80 audited runs across methods and gamma
│   └── figures/               Comparison figures for each gamma
├── tests/                     Oracle, gradient, update, and reporting tests
├── reproduce.py               Figure regeneration and test entry point
├── check_release.py           File-integrity and upload-size checks
├── DATA_SOURCES.md            Benchmark origins and redistribution scope
└── MANIFEST.sha256            Checksums of the distributed files
```

## Setup and quick checks

Run commands from this directory. Python 3.11 is recommended; benchmark runners
also support Python 3.9. Use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_release.py
python reproduce.py tests
python reproduce.py figures --benchmark dreamplace
python reproduce.py figures --benchmark openroad
python reproduce.py figures --benchmark ibm01
```

`requirements-tested.txt` pins the versions used for release validation.

Tests use small synthetic instances and independent enumeration; two extra
OpenROAD extraction tests run when optional upstream files have been downloaded.
Plotting uses bundled audits, without rerunning optimization. The manifest
verifies shipped files; regenerating PDFs or summaries may change checksums.

For comparable timings, set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
and `MKL_NUM_THREADS=1`, and use the same CPU model and available core per task.
Each seed has its own budget. Independent tasks can run on separate cores.
A GPU is needed only to reproduce the original DREAMPlace preprocessing,
not BiGAP or its baselines.

Fresh runs record CPU specifications but not host names. An optional
`EDA_HOST_GROUP` supplies an anonymous machine label; otherwise the host is
marked `unrecorded`. Matching hardware for new timing studies must be ensured
by the person running them, not inferred from missing host metadata.

## Studies

| Study | Main BiGAP configuration | Reporting |
| --- | --- | --- |
| ISPD98 ibm01 | gamma=0.01, K=3, eta=5e-5, rho=0.01, dual steps=5e-4, M=1 | Best-so-far audited g-response score |
| DREAMPlace MMS adaptec1 | gamma=0.5, K=3, eta=5e-5, rho=0.05, dual steps=0.1, M=1 | Paired F response selected by minimum observed g-response Hamming loss |
| OpenROAD GCD RC sizing | gamma=100, eta=10, rho=0.05, M=1; exact unconstrained cuts | Best-so-far value-gap penalty; BiGAP F response, baseline g response |

See [ibm01](constrained/README.md), [DREAMPlace](dreamplace/README.md), and
[OpenROAD](openroad/README.md) for protocols and reproduction commands.
Scores are benchmark-specific and not comparable across the three problems.

## Included results and storage

All final comparisons and relevant ablations are retained, including unsuccessful
gamma choices. Floating-point objectives and checkpoint times are preserved.
Absolute input paths and host identifiers are removed; CPU models and source/data
hashes are retained. DREAMPlace data serialization and path metadata changed,
but its numerical arrays did not; both file hashes are recorded in
`dreamplace/data/export_manifest.json`.

The new studies include audited trajectories rather than multi-gigabyte
per-iteration binary labels and solver snapshots. DREAMPlace also retains
coordinates needed for domain filtering and output selection. These exports
reproduce plots and statistics but cannot independently re-audit omitted label
vectors. Fresh training writes full auditable logs. Archived source hashes
describe the original runs; `MANIFEST.sha256` describes this reorganized release.

Figures are PDF only. Caches, environments, downloads, generated reruns, and
scheduler logs are excluded by `.gitignore`. No code license is assigned by
this packaging step; choose a license before public distribution and retain
upstream data terms described in `DATA_SOURCES.md`.
