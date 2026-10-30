"""Prepare and run the fixed DREAMPlace MMS comparison protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from macro_bigap import Benchmark, sha256


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/adaptec1_dreamplace_gpu.json.gz"
RESULTS_ROOT = Path(os.environ.get("EDA_RESULTS_DIR", ROOT / "results")).resolve()
RESULTS = RESULTS_ROOT / "dreamplace_adaptec1_v1"
REFERENCE = RESULTS / "reference.txt"
METHODS = ("bigap", "weighted", "cmaes", "zo", "random")
SEEDS = 8


def run_path(method, seed):
    return RESULTS / f"{method}_seed{seed}.json"


def prepare_reference():
    benchmark = Benchmark(DATA, backend="pymaxflow")
    labels, value = benchmark.exact(np.array([4.0, 4.0, 0.0]), "g", 0.01)
    minority = min(np.count_nonzero(labels), np.count_nonzero(labels == 0)) / len(labels)
    if minority < 0.02:
        raise ValueError("Reference partition is nearly one-sided")
    RESULTS.mkdir(parents=True, exist_ok=True)
    with REFERENCE.open("w", encoding="utf-8") as output:
        for name, label in zip(benchmark.names, labels):
            output.write(f"{name} {int(label)}\n")
    manifest = {
        "data": str(DATA.resolve()), "data_sha256": sha256(DATA),
        "reference": str(REFERENCE.resolve()), "reference_sha256": sha256(REFERENCE),
        "reference_x": [4.0, 4.0, 0.0], "reference_g_value": value,
        "reference_minority_fraction": minority, "methods": METHODS,
        "seeds": SEEDS, "time_budget_seconds": 300, "gamma": 0.01,
        "bigap_k_g": 3, "bigap_k_f": 3, "weighted_k_f": 3,
        "rho": 0.05, "outer_step": 5e-5,
        "theta_g": 0.1, "theta_f": 0.1, "cma_sigma": 0.4,
        "start_rule": "numpy default_rng(1000 + seed).uniform([-2,-2,-0.5],[2,2,0.5])",
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"reference": str(REFERENCE), "minority_fraction": minority,
                      "g_value": value}, indent=2))


def train(method, seed):
    if method not in METHODS or not 0 <= seed < SEEDS:
        raise ValueError("Invalid method or seed")
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    if sha256(DATA) != manifest["data_sha256"] or sha256(REFERENCE) != manifest["reference_sha256"]:
        raise ValueError("Prepared data or shared reference differs from the manifest")
    start = np.random.default_rng(1000 + seed).uniform(
        [-2.0, -2.0, -0.5], [2.0, 2.0, 0.5]
    )
    command = [sys.executable, str(ROOT / "macro_bigap.py"), "run",
               "--data", str(DATA), "--output", str(run_path(method, seed)),
               "--backend", "pymaxflow", "--method", method,
               "--reference-mode", "external", "--reference-file", str(REFERENCE),
               "--x-start", *(str(number) for number in start),
               "--seed", str(seed), "--time-budget", "300",
               "--gamma", "0.01", "--rho", "0.05", "--outer-step", "5e-5",
               "--k-g", "3", "--k-f", "3", "--theta-g", "0.1", "--theta-f", "0.1",
               "--record-interval", "1"]
    if method in {"bigap", "weighted"}:
        command.append("--reuse-trees")
    subprocess.run(command, check=True)


def audit(method, seed):
    if method not in METHODS or not 0 <= seed < SEEDS:
        raise ValueError("Invalid method or seed")
    subprocess.run([sys.executable, str(ROOT / "macro_bigap.py"), "audit",
                    "--run", str(run_path(method, seed)),
                    "--output", str(run_path(method, seed).with_suffix(".audit.json.gz")),
                    "--backend", "pymaxflow", "--stride", "30"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "train", "audit"))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args()
    if arguments.phase == "prepare":
        prepare_reference()
    elif arguments.method is None or arguments.seed is None:
        parser.error("Training and auditing require --method and --seed")
    elif arguments.phase == "train":
        train(arguments.method, arguments.seed)
    else:
        audit(arguments.method, arguments.seed)


if __name__ == "__main__":
    main()
