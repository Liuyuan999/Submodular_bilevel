"""Train the MMS gamma sweep and provide response-selection helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from macro_bigap import read_json, sha256
from run_mms_study import DATA, REFERENCE, RESULTS_ROOT, RESULTS as ORIGINAL_RESULTS


ROOT = Path(__file__).resolve().parent
RESULTS = RESULTS_ROOT / "dreamplace_adaptec1_gamma_yf_v1"
GAMMAS = (0.05, 0.1, 0.5, 1.0, 5.0, 25.0)
SEEDS = 8
TIME_BUDGET = 300.0


def run_path(gamma, seed):
    token = str(gamma).replace(".", "p")
    return RESULTS / f"gamma_{token}" / f"bigap_seed{seed}.json"


def verify_reference():
    manifest = json.loads((ORIGINAL_RESULTS / "manifest.json").read_text(encoding="utf-8"))
    if sha256(DATA) != manifest["data_sha256"] or sha256(REFERENCE) != manifest["reference_sha256"]:
        raise ValueError("The prepared placement or shared reference changed")


def train(gamma, seed):
    verify_reference()
    output = run_path(gamma, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    start = np.random.default_rng(1000 + seed).uniform(
        [-2.0, -2.0, -0.5], [2.0, 2.0, 0.5]
    )
    subprocess.run([
        sys.executable, str(ROOT / "macro_bigap.py"), "run",
        "--data", str(DATA), "--output", str(output),
        "--backend", "pymaxflow", "--method", "bigap",
        "--reference-mode", "external", "--reference-file", str(REFERENCE),
        "--x-start", *(str(number) for number in start),
        "--seed", str(seed), "--time-budget", str(TIME_BUDGET),
        "--gamma", str(gamma), "--rho", "0.05", "--outer-step", "5e-5",
        "--no-normalize-f-oracle",
        "--k-g", "3", "--k-f", "3", "--theta-g", "0.1", "--theta-f", "0.1",
        "--record-interval", "1", "--reuse-trees",
    ], check=True)


def audit(gamma, seed):
    output = run_path(gamma, seed)
    subprocess.run([
        sys.executable, str(ROOT / "macro_bigap.py"), "audit",
        "--run", str(output),
        "--output", str(output.with_suffix(".audit.json.gz")),
        "--backend", "pymaxflow", "--stride", "30",
    ], check=True)


def audit_path(gamma, seed):
    if gamma == 0.01:
        return ORIGINAL_RESULTS / f"bigap_seed{seed}.audit.json.gz"
    return run_path(gamma, seed).with_suffix(".audit.json.gz")


def trajectory(path, grid, response_kind, best_so_far=True, *, audit=None):
    data = read_json(path) if audit is None else audit
    events = [event for event in data["candidates"]
              if event["kind"] == "initial_reference" or
              response_kind is None or event["kind"] == response_kind]
    if not events or events[0]["time"] != 0:
        raise ValueError(f"Missing initial candidate in {path}")
    times = np.asarray([event["time"] for event in events], dtype=float)
    scores = np.asarray(
        [event["f_plus_gap_plus_violation"] for event in events], dtype=float
    )
    if best_so_far:
        scores = np.minimum.accumulate(scores)
    indices = np.searchsorted(times, grid, side="right") - 1
    return scores[indices], data["paper_output"]


def paired_yf_selected_by_yg_loss(path, grid, *, audit=None):
    audit = read_json(path) if audit is None else audit
    initial = audit["candidates"][0]
    if initial["kind"] != "initial_reference" or initial["time"] != 0:
        raise ValueError(f"Missing initial candidate in {path}")
    pairs = {}
    for event in audit["candidates"][1:]:
        if event["kind"] in {"g", "F"}:
            pair = pairs.setdefault(event["iteration"], {})
            if event["kind"] in pair:
                raise ValueError("Paired selection requires one response of each kind per iteration")
            pair[event["kind"]] = event
    best_yg_loss = float("inf")
    times = [0.0]
    values = [initial["f_plus_gap_plus_violation"]]
    for pair in pairs.values():
        if set(pair) != {"g", "F"}:
            raise ValueError("Incomplete paired response in candidate log")
        if pair["g"]["f_value"] < best_yg_loss:
            best_yg_loss = pair["g"]["f_value"]
            times.append(pair["F"]["time"])
            values.append(pair["F"]["f_plus_gap_plus_violation"])
    indices = np.searchsorted(times, grid, side="right") - 1
    return np.asarray(values)[indices], best_yg_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "audit"))
    parser.add_argument("--gamma", type=float, choices=GAMMAS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.gamma is None or args.seed is None or not 0 <= args.seed < SEEDS:
        parser.error("Training and auditing require a listed gamma and seed 0--7")
    (train if args.phase == "train" else audit)(args.gamma, args.seed)


if __name__ == "__main__":
    main()
