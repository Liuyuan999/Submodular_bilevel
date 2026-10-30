"""Matched-node, held-out-start 50-second DREAMPlace MMS comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from macro_bigap import read_json, sha256
from run_mms_gamma_yf import paired_yf_selected_by_yg_loss, trajectory
from run_mms_study import DATA, REFERENCE, RESULTS_ROOT, RESULTS as ORIGINAL_RESULTS


ROOT = Path(__file__).resolve().parent
RESULTS = RESULTS_ROOT / "dreamplace_adaptec1_confirm_50s_v1"
FIGURES = ROOT / "figures"
METHODS = ("bigap", "weighted", "cmaes", "zo", "random")
SEEDS = tuple(range(8, 16))
TIME_BUDGET = 50.0


def run_path(method, seed):
    return RESULTS / f"{method}_seed{seed}.json"


def audit_path(method, seed):
    return run_path(method, seed).with_suffix(".audit.json.gz")


def verify():
    manifest = json.loads((ORIGINAL_RESULTS / "manifest.json").read_text(encoding="utf-8"))
    if sha256(DATA) != manifest["data_sha256"] or sha256(REFERENCE) != manifest["reference_sha256"]:
        raise ValueError("Prepared placement or reference differs from the original study")
    return manifest


def prepare():
    original = verify()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "manifest.json").write_text(json.dumps({
        "data": str(DATA.resolve()), "data_sha256": original["data_sha256"],
        "reference": str(REFERENCE.resolve()),
        "reference_sha256": original["reference_sha256"],
        "methods": METHODS, "seeds": SEEDS,
        "time_budget_seconds": TIME_BUDGET,
        "bigap_gamma": 0.5, "baseline_gamma": 0.01,
        "k_g": 3, "k_f": 3, "rho": 0.05,
        "outer_step": 5e-5, "theta_g": 0.1, "theta_f": 0.1,
        "start_rule": "numpy default_rng(1000 + seed).uniform([-2,-2,-0.5],[2,2,0.5])",
        "selection": "BiGAP paired y_F at minimum observed f(y_g)",
        "baseline_reporting": "original audited best-so-far candidate protocol",
    }, indent=2), encoding="utf-8")


def train(method, seed):
    verify()
    if not (RESULTS / "manifest.json").exists():
        raise FileNotFoundError("Prepare the held-out study before training")
    start = np.random.default_rng(1000 + seed).uniform(
        [-2.0, -2.0, -0.5], [2.0, 2.0, 0.5]
    )
    output = run_path(method, seed)
    gamma = 0.5 if method == "bigap" else 0.01
    command = [sys.executable, str(ROOT / "macro_bigap.py"), "run",
               "--data", str(DATA), "--output", str(output),
               "--backend", "pymaxflow", "--method", method,
               "--reference-mode", "external", "--reference-file", str(REFERENCE),
               "--x-start", *(str(number) for number in start),
               "--seed", str(seed), "--time-budget", str(TIME_BUDGET),
               "--gamma", str(gamma), "--rho", "0.05", "--outer-step", "5e-5",
               "--k-g", "3", "--k-f", "3", "--theta-g", "0.1", "--theta-f", "0.1",
               "--record-interval", "1"]
    if method in {"bigap", "weighted"}:
        command.append("--reuse-trees")
    processor = next((line.partition(":")[2].strip()
                      for line in Path("/proc/cpuinfo").read_text().splitlines()
                      if line.startswith("model name")), "unknown")
    output.with_suffix(".host.json").write_text(json.dumps({
        "node": os.environ.get("EDA_HOST_GROUP", "unrecorded"), "processor": processor,
        "method": method, "seed": seed,
    }, indent=2), encoding="utf-8")
    subprocess.run(command, check=True)


def audit(method, seed):
    verify()
    subprocess.run([sys.executable, str(ROOT / "macro_bigap.py"), "audit",
                    "--run", str(run_path(method, seed)),
                    "--output", str(audit_path(method, seed)),
                    "--backend", "pymaxflow", "--stride", "30"], check=True)


def eligible_audit(path):
    audit = read_json(path)
    run = read_json(audit["run"])
    raw = read_json(run["candidates"])
    if raw["data_sha256"] != run["data_sha256"]:
        raise ValueError("Candidate log and run use different data")
    lower = np.asarray(run["settings"]["x_lower"], dtype=float)
    upper = np.asarray(run["settings"]["x_upper"], dtype=float)
    budget = float(run["settings"]["time_budget"])
    if (lower.shape != (3,) or upper.shape != (3,)
            or not np.all(np.isfinite([lower, upper])) or np.any(lower >= upper)
            or not np.isfinite(budget) or budget <= 0):
        raise ValueError("Invalid reporting domain or budget")
    coordinates = {}
    for event in raw["events"]:
        key = (event["iteration"], event["kind"], event["time"])
        if key in coordinates:
            raise ValueError("Duplicate raw candidate identity")
        coordinates[key] = event["x"]
    candidates = []
    excluded_domain = excluded_budget = 0
    previous_time = -np.inf
    for index, event in enumerate(audit["candidates"]):
        elapsed = float(event["time"])
        if not np.isfinite(elapsed) or elapsed < 0 or elapsed < previous_time:
            raise ValueError("Invalid candidate times")
        previous_time = elapsed
        if index == 0:
            if event["kind"] != "initial_reference" or elapsed != 0:
                raise ValueError("Missing time-zero reference candidate")
            location = run["records"][0]["x"]
        else:
            key = (event["iteration"], event["kind"], event["time"])
            if key not in coordinates:
                raise ValueError("Audited candidate has no matching recorded coordinates")
            location = coordinates[key]
        location = np.asarray(location, dtype=float)
        if location.shape != lower.shape or not np.all(np.isfinite(location)):
            raise ValueError("Invalid candidate coordinates")
        in_domain = np.all(location >= lower - 1e-12) and np.all(location <= upper + 1e-12)
        if index == 0 and not in_domain:
            raise ValueError("Initial reference is outside the reporting domain")
        if elapsed > budget:
            excluded_budget += 1
        elif not in_domain:
            excluded_domain += 1
        else:
            candidates.append(event)
    if not candidates:
        raise ValueError("No eligible candidates")
    return {**audit, "candidates": candidates, "reporting_filter": {
        "input_candidates": len(audit["candidates"]),
        "eligible_candidates": len(candidates),
        "excluded_outside_domain": excluded_domain,
        "excluded_after_budget": excluded_budget,
        "coordinate_tolerance": 1e-12,
    }}


def reported_candidate(path, method, *, audit=None):
    candidates = (eligible_audit(path) if audit is None else audit)["candidates"]
    if method != "bigap":
        return min(candidates, key=lambda event: event["f_plus_gap_plus_violation"])
    selected = candidates[0]
    best_yg_loss = float("inf")
    pairs = {}
    for event in candidates[1:]:
        if event["kind"] in {"g", "F"}:
            pairs.setdefault(event["iteration"], {})[event["kind"]] = event
    for pair in pairs.values():
        if set(pair) == {"g", "F"} and pair["g"]["f_value"] < best_yg_loss:
            best_yg_loss = pair["g"]["f_value"]
            selected = pair["F"]
    return selected


def summarize():
    verify()
    grid = np.linspace(0.0, TIME_BUDGET, 201)
    labels = {"bigap": "BiGAP-PD", "cmaes": "CMA-ES",
              "weighted": "Weighted-PD", "zo": "zeroth-order",
              "random": "random search"}
    colors = {"bigap": "#d62728", "cmaes": "#1f77b4",
              "weighted": "#9467bd", "zo": "#ff7f0e", "random": "#2ca02c"}
    series = {}
    summary = {"metric": "f(y) + LL gap + constraint violation",
               "time_budget_seconds": TIME_BUDGET, "seeds": SEEDS,
               "methods": {}, "paired_wins": {}, "candidate_filters": {},
               "reporting_domain": "Only recorded candidates in X and within budget; no coordinate clipping or score reevaluation",
               "response_selection": "BiGAP paired F selected by minimum g-response f; baselines best audited available g or F response"}
    for method in METHODS:
        paths = [audit_path(method, seed) for seed in SEEDS]
        audits = [eligible_audit(path) for path in paths]
        summary["candidate_filters"][method] = [audit["reporting_filter"] for audit in audits]
        if method == "bigap":
            values = np.stack([paired_yf_selected_by_yg_loss(path, grid, audit=audit)[0]
                               for path, audit in zip(paths, audits)])
        else:
            values = np.stack([trajectory(path, grid, None, audit=audit)[0]
                               for path, audit in zip(paths, audits)])
        series[method] = values
        endpoints = values[:, -1]
        selected = [reported_candidate(path, method, audit=audit)
                    for path, audit in zip(paths, audits)]
        hamming = np.asarray([event["f_value"] for event in selected])
        gaps = np.asarray([event["ll_gap"] for event in selected])
        violations = np.asarray([event["constraint_violation"] for event in selected])
        summary["methods"][method] = {
            "label": labels[method], "mean": float(np.mean(endpoints)),
            "std": float(np.std(endpoints)),
            "median": float(np.median(endpoints)),
            "q25": float(np.quantile(endpoints, 0.25)),
            "q75": float(np.quantile(endpoints, 0.75)),
            "per_seed": endpoints.tolist(),
            "hamming_mean": float(np.mean(hamming)),
            "hamming_std": float(np.std(hamming)),
            "ll_gap_mean": float(np.mean(gaps)),
            "violation_mean": float(np.mean(violations)),
        }
    for method in METHODS[1:]:
        summary["paired_wins"][method] = int(np.count_nonzero(
            series["bigap"][:, -1] < series[method][:, -1]
        ))
    hosts = [json.loads(run_path(method, seed).with_suffix(".host.json").read_text())
             for method in METHODS for seed in SEEDS]
    summary["nodes"] = sorted({host["node"] for host in hosts})
    summary["processors"] = sorted({host["processor"] for host in hosts})
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    FIGURES.mkdir(parents=True, exist_ok=True)
    for statistic in ("median", "mean"):
        for scale in ("linear", "symlog"):
            figure, axis = plt.subplots(figsize=(4.6, 4.6))
            for method in METHODS:
                values = series[method]
                if statistic == "median":
                    center = np.median(values, axis=0)
                    lower, upper = np.percentile(values, [25, 75], axis=0)
                else:
                    center = np.mean(values, axis=0)
                    deviation = np.std(values, axis=0)
                    lower, upper = center - deviation, center + deviation
                lower = np.maximum(lower, 0.0)
                axis.plot(grid, center, linewidth=2.3, color=colors[method],
                          linestyle="-" if method == "bigap" else "--",
                          label=labels[method])
                axis.fill_between(grid, lower, upper, color=colors[method], alpha=0.10)
            axis.set_xlim(0, TIME_BUDGET)
            if scale == "symlog":
                axis.set_yscale("symlog", linthresh=1)
                axis.set_ylim(bottom=0)
            axis.set_xlabel("wall-clock time (s)", fontsize=18)
            axis.set_ylabel("$f(y)$ + LL gap + violation", fontsize=17)
            axis.tick_params(labelsize=13)
            axis.grid(alpha=0.22)
            axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
                        ncol=2, fontsize=10, frameon=False)
            figure.tight_layout(pad=0.5)
            figure.savefig(FIGURES / f"dreamplace_adaptec1_confirm_50s_{statistic}_{scale}.pdf")
            plt.close(figure)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "train", "audit", "summarize"))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "summarize":
        summarize()
    else:
        if args.method is None or args.seed not in SEEDS:
            parser.error("Training and auditing require a method and seed 8--15")
        (train if args.phase == "train" else audit)(args.method, args.seed)


if __name__ == "__main__":
    main()
