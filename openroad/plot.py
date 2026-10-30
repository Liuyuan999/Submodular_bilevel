"""Render audited RC comparisons in the DREAMPlace appendix figure style."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openroad.comparison import (
    CONFIGS, SEEDS, comparison_response, optimizer_hashes, penalty_values, read_json,
)


ROOT = Path(__file__).resolve().parent
STYLES = {
    "bigap": ("BiGAP", "#d62728"),
    "cmaes": ("Nested CMA-ES", "#1f77b4"),
    "alternating": ("Alternating", "#9467bd"),
    "zero_order": ("Zeroth-order", "#ff7f0e"),
    "random": ("Random search", "#2ca02c"),
    "sequential": ("Sequential + sizing", "#8c564b"),
}


def plot(study, output, gammas):
    configurations = [("bigap", gamma) for gamma in gammas]
    configurations += [config for config in CONFIGS if config[0] != "bigap"]
    groups = {}
    instances, sources, budgets = set(), set(), set()
    for method, gamma in configurations:
        runs = []
        for seed in SEEDS:
            path = study / f"{method}_gamma{gamma:g}_seed{seed}.audit.json.gz"
            run = read_json(path)
            if (run["method"], run["gamma"], run["seed"]) != (method, gamma, seed):
                raise ValueError(f"Configuration mismatch: {path}")
            if not run["history"]:
                raise ValueError(f"Empty audit: {path}")
            instances.add(run["instance_sha256"])
            sources.add(json.dumps(optimizer_hashes(run["source_hashes"]), sort_keys=True))
            budgets.add(run["time_budget_seconds"])
            runs.append(run)
        groups[method, gamma] = runs
    if len(instances) != 1 or len(sources) != 1 or len(budgets) != 1:
        raise ValueError("Comparisons require identical data, optimizer code, and budgets")
    budget = budgets.pop()
    output.mkdir(parents=True, exist_ok=True)
    for gamma in gammas:
        series = {}
        for method in STYLES:
            runs = groups[method, gamma if method == "bigap" else 1.0]
            times = np.unique(np.concatenate((
                np.linspace(0.0, budget, 1001),
                [run["history"][0]["seconds"] for run in runs],
            )))
            traces = []
            for run in runs:
                history = run["history"]
                completed = np.asarray([record["seconds"] for record in history])
                if np.any(np.diff(completed) < 0) or completed[-1] > budget:
                    raise ValueError("Invalid optimization checkpoint times")
                positions = np.searchsorted(completed, times, side="right") - 1
                values = np.minimum.accumulate(penalty_values(
                    history, comparison_response(method, "F"), gamma))
                trace = values[np.maximum(positions, 0)].copy()
                trace[positions < 0] = np.nan
                traces.append(trace)
            traces = np.asarray(traces)
            valid = np.all(np.isfinite(traces), axis=0)
            series[method] = times[valid], traces[:, valid]
        for statistic in ("mean", "median"):
            figure, axis = plt.subplots(figsize=(4.6, 4.6))
            for method, (label, color) in STYLES.items():
                times, values = series[method]
                if statistic == "mean":
                    center = np.mean(values, axis=0)
                    deviation = np.std(values, axis=0)
                    lower, upper = center - deviation, center + deviation
                else:
                    center = np.median(values, axis=0)
                    lower, upper = np.percentile(values, [25, 75], axis=0)
                axis.step(times, center, where="post", linewidth=2.3, color=color,
                          linestyle="-" if method == "bigap" else "--", label=label)
                axis.fill_between(times, lower, upper, step="post", color=color, alpha=0.10)
            axis.set_xlim(0, budget)
            axis.set_xlabel("wall-clock time (s)", fontsize=18)
            axis.set_ylabel(r"$f(x,y)+\gamma[g(x,y)-v(x)]$", fontsize=17)
            axis.tick_params(labelsize=13)
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            axis.grid(alpha=0.22)
            axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
                        ncol=2, fontsize=10, frameon=False)
            figure.tight_layout(pad=0.5)
            path = output / f"rc_gcd_gamma{gamma:g}_{statistic}_linear.pdf"
            figure.savefig(path)
            plt.close(figure)
            print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "results/comparison_300s")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    parser.add_argument("--gamma", type=float, nargs="+", default=[0.1, 1, 10, 50, 100])
    arguments = parser.parse_args()
    plot(arguments.study, arguments.output, arguments.gamma)


if __name__ == "__main__":
    main()
