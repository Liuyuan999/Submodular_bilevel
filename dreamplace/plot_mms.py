"""Plot deferred MMS evaluations and inner-solver audits."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from macro_bigap import read_json


def comparison(args):
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for path in args.audits:
        audit = read_json(path)
        records = audit[args.source]
        times = np.asarray([record["time"] for record in records], dtype=float)
        values = np.asarray([record[args.metric] for record in records], dtype=float)
        if args.best_so_far:
            values = np.minimum.accumulate(values)
        run = read_json(audit["run"])
        label = run["method"]
        if run["method"] in {"bigap", "weighted"}:
            label += f' (K={run["settings"]["k_f"]})'
        axis.step(times, values, where="post", linewidth=2.2, label=label)
    axis.set_xlim(0, args.time_limit)
    axis.set_xlabel("wall-clock optimization time (s)", fontsize=15)
    axis.set_ylabel(args.metric.replace("_", " "), fontsize=15)
    axis.tick_params(labelsize=12)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", fontsize=11)
    figure.tight_layout()
    figure.savefig(args.output)
    plt.close(figure)


def aggregate(args):
    grid = np.linspace(0.0, args.time_limit, 601)
    grouped = {}
    for path in args.audits:
        audit = read_json(path)
        run = read_json(audit["run"])
        method = run["method"]
        seed = run["settings"]["seed"]
        events = audit["candidates"]
        times = np.asarray([event["time"] for event in events], dtype=float)
        values = np.minimum.accumulate(np.asarray(
            [event["f_plus_gap_plus_violation"] for event in events], dtype=float
        ))
        indices = np.maximum(np.searchsorted(times, grid, side="right") - 1, 0)
        if seed in grouped.setdefault(method, {}):
            raise ValueError(f"Duplicate method/seed audit: {method}/{seed}")
        grouped[method][seed] = values[indices]
    if not grouped:
        raise ValueError("No audits were provided")
    colors = {"bigap": "#d62728", "cmaes": "#1f77b4", "weighted": "#9467bd",
              "zo": "#ff7f0e", "random": "#2ca02c"}
    labels = {"bigap": "BiGAP K=3", "cmaes": "CMA-ES", "weighted": "Weighted-PD",
              "zo": "zeroth-order", "random": "random search"}
    figure, axis = plt.subplots(figsize=(9.4, 5.5))
    for method in ("bigap", "cmaes", "weighted", "zo", "random"):
        if method not in grouped:
            continue
        if len(grouped[method]) != args.expected_seeds:
            raise ValueError(f"{method} has {len(grouped[method])} seeds, expected {args.expected_seeds}")
        trajectories = np.stack([grouped[method][seed] for seed in sorted(grouped[method])])
        if args.statistic == "median":
            center = np.median(trajectories, axis=0)
            lower, upper = np.percentile(trajectories, [25, 75], axis=0)
        else:
            center = np.mean(trajectories, axis=0)
            deviation = np.std(trajectories, axis=0)
            lower, upper = center - deviation, center + deviation
        lower = np.maximum(lower, np.finfo(float).tiny) if args.log_scale else lower
        axis.plot(grid, center, color=colors[method], linewidth=2.6, label=labels[method])
        axis.fill_between(grid, lower, upper, color=colors[method], alpha=0.13)
    axis.set_xlim(0.0, args.time_limit)
    if args.log_scale:
        axis.set_yscale("log")
    axis.set_xlabel("wall-clock time (s)", fontsize=18)
    axis.set_ylabel("f(y) + LL gap + constraint violation", fontsize=18)
    axis.tick_params(labelsize=13)
    axis.grid(alpha=0.22)
    axis.legend(loc="upper right", fontsize=13)
    figure.tight_layout()
    figure.savefig(args.output)
    plt.close(figure)


def sweep(args):
    audit = read_json(args.audit)
    rows = audit["rows"]
    iterations = np.asarray([row["K"] for row in rows], dtype=float)
    saddle = np.asarray([row["mean_saddle_gap"] for row in rows], dtype=float)
    value = np.asarray([row["mean_dual_value_gap"] for row in rows], dtype=float)
    bias = np.asarray([row["norm_mean_gradient_difference"] for row in rows], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].plot(iterations, saddle[:, 0], "o-", label="g saddle gap")
    axes[0].plot(iterations, saddle[:, 1], "s-", label="F saddle gap")
    axes[0].plot(iterations, value[:, 0], "o--", label="g dual value gap")
    axes[0].plot(iterations, value[:, 1], "s--", label="F dual value gap")
    axes[0].set_ylabel("inner error", fontsize=14)
    axes[0].legend(fontsize=9)
    axes[1].plot(iterations, bias, "o-", color="tab:red")
    axes[1].set_ylabel("branch-gradient difference", fontsize=14)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("symlog", linthresh=1e-6)
        axis.set_xlabel("dual iterations K", fontsize=14)
        axis.tick_params(labelsize=11)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("comparison")
    compare.add_argument("--audits", nargs="+", type=Path, required=True)
    compare.add_argument("--source", choices=("candidates", "records"), default="candidates")
    compare.add_argument("--metric", choices=("f_plus_gap_plus_violation", "f_value",
                                               "ll_gap", "constraint_violation",
                                               "f_at_g_response", "f_at_F_response",
                                               "penalty_exact", "ll_gap_F"),
                         default="f_plus_gap_plus_violation")
    compare.add_argument("--time-limit", type=float, default=300.0)
    compare.add_argument("--best-so-far", action="store_true")
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(function=comparison)
    study = commands.add_parser("aggregate")
    study.add_argument("--audits", nargs="+", type=Path, required=True)
    study.add_argument("--statistic", choices=("median", "mean"), default="median")
    study.add_argument("--log-scale", action="store_true")
    study.add_argument("--time-limit", type=float, default=300.0)
    study.add_argument("--expected-seeds", type=int, default=8)
    study.add_argument("--output", type=Path, required=True)
    study.set_defaults(function=aggregate)
    k_plot = commands.add_parser("sweep")
    k_plot.add_argument("--audit", type=Path, required=True)
    k_plot.add_argument("--output", type=Path, required=True)
    k_plot.set_defaults(function=sweep)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.function(args)


if __name__ == "__main__":
    main()
