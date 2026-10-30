"""Feasible-candidate DREAMPlace comparisons and gamma sweeps at 300 seconds."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from macro_bigap import read_json
from run_mms_confirm_50s import eligible_audit, reported_candidate
from run_mms_gamma_yf import GAMMAS, audit_path, paired_yf_selected_by_yg_loss, trajectory
from run_mms_study import RESULTS_ROOT, RESULTS as ORIGINAL_RESULTS


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
SUMMARY = RESULTS_ROOT / "dreamplace_adaptec1_300s_feasible/summary.json"
GRID = np.linspace(0.0, 300.0, 601)
SEEDS = tuple(range(8))


def draw(curves, styles, group):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for statistic in ("mean", "median"):
        for scale in ("linear", "symlog"):
            figure, axis = plt.subplots(figsize=(4.6, 4.6))
            for label, values in curves.items():
                if statistic == "mean":
                    center = values.mean(axis=0)
                    deviation = values.std(axis=0)
                    lower, upper = center - deviation, center + deviation
                else:
                    center = np.median(values, axis=0)
                    lower, upper = np.percentile(values, [25, 75], axis=0)
                color, linestyle = styles[label]
                axis.plot(GRID, center, color=color, linestyle=linestyle,
                          linewidth=2.3, label=label)
                axis.fill_between(GRID, np.maximum(lower, 0.0), upper,
                                  color=color, alpha=0.10)
            axis.set_xlim(0, 300)
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
            stem = f"dreamplace_adaptec1_300s_{group}_{statistic}_{scale}"
            figure.savefig(FIGURES / (stem + ".pdf"))
            plt.close(figure)


def main():
    groups, rows = {}, {}
    data_hashes, starts = set(), {}
    configurations = [("bigap", gamma) for gamma in (0.01, *GAMMAS)]
    configurations += [(method, 0.01) for method in ("cmaes", "weighted", "zo", "random")]
    for method, gamma in configurations:
        traces, selected, filters = [], [], []
        for seed in SEEDS:
            path = (audit_path(gamma, seed) if method == "bigap" else
                    ORIGINAL_RESULTS / f"{method}_seed{seed}.audit.json.gz")
            audit = eligible_audit(path)
            run = read_json(audit["run"])
            if (run["settings"]["time_budget"] != 300 or run["settings"]["gamma"] != gamma
                    or run["method"] != method or run["settings"]["seed"] != seed):
                raise ValueError(f"Unexpected run configuration: {path}")
            data_hashes.add(run["data_sha256"])
            starts.setdefault(seed, set()).add(tuple(run["records"][0]["x"]))
            values = (paired_yf_selected_by_yg_loss(path, GRID, audit=audit)[0]
                      if method == "bigap" else trajectory(path, GRID, None, audit=audit)[0])
            candidate = reported_candidate(path, method, audit=audit)
            if not np.isclose(values[-1], candidate["f_plus_gap_plus_violation"], rtol=0, atol=1e-10):
                raise ValueError("Curve and table select different endpoints")
            traces.append(values)
            selected.append(candidate)
            filters.append(audit["reporting_filter"])
        values = np.stack(traces)
        groups[method, gamma] = values
        endpoints = values[:, -1]
        row = {"mean": float(endpoints.mean()), "std": float(endpoints.std()),
               "median": float(np.median(endpoints)), "per_seed": endpoints.tolist(),
               "f_mean": float(np.mean([event["f_value"] for event in selected])),
               "LL_gap_mean": float(np.mean([event["ll_gap"] for event in selected])),
               "violation_mean": float(np.mean([event["constraint_violation"] for event in selected])),
               "candidate_filters": filters, "time_to_target_grid_seconds": 0.5,
               "time_to_target": {}}
        for target in (1000, 100, 10, 1):
            times = [float(GRID[np.flatnonzero(trace <= target)[0]])
                     if np.any(trace <= target) else None for trace in values]
            row["time_to_target"][str(target)] = times
        rows[f"{method}_gamma{gamma:g}"] = row
        print(f"{method} gamma={gamma:g}: mean={row['mean']:.9g}, "
              f"std={row['std']:.9g}, median={row['median']:.9g}; "
              f"time-to-target={row['time_to_target']}", flush=True)
    if len(data_hashes) != 1 or any(len(locations) != 1 for locations in starts.values()):
        raise ValueError("Comparisons require the same dataset and matched starts")
    names = {"bigap": r"BiGAP ($\gamma=0.5$)", "weighted": "Weighted-PD",
             "cmaes": "CMA-ES", "zo": "zeroth-order", "random": "random search"}
    colors = {"bigap": "#d62728", "weighted": "#9467bd", "cmaes": "#1f77b4",
              "zo": "#ff7f0e", "random": "#2ca02c"}
    comparison = {label: groups[method, 0.5 if method == "bigap" else 0.01]
                  for method, label in names.items()}
    comparison_styles = {label: (colors[method], "-" if method == "bigap" else "--")
                         for method, label in names.items()}
    gamma_colors = ("#666666", "#1f77b4", "#17becf", "#d62728", "#ff7f0e", "#9467bd", "#8c564b")
    sweep, sweep_styles = {}, {}
    for gamma, color in zip((0.01, *GAMMAS), gamma_colors):
        label = rf"$\gamma={gamma:g}$"
        sweep[label] = groups["bigap", gamma]
        sweep_styles[label] = (color, "-")
    draw(comparison, comparison_styles, "baselines")
    draw(sweep, sweep_styles, "gamma")
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps({
        "benchmark": "DREAMPlace MMS adaptec1", "seeds": SEEDS,
        "time_budget_seconds": 300, "metric": "f(y) + positive LL gap + implication violation",
        "eligibility": "Original recorded x in X and completion time within budget; no clipping",
        "selection": "BiGAP paired y_F at smallest recorded f(y_g); baselines best audited available response",
        "cohort": "Exploratory seeds 0--7, distinct from held-out 50-second seeds 8--15",
        "hardware": "Heterogeneous CPUs in the exploratory sweep; not a matched-node timing study",
        "instance_sha256": data_hashes.pop(), "results": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(SUMMARY, flush=True)


if __name__ == "__main__":
    main()
