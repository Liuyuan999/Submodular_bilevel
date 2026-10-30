"""Matched-budget RC comparison, separate audit, and aggregate reporting."""

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np

from .model import RCModel
from .run import audit, train
from .search import train_search


CONFIGS = [("bigap", gamma) for gamma in (0.1, 1.0, 10.0, 50.0)] + [
    ("sequential", 1.0), ("alternating", 1.0), ("zero_order", 1.0),
    ("cmaes", 1.0), ("random", 1.0)]
SEEDS = tuple(range(8))
EXTRA_CONFIGS = [("bigap", 100.0)]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
PENALTY_LABEL = r"$f(x,y)+\gamma\,[g(x,y)-g(x,y_g)]$"


def source_hashes():
    paths = list((SOURCE_ROOT / "openroad").glob("*.py"))
    paths += [SOURCE_ROOT / "dreamplace" / "cut_oracle.py"]
    return {str(path.relative_to(SOURCE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt") as stream:
        json.dump(value, stream, separators=(",", ":"))
    temporary.replace(path)


def read_json(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def task_spec(task):
    configurations = CONFIGS + EXTRA_CONFIGS
    if not 0 <= task < len(configurations) * len(SEEDS):
        raise ValueError("Task must be in [0, 79]")
    method, gamma = configurations[task // len(SEEDS)]
    seed = SEEDS[task % len(SEEDS)]
    return method, gamma, seed, f"{method}_gamma{gamma:g}_seed{seed}"


def comparison_response(method, mode):
    if mode not in ("g", "F"):
        raise ValueError("Response mode must be g or F")
    return "F" if mode == "F" and method == "bigap" else "g"


def penalty_values(history, response, gamma):
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Penalty gamma must be finite and positive")
    values = []
    for record in history:
        candidate = record["responses"][response]
        gap = candidate["g"] - record["lower_optimum"]
        if not np.isfinite(gap) or gap < -1e-8:
            raise ValueError("Invalid independently audited lower-level gap")
        value = candidate["f"] + gamma * max(0.0, gap)
        if not np.isfinite(value):
            raise ValueError("Nonfinite penalty value")
        values.append(value)
    return np.asarray(values)


def optimizer_hashes(hashes):
    return {name: digest for name, digest in hashes.items()
            if name not in ("rc_sizing/comparison.py", "rc_sizing/plot.py",
                            "openroad/comparison.py", "openroad/plot.py")}


def hardware():
    return {"hostname": os.environ.get("EDA_HOST_GROUP", "unrecorded"), "python": platform.python_version(),
            "cpu": subprocess.check_output(["lscpu"], text=True),
            "affinity": sorted(os.sched_getaffinity(0)),
            "threads": {name: os.environ.get(name) for name in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def worker(arguments):
    method, gamma, seed, stem = task_spec(arguments.task)
    instance_bytes = arguments.instance.read_bytes()
    digest = hashlib.sha256(instance_bytes).hexdigest()
    model = RCModel(json.loads(instance_bytes))
    run_path = arguments.output / (stem + ".run.json.gz")
    audit_path = arguments.output / (stem + ".audit.json.gz")
    destination = run_path if arguments.phase == "train" else audit_path
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if arguments.phase == "train":
        environment = hardware()
        hashes = source_hashes()
        if method in ("cmaes", "random"):
            result = train_search(model, method, seed, arguments.time_budget, arguments.iterations)
        else:
            result = train(model, method=method, seed=seed, gamma=gamma,
                           iterations=arguments.iterations, time_budget=arguments.time_budget,
                           step=10.0, radius=0.05, batch_size=1, backend="pymaxflow")
        result.update({"hardware": environment, "source_hashes": hashes,
                       "assumptions": model.certificate()})
    else:
        run = read_json(run_path)
        if run["instance_sha256"] != digest or run["source_hashes"] != source_hashes():
            raise ValueError("Instance or implementation changed since training")
        result = audit(model, run, backend="pymaxflow")
        result.update({name: run[name] for name in
                       ("hardware", "source_hashes", "time_budget_seconds", "elapsed_training_seconds")})
        result["algorithm_output_within_budget"] = (
            run["algorithm_output"]["within_time_budget"] if run["algorithm_output"] else None)
        result["stop_reason"] = run.get("stop_reason")
        if result["stop_reason"] is None:
            result["stop_reason"] = "horizon" if run["requested_horizon_completed"] else "time_budget"
    result["instance_sha256"] = digest
    result["configuration"] = stem
    write_json(destination, result)
    print(json.dumps({"file": str(destination), "records": len(result["history"])}), flush=True)


def summarize(directory, include_gamma100=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {}
    instance_hashes, code_hashes, nodes = set(), set(), set()
    reporting_hashes = set()
    configurations = CONFIGS + (EXTRA_CONFIGS if include_gamma100 else [])
    for task in range(len(configurations) * len(SEEDS)):
        method, gamma, seed, stem = task_spec(task)
        path = directory / (stem + ".audit.json.gz")
        result = read_json(path)
        if (result["method"], result["gamma"], result["seed"]) != (method, gamma, seed):
            raise ValueError(f"Configuration does not match filename: {path}")
        if not result["history"]:
            raise ValueError(f"No completed candidates in {path}")
        groups.setdefault((method, gamma), []).append(result)
        instance_hashes.add(result["instance_sha256"])
        code_hashes.add(json.dumps(optimizer_hashes(result["source_hashes"]), sort_keys=True))
        reporting_hashes.add(result["source_hashes"]["rc_sizing/comparison.py"])
        nodes.add(result["hardware"]["hostname"])
    if len(instance_hashes) != 1 or len(code_hashes) != 1:
        raise ValueError("Results use different data or code")
    budgets = {run["time_budget_seconds"] for runs in groups.values() for run in runs}
    if len(budgets) != 1:
        raise ValueError("Results have different training budgets")
    budget = budgets.pop()
    rows, output_rows = [], []
    for (method, gamma), runs in groups.items():
        for response in runs[0]["best_by_response"]:
            selected = []
            for run in runs:
                penalties = penalty_values(run["history"], response, gamma)
                index = int(np.argmin(penalties))
                candidate = run["history"][index]["responses"][response]
                selected.append({"f": candidate["f"], "LL_gap": candidate["LL_gap"],
                                 "penalty": float(penalties[index])})
            row = {"method": method, "gamma": gamma, "response": response, "seeds": len(runs),
                   "main_comparison": response == comparison_response(method, "F")}
            for metric in ("f", "LL_gap", "penalty"):
                values = np.asarray([item[metric] for item in selected])
                row[metric + "_mean"], row[metric + "_std"] = float(values.mean()), float(values.std())
            rows.append(row)
        for run in runs:
            if run["algorithm_output"] is not None:
                metrics = run["algorithm_output"]["history"][0]["responses"]["F"]
                output_rows.append({"method": method, "gamma": gamma, "seed": run["seed"],
                                    "within_budget": run["algorithm_output_within_budget"],
                                    "f": metrics["f"], "LL_gap": metrics["LL_gap"],
                                    "penalty": float(penalty_values(run["algorithm_output"]["history"], "F", gamma)[0])})
    for filename, table in (("endpoints.csv", rows), ("algorithm1_outputs.csv", output_rows)):
        with (directory / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    colors = ["#d62728", "#f28e2b", "#b52e91", "#7f0000", "#9467bd", "#2ca02c", "#17becf", "#1f77b4", "#666666", "#8c564b"]
    names = {"bigap": "BiGAP", "sequential": "Sequential + sizing", "alternating": "Alternating",
             "zero_order": "Zeroth-order", "cmaes": "Nested CMA-ES", "random": "Random search"}
    plot_specs = [(mode, statistic, None) for mode in ("g", "F") for statistic in ("mean", "median")]
    plot_specs += [("F", statistic, gamma) for method, gamma in configurations if method == "bigap"
                   for statistic in ("mean", "median")]
    for response_mode, statistic, comparison_gamma in plot_specs:
            figure, axis = plt.subplots(figsize=(7.5, 5.8))
            for config_index, ((method, gamma), runs) in enumerate(groups.items()):
                if comparison_gamma is not None and method == "bigap" and gamma != comparison_gamma:
                    continue
                response = comparison_response(method, response_mode)
                metric_gamma = gamma if comparison_gamma is None else comparison_gamma
                times = np.unique(np.concatenate((np.linspace(0.0, budget, 1001),
                                                   [run["history"][0]["seconds"] for run in runs])))
                traces = []
                for run in runs:
                    history = run["history"]
                    indices = np.searchsorted([record["seconds"] for record in history], times, side="right") - 1
                    values = np.minimum.accumulate(penalty_values(history, response, metric_gamma))
                    trace = values[np.maximum(indices, 0)].copy()
                    trace[indices < 0] = np.nan
                    traces.append(trace)
                traces = np.asarray(traces)
                valid = np.all(np.isfinite(traces), axis=0)
                samples = traces[:, valid]
                if statistic == "mean":
                    center, spread = samples.mean(axis=0), samples.std(axis=0)
                    low, high = center - spread, center + spread
                else:
                    center = np.median(samples, axis=0)
                    low, high = np.quantile(samples, [0.25, 0.75], axis=0)
                label = (names[method] + f" ($\\gamma={gamma:g}$, $y_\\gamma$)" if method == "bigap" and response == "F"
                         else names[method] + (f" ($\\gamma={gamma:g}$, $y_g$)" if method == "bigap" else " ($y_g$)"))
                axis.step(times[valid], center, where="post", label=label, color=colors[config_index],
                          linewidth=2, linestyle="-" if method == "bigap" else "--")
                axis.fill_between(times[valid], low, high, step="post", color=colors[config_index], alpha=0.10)
            axis.set(xlim=(0, budget), xlabel="training wall-clock time (s)", ylabel=PENALTY_LABEL)
            axis.xaxis.label.set_size(18)
            axis.yaxis.label.set_size(18)
            axis.tick_params(labelsize=14)
            axis.legend(fontsize=10, loc="best", framealpha=0.95)
            axis.grid(alpha=0.2)
            suffix = "" if comparison_gamma is None else f"_gamma{comparison_gamma:g}"
            if comparison_gamma is None and response_mode == "F":
                figure.text(0.5, 0.01, "Different gamma values define different penalty objectives.",
                            ha="center", fontsize=9)
                figure.tight_layout(rect=(0, 0.035, 1, 1))
            else:
                figure.tight_layout()
            figure.savefig(directory / f"comparison_{response_mode}{suffix}_{statistic}.pdf", bbox_inches="tight")
            plt.close(figure)
    (directory / "summary.json").write_text(json.dumps({
        "instance_sha256": instance_hashes.pop(), "nodes": sorted(nodes), "time_budget_seconds": budget,
        "runs": sum(len(runs) for runs in groups.values()), "endpoints": rows, "algorithm1_outputs": output_rows,
        "comparison_response_rule": "only BiGAP uses F (y_gamma); every baseline uses g",
        "reporting_driver_hashes": sorted(reporting_hashes),
        "code_compatibility": "all solver/model/audit files must match; reporting/task driver and plotting versions may differ",
        "reporting": "best-so-far f + gamma*(g-v), recomputed per candidate before selecting the incumbent; Algorithm 1 random F output is separate; mean bands are +/- population std, median bands are IQR",
        "penalty_gap": "g(x,y)-min_z g(x,z); clamp only tiny negative solver roundoff to zero",
        "caution": "compare methods at fixed gamma; different gammas define different objectives; exact g-response baselines have zero gap and hence penalty=f; same reference start; deterministic baselines repeat identically; gamma sweep is not held-out tuning; no legalization or STA claim",
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "audit", "summarize"))
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--instance", type=Path, default=Path("openroad/data/gcd_instance.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-budget", type=float, default=300.0)
    parser.add_argument("--iterations", type=int, default=1000000)
    parser.add_argument("--include-gamma100", action="store_true")
    arguments = parser.parse_args()
    if arguments.phase == "summarize":
        summarize(arguments.output, arguments.include_gamma100)
    else:
        worker(arguments)


if __name__ == "__main__":
    main()
