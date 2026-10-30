"""Train RC placement refinements; evaluate LL gaps in a separate audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from openroad.sampling import ball_sample
from .model import RCModel
from .oracle import RCCutOracle


METHODS = ("bigap", "sequential", "alternating", "zero_order")


def train(model, method="bigap", seed=0, iterations=100, gamma=1.0, radius=0.05,
          step=10.0, time_budget=300.0, backend="auto", reuse_trees=True, batch_size=1):
    if method not in METHODS:
        raise ValueError("Unknown method")
    if iterations < 0 or not all(np.isfinite(value) for value in (radius, step, time_budget)):
        raise ValueError("Invalid run parameters")
    if radius < 0 or step <= 0 or time_budget <= 0 or (method == "zero_order" and radius == 0):
        raise ValueError("Invalid radius, stepsize, or time budget")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    if method == "bigap" and (iterations < 1 or radius == 0):
        raise ValueError("Algorithm 1 requires a positive horizon and smoothing radius")
    generator = np.random.default_rng(seed)
    output_generator = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
    positions = model.reference.copy()
    history = []
    completed_inputs = []
    started = perf_counter()
    oracle = RCCutOracle(model, gamma, backend, reuse_trees)
    reporter = RCCutOracle(model, gamma, backend, reuse_trees) if method == "bigap" else oracle

    def total_calls():
        return oracle.calls + (reporter.calls if reporter is not oracle else 0)

    def candidate():
        if method in {"bigap", "zero_order"}:
            lower_labels, penalty_labels = reporter.responses(positions)
            return {"g": lower_labels, "F": penalty_labels}
        if method == "sequential":
            return {"y": np.zeros(model.movable_count, dtype=np.uint8),
                    "g": oracle.solve(positions, "g")}
        return {"g": oracle.solve(positions, "g")}

    def penalty_value(query):
        lower_labels, penalty_labels = oracle.responses(query)
        lower = model.evaluate(query, lower_labels)
        penalty = model.evaluate(query, penalty_labels)
        return penalty["upper"] + gamma * (penalty["lower"] - lower["lower"])

    for iteration in range(iterations + 1):
        candidates = candidate()
        records = {}
        for name, labels in candidates.items():
            measured = model.evaluate(positions, labels)
            records[name] = {"labels": labels.tolist(), "f": measured["upper"], "g": measured["lower"]}
        elapsed = perf_counter() - started
        if elapsed > time_budget:
            break
        history.append({"iteration": iteration, "seconds": elapsed, "positions": positions.tolist(),
                        "responses": records, "training_cut_calls": total_calls()})
        if iteration == iterations:
            break
        if method == "bigap":
            gradient = np.zeros_like(positions)
            for _ in range(batch_size):
                query = positions + radius * ball_sample(generator, positions.shape)
                sampled_gradient, _, _ = oracle.direction(query)
                gradient += sampled_gradient
            gradient /= batch_size
        elif method == "sequential":
            gradient = model.evaluate(positions, candidates["y"], True)["upper_gradient"]
        elif method == "alternating":
            gradient = model.evaluate(positions, candidates["g"], True)["upper_gradient"]
        else:
            direction = generator.normal(size=positions.shape)
            direction /= np.linalg.norm(direction)
            forward = penalty_value(positions + radius * direction)
            backward = penalty_value(positions - radius * direction)
            gradient = positions.size * (forward - backward) * direction / (2 * radius)
        if not np.all(np.isfinite(gradient)):
            raise ArithmeticError("Nonfinite placement gradient")
        positions = model.project(positions - step * gradient)
        if perf_counter() - started >= time_budget:
            break
        completed_inputs.append(len(history) - 1)
    algorithm_output = None
    if method == "bigap" and completed_inputs:
        output_index = int(output_generator.choice(completed_inputs))
        output_positions = np.asarray(history[output_index]["positions"])
        output_labels = oracle.solve(output_positions, "F")
        measured = model.evaluate(output_positions, output_labels)
        output_seconds = perf_counter() - started
        algorithm_output = {
            "history_index": output_index, "iteration": history[output_index]["iteration"],
            "positions": output_positions.tolist(), "seconds": output_seconds,
            "responses": {"F": {"labels": output_labels.tolist(),
                                  "f": measured["upper"], "g": measured["lower"]}},
            "within_time_budget": output_seconds <= time_budget,
        }
    elapsed = perf_counter() - started
    feasible = [(record["responses"]["g"]["f"], index) for index, record in enumerate(history)
                if "g" in record["responses"]]
    selected_index = min(feasible)[1] if feasible else None
    return {"method": method, "seed": seed, "gamma": gamma, "radius": radius, "step": step,
            "algorithm": "Algorithm 1: BiGAP without coupled LL constraints" if method == "bigap" else method,
            "batch_size": batch_size, "completed_updates": len(completed_inputs),
            "algorithm_output": algorithm_output,
            "algorithm_output_rule": "uniform over inputs to completed updates, then exact F solve; excludes terminal iterate" if method == "bigap" else None,
            "requested_horizon_completed": len(completed_inputs) == iterations,
            "requested_iterations": iterations, "time_budget_seconds": time_budget,
            "elapsed_training_seconds": elapsed, "cut_calls": total_calls(),
            "optimizer_cut_calls": oracle.calls,
            "reporting_cut_calls": reporter.calls if reporter is not oracle else 0,
            "backend": oracle.graphs["g"].backend, "reuse_trees": oracle.graphs["g"].reuse_trees,
            "F_cut_scale": oracle.scale, "gradient_scale": "original f + gamma*(g-v), not divided by gamma",
            "history": history, "training_selected_g_index": selected_index,
            "selection_rule": "auxiliary best-so-far diagnostic only: smallest recorded g-response f, not Algorithm 1 output; no optimistic tie certificate",
            "clock": "includes oracle construction, all training/recording solves and objective evaluations; excludes deferred audit",
            "budget_policy": "only completed candidates within budget are recorded; final operation can overrun"}


def physical_metrics(model, positions, labels):
    positions = np.asarray(positions)
    labels = np.asarray(labels, dtype=int)
    selected = np.arange(model.movable_count), labels
    widths, heights = model.width[selected], model.height[selected]
    fixed = model.data.get("physical_fixed_cells", [])
    if fixed:
        positions = np.vstack((positions, [cell["position"] for cell in fixed]))
        widths = np.concatenate((widths, [cell["dimensions"][0] for cell in fixed]))
        heights = np.concatenate((heights, [cell["dimensions"][1] for cell in fixed]))
    left = positions - np.column_stack((widths, heights)) / 2
    right = positions + np.column_stack((widths, heights)) / 2
    horizontal = np.maximum(0.0, np.minimum(right[:, None, 0], right[None, :, 0]) -
                            np.maximum(left[:, None, 0], left[None, :, 0]))
    vertical = np.maximum(0.0, np.minimum(right[:, None, 1], right[None, :, 1]) -
                          np.maximum(left[:, None, 1], left[None, :, 1]))
    intersections = np.triu(horizontal * vertical, 1)
    return {"pairwise_overlap_area_um2": float(intersections.sum()),
            "overlapping_pairs": int(np.count_nonzero(intersections > 1e-12)),
            "total_nonfiller_area_um2": float(np.sum(widths * heights)),
            "outside_die_cells": int(np.count_nonzero(np.any(left < -1e-10, axis=1) |
                                                       np.any(right > model.die + 1e-10, axis=1)))}


def audit(model, run, backend="auto"):
    oracle = RCCutOracle(model, run["gamma"], backend, reuse_trees=False)
    records = []
    best_by_response = {}
    for index, record in enumerate(run["history"]):
        positions = np.asarray(record["positions"])
        if np.any(positions < model.lower_bound - 1e-10) or np.any(positions > model.upper_bound + 1e-10):
            raise ValueError("A recorded output is outside the placement domain")
        exact_labels = oracle.solve(positions, "g")
        exact_value = model.evaluate(positions, exact_labels)["lower"]
        evaluated = {}
        for name, response in record["responses"].items():
            measured = model.evaluate(positions, response["labels"])
            gap = measured["lower"] - exact_value
            if gap < -1e-8:
                raise ArithmeticError("Recorded candidate is below the independently computed LL optimum")
            if not np.isclose(response["f"], measured["upper"], rtol=1e-10, atol=1e-12):
                raise ArithmeticError("Recorded upper objective does not match the model")
            if not np.isclose(response["g"], measured["lower"], rtol=1e-10, atol=1e-12):
                raise ArithmeticError("Recorded lower objective does not match the model")
            gap = max(0.0, gap)
            score = measured["upper"] + gap
            previous = best_by_response.get(name)
            if previous is None or score < previous["score"]:
                best_by_response[name] = {"score": score, "index": index}
            evaluated[name] = {"f": measured["upper"], "g": measured["lower"], "LL_gap": gap,
                               "constraint_gap": 0.0, "score": score,
                               "best_score": best_by_response[name]["score"]}
        records.append({"iteration": record["iteration"], "seconds": record["seconds"],
                        "lower_optimum": exact_value, "responses": evaluated})
    for name, choice in best_by_response.items():
        record = run["history"][choice["index"]]
        labels = record["responses"][name]["labels"]
        choice["physical"] = physical_metrics(model, record["positions"], labels)
        choice["hpwl_um"] = model.physical_hpwl(record["positions"])
    output_audit = None
    if run.get("algorithm_output") is not None:
        output_run = {"method": run["method"], "seed": run["seed"], "gamma": run["gamma"],
                      "history": [run["algorithm_output"]]}
        output_audit = audit(model, output_run, backend)
    return {"method": run["method"], "seed": run["seed"], "gamma": run["gamma"],
            "algorithm_output": output_audit,
            "history": records, "best_by_response": best_by_response,
            "score": "f(x,y) + max(g(x,y)-min_z g(x,z),0); unconstrained binary LL",
            "tie_rule": "cut-selected exact g response; no optimistic tie-breaking claim",
            "audit_cut_calls": oracle.calls + (output_audit["audit_cut_calls"] if output_audit else 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "audit"))
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--method", choices=METHODS, default="bigap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--step", type=float, default=10.0)
    parser.add_argument("--time-budget", type=float, default=300.0)
    parser.add_argument("--backend", choices=("auto", "pymaxflow", "dinic"), default="auto")
    parser.add_argument("--no-reuse", action="store_true")
    arguments = parser.parse_args()
    content = arguments.instance.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    model = RCModel(json.loads(content))
    if arguments.action == "train":
        result = train(model, arguments.method, arguments.seed, arguments.iterations,
                       arguments.gamma, arguments.radius, arguments.step, arguments.time_budget,
                       arguments.backend, not arguments.no_reuse, arguments.batch_size)
    else:
        if arguments.run is None:
            parser.error("audit requires --run")
        run = json.loads(arguments.run.read_text())
        if run["instance_sha256"] != digest:
            parser.error("Run and audit instance hashes differ")
        result = audit(model, run, arguments.backend)
    result["instance_sha256"] = digest
    result["assumptions"] = model.certificate()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(arguments.output), "records": len(result["history"]),
                      "best_by_response": result.get("best_by_response")}, indent=2))


if __name__ == "__main__":
    main()
