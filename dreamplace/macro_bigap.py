"""Run and audit the macro-anchored MMS BiGAP experiment."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from cut_oracle import CutOracle, WarmBlockCutOracle


def read_json(path):
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        results_root = os.environ.get("EDA_RESULTS_DIR")
        candidates = []
        if results_root:
            relative = path
            if relative.parts and relative.parts[0] == "results":
                relative = Path(*relative.parts[1:])
            candidates.append(Path(results_root) / relative)
        candidates.append(Path(__file__).resolve().parent / path)
        path = next((candidate for candidate in candidates if candidate.exists()), path)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as destination:
        json.dump(payload, destination, separators=(",", ":"))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Benchmark:
    def __init__(self, path, backend="auto", reuse_trees=False, sfm_blocks=1,
                 normalize_f_oracle=True):
        self.path = Path(path).resolve()
        data = read_json(self.path)
        if data.get("schema") not in {"macro_anchored_mms_v1", "macro_anchored_ispd2005_v1"}:
            raise ValueError("Unsupported prepared-data schema")
        self.names = data["names"]
        self.node_count = len(self.names)
        loss_indices = np.asarray(data.get("loss_indices", list(range(self.node_count))), dtype=np.int64)
        if np.any(loss_indices < 0) or np.any(loss_indices >= self.node_count) or len(np.unique(loss_indices)) != len(loss_indices):
            raise ValueError("Invalid loss indices")
        self.loss_mask = np.zeros(self.node_count, dtype=np.float64)
        self.loss_mask[loss_indices] = 1.0
        self.coordinates = np.asarray(data["coordinates"], dtype=np.float64)
        edges = np.asarray(data["edges"], dtype=np.float64).reshape(-1, 3)
        anchors = np.asarray(data["anchors"], dtype=np.int64).reshape(-1, 2)
        self.edge_first = edges[:, 0].astype(np.int64)
        self.edge_second = edges[:, 1].astype(np.int64)
        self.edge_weight = edges[:, 2]
        self.anchor_cell = anchors[:, 0]
        self.anchor_macro = anchors[:, 1]
        self.macro_indices = np.asarray(data["macro_indices"], dtype=np.int64)
        if self.coordinates.shape != (self.node_count, 2):
            raise ValueError("Coordinate shape does not match node count")
        if not np.all(np.isfinite(self.coordinates)) or not np.all(np.isfinite(self.edge_weight)):
            raise ValueError("Coordinates and edge weights must be finite")
        if np.any(self.edge_weight < 0):
            raise ValueError("Negative pairwise weight breaks submodularity")
        for indices in (self.edge_first, self.edge_second, self.anchor_cell,
                        self.anchor_macro, self.macro_indices):
            if np.any(indices < 0) or np.any(indices >= self.node_count):
                raise ValueError("Prepared data contains an invalid node index")
        if np.any(self.edge_first == self.edge_second):
            raise ValueError("Prepared data contains a self edge")
        if not set(self.anchor_macro.tolist()).issubset(set(self.macro_indices.tolist())):
            raise ValueError("An anchor target is not a movable macro")
        if set(self.anchor_cell.tolist()) & set(self.macro_indices.tolist()):
            raise ValueError("An anchor child is a macro")
        if len(self.anchor_cell) == 0:
            raise ValueError("The constrained experiment requires at least one anchor")
        self.features = np.column_stack((self.coordinates, -np.ones(self.node_count)))
        self.pair_total = float(np.sum(self.edge_weight))
        self.oracle = CutOracle(self.node_count, self.edge_first, self.edge_second,
                                self.edge_weight, self.anchor_cell, self.anchor_macro,
                                backend=backend)
        self.reuse_trees = bool(reuse_trees and self.oracle.backend == "pymaxflow")
        self.normalize_f_oracle = bool(normalize_f_oracle)
        self.sfm_blocks = int(sfm_blocks)
        if not 1 <= self.sfm_blocks <= self.node_count:
            raise ValueError("SFM block count must lie between one and the node count")
        self.block_assignment = None
        if self.sfm_blocks > 1:
            spatial_order = np.argsort(self.coordinates[:, 0], kind="stable")
            self.block_assignment = np.empty(self.node_count, dtype=np.int64)
            self.block_assignment[spatial_order] = (
                np.arange(self.node_count) * self.sfm_blocks // self.node_count)
            self.block_assignment[self.anchor_cell] = self.block_assignment[self.anchor_macro]
        self.dual_oracles = {}
        self.reference = None
        self.f_zero = None
        self.f_one = None

    def set_reference(self, labels):
        labels = np.asarray(labels, dtype=np.uint8)
        if labels.shape != (self.node_count,) or not np.all(labels <= 1):
            raise ValueError("Reference must have one binary label per movable node")
        self.reference = labels.copy()
        self.f_zero = labels.astype(np.float64) * self.loss_mask
        self.f_one = (1.0 - labels.astype(np.float64)) * self.loss_mask

    def unary(self, location):
        score = self.features @ np.asarray(location, dtype=np.float64)
        return np.logaddexp(0.0, score), np.logaddexp(0.0, -score)

    def h_unary(self, location, which, gamma):
        zero, one = self.unary(location)
        if which == "g":
            return zero, one, 1.0
        if self.reference is None:
            raise ValueError("Set the reference before requesting the F oracle")
        return self.f_zero + gamma * zero, self.f_one + gamma * one, gamma

    def dual_unary(self, location, which, gamma):
        if which == "F" and self.normalize_f_oracle and gamma > 1.0:
            if self.reference is None:
                raise ValueError("Set the reference before requesting the F oracle")
            zero, one = self.unary(location)
            return zero + self.f_zero / gamma, one + self.f_one / gamma, 1.0, gamma
        zero, one, pair_scale = self.h_unary(location, which, gamma)
        return zero, one, pair_scale, 1.0

    def h_value(self, labels, unary_zero, unary_one, pair_scale):
        unary = float(np.sum(np.where(labels != 0, unary_one, unary_zero)))
        crossing = labels[self.edge_first] != labels[self.edge_second]
        return unary + pair_scale * float(np.sum(self.edge_weight[crossing]))

    def f_value(self, labels):
        return int(np.count_nonzero((labels != self.reference) & (self.loss_mask != 0)))

    def g_value(self, location, labels):
        zero, one = self.unary(location)
        return self.h_value(labels, zero, one, 1.0)

    def constraint(self, labels):
        return labels[self.anchor_cell].astype(np.float64) - labels[self.anchor_macro]

    def gradient_g(self, location, labels):
        score = self.features @ np.asarray(location, dtype=np.float64)
        sigmoid = np.exp(-np.logaddexp(0.0, -score))
        return self.features.T @ (sigmoid - labels)

    def exact(self, location, which, gamma):
        zero, one, pair_scale, objective_scale = self.dual_unary(location, which, gamma)
        labels = self.oracle.solve(zero, one, pair_scale=pair_scale, hard_constraints=True)
        return labels, objective_scale * self.h_value(labels, zero, one, pair_scale)

    def dual_response(self, unary_zero, unary_one, pair_scale, multiplier, which):
        shifted_one = unary_one.copy()
        np.add.at(shifted_one, self.anchor_cell, multiplier)
        np.add.at(shifted_one, self.anchor_macro, -multiplier)
        if which not in self.dual_oracles:
            if self.sfm_blocks == 1:
                self.dual_oracles[which] = CutOracle(
                    self.node_count, self.edge_first, self.edge_second, self.edge_weight,
                    self.anchor_cell, self.anchor_macro, backend=self.oracle.backend,
                    reuse_trees=self.reuse_trees
                )
            else:
                self.dual_oracles[which] = WarmBlockCutOracle(
                    self.node_count, self.edge_first, self.edge_second, self.edge_weight,
                    self.block_assignment, backend=self.oracle.backend,
                    reuse_trees=self.reuse_trees
                )
        return self.dual_oracles[which].solve(unary_zero, shifted_one, pair_scale=pair_scale)

    @property
    def cut_calls(self):
        return self.oracle.calls + sum(oracle.calls for oracle in self.dual_oracles.values())

    def uniform_ranges(self, lower, upper, radius, gamma):
        coordinate_bounds = np.max(np.abs(self.coordinates), axis=0)
        largest = np.maximum(np.abs(lower), np.abs(upper)) + radius
        score_bound = float(coordinate_bounds @ largest[:2] + largest[2])
        range_g = self.node_count * score_bound + self.pair_total
        range_f = float(self.loss_mask.sum()) + gamma * range_g
        return range_g, range_f


class DualState:
    def __init__(self, constraint_count):
        self.multiplier = np.zeros(constraint_count, dtype=np.float64)


def dual_chain(benchmark, location, which, gamma, iterations, step, state,
               saddle_radius=None):
    if iterations < 1 or step < 0:
        raise ValueError("Dual iterations must be positive and step nonnegative")
    zero, one, pair_scale, objective_scale = benchmark.dual_unary(location, which, gamma)
    multiplier = state.multiplier.copy()
    sum_multiplier = np.zeros_like(multiplier)
    sum_constraints = np.zeros_like(multiplier)
    sum_gradient = np.zeros(3, dtype=np.float64)
    sum_objective = 0.0
    exact_sfm = getattr(benchmark, "sfm_blocks", 1) == 1
    best_dual = -math.inf if exact_sfm else None
    final_labels = None
    for _ in range(iterations):
        labels = benchmark.dual_response(zero, one, pair_scale, multiplier, which)
        constraints = benchmark.constraint(labels)
        objective = benchmark.h_value(labels, zero, one, pair_scale)
        dual_value = objective_scale * (objective + float(multiplier @ constraints))
        if exact_sfm:
            best_dual = max(best_dual, dual_value)
        sum_multiplier += multiplier
        sum_constraints += constraints
        sum_objective += objective_scale * objective
        sum_gradient += objective_scale * pair_scale * benchmark.gradient_g(location, labels)
        multiplier = np.maximum(0.0, multiplier + step * constraints)
        final_labels = labels
    state.multiplier = multiplier
    average_multiplier = sum_multiplier / iterations
    average_constraints = sum_constraints / iterations
    average_objective = sum_objective / iterations
    result = {
        "gradient": sum_gradient / iterations,
        "best_dual": best_dual,
        "last_labels": final_labels,
        "average_multiplier": objective_scale * average_multiplier,
        "average_constraints": average_constraints,
        "average_objective": average_objective,
    }
    if saddle_radius is not None and not exact_sfm:
        raise ValueError("Saddle-gap certification requires exact SFM calls")
    if saddle_radius is not None:
        response = benchmark.dual_response(zero, one, pair_scale, average_multiplier, which)
        lower_value = objective_scale * (
            benchmark.h_value(response, zero, one, pair_scale)
            + float(average_multiplier @ benchmark.constraint(response)))
        upper_value = average_objective + saddle_radius * float(np.maximum(average_constraints, 0).sum())
        result["saddle_gap"] = max(0.0, upper_value - lower_value)
    return result


def reference_from_text(path, names):
    lookup = {name: index for index, name in enumerate(names)}
    labels = np.full(len(names), 2, dtype=np.uint8)
    with open(path, encoding="utf-8") as source:
        for raw_line in source:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            name, value = line.split()
            if name not in lookup or value not in {"0", "1"}:
                raise ValueError(f"Invalid reference entry: {line}")
            index = lookup[name]
            if labels[index] != 2:
                raise ValueError(f"Duplicate reference node: {name}")
            labels[index] = int(value)
    if np.any(labels == 2):
        raise ValueError("External reference does not label every movable node")
    return labels


def prepare_reference(benchmark, args):
    if args.reference_mode == "recoverable":
        location = np.asarray(args.x_reference, dtype=np.float64)
        labels, value = benchmark.exact(location, "g", args.gamma)
        minority = min(np.count_nonzero(labels), np.count_nonzero(labels == 0)) / benchmark.node_count
        if minority < args.min_class_fraction:
            raise ValueError("Recoverable reference is nearly one-sided; change --x-reference")
        source = {"mode": "recoverable", "x_reference": location.tolist(),
                  "g_optimum_at_reference": value}
    else:
        if args.reference_file is None:
            raise ValueError("--reference-file is required in external mode")
        labels = reference_from_text(args.reference_file, benchmark.names)
        source = {"mode": "external", "source": str(Path(args.reference_file).resolve()),
                  "source_sha256": sha256(args.reference_file)}
    benchmark.set_reference(labels)
    return source


def ball_sample(rng, dimension):
    direction = rng.normal(size=dimension)
    direction /= np.linalg.norm(direction)
    return direction * rng.random() ** (1.0 / dimension)


def dual_steps(benchmark, args, lower, upper):
    if args.dual_step_rule == "fixed":
        return args.theta_g, args.theta_f
    range_g, range_f = benchmark.uniform_ranges(lower, upper, args.rho, args.gamma)
    if benchmark.normalize_f_oracle and args.gamma > 1.0:
        range_f /= args.gamma
    return range_g / math.sqrt(args.k_g), range_f / math.sqrt(args.k_f)


def exact_penalty(benchmark, location, gamma, return_responses=False):
    response_g, value_g = benchmark.exact(location, "g", gamma)
    response_f, value_f = benchmark.exact(location, "F", gamma)
    value = value_f - gamma * value_g
    return (value, response_g, response_f) if return_responses else value


def candidate_event(location, labels, kind, iteration, elapsed):
    packed = np.packbits(labels)
    return {"time": elapsed, "iteration": iteration, "x": np.asarray(location).tolist(),
            "kind": kind, "labels": base64.b64encode(packed.tobytes()).decode("ascii")}


def run(args):
    if Path(args.output).exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    benchmark = Benchmark(args.data, backend=args.backend, reuse_trees=args.reuse_trees,
                          sfm_blocks=args.sfm_blocks,
                          normalize_f_oracle=args.normalize_f_oracle)
    lower = np.asarray(args.x_lower, dtype=np.float64)
    upper = np.asarray(args.x_upper, dtype=np.float64)
    location = np.asarray(args.x_start, dtype=np.float64)
    if np.any(lower >= upper) or np.any(location < lower) or np.any(location > upper):
        raise ValueError("Invalid leader box or starting point")
    if args.reference_mode == "recoverable":
        reference_location = np.asarray(args.x_reference, dtype=np.float64)
        if np.any(reference_location < lower) or np.any(reference_location > upper):
            raise ValueError("Recoverable reference location must lie inside the leader box")
    reference_start = time.perf_counter()
    source = prepare_reference(benchmark, args)
    reference_path = Path(str(args.output) + ".reference.json.gz")
    write_json(reference_path, {"schema": "binary_reference_v1", "labels": benchmark.reference.tolist(),
                                "source": source, "data_sha256": sha256(args.data)})
    reference_setup_seconds = time.perf_counter() - reference_start
    reference_setup_cut_calls = benchmark.oracle.calls
    theta_g, theta_f = dual_steps(benchmark, args, lower, upper)
    benchmark.oracle.calls = 0
    state_g = DualState(len(benchmark.anchor_cell))
    state_f = DualState(len(benchmark.anchor_cell))
    rng = np.random.default_rng(args.seed)
    output_rng = np.random.default_rng(args.seed + 1)
    records = [{"time": 0.0, "iteration": 0, "x": location.tolist()}]
    candidates = []
    first_time = time.perf_counter()
    if args.method == "cmaes":
        import cma
        strategy = cma.CMAEvolutionStrategy(
            location.tolist(), args.cma_sigma,
            {"bounds": [lower.tolist(), upper.tolist()], "seed": args.seed + 1,
             "verbose": -9, "tolflatfitness": 10 ** 9, "tolfun": 0,
             "tolfunhist": 0, "tolx": 0}
        )
        population = []
        population_scores = []
    last_record = 0.0
    iteration = 0
    best_training_value = math.inf
    output_location = location.copy()
    while iteration < args.max_iterations and time.perf_counter() - first_time < args.time_budget:
        if args.method == "bigap" and output_rng.integers(iteration + 1) == 0:
            output_location = location.copy()
        if args.method in ("random", "cmaes"):
            if args.method == "cmaes":
                if not population:
                    population = strategy.ask()
                    population_scores = []
                candidate = np.asarray(population[len(population_scores)], dtype=np.float64)
            else:
                candidate = location if iteration == 0 else rng.uniform(lower, upper)
            candidate_value, response_g, response_f = exact_penalty(
                benchmark, candidate, args.gamma, return_responses=True
            )
            event_time = time.perf_counter() - first_time
            candidates.append(candidate_event(candidate, response_g, "g", iteration, event_time))
            candidates.append(candidate_event(candidate, response_f, "F", iteration, event_time))
            if candidate_value < best_training_value:
                best_training_value = candidate_value
                location = candidate.copy()
            if args.method == "cmaes":
                population_scores.append(candidate_value)
                if len(population_scores) == len(population):
                    strategy.tell(population, population_scores)
                    population = []
        else:
            direction = np.zeros(3)
            for _ in range(args.batch_size):
                if args.method == "zo":
                    sphere = rng.normal(size=3)
                    sphere /= np.linalg.norm(sphere)
                    plus_location = location + args.rho * sphere
                    minus_location = location - args.rho * sphere
                    plus, plus_g, plus_f = exact_penalty(
                        benchmark, plus_location, args.gamma, return_responses=True
                    )
                    plus_time = time.perf_counter() - first_time
                    candidates.append(candidate_event(plus_location, plus_g, "g", iteration, plus_time))
                    candidates.append(candidate_event(plus_location, plus_f, "F", iteration, plus_time))
                    minus, minus_g, minus_f = exact_penalty(
                        benchmark, minus_location, args.gamma, return_responses=True
                    )
                    minus_time = time.perf_counter() - first_time
                    candidates.append(candidate_event(minus_location, minus_g, "g", iteration, minus_time))
                    candidates.append(candidate_event(minus_location, minus_f, "F", iteration, minus_time))
                    direction += 3.0 * (plus - minus) * sphere / (2.0 * args.rho)
                else:
                    query = location + args.rho * ball_sample(rng, 3)
                    estimate_f = dual_chain(benchmark, query, "F", args.gamma,
                                            args.k_f, theta_f, state_f)
                    if args.method == "bigap":
                        estimate_g = dual_chain(benchmark, query, "g", args.gamma,
                                                args.k_g, theta_g, state_g)
                        direction += estimate_f["gradient"] - args.gamma * estimate_g["gradient"]
                        candidates.append(candidate_event(
                            query, estimate_g["last_labels"], "g", iteration,
                            time.perf_counter() - first_time
                        ))
                    else:
                        direction += estimate_f["gradient"]
                    candidates.append(candidate_event(
                        query, estimate_f["last_labels"], "F", iteration,
                        time.perf_counter() - first_time
                    ))
            direction /= args.batch_size
            location = np.clip(location - args.outer_step * direction, lower, upper)
        iteration += 1
        elapsed = time.perf_counter() - first_time
        if elapsed <= args.time_budget and elapsed - last_record >= args.record_interval:
            records.append({"time": elapsed, "iteration": iteration, "x": location.tolist()})
            last_record = elapsed
    elapsed = time.perf_counter() - first_time
    if elapsed <= args.time_budget and records[-1]["iteration"] != iteration:
        records.append({"time": elapsed, "iteration": iteration, "x": location.tolist()})
    training_cut_calls = benchmark.cut_calls
    candidates_path = Path(str(args.output) + ".candidates.json.gz")
    write_json(candidates_path, {"schema": "macro_anchored_candidates_v1",
                                 "node_count": benchmark.node_count,
                                 "data_sha256": sha256(args.data), "events": candidates})
    paper_output = None
    if args.method == "bigap" and iteration:
        recovery_start = time.perf_counter()
        recovery = dual_chain(benchmark, output_location, "F", args.gamma,
                              args.k_f, theta_f, state_f)
        output_labels = recovery["last_labels"]
        output_path = Path(str(args.output) + ".paper_output.json.gz")
        write_json(output_path, {"schema": "macro_anchored_paper_output_v1",
                                 "x": output_location.tolist(),
                                 "labels": output_labels.tolist(),
                                 "f_value": benchmark.f_value(output_labels),
                                 "max_constraint_violation": float(np.maximum(
                                     benchmark.constraint(output_labels), 0).max(initial=0)),
                                 "data_sha256": sha256(args.data)})
        paper_output = {"path": str(output_path.resolve()),
                        "recovery_seconds": time.perf_counter() - recovery_start,
                        "recovery_cut_calls": benchmark.cut_calls - training_cut_calls}
    result = {
        "schema": "macro_anchored_run_v1",
        "method": args.method,
        "data": str(benchmark.path),
        "data_sha256": sha256(args.data),
        "reference": str(reference_path.resolve()),
        "reference_source": source,
        "reference_setup_seconds": reference_setup_seconds,
        "reference_setup_cut_calls": reference_setup_cut_calls,
        "backend": benchmark.oracle.backend,
        "reuse_trees": benchmark.reuse_trees,
        "settings": {"gamma": args.gamma, "rho": args.rho, "k_g": args.k_g, "k_f": args.k_f,
                     "sfm_blocks": args.sfm_blocks,
                     "normalize_f_oracle": args.normalize_f_oracle,
                     "batch_size": args.batch_size, "outer_step": args.outer_step,
                     "cma_sigma": args.cma_sigma,
                     "theta_g": theta_g, "theta_f": theta_f,
                     "dual_step_rule": args.dual_step_rule,
                     "x_lower": lower.tolist(), "x_upper": upper.tolist(),
                     "time_budget": args.time_budget, "seed": args.seed},
        "records": records,
        "candidates": str(candidates_path.resolve()),
        "iterations": iteration,
        "cut_calls_in_training": training_cut_calls,
        "elapsed_including_last_iteration": elapsed,
        "paper_output": paper_output,
        "elapsed_including_recovery": time.perf_counter() - first_time,
    }
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "iterations": iteration,
                      "cut_calls": benchmark.cut_calls, "records": len(records)}, indent=2))


def benchmark_for_run(result, backend, reuse_trees=False):
    data_path = Path(result["data"])
    if sha256(data_path) != result["data_sha256"]:
        raise ValueError("Prepared data changed since the run")
    benchmark = Benchmark(data_path, backend=backend, reuse_trees=reuse_trees,
                          normalize_f_oracle=result["settings"].get("normalize_f_oracle", False))
    reference = read_json(result["reference"])
    if reference["data_sha256"] != result["data_sha256"]:
        raise ValueError("Reference belongs to different prepared data")
    benchmark.set_reference(reference["labels"])
    return benchmark


def candidate_metrics(benchmark, location, labels, constrained_minimum):
    ll_gap = benchmark.g_value(location, labels) - constrained_minimum
    violation = float(np.maximum(benchmark.constraint(labels), 0).sum())
    loss = benchmark.f_value(labels)
    positive_gap = max(0.0, ll_gap)
    return {"f_value": loss, "ll_gap": ll_gap, "ll_gap_positive": positive_gap,
            "constraint_violation": violation,
            "f_plus_gap_plus_violation": loss + positive_gap + violation}


def audit(args):
    audit_start = time.perf_counter()
    result = read_json(args.run)
    if result.get("release_export"):
        raise ValueError("This compact export has audited metrics but no labels; rerun training to audit anew")
    benchmark = benchmark_for_run(result, args.backend)
    gamma = result["settings"]["gamma"]
    records = result["records"][::args.stride]
    if records[-1] != result["records"][-1]:
        records.append(result["records"][-1])
    audited = []
    for record in records:
        location = np.asarray(record["x"], dtype=np.float64)
        response_g, value_g = benchmark.exact(location, "g", gamma)
        response_f, value_f = benchmark.exact(location, "F", gamma)
        ll_gap_f = benchmark.g_value(location, response_f) - value_g
        audited.append({"time": record["time"], "iteration": record["iteration"],
                        "x": record["x"], "g_optimum": value_g,
                        "penalty_exact": value_f - gamma * value_g,
                        "f_at_g_response": benchmark.f_value(response_g),
                        "f_at_F_response": benchmark.f_value(response_f),
                        "ll_gap_F": ll_gap_f})
    candidate_log = read_json(result["candidates"])
    if candidate_log["data_sha256"] != result["data_sha256"] or candidate_log["node_count"] != benchmark.node_count:
        raise ValueError("Candidate log belongs to different prepared data")
    first_location = np.asarray(result["records"][0]["x"], dtype=np.float64)
    _, first_value = benchmark.exact(first_location, "g", gamma)
    initial_labels = benchmark.reference
    candidates = [{"time": 0.0, "iteration": 0, "kind": "initial_reference",
                   **candidate_metrics(benchmark, first_location, initial_labels, first_value)}]
    cached_location = None
    cached_value = None
    for event in candidate_log["events"]:
        if event["time"] > result["settings"]["time_budget"]:
            continue
        location = np.asarray(event["x"], dtype=np.float64)
        packed = np.frombuffer(base64.b64decode(event["labels"]), dtype=np.uint8)
        labels = np.unpackbits(packed, count=benchmark.node_count)
        if cached_location is None or not np.array_equal(location, cached_location):
            _, cached_value = benchmark.exact(location, "g", gamma)
            cached_location = location.copy()
        candidates.append({"time": event["time"], "iteration": event["iteration"],
                           "kind": event["kind"],
                           **candidate_metrics(benchmark, location, labels, cached_value)})
    recovered = None
    if result.get("paper_output") is not None:
        output = read_json(result["paper_output"]["path"])
        if output["data_sha256"] != result["data_sha256"]:
            raise ValueError("Recovered response belongs to different prepared data")
        location = np.asarray(output["x"], dtype=np.float64)
        labels = np.asarray(output["labels"], dtype=np.uint8)
        _, value_g = benchmark.exact(location, "g", gamma)
        recovered = {"x": location.tolist(),
                     **candidate_metrics(benchmark, location, labels, value_g)}
    payload = {"schema": "macro_anchored_audit_v2", "run": str(Path(args.run).resolve()),
               "records": audited, "candidates": candidates,
               "cut_calls": benchmark.oracle.calls,
               "paper_output": recovered,
               "audit_seconds": time.perf_counter() - audit_start,
               "reporting": "ll_gap is signed g(x,y)-min_{feasible z}g(x,z); the legacy score uses its positive part plus implication violation; f_at_g_response does not certify optimistic tie-breaking"}
    write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "audited": len(audited),
                      "cut_calls": benchmark.oracle.calls}, indent=2))


def sweep(args):
    sweep_start = time.perf_counter()
    result = read_json(args.run)
    if result.get("release_export"):
        raise ValueError("This compact export has audited metrics but no labels; rerun training to audit anew")
    benchmark = benchmark_for_run(result, args.backend, reuse_trees=args.reuse_trees)
    settings = result["settings"]
    lower = np.asarray(settings["x_lower"])
    upper = np.asarray(settings["x_upper"])
    location = np.asarray(result["records"][args.record_index]["x"], dtype=np.float64)
    gamma = settings["gamma"]
    rng = np.random.default_rng(args.seed)
    query_points = [location + settings["rho"] * ball_sample(rng, 3) for _ in range(args.samples)]
    exact = []
    for query in query_points:
        response_g, value_g = benchmark.exact(query, "g", gamma)
        response_f, value_f = benchmark.exact(query, "F", gamma)
        finite_difference = []
        response_stable = True
        for coordinate in range(query.size):
            direction = np.zeros_like(query)
            direction[coordinate] = args.fd_step
            plus_value, plus_g, plus_f = exact_penalty(benchmark, query + direction, gamma, True)
            minus_value, minus_g, minus_f = exact_penalty(benchmark, query - direction, gamma, True)
            finite_difference.append((plus_value - minus_value) / (2 * args.fd_step))
            response_stable &= (np.array_equal(response_g, plus_g)
                                and np.array_equal(response_g, minus_g)
                                and np.array_equal(response_f, plus_f)
                                and np.array_equal(response_f, minus_f))
        exact.append({"g": value_g, "F": value_f,
                      "gradient": gamma * (benchmark.gradient_g(query, response_f)
                                           - benchmark.gradient_g(query, response_g)),
                      "finite_difference": finite_difference,
                      "response_stable": bool(response_stable)})
    rows = []
    for iterations in args.k_values:
        range_g, range_f = benchmark.uniform_ranges(lower, upper, settings["rho"], gamma)
        if args.dual_step_rule == "theory":
            normalized_range_f = (range_f / gamma
                                  if benchmark.normalize_f_oracle and gamma > 1.0 else range_f)
            theta_g, theta_f = range_g / math.sqrt(iterations), normalized_range_f / math.sqrt(iterations)
        else:
            theta_g, theta_f = args.theta_g, args.theta_f
        errors = []
        value_gaps = []
        saddle_gaps = []
        for query, truth in zip(query_points, exact):
            estimate_g = dual_chain(benchmark, query, "g", gamma, iterations,
                                    theta_g, DualState(len(benchmark.anchor_cell)), range_g)
            estimate_f = dual_chain(benchmark, query, "F", gamma, iterations,
                                    theta_f, DualState(len(benchmark.anchor_cell)), range_f)
            estimate = estimate_f["gradient"] - gamma * estimate_g["gradient"]
            errors.append(estimate - truth["gradient"])
            value_gaps.append([truth["g"] - estimate_g["best_dual"],
                               truth["F"] - estimate_f["best_dual"]])
            saddle_gaps.append([estimate_g["saddle_gap"], estimate_f["saddle_gap"]])
        mean_error = np.mean(errors, axis=0)
        standard_error = np.std(errors, axis=0, ddof=1) / math.sqrt(args.samples) if args.samples > 1 else np.full(3, np.nan)
        rows.append({"K": iterations, "theta_g": theta_g, "theta_f": theta_f,
                     "mean_gradient_difference": mean_error.tolist(),
                     "norm_mean_gradient_difference": float(np.linalg.norm(mean_error)),
                     "coordinate_standard_error": standard_error.tolist(),
                     "mean_dual_value_gap": np.mean(value_gaps, axis=0).tolist(),
                     "mean_saddle_gap": np.mean(saddle_gaps, axis=0).tolist()})
    write_json(args.output, {"schema": "macro_anchored_k_sweep_v1",
                             "run": str(Path(args.run).resolve()), "record_index": args.record_index,
                             "samples": args.samples, "step_rule": args.dual_step_rule,
                             "sweep_seconds": time.perf_counter() - sweep_start,
                             "seed": args.seed, "rows": rows,
                             "finite_difference_step": args.fd_step,
                             "response_stable_samples": sum(item["response_stable"] for item in exact),
                             "branch_vs_finite_difference_error": [
                                 float(np.linalg.norm(item["gradient"] - item["finite_difference"]))
                                 for item in exact],
                             "caution": "Branch gradients need not equal a derivative at response switches; finite differences are diagnostics, not an A4 certificate"})
    print(json.dumps({"output": str(args.output), "rows": len(rows),
                      "cut_calls": benchmark.cut_calls}, indent=2))


def positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Expected a positive finite number")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--data", required=True, type=Path)
    runner.add_argument("--output", required=True, type=Path)
    runner.add_argument("--backend", choices=("auto", "pymaxflow", "dinic"), default="auto")
    runner.add_argument("--reuse-trees", action="store_true")
    runner.add_argument("--method", choices=("bigap", "weighted", "zo", "random", "cmaes"), default="bigap")
    runner.add_argument("--cma-sigma", type=positive_float, default=0.4)
    runner.add_argument("--reference-mode", choices=("recoverable", "external"), default="recoverable")
    runner.add_argument("--reference-file", type=Path)
    runner.add_argument("--min-class-fraction", type=float, default=0.02)
    runner.add_argument("--x-reference", nargs=3, type=float, default=(2.0, -1.0, 0.0))
    runner.add_argument("--x-start", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    runner.add_argument("--x-lower", nargs=3, type=float, default=(-4.0, -4.0, -1.0))
    runner.add_argument("--x-upper", nargs=3, type=float, default=(4.0, 4.0, 1.0))
    runner.add_argument("--gamma", type=positive_float, default=0.01)
    runner.add_argument("--rho", type=positive_float, default=0.05)
    runner.add_argument("--k-g", type=int, default=3)
    runner.add_argument("--k-f", type=int, default=3)
    runner.add_argument("--sfm-blocks", type=int, default=1)
    runner.add_argument("--normalize-f-oracle", action=argparse.BooleanOptionalAction,
                        default=True)
    runner.add_argument("--batch-size", type=int, default=1)
    runner.add_argument("--outer-step", type=positive_float, default=5e-5)
    runner.add_argument("--dual-step-rule", choices=("fixed", "theory"), default="fixed")
    runner.add_argument("--theta-g", type=positive_float, default=0.1)
    runner.add_argument("--theta-f", type=positive_float, default=0.1)
    runner.add_argument("--time-budget", type=positive_float, default=300.0)
    runner.add_argument("--record-interval", type=positive_float, default=1.0)
    runner.add_argument("--max-iterations", type=int, default=100000)
    runner.add_argument("--seed", type=int, default=0)
    runner.set_defaults(function=run)

    auditor = commands.add_parser("audit")
    auditor.add_argument("--run", required=True, type=Path)
    auditor.add_argument("--output", required=True, type=Path)
    auditor.add_argument("--stride", type=int, default=1)
    auditor.add_argument("--backend", choices=("auto", "pymaxflow", "dinic"), default="auto")
    auditor.set_defaults(function=audit)

    sweeper = commands.add_parser("sweep")
    sweeper.add_argument("--run", required=True, type=Path)
    sweeper.add_argument("--output", required=True, type=Path)
    sweeper.add_argument("--k-values", nargs="+", type=int, default=(1, 3, 5, 10, 20, 30))
    sweeper.add_argument("--samples", type=int, default=16)
    sweeper.add_argument("--record-index", type=int, default=0)
    sweeper.add_argument("--dual-step-rule", choices=("fixed", "theory"), default="theory")
    sweeper.add_argument("--theta-g", type=positive_float, default=0.1)
    sweeper.add_argument("--theta-f", type=positive_float, default=0.1)
    sweeper.add_argument("--seed", type=int, default=0)
    sweeper.add_argument("--fd-step", type=positive_float, default=1e-3)
    sweeper.add_argument("--backend", choices=("auto", "pymaxflow", "dinic"), default="auto")
    sweeper.add_argument("--reuse-trees", action="store_true")
    sweeper.set_defaults(function=sweep)

    args = parser.parse_args()
    if args.command == "run" and (args.k_g < 1 or args.k_f < 1 or args.sfm_blocks < 1
                                   or args.batch_size < 1 or args.max_iterations < 1):
        parser.error("Dual budgets, SFM blocks, batch size, and iteration cap must be positive")
    if args.command == "run" and not 0 <= args.min_class_fraction < 0.5:
        parser.error("Reference class fraction must lie in [0, 0.5)")
    if args.command == "audit" and args.stride < 1:
        parser.error("Audit stride must be positive")
    if args.command == "sweep" and (args.samples < 1 or not args.k_values or min(args.k_values) < 1):
        parser.error("Sweep samples and K values must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
