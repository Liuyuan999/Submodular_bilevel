"""Check the sourced RC model against independent binary optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .benchmark import build_instance, verify_sources
from .model import RCModel
from .oracle import RCCutOracle


def milp_response(model, positions, gamma=None):
    lower, upper = model.unaries(positions)
    scale = max(1.0, gamma) if gamma is not None else 1.0
    unary = lower if gamma is None else (upper + gamma * lower) / scale
    pairs = model.lower_pairs if gamma is None else (model.upper_pairs + gamma * model.lower_pairs) / scale
    count, pair_count = model.movable_count, len(pairs)
    rows, columns, values = [], [], []
    for index, (first, second) in enumerate(zip(model.pair_first, model.pair_second)):
        rows.extend([3 * index, 3 * index, 3 * index + 1, 3 * index + 1,
                     3 * index + 2, 3 * index + 2, 3 * index + 2])
        columns.extend([count + index, int(first), count + index, int(second),
                        count + index, int(first), int(second)])
        values.extend([1, -1, 1, -1, -1, 1, 1])
    matrix = coo_matrix((values, (rows, columns)), shape=(3 * pair_count, count + pair_count)).tocsc()
    constraints = LinearConstraint(matrix, np.full(3 * pair_count, -np.inf),
                                   np.tile([0.0, 0.0, 1.0], pair_count))
    coefficients = np.concatenate((unary, pairs))
    objective_scale = 1e6 / max(1.0, float(np.max(np.abs(coefficients))))
    result = milp(objective_scale * coefficients,
                  integrality=np.concatenate((np.ones(count), np.zeros(pair_count))),
                  bounds=Bounds(np.zeros(count + pair_count), np.ones(count + pair_count)),
                  constraints=constraints, options={"time_limit": 60.0, "mip_rel_gap": 0.0})
    if not result.success:
        raise ArithmeticError(f"Independent MILP did not certify its optimum: {result.message}")
    labels = np.rint(result.x[:count]).astype(np.uint8)
    return labels, float(result.mip_gap)


def verify(model, source_root, queries=3):
    sources = verify_sources(source_root)
    provenance = model.data.get("provenance", {})
    if provenance.get("sources") != sources:
        raise ValueError("Prepared instance source records disagree with the pinned benchmark")
    reconstructed = build_instance(source_root, max_cells=model.movable_count,
                                   placement_radius=model.data["placement_radius"])
    if reconstructed != model.data:
        raise ValueError("Prepared instance does not match reconstruction from the verified source files")
    if queries < 1:
        raise ValueError("At least one query is required")
    generator = np.random.default_rng(0)
    checks = []
    warm = {gamma: RCCutOracle(model, gamma, "pymaxflow", True) for gamma in (0.1, 1.0, 50.0)}
    for query in range(queries):
        positions = (model.reference if query == 0 else
                     generator.uniform(model.lower_bound, model.upper_bound))
        lower_milp, lower_gap = milp_response(model, positions)
        exact_lower = model.evaluate(positions, lower_milp)["lower"]
        for gamma, oracle in warm.items():
            lower_cut, penalty_cut = oracle.responses(positions)
            penalty_milp, penalty_gap = milp_response(model, positions, gamma)
            cut_value = model.evaluate(positions, penalty_cut)
            milp_value = model.evaluate(positions, penalty_milp)
            lower_error = abs(model.evaluate(positions, lower_cut)["lower"] - exact_lower)
            penalty_error = abs((cut_value["upper"] + gamma * cut_value["lower"]) -
                                (milp_value["upper"] + gamma * milp_value["lower"]))
            if lower_error > 1e-8 or penalty_error > 1e-8:
                raise ArithmeticError("Graph cut disagrees with independent MILP")
            checks.append({"query": query, "gamma": gamma, "g_value_error": lower_error,
                           "F_value_error": penalty_error, "g_milp_gap": lower_gap,
                           "F_milp_gap": penalty_gap})
    labels = generator.integers(0, 2, model.movable_count)
    positions = model.reference.copy()
    measured = model.evaluate(positions, labels, True)
    gradient_error = {"upper": 0.0, "lower": 0.0}
    for node in sorted(set([0, model.movable_count // 2, model.movable_count - 1])):
        for axis in range(2):
            forward, backward = positions.copy(), positions.copy()
            forward[node, axis] += 1e-5
            backward[node, axis] -= 1e-5
            forward_values, backward_values = model.evaluate(forward, labels), model.evaluate(backward, labels)
            for name in gradient_error:
                numerical = (forward_values[name] - backward_values[name]) / 2e-5
                error = abs(numerical - measured[name + "_gradient"][node, axis])
                gradient_error[name] = max(gradient_error[name], error)
    if max(gradient_error.values()) > 1e-7:
        raise ArithmeticError("Analytic gradient failed finite differences on the benchmark")
    zero = model.evaluate(positions, np.zeros(model.movable_count))
    lower_unary, upper_unary = model.unaries(positions)
    polynomial_errors = []
    for _ in range(5):
        labels = generator.integers(0, 2, model.movable_count)
        actual = model.evaluate(positions, labels)
        for name, unary, pairs in (("lower", lower_unary, model.lower_pairs),
                                    ("upper", upper_unary, model.upper_pairs)):
            predicted = zero[name] + unary @ labels + np.sum(
                pairs * labels[model.pair_first] * labels[model.pair_second])
            polynomial_errors.append(abs(predicted - actual[name]))
    if max(polynomial_errors) > 1e-9:
        raise ArithmeticError("Sparse quadratic does not reproduce the declared objective")
    return {"status": "passed", "source_checksums": sources,
            "source_reconstruction": "matches_prepared_instance",
            "selected_cells": model.movable_count, "continuous_dimension": model.reference.size,
            "assumptions": model.certificate(), "independent_MILP_checks": checks,
            "gradient_max_absolute_error": gradient_error,
            "polynomial_max_absolute_error": max(polynomial_errors),
            "upper_gradient_norm_at_reference": float(np.linalg.norm(measured["upper_gradient"])),
            "scope": "numerical and structural verification of the declared RC surrogate; not physical timing validation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("openroad/data/upstream"))
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    content = arguments.instance.read_bytes()
    result = verify(RCModel(json.loads(content)), arguments.source, arguments.queries)
    result["instance_sha256"] = hashlib.sha256(content).hexdigest()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "selected_cells", "continuous_dimension",
                                                  "gradient_max_absolute_error", "polynomial_max_absolute_error")}, indent=2))


if __name__ == "__main__":
    main()
