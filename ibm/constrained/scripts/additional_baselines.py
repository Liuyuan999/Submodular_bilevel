"""Literature baselines for the constrained EDA experiment.

Methods:
  nested_ga  -- classical nested real-coded GA with a lexicographic MILP follower.
  zjs_binary -- binary-upper restriction and structural test from arXiv:2606.10345.
  divide_blo -- sampled-value-function PGD from arXiv:2505.10830.

Every setup, sampling, MILP, and objective-query cost is charged to the method's
wall-clock budget.
"""
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from scipy.optimize import Bounds, LinearConstraint, milp

BASE_PATH = Path(__file__).resolve().parent / "benchmark.py"
_spec = importlib.util.spec_from_file_location("constrained_exp5_v3", BASE_PATH)
V = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = V
_spec.loader.exec_module(V)
B, M = V.B, V.M
E = os.environ.get

NESTED_POP = int(E("NESTED_POP", "12"))
NESTED_LL_TIME = float(E("NESTED_LL_TIME", "20"))
NESTED_MIP_GAP = float(E("NESTED_MIP_GAP", "0"))
NESTED_OPT_TOL = float(E("NESTED_OPT_TOL", "1e-6"))
NESTED_MUTATION = float(E("NESTED_MUTATION", "0.15"))
DIVIDE_SAMPLES = int(E("DIVIDE_SAMPLES", "32"))
DIVIDE_SAMPLE_K = int(E("DIVIDE_SAMPLE_K", "20"))
DIVIDE_GAMMA = float(E("DIVIDE_GAMMA", "20"))
DIVIDE_LAMBDA = float(E("DIVIDE_LAMBDA", "1"))
DIVIDE_ALPHA = float(E("DIVIDE_ALPHA", "1e-5"))
DIVIDE_EVAL_EVERY = int(E("DIVIDE_EVAL_EVERY", "10"))

assert NESTED_POP >= 4 and NESTED_LL_TIME > 0 and NESTED_MIP_GAP >= 0
assert 0 < NESTED_MUTATION <= 1
assert DIVIDE_SAMPLES >= 2 and DIVIDE_SAMPLE_K >= 1
assert DIVIDE_GAMMA > 0 and DIVIDE_LAMBDA > 0 and DIVIDE_ALPHA > 0
assert DIVIDE_EVAL_EVERY >= 1


def simplex_projection(vector):
    """Euclidean projection onto the unit simplex."""
    values = np.asarray(vector, dtype=float)
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    support = np.flatnonzero(
        ordered - cumulative / (np.arange(len(values)) + 1) > 0
    )
    threshold = cumulative[support[-1]] / (support[-1] + 1)
    return np.maximum(values - threshold, 0.0)


class ExactFollowerMILP:
    """Lexicographic constrained binary follower for fixed continuous x."""

    def __init__(self):
        self.n = M.n
        self.base_edges = len(M.ei)
        raw_i = np.concatenate([M.ei, B.I2])
        raw_j = np.concatenate([M.ej, B.J2])
        raw_pairs = np.sort(np.column_stack([raw_i, raw_j]), axis=1)
        unique_pairs, inverse = np.unique(raw_pairs, axis=0, return_inverse=True)
        edge_i, edge_j = unique_pairs[:, 0], unique_pairs[:, 1]
        self.edge_count = len(unique_pairs)
        self.variable_count = self.n + self.edge_count
        self.base_weights = np.bincount(
            inverse[:self.base_edges], weights=M.w, minlength=self.edge_count
        )
        self.cut_weights = np.bincount(
            inverse[self.base_edges:], minlength=self.edge_count
        )
        self.upper_weights = np.bincount(
            inverse[:self.base_edges], weights=M.a, minlength=self.edge_count
        )

        rows = np.repeat(np.arange(4 * self.edge_count), 3)
        columns = np.empty((4 * self.edge_count, 3), dtype=int)
        values = np.empty((4 * self.edge_count, 3), dtype=float)
        for edge, (left, right) in enumerate(zip(edge_i, edge_j)):
            disagreement = self.n + edge
            columns[4 * edge:4 * edge + 4] = (
                (disagreement, left, right), (disagreement, left, right),
                (disagreement, left, right), (disagreement, left, right),
            )
            values[4 * edge:4 * edge + 4] = (
                (1, -1, 1), (1, 1, -1),
                (1, -1, -1), (1, 1, 1),
            )
        disagreement_matrix = sparse.coo_matrix(
            (values.ravel(), (rows, columns.ravel())),
            shape=(4 * self.edge_count, self.variable_count),
        ).tocsr()
        lower = np.tile([0.0, 0.0, -np.inf, -np.inf], self.edge_count)
        upper = np.tile([np.inf, np.inf, 0.0, 2.0], self.edge_count)
        cut_columns = self.n + np.flatnonzero(self.cut_weights)
        cut_row = sparse.csr_matrix(
            (self.cut_weights[self.cut_weights > 0],
             (np.zeros(np.sum(self.cut_weights > 0), dtype=int), cut_columns)),
            shape=(1, self.variable_count),
        )
        self.matrix = sparse.vstack([disagreement_matrix, cut_row], format="csr")
        self.lower = np.concatenate([lower, [-np.inf]])
        self.upper = np.concatenate([upper, [B.C]])
        self.integrality = np.zeros(self.variable_count, dtype=np.uint8)
        self.integrality[:self.n] = 1
        self.bounds = Bounds(
            np.zeros(self.variable_count), np.ones(self.variable_count)
        )
        self.f_objective = np.zeros(self.variable_count)
        self.f_objective[:self.n] = np.where(M.yref > 0.5, -1.0, 1.0)
        self.f_objective[self.n:] = -self.upper_weights

    def g_objective(self, x):
        c0, c1, _, _ = M.unary(x)
        objective = np.zeros(self.variable_count)
        objective[:self.n] = c1 - c0
        objective[self.n:] = self.base_weights
        return objective, float(np.sum(c0))

    def solve(self, x, seconds):
        if seconds <= 0:
            return None
        objective_g, g_constant = self.g_objective(x)
        phase_one = milp(
            objective_g,
            integrality=self.integrality,
            bounds=self.bounds,
            constraints=LinearConstraint(self.matrix, self.lower, self.upper),
            options={
                "time_limit": max(seconds * 0.5, 0.01),
                "mip_rel_gap": NESTED_MIP_GAP,
            },
        )
        if phase_one.x is None:
            return None
        best_g_linear = float(objective_g @ phase_one.x)
        phase_two_matrix = sparse.vstack(
            [self.matrix, sparse.csr_matrix(objective_g.reshape(1, -1))],
            format="csr",
        )
        phase_two = milp(
            self.f_objective,
            integrality=self.integrality,
            bounds=self.bounds,
            constraints=LinearConstraint(
                phase_two_matrix,
                np.concatenate([self.lower, [-np.inf]]),
                np.concatenate([self.upper, [best_g_linear + NESTED_OPT_TOL]]),
            ),
            options={
                "time_limit": max(seconds * 0.5, 0.01),
                "mip_rel_gap": NESTED_MIP_GAP,
            },
        )
        solution = phase_two.x if phase_two.x is not None else phase_one.x
        y = (solution[:self.n] >= 0.5).astype(float)
        c0, c1, _, _ = M.unary(x)
        return {
            "y": y,
            "f": float(M.f_val(y)),
            "g": float(M.g_val(c0, c1, y)),
            "constraint": float(B.cval(y)),
            "phase1_success": bool(phase_one.success),
            "phase2_success": bool(phase_two.success),
            "g_milp": best_g_linear + g_constant,
        }


FOLLOWER = None


def follower():
    global FOLLOWER
    if FOLLOWER is None:
        FOLLOWER = ExactFollowerMILP()
    return FOLLOWER


def solve_follower_bounded(x, seconds):
    """Run HiGHS in a killable subprocess; its presolve can exceed time_limit."""
    if seconds <= 0:
        return None
    descriptor, output_path = tempfile.mkstemp(prefix="eda_follower_", suffix=".json")
    os.close(descriptor)
    environment = os.environ.copy()
    environment.update({
        "FOLLOWER_WORKER": "1",
        "FOLLOWER_X": ",".join(f"{value:.17g}" for value in x),
        "FOLLOWER_SECONDS": str(seconds),
        "FOLLOWER_OUTPUT": output_path,
        "METHODS": "random",
    })
    try:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=seconds, check=False,
        )
        output = Path(output_path)
        if not output.exists() or output.stat().st_size == 0:
            return None
        result = json.loads(output.read_text())
        result["y"] = np.asarray(result["y"], dtype=float)
        return result
    except subprocess.TimeoutExpired:
        return None
    finally:
        Path(output_path).unlink(missing_ok=True)


def remaining(clock, cap=NESTED_LL_TIME):
    return max(0.0, min(cap, B.TB - clock.elapsed()))


def observe_exact(clock, x, result):
    if result is None:
        return False
    info = {
        "y_best": result["y"], "gap": 0.0,
        "feasible_frac": 1.0, "gamma": clock.current_gamma,
    }
    return clock.observe(np.asarray(x), result["f"], info)


def nested_ga(x0, random_state):
    """Nested real-coded GA with a lexicographic MILP follower."""
    clock = V.V3Clock(x0)
    clock.hist, clock.traj = [], []
    clock.best = (np.inf, x0.copy())
    clock.observe(
        x0, np.inf,
        {"y_best": None, "gap": None, "feasible_frac": 0.0,
         "gamma": clock.current_gamma},
    )
    clock.method_meta = {
        "population": NESTED_POP,
        "ll_time_limit": NESTED_LL_TIME,
        "mip_relative_gap": NESTED_MIP_GAP,
        "ll_optimistic_tolerance": NESTED_OPT_TOL,
        "mutation_probability": NESTED_MUTATION,
    }
    population = np.vstack([
        x0,
        random_state.uniform(M.XLO, M.XHI, size=(NESTED_POP - 1, M.P)),
    ])
    fitness = np.full(NESTED_POP, np.inf)
    generation = 0
    while clock.active():
        for index in np.flatnonzero(~np.isfinite(fitness)):
            if not clock.active():
                break
            result = solve_follower_bounded(population[index], remaining(clock))
            if result is not None:
                fitness[index] = result["f"]
                observe_exact(clock, population[index], result)
                clock.diag.append([
                    generation, int(index), result["g"], result["constraint"],
                    int(result["phase1_success"]), int(result["phase2_success"]),
                ])
        if not clock.active() or not np.any(np.isfinite(fitness)):
            break
        elite = population[np.argmin(fitness)].copy()

        def tournament():
            candidates = random_state.integers(0, NESTED_POP, size=2)
            return population[candidates[np.argmin(fitness[candidates])]]

        children = [elite]
        scale = M.XHI - M.XLO
        while len(children) < NESTED_POP:
            parent_a, parent_b = tournament(), tournament()
            blend = random_state.uniform(-0.25, 1.25, size=M.P)
            child = blend * parent_a + (1.0 - blend) * parent_b
            mutate = random_state.random(M.P) < NESTED_MUTATION
            child[mutate] += (
                random_state.normal(0.0, 0.1, size=np.sum(mutate))
                * scale[mutate]
            )
            children.append(M.proj(child))
        population = np.asarray(children)
        fitness.fill(np.inf)
        generation += 1
    return clock


def set_function_property(values, tolerance=1e-7):
    submodular = True
    supermodular = True
    worst_sub = 0.0
    worst_super = 0.0
    for first in range(1 << M.P):
        for second in range(first, 1 << M.P):
            residual = (
                values[first] + values[second]
                - values[first | second] - values[first & second]
            )
            worst_sub = min(worst_sub, residual)
            worst_super = max(worst_super, residual)
            submodular &= residual >= -tolerance
            supermodular &= residual <= tolerance
    return submodular, supermodular, worst_sub, worst_super


def zjs_binary(x0, random_state):
    """Binary-corner interdiction restriction and exhaustive structural test."""
    del random_state
    clock = V.V3Clock(x0)
    clock.hist, clock.traj = [], []
    clock.best = (np.inf, x0.copy())
    clock.observe(
        x0, np.inf,
        {"y_best": None, "gap": None, "feasible_frac": 0.0,
         "gamma": clock.current_gamma},
    )
    corners = []
    for bits in itertools.product((0, 1), repeat=M.P):
        indicator = np.asarray(bits, dtype=float)
        x = M.XLO + indicator * (M.XHI - M.XLO)
        mask = sum(bit << index for index, bit in enumerate(bits))
        corners.append((mask, x))
    corners.sort(key=lambda item: np.linalg.norm(item[1] - x0))
    lower_values = np.full(1 << M.P, np.nan)
    for mask, x in corners:
        if not clock.active():
            break
        result = solve_follower_bounded(x, remaining(clock))
        if result is None:
            continue
        lower_values[mask] = result["g"]
        observe_exact(clock, x, result)
    if np.all(np.isfinite(lower_values)):
        structure = set_function_property(lower_values)
        clock.method_meta = {
            "all_binary_corners_evaluated": True,
            "value_function_submodular": bool(structure[0]),
            "value_function_supermodular": bool(structure[1]),
            "worst_submodularity_residual": float(structure[2]),
            "worst_supermodularity_residual": float(structure[3]),
        }
    else:
        clock.method_meta = {
            "all_binary_corners_evaluated": False,
            "evaluated_corners": int(np.sum(np.isfinite(lower_values))),
            "total_corners": int(1 << M.P),
        }
    return clock


def sample_feasible_responses(clock, random_state):
    candidates = []
    seen = set()

    def add(response):
        if response is None or B.cval(response) > 0:
            return
        packed = np.packbits(response.astype(np.uint8)).tobytes()
        if packed not in seen:
            seen.add(packed)
            candidates.append(response.astype(float).copy())

    add(np.zeros(M.n))
    add(np.ones(M.n))
    if B.cval(M.yref) <= 0:
        add(M.yref)
    while len(candidates) < DIVIDE_SAMPLES and clock.active():
        x = random_state.uniform(M.XLO, M.XHI)
        state = B.DualState()
        _, _, info = V.dualgrad_v3(
            x, "g", K_inner=DIVIDE_SAMPLE_K, state=state
        )
        add(info["y_best"])
        add(state.last_response)
    return np.asarray(candidates)


def fractional_f_and_gradient(y):
    gradient = np.where(M.yref > 0.5, -1.0, 1.0)
    difference = y[M.ei] - y[M.ej]
    sign = np.sign(difference)
    np.add.at(gradient, M.ei, -M.a * sign)
    np.add.at(gradient, M.ej, M.a * sign)
    value = (
        np.sum(np.abs(y - M.yref))
        + np.sum(M.a * (1.0 - np.abs(difference)))
    )
    return float(value), gradient


def divide_blo(x0, random_state):
    """DIVIDE-BLO with fixed feasible lower samples and continuous x."""
    clock = V.V3Clock(x0)
    samples = sample_feasible_responses(clock, random_state)
    if len(samples) < 2:
        clock.method_meta = {"failure": "fewer than two feasible lower samples"}
        return clock
    weights = random_state.exponential(size=len(samples))
    weights /= np.sum(weights)
    x = x0.copy()
    iteration = 0
    while clock.active():
        c0, c1, dc0, dc1 = M.unary(x)
        g_values = np.asarray([
            M.g_val(c0, c1, response) for response in samples
        ])
        g_gradients = np.asarray([
            M.gy(dc0, dc1, response) for response in samples
        ])
        value_weights = simplex_projection(-g_values / DIVIDE_LAMBDA)
        y_bar = weights @ samples
        _, gradient_y = fractional_f_and_gradient(y_bar)
        gradient_x = DIVIDE_GAMMA * (
            (weights - value_weights) @ g_gradients
        )
        gradient_p = samples @ gradient_y + DIVIDE_GAMMA * g_values
        x = M.proj(x - DIVIDE_ALPHA * gradient_x)
        weights = simplex_projection(weights - DIVIDE_ALPHA * gradient_p)
        iteration += 1
        if iteration % DIVIDE_EVAL_EVERY == 0 and clock.active():
            value, info = clock.evaluate(x)
            if not clock.observe(x, value, info):
                break
            clock.diag.append([
                iteration, len(samples), float(np.linalg.norm(gradient_x)),
                float(np.linalg.norm(gradient_p)), float(np.min(g_values)),
            ])
    clock.method_meta = {
        "sample_count": int(len(samples)),
        "requested_sample_count": DIVIDE_SAMPLES,
        "nonsmooth_extension": "total_variation_subgradient",
    }
    return clock


B.FUNS.update({
    "nested_ga": nested_ga,
    "zjs_binary": zjs_binary,
    "divide_blo": divide_blo,
})
B.RANDOMIZED.update({"nested_ga", "divide_blo"})

base_record = B.record


def record_with_method_meta(name, seed, clock):
    record = base_record(name, seed, clock)
    if hasattr(clock, "method_meta"):
        record["method_meta"] = clock.method_meta
    return record


B.record = record_with_method_meta
if __name__ == "__main__":
    if E("FOLLOWER_WORKER", "0") == "1":
        worker_x = np.fromstring(E("FOLLOWER_X"), sep=",")
        worker_result = follower().solve(worker_x, float(E("FOLLOWER_SECONDS")))
        if worker_result is not None:
            worker_result["y"] = worker_result["y"].tolist()
            Path(E("FOLLOWER_OUTPUT")).write_text(json.dumps(worker_result))
    else:
        V.main()
