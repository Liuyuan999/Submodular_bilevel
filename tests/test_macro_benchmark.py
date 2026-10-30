"""Small exact checks for the MMS preprocessing and BiGAP implementation."""

from __future__ import annotations

from itertools import product
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "dreamplace"
sys.path.insert(0, str(ROOT))

from cut_oracle import CutOracle, WarmBlockCutOracle
from macro_bigap import Benchmark, DualState, ball_sample, candidate_metrics, dual_chain, read_json


class CandidateMetricTests(unittest.TestCase):
    def test_signed_gap_is_kept_for_infeasible_response(self):
        class BenchmarkStub:
            def g_value(self, location, labels):
                return 3.0

            def constraint(self, labels):
                return np.array([1.0, -1.0])

            def f_value(self, labels):
                return 4

        metrics = candidate_metrics(BenchmarkStub(), None, None, 5.0)
        self.assertEqual(metrics["ll_gap"], -2.0)
        self.assertEqual(metrics["ll_gap_positive"], 0.0)
        self.assertEqual(metrics["constraint_violation"], 1.0)
        self.assertEqual(metrics["f_plus_gap_plus_violation"], 5.0)


class CutOracleTests(unittest.TestCase):
    def test_block_cut_solves_conditional_problem(self):
        edges = [(0, 1, 0.5), (1, 2, 1.4), (2, 3, 0.8),
                 (3, 4, 0.9), (4, 5, 0.6), (0, 5, 0.7)]
        assignment = np.array([0, 0, 1, 1, 2, 2])
        oracle = WarmBlockCutOracle(6, [edge[0] for edge in edges],
                                    [edge[1] for edge in edges],
                                    [edge[2] for edge in edges], assignment,
                                    backend="dinic")
        zero = np.array([0.9, 1.2, 0.1, 1.0, 0.4, 0.7])
        one = np.array([0.2, 0.1, 1.1, 0.2, 1.0, 0.3])

        def energy(labels):
            return (float(np.sum(np.where(labels, one, zero)))
                    + sum(weight for first, second, weight in edges
                          if labels[first] != labels[second]))

        for iteration in range(9):
            before = oracle.labels.copy()
            active = np.flatnonzero(assignment == iteration % 3)
            candidates = []
            for values in product((0, 1), repeat=len(active)):
                labels = before.copy()
                labels[active] = values
                candidates.append(energy(labels))
            after = oracle.solve(zero, one)
            self.assertAlmostEqual(energy(after), min(candidates))
            self.assertLessEqual(energy(after), energy(before) + 1e-10)

    def test_warm_block_cut_refines_inexact_response(self):
        oracle = WarmBlockCutOracle(4, [1, 2], [2, 3], [3, 2],
                                    [0, 0, 1, 1], backend="dinic")
        zero = np.array([4.0, 4.0, 0.0, 0.0])
        one = np.array([0.0, 0.0, 1.0, 1.0])
        first = oracle.solve(zero, one)
        second = oracle.solve(zero, one)
        np.testing.assert_array_equal(first, [1, 1, 0, 0])
        np.testing.assert_array_equal(second, [1, 1, 1, 1])
        self.assertEqual(oracle.calls, 2)
        exact = CutOracle(4, [1, 2], [2, 3], [3, 2], [], [], backend="dinic")
        np.testing.assert_array_equal(second, exact.solve(zero, one))

    @unittest.skipUnless(importlib.util.find_spec("maxflow"), "PyMaxflow is not installed")
    def test_warm_block_tree_reuse_matches_rebuild(self):
        edges_first = [0, 1, 2, 3, 4, 0, 2]
        edges_second = [1, 2, 3, 4, 5, 5, 4]
        weights = [0.6, 1.2, 0.4, 0.8, 1.1, 0.3, 0.9]
        assignments = [0, 0, 1, 1, 2, 2]
        reused = WarmBlockCutOracle(6, edges_first, edges_second, weights,
                                    assignments, backend="pymaxflow", reuse_trees=True)
        rebuilt = WarmBlockCutOracle(6, edges_first, edges_second, weights,
                                     assignments, backend="pymaxflow")
        generator = np.random.default_rng(73)
        for _ in range(24):
            zero = generator.normal(size=6)
            one = generator.normal(size=6)
            np.testing.assert_array_equal(reused.solve(zero, one, pair_scale=2.0),
                                          rebuilt.solve(zero, one, pair_scale=2.0))

    def test_cut_and_implication_match_enumeration(self):
        oracle = CutOracle(4, [0, 1, 2], [1, 2, 3], [0.7, 1.2, 0.4],
                           [0, 1], [3, 3], backend="dinic")
        zero = np.array([0.3, -0.5, 1.0, 0.2])
        one = np.array([-0.2, 0.7, -0.1, 0.9])
        for hard in (False, True):
            for pair_scale in (0.0, 0.5, 2.0):
                labels = oracle.solve(zero, one, pair_scale, hard)
                def cost(candidate):
                    return sum(one[index] if value else zero[index]
                               for index, value in enumerate(candidate)) + pair_scale * sum(
                        weight for first, second, weight in ((0, 1, 0.7), (1, 2, 1.2), (2, 3, 0.4))
                        if candidate[first] != candidate[second]
                    )
                possibilities = [candidate for candidate in product((0, 1), repeat=4)
                                 if not hard or (candidate[0] <= candidate[3] and candidate[1] <= candidate[3])]
                optimum = min(cost(candidate) for candidate in possibilities)
                self.assertAlmostEqual(cost(labels), optimum, places=8)
                if hard:
                    self.assertLessEqual(labels[0], labels[3])
                    self.assertLessEqual(labels[1], labels[3])

    def test_random_small_cuts(self):
        rng = np.random.default_rng(17)
        edges = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
        for _ in range(20):
            weights = rng.uniform(0, 2, size=len(edges))
            zero = rng.normal(size=5)
            one = rng.normal(size=5)
            oracle = CutOracle(5, [edge[0] for edge in edges],
                               [edge[1] for edge in edges], weights,
                               [0, 1], [4, 4], backend="dinic")
            for hard in (False, True):
                labels = oracle.solve(zero, one, hard_constraints=hard)
                def value(candidate):
                    return sum(one[index] if label else zero[index]
                               for index, label in enumerate(candidate)) + sum(
                        weight for (first, second), weight in zip(edges, weights)
                        if candidate[first] != candidate[second]
                    )
                candidates = [candidate for candidate in product((0, 1), repeat=5)
                              if not hard or (candidate[0] <= candidate[4] and candidate[1] <= candidate[4])]
                self.assertAlmostEqual(value(labels), min(map(value, candidates)), places=7)

    @unittest.skipUnless(importlib.util.find_spec("maxflow"), "PyMaxflow is not installed")
    def test_pymaxflow_rebuild_and_tree_reuse(self):
        rng = np.random.default_rng(31)
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
        weights = np.array([0.5, 0.8, 1.1, 0.2])
        reused = CutOracle(4, [edge[0] for edge in edges], [edge[1] for edge in edges],
                           weights, [0, 1], [3, 3], backend="pymaxflow", reuse_trees=True)
        rebuilt = CutOracle(4, [edge[0] for edge in edges], [edge[1] for edge in edges],
                            weights, [0, 1], [3, 3], backend="pymaxflow")
        for _ in range(12):
            zero = rng.normal(size=4)
            one = rng.normal(size=4)
            a = reused.solve(zero, one)
            b = rebuilt.solve(zero, one)
            def value(candidate):
                return sum(one[index] if label else zero[index]
                           for index, label in enumerate(candidate)) + sum(
                    weight for (first, second), weight in zip(edges, weights)
                    if candidate[first] != candidate[second]
                )
            optimum = min(value(candidate) for candidate in product((0, 1), repeat=4))
            self.assertAlmostEqual(value(a), optimum, places=7)
            self.assertAlmostEqual(value(b), optimum, places=7)
            hard = rebuilt.solve(zero, one, hard_constraints=True)
            self.assertLessEqual(hard[0], hard[3])
            self.assertLessEqual(hard[1], hard[3])


class BookshelfPipelineTests(unittest.TestCase):
    def write_fixture(self, root):
        (root / "tiny.aux").write_text("RowBasedPlacement : tiny.nodes tiny.nets tiny.wts tiny.pl\n")
        (root / "tiny.nodes").write_text(
            "UCLA nodes 1.0\nNumNodes : 6\nNumTerminals : 1\n"
            "c0 1 1\nc1 1 1\nc2 1 1\nM0 5 5\nM1 5 5\npad 1 1 terminal\n"
        )
        (root / "tiny.pl").write_text(
            "UCLA pl 1.0\nc0 0 0 : N\nc1 1 0 : N\nc2 8 0 : N\n"
            "M0 1 1 : N\nM1 8 1 : N\npad 0 9 : N /FIXED\n"
        )
        (root / "tiny.nets").write_text(
            "UCLA nets 1.0\nNumNets : 5\nNumPins : 10\n"
            "NetDegree : 2 n0\nc0 B\nM0 B\n"
            "NetDegree : 2 n1\nc1 B\nM0 B\n"
            "NetDegree : 2 n2\nc2 B\nM1 B\n"
            "NetDegree : 2 n3\nc0 B\nc1 B\n"
            "NetDegree : 2 n4\nM0 B\nM1 B\n"
        )
        (root / "tiny.wts").write_text(
            "UCLA wts 1.0\nNumNets : 5\nn0 3\nn1 2\nn2 3\nn3 1\nn4 1\n"
        )
        (root / "macros.txt").write_text("M0\nM1\n")
        (root / "reference.txt").write_text("c0 0\nc1 0\nc2 1\nM0 0\nM1 1\n")

    def command(self, *arguments):
        subprocess.run([sys.executable, *map(str, arguments)], check=True,
                       capture_output=True, text=True)

    def test_prepare_run_audit_and_sweep(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_fixture(root)
            prepared = root / "data.json.gz"
            run_path = root / "run.json"
            audit_path = root / "audit.json"
            sweep_path = root / "sweep.json"
            self.command(ROOT / "mms_data.py", "--aux", root / "tiny.aux",
                         "--macro-list", root / "macros.txt", "--output", prepared)
            data = read_json(prepared)
            self.assertEqual(len(data["names"]), 5)
            self.assertEqual(len(data["anchors"]), 3)
            self.assertEqual(data["metadata"]["macro_rule"]["type"], "explicit")
            self.command(ROOT / "macro_bigap.py", "run", "--data", prepared,
                         "--output", run_path, "--backend", "dinic",
                         "--reference-mode", "external", "--reference-file", root / "reference.txt",
                         "--k-g", "2", "--k-f", "2", "--max-iterations", "2",
                         "--time-budget", "5", "--record-interval", "0.001")
            run = read_json(run_path)
            self.assertEqual(run["records"][0]["time"], 0.0)
            self.assertEqual(run["iterations"], 2)
            paper_output = read_json(run["paper_output"]["path"])
            self.assertEqual(len(paper_output["labels"]), len(data["names"]))
            self.assertEqual(run["paper_output"]["recovery_cut_calls"], 2)
            self.command(ROOT / "macro_bigap.py", "audit", "--run", run_path,
                         "--output", audit_path, "--backend", "dinic")
            audit = read_json(audit_path)
            self.assertGreaterEqual(len(audit["records"]), 2)
            self.assertGreaterEqual(len(audit["candidates"]), 2)
            self.assertEqual(audit["candidates"][0]["time"], 0.0)
            self.assertEqual(audit["candidates"][0]["kind"], "initial_reference")
            self.assertEqual(audit["candidates"][0]["f_value"], 0)
            self.assertTrue(all(record["f_plus_gap_plus_violation"] >= 0
                                for record in audit["candidates"]))
            self.assertIsNotNone(audit["paper_output"])
            self.assertGreaterEqual(audit["paper_output"]["ll_gap"], 0)
            for record in audit["records"]:
                self.assertGreaterEqual(record["ll_gap_F"], 0)
            self.command(ROOT / "macro_bigap.py", "sweep", "--run", run_path,
                         "--output", sweep_path, "--backend", "dinic", "--k-values", "1", "2",
                         "--samples", "2", "--dual-step-rule", "fixed")
            sweep = read_json(sweep_path)
            self.assertEqual([row["K"] for row in sweep["rows"]], [1, 2])
            comparison_figure = root / "comparison.pdf"
            sweep_figure = root / "sweep.pdf"
            self.command(ROOT / "plot_mms.py", "comparison", "--audits", audit_path,
                         "--output", comparison_figure)
            self.command(ROOT / "plot_mms.py", "sweep", "--audit", sweep_path,
                         "--output", sweep_figure)
            self.assertGreater(comparison_figure.stat().st_size, 0)
            self.assertGreater(sweep_figure.stat().st_size, 0)
            aggregate_figure = root / "aggregate.pdf"
            self.command(ROOT / "plot_mms.py", "aggregate", "--audits", audit_path,
                         "--expected-seeds", "1", "--output", aggregate_figure)
            self.assertGreater(aggregate_figure.stat().st_size, 0)

    def test_branch_gradient_and_dual_bound(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_fixture(root)
            prepared = root / "data.json.gz"
            self.command(ROOT / "mms_data.py", "--aux", root / "tiny.aux",
                         "--macro-list", root / "macros.txt", "--output", prepared)
            benchmark = Benchmark(prepared, backend="dinic")
            labels = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
            benchmark.set_reference(labels)
            location = np.array([0.4, -0.3, 0.1])
            derivative = benchmark.gradient_g(location, labels)
            for axis in range(3):
                displaced = location.copy()
                displaced[axis] += 1e-6
                right = benchmark.g_value(displaced, labels)
                displaced[axis] -= 2e-6
                left = benchmark.g_value(displaced, labels)
                self.assertAlmostEqual(derivative[axis], (right - left) / 2e-6, places=5)
            _, optimum = benchmark.exact(location, "g", 0.01)
            estimate = dual_chain(benchmark, location, "g", 0.01, 3, 0.1,
                                  DualState(len(benchmark.anchor_cell)), 100.0)
            self.assertLessEqual(estimate["best_dual"], optimum + 1e-8)
            self.assertGreaterEqual(estimate["saddle_gap"], 0)

    def test_dual_chains_keep_independent_states_at_unit_gamma(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_fixture(root)
            prepared = root / "data.json.gz"
            self.command(ROOT / "mms_data.py", "--aux", root / "tiny.aux",
                         "--macro-list", root / "macros.txt", "--output", prepared)
            benchmark = Benchmark(prepared, backend="dinic")
            benchmark.set_reference(np.array([0, 0, 1, 0, 1], dtype=np.uint8))
            location = np.array([0.3, -0.2, 0.1])
            state_g = DualState(len(benchmark.anchor_cell))
            state_f = DualState(len(benchmark.anchor_cell))
            first = dual_chain(benchmark, location, "g", 1.0, 3, 0.2, state_g)
            dual_chain(benchmark, location, "F", 1.0, 3, 0.2, state_f)
            self.assertEqual(set(benchmark.dual_oracles), {"g", "F"})
            self.assertIsNot(benchmark.dual_oracles["g"], benchmark.dual_oracles["F"])
            self.assertEqual(benchmark.dual_oracles["g"].calls, 3)
            self.assertEqual(benchmark.dual_oracles["F"].calls, 3)
            second = dual_chain(benchmark, location, "g", 1.0, 1, 0.2, state_g)
            self.assertEqual(benchmark.dual_oracles["g"].calls, 4)
            self.assertEqual(benchmark.dual_oracles["F"].calls, 3)
            self.assertEqual(first["last_labels"].shape, second["last_labels"].shape)

    def test_normalized_f_oracle_preserves_original_objective_and_gradient(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_fixture(root)
            prepared = root / "data.json.gz"
            self.command(ROOT / "mms_data.py", "--aux", root / "tiny.aux",
                         "--macro-list", root / "macros.txt", "--output", prepared)
            original = Benchmark(prepared, backend="dinic", normalize_f_oracle=False)
            normalized = Benchmark(prepared, backend="dinic", normalize_f_oracle=True)
            reference = np.array([0, 0, 1, 0, 1], dtype=np.uint8)
            original.set_reference(reference)
            normalized.set_reference(reference)
            location = np.array([0.3, -0.2, 0.1])
            gamma = 5.0
            original_zero, original_one, original_pair = original.h_unary(location, "F", gamma)
            zero, one, pair, scale = normalized.dual_unary(location, "F", gamma)
            np.testing.assert_allclose(scale * zero, original_zero)
            np.testing.assert_allclose(scale * one, original_one)
            self.assertEqual(scale * pair, original_pair)
            self.assertTrue(Benchmark(prepared, backend="dinic").normalize_f_oracle)
            original_labels, original_value = original.exact(location, "F", gamma)
            normalized_labels, normalized_value = normalized.exact(location, "F", gamma)
            np.testing.assert_array_equal(original_labels, normalized_labels)
            self.assertAlmostEqual(original_value, normalized_value)
            for candidate in product((0, 1), repeat=original.node_count):
                labels = np.asarray(candidate, dtype=np.uint8)
                self.assertAlmostEqual(
                    scale * normalized.h_value(labels, zero, one, pair),
                    original.h_value(labels, original_zero, original_one, original_pair))
            original_state = DualState(len(original.anchor_cell))
            normalized_state = DualState(len(normalized.anchor_cell))
            original_state.multiplier[:] = 1.0
            normalized_state.multiplier[:] = 1.0 / gamma
            original_result = dual_chain(original, location, "F", gamma, 3, 0.1,
                                         original_state, 100.0)
            normalized_result = dual_chain(normalized, location, "F", gamma, 3,
                                           0.1 / gamma,
                                           normalized_state, 100.0)
            np.testing.assert_array_equal(original_result["last_labels"],
                                          normalized_result["last_labels"])
            np.testing.assert_allclose(original_result["gradient"],
                                       normalized_result["gradient"])
            np.testing.assert_allclose(original_result["average_multiplier"],
                                       normalized_result["average_multiplier"])
            self.assertAlmostEqual(original_result["best_dual"],
                                   normalized_result["best_dual"])
            self.assertAlmostEqual(original_result["saddle_gap"],
                                   normalized_result["saddle_gap"])
            np.testing.assert_allclose(original_state.multiplier,
                                       gamma * normalized_state.multiplier)

    def test_normalized_inexact_run_and_audit(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_fixture(root)
            prepared = root / "data.json.gz"
            self.command(ROOT / "mms_data.py", "--aux", root / "tiny.aux",
                         "--macro-list", root / "macros.txt", "--output", prepared)
            output = root / "normalized.json"
            audit = root / "normalized.audit.json.gz"
            self.command(ROOT / "macro_bigap.py", "run", "--data", prepared,
                         "--output", output, "--backend", "dinic", "--method", "bigap",
                         "--reference-mode", "external", "--reference-file", root / "reference.txt",
                         "--gamma", "5", "--sfm-blocks", "2", "--k-g", "3", "--k-f", "3",
                         "--max-iterations", "2", "--time-budget", "5")
            result = read_json(output)
            self.assertTrue(result["settings"]["normalize_f_oracle"])
            self.assertEqual(result["settings"]["sfm_blocks"], 2)
            self.assertEqual(result["iterations"], 2)
            self.command(ROOT / "macro_bigap.py", "audit", "--run", output,
                         "--output", audit, "--backend", "dinic")
            self.assertIsNotNone(read_json(audit)["paper_output"])

    def test_dual_update_matches_explicit_two_steps(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_fixture(root)
            prepared = root / "data.json.gz"
            self.command(ROOT / "mms_data.py", "--aux", root / "tiny.aux",
                         "--macro-list", root / "macros.txt", "--output", prepared)
            benchmark = Benchmark(prepared, backend="dinic")
            location = np.array([0.4, -0.2, 0.1])
            unary_zero, unary_one, scale = benchmark.h_unary(location, "g", 1.0)
            multiplier = np.zeros(len(benchmark.anchor_cell))
            gradients = []
            for _ in range(2):
                labels = benchmark.dual_response(unary_zero, unary_one, scale,
                                                 multiplier, "g")
                gradients.append(benchmark.gradient_g(location, labels))
                multiplier = np.maximum(0.0, multiplier + 0.2 * benchmark.constraint(labels))
            state = DualState(len(benchmark.anchor_cell))
            result = dual_chain(benchmark, location, "g", 1.0, 2, 0.2, state)
            np.testing.assert_allclose(state.multiplier, multiplier)
            np.testing.assert_allclose(result["gradient"], np.mean(gradients, axis=0))

    def test_outer_update_matches_bigap_direction(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_fixture(root)
            prepared = root / "data.json.gz"
            output = root / "run.json"
            self.command(ROOT / "mms_data.py", "--aux", root / "tiny.aux",
                         "--macro-list", root / "macros.txt", "--output", prepared)
            self.command(ROOT / "macro_bigap.py", "run", "--data", prepared,
                         "--output", output, "--backend", "dinic", "--method", "bigap",
                         "--reference-mode", "external", "--reference-file", root / "reference.txt",
                         "--k-g", "2", "--k-f", "2", "--theta-g", "0.1", "--theta-f", "0.1",
                         "--outer-step", "0.00005", "--max-iterations", "1", "--time-budget", "5")
            result = read_json(output)
            benchmark = Benchmark(prepared, backend="dinic")
            benchmark.set_reference(read_json(result["reference"])["labels"])
            start = np.asarray(result["records"][0]["x"])
            settings = result["settings"]
            query = start + settings["rho"] * ball_sample(np.random.default_rng(0), 3)
            gradient_g = dual_chain(benchmark, query, "g", settings["gamma"], 2, 0.1,
                                    DualState(len(benchmark.anchor_cell)))["gradient"]
            gradient_f = dual_chain(benchmark, query, "F", settings["gamma"], 2, 0.1,
                                    DualState(len(benchmark.anchor_cell)))["gradient"]
            expected = np.clip(start - settings["outer_step"] *
                               (gradient_f - settings["gamma"] * gradient_g),
                               settings["x_lower"], settings["x_upper"])
            np.testing.assert_allclose(result["records"][-1]["x"], expected, atol=1e-12)

    def test_comparison_methods(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_fixture(root)
            prepared = root / "data.json.gz"
            self.command(ROOT / "mms_data.py", "--aux", root / "tiny.aux",
                         "--macro-list", root / "macros.txt", "--output", prepared)
            for method in ("weighted", "zo", "random", "cmaes"):
                with self.subTest(method=method):
                    output = root / f"{method}.json"
                    self.command(ROOT / "macro_bigap.py", "run", "--data", prepared,
                                 "--output", output, "--backend", "dinic", "--method", method,
                                 "--reference-mode", "external", "--reference-file", root / "reference.txt",
                                 "--max-iterations", "1", "--time-budget",
                                 "30" if method == "cmaes" else "5")
                    result = read_json(output)
                    self.assertEqual(result["method"], method)
                    self.assertGreaterEqual(result["cut_calls_in_training"], 1)
                    self.assertGreaterEqual(len(read_json(result["candidates"])["events"]), 1)

    @unittest.skipUnless(importlib.util.find_spec("maxflow"), "PyMaxflow is not installed")
    def test_pymaxflow_bigap_tree_reuse(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_fixture(root)
            prepared = root / "data.json.gz"
            output = root / "run.json"
            self.command(ROOT / "mms_data.py", "--aux", root / "tiny.aux",
                         "--macro-list", root / "macros.txt", "--output", prepared)
            self.command(ROOT / "macro_bigap.py", "run", "--data", prepared,
                         "--output", output, "--backend", "pymaxflow", "--reuse-trees",
                         "--reference-mode", "external", "--reference-file", root / "reference.txt",
                         "--k-g", "3", "--k-f", "3", "--max-iterations", "3",
                         "--time-budget", "5")
            result = read_json(output)
            self.assertTrue(result["reuse_trees"])
            self.assertEqual(result["cut_calls_in_training"], 18)


if __name__ == "__main__":
    unittest.main()
