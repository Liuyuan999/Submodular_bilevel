"""Independent checks of RC structure, gradients, and cut optimality."""

from copy import deepcopy
from itertools import product
from pathlib import Path
import unittest

import numpy as np

from openroad.sampling import ball_sample, controlled_instance
from openroad.benchmark import build_instance, net_connections, verify_sources, wire_parameters
from openroad.model import RCModel
from openroad.oracle import RCCutOracle
from openroad.run import audit, train
from openroad.search import train_search
from openroad.comparison import comparison_response, optimizer_hashes, penalty_values, source_hashes, task_spec


def rc_instance(count=4):
    data = controlled_instance(count, 3)
    data["lower_weights"]["density"] = 0.0
    data["upper_weights"]["density"] = 0.0
    data["placement_radius"] = 0.5
    return data


class RCStructureTests(unittest.TestCase):
    def test_sparse_polynomial_matches_all_binary_objectives(self):
        model = RCModel(rc_instance())
        for positions in (model.reference, model.reference + 0.071):
            unaries = model.unaries(positions)
            zero = model.evaluate(positions, np.zeros(model.movable_count))
            for labels in product((0, 1), repeat=model.movable_count):
                labels = np.asarray(labels)
                actual = model.evaluate(positions, labels)
                for name, unary, pairs in zip(("lower", "upper"), unaries,
                                              (model.lower_pairs, model.upper_pairs)):
                    reconstructed = (zero[name] + unary @ labels +
                                     np.sum(pairs * labels[model.pair_first] * labels[model.pair_second]))
                    self.assertAlmostEqual(actual[name], reconstructed, places=12)
            self.assertTrue(np.all(model.lower_pairs <= 0))
            self.assertGreater(np.count_nonzero(model.lower_pairs < 0), 0)

    def test_cut_and_normalized_penalty_match_enumeration(self):
        model = RCModel(rc_instance(5))
        for gamma in (0.01, 1.0, 50.0, 100.0):
            oracle = RCCutOracle(model, gamma, backend="dinic")
            for offset in (0.0, -0.13, 0.27):
                positions = model.reference + offset
                lower_labels, penalty_labels = oracle.responses(positions)
                candidates = [model.evaluate(positions, labels)
                              for labels in product((0, 1), repeat=model.movable_count)]
                lower = model.evaluate(positions, lower_labels)
                penalty = model.evaluate(positions, penalty_labels)
                self.assertAlmostEqual(lower["lower"], min(value["lower"] for value in candidates), places=10)
                self.assertAlmostEqual(penalty["upper"] + gamma * penalty["lower"],
                                       min(value["upper"] + gamma * value["lower"]
                                           for value in candidates), places=10)
            self.assertEqual(oracle.scale, gamma)

    def test_fixed_label_gradients_on_and_outside_placement_box(self):
        model = RCModel(rc_instance(3))
        labels = np.asarray([1, 0, 1])
        for positions in (model.reference, model.upper_bound + 0.1):
            analytic = model.evaluate(positions, labels, True)
            for node, axis in product(range(3), range(2)):
                forward, backward = positions.copy(), positions.copy()
                forward[node, axis] += 1e-5
                backward[node, axis] -= 1e-5
                for objective in ("upper", "lower"):
                    numerical = (model.evaluate(forward, labels)[objective] -
                                 model.evaluate(backward, labels)[objective]) / 2e-5
                    self.assertAlmostEqual(numerical, analytic[objective + "_gradient"][node, axis], places=8)

    def test_equal_responses_keep_direct_upper_gradient(self):
        data = rc_instance(3)
        data["lower_weights"]["area"] = 100.0
        model = RCModel(data)
        oracle = RCCutOracle(model, 50.0, backend="dinic")
        direction, lower, penalty = oracle.direction(model.reference)
        np.testing.assert_array_equal(lower, penalty)
        direct = model.evaluate(model.reference, penalty, True)["upper_gradient"]
        np.testing.assert_allclose(direction, direct, atol=1e-12)
        self.assertGreater(np.linalg.norm(direction), 1e-5)

    def test_BiGAP_direction_matches_penalty_finite_difference(self):
        model = RCModel(rc_instance(3))
        for gamma in (0.1, 2.0, 50.0, 100.0):
            oracle = RCCutOracle(model, gamma, "dinic")
            positions = model.reference + 0.013
            analytic, _, _ = oracle.direction(positions)

            def value(query):
                lower_labels, upper_labels = oracle.responses(query)
                lower = model.evaluate(query, lower_labels)
                upper = model.evaluate(query, upper_labels)
                return upper["upper"] + gamma * (upper["lower"] - lower["lower"])

            for node, axis in product(range(3), range(2)):
                forward, backward = positions.copy(), positions.copy()
                forward[node, axis] += 1e-5
                backward[node, axis] -= 1e-5
                numerical = (value(forward) - value(backward)) / 2e-5
                self.assertAlmostEqual(numerical, analytic[node, axis], places=7)

    def test_structural_certificate_rejects_density_and_nonfinite_data(self):
        data = rc_instance()
        data["upper_weights"]["density"] = 0.01
        with self.assertRaises(ValueError):
            RCModel(data)
        data = rc_instance()
        data["lower_timing_weights"][0] = float("nan")
        with self.assertRaises(ValueError):
            RCModel(data)
        data = rc_instance()
        data["arcs"][1] = [0, 1, 0.01, 1.0, [2.0, 1.0]]
        with self.assertRaises(ValueError):
            RCModel(data)
        certificate = RCModel(rc_instance()).certificate()
        self.assertEqual(certificate["A2"]["beta"], 0)
        self.assertEqual(certificate["A3"]["constraint_count"], 0)
        self.assertEqual(certificate["A4"]["status"], "not_verified")

    def test_deferred_audit_matches_saved_outputs(self):
        model = RCModel(rc_instance(3))
        for method in ("bigap", "sequential", "alternating", "zero_order"):
            run = train(model, method=method, iterations=2, gamma=2.0, step=0.1, backend="dinic")
            saved = deepcopy(run)
            result = audit(model, run, backend="dinic")
            self.assertEqual(run, saved)
            self.assertEqual(len(result["history"]), 3)
            for record in result["history"]:
                if "g" in record["responses"]:
                    self.assertLess(record["responses"]["g"]["LL_gap"], 1e-10)
                self.assertGreater(record["seconds"], 0)
                self.assertLess(record["seconds"], run["time_budget_seconds"])

    def test_algorithm_one_minibatches_and_uniform_output(self):
        model = RCModel(rc_instance(3))
        seed, horizon, batch_size = 17, 4, 3
        radius, step, gamma = 0.8, 0.1, 0.1
        run = train(model, seed=seed, iterations=horizon, gamma=gamma, radius=radius,
                    step=step, batch_size=batch_size, backend="dinic")
        generator = np.random.default_rng(seed)
        oracle = RCCutOracle(model, gamma, "dinic", False)
        expected = model.reference.copy()
        for iteration in range(horizon):
            np.testing.assert_allclose(run["history"][iteration]["positions"], expected, atol=1e-13)
            gradients = [oracle.direction(expected + radius * ball_sample(generator, expected.shape))[0]
                         for _ in range(batch_size)]
            expected = model.project(expected - step * np.mean(gradients, axis=0))
        np.testing.assert_allclose(run["history"][-1]["positions"], expected, atol=1e-13)
        output_generator = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
        expected_index = int(output_generator.choice(list(range(horizon))))
        output = run["algorithm_output"]
        self.assertEqual(output["history_index"], expected_index)
        self.assertLess(output["iteration"], horizon)
        self.assertEqual(output["positions"], run["history"][expected_index]["positions"])
        np.testing.assert_array_equal(output["responses"]["F"]["labels"],
                                      oracle.solve(np.asarray(output["positions"]), "F"))
        self.assertEqual(run["optimizer_cut_calls"], 2 * batch_size * horizon + 1)
        self.assertEqual(run["reporting_cut_calls"], 2 * (horizon + 1))
        self.assertEqual(run["completed_updates"], horizon)
        self.assertTrue(run["requested_horizon_completed"])
        self.assertTrue(output["within_time_budget"])
        self.assertIsNotNone(audit(model, run, "dinic")["algorithm_output"])

    def test_algorithm_one_requires_positive_radius_horizon_and_batch(self):
        model = RCModel(rc_instance(3))
        for invalid in ({"radius": 0}, {"iterations": 0}, {"batch_size": 0}, {"batch_size": 1.5}):
            with self.assertRaises(ValueError):
                train(model, backend="dinic", **invalid)

    def test_nested_search_baselines_save_exact_LL_responses(self):
        model = RCModel(rc_instance(3))
        for method in ("random", "cmaes"):
            run = train_search(model, method, seed=7, iterations=20, time_budget=30, backend="dinic")
            self.assertEqual(len(run["history"]), 20)
            np.testing.assert_array_equal(run["history"][0]["positions"], model.reference)
            result = audit(model, run, "dinic")
            for record in result["history"]:
                self.assertLess(record["responses"]["g"]["LL_gap"], 1e-10)
            self.assertIsNone(run["algorithm_output"])

    def test_comparison_task_mapping_and_sources(self):
        tasks = [task_spec(task) for task in range(72)]
        self.assertEqual(len({task[3] for task in tasks}), 72)
        self.assertEqual(tasks[0][:3], ("bigap", 0.1, 0))
        self.assertEqual(tasks[-1][:3], ("random", 1.0, 7))
        hashes = source_hashes()
        self.assertIn("openroad/run.py", hashes)
        self.assertIn("openroad/placement_model.py", hashes)
        self.assertIn("dreamplace/cut_oracle.py", hashes)
        self.assertEqual(task_spec(72)[:3], ("bigap", 100.0, 0))
        self.assertEqual(task_spec(79)[:3], ("bigap", 100.0, 7))
        with self.assertRaises(ValueError):
            task_spec(80)

    def test_baselines_always_report_g_response(self):
        for method in ("sequential", "alternating", "zero_order", "cmaes", "random"):
            for mode in ("g", "F"):
                self.assertEqual(comparison_response(method, mode), "g")
        self.assertEqual(comparison_response("bigap", "g"), "g")
        self.assertEqual(comparison_response("bigap", "F"), "F")

    def test_reporting_changes_do_not_mask_solver_changes(self):
        original = source_hashes()
        reporting_change = dict(original, **{"openroad/comparison.py": "reporting-only"})
        self.assertEqual(optimizer_hashes(original), optimizer_hashes(reporting_change))
        solver_change = dict(original, **{"openroad/run.py": "changed"})
        self.assertNotEqual(optimizer_hashes(original), optimizer_hashes(solver_change))

    def test_weighted_penalty_reselects_incumbent(self):
        history = [{"lower_optimum": 3.0,
                    "responses": {"F": {"f": upper, "g": 3.0 + gap, "best_score": 1.2},
                                  "g": {"f": upper, "g": 3.0}}}
                   for upper, gap in ((1.0, 0.2), (2.0, 0.0), (1.8, 0.1))]
        self.assertEqual(int(np.argmin(penalty_values(history, "F", 1.0))), 0)
        self.assertEqual(int(np.argmin(penalty_values(history, "F", 10.0))), 1)
        np.testing.assert_allclose(np.minimum.accumulate(penalty_values(history, "F", 10.0)), [3, 2, 2])
        np.testing.assert_array_equal(penalty_values(history, "g", 1.0), penalty_values(history, "g", 100.0))

    def test_weighted_penalty_matches_direct_two_response_objective(self):
        model = RCModel(rc_instance(3))
        gamma = 100.0
        run = train(model, iterations=2, gamma=gamma, step=0.1, backend="dinic")
        evaluated = audit(model, run, "dinic")
        direct = [record["responses"]["F"]["f"] + gamma *
                  (record["responses"]["F"]["g"] - record["responses"]["g"]["g"])
                  for record in run["history"]]
        np.testing.assert_allclose(penalty_values(evaluated["history"], "F", gamma), direct, atol=1e-10)

    def test_weighted_penalty_rejects_invalid_data(self):
        history = [{"lower_optimum": 3.0, "responses": {"g": {"f": 1.0, "g": 2.9}}}]
        with self.assertRaises(ValueError):
            penalty_values(history, "g", 1.0)
        for gamma in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                penalty_values([], "g", gamma)

    def test_dynamic_cut_matches_rebuild_after_terminal_changes(self):
        try:
            import maxflow
        except ImportError:
            self.skipTest("PyMaxflow is not installed")
        model = RCModel(rc_instance(5))
        warm = RCCutOracle(model, 3.0, "pymaxflow", True)
        fresh = RCCutOracle(model, 3.0, "pymaxflow", False)
        generator = np.random.default_rng(11)
        for _ in range(20):
            positions = model.reference + generator.normal(size=model.reference.shape)
            for response in ("g", "F"):
                warm_result = model.evaluate(positions, warm.solve(positions, response))
                fresh_result = model.evaluate(positions, fresh.solve(positions, response))
                if response == "g":
                    self.assertAlmostEqual(warm_result["lower"], fresh_result["lower"], places=12)
                else:
                    self.assertAlmostEqual(warm_result["upper"] + 3 * warm_result["lower"],
                                           fresh_result["upper"] + 3 * fresh_result["lower"], places=12)


class RCBenchmarkTests(unittest.TestCase):
    def test_DEF_routes_are_not_connectivity(self):
        source = "NETS 1 ; - signal ( gate Z ) ( PIN result ) + ROUTED metal3 ( 100 200 ) ( * 300 ) ; END NETS"
        self.assertEqual(net_connections(source), [("signal", [("gate", "Z"), ("PIN", "result")])])
        with self.assertRaises(ValueError):
            net_connections("NETS 2 ; - a ( gate Z ) ; - b ( gate Z ) ; END NETS")

    def test_LEF_wire_capacitance_units(self):
        source = "LAYER metal3\n  WIDTH 0.1 ;\n CAPACITANCE CPERSQDIST 0.0002 ;\n EDGECAPACITANCE 0.00003 ;\nEND metal3\n"
        parameters = wire_parameters(source)
        self.assertAlmostEqual(parameters["cap_ff_per_um"], 0.08)
        self.assertAlmostEqual(parameters["end_cap_ff_per_branch"], 0.006)

    def test_official_source_hashes(self):
        root = Path(__file__).resolve().parents[1] / "openroad" / "data" / "upstream"
        if not root.exists():
            self.skipTest("Official source data is not present")
        self.assertEqual(len(verify_sources(root)), 4)

    def test_full_official_instance_uses_all_eligible_cells(self):
        root = Path(__file__).resolve().parents[1] / "openroad" / "data" / "upstream"
        if not root.exists():
            self.skipTest("Official source data is not present")
        instance = build_instance(root)
        model = RCModel(instance)
        self.assertEqual(model.movable_count, 295)
        self.assertEqual(instance["provenance"]["selected_count"], instance["provenance"]["eligible_count"])
        self.assertEqual(instance["provenance"]["net_count"], 497)
        self.assertEqual(len(model.arcs), 851)
        self.assertEqual(len(model.lower_pairs), 277)
        self.assertTrue(np.all(model.lower_pairs < 0))


if __name__ == "__main__":
    unittest.main()
