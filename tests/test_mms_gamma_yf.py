"""Checks for the F-response-only gamma reporting protocol."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dreamplace"))

from macro_bigap import write_json
from run_mms_gamma_yf import paired_yf_selected_by_yg_loss, trajectory
from run_mms_confirm_50s import eligible_audit, reported_candidate


class GammaYFReportingTests(unittest.TestCase):
    def reporting_fixture(self, root):
        candidates = [{"time": 0.0, "iteration": 0, "kind": "initial_reference",
                       "f_plus_gap_plus_violation": 10}]
        raw = []
        for iteration, (location, loss, score) in enumerate([
                ([1.1, 0, 0], 0, 0), ([0.5, 0, 0], 3, 6),
                ([1.0, 0, 0], 2, 4), ([0, 0, 0], 0, 0)], start=1):
            for offset, kind in enumerate(("g", "F")):
                event = {"time": iteration - 0.1 + offset * 0.1,
                         "iteration": iteration, "kind": kind,
                         "f_value": loss, "f_plus_gap_plus_violation": score}
                candidates.append(event)
                raw.append({**event, "x": location})
        raw_path, run_path, audit_path = (root / name for name in
                                         ("raw.json.gz", "run.json", "audit.json.gz"))
        write_json(raw_path, {"data_sha256": "same", "events": raw})
        write_json(run_path, {"data_sha256": "same", "candidates": str(raw_path),
                             "records": [{"time": 0, "x": [0, 0, 0]}],
                             "settings": {"x_lower": [-1, -1, -1], "x_upper": [1, 1, 1],
                                          "time_budget": 3.0}})
        write_json(audit_path, {"run": str(run_path), "candidates": candidates,
                               "paper_output": None})
        return audit_path, raw_path

    def test_feasibility_filter_precedes_selection_and_preserves_raw_data(self):
        with TemporaryDirectory() as temporary:
            path, raw_path = self.reporting_fixture(Path(temporary))
            original_audit, original_raw = path.read_bytes(), raw_path.read_bytes()
            audit = eligible_audit(path)
            self.assertEqual(audit["reporting_filter"]["excluded_outside_domain"], 2)
            self.assertEqual(audit["reporting_filter"]["excluded_after_budget"], 2)
            grid = np.array([0., 1., 2., 3.])
            values, loss = paired_yf_selected_by_yg_loss(path, grid, audit=audit)
            np.testing.assert_array_equal(values, [10, 10, 6, 4])
            self.assertEqual(loss, 2)
            selected = reported_candidate(path, "bigap")
            self.assertEqual(selected["iteration"], 3)
            self.assertEqual(selected["kind"], "F")
            baseline, _ = trajectory(path, grid, None, audit=audit)
            np.testing.assert_array_equal(baseline, [10, 10, 6, 4])
            self.assertEqual(reported_candidate(path, "cmaes")["f_plus_gap_plus_violation"], 4)
            self.assertEqual(path.read_bytes(), original_audit)
            self.assertEqual(raw_path.read_bytes(), original_raw)

    def test_missing_coordinates_are_rejected(self):
        from macro_bigap import read_json
        with TemporaryDirectory() as temporary:
            path, raw_path = self.reporting_fixture(Path(temporary))
            raw = read_json(raw_path)
            raw["events"].pop(0)
            write_json(raw_path, raw)
            with self.assertRaisesRegex(ValueError, "matching recorded coordinates"):
                eligible_audit(path)

    def test_mismatched_data_are_rejected(self):
        from macro_bigap import read_json
        with TemporaryDirectory() as temporary:
            path, raw_path = self.reporting_fixture(Path(temporary))
            raw = read_json(raw_path)
            raw["data_sha256"] = "different"
            write_json(raw_path, raw)
            with self.assertRaisesRegex(ValueError, "different data"):
                eligible_audit(path)

    def test_g_response_cannot_improve_f_only_trajectory(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json.gz"
            write_json(path, {
                "candidates": [
                    {"time": 0.0, "kind": "initial_reference", "f_plus_gap_plus_violation": 10},
                    {"time": 0.5, "iteration": 1, "kind": "g", "f_value": 4,
                     "f_plus_gap_plus_violation": 1},
                    {"time": 1.0, "iteration": 1, "kind": "F", "f_plus_gap_plus_violation": 8},
                    {"time": 1.5, "iteration": 2, "kind": "g", "f_value": 3,
                     "f_plus_gap_plus_violation": 1},
                    {"time": 2.0, "iteration": 2, "kind": "F", "f_plus_gap_plus_violation": 6},
                    {"time": 2.5, "iteration": 3, "kind": "g", "f_value": 5,
                     "f_plus_gap_plus_violation": 1},
                    {"time": 3.0, "iteration": 3, "kind": "F", "f_plus_gap_plus_violation": 9},
                ],
                "paper_output": {"f_plus_gap_plus_violation": 7},
            })
            scores, output = trajectory(path, np.array([0.0, 0.5, 1.0, 2.0, 3.0]), "F")
            np.testing.assert_array_equal(scores, [10, 10, 8, 6, 6])
            current, _ = trajectory(path, np.array([0.0, 0.5, 1.0, 2.0, 3.0]), "F",
                                    best_so_far=False)
            np.testing.assert_array_equal(current, [10, 10, 8, 6, 9])
            selected, best_yg_loss = paired_yf_selected_by_yg_loss(
                path, np.array([0.0, 1.0, 2.0, 3.0])
            )
            np.testing.assert_array_equal(selected, [10, 8, 6, 6])
            self.assertEqual(best_yg_loss, 3)
            self.assertEqual(output["f_plus_gap_plus_violation"], 7)


if __name__ == "__main__":
    unittest.main()
