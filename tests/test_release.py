"""Check that portable result exports reproduce the retained endpoints."""

import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dreamplace"))

from run_mms_confirm_50s import eligible_audit, reported_candidate
from openroad.comparison import penalty_values, read_json as read_rc
from release_archive import unpack_bundle


class ReleaseTests(unittest.TestCase):
    def test_prepared_data_hash_and_dimensions(self):
        root = ROOT / "dreamplace/data"
        manifest = json.loads((root / "export_manifest.json").read_text())
        content = (root / "adaptec1_dreamplace_gpu.json.gz").read_bytes()
        self.assertEqual(hashlib.sha256(content).hexdigest(), manifest["released_prepared_sha256"])
        data = json.loads(gzip.decompress(content))
        self.assertEqual(len(data["names"]), 210967)
        self.assertEqual(len(data["macro_indices"]), 63)
        self.assertEqual(len(data["anchors"]), 8005)

    def test_held_out_endpoints_are_preserved(self):
        with unpack_bundle(ROOT / "dreamplace/results_bundle.tar.gz") as extracted:
            root = extracted / "results/dreamplace_adaptec1_confirm_50s_v1"
            summary = json.loads((root / "summary.json").read_text())
            with patch.dict(os.environ, {"EDA_RESULTS_DIR": str(extracted / "results")}):
                for method, record in summary["methods"].items():
                    values = []
                    for seed in range(8, 16):
                        path = root / f"{method}_seed{seed}.audit.json.gz"
                        candidate = reported_candidate(path, method, audit=eligible_audit(path))
                        values.append(candidate["f_plus_gap_plus_violation"])
                    np.testing.assert_allclose(values, record["per_seed"], rtol=0, atol=1e-10)

    def test_openroad_audits_have_common_instance_and_valid_penalties(self):
        with unpack_bundle(ROOT / "openroad/results_bundle.tar.gz") as extracted:
            root = extracted / "results/comparison_300s"
            paths = sorted(root.glob("*.audit.json.gz"))
            self.assertEqual(len(paths), 80)
            expected = hashlib.sha256((ROOT / "openroad/data/gcd_instance.json").read_bytes()).hexdigest()
            for path in paths:
                result = read_rc(path)
                self.assertEqual(result["instance_sha256"], expected)
                response = "F" if result["method"] == "bigap" else "g"
                self.assertTrue(np.all(np.isfinite(penalty_values(result["history"], response, result["gamma"]))))
                self.assertLessEqual(result["history"][-1]["seconds"], 300)

    def test_exploratory_endpoints_are_preserved(self):
        with unpack_bundle(ROOT / "dreamplace/results_bundle.tar.gz") as extracted:
            root = extracted / "results"
            summary = json.loads((root / "dreamplace_adaptec1_300s_feasible/summary.json").read_text())
            with patch.dict(os.environ, {"EDA_RESULTS_DIR": str(root)}):
                for name, record in summary["results"].items():
                    method, gamma_text = name.split("_gamma")
                    gamma = float(gamma_text)
                    directory = root / "dreamplace_adaptec1_v1"
                    if method == "bigap" and gamma != 0.01:
                        directory = root / "dreamplace_adaptec1_gamma_yf_v1" / ("gamma_" + str(gamma).replace(".", "p"))
                    values = [reported_candidate(directory / f"{method}_seed{seed}.audit.json.gz", method)
                              ["f_plus_gap_plus_violation"] for seed in range(8)]
                    np.testing.assert_allclose(values, record["per_seed"], rtol=0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
