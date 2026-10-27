import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alert_adapter import export_alerts


class TestAlertAdapter(unittest.TestCase):

    def setUp(self):
        self.features = pd.DataFrame({
            "Flow Duration": [100, 200, 300],
            "Total Fwd Packets": [5, 10, 20]
        })

    def test_decision_tree_alerts(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "dt.json"

            result = export_alerts(
                self.features,
                ["Normal Traffic", "DoS", "Normal Traffic"],
                "decision_tree",
                output
            )

            self.assertEqual(len(result["alerts"]), 1)
            self.assertEqual(result["alerts"][0]["alert_id"], "dt-000002")
            self.assertEqual(json.loads(output.read_text()), result)

    def test_random_forest_alerts(self):
        with tempfile.TemporaryDirectory() as folder:
            result = export_alerts(
                self.features,
                ["Normal Traffic", "DoS", "PortScan"],
                "random_forest",
                Path(folder) / "rf.json"
            )

            self.assertEqual(len(result["alerts"]), 2)
            self.assertEqual(result["detector"], "random_forest")

    def test_invalid_detector(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                export_alerts(
                    self.features,
                    ["DoS"] * 3,
                    "unknown",
                    Path(folder) / "invalid.json"
                )

    def test_mismatched_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                export_alerts(
                    self.features,
                    ["DoS"],
                    "decision_tree",
                    Path(folder) / "invalid.json"
                )

    def test_missing_or_infinite_evidence(self):
        for invalid_value in [np.nan, np.inf, -np.inf]:
            with self.subTest(value=invalid_value):
                with tempfile.TemporaryDirectory() as folder:
                    features = pd.DataFrame({
                        "Flow Duration": [invalid_value]
                    })

                    with self.assertRaises(ValueError):
                        export_alerts(
                            features,
                            ["DoS"],
                            "random_forest",
                            Path(folder) / "invalid.json"
                        )

    def test_unapproved_evidence_excluded(self):
        features = pd.DataFrame({
            "Flow Duration": [150],
            "Total Fwd Packets": [12],
            "Internal IP": ["192.168.1.100"],
            "Attack Type": ["DoS"]
        })

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                export_alerts(
                    features,
                    ["DoS"],
                    "random_forest",
                    Path(folder) / "alert.json"
                )

        features = features.drop(columns=["Attack Type"])

        with tempfile.TemporaryDirectory() as folder:
            result = export_alerts(
                features,
                ["DoS"],
                "random_forest",
                Path(folder) / "alert.json"
            )

            evidence = result["alerts"][0]["evidence"]

            self.assertIn("Flow Duration", evidence)
            self.assertIn("Total Fwd Packets", evidence)
            self.assertNotIn("Internal IP", evidence)
            self.assertNotIn("Attack Type", evidence)
    def test_ground_truth_rejected_with_zero_alerts(self):
        features = pd.DataFrame({
            "Flow Duration": [100],
            "Attack Type": ["Normal Traffic"]
        })

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "test.json"

            with self.assertRaisesRegex(
                ValueError, "Ground-truth column"
            ):
                export_alerts(
                    features,
                    ["Normal Traffic"],
                    "decision_tree",
                    output
                )

            self.assertFalse(output.exists())

    def test_none_evidence_rejected(self):
        features = pd.DataFrame({
            "Flow Duration": [None],
            "Total Fwd Packets": [5]
        })

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "invalid.json"

            with self.assertRaises(ValueError):
                export_alerts(
                    features,
                    ["DoS"],
                    "decision_tree",
                    output
                )

            self.assertFalse(output.exists())

    def test_nonnumeric_evidence_rejected(self):
        for invalid_value in ["unknown", "100", True]:
            with self.subTest(value=invalid_value):
                with tempfile.TemporaryDirectory() as folder:
                    features = pd.DataFrame({
                        "Flow Duration": [invalid_value],
                        "Total Fwd Packets": [5]
                    })

                    output = Path(folder) / "invalid.json"

                    with self.assertRaises(ValueError):
                        export_alerts(
                            features,
                            ["DoS"],
                            "decision_tree",
                            output
                        )

                    self.assertFalse(output.exists())

    def test_failed_json_write_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "alerts.json"
            original = '{"status": "previous valid export"}'
            output.write_text(original, encoding="utf-8")

            with patch(
                "alert_adapter.json.dump",
                side_effect=OSError("Simulated write failure")
            ):
                with self.assertRaises(OSError):
                    export_alerts(
                        self.features,
                        ["Normal Traffic", "DoS", "Normal Traffic"],
                        "decision_tree",
                        output
                    )

            self.assertEqual(
                output.read_text(encoding="utf-8"),
                original
            )

if __name__ == "__main__":
    unittest.main()
