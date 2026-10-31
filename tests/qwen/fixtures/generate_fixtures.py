"""Regenerate the synthetic schema 1.1 fixtures with the team's real adapter (src/alert_adapter.py).

Synthetic rows (not CICIDS2017 records). Several values reproduce earlier Qwen failure cases:
row 2 = 101,041,626 us (~101 s, once explained as "3.18 years"), row 8 = 116,827,162 us
("3.74 years"), row 5 = negative duration (data-quality case), row 4 = DT/RF class disagreement.

Usage from the repository root (needs pandas and numpy):
    python tests/qwen/fixtures/generate_fixtures.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.alert_adapter import export_alerts  # noqa: E402  (team code, unmodified)

HERE = Path(__file__).resolve().parent
RUN_ID = "validation-20261030T000000000000Z"
DATASET = "CICIDS2017"

features = pd.DataFrame({
    "Flow Duration": [5_000, 101_041_626, 67, 21_760, -1, 5_000_000, 12_000, 116_827_162],
    "Total Fwd Packets": [3, 11, 1, 1, 1, 20, 4, 2],
    "Flow Bytes/s": [1200.0, 123.7509776, 119402.9851, 91130.51471, 0.0, 2400.5, 350.25, 0.154073759],
    "Destination Port": [443, 80, 22, 80, 8080, 21, 53, 6667],  # not allowlisted: must never be exported
})
dt_predictions = ["Normal Traffic", "DoS", "Normal Traffic", "DDoS", "Web Attacks", "Brute Force",
                  "Normal Traffic", "Bots"]
rf_predictions = ["Normal Traffic", "DoS", "Port Scanning", "DoS", "Normal Traffic", "Brute Force",
                  "Normal Traffic", "Bots"]

if __name__ == "__main__":
    export_alerts(features, dt_predictions, "decision_tree", HERE / "decision_tree_alerts.json",
                  run_id=RUN_ID, dataset=DATASET)
    export_alerts(features, rf_predictions, "random_forest", HERE / "random_forest_alerts.json",
                  run_id=RUN_ID, dataset=DATASET)
    normal = ["Normal Traffic"] * len(features)
    export_alerts(features, normal, "decision_tree", HERE / "decision_tree_alerts_empty.json",
                  run_id=RUN_ID, dataset=DATASET)
    export_alerts(features, normal, "random_forest", HERE / "random_forest_alerts_empty.json",
                  run_id=RUN_ID, dataset=DATASET)
