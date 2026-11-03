"""Write a small SYNTHETIC dataset shaped like V3, for the walkthrough and for tests. It is not CICIDS2017 data.

Writes outputs/synthetic_demo/: synthetic_train.csv, synthetic_validation.csv, dt_predictions.txt,
rf_predictions.txt and SYNTHETIC_DATA_NOT_CICIDS2017.txt. The rows are random numbers and the predictions are made
up, so nothing here says anything about classifier or Qwen quality. It can never pass as V3: the V3 check compares
real row and class counts. Needs pandas and numpy (requirements-qwen-dev.txt).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.nova_qwen.datasets import ALL_CLASSES, LABEL_COLUMN, V3_COLUMNS  # noqa: E402

# Share of rows per class: roughly V3's balance, with the rare classes raised so small files still contain them.
SHARES = {"Normal Traffic": 0.80, "DoS": 0.08, "DDoS": 0.05, "Port Scanning": 0.04, "Brute Force": 0.01,
          "Web Attacks": 0.01, "Bots": 0.01}
# Rough per-class centres, so a classifier trained on this file has something to learn.
DURATION_US = {"Normal Traffic": 2e5, "DoS": 8e7, "DDoS": 1.5e6, "Port Scanning": 60.0, "Brute Force": 5e6,
               "Web Attacks": 6e6, "Bots": 7e4}
PACKETS = {"Normal Traffic": 4, "DoS": 7, "DDoS": 3, "Port Scanning": 1, "Brute Force": 12, "Web Attacks": 9, "Bots": 2}
BANNER = ("SYNTHETIC DATA: random numbers shaped like the V3 columns, written by scripts/qwen_make_synthetic_demo.py.\n"
          "These are not CICIDS2017 records and not real classifier predictions. Never report results from them as V3.\n")


def make_frame(rows: int, rng: np.random.Generator) -> pd.DataFrame:
    labels = rng.choice(list(SHARES), size=rows, p=list(SHARES.values()))
    labels[:len(ALL_CLASSES)] = ALL_CLASSES  # every class present, even in tiny files
    rng.shuffle(labels)
    features = [c for c in V3_COLUMNS if c != LABEL_COLUMN]
    data = {c: np.round(rng.lognormal(3.0, 1.0, rows), 6) for c in features}
    centre = np.array([DURATION_US[x] for x in labels])
    duration = np.minimum(np.round(centre * rng.lognormal(0.0, 0.5, rows)), 119_000_000).astype(np.int64)
    duration[np.argmax(labels != "Normal Traffic")] = 110_000_000  # a long flow among the attacks
    packets = rng.poisson(np.array([PACKETS[x] for x in labels])) + 1
    data["Flow Duration"] = duration
    data["Total Fwd Packets"] = packets
    data["Flow Bytes/s"] = np.round(packets * 60 * 1e6 / np.maximum(duration, 1).astype(float), 6)
    data["Destination Port"] = rng.integers(1, 65535, rows)
    for i, cls in enumerate(ALL_CLASSES):  # one distinguishing column per class
        data[V3_COLUMNS[3 + i]] = np.where(labels == cls, data[V3_COLUMNS[3 + i]] + 500.0, data[V3_COLUMNS[3 + i]])
    frame = pd.DataFrame(data)[features]
    frame[LABEL_COLUMN] = labels
    return frame


def fake_predictions(labels: np.ndarray, rng: np.random.Generator, error_rate: float) -> np.ndarray:
    preds = labels.copy()
    wrong = rng.random(len(labels)) < error_rate
    preds[wrong] = rng.choice(list(SHARES), size=int(wrong.sum()))
    return preds


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "synthetic_demo")
    p.add_argument("--train-rows", type=int, default=4000)
    p.add_argument("--validation-rows", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    if a.train_rows < 100 or a.validation_rows < 100:
        p.error("--train-rows and --validation-rows must be at least 100")
    rng = np.random.default_rng(a.seed)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "SYNTHETIC_DATA_NOT_CICIDS2017.txt").write_text(BANNER, encoding="utf-8")
    make_frame(a.train_rows, rng).to_csv(a.out_dir / "synthetic_train.csv", index=False)
    validation = make_frame(a.validation_rows, rng)
    validation.to_csv(a.out_dir / "synthetic_validation.csv", index=False)
    labels = validation[LABEL_COLUMN].to_numpy(dtype=str)
    for name, error_rate in (("dt_predictions.txt", 0.02), ("rf_predictions.txt", 0.03)):
        preds = fake_predictions(labels, rng, error_rate)
        (a.out_dir / name).write_text("\n".join(preds) + "\n", encoding="utf-8")
    print(f"SYNTHETIC demo data written to {a.out_dir} (not CICIDS2017, not V3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
