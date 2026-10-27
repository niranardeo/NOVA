import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA_VERSION = "1.0"
NORMAL_LABEL = "Normal Traffic"

DETECTORS = {
    "decision_tree": "dt",
    "random_forest": "rf",
}

# Temporary evidence allowlist for development/testing.
# Final names must be verified against the processed CICIDS2017 dataset.
ALLOWED_EVIDENCE = (
    "Flow Duration",
    "Total Fwd Packets",
    "Flow Bytes/s",
)

def clean_value(value):
    """Validate and convert numeric evidence to JSON-compatible values."""

    if value is None or pd.isna(value):
        raise ValueError("Alert evidence contains missing values")

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float)
    ):
        raise ValueError("Alert evidence must be numeric")

    if not math.isfinite(value):
        raise ValueError("Alert evidence contains NaN or infinity")

    return value


def export_alerts(features, predictions, detector, output_path):
    """Export predicted attacks with their corresponding feature values."""

    if detector not in DETECTORS:
        raise ValueError(f"Unsupported detector: {detector}")

    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")

    # Reject ground-truth labels before processing any predictions.
    if "Attack Type" in features.columns:
        raise ValueError("Ground-truth column must not be exported")

    predictions = np.asarray(predictions)

    if predictions.ndim != 1 or len(features) != len(predictions):
        raise ValueError("Features and predictions must have matching rows")

    alerts = []
    prefix = DETECTORS[detector]

    for row_number, (prediction, (_, row)) in enumerate(
        zip(predictions, features.iterrows()), start=1
    ):
        if prediction == NORMAL_LABEL:
            continue

        available = [
            name for name in ALLOWED_EVIDENCE
            if name in features.columns
        ]

        if not available:
            raise ValueError("No approved evidence features available")

        evidence = {
            name: clean_value(row[name])
            for name in available
        }

        alerts.append({
            "alert_id": f"{prefix}-{row_number:06d}",
            "predicted_class": str(prediction),
            "evidence": evidence,
        })

    result = {
        "schema_version": SCHEMA_VERSION,
        "detector": detector,
        "alerts": alerts,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False
        ) as file:
            temp_path = Path(file.name)

            json.dump(
                result,
                file,
                indent=2,
                allow_nan=False
            )

            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, output)

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    print(f"{detector}: exported {len(alerts)} alerts to {output}")

    return result
