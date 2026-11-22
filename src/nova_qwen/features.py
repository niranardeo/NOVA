"""Deterministic feature-unit normalisation and data-quality notes.

CICFlowMeter reports Flow Duration in MICROSECONDS. Earlier tests showed Qwen reading raw values
as seconds (a ~101 s flow described as "3 years"), so all conversion happens here in Python and
the model only ever sees values already labelled with their unit. Raw values are kept unchanged
in the results for auditability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    raw_unit: str
    unit: str
    scale: Decimal  # value = raw * scale, computed exactly in decimal
    description: str
    integral: bool = False


FEATURES = {
    "Flow Duration": FeatureSpec(
        name="Flow Duration", raw_unit="microseconds", unit="seconds", scale=Decimal("0.000001"),
        description="time between the first and last packet of the flow"),
    "Total Fwd Packets": FeatureSpec(
        name="Total Fwd Packets", raw_unit="packets", unit="packets", scale=Decimal(1),
        description="number of packets sent in the forward direction", integral=True),
    "Flow Bytes/s": FeatureSpec(
        name="Flow Bytes/s", raw_unit="bytes per second", unit="bytes per second", scale=Decimal(1),
        description="bytes transferred per second over the whole flow"),
}


@dataclass(frozen=True)
class NormalizedFeature:
    feature: str
    raw_value: float
    raw_unit: str
    value: float
    unit: str
    display: str  # exactly the text shown to the model

    def to_record(self) -> dict:
        return {"feature": self.feature, "raw_value": self.raw_value, "raw_unit": self.raw_unit,
                "value": self.value, "unit": self.unit, "display": self.display}


def _plain(d: Decimal) -> str:
    """Decimal to plain notation (never scientific), without trailing zeros."""
    text = format(d, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def normalize_value(feature: str, raw_value) -> NormalizedFeature:
    spec = FEATURES[feature]
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)) or not math.isfinite(raw_value):
        raise ValueError(f"{feature}: {raw_value!r} is not a finite number")
    exact = Decimal(repr(raw_value)) * spec.scale
    if spec.integral and exact == exact.to_integral_value():
        display = _plain(exact.to_integral_value())
    else:
        # repr() keeps the shortest round-trip form of the exported float; scaling is exact.
        display = _plain(exact)
    return NormalizedFeature(feature=feature, raw_value=raw_value, raw_unit=spec.raw_unit,
                             value=float(exact), unit=spec.unit, display=display)


def normalize_evidence(evidence: dict) -> list:
    """Normalise every evidence value, in the contract's allowlist order."""
    return [normalize_value(name, evidence[name]) for name in FEATURES if name in evidence]


def data_quality_notes(normalized: list) -> list:
    """Deterministic plausibility checks. The model may only discuss data quality if one fires."""
    notes = []
    for f in normalized:
        if f.value < 0:
            notes.append(f"{f.feature} is negative ({f.display} {f.unit}); a negative {f.feature} is not "
                         f"physically meaningful and indicates a data-quality problem in the source record.")
        elif FEATURES[f.feature].integral and f.value != int(f.value):
            notes.append(f"{f.feature} is not a whole number ({f.display}); a packet count must be a whole "
                         f"number, which indicates a data-quality problem in the source record.")
    return notes
