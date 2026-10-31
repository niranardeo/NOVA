"""Strict reader for the classifier alert JSON contract, schema 1.1 (docs/alert_schema.md).

A file is validated completely before anything is sent to the model. Identifiers are taken
from the file and carried in code; the model is never asked to produce them.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .errors import (AlertContractError, AlertFileInvalidJSON, AlertFileMissing, AlertFileUnreadable,
                     UnsupportedSchemaVersion)

SUPPORTED_SCHEMA_VERSION = "1.1"
DETECTOR_PREFIX = {"decision_tree": "dt", "random_forest": "rf"}
NORMAL_LABEL = "Normal Traffic"
# The six attack classes present in the team's processed CICIDS2017 data.
ATTACK_CLASSES = ("DoS", "DDoS", "Port Scanning", "Brute Force", "Web Attacks", "Bots")
# Provisional evidence allowlist from docs/alert_schema.md.
ALLOWED_EVIDENCE = ("Flow Duration", "Total Fwd Packets", "Flow Bytes/s")
TOP_LEVEL_KEYS = frozenset({"schema_version", "run_id", "dataset", "detector", "alerts"})
ALERT_KEYS = frozenset({"alert_id", "record_id", "predicted_class", "evidence"})
ALERT_ID_RE = re.compile(r"(dt|rf)-([0-9]{6,})")  # ASCII digits only (\d also matches e.g. Arabic-Indic digits)
RECORD_ID_RE = re.compile(r"val-([0-9]{6,})")
MAX_ABS_EVIDENCE = 1e15  # far above any CICFlowMeter value; also keeps every value exactly representable as float


@dataclass(frozen=True)
class Alert:
    alert_id: str
    record_id: str
    record_position: int  # one-based row position in the supplied validation frame
    predicted_class: str
    evidence: dict


@dataclass(frozen=True)
class AlertExport:
    file_name: str
    sha256: str
    size_bytes: int
    schema_version: str
    run_id: str
    dataset: str
    detector: str
    alerts: tuple = field(default_factory=tuple)

    def source_record(self) -> dict:
        """Provenance block written into every results file (no absolute paths)."""
        return {
            "file_name": self.file_name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "dataset": self.dataset,
            "detector": self.detector,
            "alert_count": len(self.alerts),
        }


def _reject_constant(token):
    raise ValueError(f"non-standard JSON constant {token}")


def _no_duplicate_keys(pairs):
    keys = [k for k, _ in pairs]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise ValueError(f"duplicate keys {duplicates}")
    return dict(pairs)


def load_export(path, expected_detector: str | None = None) -> AlertExport:
    """Read and fully validate one schema 1.1 alert export."""
    path = Path(path)
    if not path.exists():
        raise AlertFileMissing(f"alert export not found: {path}")
    if not path.is_file():
        raise AlertFileUnreadable(f"alert export is not a regular file: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AlertFileUnreadable(f"cannot read alert export {path}: {exc}") from exc
    try:
        doc = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant,
                         object_pairs_hook=_no_duplicate_keys)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise AlertFileInvalidJSON(f"{path.name} is not valid JSON: {type(exc).__name__}: {exc}") from exc
    return parse_export(doc, file_name=path.name, sha256=hashlib.sha256(raw).hexdigest(),
                        size_bytes=len(raw), expected_detector=expected_detector)


def parse_export(doc, *, file_name: str = "<memory>", sha256: str = "", size_bytes: int = 0,
                 expected_detector: str | None = None) -> AlertExport:
    """Validate an already-parsed export document against schema 1.1."""
    if not isinstance(doc, dict):
        raise AlertContractError("top level must be a JSON object", code="CONTRACT_TOP_LEVEL")
    version = doc.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise UnsupportedSchemaVersion(
            f"schema_version {version!r} is not supported (expected {SUPPORTED_SCHEMA_VERSION!r})")
    keys = set(doc)
    if keys != TOP_LEVEL_KEYS:
        raise AlertContractError(
            f"top-level keys must be exactly {sorted(TOP_LEVEL_KEYS)}; missing {sorted(TOP_LEVEL_KEYS - keys)}, "
            f"unexpected {sorted(keys - TOP_LEVEL_KEYS)}", code="CONTRACT_TOP_LEVEL_KEYS")
    run_id = _identifier(doc["run_id"], "run_id")
    dataset = _identifier(doc["dataset"], "dataset")
    detector = doc["detector"]
    if not isinstance(detector, str) or detector not in DETECTOR_PREFIX:
        raise AlertContractError(f"unsupported detector {detector!r}", code="CONTRACT_DETECTOR")
    if expected_detector is not None and detector != expected_detector:
        raise AlertContractError(f"file is for detector {detector!r}, expected {expected_detector!r}",
                                 code="CONTRACT_DETECTOR_MISMATCH")
    if not isinstance(doc["alerts"], list):
        raise AlertContractError("alerts must be a JSON array", code="CONTRACT_ALERTS_TYPE")

    # record_id must name the same row as alert_id (checked per alert), so unique alert ids imply
    # unique record ids.
    alerts, seen_alert_ids = [], set()
    for index, item in enumerate(doc["alerts"]):
        alert = _parse_alert(item, index, detector)
        if alert.alert_id in seen_alert_ids:
            raise AlertContractError(f"duplicate alert_id {alert.alert_id}", code="CONTRACT_DUPLICATE_ALERT_ID")
        seen_alert_ids.add(alert.alert_id)
        alerts.append(alert)
    return AlertExport(file_name=file_name, sha256=sha256, size_bytes=size_bytes, schema_version=version,
                       run_id=run_id, dataset=dataset, detector=detector, alerts=tuple(alerts))


def _identifier(value, name: str) -> str:
    """run_id / dataset: non-empty strings that can be joined on reliably.

    Stricter than the adapter (which accepts any non-blank string): surrounding whitespace and
    control/format/separator characters are rejected because they make exact joins fail silently.
    """
    if not isinstance(value, str) or not value.strip():
        raise AlertContractError(f"{name} must be a non-empty string", code=f"CONTRACT_{name.upper()}")
    if value != value.strip():
        raise AlertContractError(f"{name} has leading or trailing whitespace: {value!r}",
                                 code=f"CONTRACT_{name.upper()}")
    bad = [c for c in value if unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp", "Cs")]
    if bad:
        raise AlertContractError(f"{name} contains control or invisible characters: {value!r}",
                                 code=f"CONTRACT_{name.upper()}")
    return value


def _parse_alert(item, index: int, detector: str) -> Alert:
    where = f"alerts[{index}]"
    if not isinstance(item, dict):
        raise AlertContractError(f"{where} must be an object", code="CONTRACT_ALERT_TYPE")
    keys = set(item)
    if keys != ALERT_KEYS:
        raise AlertContractError(
            f"{where} keys must be exactly {sorted(ALERT_KEYS)}; missing {sorted(ALERT_KEYS - keys)}, "
            f"unexpected {sorted(keys - ALERT_KEYS)}", code="CONTRACT_ALERT_KEYS")
    alert_id, record_id = item["alert_id"], item["record_id"]
    m = ALERT_ID_RE.fullmatch(alert_id) if isinstance(alert_id, str) else None
    if not m or m.group(1) != DETECTOR_PREFIX[detector]:
        raise AlertContractError(f"{where}.alert_id {alert_id!r} does not match detector {detector}",
                                 code="CONTRACT_ALERT_ID")
    r = RECORD_ID_RE.fullmatch(record_id) if isinstance(record_id, str) else None
    if not r:
        raise AlertContractError(f"{where}.record_id {record_id!r} is not 'val-' plus digits",
                                 code="CONTRACT_RECORD_ID")
    position = int(r.group(1))
    if position < 1 or int(m.group(2)) != position:
        raise AlertContractError(f"{where}: alert_id {alert_id} and record_id {record_id} name different rows",
                                 code="CONTRACT_ID_POSITION_MISMATCH")
    # The adapter writes f"{prefix}-{row:06d}" / f"val-{row:06d}"; any other spelling of the same row
    # (e.g. extra zero padding) would let one row appear twice and break exact joins.
    if alert_id != f"{DETECTOR_PREFIX[detector]}-{position:06d}" or record_id != f"val-{position:06d}":
        raise AlertContractError(f"{where}: ids {alert_id!r}/{record_id!r} are not in canonical form",
                                 code="CONTRACT_ID_NOT_CANONICAL")
    predicted = item["predicted_class"]
    if not isinstance(predicted, str) or predicted == NORMAL_LABEL:
        raise AlertContractError(f"{where}.predicted_class {predicted!r} is not an attack prediction",
                                 code="CONTRACT_PREDICTED_CLASS")
    if predicted not in ATTACK_CLASSES:
        raise AlertContractError(f"{where}.predicted_class {predicted!r} is not one of {list(ATTACK_CLASSES)}",
                                 code="CONTRACT_UNKNOWN_CLASS")
    evidence = item["evidence"]
    if not isinstance(evidence, dict) or not evidence:
        raise AlertContractError(f"{where}.evidence must be a non-empty object", code="CONTRACT_EVIDENCE_EMPTY")
    unknown = sorted(set(evidence) - set(ALLOWED_EVIDENCE))
    if unknown:
        raise AlertContractError(f"{where}.evidence has non-allowlisted features {unknown}",
                                 code="CONTRACT_EVIDENCE_FEATURE")
    for name, value in evidence.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AlertContractError(f"{where}.evidence[{name!r}] = {value!r} is not a number",
                                     code="CONTRACT_EVIDENCE_TYPE")
        if isinstance(value, float) and not math.isfinite(value):
            raise AlertContractError(f"{where}.evidence[{name!r}] is not finite", code="CONTRACT_EVIDENCE_FINITE")
        if abs(value) > MAX_ABS_EVIDENCE:
            raise AlertContractError(f"{where}.evidence[{name!r}] = {value!r} is out of range (|x| > 1e15)",
                                     code="CONTRACT_EVIDENCE_RANGE")
    return Alert(alert_id=alert_id, record_id=record_id, record_position=position,
                 predicted_class=predicted, evidence=dict(evidence))
