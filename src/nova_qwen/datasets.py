"""Dataset identity, the reserved-split guard, and export-to-CSV row alignment. Standard library only.

A file is not treated as V3 because of its name: rows, columns, class counts and evidence columns are read
from the file and compared with Kevin's notebooks (03_..._V3_Creation, 04_..._V3_Stratified_Split). That is a
profile match only: fabricated values or a different split with the same counts would match too. Those notebooks
record no file hash, so a file counts as verified only when it matches a SHA-256 the team has published.

The reserved final test split is never opened: paths are refused by name before any file access (reserved.py).
"""
from __future__ import annotations

import csv
import hashlib
import math
import re
from contextlib import contextmanager
from pathlib import Path

from .alerts import ALLOWED_EVIDENCE, ATTACK_CLASSES, NORMAL_LABEL, AlertExport
from .errors import DatasetError, DatasetMissing, NovaQwenError
from .reserved import is_reserved_name, refuse_reserved  # noqa: F401  (re-exported for the scripts)

V3 = "V3"
LABEL_COLUMN = "Attack Type"
ALL_CLASSES = (NORMAL_LABEL, *ATTACK_CLASSES)
V3_TRAIN_FILE = "cicids2017_V3_train_70.csv"
V3_VALIDATION_FILE = "cicids2017_V3_validation_15.csv"
V3_TRAIN_ROWS = 1_763_436
V3_VALIDATION_ROWS = 377_880
# Column names and order as printed by Kevin's notebook 03 (cell 4); V3 only removes rows from V2.
V3_COLUMNS = (
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
    "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "PSH Flag Count", "ACK Flag Count", "Average Packet Size", "Subflow Fwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Max", "Active Min",
    "Idle Mean", "Idle Max", "Idle Min", LABEL_COLUMN,
)
# Derived from the class percentages stored in notebook 04 (cell 7) and the split sizes; each sums exactly.
# The notebook's reserved partition differs from validation here, so a renamed test file fails this check.
V3_TRAIN_CLASSES = {"Normal Traffic": 1_465_939, "DoS": 135_531, "DDoS": 89_607, "Port Scanning": 63_090,
                    "Brute Force": 6_405, "Web Attacks": 1_500, "Bots": 1_364}
V3_VALIDATION_CLASSES = {"Normal Traffic": 314_130, "DoS": 29_042, "DDoS": 19_202, "Port Scanning": 13_520,
                         "Brute Force": 1_372, "Web Attacks": 322, "Bots": 292}
UNIT_ASSUMPTIONS = {"Flow Duration": "microseconds", "Total Fwd Packets": "packets",
                    "Flow Bytes/s": "bytes per second"}
UNIT_AUTHORITY = ("Plausibility only, not proof of the units. V3 only removes rows from V2 (notebook 03), and "
                  "docs/alert_schema.md states these units but asks for confirmation against V3 preprocessing "
                  "documentation.")
UNIT_BASIS = {"Flow Duration": "whole numbers whose maximum is between 1 s and 130 s when read as microseconds",
              "Total Fwd Packets": "non-negative whole numbers",
              "Flow Bytes/s": "numeric and finite only; this says nothing about the unit"}
FLOW_DURATION_PLAUSIBLE_MAX = 130_000_000  # CICFlowMeter's 120 s flow timeout, in microseconds, plus a margin
FLOW_DURATION_PLAUSIBLE_MIN_OF_MAX = 1_000_000  # a whole dataset whose longest flow is under 1 s is not microseconds
FLOAT_TOLERANCE = 1e-12  # pandas' default CSV float parser can differ from float(text) in the last digits
MAX_ALERT_FREE_TAIL_CHANCE = 1e-6  # V3 rows are shuffled, so a tail this unlikely for this many alerts suggests a cut-off export


def require_version_spelling(version: str) -> str:
    """A blank version or a near spelling of V3 would silently skip the V3 checks, so refuse them."""
    if not version.strip():
        raise NovaQwenError("--dataset-version must not be empty", code="BAD_ARGUMENTS")
    if version != V3 and V3 in re.sub(r"[^A-Z0-9]", "", version.upper()):
        raise NovaQwenError(f"--dataset-version {version!r}: to check against V3, write exactly V3", code="BAD_ARGUMENTS")
    return version


@contextmanager
def _readable(path):
    """A dataset file that cannot be read or parsed is invalid input (exit 2), not a crash."""
    try:
        yield
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise DatasetError(f"cannot read dataset file {path}: {type(exc).__name__}: {exc}",
                           code="DATASET_UNREADABLE") from exc


def _data_rows(reader):
    """Rows as pandas read_csv numbers them: it skips empty and whitespace-only lines, and record_id counts its rows.

    A lone quoted empty field gives [""], and pandas keeps it as a row. A lone quoted " " looks the same as an unquoted
    one through csv.reader, so that case alone is still skipped here while pandas keeps it."""
    return (row for row in reader if row and not (len(row) == 1 and row[0] and not row[0].strip(" \t")))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _new_stats() -> dict:
    return {"min": None, "max": None, "negative": 0, "non_integer": 0, "non_finite": 0, "non_numeric": 0, "missing": 0}


def _update_stats(stats: dict, text: str) -> None:
    if text == "":
        stats["missing"] += 1
        return
    try:
        value = float(text)
    except ValueError:
        stats["non_numeric"] += 1
        return
    if not math.isfinite(value):
        stats["non_finite"] += 1
        return
    stats["min"] = value if stats["min"] is None else min(stats["min"], value)
    stats["max"] = value if stats["max"] is None else max(stats["max"], value)
    stats["negative"] += value < 0
    stats["non_integer"] += value != int(value)


def inspect_csv(path) -> dict:
    """Stream one split CSV (memory stays small on a 1.76M-row file) and describe it. Reads nothing else."""
    path = refuse_reserved(path)
    if not path.is_file():
        raise DatasetMissing(f"dataset file not found: {path}")
    labels: dict = {}
    stats = {name: _new_stats() for name in ALLOWED_EVIDENCE}
    rows = malformed = 0
    with _readable(path):
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            header = next(_data_rows(reader), [])
            index = {name: i for i, name in enumerate(header)}
            label_at = index.get(LABEL_COLUMN)
            evidence_at = [(stats[name], index[name]) for name in ALLOWED_EVIDENCE if name in index]
            for row in _data_rows(reader):
                rows += 1
                if len(row) != len(header):
                    malformed += 1
                    continue
                if label_at is not None:
                    labels[row[label_at]] = labels.get(row[label_at], 0) + 1
                for target, at in evidence_at:
                    _update_stats(target, row[at])
        return {"file_name": path.name, "path": str(path), "resolved_path": str(path.resolve()),
                "is_symlink": path.is_symlink(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
                "rows": rows, "columns": len(header), "header": header, "malformed_rows": malformed,
                "label_distribution": dict(sorted(labels.items())), "evidence": stats}


def assess_units(info: dict) -> dict:
    """Are the evidence columns consistent with the units the normaliser assumes? Plausibility, not proof."""
    ev = info["evidence"]

    def clean(s):
        return not (s["missing"] or s["non_numeric"] or s["non_finite"])

    fd, pk, fb = ev["Flow Duration"], ev["Total Fwd Packets"], ev["Flow Bytes/s"]
    flow_max = fd["max"]
    flow_ok = bool(clean(fd) and not fd["non_integer"] and flow_max is not None
                   and FLOW_DURATION_PLAUSIBLE_MIN_OF_MAX < flow_max <= FLOW_DURATION_PLAUSIBLE_MAX)
    return {"assumed_units": UNIT_ASSUMPTIONS, "authority": UNIT_AUTHORITY, "basis": UNIT_BASIS,
            "consistent": {"Flow Duration": flow_ok,
                           "Total Fwd Packets": bool(clean(pk) and not pk["non_integer"] and not pk["negative"]),
                           "Flow Bytes/s": clean(fb)},
            "negative_flow_durations": fd["negative"]}


def _check_file(info: dict, role: str, version: str, expected: dict | None, expected_sha256: str | None) -> list:
    problems = []
    name = info["file_name"]
    header = info["header"]
    if LABEL_COLUMN not in header:
        problems.append(f"{role}: label column {LABEL_COLUMN!r} is missing")
    missing = [c for c in ALLOWED_EVIDENCE if c not in header]
    if missing:
        problems.append(f"{role}: evidence columns missing: {missing}")
    if info["malformed_rows"]:
        problems.append(f"{role}: {info['malformed_rows']} rows do not have {len(header)} fields")
    unknown = sorted(set(info["label_distribution"]) - set(ALL_CLASSES))
    if unknown:
        problems.append(f"{role}: unexpected class labels {unknown}")
    if not info["rows"]:
        problems.append(f"{role}: the file has no data rows")
    if expected_sha256 and info["sha256"] != expected_sha256.strip().lower():
        problems.append(f"{role}: sha256 {info['sha256']} is not the expected {expected_sha256}")
    for feature, stats in info["evidence"].items():
        if stats["missing"] or stats["non_numeric"] or stats["non_finite"]:
            problems.append(f"{role}: evidence column {feature!r} has missing or non-numeric values: {stats}")
    if version == V3:
        if name != expected["file"]:
            problems.append(f"{role}: file name {name!r} is not {expected['file']!r}")
        if info["rows"] != expected["rows"]:
            problems.append(f"{role}: {info['rows']} rows, V3 has {expected['rows']}")
        if tuple(header) != V3_COLUMNS:
            problems.append(f"{role}: columns differ from V3 ({len(header)} columns, expected {len(V3_COLUMNS)})")
        if info["label_distribution"] != dict(sorted(expected["classes"].items())):
            problems.append(f"{role}: class counts {info['label_distribution']} differ from V3 {expected['classes']}")
        if header and not all(assess_units(info)["consistent"].values()):
            problems.append(f"{role}: evidence columns are not consistent with the assumed units: "
                            f"{assess_units(info)['consistent']}")
    return problems


def check_dataset(*, validation, train=None, version: str = V3, expected_train_sha256: str | None = None,
                  expected_validation_sha256: str | None = None) -> dict:
    """Identify the training/validation files. ``train`` may be None: the Qwen stage reads only validation rows.

    v3_profile_matched: rows, columns, class counts and evidence columns match Kevin's notebooks.
    v3_validation_verified / v3_identity_verified: as above, and the file hash (or both) matches a published hash."""
    if validation is None:
        raise DatasetMissing("a validation file is required")
    profiles = {"train": {"file": V3_TRAIN_FILE, "rows": V3_TRAIN_ROWS, "classes": V3_TRAIN_CLASSES},
                "validation": {"file": V3_VALIDATION_FILE, "rows": V3_VALIDATION_ROWS,
                               "classes": V3_VALIDATION_CLASSES}}
    report = {"report": "nova-qwen-dataset-check", "dataset_version": version, "checked_against_v3_profile": version == V3,
              "train": None, "validation": None, "problems": [],
              "hash_pinned": {"validation": bool(expected_validation_sha256), "train": bool(expected_train_sha256)}}
    problems = report["problems"]
    headers = {}
    for role, path, expected_sha in (("validation", validation, expected_validation_sha256),
                                     ("train", train, expected_train_sha256)):
        if path is None:
            if expected_sha:
                problems.append(f"{role}: an expected sha256 was given, but there is no {role} file to compare")
            continue
        info = inspect_csv(path)
        problems += _check_file(info, role, version, profiles[role], expected_sha)
        headers[role] = info["header"]
        report[role] = {**{k: v for k, v in info.items() if k != "header"}, "units": assess_units(info)}
    if report["train"] and headers["train"] != headers["validation"]:
        problems.append("training and validation columns differ")
    report["passed"] = not problems
    matched = version == V3 and not problems
    report["v3_profile_matched"] = matched
    report["v3_validation_verified"] = matched and bool(expected_validation_sha256)
    report["v3_identity_verified"] = report["v3_validation_verified"] and report["train"] is not None \
        and bool(expected_train_sha256)
    return report


def check_alignment(export: AlertExport, validation_path, *, expected_rows: int | None = None,
                    require_alerts: bool = False) -> dict:
    """Does every alert's evidence equal the validation CSV at its record_id?

    record_id is a one-based row position, so this catches an export made from a filtered, sampled or reordered frame,
    or from a different file, unless the rows removed all come after the last alert. Rows without an alert cannot be
    compared, so a pass does not prove the export covered the complete frame. With expected_rows set (the V3 check), a
    alert-free tail too long to be chance for that many alerts is also a problem, because V3's rows are shuffled.
    require_alerts makes an export with no alerts a problem too. Labels are not read."""
    path = refuse_reserved(validation_path)
    if not path.is_file():
        raise DatasetMissing(f"validation file not found: {path}")
    wanted = {a.record_position: a for a in export.alerts}
    features = sorted({f for a in export.alerts for f in a.evidence})
    mismatches, problems = [], []
    mismatch_count = checked = rows = 0
    with _readable(path):
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            header = next(_data_rows(reader), [])
            index = {name: i for i, name in enumerate(header)}
            absent = [f for f in features if f not in index]
            if absent:
                problems.append(f"evidence features not in {path.name}: {absent}")
            for rows, row in enumerate(_data_rows(reader), start=1):
                alert = wanted.get(rows)
                if alert is None or absent:
                    continue
                checked += 1
                for feature, value in alert.evidence.items():
                    try:
                        same = math.isclose(float(row[index[feature]]), float(value), rel_tol=FLOAT_TOLERANCE,
                                            abs_tol=FLOAT_TOLERANCE)
                    except (ValueError, IndexError):
                        same = False
                    if not same:
                        mismatch_count += 1
                        if len(mismatches) < 5:
                            cell = row[index[feature]] if index[feature] < len(row) else None
                            mismatches.append({"alert_id": alert.alert_id, "feature": feature, "export_value": value,
                                               "csv_text": cell})
        sha256 = sha256_file(path)
    out_of_range = sorted(p for p in wanted if p > rows)
    if out_of_range:
        problems.append(f"{len(out_of_range)} alerts point past the last row ({rows}), e.g. position {out_of_range[0]}")
    if mismatch_count:
        problems.append(f"{mismatch_count} evidence values differ from the CSV at their record_id: the export was not "
                        f"made from this file's complete rows in order")
    if expected_rows is not None and rows != expected_rows:
        problems.append(f"{path.name} has {rows} rows, expected {expected_rows}")
    last_alert = max(wanted, default=None)
    trailing = None if last_alert is None else max(rows - last_alert, 0)
    if expected_rows is not None and trailing and (1 - trailing / rows) ** len(wanted) < MAX_ALERT_FREE_TAIL_CHANCE:
        problems.append(f"no alerts in the last {trailing} of {rows} rows, too long to be chance for {len(wanted)} alerts "
                        f"in shuffled rows: the export may come from a truncated frame")
    if require_alerts and last_alert is None:
        problems.append(f"the export has no alerts, so nothing was compared with the {rows} rows: it may come from an "
                        f"empty or filtered frame, or a broken model")
    return {"report": "nova-qwen-alignment-check", "export": export.source_record(), "validation_file": path.name,
            "validation_path": str(path), "validation_sha256": sha256, "validation_rows": rows,
            "expected_rows": expected_rows, "alerts_checked": checked, "evidence_values_differing": mismatch_count,
            "last_alert_position": last_alert, "rows_after_last_alert": trailing,
            "frame_completeness": "not proven by this check: rows without an alert cannot be compared",
            "examples": mismatches, "problems": problems, "passed": not problems}
