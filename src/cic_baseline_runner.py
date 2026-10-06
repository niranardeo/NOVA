"""CICIDS2017 machine-learning baseline experiment runner (train / validation only).

Trains DecisionTreeClassifier and RandomForestClassifier baselines on Kevin's
official 70% training split and evaluates them on the official 15% validation
split. The final 15% test split is intentionally NOT used by this script.

For every model the runner produces:
  * seven-class (Attack Type) metrics, classification report and confusion matrix
  * a secondary Normal-vs-Attack interpretation derived from the seven-class
    predictions (no separate binary model is trained)
  * a per-record validation prediction export
  * an errors-only export (with the original 52 features) intended as input to
    the later local-AI contextualization stage

Optional --audit-overlap adds a reporting-only exact audit of duplicates and
train/validation feature overlap (including label conflicts) and annotates the
validation predictions. It never removes, relabels or excludes any record.

Decision Tree / Random Forest here are ML baseline classifiers. They are not the
project's traditional signature/rule-based IDS.

See docs/cic_baseline_runner.md for usage and output descriptions.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger("cic_baseline_runner")

# ---------------------------------------------------------------------------
# Dataset expectations (from Kevin's preprocessing notebook,
# 02_data_preprocessing_Oct_06_2026_Kevin.ipynb)
# ---------------------------------------------------------------------------
DEFAULT_TARGET_COLUMN = "Attack Type"
EXPECTED_FEATURE_COUNT = 52
EXPECTED_CLASSES = (
    "Normal Traffic",
    "DoS",
    "DDoS",
    "Port Scanning",
    "Brute Force",
    "Web Attacks",
    "Bots",
)

# Normal-vs-Attack interpretation of the seven-class labels.
NORMAL_CLASS = "Normal Traffic"
NORMAL_LABEL = "NORMAL"
ATTACK_LABEL = "ATTACK"  # positive class for binary metrics

TRUE_POSITIVE = "true_positive"
TRUE_NEGATIVE = "true_negative"
FALSE_POSITIVE = "false_positive"
FALSE_NEGATIVE = "false_negative"

ERROR_BINARY_FALSE_POSITIVE = "binary_false_positive"
ERROR_BINARY_FALSE_NEGATIVE = "binary_false_negative"
ERROR_ATTACK_TO_ATTACK = "attack_to_attack_misclassification"

RUN_MODE_FULL = "FULL"
RUN_MODE_SMOKE = "SMOKE"
SMOKE_MARKER_FILENAME = "SMOKE_RUN_NOT_FOR_RESULTS.txt"

RECORD_ID_COLUMN = "validation_row_index"

# Overlap-audit annotation values (only present when --audit-overlap is used).
OVERLAP_LABEL_DELIMITER = "|"
OVERLAP_NOT_IN_TRAINING = "not_in_training"
OVERLAP_SAME_LABEL = "same_label_in_training"
OVERLAP_CONFLICTING_LABEL = "conflicting_label_in_training"
OVERLAP_ANNOTATION_COLUMNS = ["feature_vector_in_training", "training_labels_for_vector", "overlap_label_relation"]

# ---------------------------------------------------------------------------
# Model configuration
#
# Centralized so it can be synchronized with Jenn's model work. These are plain
# scikit-learn defaults plus random_state (and n_jobs for Random Forest). They
# are implementation/smoke-test settings, NOT the team's final configuration.
# No class weighting, resampling, scaling or hyperparameter search is applied.
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

MODEL_CONFIGS = {
    "decision_tree": {
        "estimator": DecisionTreeClassifier,
        "params": {"random_state": RANDOM_STATE},
    },
    "random_forest": {
        "estimator": RandomForestClassifier,
        "params": {"random_state": RANDOM_STATE, "n_jobs": -1},
    },
}

# Smoke/development mode defaults (stratified subsets of train and validation).
DEFAULT_SMOKE_TRAIN_ROWS = 20_000
DEFAULT_SMOKE_VALIDATION_ROWS = 5_000

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "baseline"


class DatasetValidationError(ValueError):
    """Raised when the supplied train/validation data does not match expectations."""


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------
def load_dataset(path: Path | str, target_column: str, dataset_name: str) -> pd.DataFrame:
    """Read one split CSV. The returned RangeIndex is the 0-based data-row position."""
    path = Path(path)
    if not path.is_file():
        raise DatasetValidationError(f"{dataset_name} file does not exist: {path}")
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise DatasetValidationError(
            f"Target column {target_column!r} not found in {dataset_name} file {path}"
        )
    return df


def validate_datasets(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    expected_feature_count: int | None = EXPECTED_FEATURE_COUNT,
    expected_classes: tuple[str, ...] | None = EXPECTED_CLASSES,
) -> list[str]:
    """Validate train/validation frames and return the ordered feature column list.

    Nothing is dropped, filled or modified: any problem raises DatasetValidationError.
    """
    for name, df in (("training", train_df), ("validation", validation_df)):
        if target_column not in df.columns:
            raise DatasetValidationError(f"Target column {target_column!r} not found in {name} data")

    train_features = [c for c in train_df.columns if c != target_column]
    validation_features = [c for c in validation_df.columns if c != target_column]

    if set(train_features) != set(validation_features):
        only_train = sorted(set(train_features) - set(validation_features))
        only_validation = sorted(set(validation_features) - set(train_features))
        raise DatasetValidationError(
            "Feature columns differ between training and validation. "
            f"Only in training: {only_train}. Only in validation: {only_validation}."
        )
    if train_features != validation_features:
        raise DatasetValidationError(
            "Feature columns are the same set but in a different order in training and validation."
        )
    if expected_feature_count is not None and len(train_features) != expected_feature_count:
        raise DatasetValidationError(
            f"Expected {expected_feature_count} feature columns, found {len(train_features)}."
        )

    for name, df in (("training", train_df), ("validation", validation_df)):
        non_numeric = [
            c
            for c in train_features
            if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
        ]
        if non_numeric:
            raise DatasetValidationError(f"Non-numeric feature columns in {name} data: {non_numeric}")

        nan_counts = df[train_features].isna().sum()
        nan_counts = nan_counts[nan_counts > 0]
        if not nan_counts.empty:
            raise DatasetValidationError(f"NaN values in {name} features: {nan_counts.to_dict()}")

        inf_counts = {
            c: int(np.isinf(df[c].to_numpy()).sum())
            for c in train_features
            if pd.api.types.is_float_dtype(df[c])
        }
        inf_counts = {c: n for c, n in inf_counts.items() if n > 0}
        if inf_counts:
            raise DatasetValidationError(f"Infinite values in {name} features: {inf_counts}")

        if df[target_column].isna().any():
            raise DatasetValidationError(
                f"{int(df[target_column].isna().sum())} missing target values in {name} data"
            )

        if expected_classes is not None:
            observed = set(df[target_column].astype(str).unique())
            if observed != set(expected_classes):
                raise DatasetValidationError(
                    f"Unexpected class set in {name} data. "
                    f"Missing: {sorted(set(expected_classes) - observed)}. "
                    f"Unexpected: {sorted(observed - set(expected_classes))}."
                )

    unseen = set(validation_df[target_column].astype(str)) - set(train_df[target_column].astype(str))
    if unseen:
        raise DatasetValidationError(f"Validation classes not present in training data: {sorted(unseen)}")

    return train_features


def stratified_sample(
    df: pd.DataFrame, target_column: str, n_rows: int, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Reproducible class-stratified subset. Original index (row position) is preserved.

    Only the row index is split (not the DataFrame), so no full-size copy of the
    unused remainder is created. The selected rows are identical to splitting the
    DataFrame itself with the same random_state.
    """
    if n_rows >= len(df):
        return df
    try:
        sample_index, _ = train_test_split(
            df.index.to_numpy(), train_size=n_rows, stratify=df[target_column], random_state=random_state
        )
    except ValueError as exc:
        raise DatasetValidationError(f"Cannot draw a stratified smoke sample of {n_rows} rows: {exc}") from exc
    sample = df.loc[np.sort(sample_index)]
    missing = set(df[target_column].unique()) - set(sample[target_column].unique())
    if missing:
        raise DatasetValidationError(
            f"Smoke sample of {n_rows} rows lost classes {sorted(missing)}; increase the smoke row count."
        )
    return sample


def class_distribution(labels: pd.Series, class_order: list[str]) -> dict[str, int]:
    counts = labels.astype(str).value_counts()
    ordered = {c: int(counts.get(c, 0)) for c in class_order}
    for c in counts.index:
        ordered.setdefault(c, int(counts[c]))
    return ordered


# ---------------------------------------------------------------------------
# Optional data-overlap audit (REPORTING ONLY)
#
# Never removes rows, changes labels, changes model inputs, or excludes rows
# from metrics. It only counts and annotates.
# ---------------------------------------------------------------------------
def compute_feature_vector_ids(frames: list[pd.DataFrame], feature_columns: list[str]) -> list[np.ndarray]:
    """Exact integer id per row: two rows (from any frame) share an id iff all feature values are equal.

    Built one column at a time with pd.factorize (exact value equality), re-compressing
    the combined id after each column, so it never copies or concatenates whole
    DataFrames and cannot overflow (ids stay below the total row count).
    The re-compression uses sort-based np.unique: repeated hash-based factorizing of
    ~2M distinct int64 values retained ~2.5 GB of extra memory on the full CIC data.
    """
    sizes = [len(frame) for frame in frames]
    ids = np.zeros(sum(sizes), dtype=np.int64)
    for column in feature_columns:
        codes, uniques = pd.factorize(np.concatenate([frame[column].to_numpy() for frame in frames]))
        _, ids = np.unique(ids * len(uniques) + codes, return_inverse=True)
        ids = ids.astype(np.int64, copy=False)
    return np.split(ids, np.cumsum(sizes)[:-1])


def _serialize_label_sets(vector_ids: np.ndarray, labels: np.ndarray) -> pd.Series:
    """Map feature-vector id -> sorted, '|'-joined distinct labels (deterministic)."""
    pairs = pd.DataFrame({"vector": vector_ids, "label": labels}).drop_duplicates()
    return pairs.groupby("vector")["label"].agg(lambda s: OVERLAP_LABEL_DELIMITER.join(sorted(s)))


def _within_split_audit(vector_ids: np.ndarray, labels: np.ndarray) -> dict:
    pairs = pd.DataFrame({"vector": vector_ids, "label": labels})
    labels_per_vector = pairs.groupby("vector")["label"].nunique()
    conflicting = labels_per_vector.index[labels_per_vector > 1]
    in_conflict = pairs["vector"].isin(conflicting)
    per_label = pairs.loc[in_conflict].groupby(["vector", "label"]).size()
    group_rows = per_label.groupby(level="vector").sum()
    group_majority = per_label.groupby(level="vector").max()
    label_sets = _serialize_label_sets(
        pairs.loc[in_conflict, "vector"].to_numpy(), pairs.loc[in_conflict, "label"].to_numpy()
    )
    return {
        "rows": len(pairs),
        "exact_duplicate_rows": int(pairs.duplicated().sum()),
        "duplicate_feature_vector_rows": int(pairs["vector"].duplicated().sum()),
        "unique_feature_vectors": int(len(labels_per_vector)),
        "feature_vectors_with_multiple_labels": int(len(conflicting)),
        "rows_in_feature_vectors_with_multiple_labels": int(in_conflict.sum()),
        "minimum_rows_misclassified_by_any_deterministic_classifier": int((group_rows - group_majority).sum()),
        "multiple_label_sets": {str(k): int(v) for k, v in label_sets.value_counts().sort_index().items()},
    }


def audit_data_overlap(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[dict, pd.DataFrame]:
    """Exact duplicate / cross-split overlap audit of the supplied train and validation data.

    Returns (summary dict for run_metadata.json, per-validation-row annotation frame
    indexed like validation_df). Inputs are not modified.
    """
    train_labels = train_df[target_column].astype(str).to_numpy(dtype=object)
    validation_labels = validation_df[target_column].astype(str).to_numpy(dtype=object)
    bad_labels = sorted({l for l in set(train_labels) | set(validation_labels) if OVERLAP_LABEL_DELIMITER in l})
    if bad_labels:
        raise DatasetValidationError(
            f"Labels contain the overlap delimiter {OVERLAP_LABEL_DELIMITER!r}: {bad_labels}"
        )

    train_ids, validation_ids = compute_feature_vector_ids([train_df, validation_df], feature_columns)

    shared = np.intersect1d(train_ids, validation_ids)  # unique ids present in both splits
    train_in_shared = np.isin(train_ids, shared)
    validation_in_shared = np.isin(validation_ids, shared)
    train_sets = _serialize_label_sets(train_ids[train_in_shared], train_labels[train_in_shared])
    validation_sets = _serialize_label_sets(validation_ids[validation_in_shared], validation_labels[validation_in_shared])
    shared_pairs = pd.DataFrame(
        {
            "vector": np.concatenate([train_ids[train_in_shared], validation_ids[validation_in_shared]]),
            "label": np.concatenate([train_labels[train_in_shared], validation_labels[validation_in_shared]]),
        }
    )
    labels_per_shared_vector = shared_pairs.groupby("vector")["label"].nunique()
    same_label_vectors = labels_per_shared_vector.index[labels_per_shared_vector == 1]
    conflicting_vectors = labels_per_shared_vector.index[labels_per_shared_vector > 1]
    same_label_counts = train_sets.loc[same_label_vectors].value_counts().sort_index()
    conflicting_pairs = [
        {"training_labels": str(t), "validation_labels": str(v), "feature_vectors": int(n)}
        for (t, v), n in sorted(
            pd.Series(
                list(zip(train_sets.loc[conflicting_vectors], validation_sets.loc[conflicting_vectors])),
                dtype=object,
            )
            .value_counts()
            .items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    summary = {
        "enabled": True,
        "performed": True,
        "scope": "complete supplied training and validation files (computed before any smoke sampling)",
        "method": (
            "exact equality of all feature values as parsed from the CSVs (target ignored for feature "
            "matching); exact duplicate rows compare all features plus the target"
        ),
        "records_removed": 0,
        "labels_changed": 0,
        "model_inputs_changed": False,
        "excluded_from_metrics": 0,
        "training": _within_split_audit(train_ids, train_labels),
        "validation": _within_split_audit(validation_ids, validation_labels),
        "cross_split": {
            "feature_vectors_in_both_splits": int(len(shared)),
            "same_label_feature_vectors": int(len(same_label_vectors)),
            "conflicting_label_feature_vectors": int(len(conflicting_vectors)),
            "validation_rows_with_feature_vector_in_training": int(validation_in_shared.sum()),
            "training_rows_with_feature_vector_in_validation": int(train_in_shared.sum()),
            "same_label_feature_vectors_by_label": {str(k): int(v) for k, v in same_label_counts.items()},
            "conflicting_label_pairs": conflicting_pairs,
        },
        "definitions": {
            "exact_duplicate_rows": "rows identical to an earlier row in the same split on all features AND the target",
            "duplicate_feature_vector_rows": "rows whose feature values repeat an earlier row in the same split (target ignored)",
            "feature_vectors_with_multiple_labels": "distinct feature vectors that occur with more than one target label within the split",
            "minimum_rows_misclassified_by_any_deterministic_classifier": (
                "identical inputs always receive the same prediction, so within this split at least this many rows "
                "carry a label different from whatever single label is predicted for their feature vector"
            ),
            "same_label_feature_vectors": "feature vectors in both splits whose every occurrence (train and validation) has one identical label",
            "conflicting_label_feature_vectors": "feature vectors in both splits that occur with more than one distinct label across their train and validation occurrences",
        },
    }

    training_label_strings = np.where(
        validation_in_shared,
        pd.Series(validation_ids).map(train_sets).to_numpy(dtype=object),
        "",
    ).astype(object)
    relation = np.where(
        ~validation_in_shared,
        OVERLAP_NOT_IN_TRAINING,
        np.where(training_label_strings == validation_labels, OVERLAP_SAME_LABEL, OVERLAP_CONFLICTING_LABEL),
    )
    annotations = pd.DataFrame(
        {
            "feature_vector_in_training": validation_in_shared,
            "training_labels_for_vector": training_label_strings,
            "overlap_label_relation": relation.astype(object),
        },
        index=validation_df.index,
    )
    return summary, annotations


# ---------------------------------------------------------------------------
# Prediction interpretation
# ---------------------------------------------------------------------------
def to_binary(labels) -> np.ndarray:
    """Map seven-class labels to NORMAL (Normal Traffic) or ATTACK (every other class)."""
    labels = np.asarray(labels, dtype=object)
    return np.where(labels == NORMAL_CLASS, NORMAL_LABEL, ATTACK_LABEL).astype(object)


def binary_error_types(actual_binary, predicted_binary) -> np.ndarray:
    """Classify each record as true/false positive/negative with ATTACK as the positive class."""
    actual_attack = np.asarray(actual_binary, dtype=object) == ATTACK_LABEL
    predicted_attack = np.asarray(predicted_binary, dtype=object) == ATTACK_LABEL
    return np.select(
        [
            actual_attack & predicted_attack,
            ~actual_attack & ~predicted_attack,
            ~actual_attack & predicted_attack,
            actual_attack & ~predicted_attack,
        ],
        [TRUE_POSITIVE, TRUE_NEGATIVE, FALSE_POSITIVE, FALSE_NEGATIVE],
        default="",
    ).astype(object)


def extract_prediction_confidence(classes, probabilities, predictions) -> np.ndarray:
    """Return the predict_proba value of each record's predicted class.

    Columns of predict_proba follow model.classes_, so each prediction is mapped
    to its column through classes_ rather than assuming any label order.
    """
    classes = np.asarray(classes, dtype=object)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = np.asarray(predictions, dtype=object)
    if probabilities.shape != (len(predictions), len(classes)):
        raise ValueError(
            f"Probability shape {probabilities.shape} does not match "
            f"({len(predictions)} predictions, {len(classes)} classes)"
        )
    column_of = {label: i for i, label in enumerate(classes)}
    unknown = set(predictions) - set(column_of)
    if unknown:
        raise ValueError(f"Predicted labels not in model classes_: {sorted(map(str, unknown))}")
    columns = np.fromiter((column_of[p] for p in predictions), dtype=int, count=len(predictions))
    confidence = probabilities[np.arange(len(predictions)), columns]
    if not np.allclose(confidence, probabilities.max(axis=1)):
        raise ValueError("Predicted class is not the highest-probability class for some records")
    return confidence


def build_predictions_frame(
    model_name: str,
    run_mode: str,
    record_ids,
    actual,
    predicted,
    confidence,
) -> pd.DataFrame:
    actual = np.asarray(actual, dtype=object)
    predicted = np.asarray(predicted, dtype=object)
    actual_binary = to_binary(actual)
    predicted_binary = to_binary(predicted)
    return pd.DataFrame(
        {
            RECORD_ID_COLUMN: np.asarray(record_ids),
            "model_name": model_name,
            "run_mode": run_mode,
            "actual_attack_type": actual,
            "predicted_attack_type": predicted,
            "prediction_confidence": np.asarray(confidence, dtype=float),
            "correct_multiclass_prediction": actual == predicted,
            "actual_binary": actual_binary,
            "predicted_binary": predicted_binary,
            "binary_error_type": binary_error_types(actual_binary, predicted_binary),
        }
    )


def build_error_cases(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Errors-only export: one row per misclassified validation record, with original features.

    Every binary false positive / false negative is also a seven-class error, so a
    single mask over seven-class errors (plus binary FP/FN for safety) keeps each
    record exactly once.
    """
    is_error = (~predictions["correct_multiclass_prediction"]) | predictions["binary_error_type"].isin(
        [FALSE_POSITIVE, FALSE_NEGATIVE]
    )
    errors = predictions.loc[is_error].copy()
    errors["error_category"] = np.select(
        [
            errors["binary_error_type"] == FALSE_POSITIVE,
            errors["binary_error_type"] == FALSE_NEGATIVE,
        ],
        [ERROR_BINARY_FALSE_POSITIVE, ERROR_BINARY_FALSE_NEGATIVE],
        default=ERROR_ATTACK_TO_ATTACK,
    )
    feature_rows = features.loc[errors[RECORD_ID_COLUMN].to_numpy()].reset_index(drop=True)
    metadata_columns = [
        "model_name",
        "run_mode",
        "actual_attack_type",
        "predicted_attack_type",
        "prediction_confidence",
        "actual_binary",
        "predicted_binary",
        "binary_error_type",
        "error_category",
    ] + [c for c in OVERLAP_ANNOTATION_COLUMNS if c in errors.columns]
    return pd.concat(
        [
            errors[[RECORD_ID_COLUMN]].reset_index(drop=True),
            feature_rows,
            errors[metadata_columns].reset_index(drop=True),
        ],
        axis=1,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_multiclass_metrics(actual, predicted, labels: list[str]):
    """Return (summary dict, per-class report DataFrame, confusion matrix DataFrame)."""
    summary = {"accuracy": accuracy_score(actual, predicted)}
    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            actual, predicted, labels=labels, average=average, zero_division=0
        )
        summary.update({f"{average}_precision": p, f"{average}_recall": r, f"{average}_f1": f})

    p, r, f, s = precision_recall_fscore_support(
        actual, predicted, labels=labels, average=None, zero_division=0
    )
    report = pd.DataFrame({"class": labels, "precision": p, "recall": r, "f1": f, "support": s})
    total_support = int(s.sum())
    report = pd.concat(
        [
            report,
            pd.DataFrame(
                [
                    {
                        "class": f"{average} avg",
                        "precision": summary[f"{average}_precision"],
                        "recall": summary[f"{average}_recall"],
                        "f1": summary[f"{average}_f1"],
                        "support": total_support,
                    }
                    for average in ("macro", "weighted")
                ]
            ),
        ],
        ignore_index=True,
    )

    matrix = confusion_matrix(actual, predicted, labels=labels)
    cm = pd.DataFrame(matrix, columns=labels)
    cm.insert(0, "actual_attack_type", labels)
    return summary, report, cm


def compute_binary_metrics(predictions: pd.DataFrame) -> dict:
    """Normal-vs-Attack metrics derived from seven-class predictions (ATTACK = positive)."""
    counts = predictions["binary_error_type"].value_counts()
    tp, tn, fp, fn = (int(counts.get(k, 0)) for k in (TRUE_POSITIVE, TRUE_NEGATIVE, FALSE_POSITIVE, FALSE_NEGATIVE))

    def ratio(num, den):
        return num / den if den else 0.0

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    attack_to_attack = int(
        ((~predictions["correct_multiclass_prediction"]) & (predictions["binary_error_type"] == TRUE_POSITIVE)).sum()
    )
    return {
        "positive_class": ATTACK_LABEL,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "binary_accuracy": ratio(tp + tn, tp + tn + fp + fn),
        "binary_precision": precision,
        "binary_recall": recall,
        "binary_f1": ratio(2 * precision * recall, precision + recall),
        "false_positive_rate": ratio(fp, fp + tn),
        "false_negative_rate": ratio(fn, fn + tp),
        "attack_to_attack_misclassifications": attack_to_attack,
        "total_multiclass_errors": int((~predictions["correct_multiclass_prediction"]).sum()),
    }


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def create_run_directory(output_dir: Path | str, run_mode: str) -> Path:
    run_dir = Path(output_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_mode}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_smoke_marker(run_dir: Path, smoke_train_rows: int, smoke_validation_rows: int) -> None:
    (run_dir / SMOKE_MARKER_FILENAME).write_text(
        "SMOKE / DEVELOPMENT RUN - NOT FOR RESULTS\n\n"
        "This run used small stratified subsets of the training and validation data\n"
        f"(requested: {smoke_train_rows} training rows, {smoke_validation_rows} validation rows)\n"
        "only to verify that the pipeline works. Its metrics must not be reported,\n"
        "compared, or cited as baseline results.\n"
    )


def run_baseline(
    train_path: Path | str,
    validation_path: Path | str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    target_column: str = DEFAULT_TARGET_COLUMN,
    smoke: bool = False,
    smoke_train_rows: int = DEFAULT_SMOKE_TRAIN_ROWS,
    smoke_validation_rows: int = DEFAULT_SMOKE_VALIDATION_ROWS,
    models: tuple[str, ...] | list[str] = tuple(MODEL_CONFIGS),
    expected_feature_count: int | None = EXPECTED_FEATURE_COUNT,
    expected_classes: tuple[str, ...] | None = EXPECTED_CLASSES,
    audit_overlap: bool = False,
) -> Path:
    """Run the train/validation baseline experiment and return the run directory.

    With audit_overlap=True an exact duplicate/overlap audit of the complete supplied
    files is recorded and validation predictions are annotated. The audit is
    reporting only: model inputs, labels, row counts and metrics are unaffected.
    """
    run_mode = RUN_MODE_SMOKE if smoke else RUN_MODE_FULL
    unknown_models = [m for m in models if m not in MODEL_CONFIGS]
    if unknown_models:
        raise ValueError(f"Unknown model names: {unknown_models}. Available: {list(MODEL_CONFIGS)}")

    train_path, validation_path = Path(train_path), Path(validation_path)
    started = datetime.now(timezone.utc)

    logger.info("Loading training data: %s", train_path)
    train_df = load_dataset(train_path, target_column, "training")
    logger.info("Loading validation data: %s", validation_path)
    validation_df = load_dataset(validation_path, target_column, "validation")

    feature_columns = validate_datasets(
        train_df, validation_df, target_column, expected_feature_count, expected_classes
    )
    class_order = list(expected_classes) if expected_classes is not None else sorted(
        set(train_df[target_column].astype(str))
    )
    source_rows = {"training": len(train_df), "validation": len(validation_df)}
    source_distribution = {
        "training": class_distribution(train_df[target_column], class_order),
        "validation": class_distribution(validation_df[target_column], class_order),
    }
    logger.info("Validated %d feature columns; source rows: %s", len(feature_columns), source_rows)

    overlap_annotations = None
    if audit_overlap:
        logger.info("Overlap audit (reporting only) on the complete supplied files...")
        t0 = time.perf_counter()
        overlap_audit, overlap_annotations = audit_data_overlap(
            train_df, validation_df, feature_columns, target_column
        )
        overlap_audit["audit_time_seconds"] = time.perf_counter() - t0
        cross = overlap_audit["cross_split"]
        logger.info(
            "Overlap audit: exact duplicate rows train=%d validation=%d | cross-split feature vectors=%d "
            "(same label=%d, conflicting labels=%d) | multi-label feature vectors train=%d validation=%d | "
            "records removed=0 (%.1fs)",
            overlap_audit["training"]["exact_duplicate_rows"],
            overlap_audit["validation"]["exact_duplicate_rows"],
            cross["feature_vectors_in_both_splits"],
            cross["same_label_feature_vectors"],
            cross["conflicting_label_feature_vectors"],
            overlap_audit["training"]["feature_vectors_with_multiple_labels"],
            overlap_audit["validation"]["feature_vectors_with_multiple_labels"],
            overlap_audit["audit_time_seconds"],
        )
    else:
        overlap_audit = {
            "enabled": False,
            "performed": False,
            "note": (
                "Overlap audit not performed (run with --audit-overlap). Missing counts mean "
                "'not measured', not zero."
            ),
        }

    if smoke:
        logger.info(
            "SMOKE mode: stratified sampling %d training rows and %d validation rows",
            smoke_train_rows,
            smoke_validation_rows,
        )
        train_df = stratified_sample(train_df, target_column, smoke_train_rows)
        validation_df = stratified_sample(validation_df, target_column, smoke_validation_rows)

    used_distribution = {
        "training": class_distribution(train_df[target_column], class_order),
        "validation": class_distribution(validation_df[target_column], class_order),
    }
    for split, distribution in used_distribution.items():
        logger.info("%s rows used: %d; class distribution: %s",
                    split.capitalize(), sum(distribution.values()), distribution)

    # Pop the target so the remaining frames ARE the feature matrices (no full-size copy).
    y_train = train_df.pop(target_column).astype(str).to_numpy(dtype=object)
    y_validation = validation_df.pop(target_column).astype(str).to_numpy(dtype=object)
    X_train, X_validation = train_df, validation_df
    if list(X_train.columns) != feature_columns or list(X_validation.columns) != feature_columns:
        raise RuntimeError("Feature matrix columns do not match the validated feature list")
    record_ids = validation_df.index.to_numpy()

    run_dir = create_run_directory(output_dir, run_mode)
    logger.info("Writing outputs to %s", run_dir)
    if smoke:
        write_smoke_marker(run_dir, smoke_train_rows, smoke_validation_rows)

    summary_rows, binary_rows = [], []
    model_metadata = {}

    for model_name in models:
        config = MODEL_CONFIGS[model_name]
        model = config["estimator"](**config["params"])

        logger.info("[%s] training on %d rows", model_name, len(X_train))
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        training_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        predicted = model.predict(X_validation)
        predict_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        probabilities = model.predict_proba(X_validation)
        predict_proba_seconds = time.perf_counter() - t0

        confidence = extract_prediction_confidence(model.classes_, probabilities, predicted)
        predictions = build_predictions_frame(
            model_name, run_mode, record_ids, y_validation, predicted, confidence
        )
        if overlap_annotations is not None:
            # One annotation row per validation row (unique index): never adds or drops rows.
            annotation_rows = overlap_annotations.loc[record_ids].reset_index(drop=True)
            predictions = pd.concat([predictions, annotation_rows], axis=1)
            if len(predictions) != len(record_ids):
                raise RuntimeError("Overlap annotation changed the number of prediction rows")

        summary, report, cm = compute_multiclass_metrics(y_validation, predicted, class_order)
        binary = compute_binary_metrics(predictions)
        errors = build_error_cases(predictions, X_validation)

        n_validation = len(X_validation)
        summary_rows.append(
            {
                "run_mode": run_mode,
                "model_name": model_name,
                "training_rows": len(X_train),
                "validation_rows": n_validation,
                **summary,
                "training_time_seconds": training_seconds,
                "validation_inference_time_seconds": predict_seconds,
                "validation_predict_proba_time_seconds": predict_proba_seconds,
                "avg_inference_time_per_record_seconds": predict_seconds / n_validation,
            }
        )
        binary_rows.append({"run_mode": run_mode, "model_name": model_name, **binary})

        report.insert(0, "model_name", model_name)
        report.insert(0, "run_mode", run_mode)
        report.to_csv(run_dir / f"{model_name}_classification_report.csv", index=False)
        cm.to_csv(run_dir / f"{model_name}_confusion_matrix.csv", index=False)
        predictions.to_csv(run_dir / f"{model_name}_validation_predictions.csv", index=False)
        errors.to_csv(run_dir / f"{model_name}_error_cases.csv", index=False)

        model_metadata[model_name] = {
            "estimator": f"{type(model).__module__}.{type(model).__name__}",
            "configured_params": config["params"],
            "effective_params": model.get_params(),
            "classes_": [str(c) for c in model.classes_],
            "training_time_seconds": training_seconds,
            "validation_inference_time_seconds": predict_seconds,
            "validation_predict_proba_time_seconds": predict_proba_seconds,
            "avg_inference_time_per_record_seconds": predict_seconds / n_validation,
            "error_case_rows": len(errors),
        }
        logger.info(
            "[%s] train %.2fs | predict %.2fs | accuracy %.4f | macro F1 %.4f | FP %d | FN %d | "
            "attack-to-attack %d",
            model_name,
            training_seconds,
            predict_seconds,
            summary["accuracy"],
            summary["macro_f1"],
            binary["false_positives"],
            binary["false_negatives"],
            binary["attack_to_attack_misclassifications"],
        )

    pd.DataFrame(summary_rows).to_csv(run_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(binary_rows).to_csv(run_dir / "binary_metrics.csv", index=False)

    metadata = {
        "timestamp_utc": started.isoformat(),
        "run_mode": run_mode,
        "evaluation_split": "validation",
        "smoke": {
            "enabled": smoke,
            "requested_training_rows": smoke_train_rows if smoke else None,
            "requested_validation_rows": smoke_validation_rows if smoke else None,
            "sampling": "stratified by target (sklearn train_test_split on row index)" if smoke else None,
        },
        "command_line": sys.argv,
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
        "data": {
            "training_file": train_path.name,
            "training_path": str(train_path.resolve()),
            "training_file_size_bytes": train_path.stat().st_size,
            "validation_file": validation_path.name,
            "validation_path": str(validation_path.resolve()),
            "validation_file_size_bytes": validation_path.stat().st_size,
            "record_id": (
                f"{RECORD_ID_COLUMN} = 0-based data-row position in the validation CSV "
                "(header excluded); not the row index of the complete CICIDS2017 dataset"
            ),
            "source_row_counts": source_rows,
            "used_row_counts": {"training": len(X_train), "validation": len(X_validation)},
            "source_class_distribution": source_distribution,
            "used_class_distribution": used_distribution,
        },
        "target_column": target_column,
        "feature_count": len(feature_columns),
        "feature_names": feature_columns,
        "class_names": class_order,
        "binary_mapping": {
            NORMAL_LABEL: [NORMAL_CLASS],
            ATTACK_LABEL: [c for c in class_order if c != NORMAL_CLASS],
            "positive_class": ATTACK_LABEL,
        },
        "random_state": RANDOM_STATE,
        "models": model_metadata,
        "data_overlap_audit": overlap_audit,
        "output_files": sorted(p.name for p in run_dir.iterdir()) + ["run_metadata.json"],
    }
    if overlap_annotations is not None:
        metadata["data_overlap_audit"]["prediction_annotation"] = {
            "columns": OVERLAP_ANNOTATION_COLUMNS,
            "reference_training_data": (
                "complete supplied training file (before any smoke sampling); in SMOKE mode a matching "
                "training row is not necessarily in the smoke training subset the model saw"
            ),
            "label_delimiter": OVERLAP_LABEL_DELIMITER,
            "values": [OVERLAP_NOT_IN_TRAINING, OVERLAP_SAME_LABEL, OVERLAP_CONFLICTING_LABEL],
        }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    logger.info("Run complete (%s): %s", run_mode, run_dir)
    return run_dir


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train Decision Tree and Random Forest baselines on the CICIDS2017 training split "
            "and evaluate them on the validation split."
        )
    )
    parser.add_argument("--train", required=True, type=Path, help="Path to the 70%% training CSV.")
    parser.add_argument("--validation", required=True, type=Path, help="Path to the 15%% validation CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Parent directory for timestamped run folders (default: NOVA-main/outputs/baseline).",
    )
    parser.add_argument(
        "--target-column", default=DEFAULT_TARGET_COLUMN, help="Target column (default: %(default)s)."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS),
        default=list(MODEL_CONFIGS),
        help="Models to run (default: all).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="SMOKE/development mode: small stratified subsets of train and validation.",
    )
    parser.add_argument(
        "--smoke-train-rows",
        type=int,
        default=None,
        help=f"Training rows in smoke mode (default: {DEFAULT_SMOKE_TRAIN_ROWS}).",
    )
    parser.add_argument(
        "--smoke-validation-rows",
        type=int,
        default=None,
        help=f"Validation rows in smoke mode (default: {DEFAULT_SMOKE_VALIDATION_ROWS}).",
    )
    parser.add_argument(
        "--audit-overlap",
        action="store_true",
        help=(
            "Reporting-only exact audit of duplicates and train/validation feature overlap on the complete "
            "supplied files; annotates validation predictions. Never removes or changes records."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.smoke and (args.smoke_train_rows is not None or args.smoke_validation_rows is not None):
        parser.error("--smoke-train-rows / --smoke-validation-rows require --smoke")
    for option in ("smoke_train_rows", "smoke_validation_rows"):
        value = getattr(args, option)
        if value is not None and value <= 0:
            parser.error(f"--{option.replace('_', '-')} must be positive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run_baseline(
            train_path=args.train,
            validation_path=args.validation,
            output_dir=args.output_dir,
            target_column=args.target_column,
            smoke=args.smoke,
            smoke_train_rows=args.smoke_train_rows or DEFAULT_SMOKE_TRAIN_ROWS,
            smoke_validation_rows=args.smoke_validation_rows or DEFAULT_SMOKE_VALIDATION_ROWS,
            models=args.models,
            audit_overlap=args.audit_overlap,
        )
    except DatasetValidationError as exc:
        logger.error("Dataset validation failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
