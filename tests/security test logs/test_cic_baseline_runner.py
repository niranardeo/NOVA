"""Unit tests for src/cic_baseline_runner.py.

Uses only small synthetic datasets written to temporary directories; the real
CICIDS2017 files are never read. Run from NOVA-main with:

    python -m unittest discover -s tests -v
"""

import contextlib
import io
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cic_baseline_runner as runner  # noqa: E402

TARGET = runner.DEFAULT_TARGET_COLUMN
CLASSES = list(runner.EXPECTED_CLASSES)
FEATURES = [f"feature_{i:02d}" for i in range(runner.EXPECTED_FEATURE_COUNT)]


def setUpModule():
    logging.disable(logging.CRITICAL)


def tearDownModule():
    logging.disable(logging.NOTSET)


def make_split(rows_per_class: int, seed: int) -> pd.DataFrame:
    """Synthetic split with 52 numeric features and all seven classes (class-dependent shift)."""
    rng = np.random.default_rng(seed)
    frames = []
    for class_index, label in enumerate(CLASSES):
        values = rng.normal(loc=class_index * 3.0, scale=0.5, size=(rows_per_class, len(FEATURES)))
        frame = pd.DataFrame(values, columns=FEATURES)
        frame[TARGET] = label
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write_csv(self, df: pd.DataFrame, name: str) -> Path:
        path = self.tmp / name
        df.to_csv(path, index=False)
        return path

    def write_text_csv(self, text: str, name: str) -> Path:
        path = self.tmp / name
        path.write_text(text)
        return path


class LoadingAndValidationTests(TempDirTestCase):
    def test_loads_train_and_validation(self):
        train = make_split(10, seed=1)
        validation = make_split(4, seed=2)
        train_df = runner.load_dataset(self.write_csv(train, "train.csv"), TARGET, "training")
        validation_df = runner.load_dataset(self.write_csv(validation, "val.csv"), TARGET, "validation")

        features = runner.validate_datasets(train_df, validation_df)

        self.assertEqual(features, FEATURES)
        self.assertEqual(len(train_df), 70)
        self.assertEqual(len(validation_df), 28)
        self.assertEqual(list(validation_df.index), list(range(28)))

    def test_missing_file_raises(self):
        with self.assertRaisesRegex(runner.DatasetValidationError, "does not exist"):
            runner.load_dataset(self.tmp / "missing.csv", TARGET, "training")

    def test_missing_target_column_raises(self):
        train = make_split(3, seed=1).drop(columns=[TARGET])
        with self.assertRaisesRegex(runner.DatasetValidationError, "Target column"):
            runner.load_dataset(self.write_csv(train, "train.csv"), TARGET, "training")

    def test_missing_target_column_in_validate_raises(self):
        train = make_split(3, seed=1)
        validation = make_split(3, seed=2).drop(columns=[TARGET])
        with self.assertRaisesRegex(runner.DatasetValidationError, "Target column"):
            runner.validate_datasets(train, validation)

    def test_mismatched_feature_names_raise(self):
        train = make_split(3, seed=1)
        validation = make_split(3, seed=2).rename(columns={"feature_05": "renamed"})
        with self.assertRaisesRegex(runner.DatasetValidationError, "differ"):
            runner.validate_datasets(train, validation)

    def test_mismatched_feature_order_raises(self):
        train = make_split(3, seed=1)
        validation = make_split(3, seed=2)
        reordered = [FEATURES[1], FEATURES[0]] + FEATURES[2:] + [TARGET]
        with self.assertRaisesRegex(runner.DatasetValidationError, "different order"):
            runner.validate_datasets(train, validation[reordered])

    def test_wrong_feature_count_raises(self):
        train = make_split(3, seed=1).drop(columns=["feature_51"])
        validation = make_split(3, seed=2).drop(columns=["feature_51"])
        with self.assertRaisesRegex(runner.DatasetValidationError, "Expected 52"):
            runner.validate_datasets(train, validation)

    def test_non_numeric_feature_raises(self):
        train = make_split(3, seed=1)
        train["feature_10"] = train["feature_10"].astype(str) + "x"
        with self.assertRaisesRegex(runner.DatasetValidationError, "Non-numeric"):
            runner.validate_datasets(train, make_split(3, seed=2))

    def test_nan_in_csv_detected(self):
        validation = make_split(3, seed=2)
        validation.loc[4, "feature_07"] = np.nan  # written as an empty CSV cell
        train_df = runner.load_dataset(self.write_csv(make_split(3, seed=1), "train.csv"), TARGET, "training")
        validation_df = runner.load_dataset(self.write_csv(validation, "val.csv"), TARGET, "validation")
        with self.assertRaisesRegex(runner.DatasetValidationError, "NaN.*feature_07"):
            runner.validate_datasets(train_df, validation_df)

    def test_positive_and_negative_infinity_in_csv_detected(self):
        for token in ("inf", "-inf"):
            with self.subTest(token=token):
                train = make_split(3, seed=1)
                text = train.to_csv(index=False).splitlines()
                cells = text[3].split(",")
                cells[2] = token  # feature_02 of data row 2
                text[3] = ",".join(cells)
                path = self.write_text_csv("\n".join(text) + "\n", f"train_{token}.csv")
                train_df = runner.load_dataset(path, TARGET, "training")
                with self.assertRaisesRegex(runner.DatasetValidationError, "Infinite.*feature_02"):
                    runner.validate_datasets(train_df, make_split(3, seed=2))

    def test_unexpected_class_raises(self):
        validation = make_split(3, seed=2)
        validation.loc[0, TARGET] = "Heartbleed"
        with self.assertRaisesRegex(runner.DatasetValidationError, "Unexpected class set"):
            runner.validate_datasets(make_split(3, seed=1), validation)

    def test_negative_feature_values_are_accepted_unchanged(self):
        train = make_split(3, seed=1)
        train["feature_00"] = -1  # e.g. CICIDS2017 Init_Win_bytes sentinel values
        runner.validate_datasets(train, make_split(3, seed=2))
        self.assertTrue((train["feature_00"] == -1).all())


class BinaryInterpretationTests(unittest.TestCase):
    def test_normal_attack_mapping(self):
        mapped = runner.to_binary(CLASSES)
        self.assertEqual(list(mapped), ["NORMAL"] + ["ATTACK"] * 6)

    def _error_type(self, actual, predicted):
        frame = runner.build_predictions_frame("m", "FULL", [0], [actual], [predicted], [1.0])
        return frame.loc[0, "binary_error_type"], bool(frame.loc[0, "correct_multiclass_prediction"])

    def test_true_positive(self):
        self.assertEqual(self._error_type("DoS", "DoS"), ("true_positive", True))

    def test_true_negative(self):
        self.assertEqual(self._error_type("Normal Traffic", "Normal Traffic"), ("true_negative", True))

    def test_false_positive(self):
        self.assertEqual(self._error_type("Normal Traffic", "Port Scanning"), ("false_positive", False))

    def test_false_negative(self):
        self.assertEqual(self._error_type("Bots", "Normal Traffic"), ("false_negative", False))

    def test_attack_to_attack_is_multiclass_error_but_binary_true_positive(self):
        self.assertEqual(self._error_type("DoS", "DDoS"), ("true_positive", False))

    def test_error_cases_categories_dedup_and_features(self):
        actual = ["Normal Traffic", "Normal Traffic", "DoS", "Bots", "DoS", "Web Attacks"]
        predicted = ["Normal Traffic", "DoS", "DoS", "Normal Traffic", "DDoS", "Brute Force"]
        record_ids = [10, 11, 12, 13, 14, 15]
        features = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6], "f2": [-1.5, 0, 0, 0, 0, 9.25]}, index=record_ids)
        predictions = runner.build_predictions_frame(
            "decision_tree", "FULL", record_ids, actual, predicted, [0.9] * 6
        )
        errors = runner.build_error_cases(predictions, features)

        self.assertEqual(list(errors[runner.RECORD_ID_COLUMN]), [11, 13, 14, 15])
        self.assertTrue(errors[runner.RECORD_ID_COLUMN].is_unique)
        self.assertEqual(
            list(errors["error_category"]),
            [
                "binary_false_positive",
                "binary_false_negative",
                "attack_to_attack_misclassification",
                "attack_to_attack_misclassification",
            ],
        )
        self.assertEqual(list(errors["f1"]), [2, 4, 5, 6])
        self.assertEqual(list(errors["f2"]), [0, 0, 0, 9.25])
        self.assertEqual(list(errors.columns[:3]), [runner.RECORD_ID_COLUMN, "f1", "f2"])

    def test_binary_metrics_values(self):
        actual = ["Normal Traffic", "Normal Traffic", "Normal Traffic", "DoS", "DoS", "Bots", "DDoS"]
        predicted = ["Normal Traffic", "Normal Traffic", "DoS", "DoS", "DDoS", "Normal Traffic", "DDoS"]
        predictions = runner.build_predictions_frame("m", "FULL", range(7), actual, predicted, [1.0] * 7)
        metrics = runner.compute_binary_metrics(predictions)

        self.assertEqual(
            (metrics["true_positives"], metrics["true_negatives"],
             metrics["false_positives"], metrics["false_negatives"]),
            (3, 2, 1, 1),
        )
        self.assertAlmostEqual(metrics["binary_accuracy"], 5 / 7)
        self.assertAlmostEqual(metrics["binary_precision"], 3 / 4)
        self.assertAlmostEqual(metrics["binary_recall"], 3 / 4)
        self.assertAlmostEqual(metrics["binary_f1"], 3 / 4)
        self.assertAlmostEqual(metrics["false_positive_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["false_negative_rate"], 1 / 4)
        self.assertEqual(metrics["attack_to_attack_misclassifications"], 1)
        self.assertEqual(metrics["total_multiclass_errors"], 3)


class ConfidenceTests(unittest.TestCase):
    def test_confidence_uses_classes_order(self):
        classes = np.array(["DoS", "Normal Traffic", "Bots"], dtype=object)
        probabilities = np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
        predictions = np.array(["Normal Traffic", "DoS", "Bots"], dtype=object)
        confidence = runner.extract_prediction_confidence(classes, probabilities, predictions)
        np.testing.assert_allclose(confidence, [0.7, 0.6, 0.6])

    def test_confidence_rejects_non_argmax_prediction(self):
        classes = np.array(["DoS", "Normal Traffic"], dtype=object)
        with self.assertRaises(ValueError):
            runner.extract_prediction_confidence(classes, [[0.9, 0.1]], ["Normal Traffic"])

    def test_confidence_rejects_unknown_label(self):
        with self.assertRaises(ValueError):
            runner.extract_prediction_confidence(["DoS"], [[1.0]], ["Bots"])

    def test_confidence_matches_fitted_model(self):
        from sklearn.ensemble import RandomForestClassifier

        train = make_split(8, seed=3)
        model = RandomForestClassifier(n_estimators=5, random_state=42).fit(train[FEATURES], train[TARGET])
        X = make_split(2, seed=4)[FEATURES]
        predicted = model.predict(X)
        probabilities = model.predict_proba(X)
        confidence = runner.extract_prediction_confidence(model.classes_, probabilities, predicted)
        expected = [probabilities[i, list(model.classes_).index(p)] for i, p in enumerate(predicted)]
        np.testing.assert_allclose(confidence, expected)


class EndToEndTests(TempDirTestCase):
    EXPECTED_FILES = {
        "metrics_summary.csv",
        "binary_metrics.csv",
        "run_metadata.json",
        *(
            f"{model}_{suffix}"
            for model in ("decision_tree", "random_forest")
            for suffix in (
                "classification_report.csv",
                "confusion_matrix.csv",
                "validation_predictions.csv",
                "error_cases.csv",
            )
        ),
    }

    def setUp(self):
        super().setUp()
        self.train_path = self.write_csv(make_split(20, seed=11), "train.csv")
        validation = make_split(10, seed=12)
        # A few noisy rows so error-case exports are non-empty.
        validation.loc[:3, FEATURES] = validation.loc[:3, FEATURES] + 9.0
        self.validation_path = self.write_csv(validation, "val.csv")
        self.output_dir = self.tmp / "outputs"

    def test_full_run_creates_expected_outputs(self):
        run_dir = runner.run_baseline(self.train_path, self.validation_path, self.output_dir)

        self.assertTrue(run_dir.name.endswith("_FULL"))
        self.assertEqual({p.name for p in run_dir.iterdir()}, self.EXPECTED_FILES)

        metadata = json.loads((run_dir / "run_metadata.json").read_text())
        self.assertEqual(metadata["run_mode"], "FULL")
        self.assertEqual(metadata["feature_count"], 52)
        self.assertEqual(metadata["class_names"], CLASSES)
        self.assertEqual(metadata["data"]["used_row_counts"], {"training": 140, "validation": 70})
        self.assertEqual(metadata["random_state"], 42)
        self.assertEqual(metadata["models"]["random_forest"]["configured_params"]["n_jobs"], -1)

        predictions = pd.read_csv(run_dir / "random_forest_validation_predictions.csv")
        self.assertEqual(len(predictions), 70)
        self.assertEqual(list(predictions[runner.RECORD_ID_COLUMN]), list(range(70)))
        self.assertTrue(set(predictions["binary_error_type"]) <= {
            "true_positive", "true_negative", "false_positive", "false_negative"})

        cm = pd.read_csv(run_dir / "decision_tree_confusion_matrix.csv")
        self.assertEqual(list(cm.columns), ["actual_attack_type"] + CLASSES)
        self.assertEqual(int(cm[CLASSES].to_numpy().sum()), 70)

        errors = pd.read_csv(run_dir / "decision_tree_error_cases.csv")
        dt_predictions = pd.read_csv(run_dir / "decision_tree_validation_predictions.csv")
        self.assertEqual(len(errors), int((~dt_predictions["correct_multiclass_prediction"]).sum()))
        self.assertTrue(set(FEATURES) <= set(errors.columns))

        summary = pd.read_csv(run_dir / "metrics_summary.csv")
        self.assertEqual(list(summary["run_mode"]), ["FULL", "FULL"])

    def test_smoke_run_is_marked(self):
        run_dir = runner.run_baseline(
            self.train_path,
            self.validation_path,
            self.output_dir,
            smoke=True,
            smoke_train_rows=70,
            smoke_validation_rows=35,
        )

        self.assertTrue(run_dir.name.endswith("_SMOKE"))
        self.assertTrue((run_dir / runner.SMOKE_MARKER_FILENAME).is_file())
        metadata = json.loads((run_dir / "run_metadata.json").read_text())
        self.assertEqual(metadata["run_mode"], "SMOKE")
        self.assertTrue(metadata["smoke"]["enabled"])
        self.assertEqual(metadata["data"]["used_row_counts"], {"training": 70, "validation": 35})
        self.assertEqual(metadata["data"]["source_row_counts"], {"training": 140, "validation": 70})
        self.assertTrue(all(v == 5 for v in metadata["data"]["used_class_distribution"]["validation"].values()))
        for name in ("metrics_summary.csv", "binary_metrics.csv", "random_forest_error_cases.csv",
                     "decision_tree_validation_predictions.csv", "decision_tree_classification_report.csv"):
            frame = pd.read_csv(run_dir / name)
            if len(frame):
                self.assertEqual(set(frame["run_mode"]), {"SMOKE"}, name)

        # Smoke record ids are the original validation row positions.
        predictions = pd.read_csv(run_dir / "decision_tree_validation_predictions.csv")
        self.assertTrue(predictions[runner.RECORD_ID_COLUMN].between(0, 69).all())
        self.assertTrue(predictions[runner.RECORD_ID_COLUMN].is_unique)

    def test_full_run_has_no_smoke_marker(self):
        run_dir = runner.run_baseline(
            self.train_path, self.validation_path, self.output_dir, models=["decision_tree"]
        )
        self.assertFalse((run_dir / runner.SMOKE_MARKER_FILENAME).exists())

    def test_stratified_sample_keeps_every_class(self):
        df = make_split(20, seed=5)
        sample = runner.stratified_sample(df, TARGET, 35)
        self.assertEqual(len(sample), 35)
        self.assertEqual(set(sample[TARGET]), set(CLASSES))
        pd.testing.assert_frame_equal(sample, df.loc[sample.index])

    def test_stratified_sample_matches_dataframe_split(self):
        # Index-only splitting must select exactly the rows a DataFrame split would.
        from sklearn.model_selection import train_test_split

        df = make_split(20, seed=5)
        expected, _ = train_test_split(df, train_size=35, stratify=df[TARGET], random_state=42)
        sample = runner.stratified_sample(df, TARGET, 35)
        self.assertEqual(list(sample.index), sorted(expected.index))

    def test_stratified_sample_too_small_raises_validation_error(self):
        with self.assertRaises(runner.DatasetValidationError):
            runner.stratified_sample(make_split(20, seed=5), TARGET, 3)


class CommandLineTests(TempDirTestCase):
    def test_audit_overlap_flag_defaults_off(self):
        parser = runner.build_parser()
        base = ["--train", "a.csv", "--validation", "b.csv"]
        self.assertFalse(parser.parse_args(base).audit_overlap)
        self.assertTrue(parser.parse_args(base + ["--audit-overlap"]).audit_overlap)

    def test_parser_has_no_test_option(self):
        parser = runner.build_parser()
        option_strings = {s for action in parser._actions for s in action.option_strings}
        self.assertNotIn("--test", option_strings)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parser.parse_args(["--train", "a.csv", "--validation", "b.csv", "--test", "c.csv"])

    def test_smoke_row_options_require_smoke(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            runner.main(["--train", "a.csv", "--validation", "b.csv", "--smoke-train-rows", "10"])

    def test_main_returns_error_code_for_missing_file(self):
        code = runner.main(
            ["--train", str(self.tmp / "nope.csv"), "--validation", str(self.tmp / "nope2.csv"),
             "--output-dir", str(self.tmp / "out")]
        )
        self.assertEqual(code, 2)
        self.assertFalse((self.tmp / "out").exists())


N, D, P = "Normal Traffic", "DoS", "Port Scanning"


def overlap_fixture():
    """Tiny two-feature splits with hand-checkable duplicate/overlap structure."""
    train = pd.DataFrame(
        [
            (1, 1.0, N),
            (1, 1.0, N),         # exact duplicate of row 0
            (2, 2.0, N),
            (2, 2.0, D),         # (2,2) carries two labels in training (inserted N before D)
            (3, 3.0, N),         # shared with validation, same label
            (4, 4.0, N),         # shared with validation, validation says Port Scanning
            (5, 5.0, P),
            (6, 6.0, N),         # shared; validation has (6,6) as both N and D
            (9, 9.000001, D),    # near-identical to validation (9, 9.0) but NOT equal
        ],
        columns=["f1", "f2", TARGET],
    )
    validation = pd.DataFrame(
        [
            (3, 3.0, N),         # 0: same label in training
            (4, 4.0, P),         # 1: conflicting with training
            (7, 7.0, N),         # 2: not in training
            (7, 7.0, N),         # 3: exact duplicate of row 2
            (8, 8.0, N),         # 4
            (8, 8.0, P),         # 5: (8,8) carries two labels in validation
            (6, 6.0, N),         # 6: training label N == own label
            (6, 6.0, D),         # 7: training label N != own label
            (2, 2.0, N),         # 8: training labels D|N
            (9, 9.0, D),         # 9: not in training (exact match required)
        ],
        columns=["f1", "f2", TARGET],
    )
    return train, validation


class OverlapAuditTests(unittest.TestCase):
    def setUp(self):
        self.train, self.validation = overlap_fixture()
        self.summary, self.annotations = runner.audit_data_overlap(
            self.train, self.validation, ["f1", "f2"], TARGET
        )

    def test_exact_duplicate_counting(self):
        self.assertEqual(self.summary["training"]["exact_duplicate_rows"], 1)
        self.assertEqual(self.summary["validation"]["exact_duplicate_rows"], 1)
        self.assertEqual(self.summary["training"]["duplicate_feature_vector_rows"], 2)
        self.assertEqual(self.summary["validation"]["duplicate_feature_vector_rows"], 3)
        self.assertEqual(self.summary["training"]["unique_feature_vectors"], 7)
        self.assertEqual(self.summary["validation"]["unique_feature_vectors"], 7)

    def test_feature_only_cross_split_matching(self):
        cross = self.summary["cross_split"]
        self.assertEqual(cross["feature_vectors_in_both_splits"], 4)  # (2,2) (3,3) (4,4) (6,6); not (9,9)
        self.assertEqual(cross["validation_rows_with_feature_vector_in_training"], 5)
        self.assertEqual(cross["training_rows_with_feature_vector_in_validation"], 5)

    def test_same_label_cross_split_overlap(self):
        cross = self.summary["cross_split"]
        self.assertEqual(cross["same_label_feature_vectors"], 1)
        self.assertEqual(cross["same_label_feature_vectors_by_label"], {N: 1})

    def test_conflicting_label_cross_split_overlap(self):
        cross = self.summary["cross_split"]
        self.assertEqual(cross["conflicting_label_feature_vectors"], 3)
        self.assertEqual(
            cross["conflicting_label_pairs"],
            [
                {"training_labels": f"{D}|{N}", "validation_labels": N, "feature_vectors": 1},
                {"training_labels": N, "validation_labels": f"{D}|{N}", "feature_vectors": 1},
                {"training_labels": N, "validation_labels": P, "feature_vectors": 1},
            ],
        )

    def test_within_training_conflicting_labels(self):
        training = self.summary["training"]
        self.assertEqual(training["feature_vectors_with_multiple_labels"], 1)
        self.assertEqual(training["rows_in_feature_vectors_with_multiple_labels"], 2)
        self.assertEqual(training["minimum_rows_misclassified_by_any_deterministic_classifier"], 1)
        self.assertEqual(training["multiple_label_sets"], {f"{D}|{N}": 1})

    def test_within_validation_conflicting_labels(self):
        validation = self.summary["validation"]
        self.assertEqual(validation["feature_vectors_with_multiple_labels"], 2)
        self.assertEqual(validation["rows_in_feature_vectors_with_multiple_labels"], 4)
        self.assertEqual(validation["minimum_rows_misclassified_by_any_deterministic_classifier"], 2)
        self.assertEqual(validation["multiple_label_sets"], {f"{D}|{N}": 1, f"{N}|{P}": 1})

    def test_feature_vector_in_training_annotation(self):
        self.assertEqual(
            list(self.annotations["feature_vector_in_training"]),
            [True, True, False, False, False, False, True, True, True, False],
        )
        self.assertEqual(
            list(self.annotations["overlap_label_relation"]),
            [
                runner.OVERLAP_SAME_LABEL,
                runner.OVERLAP_CONFLICTING_LABEL,
                runner.OVERLAP_NOT_IN_TRAINING,
                runner.OVERLAP_NOT_IN_TRAINING,
                runner.OVERLAP_NOT_IN_TRAINING,
                runner.OVERLAP_NOT_IN_TRAINING,
                runner.OVERLAP_SAME_LABEL,
                runner.OVERLAP_CONFLICTING_LABEL,
                runner.OVERLAP_CONFLICTING_LABEL,
                runner.OVERLAP_NOT_IN_TRAINING,
            ],
        )

    def test_multiple_training_labels_serialized_deterministically(self):
        labels = list(self.annotations["training_labels_for_vector"])
        self.assertEqual(labels[8], f"{D}|{N}")  # sorted, regardless of training row order
        self.assertEqual(labels[0], N)
        self.assertEqual(labels[2], "")
        # Reversing training row order must not change the serialization.
        _, reversed_annotations = runner.audit_data_overlap(
            self.train.iloc[::-1].reset_index(drop=True), self.validation, ["f1", "f2"], TARGET
        )
        self.assertEqual(list(reversed_annotations["training_labels_for_vector"]), labels)

    def test_one_annotation_row_per_validation_row_and_index_preserved(self):
        validation = self.validation.set_axis(range(100, 110))  # e.g. smoke row positions
        _, annotations = runner.audit_data_overlap(self.train, validation, ["f1", "f2"], TARGET)
        self.assertEqual(len(annotations), len(validation))
        self.assertEqual(list(annotations.index), list(validation.index))
        self.assertTrue(annotations.index.is_unique)

    def test_audit_does_not_modify_inputs_and_reports_no_removals(self):
        train_before, validation_before = self.train.copy(), self.validation.copy()
        summary, _ = runner.audit_data_overlap(self.train, self.validation, ["f1", "f2"], TARGET)
        pd.testing.assert_frame_equal(self.train, train_before)
        pd.testing.assert_frame_equal(self.validation, validation_before)
        self.assertEqual(summary["records_removed"], 0)
        self.assertEqual(summary["labels_changed"], 0)
        self.assertFalse(summary["model_inputs_changed"])

    def test_no_overlap_case(self):
        validation = self.validation.copy()
        validation["f1"] = validation["f1"] + 1000
        summary, annotations = runner.audit_data_overlap(self.train, validation, ["f1", "f2"], TARGET)
        self.assertEqual(summary["cross_split"]["feature_vectors_in_both_splits"], 0)
        self.assertEqual(summary["cross_split"]["conflicting_label_pairs"], [])
        self.assertFalse(annotations["feature_vector_in_training"].any())
        self.assertEqual(set(annotations["training_labels_for_vector"]), {""})

    def test_label_containing_delimiter_rejected(self):
        validation = self.validation.copy()
        validation.loc[0, TARGET] = "Odd|Label"
        with self.assertRaises(runner.DatasetValidationError):
            runner.audit_data_overlap(self.train, validation, ["f1", "f2"], TARGET)


class OverlapEndToEndTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        train = make_split(20, seed=21)
        validation = make_split(10, seed=22)
        validation.loc[:3, FEATURES] = validation.loc[:3, FEATURES] + 9.0  # guarantee some errors
        # Validation row 10: copy features of a training row with the SAME label.
        same_label = validation.loc[10, TARGET]
        self.same_source = int(train.index[train[TARGET] == same_label][0])
        validation.loc[10, FEATURES] = train.loc[self.same_source, FEATURES].to_numpy()
        # Validation row 11: copy features of a training row with a DIFFERENT label.
        other = validation.loc[11, TARGET]
        self.conflict_source = int(train.index[train[TARGET] != other][0])
        validation.loc[11, FEATURES] = train.loc[self.conflict_source, FEATURES].to_numpy()
        self.conflict_training_label = train.loc[self.conflict_source, TARGET]
        self.train_path = self.write_csv(train, "train.csv")
        self.validation_path = self.write_csv(validation, "val.csv")

    def run_pair(self, **kwargs):
        audited = runner.run_baseline(
            self.train_path, self.validation_path, self.tmp / "out_audit", audit_overlap=True, **kwargs
        )
        plain = runner.run_baseline(self.train_path, self.validation_path, self.tmp / "out_plain", **kwargs)
        return audited, plain

    @staticmethod
    def read_predictions(run_dir, model):
        frame = pd.read_csv(run_dir / f"{model}_validation_predictions.csv")
        if "training_labels_for_vector" in frame:
            frame["training_labels_for_vector"] = frame["training_labels_for_vector"].fillna("")
        return frame

    def test_audit_metadata_enabled(self):
        audited, _ = self.run_pair()
        audit = json.loads((audited / "run_metadata.json").read_text())["data_overlap_audit"]
        self.assertTrue(audit["enabled"])
        self.assertTrue(audit["performed"])
        self.assertEqual(audit["records_removed"], 0)
        self.assertEqual(audit["cross_split"]["feature_vectors_in_both_splits"], 2)
        self.assertEqual(audit["cross_split"]["same_label_feature_vectors"], 1)
        self.assertEqual(audit["cross_split"]["conflicting_label_feature_vectors"], 1)
        self.assertEqual(audit["training"]["exact_duplicate_rows"], 0)
        self.assertEqual(audit["prediction_annotation"]["columns"], runner.OVERLAP_ANNOTATION_COLUMNS)

    def test_audit_metadata_disabled(self):
        _, plain = self.run_pair(models=["decision_tree"])
        audit = json.loads((plain / "run_metadata.json").read_text())["data_overlap_audit"]
        self.assertFalse(audit["enabled"])
        self.assertFalse(audit["performed"])
        self.assertNotIn("cross_split", audit)
        self.assertNotIn("records_removed", audit)  # absent, not a misleading zero
        predictions = self.read_predictions(plain, "decision_tree")
        for column in runner.OVERLAP_ANNOTATION_COLUMNS:
            self.assertNotIn(column, predictions.columns)

    def test_predictions_annotated_one_row_per_validation_row(self):
        audited, _ = self.run_pair()
        for model in ("decision_tree", "random_forest"):
            predictions = self.read_predictions(audited, model).set_index(runner.RECORD_ID_COLUMN)
            self.assertEqual(len(predictions), 70)
            self.assertTrue(predictions.index.is_unique)
            self.assertEqual(int(predictions["feature_vector_in_training"].sum()), 2)
            self.assertEqual(predictions.loc[10, "overlap_label_relation"], runner.OVERLAP_SAME_LABEL)
            self.assertEqual(predictions.loc[11, "overlap_label_relation"], runner.OVERLAP_CONFLICTING_LABEL)
            self.assertEqual(predictions.loc[11, "training_labels_for_vector"], self.conflict_training_label)
            self.assertEqual(predictions.loc[0, "training_labels_for_vector"], "")

            errors = pd.read_csv(audited / f"{model}_error_cases.csv")
            for column in runner.OVERLAP_ANNOTATION_COLUMNS:
                self.assertIn(column, errors.columns)
            self.assertTrue(errors[runner.RECORD_ID_COLUMN].is_unique)
            self.assertTrue(set(errors[runner.RECORD_ID_COLUMN]) <= set(predictions.index))

    def test_auditing_does_not_alter_rows_labels_predictions_or_metrics(self):
        audited, plain = self.run_pair()
        meta_a = json.loads((audited / "run_metadata.json").read_text())
        meta_p = json.loads((plain / "run_metadata.json").read_text())
        self.assertEqual(meta_a["data"]["used_row_counts"], meta_p["data"]["used_row_counts"])
        self.assertEqual(meta_a["data"]["used_class_distribution"], meta_p["data"]["used_class_distribution"])
        for model in ("decision_tree", "random_forest"):
            a = self.read_predictions(audited, model)
            p = self.read_predictions(plain, model)
            pd.testing.assert_frame_equal(a[p.columns], p)
        timing = [c for c in pd.read_csv(plain / "metrics_summary.csv").columns if "time" in c]
        pd.testing.assert_frame_equal(
            pd.read_csv(audited / "metrics_summary.csv").drop(columns=timing),
            pd.read_csv(plain / "metrics_summary.csv").drop(columns=timing),
        )
        pd.testing.assert_frame_equal(
            pd.read_csv(audited / "binary_metrics.csv"), pd.read_csv(plain / "binary_metrics.csv")
        )

    def test_smoke_audit_covers_full_files_but_models_use_subset(self):
        audited, plain = self.run_pair(smoke=True, smoke_train_rows=70, smoke_validation_rows=35)
        metadata = json.loads((audited / "run_metadata.json").read_text())
        self.assertEqual(metadata["run_mode"], "SMOKE")
        self.assertEqual(metadata["data_overlap_audit"]["training"]["rows"], 140)
        self.assertEqual(metadata["data_overlap_audit"]["validation"]["rows"], 70)
        self.assertEqual(metadata["data"]["used_row_counts"], {"training": 70, "validation": 35})
        predictions = self.read_predictions(audited, "random_forest")
        self.assertEqual(len(predictions), 35)
        self.assertEqual(set(predictions["run_mode"]), {"SMOKE"})
        p = self.read_predictions(plain, "random_forest")
        pd.testing.assert_frame_equal(predictions[p.columns], p)


if __name__ == "__main__":
    unittest.main()
