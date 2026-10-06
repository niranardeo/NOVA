# CICIDS2017 Baseline Runner (`src/cic_baseline_runner.py`)

Owner: Paul (Team 1 – AI-Enhanced Network Anomaly Detection)

## What it does

A standalone, reproducible Python script (no Jupyter needed) that:

1. loads Kevin's official **70% training** and **15% validation** CICIDS2017 splits;
2. validates them (columns, order, feature count, numeric types, NaN, ±infinity, class set);
3. trains two **machine-learning baseline classifiers**, `DecisionTreeClassifier` and
   `RandomForestClassifier`, on the training split only;
4. evaluates them on the validation split only, with seven-class (`Attack Type`) metrics;
5. derives a secondary **Normal-vs-Attack** view from those same seven-class predictions;
6. exports per-record predictions and an **errors-only** dataset for the later
   local-AI contextualization stage;
7. optionally (`--audit-overlap`) runs a **reporting-only** exact audit of duplicates and
   train/validation feature overlap, and annotates each validation prediction with it.

It complements Jenn's notebook-based model development rather than replacing it.

**Terminology.**
- The Decision Tree and Random Forest models are *ML baseline classifiers*.
- They are **not** the project's traditional signature/rule-based IDS (e.g. Suricata
  rules and thresholds), which is a separate concept in the capstone.
- They are not the later LLM/local-AI layer either, which is a contextualization/analysis
  component. This runner does not call any LLM.

## Data inputs

| Item | Detail |
|---|---|
| Original source | CICIDS2017, as the cleaned Kaggle release linked in the README ("CICIDS2017 Cleaned and Preprocessed", 2,520,751 rows × 53 columns) |
| Authoritative split | `02_data_preprocessing_Oct_06_2026_Kevin.ipynb`: stratified `train_test_split(random_state=42)` into 70% / 15% / 15% |
| Training file | `training_dataset_CIC_70.csv`: 1,764,525 rows |
| Validation file | `validation_dataset_CIC_15.csv`: 378,113 rows |
| Target | `Attack Type` (7 classes: Normal Traffic, DoS, DDoS, Port Scanning, Brute Force, Web Attacks, Bots) |
| Features | 52 numeric columns, same names and order in both files |

Kevin's cleaned CIC pipeline is the authoritative source. This runner **does not re-clean,
re-split, deduplicate, scale, resample, or select features**. It only validates the data and
fails loudly if something is unexpected. Nothing is dropped, filled or relabelled.

The files live outside the repository in the sibling `Capstone/data/` folder (downloadable
from the OneDrive link in the README). Never copy them into `NOVA-main` or commit them.

**Negative feature values are expected and kept unchanged.** Both splits contain negative
values that are part of the original CICIDS2017 flow features and are retained by Kevin's
dataset. Most are `Init_Win_bytes_forward` / `Init_Win_bytes_backward` (−1 means "not
observed"), plus a small number in `Flow Duration`, `Flow IAT Min/Mean/Max`,
`Flow Packets/s`, `Flow Bytes/s`, `Fwd IAT Min`, `Fwd/Bwd Header Length` and
`min_seg_size_forward`. The runner does not modify, normalize, replace or reject them.

### Why only train and validation

The final **15% test split is intentionally absent** from this workspace and is never
referenced by the runner: there is no `--test` option and no logic that looks for a test
file. The test set is held back, untouched, for a later final-evaluation phase so that
model choices made against validation results cannot leak into the final score.
**Validation results are not final test results.**

## Known CICIDS2017 data characteristics

These figures come from the runner's exact audit (`--audit-overlap`) of the complete
official files. They match the duplicate and overlap checks in Jenn's Session 03 notebook.
**The runner does not remove or change any of these records.** Kevin's official split is
unchanged, and how to treat them is **awaiting a team methodology decision**.

| Finding | Count |
|---|---|
| Exact duplicate rows (all 52 features + label) within training | **82** |
| Exact duplicate rows within validation | **4** |
| Distinct feature vectors that appear in **both** training and validation (all 52 features compared, label ignored) | **190** |
| …of which every occurrence has the **same label** (all `Normal Traffic`) | **32** |
| …of which the occurrences have **conflicting labels** | **158** |
| Feature vectors within training that carry more than one label | **348** |
| Feature vectors within validation that carry more than one label | **15** |

Each of the 190 shared feature vectors corresponds to exactly one training row and one
validation row. The 158 conflicting cross-split pairs are:

| Training label | Validation label | Feature vectors |
|---|---|---|
| Normal Traffic | Port Scanning | 71 |
| Port Scanning | Normal Traffic | 61 |
| DoS | Normal Traffic | 14 |
| Normal Traffic | DoS | 12 |

Within training, the multi-label vectors are:
- Normal Traffic / Port Scanning: 288;
- Normal Traffic / DoS: 57;
- Normal Traffic / DDoS: 3.

Within validation:
- Normal Traffic / Port Scanning: 9;
- Normal Traffic / DoS: 6.

Every label conflict pairs `Normal Traffic` with an attack class.

### What these findings mean

The 190 should not be described simply as "190 leaked records". They fall into two
different kinds:

- **Same-label cross-split duplication (32 vectors).** The same input *and* the same label
  occur in both training and validation. This is a direct train/validation independence
  concern: for these rows the model is evaluated on an example it was trained on. The rows
  are all `Normal Traffic`, which makes up about 0.008% of validation.
- **Cross-split feature duplication with label conflict (158 vectors).** The same 52 feature
  values occur in both splits, but with *different* labels.
  - This does not give the ordinary "memorized the answer" advantage.
  - It is still a dataset-quality and evaluation-independence characteristic, because the
    two splits share identical inputs.
  - A trained model gives the same output for the validation row and its training twin, so
    it cannot be right on both.
  - That does **not** mean the validation row is automatically misclassified: the model may
    predict the validation label rather than the training one.
- **Within-split label conflicts (348 training, 15 validation vectors).**
  - Identical inputs carrying different labels show that, for these cases, the 52 available
    features cannot uniquely determine the supplied label.
  - The precise, supported consequence: identical inputs always receive the same prediction,
    so in each such group at least one row is misclassified by *any* deterministic
    classifier.
  - For validation this means **at least 15 rows are necessarily misclassified**. Each
    conflict pairs Normal Traffic with an attack, so this holds for both the seven-class and
    the Normal-vs-Attack views.
  - The audit reports this as `minimum_rows_misclassified_by_any_deterministic_classifier`.
  - Beyond that lower bound, no specific model error should be called "unavoidable" without
    further evidence.

The audit compares feature values exactly as parsed from the CSVs. Note that scikit-learn
tree models convert features to 32-bit floats internally. Vectors identical in the audit are
also identical to the model, and a few more near-identical vectors could become
indistinguishable to it. The audit counts are therefore a lower bound from the model's
point of view.

**How to interpret results.**
- Smoke results are implementation checks only and are **not** capstone findings.
- Any final baseline results on this split should be read with these characteristics in
  mind, unless the team chooses to regenerate the split.

The options the team can choose from:
1. keep the split and document it (current state);
2. remove overlapping rows from validation only (unequal treatment across splits);
3. deduplicate the complete dataset with a policy for conflicting labels, then regenerate
   train, validation **and** test consistently. This would change the final test set, so it
   must be decided before any test evaluation.

## Setup

The repository's `requirements.txt` pins the versions recorded in Kevin's audit notebook
(pandas 3.0.3, NumPy 2.4.6, scikit-learn 1.9.0). A reusable virtual environment lives at
`Capstone/.venv`, outside the repository:

```bash
cd Capstone
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r NOVA-main/requirements.txt
```

## Running

Run every command from `NOVA-main/`. `PYTHONDONTWRITEBYTECODE=1` is optional; it stops
Python from creating `__pycache__` folders.

### Unit tests

The tests use synthetic data only and never read the real CIC files:

```bash
PYTHONDONTWRITEBYTECODE=1 ../.venv/bin/python -m unittest discover -s tests -v
```

### Smoke / development mode

This mode uses small, reproducible, class-stratified subsets of **both** splits
(default 20,000 training and 5,000 validation rows, `random_state=42`). Use it to check
that the pipeline works, not to produce results.

```bash
../.venv/bin/python src/cic_baseline_runner.py --train ../data/training_dataset_CIC_70.csv --validation ../data/validation_dataset_CIC_15.csv --output-dir outputs/baseline --smoke
```

The same smoke run with the overlap audit:

```bash
../.venv/bin/python src/cic_baseline_runner.py --train ../data/training_dataset_CIC_70.csv --validation ../data/validation_dataset_CIC_15.csv --output-dir outputs/baseline --smoke --audit-overlap
```

You can change the sizes with `--smoke-train-rows N` and `--smoke-validation-rows N`.
Smoke runs are marked in four ways, so they cannot be mistaken for results:
- the folder name ends in `_SMOKE`;
- the folder contains a `SMOKE_RUN_NOT_FOR_RESULTS.txt` file;
- every metrics, prediction and error CSV has `run_mode = SMOKE`;
- `run_metadata.json` has `"run_mode": "SMOKE"`.

### Full train/validation experiment

Don't run this yet: see "Synchronization status" below. Without the audit:

```bash
../.venv/bin/python src/cic_baseline_runner.py --train ../data/training_dataset_CIC_70.csv --validation ../data/validation_dataset_CIC_15.csv --output-dir outputs/baseline
```

With the overlap audit and annotated predictions:

```bash
../.venv/bin/python src/cic_baseline_runner.py --train ../data/training_dataset_CIC_70.csv --validation ../data/validation_dataset_CIC_15.csv --output-dir outputs/baseline --audit-overlap
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `--train` | required | 70% training CSV |
| `--validation` | required | 15% validation CSV |
| `--output-dir` | `NOVA-main/outputs/baseline` | parent folder for timestamped run folders |
| `--target-column` | `Attack Type` | target column |
| `--models` | `decision_tree random_forest` | run one or both models |
| `--smoke` | off | smoke/development mode |
| `--smoke-train-rows` / `--smoke-validation-rows` | 20000 / 5000 | smoke subset sizes (only with `--smoke`) |
| `--audit-overlap` | off | reporting-only duplicate and overlap audit, plus prediction annotation |

If dataset validation fails, the runner exits with code 2 and prints a clear error message.

### Memory and runtime (measured on the development Mac)

- **Loading both full CSVs** takes about 4 s and roughly 1.9 GB of RAM. Smoke mode also
  loads the full files, because the stratified subset and the data checks need all rows.
- **Smoke sampling** selects row indices only, with no copy of the unused rows. The model
  frames are the loaded frames with the target column removed, so no copy is made there
  either.
- **Smoke run with `--audit-overlap`:** about 15 s wall-clock and 2.56 GB peak. The original
  smoke run before these optimizations, without the audit, peaked at 2.77 GB.
- **The audit itself:** about 5–8 s and a few hundred MB on the full files. It builds exact
  per-row feature-vector ids one column at a time, never copying the whole DataFrame.
- **The full Random Forest** (100 unlimited-depth trees on 1.76M rows) has not been run.
  Expect several additional GB of RAM and much longer training. Plan hardware and time
  before running it.

## Model configuration

All model settings live in `MODEL_CONFIGS` at the top of `src/cic_baseline_runner.py`:

| Model | Settings |
|---|---|
| DecisionTreeClassifier | scikit-learn defaults + `random_state=42` |
| RandomForestClassifier | scikit-learn defaults (100 trees, unlimited depth) + `random_state=42`, `n_jobs=-1` |

The following are also not used:
- no `class_weight` (left at `None`);
- no resampling (SMOTE, under- or oversampling);
- no scaling (tree models do not need it);
- no PCA or feature selection;
- no hyperparameter search.

**These are reproducible implementation baselines, not the team's final configuration.**
`run_metadata.json` records both the configured and the full effective parameters.

### Synchronization status with Jenn's work

- **Same pipeline.** Jenn's Session 03 notebook (`03_baseline_model_development`) uses the
  same official train/validation pipeline as this runner:
  - the same two files, 52 features, `Attack Type` and seven classes;
  - training on 70% and evaluating on 15%, with the test split untouched;
  - `random_state = 42`.
- **Same data checks.** Her duplicate and overlap checks agree with the figures above.
- **Parameters not yet set.** She has not yet established final Decision Tree or Random
  Forest parameters, so the runner keeps scikit-learn defaults.
- **Before results count as the team's baseline**, `MODEL_CONFIGS` must be synchronized with
  the team's agreed parameters, and the team must decide how to handle the duplicate and
  overlap findings.

## Outputs

Each run writes to `outputs/baseline/<YYYYmmdd_HHMMSS>_<FULL|SMOKE>/`. An existing folder is
never overwritten. `outputs/` is generated content and is ignored by `.gitignore`.

| File | Contents |
|---|---|
| `metrics_summary.csv` | One row per model, with: `run_mode`; training/validation row counts; accuracy; macro and weighted precision/recall/F1; training time; validation `predict` time; `predict_proba` time; average inference time per record (seconds) |
| `binary_metrics.csv` | One row per model, Normal-vs-Attack view: TP, TN, FP, FN; binary accuracy/precision/recall/F1; false-positive rate; false-negative rate; `attack_to_attack_misclassifications`; `total_multiclass_errors` |
| `<model>_classification_report.csv` | Per-class precision, recall, F1 and support, plus `macro avg` and `weighted avg` rows (`zero_division=0`) |
| `<model>_confusion_matrix.csv` | 7×7 seven-class confusion matrix. Rows are actual classes (`actual_attack_type`) and columns are predicted classes, in the fixed class order listed above |
| `<model>_validation_predictions.csv` | One row per validation record used: `validation_row_index`, `model_name`, `run_mode`, `actual_attack_type`, `predicted_attack_type`, `prediction_confidence`, `correct_multiclass_prediction`, `actual_binary`, `predicted_binary`, `binary_error_type`. Audit columns are added when the audit is on |
| `<model>_error_cases.csv` | Errors only, one row per misclassified record. Contains `validation_row_index`, then the **original 52 feature values**, then `model_name`, `run_mode`, `actual_attack_type`, `predicted_attack_type`, `prediction_confidence`, `actual_binary`, `predicted_binary`, `binary_error_type`, `error_category`. Audit columns are added when the audit is on |
| `run_metadata.json` | Reproducibility record (listed below) |
| `SMOKE_RUN_NOT_FOR_RESULTS.txt` | Smoke runs only |

`run_metadata.json` records:
- timestamp, run mode, smoke settings and the command line;
- Python, pandas, NumPy and scikit-learn versions, and the platform;
- input filenames, paths and file sizes (no hashing of the multi-GB files);
- source and used row counts, and source and used class distributions;
- feature count and names, and class names;
- the binary mapping and `random_state`;
- per-model configured and effective parameters, `classes_`, timings and error-case count;
- `data_overlap_audit` (see below);
- the list of output files.

**Record ID.** `validation_row_index` is the 0-based data-row position in
`validation_dataset_CIC_15.csv`, with the header excluded, so the CSV line number is
`validation_row_index + 2`. It is **not** the row index of the complete CICIDS2017
dataset: Kevin exported the splits with `index=False`, so the original indices were not
kept. In smoke mode the sampled rows keep their original positions.

**Prediction confidence** is the `predict_proba` value of the predicted class. Each
predicted label is mapped to its column through `model.classes_`, and the runner checks
that the result equals the row's maximum probability. Neither model's score is a
calibrated probability:
- **Decision Tree:** the score is the class fraction in the leaf. A default, unlimited-depth
  tree has pure leaves, so its confidence is almost always 1.0 and says little.
- **Random Forest:** the score is the average of the trees' leaf class fractions, so it
  varies more and is more informative.

### Overlap audit output (`--audit-overlap`)

With the audit **off**, `data_overlap_audit` is `{"enabled": false, "performed": false, ...}`
with a note saying the overlap was not measured. No counts are written, because a missing
count means "not measured", not zero. The annotation columns are omitted from the CSVs.

With the audit **on**, `data_overlap_audit` contains:

- **Flags:**
  - `enabled` and `performed`;
  - `scope`: the complete supplied files, computed before any smoke sampling;
  - `method`.
- **What was changed (all zero):** `records_removed: 0`, `labels_changed: 0`,
  `model_inputs_changed: false`, `excluded_from_metrics: 0`.
- **`training` and `validation`**, each with: `rows`, `exact_duplicate_rows`,
  `duplicate_feature_vector_rows`, `unique_feature_vectors`,
  `feature_vectors_with_multiple_labels`, `rows_in_feature_vectors_with_multiple_labels`,
  `minimum_rows_misclassified_by_any_deterministic_classifier` and `multiple_label_sets`.
- **`cross_split`**, with: `feature_vectors_in_both_splits`, `same_label_feature_vectors`,
  `conflicting_label_feature_vectors`, the training and validation row counts involved,
  `same_label_feature_vectors_by_label` and `conflicting_label_pairs`.
- **Also:** `definitions` for every count, `audit_time_seconds`, and
  `prediction_annotation`, which describes the columns below.

Three columns are appended to both prediction and error-case CSVs. Each validation row stays
exactly one row:

| Column | Meaning |
|---|---|
| `feature_vector_in_training` | `True` if the row's exact 52 feature values occur in the supplied training file |
| `training_labels_for_vector` | sorted, `|`-separated distinct training labels for that feature vector (e.g. `Normal Traffic` or `DoS|Normal Traffic`); empty when not in training |
| `overlap_label_relation` | `not_in_training`, `same_label_in_training` (training labels are exactly this row's label) or `conflicting_label_in_training` |

The annotation always refers to the **complete supplied training file**. In smoke mode the
matching training row is usually *not* in the 20,000-row subset the model actually saw.

## Seven-class vs Normal-vs-Attack evaluation

**Seven-class evaluation** scores the model's actual task: predicting one of the 7
`Attack Type` labels. The classes are highly imbalanced (about 83% Normal Traffic, under
0.1% each for Web Attacks and Bots), so **macro** precision, recall and F1 and the
per-class rows matter more than accuracy.

The **Normal-vs-Attack** view trains **no second model**. It reinterprets the seven-class
labels:

- `Normal Traffic` → **NORMAL**
- `DoS`, `DDoS`, `Port Scanning`, `Brute Force`, `Web Attacks`, `Bots` → **ATTACK**
  (the positive class)

| `binary_error_type` | Actual | Predicted |
|---|---|---|
| `true_positive` | ATTACK | ATTACK |
| `true_negative` | NORMAL | NORMAL |
| `false_positive` | NORMAL | ATTACK (a benign flow raised as an attack) |
| `false_negative` | ATTACK | NORMAL (a missed attack) |

The rates are defined as:
- false-positive rate = FP / (FP + TN);
- false-negative rate = FN / (FN + TP).

When the team discusses per-class "false positives" from a confusion matrix, keep that term
separate from these binary false positives and false negatives.

### Attack-to-attack errors are kept separate

A record such as *actual DoS, predicted DDoS* is a **seven-class error** but a **binary
true positive**: the attack was still detected, only mislabelled. Merging these with
FP/FN would overstate missed detections and false alarms. They are therefore:

- counted in `binary_metrics.csv` as `attack_to_attack_misclassifications`, not as FP/FN;
- marked `correct_multiclass_prediction = False` with `binary_error_type = true_positive`
  in the predictions file;
- tagged `error_category = attack_to_attack_misclassification` in the error-case export.

## Error-case exports and the later local-AI stage

`<model>_error_cases.csv` holds every validation record the model got wrong at the
seven-class level. That set already includes every binary FP and FN, so each record appears
**once per model**. Each row is self-contained for later root-cause analysis:

- the untouched 52 flow features (checked to match the source CSV row exactly);
- the true and predicted labels, and the model's confidence;
- the binary interpretation and an `error_category`:
  - `binary_false_positive`: a false alarm on normal traffic;
  - `binary_false_negative`: a missed attack;
  - `attack_to_attack_misclassification`: detected, but given the wrong attack type.

With `--audit-overlap`, each error also says whether its exact feature vector exists in
training and under which labels. That lets the planned locally hosted AI stage separate
three kinds of error:
- ordinary model mistakes;
- cases where identical inputs carry conflicting labels;
- cases where the same input and label appeared in training.

For example, the AI stage could explain why a benign flow looked like Port Scanning, or flag
missed attacks for analyst review. `validation_row_index` links each case back to the
validation CSV and to the other model's export. **No LLM is invoked by this runner.**

## Blocked on team decisions

- Whether to keep, document-only, or regenerate the official split given the duplicate,
  overlap and label-conflict findings above.
- The final Decision Tree and Random Forest parameters, agreed with Jenn's model work.
- Authorization to run the full train/validation baseline.
- Final-test evaluation, only when the team reaches that phase.
