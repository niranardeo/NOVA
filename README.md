# NOVA — Network Observation and Validation with AI

NOVA is an IT 691 capstone project that combines network intrusion
classification with AI-assisted security alert explanations.

The project uses the CICIDS2017 dataset, Decision Tree and Random
Forest classifiers, a structured JSON alert adapter, and a Qwen
consumer powered by Ollama.

Project Purpose:
NOVA (Network Observation and Validation with AI) is a cybersecurity research prototype designed to identify suspicious network activity and improve security alert interpretation. Traditional detection methods can produce false positives, miss threats, or provide limited context for security investigations.

The project combines Decision Tree and Random Forest classifiers with a locally hosted Qwen AI model to provide additional explanations for detected activity.

## Intended Users

Security analysts, incident response teams, and cybersecurity researchers.

## Research Objective
NOVA investigates how effectively machine-learning models identify unusual network traffic and whether AI-assisted explanations provide useful context beyond classifier results alone.

The prototype supports human decision-making without modifying original predictions or automatically responding to threats.

## Code Branch
The integrated code and documentation were merged into the main branch through pull requests #95 and #96.
Commit: e03130c 

## System Architecture

CICIDS2017 V3 Dataset
        |
        v
Decision Tree / Random Forest
        |
        v
JSON Alert Adapter (Schema 1.1)
        |
        v
Qwen Consumer (Ollama)
        |
        v
Validated Explanations and Analyst Review

The classifiers predict traffic categories. The alert adapter
transforms non-normal predictions into structured JSON records.
Qwen consumes these alerts to generate evidence-based explanations.

Qwen does not change classifier predictions.

## Dataset Files

The full CICIDS2017 dataset and generated train/test splits are not
included in this repository because of their size.

Original cleaned dataset:
[CICIDS2017 Cleaned and Preprocessed](https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed)

Project-specific V2 and V3 datasets:
[NOVA Dataset Files](https://paceuniversity-my.sharepoint.com/:f:/r/personal/kv56727n_pace_edu/Documents/Nova_IT691_Dataset_Version_2_and_3?d=w6b005fdd5ccd4940845187154d6eef45&csf=1&web=1&e=xgkgab)

The V3 dataset uses a 70/15/15 stratified
training/validation/test split.

The reserved test dataset was not used in the reported
preliminary classifier experiments or Qwen integration tests.

## Baseline Classification

The V3 classifier experiments use:

- Decision Tree
- Random Forest
- 52 input features
- 7 traffic classes
- 10% stratified training sample
- Random state 42

The Decision Tree uses a maximum depth of 15 and minimum leaf
size of 10.

The Random Forest uses 50 estimators, maximum depth 15,
minimum leaf size 10, and n_jobs=2.

Both classifiers were evaluated on the complete V3 validation
split of 377,880 records.

### Preliminary Validation Results

| Metric | Decision Tree | Random Forest |
|---|---:|---:|
| Accuracy | 99.77% | 99.73% |
| Macro F1-score | 0.92 | 0.91 |
| Bots recall | 0.48 | 0.39 |
| Web Attacks recall | 0.82 | 0.73 |

These are preliminary validation results, not final
held-out test results.

See [Baseline Experiment Results](docs/baseline_experiment_results.md).

## Standalone CICIDS2017 Baseline Runner

A separate Python baseline runner supports reproducible Decision
Tree and Random Forest experiments and exports classification
metrics, false positives, false negatives, and structured error
cases for later analysis.

This runner documents an earlier cleaned-dataset workflow and
should not be confused with the sampled V3 notebook experiment.

See [Baseline Runner Documentation](docs/cic_baseline_runner.md).

## JSON Alert Adapter

The alert adapter converts classifier predictions into structured
JSON alerts using schema version 1.1.

Each alert includes identifiers, a predicted traffic class,
and approved evidence fields.

The adapter preserves classifier predictions and does not
generate explanations itself.

Relevant files:

- src/alert_adapter.py
- tests/test_alert_adapter.py
- docs/alert_schema.md

## Qwen AI Integration

The Qwen consumer processes JSON alerts and generates
evidence-based explanations using a separately installed
Ollama server.

The recorded integration used qwen3.5:4b-q4_K_M.

The consumer uses Python's standard library for its runtime
dependencies. Dataset export and end-to-end workflows require
additional Python packages.

The integration supports alert validation, traceability,
response checking, and abstention.

See:

- [Qwen Integration Guide](docs/qwen_integration.md)
- [Qwen Testing Guide](docs/qwen_testing_guide.md)
- [Qwen V3 Integration Results](docs/qwen_v3_integration_results.md)

### Recorded Integration Results

The team's V3 integration report documents:

- 126,498 classifier-generated validation alerts aligned
  with the source validation dataset
- 20 real V3 alerts processed through the Qwen consumer
- 8 explained, 12 abstained, 0 rejected, 0 failed
- Preserved alert identifiers, predicted classes, and evidence

These results demonstrate the tested integration workflow.
They do not establish explanation accuracy or usefulness.

Human explanation-quality evaluation remains pending.

## Installation

Create and activate a Python virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install the baseline dependencies:

    python -m pip install -r requirements.txt

Install the Qwen development/test dependencies when needed:

    python -m pip install -r requirements-qwen-dev.txt

For model-loading and end-to-end export requirements, see:

    requirements-qwen-e2e.txt

Ollama and the Qwen model must be installed separately for
live inference. Offline tests do not require Ollama.

## Testing

Run the alert adapter tests:

    python -m unittest discover -s tests -p "test_alert_adapter.py" -v

Run the Qwen offline test suite:

    python -m pytest -p no:cacheprovider tests/qwen -q

The integration review on Ubuntu recorded:

- 12 passing alert adapter tests
- 965 passing Qwen offline tests

These tests do not substitute for live Ollama inference or
human explanation review.

## V3 Alert Export

The official exporter supports saved prediction arrays or
trusted saved classifier models.

View its options:

    python scripts/qwen_export_v3_alerts.py --help

For prediction-array exports, see the usage examples in
the script's help text.

The exporter requires the complete validation dataset
to preserve record alignment and refuses reserved test
dataset filenames.

It produces structured JSON exports and a manifest
containing dataset and prediction information.

## Reproducibility and Limitations

- Large datasets and trained model artifacts are excluded
  from Git.
- The V3 experiments used sampled training data.
- The reserved test split has not been evaluated.
- Potential preprocessing leakage requires further review.
- Bots and Web Attacks have weaker recall.
- Dataset profile matching is not the same as pinning
  and verifying dataset hashes.
- Qwen explanation quality has not received completed
  human evaluation.
- The recorded live Qwen run used a separate Ollama
  environment, not this Ubuntu review VM.

## Project Documentation

- [Baseline Experiment Results](docs/baseline_experiment_results.md)
- [Alert Schema](docs/alert_schema.md)
- [Qwen Integration](docs/qwen_integration.md)
- [Qwen Testing Guide](docs/qwen_testing_guide.md)
- [Qwen V3 Results](docs/qwen_v3_integration_results.md)
- [Qwen Review Rubric](docs/qwen_review_rubric.md)
- Project plan and contributions (https://github.com/niranardeo/NOVA/blob/main/IT%20691%20Fa2026%20Capstone%20TEAM%201%20NOVA_Project_Plan_1123.xlsx)

## Project Status

The V3 classifier, JSON adapter, and Qwen consumer have
completed the documented integration checks.

Final project activities include human explanation review,
documentation reconciliation, and team release approval.

## NOVA Capstone Video Presentation
Link: https://paceuniversity-my.sharepoint.com/:v:/g/personal/kv56727n_pace_edu/IQDj1DyQcmTgT5XD3vanCcvPAbS80t5YNCKJxQXJ081DbR8?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=dlf3A1


