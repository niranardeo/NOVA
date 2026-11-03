# NOVA — Baseline Model Experiment Results

**Project:** NOVA — IT 691 Capstone
**Dataset:** CICIDS2017, processed V3
**Models:** Decision Tree and Random Forest
**Status:** Preliminary sampled-training experiments

## 1. Overview

This report documents the initial Decision Tree and Random Forest
experiments conducted using the official CICIDS2017 V3 training and
validation splits.

Both models were trained on a reproducible 10% stratified sample
of the official training dataset and evaluated against the complete
official validation dataset.

These results are not full-training or reserved-test results.

## 2. Dataset Preparation

| Property | Value |
|---|---|
| Official training records | 1,763,436 |
| Sampled training records | 176,343 |
| Official validation records | 377,880 |
| Input features | 52 |
| Traffic classes | 7 |
| Sampling method | Class-stratified reservoir sampling |
| Sampling random seed | 42 |
| Missing values in saved training sample | 0 |

### Sampled Training Class Distribution

| Traffic Class | Records |
|---|---:|
| Normal Traffic | 146,594 |
| DoS | 13,553 |
| DDoS | 8,961 |
| Port Scanning | 6,309 |
| Brute Force | 640 |
| Web Attacks | 150 |
| Bots | 136 |
| **Total** | **176,343** |

The dataset is highly imbalanced, with Normal Traffic accounting
for approximately 83% of the training records.

The rare Bots and Web Attacks classes require particular attention
during model evaluation.

## 3. Model Configurations

| Parameter | Decision Tree | Random Forest |
|---|---|---|
| Maximum depth | 15 | 15 |
| Minimum samples per leaf | 10 | 10 |
| Number of trees | 1 | 50 |
| Random state | 42 | 42 |
| Parallel jobs | N/A | 1 |
| Training records | 176,343 | 176,343 |

The Random Forest worker count was limited to one to reduce
memory pressure on the Ubuntu virtual machine.

## 4. Validation Results

Both models were evaluated against all 377,880 records in the
official V3 validation dataset.

| Metric | Decision Tree | Random Forest |
|---|---:|---:|
| Accuracy | 99.77% | 99.73% |
| Macro precision | 0.95 | 0.98 |
| Macro recall | 0.90 | 0.87 |
| Macro F1-score | 0.92 | 0.91 |
| Training time | 8.55 seconds | 28.48 seconds |

### Per-Class F1-Scores

| Traffic Class | Decision Tree | Random Forest |
|---|---:|---:|
| Bots | 0.55 | 0.54 |
| Brute Force | 0.99 | 0.98 |
| DDoS | 1.00 | 1.00 |
| DoS | 0.99 | 0.99 |
| Normal Traffic | 1.00 | 1.00 |
| Port Scanning | 1.00 | 1.00 |
| Web Attacks | 0.90 | 0.84 |

### Rare Attack Recall

| Traffic Class | Decision Tree | Random Forest |
|---|---:|---:|
| Bots | 0.48 | 0.39 |
| Web Attacks | 0.82 | 0.73 |

## 5. Initial Findings

Both models achieved high overall accuracy on the validation split.

The Decision Tree performed slightly better in overall accuracy,
macro F1-score, and recall for Bots and Web Attacks.

Random Forest achieved higher Bots precision (0.85 versus 0.65),
but lower Bots recall.

These findings demonstrate why class-level precision, recall,
and F1-score are important in cybersecurity classification.
High overall accuracy does not guarantee strong detection of
rare attacks.

## 6. Infrastructure and Reproducibility

Experiments were conducted using an Ubuntu Server virtual machine
with approximately 1.4 GiB RAM and 2 GiB swap.

The V3 datasets were accessed through a VirtualBox shared folder.
Chunked processing was used for sample preparation and validation.

The trained models and validation reports are stored locally
under data/processed/, which is excluded from Git.

The existing model notebook and alert adapter implementation
are maintained in the NOVA repository.

## 7. Limitations

- Models were trained on a 10% sample, not the complete training split.
- The reserved final test dataset was not used.
- Potential data leakage and preprocessing effects require further review.
- Minority-class recall remains weaker, particularly for Bots.
- No statistical significance analysis has been performed.
- The model artifacts are stored locally and are not included in Git.
- End-to-end classifier-to-Qwen integration has not yet been verified.
- Evidence-unit definitions require confirmation against V3 preprocessing.

## 8. Next Steps

1. Review the preliminary model comparison with the team.
2. Investigate Bots and Web Attacks misclassifications.
3. Review preprocessing and potential leakage risks.
4. Coordinate validation prediction exports through the alert adapter.
5. Test classifier-to-Qwen integration using schema version 1.1.
6. Preserve the reserved final test dataset for the agreed final evaluation.

## 9. Related Project Files

- `notebooks/03_baseline_model_development_Oct_06_2026_Jennifer.ipynb`
- `src/alert_adapter.py`
- `tests/test_alert_adapter.py`
- `docs/alert_schema.md`

**Note:** This document records preliminary validation experiments
and should not be interpreted as final model performance.
