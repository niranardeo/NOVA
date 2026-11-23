# NOVA — Qwen Explanation and Alert Evaluation Criteria

## 1. Purpose

This document defines the proposed evaluation criteria for
Qwen3.5-4B when processing network anomaly alerts generated
by NOVA's Decision Tree and Random Forest classifiers.

The goal is to measure whether AI-generated explanations
are accurate, evidence-based, consistent, and useful for
security analysts.

These criteria are preliminary and require team review.

## 2. Evaluation Criteria

### Explanation Accuracy
Does Qwen explain the classifier prediction correctly
without presenting the prediction as a confirmed attack?

### Evidence Grounding
Does Qwen use the network-flow evidence provided in the
JSON alert without inventing additional information?

### Alert Prioritization
Does Qwen assign a priority consistent with agreed
security risk criteria?

Priority levels and their definitions must be agreed
upon before testing.

### Consistency
Does Qwen produce reasonably consistent explanations
and priorities when processing identical alerts?

### Uncertainty Handling
Does Qwen acknowledge missing information, insufficient
evidence, and limitations in the classifier prediction?

## 3. Proposed Scoring Method

Each criterion will receive a score from 0 to 2:

0 = Does not meet the criterion
1 = Partially meets the criterion
2 = Fully meets the criterion

Maximum score per evaluated response: 10 points.

Scores will be assigned using a human-reviewed rubric.
The team must define specific scoring examples before
formal evaluation.

## 4. Proposed Test Scenarios

- DoS prediction with available network-flow evidence
- PortScan prediction with available network-flow evidence
- Prediction with limited supporting evidence
- Identical alert processed multiple times
- Empty alert list
- Conflicting predictions from different classifiers

## 5. Proposed Evaluation Metrics

- Average explanation quality score
- Percentage of explanations using only supplied evidence
- Percentage of responses containing unsupported claims
- Priority agreement rate against human-reviewed labels
- Percentage of responses acknowledging insufficient evidence
- Consistency across repeated evaluations

## 6. Responsibilities

Classifier integration:
Prepare and validate structured JSON alerts from
Decision Tree and Random Forest predictions.

Qwen integration:
Process the agreed JSON format and generate explanations
and proposed alert priorities.

Joint evaluation:
Agree on the response schema, scoring rubric, test cases,
and evaluation results.

## 7. Current Status

The JSON alert adapter has passed seven component tests.

Synthetic Decision Tree and Random Forest alerts have
been generated and structurally validated.

Qwen integration and AI explanation evaluation have
not yet been performed.

The evaluation criteria require review and agreement
with the Qwen developer before formal testing.
