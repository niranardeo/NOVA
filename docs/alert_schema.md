# NOVA Classifier Alert JSON Contract

## Purpose

NOVA exports network anomaly predictions from two independent
classifiers for downstream AI explanation.

Supported detectors:
- decision_tree
- random_forest

## JSON Structure

{
  "schema_version": "1.0",
  "detector": "random_forest",
  "alerts": [
    {
      "alert_id": "rf-000002",
      "predicted_class": "DoS",
      "evidence": {
        "Flow Duration": 200,
        "Total Fwd Packets": 10
      }
    }
  ]
}

The example above is synthetic, not a real model prediction.

## Field Definitions

- schema_version: Version of the JSON contract.
- detector: Classifier that generated the predictions.
- alerts: Array of predicted attack records.
- alert_id: Identifier unique within a detector's export.
- predicted_class: Attack class predicted by the classifier.
- evidence: Original feature values associated with the prediction.

## Processing Rules

1. Decision Tree and Random Forest produce separate JSON files.
2. Predictions labeled "Normal Traffic" are excluded.
3. Alert evidence is taken from the corresponding input feature row.
4. Ground-truth validation labels are not included.
5. Missing or infinite numeric evidence is rejected.
6. Unsupported detector names are rejected.
7. Prediction counts must match feature row counts.
8. Only explicitly allowlisted network features are exported as evidence.
9. The `Attack Type` column is rejected before processing predictions,
   including when no attack alerts are generated.
10. Unapproved feature columns are excluded from the JSON output.

## Current Limitations

- Alert IDs are local to each export and may repeat across runs.
- No confidence score is currently exported.
- Evidence is restricted to a provisional allowlist:
  `Flow Duration`, `Total Fwd Packets`, and `Flow Bytes/s`.
- The final evidence allowlist must be reviewed against the
  actual processed CICIDS2017 feature names.
- Alert IDs are not globally unique across different export runs.
- Evidence privacy and data-sharing approval remain required.
- The adapter has not yet been connected to live classifier outputs.
- This contract is provisional until reviewed by the Qwen teammate.

## Qwen Integration

The downstream AI component should use the provided prediction
and evidence to generate a grounded explanation.

It must not treat the predicted class as verified ground truth.

Suggested response fields:
- alert_id
- explanation
- priority
- evidence_used
- limitations
- abstain

The final response schema must be agreed upon by both components.
