# NOVA Classifier Alert JSON Contract

## Purpose

NOVA exports network anomaly predictions from two independent
classifiers for downstream AI explanation.

Supported detectors:
- decision_tree
- random_forest

## JSON Structure

{
  "schema_version": "1.1",
  "run_id": "validation-20261027T183300000000Z",
  "dataset": "CICIDS2017",
  "detector": "random_forest",
  "alerts": [
    {
      "alert_id": "rf-000002",
      "record_id": "val-000002",
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
- run_id: Identifier shared by classifier exports from the same
  validation execution. The notebook generates a UTC timestamp-based ID.
- dataset: Name of the dataset used for the predictions.
- record_id: One-based position of the record in the validation
  dataset, formatted as val-000002. It is not an original
  CICIDS2017 source-record identifier.

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
11. run_id and dataset are required non-empty strings.
12. Each exported alert includes a record_id corresponding to
    its one-based validation dataset row position.
13. Decision Tree and Random Forest exports from the same
    validation execution share the same run_id and record_id
    convention.
14. Evidence values must be numeric, finite, and non-boolean.
    Missing values, NaN, infinity, and numeric strings are rejected.
15. JSON exports are written to a temporary file and atomically
    replace the destination after successful serialization.
16. If serialization fails, an existing destination file is
    preserved and the temporary file is cleaned up.

## Current Limitations

## Current Limitations

- Alert IDs are unique only within a detector export and may
  repeat across validation runs. Use run_id, detector, and
  alert_id together for traceability.
- record_id represents the validation row position, not the
  original CICIDS2017 source-record identifier.
- No prediction confidence score is currently exported.
- Evidence is restricted to a provisional allowlist:
  Flow Duration, Total Fwd Packets, and Flow Bytes/s.
- Evidence units are not yet explicitly represented in the
  JSON contract. This can cause incorrect AI explanations.
- The evidence allowlist and units require confirmation
  against the processed CICIDS2017 dataset.
- Random Forest reproducibility across platforms requires
  further investigation.
- Potential training and validation data overlap requires
  investigation before final performance claims.
- Qwen testing identified unsupported statements, incorrect
  evidence units, and hallucinated alerts from empty inputs.
- Empty-alert handling and evidence-grounded explanations
  require further integration testing.
- Evidence privacy and data-sharing approval remain required.
- The complete classifier-to-Qwen pipeline has not yet been
  verified using schema version 1.1.

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


## Qwen Handoff — Integration Guidance

### Current Integration Status

NOVA uses Decision Tree and Random Forest classifiers to
identify potentially malicious network traffic.

The notebook includes a JSON export integration for both
classifiers using src/alert_adapter.py.

Synthetic JSON alerts have been generated and validated.
Actual CICIDS2017 model predictions have not yet been
exported through this integration.

### Current Integration Status

NOVA uses Decision Tree and Random Forest classifiers to
identify potentially malicious network traffic.

The baseline notebook includes JSON export integration for
both classifiers using src/alert_adapter.py.

Independent Ubuntu validation confirmed successful exports
from actual CICIDS2017 validation predictions:

- Decision Tree: 63,465 alerts exported.
- Random Forest: 63,324 alerts exported.
- Exported alerts were matched to their source validation
  rows with zero mismatches.
- The baseline notebook executed successfully on Ubuntu.

These results were obtained using the earlier alert contract.
Schema version 1.1 introduces additional traceability metadata
and stricter validation that still require end-to-end verification.

Qwen3.5-4B has been tested separately using Ollama, but the
complete classifier-to-Qwen integration has not yet been
verified end to end.

### JSON Files for Qwen

Decision Tree:
outputs/decision_tree_alerts.json

Random Forest:
outputs/random_forest_alerts.json

Each file contains predicted attack classifications and
approved network-flow evidence.

### Qwen Responsibilities

1. Read and validate the JSON alerts.
2. Explain predicted attacks using the supplied evidence.
3. Identify uncertainty and limitations.
4. Never treat predictions as confirmed attacks.
5. Never invent missing evidence or confidence scores.
6. Preserve alert IDs for traceability.
7. Handle empty alert lists correctly.

### Remaining Qwen Integration Work

The complete classifier-to-Qwen pipeline has not yet been
verified end to end using schema version 1.1.

The final Qwen response format requires team agreement.

Before final integration, the team must verify that Qwen:

- Handles empty alert lists without inventing attacks.
- Uses only evidence provided in the input JSON.
- Does not invent missing feature values or confidence scores.
- Interprets network-flow feature units correctly.
- Preserves run_id, detector, record_id, and alert_id
  for traceability.
- Clearly distinguishes predicted attacks from confirmed
  security incidents.

See the main Current Limitations section for outstanding
classifier, dataset, and evidence-quality concerns.
