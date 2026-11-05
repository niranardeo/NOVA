# Qwen Integration Walkthrough (10–15 minutes)

A practical demo for the team. **It uses synthetic data, so nothing here is a V3 result.** It needs no dataset. Steps 1–5
work offline; step 6 needs Ollama and the model.

| Part | Time |
|---|---|
| 1. What NOVA does | 1 min |
| 2. From classifiers to JSON | 2 min |
| 3. Reading an export | 1 min |
| 4. Unit normalisation | 2 min |
| 5. Identifiers and verification, offline | 2 min |
| 6. A real Qwen call | 3–4 min |
| 7. Limits and next steps | 2 min |

## Setup (before the meeting)

From the repository root:

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
python -m pip install -r requirements-qwen-dev.txt
```

For step 6, start Ollama and check the model. This should report `"digest_verified": true`:

```bash
python -m src.nova_qwen check-ollama --inference
```

## 1. What NOVA does

NOVA studies whether a local AI model can help analysts interpret network anomaly alerts. Decision Tree and Random Forest
classifiers flag flows from CICIDS2017. Qwen3.5-4B, running locally through Ollama, explains each flagged alert in plain
language, using only the evidence it is given. A person stays in charge.

```
V3 validation rows -> Decision Tree / Random Forest -> alert_adapter.py -> schema 1.1 JSON (one file per classifier)
  -> nova_qwen: validate -> normalise units in Python -> Qwen via Ollama -> check the answer -> verified results
  -> human review worksheet
```

## 2. Decision Tree, Random Forest and Jenn's JSON

Jenn's V3 experiments are in `docs/baseline_experiment_results.md`. They are preliminary: trained on a 10% sample,
evaluated on all 377,880 validation rows, and never run on the reserved test split.

Her adapter (`src/alert_adapter.py`) writes one JSON file per classifier. Rows predicted as normal are left out. Look at
one alert from the bundled synthetic fixture, which her real adapter wrote:

```bash
python -c "import json; d=json.load(open('tests/qwen/fixtures/decision_tree_alerts.json')); print({k: d[k] for k in ('schema_version','run_id','dataset','detector')}); print(d['alerts'][0])"
```

Point out `record_id`. It is the row's position in the validation frame, which is why exports must come from the complete
file in its original order.

## 3. How the consumer reads an export

It is strict: unknown fields, unknown classes, ground-truth labels and non-canonical IDs are all refused.

```bash
python -m src.nova_qwen validate-export tests/qwen/fixtures/decision_tree_alerts.json --detector decision_tree
```

## 4. How units are normalised

CICFlowMeter reports Flow Duration in microseconds. Earlier tests showed Qwen reading a raw 101,041,626 as about 3 years, so the conversion now happens in Python and the model only ever sees seconds:

```bash
python -c "from src.nova_qwen.features import normalize_evidence as n; [print(f.feature, f.raw_value, f.raw_unit, '->', f.display, f.unit) for f in n({'Flow Duration': 101041626.0, 'Total Fwd Packets': 11.0, 'Flow Bytes/s': 123.7509776})]"
```

The team's units are documented in `docs/alert_schema.md`, which asks for confirmation against the V3 preprocessing documentation. That confirmation is still open.

## 5. Identifiers, the synthetic dataset and verification

Make a small **synthetic** dataset with V3's columns. It is random numbers and made-up predictions:

```bash
python scripts/qwen_make_synthetic_demo.py
```

Turn its predictions into schema 1.1 exports with Jenn's unmodified adapter:

```bash
python scripts/qwen_export_v3_alerts.py --validation outputs/synthetic_demo/synthetic_validation.csv --dt-predictions outputs/synthetic_demo/dt_predictions.txt --rf-predictions outputs/synthetic_demo/rf_predictions.txt --dataset SYNTHETIC-NOT-CICIDS2017 --dataset-version SYNTHETIC-DEMO --out-dir outputs/synthetic_demo
```

Check that every alert's evidence equals the CSV at its `record_id`:

```bash
python -m src.nova_qwen check-alignment outputs/synthetic_demo/decision_tree_alerts.json --validation outputs/synthetic_demo/synthetic_validation.csv --dataset-version SYNTHETIC-DEMO
```

Show that the tooling is not fooled. The V3 check fails on this file, and the reserved split is refused by name:

```bash
python -m src.nova_qwen check-dataset --validation outputs/synthetic_demo/synthetic_validation.csv
```

```bash
python -m src.nova_qwen check-dataset --validation demo_RESERVED_copy.csv
```

The second command exits with code 2 and prints the error code `RESERVED_SPLIT_REFUSED`. The name alone triggers the refusal: no such file exists here, and nothing is opened. Do not use the real reserved file for this demo.

Show that a mistyped limit cannot queue tens of thousands of model calls. Both are refused before any inference:

```bash
python scripts/qwen_e2e_from_notebook.py --skip-notebook --limit -1
```

```bash
python scripts/qwen_e2e_from_notebook.py --skip-notebook --limit 100000
```

## 6. A real Qwen call (needs Ollama)

Run the production script on the synthetic exports with 2 alerts per classifier. It takes about 2 minutes on CPU.

```bash
python scripts/qwen_e2e_from_notebook.py --skip-notebook --dataset-version SYNTHETIC-DEMO --validation outputs/synthetic_demo/synthetic_validation.csv --exports-dir outputs/synthetic_demo --out-dir outputs/synthetic_demo/e2e --limit 2
```

The answers are from the real model, but about synthetic rows. The last line ends with `(NOT a verified V3 run)`. Abstentions are counted separately, and Qwen may well abstain on random numbers.

See one answer next to its evidence and identifiers:

```bash
python -c "import json; r=json.load(open('outputs/synthetic_demo/e2e/decision_tree_explanations.json'))['results'][0]; print(r['alert_id'], r['record_id'], r['predicted_class'], r['status']); print(r['evidence_normalized'][0]['display'], r['evidence_normalized'][0]['unit']); print(json.dumps(r['response'], indent=2))"
```

Check the results against the export independently. It recomputes identifiers, classes, evidence and normalisation from the source and re-validates each answer:

```bash
python -m src.nova_qwen verify outputs/synthetic_demo/e2e/decision_tree_explanations.json --export outputs/synthetic_demo/decision_tree_alerts.json
```

Open the worksheet named by `review_worksheet` in `outputs/synthetic_demo/e2e/summary.json` to show the review columns, and `docs/qwen_review_rubric.md` for how to score them. On a first run it is `review_worksheet.csv`; a re-run never overwrites a worksheet and writes `review_worksheet_<timestamp>.csv` instead.

**No Ollama in the room?** Show an offline empty-export case instead. It never contacts the model:

```bash
python -m src.nova_qwen explain tests/qwen/fixtures/decision_tree_alerts_empty.json --out outputs/synthetic_demo/empty.json --overwrite
```

## 7. Smoke tests, limitations and next steps

The one-command check for teammates (offline, about 2 minutes):

```bash
python scripts/qwen_smoke.py
```

**Current limits to state clearly:**
- Qwen explains alerts. It does not change predictions, and it cannot recover a false negative it never receives.
- Good explanations are not evidence that false positives or false negatives went down.
- `priority` is always `null`: no policy is approved.
- The checks are rule-based. Human review decides quality, and it has not been done yet.
- About 25 s per alert on CPU, so only small samples go to Qwen.
- A bounded run on Jenn's real V3 exports passed (20 alerts), but row alignment against the V3 CSV and the human review are still open. See `docs/qwen_v3_integration_results.md`.

**Next:** Kevin provides V3 hashes and confirms units; the alignment check runs against the V3 validation CSV; the team does the human review; nobody touches the reserved test split until the configuration is frozen.
