# NOVA Qwen Integration (`src/nova_qwen`)

Owner: Paul (Qwen developer). Status: **ready for independent team testing**. The response format is
**provisional** until the team approves it (see "What is provisional"). Team testing guide: see the release
notes that accompany the branch; quick start below.

```
processed CICIDS2017 features
  -> Decision Tree / Random Forest            (notebooks/03_..._Jennifer.ipynb, unchanged)
  -> schema 1.1 alert exports                 (src/alert_adapter.py, unchanged)
       outputs/decision_tree_alerts.json, outputs/random_forest_alerts.json
  -> Qwen consumer                            (src/nova_qwen)
       strict 1.1 validation -> unit normalisation in Python -> one alert per request
       -> Ollama /api/chat (qwen3.5:4b-q4_K_M, JSON-schema constrained) -> deterministic answer checks
  -> validated explanations                   outputs/qwen/<detector>_explanations.json
```

## Requirements

| Item | Requirement |
|---|---|
| Python | **3.12 or 3.13** recommended and tested (macOS 3.13.7; Ubuntu 24.04 3.12.3; Debian `python:3.12-slim` (x86-64) and `python:3.13-slim` (arm64)). The tests and the notebook need **≥ 3.11** (pandas 3.0.3). The consumer itself uses only the standard library. macOS's built-in `/usr/bin/python3` (3.9) is too old; install 3.12 or 3.13, e.g. from python.org or `brew install python@3.12`. |
| Ollama | 0.40.11 tested (`ollama/ollama@sha256:dea72d1e…`). Native install (https://ollama.com/download) or the Docker setup in `docker/qwen/`. |
| Model | `qwen3.5:4b-q4_K_M`, digest `d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04` (about 3.3 GB on disk; the pull's main layer is 2.6 GB). |
| Memory | At least 8 GB free for the model with an 8K context (about 5 GB was observed in use). The one-step notebook-plus-Qwen run needs about 12 GB free (notebook about 5 GB plus model about 5 GB); otherwise use the two-step mode. |
| Speed | CPU-only: about 20–30 s per alert in a 14-vCPU Linux VM. `--all` on one classifier's ~63k alerts would take **weeks** on CPU; use `--limit` or `--alert-id`. |
| Data | Only for the end-to-end run: the approved `training_dataset_CIC_70.csv` and `validation_dataset_CIC_15.csv` in `data/processed/`. **Never put the reserved test split there.** |

## Setup (macOS / Ubuntu)

Run these from the repository root. Create a virtual environment with Python 3.12 or 3.13 (use whichever is installed; `python3 --version` must not show 3.9 or 3.10).

**Ubuntu / WSL2 only:** the `venv` module is a separate package. Install it first (Ubuntu 24.04's `python3` is 3.12):

```bash
sudo apt install python3 python3-venv
```

Create the environment (on Ubuntu use `python3 -m venv .venv`; on macOS `python3.13` or `python3.12`):

```bash
python3.13 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install the test dependencies. The consumer itself needs none; `requirements-qwen.txt` is intentionally empty.

```bash
python -m pip install -r requirements-qwen-dev.txt
```

Pull the pinned model (about 3.3 GB on disk, once):

```bash
ollama pull qwen3.5:4b-q4_K_M
```

Check that Ollama is reachable, the model is present, the digest matches and inference works:

```bash
python -m src.nova_qwen check-ollama --inference
```

**Windows:** use WSL2 with Ubuntu and follow the Ubuntu steps (untested on native Windows).

**Docker instead of a native Ollama:** see the header of `docker/qwen/compose.yaml`. It is optional. On Linux, add
`--user "$(id -u):$(id -g)"` to `docker compose run` so result files in `outputs/` are owned by you.

## Usage

No data yet? The commands below also work on the bundled fixtures `tests/qwen/fixtures/decision_tree_alerts.json` and
`tests/qwen/fixtures/random_forest_alerts.json` (5 alerts each, written by the unmodified adapter). With the fixtures,
pass only IDs they contain to `--alert-id` (DT: dt-000002, dt-000004, dt-000005, dt-000006, dt-000008; RF:
rf-000002, rf-000003, rf-000004, rf-000006, rf-000008); `dt-002550` in the example below exists only in a real export.
`explain` needs Ollama, except on the `*_alerts_empty.json` fixtures.

Validate an export (no model call):

```bash
python -m src.nova_qwen validate-export outputs/decision_tree_alerts.json
```

Explain the first 10 alerts of each classifier. A new run defaults to `--limit 10`.

```bash
python -m src.nova_qwen explain outputs/decision_tree_alerts.json --detector decision_tree --limit 10 --out outputs/qwen/decision_tree_explanations.json
```

```bash
python -m src.nova_qwen explain outputs/random_forest_alerts.json --detector random_forest --limit 10 --out outputs/qwen/random_forest_explanations.json
```

Explain specific alerts:

```bash
python -m src.nova_qwen explain outputs/decision_tree_alerts.json --alert-id dt-000002 --alert-id dt-002550 --out outputs/qwen/dt_selected.json
```

Continue an interrupted run with the same export, selection and settings. Without selection flags, the previous selection is used:

```bash
python -m src.nova_qwen explain outputs/decision_tree_alerts.json --out outputs/qwen/decision_tree_explanations.json --resume
```

Check a results file against its source export (identifiers, classes, raw evidence, the selection, the recorded model output, re-validation):

```bash
python -m src.nova_qwen verify outputs/qwen/decision_tree_explanations.json --export outputs/decision_tree_alerts.json
```

Compare DT and RF for the same run. This refuses if the run, the dataset or the row positions differ:

```bash
python -m src.nova_qwen correlate --dt-export outputs/decision_tree_alerts.json --rf-export outputs/random_forest_alerts.json --dt-results outputs/qwen/decision_tree_explanations.json --rf-results outputs/qwen/random_forest_explanations.json --out outputs/qwen/correlation.json
```

Build a human-review worksheet. It writes `review_worksheet.csv` plus `review_worksheet.rubric.txt`:

```bash
python -m src.nova_qwen worksheet outputs/qwen/decision_tree_explanations.json outputs/qwen/random_forest_explanations.json --out outputs/qwen/review_worksheet.csv
```

Other `explain` options:
- `--offset N` skips the first N alerts. It cannot be combined with `--alert-id`, and an offset past the end is refused.
- `--all` explains every alert.
- `--overwrite` replaces an existing results file.
- `--strict` exits 1 if any answer was rejected.
- `--max-attempts`, `--retries`, `--timeout`, `--num-ctx`, `--num-predict` and `--seed` override the settings.
- `verify --allow-incomplete` accepts an interrupted file.

**Resume rules.**
- `--resume` with a *different* selection, export, model digest, component or prompt version, or settings is refused (`RESUME_MISMATCH`). Earlier results are never silently dropped.
- Earlier accepted answers are reused only if the alert is unchanged and the answer still passes today's checks.
- Failed and rejected alerts are reprocessed.
- `--resume` on a file that does not exist yet starts a normal run.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | `verify` failed, an alert failed, or (with `--strict`) an answer was rejected |
| 2 | Invalid input, arguments, results file or resume request |
| 3 | Ollama or model problem: unavailable, model missing, digest mismatch, or `RUN_ABORTED` (3 consecutive alerts could not reach the model; the file is kept as `interrupted`, so fix Ollama and use `--resume`) |
| 4 | Output problem: would overwrite an input or a file this tool did not write, file exists (use `--overwrite` or `--resume`), or the write failed |
| 130 | Interrupted (Ctrl-C). Partial results are kept with `run_status: "interrupted"`; use `--resume`. |

## Settings

The defaults below are recorded in every results file. Settings with an environment variable or flag can be
overridden. `think=false` and `temperature=0` are **fixed** for determinism.

| Setting | Default | Override |
|---|---|---|
| Ollama URL | `http://127.0.0.1:11434` | `NOVA_OLLAMA_URL`, `--url` |
| Model | `qwen3.5:4b-q4_K_M` | `NOVA_QWEN_MODEL`, `--model` |
| Required digest | `d8b0f5e9…38c04` (`''` disables the check; not recommended) | `NOVA_QWEN_EXPECTED_DIGEST`, `--expected-digest` |
| Seed | 42 | `NOVA_QWEN_SEED`, `--seed` |
| num_ctx / num_predict | 8192 / 512 | `NOVA_QWEN_NUM_CTX`, `NOVA_QWEN_NUM_PREDICT`, `--num-ctx`, `--num-predict` |
| Overall time limit per model request | 180 s (connect 5 s, fixed) | `NOVA_QWEN_TIMEOUT_SECONDS`, `--timeout` |
| Transport retries / answers per alert | 2 / 2 (first answer + 1 corrective re-ask) | `--retries`, `--max-attempts` |
| Stop after consecutive unreachable alerts | 3 | fixed |
| Non-loopback Ollama host | refused | `NOVA_QWEN_ALLOW_REMOTE_OLLAMA=1`, `--allow-remote-ollama` |

## How each known failure is handled

| Earlier failure (audits) | Handling in this component |
|---|---|
| Flow Duration (µs) read as seconds ("101 s = 3.18 years") | Converted to seconds **in Python** (`features.py`, exact decimal arithmetic). The model only sees `"value": "101.041626", "unit": "seconds"`; raw values are kept in `evidence` and `evidence_normalized[].raw_value`. Common forms of time statements that do not equal the duration are rejected (`TIME_UNIT_MISMATCH`), including spelled-out ("about three years"), hyphenated ("101041626-second") and multiplier ("101 million seconds") forms, and unattached hour/day/year words. Known gaps: a bare `m` ("101 m") is not read as minutes, and a sub-second unit written *before* the number ("Flow Duration in microseconds was 101.041626") is not checked; see Known limitations. |
| Empty alert list → invented alert | An empty export never contacts Ollama, not even the version endpoint. Results are `[]` with `model.contacted: false`. |
| "Physically impossible… data corruption" without cause | Plausibility checks run in Python (negative values, fractional packet counts) and are passed as `data_quality_notes`. Corruption or impossibility claims without a note are rejected (`UNSUPPORTED_DATA_QUALITY_CLAIM`). |
| Over-length prompts silently truncated (HTTP 200) | One alert per request. Before sending, `UTF-8 bytes + 128 + num_predict ≤ num_ctx` is required (Qwen's byte-level tokenizer cannot produce more tokens than bytes), so Ollama cannot truncate. Otherwise `PROMPT_BUDGET_EXCEEDED` and nothing is sent; this also applies to corrective re-asks. Truncated *generations* (`done_reason != "stop"`) are rejected. |
| Classifier context lost or merged | Identifiers, detector, run and predicted class are copied from the export in code and never generated by the model. DT and RF results stay in separate files. `correlate` joins them only after checking run, dataset and identical evidence on every shared `record_id`. |
| Invalid, incomplete or unsupported output | JSON-schema constrained decoding, then strict validation. Invalid answers get one corrective re-ask; if still invalid, `status: "rejected"`, `response: null`, and the raw answer is kept only for audit. |

## Deterministic answer checks (`response.py`)

These are **errors**; the answer is rejected:
- **Structure:** wrong keys, wrong types, over-long text, no observations, no limitations, an explanation of fewer than 5 words (unless abstaining).
- **Truncation:** generation did not end with `done_reason: "stop"`.
- **Policy:** `priority` not null; `evidence_used` naming features that were not supplied.
- **Numbers:** any number not equal to a supplied value, up to the rounding of the written number. A sentence that names exactly one feature and gives one value must give *that* feature's value. The raw microsecond duration is accepted only with a microsecond unit.
- **Time units:** see the Flow Duration row above.
- **Certainty:** "confirms / definitely / proves / certainly" wording; asserting that the traffic *is* an attack ("This is a DoS attack"); confidence, probability or percentage claims.
- **Unsupported content:** data-quality claims without a note; invented IP addresses, port numbers, protocols or TCP flags.

**Negation is scoped to its clause.** A claim is excused only by a negation earlier in the *same clause* (for example "not", "no", "cannot", "lacks", "restricts"). A clause ends at `;` or `:`; at *but, although, however, while, yet, because, since, so* or *therefore*; where a comma or "and/or" starts a new subject ("…, this confirms"); and after factive words ("not surprising that this is…").
- **Passes:** negated lists stay together, so "The evidence cannot confirm the attack type, identify the IPs, or establish intent" passes.
- **Rejected:** "Payloads are not available, this confirms a DoS attack" and "There is no doubt that…".
- **Hedges:** "would be needed to confirm", "a possible DoS attack" and "classified as DoS" are statements about the prediction, not certainty claims.

**Field roles.** `limitations` exist to say what is *not* known. There, protocol names, confidence words and unattached time words are allowed ("no TCP flags or confidence scores are available"). In `observations` and `explanation` they are rejected unless negated.

These are **warnings**, kept for human review: other attack classes mentioned, a data-quality note ignored, an observation that names no feature, and thinking text returned.

**Passing these checks does not prove an explanation is correct or useful.** A false statement phrased without
numbers, units or forbidden wording can pass, for example calling a packet *count* a *rate*. Semantic quality
must be judged by people with `worksheet` and its rubric. The rubric is *adapted from* the team's
`docs/qwen_evaluation_criteria.md`, not a one-to-one copy (its header line says "maps to"): H1 and H6 are scored 0–2 and
H2–H5 0–1, and it has no prioritization or consistency criterion. Report each column separately rather than one average.

## Results file (`nova-qwen-explanations`, version `0.1.0-provisional`)

Abbreviated example. A real file also contains `model.capabilities/expected_digest/family/parameter_size/size_bytes`,
all `model.settings`, and per attempt `model`, `done`, `wall_seconds`, `total_duration_seconds`,
`load_duration_seconds`, `thinking_chars` and `raw_truncated`. Rejected results add `rejected_response`, and
resumed runs add a top-level `resumed_from`.

```json
{
  "output_schema": "nova-qwen-explanations", "output_schema_version": "0.1.0-provisional",
  "generator": {"component": "nova_qwen", "version": "0.1.0", "prompt_version": "nova-qwen-prompt-1.0",
                "response_schema_version": "nova-qwen-response-0.1-provisional"},
  "run_status": "complete", "created_utc": "...", "updated_utc": "...",
  "source": {"file_name": "decision_tree_alerts.json", "sha256": "...", "size_bytes": 15504498, "schema_version": "1.1",
             "run_id": "validation-...Z", "dataset": "CICIDS2017", "detector": "decision_tree", "alert_count": 63465},
  "selection": {"mode": "range", "alert_ids": null, "offset": 0, "limit": 1, "selected": 1},
  "model": {"contacted": true, "name": "qwen3.5:4b-q4_K_M", "digest": "d8b0f5e9...", "digest_verified": true,
            "ollama_version": "0.40.11", "quantization": "Q4_K_M", "settings": {"num_ctx": 8192, "seed": 42}},
  "summary": {"explained": 1, "abstained": 0, "rejected": 0, "failed": 0, "selected": 1, "processed": 1, "pending": 0},
  "results": [{
    "alert_id": "dt-000002", "record_id": "val-000002", "record_position": 2, "detector": "decision_tree",
    "run_id": "validation-...Z", "dataset": "CICIDS2017", "schema_version": "1.1",
    "predicted_class": "DoS", "prediction_status": "unverified classifier prediction",
    "evidence": {"Flow Duration": 101041626.0, "Total Fwd Packets": 11.0, "Flow Bytes/s": 123.7509776},
    "evidence_normalized": [{"feature": "Flow Duration", "raw_value": 101041626.0, "raw_unit": "microseconds",
                             "value": 101.041626, "unit": "seconds", "display": "101.041626"}],
    "data_quality_notes": [],
    "status": "explained",
    "response": {"observations": ["..."], "explanation": "...", "limitations": ["..."],
                 "evidence_used": ["Flow Duration"], "priority": null, "abstain": false},
    "validation": {"passed": true, "errors": [], "warnings": []},
    "attempts": [{"attempt": 1, "done_reason": "stop", "prompt_eval_count": 551, "eval_count": 291,
                  "request_bytes": 2705, "validation_errors": [], "raw_response": "..."}],
    "error": null
  }]
}
```

`status` values:
- **explained:** a validated answer.
- **abstained:** a validated answer that declines to interpret the evidence.
- **rejected:** the model answered but failed the checks; `response` is null, and `rejected_response` and `attempts[].raw_response` are kept for audit.
- **failed:** no answer, because of a transport or model error or a budget refusal; see `error`.

Results are written atomically after **every** alert, with normal file permissions (0644 under the usual umask).

### What is provisional (needs team approval)

- **Response fields:** `observations`, `explanation`, `limitations`, `evidence_used`, `priority`, `abstain`, adapted from the "Suggested response fields" in `docs/alert_schema.md`. `alert_id` and the other identifiers are attached in code.
- **Priority:** `priority` is always `null`, because **no priority policy has been approved**. None is invented.
- **Evidence allowlist and units:** the 3-feature allowlist is the adapter's provisional allowlist. Units follow CICFlowMeter conventions (Flow Duration in microseconds).
- **Rules and prompt:** the deterministic rejection rules and the prompt (`nova-qwen-prompt-1.0`).
- **Consumer-side tightening:** the reader is stricter than the adapter. It rejects `run_id`/`dataset` with surrounding whitespace or control characters, non-canonical IDs, unknown classes and values above 1e15.

## Security notes

- **Network:** alert evidence is sent only to the configured Ollama URL. It must be loopback unless explicitly allowed. URLs with credentials, paths, port 0 or invalid ports, or other schemes are refused.
- **HTTP behaviour:** no proxy environment variables are used, redirects are not followed, every request has an overall wall-clock deadline (a watchdog closes the socket), responses are size-limited and must be strict JSON with the documented field types, and error text is capped.
- **Ground truth:** labels are never read. The reader rejects any field outside the 1.1 contract, including `ground_truth` or `Attack Type`.
- **Files:** the `explain` and `worksheet` commands never overwrite an input export, a file they did not write, or (without `--overwrite`) an existing results file or review worksheet. Worksheet cells are neutralised against spreadsheet formula injection. Exception: `scripts/qwen_e2e_from_notebook.py` regenerates its own `outputs/qwen_e2e/*_explanations.json`, `correlation.json` and `summary.json` on every run; it never overwrites an existing worksheet (a re-run writes `review_worksheet_<timestamp>.csv`, named in `summary.json` → `review_worksheet`).

## Testing

Offline tests (no Ollama; about 15 s):

```bash
python -m pytest -p no:cacheprovider -q tests/qwen
```

One-command smoke test, offline:

```bash
python scripts/qwen_smoke.py
```

Smoke test against your Ollama:

```bash
python scripts/qwen_smoke.py --live
```

Live suite (about 8–15 minutes on CPU):

```bash
NOVA_QWEN_LIVE=1 python -m pytest -p no:cacheprovider -v tests/qwen_live
```

### End-to-end: classifier → adapter → Qwen

This needs the approved data in `data/processed/` and the end-to-end dependencies:

```bash
python -m pip install -r requirements-qwen-e2e.txt
```

First create the data folder and put the two approved CSVs in `data/processed/`. Only `data/raw/` and `data/processed/`
are gitignored, not the rest of `data/`, so keep other data files out of `data/`:

```bash
mkdir -p data/processed
```

One step, if you have about 12 GB of free memory:

```bash
python scripts/qwen_e2e_from_notebook.py --limit 5
```

Low-memory machines use two steps. Stop Ollama first, then run the notebook only:

```bash
python scripts/qwen_e2e_from_notebook.py --notebook-only
```

Then start Ollama again and run the Qwen part:

```bash
python scripts/qwen_e2e_from_notebook.py --skip-notebook --limit 5
```

Outputs are in `outputs/qwen_e2e/`. The script regenerates its explanations, `correlation.json` and `summary.json` on
every run, but never overwrites a worksheet: the first run writes `review_worksheet.csv`, a re-run writes
`review_worksheet_<timestamp>.csv`. Score the sheet named in `summary.json` → `review_worksheet`. Notebook 03 also
writes `decision_tree_confusion_matrix.png` in the repository root; it is not gitignored, so do not commit it.

## Known limitations

- **CPU throughput:** about 20–30 s per alert. Full-export explanation needs a GPU or sampling.
- **Semantic checks:** these are pattern-based heuristics. They were replayed over the 36 distinct real Qwen answers collected during testing (24 first attempts and 12 corrective re-asks): 30 pass the final validator. Of the 24 first attempts, 18 pass and 6 are rejected (an approximation "over 91,000", derived rates and an asserted protocol); all 12 re-ask answers pass. 5 of the 6 rejected answers were followed by an accepted re-ask; the "over 91,000" answer came from an early run whose re-ask did not fit the context, and that alert was explained in a later run. Unusual phrasings can still slip through or be rejected. Use the human-review worksheet.
- **Known validator gaps:** a bare `m` for minutes ("lasted 101 m") is not checked as a time unit; a sub-second unit written before the number ("Flow Duration in microseconds was 101.041626", "Flow Duration (ms): 101.041626") is not checked either (hour, day, week and year words before the number are rejected); a value given for the wrong feature is missed unless the sentence names exactly one feature and gives one number, for example "The byte rate was 11 bytes per second" or a multi-feature sentence with swapped values (see Cross-feature values); negation is scoped to clauses by keyword rules, so unusual sentence structures can be excused or rejected wrongly.
- **No derived numbers:** derived numbers are rejected even when correct, for example "9 packets over 98.68 s ≈ 0.091 packets per second". This is deliberate: earlier tests showed the model's arithmetic is unreliable, so the prompt forbids new numbers and the validator enforces it.
- **Cross-feature values:** a value that belongs to another feature is caught only when a sentence names one feature and gives one number.
- **Interpretations:** with only 3 features, interpretations are necessarily shallow (audit finding NOVA-25).
- **record_id joins:** `record_id` is a row position (schema 1.1). `correlate` checks what it can (same run and dataset, identical evidence on shared records) and warns when nothing can be cross-checked.
- **Random Forest platforms:** RF predictions differ slightly between macOS and Linux (audit NOVA-08). Correlate only exports from the same run.
- **Adapter permissions:** `src/alert_adapter.py` writes exports with mode 0600 (audit NOVA-28). Run the notebook and the consumer as the same OS user.

## Troubleshooting

| Symptom (CLI error code) | Fix |
|---|---|
| `OLLAMA_UNAVAILABLE` | Start Ollama (`ollama serve`, the desktop app, or `docker compose … up -d ollama`). Check `NOVA_OLLAMA_URL`. |
| `MODEL_NOT_AVAILABLE` | `ollama pull qwen3.5:4b-q4_K_M` |
| `MODEL_DIGEST_MISMATCH` | You have different weights under that tag. Re-pull, or deliberately set `NOVA_QWEN_EXPECTED_DIGEST` and report it. |
| `OLLAMA_TIMEOUT` | The CPU is slow or the first model load is in progress. Raise `--timeout`; first loads can take a minute. |
| `RUN_ABORTED`, or HTTP 500 "model runner has unexpectedly stopped" | Usually not enough memory. Give Docker or the VM at least 8 GB, then rerun with `--resume`. |
| `RESUME_MISMATCH` | Different export, selection, model or settings than the file you are resuming. Run `--resume` without selection flags, or use a new `--out`. |
| `CONFIG_REMOTE_HOST` | Ollama is not on this machine. Only if that is intended: `--allow-remote-ollama`. |
| `OUTPUT_EXISTS` / `OUTPUT_NOT_OURS` / `OUTPUT_IS_INPUT` | Use `--resume` or `--overwrite`, or choose another `--out`. The tool never overwrites inputs or files it did not write. |
| `ALERT_FILE_UNREADABLE` … `Permission denied` | The adapter wrote the export 0600 as another user. Run as the same user, or `chmod 644` the export. |
| `UNSUPPORTED_SCHEMA_VERSION` | The export is not schema 1.1. Use exports from `feature/data-exploration` @ `5cbbabc9` or later. If the message says `schema_version None`, the file is not an alert export (for example a results file, `correlation.json` or `summary.json`); pass `outputs/<detector>_alerts.json`. |
| `No module named 'src'` | Run the command from the repository root (the folder that contains `src/`). |
| Many `rejected` answers | Look at `validation.errors` in the results file and report it; do not loosen the checks. |
