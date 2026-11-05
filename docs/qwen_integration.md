# NOVA Qwen Integration (`src/nova_qwen`)

Owner: Paul (Qwen developer). Status: **ready for independent team testing on the V3 workflow. A bounded run on Jenn's
real V3 validation exports passed** (20 alerts; see `docs/qwen_v3_integration_results.md`). Row alignment against the V3
CSV and the human review of explanation quality are still open. The response format is
**provisional** until the team approves it (see "What is provisional").

Team documents: testing guide `docs/qwen_testing_guide.md`, review rubric `docs/qwen_review_rubric.md`, walkthrough
`docs/qwen_integration_walkthrough.md`, V3 results `docs/qwen_v3_integration_results.md`, open requests
`docs/qwen_v3_team_requests.md`.

```
CICIDS2017 V3 validation CSV (complete, in file order)
  -> Decision Tree / Random Forest predictions (Jenn's classifier work, unchanged)
  -> schema 1.1 alert exports                 (src/alert_adapter.py, unchanged)
       outputs/decision_tree_alerts.json, outputs/random_forest_alerts.json
  -> dataset and row-alignment checks         (no model involved)
  -> a small representative sample            (the full export is never sent to Ollama)
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
| Memory | At least 8 GB free for the model with an 8K context (about 5 GB was observed in use). Notebook 03 as committed needs about 6 GB; together with the model, about 12 GB. If that is too much, stop Ollama, run `--notebook-only`, start Ollama, then run `--skip-notebook`. In Docker, set the memory limit accordingly. |
| Speed | CPU-only: about 20–30 s per alert in a 14-vCPU Linux VM. One classifier's ~63k alerts would take **weeks** on CPU, so more than 25 alerts in a run needs `--allow-large-run`. |
| Data | Only for the end-to-end run: the approved `cicids2017_V3_train_70.csv` and `cicids2017_V3_validation_15.csv`, in a folder holding only those files. **Never put `cicids2017_V3_test_RESERVED_15.csv` there**; the tools refuse test and reserved file names. |

## Setup (macOS / Ubuntu)

Run these from the repository root. Check your Python first: `python3 --version` must show 3.11 or newer (3.12 and 3.13 are tested). If it shows 3.9 or 3.10, install a newer Python and use that in the command below.

**Ubuntu / WSL2 only:** the `venv` module is a separate package. Install it first (Ubuntu 24.04's `python3` is 3.12):

```bash
sudo apt install python3 python3-venv
```

Create the environment:

```bash
python3 -m venv .venv
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
- `--all` explains every alert. A run selecting more than 25 alerts is refused before the model is contacted unless you also pass `--allow-large-run`; this applies to `--limit`, `--all`, `--alert-id` and resumed runs.
- `--dataset-version V3` records that version in the results file as declared and unverified. `scripts/qwen_e2e_from_notebook.py` records what its checks established.
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
| 0 | Finished with nothing in row 1. For `explain` this does not mean every answer was accepted: rejected answers still exit 0 unless you pass `--strict`, so read `summary` |
| 1 | `verify`, `check-dataset` or `check-alignment` failed, an alert failed, or (with `--strict`) an answer was rejected |
| 2 | Invalid input, arguments, results file or resume request; a missing dataset file; an unreachable input path; a refused test split; a selection above the limit |
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
must be judged by people with `worksheet` and `docs/qwen_review_rubric.md`. The worksheet uses the five criteria of
`docs/qwen_evaluation_criteria.md`, each 0–2. Prioritization stays `N/A` until a policy is approved, so totals are out of 8.
Report each column separately rather than one average.

## Results file (`nova-qwen-explanations`, version `0.1.0-provisional`)

Abbreviated, illustrative example of the file's shape. The sample values come from an older pre-V3 export; the real V3 results are summarised in `docs/qwen_v3_integration_results.md`. A real file also contains `model.capabilities/expected_digest/family/parameter_size/size_bytes`,
all `model.settings`, and per attempt `model`, `done`, `wall_seconds`, `total_duration_seconds`,
`load_duration_seconds`, `thinking_chars` and `raw_truncated`. Rejected results add `rejected_response`, and
resumed runs add a top-level `resumed_from`. `provenance` records the dataset identity this run's caller established (null if
none was given or stored). A resume that establishes less keeps the earlier, stronger identity in
`provenance.previous_provenance`. `source.dataset` is copied from the export and is not proof of the version: the notebook writes
`CICIDS2017` regardless of the file it read.

```json
{
  "output_schema": "nova-qwen-explanations", "output_schema_version": "0.1.0-provisional",
  "generator": {"component": "nova_qwen", "version": "0.2.0", "prompt_version": "nova-qwen-prompt-1.0",
                "response_schema_version": "nova-qwen-response-0.1-provisional"},
  "run_status": "complete", "created_utc": "...", "updated_utc": "...",
  "source": {"file_name": "decision_tree_alerts.json", "sha256": "...", "size_bytes": 15504498, "schema_version": "1.1",
             "run_id": "validation-...Z", "dataset": "CICIDS2017", "detector": "decision_tree", "alert_count": 63465},
  "provenance": {"dataset_version": "V2-LEGACY-REGRESSION", "verified": false, "v3_profile_matched": false, "validation_sha256": "..."},
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
- **abstained:** a validated answer where the model set `abstain: true`. It does not mean the answer declined to interpret the evidence: in the real V3 run every abstained answer still did (see `docs/qwen_review_rubric.md`).
- **rejected:** the model answered but failed the checks; `response` is null, and `rejected_response` and `attempts[].raw_response` are kept for audit.
- **failed:** no answer, because of a transport or model error or a budget refusal; see `error`.

Results are written atomically after **every** alert, with normal file permissions (0644 under the usual umask).

### What is provisional (needs team approval)

- **Response fields:** `observations`, `explanation`, `limitations`, `evidence_used`, `priority`, `abstain`, adapted from the "Suggested response fields" in `docs/alert_schema.md`. `alert_id` and the other identifiers are attached in code.
- **Priority:** `priority` is always `null`, because **no priority policy has been approved**. None is invented.
- **Evidence allowlist and units:** the 3-feature allowlist is the adapter's provisional allowlist. Units follow CICFlowMeter conventions and `docs/alert_schema.md` (Flow Duration in microseconds, Total Fwd Packets as a count, Flow Bytes/s in bytes per second). `check-dataset` tests whether a file is consistent with them, but that is plausibility, not proof: **the V3 unit assumption still needs the team's confirmation.**
- **Dataset field value:** Jenn's V3 validation exports use `CICIDS2017-V3-validation`. The consumer accepts any clean non-empty value and does not rewrite it; a V3 run whose exports do not name V3 says so in `summary.json`.
- **Rules and prompt:** the deterministic rejection rules and the prompt (`nova-qwen-prompt-1.0`).
- **Consumer-side tightening:** the reader is stricter than the adapter. It rejects `run_id`/`dataset` with surrounding whitespace or control characters, non-canonical IDs, unknown classes and values above 1e15.

## Security notes

- **Network:** alert evidence is sent only to the configured Ollama URL. It must be loopback unless explicitly allowed. URLs with credentials, paths, port 0 or invalid ports, or other schemes are refused.
- **HTTP behaviour:** no proxy environment variables are used, redirects are not followed, every request has an overall wall-clock deadline (a watchdog closes the socket), responses are size-limited and must be strict JSON with the documented field types, and error text is capped.
- **Ground truth:** the consumer never reads labels. The reader rejects any field outside the 1.1 contract, including `ground_truth` or `Attack Type`. `check-dataset` counts labels in the files it is given, to identify them; `scripts/qwen_export_v3_alerts.py` reads labels only to compute the manifest's metrics. Neither passes labels to an export or to the model.
- **Reserved test split:** every dataset, export, results and output path is checked by name, and through symlinks, before the file is touched. Device paths such as `/dev/stdin` are refused too, because they hide the file's name. A renamed copy cannot be caught by name; V3's class counts differ between validation and the reserved split, so `check-dataset` catches it afterwards, but it will have been read. Keep it out of your working folders.
- **Model files:** `scripts/qwen_export_v3_alerts.py --dt-model/--rf-model` loads joblib or pickle files, which can run code. It requires `--trust-model-files`; use only models you made yourself. `.npy` predictions are loaded without pickle.
- **Files:** the `explain` and `worksheet` commands never overwrite an input export, a file they did not write, or (without `--overwrite`) an existing results file or review worksheet. Worksheet cells are neutralised against spreadsheet formula injection. Exception: `scripts/qwen_e2e_from_notebook.py` regenerates its own `*_explanations.json`, `correlation.json` and `summary.json` on every run, and in notebook mode its executed notebook and `notebook_run.json`; it never overwrites an existing worksheet or rubric file (a re-run writes `review_worksheet_<timestamp>.csv` and its `.rubric.txt`, named in `summary.json` → `review_worksheet`).

## Testing

Step-by-step instructions and expected results are in `docs/qwen_testing_guide.md`.

Offline tests (no Ollama; about 2 minutes):

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

Live suite (about 10–15 minutes on CPU):

```bash
NOVA_QWEN_LIVE=1 python -m pytest -p no:cacheprovider -v tests/qwen_live
```

### The V3 workflow: classifier -> adapter -> Qwen

V3 is the default dataset. The tools never fall back to another dataset if V3 is missing, and a file is not accepted as V3
because of its name. Rows, columns, class counts and evidence columns must match Kevin's notebooks.

Identify the two approved files. Replace the folder with your own:

```bash
python -m src.nova_qwen check-dataset --train /path/to/cicids2017_V3_train_70.csv --validation /path/to/cicids2017_V3_validation_15.csv
```

Create exports from saved models or predictions, with the unmodified adapter and the complete validation file in order.
Details and alternatives are in the testing guide.

```bash
python scripts/qwen_export_v3_alerts.py --validation /path/to/cicids2017_V3_validation_15.csv --dt-predictions dt_pred.npy --rf-predictions rf_pred.npy --dataset CICIDS2017-V3-validation --out-dir outputs
```

Check that each export's evidence equals the CSV at each `record_id`. An export from a filtered, sampled or reordered frame fails, unless the rows removed all come after the last alert. Rows without an alert cannot be compared, so a pass does not prove the export covered the complete frame. For V3, an alert-free tail too long to be chance for that number of alerts, or an export with no alerts at all, is also a failure, because V3's rows are shuffled, and the report shows `rows_after_last_alert`:

```bash
python -m src.nova_qwen check-alignment outputs/decision_tree_alerts.json --validation /path/to/cicids2017_V3_validation_15.csv
```

Explain a small representative sample and write verified results, correlation, a worksheet and `summary.json`:

```bash
python scripts/qwen_e2e_from_notebook.py --skip-notebook --data-dir /path/to/folder
```

Main options of `qwen_e2e_from_notebook.py`:
- `--limit N`: alerts per classifier, default 6. A negative, zero or non-numeric value is refused. More than 25 needs `--allow-large-run`.
- `--select representative|first`: `representative` (default) covers DoS, DDoS, Bots, Web Attacks, a DT/RF class disagreement (both alerted, different classes) and the longest flow when they exist, then a negative duration, a row only one classifier flagged, and the other classes if there is room. Both classifiers explain the same row where possible. `summary.json` lists what was unavailable, or did not fit.
- `--data-dir`, or `--train` and `--validation`: where the CSVs are. A file you name must exist. `--expected-train-sha256` and `--expected-validation-sha256` compare against published hashes; a mismatch stops the run.
- `--exports-dir`, `--out-dir`: where the exports are, and where results go.
- `--dataset-version NAME`: anything other than `V3` runs the general checks only, and is never reported as V3.
- `--no-dataset-check`: explain exports without a dataset file. The run is reported as **NOT a verified V3 run**.
- `--resume` with `--skip-notebook`: keep good results in `--out-dir` and redo only the rest (same exports, selection and settings).
- `--notebook-only`, or no `--skip-notebook`: execute notebook 03 as committed (needs `requirements-qwen-e2e.txt`). The notebook's own path decides what it reads, so the script parses it, refuses if it cannot tell which files are read, and hashes those files. It also stops if the notebook did not write new exports. This is not the sampler in Jenn's report.

`summary.json` records the dataset identity, the files, hashes and paths used, row alignment, the selection and why each
alert was chosen, each classifier's counts, and the overall verdict. See the verdict table in the testing guide.
`dataset.v3_profile_matched` is true if the validation file matches Kevin's notebooks and every alert in both exports
matched its row. `dataset.real_v3_verified` also needs a matching published hash, because fabricated values or a different
split with the same counts would match the profile. No hash has been published yet.

A run refused before the notebook or inference changes nothing in `--out-dir`, including a `--resume` it cannot finish. A
run that starts (in notebook mode, when the notebook starts) replaces `summary.json` at once, so an earlier verdict cannot
survive it. In notebook mode, a refusal after the notebook has finished still leaves the new exports and the notebook's
own files, because that run did happen. Files in `--out-dir` that this tool did not write are refused, not replaced; a
worksheet or rubric file already there is kept, and the run writes a timestamped pair beside it.

## Known limitations

- **Scope:** Qwen explains alerts the classifiers raised. It does not change predictions and never sees normal-classified rows, so it **cannot recover a false negative**. Good explanations are also not evidence of fewer false positives. See `docs/qwen_review_rubric.md`.
- **CPU throughput:** 13–24 s per model request on CPU in the real V3 runs (median 16–19 s), and up to 40 s for an alert that needed a corrective re-ask. Full-export explanation needs a GPU or sampling.
- **Semantic checks:** these are pattern-based heuristics. Replayed over the 68 distinct real Qwen answers saved during testing (43 from older and synthetic runs, 25 from the two real V3 runs), the current validator accepts 58 and rejects 10. The rejections are derived or approximate numbers ("over 91,000", "approximately 80 kilobytes per second", "0.47 packets per second", "approx. 400KB") and one asserted protocol. Every rejected answer in the real V3 run was followed by an accepted re-ask. Unusual phrasings can still slip through or be rejected. Use the human-review worksheet.
- **Known validator gaps:** a bare `m` for minutes ("lasted 101 m") is not checked as a time unit; a sub-second unit written before the number ("Flow Duration in microseconds was 101.041626", "Flow Duration (ms): 101.041626") is not checked either (hour, day, week and year words before the number are rejected); a value given for the wrong feature is missed unless the sentence names exactly one feature and gives one number, for example "The byte rate was 11 bytes per second" or a multi-feature sentence with swapped values (see Cross-feature values); negation is scoped to clauses by keyword rules, so unusual sentence structures can be excused or rejected wrongly.
- **Not caught by any rule (from the real V3 run):** a rate described as an amount in words ("significant volume transferred in that instant" for 52,434 bytes per second over 0.000267 s, about 14 bytes); general-knowledge statements about what attacks "typically" do; and named techniques that are not in the evidence (for example "slowloris" or "beaconing"). These need the human review.
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
| `RESUME_MISMATCH` | Different export, selection, model or settings than the file you are resuming. Run `--resume` without selection flags, or use a new `--out`. With the end-to-end script, rerun with the original selection, or move a file left over from an older run out of `--out-dir`. Without `--resume` the run starts over. |
| `CONFIG_REMOTE_HOST` | Ollama is not on this machine. Only if that is intended: `--allow-remote-ollama`. |
| `OUTPUT_EXISTS` / `OUTPUT_NOT_OURS` / `OUTPUT_IS_INPUT` | Use `--resume` or `--overwrite`, or choose another `--out`. The tool never overwrites inputs or files it did not write. |
| `ALERT_FILE_UNREADABLE` … `Permission denied` | The adapter wrote the export 0600 as another user. Run as the same user, or `chmod 644` the export. |
| `DATASET_MISSING` / `EXPORT_MISSING` | The named file is not there. No other dataset is substituted. See the testing guide, step 6. |
| `DATASET_UNREADABLE` | The CSV is not readable as UTF-8 CSV. Download it again. |
| `SAME_CLASSIFIER_SOURCE` / `MODEL_DETECTOR_MISMATCH` | The same file, or identical predictions on V3, was given for both classifiers, or the model files look swapped. |
| `NOTEBOOK_TOOLS_MISSING` | Notebook mode needs nbconvert and ipykernel. `python -m pip install -r requirements-qwen-e2e.txt` |
| `MODEL_NEEDS_JOBLIB` | A compressed model file was given without joblib installed. `python -m pip install -r requirements-qwen-e2e.txt` |
| `EXPORT_MOVE_INCOMPLETE` | The final move of the export files failed partway, so `--out-dir` may hold a mixed set. Fix the cause and run the export again. |
| `RESERVED_SPLIT_REFUSED` | The file name marks a test or reserved split. Use the validation file. If it is your own `--out` file, choose another name; a results file already written under such a name can be renamed and then resumed or read. |
| `LARGE_RUN_NOT_AUTHORIZED` | Select fewer alerts, or pass `--allow-large-run` if the run really should take that long. |
| `PREDICTIONS_NOT_COMPLETE` | Exports need one prediction per validation row, in file order. A sample or SMOKE subset cannot be exported. |
| Dataset check failed on V3 | A row, column or class count differs from Kevin's notebooks, so this is not the V3 file as described. Check with Kevin before overriding with another `--dataset-version`. |
| `UNSUPPORTED_SCHEMA_VERSION` | The export is not schema 1.1. Use exports from `feature/data-exploration` @ `5cbbabc9` or later. If the message says `schema_version None`, the file is not an alert export (for example a results file, `correlation.json` or `summary.json`); pass `outputs/<detector>_alerts.json`. |
| `No module named 'src'` | Run the command from the repository root (the folder that contains `src/`). |
| Many `rejected` answers | Look at `validation.errors` in the results file and report it; do not loosen the checks. |
