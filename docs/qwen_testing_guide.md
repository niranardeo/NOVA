# Qwen Integration Testing Guide (V3 workflow)

For Jenn, Kevin and anyone testing branch `paul/qwen-integration-schema11`. Please test it on your own machine and send
back the report at the end.

**What it does.** It reads the schema 1.1 alert exports, converts Flow Duration from microseconds to seconds in Python,
explains a small sample of alerts with Qwen3.5-4B through Ollama, checks every answer, and writes results that keep every
identifier, predicted class and raw evidence value exactly as exported.

**What it does not do.**
- It changes nothing in the classifier notebook, the adapter or its tests.
- It does not decide priorities (`priority` is always `null`, because none is approved).
- It does not change predictions or recover missed attacks. See `docs/qwen_review_rubric.md`.

**Never copy, open, or point any tool at `cicids2017_V3_test_RESERVED_15.csv`.** The tools refuse test or reserved file
names, but a renamed copy can only be caught after it has been opened, so please keep it out of your working folders.

## Requirements

- macOS or Ubuntu. On Windows, use WSL2 with Ubuntu (not tested).
- Python 3.12 or 3.13 (tested). 3.11 is the minimum for the test dependencies. Check yours:

  ```bash
  python3 --version
  ```

  If it prints 3.11 or newer, use `python3` below. Otherwise install 3.12 (macOS's built-in 3.9 and Ubuntu 22.04's 3.10 are too old), and use that interpreter in step 2, for example `python3.12 -m venv .venv`. Homebrew's `python@3.12` provides `python3.12`, not `python3`.
- Ubuntu 24.04: install the venv module first (its `python3` is 3.12). Tested in a clean container.

  ```bash
  sudo apt install python3 python3-venv
  ```

- For live tests: Ollama with `qwen3.5:4b-q4_K_M` (about 3.3 GB on disk), 8 GB of free memory, about 25 s per alert on CPU.
- For the real V3 run: alert exports and/or the two approved V3 CSVs. See step 6.

## 1. Get the branch

Run every command from the repository root, the folder that contains `src/`.

```bash
git clone https://github.com/niranardeo/NOVA.git && cd NOVA
```

If you already have a clone, skip that. Then:

```bash
git fetch origin
```

```bash
git switch paul/qwen-integration-schema11
```

If you tested an earlier version, bring your copy up to date. If this refuses because of local changes, commit or stash them first.

```bash
git pull --ff-only origin paul/qwen-integration-schema11
```

Confirm it contains Jenn's latest work. This should print `BASE OK`:

```bash
git merge-base --is-ancestor 7f069d34970b3a057f4e48f3978880382bb1f5bc HEAD && echo "BASE OK" || echo "BASE MISSING - stop and tell Paul"
```

## 2. Set up Python

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

This must show 3.11 or newer:

```bash
python --version
```

```bash
python -m pip install -r requirements-qwen-dev.txt
```

## 3. Offline tests (no Ollama, a couple of minutes)

The smoke test should end with `ALL PASS: 7/7 checks passed`:

```bash
python scripts/qwen_smoke.py
```

Jenn's own tests. Expect `Ran 12 tests` and `OK`:

```bash
python -m unittest discover -s tests
```

Everything. Expect **`939 passed, 11 skipped, 12 subtests passed`**; the 11 skips are the opt-in live tests. If you run as root (some containers), 11 file-permission tests also skip: `928 passed, 22 skipped`.

```bash
python -m pytest -p no:cacheprovider -q tests
```

## 4. Install Ollama and the model

Install Ollama from https://ollama.com/download and start it. Then:

```bash
ollama pull qwen3.5:4b-q4_K_M
```

This must report `"digest_verified": true`, digest `d8b0f5e9760c…` and `"ok": true`:

```bash
python -m src.nova_qwen check-ollama --inference
```

Prefer Docker for Ollama? See the header of `docker/qwen/compose.yaml`.

## 5. Live tests against your Ollama

Smoke test, 2 alerts per classifier, about 2–4 minutes. It should end with `ALL PASS: 12/12 checks passed`:

```bash
python scripts/qwen_smoke.py --live-only
```

Full live suite, about 10–15 minutes on CPU. Expect **11 passed**:

```bash
NOVA_QWEN_LIVE=1 python -m pytest -p no:cacheprovider -v tests/qwen_live
```

## 6. The real V3 run

This has three parts, and each can be done by a different person.

### 6a. Get only the two approved files

From `Nova_IT691_Dataset_Version_2_and_3/V3_Stratified_Split` (OneDrive link in the `main` README), download **individually**:

- `cicids2017_V3_train_70.csv`
- `cicids2017_V3_validation_15.csv`

Do not use "Download all": the reserved test file is in that folder. Keep them in a folder holding only these two files,
outside the repository or in `data/processed/` (gitignored). Set a shell variable to that folder (this is your own path):

```bash
V3_DIR=/path/to/folder/with/the/two/files
```

Check them. It compares rows, columns, class counts and evidence columns with Kevin's notebooks, and prints the SHA-256 of each file. That is a profile match, which fabricated values or a different split with the same counts would also pass. Only a matching published hash makes a file verified, so if Kevin has published hashes, add `--expected-train-sha256` and `--expected-validation-sha256`.

```bash
python -m src.nova_qwen check-dataset --train "$V3_DIR/cicids2017_V3_train_70.csv" --validation "$V3_DIR/cicids2017_V3_validation_15.csv"
```

### 6b. Create the alert exports

The exports must come from the **complete validation file in its original row order**, or `record_id` loses its meaning. Do not export from a sample or SMOKE subset. Use one route:

**Route 1: saved models or predictions from your classifier run (best for Jenn's VM; uses little memory).** With saved predictions, only the evidence and label columns are loaded. With saved models, all columns are read one chunk at a time to predict, and only the evidence and label columns are kept for the export and metrics. With saved models:

```bash
python -m pip install -r requirements-qwen-e2e.txt
```

```bash
python scripts/qwen_export_v3_alerts.py --validation "$V3_DIR/cicids2017_V3_validation_15.csv" --dt-model DT_MODEL_FILE --rf-model RF_MODEL_FILE --trust-model-files --dataset CICIDS2017-V3 --out-dir outputs
```

Replace `DT_MODEL_FILE` and `RF_MODEL_FILE` with your saved models. Loading a model file runs code from it, so use only models you made yourself. With saved predictions instead, use `--dt-predictions FILE --rf-predictions FILE`: text with one class name per line, or a `.npy` string array, one per validation row, in file order. `predict()` returns an object array, which is refused because loading it would need pickle, so save it with `np.save("dt_pred.npy", y_pred_dt.astype(str))`.

Add `--expected-validation-sha256` once Kevin has published the hash. A failure before the final move leaves `--out-dir` as it was. Trust the folder only after the run prints `EXPORT DONE`. A hard kill (`kill -9` or the out-of-memory killer) can leave a hidden `.export-staging-*` folder there. No tool reads it, and it is safe to delete once no export is running.

It writes `outputs/decision_tree_alerts.json`, `outputs/random_forest_alerts.json` and `outputs/export_manifest.json`. The manifest records the file hash, the model settings, and the accuracy and per-class recall the predictions really give. Please compare those with your report.

**Route 2: your own notebook run.** Run the export cell at the end of notebook 03 after the models are trained. Keep `X_val` complete and in file order.

**Route 3: execute notebook 03 as committed (about 6 GB of memory with Ollama stopped).** It reads `/media/sf_NOVA_Data`, so that path must exist. This is not the sampler in Jenn's report (`train_test_split` and `n_jobs=2`, against reservoir sampling and `n_jobs=1`), so treat the result as a separate reproduction, not her experiment. It does not fit Jenn's VM (about 1.4 GiB RAM plus 2 GiB swap); use Route 1 there. On a machine with enough memory where that path exists:

```bash
python -m pip install -r requirements-qwen-e2e.txt
```

```bash
python scripts/qwen_e2e_from_notebook.py --notebook-only
```

On another machine, Docker can provide that path. Give Docker enough memory in its settings. Tested on Docker Desktop for Mac; on Linux, run `sudo chown -R "$USER" .` from the repository root afterwards:

```bash
docker run --rm -v "$PWD":/work -v "$V3_DIR":/media/sf_NOVA_Data:ro -w /work python:3.12-slim bash -c "python -m pip -q install -r requirements-qwen-e2e.txt && python scripts/qwen_e2e_from_notebook.py --notebook-only"
```

Without Docker, it prints the exact command for step 6c when it finishes; run it from the repository root. After the Docker route, that printed command holds container paths (`/media/sf_NOVA_Data`, `/work`) and is refused on the host, so use the step 6c command below with `--data-dir "$V3_DIR"` instead. The notebook also writes `decision_tree_confusion_matrix.png` in the repository root. Do not commit it.

### 6c. Explain a small sample with Qwen

Do not send a whole export to Ollama: one classifier's tens of thousands of alerts would take weeks on CPU. This sends 6 alerts per classifier by default (about 5–6 minutes on CPU) and refuses more than 25 per classifier without `--allow-large-run`.

```bash
python scripts/qwen_e2e_from_notebook.py --skip-notebook --data-dir "$V3_DIR"
```

If the exports are in another folder, add `--exports-dir FOLDER`. If you have the exports but not the CSVs, use `--skip-notebook --no-dataset-check`; the run still works but is reported as **NOT a verified V3 run**.

Before any inference, this checks the dataset, confirms every alert's evidence equals its validation row, and confirms both exports come from one run. The sample covers DoS, DDoS, Bots, Web Attacks, a DT/RF class disagreement and the longest flow, when those exist; `summary.json` says what was not available or did not fit.

If it stops partway, fix the problem and rerun the same command with `--resume`: good answers are kept and only the rest are redone. If a results file in that folder is left over from an older run with a different selection or export, `--resume` refuses before anything changes and names the file. Move that file out of the folder and rerun with `--resume`.

The last line is the verdict. For a run that reached the verdict, a suffix says what is known about the data:

- No suffix: the validation file matched the V3 profile and a published hash, and every alert matched its row.
- `(V3 profile matched; file hash not pinned)`: the same checks passed, but with no hash to compare. Add `--expected-validation-sha256` once Kevin publishes one.
- `(NOT a verified V3 run)`: another dataset, or no dataset file, or no alerts to compare.

| Last line | Meaning | Exit |
|---|---|---|
| `E2E PASS` | Complete, verified, something explained, nothing rejected or failed. Abstentions are counted separately | 0 |
| `E2E NO_EXPLANATIONS` | Works, but at least one classifier only abstained | 1 |
| `E2E REJECTED_ANSWERS` | Unresolved rejected answers | 1 |
| `E2E FAILED` | A technical error, a failed verification, a notebook that failed, or a results file that could not be written | 1, 3 for Ollama and model problems, or 130 if interrupted |
| `E2E INCOMPLETE` | Selected alerts were not processed, or the run stopped before its correlation and worksheet. See `summary.json` | 3 if the model was lost, or 130 if interrupted |
| `E2E SKIPPED_NO_ALERTS` | An export has no alerts. Nothing was sent for it | 1 |
| `E2E BLOCKED: missing input` | A dataset or export file is missing. Nothing was run, and no other dataset was substituted | 2 |
| `E2E NOT RUN: interrupted before anything changed` | Ctrl-C before the run replaced anything. Earlier files still belong together | 130 |
| `E2E NOT RUN: invalid input` | A bad argument, a refused file, a failed dataset check, or files in `--out-dir` this tool did not write. Nothing was sent to the model | 2 |

Results are in `outputs/qwen_e2e/`: `*_explanations.json`, `correlation.json`, `summary.json` and a review worksheet.

- **Re-runs.** A run refused or blocked before the notebook or inference changes nothing, so the earlier files still belong together; check `generated_utc` in `summary.json`. A run that starts (in notebook mode, when the notebook starts) replaces `summary.json` at once, so an earlier verdict cannot survive it. In notebook mode, a refusal after the notebook has finished still leaves the new exports and the notebook's own files, because that run did happen. If a run stops early, `summary.json` says INCOMPLETE or FAILED and `classifiers.<name>.results_file` names only the results from this run.
- **Worksheets are never overwritten.** If a worksheet or rubric file is already there, a re-run writes `review_worksheet_<timestamp>.csv` beside it, and `summary.json` → `review_worksheet` names the current one.

### 6d. Human review

Open the worksheet and score the rows using `docs/qwen_review_rubric.md`. The automated checks do not judge whether an explanation is sensible; your scores do.

### No V3 data yet?

Everything except the V3 identity check works on synthetic data. See `docs/qwen_integration_walkthrough.md`.

## 7. Report your results

Reply to Paul, or comment on the pull request, with:

```
Name / machine:            (e.g. Kevin, Ubuntu 24.04 x86-64, 16 GB)
Python version:            (python --version)
Ollama version + digest:   (from check-ollama)
Step 3 offline:            smoke ALL PASS? pytest: __ passed / __ failed / __ skipped; unittest 12 OK?
Step 5 live:               smoke ALL PASS? live pytest: __ passed / __ failed
Step 6a dataset:           check-dataset passed? SHA-256 of the two files
Step 6c real V3 run:       last line; per classifier explained/abstained/rejected/failed/incomplete/skipped;
                           v3_profile_matched and real_v3_verified in summary.json
Human review:              rows scored per classifier; average of EACH column; worst example (alert id + why)
Problems:                  exact command, full error output (the JSON error has a code)
```

Attach `outputs/qwen_e2e/summary.json` (it includes your local file paths), `outputs/export_manifest.json` if you made one, and your scored worksheet. **Do not attach** datasets, the test split, the `*_alerts.json` exports, `correlation.json` or model files.

Full reference: `docs/qwen_integration.md`.
