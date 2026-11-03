# Qwen V3 Integration Results

**Real V3 end-to-end verification: BLOCKED BY MISSING INPUT.** No real V3 alert has been processed. Everything below is either code, synthetic data, or the older pre-V3 exports, and each is labelled as such.

| Question | Answer |
|---|---|
| Is the code ready for V3? | Yes, as far as it can be shown without V3 files |
| Dataset version used in tests | Synthetic data with V3's columns, and `V2-LEGACY-REGRESSION` (older exports). Never real V3 |
| Were real V3 alerts processed? | **No** |
| Was the reserved test split used? | **No.** It was not read, loaded, hashed or inspected |
| Is explanation quality evaluated? | **No.** Human review has not been done |

## 1. What is missing for the real V3 test

| File | Owner | Why |
|---|---|---|
| `decision_tree_alerts.json`, `random_forest_alerts.json` from one run on the complete validation file (or saved models or predictions to build them) | Jenn | Not in the repository: `outputs/` is ignored by Git, and her report keeps models local |
| `cicids2017_V3_validation_15.csv` | Team share | Needed to check identity and row alignment. Not downloaded for this work |
| `cicids2017_V3_train_70.csv` | Team share | Optional for the run, needed for full file identity |
| Published SHA-256 of the two files | Kevin | Without it, a run can only be a profile match, not verified |

Once the exports and the validation CSV are available, this runs the bounded test:

```bash
python scripts/qwen_e2e_from_notebook.py --skip-notebook --data-dir "$V3_DIR"
```

## 2. Source revisions

| Item | Value |
|---|---|
| `paul/qwen-integration-schema11` | `6bffaef271314e898476b4a82a38d6b873f1a393` |
| `feature/data-exploration` | `7f069d34970b3a057f4e48f3978880382bb1f5bc` |
| Common ancestor | `5cbbabc9ec29adf8fb2f6f1b507c9cbcce250118` |
| Jenn's ZIP | `10f7a025…` Same contents as `7f069d3` by `diff -r`. A ZIP carries no Git history, so Git is the authority |
| Jenn's report PDF | `bf07ac33…` |

The update was built and tested on a merge of those two branches. They change no file in common.

## 3. Jenn's reported results (supplied, not reproduced)

These are Jenn's preliminary results from her report. I have not reproduced them and do not replace them.

| | Decision Tree | Random Forest |
|---|---|---|
| Accuracy | 99.77% | 99.73% |
| Macro F1 | 0.92 | 0.91 |
| Bots recall / precision | 0.48 / 0.65 | 0.39 / 0.85 |
| Web Attacks recall | 0.82 | 0.73 |

Trained on 176,343 of 1,763,436 rows and evaluated on all 377,880 validation rows. The report's numbers agree with each other: macro F1 recomputed from the per-class values gives 0.919 and 0.907.

## 4. Notebook findings (for Jenn to confirm)

I did not edit the notebook or question the results. These are the differences between the committed notebook and the report.

**A. Stale outputs.** Commit `06eb726` changed only path strings in cells 2, 17 and 21. The saved outputs still show the old rows (1,764,525 / 378,113 / 176,452), so the committed outputs are not the V3 run. The source now names V3 files under `/media/sf_NOVA_Data`.

**B. Sampling.** The report describes class-stratified reservoir sampling. The notebook uses `train_test_split(train_size=0.10, stratify=...)` and has no reservoir, `chunksize` or `joblib` code. On the class counts printed by notebook 04, a real `train_test_split` gives Normal 146,593 and Brute Force 641; the report has 146,594 and 640, which equals rounding 10% per class. That rests on notebook 04's counts, and one row of difference would also explain it, so `check-dataset` on the real file settles it.

**C. `n_jobs`.** The report says 1 for Random Forest. The notebook says 2 in cell 30, and 1 appears nowhere. On synthetic numbers, both settings gave identical predictions, so this may not change results, but only Jenn can confirm what ran.

**D. Clean run.** Cell 12 fits a Decision Tree on the full training file, then cells 19 and 24 refit on the sample. Cells 11–15, 28, 33 and 34 never ran in the saved copy, and the execution counter resets at cell 21. The unmodified notebook does run top to bottom in Docker on labelled synthetic data (40,000 / 8,000 rows; 1,642 and 1,645 alerts), so its code works. This cannot show it reproduces the report: the exact V3 code is still needed from Jenn.

**E. Dataset identity.** Cell 34 hardcodes `dataset="CICIDS2017"`, so V2 and V3 exports look the same. I recommend `CICIDS2017-V3`, pending a team decision. The schema only requires a non-empty string, so the consumer accepts either, and my run reports record the V3 identity separately from the files. Nobody's exports are rewritten.

## 5. What changed

Only Qwen-owned files changed. `src/alert_adapter.py`, `tests/test_alert_adapter.py`, the notebooks, `docs/alert_schema.md`, `docs/baseline_experiment_results.md` and `docs/qwen_evaluation_criteria.md` are byte-identical to `origin/feature/data-exploration`.

| Defect, reproduced before the fix | Now |
|---|---|
| `--limit -1` sent 8 of 10 alerts, exit 0, `E2E PASS` | Refused before any request |
| `--limit 100000` accepted | More than 25 per classifier needs `--allow-large-run` |
| Every answer rejected printed `E2E PASS`, exit 0 | `E2E REJECTED_ANSWERS`, exit 1 |
| Unknown alert prefix silently ignored | Refused |
| Old file names demanded, with V3 files present | V3 is the default; no fallback; the paths and identity are recorded |
| Stale notebook record credited to other exports | Counted only if its export hashes match |

Added: dataset and alignment checks (`check-dataset`, `check-alignment`), representative bounded selection, the verdict logic, `scripts/qwen_export_v3_alerts.py` (saved models or predictions to exports through the unmodified adapter), `scripts/qwen_make_synthetic_demo.py`, the worksheet columns, and the team docs.

## 6. Test results

Final code, frozen before these runs, except where the result says otherwise.

| Group | Test | Environment | Result |
|---|---|---|---|
| A | Jenn's 12 adapter tests, unmodified | macOS, Python 3.13 | **PASS** `Ran 12 tests`, `OK` |
| A | Do those tests reach the branches they name? | Mutated copies of the adapter | **PASS with 2 gaps**: 7 of 9 removed branches caught. "No approved evidence features available" and the DataFrame `TypeError` are not |
| B | Whole offline suite, in a fresh virtual environment | macOS, Python 3.13 | **PASS** `939 passed, 11 skipped, 12 subtests passed` |
| B | Offline smoke | macOS | **PASS** `ALL PASS: 7/7` |
| B | Mutation check of the Qwen guards | macOS | **PASS after closing survivors.** 176 guards removed one at a time, 173 caught, on the code just before the last small fixes. New tests close the 3 survivors. A re-check of those 3 and 4 new guards: all caught. The full 180 were not repeated on the frozen code |
| C | Live suite against real Ollama | `qwen3.5:4b-q4_K_M`, digest verified | **PASS** `11 passed` in 6:01, including the production script on the synthetic demo |
| C | Live smoke | Same | **PASS** `ALL PASS: 12/12` (explained 3, abstained 1, rejected 0, failed 0) |
| C | Regression on older exports, 6 alerts per classifier | Same; **not V3** | **PASS on earlier code, NOT RE-RUN on the final code.** See section 7 |
| C | Consumer on exports written by Jenn's notebook on **synthetic** data | Same; **not V3** | **NOT TESTED** |
| D | Real V3, bounded | | **BLOCKED BY MISSING INPUT** |
| E | Ubuntu 24.04 container, clean copy, non-root | Docker on macOS (arm64) | **PASS** `939 passed, 11 skipped, 12 subtests passed`, `ALL PASS: 7/7`, Python 3.12 |
| E | Run as root | Same | `928 passed, 22 skipped, 12 subtests passed` (the 11 extra skips are file-permission tests) |
| E | Docker image, default offline command | `nova-qwen:0.2`, Python 3.12 | **PASS** `927 passed` (`tests/qwen` only). Build and `docker compose config` also pass |
| E | Live smoke inside the image | Same | **NOT TESTED** on the final image |
| E | Walkthrough, every command run literally | Clean copy | **NOT TESTED** as a literal run. Its real-model step is covered by the live suite's synthetic demo test |
| | Windows, native GPU, Ubuntu VM | | **NOT TESTED** |
| | Jenn's real VM (low-memory export route) | | **NOT TESTED** |

Skips: the 11 skipped tests are the opt-in live tests, which ran separately above.

Three rounds of adversarial review were run against this code. All findings from the first two rounds were fixed. The third was stopped early: its three medium findings and one test that could not fail are fixed; the remaining low-severity items are in section 11. Its refutation step did not finish, so those findings were not independently confirmed.

## 7. Regression on older exports (`V2-LEGACY-REGRESSION`, not V3)

This shows the pipeline works on real classifier output at real size. **It says nothing about V3.** These files predate V3 and have a different class mix.

This run used the code as it stood before the last two review rounds, and was not repeated on the final code. Those rounds did not change prompts, validation rules or model settings. Only 12 of 126,789 alerts were sent.

| | Explained | Abstained | Rejected | Failed |
|---|---|---|---|---|
| Decision Tree (63,465 alerts in export, 6 sent) | 4 | 2 | 0 | 0 |
| Random Forest (63,324 alerts in export, 6 sent) | 1 | 5 | 0 | 0 |

- **Verdict:** `E2E PASS (NOT a verified V3 run)`. Every alert's evidence matched its row in the older validation CSV.
- **Correlation:** both, same class 63,032; both, different class 22; Decision Tree only 411; Random Forest only 270.
- **Bots:** `dt-006266` explained, `rf-006266` abstained. **Web Attacks:** `dt-002126` and `rf-002126` both abstained. One or two alerts per class is an observation, not a rate.
- **Latency:** 16–118 s per alert on CPU, while another heavy job was competing for the machine, so treat this as an upper range. The usual figure unloaded is about 25 s.

## 8. Evidence units

The conversion is unchanged: Flow Duration is treated as microseconds and shown in seconds, Total Fwd Packets as a count, Flow Bytes/s as bytes per second. Raw values are preserved next to the converted ones. **The V3 unit assumption requires confirmation from Kevin.** Notebooks 03 and 04 only remove rows, but no notebook shows how `WHOLE_SET` became `_V2`. `check-dataset` flags implausible values but cannot prove a unit.

## 9. Traceability

Every result keeps `alert_id`, `record_id`, `detector`, `run_id`, predicted class and raw evidence exactly as exported, plus the source export hash. `verify` recomputes them from the export. `check-alignment` confirms each alert's evidence equals the CSV at its `record_id`. Rows without an alert cannot be compared, so a pass does not prove the frame was complete. For V3, an alert-free tail too long to be chance is also a failure.

## 10. What these results do and do not show

| Question | Status |
|---|---|
| Classifier performance | Jenn's preliminary results. Qwen does not change any prediction |
| Explanation correctness and usefulness | **Not evaluated.** Needs the human review in `docs/qwen_review_rubric.md` |
| Evidence grounding | Automated checks catch common errors. They cannot judge whether an explanation is sensible |
| Prioritization | **Not measured.** `priority` stays `null`, because no policy is approved |
| Abstention | Counted separately. Rates on old exports say nothing about V3 |
| Latency | Section 7. CPU only, small samples |

- **Qwen cannot recover a false negative.** It receives only alerts the classifiers raised. A missed attack is classified as normal and never reaches it.
- **Good explanations are not evidence of fewer false positives or false negatives.** Qwen explains true and false positives alike, filters nothing, and sees no labels.
- **Bots and Web Attacks** have the weakest recall in Jenn's report, so the worksheet flags those rows for reviewers: three flow features cannot show payloads, URLs or command-and-control behaviour. Whatever Qwen says about these rows does not improve that recall.
- **No final test performance is claimed.** The reserved test split stays closed until the configuration is frozen.

## 11. Known limitations

- **General-knowledge statements pass the checks.** Real answers said things like "contradicts typical web attack patterns involving active data exfiltration or scanning". These are not numbers or invented fields, so the rules let them through. A reviewer has to catch them.
- **Abstention can differ by classifier for the same flow**, including when both predict the same class. Treat it as model behaviour, not a finding about the classifiers.
- **Low-severity items left open from the last review:**
  - A results file edited by hand can still cause a traceback in `explain --resume` (malformed `alert_id`, `attempts` or `selection` fields). The end-to-end script reports `E2E FAILED: unexpected error` instead.
  - A lone surrogate arriving as a real character, rather than as an escape, ends the run with an error instead of being rejected as one answer.
  - Closing the terminal (SIGHUP) during an export can leave a hidden `.export-staging-*` folder, as a hard kill can. Nothing reads it.
  - A test-like `--out` is refused with a message about reading a test split, and `check-dataset` and `check-alignment` only refuse it after printing their report.
  - A notebook-mode stop after the notebook has started can show a last line such as `E2E NOT RUN` next to a `summary.json` saying INCOMPLETE.
  - `--notebook-only` drops `--limit`, `--select` and `--alert-id` from the command it prints.
  - A few of the newest guards have no test that fails if they are removed (for example, a new test-like `--out` on the report commands).

## 12. Remaining work

1. Jenn: V3 exports or saved models, and the code behind her reported results.
2. Kevin: SHA-256 values and unit confirmation.
3. Team: decide the `dataset` value.
4. Me: run the bounded V3 test, and add the real V3 results here.
5. Team: human review with the worksheet.
