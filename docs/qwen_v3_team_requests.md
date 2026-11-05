# Open Requests for the V3 Qwen Integration

Only what is needed to finish the integration. The findings behind these are in `docs/qwen_v3_integration_results.md`.

## Done

- **Jenn: V3 alert exports.** Received on 4 November (`decision_tree_v3_validation_alerts.json`, `random_forest_v3_validation_alerts.json`). Thank you. A bounded run of 20 alerts passed, and the exports themselves pass every structural check.
- **Dataset label.** The exports use `CICIDS2017-V3-validation`, and the consumer now treats that as the V3 label. Nothing is rewritten.

## Jenn

1. **The V3 validation CSV, or a check on your side.** Without `cicids2017_V3_validation_15.csv` I cannot confirm that each alert's evidence equals its validation row, so my run is reported as "NOT a verified V3 run". Your README says this alignment passed on your side, which is good. If the CSV can be shared, I will run `check-alignment`. If not, please send the output of these commands, run on the VM from the repository root on my branch. The module is not on `main` or `feature/data-exploration`; see step 1 of `docs/qwen_testing_guide.md`:

```bash
git fetch origin && git switch paul/qwen-integration-schema11
```

Replace `/path/to/` with the folder that holds your exports:

```bash
python -m src.nova_qwen check-alignment /path/to/decision_tree_v3_validation_alerts.json --validation /media/sf_NOVA_Data/cicids2017_V3_validation_15.csv
```

```bash
python -m src.nova_qwen check-alignment /path/to/random_forest_v3_validation_alerts.json --validation /media/sf_NOVA_Data/cicids2017_V3_validation_15.csv
```

2. **True labels of the 9 records where the classifiers disagree** (`val-015887`, `val-023930`, `val-036230`, `val-056621`, `val-101548`, `val-211741`, `val-238173`, `val-266261`, `val-332145`). Qwen cannot settle a disagreement, and it is never shown labels. The labels would show which classifier was right on these records, which belongs in your results, not in Qwen's.

3. **Which code produced the V3 results?** The report describes reservoir sampling and `n_jobs=1`; the committed notebook uses `train_test_split` and `n_jobs=2`, and its saved outputs still show the old row counts. The README's "10% sample (176,343 records)" matches the report. Please commit or share the code you actually ran.

4. **Small things, no rush.**
   - Tests for two adapter branches: "No approved evidence features available" and the "features must be a pandas DataFrame" `TypeError`. All 12 tests still pass with either branch removed.
   - `docs/qwen_evaluation_criteria.md` section 7 still says seven tests (there are 12) and that Qwen integration has not been performed. The bounded integration has now run; the explanation evaluation has not, so please keep that part.

## Kevin

1. **SHA-256 of the approved files,** `cicids2017_V3_train_70.csv` and `cicids2017_V3_validation_15.csv`, so everyone can confirm the same bytes. Run `shasum -a 256` on these two files only. This does not need the reserved file, so please do not hash or open it.
2. **Units.** Do `WHOLE_SET`, `V2` and `V3` keep CICFlowMeter's units unchanged: Flow Duration in microseconds, Total Fwd Packets as a count, Flow Bytes/s in bytes per second? The V3 exports are consistent with that (the longest flow is 119,816,717, just under CICFlowMeter's 120-second timeout in microseconds), but consistent is not confirmed.
3. **"Initial testing" data.** Notebook 02 wrote `initial_test_dataset_CIC_15.csv`, a 15% split of the old `WHOLE_SET`. Is anyone meant to use it? I have not opened any test file.

## Team (Nira to coordinate)

1. **Human review.** Score the 20 rows in the review worksheet with `docs/qwen_review_rubric.md`, ideally two reviewers. Until then there is no quality score to report.
2. **Abstention.** On the same flow and class, the model abstained for one classifier and explained for the other. Should the report treat `abstain` as informative at all? I suggest reporting it but not interpreting it.

## Paul (me)

1. Run `check-alignment` once the CSV or Jenn's output is available, and update the results.
2. Rerun the same bounded sample if the prompt or validator changes, so results stay comparable.
