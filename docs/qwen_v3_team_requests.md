# Open Requests for the V3 Qwen Integration

Only what is needed to finish the integration. Findings behind these are in `docs/qwen_v3_integration_results.md`.

## Jenn

**1. V3 alert exports (this is what blocks the real V3 test).** No V3 export or saved model is in the repository. That is expected for the models, which your report keeps local. The report lists exports as a next step, and `outputs/` is ignored by Git, so I cannot tell whether any exist yet. Please send `decision_tree_alerts.json` and `random_forest_alerts.json` from one run, made from the complete validation file in file order. If you do not have them yet, the easiest way is from your saved models on the VM (it loads your models without changing them, and it is built for low memory, though it has not been run on your VM):

```bash
python scripts/qwen_export_v3_alerts.py --validation /media/sf_NOVA_Data/cicids2017_V3_validation_15.csv --dt-model DT_MODEL_FILE --rf-model RF_MODEL_FILE --trust-model-files --dataset CICIDS2017-V3 --out-dir outputs
```

Replace `DT_MODEL_FILE` and `RF_MODEL_FILE` with your saved models. Please also send `outputs/export_manifest.json`, and check that its accuracy matches your report: Decision Tree 99.77%, Random Forest 99.73%.

**2. Which code produced the V3 results?** The committed notebook and your report describe different runs:

| Item | Report | Committed notebook |
|---|---|---|
| Sampling | Class-stratified reservoir sampling | `train_test_split(train_size=0.10, stratify=...)` |
| Sampled classes | Normal 146,594; Brute Force 640 | On the class counts printed by notebook 04 (labels only, no CSV read), that call gives Normal 146,593; Brute Force 641 |
| Random Forest `n_jobs` | 1 | 2 |
| Saved outputs | V3 rows (1,763,436 / 377,880) | Still the old rows (1,764,525 / 378,113) |

I am not questioning your results. Editing paths without re-running explains the saved-outputs row. The other three rows would remain after a re-run, because the committed notebook still uses `train_test_split` and `n_jobs=2`, and has no reservoir sampling or chunking. The class-count row holds only if your training file has notebook 04's class counts; a difference of one row would also explain it, so `check-dataset` on the real file settles that. Could you please commit or share the code you actually ran (the sampler and the chunked validation), and confirm that the report's settings are the ones used?

**3. Small things, no rush.**
- Tests for two adapter branches: "No approved evidence features available" (removing it from a copy exports an alert with empty evidence) and the "features must be a pandas DataFrame" `TypeError` (lower priority: removing it only changes the error type). All 12 tests still passed with either removed. The other seven branches I removed were all caught, so your fix to the tests works.
- `docs/qwen_evaluation_criteria.md` section 7 still says seven tests (there are now 12) and that Qwen integration has not been performed. The integration code now exists, though real V3 verification is still waiting on exports. Explanation evaluation has not been done, so please keep that part.

## Kevin

1. **SHA-256 of the approved files**, `cicids2017_V3_train_70.csv` and `cicids2017_V3_validation_15.csv`, so everyone can confirm the same bytes. (Run `shasum -a 256` on these two files only. This request does not need the reserved file, so please do not hash or open it; I will not access it either.)
2. **Units.** Do `WHOLE_SET`, `V2` and `V3` keep CICFlowMeter's units unchanged: Flow Duration in microseconds, Total Fwd Packets as a count, Flow Bytes/s in bytes per second? Notebooks 03 and 04 only remove rows, but no notebook shows how `WHOLE_SET` became `_V2`.
3. **"Initial testing" data.** The folder was described as holding initial testing data. Notebook 04 writes only `cicids2017_V3_test_RESERVED_15.csv` as its test file, but notebook 02 wrote `initial_test_dataset_CIC_15.csv`, a stratified 15% split of the old `WHOLE_SET` (378,113 rows). Is that the file meant, or is there a separate V3 one? Since it predates V3 and comes from the old split, is anyone meant to use it? I have not opened any test file.

## Team decision (Nira to coordinate)

**The `dataset` value in exports.** I recommend `CICIDS2017-V3`. The schema only requires a non-empty string, and my consumer accepts either value. Notebook cell 34 still writes `CICIDS2017`, which makes V2 and V3 exports look the same. Until this is decided, my run reports record the V3 identity separately, from the files, and I will not rewrite anyone's exports.

## Paul (me)

1. Download the two approved CSVs individually, then run `check-dataset`.
2. Run the bounded real V3 test once Jenn's exports arrive, and update `docs/qwen_v3_integration_results.md`.
