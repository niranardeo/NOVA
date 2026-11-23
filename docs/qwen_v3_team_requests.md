# Open Requests for the V3 Qwen Integration

**The integration is finished.** Nothing more is needed from anyone to complete it. The findings are in `docs/qwen_v3_integration_results.md`.

## Done

- **Jenn's V3 alert exports**, received on 4 November. They pass every structural check, and every alert matches its row in the V3 validation file.
- **Jenn's reported results**, reproduced exactly from her exports and the validation labels.
- **Labels of the 9 disagreement records**, read from the validation file: Random Forest was right on 5, the Decision Tree on none, and both were wrong on 4.
- **The V3 files**, which match Kevin's profile. The team decided not to pin hashes; the SHA-256 values are recorded in the results.
- **The dataset label.** The exports use `CICIDS2017-V3-validation`, and the consumer accepts it as it is.

## For the report (team, Nira to coordinate)

1. **Human review.** Score the 20 rows in the review worksheet with `docs/qwen_review_rubric.md`, ideally two reviewers. Until then there is no quality score to report.
2. **Abstention.** On the same flow and class, the model abstained for one classifier and explained for the other. Should the report interpret `abstain` at all? I suggest reporting it but not interpreting it.

## Optional

- **Kevin: units.** Do `WHOLE_SET`, `V2` and `V3` keep CICFlowMeter's units unchanged (Flow Duration in microseconds, Total Fwd Packets as a count, Flow Bytes/s in bytes per second)? The data is consistent with that, and the report states it as an assumption otherwise.
- **The test split.** In `V3_Stratified_Split`, the held-out file is called `cicids2017_V3_initial_test_15.csv`, not `cicids2017_V3_test_RESERVED_15.csv` as in Kevin's notebook 04 (on `main`). The tools refuse it by name. Please make sure nobody opens it before the final evaluation.
- **Jenn: training code.** For reproducibility, commit the code that trained the reported models (the report describes reservoir sampling and `n_jobs=1`; the committed notebook uses `train_test_split` and `n_jobs=2`). This does not affect the integration.
- **Jenn: two adapter tests**, for the "No approved evidence features available" branch and the "features must be a pandas DataFrame" `TypeError`. All 12 tests still pass with either branch removed. Also, `docs/qwen_evaluation_criteria.md` section 7 still says seven tests and that Qwen integration has not been performed.

## Paul (me)

- Rerun the same 20-alert sample after any prompt or validator change, so results stay comparable (testing guide, step "6c (handoff)").
