# Qwen V3 Integration Results

**The V3 classifier-to-Qwen integration is complete and checked against the V3 dataset:** both V3 files match Kevin's profile, and every alert matched its validation row. The file hashes are not pinned, so in the tools' own terms the run is a profile match, not "verified". Twenty real alerts from Jenn's V3 validation exports went through the production consumer on the pinned model.
- Every alert in both exports (126,498) matched its row in the V3 validation file.
- Every identifier, predicted class and evidence value came back unchanged.
- Every final answer passed the automated checks.

That shows the pipeline is correct. It does not show the explanations are good: the human review has not been done, and a claim-by-claim check found recurring problems (section 9).

| Question | Answer |
|---|---|
| Pipeline verdict | `E2E PASS (V3 profile matched; file hash not pinned)`. Both V3 files match Kevin's profile, and every alert matched its row. The team decided not to pin hashes; the SHA-256 values are recorded in section 1 in case that changes |
| Real V3 alerts processed | 20 of 126,498 (10 per classifier), chosen deterministically |
| Explained / abstained / rejected / failed | **8 / 12 / 0 / 0** |
| Identifiers, classes and evidence preserved | **Yes, all 20**, checked independently against the source files |
| Jenn's reported classifier results | **Reproduced exactly** from her exports and the validation labels (section 4) |
| Explanation quality | **Not evaluated by people yet.** Provisional findings in section 9 |
| Reserved test split | Not extracted, opened, hashed or read |

## 1. Inputs

| File | SHA-256 | Size |
|---|---|---|
| `decision_tree_v3_validation_alerts.json` (from `NOVA_V3_Qwen_Integration_Handoff.zip`, `b62c2765…91f87`) | `207fd4be41de0637e3c79702b6b2dcf9154cb3b54bf2f2830e8f229f6197bfcd` | 63,453 alerts |
| `random_forest_v3_validation_alerts.json` (same ZIP) | `0c3146909ca585eb62d156035668308cc71cea73b62b73b2a99954081dc05d31` | 63,045 alerts |
| `cicids2017_V3_validation_15.csv` (from `V3_Stratified_Split.zip`) | `4e9943a5c2a99970a8c05e6e6d8ab7840b2a8c3bdf911e7d5d4eff2616d6530c` | 377,880 rows |
| `cicids2017_V3_train_70.csv` (same ZIP) | `47c4260a0c0ea06b9998634eea09be4ea4beffc26d423a1723e22a0e10582d12` | 1,763,436 rows |

- **The exports:** schema `1.1`, run `v3-validation-sample10-20261104`, dataset `CICIDS2017-V3-validation`, kept as Jenn wrote it. Their hashes were unchanged after every run, and none of these files is in the repository.
- **The test split.** `V3_Stratified_Split.zip` also contains `cicids2017_V3_initial_test_15.csv`. Only the two files above were extracted, by name; that file was never extracted, opened or hashed. Its name differs from `cicids2017_V3_test_RESERVED_15.csv` in Kevin's notebook, but it is presumably the same held-out split. The tools refuse it by name. **Please treat it as the reserved test split, and do not extract it.**

## 2. Code and model

| Item | Value |
|---|---|
| Code | `paul/qwen-integration-schema11` at `684061d` (pushed). The tested tree is byte-identical to this commit |
| Component | `nova_qwen` 0.2.0, prompt `nova-qwen-prompt-1.0` |
| Model | `qwen3.5:4b-q4_K_M`, digest `d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04` (verified), Q4_K_M, 4.2B |
| Runtime | Ollama 0.40.11 in Docker on a Mac, CPU only |
| Settings | temperature 0, seed 42, thinking off, context 8,192, at most 512 output tokens, at most 2 attempts per alert |

## 3. Data checks (before any model call)

**Dataset identity (`check-dataset`): PASS.** Both files match Kevin's V3 notebooks exactly:
- rows: 1,763,436 training and 377,880 validation;
- the 53 columns;
- the class counts of every class.

The evidence values are consistent with the assumed units, which is not proof of them (section 12).

**Row alignment (`check-alignment`): PASS for both exports.** All 63,453 Decision Tree and 63,045 Random Forest alerts were compared with the validation file at their `record_id`, with 0 evidence values differing. The last alert is at row 377,870, 10 rows before the end, which is ordinary for a shuffled split. The validation file has 16 rows with a negative Flow Duration. All 16 are labelled Normal Traffic and none was exported.

**Export structure: PASS.** This used the production `validate-export` and `correlate` commands, plus a separate checker that shares no code with the consumer:
- schema 1.1, the expected detector in each file, and the same `run_id`, `dataset` and schema version in both;
- canonical, matching and unique `dt-`/`rf-` and `val-` identifiers;
- positions within 1–377,880;
- no Normal Traffic predictions;
- exactly the three approved evidence features, all finite.

An empty export with the same metadata produced zero results and never contacted the model.

| Cross-classifier comparison | Records |
|---|---:|
| Flagged by both | 62,858 |
| … same predicted class | 62,849 |
| … different predicted class | 9 |
| Decision Tree only | 595 |
| Random Forest only | 187 |
| Flagged by both, but evidence differs | 0 |

## 4. Jenn's classifier results, reproduced from her exports

A row without an alert was predicted Normal Traffic. Comparing the full predictions with the validation labels gives exactly the figures in `docs/baseline_experiment_results.md`. So the exports are the experiment she reported. The labels were read only for this check, and Qwen never sees them.

| Metric | Decision Tree, report → recomputed | Random Forest, report → recomputed |
|---|---|---|
| Accuracy | 99.77% → 99.772% | 99.73% → 99.729% |
| Macro precision / recall / F1 | 0.95 / 0.90 / 0.92 → 0.945 / 0.895 / 0.917 | 0.98 / 0.87 / 0.91 → 0.976 / 0.866 / 0.906 |
| Bots precision / recall | 0.65 / 0.48 → 0.647 / 0.476 | 0.85 / 0.39 → 0.846 / 0.394 |
| Web Attacks recall | 0.82 → 0.823 | 0.73 → 0.730 |
| Attacks missed (false negatives) / normal rows flagged (false positives) | 552 / 255 | 851 / 146 |

The open question about the training code (reservoir sampling and `n_jobs=1` in the report, against `train_test_split` and `n_jobs=2` in the committed notebook) concerns reproducing the training, not these predictions. It does not affect the integration.

## 5. Tests

| Category | Result |
|---|---|
| Jenn's adapter tests (`unittest`) | **PASS** 12 of 12 |
| Offline suite, macOS, Python 3.13.7 | **PASS** 977 passed (965 Qwen tests and Jenn's 12), 0 failed, 11 skipped (the live tests), 12 subtests |
| Offline smoke | **PASS** 7 of 7 |
| Ubuntu 24.04 container, non-root, Python 3.12.3 | **PASS** 12 of 12 (Jenn's), 977 passed and 11 skipped, smoke 7 of 7 |
| Model and digest check | **PASS** |
| Live suite against Ollama | **PASS** 11 of 11 |
| Live smoke | **PASS** 12 of 12 |
| V3 dataset check (train and validation) | **PASS** |
| Row alignment, both exports | **PASS** 126,498 of 126,498 alerts, 0 differences |
| Export validation and cross-classifier correlation | **PASS** |
| Real V3 bounded inference with the dataset checks | **PASS** 20 of 20 processed, 0 rejected, 0 failed |
| Independent result verification (`verify`) | **PASS** 10 of 10 per classifier |
| Independent end-to-end check against the source files | **PASS** 20 of 20, 0 problems |
| Worksheet generation | **PASS** 20 rows |
| Pinned file hash | **SKIPPED** by team decision |
| Docker image `nova-qwen:0.2` rebuilt; Windows, GPU, an Ubuntu VM, Jenn's VM | **NOT TESTED** |

The suites ran on a tree byte-identical to `684061d`. The real V3 run and the data checks ran on `684061d` itself.

## 6. Sample

Ten alerts per classifier: the representative selection (`--limit 8`) plus two more disagreement records requested by ID. Nothing is random, so the same command picks the same alerts. The true label is shown for reference only; Qwen never saw it.

| Record | Flow Duration (s) | Fwd packets | Bytes/s | True label | Decision Tree | Random Forest | Why chosen |
|---|---:|---:|---:|---|---|---|---|
| `val-000001` | 0.145596 | 3 | 79,899.17 | DDoS | DDoS: explained | DDoS: abstained | class |
| `val-000003` | 85.492058 | 10 | 144.05 | DoS | DoS: abstained | DoS: abstained | class |
| `val-000011` | 0.000015 | 1 | 400,000 | Port Scanning | Port Scanning: explained | Port Scanning: abstained | class |
| `val-000017` | 99.179788 | 7 | 120.62 | DoS | DoS: abstained | — | fill |
| `val-000330` | 5.446992 | 3 | 0 | Web Attacks | Web Attacks: abstained | Web Attacks: abstained | class |
| `val-000440` | 10.674295 | 5 | 384.10 | DoS | — | DoS: abstained | Random Forest only |
| `val-000862` | 0.071796 | 4 | 4,791.35 | Bots | Bots: explained | Bots: explained | class |
| `val-015887` | 0.000267 | 2 | 52,434.46 | Brute Force | **DoS (wrong)**: abstained | Brute Force: abstained | disagreement; class (Brute Force, Random Forest) |
| `val-023930` | 109.698999 | 2 | 0.13 | Port Scanning | **Bots (wrong)**: explained | Port Scanning: abstained | requested by ID (a disagreement) |
| `val-036230` | 0.00096 | 2 | 0 | Brute Force | **DoS (wrong)**: explained | **Port Scanning (wrong)**: explained | requested by ID (a disagreement) |
| `val-166456` | 119.816717 | 16 | 28.85 | Brute Force | Brute Force: explained | — | longest flow; Decision Tree only; class |
| `val-223443` | 119.794368 | 18 | 47.80 | DoS | — | DoS: abstained | longest Random Forest flow |

Covered: all six classes, three of the nine disagreements, one record flagged by each classifier alone, the longest flows (just under the 120-second flow timeout) and several sub-millisecond flows. **This is a test of the pipeline, not a measurement:** 20 hand-chosen alerts give no rate for anything.

## 7. Results

| | Explained | Abstained | Rejected | Failed |
|---|---:|---:|---:|---:|
| Decision Tree | 6 | 4 | 0 | 0 |
| Random Forest | 2 | 8 | 0 | 0 |
| **Total** | **8** | **12** | **0** | **0** |

- **Traceability.** For all 20, the run ID, dataset, detector, record ID, alert ID, predicted class and raw evidence equal the source file. Qwen never changed a prediction or an identifier.
- **Units.** Flow Duration was converted from microseconds to seconds in Python for all 20, exactly (for example 15 → 0.000015 s, 119,816,717 → 119.816717 s), with the raw value kept beside it. Every answer quoted durations in seconds.
- **Priority.** `null` in all 20.
- **Re-asks.** Four first answers were rejected and re-asked, and all four second answers were accepted.
  - Three of the four included a calculated rate: "approximately 80 kilobytes per second" twice, and "approximately 0.47 packets per second".
  - One gave an invented payload size, "approx. 400KB". At 400,000 bytes per second over 0.000015 s, the whole flow carried about 6 bytes.
- **Repeatability.** The same 20 alerts were run four times: on the pushed code, after the first fixes, on the final code, and once more with the dataset checks. Every first answer was byte-identical in all four runs. The last three runs were identical throughout.
- **Latency (verified run).** 24 model requests, 13.2–22.5 s each (median 18.0 s, mean 18.2 s). Per alert, including a re-ask: 13.2–38.7 s (median 18.4 s, mean 21.8 s). The mean per request was 15.9–18.6 s across the four runs. At this run's rate, explaining all 126,498 alerts one after another on this machine would take about 32 days, or about 27 if no alert needed a re-ask. Four re-asks in 20 hand-chosen alerts is not a measured re-ask rate.

## 8. Abstention

12 of 20 answers abstained (4 Decision Tree, 8 Random Forest). The abstentions are the model's own choice; the validator rejected none of the final answers.

- **Abstaining did not mean declining to interpret.** Every abstained answer still interpreted the evidence: for example "consistent with a slow, sustained attack pattern", or "suggests a potential false positive".
- **Abstaining depends on the classifier's name, not only the evidence.** For the same row and the same predicted class, the prompt differs only in the classifier name. On 2 of the 5 such pairs, the model explained for one and abstained for the other (`val-000001` DDoS, `val-000011` Port Scanning).
- **Abstaining did not track correctness.** The model abstained on 1 of the 4 wrong predictions and on 11 of the 16 correct ones.

So the abstention count should be reported, but not read as a measure of evidence quality. I did not change the prompt to reduce abstentions: a prompt change would need the team's agreement and a re-run.

## 9. Explanation quality (provisional)

**The 0–2 human scores have not been given.** What follows are objective checks against the source evidence and provisional notes. The human review in the worksheet decides quality.

**What held in all 20 answers:**
- every value quoted was supplied or a correct rounding of it ("approx 0.13 bytes/s" for 0.127621948);
- no IP address, port, protocol, TCP flag or payload was asserted as present (several were listed as unavailable, which is correct);
- no answer stated an attack as confirmed;
- every answer gave specific limitations: three features only, no packet contents, no surrounding traffic, no ground truth.

**What went wrong, and was not caught by any rule.** The counts come from a claim-by-claim check of all 20 answers against the evidence, with a second pass that challenged the first. They are provisional: they find problems but do not score quality.

| Problem | Answers | Examples |
|---|---:|---|
| Relies on an outside baseline ("typical", "usually", or calling a value "high" or "low" with nothing to compare it to) | 18 | "atypical for most DoS attacks, which usually involve high volume" (`dt-000003`) |
| Says why the classifier predicted the class, though no model internals were supplied | 20 | Every answer opens "The classifier predicted '<class>' based on …"; also "the model associates this specific combination with DDoS" (`dt-000001`) |
| Names a technique or pattern that is not in the evidence | 9 | "slowloris or targeted probing" (`dt-000017`); "beaconing" (`dt-023930`); "reconnaissance" (`rf-000003`) |
| Describes a byte rate as a large amount, which the arithmetic contradicts | 5 | Clear in 2: "significant volume transferred in that instant" (`rf-015887`) and "rapid, high-volume bursts" (`dt-015887`), for about 14 bytes. Borderline in 3: "a large amount of data" (`dt-000001`) and "a single large transfer" (`rf-000001`), both for about 11.6 KB, and "byte volume" for a rate (`rf-000440`) |
| Contradicts itself | 4 | "many short connections" for one 120-second flow (`dt-166456`); arguing for and against a DoS fit in the same answer (`dt-000003`) |
| Leans past the evidence | 8 | "a slow, automated probe attempting to map services rather than legitimate communication" (`rf-023930`) |

Three more answers read an amount off the rate ("minimal data transfer", "very small data transfer"); there the arithmetic happens to agree. Several answers also call Total Fwd Packets the flow's packet count, although it counts forward packets only. Most of these problems are mild, and every answer says what the evidence cannot establish, though only 13 of the 20 call the prediction unverified. Even so, a reader could take the general-knowledge statements as findings about this flow. The reviewers should look for that first.

**Against the true labels** (for this report only; Qwen never sees labels):
- 16 of the 20 predictions were correct, and all 4 wrong ones are in the disagreement records.
- Qwen gave a plausible explanation for 3 of the 4 wrong predictions.
- It argued against the label in 6 of the 16 correct predictions (`dt-000003`, `dt-000017`, `rf-000330`, `rf-000440`, `rf-223443`, `rf-015887`); for example "atypical for sustained brute-force attempts" for Random Forest's correct Brute Force on `val-015887`.
- Both answers that suggested a "potential false positive" (`dt-000003`, `rf-000330`) concern correct predictions.

In this small sample, Qwen's caution did not follow whether the classifier was right. That is one more reason not to claim that explanations reduce false positives.

**Bots and Web Attacks** are the classes with the weakest recall. None of the three Bots answers or the two Web Attacks answers mentioned URLs, payloads or command-and-control traffic as observed. The Bots answers did use general patterns ("beaconing", "rapid scanning or automated probing"), and one Web Attacks answer named "slowloris or connection floods".

## 10. Defects found by the real run, and fixes (in `684061d`)

Each fix has a regression test that fails on `0fa9896` and passes on `684061d`. Replaying all 68 distinct real answers saved during testing through the fixed validator changes exactly two verdicts: the first two below.

| Defect | Cause | Fix |
|---|---|---|
| "approx. 400KB" accepted (`rf-000011`) | A number glued to a unit, such as `400KB` or `0.47pps`, was never read as a number | Such numbers are checked like any other; ordinals such as "1st" are not quantities |
| "a potential false positive or an anomaly rather than confirmed malicious activity" rejected (`rf-000330`) | A confirmation word right after "rather than" was read as a claim | It is read as denied; "rather than" excuses nothing else in the clause |
| `summary.json` recommended `CICIDS2017-V3` although the exports say `CICIDS2017-V3-validation` | The note predated Jenn's label | A label that names V3 is accepted as it is |

A review of these fixes found four more gaps of the same kind, fixed and tested the same way:
- a number glued to a unit right after a feature name ("Flow Bytes/s is 1B/s") is now compared with that feature's value;
- rank statistics such as "99th percentile" are rejected as invented;
- `CICIDS2017V3` counts as naming V3, and `CICIDS2017V2` is refused for a V3 run;
- "μsecs" is read as microseconds.

None of these four occurred in the real answers. No defect was found in Jenn's adapter, her exports or the V3 files.

## 11. Classifier disagreements

Qwen handles each alert on its own and never sees the other classifier's prediction or the true label. So it cannot compare the two predictions, and it did not try to name a winner. For the three disagreement records processed:

| Record | Evidence | True label | Decision Tree | Random Forest |
|---|---|---|---|---|
| `val-015887` | 0.000267 s, 2 packets, 52,434 bytes/s | Brute Force | DoS (wrong): abstained. "consistent with a DoS prediction only if the model associates rapid, high-volume bursts with denial-of-service patterns" | Brute Force (right): abstained. "atypical for sustained brute-force attempts", and "significant volume transferred in that instant" (wrong: about 14 bytes) |
| `val-023930` | 109.7 s, 2 packets, 0.13 bytes/s | Port Scanning | Bots (wrong): explained. "bot-like behavior, which often involves slow, low-volume scanning or beaconing" | Port Scanning (right): abstained. "a slow, automated probe attempting to map services rather than legitimate communication" |
| `val-036230` | 0.00096 s, 2 packets, 0 bytes/s | Brute Force | DoS (wrong): explained. "possibly indicating a probe or initial phase of an attack" | Port Scanning (wrong): explained. "rapid connection attempts without successful data exchange" |

- **Both predictions stayed separate.** Each result keeps its own alert ID and predicted class, and the worksheet groups the pair by record so a reviewer can compare them.
- **The explanation follows the label it is given.**
  - `val-023930`: the same evidence is "bot-like" beaconing for one answer and a "slow, automated probe" for the other.
  - `val-036230`: both labels were explained, and both are wrong.
  - `val-015887`: both answers abstained and doubted their label, including the correct one.
- **Did the answers acknowledge the limits?** Yes. All six said the evidence cannot establish an attack, and listed what is missing.
- **Could this help an analyst?** Side by side, a pair shows that the same three values fit two stories. That is a useful warning, but it does not resolve anything.
- **All nine disagreements, by true label:**
  - Random Forest was right on 5 (`val-015887`, `val-023930`, `val-101548`, `val-211741`, `val-266261`).
  - The Decision Tree was right on none.
  - On the other 4 (`val-036230`, `val-056621`, `val-238173`, `val-332145`) both were wrong: the true class is Brute Force, and they said DoS and Port Scanning.
  - Resolving such cases needs labels or more evidence (destination ports, TCP flags, backward packets, timing between packets), not explanations.

## 12. What these results do and do not show

| Question | Status |
|---|---|
| Classifier performance | Jenn's preliminary validation results, **reproduced exactly** from her exports (section 4). Qwen changes no prediction |
| Pipeline correctness | **Shown** on 20 real alerts, with every alert in both exports aligned to the V3 validation file (profile matched; hashes not pinned) |
| Explanation correctness, grounding, usefulness | **Not evaluated by people.** Provisional findings in section 9 |
| Prioritization | Not measured: `priority` stays `null` |
| Effect on false positives or false negatives | **Not shown, and not claimed.** Qwen explains alerts already raised, so it cannot recover any of the 552 (DT) or 851 (RF) missed attacks. It explains true and false positives alike, and its caution did not follow correctness in this sample |

No result here comes from the test split, and no final performance is claimed.

**Known limitations:**
- **Units are an assumption.** Flow Duration is treated as microseconds, Total Fwd Packets as a count and Flow Bytes/s as bytes per second. The data is consistent with this (the longest flow is just under CICFlowMeter's 120-second timeout), but the V3 preprocessing has not been confirmed.
- **File hashes are not pinned**, by team decision.
- **Small, hand-chosen sample:** no rates for explanation quality, abstention or errors.
- **Pattern-based checks:** they catch invented numbers and asserted protocols, not wrong reasoning in words.
- **CPU only:** about 16–19 s per request.
- **The results files name the input links.** The run reads the handoff files through links named `decision_tree_alerts.json` and `random_forest_alerts.json`; the recorded SHA-256 identifies Jenn's files.

## 13. Remaining work

Nothing further is needed for the integration. For the report:
1. Team: human review of the 20 rows with `docs/qwen_review_rubric.md`, ideally two reviewers.
2. Team: decide whether the report interprets abstention at all (section 8).
3. Kevin, optional: confirm the units, or keep them as a stated assumption.
4. Optional: pin the SHA-256 values in section 1 if the team later wants a fully verified label.

To reproduce the verified run, see the testing guide, step "6c (handoff)".
