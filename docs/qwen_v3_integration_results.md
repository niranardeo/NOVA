# Qwen V3 Integration Results

**The real V3 classifier-to-Qwen integration works on a bounded sample.** Twenty real alerts from Jenn's V3 validation exports went through the production consumer on the pinned model. Every identifier, predicted class and evidence value came back unchanged, and every final answer passed the automated checks. That says the pipeline is correct. It does not yet say the explanations are good: the human review has not been done, and a claim-by-claim check found recurring problems (section 8).

| Question | Answer |
|---|---|
| Real V3 alerts processed? | **Yes:** 20 of 126,498 (10 per classifier), chosen deterministically |
| Pipeline verdict | `E2E PASS (NOT a verified V3 run)`. The suffix means no V3 CSV was read, so rows were not compared with the dataset |
| Explained / abstained / rejected / failed | **8 / 12 / 0 / 0** |
| Identifiers, classes and evidence preserved | **Yes, all 20**, checked independently against the source files |
| Explanation quality | **Not evaluated by people yet.** Provisional findings in section 8 |
| Reserved test split | Not used, opened, hashed or listed |

## 1. Inputs

| File | SHA-256 | Alerts |
|---|---|---|
| `NOVA_V3_Qwen_Integration_Handoff.zip` | `b62c2765cac8b308c9c9e6b0af10fc04d7e27d2a24bca61032a2afce07d91f87` | |
| `decision_tree_v3_validation_alerts.json` | `207fd4be41de0637e3c79702b6b2dcf9154cb3b54bf2f2830e8f229f6197bfcd` | 63,453 |
| `random_forest_v3_validation_alerts.json` | `0c3146909ca585eb62d156035668308cc71cea73b62b73b2a99954081dc05d31` | 63,045 |

Both files: schema `1.1`, run `v3-validation-sample10-20261104`, dataset `CICIDS2017-V3-validation`, preliminary validation predictions from classifiers trained on a 10% sample (176,343 rows) of the V3 training set. The hashes were the same before and after every run, and the files are not in the repository. The dataset value is kept as Jenn wrote it.

## 2. Code and model

| Item | Value |
|---|---|
| Branch | `paul/qwen-integration-schema11` at `0fa9896`, plus the fixes in section 9 (uncommitted) |
| Component | `nova_qwen` 0.2.0, prompt `nova-qwen-prompt-1.0` |
| Model | `qwen3.5:4b-q4_K_M`, digest `d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04` (verified), Q4_K_M, 4.2B |
| Runtime | Ollama 0.40.11 in Docker on a Mac, CPU only |
| Settings | temperature 0, seed 42, thinking off, context 8,192, at most 512 output tokens, at most 2 attempts per alert |

## 3. Export validation (before any model call)

All of these passed, using both the production `validate-export` and `correlate` commands and a separate checker that shares no code with the consumer:

- valid JSON, schema 1.1, the expected detector in each file, and identical `run_id`, `dataset` and `schema_version` in both;
- every alert has the canonical `dt-`/`rf-` and `val-` identifiers, with matching numbers, and no duplicates;
- positions are in 1–377,880 and strictly ascending; the last alert is at row 377,870;
- no `Normal Traffic` predictions, and only the six attack classes;
- every alert has exactly the three approved evidence features, all finite numbers; Flow Duration is a whole number from 1 to 119,816,717;
- an empty export with the same metadata produced zero results and never contacted the model.

| Cross-classifier comparison | Records |
|---|---:|
| Flagged by both | 62,858 |
| … same predicted class | 62,849 |
| … different predicted class | 9 |
| Decision Tree only | 595 |
| Random Forest only | 187 |
| Flagged by either | 63,640 |
| Flagged by both, but evidence differs | 0 |

This reproduces the counts in the independent preflight review exactly. The 9 disagreements are Bots vs Port Scanning (4), DoS vs Port Scanning (4) and DoS vs Brute Force (1).

**Limitation.** Without `cicids2017_V3_validation_15.csv` I could not check that each alert's evidence equals its row in the dataset. Jenn's README says this check passed on her side. Structural validity is proven here; dataset alignment is not.

## 4. Tests

| Category | Result |
|---|---|
| Jenn's adapter tests (`unittest`) | **PASS** 12 of 12 |
| Offline suite, macOS, Python 3.13.7 | **PASS** 977 passed (965 Qwen tests and Jenn's 12), 0 failed, 11 skipped (the live tests), 12 subtests |
| Offline smoke | **PASS** 7 of 7 |
| Model and digest check | **PASS** |
| Live suite against Ollama | **PASS** 11 of 11, 0 skipped |
| Live smoke | **PASS** 12 of 12 |
| Export validation (CLI and independent checker) | **PASS** both files |
| Cross-classifier correlation (CLI and independent checker) | **PASS**, counts agree |
| Real V3 bounded inference | **PASS** 20 of 20 processed, 0 rejected, 0 failed |
| Independent result verification (`verify`) | **PASS** 10 of 10 per classifier |
| Independent end-to-end check against the source files | **PASS** 20 of 20, 0 problems |
| Worksheet generation (script and CLI) | **PASS** 20 rows |
| Ubuntu 24.04 container, non-root, Python 3.12.3 | **PASS** 12 of 12 (Jenn's), 977 passed and 11 skipped, smoke 7 of 7 |
| Alignment against the V3 validation CSV | **BLOCKED**: CSV not available |
| Docker image `nova-qwen:0.2` rebuilt with the fixes | **NOT TESTED** |
| Windows, GPU, an Ubuntu VM, Jenn's VM | **NOT TESTED** |

All of these ran on the final code; its hash was the same before and after. Before the fixes, the pushed commit gave 939 passed, 11 skipped, and Jenn's 12 passed.

## 5. Sample

Ten alerts per classifier: the representative selection (`--limit 8`) plus two more disagreement records requested by ID. Nothing is random, so the same command picks the same alerts.

| Record | Flow Duration (s) | Fwd packets | Bytes/s | Decision Tree | Random Forest | Why chosen |
|---|---:|---:|---:|---|---|---|
| `val-000001` | 0.145596 | 3 | 79,899.17 | DDoS: explained | DDoS: abstained | class |
| `val-000003` | 85.492058 | 10 | 144.05 | DoS: abstained | DoS: abstained | class |
| `val-000011` | 0.000015 | 1 | 400,000 | Port Scanning: explained | Port Scanning: abstained | class |
| `val-000017` | 99.179788 | 7 | 120.62 | DoS: abstained | — | fill |
| `val-000330` | 5.446992 | 3 | 0 | Web Attacks: abstained | Web Attacks: abstained | class |
| `val-000440` | 10.674295 | 5 | 384.10 | — | DoS: abstained | Random Forest only |
| `val-000862` | 0.071796 | 4 | 4,791.35 | Bots: explained | Bots: explained | class |
| `val-015887` | 0.000267 | 2 | 52,434.46 | DoS: abstained | Brute Force: abstained | disagreement; class (Brute Force, Random Forest) |
| `val-023930` | 109.698999 | 2 | 0.13 | Bots: explained | Port Scanning: abstained | disagreement |
| `val-036230` | 0.00096 | 2 | 0 | DoS: explained | Port Scanning: explained | disagreement |
| `val-166456` | 119.816717 | 16 | 28.85 | Brute Force: explained | — | longest flow; Decision Tree only; class |
| `val-223443` | 119.794368 | 18 | 47.80 | — | DoS: abstained | longest Random Forest flow |

Covered: all six classes, three of the nine disagreements, one record flagged by each classifier alone, the longest flows (just under the 120-second flow timeout) and several sub-millisecond flows. The selection has no shortest-flow rule; the exports go down to 1 microsecond. There are no negative durations in these exports. **This is a test of the pipeline, not a measurement:** 20 hand-chosen alerts give no rate for anything.

## 6. Results

| | Explained | Abstained | Rejected | Failed |
|---|---:|---:|---:|---:|
| Decision Tree | 6 | 4 | 0 | 0 |
| Random Forest | 2 | 8 | 0 | 0 |
| **Total** | **8** | **12** | **0** | **0** |

- **Traceability.** For all 20, the run ID, dataset, detector, record ID, alert ID, predicted class and raw evidence equal the source file. Qwen never changed a prediction or an identifier.
- **Units.** Flow Duration was converted from microseconds to seconds in Python for all 20, exactly (for example 15 → 0.000015 s, 119,816,717 → 119.816717 s), with the raw value kept beside it. Every answer quoted durations in seconds. The microseconds convention itself is still **unconfirmed** (section 10).
- **Priority.** `null` in all 20.
- **Re-asks.** Four first answers were rejected and re-asked, and all four second answers were accepted. Three of the four first answers included a calculated rate ("approximately 80 kilobytes per second" twice, "approximately 0.47 packets per second"), and one gave an invented payload size, "approx. 400KB". For one packet at 400,000 bytes per second over 0.000015 s, the real amount is about 6 bytes.
- **Repeatability.** The same 20 alerts were run three times (pushed code, after the first fixes, final code). Every first answer was byte-identical in all three runs. The only differences are that the fixed validator re-asked `rf-000011` and accepted `rf-000330` the first time.
- **Latency (final run).** 24 model requests, 13.4–20.1 s each (median 15.8 s, mean 15.9 s). Per alert, including a re-ask: 14.7–36.7 s (median 16.0 s, mean 19.1 s). The whole run took 6 min 24 s. An earlier run of the same sample averaged 18.6 s per request, so expect some variation. At the final run's rate, explaining all 126,498 alerts one after another on this machine would take about 28 days, or about 23 if no alert needed a re-ask. Four re-asks in 20 hand-chosen alerts is not a measured re-ask rate.

## 7. Abstention

12 of 20 answers abstained (4 Decision Tree, 8 Random Forest). The abstentions are the model's own choice; the validator rejected none of the final answers. Two findings matter for the report:

- **Abstaining did not mean declining to interpret.** Every abstained answer still interpreted the evidence: for example "consistent with a slow, sustained attack pattern", or "suggests a potential false positive".
- **Abstaining depends on the classifier's name, not only the evidence.** For the same row and the same predicted class, the prompt differs only in the classifier name. On 2 of the 5 such pairs, the model explained for one and abstained for the other (`val-000001` DDoS, `val-000011` Port Scanning). The answer texts were often nearly identical.

So the abstention count should be reported, but not read as a measure of evidence quality. I did not change the prompt to reduce abstentions: a prompt change would need the team's agreement and a re-run.

## 8. Explanation quality (provisional)

**The 0–2 human scores have not been given.** What follows are objective checks against the source evidence, plus my provisional notes. The human review in the worksheet decides quality.

**What held in all 20 answers:**
- every value quoted was supplied or a correct rounding of it ("approx 0.13 bytes/s" for 0.127621948);
- no IP address, port, protocol, TCP flag or payload was asserted as present (several were listed as unavailable, which is correct);
- no answer stated an attack as confirmed;
- every answer gave specific limitations: three features only, no packet contents, no surrounding traffic, no ground truth.

**What went wrong, and was not caught by any rule:**

The counts below come from a claim-by-claim check of all 20 answers against the evidence, with a second pass that challenged the first. They are provisional: they find problems but do not score quality.

| Problem | Answers | Examples |
|---|---:|---|
| Relies on an outside baseline ("typical", "usually", or calling a value "high" or "low" with nothing to compare it to) | 18 | "atypical for most DoS attacks, which usually involve high volume" (`dt-000003`) |
| Says why the classifier predicted the class, though no model internals were supplied | 20 | Every answer opens "The classifier predicted '<class>' based on …"; also "the model associates this specific combination with DDoS" (`dt-000001`) |
| Names a technique or pattern that is not in the evidence | 9 | "slowloris or targeted probing" (`dt-000017`); "beaconing" (`dt-023930`); "reconnaissance" (`rf-000003`) |
| Describes a byte rate as an amount | 5 | Clear in 2: "significant volume transferred in that instant" (`rf-015887`) and "rapid, high-volume bursts" (`dt-015887`), for about 14 bytes. Borderline in 3: "a large amount of data" (`dt-000001`) and "a single large transfer" (`rf-000001`), both for about 11.6 KB, and "byte volume" for a rate (`rf-000440`) |
| Contradicts itself | 4 | "many short connections" for one 120-second flow (`dt-166456`); arguing for and against a DoS fit in the same answer (`dt-000003`) |
| Leans past the evidence | 8 | "a slow, automated probe attempting to map services rather than legitimate communication" (`rf-023930`) |

Several answers also call Total Fwd Packets the flow's packet count, although it counts forward packets only. Most of these are mild, and every answer still ends by saying the prediction is unverified. Even so, a reader could take the general-knowledge statements as findings about this flow. The reviewers should look for that first.

**Bots and Web Attacks** (the classes with the weakest recall in Jenn's results): none of the three Bots answers or the two Web Attacks answers mentioned URLs, payloads or command-and-control traffic as observed. The Bots answers did use general patterns, though ("beaconing", "rapid scanning or automated probing"), and one Web Attacks answer named "slowloris or connection floods". Nothing Qwen writes about these rows changes the classifiers' recall.

## 9. Defects found by the real run, and fixes

Each fix has a regression test that fails on the pushed code and passes now. Replaying all 68 distinct real answers saved during testing (25 from the two real V3 runs) through the fixed validator changes exactly two verdicts: the first two below.

| Defect | Cause | Fix |
|---|---|---|
| "approx. 400KB" accepted (`rf-000011`) | A number glued to a unit, such as `400KB` or `0.47pps`, was never read as a number | `src/nova_qwen/response.py`: such numbers are checked like any other; ordinals such as "1st" are not quantities |
| "a potential false positive or an anomaly rather than confirmed malicious activity" rejected (`rf-000330`) | A confirmation word right after "rather than" was read as a claim | `src/nova_qwen/response.py`: a confirmation word right after "rather than" is read as denied; "rather than" excuses nothing else in the clause |
| `summary.json` recommended `CICIDS2017-V3` although the exports say `CICIDS2017-V3-validation` | The note predated Jenn's label | `scripts/qwen_e2e_from_notebook.py`: a label that names V3 is accepted as it is; only a label that does not name V3 gets a note |

A review of these fixes found four more gaps of the same kind, fixed the same way and tested: a number glued to a unit right after a feature name ("Flow Bytes/s is 1B/s") is now compared with that feature's value; rank statistics such as "99th percentile" are rejected as invented; `CICIDS2017V3` counts as naming V3, and `CICIDS2017V2` is refused for a V3 run; and "μsecs" is read as microseconds. None of them occurred in the real answers.

The export bridge's examples now use Jenn's label as well. No defect was found in Jenn's adapter or exports.

## 10. Classifier disagreements

Qwen handles each alert on its own and never sees the other classifier's prediction. So it cannot compare them, and it did not try to name a winner. For the three disagreement records processed:

| Record | Evidence | Decision Tree | Random Forest |
|---|---|---|---|
| `val-015887` | 0.000267 s, 2 packets, 52,434 bytes/s | DoS: abstained. "consistent with a DoS prediction only if the model associates rapid, high-volume bursts with denial-of-service patterns" | Brute Force: abstained. "atypical for sustained brute-force attempts", and "significant volume transferred in that instant" (wrong: about 14 bytes) |
| `val-023930` | 109.7 s, 2 packets, 0.13 bytes/s | Bots: explained. "bot-like behavior, which often involves slow, low-volume scanning or beaconing" | Port Scanning: abstained. "a slow, automated probe attempting to map services rather than legitimate communication" |
| `val-036230` | 0.00096 s, 2 packets, 0 bytes/s | DoS: explained. "possibly indicating a probe or initial phase of an attack" | Port Scanning: explained. "rapid connection attempts without successful data exchange" |

- **Both predictions stayed separate.** Each result keeps its own alert ID and predicted class, and the worksheet groups the pair by record so a reviewer can compare them.
- **On two of the three, each answer made its own label sound plausible.** On `val-023930`, identical evidence is "bot-like" beaconing for one answer and a "slow, automated probe" for the other. On `val-036230`, the same zero-byte, 2-packet flow is "a probe or initial phase of an attack" for one label and "rapid connection attempts without successful data exchange" for the other. The exception is `val-015887`, where both answers abstained and doubted their label. That is the main limitation of this design: the explanation follows the label it is given.
- **Did the answers acknowledge the limits?** Yes. All six said the evidence cannot establish an attack, and listed what is missing.
- **Could this help an analyst?** Side by side, the pair shows that the same three values fit two stories. That is a useful warning, but it does not resolve anything.
- **What would resolve it:** the true label of the record (available to Jenn in the validation CSV, never to Qwen), plus evidence these three features do not include: destination ports contacted, TCP flags, backward packets and timing between packets. Those would need a separate, approved feature change.

## 11. What these results do and do not show

| Question | Status |
|---|---|
| Classifier performance | Jenn's preliminary validation results: DT accuracy 99.77%, RF 99.73%. Not reproduced here; Qwen changes no prediction |
| Pipeline correctness | **Shown** on 20 real alerts |
| Explanation correctness, grounding, usefulness | **Not evaluated by people.** Provisional findings in section 8 |
| Prioritization | Not measured: `priority` stays `null` |
| Latency | Section 6 |
| Effect on false positives or false negatives | **Not shown, and not claimed.** Qwen explains alerts already raised, so it cannot recover a missed attack. It explains true and false positives alike, and sees no labels |

No result here comes from the reserved test split, and no final performance is claimed.

## 12. Known limitations

- **No dataset alignment check** (section 3), so the run is labelled "NOT a verified V3 run".
- **Units are a provisional assumption.** The exports are consistent with microseconds (the longest flow is just under the 120-second timeout), but Kevin has not confirmed the V3 preprocessing.
- **Small, hand-chosen sample.** It cannot give rates for explanation quality, abstention or errors.
- **Pattern-based checks.** They catch invented numbers and asserted protocols, not wrong reasoning in words.
- **CPU only.** About 16–19 s per request, so only bounded samples are practical.
- **Results files name the input link.** The run reads the handoff files through links named `decision_tree_alerts.json` and `random_forest_alerts.json` (see the testing guide). The recorded SHA-256 identifies Jenn's files.

## 13. Remaining work

1. Jenn: the alignment check output (or the V3 CSV), and the true labels of the 9 disagreement records.
2. Kevin: SHA-256 of the two approved files, and confirmation of the units.
3. Team: human review of the 20 rows with `docs/qwen_review_rubric.md`.
4. Team: decide whether the report should interpret abstention at all (section 7).
5. Me: rerun this exact sample after any prompt or validator change, so results stay comparable. To reproduce it, see the testing guide, step "6c (handoff)".
