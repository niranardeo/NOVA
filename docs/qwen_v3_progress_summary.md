# Qwen V3 Integration: Progress Summary

For Nira, as input to the team report. Full detail is in `docs/qwen_v3_integration_results.md`.

**Status: the V3 classifier-to-Qwen integration is complete and checked against the V3 dataset** (profile matched, every alert row-aligned; file hashes not pinned by team decision, so the run reads `E2E PASS (V3 profile matched; file hash not pinned)`). Twenty real alerts from Jenn's V3 validation exports went through Qwen: 8 explained, 12 abstained, 0 rejected, 0 failed. Every identifier, prediction and evidence value was preserved, and every alert in both exports (126,498) matched its row in the V3 validation file. Explanation quality has not been judged by people yet.

## 1. Jenn's classifier results (preliminary, hers)

- Decision Tree: 99.77% accuracy, macro F1 0.92. Random Forest: 99.73%, macro F1 0.91.
- Trained on a 10% sample of V3 training data; evaluated on all 377,880 validation rows; never on the test split.
- Weakest classes: Bots (recall 0.48 DT, 0.39 RF) and Web Attacks (0.82 DT, 0.73 RF).
- **Reproduced exactly from her exports and the validation labels**, so the exports are the experiment she reported.
- The two classifiers agree on 62,849 alerted records and disagree on the class of 9. Of those 9, Random Forest was right on 5, the Decision Tree on none, and both were wrong on 4. 595 records are flagged by DT only and 187 by RF only.

## 2. My Qwen implementation

- It reads the exports, converts Flow Duration from microseconds to seconds in Python, asks Qwen3.5-4B (local, through Ollama) to explain each alert, checks every answer, and writes a review worksheet.
- It never changes a prediction, and `priority` stays `null`, because no policy is approved.
- Tests: Jenn's 12 adapter tests pass; my offline suite passes 965 tests (977 including Jenn's 12) on macOS and in an Ubuntu 24.04 container; the live tests against the real model pass 11 of 11.

## 3. Real V3 integration results

- 20 alerts (10 per classifier): all six attack classes, three disagreements, records flagged by one classifier only, the longest flows, and sub-millisecond flows.
- 8 explained, 12 abstained, 0 rejected, 0 failed, in all four runs. Every first answer was identical in all four, and the last three runs were identical throughout.
- The real data found one validator gap (an invented "400KB" payload size was accepted), one false rejection, and an outdated dataset-label note. All are fixed and tested, along with four smaller gaps found in a review of the fixes.
- About 16–19 s per request on CPU, so only small samples are practical.

## 4. Explanation quality (provisional, not scored)

- **Held in all 20:** no invented IP, port or protocol was asserted as present, no attack was stated as confirmed, durations were quoted correctly in seconds, and every answer listed what the evidence cannot show.
- **Recurring problems no rule catches:** statements about what attacks "typically" do, named techniques that are not in the evidence ("slowloris", "beaconing"), and rates described as amounts ("significant volume" for about 14 bytes).
- **Abstention is not reliable.** With identical evidence and class, the model abstained for one classifier and explained for the other in 2 of 5 cases. It abstained on 1 of the 4 wrong predictions and on 11 of the 16 correct ones.
- **Explanations follow the label, not the truth.** Qwen explained 3 of the 4 wrong predictions plausibly, while 6 of its 16 answers on correct predictions called the evidence atypical of or inconsistent with the label (for example `rf-015887`, `dt-000003`, `rf-000330`). It cannot tell which classifier is right, and it does not try.

## 5. What we should not claim

- That Qwen reduces false positives or false negatives. It sees only alerts already raised, sees no labels, filters nothing, and its caution did not follow correctness in our sample.
- Any quality score, until the human review is done.
- Anything about the test split.

## 6. Outstanding work

1. Human review of the 20 rows with the rubric (team).
2. A decision on whether to interpret abstention in the report (team).
3. Optional: Kevin confirms the feature units, or we keep them as a stated assumption.
