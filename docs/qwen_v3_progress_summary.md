# Qwen V3 Integration: Progress Summary

For Nira, as input to the team report. Full detail is in `docs/qwen_v3_integration_results.md`.

**Status: the real V3 classifier-to-Qwen pipeline works.** Twenty real alerts from Jenn's V3 validation exports went through Qwen (8 explained, 12 abstained, 0 rejected, 0 failed), with every identifier, prediction and evidence value preserved. Explanation quality has not been judged by people yet.

## 1. Jenn's classifier results (preliminary, hers)

- Decision Tree: 99.77% accuracy, macro F1 0.92. Random Forest: 99.73%, macro F1 0.91.
- Trained on a 10% sample of V3 training data; evaluated on all 377,880 validation rows; never on the reserved test split.
- Weakest classes: Bots (recall 0.48 DT, 0.39 RF) and Web Attacks (0.82 DT, 0.73 RF).
- Her exports hold 63,453 (DT) and 63,045 (RF) alerts. The two classifiers agree on 62,849 records and disagree on the class of 9; 595 are flagged by DT only and 187 by RF only.

## 2. My Qwen implementation

- It reads the exports, converts Flow Duration from microseconds to seconds in Python, asks Qwen3.5-4B (local, through Ollama) to explain each alert, checks every answer, and writes a review worksheet.
- It never changes a prediction, and `priority` stays `null`, because no policy is approved.
- Tests: Jenn's 12 adapter tests pass; my offline suite passes 965 tests (977 including Jenn's 12) on macOS and in an Ubuntu 24.04 container; the live tests against the real model pass 11 of 11.

## 3. Real V3 integration results

- 20 alerts (10 per classifier), covering all six attack classes, three disagreements, records flagged by one classifier only, the longest flows, and sub-millisecond flows.
- 8 explained, 12 abstained, 0 rejected, 0 failed.
- All 20 traced back exactly to Jenn's files.
- The real data found one validator gap (an invented "400KB" payload size was accepted), one false rejection, and an outdated dataset-label note. All three are fixed and tested, along with four smaller gaps a review of the fixes found.
- About 16–19 s per request on CPU, so only small samples are practical.
- Not yet checked against the V3 validation CSV, because I don't have it. Jenn reports that check passed on her side.

## 4. Explanation quality (provisional, not scored)

- **Held in all 20:** no invented IP, port or protocol was asserted as present, no attack was stated as confirmed, durations were quoted correctly in seconds, and every answer listed what the evidence cannot show.
- **Recurring problems no rule catches:** statements about what attacks "typically" do, named techniques that are not in the evidence ("slowloris", "beaconing"), and rates described as amounts ("significant volume" for about 14 bytes).
- **Abstention is not reliable:** with identical evidence and class, the model abstained for one classifier and explained for the other in 2 of 5 cases, and abstained answers still interpreted the evidence.
- **Disagreements:** on two of the three disagreement records, Qwen made each classifier's label sound plausible for the same flow; on the third (`val-015887`), both answers abstained and doubted their label. It cannot tell which classifier is right, and it does not try.

## 5. What we should not claim

- That Qwen reduces false positives or false negatives. It sees only alerts already raised, sees no labels, and filters nothing.
- Any quality score, until the human review is done.
- Anything about the reserved test split.

## 6. Outstanding work

1. Human review of the 20 rows with the rubric (team).
2. Alignment check against the V3 CSV (Jenn's output, or the CSV for me to run it).
3. True labels of the 9 disagreement records (Jenn).
4. Hashes of the two approved files and confirmation of the units (Kevin).
