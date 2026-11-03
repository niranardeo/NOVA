# Qwen Explanation Review Rubric

This turns `docs/qwen_evaluation_criteria.md` into something a reviewer can score. Three columns use that document's
criteria as written, and prioritization is not applicable yet. Consistency is a **proposed** substitute that needs Jenn's
agreement: the document asks about identical alerts processed repeatedly, and that is already measured automatically
(see below). The scoring anchors are also **proposed**, because the criteria say the team must define examples before
formal evaluation. Please change anything the team disagrees with.

## What the worksheet is for

`review_worksheet.csv` has one row per alert sent to Qwen, showing the evidence, the answer and the automated results.
The automated checks catch common forms of wrong units, invented numbers, invented protocols and "confirmed attack"
wording. They cannot tell whether an explanation is sensible. That judgement is the point of this review.

## Scoring (0 = does not meet, 1 = partially meets, 2 = fully meets)

| Column | Question | Score 2 looks like | Score 0 looks like |
|---|---|---|---|
| `explanation_accuracy_0_2` | Does it explain the prediction correctly, without presenting it as a confirmed attack? | Describes what the values could mean for the predicted class and calls it a prediction | "This is a DoS attack", or a statement that contradicts the values |
| `evidence_grounding_0_2` | Does it use only the supplied evidence? | Every value and unit matches `evidence_as_shown_to_model`, and nothing is added | Invents ports, IPs, intent, thresholds or "typical" values, or calls a packet count a rate |
| `uncertainty_handling_0_2` | Does it acknowledge what is missing? | Says what the three features cannot show. An abstention on thin evidence can score 2 | Sounds certain and lists no real limits |
| `consistency_0_2` (proposed) | Are answers consistent for the same flow? | Rows sharing a `consistency_group` do not contradict each other on the facts | The same flow is described with conflicting facts |
| `prioritization_0_2_or_NA` | Does it assign a priority consistent with agreed criteria? | Not applicable yet | Not applicable yet |

**Consistency.** Rows with the same `consistency_group` are the same validation row in the same run, seen by both
classifiers. Their predicted classes may differ, and that alone is not an inconsistency. One classifier's answer
abstaining while the other's explains is worth a note. Leave this column blank for rows with no group. The criteria
document's own question, repeating an identical alert, is covered by the live suite: under the fixed seed the output is
byte-identical (`test_repeatability_under_fixed_settings`).

**Prioritization is `N/A`.** No priority policy is approved, so Qwen always returns `priority: null`, and I will not
invent one. Until the team approves a policy, report totals **out of 8** (out of 6 for rows with no consistency score),
not the 10 in the criteria document.

**`analyst_usefulness_0_2_not_in_total`** asks whether this would help an analyst judge the alert. It is not one of the
five criteria, so report it separately.

Fill in `reviewer` and `review_notes` too. Leave a row blank if you did not review it.

## How to read rows with other statuses

| `status` | What it means | How to score |
|---|---|---|
| `explained` | A validated answer | Score normally |
| `abstained` | A validated answer that declines to interpret the evidence | Score normally. Do not mark it down just for abstaining |
| `rejected` | The model answered, and the answer failed the automated checks | Leave scores blank and note why in `review_notes` if useful |
| `failed` | No answer, because of a technical error | Leave blank |

## Bots and Web Attacks

Jenn's preliminary V3 results show weaker recall for these classes: Bots 0.48 (DT) and 0.39 (RF), Web Attacks 0.82 (DT)
and 0.73 (RF). These rows carry a `class_review_note`. When reviewing them, please check:

- Does the answer overstate what three flow features can say about a rare class? A Web Attacks alert should not describe
  URLs or payloads, and a Bots alert should not describe command-and-control behaviour. Neither is in the evidence.
- Does it treat the prediction as unverified?
- Is the limitation specific, or boilerplate?

## What this review does and does not measure

Keep these separate when reporting:

| Question | Measured by |
|---|---|
| Classifier detection performance | Jenn's validation results. Qwen does not change any prediction |
| Explanation correctness | `explanation_accuracy_0_2` |
| Evidence grounding | `evidence_grounding_0_2`, plus the automated checks |
| Alert prioritization | Not measured: no approved policy |
| Abstention behaviour | `explained` and `abstained` counts, plus `uncertainty_handling_0_2` |
| Analyst usefulness | `analyst_usefulness_0_2_not_in_total` |
| Inference latency | `attempts[].total_duration_seconds` in the results files |

**Qwen cannot recover a classifier false negative under this design.** It receives only alerts the classifiers raised.
A missed attack is classified as normal and never reaches Qwen. Whether AI can reduce false negatives would need a
separately evaluated workflow that shows Qwen normal-classified flows, so it remains an unproven research claim.

**Good explanations also do not show that false positives went down.** Qwen explains false positives and true positives
alike. It does not filter alerts, and it sees no ground truth. Reducing false positives would need a prioritization or
triage policy, evaluated against labels.

## Reporting

Report the following per classifier, for each column separately, rather than one blended number:

- rows scored and reviewer count;
- the average of each column, and how many rows scored 0;
- the worst example (alert id and why);
- anything you would change in this rubric.

Do not attach datasets, the test split or `*_alerts.json` exports. The worksheet is fine to share: its only data values are
the three evidence features of the alerts sent to Qwen.
