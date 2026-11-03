# Qwen V3 Integration: Progress Summary

For Nira, as input to the team report. Full detail is in `docs/qwen_v3_integration_results.md`.

**Status: the Qwen component is updated for V3 and tested, but it has not yet run on real V3 alerts.** That test is blocked until the V3 exports (or saved models) and the V3 validation CSV are available.

**Done**
- The consumer defaults to the V3 file names, takes configurable paths, and never falls back to the older dataset.
- It checks the dataset's identity and confirms every alert matches its validation row before any model call.
- Alert counts are bounded: a negative or zero `--limit` is refused, and more than 25 per classifier needs explicit approval.
- Run verdicts now separate explained, abstained, rejected, failed, skipped and incomplete. A run where the model's answers were all rejected used to print PASS, and no longer does.
- Added a low-memory way to build exports from saved models or predictions with Jenn's unmodified adapter, plus a review worksheet built on Jenn's criteria, and team docs in the repository.

**Tested**
- Jenn's 12 adapter tests pass, unmodified.
- Qwen offline suite: `939 passed, 11 skipped, 12 subtests passed` on macOS, and the same suite on a clean Ubuntu 24.04 container.
- Real Ollama with `qwen3.5:4b-q4_K_M`, digest verified: live suite 11 passed and live smoke 12/12 on the final code.
- A bounded regression on the older (pre-V3) exports, 12 of 126,789 alerts, ended PASS with no rejected or failed answers. It ran before the last fixes and was not repeated, and it is not a V3 result.

**Not done**
- Real V3 end-to-end run: BLOCKED BY MISSING INPUT.
- Human review of explanation quality: not started, so there is no quality score to report yet.
- Windows, a native GPU and an Ubuntu VM: not tested.

**What can be said in the report**
- Classifier performance is Jenn's work, reported separately and still preliminary.
- Qwen explains alerts the classifiers raised. It does not change predictions, and it cannot recover a false negative it never receives.
- Good explanations are not evidence that false positives or false negatives went down, so we should not claim that.
- `priority` stays `null` until the team approves a policy.
- Nothing has been evaluated on the reserved test split.

**Needed from the team** (`docs/qwen_v3_team_requests.md`)
- Jenn: V3 exports or saved models, and the code that produced her reported results.
- Kevin: SHA-256 of the two approved files, and confirmation of the feature units.
- Team: a decision on the `dataset` value in exports (I recommend `CICIDS2017-V3`).
