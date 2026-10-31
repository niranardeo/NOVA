"""Response validation. Each negative case changes ONE thing in an otherwise valid answer and
asserts that exactly the intended error code fires. Several cases are the literal failure texts
observed from Qwen in the earlier audits."""
import copy

import pytest

from src.nova_qwen.features import data_quality_notes, normalize_evidence
from src.nova_qwen.response import parse_content, validate_response

EVIDENCE = {"Flow Duration": 101041626.0, "Total Fwd Packets": 11.0, "Flow Bytes/s": 123.7509776}
NORM = normalize_evidence(EVIDENCE)
NOTES = data_quality_notes(NORM)
VALID = {
    "observations": ["Flow Duration is 101.041626 seconds.", "Total Fwd Packets is 11 packets.",
                     "Flow Bytes/s is 123.7509776 bytes per second."],
    "explanation": "A flow lasting about 101 seconds with 11 forward packets and a low byte rate could be "
                   "consistent with a slow DoS pattern, but these values cannot establish that an attack occurred.",
    "limitations": ["Only three flow features are available.", "No packet contents or ground truth are available."],
    "evidence_used": ["Flow Duration", "Total Fwd Packets", "Flow Bytes/s"],
    "priority": None,
    "abstain": False,
}


def codes(answer, *, norm=NORM, notes=NOTES, cls="DoS", done="stop"):
    v = validate_response(answer, norm, notes, cls, done)
    return sorted({e["code"] for e in v.errors}), sorted({w["code"] for w in v.warnings})


def edit(**changes):
    a = copy.deepcopy(VALID)
    a.update(changes)
    return a


def test_valid_answer_passes_without_warnings():
    assert codes(VALID) == ([], [])


@pytest.mark.parametrize("text", [
    "Flow Duration is 101.041626 seconds.", "The flow lasted about 101 seconds.", "It lasted 101.04 s.",
    "It lasted 101,041,626 microseconds.", "Roughly 101041.626 milliseconds elapsed.",
    "A duration of 1.68 minutes was observed.",
])
def test_correct_duration_statements_pass(text):
    assert codes(edit(observations=[text]))[0] == []


@pytest.mark.parametrize("text", [
    "Flow Duration of 101041626.0 seconds (approximately 3.18 years).",   # audit regression 1
    "The flow lasted 101041626 seconds.",
    "The flow lasted about 3.2 years.",
    "Flow Duration (119,989,925 microseconds, approximately 33.3 hours).",  # wrong conversion
    "The duration was 101 milliseconds.",
])
def test_wrong_duration_units_are_rejected(text):
    assert "TIME_UNIT_MISMATCH" in codes(edit(observations=[text]))[0]


def test_rates_are_not_mistaken_for_durations():
    assert codes(edit(observations=["Flow Bytes/s is 123.7509776 bytes per second.",
                                    "Flow Bytes/s is about 123.75 bytes/s."])) == ([], [])


def test_small_durations_including_scientific_notation():
    norm = normalize_evidence({"Flow Duration": 67.0})
    base = edit(observations=["Flow Duration is 0.000067 seconds."], evidence_used=["Flow Duration"],
                explanation="A very short flow could fit the prediction but cannot confirm nothing more.",
                limitations=["Only one feature."])
    assert codes(base, norm=norm, notes=[]) == ([], [])
    for ok in ("It lasted 6.7e-05 seconds.", "It lasted 67 microseconds.", "It lasted 0.067 ms."):
        assert codes(edit(**{**base, "observations": [ok]}), norm=norm, notes=[])[0] == []
    assert codes(edit(**{**base, "observations": ["It lasted 67 seconds."]}), norm=norm, notes=[])[0] == \
        ["TIME_UNIT_MISMATCH"]


def test_invented_number_is_rejected():
    assert codes(edit(explanation="Typical DoS flows exceed 5000 packets, so this is unusual but unverified."))[0] \
        == ["NUMBER_NOT_IN_EVIDENCE"]


def test_feature_count_may_be_mentioned():
    assert codes(edit(limitations=["Only 3 features were provided."]))[0] == []


def test_time_mentioned_without_duration_evidence():
    norm = normalize_evidence({"Total Fwd Packets": 11.0})
    a = edit(observations=["Total Fwd Packets is 11 packets."], evidence_used=["Total Fwd Packets"],
             explanation="The flow lasted 4 minutes, which cannot be verified.", limitations=["One feature only."])
    assert codes(a, norm=norm, notes=[])[0] == ["TIME_WITHOUT_DURATION"]


@pytest.mark.parametrize("sentence, code", [
    ("This confirms a DoS attack.", "CONFIRMATION_CLAIM"),
    ("The flow is definitely malicious.", "CONFIRMATION_CLAIM"),
    ("These values prove the prediction.", "CONFIRMATION_CLAIM"),
    ("The long duration is confirming the attack.", "CONFIRMATION_CLAIM"),
    ("These values proved it was malicious.", "CONFIRMATION_CLAIM"),
    ("This traffic is a DoS attack.", "ASSERTIVE_CLASSIFICATION"),
    ("The probability of an attack is high.", "CONFIDENCE_CLAIM"),
    ("Model confidence is high for this alert.", "CONFIDENCE_CLAIM"),
    ("The record looks corrupted.", "UNSUPPORTED_DATA_QUALITY_CLAIM"),
    ("Such a duration is physically impossible, suggesting data corruption.", "UNSUPPORTED_DATA_QUALITY_CLAIM"),
    ("The SYN flood pattern is visible.", "INVENTED_NETWORK_DETAIL"),
])
def test_forbidden_claims(sentence, code):
    errs = codes(edit(explanation=VALID["explanation"] + " " + sentence))[0]
    assert code in errs and set(errs) <= {code, "CONFIRMATION_CLAIM", "ASSERTIVE_CLASSIFICATION"}, errs


@pytest.mark.parametrize("sentence, code", [
    ("The classifier is 95% sure.", "CONFIDENCE_CLAIM"),
    ("Traffic came from 192.168.1.20.", "INVENTED_NETWORK_DETAIL"),
    ("It targeted port 80 repeatedly.", "INVENTED_NETWORK_DETAIL"),
])
def test_forbidden_claims_with_invented_numbers(sentence, code):
    # The invented number is ALSO reported; the intended check must still fire.
    assert codes(edit(explanation=VALID["explanation"] + " " + sentence))[0] == sorted({code, "NUMBER_NOT_IN_EVIDENCE"})


@pytest.mark.parametrize("sentence", [
    "The prediction has not been confirmed.",
    "No confidence score or probability is available.",
    "No TCP flags, ports or IP addresses are available.",
    "There is no indication that the data is corrupted.",
    "The traffic is not necessarily a DoS attack.",
])
def test_negated_mentions_are_allowed(sentence):
    assert codes(edit(limitations=[sentence]))[0] == []


def test_concrete_network_values_rejected_even_when_negated():
    assert codes(edit(limitations=["It did not come from 10.0.0.1."]))[0] == \
        ["INVENTED_NETWORK_DETAIL", "NUMBER_NOT_IN_EVIDENCE"]


def test_data_quality_discussion_allowed_when_a_note_was_supplied():
    norm = normalize_evidence({"Flow Duration": -1.0, "Total Fwd Packets": 1.0})
    notes = data_quality_notes(norm)
    a = edit(observations=["Flow Duration is -0.000001 seconds.", "Total Fwd Packets is 1 packets."],
             evidence_used=["Flow Duration", "Total Fwd Packets"],
             explanation="The negative duration is physically impossible and indicates a data-quality problem.",
             limitations=["The record may be unreliable."])
    assert codes(a, norm=norm, notes=notes) == ([], [])
    ignored = edit(**{**a, "explanation": "The values could fit the prediction but cannot establish it."})
    assert codes(ignored, norm=norm, notes=notes)[1] == ["DATA_QUALITY_NOTE_IGNORED"]


@pytest.mark.parametrize("change, code", [
    ({"priority": "high"}, "PRIORITY_NOT_NULL"),
    ({"evidence_used": ["Flow Duration", "Destination Port"]}, "EVIDENCE_USED_UNKNOWN"),
    ({"evidence_used": ["Flow Duration", "Flow Duration"]}, "EVIDENCE_USED_DUPLICATE"),
    ({"evidence_used": []}, "EVIDENCE_USED_EMPTY"),
    ({"observations": []}, "OBSERVATIONS_EMPTY"),
    ({"explanation": "  "}, "EXPLANATION_EMPTY"),
    ({"abstain": True, "limitations": []}, "LIMITATIONS_EMPTY"),
    ({"limitations": []}, "LIMITATIONS_EMPTY"),
    ({"abstain": True, "observations": []}, "OBSERVATIONS_EMPTY"),
    ({"explanation": "."}, "EXPLANATION_EMPTY"),
    ({"explanation": "Could be."}, "EXPLANATION_EMPTY"),
    ({"explanation": "word " * 300}, "RESPONSE_LENGTH"),
    ({"observations": ["a"] * 7}, "RESPONSE_LENGTH"),
    ({"limitations": ["y" * 401]}, "RESPONSE_LENGTH"),
])
def test_structural_policy_errors(change, code):
    assert codes(edit(**change))[0] == [code]


@pytest.mark.parametrize("change", [
    {"observations": "Flow Duration is 101 s"}, {"observations": [1, 2]}, {"observations": ["ok", ""]},
    {"limitations": None}, {"explanation": 5}, {"abstain": "false"}, {"evidence_used": "Flow Duration"},
])
def test_type_errors(change):
    assert codes(edit(**change))[0] == ["RESPONSE_TYPE"]


def test_missing_and_extra_keys():
    a = edit()
    del a["limitations"]
    assert codes(a)[0] == ["RESPONSE_KEYS"]
    assert codes(edit(alert_id="dt-000002"))[0] == ["RESPONSE_KEYS"]
    assert codes(edit(confidence=0.9))[0] == ["RESPONSE_KEYS"]


def test_truncated_generation_is_rejected_even_if_json_is_valid():
    assert codes(VALID, done="length")[0] == ["RESPONSE_TRUNCATED"]
    assert codes(VALID, done=None)[0] == ["RESPONSE_TRUNCATED"]


def test_abstention_is_valid_with_observations_and_a_reason():
    assert codes(edit(abstain=True, explanation="", evidence_used=[]))[0] == []


def test_warnings_do_not_reject():
    errs, warns = codes(edit(explanation="These values could fit DoS or DDoS; the prediction is unverified."))
    assert errs == [] and warns == ["OTHER_CLASS_MENTIONED"]
    errs, warns = codes(edit(observations=["The flow was long."]))
    assert errs == [] and warns == ["OBSERVATION_WITHOUT_FEATURE"]


@pytest.mark.parametrize("content, ok", [
    ('{"a": 1}', True), ("[1, 2]", False), ('{"a": 1', False), ("", False), ('{"a": 1} trailing', False),
])
def test_parse_content(content, ok):
    obj, err = parse_content(content)
    assert (obj is not None) == ok and (err is None) == ok


def test_non_object_answer():
    assert codes(None)[0] == ["RESPONSE_NOT_OBJECT"]


# --------------------------------------------------------------------------- review regressions (adversarial review)
@pytest.mark.parametrize("text", [
    "Flow Duration was about three years.", "Flow Duration lasted roughly seventeen hours.",
    "The flow lasted a few minutes.", "Flow Duration was 101.041626 nanoseconds.", "Flow Duration was 101.041626 ns.",
    "Flow Duration was 101.041626 msec.", "Flow Duration was 101.041626 usec.", "Flow Duration was 101.041626 microsec.",
    "Flow Duration was 101.041626 \u00b5sec.", "Flow Duration was 101.041626 h.", "It was a 101041626-second flow.",
    "It was a 101.041626-hour flow.", "Flow Duration was 101 million seconds.", "The flow persisted for hours.",
    "Flow Duration was 1.68e2 seconds.", "Flow Duration was 101041626 seconds.",
])
def test_wrong_or_uncheckable_durations_rejected(text):
    assert "TIME_UNIT_MISMATCH" in codes(edit(observations=[text]))[0]


@pytest.mark.parametrize("text", [
    "Flow Duration was 1 minute 41 seconds.", "Flow Duration was 1 min 41 s.", "Flow Duration was 101041.626 msec.",
    "Flow Duration was 101041626 usec.", "It was a 101.041626-second flow.", "Flow Duration was 101.041626 seconds.",
    "The dataset is CICIDS 2017 and Flow Duration was 101.04 seconds.",
])
def test_correct_durations_and_constants_pass(text):
    assert codes(edit(observations=[text]))[0] == []


def test_negative_duration_without_sign_when_called_negative():
    norm = normalize_evidence({"Flow Duration": -1.0, "Total Fwd Packets": 1.0})
    notes = data_quality_notes(norm)
    a = edit(observations=["Flow Duration is negative, 1 microsecond.", "Total Fwd Packets is 1 packet."],
             evidence_used=["Flow Duration", "Total Fwd Packets"],
             explanation="The negative duration indicates a data-quality problem, so the prediction cannot be assessed.",
             limitations=["The record may be unreliable."])
    assert codes(a, norm=norm, notes=notes)[0] == []
    b = edit(**{**a, "observations": ["Flow Duration lasted 1 microsecond.", "Total Fwd Packets is 1 packet."]})
    assert codes(b, norm=norm, notes=notes)[0] == ["TIME_UNIT_MISMATCH"]


@pytest.mark.parametrize("sentence, code", [
    ("This flow is a DoS attack, not normal traffic.", "ASSERTIVE_CLASSIFICATION"),
    ("This is a DoS attack.", "ASSERTIVE_CLASSIFICATION"),
    ("It is a DoS attack.", "ASSERTIVE_CLASSIFICATION"),
    ("This is definitely a DoS attack, not benign traffic.", "CONFIRMATION_CLAIM"),
    ("The attack is confirmed, although packet contents are not available.", "CONFIRMATION_CLAIM"),
    ("The prediction is not verified, but this confirms a DoS attack.", "CONFIRMATION_CLAIM"),
    ("The model has high confidence in DoS even without packet contents.", "CONFIDENCE_CLAIM"),
    ("The flow used TCP SYN packets, but payloads are not available.", "INVENTED_NETWORK_DETAIL"),
    ("Certainly this pattern is malicious.", "CONFIRMATION_CLAIM"),
    ("The values undoubtedly show an attack.", "CONFIRMATION_CLAIM"),
    ("The values unquestionably show an attack.", "CONFIRMATION_CLAIM"),
    ("The duration clearly indicates an attack.", "CONFIRMATION_CLAIM"),
    ("The outcome is certain for this flow.", "CONFIRMATION_CLAIM"),
    ("This is malicious without a doubt.", "CONFIRMATION_CLAIM"),
    ("The classifier may have high confidence in this.", "CONFIDENCE_CLAIM"),
    ("These values confirm the attack.", "CONFIRMATION_CLAIM"),
    ("This traffic was a port scan.", "ASSERTIVE_CLASSIFICATION"),
])
def test_claims_that_previously_bypassed_the_checks(sentence, code):
    errs = codes(edit(explanation=VALID["explanation"] + " " + sentence))[0]
    assert code in errs, errs


@pytest.mark.parametrize("sentence", [
    "Further analysis would be needed to confirm the prediction.",
    "Confirmation would require packet captures.",
    "Certain attack types can produce long flows, but this is not established here.",
    "This is a possible DoS attack pattern, not a confirmed one.",
    "There is no indication that the data is corrupted.",
    "The prediction cannot be determined with certainty.",
    "It is not necessarily an attack.",
    "Data-quality problems cannot be ruled out without more features.",
])
def test_hedged_statements_are_not_claims(sentence):
    assert codes(edit(limitations=[sentence]))[0] == []


@pytest.mark.parametrize("obs", [
    "Total Fwd Packets was 101 packets.", "Flow Bytes/s was 11 bytes per second.",
    "Flow Bytes/s was 101041626 bytes per second.", "Flow Duration (in seconds) was 101041626.",
])
def test_values_must_belong_to_the_named_feature(obs):
    assert "NUMBER_NOT_IN_EVIDENCE" in codes(edit(observations=[obs]))[0]


def test_two_values_in_one_sentence_are_checked_against_all_features():
    assert codes(edit(observations=["Flow Bytes/s is 123.7509776 bytes per second over 11 packets."]))[0] == []


@pytest.mark.parametrize("text, ok", [
    ("Flow Duration was 101.04 seconds.", True), ("Flow Duration was 101.05 seconds.", False),
    ("Flow Duration was about 101 seconds.", True), ("Flow Duration was about 102 seconds.", False),
    ("Flow Duration was 1.68 minutes.", True), ("Flow Duration was 1.6 minutes.", False),
    ("Flow Bytes/s was 123.75 bytes per second.", True), ("Flow Bytes/s was 123.76 bytes per second.", False),
    ("Flow Bytes/s was 125 bytes per second.", False),
])
def test_tolerance_is_the_rounding_of_the_written_number(text, ok):
    errs = codes(edit(observations=[text]))[0]
    assert (errs == []) == ok, errs


def test_bare_seconds_without_duration_are_still_number_checked():
    norm = normalize_evidence({"Total Fwd Packets": 11.0})
    a = edit(observations=["Total Fwd Packets is 11 packets."], evidence_used=["Total Fwd Packets"],
             explanation="Eleven packets could fit the prediction, but this cannot establish an attack.",
             limitations=["One feature only."])
    assert codes(a, norm=norm, notes=[])[0] == []
    b = edit(**{**a, "explanation": a["explanation"] + " It took 42 s."})
    assert codes(b, norm=norm, notes=[])[0] == ["TIME_WITHOUT_DURATION"]
    c = edit(**{**a, "explanation": a["explanation"] + " It took 42s."})
    assert codes(c, norm=norm, notes=[])[0] == ["TIME_WITHOUT_DURATION"]


@pytest.mark.parametrize("content", ['{"a": NaN}', '{"a": Infinity}', "[" * 100000 + "]" * 100000])
def test_parse_content_rejects_non_strict_json(content):
    obj, err = parse_content(content)
    assert obj is None and err


@pytest.mark.parametrize("sentence", [
    "No packet-level details (payloads, TCP flags) are provided.",          # live run, round 2
    "No packet-level details (e.g. TCP flags, payloads) are available.",
    "The evidence lacks payloads, TCP flags, and port numbers.",
    "There are no IP addresses, ports, or protocol details such as TCP or UDP.",
])
def test_negated_lists_are_not_invented_details(sentence):
    assert codes(edit(limitations=[sentence]))[0] == []


@pytest.mark.parametrize("sentence", [
    "The prediction is not verified, but the flow used TCP SYN packets.",
    "No payloads are available; the flow used TCP.",
    "Payloads are unavailable, however TCP SYN flags were set.",
])
def test_negation_does_not_cross_a_real_clause_boundary(sentence):
    # explanation/observations are strict fields
    assert "INVENTED_NETWORK_DETAIL" in codes(edit(explanation=VALID["explanation"] + " " + sentence))[0]
