"""Validation of a model answer.

Valid JSON is not the same as a correct explanation. Besides the structure, every check that can
be decided deterministically is enforced here.

Errors (the answer is rejected):
  structure and types; truncated generation; non-null priority; evidence_used naming features that were
  not supplied; missing observations or limitations; numbers that do not equal a supplied value (a number
  written right after a feature name must be THAT feature's value); time statements that do not equal the
  normalised Flow Duration (including spelled-out, hyphenated, abbreviated and multiplier forms);
  affirmative certainty ("confirms", "no doubt", "enough to prove"); asserting that the traffic IS an
  attack; confidence/probability claims; data-corruption claims without a data-quality note; invented
  network details (IP addresses, port numbers, protocols, TCP flags).
Warnings (kept for human review): other attack classes mentioned, data-quality note not mentioned,
observations that name no feature.

Field roles. "limitations" exist to say what is NOT known, so there protocol names, confidence words and
unattached time words are legitimate ("no TCP flags or confidence scores are available"); certainty,
assertion, number, unit and data-quality checks still apply everywhere.
Negation scope: a claim is excused only by a negation earlier in the SAME clause. Clauses end at ; : at
but/although/however/while/yet/because/since/so/therefore, and where a comma, "and/or" or "that" starts a new
subject. This keeps negated lists together ("cannot confirm X, identify Y, or establish Z") while
"Payloads are not available, this confirms a DoS attack" is still rejected.

Semantic quality beyond these checks still needs human review (python -m src.nova_qwen worksheet).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .alerts import ATTACK_CLASSES
from .ollama import strict_json

RESPONSE_KEYS = ("observations", "explanation", "limitations", "evidence_used", "priority", "abstain")
MAX_ITEMS, MAX_ITEM_CHARS, MAX_EXPLANATION_CHARS, MIN_EXPLANATION_WORDS = 6, 400, 1200, 5
ALLOWED_CONSTANTS = (2017.0,)  # the dataset name "CICIDS2017" / "CICIDS 2017" appears in the system prompt

NUM = r"[-+]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?"
NUMBER_RE = re.compile(rf"(?<![\w.]){NUM}(?![\w])")
TIME_UNITS = {
    "nanosecond": 1e-9, "nanoseconds": 1e-9, "ns": 1e-9, "nsec": 1e-9, "nsecs": 1e-9,
    "microsecond": 1e-6, "microseconds": 1e-6, "µs": 1e-6, "μs": 1e-6, "us": 1e-6, "usec": 1e-6, "usecs": 1e-6,
    "µsec": 1e-6, "μsec": 1e-6, "microsec": 1e-6, "microsecs": 1e-6,
    "millisecond": 1e-3, "milliseconds": 1e-3, "ms": 1e-3, "msec": 1e-3, "msecs": 1e-3, "millisec": 1e-3,
    "millisecs": 1e-3,
    "second": 1.0, "seconds": 1.0, "sec": 1.0, "secs": 1.0, "s": 1.0,
    "minute": 60.0, "minutes": 60.0, "min": 60.0, "mins": 60.0,
    "hour": 3600.0, "hours": 3600.0, "hr": 3600.0, "hrs": 3600.0, "h": 3600.0,
    "day": 86400.0, "days": 86400.0, "week": 604800.0, "weeks": 604800.0, "wk": 604800.0, "wks": 604800.0,
    "month": 2629800.0, "months": 2629800.0, "year": 31557600.0, "years": 31557600.0, "yr": 31557600.0,
    "yrs": 31557600.0,
}
MULTIPLIERS = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
UNIT_ALT = "|".join(sorted((re.escape(u) for u in TIME_UNITS), key=len, reverse=True))
TIME_RE = re.compile(rf"(?<![\w.])({NUM})(?:\s*(thousand|million|billion|trillion))?\s*(?:-|\(\s*)?\s*({UNIT_ALT})"
                     rf"(?![\w/])", re.I)
COMPOUND_RE = re.compile(rf"(?<![\w.])({NUM})\s*({UNIT_ALT})(?:\s*,?\s*(?:and\s+)?({NUM})\s*({UNIT_ALT})){{1,2}}(?![\w/])",
                         re.I)
PART_RE = re.compile(rf"({NUM})\s*({UNIT_ALT})(?![\w/])", re.I)
ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
SCALES = {"hundred": 100, "thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9}
VAGUE = r"a few|several|many|dozens of|a dozen|a couple of|half an?|a fraction of an?"
NUMBER_WORD = "|".join(sorted(list(ONES) + list(TENS) + list(SCALES), key=len, reverse=True))
TIME_WORD_UNITS = (r"nanoseconds?|microseconds?|milliseconds?|seconds?|minutes?|hours?|days?|weeks?|months?|years?|"
                   r"yrs?|wks?|hrs?|mins?|secs?")
SPELLED_TIME_RE = re.compile(rf"\b((?:{VAGUE})|(?:(?:{NUMBER_WORD})(?:(?:[\s-]+|\s+and\s+)(?:{NUMBER_WORD}))*))"
                             rf"[\s-]+({TIME_WORD_UNITS})\b", re.I)
BIG_TIME_WORD_RE = re.compile(r"\b(minutes?|hours?|days?|weeks?|months?|years?|yrs?|wks?|decades?)\b", re.I)
RATE_RE = re.compile(r"\b(?:bytes?|packets?|bits?)\s*(?:per|/)\s*(?:second|sec|s)\b|\bper\s+(?:second|sec)\b|/s\b",
                     re.I)

NEGATION_RE = re.compile(r"\b(?:no|not|never|none|without|cannot|can't|isn't|aren't|doesn't|didn't|wasn't|"
                         r"lacks?|lacking|absent|unavailable|unverified|neither|nor|unconfirmed|nothing|insufficient|"
                         r"unable|restricts?|limits?|prevents?|precludes?)\b|n't\b", re.I)
# A negation covers the rest of its CLAUSE (so it governs lists: "cannot confirm X, identify Y, or establish Z";
# "no packet details (payloads, TCP flags)"). A clause ends at ; : , at contrastive/causal conjunctions, and where
# a comma or "and/or" introduces a new subject ("..., this confirms", "... and the values confirm"), and after
# factive predicates ("not surprising that this is an attack").
SUBJECT = r"(?:this|it|that|these|those|which|they|there|the (?:values?|evidence|data|flow|traffic|features?|record|alert|model))\b"
CLAUSE_BOUNDARY_RE = re.compile(rf"[;:]|\b(?:but|although|though|however|while|whereas|yet|because|since|so|therefore|"
                                rf"thus|hence)\b|,\s*(?:and\s+|or\s+)?(?={SUBJECT})|\b(?:and|or)\s+(?={SUBJECT})|"
                                rf"\b(?:surprising|unsurprising|notable|noteworthy|interesting|important)\s+that\s+(?={SUBJECT})",
                                re.I)
# Only FACTIVE predicates make "that" start an independent claim ("not surprising that this is an attack" asserts it);
# "no indication that the data is corrupted" or "not clear that this is an attack" keep the negation.
ABBREVIATIONS_RE = re.compile(r"\b(e\.g|i\.e|etc|vs|approx)\.", re.I)
# Affirmations that contain a negation word: never excused.
INTENSIFIER_RE = re.compile(r"\b(?:no|without|beyond)\s+(?:any\s+|a\s+)?(?:doubt|question)\b|"
                            r"\bunmistakabl[ey]\b|\bindisputabl[ey]\b|\bundeniabl[ey]\b", re.I)
# "certain" alone usually means "some" and the noun "confirmation" is used in hedges, so neither is listed.
CONFIRM_RE = re.compile(r"\b(?:confirm|confirms|confirmed|confirming|definitely|definitive(?:ly)?|certainly|certainty|"
                        r"undoubtedly|unquestionably|prove|proves|proved|proven|proving|demonstrates?|demonstrated|"
                        r"establish(?:es|ed)?|clearly (?:is|was|an?|indicates?|shows?|demonstrates?)|is certain)\b",
                        re.I)
HEDGE_RE = re.compile(r"\b(?:to|needs?|needed|would|could|might|may|must|should|until|unless|if|whether|"
                      r"cannot be|before)\b", re.I)
SUFFICIENCY_RE = re.compile(r"\b(?:enough|sufficient|sufficiently|suffices?)\b", re.I)
ATTACK_NOUNS = (r"attack|dos|ddos|denial[- ]of[- ]service|port ?scan(?:ning)?|scan|brute[- ]force|bot(?:net)?|"
                r"web attacks?|malicious|intrusion|exploit|flood")
ASSERTIVE_RE = re.compile(rf"\b(?:(?:this|that|it|the)(?:\s+(?:flow|traffic|record|connection|alert|activity|session|"
                          rf"host))?\s+(?:is|was|represents|constitutes)|it's|that's)\s+(?:an?\s+|the\s+)?(?:[\w-]+\s+){{0,2}}"
                          rf"(?:{ATTACK_NOUNS})\b", re.I)
# Words that turn an assertion into a statement about the PREDICTION or a possibility.
PREDICTION_CONTEXT_RE = re.compile(r"\b(?:possible|possibly|potential|potentially|suspected|predicted|prediction|"
                                   r"predicts?|alleged|apparent|likely|probable|presumed|classified|classifier|"
                                   r"classification|labell?ed|flagged|identified|categori[sz]ed|detected|marked|"
                                   r"assigned|considered|model|consistent with)\b", re.I)
CONFIDENCE_RE = re.compile(r"[0-9]\s*%|\bper ?cent\b|\bconfiden(?:ce|t)\b|\bprobabilit(?:y|ies)\b|\bprobable\b|"
                           r"\blikelihood\b|\bodds\b|\bscore of\b", re.I)
DATA_QUALITY_RE = re.compile(r"\b(?:corrupt\w*|impossible|invalid (?:data|value|record)|erroneous|data error|"
                             r"unrealistic|implausible|glitch\w*|faulty|nonsensical)\b", re.I)   # claims
DATA_QUALITY_MENTION_RE = re.compile(DATA_QUALITY_RE.pattern + r"|\bdata[- ]quality\b|\bartef?act\b|\bartifact\b|"
                                     r"\bnegative\b", re.I)
CONCRETE_NETWORK_RE = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b|\bports?\s*(?:number\s*)?[0-9]+\b", re.I)
PROTOCOL_RE = re.compile(r"\b(?:TCP|UDP|ICMP|SYN|ACK|RST|FIN|PSH|URG|HTTP|HTTPS|DNS|SSH|FTP|TLS|SSL|SMTP)\b")
CLAUSE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")


@dataclass
class Validation:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def error(self, code: str, message: str) -> None:
        self.errors.append({"code": code, "message": message})

    def warn(self, code: str, message: str) -> None:
        self.warnings.append({"code": code, "message": message})

    def to_record(self) -> dict:
        return {"passed": self.passed, "errors": self.errors, "warnings": self.warnings}


def parse_content(content):
    """The whole content must be exactly one strict JSON object (finite numbers, bounded nesting)."""
    if not isinstance(content, str):
        return None, "content is not a string"
    try:
        value = strict_json(content)
    except ValueError as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(value, dict):
        return None, f"expected a JSON object, got {type(value).__name__}"
    return value, None


def _to_float(token: str):
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def _decimals(token: str) -> int:
    mantissa = token.lower().split("e")[0]
    return len(mantissa.split(".")[1]) if "." in mantissa else 0


def _rounding(token: str, scale: float) -> float:
    """Half a unit in the last written place, in seconds-of-scale."""
    exp = int(token.lower().split("e")[1]) if "e" in token.lower() else 0
    return 0.5 * 10 ** (exp - _decimals(token)) * scale


def _matches(token: str, value: float, scale: float = 1.0) -> bool:
    """Does a written number equal ``value``, up to the rounding of the written number?"""
    x = _to_float(token)
    if x is None:
        return False
    return abs(x * scale - value) <= _rounding(token, scale) + 1e-9 * max(1.0, abs(value))


def _words_to_number(words: str):
    """'one hundred and one' -> 101; None for vague quantities ('a few') or unknown words."""
    total, current, seen = 0, 0, False
    for w in re.split(r"[\s-]+", words.lower().strip()):
        if w == "and":
            continue
        if w in ONES:
            current += ONES[w]
        elif w in TENS:
            current += TENS[w]
        elif w in SCALES:
            current = max(current, 1) * SCALES[w]
            if SCALES[w] >= 1000:
                total, current = total + current, 0
        else:
            return None
        seen = True
    return total + current if seen else None


def _window_before(text: str, start: int, strict: bool = True) -> str:
    """The part of the current clause before ``start``. Boundaries are found in the FULL text, because some
    are defined by what follows them ("that this is ..."), which may be the claim itself."""
    clause_start = 0
    for m in CLAUSE_BOUNDARY_RE.finditer(text):
        if m.end() > start:
            break
        clause_start = m.end()
    return text[clause_start:start]


def _negated_before(text: str, start: int, strict: bool = True) -> bool:
    return bool(NEGATION_RE.search(_window_before(text, start, strict)))


def _sentences(text: str) -> list:
    """Split into sentences without breaking at "e.g." / "i.e." / "etc." (same length, so offsets stay valid)."""
    protected = ABBREVIATIONS_RE.sub(lambda m: m.group(0)[:-1] + "․", text)
    return [s.replace("․", ".") for s in CLAUSE_SPLIT_RE.split(protected) if s.strip()]


def validate_response(obj, normalized: list, notes: list, predicted_class: str,
                      done_reason: str | None = "stop") -> Validation:
    v = Validation()
    if done_reason != "stop":
        v.error("RESPONSE_TRUNCATED", f"generation ended with done_reason={done_reason!r}, not 'stop'")
    if not isinstance(obj, dict):
        v.error("RESPONSE_NOT_OBJECT", "the answer is not a JSON object")
        return v

    keys = set(obj)
    missing, extra = [k for k in RESPONSE_KEYS if k not in keys], sorted(keys - set(RESPONSE_KEYS))
    if missing or extra:
        v.error("RESPONSE_KEYS", f"missing {missing}, unexpected {extra}")
    if missing:
        return v  # type and content checks need every key; the missing ones are already reported
    observations, limitations = obj.get("observations"), obj.get("limitations")
    explanation, evidence_used, abstain = obj.get("explanation"), obj.get("evidence_used"), obj.get("abstain")
    for name, value in (("observations", observations), ("limitations", limitations)):
        if not isinstance(value, list) or not all(isinstance(x, str) and x.strip() for x in value):
            v.error("RESPONSE_TYPE", f"{name} must be a list of non-empty strings")
        elif len(value) > MAX_ITEMS or any(len(x) > MAX_ITEM_CHARS for x in value):
            v.error("RESPONSE_LENGTH", f"{name} exceeds {MAX_ITEMS} items or {MAX_ITEM_CHARS} characters per item")
    if not isinstance(explanation, str):
        v.error("RESPONSE_TYPE", "explanation must be a string")
    elif len(explanation) > MAX_EXPLANATION_CHARS:
        v.error("RESPONSE_LENGTH", f"explanation exceeds {MAX_EXPLANATION_CHARS} characters")
    if not isinstance(abstain, bool):
        v.error("RESPONSE_TYPE", "abstain must be true or false")
    if obj.get("priority") is not None:
        v.error("PRIORITY_NOT_NULL", f"priority must be null (no approved policy), got {obj.get('priority')!r}")

    supplied = [f.feature for f in normalized]
    if not isinstance(evidence_used, list) or not all(isinstance(x, str) for x in evidence_used):
        v.error("RESPONSE_TYPE", "evidence_used must be a list of feature names")
        evidence_used = []
    else:
        unknown = [x for x in evidence_used if x not in supplied]
        if unknown:
            v.error("EVIDENCE_USED_UNKNOWN", f"evidence_used names features that were not supplied: {unknown}")
        if len(set(evidence_used)) != len(evidence_used):
            v.error("EVIDENCE_USED_DUPLICATE", "evidence_used lists a feature more than once")
    if any(e["code"] in ("RESPONSE_KEYS", "RESPONSE_TYPE") for e in v.errors):
        return v  # content checks below need a well-formed object

    if not observations:
        v.error("OBSERVATIONS_EMPTY", "observations must restate the evidence (also when abstaining)")
    if not limitations:
        v.error("LIMITATIONS_EMPTY", "limitations must state what the evidence cannot show")
    if abstain is False:
        if len(re.findall(r"[A-Za-z]{2,}", explanation)) < MIN_EXPLANATION_WORDS:
            v.error("EXPLANATION_EMPTY", "explanation is empty or not substantive although abstain is false")
        if not evidence_used:
            v.error("EVIDENCE_USED_EMPTY", "evidence_used is empty although abstain is false")

    roles = [(t, True) for t in observations] + [(explanation, True)] + [(t, False) for t in limitations]
    reported = set()
    for text, strict in roles:
        _check_numbers_and_units(v, text, normalized, strict, reported)
        _check_claims(v, text, notes, strict, reported)
    full = " ".join(t for t, _ in roles)
    concrete = CONCRETE_NETWORK_RE.search(full)
    if concrete:
        v.error("INVENTED_NETWORK_DETAIL", f"mentions a network value that is not in the evidence: {concrete.group(0)!r}")

    if notes and not DATA_QUALITY_MENTION_RE.search(full):
        v.warn("DATA_QUALITY_NOTE_IGNORED", "a data-quality note was supplied but the answer does not mention it")
    others = [c for c in ATTACK_CLASSES if c != predicted_class and re.search(rf"\b{re.escape(c)}\b", full)]
    if others:
        v.warn("OTHER_CLASS_MENTIONED", f"mentions other attack classes {others}; check it does not change the prediction")
    for o in observations:
        if not any(name.lower() in o.lower() for name in supplied):
            v.warn("OBSERVATION_WITHOUT_FEATURE", f"observation names no supplied feature: {o[:120]!r}")
    return v


def _once(v: Validation, reported: set, code: str, message: str, key: str) -> None:
    if (code, key) not in reported:
        reported.add((code, key))
        v.error(code, message)


def _check_claims(v: Validation, text: str, notes: list, strict: bool, reported: set) -> None:
    for sentence in _sentences(text):
        short = sentence.strip()[:160]
        if INTENSIFIER_RE.search(sentence):
            _once(v, reported, "CONFIRMATION_CLAIM", f"states the prediction as certain: {short!r}", short)
        for m in CONFIRM_RE.finditer(sentence):
            window = _window_before(sentence, m.start(), strict)
            negated = bool(NEGATION_RE.search(window))
            sufficiency = bool(SUFFICIENCY_RE.search(window))
            if not negated and (sufficiency or not HEDGE_RE.search(window)):
                _once(v, reported, "CONFIRMATION_CLAIM", f"states the prediction as confirmed or certain: {short!r}",
                      short)
                break
        for m in ASSERTIVE_RE.finditer(sentence):
            excused = (NEGATION_RE.search(m.group(0)) or PREDICTION_CONTEXT_RE.search(m.group(0))
                       or PREDICTION_CONTEXT_RE.search(_window_before(sentence, m.start(), strict))
                       or _negated_before(sentence, m.start(), strict))
            if not excused:
                _once(v, reported, "ASSERTIVE_CLASSIFICATION",
                      f"asserts the traffic IS an attack instead of a prediction: {short!r}", short)
                break
        if not notes:
            for m in DATA_QUALITY_RE.finditer(sentence):
                if not _negated_before(sentence, m.start(), strict):
                    _once(v, reported, "UNSUPPORTED_DATA_QUALITY_CLAIM",
                          f"claims a data problem although no data-quality note was supplied: {short!r}", short)
                    break
        if not strict:
            continue  # limitations may name missing protocols / scores ("no TCP flags or confidence score")
        for m in CONFIDENCE_RE.finditer(sentence):
            if not _negated_before(sentence, m.start(), strict):
                _once(v, reported, "CONFIDENCE_CLAIM", f"mentions a confidence, probability or percentage: {short!r}",
                      short)
                break
        for m in PROTOCOL_RE.finditer(sentence):
            if not _negated_before(sentence, m.start(), strict):
                _once(v, reported, "INVENTED_NETWORK_DETAIL",
                      f"asserts a protocol or flag that is not in the evidence: {m.group(0)!r} in {short!r}", short)
                break


def _check_numbers_and_units(v: Validation, text: str, normalized: list, strict: bool, reported: set) -> None:
    duration = next((f for f in normalized if f.feature == "Flow Duration"), None)
    others = [f for f in normalized if f.feature != "Flow Duration"]
    # Plain numbers may equal any non-duration value, the duration in seconds, the feature count or the
    # dataset year. The raw microsecond duration is only acceptable WITH a microsecond unit (checked as a time).
    general = [f.value for f in others] + ([duration.value] if duration else []) + \
        [float(len(normalized))] + list(ALLOWED_CONSTANTS)

    def time_ok(value_seconds: float, tol: float, negative_context: bool) -> bool:
        target = duration.value
        slack = tol + 1e-9 * max(1.0, abs(target))
        return abs(value_seconds - target) <= slack or (negative_context and abs(value_seconds - abs(target)) <= slack)

    for original in _sentences(text):
        sentence = RATE_RE.sub(lambda m: " " * len(m.group(0)), original)  # same length: offsets stay valid
        negative_context = "negative" in original.lower()
        spans = []
        # 1. compound durations ("1 minute 41 seconds"): integer leading parts, each later part smaller than one
        #    unit of the previous; checked once, with the rounding of the LAST part only
        for m in COMPOUND_RE.finditer(sentence):
            parts = [(n, TIME_UNITS.get(u.lower()) or TIME_UNITS.get(u)) for n, u in PART_RE.findall(m.group(0))]
            nums = [_to_float(n) for n, _ in parts]
            if any(x is None for x in nums) or not all(float(x).is_integer() for x in nums[:-1]) or not all(
                    parts[i][1] < parts[i - 1][1] and nums[i] * parts[i][1] < parts[i - 1][1] for i in range(1, len(parts))):
                continue  # e.g. a restatement "0.02176 seconds, 21.76 ms": each part is checked on its own
            spans.append(m.span())
            total = sum(x * f for x, (_, f) in zip(nums, parts))
            if duration is None:
                _once(v, reported, "TIME_WITHOUT_DURATION",
                      f"mentions a time ({m.group(0).strip()!r}) but no duration was supplied", m.group(0))
            elif not time_ok(total, _rounding(parts[-1][0], parts[-1][1]), negative_context):
                _once(v, reported, "TIME_UNIT_MISMATCH",
                      f"{m.group(0).strip()!r} does not equal the Flow Duration of {duration.display} seconds", m.group(0))
        # 2. single numeric time mentions, incl. multiplier ("101 million seconds") and hyphen ("101-second")
        for m in TIME_RE.finditer(sentence):
            if any(a <= m.start() < b for a, b in spans):
                continue
            token, mult, unit = m.group(1), m.group(2), m.group(3)
            factor = (TIME_UNITS.get(unit.lower()) or TIME_UNITS.get(unit)) * (MULTIPLIERS[mult.lower()] if mult else 1.0)
            spans.append(m.span())
            if duration is None:
                _once(v, reported, "TIME_WITHOUT_DURATION",
                      f"mentions a time ({m.group(0).strip()!r}) but no duration was supplied", m.group(0))
            elif not time_ok((_to_float(token) or 0.0) * factor, _rounding(token, factor), negative_context):
                _once(v, reported, "TIME_UNIT_MISMATCH",
                      f"{m.group(0).strip()!r} does not equal the Flow Duration of {duration.display} seconds", m.group(0))
        # 3. spelled-out durations: "five seconds" is checked, "a few minutes" cannot be verified
        for m in SPELLED_TIME_RE.finditer(sentence):
            if any(a <= m.start() < b for a, b in spans) or _negated_before(sentence, m.start(), strict):
                continue
            spans.append(m.span())
            unit = m.group(2).lower()
            factor = TIME_UNITS.get(unit) or TIME_UNITS.get(unit.rstrip("s"))
            amount = _words_to_number(m.group(1))
            if duration is None:
                _once(v, reported, "TIME_WITHOUT_DURATION",
                      f"mentions a time ({m.group(0).strip()!r}) but no duration was supplied", m.group(0))
            elif amount is None or factor is None or not time_ok(amount * factor, 0.5 * factor, negative_context):
                _once(v, reported, "TIME_UNIT_MISMATCH",
                      f"{m.group(0).strip()!r} does not equal (or cannot be checked against) the Flow Duration of "
                      f"{duration.display} seconds; quote the value with its unit", m.group(0))
        # 4. larger time units not attached to a checked value ("lasting hours"); limitations may say what is
        #    missing ("no statistics over minutes or hours")
        if strict:
            for m in BIG_TIME_WORD_RE.finditer(sentence):
                if any(a <= m.start() < b for a, b in spans) or _negated_before(sentence, m.start(), strict):
                    continue
                _once(v, reported, "TIME_UNIT_MISMATCH", f"time word {m.group(0)!r} is not attached to a value that "
                                                         f"can be checked against the Flow Duration",
                      f"word:{m.group(0).lower()}")
        # 5. a number written right after a feature name must be THAT feature's value; every other number must
        #    equal some supplied value
        attached = {}
        for f in normalized:
            pattern = (rf"{re.escape(f.feature)}\s*(?:\([^)]*\)\s*)?(?:is|was|of|=|:|at|equals|reached|measured)?\s*"
                       rf"(?:about|approximately|roughly|around|exactly|only|just)?\s*({NUM})(?![\w])")
            for am in re.finditer(pattern, original, re.I):
                attached[am.start(1)] = f
        for m in NUMBER_RE.finditer(sentence):
            if any(a <= m.start() < b for a, b in spans):
                continue
            token = m.group(0)
            feature = attached.get(m.start())
            pool = [feature.value] if feature is not None else general
            if any(_matches(token, a) for a in pool) or (negative_context and any(_matches(token, abs(a)) for a in pool)):
                continue
            what = f"the {feature.feature} value {feature.display}" if feature is not None else "any supplied value"
            _once(v, reported, "NUMBER_NOT_IN_EVIDENCE", f"the number {token} does not equal {what}", f"num:{token}")
