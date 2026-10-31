"""Prompt construction, response schema and context-budget guard.

One alert per request. Identifiers (run_id, alert_id, record_id, detector name) are NOT sent to
the model as fields to copy; they are attached to the answer in code. The model receives only the
classifier's name, its unverified predicted class, unit-labelled evidence and data-quality notes.
"""
from __future__ import annotations

import json

from .errors import PromptBudgetExceeded

PROMPT_VERSION = "nova-qwen-prompt-1.0"
RESPONSE_SCHEMA_VERSION = "nova-qwen-response-0.1-provisional"

DETECTOR_NAMES = {"decision_tree": "Decision Tree", "random_forest": "Random Forest"}

SYSTEM_PROMPT = """You help a security analyst review alerts from a machine-learning network-flow \
classifier (a Decision Tree or Random Forest trained on CICIDS2017 flow features).

Rules:
1. predicted_class is an UNVERIFIED classifier prediction. Never state or imply that an attack is \
confirmed, proven or certain.
2. Use only the evidence in the input. Every value is already converted to the unit written next to \
it. Quote values exactly as written, with their units. Do not convert units and do not calculate \
new numbers.
3. Do not invent anything that is not in the input: no IP addresses, ports, protocols, TCP flags, \
timestamps, payloads, attacker intent, thresholds, typical or baseline values, confidence scores \
or probabilities.
4. Do not claim data corruption, impossibility or errors unless the input lists a data-quality \
note. If it lists one, mention it.
5. "observations": short factual restatements of the evidence values you used, one per feature.
6. "explanation": your cautious interpretation in at most 120 words. Say whether the observations \
could be consistent with the predicted class and what they cannot establish. Separate what was \
observed from what you infer.
7. "limitations": what this evidence cannot show (for example: only a few features, no packet \
contents, no surrounding traffic, no ground truth).
8. "evidence_used": the feature names you used, written exactly as given.
9. "priority": must be null, because no priority policy has been approved.
10. "abstain": true if the evidence is too limited to say anything useful about the prediction; \
still fill observations and limitations.
Respond with one JSON object only."""

TEMPLATE_OVERHEAD_TOKENS = 128  # chat-template and role tokens added by Ollama around the messages


def build_user_message(detector: str, predicted_class: str, normalized: list, notes: list) -> str:
    payload = {
        "classifier": DETECTOR_NAMES[detector],
        "predicted_class": predicted_class,
        "prediction_status": "unverified classifier prediction",
        "evidence": [{"feature": f.feature, "value": f.display, "unit": f.unit} for f in normalized],
        "data_quality_notes": list(notes),
    }
    return "Explain this classifier alert.\n" + json.dumps(payload, indent=2, ensure_ascii=False)


def build_messages(detector: str, predicted_class: str, normalized: list, notes: list) -> list:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(detector, predicted_class, normalized, notes)}]


def response_schema(feature_names: list) -> dict:
    """JSON schema passed as Ollama ``format``; evidence_used is restricted to this alert's features."""
    return {
        "type": "object",
        "properties": {
            "observations": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            "explanation": {"type": "string"},
            "limitations": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            "evidence_used": {"type": "array", "items": {"type": "string", "enum": list(feature_names)},
                              "maxItems": len(feature_names)},
            "priority": {"type": "null"},
            "abstain": {"type": "boolean"},
        },
        "required": ["observations", "explanation", "limitations", "evidence_used", "priority", "abstain"],
        "additionalProperties": False,
    }


def feedback_message(issues: list) -> str:
    lines = ["Your previous answer was rejected by automatic checks:"]
    lines += [f"- {i['code']}: {i['message']}" for i in issues]
    lines.append("Answer again with one corrected JSON object that follows every rule.")
    return "\n".join(lines)


def request_bytes(messages: list, schema: dict) -> int:
    return sum(len(m["content"].encode("utf-8")) for m in messages) + len(json.dumps(schema).encode("utf-8"))


def check_budget(messages: list, schema: dict, num_ctx: int, num_predict: int) -> int:
    """Refuse requests that could be truncated.

    Qwen uses a byte-level BPE tokenizer, so a text never has more tokens than UTF-8 bytes. If
    bytes + template overhead + num_predict fits in num_ctx, Ollama cannot truncate the prompt
    (Ollama otherwise truncates silently and still returns HTTP 200). The schema is counted too,
    as a conservative allowance in case a runtime injects it into the prompt.
    """
    size = request_bytes(messages, schema)
    needed = size + TEMPLATE_OVERHEAD_TOKENS + num_predict
    if needed > num_ctx:
        raise PromptBudgetExceeded(
            f"request needs up to {needed} tokens ({size} bytes + {TEMPLATE_OVERHEAD_TOKENS} template + "
            f"{num_predict} output) but num_ctx is {num_ctx}",
            details={"request_bytes": size, "upper_bound_tokens": needed, "num_ctx": num_ctx})
    return size
