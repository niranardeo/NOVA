"""Shared helpers for the offline Qwen consumer tests (imported by conftest.py and the tests).

FakeOllama is a real HTTP server on 127.0.0.1 (random port) that implements the Ollama endpoints the
consumer uses. Tests script its behaviour (valid answers, invalid answers, HTTP errors, delays,
dropped connections), so the client is exercised over real sockets without a model. Live tests
against a real Ollama are in tests/qwen_live/ and are opt-in.
"""
from __future__ import annotations

import json
import shutil
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nova_qwen.config import DEFAULT_EXPECTED_DIGEST, DEFAULT_MODEL, QwenSettings  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DT_FIXTURE = FIXTURES / "decision_tree_alerts.json"
RF_FIXTURE = FIXTURES / "random_forest_alerts.json"
DT_EMPTY = FIXTURES / "decision_tree_alerts_empty.json"
RF_EMPTY = FIXTURES / "random_forest_alerts_empty.json"
DIGEST = DEFAULT_EXPECTED_DIGEST
MODEL = DEFAULT_MODEL


def evidence_from_request(body: dict) -> dict:
    """Recover the prompt payload (classifier, class, evidence, notes) from a chat request."""
    user = [m for m in body["messages"] if m["role"] == "user"][0]["content"]
    return json.loads(user.split("\n", 1)[1])


def valid_answer(payload: dict, *, abstain: bool = False) -> dict:
    """A rule-following answer built only from the evidence shown to the model."""
    ev = payload["evidence"]
    answer = {
        "observations": [f"{e['feature']} was {e['value']} {e['unit']}." for e in ev],
        "explanation": "These observations could be consistent with the predicted class, but they do not "
                       "establish it; the prediction remains unverified.",
        "limitations": ["Only a few flow features were provided; packet contents and surrounding traffic "
                        "are not available."],
        "evidence_used": [e["feature"] for e in ev],
        "priority": None,
        "abstain": abstain,
    }
    if payload.get("data_quality_notes"):
        answer["limitations"].append("A data-quality note was supplied: " + payload["data_quality_notes"][0])
    return answer


def chat_body(content, *, done_reason="stop", prompt_eval_count=300, eval_count=120, thinking=None):
    msg = {"role": "assistant", "content": content if isinstance(content, str) else json.dumps(content)}
    if thinking:
        msg["thinking"] = thinking
    return {"model": MODEL, "message": msg, "done": done_reason is not None, "done_reason": done_reason,
            "prompt_eval_count": prompt_eval_count, "eval_count": eval_count, "total_duration": 1_500_000_000,
            "load_duration": 10_000_000}


class PickledModel:
    """A stand-in for a saved classifier, importable when a test loads it back from a file."""

    feature_names_in_ = None

    def __init__(self, label="DoS", every=2, params=None):
        self.label, self.every, self.params = label, every, params or {}

    def predict(self, features):
        import numpy as np
        return np.array([self.label if i % self.every == 0 else "Normal Traffic" for i in range(len(features))])

    def get_params(self):
        return self.params


class RandomForestStandIn(PickledModel):
    """Named like a Random Forest, to test that swapped model files are noticed."""


class PipelineStandIn(PickledModel):
    """Shaped like a scikit-learn Pipeline: the real classifier is the last step."""

    def __init__(self, final):
        super().__init__()
        self.steps = [("scale", None), ("classifier", final)]

    def get_params(self):
        return {"classifier__max_depth": 7}


class SearchStandIn(PickledModel):
    """Shaped like a fitted GridSearchCV: its own settings are not the model's."""

    def __init__(self, best):
        super().__init__()
        self.best_estimator_ = best

    def get_params(self):
        return {"n_jobs": 99}


class DecisionTreeStandIn(PickledModel):
    """Named like a Decision Tree, with settings to record."""

    def get_params(self):
        return {"max_depth": 7, "random_state": 42}


class RaisingModel(PickledModel):
    def predict(self, features):
        raise ValueError("this model cannot predict")


class RunsOnLoad:
    """Unpickling this creates the marker file, so a test can see whether a model file was ever loaded."""

    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return (open, (self.marker, "w"))


class FakeOllama:
    """Scriptable Ollama stand-in. ``chat_actions`` is consumed in order; when empty, ``default_chat``."""

    def __init__(self):
        self.requests = []
        self.chat_actions = []
        self.model_entry = {"name": MODEL, "model": MODEL, "digest": DIGEST, "size": 3324173934,
                            "details": {"family": "qwen35", "parameter_size": "4.2B", "quantization_level": "Q4_K_M"}}
        self.capabilities = ["completion", "vision", "tools", "thinking"]
        self.version_action = None
        self.default_chat = lambda body: ("json", 200, chat_body(valid_answer(evidence_from_request(body))))
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _body(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                return json.loads(raw) if raw else None

            def _send(self, action):
                kind = action[0]
                if kind == "sleep":
                    time.sleep(action[1])
                    return self._send(action[2])
                if kind == "trickle":  # ("trickle", total_bytes, seconds_between_bytes): never-ending slow body
                    total, interval = action[1], action[2]
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(total))
                    self.end_headers()
                    try:
                        for _ in range(total):
                            self.wfile.write(b" ")
                            self.wfile.flush()
                            time.sleep(interval)
                    except OSError:
                        pass
                    return
                if kind == "drop":
                    self.close_connection = True
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    return
                if kind == "json":
                    status, obj = action[1], action[2]
                    data = json.dumps(obj).encode()
                    ctype = "application/json"
                else:  # raw
                    status, data, ctype = action[1], action[2], action[3]
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                if len(action) > 4 and action[4]:
                    for k, v in action[4].items():
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                fake.requests.append(("GET", self.path, None))
                if self.path == "/api/version":
                    return self._send(fake.version_action or ("json", 200, {"version": "0.40.11"}))
                if self.path == "/api/tags":
                    return self._send(("json", 200, {"models": [fake.model_entry] if fake.model_entry else []}))
                if self.path == "/api/ps":
                    return self._send(("json", 200, {"models": []}))
                return self._send(("json", 404, {"error": "not found"}))

            def do_POST(self):
                body = self._body()
                fake.requests.append(("POST", self.path, body))
                if self.path == "/api/show":
                    return self._send(("json", 200, {"capabilities": fake.capabilities}))
                if self.path == "/api/chat":
                    action = fake.chat_actions.pop(0) if fake.chat_actions else fake.default_chat
                    if callable(action):
                        action = action(body)
                    return self._send(action)
                return self._send(("json", 404, {"error": "not found"}))

        class QuietServer(ThreadingHTTPServer):
            def handle_error(self, request, client_address):
                pass  # tests deliberately time out / drop connections; the client side is asserted

        self.server = QuietServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()

    def chat_requests(self):
        return [b for m, p, b in self.requests if p == "/api/chat"]

    def close(self):
        self.server.shutdown()
        self.server.server_close()



def make_settings(url: str, **kw) -> QwenSettings:
    base = dict(url=url, timeout_seconds=5.0, connect_timeout_seconds=2.0, retry_backoff_seconds=0.0,
                transport_retries=2, max_attempts=2)
    base.update(kw)
    return QwenSettings(**base).validate()



def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, doc):
    Path(path).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return Path(path)
