"""Minimal, dependency-free Ollama HTTP client.

Security and robustness choices:
* the standard library's http.client is used directly: no proxy environment variables are
  honoured (alert evidence never goes through an unexpected proxy) and redirects are not followed;
* a connect timeout, plus an overall wall-clock deadline per request enforced by a watchdog that
  closes the socket (a server that trickles bytes cannot keep a request alive);
* response bodies are read in bounded chunks, size-limited, and must be strict JSON objects
  (no NaN/Infinity, bounded nesting) with the documented field types;
* failures become typed errors (unavailable, timeout, HTTP status, protocol), and only transient
  ones are retried, a bounded number of times.
"""
from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

from .config import QwenSettings
from .errors import (ModelDigestMismatch, ModelNotAvailable, OllamaError, OllamaHTTPError, OllamaProtocolError,
                     OllamaTimeout, OllamaUnavailable)

MAX_ERROR_TEXT = 300


@dataclass(frozen=True)
class ChatResult:
    content: str
    thinking: str
    done: bool
    done_reason: str | None
    prompt_eval_count: int | None
    eval_count: int | None
    total_duration_seconds: float | None
    load_duration_seconds: float | None
    model: str | None
    wall_seconds: float

    def metadata(self) -> dict:
        return {"model": self.model, "done": self.done, "done_reason": self.done_reason,
                "prompt_eval_count": self.prompt_eval_count, "eval_count": self.eval_count,
                "total_duration_seconds": self.total_duration_seconds,
                "load_duration_seconds": self.load_duration_seconds, "wall_seconds": round(self.wall_seconds, 3),
                "thinking_chars": len(self.thinking)}


@dataclass(frozen=True)
class ModelInfo:
    name: str
    digest: str
    expected_digest: str
    digest_verified: bool
    ollama_version: str
    size_bytes: int | None
    quantization: str | None
    parameter_size: str | None
    family: str | None
    capabilities: tuple

    def to_record(self) -> dict:
        return {"name": self.name, "digest": self.digest, "expected_digest": self.expected_digest or None,
                "digest_verified": self.digest_verified, "ollama_version": self.ollama_version,
                "size_bytes": self.size_bytes, "quantization": self.quantization,
                "parameter_size": self.parameter_size, "family": self.family,
                "capabilities": list(self.capabilities)}


def _reject_constant(token):
    raise ValueError(f"non-standard JSON constant {token}")


def _finite_float(token: str) -> float:
    value = float(token)
    if value in (float("inf"), float("-inf")):
        raise ValueError(f"number {token} overflows")
    return value


def strict_json(text: str):
    """json.loads without NaN/Infinity or overflowing numbers (1e999); deep nesting surfaces as ValueError."""
    try:
        return json.loads(text, parse_constant=_reject_constant, parse_float=_finite_float)
    except RecursionError as exc:
        raise ValueError("JSON nesting too deep") from exc


def _opt_str(value, field: str):
    if value is None or isinstance(value, str):
        return value
    raise OllamaProtocolError(f"{field} has type {type(value).__name__}, expected a string")


def _opt_int(value, field: str):
    if value is None or (isinstance(value, int) and not isinstance(value, bool)):
        return value
    raise OllamaProtocolError(f"{field} has type {type(value).__name__}, expected an integer")


def _short(text) -> str:
    text = str(text)
    return text if len(text) <= MAX_ERROR_TEXT else text[:MAX_ERROR_TEXT] + f"... [{len(text)} chars]"


class OllamaClient:
    def __init__(self, settings: QwenSettings, *, sleep=time.sleep):
        self.settings = settings.validate()
        parts = urlsplit(settings.url)
        self._https = parts.scheme == "https"
        self._host = parts.hostname
        self._port = parts.port or (443 if self._https else 80)
        self._sleep = sleep

    # ------------------------------------------------------------------ transport
    def _request(self, method: str, path: str, payload: dict | None = None, *, timeout: float | None = None) -> dict:
        deadline_s = timeout if timeout is not None else self.settings.timeout_seconds
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        cls = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
        conn = cls(self._host, self._port, timeout=min(self.settings.connect_timeout_seconds, deadline_s))
        start = time.monotonic()
        expired = threading.Event()

        def watchdog():
            expired.set()
            sock = conn.sock
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        timer = threading.Timer(deadline_s, watchdog)
        timer.daemon = True
        timer.start()
        chunks, size = [], 0
        try:
            try:
                conn.connect()
            except socket.timeout as exc:
                raise OllamaTimeout(f"connecting to Ollama at {self.settings.url} timed out") from exc
            except OSError as exc:
                if expired.is_set():
                    raise OllamaTimeout(f"Ollama {path} exceeded the {deadline_s}s deadline") from exc
                raise OllamaUnavailable(f"cannot connect to Ollama at {self.settings.url}: {exc}") from exc
            conn.sock.settimeout(deadline_s)
            headers = {"Accept": "application/json"}
            if body is not None:
                headers["Content-Type"] = "application/json"
            try:
                conn.request(method, path, body=body, headers=headers)
                resp = conn.getresponse()
                while size <= self.settings.max_response_bytes:
                    chunk = resp.read(min(65536, self.settings.max_response_bytes + 1 - size))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                if expired.is_set():
                    raise OllamaTimeout(f"Ollama {path} exceeded the {deadline_s}s deadline")
            except socket.timeout as exc:
                raise OllamaTimeout(f"Ollama {path} did not answer within {deadline_s}s") from exc
            except (ConnectionError, http.client.HTTPException, OSError) as exc:
                if expired.is_set():
                    raise OllamaTimeout(f"Ollama {path} exceeded the {deadline_s}s deadline") from exc
                raise OllamaUnavailable(f"connection to Ollama failed during {path}: {exc!r}") from exc
        finally:
            timer.cancel()
            conn.close()
        data = b"".join(chunks)
        if size > self.settings.max_response_bytes:
            raise OllamaProtocolError(f"Ollama {path} response exceeds {self.settings.max_response_bytes} bytes")
        text = data.decode("utf-8", errors="replace")
        if resp.status >= 300:
            message = text
            try:
                parsed_error = strict_json(text)
                if isinstance(parsed_error, dict) and isinstance(parsed_error.get("error"), str):
                    message = parsed_error["error"]
            except ValueError:
                pass
            if resp.status == 404 and "not found" in message.lower():
                raise ModelNotAvailable(f"Ollama: {_short(message)}", details={"status": 404, "path": path})
            raise OllamaHTTPError(f"Ollama {path} returned HTTP {resp.status}: {_short(message)}",
                                  status=resp.status, body=text[:2000])
        try:
            parsed = strict_json(text)
        except ValueError as exc:
            raise OllamaProtocolError(f"Ollama {path} returned invalid JSON: {_short(exc)}") from exc
        if not isinstance(parsed, dict):
            raise OllamaProtocolError(f"Ollama {path} returned {type(parsed).__name__}, not an object")
        parsed["_wall_seconds"] = time.monotonic() - start
        return parsed

    def _with_retries(self, func):
        """Retry transient failures (unavailable, timeout, HTTP 5xx) with linear backoff."""
        attempt = 0
        while True:
            try:
                return func()
            except OllamaError as exc:
                if not exc.transient or attempt >= self.settings.transport_retries:
                    exc.details.setdefault("transport_attempts", attempt + 1)
                    raise
                attempt += 1
                self._sleep(self.settings.retry_backoff_seconds * attempt)

    # ------------------------------------------------------------------ API
    def version(self) -> str:
        v = self._request("GET", "/api/version", timeout=self.settings.connect_timeout_seconds * 2).get("version")
        if not isinstance(v, str):
            raise OllamaProtocolError("/api/version did not return a version string")
        return v

    def list_models(self) -> list:
        models = self._request("GET", "/api/tags", timeout=30).get("models")
        if not isinstance(models, list) or not all(isinstance(m, dict) for m in models):
            raise OllamaProtocolError("/api/tags did not return a list of model objects")
        return models

    def show(self, model: str) -> dict:
        return self._request("POST", "/api/show", {"model": model}, timeout=60)

    def check_model(self) -> ModelInfo:
        """Verify that the configured model exists, and that its digest matches the pin.

        Never falls back to another model."""
        s = self.settings
        version = self.version()
        entry = next((m for m in self.list_models() if m.get("name") == s.model or m.get("model") == s.model), None)
        if entry is None:
            raise ModelNotAvailable(f"model {s.model!r} is not available in Ollama {version}. Run: ollama pull {s.model}",
                                    details={"model": s.model})
        digest = entry.get("digest")
        if not isinstance(digest, str):
            raise OllamaProtocolError("/api/tags model entry has no digest string")
        verified = bool(s.expected_digest) and digest == s.expected_digest
        if s.expected_digest and not verified:
            raise ModelDigestMismatch(
                f"model {s.model!r} has digest {digest}, expected {s.expected_digest}. Different weights would make "
                f"results incomparable; pull the pinned model or set NOVA_QWEN_EXPECTED_DIGEST deliberately.",
                details={"actual": digest, "expected": s.expected_digest})
        details = entry.get("details")
        details = details if isinstance(details, dict) else {}
        try:
            capabilities = self.show(s.model).get("capabilities")
        except OllamaHTTPError:
            capabilities = None
        if not isinstance(capabilities, list) or not all(isinstance(c, str) for c in capabilities):
            capabilities = []
        size = entry.get("size")
        return ModelInfo(name=s.model, digest=digest, expected_digest=s.expected_digest, digest_verified=verified,
                         ollama_version=version, size_bytes=size if isinstance(size, int) else None,
                         quantization=_opt_str(details.get("quantization_level"), "details.quantization_level"),
                         parameter_size=_opt_str(details.get("parameter_size"), "details.parameter_size"),
                         family=_opt_str(details.get("family"), "details.family"), capabilities=tuple(capabilities))

    def chat(self, messages: list, *, schema: dict | None, send_think: bool = True) -> ChatResult:
        s = self.settings
        payload = {"model": s.model, "messages": messages, "stream": False, "options": s.options(),
                   "keep_alive": s.keep_alive}
        if send_think:
            payload["think"] = s.think
        if schema is not None:
            payload["format"] = schema

        def call():
            return self._request("POST", "/api/chat", payload)

        body = self._with_retries(call)
        if body.get("error"):
            raise OllamaProtocolError(f"/api/chat returned an error field: {_short(body['error'])}")
        message = body.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaProtocolError("/api/chat response has no message.content string")
        model = _opt_str(body.get("model"), "model")
        if model is not None and model != s.model:
            raise OllamaProtocolError(f"/api/chat answered with model {model!r}, requested {s.model!r}")

        def seconds(key):
            v = body.get(key)
            return round(v / 1e9, 3) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

        return ChatResult(content=message["content"], thinking=_opt_str(message.get("thinking"), "thinking") or "",
                          done=body.get("done") is True, done_reason=_opt_str(body.get("done_reason"), "done_reason"),
                          prompt_eval_count=_opt_int(body.get("prompt_eval_count"), "prompt_eval_count"),
                          eval_count=_opt_int(body.get("eval_count"), "eval_count"),
                          total_duration_seconds=seconds("total_duration"),
                          load_duration_seconds=seconds("load_duration"), model=model,
                          wall_seconds=body.get("_wall_seconds", 0.0))
