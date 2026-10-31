"""Model and inference settings for the NOVA Qwen consumer.

Defaults are the team's agreed deployment: Qwen3.5-4B (Q4_K_M) served by a local Ollama.
Every value can be overridden with an environment variable or a CLI flag, and every value
actually used is recorded in each results file.
"""
from __future__ import annotations

import ipaddress
import math
import os
from dataclasses import asdict, dataclass, field, replace
from urllib.parse import urlsplit

from .errors import ConfigError

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3.5:4b-q4_K_M"
# Digest of qwen3.5:4b-q4_K_M as published on the Ollama library and verified locally
# (see docs/qwen_integration.md). A different digest means different weights.
DEFAULT_EXPECTED_DIGEST = "d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04"

ENV = {
    "url": "NOVA_OLLAMA_URL",
    "model": "NOVA_QWEN_MODEL",
    "expected_digest": "NOVA_QWEN_EXPECTED_DIGEST",
    "timeout_seconds": "NOVA_QWEN_TIMEOUT_SECONDS",
    "num_ctx": "NOVA_QWEN_NUM_CTX",
    "num_predict": "NOVA_QWEN_NUM_PREDICT",
    "seed": "NOVA_QWEN_SEED",
    "allow_remote": "NOVA_QWEN_ALLOW_REMOTE_OLLAMA",
}


@dataclass(frozen=True)
class QwenSettings:
    """All settings that influence a model request. Recorded verbatim in every results file."""

    url: str = DEFAULT_OLLAMA_URL
    model: str = DEFAULT_MODEL
    # Empty string disables digest pinning (not recommended; recorded in the output).
    expected_digest: str = DEFAULT_EXPECTED_DIGEST
    think: bool = False
    temperature: float = 0.0
    seed: int = 42
    # 8192 leaves room for one corrective re-ask (first request ~2.7 KB + previous answer + feedback)
    # under the byte-based no-truncation guarantee in prompt.check_budget.
    num_ctx: int = 8192
    num_predict: int = 512
    # Overall wall-clock limit for one HTTP request (also the socket read timeout).
    timeout_seconds: float = 180.0
    connect_timeout_seconds: float = 5.0
    keep_alive: str = "10m"
    # Transport retries (connection refused, timeout, HTTP 5xx) per model request.
    transport_retries: int = 2
    retry_backoff_seconds: float = 2.0
    # Model answers per alert: the first answer plus corrective re-asks after validation errors.
    max_attempts: int = 2
    # Stop the run (resumable) after this many consecutive alerts failed for transport reasons.
    abort_after_consecutive_failures: int = 3
    allow_remote: bool = False
    max_response_bytes: int = 1_000_000
    extra: dict = field(default_factory=dict)

    def options(self) -> dict:
        """The Ollama ``options`` object sent with every chat request."""
        return {
            "temperature": self.temperature,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }

    def to_record(self) -> dict:
        record = asdict(self)
        record.pop("extra")
        return record

    def validate(self) -> "QwenSettings":
        _check_url(self.url, self.allow_remote)
        if not self.model or any(c.isspace() for c in self.model):
            raise ConfigError(f"invalid model name {self.model!r}", code="CONFIG_MODEL")
        if self.expected_digest and not _is_hex_digest(self.expected_digest):
            raise ConfigError("expected digest must be 64 hex characters or empty", code="CONFIG_DIGEST")
        if self.num_ctx < 1024:
            raise ConfigError("num_ctx must be at least 1024", code="CONFIG_NUM_CTX")
        if not 64 <= self.num_predict < self.num_ctx:
            raise ConfigError("num_predict must be >= 64 and < num_ctx", code="CONFIG_NUM_PREDICT")
        for name in ("timeout_seconds", "connect_timeout_seconds", "retry_backoff_seconds", "temperature"):
            if not math.isfinite(getattr(self, name)):
                raise ConfigError(f"{name} must be a finite number", code="CONFIG_NOT_FINITE")
        if not (0 < self.timeout_seconds <= 86400 and 0 < self.connect_timeout_seconds <= 600):
            raise ConfigError("timeout must be in (0, 86400] s and connect timeout in (0, 600] s", code="CONFIG_TIMEOUT")
        if self.max_attempts < 1 or self.transport_retries < 0 or self.abort_after_consecutive_failures < 1:
            raise ConfigError("max_attempts >= 1, transport_retries >= 0 and abort_after_consecutive_failures >= 1 "
                              "required", code="CONFIG_RETRIES")
        if self.temperature < 0:
            raise ConfigError("temperature must be >= 0", code="CONFIG_TEMPERATURE")
        return self

    @classmethod
    def from_env(cls, environ: dict | None = None, **overrides) -> "QwenSettings":
        env = os.environ if environ is None else environ
        values: dict = {}
        try:
            if env.get(ENV["url"]):
                values["url"] = env[ENV["url"]]
            if env.get(ENV["model"]):
                values["model"] = env[ENV["model"]]
            if ENV["expected_digest"] in env:
                values["expected_digest"] = env[ENV["expected_digest"]].strip().lower()
            if env.get(ENV["timeout_seconds"]):
                values["timeout_seconds"] = float(env[ENV["timeout_seconds"]])
            if env.get(ENV["num_ctx"]):
                values["num_ctx"] = int(env[ENV["num_ctx"]])
            if env.get(ENV["num_predict"]):
                values["num_predict"] = int(env[ENV["num_predict"]])
            if env.get(ENV["seed"]):
                values["seed"] = int(env[ENV["seed"]])
            if env.get(ENV["allow_remote"]):
                values["allow_remote"] = env[ENV["allow_remote"]].strip().lower() in ("1", "true", "yes")
        except ValueError as exc:
            raise ConfigError(f"invalid numeric environment setting: {exc}", code="CONFIG_ENV") from exc
        values.update({k: v for k, v in overrides.items() if v is not None})
        if isinstance(values.get("expected_digest"), str):
            values["expected_digest"] = values["expected_digest"].strip().lower()
        return replace(cls(), **values).validate()


def _is_hex_digest(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _check_url(url: str, allow_remote: bool) -> None:
    """Only plain http(s) URLs without credentials; loopback unless explicitly allowed."""
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise ConfigError(f"invalid Ollama URL {url!r}", code="CONFIG_URL") from exc
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ConfigError(f"Ollama URL must be http(s)://host[:port], got {url!r}", code="CONFIG_URL")
    if parts.username or parts.password:
        raise ConfigError("credentials in the Ollama URL are not allowed", code="CONFIG_URL")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise ConfigError("Ollama URL must not contain a path, query or fragment", code="CONFIG_URL")
    try:
        port = parts.port
    except ValueError as exc:
        raise ConfigError(f"invalid port in Ollama URL {url!r}: {exc}", code="CONFIG_URL") from exc
    if port == 0:
        raise ConfigError("port 0 is not a valid Ollama port", code="CONFIG_URL")
    host = parts.hostname
    if allow_remote:
        return
    if host == "localhost":
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise ConfigError(
        f"Ollama host {host!r} is not loopback. Alert evidence would leave this machine; set "
        f"{ENV['allow_remote']}=1 or pass --allow-remote-ollama if that is intended.",
        code="CONFIG_REMOTE_HOST",
    )
