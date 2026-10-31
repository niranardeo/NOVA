"""Typed errors for the NOVA Qwen consumer.

Every error carries a stable machine-readable ``code`` so that callers and tests can tell
exactly which check failed (a test asserting only "some ValueError" cannot tell a metadata
error from an evidence error).
"""
from __future__ import annotations


class NovaQwenError(Exception):
    """Base class for all errors raised by the Qwen consumer."""

    code = "NOVA_QWEN_ERROR"

    def __init__(self, message: str, *, code: str | None = None, details: dict | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": str(self), "details": self.details}


# ----------------------------------------------------------------------------- input
class AlertFileError(NovaQwenError):
    """The classifier alert export cannot be used. Nothing is sent to the model."""

    code = "ALERT_FILE_ERROR"


class AlertFileMissing(AlertFileError):
    code = "ALERT_FILE_MISSING"


class AlertFileUnreadable(AlertFileError):
    code = "ALERT_FILE_UNREADABLE"


class AlertFileInvalidJSON(AlertFileError):
    code = "ALERT_FILE_INVALID_JSON"


class UnsupportedSchemaVersion(AlertFileError):
    code = "UNSUPPORTED_SCHEMA_VERSION"


class AlertContractError(AlertFileError):
    """The file parses but violates the schema 1.1 alert contract."""

    code = "ALERT_CONTRACT_VIOLATION"


class IncompatibleExports(NovaQwenError):
    """Two exports cannot be correlated (different run, dataset or row positions)."""

    code = "INCOMPATIBLE_EXPORTS"


# ----------------------------------------------------------------------------- config
class ConfigError(NovaQwenError):
    code = "CONFIG_ERROR"


# ----------------------------------------------------------------------------- model
class OllamaError(NovaQwenError):
    """Base class for Ollama communication problems."""

    code = "OLLAMA_ERROR"
    transient = False


class OllamaUnavailable(OllamaError):
    code = "OLLAMA_UNAVAILABLE"
    transient = True


class OllamaTimeout(OllamaError):
    code = "OLLAMA_TIMEOUT"
    transient = True


class OllamaHTTPError(OllamaError):
    code = "OLLAMA_HTTP_ERROR"

    def __init__(self, message: str, *, status: int, body: str = "", **kw):
        super().__init__(message, **kw)
        self.status = status
        self.body = body
        # 5xx (e.g. "model runner has unexpectedly stopped") may succeed on retry.
        self.transient = status >= 500


class OllamaProtocolError(OllamaError):
    """The server answered, but not with the JSON structure the API documents."""

    code = "OLLAMA_PROTOCOL_ERROR"


class ModelNotAvailable(OllamaError):
    code = "MODEL_NOT_AVAILABLE"


class ModelDigestMismatch(OllamaError):
    code = "MODEL_DIGEST_MISMATCH"


class PromptBudgetExceeded(NovaQwenError):
    """The request could exceed the context window, so Ollama could silently truncate it."""

    code = "PROMPT_BUDGET_EXCEEDED"


# ----------------------------------------------------------------------------- output
class OutputError(NovaQwenError):
    code = "OUTPUT_ERROR"


class ResultsFileError(NovaQwenError):
    """A results file given as INPUT (verify, worksheet, correlate, resume) is missing or malformed."""

    code = "RESULTS_FILE_ERROR"


class ResumeMismatch(NovaQwenError):
    code = "RESUME_MISMATCH"


class ResultsVerificationError(NovaQwenError):
    code = "RESULTS_VERIFICATION_FAILED"
