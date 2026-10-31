"""Ollama client and settings, exercised over real sockets against the scripted FakeOllama."""
import json

import pytest

from qwen_test_support import DIGEST, MODEL, chat_body, make_settings
from src.nova_qwen.config import DEFAULT_EXPECTED_DIGEST, DEFAULT_MODEL, QwenSettings
from src.nova_qwen.errors import (ConfigError, ModelDigestMismatch, ModelNotAvailable, OllamaHTTPError,
                                  OllamaProtocolError, OllamaTimeout, OllamaUnavailable)
from src.nova_qwen.ollama import OllamaClient

MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
SCHEMA = {"type": "object"}


def client_for(url, **kw):
    sleeps = []
    c = OllamaClient(make_settings(url, **kw), sleep=sleeps.append)
    return c, sleeps


# --------------------------------------------------------------------------- settings
def test_defaults_are_the_agreed_deployment():
    s = QwenSettings()
    assert s.model == DEFAULT_MODEL == "qwen3.5:4b-q4_K_M"
    assert s.expected_digest == DEFAULT_EXPECTED_DIGEST
    assert (s.think, s.temperature, s.seed, s.num_ctx, s.num_predict) == (False, 0.0, 42, 8192, 512)
    assert s.url == "http://127.0.0.1:11434"
    assert s.options() == {"temperature": 0.0, "seed": 42, "num_ctx": 8192, "num_predict": 512}


@pytest.mark.parametrize("url, code", [
    ("http://10.1.2.3:11434", "CONFIG_REMOTE_HOST"), ("http://ollama.example.com", "CONFIG_REMOTE_HOST"),
    ("ftp://127.0.0.1", "CONFIG_URL"), ("http://user:pw@127.0.0.1:11434", "CONFIG_URL"),
    ("http://127.0.0.1:11434/api", "CONFIG_URL"), ("127.0.0.1:11434", "CONFIG_URL"),
])
def test_unsafe_urls_rejected(url, code):
    with pytest.raises(ConfigError) as e:
        QwenSettings(url=url).validate()
    assert e.value.code == code


@pytest.mark.parametrize("url", ["http://localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434",
                                 "https://127.0.0.1"])
def test_loopback_urls_accepted(url):
    QwenSettings(url=url).validate()


def test_remote_host_only_with_explicit_opt_in():
    QwenSettings(url="http://ollama:11434", allow_remote=True).validate()


@pytest.mark.parametrize("kw, code", [
    ({"expected_digest": "abc"}, "CONFIG_DIGEST"), ({"num_ctx": 512}, "CONFIG_NUM_CTX"),
    ({"num_predict": 8192}, "CONFIG_NUM_PREDICT"), ({"num_predict": 10}, "CONFIG_NUM_PREDICT"),
    ({"timeout_seconds": 0}, "CONFIG_TIMEOUT"), ({"max_attempts": 0}, "CONFIG_RETRIES"),
    ({"transport_retries": -1}, "CONFIG_RETRIES"), ({"temperature": -1}, "CONFIG_TEMPERATURE"),
    ({"model": "qwen 3.5"}, "CONFIG_MODEL"), ({"model": ""}, "CONFIG_MODEL"),
])
def test_invalid_settings(kw, code):
    with pytest.raises(ConfigError) as e:
        QwenSettings(**kw).validate()
    assert e.value.code == code


def test_environment_overrides():
    env = {"NOVA_OLLAMA_URL": "http://localhost:9999", "NOVA_QWEN_MODEL": "qwen3.5:4b", "NOVA_QWEN_NUM_CTX": "6000",
           "NOVA_QWEN_TIMEOUT_SECONDS": "30", "NOVA_QWEN_EXPECTED_DIGEST": DIGEST.upper(), "NOVA_QWEN_SEED": "7"}
    s = QwenSettings.from_env(env)
    assert (s.url, s.model, s.num_ctx, s.timeout_seconds, s.expected_digest, s.seed) == \
        ("http://localhost:9999", "qwen3.5:4b", 6000, 30.0, DIGEST, 7)
    assert QwenSettings.from_env(env, num_ctx=7000).num_ctx == 7000  # explicit argument wins
    assert QwenSettings.from_env({"NOVA_QWEN_EXPECTED_DIGEST": ""}).expected_digest == ""


def test_bad_numeric_environment_value():
    with pytest.raises(ConfigError) as e:
        QwenSettings.from_env({"NOVA_QWEN_NUM_CTX": "lots"})
    assert e.value.code == "CONFIG_ENV"


# --------------------------------------------------------------------------- model checks
def test_check_model_verifies_digest(fake_ollama):
    c, _ = client_for(fake_ollama.url)
    info = c.check_model()
    assert info.name == MODEL and info.digest == DIGEST and info.digest_verified
    assert info.ollama_version == "0.40.11" and "thinking" in info.capabilities
    assert info.quantization == "Q4_K_M" and info.parameter_size == "4.2B"


def test_model_missing_never_falls_back(fake_ollama):
    fake_ollama.model_entry = {"name": "llama3:8b", "model": "llama3:8b", "digest": "0" * 64}
    c, _ = client_for(fake_ollama.url)
    with pytest.raises(ModelNotAvailable) as e:
        c.check_model()
    assert e.value.code == "MODEL_NOT_AVAILABLE" and "ollama pull qwen3.5:4b-q4_K_M" in str(e.value)
    assert not fake_ollama.chat_requests()


def test_digest_mismatch_refused(fake_ollama):
    fake_ollama.model_entry = dict(fake_ollama.model_entry, digest="1" * 64)
    c, _ = client_for(fake_ollama.url)
    with pytest.raises(ModelDigestMismatch) as e:
        c.check_model()
    assert e.value.code == "MODEL_DIGEST_MISMATCH" and e.value.details["actual"] == "1" * 64


def test_digest_pinning_can_be_disabled_explicitly(fake_ollama):
    fake_ollama.model_entry = dict(fake_ollama.model_entry, digest="1" * 64)
    c, _ = client_for(fake_ollama.url, expected_digest="")
    info = c.check_model()
    assert info.digest == "1" * 64 and info.digest_verified is False


# --------------------------------------------------------------------------- chat requests
def test_chat_request_payload_and_result(fake_ollama):
    fake_ollama.chat_actions.append(("json", 200, chat_body('{"ok": true}', prompt_eval_count=321, eval_count=12)))
    c, _ = client_for(fake_ollama.url)
    r = c.chat(MESSAGES, schema=SCHEMA)
    body = fake_ollama.chat_requests()[0]
    assert body["model"] == MODEL and body["stream"] is False and body["think"] is False
    assert body["format"] == SCHEMA and body["messages"] == MESSAGES
    assert body["options"] == {"temperature": 0.0, "seed": 42, "num_ctx": 8192, "num_predict": 512}
    assert (r.content, r.done_reason, r.prompt_eval_count, r.eval_count) == ('{"ok": true}', "stop", 321, 12)


def test_think_parameter_can_be_omitted(fake_ollama):
    fake_ollama.chat_actions.append(("json", 200, chat_body('{"ok": true}')))
    c, _ = client_for(fake_ollama.url)
    c.chat(MESSAGES, schema=SCHEMA, send_think=False)
    assert "think" not in fake_ollama.chat_requests()[0]


def test_connection_refused_is_retried_then_typed(unused_port):
    c, sleeps = client_for(f"http://127.0.0.1:{unused_port}", transport_retries=2, retry_backoff_seconds=1.0)
    with pytest.raises(OllamaUnavailable) as e:
        c.chat(MESSAGES, schema=SCHEMA)
    assert e.value.code == "OLLAMA_UNAVAILABLE" and e.value.details["transport_attempts"] == 3
    assert sleeps == [1.0, 2.0]


def test_read_timeout(fake_ollama):
    fake_ollama.chat_actions += [("sleep", 1.5, ("json", 200, chat_body("{}")))] * 2
    c, _ = client_for(fake_ollama.url, timeout_seconds=0.5, transport_retries=1)
    with pytest.raises(OllamaTimeout) as e:
        c.chat(MESSAGES, schema=SCHEMA)
    assert e.value.code == "OLLAMA_TIMEOUT" and e.value.details["transport_attempts"] == 2


def test_dropped_connection(fake_ollama):
    fake_ollama.chat_actions += [("drop",)] * 3
    c, _ = client_for(fake_ollama.url, transport_retries=2)
    with pytest.raises(OllamaUnavailable):
        c.chat(MESSAGES, schema=SCHEMA)
    assert len(fake_ollama.chat_requests()) == 3


def test_server_error_retried_then_succeeds(fake_ollama):
    fake_ollama.chat_actions += [("json", 500, {"error": "model runner has unexpectedly stopped"}),
                                 ("json", 200, chat_body('{"ok": 1}'))]
    c, sleeps = client_for(fake_ollama.url)
    assert c.chat(MESSAGES, schema=SCHEMA).content == '{"ok": 1}'
    assert len(fake_ollama.chat_requests()) == 2 and len(sleeps) == 1


def test_client_error_not_retried(fake_ollama):
    fake_ollama.chat_actions += [("json", 400, {"error": "invalid format"})]
    c, sleeps = client_for(fake_ollama.url)
    with pytest.raises(OllamaHTTPError) as e:
        c.chat(MESSAGES, schema=SCHEMA)
    assert e.value.status == 400 and "invalid format" in str(e.value) and not sleeps
    assert len(fake_ollama.chat_requests()) == 1


def test_model_not_found_on_chat(fake_ollama):
    fake_ollama.chat_actions += [("json", 404, {"error": "model 'qwen3.5:4b-q4_K_M' not found"})]
    c, _ = client_for(fake_ollama.url)
    with pytest.raises(ModelNotAvailable):
        c.chat(MESSAGES, schema=SCHEMA)


def test_redirects_are_not_followed(fake_ollama):
    fake_ollama.chat_actions += [("raw", 307, b"", "text/plain", {"Location": "http://example.com/steal"})]
    c, _ = client_for(fake_ollama.url)
    with pytest.raises(OllamaHTTPError) as e:
        c.chat(MESSAGES, schema=SCHEMA)
    assert e.value.status == 307 and len(fake_ollama.requests) == 1


@pytest.mark.parametrize("action, code", [
    (("raw", 200, b"<html>proxy error</html>", "text/html"), "OLLAMA_PROTOCOL_ERROR"),
    (("json", 200, ["not", "an", "object"]), "OLLAMA_PROTOCOL_ERROR"),
    (("json", 200, {"done": True}), "OLLAMA_PROTOCOL_ERROR"),
    (("json", 200, {"message": {"content": None}}), "OLLAMA_PROTOCOL_ERROR"),
])
def test_protocol_errors(fake_ollama, action, code):
    fake_ollama.chat_actions.append(action)
    c, _ = client_for(fake_ollama.url)
    with pytest.raises(OllamaProtocolError) as e:
        c.chat(MESSAGES, schema=SCHEMA)
    assert e.value.code == code


def test_oversized_response_rejected(fake_ollama):
    fake_ollama.chat_actions.append(("json", 200, chat_body("x" * 5000)))
    c, _ = client_for(fake_ollama.url, max_response_bytes=1000)
    with pytest.raises(OllamaProtocolError) as e:
        c.chat(MESSAGES, schema=SCHEMA)
    assert "exceeds 1000 bytes" in str(e.value)


def test_proxy_environment_is_ignored(fake_ollama, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9")
    c, _ = client_for(fake_ollama.url)
    assert c.version() == "0.40.11"


def test_truncated_generation_metadata_is_reported(fake_ollama):
    fake_ollama.chat_actions.append(("json", 200, chat_body('{"observations": ["Fl', done_reason="length")))
    c, _ = client_for(fake_ollama.url)
    r = c.chat(MESSAGES, schema=SCHEMA)
    assert r.done_reason == "length" and r.content == '{"observations": ["Fl' and r.done is True
