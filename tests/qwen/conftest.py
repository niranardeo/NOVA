"""pytest fixtures for the offline Qwen tests. Helpers live in qwen_test_support.py (a unique module
name, so imports cannot collide with tests/qwen_live/conftest.py)."""
import shutil
import socket

import pytest

from qwen_test_support import FIXTURES, FakeOllama, make_settings


@pytest.fixture
def fake_ollama():
    fake = FakeOllama()
    yield fake
    fake.close()


@pytest.fixture
def settings(fake_ollama):
    return make_settings(fake_ollama.url)


@pytest.fixture
def unused_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def work(tmp_path):
    """A temp dir holding copies of the fixtures (tests may modify them)."""
    for f in FIXTURES.glob("*.json"):
        shutil.copy2(f, tmp_path / f.name)
    return tmp_path
