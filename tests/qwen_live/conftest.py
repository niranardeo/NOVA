"""Live tests against a REAL Ollama with qwen3.5:4b-q4_K_M. Opt-in only:

    NOVA_QWEN_LIVE=1 python -m pytest -p no:cacheprovider tests/qwen_live -v

They take several minutes on CPU (about 20-30 s per alert) and never download a model.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "tests" / "qwen" / "fixtures"


HERE = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config, items):
    """Skip ONLY the tests in this directory unless opted in. (This hook receives every collected item
    in the session, so it must not mark tests outside tests/qwen_live/.)"""
    if os.environ.get("NOVA_QWEN_LIVE") == "1":
        return
    skip = pytest.mark.skip(reason="live Ollama tests are opt-in: set NOVA_QWEN_LIVE=1")
    for item in items:
        if HERE in Path(str(item.fspath)).resolve().parents:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def live_settings():
    from src.nova_qwen.config import QwenSettings
    return QwenSettings.from_env()


@pytest.fixture(scope="session")
def live_client(live_settings):
    from src.nova_qwen.ollama import OllamaClient
    return OllamaClient(live_settings)
