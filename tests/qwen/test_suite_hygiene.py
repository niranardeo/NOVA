"""Guards for the test suite itself.

A conftest hook that skips the opt-in live tests once skipped EVERY test in the session (pytest
passes all collected items to pytest_collection_modifyitems). This check makes sure the team's tests
and the offline Qwen tests still run when the live directory is collected too."""
import os
import re
import subprocess
import sys

from qwen_test_support import ROOT


def test_live_opt_in_does_not_skip_other_tests():
    env = {k: v for k, v in os.environ.items() if k != "NOVA_QWEN_LIVE"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q",
                        "tests/test_alert_adapter.py", "tests/qwen/test_features_prompt.py", "tests/qwen_live"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    last = r.stdout.strip().splitlines()[-1]
    assert r.returncode == 0, r.stdout[-2000:]
    passed = int(last.split(" passed")[0].split()[-1])
    skipped = int(last.split(" skipped")[0].split()[-1])
    collected = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--collect-only",
                                "tests/test_alert_adapter.py", "tests/qwen/test_features_prompt.py"],
                               cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    expected = int(collected.stdout.strip().splitlines()[-1].split()[0])  # "N tests collected"
    assert expected >= 12 + 20
    assert passed == expected and skipped == 11, last


def test_team_tests_still_discoverable_by_unittest():
    """The team runs `python -m unittest discover -s tests`; the Qwen tests must not break that."""
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=ROOT,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True, timeout=300)
    ran = re.search(r"^Ran (\d+) tests? in ", r.stderr, re.M)
    assert r.returncode == 0 and ran and int(ran.group(1)) >= 12 and "OK" in r.stderr, r.stderr[-1500:]
