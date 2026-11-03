"""One-command smoke test for the NOVA Qwen consumer. Prints PASS/FAIL per check; exit 0 only if all pass.
Uses only the bundled SYNTHETIC fixtures, so it needs no dataset and says nothing about classifier results.

Offline, no Ollama needed (the default):

    python scripts/qwen_smoke.py

Also runs the consumer against a real Ollama + qwen3.5:4b-q4_K_M:

    python scripts/qwen_smoke.py --live

Run from anywhere; paths are resolved from this file. Results are written under outputs/qwen_smoke/
(gitignored). The live mode never downloads a model; pull it first (see docs/qwen_integration.md).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nova_qwen.consumer import MAX_UNCONFIRMED_ALERTS  # noqa: E402
from src.nova_qwen.outcome import PASS, classifier_outcome, counts_line, overall_outcome  # noqa: E402

FIX = ROOT / "tests" / "qwen" / "fixtures"
OUT = ROOT / "outputs" / "qwen_smoke"
results: list = []


def limit_value(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from exc
    if not 1 <= value <= MAX_UNCONFIRMED_ALERTS:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_UNCONFIRMED_ALERTS}, got {value}")
    return value


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    return ok


def cli(*args, timeout=3600):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-m", "src.nova_qwen", *map(str, args)], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=timeout)


def offline() -> None:
    check("python version >= 3.11 (3.12/3.13 tested)", sys.version_info >= (3, 11), sys.version.split()[0])
    for name in ("decision_tree_alerts.json", "random_forest_alerts.json"):
        r = cli("validate-export", FIX / name)
        check(f"validate-export {name}", r.returncode == 0, r.stderr.strip()[:200])
    try:
        OUT.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        check(f"results directory {OUT} is writable", False, str(exc))
        return
    empty_out = OUT / "empty_offline_results.json"
    r = cli("explain", FIX / "decision_tree_alerts_empty.json", "--out", empty_out, "--overwrite",
            "--url", "http://127.0.0.1:9")  # nothing listens on port 9: proves no model contact
    doc = json.loads(empty_out.read_text()) if empty_out.exists() else {}
    check("empty export -> no model contact, zero results",
          r.returncode == 0 and doc.get("results") == [] and doc.get("model", {}).get("contacted") is False,
          r.stderr.strip()[:200])
    r = cli("check-dataset", "--validation", OUT / "absent" / "cicids2017_V3_test_RESERVED_15.csv")  # no such file exists
    check("reserved test split is refused by name, without being opened",
          r.returncode == 2 and "RESERVED_SPLIT_REFUSED" in r.stderr, r.stderr.strip()[-120:])
    r = subprocess.run([sys.executable, "scripts/qwen_e2e_from_notebook.py", "--skip-notebook", "--limit", "-1"], cwd=ROOT,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True, timeout=120)
    check("end-to-end script refuses a negative --limit before any inference",
          r.returncode == 2 and "argument --limit" in r.stderr, r.stderr.strip()[-120:])
    try:
        import pytest  # noqa: F401
    except ImportError:
        check("offline test suite (pytest)", False, "pytest missing: pip install -r requirements-qwen-dev.txt")
        return
    r = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "tests/qwen"], cwd=ROOT,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True, timeout=1800)
    summary = (r.stdout.strip().splitlines() or ["no output"])[-1]
    check("offline test suite tests/qwen", r.returncode == 0, summary)


def live(limit: int) -> None:
    r = cli("check-ollama", "--inference", timeout=600)
    info = json.loads(r.stdout)["model"] if r.returncode == 0 else {}
    check("Ollama reachable, pinned model present, digest verified, tiny inference ok", info.get("digest_verified") is True,
          (f"{info.get('name')} {str(info.get('digest'))[:12]}" if r.returncode == 0 else r.stderr.strip()[:300]))
    if r.returncode != 0:
        return
    try:
        OUT.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        check(f"results directory {OUT} is writable", False, str(exc))
        return
    outs, outcomes = {}, {}
    for det in ("decision_tree", "random_forest"):
        export = FIX / f"{det}_alerts.json"
        out = OUT / f"{det}_live_results.json"
        expected = min(limit, len(json.loads(export.read_text())["alerts"]))
        r = cli("explain", export, "--out", out, "--overwrite", "--limit", limit, "--detector", det)
        doc = json.loads(out.read_text()) if out.exists() else {}
        s = doc.get("summary", {})
        check(f"live explain {det} ({expected} alerts)", r.returncode == 0 and s.get("failed") == 0 and
              s.get("processed") == expected, json.dumps(s) if s else r.stderr.strip()[:300])
        check(f"{det}: accepted answers", s.get("explained", 0) + s.get("abstained", 0) >= 1,
              f"explained={s.get('explained')} abstained={s.get('abstained')} rejected={s.get('rejected')}")
        check(f"{det}: no unresolved rejected or failed answers", s.get("rejected") == 0 and s.get("failed") == 0,
              f"rejected={s.get('rejected')} failed={s.get('failed')}")
        r = cli("verify", out, "--export", export)
        check(f"verify {det} results against the export (ids, classes, evidence, re-validation)", r.returncode == 0,
              r.stdout.strip()[-300:] if r.returncode else "")
        report = json.loads(r.stdout) if r.stdout.strip().startswith("{") else None
        outcomes[det] = classifier_outcome(doc or None, report, len(json.loads(export.read_text())["alerts"]),
                                           selected=expected)
        print(f"      {det}: {outcomes[det]['verdict']} ({counts_line(outcomes[det]['counts'])})", flush=True)
        outs[det] = out
    overall = overall_outcome(outcomes)
    check("Qwen outcome across both classifiers is a clean pass (abstentions are reported separately)",
          overall["verdict"] == PASS, f"{overall['verdict']}: {counts_line(overall['totals'] or None)}")
    r = cli("correlate", "--dt-export", FIX / "decision_tree_alerts.json", "--rf-export",
            FIX / "random_forest_alerts.json", "--dt-results", outs["decision_tree"], "--rf-results",
            outs["random_forest"], "--out", OUT / "correlation.json")
    check("correlate DT/RF of the same run", r.returncode == 0, r.stderr.strip()[:200])
    r = cli("worksheet", outs["decision_tree"], outs["random_forest"], "--out", OUT / "review_worksheet.csv",
            "--overwrite")  # smoke outputs are disposable; real review sheets are protected without --overwrite
    check("human-review worksheet written", r.returncode == 0, str(OUT / "review_worksheet.csv"))


def main() -> int:
    global OUT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live", action="store_true", help="also test against a running Ollama")
    p.add_argument("--live-only", action="store_true", help="skip the offline part")
    p.add_argument("--limit", type=limit_value, default=2, help="alerts per classifier in live mode (default 2)")
    p.add_argument("--out-dir", type=Path, default=OUT, help="where results go (default outputs/qwen_smoke)")
    a = p.parse_args()
    OUT = a.out_dir.resolve()
    print(f"NOVA Qwen smoke test - repository {ROOT}")
    if not a.live_only:
        offline()
    if a.live or a.live_only:
        live(a.limit)
    failed = [n for n, ok in results if not ok]
    print(f"\n{'ALL PASS' if not failed else 'FAILED'}: {len(results) - len(failed)}/{len(results)} checks passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
