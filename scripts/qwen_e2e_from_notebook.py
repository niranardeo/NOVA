"""True end-to-end run: CICIDS2017 processed features -> Decision Tree / Random Forest (Jenn's notebook 03,
unmodified) -> schema 1.1 alert exports (src/alert_adapter.py) -> Qwen consumer -> validated explanations.

    python scripts/qwen_e2e_from_notebook.py --limit 5
    python scripts/qwen_e2e_from_notebook.py --skip-notebook --limit 5   # reuse existing outputs/*_alerts.json
    # Machines with < 12 GB free memory (notebook ~5 GB + model ~5 GB): two steps
    python scripts/qwen_e2e_from_notebook.py --notebook-only            # with Ollama stopped
    python scripts/qwen_e2e_from_notebook.py --skip-notebook --limit 5  # after starting Ollama

Prerequisites: pip install -r requirements-qwen-e2e.txt; Ollama running with qwen3.5:4b-q4_K_M; the APPROVED
files data/processed/training_dataset_CIC_70.csv and data/processed/validation_dataset_CIC_15.csv (the
notebook's own paths; data/ is gitignored). This script reads no dataset itself: it only checks that those two
files exist and hashes them, and notebook 03 reads only those two files. Keep the reserved test split out of
data/processed.

Writes: outputs/decision_tree_alerts.json, outputs/random_forest_alerts.json (by the notebook),
outputs/qwen_e2e/{executed notebook, *_explanations.json, correlation.json, review_worksheet.csv, summary.json}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NOTEBOOK = ROOT / "notebooks" / "03_baseline_model_development_Oct_06_2026_Jennifer.ipynb"
DATA = ROOT / "data" / "processed"
APPROVED = ("training_dataset_CIC_70.csv", "validation_dataset_CIC_15.csv")
OUT = ROOT / "outputs" / "qwen_e2e"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_notebook() -> dict:
    missing = [n for n in APPROVED if not (DATA / n).is_file()]
    if missing:
        raise SystemExit(f"missing approved data files in {DATA}: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
           "--ExecutePreprocessor.timeout=3600", "--ExecutePreprocessor.kernel_name=python3",
           "--output-dir", str(OUT), "--output", "03_executed_by_e2e.ipynb", str(NOTEBOOK)]
    print("running notebook 03 (unmodified) ...", flush=True)
    r = subprocess.run(cmd, cwd=NOTEBOOK.parent, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"notebook execution failed:\n{r.stderr[-3000:]}")
    return {"seconds": round(time.time() - start, 1),
            "data_sha256": {n: sha256(DATA / n) for n in APPROVED}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=5, help="alerts per classifier sent to Qwen (default 5)")
    p.add_argument("--alert-id", action="append", default=[], help="also explain this alert id (repeatable)")
    p.add_argument("--skip-notebook", action="store_true", help="use existing outputs/*_alerts.json")
    p.add_argument("--notebook-only", action="store_true",
                   help="only run the notebook (low-memory machines: stop Ollama first, then rerun with --skip-notebook)")
    a = p.parse_args()
    if a.notebook_only:
        info = run_notebook()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "notebook_run.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(json.dumps(info, indent=2))
        return 0

    from src.nova_qwen.alerts import load_export
    from src.nova_qwen.config import QwenSettings
    from src.nova_qwen.consumer import run
    from src.nova_qwen.correlate import correlate
    from src.nova_qwen.output import utc_now, write_json_atomic
    from src.nova_qwen.verify import verify_results
    from src.nova_qwen.worksheet import RUBRIC, rows_from_results, to_csv

    settings = QwenSettings.from_env()
    if not a.skip_notebook:
        # Fail fast: check Ollama and the pinned model BEFORE spending minutes on the notebook.
        from src.nova_qwen.ollama import OllamaClient
        OllamaClient(settings).check_model()
    previous = OUT / "notebook_run.json"
    notebook = (json.loads(previous.read_text()) if previous.is_file() else None) if a.skip_notebook else run_notebook()
    summary = {"notebook": notebook, "classifiers": {}}
    exports = {d: ROOT / "outputs" / f"{d}_alerts.json" for d in ("decision_tree", "random_forest")}
    for path in exports.values():
        if not path.is_file():
            raise SystemExit(f"alert export not found (run without --skip-notebook): {path}")
    loaded = {d: load_export(p, expected_detector=d) for d, p in exports.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    docs = {}
    ok = True
    for det, export in loaded.items():
        ids = [i for i in a.alert_id if i.startswith(export.alerts[0].alert_id[:3])] if export.alerts else []
        chosen = list(dict.fromkeys([x.alert_id for x in export.alerts[:a.limit]] + ids))
        out = OUT / f"{det}_explanations.json"
        print(f"{det}: {len(export.alerts)} alerts in export (run_id {export.run_id}); explaining {len(chosen)}",
              flush=True)
        doc = run(export, settings, out, alert_ids=chosen or None, limit=a.limit,
                  progress=lambda i, n, r: print(f"  [{i}/{n}] {r['alert_id']} {r['predicted_class']}: {r['status']}",
                                                 flush=True))
        report = verify_results(doc, export)
        docs[det] = doc
        summary["classifiers"][det] = {"export": export.source_record(), "results_file": out.name,
                                       "summary": doc["summary"], "verify": report, "model": doc["model"].get("digest")}
        ok &= report["passed"] and doc["summary"]["failed"] == 0
    corr = correlate(loaded["decision_tree"], loaded["random_forest"], docs["decision_tree"], docs["random_forest"])
    write_json_atomic(OUT / "correlation.json", corr)
    worksheet = OUT / "review_worksheet.csv"
    if worksheet.exists():  # never overwrite a sheet that may already hold human scores
        worksheet = OUT / f"review_worksheet_{utc_now().replace(':', '').replace('.', '')}.csv"
    worksheet.write_text(to_csv(rows_from_results(list(docs.values()))), encoding="utf-8")
    rubric = worksheet.with_name(worksheet.stem + ".rubric.txt")
    rubric.write_text(RUBRIC + "\n", encoding="utf-8")
    summary["review_worksheet"] = worksheet.name
    summary["correlation_counts"] = corr["counts"]
    write_json_atomic(OUT / "summary.json", summary)
    print(json.dumps({d: {"summary": v["summary"], "verified": v["verify"]["passed"]}
                      for d, v in summary["classifiers"].items()} | {"correlation": corr["counts"]}, indent=2))
    print("E2E", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def entry() -> int:
    from src.nova_qwen.errors import NovaQwenError
    try:
        return main()
    except NovaQwenError as exc:
        print(json.dumps({"error": exc.to_dict()}, indent=2), file=sys.stderr)
        print("E2E FAIL")
        return 3 if exc.code.startswith(("OLLAMA", "MODEL", "RUN_ABORTED")) else 2


if __name__ == "__main__":
    sys.exit(entry())
