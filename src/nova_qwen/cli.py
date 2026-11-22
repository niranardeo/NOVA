"""Command-line interface.  Run from the repository root:  python -m src.nova_qwen <command> --help

Exit codes: 0 finished (for explain this does not mean every answer was accepted: read the summary, or use --strict);
1 verification or a dataset/alignment check failed, an alert failed, or (with --strict) an answer was rejected;
2 invalid input, arguments or resume request, a missing dataset file, a refused test split, or a selection above the
limit; 3 Ollama or model problem (incl. a run stopped because
Ollama became unreachable); 4 output problem; 130 interrupted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .alerts import load_export
from .config import QwenSettings
from .errors import NovaQwenError, OllamaError, OutputError
from .output import check_output_path, load_results, write_json_atomic, write_text_atomic
from .reserved import refuse_reserved

EXIT_OK, EXIT_FAILED, EXIT_INPUT, EXIT_OLLAMA, EXIT_OUTPUT, EXIT_INTERRUPTED = 0, 1, 2, 3, 4, 130
DEFAULT_LIMIT = 10


def _settings(args) -> QwenSettings:
    return QwenSettings.from_env(
        url=getattr(args, "url", None), model=getattr(args, "model", None),
        expected_digest=getattr(args, "expected_digest", None),
        timeout_seconds=getattr(args, "timeout", None), num_ctx=getattr(args, "num_ctx", None),
        num_predict=getattr(args, "num_predict", None), seed=getattr(args, "seed", None),
        max_attempts=getattr(args, "max_attempts", None), transport_retries=getattr(args, "retries", None),
        allow_remote=True if getattr(args, "allow_remote_ollama", False) else None)


def _add_model_args(p):
    p.add_argument("--url", help="Ollama base URL (default $NOVA_OLLAMA_URL or http://127.0.0.1:11434)")
    p.add_argument("--model", help="model tag (default $NOVA_QWEN_MODEL or qwen3.5:4b-q4_K_M)")
    p.add_argument("--expected-digest", help="required model digest; '' disables pinning (not recommended)")
    p.add_argument("--allow-remote-ollama", action="store_true", help="permit a non-loopback Ollama host")
    p.add_argument("--timeout", type=float, help="overall time limit per model request in seconds (default 180)")


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def cmd_check_ollama(args) -> int:
    from .ollama import OllamaClient
    from .response import parse_content
    settings = _settings(args)
    client = OllamaClient(settings)
    info = client.check_model()
    report = {"ollama_url": settings.url, "model": info.to_record(), "inference_test": None}
    if args.inference:
        messages = [{"role": "system", "content": "Respond with JSON only."},
                    {"role": "user", "content": 'Reply with {"ready": true}.'}]
        schema = {"type": "object", "properties": {"ready": {"type": "boolean"}}, "required": ["ready"]}
        chat = client.chat(messages, schema=schema, send_think="thinking" in info.capabilities or not info.capabilities)
        parsed, _ = parse_content(chat.content)
        ok = chat.done_reason == "stop" and isinstance(parsed, dict) and parsed.get("ready") is True
        report["inference_test"] = {"content": chat.content[:500], **chat.metadata(), "ok": ok}
        _print(report)
        return EXIT_OK if ok else EXIT_OLLAMA
    _print(report)
    return EXIT_OK


def cmd_validate_export(args) -> int:
    export = load_export(args.export, expected_detector=args.detector)
    classes = {}
    for a in export.alerts:
        classes[a.predicted_class] = classes.get(a.predicted_class, 0) + 1
    _print({"valid": True, **export.source_record(), "predicted_classes": classes})
    return EXIT_OK


def cmd_explain(args) -> int:
    from .consumer import MAX_UNCONFIRMED_ALERTS, run
    settings = _settings(args)
    export = load_export(args.export, expected_detector=args.detector)
    out = check_output_path(args.out, [args.export], overwrite=args.overwrite, resume=args.resume)
    explicit = args.all or args.alert_id or args.limit is not None or args.offset
    from_previous = bool(args.resume and not explicit and out.exists())
    limit = None if args.all else (args.limit if args.limit is not None else DEFAULT_LIMIT)
    max_alerts = None if args.allow_large_run else MAX_UNCONFIRMED_ALERTS
    provenance = {"dataset_version": args.dataset_version, "verified": False} if args.dataset_version else None

    def progress(i, n, r):
        if not args.quiet:
            print(f"[{i}/{n}] {r['alert_id']} ({r['predicted_class']}): {r['status']}"
                  + (f" {r['error']['code']}" if r.get("error") else ""), file=sys.stderr)

    doc = run(export, settings, out, limit=limit, offset=args.offset, alert_ids=args.alert_id,
              resume=args.resume, selection_from_previous=from_previous, progress=progress, max_alerts=max_alerts,
              provenance=provenance)
    _print({"output": str(out), "run_status": doc["run_status"], "summary": doc["summary"],
            "selection": doc["selection"], "resumed_from": doc.get("resumed_from"),
            "model": {k: doc["model"].get(k) for k in ("name", "digest", "digest_verified", "contacted")}})
    s = doc["summary"]
    if s["failed"] or (args.strict and s["rejected"]):
        return EXIT_FAILED
    return EXIT_OK


def _write_report(out, report: dict, marker: str) -> None:
    if out:
        out = Path(out)
        _guard_report_path(out, [], lambda head: f'"report": "{marker}"' in head, overwrite=True)
        write_json_atomic(out, report)


def cmd_check_dataset(args) -> int:
    from .datasets import check_dataset, require_version_spelling
    report = check_dataset(train=args.train, validation=args.validation,
                           version=require_version_spelling(args.dataset_version),
                           expected_train_sha256=args.expected_train_sha256,
                           expected_validation_sha256=args.expected_validation_sha256)
    _print(report)  # before writing, so a problem with --out cannot lose the result of a long scan
    _write_report(args.out, report, "nova-qwen-dataset-check")
    return EXIT_OK if report["passed"] else EXIT_FAILED


def cmd_check_alignment(args) -> int:
    from .datasets import V3, V3_VALIDATION_ROWS, check_alignment, require_version_spelling
    version = require_version_spelling(args.dataset_version)
    export = load_export(args.export, expected_detector=args.detector)
    report = {**check_alignment(export, args.validation, expected_rows=V3_VALIDATION_ROWS if version == V3 else None,
                                require_alerts=version == V3),
              "dataset_version": version}
    _print(report)
    _write_report(args.out, report, "nova-qwen-alignment-check")
    return EXIT_OK if report["passed"] else EXIT_FAILED


def cmd_verify(args) -> int:
    from .verify import verify_results
    report = verify_results(load_results(args.results), load_export(args.export),
                            allow_incomplete=args.allow_incomplete)
    _print(report)
    return EXIT_OK if report["passed"] else EXIT_FAILED


def _guard_report_path(out: Path, inputs: list, marker_ok, overwrite: bool) -> None:
    """Refuse to overwrite inputs, unrelated files, or (without --overwrite) an earlier report."""
    try:
        out.exists()
    except OSError as exc:
        raise OutputError(f"cannot reach {out}: {exc}", code="OUTPUT_WRITE_FAILED") from exc
    if any(Path(p).resolve() == out.resolve() for p in inputs):
        raise OutputError(f"{out} is one of the inputs", code="OUTPUT_IS_INPUT")
    refuse_reserved(out)  # also for a new file: no tool here could read or replace it afterwards
    if out.exists():
        if out.is_dir():
            raise OutputError(f"{out} is a directory", code="OUTPUT_IS_DIRECTORY")
        try:
            with out.open(encoding="utf-8") as fh:
                head = fh.read(4096)
        except (OSError, UnicodeDecodeError):
            head = ""
        if not marker_ok(head):
            raise OutputError(f"refusing to overwrite {out}: it was not written by this tool", code="OUTPUT_NOT_OURS")
        if not overwrite:
            raise OutputError(f"{out} already exists (it may contain review work); use --overwrite to replace it",
                              code="OUTPUT_EXISTS")


def cmd_correlate(args) -> int:
    from .correlate import correlate
    dt = load_export(args.dt_export, expected_detector="decision_tree")
    rf = load_export(args.rf_export, expected_detector="random_forest")
    report = correlate(dt, rf, load_results(args.dt_results) if args.dt_results else None,
                       load_results(args.rf_results) if args.rf_results else None)
    if args.out:
        out = Path(args.out)
        inputs = [p for p in (args.dt_export, args.rf_export, args.dt_results, args.rf_results) if p]
        _guard_report_path(out, inputs, lambda head: '"report": "nova-qwen-correlation"' in head, overwrite=True)
        write_json_atomic(out, report)
    _print({"compatibility": report["compatibility"], "counts": report["counts"], "warnings": report["warnings"],
            "disagreements": report["disagreements"][:20]})
    return EXIT_OK


def cmd_worksheet(args) -> int:
    from .worksheet import RUBRIC, is_worksheet_header, rows_from_results, to_csv
    out = Path(args.out)
    rubric = out.with_name(out.stem + ".rubric.txt")
    _guard_report_path(out, args.results, is_worksheet_header,
                       overwrite=args.overwrite)
    _guard_report_path(rubric, args.results, lambda head: head.startswith("Scoring guide"), overwrite=True)
    rows = rows_from_results([load_results(p) for p in args.results])
    write_text_atomic(out, to_csv(rows))
    write_text_atomic(rubric, RUBRIC + "\n")
    _print({"worksheet": str(out), "rubric": str(rubric), "rows": len(rows)})
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.nova_qwen",
                                     description="NOVA Qwen consumer for classifier alert exports (schema 1.1).")
    parser.add_argument("--version", action="version", version=f"nova_qwen {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check-ollama", help="verify Ollama, the pinned model and its digest")
    _add_model_args(p)
    p.add_argument("--inference", action="store_true", help="also run one tiny inference request")
    p.set_defaults(func=cmd_check_ollama)

    p = sub.add_parser("validate-export", help="validate a schema 1.1 alert export (no model call)")
    p.add_argument("export")
    p.add_argument("--detector", choices=["decision_tree", "random_forest"])
    p.set_defaults(func=cmd_validate_export)

    p = sub.add_parser("explain", help="explain alerts from one export with Qwen and write a results file")
    p.add_argument("export", help="decision_tree_alerts.json or random_forest_alerts.json (schema 1.1)")
    p.add_argument("--out", required=True, help="results file to write (e.g. outputs/qwen/decision_tree_explanations.json)")
    p.add_argument("--detector", choices=["decision_tree", "random_forest"], help="require this detector")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--limit", type=int, help=f"maximum alerts to explain (default {DEFAULT_LIMIT} for a new run)")
    group.add_argument("--all", action="store_true",
                       help="explain every alert (about 25 s per alert on CPU; more than 25 needs --allow-large-run)")
    group.add_argument("--alert-id", action="append", help="explain this alert id (repeatable)")
    p.add_argument("--offset", type=int, default=0, help="skip this many alerts before --limit (not with --alert-id)")
    p.add_argument("--resume", action="store_true",
                   help="continue an existing results file for the same export, selection and settings "
                        "(without selection flags, the previous selection is used)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing results file")
    p.add_argument("--allow-large-run", action="store_true",
                   help="permit more than 25 alerts in this run (a full export would take weeks on CPU)")
    p.add_argument("--dataset-version", help="record this dataset version in the results file as declared, unverified "
                                             "(use check-dataset to verify files)")
    p.add_argument("--strict", action="store_true", help="exit 1 if any answer was rejected")
    p.add_argument("--num-ctx", type=int)
    p.add_argument("--num-predict", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--max-attempts", type=int, help="model answers per alert incl. corrective re-asks (default 2)")
    p.add_argument("--retries", type=int, help="transport retries per request (default 2)")
    p.add_argument("--quiet", action="store_true")
    _add_model_args(p)
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("verify", help="check a results file against its source export")
    p.add_argument("results")
    p.add_argument("--export", required=True)
    p.add_argument("--allow-incomplete", action="store_true", help="accept an interrupted/in-progress file")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("correlate", help="compare DT and RF exports (and results) of the same run")
    p.add_argument("--dt-export", required=True)
    p.add_argument("--rf-export", required=True)
    p.add_argument("--dt-results")
    p.add_argument("--rf-results")
    p.add_argument("--out")
    p.set_defaults(func=cmd_correlate)

    p = sub.add_parser("check-dataset", help="identify the training/validation CSVs (never reads a test split)")
    p.add_argument("--validation", required=True)
    p.add_argument("--train", help="optional: the Qwen stage itself reads only the validation file")
    p.add_argument("--dataset-version", default="V3",
                   help="V3 (default) compares rows, columns and class counts with Kevin's notebooks; any other "
                        "value runs the general checks only and is never reported as V3")
    p.add_argument("--expected-train-sha256")
    p.add_argument("--expected-validation-sha256")
    p.add_argument("--out", help="also write the report to this file")
    p.set_defaults(func=cmd_check_dataset)

    p = sub.add_parser("check-alignment",
                       help="check that an export's record_ids and evidence match the validation CSV rows")
    p.add_argument("export")
    p.add_argument("--validation", required=True)
    p.add_argument("--detector", choices=["decision_tree", "random_forest"])
    p.add_argument("--dataset-version", default="V3", help="V3 (default) also requires V3's validation row count")
    p.add_argument("--out", help="also write the report to this file")
    p.set_defaults(func=cmd_check_alignment)

    p = sub.add_parser("worksheet", help="build a human-review CSV from results files")
    p.add_argument("results", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--overwrite", action="store_true", help="replace an existing worksheet (loses review work)")
    p.set_defaults(func=cmd_worksheet)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("interrupted; partial results (if any) were saved with run_status=interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except OllamaError as exc:
        print(json.dumps({"error": exc.to_dict()}, indent=2), file=sys.stderr)
        return EXIT_OLLAMA
    except OutputError as exc:
        print(json.dumps({"error": exc.to_dict()}, indent=2), file=sys.stderr)
        return EXIT_OUTPUT
    except NovaQwenError as exc:  # input, contract, config, results-file and resume problems
        print(json.dumps({"error": exc.to_dict()}, indent=2), file=sys.stderr)
        return EXIT_INPUT
    except OSError as exc:  # an input path that cannot be reached: Path.exists and is_file raise on EACCES, ENAMETOOLONG
        print(json.dumps({"error": {"code": "FILE_ERROR", "message": str(exc)}}, indent=2), file=sys.stderr)
        return EXIT_INPUT
