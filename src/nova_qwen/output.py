"""Versioned results document and safe result-file writing."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .errors import OutputError, ResultsFileError

OUTPUT_SCHEMA = "nova-qwen-explanations"
# 0.x = provisional: the response fields and statuses still need team approval (docs/qwen_integration.md).
OUTPUT_SCHEMA_VERSION = "0.1.0-provisional"
STATUSES = ("explained", "abstained", "rejected", "failed")
RUN_STATUSES = ("in_progress", "complete", "interrupted")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def is_results_document(doc) -> bool:
    return isinstance(doc, dict) and doc.get("output_schema") == OUTPUT_SCHEMA


RESULT_KEYS = ("alert_id", "record_id", "record_position", "detector", "run_id", "dataset", "schema_version",
               "predicted_class", "prediction_status", "evidence", "evidence_normalized", "data_quality_notes",
               "status", "response", "validation", "attempts", "error")


def load_results(path) -> dict:
    """Read a results file given as input. Problems are input errors (exit 2), never tracebacks."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResultsFileError(f"results file not found: {path}", code="RESULTS_MISSING") from exc
    except (OSError, ValueError, RecursionError) as exc:
        raise ResultsFileError(f"cannot read results file {path}: {exc}", code="RESULTS_UNREADABLE") from exc
    if not is_results_document(doc):
        raise ResultsFileError(f"{path} is not a {OUTPUT_SCHEMA} document", code="RESULTS_NOT_OURS")
    problems = []
    for key, typ in (("source", dict), ("selection", dict), ("model", dict), ("results", list)):
        if not isinstance(doc.get(key), typ):
            problems.append(f"{key} must be a {typ.__name__}")
    for i, r in enumerate(doc.get("results") if isinstance(doc.get("results"), list) else []):
        if not isinstance(r, dict) or any(k not in r for k in RESULT_KEYS):
            problems.append(f"results[{i}] is missing required keys")
        elif not isinstance(r["evidence"], dict) or not isinstance(r["evidence_normalized"], list):
            problems.append(f"results[{i}] has malformed evidence")
    if problems:
        raise ResultsFileError(f"{path} is a malformed results file: {problems[:5]}", code="RESULTS_MALFORMED")
    return doc


def check_output_path(output_path, input_paths, *, overwrite: bool, resume: bool) -> Path:
    """Refuse to clobber inputs or unrelated files."""
    out = Path(output_path)
    resolved = out.resolve()
    for p in input_paths:
        if Path(p).resolve() == resolved:
            raise OutputError("the output path is the same as an input alert export", code="OUTPUT_IS_INPUT")
    if out.exists():
        if out.is_dir():
            raise OutputError(f"output path is a directory: {out}", code="OUTPUT_IS_DIRECTORY")
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
        if not is_results_document(existing):
            raise OutputError(f"refusing to overwrite {out}: it is not a {OUTPUT_SCHEMA} results file",
                              code="OUTPUT_NOT_OURS")
        if not (overwrite or resume):
            raise OutputError(f"{out} already exists; use --resume to continue it or --overwrite to replace it",
                              code="OUTPUT_EXISTS")
    return out


def _current_umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def write_json_atomic(path, doc: dict) -> None:
    """Serialize fully, write a same-directory temp file, fsync, then os.replace.

    The final file gets normal permissions (0o666 minus the umask), so a consumer running as another
    user can read it, unlike a raw mkstemp file which is always 0o600."""
    write_text_atomic(path, json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_text_atomic(path, text: str) -> None:
    path = Path(path)
    try:
        _write(path, text)
    except OSError as exc:
        raise OutputError(f"cannot write {path}: {exc}", code="OUTPUT_WRITE_FAILED") from exc


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name[:100]}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o666 & ~_current_umask())
        os.replace(tmp, path)
        tmp = None
        _fsync_directory(path.parent)
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass


def _fsync_directory(directory: Path) -> None:
    """Make the rename durable where the OS supports opening directories (POSIX)."""
    if os.name != "posix":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
