"""End-to-end run for the V3 workflow: classifier alert exports -> Qwen consumer -> verified explanations.

The usual route is to consume exports that already exist (Jenn's, or from scripts/qwen_export_v3_alerts.py):

    python scripts/qwen_e2e_from_notebook.py --skip-notebook --data-dir /path/to/folder/with/the/two/V3/files

Or execute notebook 03 as committed first. It reads the paths written in the notebook, so those paths must exist
(/media/sf_NOVA_Data today). It needs about 6 GB of memory with Ollama stopped, or about 12 GB together with the model,
and it is not the sampler Jenn's report describes:

Notebook only (stop Ollama first if memory is tight):

    python scripts/qwen_e2e_from_notebook.py --notebook-only

Notebook, then Qwen:

    python scripts/qwen_e2e_from_notebook.py

Before any inference this checks the dataset identity, that every alert matches its validation row, and that the
requested alert count is within the limit. V3 is the default; it never falls back to another dataset. Any other
dataset must be named with --dataset-version and is never reported as V3. The reserved test split is refused by
name, and is never opened.

Exit codes: 0 PASS; 1 the run finished without a clean pass, or the notebook or a file write failed (see the last line);
2 not run: invalid or missing input; 3 Ollama or model problem; 130 interrupted.

Writes to --out-dir (default outputs/qwen_e2e): *_explanations.json, correlation.json, review_worksheet.csv and its
.rubric.txt, summary.json, and in notebook mode the executed notebook and notebook_run.json. A run refused before the
notebook or inference changes none of them. A run that starts (in notebook mode, when the notebook starts) replaces
summary.json at once, so an earlier verdict cannot outlive it. In notebook mode, a refusal after the notebook has
finished still leaves the new exports and the notebook's own files, because that run did happen. Keep datasets out of
the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nova_qwen.alerts import load_export  # noqa: E402
from src.nova_qwen.config import QwenSettings  # noqa: E402
from src.nova_qwen.consumer import MAX_UNCONFIRMED_ALERTS, TRANSPORT_CODES, RunAborted, run  # noqa: E402
from src.nova_qwen.correlate import check_compatible, correlate  # noqa: E402
from src.nova_qwen.datasets import (V3, V3_TRAIN_FILE, V3_VALIDATION_FILE, V3_VALIDATION_ROWS, check_alignment,  # noqa: E402
                                    check_dataset, is_reserved_name, refuse_reserved, require_version_spelling)
from src.nova_qwen.errors import (DatasetError, DatasetMissing, LargeRunNotAuthorized, NovaQwenError,  # noqa: E402
                                  OllamaError, OutputError, ReservedSplitRefused, ResumeMismatch)
from src.nova_qwen.ollama import OllamaClient  # noqa: E402
from src.nova_qwen.outcome import FAILED, INCOMPLETE, PASS, classifier_outcome, counts_line, overall_outcome  # noqa: E402
from src.nova_qwen.output import check_output_path, utc_now, write_json_atomic, write_text_atomic  # noqa: E402
from src.nova_qwen.selection import add_requested, plan_first, plan_representative  # noqa: E402
from src.nova_qwen.verify import verify_results  # noqa: E402
from src.nova_qwen.worksheet import RUBRIC, rows_from_results, to_csv  # noqa: E402

NOTEBOOK = ROOT / "notebooks" / "03_baseline_model_development_Oct_06_2026_Jennifer.ipynb"
DEFAULT_DATA_DIR = ROOT / "data" / "processed"
DEFAULT_EXPORTS = ROOT / "outputs"
DEFAULT_OUT = ROOT / "outputs" / "qwen_e2e"
DETECTORS = ("decision_tree", "random_forest")
PREFIX = {"decision_tree": "dt-", "random_forest": "rf-"}
RECOMMENDED_DATASET_FIELD = "CICIDS2017-V3-validation"  # the value in Jenn's V3 validation exports
NAMES_V3_RE = re.compile(r"(?i)(?<![a-z])v3(?![a-z0-9])")  # "CICIDS2017V3" names V3 too
# Why the two can differ is documented in docs/qwen_v3_integration_results.md.
NOTEBOOK_CAVEAT = ("Notebook executed as committed (train_test_split, n_jobs=2). Jenn's report describes "
                   "class-stratified reservoir sampling and n_jobs=1, so this is not her reported experiment.")
UNIT_ASSUMPTION = ("Flow Duration is treated as microseconds, Total Fwd Packets as a packet count and Flow Bytes/s as "
                   "bytes per second. Requires confirmation against V3 preprocessing.")
SCOPE_NOTE = ("Qwen explains alerts the classifiers already raised. It does not change predictions, does not see "
              "normal-classified rows, and cannot recover a classifier false negative.")
STARTED = False  # set once anything in --out-dir has been replaced, so an interrupt can say which side of that it fell on
INTERRUPTED_NOTE = ("interrupted. A results file being written at that moment has run_status=interrupted; other files in "
                    "--out-dir may be from an earlier run unless summary.json names them.")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=positive_int, default=6,
                   help=f"alerts per classifier sent to Qwen (default 6, above {MAX_UNCONFIRMED_ALERTS} needs "
                        "--allow-large-run)")
    p.add_argument("--allow-large-run", action="store_true",
                   help=f"permit more than {MAX_UNCONFIRMED_ALERTS} alerts per classifier (about 25 s each on CPU)")
    p.add_argument("--select", choices=["representative", "first"], default="representative",
                   help="representative covers classes, a DT/RF class disagreement and unusual durations (default); "
                        "first takes the first alerts in row order")
    p.add_argument("--alert-id", action="append", default=[], help="also explain this alert id (repeatable)")
    p.add_argument("--skip-notebook", action="store_true", help="use existing alert exports")
    p.add_argument("--notebook-only", action="store_true",
                   help="only run the notebook (if memory is tight, stop Ollama first and rerun with --skip-notebook)")
    p.add_argument("--notebook", type=Path, default=NOTEBOOK, help="notebook to execute (default: Jenn's notebook 03)")
    p.add_argument("--dataset-version", default=V3,
                   help="V3 (default) must match Kevin's notebooks. Anything else, such as SYNTHETIC-DEMO or "
                        "V2-LEGACY-REGRESSION, runs general checks only and is never reported as V3")
    p.add_argument("--data-dir", type=Path, help=f"folder holding {V3_TRAIN_FILE} and {V3_VALIDATION_FILE} "
                                                 "(default data/processed)")
    p.add_argument("--train", type=Path, help="training CSV (optional with --skip-notebook)")
    p.add_argument("--validation", type=Path, help="validation CSV the exports were made from")
    p.add_argument("--expected-train-sha256")
    p.add_argument("--expected-validation-sha256",
                   help="the published hash of the validation file; without it a V3 run counts as a profile match only")
    p.add_argument("--no-dataset-check", action="store_true",
                   help="with --skip-notebook: explain exports without a dataset file. The run is then reported as "
                        "dataset identity and row alignment NOT verified")
    p.add_argument("--exports-dir", type=Path, default=DEFAULT_EXPORTS, help="folder holding the *_alerts.json exports")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="folder for this run's results")
    p.add_argument("--resume", action="store_true",
                   help="with --skip-notebook: keep good results in --out-dir and redo only the rest (same exports, "
                        "selection and settings)")
    return p


def validate_args(a) -> None:
    """Everything here runs before any dataset read, notebook or model call."""
    if a.skip_notebook and a.notebook_only:
        raise NovaQwenError("--skip-notebook and --notebook-only cannot be combined", code="BAD_ARGUMENTS")
    if a.resume and not a.skip_notebook:
        raise NovaQwenError("--resume needs --skip-notebook", code="BAD_ARGUMENTS")
    if a.no_dataset_check and not a.skip_notebook:
        raise NovaQwenError("--no-dataset-check needs --skip-notebook", code="BAD_ARGUMENTS")
    if a.no_dataset_check and (a.validation or a.train or a.data_dir):
        raise NovaQwenError("--no-dataset-check cannot be combined with dataset paths", code="BAD_ARGUMENTS")
    if a.no_dataset_check and (a.expected_train_sha256 or a.expected_validation_sha256):
        raise NovaQwenError("--no-dataset-check reads no file, so an expected hash cannot be compared",
                            code="BAD_ARGUMENTS")
    require_version_spelling(a.dataset_version)
    if a.data_dir and (a.train or a.validation):
        raise NovaQwenError("use --data-dir, or --train/--validation, not both", code="BAD_ARGUMENTS")
    if a.dataset_version != V3 and a.data_dir:
        raise NovaQwenError("--data-dir assumes V3 file names; with another dataset use --train/--validation",
                            code="BAD_ARGUMENTS")
    if not a.skip_notebook and a.exports_dir.resolve() != DEFAULT_EXPORTS.resolve():
        raise NovaQwenError("the notebook always writes to outputs/; --exports-dir needs --skip-notebook",
                            code="BAD_ARGUMENTS")
    for path in (a.train, a.validation):
        if path is not None:
            refuse_reserved(path)
    wrong = [i for i in a.alert_id if not i.startswith(tuple(PREFIX.values()))]
    if wrong:
        raise NovaQwenError(f"--alert-id must start with dt- or rf-: {wrong}", code="BAD_SELECTION")
    if not a.allow_large_run:
        for det, prefix in PREFIX.items():
            total = a.limit + len({i for i in a.alert_id if i.startswith(prefix)})
            if total > MAX_UNCONFIRMED_ALERTS:
                raise LargeRunNotAuthorized(
                    f"up to {total} {det} alerts requested; the limit without --allow-large-run is "
                    f"{MAX_UNCONFIRMED_ALERTS} per classifier (about 25 s per alert on CPU)",
                    details={"requested": total, "limit": MAX_UNCONFIRMED_ALERTS})
    try:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=a.out_dir):
            pass
    except OSError as exc:
        raise NovaQwenError(f"--out-dir {a.out_dir} cannot be written: {exc}", code="BAD_ARGUMENTS") from exc


def notebook_data_files(notebook: Path) -> tuple:
    """The (training, validation) CSVs a notebook will really read, from its own source. Stops on anything else."""
    refuse_reserved(notebook)
    try:
        nb = json.loads(notebook.read_text(encoding="utf-8"))
        source = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    except FileNotFoundError as exc:
        raise DatasetMissing(f"notebook not found: {notebook}") from exc
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DatasetError(f"cannot read notebook {notebook}: {exc}") from exc
    dirs = set(re.findall(r"""DATA_DIR\s*=\s*Path\(\s*["']([^"']+)["']\s*\)""", source))
    names = set(re.findall(r"""["']([^"'\s]+\.csv)["']""", source))
    reserved = sorted(n for n in names if is_reserved_name(n))
    if reserved:
        raise ReservedSplitRefused(f"{notebook.name} names a test or reserved file: {reserved}")
    train = [n for n in names if "train" in n.lower()]
    validation = [n for n in names if "valid" in n.lower()]
    if len(dirs) != 1 or len(train) != 1 or len(validation) != 1 or len(names) != 2:
        raise DatasetError(f"cannot tell which files {notebook.name} reads: DATA_DIR values {sorted(dirs)}, "
                           f"CSV names {sorted(names)}. Refusing to record dataset hashes I cannot attribute.")
    base = Path(next(iter(dirs)))
    base = base if base.is_absolute() else (notebook.parent / base).resolve()
    return base / train[0], base / validation[0]


def resolve_dataset(a) -> tuple:
    """(training path or None, validation path or None) for this run. Never substitutes another dataset."""
    if a.no_dataset_check:
        return None, None
    if not a.skip_notebook:
        nb_train, nb_validation = notebook_data_files(a.notebook)
        if a.dataset_version == V3 and (nb_train.name, nb_validation.name) != (V3_TRAIN_FILE, V3_VALIDATION_FILE):
            raise DatasetError(f"{a.notebook.name} reads {nb_train.name} and {nb_validation.name}, not the V3 files")
        given = (a.data_dir / nb_train.name, a.data_dir / nb_validation.name) if a.data_dir else (a.train, a.validation)
        for label, have, reads in (("training", given[0], nb_train), ("validation", given[1], nb_validation)):
            if have is not None and Path(have).resolve() != reads.resolve():
                raise DatasetError(f"--{label} is {have}, but {a.notebook.name} reads {reads}. The notebook's own path "
                                   f"decides what it reads, so these must match.")
        for path in (nb_train, nb_validation):
            if not path.is_file():
                raise DatasetMissing(f"{a.notebook.name} reads {path}, which does not exist on this machine. Provide "
                                     f"it at that path (see docs/qwen_testing_guide.md), or use exports with "
                                     f"--skip-notebook.")
        return nb_train, nb_validation
    if a.dataset_version == V3:
        folder = a.data_dir or DEFAULT_DATA_DIR
        validation = a.validation or folder / V3_VALIDATION_FILE
        train = a.train or (None if a.validation else folder / V3_TRAIN_FILE)
    else:
        validation, train = a.validation, a.train
    if validation is None or not Path(validation).is_file():
        where = "Pass --data-dir or --validation" if a.dataset_version == V3 else "Pass --validation"
        raise DatasetMissing(
            f"{a.dataset_version} validation CSV not found: {validation}. {where}. No other dataset is substituted. "
            f"To explain exports without it, pass --no-dataset-check (the run is then reported as dataset identity "
            f"and row alignment NOT verified).")
    if a.train is not None and not Path(a.train).is_file():
        raise DatasetMissing(f"--train {a.train} not found. Omit --train to run without a training file.")
    return (train if train and Path(train).is_file() else None), validation


def run_notebook(a, train: Path, validation: Path, out_dir: Path) -> dict:
    exports = {d: DEFAULT_EXPORTS / f"{d}_alerts.json" for d in DETECTORS}
    before = {d: p.stat().st_mtime_ns if p.is_file() else None for d, p in exports.items()}
    start = time.time()
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
           "--ExecutePreprocessor.timeout=3600", "--ExecutePreprocessor.kernel_name=python3",
           "--output-dir", str(out_dir.resolve()), "--output", "03_executed_by_e2e.ipynb", str(a.notebook.resolve())]
    print("running the notebook as committed ...", flush=True)
    r = subprocess.run(cmd, cwd=a.notebook.resolve().parent, capture_output=True, text=True)
    if r.returncode != 0:
        raise NovaQwenError(f"notebook execution failed:\n{r.stderr[-3000:]}", code="NOTEBOOK_FAILED")
    not_written = [str(p) for d, p in exports.items() if not p.is_file() or p.stat().st_mtime_ns == before[d]]
    if not_written:
        raise NovaQwenError(f"{a.notebook.name} finished but did not write {not_written}. Refusing to attribute "
                            f"existing files to this run.", code="NOTEBOOK_FAILED")
    info = {"seconds": round(time.time() - start, 1), "dataset_version": a.dataset_version,
            "notebook": {"file_name": a.notebook.name, "sha256": sha256(a.notebook)},
            "data_files": {p.name: {"path": str(p), "sha256": sha256(p)} for p in (train, validation)},
            "exports_sha256": {d: sha256(p) for d, p in exports.items()},
            "note": NOTEBOOK_CAVEAT if a.notebook.resolve() == NOTEBOOK.resolve() else "Notebook executed as given."}
    write_json_atomic(out_dir / "notebook_run.json", info)
    return info


def previous_notebook_run(out_dir: Path, exports: dict) -> tuple:
    """A stored notebook record counts only if it produced exactly these exports."""
    path = out_dir / "notebook_run.json"
    if not path.is_file():
        return None, "no notebook record in this --out-dir, so the origin of these exports is not recorded"
    refuse_reserved(path)
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None, "notebook_run.json is unreadable and was ignored"
    recorded = info.get("exports_sha256") if isinstance(info, dict) else None
    if recorded != {d: sha256(p) for d, p in exports.items()}:
        return None, "notebook_run.json describes different exports and was ignored"
    return info, "this notebook run produced these exports"


def dataset_field_note(version: str, fields: dict, files_checked: bool) -> str:
    values = sorted(set(fields.values()))
    if version != V3:
        return f"The exports say dataset {values}."
    source = ("V3 identity is recorded here, from the files, not from the label." if files_checked else
              "No dataset file was read, so V3 is only declared for this run, not established.")
    if all(NAMES_V3_RE.search(v) for v in values):
        return f"The exports say dataset {values}. {source}"
    return f"The exports say dataset {values}, which does not name V3. {source} Jenn's V3 exports use " \
           f"{RECOMMENDED_DATASET_FIELD!r}."


def refuse_contradicting_labels(version: str, fields: dict) -> None:
    for field in fields.values():
        named = re.search(r"(?i)(?<![a-z])v(\d+)(?![a-z0-9])", field)
        if version != V3 and "v3" in field.lower():
            raise DatasetError(f"--dataset-version {version} but the exports say {fields}. Refusing to mix them.")
        if version == V3 and named and named.group(1) != "3":
            raise DatasetError(f"--dataset-version V3 but the exports say {fields}. Refusing to mix them.")


def file_state(path: Path) -> str | None:
    try:
        return sha256(path)
    except OSError:
        return None


def read_if_changed(path: Path, before: str | None) -> dict | None:
    """The results file, only if this attempt changed it: it may otherwise still hold an earlier run's answers.
    Compares contents, not timestamps, so a clock that moved between or during runs cannot misattribute it."""
    if file_state(path) == before:
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def lost_the_model(doc: dict | None, settings: QwenSettings) -> bool:
    """The consumer's abort rule, for a run too short for it to fire: its last alerts could not reach the model."""
    results = (doc or {}).get("results") or []
    last = results[-settings.abort_after_consecutive_failures:]
    return bool(last) and all(r.get("status") == "failed" and (r.get("error") or {}).get("code") in TRANSPORT_CODES
                              for r in last)


def guard_own_files(out_dir: Path, exports: dict, resume: bool, notebook: bool) -> None:
    """Refuse an --out-dir holding files this tool did not write, before anything in it is replaced."""
    try:
        for det in DETECTORS:
            check_output_path(out_dir / f"{det}_explanations.json", list(exports.values()), overwrite=True, resume=resume)
    except OutputError as exc:  # nothing has run yet, so this is an invalid request, not a failed run
        raise NovaQwenError(str(exc), code=exc.code) from exc
    owned = {"correlation.json": lambda d: d.get("report") == "nova-qwen-correlation",
             "summary.json": lambda d: d.get("report") == "nova-qwen-e2e-summary"}
    if notebook:  # only a notebook run replaces these two
        owned["notebook_run.json"] = lambda d: "exports_sha256" in d
        owned["03_executed_by_e2e.ipynb"] = lambda d: "cells" in d and "nbformat" in d
    for name, is_ours in owned.items():
        path = out_dir / name
        if path.exists():
            refuse_reserved(path)
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
                ours = isinstance(doc, dict) and is_ours(doc)
            except (OSError, ValueError):
                ours = False
            if not ours:
                raise NovaQwenError(f"refusing to overwrite {path}: it was not written by this tool",
                                    code="OUTPUT_NOT_OURS")


def notebook_tools_installed() -> bool:
    return importlib.util.find_spec("nbconvert") is not None and importlib.util.find_spec("ipykernel") is not None


def retire_earlier_run(out_dir: Path) -> None:
    """The notebook replaces the exports, so an earlier verdict stops describing them as soon as it starts."""
    global STARTED
    STARTED = True
    write_json_atomic(out_dir / "summary.json",
                      {"report": "nova-qwen-e2e-summary", "generated_utc": utc_now(),
                       "outcome": {"verdict": INCOMPLETE, "classifiers": {}, "totals": {}, "success": False},
                       "note": "The notebook was started, which can replace the alert exports. No Qwen run has finished "
                               "since; any results file in this folder is from an earlier run."})
    (out_dir / "correlation.json").unlink(missing_ok=True)


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    validate_args(a)
    settings = QwenSettings.from_env()
    if a.dataset_version != V3:
        print(f"NOTE: dataset version {a.dataset_version!r}. This run is NOT a V3 result.", flush=True)
    out_dir = a.out_dir
    train, validation = resolve_dataset(a)
    exports = {d: a.exports_dir / f"{d}_alerts.json" for d in DETECTORS}
    guard_own_files(out_dir, exports, a.resume, notebook=not a.skip_notebook)
    if not a.skip_notebook and not notebook_tools_installed():
        raise NovaQwenError("running the notebook needs nbconvert and ipykernel: python -m pip install -r "
                            "requirements-qwen-e2e.txt. Nothing was changed.", code="NOTEBOOK_TOOLS_MISSING")
    expected_rows = V3_VALIDATION_ROWS if a.dataset_version == V3 else None
    check = None
    if validation is not None:
        check = check_dataset(train=train, validation=validation, version=a.dataset_version,
                              expected_train_sha256=a.expected_train_sha256,
                              expected_validation_sha256=a.expected_validation_sha256)
        if not check["passed"]:
            raise DatasetError(f"dataset check failed for version {a.dataset_version}: {check['problems']}",
                               details={"problems": check["problems"]})
    if a.notebook_only:
        retire_earlier_run(out_dir)
        info = run_notebook(a, train, validation, out_dir)
        print(json.dumps(info, indent=2))
        next_step = ["python", "scripts/qwen_e2e_from_notebook.py", "--skip-notebook", "--dataset-version", a.dataset_version,
                     "--train", str(train), "--validation", str(validation), "--out-dir", str(out_dir.resolve())]
        for flag, value in (("--expected-train-sha256", a.expected_train_sha256),
                            ("--expected-validation-sha256", a.expected_validation_sha256)):
            if value:
                next_step += [flag, value]
        print(f"Next, from the repository root: {shlex.join(next_step)}")
        print("NOTEBOOK DONE: no Qwen inference was run.")
        return 0

    if not a.skip_notebook:  # fail fast: check the model BEFORE spending minutes on the notebook
        OllamaClient(settings).check_model()
        retire_earlier_run(out_dir)
    notebook = run_notebook(a, train, validation, out_dir) if not a.skip_notebook else None

    for path in exports.values():
        if not path.is_file():
            raise DatasetMissing(f"alert export not found: {path}. Jenn's classifier run or "
                                 f"scripts/qwen_export_v3_alerts.py creates these.", code="EXPORT_MISSING")
    loaded = {d: load_export(p, expected_detector=d) for d, p in exports.items()}
    if notebook is None:
        notebook, notebook_note = previous_notebook_run(out_dir, exports)
    else:
        notebook_note = "this run executed the notebook"
    fields = {d: e.dataset for d, e in loaded.items()}
    refuse_contradicting_labels(a.dataset_version, fields)
    check_compatible(loaded["decision_tree"], loaded["random_forest"])

    alignment = {}
    if validation is not None:
        for det, export in loaded.items():
            alignment[det] = check_alignment(export, validation, expected_rows=expected_rows)
            if not alignment[det]["passed"]:
                raise DatasetError(f"{det} export does not match {Path(validation).name}: {alignment[det]['problems']}",
                                   details={"examples": alignment[det]["examples"]})

    plan = (plan_representative if a.select == "representative" else plan_first)(
        loaded["decision_tree"], loaded["random_forest"], a.limit)
    for det, export in loaded.items():
        requested = [i for i in a.alert_id if i.startswith(PREFIX[det])]
        known = {x.alert_id for x in export.alerts}
        missing = [i for i in requested if i not in known]
        if missing:
            raise NovaQwenError(f"alert ids not found in {export.file_name}: {missing}", code="ALERT_ID_NOT_FOUND")
        add_requested(plan, export, loaded[DETECTORS[1 - DETECTORS.index(det)]], requested)

    compared = bool(alignment) and all(r["alerts_checked"] for r in alignment.values())
    profile_ok = bool(check and check["v3_profile_matched"] and compared)
    v3_verified = bool(profile_ok and check["v3_validation_verified"])
    identity = {"declared_version": a.dataset_version, "real_v3_verified": v3_verified, "v3_profile_matched": profile_ok,
                "verification_basis": {"method": "file name, rows, columns, class counts and evidence columns, compared "
                                                 "with Kevin's notebooks; verified also needs a matching published hash",
                                       "validation_hash_pinned": bool(check and check["hash_pinned"]["validation"]),
                                       "train_hash_pinned": bool(check and check["hash_pinned"]["train"]),
                                       "training_file_checked": bool(check and check["train"]),
                                       "alerts_compared_with_csv": {d: r["alerts_checked"] for d, r in alignment.items()}},
                "file_check": check, "alignment": alignment or "NOT CHECKED",
                "export_dataset_field": fields,
                "field_note": dataset_field_note(a.dataset_version, fields, check is not None)}
    provenance = {"dataset_version": a.dataset_version, "verified": v3_verified, "v3_profile_matched": profile_ok,
                  "validation_sha256": check["validation"]["sha256"] if check else None}
    paths = {"cwd": os.getcwd(), "train": os.path.abspath(train) if train else None,
             "validation": os.path.abspath(validation) if validation else None,
             "exports_dir": os.path.abspath(a.exports_dir), "out_dir": os.path.abspath(out_dir)}

    # With --skip-notebook, a problem up to the first summary write leaves the earlier results and verdict together.
    if any(e.alerts for e in loaded.values()):
        OllamaClient(settings).check_model()
    max_alerts = None if a.allow_large_run else MAX_UNCONFIRMED_ALERTS
    requests = {d: {"alert_ids": plan[d]["alert_ids"] or None, "limit": a.limit if not plan[d]["alert_ids"] else None}
                for d in DETECTORS}
    if a.resume:  # the consumer's refusals fire for BOTH classifiers here, before anything changes
        for det, export in loaded.items():
            try:
                run(export, settings, out_dir / f"{det}_explanations.json", **requests[det], max_alerts=max_alerts,
                    provenance=provenance, resume=True, check_only=True)
            except ResumeMismatch as exc:
                reason = str(exc).replace(" Run --resume without selection flags to continue the previous one.", "")
                raise ResumeMismatch(f"{det}: {reason} Nothing was sent and nothing changed. Rerun with the original "
                                     f"selection, or if that file is left over from an older run, move it out of "
                                     f"--out-dir and rerun with --resume. Without --resume the run starts over.") from exc
    started = utc_now()
    global STARTED
    STARTED = True
    base_summary = {"report": "nova-qwen-e2e-summary", "generated_utc": started, "dataset": identity, "paths": paths,
                    "unit_assumption": UNIT_ASSUMPTION, "notebook": notebook, "notebook_note": notebook_note,
                    "selection": plan, "limits": {"limit_per_classifier": a.limit, "large_run_allowed": a.allow_large_run},
                    "scope_note": SCOPE_NOTE}
    write_json_atomic(out_dir / "summary.json",
                      {**base_summary, "outcome": {"verdict": INCOMPLETE, "classifiers": {}, "totals": {}, "success": False},
                       "note": "This run started and has not reached its final summary."})
    (out_dir / "correlation.json").unlink(missing_ok=True)  # it describes results about to be replaced

    docs, verifies = {}, {}
    stopped = None  # why the loop ended early
    interrupted = False
    for det, export in loaded.items():
        chosen = plan[det]["alert_ids"]
        print(f"{det}: {len(export.alerts)} alerts in export (run_id {export.run_id}); sending {len(chosen)} to Qwen",
              flush=True)
        out = out_dir / f"{det}_explanations.json"
        before = file_state(out)
        try:
            docs[det] = run(export, settings, out, **requests[det], max_alerts=max_alerts, provenance=provenance,
                            resume=a.resume,
                            progress=lambda i, n, r: print(f"  [{i}/{n}] {r['alert_id']} {r['predicted_class']}: "
                                                           f"{r['status']}", flush=True))
            verifies[det] = verify_results(docs[det], export)
        except (OllamaError, KeyboardInterrupt) as exc:  # includes RunAborted; this classifier may hold partial results
            stopped = exc
            interrupted = isinstance(exc, KeyboardInterrupt)
            partial = read_if_changed(out, before)
            if partial is not None:
                docs[det] = partial
                verifies[det] = verify_results(partial, export, allow_incomplete=True)
            break

    outcomes = {d: classifier_outcome(docs.get(d), verifies.get(d), len(loaded[d].alerts),
                                      selected=len(plan[d]["alert_ids"])) for d in DETECTORS}
    outcome = overall_outcome(outcomes)
    if stopped is not None and outcome["verdict"] != FAILED:  # every result may be good, but no correlation or worksheet
        outcome.update(verdict=INCOMPLETE, success=False)
    summary = {**base_summary, "generated_utc": utc_now(),
               "classifiers": {d: {"export": loaded[d].source_record(),
                                   "results_file": f"{d}_explanations.json" if d in docs else None,
                                   "outcome": outcomes[d], "summary": (docs.get(d) or {}).get("summary"),
                                   "verify": verifies.get(d), "model": ((docs.get(d) or {}).get("model") or {}).get("digest")}
                               for d in DETECTORS},
               "outcome": outcome, "correlation_counts": None, "review_worksheet": None}
    if stopped is None:
        corr = correlate(loaded["decision_tree"], loaded["random_forest"], docs["decision_tree"], docs["random_forest"])
        write_json_atomic(out_dir / "correlation.json", corr)
        worksheet = out_dir / "review_worksheet.csv"
        if worksheet.exists() or worksheet.with_suffix(".rubric.txt").exists():  # a sheet may already hold human scores
            worksheet = out_dir / f"review_worksheet_{utc_now().replace(':', '').replace('.', '')}.csv"
        write_text_atomic(worksheet, to_csv(rows_from_results(list(docs.values()))))
        write_text_atomic(worksheet.with_suffix(".rubric.txt"), RUBRIC + "\n")
        summary["review_worksheet"] = worksheet.name
        summary["correlation_counts"] = corr["counts"]
    else:
        summary["note"] = ("This run stopped early. Any other results file or worksheet in this folder is from an "
                           "earlier run; only the results_file values above belong to this one.")
    write_json_atomic(out_dir / "summary.json", summary)
    for det in DETECTORS:
        print(f"{det}: {outcomes[det]['verdict']} ({counts_line(outcomes[det]['counts'])}; "
              f"verified={outcomes[det]['verified']})")
        for reason in outcomes[det]["reasons"]:
            print(f"  - {reason}")
    print(f"dataset: declared {a.dataset_version}; V3 profile and row alignment matched: {profile_ok}; "
          f"validation hash pinned and matched: {v3_verified}")
    if interrupted:
        print(INTERRUPTED_NOTE, file=sys.stderr)
    elif stopped is not None:
        print(json.dumps({"error": stopped.to_dict()}, indent=2), file=sys.stderr)
    qualifier = "" if v3_verified else " (V3 profile matched; file hash not pinned)" if profile_ok \
        else " (NOT a verified V3 run)"
    print(f"E2E {outcome['verdict']}{qualifier}")
    if interrupted:
        return 130
    if stopped is not None or any(lost_the_model(docs.get(d), settings) for d in DETECTORS):
        return 3
    return 0 if outcome["verdict"] == PASS else 1


def entry(argv=None) -> int:
    try:
        return main(argv)
    except SystemExit as exc:  # argparse rejected an argument (--help exits 0)
        if exc.code == 2:
            print("E2E NOT RUN: invalid input")
        raise
    except KeyboardInterrupt:
        if not STARTED:
            print("E2E NOT RUN: interrupted before anything changed")
            return 130
        print(INTERRUPTED_NOTE, file=sys.stderr)
        print("E2E INCOMPLETE")
        return 130
    except (DatasetMissing, ReservedSplitRefused, DatasetError, LargeRunNotAuthorized) as exc:
        print(json.dumps({"error": exc.to_dict()}, indent=2), file=sys.stderr)
        blocked = isinstance(exc, DatasetMissing)
        print("E2E BLOCKED: missing input" if blocked else "E2E NOT RUN: invalid input")
        return 2
    except NovaQwenError as exc:
        print(json.dumps({"error": exc.to_dict()}, indent=2), file=sys.stderr)
        if exc.code.startswith(("OLLAMA", "MODEL", "RUN_ABORTED")):
            print("E2E FAILED: Ollama or model problem")
            return 3
        if isinstance(exc, OutputError) or exc.code == "NOTEBOOK_FAILED":
            print(f"E2E FAILED: {exc.code}")
            return 1
        print("E2E NOT RUN: invalid input")
        return 2
    except OSError as exc:  # for example an --exports-dir that cannot be read
        print(json.dumps({"error": {"code": "FILE_ERROR", "message": str(exc)}}, indent=2), file=sys.stderr)
        print("E2E FAILED: file error")
        return 1
    except Exception as exc:  # a stored file no check anticipated, or a bug: summary.json still says INCOMPLETE
        print(json.dumps({"error": {"code": "UNEXPECTED_ERROR", "type": type(exc).__name__, "message": str(exc)}},
                         indent=2), file=sys.stderr)
        print("E2E FAILED: unexpected error")
        return 1


if __name__ == "__main__":
    sys.exit(entry())
