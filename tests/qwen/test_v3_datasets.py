"""Dataset identity, the reserved-split guard and export-to-CSV row alignment (synthetic files only)."""
import csv
import os
import sys

import pytest

from qwen_test_support import DT_EMPTY, DT_FIXTURE, write_json, load_json
from src.nova_qwen.alerts import load_export
from src.nova_qwen import datasets as D
from src.nova_qwen.errors import DatasetError, DatasetMissing, ReservedSplitRefused
from src.nova_qwen.output import check_output_path, load_results

HEADER = list(D.V3_COLUMNS)


def row(duration=1000, packets=3, rate=2.5, label="Normal Traffic"):
    values = dict.fromkeys(HEADER, "0")
    values.update({"Flow Duration": str(duration), "Total Fwd Packets": str(packets), "Flow Bytes/s": str(rate),
                   D.LABEL_COLUMN: label})
    return [values[c] for c in HEADER]


def write_csv(path, rows, header=HEADER):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


@pytest.mark.parametrize("name", ["cicids2017_V3_test_RESERVED_15.csv", "test.csv", "testing_dataset_CIC_15.csv",
                                  "V3_test_15.csv", "RESERVED.csv", "my-reserved-copy.csv", "TEST_split.CSV"])
def test_reserved_names_are_refused(name):
    assert D.is_reserved_name(name)


@pytest.mark.parametrize("name", ["cicids2017_V3_validation_15.csv", "cicids2017_V3_train_70.csv",
                                  "latest_validation.csv", "contest.csv", "synthetic_validation.csv"])
def test_approved_names_are_not_refused(name):
    assert not D.is_reserved_name(name)


def test_reserved_file_is_refused_before_it_is_opened(tmp_path):
    target = tmp_path / "cicids2017_V3_test_RESERVED_15.csv"
    target.write_text("must never be read")
    wanted, opened = str(target), []
    # An audit hook sees every open (io.open, Path.open, os.open), which patching builtins.open does not.
    sys.addaudithook(lambda event, args: opened.append(args) if event == "open" and str(args[0]) == wanted else None)
    export = load_export(DT_FIXTURE)
    with pytest.raises(ReservedSplitRefused):
        D.inspect_csv(target)
    with pytest.raises(ReservedSplitRefused):
        D.check_dataset(validation=target)
    with pytest.raises(ReservedSplitRefused):
        D.check_alignment(export, target)
    with pytest.raises(ReservedSplitRefused):
        load_export(target)
    with pytest.raises(ReservedSplitRefused):
        load_results(target)
    with pytest.raises(ReservedSplitRefused):
        check_output_path(target, [], overwrite=True, resume=False)
    assert not opened


@pytest.mark.parametrize("name", ["cicids2017_V3_te\u017ft_RESERVED_15.csv", "cicids2017_V3_teſt_15.csv", "ＲＥＳＥＲＶＥＤ.csv"])
def test_look_alike_spellings_of_reserved_names_are_refused(name):
    assert D.is_reserved_name(name)


def test_device_paths_are_refused(tmp_path):
    with pytest.raises(ReservedSplitRefused):
        D.inspect_csv("/dev/stdin")


def test_shared_memory_files_are_ordinary_files_but_still_judged_by_name():
    assert D.refuse_reserved("/dev/shm/x/decision_tree_alerts.json")
    for path in ("/dev/shm/x/cicids2017_V3_test_RESERVED_15.csv", "/dev/shm/../stdin", "/dev/null", "/proc/self/fd/0"):
        with pytest.raises(ReservedSplitRefused):
            D.refuse_reserved(path)


def test_a_name_with_the_word_preserved_is_not_refused():
    assert not D.is_reserved_name("cicids2017_V3_validation_15_order_preserved.csv")
    assert D.is_reserved_name("finalreserved_15.csv")


def test_an_output_with_a_test_like_name_is_refused_whether_or_not_it_exists(tmp_path):
    target = tmp_path / "test_run.json"
    with pytest.raises(ReservedSplitRefused):
        check_output_path(target, [], overwrite=False, resume=False)
    assert not target.exists()
    target.write_text("{}")
    with pytest.raises(ReservedSplitRefused):
        check_output_path(target, [], overwrite=True, resume=False)


def test_symlink_to_reserved_file_is_refused(tmp_path):
    target = tmp_path / "cicids2017_V3_test_RESERVED_15.csv"
    target.write_text("must never be read")
    link = tmp_path / "cicids2017_V3_validation_15.csv"
    link.symlink_to(target)
    with pytest.raises(ReservedSplitRefused):
        D.inspect_csv(link)


def test_missing_file_is_reported_not_substituted(tmp_path):
    with pytest.raises(DatasetMissing):
        D.check_dataset(validation=tmp_path / D.V3_VALIDATION_FILE)


def small_v3(monkeypatch, tmp_path, flows=None):
    """Files named like V3, with the V3 profile narrowed to their size, so the checks themselves can be exercised."""
    flows = flows or [119_000_000, 2000, 3000]
    monkeypatch.setattr(D, "V3_VALIDATION_ROWS", 3)
    monkeypatch.setattr(D, "V3_TRAIN_ROWS", 3)
    classes = {"Normal Traffic": 2, "DoS": 1}
    monkeypatch.setattr(D, "V3_VALIDATION_CLASSES", classes)
    monkeypatch.setattr(D, "V3_TRAIN_CLASSES", classes)
    rows = [row(f, label="DoS" if i == 0 else "Normal Traffic") for i, f in enumerate(flows)]
    return write_csv(tmp_path / D.V3_TRAIN_FILE, rows), write_csv(tmp_path / D.V3_VALIDATION_FILE, rows)


def test_file_matching_the_v3_profile_passes(monkeypatch, tmp_path):
    train, validation = small_v3(monkeypatch, tmp_path)
    report = D.check_dataset(train=train, validation=validation)
    assert report["passed"] and report["v3_profile_matched"]
    assert not report["v3_validation_verified"] and not report["v3_identity_verified"]  # no published hash compared
    assert report["validation"]["sha256"] == D.sha256_file(validation) and report["validation"]["rows"] == 3
    assert report["validation"]["label_distribution"] == {"DoS": 1, "Normal Traffic": 2}
    assert report["validation"]["units"]["consistent"] == {"Flow Duration": True, "Total Fwd Packets": True,
                                                           "Flow Bytes/s": True}


def test_a_matching_published_hash_is_what_makes_a_file_verified(monkeypatch, tmp_path):
    train, validation = small_v3(monkeypatch, tmp_path)
    t, v = D.sha256_file(train), D.sha256_file(validation)
    only_validation = D.check_dataset(validation=validation, expected_validation_sha256=v)
    assert only_validation["v3_validation_verified"] and not only_validation["v3_identity_verified"]
    both = D.check_dataset(train=train, validation=validation, expected_train_sha256=t, expected_validation_sha256=v)
    assert both["v3_validation_verified"] and both["v3_identity_verified"]
    assert both["hash_pinned"] == {"validation": True, "train": True}
    wrong = D.check_dataset(validation=validation, expected_validation_sha256="0" * 64)
    assert not wrong["passed"] and not wrong["v3_profile_matched"] and not wrong["v3_validation_verified"]


def test_validation_alone_is_not_full_v3_identity(monkeypatch, tmp_path):
    _, validation = small_v3(monkeypatch, tmp_path)
    report = D.check_dataset(validation=validation)
    assert report["v3_profile_matched"] and not report["v3_identity_verified"] and report["train"] is None


def test_an_expected_hash_with_no_file_is_a_problem_not_ignored(monkeypatch, tmp_path):
    _, validation = small_v3(monkeypatch, tmp_path)
    report = D.check_dataset(validation=validation, expected_train_sha256="0" * 64)
    assert not report["passed"] and any("no train file" in p for p in report["problems"])


def test_a_v3_file_name_alone_does_not_make_it_v3(tmp_path):
    path = write_csv(tmp_path / D.V3_VALIDATION_FILE, [row()])
    report = D.check_dataset(validation=path)
    assert not report["passed"] and not report["v3_validation_verified"]
    assert any("rows, V3 has 377880" in p for p in report["problems"])
    assert any("class counts" in p for p in report["problems"])


def test_class_count_mismatch_fails_v3_identity(monkeypatch, tmp_path):
    _, validation = small_v3(monkeypatch, tmp_path)
    monkeypatch.setattr(D, "V3_VALIDATION_CLASSES", {"Normal Traffic": 1, "DoS": 2})
    report = D.check_dataset(validation=validation)
    assert any("class counts" in p for p in report["problems"]) and not report["v3_profile_matched"]


OLD_VALIDATION_ROWS = 378_113
OLD_VALIDATION_CLASSES = {"Bots": 292, "Brute Force": 1372, "DDoS": 19202, "DoS": 29062, "Normal Traffic": 314259,
                          "Port Scanning": 13604, "Web Attacks": 322}


def _inspected_as(monkeypatch, tmp_path, rows, classes):
    """check_dataset as if the file had these counts, without generating a 46 MB file."""
    path = write_csv(tmp_path / D.V3_VALIDATION_FILE, [row(119_000_000), row(2000)])
    real = D.inspect_csv
    monkeypatch.setattr(D, "inspect_csv", lambda p: {**real(p), "rows": rows,
                                                     "label_distribution": dict(sorted(classes.items()))})
    return D.check_dataset(validation=path)


def test_v3_counts_themselves_match_the_profile(monkeypatch, tmp_path):
    assert _inspected_as(monkeypatch, tmp_path, D.V3_VALIDATION_ROWS, D.V3_VALIDATION_CLASSES)["v3_profile_matched"]


def test_old_dataset_counts_do_not_pass_as_v3_by_renaming(monkeypatch, tmp_path):
    report = _inspected_as(monkeypatch, tmp_path, OLD_VALIDATION_ROWS, OLD_VALIDATION_CLASSES)
    assert any("378113 rows, V3 has 377880" in p for p in report["problems"])
    assert any("class counts" in p for p in report["problems"]) and not report["v3_profile_matched"]


def test_renamed_reserved_counts_are_told_apart_by_class_counts(monkeypatch, tmp_path):
    """V3's reserved partition differs from validation in Brute Force and Web Attacks (notebook 04 percentages)."""
    counts = {**D.V3_VALIDATION_CLASSES, "Brute Force": 1373, "Web Attacks": 321}
    report = _inspected_as(monkeypatch, tmp_path, D.V3_VALIDATION_ROWS, counts)
    assert len(report["problems"]) == 1 and "class counts" in report["problems"][0]
    assert not report["v3_profile_matched"]


def test_v3_columns_must_match_in_name_and_order(monkeypatch, tmp_path):
    _, validation = small_v3(monkeypatch, tmp_path)
    with open(validation, newline="") as fh:
        rows = list(csv.reader(fh))[1:]
    swapped = list(HEADER)
    swapped[5], swapped[6] = swapped[6], swapped[5]  # neither is an evidence column
    report = D.check_dataset(validation=write_csv(validation, rows, swapped))
    assert len(report["problems"]) == 1 and "columns differ from V3" in report["problems"][0]
    assert not report["passed"] and not report["v3_profile_matched"]


def test_v3_file_name_must_match(monkeypatch, tmp_path):
    _, validation = small_v3(monkeypatch, tmp_path)
    report = D.check_dataset(validation=validation.rename(tmp_path / "latest_validation.csv"))
    assert len(report["problems"]) == 1 and "file name" in report["problems"][0]
    assert not report["passed"] and not report["v3_profile_matched"]


def test_header_only_file_is_not_a_pass(tmp_path):
    report = D.check_dataset(validation=write_csv(tmp_path / "v.csv", []), version="X")
    assert not report["passed"] and any("no data rows" in p for p in report["problems"])


def test_a_validation_file_is_required():
    with pytest.raises(DatasetMissing):
        D.check_dataset(validation=None)


@pytest.mark.parametrize("content", [b"\xff\xfe\x00bad", b'a,b\n"unterminated,1\n' + b"x" * 200_000 + b"\n"])
def test_unreadable_files_are_input_errors_not_crashes(tmp_path, content):
    path = tmp_path / "v.csv"
    path.write_bytes(content)
    with pytest.raises(DatasetError) as info:
        D.inspect_csv(path)
    assert info.value.code == "DATASET_UNREADABLE"
    export = load_export(DT_FIXTURE)
    with pytest.raises(DatasetError):
        D.check_alignment(export, path)


def test_blank_lines_are_counted_as_pandas_counts_them(tmp_path):
    """pandas skips blank lines, and record_id counts its rows, so positions here must skip them too."""
    export, rows = fixture_rows()
    path = tmp_path / "v.csv"
    write_csv(path, rows)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:3] + ["", "   "] + lines[3:]) + "\n\n")
    report = D.check_alignment(export, path)
    assert report["passed"] and report["validation_rows"] == 8
    assert D.inspect_csv(path)["rows"] == 8


@pytest.mark.parametrize("bad", ["", " ", "v3", "V3 ", " V3", "V-3", "CICIDS2017-V3", "V_3"])
def test_near_spellings_of_v3_are_refused(bad):
    with pytest.raises(D.NovaQwenError):
        D.require_version_spelling(bad)


@pytest.mark.parametrize("ok", ["V3", "SYNTHETIC-DEMO", "V2-LEGACY-REGRESSION", "X"])
def test_ordinary_version_names_are_allowed(ok):
    assert D.require_version_spelling(ok) == ok


def test_recorded_path_shows_a_symlink_target(tmp_path):
    target = write_csv(tmp_path / "real.csv", [row()])
    link = tmp_path / "v.csv"
    link.symlink_to(target)
    info = D.inspect_csv(link)
    assert info["is_symlink"] and info["resolved_path"] == str(target.resolve()) and info["path"] == str(link)


def test_profile_counts_sum_to_the_split_sizes():
    assert sum(D.V3_TRAIN_CLASSES.values()) == D.V3_TRAIN_ROWS
    assert sum(D.V3_VALIDATION_CLASSES.values()) == D.V3_VALIDATION_ROWS
    assert len(D.V3_COLUMNS) == 53 and D.V3_COLUMNS[-1] == D.LABEL_COLUMN


def test_non_v3_version_is_never_reported_as_v3(tmp_path):
    path = write_csv(tmp_path / "synthetic_validation.csv", [row()])
    report = D.check_dataset(validation=path, version="SYNTHETIC-DEMO")
    assert report["passed"] and not report["checked_against_v3_profile"]
    assert not report["v3_validation_verified"] and not report["v3_identity_verified"]


def test_expected_hash_mismatch_fails(tmp_path):
    path = write_csv(tmp_path / "v.csv", [row()])
    report = D.check_dataset(validation=path, version="X", expected_validation_sha256="0" * 64)
    assert not report["passed"] and any("sha256" in p for p in report["problems"])
    assert report["hash_pinned"] == {"validation": True, "train": False}
    good = D.check_dataset(validation=path, version="X", expected_validation_sha256=D.sha256_file(path).upper())
    assert good["passed"]


@pytest.mark.parametrize("case, rows, header, expect", [
    ("no label column", [row()[:-1]], HEADER[:-1], "label column"),
    ("no evidence column", [row()[1:]], HEADER[1:], None),
    ("short row", [row()[:-2]], HEADER, "fields"),
    ("unknown class", [row(label="Benign")], HEADER, "unexpected class labels"),
    ("missing evidence", [row(duration="")], HEADER, "missing or non-numeric"),
    ("non numeric evidence", [row(rate="fast")], HEADER, "missing or non-numeric"),
    ("infinite evidence", [row(rate="inf")], HEADER, "missing or non-numeric"),
])
def test_general_checks(tmp_path, case, rows, header, expect):
    header = [c for c in header]
    if case == "no evidence column":
        header = [c for c in HEADER if c != "Flow Duration"]
        rows = [[v for c, v in zip(HEADER, row()) if c != "Flow Duration"]]
        expect = "evidence columns missing"
    report = D.check_dataset(validation=write_csv(tmp_path / "v.csv", rows, header), version="X")
    assert not report["passed"] and any(expect in p for p in report["problems"]), report["problems"]


def test_training_and_validation_columns_must_match(tmp_path):
    other = list(HEADER)
    other[0], other[1] = other[1], other[0]
    report = D.check_dataset(train=write_csv(tmp_path / "t.csv", [row()], other),
                             validation=write_csv(tmp_path / "v.csv", [row()]), version="X")
    assert any("columns differ" in p for p in report["problems"])


@pytest.mark.parametrize("flows, consistent", [
    ([119_999_998, 50_000, -1], True),      # the shape seen in the old CICIDS2017 files, incl. a negative duration
    ([120.0, 0.05, 3.0], False),            # looks like seconds
    ([119.5, 2.5, 3.5], False),             # fractional
    ([5_000_000_000, 10, 20], False),       # far beyond the 120 s flow timeout
])
def test_flow_duration_unit_plausibility(tmp_path, flows, consistent):
    path = write_csv(tmp_path / "v.csv", [row(f) for f in flows])
    units = D.assess_units(D.inspect_csv(path))
    assert units["consistent"]["Flow Duration"] is consistent
    assert units["assumed_units"]["Flow Duration"] == "microseconds"
    assert units["authority"].startswith("Plausibility only") and "says nothing about the unit" in units["basis"]["Flow Bytes/s"]


def test_unit_problem_fails_only_the_v3_profile(monkeypatch, tmp_path):
    _, validation = small_v3(monkeypatch, tmp_path, flows=[120, 2, 3])
    assert any("assumed units" in p for p in D.check_dataset(validation=validation)["problems"])
    assert D.check_dataset(validation=validation, version="X")["passed"]


def fixture_rows():
    """A CSV whose rows match the DT fixture's alerts at their record_id (row 1 has no alert)."""
    export = load_export(DT_FIXTURE)
    by_position = {a.record_position: a for a in export.alerts}
    rows = []
    for position in range(1, 9):
        a = by_position.get(position)
        rows.append(row(a.evidence["Flow Duration"], a.evidence["Total Fwd Packets"], a.evidence["Flow Bytes/s"])
                    if a else row(11, 11, 11.0))
    return export, rows


def test_alignment_passes_when_every_alert_matches_its_row(tmp_path):
    export, rows = fixture_rows()
    report = D.check_alignment(export, write_csv(tmp_path / "v.csv", rows))
    assert report["passed"] and report["alerts_checked"] == 5 and report["validation_rows"] == 8


def test_alignment_never_uses_the_label_column(tmp_path):
    export, rows = fixture_rows()
    for r in rows:
        r[-1] = "not a real label"
    assert D.check_alignment(export, write_csv(tmp_path / "v.csv", rows))["passed"]


def test_alignment_detects_a_reordered_file(tmp_path):
    export, rows = fixture_rows()
    report = D.check_alignment(export, write_csv(tmp_path / "v.csv", list(reversed(rows))))
    assert not report["passed"] and report["evidence_values_differing"] > 0 and report["examples"]


def test_alignment_detects_a_frame_with_rows_removed_before_an_alert(tmp_path):
    """An export made from a filtered frame numbers rows of that frame, so its ids point at the wrong CSV rows."""
    export, rows = fixture_rows()
    rows.insert(0, row(777, 7, 7.0))
    assert not D.check_alignment(export, write_csv(tmp_path / "v.csv", rows))["passed"]


def test_alignment_reports_what_it_cannot_prove_and_catches_a_long_tail_for_v3(tmp_path):
    """A head(n) export leaves every id lined up. Only the alert-free tail gives it away, and only for V3's size."""
    export, rows = fixture_rows()
    rows += [row(11, 11, 11.0) for _ in range(192)]  # 200 rows; the last alert is at row 8
    path = write_csv(tmp_path / "v.csv", rows)
    general = D.check_alignment(export, path)
    assert general["passed"] and general["last_alert_position"] == 8 and general["rows_after_last_alert"] == 192
    assert "not proven" in general["frame_completeness"]
    report = D.check_alignment(export, path, expected_rows=200)
    assert not report["passed"] and any("truncated" in p for p in report["problems"])


def test_alignment_does_not_call_a_chance_tail_truncated(tmp_path):
    export, rows = fixture_rows()
    rows += [row(11, 11, 11.0)] * 2  # 10 rows: 2 after the last of 5 alerts is ordinary by chance
    report = D.check_alignment(export, write_csv(tmp_path / "v.csv", rows), expected_rows=10)
    assert report["rows_after_last_alert"] == 2 and report["passed"]


def test_alignment_needs_alerts_only_when_asked(tmp_path):
    export = load_export(DT_EMPTY)
    path = write_csv(tmp_path / "v.csv", [row()])
    assert D.check_alignment(export, path, expected_rows=1)["passed"]
    report = D.check_alignment(export, path, expected_rows=1, require_alerts=True)
    assert not report["passed"] and any("nothing was compared" in p for p in report["problems"])


def test_rows_are_numbered_as_pandas_numbers_them(tmp_path):
    pd = pytest.importorskip("pandas")
    export, rows = fixture_rows()
    path = write_csv(tmp_path / "v.csv", rows)
    body = path.read_text()
    header, rest = body.split("\n", 1)
    leading_blank = tmp_path / "blank_first.csv"
    leading_blank.write_text(f"\n \n{header}\n{rest}")
    assert len(pd.read_csv(leading_blank)) == 8 and D.inspect_csv(leading_blank)["rows"] == 8
    assert D.check_alignment(export, leading_blank)["passed"]
    quoted = tmp_path / "quoted_line.csv"
    quoted.write_text(f'{header}\n""\n{rest}')
    assert len(pd.read_csv(quoted)) == 9  # the lone quoted field is a row, so every later position moves by one
    assert D.inspect_csv(quoted)["rows"] == 9
    assert not D.check_alignment(export, quoted)["passed"]


def test_alignment_report_identifies_the_file_it_compared(tmp_path):
    export, rows = fixture_rows()
    path = write_csv(tmp_path / "v.csv", rows)
    report = D.check_alignment(export, path)
    assert report["validation_sha256"] == D.sha256_file(path) and report["validation_path"] == str(path)
    assert report["expected_rows"] is None


def test_alignment_detects_alerts_past_the_end(tmp_path):
    export, rows = fixture_rows()
    report = D.check_alignment(export, write_csv(tmp_path / "v.csv", rows[:4]))
    assert not report["passed"] and any("past the last row" in p for p in report["problems"])


def test_alignment_checks_expected_row_count(tmp_path):
    export, rows = fixture_rows()
    report = D.check_alignment(export, write_csv(tmp_path / "v.csv", rows), expected_rows=377_880)
    assert not report["passed"] and any("expected 377880" in p for p in report["problems"])


def test_alignment_reports_missing_evidence_column(tmp_path):
    export, rows = fixture_rows()
    header = [c if c != "Flow Bytes/s" else "Other" for c in HEADER]
    report = D.check_alignment(export, write_csv(tmp_path / "v.csv", rows, header))
    assert not report["passed"] and any("not in" in p for p in report["problems"])


def test_alignment_refuses_reserved_file(tmp_path):
    export, rows = fixture_rows()
    with pytest.raises(ReservedSplitRefused):
        D.check_alignment(export, write_csv(tmp_path / "cicids2017_V3_test_RESERVED_15.csv", rows))


def test_alignment_of_an_empty_export_has_nothing_to_contradict(tmp_path, work):
    doc = load_json(work / "decision_tree_alerts.json")
    doc["alerts"] = []
    export = load_export(write_json(tmp_path / "e.json", doc))
    report = D.check_alignment(export, write_csv(tmp_path / "v.csv", [row()]))
    assert report["passed"] and report["alerts_checked"] == 0 and report["last_alert_position"] is None


def test_alignment_tolerates_only_float_parser_noise(tmp_path, work):
    pd = pytest.importorskip("pandas")
    from src.alert_adapter import export_alerts
    rates = [0.154073759, 123.7509776, 1e-3, 91130.51471, 2400.5000000001, 0.30000000000000004]
    rows = [row(10 + i, 2, r, "Normal Traffic") for i, r in enumerate(rates)]
    path = write_csv(tmp_path / "v.csv", rows)
    frame = pd.read_csv(path, usecols=["Flow Duration", "Total Fwd Packets", "Flow Bytes/s"])
    export_alerts(frame, ["DoS"] * len(rates), "decision_tree", tmp_path / "x.json", run_id="r", dataset="d")
    assert D.check_alignment(load_export(tmp_path / "x.json"), path)["passed"]
    rows[3] = row(13, 2, 91130.6, "Normal Traffic")  # a real difference, in the 6th significant digit
    assert not D.check_alignment(load_export(tmp_path / "x.json"), write_csv(path, rows))["passed"]
    rows[3] = row(13, 2, repr(91130.51471 * (1 + 1e-11)), "Normal Traffic")  # ten times the tolerance
    assert not D.check_alignment(load_export(tmp_path / "x.json"), write_csv(path, rows))["passed"]


def test_module_uses_only_the_standard_library():
    """The consumer must keep running with no installed packages, including this dataset check."""
    imports = [ln for ln in open(D.__file__, encoding="utf-8") if ln.startswith(("import ", "from "))]
    assert not [ln for ln in imports if any(x in ln for x in ("pandas", "numpy", "sklearn"))]
