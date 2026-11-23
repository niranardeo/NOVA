"""The guard that keeps the reserved final test split out of every tool in this package.

Paths are refused by name, before the file is touched. A renamed copy cannot be caught this way; V3's class counts
differ between validation and the reserved split, so check-dataset catches one afterwards, but it will have been read.
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

from .errors import ReservedSplitRefused


def is_reserved_name(name: str) -> bool:
    """True for names such as cicids2017_V3_test_RESERVED_15.csv or testing_dataset.csv, not latest_validation.csv.

    Names are case-folded first, because macOS file systems treat look-alike spellings as the same file. The word
    "preserved" is exempt; anything else containing "reserved", or a word starting with "test", is refused."""
    folded = unicodedata.normalize("NFKC", name).casefold()
    tokens = [t for t in re.split(r"[^a-z0-9]+", folded) if t]
    return any(t.startswith("test") or "reserved" in t.replace("preserved", "") for t in tokens)


def refuse_reserved(path) -> Path:
    """Return the path, or refuse it by name. Nothing is opened. The symlink target is compared too."""
    path = Path(path)
    if os.path.isdir(path):  # not a file: the caller reports that. Unlike Path.is_dir, this never raises
        return path
    resolved = path.resolve()
    # /dev/stdin and /dev/fd/N hide the real file's name; /dev/shm holds ordinary files under their real names
    if resolved.parts[1:2] in (("dev",), ("proc",)) and resolved.parts[1:3] != ("dev", "shm"):
        raise ReservedSplitRefused(f"refusing {path}: a device or descriptor path hides which file would be read")
    for name in (path.name, resolved.name):
        if is_reserved_name(name):
            raise ReservedSplitRefused(f"refusing {path}: the name marks it as a test or reserved split, which this "
                                       f"tool never reads")
    return path
