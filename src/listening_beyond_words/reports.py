"""The writer of the tables of results: aggregate numbers only, at full precision.

Engineering conventions, section 7, and pilot analysis, section "Data protection": the
tables that are copied into ``reports/`` and committed hold aggregate numbers only, such
as per-fold AUROCs, means, intervals and counts, and their writer refuses any column that
identifies a participant or a segment. ``write_table`` writes one table as CSV, all or
nothing, and returns the SHA-256 that the job's ``run.json`` records, so that a copy in
``reports/`` can be checked against it.
"""

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from listening_beyond_words.runinfo import StepError

# Column names that hold a participant's or a segment's ID: a participant is named by their
# session ID (engineering conventions, section 4, "Naming"). A name is compared in lower case
# with everything but its letters and digits dropped, so "Session ID", "session-id" and
# "SessionID" are refused as "session_id" is, while a count such as "segments" stays allowed.
ID_COLUMNS: Final = frozenset(
    {"session", "session_id", "segment", "segment_id", "pid", "participant", "participant_id"}
)
_ID_NAMES: Final = frozenset(re.sub(r"[^0-9a-z]", "", name) for name in ID_COLUMNS)
# A segment's ID, {pid}_{index:03d} (section 4), alone or inside a text; a pid of 3 digits,
# or of 4 as the tests' fake IDs from 9001.
SEGMENT_ID: Final = re.compile(r"(?<![0-9])[0-9]{3,4}_[0-9]{3}(?![0-9])")
_WHOLE_NUMBER: Final = re.compile(r"[+-]?[0-9]+")
_NO_VALUE_HINT: Final = (
    'a value that cannot exist is written as text, such as "not available (cut: no B1)"'
    ' or "not defined" (pilot analysis, sections "Runs and pins" and "The main numbers")'
)


def _no_value(value: Any) -> bool:
    """Whether a cell holds no value: missing, NaN, or text with nothing in it."""
    if isinstance(value, str):
        return not value.strip()
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float | np.floating) and math.isnan(value)


def _not_finite(value: Any) -> bool:
    """Whether a cell holds an infinite number."""
    return isinstance(value, float | np.floating) and math.isinf(value)


def _names_an_id(name: str) -> bool:
    """Whether a column name is an ID's, in any case and spelling (``ID_COLUMNS``)."""
    return re.sub(r"[^0-9a-z]", "", name.lower()) in _ID_NAMES


def write_table(table: pd.DataFrame, path: Path) -> str:
    """Write an aggregate table as CSV, at full precision and all or nothing; return its SHA-256.

    Before anything is written, the table is refused if:

    - a column is named as an ID (``ID_COLUMNS``, in any case and with any separators,
      such as "Session ID"), has a name that is not text, is a whole number or holds a
      segment ID, as a session or segment ID would be in a table turned on its side, or
      is given twice;
    - a cell holds a segment ID (``SEGMENT_ID``);
    - its index is anything but 0 to n − 1, such as one by participant; the index is not
      written;
    - a cell holds no value (missing, NaN or empty text) or an infinite number: in the
      tables of the analysis a value that cannot exist is written as text, never blank;
    - the file exists already: a job never overwrites its output (engineering conventions,
      section 9).

    Floats are written as Python writes them, the shortest text that reads back as the
    same float; exact fractions as ``p/q``; whole numbers, text, and True and False as
    they are. The text goes to a temporary file beside ``path`` that is then renamed onto
    it, so a job that dies while writing leaves no half table.

    Args:
        table: The table, one row per aggregate, with names of text as columns.
        path: The CSV file, in the job's own folder.

    Returns:
        The SHA-256 of the file written.

    Raises:
        StepError: If the table is refused. The message names the rule and a column name,
            never a cell or a name that is a number.
        FileExistsError: If the file exists already.
    """
    if isinstance(table.columns, pd.MultiIndex):
        raise StepError("a table of results has one row of column names, not several")
    names: list[Any] = list(table.columns)  # any labels at all, whatever the stubs say
    for name in names:
        if (
            not isinstance(name, str)
            or _WHOLE_NUMBER.fullmatch(name.strip())
            or SEGMENT_ID.search(name)
        ):
            raise StepError(
                "a column of a table of results must be named by text, not by a number or a"
                " segment ID, as a session or segment ID would name it"
            )
        if _names_an_id(name):
            raise StepError(
                f"the column {name!r} identifies a participant or a segment: the tables of"
                " results hold aggregate numbers only (engineering conventions, section 7)"
            )
    if len(set(names)) != len(names):
        raise StepError("a column of a table of results is given twice")
    index = table.index
    if (
        isinstance(index, pd.MultiIndex)
        or not index.equals(pd.RangeIndex(len(table)))
        or (index.name is not None and _names_an_id(str(index.name)))
    ):
        raise StepError(
            "a table of results must be indexed 0 to n - 1: an index by participant or"
            " segment would identify them"
        )
    cells = [value for name in names for value in table[name].tolist()]
    empty = sum(_no_value(value) for value in cells)
    if empty:
        raise StepError(f"{empty} cells of a table of results hold no value: {_NO_VALUE_HINT}")
    if any(_not_finite(value) for value in cells):
        raise StepError("a cell of a table of results holds an infinite number")
    if any(isinstance(value, str) and SEGMENT_ID.search(value) for value in cells):
        raise StepError("a cell of a table of results holds a segment ID")
    if path.exists():
        raise FileExistsError(f"{path.name} exists already: a job never overwrites its output")
    data = table.to_csv(index=False, lineterminator="\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return hashlib.sha256(data).hexdigest()
