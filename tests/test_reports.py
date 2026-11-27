"""The writer of the tables of results: aggregates at full precision, never an identifier.

Engineering conventions, section 7, and docs/pilot_analysis.md, section "Data protection":
the tables copied into reports/ hold aggregate numbers only, so their writer refuses every
column, index and cell that identifies a participant or a segment. Synthetic tables only.
"""

import csv
import hashlib
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from listening_beyond_words.reports import ID_COLUMNS, write_table
from listening_beyond_words.runinfo import StepError

FLOATS = [0.1, 1 / 3, 5e-324, -0.0, 0.6000000000000001, 1e22, 0.30000000000000004]


def aggregates() -> pd.DataFrame:
    """A table as the analysis writes them: names, counts, floats, exact values and text."""
    return pd.DataFrame(
        {
            "number": [f"auroc_b{index}" for index in range(len(FLOATS))],
            "round": [1, 2, 1, 2, 1, 2, 1],
            "segments": [412, 398, 377, 405, 390, 401, 388],  # a count, not an ID
            "value": FLOATS,
            "exact": [Fraction(3, 100), Fraction(-1, 25), Fraction(1, 3), 0, 1, 2, 3],
            "passes": [True, False, True, True, False, True, True],
            "note": ["not available (cut: no B1)", "not defined", "x0", "rev", "-", "r1", "f1"],
        }
    )


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_a_table_is_written_at_full_precision_and_returns_its_sha256(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"
    sha256 = write_table(aggregates(), path)
    data = path.read_bytes()
    assert sha256 == hashlib.sha256(data).hexdigest()
    assert data.endswith(b"\n")
    assert b"\r" not in data
    back = rows(path)
    assert list(back[0]) == list(aggregates().columns)
    for row, expected in zip(back, FLOATS, strict=True):
        found = float(row["value"])  # the shortest text reads back as the same float
        assert found == expected
        assert math.copysign(1.0, found) == math.copysign(1.0, expected)
    assert [Fraction(row["exact"]) for row in back][:3] == [
        Fraction(3, 100),
        Fraction(-1, 25),
        Fraction(1, 3),
    ]
    assert back[0]["exact"] == "3/100"
    assert [row["passes"] for row in back][:2] == ["True", "False"]
    assert back[0]["note"] == "not available (cut: no B1)"
    assert [int(row["segments"]) for row in back] == [412, 398, 377, 405, 390, 401, 388]
    assert not (tmp_path / "summary.csv.tmp").exists()


@pytest.mark.parametrize(
    "name",
    [
        "session",
        "Session",
        "SESSION",
        " session ",
        "segment",
        "Segment_ID",
        "segment_id",
        "pid",
        "PID",
        "participant",
        "participant_id",
        "session_id",
        "Session ID",
        "session-id",
        "SessionID",
        "Segment ID",
        "segment.id",
        "Participant ID",
    ],
)
def test_each_id_column_is_refused_in_any_case_and_spelling(tmp_path: Path, name: str) -> None:
    path = tmp_path / "fold_values.csv"
    table = aggregates().assign(**{name: range(9001, 9008)})
    with pytest.raises(StepError, match="identifies a participant or a segment"):
        write_table(table, path)
    assert list(tmp_path.iterdir()) == []  # nothing is written


def test_the_id_names_are_the_conventions_and_the_plans() -> None:
    assert {"session", "segment", "pid", "segment_id"} <= ID_COLUMNS


def test_an_index_by_participant_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "fold_values.csv"
    table = aggregates()
    for index in (
        pd.Index(range(9001, 9008), name="session"),
        pd.Index(range(9001, 9008)),
        pd.RangeIndex(1, 8),
        pd.RangeIndex(7, name="session"),
        pd.RangeIndex(7, name="Session ID"),
        pd.MultiIndex.from_arrays([range(7), range(7)]),
    ):
        with pytest.raises(StepError, match="indexed 0 to n - 1"):
            write_table(table.set_axis(index), path)
    with pytest.raises(StepError, match="indexed 0 to n - 1"):
        write_table(table.set_index("segments"), path)
    assert not path.exists()
    write_table(table.set_axis(pd.Index(range(7), dtype=np.int64)), path)  # 0 to n - 1
    assert len(rows(path)) == 7


@pytest.mark.parametrize(
    ("columns", "match"),
    [
        ([9001, 9002], "named by text"),
        (["9001", "9002"], "named by text"),
        ([" 300 ", "x0"], "named by text"),
        (["303_007", "303_008"], "not by a number or a segment ID"),
        (["x0", "worst 9001_003"], "not by a number or a segment ID"),
        (["x0", "x0"], "given twice"),
    ],
)
def test_a_column_named_by_a_number_or_twice_is_refused(
    tmp_path: Path, columns: list[Any], match: str
) -> None:
    table = pd.DataFrame([[0.5, 0.6]], columns=columns)
    with pytest.raises(StepError, match=match):
        write_table(table, tmp_path / "table.csv")
    two_rows = pd.DataFrame({("a", "b"): [1], ("a", "c"): [2]})
    with pytest.raises(StepError, match="one row of column names"):
        write_table(two_rows, tmp_path / "table.csv")


@pytest.mark.parametrize("empty", [math.nan, None, "", "  ", pd.NA, np.float64("nan")])
def test_a_cell_without_a_value_is_refused(tmp_path: Path, empty: Any) -> None:
    table = aggregates().astype({"value": object, "note": object})
    table.loc[3, "note"] = empty
    table.loc[5, "value"] = empty
    with pytest.raises(StepError, match=r"^2 cells .* hold no value: a value that cannot exist"):
        write_table(table, tmp_path / "summary.csv")
    assert list(tmp_path.iterdir()) == []


def test_an_infinite_number_and_a_segment_id_in_a_cell_are_refused(tmp_path: Path) -> None:
    for column, value, match in (
        ("value", math.inf, "infinite number"),
        ("value", -math.inf, "infinite number"),
        ("note", "9001_003", "segment ID"),
        ("note", "worst segment 300_012", "segment ID"),
    ):
        table = aggregates().astype({"value": object})
        table.loc[2, column] = value
        with pytest.raises(StepError, match=match) as refusal:
            write_table(table, tmp_path / "summary.csv")
        assert "300_012" not in str(refusal.value)
        assert "9001" not in str(refusal.value)
    assert list(tmp_path.iterdir()) == []


def test_an_existing_file_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "checks.csv"
    first = write_table(aggregates(), path)
    with pytest.raises(FileExistsError, match="never overwrites"):
        write_table(aggregates().iloc[:2], path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == first


def test_the_same_table_gives_the_same_bytes(tmp_path: Path) -> None:
    assert write_table(aggregates(), tmp_path / "a.csv") == write_table(
        aggregates(), tmp_path / "b.csv"
    )
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
    empty = pd.DataFrame({"number": pd.Series([], dtype=str), "value": pd.Series([], dtype=float)})
    write_table(empty, tmp_path / "empty.csv")
    assert (tmp_path / "empty.csv").read_text(encoding="utf-8") == "number,value\n"
