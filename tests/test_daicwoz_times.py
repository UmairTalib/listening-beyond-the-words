import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from listening_beyond_words.daicwoz import (
    best_offset,
    inside,
    load_transcripts,
    merge_intervals,
    read_turns,
)
from listening_beyond_words.settings import Workspace

HEADER = "start_time\tstop_time\tspeaker\tvalue"


def transcript(*rows: str) -> bytes:
    return ("\r\n".join([HEADER, *rows]) + "\r\n").encode()


def test_read_turns_keeps_times_and_speaker_but_no_words() -> None:
    data = transcript(
        "5.0\t6.0\tParticipant\tsecond",
        "1.0\t2.5\tEllie\tfirst",
        "x\t1\tEllie\tnot a number",
        "3\t2\tEllie\tbackwards",
        "7.0",
    )
    turns = read_turns(data)
    assert list(turns.columns) == ["start", "stop", "speaker"]
    assert turns.to_dict("list") == {
        "start": [1.0, 5.0],
        "stop": [2.5, 6.0],
        "speaker": ["ellie", "participant"],
    }


def test_read_turns_refuses_another_table() -> None:
    with pytest.raises(ValueError, match="not a DAIC-WOZ transcript"):
        read_turns(b"a\tb\n1\t2\n")


def test_merge_intervals_joins_overlapping_and_touching_ones() -> None:
    starts, stops = merge_intervals(np.array([5.0, 1.0, 2.0, 8.0]), np.array([6.0, 3.0, 4.0, 9.0]))
    assert starts.tolist() == [1.0, 5.0, 8.0]
    assert stops.tolist() == [4.0, 6.0, 9.0]


def test_inside_includes_the_ends_and_handles_no_intervals() -> None:
    points = np.array([0.5, 1.0, 2.0, 3.5, 4.0, 4.5])
    starts, stops = np.array([1.0, 3.0]), np.array([2.0, 4.0])
    assert inside(points, starts, stops).tolist() == [False, True, True, True, True, False]
    assert not inside(points, np.array([]), np.array([])).any()


def test_best_offset_finds_a_known_shift() -> None:
    rng = np.random.default_rng(0)
    starts = np.arange(0.0, 300.0, 10.0)
    stops = starts + 4.0
    points = starts + rng.uniform(0.5, 3.5, len(starts)) + 34.3  # the audio runs 34.3 s later
    shift, share = best_offset(points, starts, stops, np.round(np.arange(-60, 60.01, 0.1), 1))
    assert share == 1.0
    assert abs(shift - 34.3) <= 0.55


def test_best_offset_prefers_no_shift_on_ties_and_handles_empty_input() -> None:
    shift, share = best_offset(
        np.array([1.5]), np.array([1.0]), np.array([2.0]), np.array([-0.4, 0.0, 0.4])
    )
    assert (shift, share) == (0.0, 1.0)
    assert best_offset(np.array([]), np.array([1.0]), np.array([2.0]), np.array([0.0])) == (
        0.0,
        0.0,
    )


def make_run(root: Path, files: dict[int, bytes]) -> str:
    run = root / "daicwoz" / "run-test"
    (run / "transcripts").mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for session, data in files.items():
        (run / "transcripts" / f"{session}_TRANSCRIPT.csv").write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        rows.append({"session": session, "status": "ok", "transcript_sha256": sha})
    rows.append({"session": 9009, "status": "archive damaged", "transcript_sha256": None})
    pd.DataFrame(rows).to_csv(run / "sessions.tsv", sep="\t", index=False)
    return hashlib.sha256((run / "sessions.tsv").read_bytes()).hexdigest()


def test_load_transcripts_checks_every_file_and_takes_extra_sessions(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path.resolve())
    one = transcript("1\t2\tEllie\ta")
    two = transcript("1\t2\tParticipant\tb")
    three = transcript("3\t4\tEllie\tc")
    sha = make_run(workspace.root, {9001: one, 9002: two})
    extra = workspace.root / "daicwoz" / "extra" / "9003_TRANSCRIPT.csv"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(three)
    pin = {
        "run": "daicwoz/run-test",
        "sha256": sha,
        "extra": [
            {
                "session": 9003,
                "file": "daicwoz/extra/9003_TRANSCRIPT.csv",
                "sha256": hashlib.sha256(three).hexdigest(),
            }
        ],
    }
    assert load_transcripts(workspace, pin, sessions=(9001, 9002, 9003)) == {
        9001: one,
        9002: two,
        9003: three,
    }
    with pytest.raises(ValueError, match="no DAIC-WOZ transcript"):
        load_transcripts(workspace, {**pin, "extra": []}, sessions=(9001, 9002, 9003))


def test_load_transcripts_stops_on_a_changed_file(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path.resolve())
    sha = make_run(workspace.root, {9001: transcript("1\t2\tEllie\ta")})
    path = workspace.root / "daicwoz" / "run-test" / "transcripts" / "9001_TRANSCRIPT.csv"
    path.write_bytes(transcript("1\t2\tEllie\tchanged"))
    with pytest.raises(ValueError, match="does not match"):
        load_transcripts(workspace, {"run": "daicwoz/run-test", "sha256": sha}, sessions=(9001,))
