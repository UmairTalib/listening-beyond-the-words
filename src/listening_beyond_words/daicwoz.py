"""DAIC-WOZ human transcripts: finding them in the official session archives and reading them.

Each official session archive ``NNN_P.zip`` holds ``NNN_TRANSCRIPT.csv``, a tab-separated
table with the columns ``start_time``, ``stop_time``, ``speaker`` (``Ellie`` or
``Participant``) and ``value``. Unlike the E-DAIC transcripts, it lists the interviewer's
lines with their times. The DAIC-WOZ documentation names sessions without Ellie lines
(451, 458 and 480), sessions whose transcript times are offset from the audio, and sessions
with interruptions; later steps take those from the documentation, not from this module.
"""

import csv
import hashlib
import io
import re
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from listening_beyond_words.runinfo import check_sha256
from listening_beyond_words.sessions import PILOT_IDS
from listening_beyond_words.settings import Workspace

COUNT_KEYS = ("parsed", "ellie_lines", "participant_lines", "other_lines", "bad_rows")


def member_name(filename: str) -> str:
    r"""Return a member's file name without its folders, whichever separator the zip tool used.

    Args:
        filename: The member's name as stored in the archive.

    Returns:
        The part after the last ``/`` or ``\``.
    """
    return re.split(r"[/\\]", filename)[-1]


def find_member(archive: zipfile.ZipFile, name: str) -> zipfile.ZipInfo | None:
    """Return the member whose file name, without folders, is ``name`` in any case.

    Args:
        archive: An open archive.
        name: The file name to look for, for example ``300_TRANSCRIPT.csv``.

    Returns:
        The member, or None if the archive has none of that name.

    Raises:
        ValueError: If more than one member has that name.
    """
    wanted = name.lower()
    matches = [info for info in archive.infolist() if member_name(info.filename).lower() == wanted]
    if len(matches) > 1:
        raise ValueError(f"{len(matches)} members are named {name}")
    return matches[0] if matches else None


def turn_counts(data: bytes) -> dict[str, int]:
    """Count a transcript's lines per speaker, without keeping any of their text.

    The file is read as tab-separated text without quoting, so a quote inside a line
    cannot swallow the lines after it. A byte-order mark is ignored.

    Args:
        data: The bytes of a DAIC-WOZ transcript file.

    Returns:
        ``parsed`` (1 if the header names start_time, stop_time and speaker, else 0),
        ``ellie_lines``, ``participant_lines``, ``other_lines`` (any other speaker label)
        and ``bad_rows`` (rows that are too short or whose times are not two numbers
        in order).
    """
    counts = dict.fromkeys(COUNT_KEYS, 0)
    try:
        rows = _rows(data)
    except ValueError:
        return counts
    header = [cell.strip().lower() for cell in rows[0]] if rows else []
    if not {"start_time", "stop_time", "speaker"} <= set(header):
        return counts
    i_start, i_stop, i_speaker = (header.index(c) for c in ("start_time", "stop_time", "speaker"))
    speakers: Counter[str] = Counter()
    bad = 0
    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue
        if len(row) <= max(i_start, i_stop, i_speaker):
            bad += 1
            continue
        speakers[row[i_speaker].strip().lower()] += 1
        try:
            in_order = 0.0 <= float(row[i_start]) <= float(row[i_stop])
        except ValueError:
            in_order = False
        bad += not in_order
    counts.update(
        parsed=1,
        ellie_lines=speakers["ellie"],
        participant_lines=speakers["participant"],
        other_lines=sum(n for s, n in speakers.items() if s not in ("ellie", "participant")),
        bad_rows=bad,
    )
    return counts


def _rows(data: bytes) -> list[list[str]]:
    """The cells of a tab-separated transcript, without quoting and ignoring a byte-order mark."""
    text = data.decode("utf-8-sig", errors="replace")
    try:
        return list(
            csv.reader(io.StringIO(text, newline=""), delimiter="\t", quoting=csv.QUOTE_NONE)
        )
    except csv.Error as error:
        raise ValueError(f"unreadable transcript: {error}") from None


def read_turns(data: bytes) -> pd.DataFrame:
    """Return a transcript's turns with their times and speaker, but none of their words.

    Rows that are too short or whose times are not two numbers in order are left out;
    ``turn_counts`` counts them.

    Args:
        data: The bytes of a DAIC-WOZ transcript file.

    Returns:
        One row per turn, in time order, with ``start`` and ``stop`` in seconds and
        ``speaker`` in lower case (``ellie``, ``participant`` or another label).

    Raises:
        ValueError: If the header does not name start_time, stop_time and speaker.
    """
    rows = _rows(data)
    header = [cell.strip().lower() for cell in rows[0]] if rows else []
    if not {"start_time", "stop_time", "speaker"} <= set(header):
        raise ValueError("not a DAIC-WOZ transcript: no start_time, stop_time and speaker")
    i_start, i_stop, i_speaker = (header.index(c) for c in ("start_time", "stop_time", "speaker"))
    turns = []
    for row in rows[1:]:
        if len(row) <= max(i_start, i_stop, i_speaker):
            continue
        try:
            start, stop = float(row[i_start]), float(row[i_stop])
        except ValueError:
            continue
        if 0.0 <= start <= stop:
            turns.append((start, stop, row[i_speaker].strip().lower()))
    table = pd.DataFrame(turns, columns=["start", "stop", "speaker"])
    return table.sort_values(["start", "stop"], ignore_index=True)


def merge_intervals(starts: np.ndarray, stops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the union of intervals as sorted, non-overlapping starts and stops.

    Args:
        starts: Start of each interval in seconds.
        stops: End of each interval in seconds, not before its start.

    Returns:
        The starts and stops of the merged intervals.
    """
    if len(starts) == 0:
        return np.empty(0), np.empty(0)
    order = np.argsort(starts, kind="stable")
    merged_starts, merged_stops = [float(starts[order[0]])], [float(stops[order[0]])]
    for i in order[1:]:
        if starts[i] <= merged_stops[-1]:
            merged_stops[-1] = max(merged_stops[-1], float(stops[i]))
        else:
            merged_starts.append(float(starts[i]))
            merged_stops.append(float(stops[i]))
    return np.array(merged_starts), np.array(merged_stops)


def inside(points: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Return, for every point, whether it lies inside any interval (ends included).

    Args:
        points: Times in seconds.
        starts: Start of each interval; the intervals may overlap.
        stops: End of each interval.

    Returns:
        A boolean array, one value per point.
    """
    points = np.asarray(points, dtype=float)
    merged_starts, merged_stops = merge_intervals(np.asarray(starts), np.asarray(stops))
    result = np.zeros(len(points), dtype=bool)
    if len(merged_starts) == 0:
        return result
    index = np.searchsorted(merged_starts, points, side="right") - 1
    valid = index >= 0
    result[valid] = points[valid] <= merged_stops[index[valid]]
    return result


def best_offset(
    points: np.ndarray, starts: np.ndarray, stops: np.ndarray, shifts: np.ndarray
) -> tuple[float, float]:
    """Find the shift of the intervals that puts the most points inside them.

    Args:
        points: Times in seconds, for example the middles of aligned words.
        starts: Start of each interval, for example the participant's turns.
        stops: End of each interval.
        shifts: Candidate shifts in seconds, added to the interval times.

    Returns:
        The best shift and the share of points inside the shifted intervals. Ties go to
        the shift closest to zero; with no points or no intervals the result is (0, 0).
    """
    if len(points) == 0 or len(starts) == 0:
        return 0.0, 0.0
    shares = [float(inside(points, starts + d, stops + d).mean()) for d in shifts]
    best = max(range(len(shifts)), key=lambda k: (shares[k], -abs(float(shifts[k]))))
    return float(shifts[best]), shares[best]


def load_transcripts(
    workspace: Workspace, pin: Mapping[str, Any], sessions: Iterable[int] = PILOT_IDS
) -> dict[int, bytes]:
    """Read the pinned DAIC-WOZ transcripts and check every file against its SHA-256.

    Args:
        workspace: The workspace.
        pin: The ``daicwoz_transcripts`` entry of ``configs/pilot/data.yaml``: ``run``
            (a run folder of ``extract_daicwoz_transcripts.py``), ``sha256`` (of its
            ``sessions.tsv``, which lists every transcript's SHA-256) and ``extra``
            (sessions whose transcript comes from elsewhere: ``session``, ``file``, ``sha256``).
        sessions: The sessions that must all have a transcript.

    Returns:
        The bytes of each session's transcript.

    Raises:
        ValueError: If a file does not match its SHA-256, a session has two transcripts,
            or a required session has none.
    """
    run = workspace.resolve(pin["run"])
    table_path = run / "sessions.tsv"
    check_sha256(table_path, pin["sha256"])
    table = pd.read_csv(table_path, sep="\t")
    found: dict[int, bytes] = {}
    ok = table[table["status"] == "ok"]
    for session, expected in zip(
        ok["session"].astype(int).tolist(),
        ok["transcript_sha256"].astype(str).tolist(),
        strict=True,
    ):
        data = (run / "transcripts" / f"{session}_TRANSCRIPT.csv").read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"session {session}: transcript does not match its SHA-256")
        found[session] = data
    for extra in pin.get("extra") or []:
        session = int(extra["session"])
        if session in found:
            raise ValueError(f"session {session} has two transcripts")
        path = workspace.resolve(extra["file"])
        check_sha256(path, extra["sha256"])
        found[session] = path.read_bytes()
    missing = sorted(set(sessions) - set(found))
    if missing:
        raise ValueError(f"no DAIC-WOZ transcript for sessions {missing}")
    return found
