"""DAIC-WOZ human transcripts: finding them in the official session archives and reading them.

Each official session archive ``NNN_P.zip`` holds ``NNN_TRANSCRIPT.csv``, a tab-separated
table with the columns ``start_time``, ``stop_time``, ``speaker`` (``Ellie`` or
``Participant``) and ``value``. Unlike the E-DAIC transcripts, it lists the interviewer's
lines with their times. The DAIC-WOZ documentation names sessions without Ellie lines
(451, 458 and 480), sessions whose transcript times are offset from the audio, and sessions
with interruptions; later steps take those from the documentation, not from this module.
"""

import csv
import io
import re
import zipfile
from collections import Counter

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
    text = data.decode("utf-8-sig", errors="replace")
    try:
        rows = list(
            csv.reader(io.StringIO(text, newline=""), delimiter="\t", quoting=csv.QUOTE_NONE)
        )
    except csv.Error:
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
