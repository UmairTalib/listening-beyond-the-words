"""E-DAIC transcripts: reading them and summarising what they contain.

Each session has ``NNN_Transcript.csv`` with the columns Start_Time, End_Time,
Text and Confidence: one line per utterance found by Google's speech recognition
(AVEC 2019). The files have no speaker column, so whether the interviewer's lines
are included has to be measured; ``INTERVIEWER_PHRASES`` serves that purpose.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

COLUMNS: Final = {
    "Start_Time": "start",
    "End_Time": "end",
    "Text": "text",
    "Confidence": "confidence",
}

# Lines of the virtual interviewer (Ellie) from the published DAIC-WOZ interview
# protocol, in a wording that participants rarely use themselves.
INTERVIEWER_PHRASES: Final = (
    "thanks for coming in today",
    "safe and secure environment",
    "your answers are totally confidential",
    "where are you from originally",
    "what do you do to relax",
    "how easy is it for you to get a good nights sleep",
    "have you been diagnosed with depression",
    "what are you most proud of in your life",
    "what advice would you give yourself",
    "the last time you felt really happy",
)
FILLERS: Final = frozenset({"um", "uh", "umm", "uhm", "mm", "hmm", "mhm", "erm"})
TYPOGRAPHIC_APOSTROPHE: Final = chr(0x2019)
LOW_CONFIDENCE: Final = 0.5


def normalise(text: str) -> str:
    """Return lower-case text with letters, digits and single spaces only.

    Apostrophes are dropped (``I'm`` becomes ``im``); every other character
    that is not a letter or digit becomes a space.

    Args:
        text: Raw transcript text.

    Returns:
        The normalised text.
    """
    text = text.lower().replace("'", "").replace(TYPOGRAPHIC_APOSTROPHE, "")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def read_transcript(path: Path) -> pd.DataFrame:
    """Read one E-DAIC transcript.

    Args:
        path: A ``NNN_Transcript.csv`` file.

    Returns:
        One row per line, sorted by start time, with the columns ``start`` and
        ``end`` (seconds), ``text`` and ``confidence`` (NaN where it is missing).

    Raises:
        ValueError: If the columns are not those of the E-DAIC format, or a start
            or end time is not a number.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(raw.columns) != list(COLUMNS):
        raise ValueError(f"{path.name}: unexpected columns {list(raw.columns)}")
    frame = raw.rename(columns=COLUMNS)
    frame["start"] = pd.to_numeric(frame["start"])
    frame["end"] = pd.to_numeric(frame["end"])
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    return frame.sort_values("start", kind="stable").reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class TranscriptSummary:
    """What one transcript contains, as numbers only (no text).

    Attributes:
        lines: Number of lines.
        words: Number of words after normalisation, fillers included.
        speech_seconds: Total time inside the lines.
        first_start: Start of the first line, in seconds.
        last_end: Latest end of any line, in seconds.
        overlaps: Lines that start before the previous line ends.
        gaps: Pause between each pair of consecutive lines, in seconds (0 where they overlap).
        confidences: Confidence of each line that has one.
        missing_confidence: Lines without a confidence value.
        fillers: Filler words such as "um" and "uh".
        tags: Lines with a bracketed tag such as ``<laughter>``.
        interviewer_phrases: The interviewer phrases that occur in the transcript.
    """

    lines: int
    words: int
    speech_seconds: float
    first_start: float
    last_end: float
    overlaps: int
    gaps: tuple[float, ...]
    confidences: tuple[float, ...]
    missing_confidence: int
    fillers: int
    tags: int
    interviewer_phrases: frozenset[str]


def summarise(frame: pd.DataFrame) -> TranscriptSummary:
    """Summarise one transcript as read by ``read_transcript``.

    Args:
        frame: The transcript lines, sorted by start time.

    Returns:
        Counts and timings of the transcript.
    """
    texts = [normalise(text) for text in frame["text"]]
    tokens = [token for text in texts for token in text.split()]
    joined = f" {' '.join(texts)} "
    starts = frame["start"].to_numpy(dtype=float)
    ends = frame["end"].to_numpy(dtype=float)
    confidence = frame["confidence"].to_numpy(dtype=float)
    return TranscriptSummary(
        lines=len(frame),
        words=len(tokens),
        speech_seconds=float(np.clip(ends - starts, 0.0, None).sum()),
        first_start=float(starts.min()) if len(frame) else 0.0,
        last_end=float(ends.max()) if len(frame) else 0.0,
        overlaps=int((starts[1:] < ends[:-1]).sum()),
        gaps=tuple(float(gap) for gap in np.clip(starts[1:] - ends[:-1], 0.0, None)),
        confidences=tuple(float(value) for value in confidence[~np.isnan(confidence)]),
        missing_confidence=int(np.isnan(confidence).sum()),
        fillers=sum(token in FILLERS for token in tokens),
        tags=int(frame["text"].str.contains(r"[<\[]", regex=True).sum()),
        interviewer_phrases=frozenset(p for p in INTERVIEWER_PHRASES if f" {p} " in joined),
    )
