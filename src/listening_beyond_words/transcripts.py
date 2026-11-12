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
# protocol: her questions, follow-ups and comments, in her own wording, which a
# participant does not use when answering (a participant says "my life", Ellie says
# "your life"; a participant may say "diagnosed with PTSD", so only Ellie's whole
# question is listed). Normalised as by ``normalise``.
INTERVIEWER_PHRASES: Final = (
    # Opening
    "thanks for coming in today",
    "safe and secure environment",
    "your answers are totally confidential",
    "created to talk to people",
    "think of me as a friend",
    "here to learn about people",
    "ask a few questions to get us started",
    "are you okay with this",
    "are you ok with this",
    "how are you doing today",
    # Questions
    "where are you from originally",
    "why did you move to la",
    "why did you move to l a",
    "whyd you move to la",
    "how do you like la",
    "how do you like l a",
    "things you really like about la",
    "things you really like about l a",
    "things you dont really like about la",
    "things you dont really like about l a",
    "how often do you go back to your hometown",
    "what did you study at school",
    "whats your dream job",
    "what is your dream job",
    "do you travel a lot",
    "what do you enjoy about traveling",
    "one of your trips",
    "do you consider yourself an introvert",
    "consider yourself more shy or outgoing",
    "what do you do to relax",
    "how are you at controlling your temper",
    "the last time you argued with someone",
    "how did you feel in that moment",
    "how close are you to them",
    "one of your most memorable experiences",
    "how easy is it for you to get a good nights sleep",
    "what are you like when you dont sleep well",
    "how have you been feeling lately",
    "changes in your behavior or thoughts",
    "have you been diagnosed with depression",
    "have you ever been diagnosed with p t s d",
    "have you ever been diagnosed with ptsd",
    "have you ever served in the military",
    "how long ago were you diagnosed",
    "what got you to seek help",
    "do you still go to therapy",
    "do you feel like therapy is useful",
    "the last time you felt really happy",
    "what advice would you give yourself",
    "what are you most proud of in your life",
    "how would your best friend describe you",
    "how would your best friends describe you",
    "is there anything you regret",
    "whats something you feel guilty about",
    "what is something you feel guilty about",
    "some of your best qualities",
    "usually put you in a good mood",
    "positive influence in your life",
    "things that make you really mad",
    # Follow-ups and comments
    "tell me more about that",
    "can you tell me about that",
    "give me an example of that",
    "that sounds really hard",
    "im sorry to hear that",
    "i see what you mean",
    "thats so good to hear",
    "that sounds like a great situation",
    # Closing
    "thanks for sharing your thoughts",
    "asked everything i need",
    "it was great chatting with you",
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
