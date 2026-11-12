"""Segments: exactly 40 consecutive participant words, with turns joined by 300 ms gaps.

A segment is cut by words, not by time, so the number of words per clip cannot
carry the speaking rate into the words-only view. Within a line the original
audio is kept, pauses included; between lines the audio is replaced by a fixed
gap. Words left over at the end of a session (fewer than 40) are not used.
"""

from typing import Final, cast

import pandas as pd

SEGMENT_WORDS: Final = 40
GAP_SECONDS: Final = 0.3


def build_segments(
    words: pd.DataFrame, n_words: int = SEGMENT_WORDS, gap: float = GAP_SECONDS
) -> pd.DataFrame:
    """Cut each session's words into consecutive segments of exactly ``n_words``.

    Args:
        words: One row per aligned word, with the columns ``session``, ``line``,
            ``word_index``, ``word``, ``start`` and ``end`` (seconds in the session).
        n_words: Words per segment.
        gap: Seconds of silence between two lines of one segment.

    Returns:
        One row per segment with the columns ``segment``, ``session``, ``index``,
        ``starts`` and ``ends`` (the audio parts, one per line), ``parts``,
        ``seconds`` (audio plus gaps) and ``text``.
    """
    rows = []
    ordered = words.sort_values(["session", "start", "word_index"], kind="stable")
    for key, group in ordered.groupby("session", sort=True):
        session = int(cast(int, key))
        usable = len(group) - len(group) % n_words
        for index, first in enumerate(range(0, usable, n_words)):
            chunk = group.iloc[first : first + n_words]
            parts = chunk.groupby("line", sort=False).agg(
                start=("start", "min"), end=("end", "max")
            )
            parts = parts.sort_values("start")
            seconds = float((parts["end"] - parts["start"]).sum()) + gap * (len(parts) - 1)
            rows.append(
                {
                    "segment": f"{session}_{index:03d}",
                    "session": session,
                    "index": index,
                    "starts": parts["start"].round(3).tolist(),
                    "ends": parts["end"].round(3).tolist(),
                    "parts": len(parts),
                    "seconds": round(seconds, 3),
                    "text": " ".join(chunk["word"]),
                }
            )
    return pd.DataFrame(rows)
