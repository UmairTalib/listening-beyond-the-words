"""Segments: exactly 40 consecutive kept participant words, with parts joined by 300 ms gaps.

A segment is cut by words, not by time, so the number of words per clip cannot
carry the speaking rate into the words-only view. Within a part the original
audio is kept, pauses included; between parts the audio is replaced by a fixed
gap. A part is a run of kept words from one transcript line with no dropped word
and no interviewer turn between them (``cut.py``, column ``part``); words without
a part column are split into parts by line. Words left over at the end of a
session (fewer than 40) are not used.
"""

import math
from collections.abc import Mapping
from typing import Final, cast

import numpy as np
import pandas as pd

SEGMENT_WORDS: Final = 40
GAP_SECONDS: Final = 0.3


def in_segment_order(words: pd.DataFrame) -> pd.DataFrame:
    """Sort the words in the order in which segments take them: by session, then by time.

    Words that start at the same time are ordered by their position in the line, then by
    line, so the order never depends on the order of the rows.
    """
    return words.sort_values(["session", "start", "word_index", "line"], kind="stable")


def _kept(group: pd.DataFrame) -> np.ndarray:
    """Which words of a table are kept: the ``kept`` column, or every word if there is none."""
    if "kept" in group.columns:
        return group["kept"].to_numpy(dtype=bool)
    return np.ones(len(group), dtype=bool)


def build_segments(
    words: pd.DataFrame, n_words: int = SEGMENT_WORDS, gap: float = GAP_SECONDS
) -> pd.DataFrame:
    """Cut each session's kept words into consecutive segments of exactly ``n_words``.

    Args:
        words: One row per aligned word, with the columns ``session``, ``line``,
            ``word_index``, ``word``, ``start`` and ``end`` (seconds in the session),
            and optionally ``kept`` (only kept words are used) and ``part`` (the part
            key from ``cut.py``; without it, each line is a part).
        n_words: Words per segment.
        gap: Seconds of silence between two parts of one segment.

    Returns:
        One row per segment with the columns ``segment``, ``session``, ``index``,
        ``starts`` and ``ends`` (the audio parts), ``parts``, ``seconds`` (audio plus
        gaps) and ``text``.

    Raises:
        ValueError: If the words have a ``kept`` column but no ``part`` column, which
            would join the words on both sides of a dropped word into one part.
    """
    if "kept" in words.columns and "part" not in words.columns:
        raise ValueError("words with a kept column need their part keys (cut.cut_words)")
    ordered = in_segment_order(words)
    ordered = ordered[_kept(ordered)]
    key = "part" if "part" in ordered.columns else "line"
    rows = []
    for session_key, group in ordered.groupby("session", sort=True):
        session = int(cast(int, session_key))
        usable = len(group) - len(group) % n_words
        for index, first in enumerate(range(0, usable, n_words)):
            chunk = group.iloc[first : first + n_words]
            parts = chunk.groupby(key, sort=False).agg(start=("start", "min"), end=("end", "max"))
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


def segment_neighbours(words: pd.DataFrame, n_words: int = SEGMENT_WORDS) -> pd.DataFrame:
    """Return, for every segment, the aligned words just outside it and the dropped words inside it.

    The audio cut of a segment uses these times so that its margins stop halfway to
    the nearest other aligned word, kept or dropped, which is often in the same line,
    only a few frames away. The segments are those of ``build_segments`` on the same
    words, in the same order.

    Args:
        words: One row per aligned word, as for ``build_segments``; dropped words
            (``kept`` false) count as neighbours too.
        n_words: Words per segment.

    Returns:
        One row per segment with the columns ``segment``, ``before`` (end of the word
        just before the segment, -inf if there is none), ``after`` (start of the word
        just after it, inf if there is none), ``other_starts`` and ``other_ends`` (the
        dropped words between its first and last word), in seconds, rounded like the
        part times of ``build_segments``.
    """
    rows = []
    for session_key, group in in_segment_order(words).groupby("session", sort=True):
        session = int(cast(int, session_key))
        starts = group["start"].to_numpy(dtype=float)
        ends = group["end"].to_numpy(dtype=float)
        kept = _kept(group)
        position = np.flatnonzero(kept)
        usable = len(position) - len(position) % n_words
        for index, first in enumerate(range(0, usable, n_words)):
            low, high = position[first], position[first + n_words - 1]
            between = np.arange(low + 1, high)
            others = between[~kept[between]]
            rows.append(
                {
                    "segment": f"{session}_{index:03d}",
                    "before": float(np.round(ends[low - 1], 3)) if low > 0 else -math.inf,
                    "after": (
                        float(np.round(starts[high + 1], 3)) if high + 1 < len(group) else math.inf
                    ),
                    "other_starts": np.round(starts[others], 3).tolist(),
                    "other_ends": np.round(ends[others], 3).tolist(),
                }
            )
    return pd.DataFrame(rows, columns=["segment", "before", "after", "other_starts", "other_ends"])


def segment_walls(segments: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    """Return, for every segment, the zones next to its parts, which its cut must not enter.

    For each part these are the zone that ends last at or before the part's start and
    the zone that starts first at or after the part's end. They are the only zones that
    can stop a margin of the part, whatever the margin.

    Args:
        segments: Segments from ``build_segments``.
        zones: One row per zone (interviewer turn or widened interruption) with the
            columns ``session``, ``start`` and ``end`` in seconds (``cut.zone_table``).

    Returns:
        One row per segment with the columns ``segment``, ``wall_starts`` and
        ``wall_ends``, in time order and without repeats.

    Raises:
        ValueError: If a zone overlaps a part.
    """
    empty = np.empty(0)
    by_session = {
        int(cast(int, session)): (
            group["start"].to_numpy(dtype=float),
            group["end"].to_numpy(dtype=float),
        )
        for session, group in zones.groupby("session", sort=True)
    }
    rows = []
    for segment, session, part_starts, part_ends in zip(
        segments["segment"].tolist(),
        segments["session"].tolist(),
        segments["starts"].tolist(),
        segments["ends"].tolist(),
        strict=True,
    ):
        zone_starts, zone_ends = by_session.get(int(session), (empty, empty))
        chosen: set[int] = set()
        for start, end in zip(part_starts, part_ends, strict=True):
            if ((zone_starts < end) & (zone_ends > start)).any():
                raise ValueError(f"segment {segment}: a zone overlaps a part")
            earlier = np.flatnonzero(zone_ends <= start)
            if len(earlier):
                chosen.add(int(earlier[np.argmax(zone_ends[earlier])]))
            later = np.flatnonzero(zone_starts >= end)
            if len(later):
                chosen.add(int(later[np.argmin(zone_starts[later])]))
        order = sorted(chosen, key=lambda i: (zone_starts[i], zone_ends[i]))
        rows.append(
            {
                "segment": segment,
                "wall_starts": [float(zone_starts[i]) for i in order],
                "wall_ends": [float(zone_ends[i]) for i in order],
            }
        )
    return pd.DataFrame(rows, columns=["segment", "wall_starts", "wall_ends"])


def segment_bounds(segments: pd.DataFrame, turns: Mapping[int, np.ndarray]) -> pd.DataFrame:
    """Return, for every segment, the stretch that each part's window must stay inside.

    In the sessions given in ``turns`` (those without interviewer turns), a part's window
    stays inside the participant turn that holds its words, widened to the part's own
    words where a word runs past the turn. Elsewhere a part is unbounded.

    Args:
        segments: Segments from ``build_segments``.
        turns: For each bounded session, its participant turns in seconds, merged and in
            time order, shape (n, 2).

    Returns:
        One row per segment with the columns ``segment``, ``bound_starts`` and
        ``bound_ends`` (-inf and inf where a part is unbounded).

    Raises:
        ValueError: If the middle of a part lies in no turn of its session.
    """
    rows = []
    for segment, session, part_starts, part_ends in zip(
        segments["segment"].tolist(),
        segments["session"].tolist(),
        segments["starts"].tolist(),
        segments["ends"].tolist(),
        strict=True,
    ):
        lows, highs = [-math.inf] * len(part_starts), [math.inf] * len(part_starts)
        session_turns = turns.get(int(session))
        if session_turns is not None:
            for i, (start, end) in enumerate(zip(part_starts, part_ends, strict=True)):
                # The part's middle lies between its first and last word's middles, so it
                # lies in the turn that holds them all.
                middle = (start + end) / 2
                k = int(np.searchsorted(session_turns[:, 0], middle, side="right")) - 1
                if k < 0 or middle > session_turns[k, 1]:
                    raise ValueError(f"segment {segment}: a part lies in no participant turn")
                lows[i] = min(float(session_turns[k, 0]), float(start))
                highs[i] = max(float(session_turns[k, 1]), float(end))
        rows.append({"segment": segment, "bound_starts": lows, "bound_ends": highs})
    return pd.DataFrame(rows, columns=["segment", "bound_starts", "bound_ends"])
