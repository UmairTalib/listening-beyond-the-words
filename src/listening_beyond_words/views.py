"""Views of a segment. So far the original clip, x0.

The original clip is the participant's own audio. Each part of a segment (a run of
its kept words from one transcript line, ``cut.py``) is cut from the session recording, and the parts
are joined by exactly 300 ms of silence (pilot protocol, fixed settings).

The aligner's word times end a little early, so each part keeps a margin of audio
before its first word and after its last word. The margin never reaches more than
halfway to the nearest other aligned participant word, kept or dropped by the cut
rule, which is often in the same line only a few frames away, and never past the
recording. It stops a little short of the interviewer's turns and the widened
interruptions, and in the sessions without interviewer turns it stops at the edge of
the participant's turn. Each part fades in and out so that the joins do not click; where the
full margin fits, the fade lies inside it and never touches a word. The margin, the
fade and the distance to the interviewer's turns are set in ``configs/pilot/data.yaml``
(pilot protocol, sections "Speech used" and "Interviewer detector").
"""

import math
from collections.abc import Sequence
from typing import Any, Final

import numpy as np

from listening_beyond_words.segments import GAP_SECONDS

SAMPLING_RATE: Final = 16_000
# Word ends from the aligner can lie up to about 0.6 ms past the last sample.
END_TOLERANCE_SECONDS: Final = 0.001

Window = tuple[float, float]


def part_windows(
    starts: Sequence[float],
    ends: Sequence[float],
    duration: float,
    *,
    margin: float,
    before: float = -math.inf,
    after: float = math.inf,
    others: Sequence[Window] = (),
    walls: Sequence[Window] = (),
    wall_guard: float = 0.0,
    bounds: Sequence[Window] | None = None,
) -> list[Window]:
    """Return the stretch of the recording that each part of a segment keeps.

    A part keeps its words plus a margin on each side. The margin is at most
    ``margin`` and at most half the gap to the nearest other aligned participant
    word: the neighbouring part of the segment, a dropped word, or the word just
    outside the segment. It also stops ``wall_guard`` short of a wall (an
    interviewer turn or a widened interruption).

    Args:
        starts: Start of each part's first word, in seconds in the session, in time order.
        ends: End of each part's last word, in seconds in the session.
        duration: Length of the session recording in seconds.
        margin: Most audio kept on each side of a part, in seconds.
        before: End of the word just before the segment; -inf if there is none.
        after: Start of the word just after the segment; inf if there is none.
        others: (start, end) of other words that the segment does not use, such as the
            words that the cut rule dropped between its parts.
        walls: (start, end) of stretches that no window may enter.
        wall_guard: Distance in seconds that every window keeps from every wall.
        bounds: For each part, the (start, end) stretch its window must stay inside,
            such as its participant turn in the sessions without interviewer turns;
            None leaves every part unbounded. A bound must hold the part's words.

    Returns:
        One (start, end) window per part, in seconds, inside the recording.

    Raises:
        ValueError: If there are no parts, a part has no end, the parts are not in
            time order or overlap their neighbours or another word, a part comes
            within ``wall_guard`` of a wall, a bound does not hold its part's words, or
            a part lies outside the recording.
    """
    starts = [float(start) for start in starts]
    ends = [float(end) for end in ends]
    if not starts or len(starts) != len(ends):
        raise ValueError(f"expected one end per start and at least one part: {starts}, {ends}")
    previous = [float(before), *ends[:-1]]
    following = [*starts[1:], float(after)]
    for other_start, other_end in others:
        for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
            if other_start < end and other_end > start:
                raise ValueError(
                    f"another word {other_start}-{other_end} overlaps part {start}-{end}"
                )
            if other_end <= start:
                previous[i] = max(previous[i], float(other_end))
            if other_start >= end:
                following[i] = min(following[i], float(other_start))
    floor = [-math.inf] * len(starts)
    ceiling = [math.inf] * len(starts)
    for wall_start, wall_end in walls:
        for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
            if wall_start < end + wall_guard and wall_end > start - wall_guard:
                raise ValueError(
                    f"part {start}-{end} comes within {wall_guard} s of a wall {wall_start}-{wall_end}"
                )
            if wall_end <= start:
                floor[i] = max(floor[i], float(wall_end) + wall_guard)
            if wall_start >= end:
                ceiling[i] = min(ceiling[i], float(wall_start) - wall_guard)
    if bounds is not None:
        if len(bounds) != len(starts):
            raise ValueError(f"expected one bound per part, got {len(bounds)} for {len(starts)}")
        for i, ((bound_start, bound_end), start, end) in enumerate(
            zip(bounds, starts, ends, strict=True)
        ):
            if not bound_start <= start <= end <= bound_end:
                raise ValueError(
                    f"bound {bound_start}-{bound_end} does not hold part {start}-{end}"
                )
            floor[i] = max(floor[i], float(bound_start))
            ceiling[i] = min(ceiling[i], float(bound_end))
    windows = []
    for start, end, previous_end, next_start, lowest, highest in zip(
        starts, ends, previous, following, floor, ceiling, strict=True
    ):
        if not previous_end <= start <= end <= next_start:
            raise ValueError(
                f"part {start}-{end} is out of time order or overlaps a neighbouring word"
                f" (previous word ends {previous_end}, next word starts {next_start})"
            )
        if start < 0 or end > duration + END_TOLERANCE_SECONDS:
            raise ValueError(f"part {start}-{end} lies outside the {duration} s recording")
        low = max(0.0, start - min(margin, (start - previous_end) / 2), lowest)
        high = min(duration, end + min(margin, (next_start - end) / 2), highest)
        if high <= low:
            raise ValueError(f"part {start}-{end} lies outside the {duration} s recording")
        windows.append((low, high))
    return windows


def segment_windows(
    segment: Any, duration: float, *, margin: float, wall_guard: float
) -> list[Window]:
    """Return the window of each part of one segment, from its row of the segments table.

    Args:
        segment: A row of ``segments.parquet`` (``scripts/build_segments.py``), such as a
            named tuple from ``itertuples``, with the fields ``starts``, ``ends``,
            ``before``, ``after``, ``other_starts``, ``other_ends``, ``wall_starts``,
            ``wall_ends``, ``bound_starts`` and ``bound_ends``.
        duration: Length of the session recording in seconds.
        margin: Most audio kept on each side of a part, in seconds.
        wall_guard: Distance in seconds that every window keeps from every wall.

    Returns:
        One (start, end) window per part, from ``part_windows``.
    """
    return part_windows(
        segment.starts,
        segment.ends,
        duration,
        margin=margin,
        before=segment.before,
        after=segment.after,
        others=list(zip(segment.other_starts, segment.other_ends, strict=True)),
        walls=list(zip(segment.wall_starts, segment.wall_ends, strict=True)),
        wall_guard=wall_guard,
        bounds=list(zip(segment.bound_starts, segment.bound_ends, strict=True)),
    )


def windows_overlap(first: Sequence[Window], second: Sequence[Window]) -> bool:
    """Return whether any window of ``first`` shares audio with any window of ``second``.

    Windows that only touch, one ending exactly where the other starts, share no audio.

    Args:
        first: (start, end) windows in seconds.
        second: (start, end) windows in seconds, of the same recording.

    Returns:
        True if some pair of windows overlaps.
    """
    return any(
        low < other_high and high > other_low
        for low, high in first
        for other_low, other_high in second
    )


def cut_original(
    audio: np.ndarray,
    windows: Sequence[Window],
    *,
    fade: float,
    rate: int = SAMPLING_RATE,
    gap: float = GAP_SECONDS,
) -> np.ndarray:
    """Cut the original clip (x0) of one segment from its session recording.

    Args:
        audio: The whole session recording as floats in [-1, 1], mono, shape (samples,).
        windows: The window of each part in seconds, from ``part_windows``.
        fade: Seconds of fade-in at the start and fade-out at the end of each part.
        rate: Sampling rate of ``audio`` in Hz.
        gap: Seconds of silence between two parts.

    Returns:
        The clip as float32, shape (samples,). The recording itself is not changed.

    Raises:
        ValueError: If ``audio`` is not mono or not floating point, or there are no windows.
    """
    if audio.ndim != 1:
        raise ValueError(f"expected mono audio of shape (samples,), got shape {audio.shape}")
    if not np.issubdtype(audio.dtype, np.floating):
        raise ValueError(f"expected audio as floats in [-1, 1], got {audio.dtype}")
    if len(windows) == 0:
        raise ValueError("a clip needs at least one part")
    silence = np.zeros(round(gap * rate), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for low, high in windows:
        piece = np.array(audio[round(low * rate) : round(high * rate)], dtype=np.float32)
        n = min(round(fade * rate), len(piece) // 2)
        if n > 0:
            # Raised-cosine ramp from 0, so each part starts and ends at exactly 0.
            ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)).astype(np.float32)
            piece[:n] *= ramp
            piece[-n:] *= ramp[::-1]
        if pieces:
            pieces.append(silence)
        pieces.append(piece)
    return np.concatenate(pieces)
