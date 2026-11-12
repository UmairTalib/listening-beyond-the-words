"""Views of a segment. So far the original clip, x0.

The original clip is the participant's own audio. Each part of a segment (its
words from one transcript line) is cut from the session recording, and the parts
are joined by exactly 300 ms of silence (pilot protocol, fixed settings).

The aligner's word times end a little early, so each part keeps a margin of audio
before its first word and after its last word. The margin never reaches more than
halfway to the nearest other kept participant word, which is often in the same
line only a few frames away, and never past the recording. Each part fades in and
out so that the joins do not click; where the full margin fits, the fade lies
inside it and never touches a word. The margin and the fade are set in
``configs/pilot/data.yaml`` (pilot protocol, section "Speech used").
"""

import math
from collections.abc import Sequence
from typing import Final

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
) -> list[Window]:
    """Return the stretch of the recording that each part of a segment keeps.

    A part keeps its words plus a margin on each side. The margin is at most
    ``margin`` and at most half the gap to the nearest other kept participant word:
    the neighbouring part of the segment, or the word just outside the segment.

    Args:
        starts: Start of each part's first word, in seconds in the session, in time order.
        ends: End of each part's last word, in seconds in the session.
        duration: Length of the session recording in seconds.
        margin: Most audio kept on each side of a part, in seconds.
        before: End of the kept word just before the segment; -inf if there is none.
        after: Start of the kept word just after the segment; inf if there is none.

    Returns:
        One (start, end) window per part, in seconds, inside the recording.

    Raises:
        ValueError: If there are no parts, a part has no end, the parts are not in
            time order or overlap their neighbours, or a part lies outside the recording.
    """
    starts = [float(start) for start in starts]
    ends = [float(end) for end in ends]
    if not starts or len(starts) != len(ends):
        raise ValueError(f"expected one end per start and at least one part: {starts}, {ends}")
    previous = [float(before), *ends[:-1]]
    following = [*starts[1:], float(after)]
    windows = []
    for start, end, previous_end, next_start in zip(starts, ends, previous, following, strict=True):
        if not previous_end <= start <= end <= next_start:
            raise ValueError(
                f"part {start}-{end} is out of time order or overlaps a neighbouring word"
                f" (previous word ends {previous_end}, next word starts {next_start})"
            )
        if start < 0 or end > duration + END_TOLERANCE_SECONDS:
            raise ValueError(f"part {start}-{end} lies outside the {duration} s recording")
        low = max(0.0, start - min(margin, (start - previous_end) / 2))
        high = min(duration, end + min(margin, (next_start - end) / 2))
        if high <= low:
            raise ValueError(f"part {start}-{end} lies outside the {duration} s recording")
        windows.append((low, high))
    return windows


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
