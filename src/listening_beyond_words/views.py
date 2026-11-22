"""Views of a segment: the original clip x0, the muffled clip x2 and the reversed clip rev.

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

The other views are built from the same parts (pilot protocol, section "Clip
versions"), so every view has the same pieces at the same places and the same silent
gaps. x2 filters each faded part of x0 on its own with a low-pass, fades it again
and joins the parts; the clip then gets the RMS of x0. rev plays each part's audio,
before its fades, backwards in 250 ms windows from the part's first sample, then fades
it like x0. The settings are in ``configs/pilot/data.yaml`` (``views``).
"""

import functools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from scipy.signal import butter, sosfiltfilt

from listening_beyond_words.segments import GAP_SECONDS

SAMPLING_RATE: Final = 16_000
PCM16_SCALE: Final = 32_768  # a 16-bit sample n stands for n / 32768
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


def cut_parts(
    audio: np.ndarray, windows: Sequence[Window], *, rate: int = SAMPLING_RATE
) -> list[np.ndarray]:
    """Cut the audio of each part of one segment from its session recording, before any fade.

    Args:
        audio: The whole session recording as floats in [-1, 1], mono, shape (samples,).
        windows: The window of each part in seconds, from ``part_windows``.
        rate: Sampling rate of ``audio`` in Hz.

    Returns:
        A float32 copy of each part's window, in the order of ``windows``. The recording
        itself is not changed.

    Raises:
        ValueError: If ``audio`` is not mono or not floating point, or there are no windows.
    """
    if audio.ndim != 1:
        raise ValueError(f"expected mono audio of shape (samples,), got shape {audio.shape}")
    if not np.issubdtype(audio.dtype, np.floating):
        raise ValueError(f"expected audio as floats in [-1, 1], got {audio.dtype}")
    if len(windows) == 0:
        raise ValueError("a clip needs at least one part")
    return [
        np.array(audio[round(low * rate) : round(high * rate)], dtype=np.float32)
        for low, high in windows
    ]


def fade_part(part: np.ndarray, *, fade: float, rate: int = SAMPLING_RATE) -> np.ndarray:
    """Return a copy of one part with x0's raised-cosine fades at both ends.

    The fade lasts ``fade`` seconds, or half the part if the part is shorter. The ramp
    starts at 0, so a part of two samples or more starts and ends at exactly 0. It is
    computed in the part's own float type: float32 parts are faded exactly as x0 always was.

    Args:
        part: One part as floats, shape (samples,).
        fade: Seconds of fade-in at the start and fade-out at the end.
        rate: Sampling rate in Hz.

    Returns:
        The faded copy, in the type of ``part``. ``part`` itself is not changed.

    Raises:
        ValueError: If ``part`` is not one-dimensional floats.
    """
    if part.ndim != 1 or not np.issubdtype(part.dtype, np.floating):
        raise ValueError(
            f"expected a part as floats of shape (samples,), got {part.dtype} {part.shape}"
        )
    piece = part.copy()
    n = min(round(fade * rate), len(piece) // 2)
    if n > 0:
        # Raised-cosine ramp from 0, so each part starts and ends at exactly 0.
        ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)).astype(piece.dtype)
        piece[:n] *= ramp
        piece[-n:] *= ramp[::-1]
    return piece


def join_parts(
    parts: Sequence[np.ndarray], *, rate: int = SAMPLING_RATE, gap: float = GAP_SECONDS
) -> tuple[np.ndarray, list[int], list[int]]:
    """Join the parts of one clip with silence between them.

    Args:
        parts: The parts in clip order, all of one float type, each of shape (samples,).
        rate: Sampling rate in Hz.
        gap: Seconds of silence between two parts: exactly ``round(gap * rate)`` zeros.

    Returns:
        The clip, in the parts' type; the first sample of each part in the clip; and the
        sample just after each part, so that ``clip[start:end]`` is the part.

    Raises:
        ValueError: If there are no parts or the parts differ in type.
    """
    if len(parts) == 0:
        raise ValueError("a clip needs at least one part")
    kinds = {part.dtype for part in parts}
    if len(kinds) != 1:
        raise ValueError(f"the parts of a clip must share one type, got {sorted(map(str, kinds))}")
    silence = np.zeros(round(gap * rate), dtype=parts[0].dtype)
    pieces: list[np.ndarray] = []
    starts: list[int] = []
    ends: list[int] = []
    position = 0
    for part in parts:
        if pieces:
            pieces.append(silence)
            position += len(silence)
        starts.append(position)
        pieces.append(part)
        position += len(part)
        ends.append(position)
    return np.concatenate(pieces), starts, ends


def cut_original(
    audio: np.ndarray,
    windows: Sequence[Window],
    *,
    fade: float,
    rate: int = SAMPLING_RATE,
    gap: float = GAP_SECONDS,
) -> np.ndarray:
    """Cut the original clip (x0) of one segment from its session recording.

    Each part is cut (``cut_parts``), faded (``fade_part``) and joined to the next by
    the gap (``join_parts``).

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
    parts = cut_parts(audio, windows, rate=rate)
    clip, _, _ = join_parts(
        [fade_part(part, fade=fade, rate=rate) for part in parts], rate=rate, gap=gap
    )
    return clip


@functools.lru_cache(maxsize=8)
def _lowpass(cutoff_hz: float, order: int, rate: int) -> tuple[np.ndarray, int]:
    """The Butterworth low-pass as second-order sections, and scipy's default padding for it."""
    sos = butter(order, cutoff_hz, btype="lowpass", fs=rate, output="sos")
    # sosfiltfilt pads each end by 3 x (2 x sections + 1) samples, less any zero coefficients.
    taps = 2 * len(sos) + 1 - min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
    return sos, 3 * taps


def muffle(
    part: np.ndarray, *, cutoff_hz: float, order: int, rate: int = SAMPLING_RATE
) -> np.ndarray:
    """Low-pass one part of x2: a Butterworth filter run forwards and then backwards.

    Each pass has the given order and its -3 dB point at ``cutoff_hz``; the two passes
    together are 6 dB down there and shift nothing in time. The ends are padded as scipy
    does by default (odd extension). The work is done in float64.

    Args:
        part: One part as floats, shape (samples,).
        cutoff_hz: The -3 dB point of each pass in Hz.
        order: The order of each pass.
        rate: Sampling rate in Hz.

    Returns:
        The filtered part as float64, as long as ``part``. ``part`` itself is not changed.

    Raises:
        ValueError: If ``part`` is not one-dimensional floats, or too short for the
            filter's padding (27 samples or fewer for order 8).
    """
    if part.ndim != 1 or not np.issubdtype(part.dtype, np.floating):
        raise ValueError(
            f"expected a part as floats of shape (samples,), got {part.dtype} {part.shape}"
        )
    sos, pad = _lowpass(float(cutoff_hz), int(order), int(rate))
    if len(part) <= pad:
        raise ValueError(
            f"a part of {len(part)} samples is too short for the x2 filter, which needs"
            f" more than {pad}"
        )
    # A copy of the cached sections, so that no call can change them for the next.
    return np.asarray(sosfiltfilt(sos.copy(), part.astype(np.float64)), dtype=np.float64)


def reverse_windows(part: np.ndarray, *, window: int) -> np.ndarray:
    """Play one part backwards window by window (rev).

    The windows start at the part's first sample; a shorter last window is reversed too.

    Args:
        part: One part, shape (samples,).
        window: Samples per window.

    Returns:
        A new array of the same type and length. ``part`` itself is not changed.

    Raises:
        ValueError: If ``part`` is not one-dimensional or the window is not positive.
    """
    if part.ndim != 1:
        raise ValueError(f"expected a part of shape (samples,), got shape {part.shape}")
    if window < 1:
        raise ValueError(f"a window needs at least one sample, got {window}")
    reversed_part = np.empty_like(part)
    for start in range(0, len(part), window):
        reversed_part[start : start + window] = part[start : start + window][::-1]
    return reversed_part


def reverse_whole(part: np.ndarray) -> np.ndarray:
    """Play one whole part backwards: the backup of check F4, built only if rev fails it.

    Args:
        part: One part, shape (samples,).

    Returns:
        A reversed copy. ``part`` itself is not changed.

    Raises:
        ValueError: If ``part`` is not one-dimensional.
    """
    if part.ndim != 1:
        raise ValueError(f"expected a part of shape (samples,), got shape {part.shape}")
    return part[::-1].copy()


def clip_rms(clip: np.ndarray) -> float:
    """Return the RMS of a whole clip, gaps included, computed in float64.

    Args:
        clip: A clip as floats, shape (samples,).

    Returns:
        The RMS; NaN if the clip holds NaN.

    Raises:
        ValueError: If the clip is empty.
    """
    if clip.size == 0:
        raise ValueError("an empty clip has no RMS")
    return float(np.sqrt(np.mean(np.square(clip, dtype=np.float64))))


def clip_peak(clip: np.ndarray) -> float:
    """Return the largest absolute value of a whole clip, computed in float64.

    Args:
        clip: A clip as floats, shape (samples,).

    Returns:
        The peak; NaN if the clip holds NaN.

    Raises:
        ValueError: If the clip is empty.
    """
    if clip.size == 0:
        raise ValueError("an empty clip has no peak")
    return float(np.max(np.abs(clip.astype(np.float64))))


def set_rms(clip: np.ndarray, target: float) -> tuple[np.ndarray, float]:
    """Scale a whole clip to a target RMS.

    Args:
        clip: A clip as floats, shape (samples,).
        target: The RMS it should have, above 0.

    Returns:
        The scaled clip as float64 and the gain applied to it.

    Raises:
        ValueError: If the target is not a positive finite number, or the clip is silent
            or holds NaN or inf.
    """
    if not (math.isfinite(target) and target > 0.0):
        raise ValueError(f"the target RMS must be a positive number, got {target}")
    if not np.all(np.isfinite(clip)):
        raise ValueError("the clip holds NaN or inf")
    rms = clip_rms(clip)
    if rms == 0.0:
        raise ValueError("a silent clip has no RMS to set")
    gain = target / rms
    return clip.astype(np.float64) * gain, gain


def limit_peak(clip: np.ndarray, limit: float) -> tuple[np.ndarray, float, bool]:
    """Scale a whole clip down to a peak of ``limit``, only if its peak is above it.

    Args:
        clip: A clip as floats, shape (samples,).
        limit: The highest peak allowed, above 0.

    Returns:
        The clip (scaled, or unchanged as given), the factor applied (1.0 if none), and
        whether the clip was scaled.

    Raises:
        ValueError: If the limit is not positive, or the clip holds NaN or inf.
    """
    if not (math.isfinite(limit) and limit > 0.0):
        raise ValueError(f"the peak limit must be a positive number, got {limit}")
    if not np.all(np.isfinite(clip)):
        raise ValueError("the clip holds NaN or inf")
    peak = clip_peak(clip)
    if peak > limit:
        factor = limit / peak
        return clip * factor, factor, True
    return clip, 1.0, False


def to_pcm16(clip: np.ndarray) -> np.ndarray:
    """Turn a clip into the 16-bit whole numbers that its file stores: round(x * 32768).

    A 16-bit file read as floats gives n / 32768 exactly, so audio read from a 16-bit
    recording comes back as the same numbers. Values exactly halfway round to even.

    Args:
        clip: A clip as floats in [-1, 1), shape (samples,).

    Returns:
        The samples as int16.

    Raises:
        ValueError: If the clip holds NaN or inf, or a value that rounds outside
            -32768 to 32767; nothing is clipped silently.
    """
    if not np.issubdtype(clip.dtype, np.floating):
        raise ValueError(f"expected a clip as floats, got {clip.dtype}")
    scaled = np.rint(clip.astype(np.float64) * PCM16_SCALE)
    if not np.all(np.isfinite(scaled)):
        raise ValueError("the clip holds NaN or inf")
    if scaled.size and (scaled.min() < -PCM16_SCALE or scaled.max() > PCM16_SCALE - 1):
        raise ValueError(
            f"the clip runs from {scaled.min() / PCM16_SCALE} to {scaled.max() / PCM16_SCALE},"
            " outside the 16-bit range"
        )
    return scaled.astype(np.int16)


def stored_windows(segment: Any) -> list[Window]:
    """Return the x0 window of each part of one segment, as stored with the segments.

    Args:
        segment: A row of ``segments.parquet`` with the fields ``window_starts`` and
            ``window_ends`` (seconds in the session), such as a named tuple from
            ``itertuples``.

    Returns:
        One (start, end) window per part.
    """
    return list(
        zip(map(float, segment.window_starts), map(float, segment.window_ends), strict=True)
    )


@dataclass(frozen=True, slots=True)
class ViewSettings:
    """Settings of x0, x2 and rev, from ``configs/pilot/data.yaml`` (``x0_cut`` and ``views``).

    Attributes:
        fade: Seconds of fade at both ends of each part, in every view.
        cutoff_hz: The -3 dB point of each pass of x2's low-pass, in Hz.
        order: The order of each pass of x2's low-pass.
        window: Samples per window of rev.
        peak_limit: Highest peak of an x2 clip after its RMS is set.
        rate: Sampling rate in Hz.
        gap: Seconds of silence between two parts.
    """

    fade: float
    cutoff_hz: float
    order: int
    window: int
    peak_limit: float
    rate: int = SAMPLING_RATE
    gap: float = GAP_SECONDS

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ViewSettings":
        """Read the settings from the whole data config.

        Args:
            config: The data config as loaded from YAML, with ``x0_cut`` and ``views``.

        Returns:
            The settings.

        Raises:
            ValueError: If the files are not 16-bit, the window is not a whole number of
                samples, or a setting is out of range.
        """
        views = config["views"]
        if views["flac_bits"] != 16:
            raise ValueError(f"the views are stored as 16-bit FLAC, not {views['flac_bits']}-bit")
        window_ms = views["rev"]["window_ms"]
        window = round(window_ms * SAMPLING_RATE / 1000)
        if window * 1000 != window_ms * SAMPLING_RATE or window < 1:
            raise ValueError(f"rev's window of {window_ms} ms is not a whole number of samples")
        settings = cls(
            fade=config["x0_cut"]["fade_ms"] / 1000,
            cutoff_hz=float(views["x2"]["cutoff_hz"]),
            order=int(views["x2"]["butterworth_order"]),
            window=window,
            peak_limit=float(views["peak_limit"]),
        )
        if not 0.0 < settings.cutoff_hz < settings.rate / 2:
            raise ValueError(f"x2's cutoff must lie between 0 and {settings.rate / 2} Hz")
        if settings.order < 1:
            raise ValueError(f"x2's filter order must be at least 1, got {settings.order}")
        if not 0.0 < settings.peak_limit < 1.0:
            raise ValueError(f"the peak limit must lie between 0 and 1, got {settings.peak_limit}")
        if settings.fade < 0.0:
            raise ValueError(f"the fade must not be negative, got {settings.fade}")
        return settings


@dataclass(frozen=True, slots=True)
class SegmentViews:
    """The views of one segment, before they are stored.

    Attributes:
        x0: The original clip: the faded parts joined, float32, never rescaled.
        x2: The muffled clip, float64, at the RMS of x0 (lower if its peak was limited).
        rev: The reversed clip, float32.
        part_starts: The first sample of each part, the same in every view.
        part_ends: The sample just after each part, the same in every view.
        x2_gain: The gain applied to x2 after the filter: the RMS gain times the peak factor.
        x2_peak_limited: Whether x2 was scaled down to the peak limit.
    """

    x0: np.ndarray
    x2: np.ndarray
    rev: np.ndarray
    part_starts: tuple[int, ...]
    part_ends: tuple[int, ...]
    x2_gain: float
    x2_peak_limited: bool


def build_segment_views(
    audio: np.ndarray, windows: Sequence[Window], settings: ViewSettings
) -> SegmentViews:
    """Build x0, x2 and rev of one segment from its session recording.

    - x0: each part is cut and faded, and the parts are joined with the gaps.
    - x2: each faded part of x0 is low-passed (``muffle``) and faded again, the parts are
      joined, the clip is set to the RMS of x0, and then scaled down to the peak limit if
      its peak is above it.
    - rev: each part, before its fades, is played backwards in windows
      (``reverse_windows``) and then faded; the parts are joined.

    Args:
        audio: The whole session recording as floats in [-1, 1], mono, shape (samples,).
        windows: The x0 window of each part in seconds, from ``stored_windows``.
        settings: The view settings.

    Returns:
        The three views and where their parts are.

    Raises:
        ValueError: If the audio or windows are refused by ``cut_parts``, a part is too
            short for x2's filter, or x0 is silent.
    """
    rate, fade, gap = settings.rate, settings.fade, settings.gap
    raw = cut_parts(audio, windows, rate=rate)
    faded = [fade_part(part, fade=fade, rate=rate) for part in raw]
    x0, starts, ends = join_parts(faded, rate=rate, gap=gap)
    muffled = [
        fade_part(
            muffle(part, cutoff_hz=settings.cutoff_hz, order=settings.order, rate=rate),
            fade=fade,
            rate=rate,
        )
        for part in faded
    ]
    x2_joined, x2_starts, _ = join_parts(muffled, rate=rate, gap=gap)
    target = clip_rms(x0)
    if target == 0.0:
        raise ValueError("x0 is silent, so x2 has no RMS to match")
    x2, rms_gain = set_rms(x2_joined, target)
    x2, peak_factor, limited = limit_peak(x2, settings.peak_limit)
    reversed_parts = [
        fade_part(reverse_windows(part, window=settings.window), fade=fade, rate=rate)
        for part in raw
    ]
    rev, rev_starts, _ = join_parts(reversed_parts, rate=rate, gap=gap)
    if not (x2_starts == rev_starts == starts and len(x2) == len(rev) == len(x0)):
        raise ValueError("the views of a segment differ in their parts")
    return SegmentViews(
        x0=x0,
        x2=x2,
        rev=rev,
        part_starts=tuple(starts),
        part_ends=tuple(ends),
        x2_gain=rms_gain * peak_factor,
        x2_peak_limited=limited,
    )


def check_segment_views(
    views: Mapping[str, np.ndarray],
    starts: Sequence[int],
    ends: Sequence[int],
    *,
    gap: int,
) -> None:
    """Check that the views of one segment fit together, as they are stored.

    Every view has the same length and the same parts; every gap sample is exactly 0;
    and every part starts and ends at exactly 0.

    Args:
        views: Each view's clip by name, as floats or 16-bit numbers.
        starts: The first sample of each part.
        ends: The sample just after each part.
        gap: Samples of silence between two parts.

    Raises:
        ValueError: If any of these does not hold.
    """
    lengths = {name: len(clip) for name, clip in views.items()}
    if not lengths or len(set(lengths.values())) != 1:
        raise ValueError(f"the views differ in length: {lengths}")
    length = next(iter(lengths.values()))
    if len(starts) == 0 or len(starts) != len(ends):
        raise ValueError("expected one end per start and at least one part")
    if starts[0] != 0 or ends[-1] != length:
        raise ValueError("the parts do not span the clip")
    for start, end in zip(starts, ends, strict=True):
        if end - start < 2:
            raise ValueError(f"a part from sample {start} to {end} is too short to fade")
    for end, next_start in zip(ends[:-1], starts[1:], strict=True):
        if next_start - end != gap:
            raise ValueError(f"a gap of {next_start - end} samples, not {gap}")
    for name, clip in views.items():
        for end, next_start in zip(ends[:-1], starts[1:], strict=True):
            if np.any(clip[end:next_start] != 0):
                raise ValueError(f"{name}: a gap is not silent")
        for start, end in zip(starts, ends, strict=True):
            if clip[start] != 0 or clip[end - 1] != 0:
                raise ValueError(f"{name}: a part does not start and end at 0")
