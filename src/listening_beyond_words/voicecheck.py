"""The voice rule of the word check: how closely the re-voiced clip x1 follows the speaker's voice.

Pilot protocol, section "Clip versions", bullet "Word check"; the settings are in
``configs/pilot/data.yaml`` (``word_check.voice``). x1 keeps the words and the gender
of x0 on purpose, but should not keep the speaker's own voice. Two tests look for it:

- The ICC gate: three eGeMAPS measures (pitch variability, loudness variability and
  voiced stretches per second) are taken on x0 and on x1 for every sampled segment.
  ICC(C,1) says how closely x1 follows x0, segment by segment; it must be at most 0.30
  for each measure within each gender. Each gender has two x1 voices, and a gap between
  them would lower the ICC, so both views are first centred within each x1 voice
  (``voice_iccs``).
- The probe: a classifier predicts the PHQ-8 label from all 88 measures, averaged per
  participant. Each x1 voice is balanced against the label, so the measures are centred
  within the voice, with means from the training participants only, and gender is then
  added back as a feature; x0 goes through the same steps with the same voice groups.
  Its control uses gender alone (``probe_aurocs``, ``control_aurocs``).

Each part of a clip is measured on its own, without the gaps, by the view's own part
bounds (``part_measures``), and the parts are then averaged with weights that follow
what each measure is taken on (``segment_measures``). opensmile counts a frame as
unvoiced below a fixed level, and x0 keeps its recording's level, so the measures are
taken on a copy of each x0 and x1 clip at x1's level (``level_copy``); no stored clip
changes. opensmile is imported inside ``Smile.load`` only, so that everything else runs
without it.
"""

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Final, Protocol

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from listening_beyond_words.labels import GENDERS
from listening_beyond_words.runinfo import StepError
from listening_beyond_words.views import SAMPLING_RATE, limit_peak, set_rms

# opensmile 2.6.0, eGeMAPSv02, at 16 kHz.
FRAME_SAMPLES: Final = 160  # one descriptor frame every 10 ms
PITCH_WINDOW_SAMPLES: Final = 960  # the 60 ms pitch window: a shorter part gets no values
PITCH_TRACK: Final = "F0semitoneFrom27.5Hz_sma3nz"  # the smoothed pitch, 0 where unvoiced
# A part of N samples gives (N - 960) // 160 + 2 frames. opensmile counts the voiced
# stretches of the pitch track by the nonX rule of its functionalSegments.cpp: a stretch
# starts once 3 frames are voiced and ends after 2 unvoiced frames in a row, so a gap of one
# frame is bridged. VoicedSegmentsPerSec is that count divided by 0.01 s times the number of
# frames that reached the functional: the first frames of the track, on synthetic speech
# usually 2 to 8 fewer than the track has, and in a few short parts far fewer
# (``stretch_count``).
STRETCH_START_FRAMES: Final = 3  # segMinLng
STRETCH_END_FRAMES: Final = 2  # pauseMinLng
USUAL_SHORTFALL: Final = range(2, 9)  # frames of the track that the functional usually misses
STRETCH_TOLERANCE: Final = 1e-6  # relative: opensmile computes VoicedSegmentsPerSec in float32
FEMALE: Final = "female"  # gender enters the probe as 1 for women, 0 for men

# The three measures of the ICC gate (data.yaml, word_check.voice.measures).
PITCH_VARIABILITY: Final = "F0semitoneFrom27.5Hz_sma3nz_stddevNorm"
LOUDNESS_VARIABILITY: Final = "loudness_sma3_stddevNorm"
VOICED_STRETCHES: Final = "VoicedSegmentsPerSec"
STRETCH_LENGTH_MEAN: Final = "MeanVoicedSegmentLengthSec"
STRETCH_LENGTH_SD: Final = "StddevVoicedSegmentLengthSec"
SOUND_LEVEL: Final = "equivalentSoundLevel_dBp"

# What each functional is taken on, and so how the parts of a segment are averaged.
VOICED: Final = "voiced"  # frames with pitch only: weighted by voiced time
UNVOICED: Final = "unvoiced"  # frames without pitch only: weighted by unvoiced time
ALL_FRAMES: Final = "all"  # every frame: weighted by length
LEVEL: Final = "level"  # the sound level: a power mean, weighted by length
TIMING: Final = "timing"  # stretches and peaks: each with its own weight (TIMING_WEIGHTS)

# The 88 eGeMAPSv02 functionals in opensmile's order. opensmile skips the frames where a
# descriptor is 0, and sets the voiced-only descriptors to 0 where there is no pitch.
# The three formant amplitudes relative to the pitch are the exception: opensmile 2.6.0
# sets them to -201 dB where there is no pitch, not to 0, so their two functionals take
# in every frame (checked on synthetic tones, and their length-weighted part values come
# closer to the value of the joined parts).
FUNCTIONAL_CLASSES: Final[dict[str, str]] = {
    "F0semitoneFrom27.5Hz_sma3nz_amean": VOICED,
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": VOICED,
    "F0semitoneFrom27.5Hz_sma3nz_percentile20.0": VOICED,
    "F0semitoneFrom27.5Hz_sma3nz_percentile50.0": VOICED,
    "F0semitoneFrom27.5Hz_sma3nz_percentile80.0": VOICED,
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2": VOICED,
    "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope": VOICED,
    "F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope": VOICED,
    "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope": VOICED,
    "F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope": VOICED,
    "loudness_sma3_amean": ALL_FRAMES,
    "loudness_sma3_stddevNorm": ALL_FRAMES,
    "loudness_sma3_percentile20.0": ALL_FRAMES,
    "loudness_sma3_percentile50.0": ALL_FRAMES,
    "loudness_sma3_percentile80.0": ALL_FRAMES,
    "loudness_sma3_pctlrange0-2": ALL_FRAMES,
    "loudness_sma3_meanRisingSlope": ALL_FRAMES,
    "loudness_sma3_stddevRisingSlope": ALL_FRAMES,
    "loudness_sma3_meanFallingSlope": ALL_FRAMES,
    "loudness_sma3_stddevFallingSlope": ALL_FRAMES,
    "spectralFlux_sma3_amean": ALL_FRAMES,
    "spectralFlux_sma3_stddevNorm": ALL_FRAMES,
    "mfcc1_sma3_amean": ALL_FRAMES,
    "mfcc1_sma3_stddevNorm": ALL_FRAMES,
    "mfcc2_sma3_amean": ALL_FRAMES,
    "mfcc2_sma3_stddevNorm": ALL_FRAMES,
    "mfcc3_sma3_amean": ALL_FRAMES,
    "mfcc3_sma3_stddevNorm": ALL_FRAMES,
    "mfcc4_sma3_amean": ALL_FRAMES,
    "mfcc4_sma3_stddevNorm": ALL_FRAMES,
    "jitterLocal_sma3nz_amean": VOICED,
    "jitterLocal_sma3nz_stddevNorm": VOICED,
    "shimmerLocaldB_sma3nz_amean": VOICED,
    "shimmerLocaldB_sma3nz_stddevNorm": VOICED,
    "HNRdBACF_sma3nz_amean": VOICED,
    "HNRdBACF_sma3nz_stddevNorm": VOICED,
    "logRelF0-H1-H2_sma3nz_amean": VOICED,
    "logRelF0-H1-H2_sma3nz_stddevNorm": VOICED,
    "logRelF0-H1-A3_sma3nz_amean": VOICED,
    "logRelF0-H1-A3_sma3nz_stddevNorm": VOICED,
    "F1frequency_sma3nz_amean": VOICED,
    "F1frequency_sma3nz_stddevNorm": VOICED,
    "F1bandwidth_sma3nz_amean": VOICED,
    "F1bandwidth_sma3nz_stddevNorm": VOICED,
    "F1amplitudeLogRelF0_sma3nz_amean": ALL_FRAMES,  # -201 dB without pitch
    "F1amplitudeLogRelF0_sma3nz_stddevNorm": ALL_FRAMES,
    "F2frequency_sma3nz_amean": VOICED,
    "F2frequency_sma3nz_stddevNorm": VOICED,
    "F2bandwidth_sma3nz_amean": VOICED,
    "F2bandwidth_sma3nz_stddevNorm": VOICED,
    "F2amplitudeLogRelF0_sma3nz_amean": ALL_FRAMES,  # -201 dB without pitch
    "F2amplitudeLogRelF0_sma3nz_stddevNorm": ALL_FRAMES,
    "F3frequency_sma3nz_amean": VOICED,
    "F3frequency_sma3nz_stddevNorm": VOICED,
    "F3bandwidth_sma3nz_amean": VOICED,
    "F3bandwidth_sma3nz_stddevNorm": VOICED,
    "F3amplitudeLogRelF0_sma3nz_amean": ALL_FRAMES,  # -201 dB without pitch
    "F3amplitudeLogRelF0_sma3nz_stddevNorm": ALL_FRAMES,
    "alphaRatioV_sma3nz_amean": VOICED,
    "alphaRatioV_sma3nz_stddevNorm": VOICED,
    "hammarbergIndexV_sma3nz_amean": VOICED,
    "hammarbergIndexV_sma3nz_stddevNorm": VOICED,
    "slopeV0-500_sma3nz_amean": VOICED,
    "slopeV0-500_sma3nz_stddevNorm": VOICED,
    "slopeV500-1500_sma3nz_amean": VOICED,
    "slopeV500-1500_sma3nz_stddevNorm": VOICED,
    "spectralFluxV_sma3nz_amean": VOICED,
    "spectralFluxV_sma3nz_stddevNorm": VOICED,
    "mfcc1V_sma3nz_amean": VOICED,
    "mfcc1V_sma3nz_stddevNorm": VOICED,
    "mfcc2V_sma3nz_amean": VOICED,
    "mfcc2V_sma3nz_stddevNorm": VOICED,
    "mfcc3V_sma3nz_amean": VOICED,
    "mfcc3V_sma3nz_stddevNorm": VOICED,
    "mfcc4V_sma3nz_amean": VOICED,
    "mfcc4V_sma3nz_stddevNorm": VOICED,
    "alphaRatioUV_sma3nz_amean": UNVOICED,
    "hammarbergIndexUV_sma3nz_amean": UNVOICED,
    "slopeUV0-500_sma3nz_amean": UNVOICED,
    "slopeUV500-1500_sma3nz_amean": UNVOICED,
    "spectralFluxUV_sma3nz_amean": UNVOICED,
    "loudnessPeaksPerSec": TIMING,
    "VoicedSegmentsPerSec": TIMING,
    "MeanVoicedSegmentLengthSec": TIMING,
    "StddevVoicedSegmentLengthSec": TIMING,
    "MeanUnvoicedSegmentLength": TIMING,
    "StddevUnvoicedSegmentLength": TIMING,
    "equivalentSoundLevel_dBp": LEVEL,
}
FUNCTIONAL_NAMES: Final = tuple(FUNCTIONAL_CLASSES)

# The column of the part table that weights each class, and each timing measure.
CLASS_WEIGHTS: Final[dict[str, str]] = {
    VOICED: "voiced_seconds",
    UNVOICED: "unvoiced_seconds",
    ALL_FRAMES: "seconds",
    LEVEL: "seconds",
}
TIMING_WEIGHTS: Final[dict[str, str]] = {
    "loudnessPeaksPerSec": "seconds",
    # The first voiced stretch of each part is not counted: sum of (n - 1) over the length.
    "VoicedSegmentsPerSec": "seconds",
    "MeanVoicedSegmentLengthSec": "stretches",
    "StddevVoicedSegmentLengthSec": "stretches",  # pooled over all stretches
    "MeanUnvoicedSegmentLength": "unvoiced_seconds",
    "StddevUnvoicedSegmentLength": "unvoiced_seconds",
}

PART_COLUMNS: Final = (
    "part",
    "samples",
    "seconds",
    "measured",
    "frames",
    "voiced_seconds",
    "unvoiced_seconds",
    "stretches",
)


def weight_column(name: str) -> str:
    """Return the column of the part table that weights a functional across the parts.

    Args:
        name: One of the 88 functionals.

    Returns:
        ``seconds``, ``voiced_seconds``, ``unvoiced_seconds`` or ``stretches``.

    Raises:
        KeyError: If the name is not one of the 88.
    """
    return TIMING_WEIGHTS.get(name) or CLASS_WEIGHTS[FUNCTIONAL_CLASSES[name]]


# Level ------------------------------------------------------------------------------


def level_copy(samples: np.ndarray, rms: float, peak: float) -> tuple[np.ndarray, bool]:
    """Return a copy of a whole clip at x1's level, for the voice measures only.

    The copy gets an RMS of ``rms`` over the whole clip, gaps included
    (``views.set_rms``), and is then scaled down as a whole to a peak of ``peak`` if its
    peak is above it (``views.limit_peak``), as x1 itself was built. x1 comes back as it
    is stored, apart from rounding; x0 comes back louder. opensmile counts a frame as
    unvoiced below a fixed level, so x0 and x1 are measured at the same level.

    Args:
        samples: The clip as floats, shape (samples,); it is not changed.
        rms: The RMS of the copy (``word_check.voice.level_rms``).
        peak: The highest peak of the copy (``views.peak_limit``), between 0 and 1:
            opensmile cannot take values of 1 or more.

    Returns:
        The copy as float64, and whether the peak guard scaled it down.

    Raises:
        ValueError: If the clip is not one-dimensional floats, is silent or holds NaN or
            inf, or a setting is out of range.
    """
    if samples.ndim != 1 or not np.issubdtype(samples.dtype, np.floating):
        raise ValueError(
            f"expected a clip as floats of shape (samples,), got {samples.dtype} {samples.shape}"
        )
    if not 0.0 < peak < 1.0:
        raise ValueError(f"the peak limit must lie between 0 and 1, got {peak}")
    scaled, _ = set_rms(samples, rms)
    copy, _, limited = limit_peak(scaled, peak)
    return copy, limited


# opensmile --------------------------------------------------------------------------


class Extractor(Protocol):
    """What the voice measures need of opensmile; the tests pass a stand-in."""

    def functionals(self, part: np.ndarray) -> pd.Series:
        """Return the 88 eGeMAPSv02 functionals of a stretch of 16 kHz audio, by name."""
        ...

    def pitch(self, part: np.ndarray) -> np.ndarray:
        """Return its smoothed pitch track: one value per 10 ms frame, 0 where unvoiced."""
        ...


def _smile_input(part: np.ndarray) -> np.ndarray:
    """Check one stretch of audio for opensmile and return it as float64.

    opensmile multiplies the values by 32768 and stores them as 16-bit numbers, so a
    value of 1 or more would wrap round.

    Raises:
        ValueError: If the audio is not one-dimensional floats or holds a value outside
            [-1, 1), NaN or inf.
    """
    if part.ndim != 1 or not np.issubdtype(part.dtype, np.floating):
        raise ValueError(
            f"expected audio as floats of shape (samples,), got {part.dtype} {part.shape}"
        )
    audio = np.asarray(part, dtype=np.float64)
    if not np.all(np.isfinite(audio)):
        raise ValueError("the audio holds NaN or inf")
    if audio.size and (audio.min() < -1.0 or audio.max() >= 1.0):
        raise ValueError("opensmile takes values from -1 up to, not including, 1")
    return audio


class Smile:
    """opensmile 2.6.0 with eGeMAPSv02, loaded once and reused for every part.

    Build it with ``Smile.load`` on OMNI, where opensmile is installed. The constructor
    takes the two opensmile objects directly, so that the tests can pass stand-ins.

    Args:
        functionals: An ``opensmile.Smile`` for eGeMAPSv02 functionals.
        descriptors: An ``opensmile.Smile`` for eGeMAPSv02 low-level descriptors.
    """

    def __init__(self, functionals: Any, descriptors: Any) -> None:
        self._functionals = functionals
        self._descriptors = descriptors

    @classmethod
    def load(cls) -> "Smile":
        """Load opensmile's eGeMAPSv02 functionals and low-level descriptors and check them.

        Returns:
            The extractor.

        Raises:
            StepError: If the functionals are not the 88 of ``FUNCTIONAL_CLASSES`` in that
                order, or the descriptors have no pitch track.
        """
        import opensmile

        feature_set = opensmile.FeatureSet.eGeMAPSv02
        functionals = opensmile.Smile(
            feature_set=feature_set, feature_level=opensmile.FeatureLevel.Functionals
        )
        descriptors = opensmile.Smile(
            feature_set=feature_set, feature_level=opensmile.FeatureLevel.LowLevelDescriptors
        )
        if list(functionals.feature_names) != list(FUNCTIONAL_NAMES):
            raise StepError(
                "opensmile's eGeMAPSv02 functionals are not the 88 that the voice measures class"
            )
        if PITCH_TRACK not in list(descriptors.feature_names):
            raise StepError(f"opensmile's eGeMAPSv02 descriptors have no {PITCH_TRACK}")
        return cls(functionals, descriptors)

    def functionals(self, part: np.ndarray) -> pd.Series:
        """Return the 88 eGeMAPSv02 functionals of a stretch of 16 kHz audio.

        Args:
            part: The audio as floats in [-1, 1), shape (samples,).

        Returns:
            The values as float64, by name, in opensmile's order; all NaN for fewer than
            960 samples.

        Raises:
            ValueError: If the audio is refused (``_smile_input``).
            StepError: If opensmile does not give one row of values.
        """
        table = self._functionals.process_signal(_smile_input(part), SAMPLING_RATE)
        if len(table) != 1:
            raise StepError(f"opensmile gave {len(table)} rows of functionals for one part")
        row = table.iloc[0]
        return pd.Series(
            row.to_numpy(dtype=np.float64), index=[str(name) for name in row.index], dtype=float
        )

    def pitch(self, part: np.ndarray) -> np.ndarray:
        """Return the smoothed pitch track of a stretch of 16 kHz audio.

        Args:
            part: The audio as floats in [-1, 1), shape (samples,).

        Returns:
            ``F0semitoneFrom27.5Hz_sma3nz`` as float64, one value per 10 ms frame; 0
            where opensmile finds no pitch.

        Raises:
            ValueError: If the audio is refused (``_smile_input``).
        """
        table = self._descriptors.process_signal(_smile_input(part), SAMPLING_RATE)
        return np.asarray(table[PITCH_TRACK].to_numpy(dtype=np.float64))


# Voice measures ---------------------------------------------------------------------


def _check_bounds(clip: np.ndarray, starts: Sequence[int], ends: Sequence[int]) -> None:
    """Refuse part bounds that do not lie in order inside the clip."""
    if clip.ndim != 1 or not np.issubdtype(clip.dtype, np.floating):
        raise ValueError(
            f"expected a clip as floats of shape (samples,), got {clip.dtype} {clip.shape}"
        )
    if len(starts) == 0 or len(starts) != len(ends):
        raise ValueError("expected one end per start and at least one part")
    previous = 0
    for start, end in zip(starts, ends, strict=True):
        if not previous <= start < end <= len(clip):
            raise ValueError(
                f"a part from sample {start} to {end} is out of order or outside the clip"
            )
        previous = end


def _pitch_frames(
    extractor: Extractor, piece: np.ndarray, segment: str | None, position: int
) -> np.ndarray:
    """The pitch track of one part, after checking its frame count against its length."""
    pitch = np.asarray(extractor.pitch(piece), dtype=np.float64)
    expected = (len(piece) - PITCH_WINDOW_SAMPLES) // FRAME_SAMPLES + 2
    if pitch.ndim != 1 or len(pitch) != expected:
        raise StepError(
            f"part {position}: the pitch track has {pitch.size} frames for {len(piece)}"
            f" samples, not {expected}",
            segment=segment,
            part=position,
        )
    if not np.all(np.isfinite(pitch)):
        raise StepError(
            f"part {position}: the pitch track holds NaN or inf", segment=segment, part=position
        )
    return pitch


def stretch_counts(track: np.ndarray) -> np.ndarray:
    """Count the voiced stretches in the first m frames of a pitch track, for every m.

    opensmile 2.6.0's nonX rule (functionalSegments.cpp) with eGeMAPSv02's settings: a
    frame is voiced where the track is not 0; a stretch starts once 3 frames are voiced
    (a single unvoiced frame between them is skipped) and ends after 2 unvoiced frames in a
    row, so a gap of one frame is bridged; a stretch still open at the last frame counts.

    Args:
        track: A smoothed pitch track (``Smile.pitch``), one value per frame.

    Returns:
        Shape (frames + 1,): element m is the count over the first m frames.
    """
    counts = np.zeros(len(track) + 1, dtype=np.int64)
    closed = voiced = unvoiced = 0
    starting = inside = False
    for position, value in enumerate(np.asarray(track, dtype=np.float64), start=1):
        if value != 0:
            unvoiced = 0
            if not inside:
                starting = True
                voiced += 1
                if voiced >= STRETCH_START_FRAMES:
                    starting, inside, voiced = False, True, 0
        elif starting or inside:
            unvoiced += 1
            if unvoiced >= STRETCH_END_FRAMES:
                closed += int(inside)
                starting, inside, voiced, unvoiced = False, False, 0, 0
        counts[position] = closed + int(inside)
    return counts


def stretch_count(vsps: float, track: np.ndarray, segment: str | None, position: int) -> int:
    """Recover how many voiced stretches opensmile counted in a part.

    VoicedSegmentsPerSec is that count n divided by 0.01 s times the m frames that reached
    opensmile's functional, the first m of the pitch track, and m is not printed. So
    n = VoicedSegmentsPerSec x m x 0.01 must be a whole number, and n stretches need at
    least 3n + 2(n - 1) frames. If every m from 1 to the track's length that meets both
    gives the same n, that is the count. Otherwise only the m whose first frames hold n
    stretches by opensmile's own rule (``stretch_counts``) are kept; of those, an m that
    misses 2 to 8 frames (``USUAL_SHORTFALL``) goes first, the one that misses the fewest.

    Args:
        vsps: opensmile's VoicedSegmentsPerSec of the part.
        track: The part's pitch track (``Smile.pitch``).
        segment: The segment ID, for the messages of errors.
        position: The part's position in the clip, from 0.

    Returns:
        The count, 0 or more.

    Raises:
        StepError: If VoicedSegmentsPerSec is not a number of 0 or more, or opensmile's
            output is inconsistent: no whole count fits it, or none that the track holds.
            The message names the part, never its values.
    """
    if not math.isfinite(vsps) or vsps < 0:
        raise StepError(
            f"part {position}: {VOICED_STRETCHES} is not a number of zero or more",
            segment=segment,
            part=position,
        )
    if vsps == 0:
        return 0
    frames = len(track)
    used = np.arange(frames, 0, -1)  # frames that reached the functional, the most first
    raw = vsps * used * FRAME_SAMPLES / SAMPLING_RATE
    counts = np.round(raw)
    needed = STRETCH_START_FRAMES * counts + STRETCH_END_FRAMES * (counts - 1)
    fits = (counts >= 1) & (np.abs(raw - counts) <= STRETCH_TOLERANCE * counts) & (used >= needed)
    candidates = [(frames - int(m), int(n)) for m, n in zip(used[fits], counts[fits], strict=True)]
    if not candidates:
        raise StepError(
            f"part {position}: {VOICED_STRETCHES} gives no whole number of voiced stretches"
            " that the part's frames can hold",
            segment=segment,
            part=position,
        )
    if len({count for _, count in candidates}) == 1:
        return candidates[0][1]
    held = stretch_counts(track)
    kept = [(missed, count) for missed, count in candidates if held[frames - missed] == count]
    if not kept:
        raise StepError(
            f"part {position}: no count of voiced stretches that {VOICED_STRETCHES} allows"
            " is held by the part's pitch track",
            segment=segment,
            part=position,
        )
    usual = [fit for fit in kept if fit[0] in USUAL_SHORTFALL]
    return min(usual or kept)[1]


def part_measures(
    clip: np.ndarray,
    starts: Sequence[int],
    ends: Sequence[int],
    extractor: Extractor,
    min_samples: int,
    *,
    segment: str | None = None,
) -> pd.DataFrame:
    """Measure each part of one clip on its own, without the gaps.

    For each part of at least ``min_samples`` samples:

    - the 88 functionals of ``extractor``;
    - its length, and its voiced and unvoiced time from the pitch track (frames with
      pitch above 0, 10 ms each);
    - its count of voiced stretches as opensmile counts them, recovered from
      VoicedSegmentsPerSec and the pitch track (``stretch_count``).

    A shorter part is left out: it gets a row without values, and ``measured`` False.

    Args:
        clip: The clip as floats in [-1, 1), shape (samples,), such as ``level_copy``'s.
        starts: The first sample of each part, from this view's own manifest.
        ends: The sample just after each part.
        extractor: opensmile (``Smile``), or a stand-in.
        min_samples: Parts with fewer samples are left out
            (``word_check.voice.min_part_samples``); at least 960, as opensmile gives no
            values below.
        segment: The segment ID, for the messages of errors.

    Returns:
        One row per part in clip order, with the columns of ``PART_COLUMNS`` (``part``
        is the position from 0; ``frames`` and ``stretches`` are nullable whole numbers)
        and then the 88 functionals.

    Raises:
        ValueError: If the clip is not one-dimensional floats, the bounds are out of
            order or outside the clip, or ``min_samples`` is below 960.
        StepError: If the extractor does not give the 88 functionals, a pitch track does
            not have (N - 960) // 160 + 2 frames for N samples or holds NaN, or no count
            of voiced stretches fits opensmile's output (``stretch_count``). It names the
            part.
    """
    _check_bounds(clip, starts, ends)
    if min_samples < PITCH_WINDOW_SAMPLES:
        raise ValueError(
            f"opensmile gives no values below {PITCH_WINDOW_SAMPLES} samples, so parts must"
            f" have at least that many, not {min_samples}"
        )
    rows: list[dict[str, Any]] = []
    for position, (start, end) in enumerate(zip(starts, ends, strict=True)):
        samples = int(end) - int(start)
        row: dict[str, Any] = {
            "part": position,
            "samples": samples,
            "seconds": samples / SAMPLING_RATE,
            "measured": samples >= min_samples,
            "frames": pd.NA,
            "voiced_seconds": math.nan,
            "unvoiced_seconds": math.nan,
            "stretches": pd.NA,
        }
        if samples >= min_samples:
            piece = clip[int(start) : int(end)]
            values = extractor.functionals(piece)
            missing = [name for name in FUNCTIONAL_NAMES if name not in values.index]
            if missing:
                raise StepError(
                    f"part {position}: {len(missing)} of the 88 functionals are missing",
                    segment=segment,
                    part=position,
                )
            pitch = _pitch_frames(extractor, piece, segment, position)
            voiced = int(np.count_nonzero(pitch > 0))
            row.update(
                frames=len(pitch),
                voiced_seconds=voiced * FRAME_SAMPLES / SAMPLING_RATE,
                unvoiced_seconds=(len(pitch) - voiced) * FRAME_SAMPLES / SAMPLING_RATE,
                stretches=stretch_count(float(values[VOICED_STRETCHES]), pitch, segment, position),
            )
            row.update({name: float(values[name]) for name in FUNCTIONAL_NAMES})
        else:
            row.update(dict.fromkeys(FUNCTIONAL_NAMES, math.nan))
        rows.append(row)
    table = pd.DataFrame(rows, columns=[*PART_COLUMNS, *FUNCTIONAL_NAMES])
    table["frames"] = table["frames"].astype("Int64")
    table["stretches"] = table["stretches"].astype("Int64")
    table["measured"] = table["measured"].astype(bool)
    for column in ("seconds", "voiced_seconds", "unvoiced_seconds", *FUNCTIONAL_NAMES):
        table[column] = table[column].astype(float)
    return table


def voiced_seconds(
    clip: np.ndarray,
    starts: Sequence[int],
    ends: Sequence[int],
    extractor: Extractor,
    min_samples: int,
    *,
    segment: str | None = None,
) -> float:
    """Return the voiced time of a clip's parts from their pitch tracks alone.

    The job compares x0's voiced time at its stored level with that of its level copy,
    to show what measuring at x1's level changes. Parts are taken as in
    ``part_measures``, with the same frame-count check, but without the functionals.

    Args:
        clip: The clip as floats in [-1, 1), shape (samples,).
        starts: The first sample of each part.
        ends: The sample just after each part.
        extractor: opensmile (``Smile``), or a stand-in.
        min_samples: Parts with fewer samples are left out; at least 960.
        segment: The segment ID, for the messages of errors.

    Returns:
        The voiced time in seconds, summed over the parts measured.

    Raises:
        ValueError: As ``part_measures``.
        StepError: If a pitch track has the wrong number of frames or holds NaN.
    """
    _check_bounds(clip, starts, ends)
    if min_samples < PITCH_WINDOW_SAMPLES:
        raise ValueError(f"parts must have at least {PITCH_WINDOW_SAMPLES} samples")
    frames = 0
    for position, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if int(end) - int(start) >= min_samples:
            pitch = _pitch_frames(extractor, clip[int(start) : int(end)], segment, position)
            frames += int(np.count_nonzero(pitch > 0))
    return frames * FRAME_SAMPLES / SAMPLING_RATE


def part_weights(parts: pd.DataFrame) -> pd.DataFrame:
    """Return the weight that each part gets in the segment value of each functional.

    A part without a value for a functional gets 0: a part that was left out, a value that
    is NaN, or no voiced time for a voiced-only functional (opensmile prints 0 there, not
    "no value"), and so on. ``(part_weights(parts) == 0).sum()`` counts, per functional,
    the parts without a value.

    Args:
        parts: One clip's rows of ``part_measures``.

    Returns:
        One row per part and one column per functional, in the order of ``parts``.
    """
    measured = parts["measured"].to_numpy(dtype=bool)
    weights: dict[str, np.ndarray] = {}
    for name in FUNCTIONAL_NAMES:
        raw = parts[weight_column(name)].to_numpy(dtype=np.float64, na_value=np.nan)
        values = parts[name].to_numpy(dtype=np.float64)
        usable = measured & np.isfinite(values) & np.isfinite(raw) & (raw > 0)
        weights[name] = np.where(usable, raw, 0.0)
    return pd.DataFrame(weights, index=parts.index, columns=list(FUNCTIONAL_NAMES))


def segment_measures(parts: pd.DataFrame) -> pd.Series:
    """Average one clip's part values into one value per functional.

    Each functional is a mean of its part values weighted by ``part_weights``: by voiced
    time for those taken on voiced frames only, by unvoiced time for those on unvoiced
    frames only, by length for those on every frame, and each timing measure by its own
    weight (``TIMING_WEIGHTS``). Three take another form of mean:

    - the sound level is a power mean, 10 log10(sum of d 10^(L / 10) / sum of d), the
      level of the parts joined without the gaps;
    - the spread of the voiced-stretch lengths is pooled over all stretches of the clip
      (a population standard deviation, as opensmile's within each part);
    - VoicedSegmentsPerSec does not count the first voiced stretch of each part: the sum
      of max(n - 1, 0) over the sum of the lengths, over every part measured.

    Args:
        parts: One clip's rows of ``part_measures``.

    Returns:
        The 88 values, by name; NaN for a functional that no part has a value for.
    """
    weights = part_weights(parts)
    counts = parts["stretches"].to_numpy(dtype=np.float64, na_value=np.nan)
    result: dict[str, float] = {}
    for name in FUNCTIONAL_NAMES:
        values = parts[name].to_numpy(dtype=np.float64)
        weight = weights[name].to_numpy(dtype=np.float64)
        has = weight > 0
        if not has.any():
            result[name] = math.nan
        elif name == VOICED_STRETCHES:
            # Every part measured counts its length, also one without a voiced stretch.
            result[name] = float(np.maximum(counts[has] - 1, 0).sum() / weight[has].sum())
        elif name == STRETCH_LENGTH_SD:
            means = parts[STRETCH_LENGTH_MEAN].to_numpy(dtype=np.float64)
            pooled = has & np.isfinite(means)
            if not pooled.any():
                result[name] = math.nan
            else:
                count = weight[pooled]
                centre = float(np.sum(count * means[pooled]) / count.sum())
                spread = values[pooled] ** 2 + (means[pooled] - centre) ** 2
                result[name] = math.sqrt(float(np.sum(count * spread) / count.sum()))
        elif FUNCTIONAL_CLASSES[name] == LEVEL:
            power = np.sum(weight[has] * 10.0 ** (values[has] / 10.0)) / weight[has].sum()
            result[name] = float(10.0 * np.log10(power))
        else:
            result[name] = float(np.sum(weight[has] * values[has]) / weight[has].sum())
    return pd.Series(result, index=list(FUNCTIONAL_NAMES), dtype=float)


def participant_means(values: pd.DataFrame, sessions: pd.Series) -> pd.DataFrame:
    """Average segment values per participant, over the segments that have a value.

    Args:
        values: One row per segment, indexed by segment ID, one column per measure.
        sessions: Each segment's participant (session ID), indexed by segment ID.

    Returns:
        One row per participant, indexed by session ID in ascending order; NaN for a
        measure that none of the participant's segments has a value for.

    Raises:
        ValueError: If a segment has no participant.
    """
    owners = sessions.reindex(values.index)
    if owners.isna().any():
        raise ValueError(f"{int(owners.isna().sum())} segment(s) have no participant")
    means = values.groupby(owners.to_numpy(), sort=True).mean()
    means.index.name = "session"
    return means


# ICC ---------------------------------------------------------------------------------


def icc_c1(table: np.ndarray) -> float:
    """Return ICC(C,1), the Shrout-Fleiss ICC(3,1), of an n x k table.

    The two-way form: with the mean squares of the rows (MSR) and of the error (MSE),
    ICC(C,1) = (MSR - MSE) / (MSR + (k - 1) MSE). For two columns it equals
    2 cov / (var0 + var1). Adding a constant to a column changes nothing; scaling one does.
    A negative value is returned as it is.

    Args:
        table: Shape (n, k): one row per target (a segment), one column per rater (a
            view), with n and k at least 2.

    Returns:
        The ICC.

    Raises:
        ValueError: If the table is not 2-D with at least 2 rows and 2 columns, holds NaN
            or inf, or every column is constant (even at different constants), which
            makes the denominator 0.
    """
    values = np.asarray(table, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError(f"expected a table of at least 2 x 2, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("the table holds NaN or inf")
    if np.all(values == values[0]):
        raise ValueError("ICC(C,1) is undefined when every column is constant")
    n, k = values.shape
    # Centring each column removes the columns' sum of squares, which ICC(C,1) leaves
    # out, and keeps the rest exact: the denominator is then the sum of the columns'
    # variances.
    centred = values - values.mean(axis=0)
    ss_within = float(np.sum(centred**2))
    ss_rows = k * float(np.sum(centred.mean(axis=1) ** 2))
    ms_rows = ss_rows / (n - 1)
    ms_error = (ss_within - ss_rows) / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error
    if not denominator > 0:
        raise ValueError("ICC(C,1) is undefined when every column is constant")
    return (ms_rows - ms_error) / denominator


def centre_within(
    values: np.ndarray, groups: Sequence[Any] | np.ndarray, *, fit: np.ndarray | None = None
) -> np.ndarray:
    """Subtract each group's mean from its members.

    Args:
        values: Shape (n,) or (n, p): one row per item. NaN stays NaN and is left out of
            the means.
        groups: The group of each row, such as a participant's x1 voice.
        fit: Which rows the means come from, such as the training participants; all
            rows if None. Every row is centred.

    Returns:
        The centred copy as float64, in the shape of ``values``.

    Raises:
        ValueError: If the shapes differ, or a group has no row with a value to take a
            mean from.
    """
    data = np.asarray(values, dtype=np.float64)
    labels = np.asarray(groups, dtype=object)
    if data.ndim not in (1, 2) or len(data) != len(labels) or labels.ndim != 1:
        raise ValueError(f"expected one group per row, got {labels.shape} for {data.shape}")
    source = np.ones(len(labels), dtype=bool) if fit is None else np.asarray(fit, dtype=bool)
    if source.shape != labels.shape:
        raise ValueError(f"expected one fit flag per row, got {source.shape} for {labels.shape}")
    matrix = data if data.ndim == 2 else data[:, np.newaxis]
    centred = matrix.copy()
    for group in pd.unique(labels):
        members = labels == group
        rows = matrix[members & source]
        finite = np.isfinite(rows)
        counts = finite.sum(axis=0)
        if rows.shape[0] == 0 or np.any(counts == 0):
            raise ValueError(f"the group {group} has no row with a value to take a mean from")
        means = np.where(finite, rows, 0.0).sum(axis=0) / counts
        centred[members] = matrix[members] - means
    return centred.reshape(data.shape)


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    """Pearson's r of two columns; NaN when either has no variance or fewer than 2 rows."""
    if len(first) < 2:
        return math.nan
    a = first - first.mean()
    b = second - second.mean()
    scale = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    return float(np.sum(a * b)) / scale if scale > 0 else math.nan


def _reported(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    """An ICC that is reported only: NaN, not an error, where it cannot be computed."""
    try:
        icc = icc_c1(np.column_stack([first, second]))
    except ValueError:
        icc = math.nan
    return {"icc": icc, "pearson": _pearson(first, second), "segments": len(first)}


def _view_values(table: pd.DataFrame, index: pd.Index, measure: str, view: str) -> np.ndarray:
    """One measure of one view for the given segments, in their order."""
    if measure not in table.columns:
        raise ValueError(f"{view} has no column {measure}")
    missing = index.difference(table.index)
    if len(missing):
        raise ValueError(f"{view} has no row for {len(missing)} segment(s)")
    return table[measure].reindex(index).to_numpy(dtype=np.float64)


def voice_iccs(
    x0: pd.DataFrame,
    x1: pd.DataFrame,
    segments: pd.DataFrame,
    measures: Sequence[str],
    *,
    whole_x0: pd.DataFrame | None = None,
    whole_x1: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute the ICC gate of x1's voice and everything reported beside it.

    For each measure, a segment counts if it has a value in both views; the others are
    left out and counted. Within each gender, each x1 voice group's mean over those
    paired segments is subtracted from both views (x0 grouped by the participant's x1
    voice), and ICC(C,1) of the centred views is the gate value: six values for three
    measures and two genders. Reported beside it, without the centring: the ICC within
    each gender, the pooled ICC over all paired segments, and the same two for the whole
    clips; Pearson's r beside each; and each voice's offset.

    Args:
        x0: x0's measures, one row per sampled segment (indexed by segment ID), one
            column per measure (``segment_measures``); NaN where there is no value.
        x1: x1's measures, in the same form.
        segments: One row per sampled segment, indexed by segment ID, with the columns
            ``gender`` and ``voice`` (the participant's x1 voice).
        measures: The measures of the gate (``word_check.voice.measures``).
        whole_x0: x0's measures of each whole clip, gaps included, in the same form;
            if given with ``whole_x1``, the whole-clip ICCs are reported.
        whole_x1: x1's measures of each whole clip.

    Returns:
        A dictionary of plain numbers, keyed by measure and then gender:

        - ``gate``: {measure: {gender: {icc, pearson, segments}}}, the six gate values;
        - ``within_gender``: the same without the centring;
        - ``pooled``: {measure: {icc, pearson, segments}}, all genders, without centring;
        - ``whole_clip`` and ``whole_clip_pooled``: as ``within_gender`` and ``pooled``
          for the whole clips (only if they are given);
        - ``offsets``: {measure: {voice: {x0, x1}}}, each voice group's mean minus its
          gender's mean over the paired segments, what the centring removes;
        - ``left_out``: {measure: {gender: count}}, segments without a value in a view.

    Raises:
        ValueError: If a table lacks a segment or a measure, only one of the whole-clip
            tables is given, or a voice belongs to more than one gender.
        StepError: If a gate value cannot be computed: fewer than 2 paired segments, or no
            variance left in either view after the centring. It is never compared then.
    """
    if (whole_x0 is None) != (whole_x1 is None):
        raise ValueError("give the whole-clip measures of both views, or of neither")
    index = pd.Index(segments.index)
    gender = segments["gender"].to_numpy(dtype=object)
    voice = segments["voice"].to_numpy(dtype=object)
    genders_of_voice = segments.groupby("voice")["gender"].nunique()
    if (genders_of_voice > 1).any():
        raise ValueError("a voice belongs to more than one gender")
    genders = sorted({str(value) for value in gender})
    result: dict[str, Any] = {
        "gate": {},
        "within_gender": {},
        "pooled": {},
        "offsets": {},
        "left_out": {},
    }
    for measure in measures:
        first = _view_values(x0, index, measure, "x0")
        second = _view_values(x1, index, measure, "x1")
        paired = np.isfinite(first) & np.isfinite(second)
        gate: dict[str, Any] = {}
        raw: dict[str, Any] = {}
        offsets: dict[str, Any] = {}
        left_out: dict[str, int] = {}
        for name in genders:
            in_gender = gender == name
            rows = in_gender & paired
            left_out[name] = int(np.count_nonzero(in_gender & ~paired))
            a, b, groups = first[rows], second[rows], voice[rows]
            raw[name] = _reported(a, b)
            if len(a) < 2:
                raise StepError(
                    f"the ICC gate of {measure} for {name} cannot be computed: {len(a)}"
                    " paired segment(s)"
                )
            centred_a = centre_within(a, groups)
            centred_b = centre_within(b, groups)
            # Read on the values themselves: centring a voice group whose values are all
            # equal can leave rounding residue instead of zeros, and no ICC may come from it.
            flat = not any(
                np.ptp(values[groups == group]) > 0
                for values in (a, b)
                for group in pd.unique(groups)
            )
            try:
                icc = math.nan if flat else icc_c1(np.column_stack([centred_a, centred_b]))
            except ValueError:
                flat = True
            if flat:
                raise StepError(
                    f"the ICC gate of {measure} for {name} cannot be computed: no variance"
                    f" in either view after the voice centring ({len(a)} paired segments)"
                )
            gate[name] = {
                "icc": icc,
                "pearson": _pearson(centred_a, centred_b),
                "segments": len(a),
            }
            for group in sorted({str(value) for value in groups}):
                members = groups == group
                offsets[group] = {
                    "x0": float(a[members].mean() - a.mean()),
                    "x1": float(b[members].mean() - b.mean()),
                }
        result["gate"][measure] = gate
        result["within_gender"][measure] = raw
        result["pooled"][measure] = _reported(first[paired], second[paired])
        result["offsets"][measure] = offsets
        result["left_out"][measure] = left_out
        if whole_x0 is not None and whole_x1 is not None:
            whole_first = _view_values(whole_x0, index, measure, "x0's whole clips")
            whole_second = _view_values(whole_x1, index, measure, "x1's whole clips")
            whole_paired = np.isfinite(whole_first) & np.isfinite(whole_second)
            result.setdefault("whole_clip", {})[measure] = {
                name: _reported(
                    whole_first[whole_paired & (gender == name)],
                    whole_second[whole_paired & (gender == name)],
                )
                for name in genders
            }
            result.setdefault("whole_clip_pooled", {})[measure] = _reported(
                whole_first[whole_paired], whole_second[whole_paired]
            )
    return result


# Probe -------------------------------------------------------------------------------


def _aligned(index: pd.Index, values: pd.Series, name: str) -> np.ndarray:
    """One participant-level input in the order of ``index``; every participant once."""
    if values.index.has_duplicates or len(values) != len(index) or set(values.index) != set(index):
        raise ValueError(f"{name} must hold each participant of the probe exactly once")
    return values.reindex(index).to_numpy()


def _female(gender: np.ndarray) -> np.ndarray:
    """Gender as one feature: 1 for women, 0 for men."""
    unknown = {str(value) for value in gender} - GENDERS
    if unknown:
        raise ValueError(f"unknown gender values: {sorted(unknown)}")
    return (gender == FEMALE).astype(np.float64)


def _fold_scores(
    matrix: np.ndarray,
    female: np.ndarray,
    groups: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    c: float,
    max_iter: int,
) -> np.ndarray:
    """Score each participant with the model of its own fold, trained on the other folds."""
    scores = np.full(len(labels), np.nan)
    for fold in sorted(set(folds.tolist())):
        test = folds == fold
        train = ~test
        for part, rows in (("training", train), ("test", test)):
            if len(set(labels[rows].tolist())) != 2:
                raise ValueError(f"fold {fold}: the {part} participants lack a class")
        centred = centre_within(matrix, groups, fit=train)
        # A missing value takes its voice's mean over the training participants: 0 here.
        centred = np.where(np.isnan(centred), 0.0, centred)
        design = np.column_stack([centred, female])
        scaler = StandardScaler().fit(design[train])
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=max_iter)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(scaler.transform(design[train]), labels[train])
        except ConvergenceWarning:
            raise StepError(
                f"fold {fold}: the voice probe did not converge in {max_iter} iterations"
            ) from None
        scores[test] = model.decision_function(scaler.transform(design[test]))
    return scores


def _fold_aurocs(scores: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """The AUROC of each fold's participants, folds in ascending order; ties count one half."""
    return np.array(
        [
            float(roc_auc_score(labels[folds == fold], scores[folds == fold]))
            for fold in sorted(set(folds.tolist()))
        ]
    )


def probe_scores(
    features: pd.DataFrame,
    gender: pd.Series,
    voices: pd.Series,
    labels: pd.Series,
    folds: pd.Series,
    c: float,
    max_iter: int,
) -> pd.Series:
    """Score each participant with the voice probe of its own fold.

    In each fold, the model learns from the other folds' participants:

    1. Each measure is centred within each x1 voice, with that voice's mean over the
       training participants, for training and test participants alike. A missing value
       then takes its voice's training mean, which is 0 after the centring.
    2. Gender is added as one more feature (1 for women), after the centring and not
       centred: each voice has one gender, so the centring removes gender, and the
       gender-only control keeps it.
    3. Every feature is scaled with the training participants' mean and standard
       deviation, and a logistic regression (L2, lbfgs) is fitted on them.

    x0 and x1 go through this same function with the same voice groups, so their probes
    differ only in the audio.

    Args:
        features: One row per participant, indexed by session ID, one column per measure
            (``participant_means``); NaN where a participant has no value. Missing values
            do not depend on the fold, so ``features.isna().sum()`` counts them.
        gender: Each participant's gender (``female`` or ``male``), by session ID.
        voices: Each participant's x1 voice, by session ID.
        labels: Each participant's label (0 or 1), by session ID.
        folds: Each participant's round-1 fold, by session ID.
        c: The inverse strength of the L2 penalty (``word_check.voice.probe_logistic_c``).
        max_iter: The most iterations of lbfgs (``word_check.voice.probe_max_iter``).

    Returns:
        Each participant's log-odds from its own fold's model, by session ID in ascending
        order.

    Raises:
        ValueError: If the inputs do not name the same participants, a gender is unknown,
            a fold's training or test participants lack a class, or a voice has no
            training value for a measure.
        StepError: If a fit does not converge.
    """
    table = features.sort_index()
    index = pd.Index(table.index)
    if index.has_duplicates:
        raise ValueError("features must hold each participant once")
    scores = _fold_scores(
        table.to_numpy(dtype=np.float64),
        _female(_aligned(index, gender, "gender")),
        _aligned(index, voices, "voices"),
        _aligned(index, labels, "labels").astype(int),
        _aligned(index, folds, "folds").astype(int),
        c,
        max_iter,
    )
    return pd.Series(scores, index=index, name="score")


def probe_aurocs(
    features: pd.DataFrame,
    gender: pd.Series,
    voices: pd.Series,
    labels: pd.Series,
    folds: pd.Series,
    c: float,
    max_iter: int,
) -> np.ndarray:
    """Return the fold AUROCs of the voice probe; their mean is the probe's result.

    The scores come from ``probe_scores``; each fold's AUROC is taken over its own
    participants, ties counting one half.

    Args:
        features: As ``probe_scores``.
        gender: As ``probe_scores``.
        voices: As ``probe_scores``.
        labels: As ``probe_scores``.
        folds: As ``probe_scores``.
        c: As ``probe_scores``.
        max_iter: As ``probe_scores``.

    Returns:
        One AUROC per fold, folds in ascending order.

    Raises:
        ValueError: As ``probe_scores``.
        StepError: As ``probe_scores``.
    """
    scores = probe_scores(features, gender, voices, labels, folds, c, max_iter)
    index = pd.Index(scores.index)
    return _fold_aurocs(
        scores.to_numpy(),
        _aligned(index, labels, "labels").astype(int),
        _aligned(index, folds, "folds").astype(int),
    )


def control_aurocs(
    gender: pd.Series, labels: pd.Series, folds: pd.Series, c: float, max_iter: int
) -> np.ndarray:
    """Return the fold AUROCs of the probe's control, which knows only the gender.

    It is built as the probe, with no measure: gender, scaled with the training
    participants' statistics, and the same logistic regression. With stratified folds,
    every fold's counts are fixed, so its result does not depend on the seed.

    Args:
        gender: Each participant's gender (``female`` or ``male``), by session ID.
        labels: Each participant's label (0 or 1), by session ID.
        folds: Each participant's round-1 fold, by session ID.
        c: As ``probe_scores``.
        max_iter: As ``probe_scores``.

    Returns:
        One AUROC per fold, folds in ascending order.

    Raises:
        ValueError: As ``probe_scores``.
        StepError: If a fit does not converge.
    """
    index = pd.Index(gender.sort_index().index)
    if index.has_duplicates:
        raise ValueError("gender must hold each participant once")
    label_values = _aligned(index, labels, "labels").astype(int)
    fold_values = _aligned(index, folds, "folds").astype(int)
    scores = _fold_scores(
        np.empty((len(index), 0)),
        _female(_aligned(index, gender, "gender")),
        np.zeros(len(index), dtype=object),
        label_values,
        fold_values,
        c,
        max_iter,
    )
    return _fold_aurocs(scores, label_values, fold_values)


def gate_values(iccs: Mapping[str, Any]) -> dict[str, float]:
    """Return the six gate ICCs of ``voice_iccs`` as one flat mapping.

    Args:
        iccs: The result of ``voice_iccs``.

    Returns:
        ``"<measure> <gender>"`` to its gate ICC, in the order of the result.
    """
    return {
        f"{measure} {gender}": float(entry["icc"])
        for measure, by_gender in iccs["gate"].items()
        for gender, entry in by_gender.items()
    }
