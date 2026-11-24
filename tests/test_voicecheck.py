import ast
import inspect
import math
import warnings
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from listening_beyond_words import voicecheck
from listening_beyond_words.partitions import make_partitions
from listening_beyond_words.revoice import assign_voices
from listening_beyond_words.runinfo import StepError
from listening_beyond_words.views import PCM16_SCALE, SAMPLING_RATE, clip_peak, clip_rms, to_pcm16
from listening_beyond_words.voicecheck import (
    ALL_FRAMES,
    FUNCTIONAL_CLASSES,
    FUNCTIONAL_NAMES,
    LEVEL,
    LOUDNESS_VARIABILITY,
    PITCH_TRACK,
    PITCH_VARIABILITY,
    TIMING,
    TIMING_WEIGHTS,
    UNVOICED,
    VOICED,
    VOICED_STRETCHES,
    Smile,
    centre_within,
    control_aurocs,
    gate_values,
    icc_c1,
    level_copy,
    part_measures,
    part_weights,
    participant_means,
    probe_aurocs,
    probe_scores,
    segment_measures,
    voice_iccs,
    voiced_seconds,
    weight_column,
)

MODULE = Path(voicecheck.__file__)
MIN_SAMPLES = 960  # word_check.voice.min_part_samples
GATE = (PITCH_VARIABILITY, LOUDNESS_VARIABILITY, VOICED_STRETCHES)
VOICES = {"female": ["af_heart", "af_bella"], "male": ["am_michael", "am_fenrir"]}
# The pilot's participants by gender and PHQ-8 class.
COUNTS = {("female", 0): 56, ("female", 1): 31, ("male", 0): 76, ("male", 1): 26}
SHROUT_FLEISS = np.array(
    [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]],
    dtype=float,
)


def names_of(kind: str) -> list[str]:
    return [name for name, value in FUNCTIONAL_CLASSES.items() if value == kind]


# The 88 functionals ------------------------------------------------------------------


def test_the_88_functionals_have_one_class_each() -> None:
    assert len(FUNCTIONAL_NAMES) == 88
    assert len(set(FUNCTIONAL_NAMES)) == 88
    counts = {kind: len(names_of(kind)) for kind in (VOICED, UNVOICED, ALL_FRAMES, LEVEL, TIMING)}
    # The formant amplitudes relative to the pitch take in every frame (-201 dB without
    # pitch), so 6 of the 56 functionals named "voiced" are weighted by length.
    assert counts == {VOICED: 50, UNVOICED: 5, ALL_FRAMES: 26, LEVEL: 1, TIMING: 6}
    assert all("amplitudeLogRelF0" in name for name in names_of(ALL_FRAMES) if "sma3nz" in name)
    assert all("UV" in name.split("_sma3")[0] for name in names_of(UNVOICED))


def test_the_mean_and_spread_of_one_descriptor_share_a_class() -> None:
    for name in FUNCTIONAL_NAMES:
        if name.endswith("_amean"):
            twin = name.removesuffix("_amean") + "_stddevNorm"
            if twin in FUNCTIONAL_CLASSES:
                assert FUNCTIONAL_CLASSES[twin] == FUNCTIONAL_CLASSES[name], name


def test_every_timing_measure_has_a_named_weight() -> None:
    assert set(TIMING_WEIGHTS) == set(names_of(TIMING))
    assert weight_column("loudnessPeaksPerSec") == "seconds"
    assert weight_column(VOICED_STRETCHES) == "seconds"
    assert weight_column("MeanVoicedSegmentLengthSec") == "stretches"
    assert weight_column("StddevVoicedSegmentLengthSec") == "stretches"
    assert weight_column("MeanUnvoicedSegmentLength") == "unvoiced_seconds"
    assert weight_column("StddevUnvoicedSegmentLength") == "unvoiced_seconds"
    assert weight_column(PITCH_VARIABILITY) == "voiced_seconds"
    assert weight_column(LOUDNESS_VARIABILITY) == "seconds"
    assert weight_column("alphaRatioUV_sma3nz_amean") == "unvoiced_seconds"
    assert weight_column("equivalentSoundLevel_dBp") == "seconds"


def test_the_gate_measures_are_classed() -> None:
    assert FUNCTIONAL_CLASSES[PITCH_VARIABILITY] == VOICED
    assert FUNCTIONAL_CLASSES[LOUDNESS_VARIABILITY] == ALL_FRAMES
    assert FUNCTIONAL_CLASSES[VOICED_STRETCHES] == TIMING


def test_opensmile_is_imported_only_inside_the_loader() -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    top = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in (
            node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")]
        )
    }
    assert "opensmile" not in top
    inside = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "load"
        for child in ast.walk(node)
        if isinstance(child, ast.Import) and child.names[0].name == "opensmile"
    ]
    assert len(inside) == 1


# A stand-in for opensmile -------------------------------------------------------------


@dataclass
class FakePart:
    functionals: pd.Series
    pitch: np.ndarray


def frames_of(samples: int) -> int:
    return (samples - 960) // 160 + 2


def fake_part(
    samples: int,
    voiced: float,
    *,
    stretches: int = 1,
    value: float = 0.0,
    values: dict[str, float] | None = None,
    lengths: list[float] | None = None,
    vsps: float | None = None,
    frames: int | None = None,
) -> FakePart:
    """A part as opensmile would give it: ``voiced`` seconds of pitch, ``stretches`` stretches.

    The voiced frames come first, as ``stretches`` runs 2 frames apart; the track then ends
    unvoiced, and opensmile divides the count by 2 frames fewer than the track has.
    """
    count = frames_of(samples) if frames is None else frames
    pitch = np.zeros(count)
    start = 0
    for run in np.array_split(np.arange(round(voiced * 100)), max(stretches, 1)):
        pitch[start : start + len(run)] = 30.0
        start += len(run) + 2
    functionals = dict.fromkeys(FUNCTIONAL_NAMES, value)
    functionals.update(values or {})
    functionals[VOICED_STRETCHES] = stretches / ((count - 2) * 0.01) if vsps is None else vsps
    spans = [0.2] * stretches if lengths is None else lengths
    functionals["MeanVoicedSegmentLengthSec"] = float(np.mean(spans)) if spans else 0.0
    functionals["StddevVoicedSegmentLengthSec"] = float(np.std(spans)) if spans else 0.0
    return FakePart(pd.Series(functionals, dtype=float), pitch)


class FakeExtractor:
    """opensmile's stand-in: each part's outputs are set by its length in samples."""

    def __init__(self, parts: dict[int, FakePart]) -> None:
        self.parts = parts
        self.pieces: list[np.ndarray] = []

    def functionals(self, part: np.ndarray) -> pd.Series:
        self.pieces.append(part.copy())
        return self.parts[len(part)].functionals

    def pitch(self, part: np.ndarray) -> np.ndarray:
        return self.parts[len(part)].pitch


def clip_of(lengths: list[int], *, gap: int = 4800, fill: float = 0.1) -> tuple:
    """A clip of parts of the given lengths, joined by silent gaps, with its part bounds."""
    pieces: list[np.ndarray] = []
    starts: list[int] = []
    ends: list[int] = []
    position = 0
    for index, length in enumerate(lengths):
        if index:
            pieces.append(np.zeros(gap))
            position += gap
        starts.append(position)
        pieces.append(np.full(length, fill))
        position += length
        ends.append(position)
    return np.concatenate(pieces), starts, ends


def measure(parts: dict[int, FakePart]) -> tuple[pd.DataFrame, pd.Series, FakeExtractor]:
    clip, starts, ends = clip_of(list(parts))
    extractor = FakeExtractor(parts)
    table = part_measures(clip, starts, ends, extractor, MIN_SAMPLES, segment="9001_000")
    return table, segment_measures(table), extractor


# Voice measures ----------------------------------------------------------------------


def test_voiced_only_measures_are_weighted_by_voiced_time() -> None:
    # 1.0 s voiced at 0.10 and 0.5 s voiced at 0.40 give 0.20; by length (1.5 s and
    # 1.51 s) they would give about 0.25.
    _, values, _ = measure(
        {24_000: fake_part(24_000, 1.0, value=0.10), 24_160: fake_part(24_160, 0.5, value=0.40)}
    )
    for name in names_of(VOICED):
        assert values[name] == pytest.approx(0.20, abs=1e-12), name


def test_measures_on_every_frame_are_weighted_by_length() -> None:
    # 2 s at 0.10 and 1 s at 0.40 give 0.20; by voiced time (0.5 s each) they would give 0.25.
    _, values, _ = measure(
        {32_000: fake_part(32_000, 0.5, value=0.10), 16_000: fake_part(16_000, 0.5, value=0.40)}
    )
    for name in [*names_of(ALL_FRAMES), "loudnessPeaksPerSec"]:
        assert values[name] == pytest.approx(0.20, abs=1e-12), name
    assert values["F1amplitudeLogRelF0_sma3nz_amean"] == pytest.approx(0.20, abs=1e-12)


def test_unvoiced_only_measures_and_unvoiced_stretch_lengths_are_weighted_by_unvoiced_time() -> (
    None
):
    # 196 frames with 1.5 s voiced leave 0.46 s unvoiced; 96 frames with 0.2 s, 0.76 s.
    table, values, _ = measure(
        {32_000: fake_part(32_000, 1.5, value=0.10), 16_000: fake_part(16_000, 0.2, value=0.40)}
    )
    np.testing.assert_allclose(table["unvoiced_seconds"], [0.46, 0.76], atol=1e-12)
    expected = (0.46 * 0.10 + 0.76 * 0.40) / 1.22
    for name in [*names_of(UNVOICED), "MeanUnvoicedSegmentLength", "StddevUnvoicedSegmentLength"]:
        assert values[name] == pytest.approx(expected, abs=1e-12), name


def test_the_sound_level_is_a_power_mean() -> None:
    level = "equivalentSoundLevel_dBp"
    _, values, _ = measure(
        {
            32_000: fake_part(32_000, 0.5, values={level: 60.0}),
            16_000: fake_part(16_000, 0.5, values={level: 70.0}),
        }
    )
    assert values[level] == pytest.approx(10 * math.log10((2 * 1e6 + 1 * 1e7) / 3), abs=1e-9)
    assert values[level] == pytest.approx(66.0206, abs=1e-4)


def test_voiced_stretches_per_second_leave_out_the_first_stretch_of_each_part() -> None:
    table, values, _ = measure(
        {32_000: fake_part(32_000, 1.0, stretches=3), 16_000: fake_part(16_000, 0.5, stretches=1)}
    )
    assert table["stretches"].tolist() == [3, 1]
    assert values[VOICED_STRETCHES] == pytest.approx((2 + 0) / 3, abs=1e-12)
    assert round(values[VOICED_STRETCHES], 3) == 0.667


def test_voiced_stretch_lengths_are_pooled_by_their_counts() -> None:
    first, second = [0.3, 0.2, 0.4], [0.45]
    _, values, _ = measure(
        {
            32_000: fake_part(32_000, 1.0, stretches=3, lengths=first),
            16_000: fake_part(16_000, 0.5, stretches=1, lengths=second),
        }
    )
    assert values["MeanVoicedSegmentLengthSec"] == pytest.approx(np.mean(first + second))
    assert values["StddevVoicedSegmentLengthSec"] == pytest.approx(np.std(first + second))


def test_a_part_under_960_samples_is_left_out_and_counted() -> None:
    parts = {
        32_000: fake_part(32_000, 1.0, stretches=3, value=0.1),
        959: fake_part(32_000, 1.0, value=50.0),  # never asked for
        16_000: fake_part(16_000, 0.5, value=0.4),
    }
    table, values, extractor = measure(parts)
    assert table["measured"].tolist() == [True, False, True]
    assert table.loc[1, list(FUNCTIONAL_NAMES)].isna().all()
    assert pd.isna(table.loc[1, "stretches"])
    assert table.loc[1, "samples"] == 959
    assert sorted(len(piece) for piece in extractor.pieces) == [16_000, 32_000]
    without = segment_measures(table.drop(index=1))
    pd.testing.assert_series_equal(values, without)
    left_out = (part_weights(table) == 0).sum()
    assert (left_out >= 1).all()
    assert left_out[LOUDNESS_VARIABILITY] == 1


def test_an_unvoiced_part_has_no_weight_for_voiced_measures_but_its_length_counts() -> None:
    table, values, _ = measure(
        {
            32_000: fake_part(32_000, 1.0, stretches=3, value=0.25),
            16_000: fake_part(16_000, 0.0, stretches=0, value=0.0),  # opensmile prints 0
        }
    )
    assert table["voiced_seconds"].tolist() == [1.0, 0.0]
    assert values[PITCH_VARIABILITY] == pytest.approx(0.25)
    assert values[VOICED_STRETCHES] == pytest.approx(2 / 3)
    left_out = (part_weights(table) == 0).sum()
    assert left_out[PITCH_VARIABILITY] == 1
    assert left_out[VOICED_STRETCHES] == 0


def test_a_clip_without_voiced_frames_has_no_value_for_voiced_measures() -> None:
    _, values, _ = measure(
        {
            32_000: fake_part(32_000, 0.0, stretches=0, value=0.3),
            16_000: fake_part(16_000, 0.0, stretches=0, value=0.3),
        }
    )
    assert values[names_of(VOICED)].isna().all()
    assert np.isnan(values["MeanVoicedSegmentLengthSec"])
    assert values[VOICED_STRETCHES] == 0.0
    assert values[LOUDNESS_VARIABILITY] == pytest.approx(0.3)


def track_of(voicing: str) -> np.ndarray:
    """A pitch track from a string of 1s (voiced frames) and 0s."""
    return np.array([31.0 if flag == "1" else 0.0 for flag in voicing])


def short_part(voicing: str, stretches: int, used: int) -> FakePart:
    """A part whose pitch track is ``voicing``, with opensmile's count over ``used`` frames."""
    part = fake_part(frames_of_samples(len(voicing)), 0.0, vsps=stretches / (used * 0.01))
    part.pitch = track_of(voicing)
    return part


def frames_of_samples(frames: int) -> int:
    """The fewest samples that give a part ``frames`` frames."""
    return (frames - 2) * 160 + 960


@pytest.mark.parametrize(
    ("voicing", "counts"),
    [
        ("", [0]),
        ("11", [0, 0, 0]),
        ("111", [0, 0, 0, 1]),
        ("1101", [0, 0, 0, 0, 1]),  # one unvoiced frame inside a stretch that starts
        ("1110111", [0, 0, 0, 1, 1, 1, 1, 1]),  # a gap of one frame is bridged
        ("11100111", [0, 0, 0, 1, 1, 1, 1, 1, 2]),  # two frames end a stretch
        ("1100111", [0, 0, 0, 0, 0, 0, 0, 1]),  # 2 voiced frames start none
        ("0111000111110", [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]),
    ],
)
def test_the_stretches_are_counted_by_opensmiles_own_rule(voicing: str, counts: list[int]) -> None:
    # The nonX rule of functionalSegments.cpp: a stretch starts once 3 frames are voiced and
    # ends after 2 unvoiced frames in a row; one still open at the last frame counts.
    assert voicecheck.stretch_counts(track_of(voicing)).tolist() == counts


@pytest.mark.parametrize(
    ("voicing", "stretches", "used", "expected"),
    [
        # 110 ms, a word and a quiet tail; opensmile used 5 of the 7 frames. The rule
        # before 2026-11-24 read 0.70 stretches here and stopped the job.
        ("1110000", 1, 5, 1),
        # Voiced in every frame, but opensmile used only the first 4 of 15 frames.
        ("1" * 15, 1, 4, 1),
        # 4 and 8 frames allow 1 and 2 stretches, and the track holds both: 3 frames
        # missed is usual, and fewer than 7.
        ("11100111111", 2, 8, 2),
        # 4 and 8 frames both fit the track: 4 frames missed is usual, 0 is not.
        ("11100111", 1, 4, 1),
        # 5 frames give 1 stretch and 10 give 2, but the track holds only one.
        ("1111100000", 1, 5, 1),
    ],
)
def test_the_count_of_a_short_part_follows_opensmiles_own_frames(
    voicing: str, stretches: int, used: int, expected: int
) -> None:
    samples = frames_of_samples(len(voicing))
    table, _, _ = measure({samples: short_part(voicing, stretches, used)})
    assert table["stretches"].tolist() == [expected]


def test_opensmile_output_that_fits_no_count_stops_the_job_without_its_value() -> None:
    no_whole = {32_000: fake_part(32_000, 1.0, vsps=7.3), 16_000: fake_part(16_000, 0.5)}
    with pytest.raises(StepError) as caught:
        measure(no_whole)
    message = (
        "part 0: VoicedSegmentsPerSec gives no whole number of voiced stretches that the"
        " part's frames can hold"
    )
    assert (str(caught.value), caught.value.segment, caught.value.part) == (
        message,
        "9001_000",
        0,
    )
    # 1 stretch needs 3 frames: a part of 2 frames cannot hold the 1 that 50 per second gives.
    with pytest.raises(StepError, match="part 0: VoicedSegmentsPerSec gives no whole number"):
        measure({960: fake_part(960, 0.0, vsps=50.0)})
    # 47 and 94 frames allow 1 and 2 stretches; the unvoiced track holds neither.
    unvoiced = fake_part(16_000, 0.0, stretches=0, vsps=2 / 0.94)
    with pytest.raises(StepError) as caught:
        measure({16_000: unvoiced})
    assert str(caught.value) == (
        "part 0: no count of voiced stretches that VoicedSegmentsPerSec allows is held by"
        " the part's pitch track"
    )


def test_a_wrong_number_of_frames_stops_the_job() -> None:
    parts = {
        32_000: fake_part(32_000, 1.0),
        16_000: fake_part(16_000, 0.5, frames=frames_of(16_000) - 1),
    }
    with pytest.raises(StepError, match="part 1: the pitch track has 95 frames") as caught:
        measure(parts)
    assert caught.value.part == 1


def test_missing_functionals_and_a_broken_count_stop_the_job() -> None:
    part = fake_part(16_000, 0.5)
    part.functionals = part.functionals.drop(index=[LOUDNESS_VARIABILITY])
    with pytest.raises(StepError, match="1 of the 88 functionals are missing"):
        measure({16_000: part})
    with pytest.raises(StepError, match="not a number"):
        measure({16_000: fake_part(16_000, 0.5, vsps=math.nan)})


def test_only_each_views_own_part_bounds_reach_the_extractor() -> None:
    # The gaps hold a marker that no part holds, so a gap sample would show.
    parts = {32_000: fake_part(32_000, 1.0), 16_000: fake_part(16_000, 0.5)}
    clip, starts, ends = clip_of([32_000, 16_000], gap=4800, fill=0.1)
    clip[ends[0] : starts[1]] = -0.5
    extractor = FakeExtractor(parts)
    part_measures(clip, starts, ends, extractor, MIN_SAMPLES)
    assert [len(piece) for piece in extractor.pieces] == [32_000, 16_000]
    for piece, start, end in zip(extractor.pieces, starts, ends, strict=True):
        np.testing.assert_array_equal(piece, clip[start:end])
        assert not np.any(piece == -0.5)
    # Another view's bounds in the same clip are cut at its own places.
    other = FakeExtractor({16_000: fake_part(16_000, 0.5), 30_000: fake_part(30_000, 1.0)})
    part_measures(clip, [2_000, ends[1] - 16_000], [32_000, ends[1]], other, MIN_SAMPLES)
    np.testing.assert_array_equal(other.pieces[0], clip[2_000:32_000])


def test_part_bounds_and_settings_are_checked() -> None:
    clip, starts, ends = clip_of([16_000])
    extractor = FakeExtractor({16_000: fake_part(16_000, 0.5)})
    with pytest.raises(ValueError, match="at least that many"):
        part_measures(clip, starts, ends, extractor, 959)
    with pytest.raises(ValueError, match="outside the clip"):
        part_measures(clip, [0], [len(clip) + 1], extractor, MIN_SAMPLES)
    with pytest.raises(ValueError, match="one end per start"):
        part_measures(clip, [], [], extractor, MIN_SAMPLES)
    with pytest.raises(ValueError, match="as floats"):
        part_measures(to_pcm16(clip), starts, ends, extractor, MIN_SAMPLES)


def test_the_part_table_holds_lengths_times_and_counts() -> None:
    table, _, _ = measure(
        {32_000: fake_part(32_000, 1.2, stretches=2), 16_000: fake_part(16_000, 0.3)}
    )
    assert list(table.columns[:8]) == list(voicecheck.PART_COLUMNS)
    assert table["part"].tolist() == [0, 1]
    assert table["seconds"].tolist() == [2.0, 1.0]
    assert table["frames"].tolist() == [196, 96]
    np.testing.assert_allclose(table["voiced_seconds"], [1.2, 0.3], atol=1e-12)
    np.testing.assert_allclose(table["unvoiced_seconds"], [0.76, 0.66], atol=1e-12)


def test_voiced_time_alone_comes_from_the_pitch_tracks() -> None:
    parts = {
        32_000: fake_part(32_000, 1.2),
        900: fake_part(32_000, 1.0),
        16_000: fake_part(16_000, 0.3),
    }
    clip, starts, ends = clip_of(list(parts))
    assert voiced_seconds(clip, starts, ends, FakeExtractor(parts), MIN_SAMPLES) == pytest.approx(
        1.5
    )


def test_the_level_copy_leaves_x1_as_it_is() -> None:
    rng = np.random.default_rng(9001)
    built = rng.normal(0.0, 1.0, 48_000)
    built *= 0.05 / clip_rms(built)
    stored = to_pcm16(built) / PCM16_SCALE  # as x1 is read back from its file
    kept = stored.copy()
    copy, limited = level_copy(stored, 0.05, 0.99)
    assert not limited
    assert np.max(np.abs(copy - stored)) < 1e-6
    assert copy.dtype == np.float64
    np.testing.assert_array_equal(stored, kept)


def test_the_level_copy_limits_a_spiky_x0_clip_and_counts_it() -> None:
    rng = np.random.default_rng(9002)
    samples = (rng.normal(0.0, 0.008, 48_000) * PCM16_SCALE).round().astype(np.int16)
    samples[1_000] = 16_000  # one spike, far above the rest
    stored = samples / PCM16_SCALE
    kept = stored.copy()
    copy, limited = level_copy(stored, 0.05, 0.99)
    assert limited
    assert clip_peak(copy) == pytest.approx(0.99, abs=1e-12)
    assert clip_rms(copy) < 0.05
    np.testing.assert_array_equal(stored, kept)
    quiet, touched = level_copy(rng.normal(0.0, 0.008, 48_000), 0.05, 0.99)
    assert not touched
    assert clip_rms(quiet) == pytest.approx(0.05, rel=1e-12)


def test_the_level_copy_refuses_what_opensmile_cannot_take() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        level_copy(np.full(100, 0.1), 0.05, 1.0)
    with pytest.raises(ValueError, match="as floats"):
        level_copy(np.full(100, 3, dtype=np.int16), 0.05, 0.99)
    with pytest.raises(ValueError, match="silent"):
        level_copy(np.zeros(100), 0.05, 0.99)


def test_participant_means_skip_segments_without_a_value() -> None:
    values = pd.DataFrame(
        {"a": [1.0, 3.0, np.nan, np.nan], "b": [2.0, np.nan, 5.0, 7.0]},
        index=["9002_000", "9002_001", "9001_000", "9001_001"],
    )
    sessions = pd.Series([9002, 9002, 9001, 9001], index=values.index)
    means = participant_means(values, sessions)
    assert means.index.tolist() == [9001, 9002]
    assert np.isnan(means.loc[9001, "a"])
    assert means.loc[9001, "b"] == 6.0
    assert means.loc[9002, "a"] == 2.0
    with pytest.raises(ValueError, match="no participant"):
        participant_means(values, sessions.iloc[:3])


# The opensmile wrapper, with stand-ins ---------------------------------------------------


class FakeOpensmile:
    """An ``opensmile.Smile`` stand-in that records what it is given."""

    def __init__(self, table: pd.DataFrame) -> None:
        self.table = table
        self.calls: list[tuple[np.ndarray, int]] = []

    def process_signal(self, signal: np.ndarray, sampling_rate: int) -> pd.DataFrame:
        self.calls.append((signal, sampling_rate))
        return self.table


def test_smile_gives_the_88_functionals_and_the_pitch_track() -> None:
    row = pd.DataFrame([np.arange(88, dtype=np.float32)], columns=list(FUNCTIONAL_NAMES))
    track = pd.DataFrame({"Loudness_sma3": [0.1, 0.2, 0.3], PITCH_TRACK: [0.0, 30.5, 31.0]})
    functionals, descriptors = FakeOpensmile(row), FakeOpensmile(track)
    smile = Smile(functionals, descriptors)
    audio = np.full(1_600, 0.25, dtype=np.float32)
    values = smile.functionals(audio)
    assert list(values.index) == list(FUNCTIONAL_NAMES)
    assert values.dtype == np.float64
    assert values[LOUDNESS_VARIABILITY] == 11.0
    np.testing.assert_array_equal(smile.pitch(audio), [0.0, 30.5, 31.0])
    signal, rate = functionals.calls[0]
    assert rate == SAMPLING_RATE
    assert signal.dtype == np.float64
    np.testing.assert_array_equal(signal, audio)


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        (np.full(10, 1_000, dtype=np.int16), "as floats"),
        (np.zeros((2, 10)), "as floats"),
        (np.full(10, 1.0), "up to, not including, 1"),
        (np.full(10, -1.001), "up to, not including, 1"),
        (np.array([0.0, np.nan]), "NaN"),
    ],
)
def test_smile_refuses_audio_that_opensmile_would_misread(audio: np.ndarray, message: str) -> None:
    row = pd.DataFrame([np.zeros(88)], columns=list(FUNCTIONAL_NAMES))
    smile = Smile(FakeOpensmile(row), FakeOpensmile(pd.DataFrame({PITCH_TRACK: [0.0]})))
    with pytest.raises(ValueError, match=message):
        smile.functionals(audio)
    with pytest.raises(ValueError, match=message):
        smile.pitch(audio)
    assert smile.functionals(np.full(10, -1.0)).size == 88  # -1 itself is fine


def test_smile_refuses_more_than_one_row_of_functionals() -> None:
    rows = pd.DataFrame(np.zeros((2, 88)), columns=list(FUNCTIONAL_NAMES))
    smile = Smile(FakeOpensmile(rows), FakeOpensmile(pd.DataFrame({PITCH_TRACK: [0.0]})))
    with pytest.raises(StepError, match="2 rows"):
        smile.functionals(np.zeros(10))


# ICC ---------------------------------------------------------------------------------


def test_icc_matches_the_shrout_fleiss_example() -> None:
    assert icc_c1(SHROUT_FLEISS) == pytest.approx(float(Fraction(920, 1287)), abs=1e-12)
    assert round(icc_c1(SHROUT_FLEISS), 4) == 0.7148


def test_icc_of_two_views_is_twice_the_covariance_over_the_sum_of_variances() -> None:
    rng = np.random.default_rng(9003)
    x0 = rng.normal(size=80)
    x1 = 0.4 * x0 + rng.normal(size=80) + 3.0
    covariance = np.cov(x0, x1)
    expected = 2 * covariance[0, 1] / (covariance[0, 0] + covariance[1, 1])
    assert icc_c1(np.column_stack([x0, x1])) == pytest.approx(expected, abs=1e-12)


def test_icc_ignores_a_shift_of_one_view_but_not_a_scale() -> None:
    rng = np.random.default_rng(9004)
    x0 = rng.normal(size=60)
    x1 = 0.5 * x0 + rng.normal(size=60)
    base = icc_c1(np.column_stack([x0, x1]))
    assert icc_c1(np.column_stack([x0, x1 + 10])) == pytest.approx(base, abs=1e-12)
    assert abs(icc_c1(np.column_stack([x0, 3 * x1])) - base) > 0.01


def test_icc_is_one_for_equal_views_and_minus_one_for_opposite_views() -> None:
    x0 = np.random.default_rng(9005).normal(size=40)
    assert icc_c1(np.column_stack([x0, x0])) == pytest.approx(1.0, abs=1e-12)
    assert icc_c1(np.column_stack([x0, -x0])) == pytest.approx(-1.0, abs=1e-12)
    assert icc_c1(np.column_stack([x0, np.full(40, 2.0)])) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "table",
    [
        np.ones((5, 2)),
        np.column_stack([np.ones(5), np.full(5, 2.0)]),
        np.column_stack([np.full(5, 0.1), np.full(5, 0.7), np.full(5, -3.0)]),
        # The mean of three 0.1s is not exactly 0.1, so the centred columns are not 0.
        np.column_stack([np.full(3, 0.1), np.full(3, 0.7)]),
    ],
)
def test_icc_refuses_a_table_whose_columns_are_each_constant(table: np.ndarray) -> None:
    with pytest.raises(ValueError, match="every column is constant"):
        icc_c1(table)


def test_icc_refuses_missing_values_and_too_small_tables() -> None:
    with pytest.raises(ValueError, match="NaN"):
        icc_c1(np.array([[1.0, 2.0], [np.nan, 1.0], [3.0, 3.0]]))
    with pytest.raises(ValueError, match="at least 2 x 2"):
        icc_c1(np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="at least 2 x 2"):
        icc_c1(np.array([1.0, 2.0, 3.0]))


def test_centring_within_groups_uses_the_fit_rows_only() -> None:
    values = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, np.nan], [100.0, 0.0]])
    groups = ["a", "a", "a", "b"]
    centred = centre_within(values, groups)
    np.testing.assert_allclose(centred[:3, 0], [-2.0, 0.0, 2.0])
    np.testing.assert_allclose(centred[:2, 1], [-5.0, 5.0])
    assert np.isnan(centred[2, 1])
    assert centred[3].tolist() == [0.0, 0.0]
    fitted = centre_within(values, groups, fit=np.array([True, False, False, True]))
    np.testing.assert_allclose(fitted[:3, 0], [0.0, 2.0, 4.0])
    np.testing.assert_allclose(centre_within(values[:, 0], groups), centred[:, 0])
    with pytest.raises(ValueError, match="group b has no row"):
        centre_within(values, groups, fit=np.array([True, True, True, False]))


def segment_table(rng: np.random.Generator, iccs: dict[str, float], per_voice: int = 120) -> tuple:
    """Synthetic segments: x0 and x1 share a part of size icc (the true ICC) per gender."""
    rows = []
    for gender, voices in VOICES.items():
        for voice in voices:
            for index in range(per_voice):
                rows.append(
                    {"segment": f"{9001 + len(rows)}_{index:03d}", "gender": gender, "voice": voice}
                )
    segments = pd.DataFrame(rows).set_index("segment")
    rho = segments["gender"].map(iccs).to_numpy()
    shared = rng.normal(size=len(segments))
    views = {}
    for view in ("x0", "x1"):
        noise = rng.normal(size=(len(segments), len(GATE)))
        values = np.sqrt(rho)[:, None] * shared[:, None] + np.sqrt(1 - rho)[:, None] * noise
        views[view] = pd.DataFrame(values, index=segments.index, columns=list(GATE))
    return views["x0"], views["x1"], segments


def test_the_gate_reads_each_gender_on_its_own() -> None:
    x0, x1, segments = segment_table(np.random.default_rng(9006), {"female": 0.5, "male": 0.1})
    result = voice_iccs(x0, x1, segments, GATE)
    for measure in GATE:
        assert result["gate"][measure]["female"]["icc"] > 0.30
        assert result["gate"][measure]["male"]["icc"] <= 0.30
        assert result["gate"][measure]["female"]["segments"] == 240
    flat = gate_values(result)
    assert len(flat) == 6
    assert flat[f"{PITCH_VARIABILITY} female"] == result["gate"][PITCH_VARIABILITY]["female"]["icc"]


def test_a_gap_between_one_genders_voices_leaves_the_gate_unchanged() -> None:
    x0, x1, segments = segment_table(np.random.default_rng(9007), {"female": 0.4, "male": 0.4})
    before = voice_iccs(x0, x1, segments, GATE)
    shifted = x1.copy()
    shifted.loc[segments["voice"] == "af_bella"] += 3.0
    after = voice_iccs(x0, shifted, segments, GATE)
    for measure in GATE:
        female_before = before["gate"][measure]["female"]["icc"]
        assert after["gate"][measure]["female"]["icc"] == pytest.approx(female_before, abs=1e-12)
        # Without the centring the gap would lower it, towards passing.
        assert after["within_gender"][measure]["female"]["icc"] < female_before - 0.1
        assert after["gate"][measure]["male"] == before["gate"][measure]["male"]
        assert after["offsets"][measure]["am_fenrir"] == before["offsets"][measure]["am_fenrir"]
        offsets = after["offsets"][measure]
        assert offsets["af_bella"]["x1"] - offsets["af_heart"]["x1"] == pytest.approx(
            before["offsets"][measure]["af_bella"]["x1"]
            - before["offsets"][measure]["af_heart"]["x1"]
            + 3.0
        )


def test_x0_is_centred_by_the_participants_x1_voice_too() -> None:
    # A gap between x0's groups of one x1 voice and the other is removed as x1's is.
    x0, x1, segments = segment_table(np.random.default_rng(9028), {"female": 0.4, "male": 0.4})
    before = voice_iccs(x0, x1, segments, GATE)
    shifted = x0.copy()
    shifted.loc[segments["voice"] == "am_fenrir"] -= 2.5
    after = voice_iccs(shifted, x1, segments, GATE)
    for measure in GATE:
        male_before = before["gate"][measure]["male"]["icc"]
        assert after["gate"][measure]["male"]["icc"] == pytest.approx(male_before, abs=1e-12)
        assert after["within_gender"][measure]["male"]["icc"] < male_before - 0.1


def test_a_segment_without_a_value_in_a_view_is_left_out_and_counted() -> None:
    x0, x1, segments = segment_table(np.random.default_rng(9008), {"female": 0.3, "male": 0.3})
    dropped = segments.index[5]
    holed = x1.copy()
    holed.loc[dropped, PITCH_VARIABILITY] = np.nan
    result = voice_iccs(x0, holed, segments, GATE)
    kept = segments.index != dropped
    reference = voice_iccs(x0[kept], x1[kept], segments[kept], GATE)
    assert result["left_out"][PITCH_VARIABILITY] == {"female": 1, "male": 0}
    assert result["left_out"][LOUDNESS_VARIABILITY] == {"female": 0, "male": 0}
    assert result["gate"][PITCH_VARIABILITY]["female"] == pytest.approx(
        reference["gate"][PITCH_VARIABILITY]["female"]
    )
    assert result["gate"][PITCH_VARIABILITY]["female"]["segments"] == 239


def test_the_centring_means_come_from_the_paired_segments_only() -> None:
    x0, x1, segments = segment_table(np.random.default_rng(9009), {"female": 0.3, "male": 0.3})
    extra = pd.DataFrame({"gender": ["female"], "voice": ["af_heart"]}, index=["9999_000"])
    x0_extra = pd.concat([x0, pd.DataFrame([[np.nan] * 3], index=extra.index, columns=list(GATE))])
    x1_extra = pd.concat([x1, pd.DataFrame([[1_000.0] * 3], index=extra.index, columns=list(GATE))])
    result = voice_iccs(x0_extra, x1_extra, pd.concat([segments, extra]), GATE)
    reference = voice_iccs(x0, x1, segments, GATE)
    for measure in GATE:
        for gender in VOICES:
            assert result["gate"][measure][gender] == pytest.approx(
                reference["gate"][measure][gender]
            )
        assert result["offsets"][measure]["af_heart"] == pytest.approx(
            reference["offsets"][measure]["af_heart"]
        )
        assert result["left_out"][measure]["female"] == 1


def test_a_negative_gate_value_is_kept_and_passes() -> None:
    x0, x1, segments = segment_table(np.random.default_rng(9010), {"female": 0.0, "male": 0.0})
    x1 = -x0 + 0.2 * x1
    result = voice_iccs(x0, x1, segments, GATE)
    value = result["gate"][PITCH_VARIABILITY]["female"]["icc"]
    assert value < -0.5
    assert value <= 0.30


def test_the_reported_iccs_stand_beside_the_gate() -> None:
    x0, x1, segments = segment_table(np.random.default_rng(9011), {"female": 0.4, "male": 0.2})
    result = voice_iccs(x0, x1, segments, GATE, whole_x0=x0 * 2.0, whole_x1=x1 + 1.0)
    entry = result["within_gender"][LOUDNESS_VARIABILITY]["female"]
    female = segments["gender"] == "female"
    a = x0.loc[female, LOUDNESS_VARIABILITY].to_numpy()
    b = x1.loc[female, LOUDNESS_VARIABILITY].to_numpy()
    assert entry["icc"] == pytest.approx(icc_c1(np.column_stack([a, b])))
    assert entry["pearson"] == pytest.approx(np.corrcoef(a, b)[0, 1])
    pooled = result["pooled"][LOUDNESS_VARIABILITY]
    assert pooled["segments"] == 480
    both = np.column_stack([x0[LOUDNESS_VARIABILITY], x1[LOUDNESS_VARIABILITY]])
    assert pooled["icc"] == pytest.approx(icc_c1(both))
    whole = result["whole_clip"][LOUDNESS_VARIABILITY]["female"]
    assert whole["pearson"] == pytest.approx(entry["pearson"])
    assert whole["icc"] != pytest.approx(entry["icc"])  # scaling x0 changes the ICC
    assert result["whole_clip_pooled"][LOUDNESS_VARIABILITY]["segments"] == 480
    assert "whole_clip" not in voice_iccs(x0, x1, segments, GATE)
    with pytest.raises(ValueError, match="both views, or of neither"):
        voice_iccs(x0, x1, segments, GATE, whole_x0=x0)


def test_a_gate_value_that_cannot_be_computed_stops_the_job() -> None:
    x0, x1, segments = segment_table(np.random.default_rng(9012), {"female": 0.3, "male": 0.3})
    # Each voice group constant: nothing is left after the centring.
    flat0 = x0.copy()
    flat1 = x1.copy()
    for voice, group in segments.groupby("voice"):
        flat0.loc[group.index, PITCH_VARIABILITY] = len(voice)
        flat1.loc[group.index, PITCH_VARIABILITY] = len(voice) * 2.0
    with pytest.raises(StepError, match=f"ICC gate of {PITCH_VARIABILITY} for female"):
        voice_iccs(flat0, flat1, segments, GATE)
    # Constants that are not whole numbers leave rounding residue after the centring (the
    # mean of many 0.1s is not exactly 0.1); no ICC is read from it either.
    constant = segments["voice"].map(
        {"af_heart": 0.1, "af_bella": 0.7, "am_michael": 0.3, "am_fenrir": 0.9}
    )
    flat0[PITCH_VARIABILITY] = 1.1 * constant
    flat1[PITCH_VARIABILITY] = 0.37 * constant
    with pytest.raises(StepError, match=f"ICC gate of {PITCH_VARIABILITY} for female"):
        voice_iccs(flat0, flat1, segments, GATE)
    few = x1.copy()
    few.loc[segments["gender"] == "male", VOICED_STRETCHES] = np.nan
    few.loc[segments.index[segments["gender"] == "male"][0], VOICED_STRETCHES] = 1.0
    with pytest.raises(StepError, match="1 paired segment"):
        voice_iccs(x0, few, segments, GATE)


def test_the_gate_refuses_a_voice_in_two_genders_and_missing_rows() -> None:
    x0, x1, segments = segment_table(np.random.default_rng(9013), {"female": 0.3, "male": 0.3})
    mixed = segments.copy()
    mixed.iloc[0, mixed.columns.get_loc("voice")] = "am_michael"
    with pytest.raises(ValueError, match="more than one gender"):
        voice_iccs(x0, x1, mixed, GATE)
    with pytest.raises(ValueError, match="no row for 1 segment"):
        voice_iccs(x0.iloc[1:], x1, segments, GATE)


# Probe -------------------------------------------------------------------------------


def participants(seed: int, folds_seed: int = 101) -> pd.DataFrame:
    """Synthetic participants with the pilot's counts, voices and round-1 folds."""
    rng = np.random.default_rng(seed)
    cells = [cell for cell, count in COUNTS.items() for _ in range(count)]
    order = rng.permutation(len(cells))
    table = pd.DataFrame(
        {"gender": [cells[i][0] for i in order], "label": [cells[i][1] for i in order]},
        index=pd.Index(range(9001, 9001 + len(cells)), name="session"),
    )
    table["voice"] = pd.Series(assign_voices(table, VOICES))
    folds = make_partitions(table[["label", "gender"]], seeds=(folds_seed,))
    table["fold"] = folds.set_index("session")["fold"]
    return table


def control_from_counts(table: pd.DataFrame) -> float:
    """The gender-only control from the fold counts: women rank above men if more of them are positive in training."""
    aurocs = []
    for fold in sorted(table["fold"].unique()):
        test, train = table[table["fold"] == fold], table[table["fold"] != fold]
        rates = train.groupby("gender")["label"].mean()
        first, second = (
            ("female", "male") if rates["female"] > rates["male"] else ("male", "female")
        )
        count = test.groupby(["gender", "label"]).size()
        positives, negatives = int(test["label"].sum()), int((test["label"] == 0).sum())
        wins = Fraction(int(count[(first, 1)] * count[(second, 0)]))
        ties = Fraction(
            int(count[(first, 1)] * count[(first, 0)] + count[(second, 1)] * count[(second, 0)]), 2
        )
        aurocs.append((wins + ties) / (positives * negatives))
    return float(sum(aurocs) / len(aurocs))


def probe_inputs(table: pd.DataFrame) -> tuple:
    return table["gender"], table["voice"], table["label"], table["fold"]


def noise_features(table: pd.DataFrame, rng: np.random.Generator, count: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        rng.normal(size=(len(table), count)),
        index=table.index,
        columns=[f"m{index}" for index in range(count)],
    )


@pytest.mark.parametrize(("seed", "folds_seed"), [(9014, 101), (9015, 102)])
def test_the_control_equals_the_value_from_the_fold_counts(seed: int, folds_seed: int) -> None:
    table = participants(seed, folds_seed)
    aurocs = control_aurocs(table["gender"], table["label"], table["fold"], 0.1, 5000)
    assert len(aurocs) == 5
    assert aurocs.mean() == pytest.approx(control_from_counts(table), abs=1e-12)
    assert round(aurocs.mean(), 6) == 0.559861
    assert control_aurocs(table["gender"], table["label"], table["fold"], 1.0, 5000).mean() == (
        pytest.approx(aurocs.mean(), abs=1e-12)
    )


def test_the_probe_finds_a_planted_measure() -> None:
    table = participants(9016)
    rng = np.random.default_rng(9016)
    features = noise_features(table, rng)
    features["planted"] = table["label"] + 0.1 * rng.normal(size=len(table))
    aurocs = probe_aurocs(features, *probe_inputs(table), 0.1, 5000)
    assert aurocs.mean() > 0.95


def test_the_probe_centres_adds_gender_scales_and_fits_on_the_training_rows() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    table = participants(9024)
    rng = np.random.default_rng(9024)
    features = noise_features(table, rng, count=6) * [1.0, 10.0, 100.0, 0.1, 1.0, 5.0]
    features["hint"] = table["label"] + rng.normal(size=len(table))
    scores = probe_scores(features, *probe_inputs(table), 0.1, 5000)
    train, test = table["fold"] != 4, table["fold"] == 4
    centred = features.copy()
    for voice in VOICES["female"] + VOICES["male"]:
        rows = table["voice"] == voice
        centred.loc[rows] -= features.loc[rows & train].mean()
    design = centred.assign(female=(table["gender"] == "female").astype(float)).to_numpy()
    scaler = StandardScaler().fit(design[train.to_numpy()])
    model = LogisticRegression(C=0.1, max_iter=5000).fit(
        scaler.transform(design[train.to_numpy()]), table.loc[train, "label"]
    )
    expected = model.decision_function(scaler.transform(design[test.to_numpy()]))
    np.testing.assert_allclose(scores[test].to_numpy(), expected, rtol=1e-9, atol=1e-9)
    other_c = probe_scores(features, *probe_inputs(table), 1.0, 5000)
    assert not np.allclose(other_c[test].to_numpy(), expected, atol=1e-3)


def test_a_constant_added_to_one_voice_changes_nothing() -> None:
    table = participants(9017)
    features = noise_features(table, np.random.default_rng(9017))
    features["label_hint"] = table["label"] * 0.5 + np.random.default_rng(1).normal(size=len(table))
    base = probe_aurocs(features, *probe_inputs(table), 0.1, 5000)
    shifted = features.copy()
    shifted.loc[table["voice"] == "am_fenrir"] += 7.0
    np.testing.assert_allclose(
        probe_aurocs(shifted, *probe_inputs(table), 0.1, 5000), base, atol=1e-12
    )


def test_a_test_participants_values_change_no_other_score_of_its_fold() -> None:
    table = participants(9018)
    features = noise_features(table, np.random.default_rng(9018))
    base = probe_scores(features, *probe_inputs(table), 0.1, 5000)
    target = table.index[(table["fold"] == 2) & (table["voice"] == "af_heart")][0]
    changed = features.copy()
    changed.loc[target] = changed.loc[target] * 10 + 3
    scores = probe_scores(changed, *probe_inputs(table), 0.1, 5000)
    same_fold = table.index[(table["fold"] == 2) & (table.index != target)]
    np.testing.assert_allclose(scores[same_fold], base[same_fold], rtol=0, atol=1e-12)
    assert scores[target] != pytest.approx(base[target])


def test_gender_is_added_after_the_centring_and_not_centred() -> None:
    # With measures that carry nothing, the probe is its control. Gender centred within
    # each voice would be 0 for everyone and give 0.5 in every fold.
    table = participants(9019)
    flat = pd.DataFrame({"m0": 1.0, "m1": -2.0}, index=table.index)
    probe = probe_aurocs(flat, *probe_inputs(table), 0.1, 5000)
    control = control_aurocs(table["gender"], table["label"], table["fold"], 0.1, 5000)
    np.testing.assert_allclose(probe, control, atol=1e-12)
    assert not np.allclose(probe, 0.5)


def test_a_missing_value_takes_its_voices_training_mean() -> None:
    # The job's count of such values is tested end to end (test_check_views.py).
    table = participants(9020)
    features = noise_features(table, np.random.default_rng(9020))
    target = table.index[(table["fold"] == 3) & (table["voice"] == "am_michael")][0]
    holed = features.copy()
    holed.loc[target, "m4"] = np.nan
    training = (table["fold"] != 3) & (table["voice"] == "am_michael")
    filled = features.copy()
    filled.loc[target, "m4"] = features.loc[training, "m4"].mean()
    with_hole = probe_scores(holed, *probe_inputs(table), 0.1, 5000)
    with_mean = probe_scores(filled, *probe_inputs(table), 0.1, 5000)
    assert with_hole[target] == pytest.approx(with_mean[target], abs=1e-12)
    fold = table.index[table["fold"] == 3]
    np.testing.assert_allclose(with_hole[fold], with_mean[fold], atol=1e-12)


def test_x0_and_x1_go_through_the_same_function() -> None:
    table = participants(9021)
    features = noise_features(table, np.random.default_rng(9021))
    assert "view" not in inspect.signature(probe_aurocs).parameters
    first = probe_aurocs(features, *probe_inputs(table), 0.1, 5000)
    reordered = features.sample(frac=1.0, random_state=1)
    np.testing.assert_allclose(probe_aurocs(reordered, *probe_inputs(table), 0.1, 5000), first)


def test_the_probe_refuses_what_it_cannot_read() -> None:
    table = participants(9022)
    features = noise_features(table, np.random.default_rng(9022))
    gender, voices, labels, folds = probe_inputs(table)
    with pytest.raises(ValueError, match="labels must hold each participant"):
        probe_aurocs(features, gender, voices, labels.iloc[1:], folds, 0.1, 5000)
    with pytest.raises(ValueError, match="unknown gender"):
        probe_aurocs(features, gender.replace("male", "m"), voices, labels, folds, 0.1, 5000)
    lonely = labels.copy()
    lonely[folds == 1] = 0
    with pytest.raises(ValueError, match="fold 1: the test participants lack a class"):
        probe_aurocs(features, gender, voices, lonely, folds, 0.1, 5000)


def test_a_probe_that_does_not_converge_stops_the_job() -> None:
    table = participants(9023)
    features = noise_features(table, np.random.default_rng(9023), count=40)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(StepError, match="did not converge in 1 iterations"):
            probe_aurocs(features, *probe_inputs(table), 100.0, 1)


# opensmile itself (OMNI only) ----------------------------------------------------------


def tone(
    samples: int, amplitude: float = 0.1, pitch_hz: float = 150.0, vibrato: float = 0.0
) -> np.ndarray:
    """A harmonic tone with 5 ms ramps, synthetic."""
    t = np.arange(samples) / SAMPLING_RATE
    frequency = pitch_hz * (1 + vibrato * np.sin(2 * np.pi * 4 * t))
    phase = 2 * np.pi * np.cumsum(frequency) / SAMPLING_RATE
    wave = sum(np.sin(k * phase) / k for k in range(1, 13))
    wave = wave / np.max(np.abs(wave)) * amplitude
    ramp = min(80, samples // 2)
    window = np.ones(samples)
    window[:ramp] = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp) / ramp)
    window[samples - ramp :] = window[:ramp][::-1]
    return wave * window


@pytest.fixture(scope="module")
def smile() -> Smile:
    pytest.importorskip("opensmile")
    return Smile.load()


@pytest.mark.slow
def test_opensmile_names_the_88_functionals_in_order(smile: Smile) -> None:
    import opensmile

    names = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    ).feature_names
    assert list(names) == list(FUNCTIONAL_NAMES)


@pytest.mark.slow
def test_opensmile_classes_equal_the_functional_classes(smile: Smile) -> None:
    rng = np.random.default_rng(9024)
    noise = smile.functionals(0.05 * rng.standard_normal(SAMPLING_RATE))
    assert not np.any(smile.pitch(0.05 * rng.standard_normal(SAMPLING_RATE)) > 0)
    # Without pitch, the voiced-only functionals print 0; every other one has a value,
    # except the spread of the formant amplitudes, which are -201 dB in every frame then.
    assert (noise[names_of(VOICED)] == 0.0).all()
    floor = [name for name in names_of(ALL_FRAMES) if "amplitudeLogRelF0" in name]
    assert (noise[[name for name in floor if name.endswith("_amean")]] < -150).all()
    assert (noise[[name for name in floor if name.endswith("_stddevNorm")]] == 0.0).all()
    assert (noise[[n for n in names_of(ALL_FRAMES) if n not in floor]] != 0.0).all()
    assert (noise[names_of(UNVOICED)] != 0.0).all()
    assert (noise[names_of(LEVEL)] != 0.0).all()
    # With pitch in every frame, the unvoiced-only functionals print 0.
    steady = tone(2 * SAMPLING_RATE, amplitude=0.2)
    assert np.all(smile.pitch(steady) > 0)
    voiced = smile.functionals(steady)
    assert (voiced[names_of(UNVOICED)] == 0.0).all()
    assert (voiced[[n for n in names_of(VOICED) if n.endswith("_amean")]] != 0.0).all()


@pytest.mark.slow
@pytest.mark.parametrize("samples", [960, 961, 1_119, 1_120, 1_600, 16_000, 31_999])
def test_opensmile_gives_the_expected_number_of_frames(smile: Smile, samples: int) -> None:
    assert len(smile.pitch(tone(samples))) == (samples - 960) // 160 + 2
    assert not smile.functionals(tone(samples)).isna().any()


@pytest.mark.slow
def test_opensmile_gives_no_values_below_960_samples(smile: Smile) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "Segment too short"
        assert smile.functionals(tone(959)).isna().all()


@pytest.mark.slow
def test_three_bursts_give_three_voiced_stretches(smile: Smile) -> None:
    silence = [np.zeros(round(seconds * SAMPLING_RATE)) for seconds in (0.1, 0.2, 0.15, 0.1)]
    bursts = [
        tone(round(s * SAMPLING_RATE), pitch_hz=f) for s, f in ((0.3, 150), (0.25, 160), (0.4, 140))
    ]
    clip = np.concatenate(
        [silence[0], bursts[0], silence[1], bursts[1], silence[2], bursts[2], silence[3]]
    )
    table = part_measures(clip, [0], [len(clip)], smile, MIN_SAMPLES)
    assert table["stretches"].tolist() == [3]
    assert table.loc[0, "voiced_seconds"] > 0.8


@pytest.mark.slow
def test_short_parts_give_opensmiles_count_of_voiced_stretches(smile: Smile) -> None:
    # 110 ms: a word that ends in a quiet unvoiced tail. The rule before 2026-11-24 stopped
    # the job here.
    tail = 0.01 * np.random.default_rng(9026).standard_normal(792)
    word = np.concatenate([tone(968, amplitude=0.2), tail])
    assert part_measures(word, [0], [len(word)], smile, MIN_SAMPLES)["stretches"].tolist() == [1]
    # 100 to 200 ms of steady voice: opensmile counts one stretch in each.
    for samples in range(1_600, 3_201, 160):
        steady = tone(samples, amplitude=0.2)
        table = part_measures(steady, [0], [samples], smile, MIN_SAMPLES)
        assert table["stretches"].tolist() == [1], samples


@pytest.mark.slow
def test_each_count_is_held_by_the_first_frames_of_the_pitch_track(smile: Smile) -> None:
    # Random speech-like parts of 60 ms to 1 s, many of them short: no count stops the job,
    # and each is opensmile's own rule over the first frames that give VoicedSegmentsPerSec.
    rng = np.random.default_rng(9027)
    for _ in range(300):
        pieces = []
        for _ in range(int(rng.integers(1, 4))):
            pieces.append(tone(int(rng.integers(300, 4_000)), 0.2, float(rng.uniform(90, 250))))
            gap = int(rng.integers(80, 2_000))
            pieces.append(float(rng.uniform(0.0005, 0.02)) * rng.standard_normal(gap))
        part = np.concatenate(pieces)
        table = part_measures(part, [0], [len(part)], smile, MIN_SAMPLES)
        count = int(table.loc[0, "stretches"])
        vsps = float(smile.functionals(part)[VOICED_STRETCHES])
        held = voicecheck.stretch_counts(smile.pitch(part))
        used = [m for m in range(1, len(held)) if abs(vsps * m * 0.01 - count) <= 1e-6 * count]
        assert count == 0 if vsps == 0 else any(held[m] == count for m in used)


@pytest.mark.slow
def test_pitch_variability_is_the_spread_of_the_voiced_pitch(smile: Smile) -> None:
    # Silence around the tone: opensmile's functionals leave out up to 5 frames at the
    # end of a part, which then hold no pitch.
    silence = np.zeros(SAMPLING_RATE // 10)
    clip = np.concatenate([silence, tone(SAMPLING_RATE, amplitude=0.2, vibrato=0.05), silence])
    pitch = smile.pitch(clip)
    voiced = pitch[pitch > 0]
    assert smile.functionals(clip)[PITCH_VARIABILITY] == pytest.approx(
        voiced.std() / voiced.mean(), rel=1e-5
    )


@pytest.mark.slow
def test_silence_gives_no_voiced_time(smile: Smile) -> None:
    rng = np.random.default_rng(9025)
    for clip in (np.zeros(SAMPLING_RATE), 1e-4 * rng.standard_normal(SAMPLING_RATE)):
        table = part_measures(clip, [0], [len(clip)], smile, MIN_SAMPLES)
        assert table.loc[0, "voiced_seconds"] == 0.0
        assert table["stretches"].tolist() == [0]
        assert np.isnan(segment_measures(table)[PITCH_VARIABILITY])
