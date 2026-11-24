import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.signal import butter, sosfiltfilt

from listening_beyond_words.segments import build_segments, segment_neighbours
from listening_beyond_words.views import (
    ViewSettings,
    build_segment_views,
    check_segment_views,
    clip_peak,
    clip_rms,
    cut_original,
    cut_parts,
    fade_part,
    join_parts,
    limit_peak,
    muffle,
    part_windows,
    reverse_whole,
    reverse_windows,
    segment_windows,
    set_rms,
    stored_windows,
    to_pcm16,
    windows_overlap,
)

RATE = 16_000
MARGIN = 0.1  # the pilot's settings, configs/pilot/data.yaml
FADE = 0.01
CONFIG = Path(__file__).parents[1] / "configs" / "pilot" / "data.yaml"
SETTINGS = ViewSettings(fade=FADE, cutoff_hz=400.0, order=8, window=4000, peak_limit=0.99)


def noise(seconds: float) -> np.ndarray:
    """Synthetic recording: uniform noise, never exactly 0, at 16 kHz."""
    rng = np.random.default_rng(9001)
    return rng.uniform(0.1, 0.5, round(seconds * RATE)).astype(np.float32)


def words_in_lines(lines: list[tuple[float, int]]) -> pd.DataFrame:
    """Words of 0.2 s every 0.25 s, line after line; each line is (start time, word count)."""
    rows = [
        {
            "session": 9001,
            "line": line,
            "word_index": index,
            "word": f"w{line}x{index}",
            "start": start + 0.25 * index,
            "end": start + 0.25 * index + 0.2,
        }
        for line, (start, count) in enumerate(lines)
        for index in range(count)
    ]
    return pd.DataFrame(rows)


def test_parts_keep_a_margin_around_their_words() -> None:
    windows = part_windows([1.0, 5.0], [2.0, 6.0], duration=10.0, margin=MARGIN)
    np.testing.assert_allclose(windows, [(0.9, 2.1), (4.9, 6.1)])


def test_margin_stops_halfway_to_a_close_part() -> None:
    windows = part_windows([1.0, 2.06], [2.0, 3.0], duration=10.0, margin=MARGIN)
    np.testing.assert_allclose(windows, [(0.9, 2.03), (2.03, 3.1)])


def test_margin_stops_halfway_to_the_words_outside_the_segment() -> None:
    windows = part_windows([1.0], [2.0], duration=10.0, margin=MARGIN, before=0.96, after=2.08)
    np.testing.assert_allclose(windows, [(0.98, 2.04)])


def test_margin_is_full_when_the_outside_words_are_far_or_missing() -> None:
    far = part_windows([1.0], [2.0], duration=10.0, margin=MARGIN, before=0.2, after=3.0)
    missing = part_windows(
        [1.0], [2.0], duration=10.0, margin=MARGIN, before=-math.inf, after=math.inf
    )
    np.testing.assert_allclose(far, [(0.9, 2.1)])
    assert missing == far


def test_margin_stops_at_the_ends_of_the_recording() -> None:
    windows = part_windows([0.05], [9.97], duration=10.0, margin=MARGIN)
    np.testing.assert_allclose(windows, [(0.0, 10.0)])


@pytest.mark.parametrize(
    ("starts", "ends", "before", "after", "message"),
    [
        ([], [], -math.inf, math.inf, "at least one part"),
        ([1.0, 2.0], [1.5], -math.inf, math.inf, "at least one part"),
        ([1.0], [0.5], -math.inf, math.inf, "time order"),  # ends before it starts
        ([3.0, 1.0], [4.0, 2.0], -math.inf, math.inf, "time order"),
        ([1.0, 1.8], [2.0, 3.0], -math.inf, math.inf, "time order"),  # parts overlap
        ([1.0], [2.0], 1.2, math.inf, "time order"),  # overlaps the word before
        ([1.0], [2.0], -math.inf, 1.9, "time order"),  # overlaps the word after
        ([12.0], [13.0], -math.inf, math.inf, "outside"),  # after the recording ends
        ([10.05], [10.5], -math.inf, math.inf, "outside"),  # starts just after the end
        ([9.95], [10.4], -math.inf, math.inf, "outside"),  # runs past the end
        ([-0.01], [0.5], -math.inf, math.inf, "outside"),  # starts before the recording
    ],
)
def test_impossible_parts_are_refused(
    starts: list[float], ends: list[float], before: float, after: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        part_windows(starts, ends, duration=10.0, margin=MARGIN, before=before, after=after)


def test_neighbouring_segments_never_share_audio() -> None:
    words = words_in_lines([(0.0, 30), (20.0, 30), (40.0, 25)])
    table = build_segments(words).merge(segment_neighbours(words), on="segment")
    windows = [
        part_windows(row.starts, row.ends, 60.0, margin=MARGIN, before=row.before, after=row.after)
        for row in table.itertuples()
    ]
    assert windows[0][-1][1] <= windows[1][0][0]
    # Words 0.05 s apart: the margin is half of that on both sides of the split.
    assert windows[0][-1][1] == pytest.approx(20.0 + 9 * 0.25 + 0.2 + 0.025)
    assert windows[1][0][0] == pytest.approx(20.0 + 10 * 0.25 - 0.025)


def test_clip_joins_parts_with_exactly_300_ms_of_silence() -> None:
    audio = noise(10.0)
    clip = cut_original(audio, [(0.9, 2.1), (4.9, 6.1)], fade=FADE)
    part = round(1.2 * RATE)
    gap = round(0.3 * RATE)
    assert clip.dtype == np.float32
    assert len(clip) == 2 * part + gap
    assert np.all(clip[part : part + gap] == 0.0)


def test_clip_keeps_the_audio_unchanged_between_the_fades() -> None:
    audio = noise(10.0)
    clip = cut_original(audio, [(0.9, 2.1), (4.9, 6.1)], fade=FADE)
    fade = round(0.01 * RATE)
    part = round(1.2 * RATE)
    gap = round(0.3 * RATE)
    first = slice(round(0.9 * RATE), round(2.1 * RATE))
    second = slice(round(4.9 * RATE), round(6.1 * RATE))
    np.testing.assert_array_equal(clip[fade : part - fade], audio[first][fade:-fade])
    np.testing.assert_array_equal(clip[part + gap + fade : -fade], audio[second][fade:-fade])


def test_each_part_fades_in_and_out_from_exactly_zero() -> None:
    audio = noise(10.0)
    clip = cut_original(audio, [(0.9, 2.1)], fade=FADE)
    gain = clip / audio[round(0.9 * RATE) : round(2.1 * RATE)]
    fade = round(0.01 * RATE)
    assert gain[0] == 0.0
    assert gain[-1] == 0.0
    assert np.all(np.diff(gain[:fade]) > 0)
    assert np.all(np.diff(gain[-fade:]) < 0)
    assert np.all(gain[:fade] < 1.0)
    np.testing.assert_array_equal(gain[fade:-fade], 1.0)


def test_fade_shrinks_to_fit_a_very_short_part() -> None:
    audio = noise(1.0)
    clip = cut_original(audio, [(0.5, 0.505)], fade=FADE)
    assert len(clip) == round(0.005 * RATE)
    assert clip[0] == 0.0
    assert clip[-1] == 0.0
    assert np.all(np.abs(clip) <= np.abs(audio[round(0.5 * RATE) : round(0.505 * RATE)]))


def test_cutting_does_not_change_the_recording() -> None:
    audio = noise(10.0)
    copy = audio.copy()
    cut_original(audio, [(0.9, 2.1), (4.9, 6.1)], fade=FADE)
    np.testing.assert_array_equal(audio, copy)


def test_cut_refuses_stereo_audio_and_empty_windows() -> None:
    with pytest.raises(ValueError, match="mono"):
        cut_original(np.zeros((100, 2), dtype=np.float32), [(0.0, 0.001)], fade=FADE)
    with pytest.raises(ValueError, match="at least one part"):
        cut_original(noise(1.0), [], fade=FADE)


def test_a_word_end_a_fraction_of_a_sample_past_the_end_is_accepted() -> None:
    windows = part_windows([9.5], [10.0005], duration=10.0, margin=MARGIN)
    np.testing.assert_allclose(windows, [(9.4, 10.0)])


def test_cut_refuses_integer_audio() -> None:
    with pytest.raises(ValueError, match="floats"):
        cut_original(np.zeros(1000, dtype=np.int16), [(0.0, 0.01)], fade=FADE)


def test_data_config_sets_a_fade_that_fits_inside_the_margin() -> None:
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "pilot" / "data.yaml").read_text(encoding="utf-8")
    )
    cut = config["x0_cut"]
    assert set(cut) == {"margin_ms", "fade_ms", "wall_guard_ms"}
    assert 0 < cut["fade_ms"] < cut["margin_ms"]
    assert 0 < cut["wall_guard_ms"] < cut["margin_ms"]


def test_margin_stops_halfway_to_a_dropped_word() -> None:
    between = part_windows([1.0, 3.0], [2.0, 4.0], 10.0, margin=MARGIN, others=[(2.06, 2.5)])
    np.testing.assert_allclose(between, [(0.9, 2.03), (2.9, 4.1)])
    outside = part_windows([1.0], [2.0], 10.0, margin=MARGIN, before=0.5, others=[(0.9, 0.96)])
    np.testing.assert_allclose(outside, [(0.98, 2.1)])


def test_margin_stops_short_of_a_wall() -> None:
    windows = part_windows(
        [1.0, 3.0],
        [2.0, 4.0],
        10.0,
        margin=MARGIN,
        walls=[(2.05, 2.9), (4.1, 5.0)],
        wall_guard=0.02,
    )
    np.testing.assert_allclose(windows, [(0.9, 2.03), (2.92, 4.08)])


@pytest.mark.parametrize(
    ("others", "walls", "message"),
    [
        ([(1.5, 1.6)], [], "overlaps part"),  # another word inside a part
        ([], [(1.5, 1.6)], "comes within"),  # a wall inside a part
        ([], [(2.01, 2.5)], "comes within"),  # a wall closer than the guard
        ([], [(0.5, 0.99)], "comes within"),
    ],
)
def test_parts_that_touch_other_words_or_walls_are_refused(
    others: list[tuple[float, float]], walls: list[tuple[float, float]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        part_windows([1.0], [2.0], 10.0, margin=MARGIN, others=others, walls=walls, wall_guard=0.02)


def test_a_dropped_word_that_touches_a_part_leaves_no_margin() -> None:
    windows = part_windows([1.0], [2.0], 10.0, margin=MARGIN, others=[(0.8, 1.0), (2.0, 2.3)])
    assert windows == [(1.0, 2.0)]


def test_segment_windows_read_every_limit_from_the_segment_row() -> None:
    row = SimpleNamespace(
        starts=[1.0, 3.0],
        ends=[2.0, 4.0],
        before=0.5,
        after=4.5,
        other_starts=[2.06],
        other_ends=[2.5],
        wall_starts=[4.3],
        wall_ends=[5.0],
        bound_starts=[-math.inf, -math.inf],
        bound_ends=[math.inf, math.inf],
    )
    windows = segment_windows(row, 10.0, margin=MARGIN, wall_guard=0.02)
    np.testing.assert_allclose(windows, [(0.9, 2.03), (2.9, 4.1)])
    tight = segment_windows(
        SimpleNamespace(**{**vars(row), "wall_starts": [4.05]}),
        10.0,
        margin=MARGIN,
        wall_guard=0.02,
    )
    assert tight[-1][1] == pytest.approx(4.03)


@pytest.mark.parametrize(
    ("first", "second", "shared"),
    [
        ([(0.0, 1.0)], [(1.0, 2.0)], False),  # touching windows share no audio
        ([(0.0, 1.5)], [(1.0, 2.0)], True),
        ([], [(0.0, 1.0)], False),
        ([(3.0, 4.0), (0.0, 1.0)], [(0.5, 0.6)], True),
    ],
)
def test_windows_overlap(
    first: list[tuple[float, float]], second: list[tuple[float, float]], shared: bool
) -> None:
    assert windows_overlap(first, second) is shared
    assert windows_overlap(second, first) is shared


def test_a_bound_keeps_the_window_inside_the_participants_turn() -> None:
    windows = part_windows(
        [1.0, 3.0], [2.0, 4.0], 10.0, margin=MARGIN, bounds=[(0.95, 2.0), (2.5, 4.05)]
    )
    np.testing.assert_allclose(windows, [(0.95, 2.0), (2.9, 4.05)])
    free = part_windows([1.0], [2.0], 10.0, margin=MARGIN, bounds=[(-math.inf, math.inf)])
    np.testing.assert_allclose(free, [(0.9, 2.1)])
    with pytest.raises(ValueError, match="does not hold"):
        part_windows([1.0], [2.0], 10.0, margin=MARGIN, bounds=[(1.1, 3.0)])
    with pytest.raises(ValueError, match="one bound per part"):
        part_windows([1.0], [2.0], 10.0, margin=MARGIN, bounds=[])


def frozen_cut_original(
    audio: np.ndarray, windows: list[tuple[float, float]], *, fade: float, rate: int = RATE
) -> np.ndarray:
    """``cut_original`` as it was when the listening checks were cut (commit 2645d45)."""
    silence = np.zeros(round(0.3 * rate), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for low, high in windows:
        piece = np.array(audio[round(low * rate) : round(high * rate)], dtype=np.float32)
        n = min(round(fade * rate), len(piece) // 2)
        if n > 0:
            ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)).astype(np.float32)
            piece[:n] *= ramp
            piece[-n:] *= ramp[::-1]
        if pieces:
            pieces.append(silence)
        pieces.append(piece)
    return np.concatenate(pieces)


@pytest.mark.parametrize(
    ("windows", "fade", "float64"),
    [
        ([(0.9, 2.1), (4.9, 6.1), (7.0, 7.3)], FADE, False),
        ([(0.9, 2.1), (4.9, 6.1)], FADE, True),  # a float64 recording is cut as float32
        ([(0.9, 2.1), (4.9, 6.1)], 0.0, False),  # no fade
        ([(1.0, 1.0 + 1 / RATE), (2.0, 3.0)], FADE, False),  # a part of 1 sample
        ([(1.0, 1.0 + 3 / RATE)], FADE, False),  # a part shorter than two fades
        ([(1.0, 1.0 + 7 / RATE), (2.0, 2.0 + 320 / RATE)], FADE, False),
    ],
)
def test_cut_original_is_unchanged_bit_for_bit(
    windows: list[tuple[float, float]], fade: float, float64: bool
) -> None:
    audio = noise(10.0)
    if float64:
        audio = audio.astype(np.float64) + 1e-9
    expected = frozen_cut_original(audio, windows, fade=fade)
    clip = cut_original(audio, windows, fade=fade)
    assert clip.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(clip, expected)


def test_cut_parts_copies_each_window_before_any_fade() -> None:
    audio = noise(10.0)
    parts = cut_parts(audio, [(0.9, 2.1), (4.9, 6.1)])
    assert [part.dtype for part in parts] == [np.float32, np.float32]
    np.testing.assert_array_equal(parts[0], audio[round(0.9 * RATE) : round(2.1 * RATE)])
    np.testing.assert_array_equal(parts[1], audio[round(4.9 * RATE) : round(6.1 * RATE)])
    parts[0][:] = 0.0
    assert np.all(audio[round(0.9 * RATE) : round(2.1 * RATE)] > 0.0)


@pytest.mark.parametrize(
    ("audio", "windows", "message"),
    [
        (np.zeros((100, 2), dtype=np.float32), [(0.0, 0.001)], "mono"),
        (np.zeros(1000, dtype=np.int16), [(0.0, 0.01)], "floats"),
        (np.zeros(1000, dtype=np.float32), [], "at least one part"),
    ],
)
def test_cut_parts_keeps_the_three_errors_of_the_cut(
    audio: np.ndarray, windows: list[tuple[float, float]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        cut_parts(audio, windows)


def test_fade_part_returns_a_faded_copy_in_the_parts_own_type() -> None:
    part = noise(1.0).astype(np.float64)
    faded = fade_part(part, fade=FADE)
    n = round(FADE * RATE)
    assert faded.dtype == np.float64
    assert faded[0] == 0.0
    assert faded[-1] == 0.0
    np.testing.assert_array_equal(faded[n:-n], part[n:-n])
    assert not np.array_equal(faded[:n], part[:n])
    assert np.all(part > 0.0)  # the input is not changed
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)
    np.testing.assert_array_equal(faded[:n], part[:n] * ramp)


def test_fade_part_refuses_integers_and_two_dimensions() -> None:
    with pytest.raises(ValueError, match="floats"):
        fade_part(np.ones(100, dtype=np.int16), fade=FADE)
    with pytest.raises(ValueError, match="floats"):
        fade_part(np.ones((100, 2), dtype=np.float32), fade=FADE)


def test_join_parts_puts_exactly_4800_zeros_between_parts_and_returns_their_bounds() -> None:
    parts = [np.ones(10, np.float32), np.full(5, 2.0, np.float32), np.full(7, 3.0, np.float32)]
    clip, starts, ends = join_parts(parts)
    assert clip.dtype == np.float32
    assert starts == [0, 4810, 9615]
    assert ends == [10, 4815, 9622]
    assert len(clip) == 9622
    for part, start, end in zip(parts, starts, ends, strict=True):
        np.testing.assert_array_equal(clip[start:end], part)
    assert np.all(clip[10:4810] == 0.0)
    assert np.all(clip[4815:9615] == 0.0)


def test_join_parts_refuses_no_parts_and_mixed_types() -> None:
    with pytest.raises(ValueError, match="at least one part"):
        join_parts([])
    with pytest.raises(ValueError, match="one type"):
        join_parts([np.ones(3, np.float32), np.ones(3, np.float64)])


def speech_like(seconds: float, seed: int = 9001) -> np.ndarray:
    """Synthetic recording around zero, as 16-bit audio read as floats (n / 32768)."""
    rng = np.random.default_rng(seed)
    samples = np.clip(np.rint(rng.normal(0.0, 0.05, round(seconds * RATE)) * 32768), -32768, 32767)
    return (samples / 32768).astype(np.float32)


def sine(frequency: float, seconds: float = 1.0) -> np.ndarray:
    return np.sin(2 * np.pi * frequency * np.arange(round(seconds * RATE)) / RATE)


def middle_gain_db(frequency: float) -> float:
    """Gain of x2's filter on a 1 s sine, measured on its middle half, away from the edges."""
    x = sine(frequency)
    y = muffle(x, cutoff_hz=400, order=8)
    middle = slice(len(x) // 4, 3 * len(x) // 4)
    return float(20 * np.log10(clip_rms(y[middle]) / clip_rms(x[middle])))


def test_x2_filter_removes_twice_the_cutoff() -> None:
    assert middle_gain_db(800) <= -60.0  # expected -97.2 dB


def test_x2_filter_is_6_db_down_at_the_cutoff() -> None:
    assert middle_gain_db(400) == pytest.approx(-6.02, abs=0.05)


def test_x2_filter_keeps_low_frequencies() -> None:
    assert abs(middle_gain_db(100)) <= 0.1


def test_x2_filter_shifts_nothing_in_time() -> None:
    impulse = np.zeros(4001)
    impulse[2000] = 1.0
    response = muffle(impulse, cutoff_hz=400, order=8)
    assert int(np.argmax(response)) == 2000
    np.testing.assert_allclose(response, response[::-1], atol=1e-12)


def test_x2_filter_keeps_the_length_works_in_float64_and_leaves_its_input() -> None:
    part = speech_like(1.0)
    copy = part.copy()
    muffled = muffle(part, cutoff_hz=400, order=8)
    assert muffled.dtype == np.float64
    assert len(muffled) == len(part)
    np.testing.assert_array_equal(part, copy)


def test_x2_filter_refuses_a_part_shorter_than_its_padding() -> None:
    with pytest.raises(ValueError, match="too short for the x2 filter"):
        muffle(np.ones(27, np.float32), cutoff_hz=400, order=8)
    assert len(muffle(np.ones(28, np.float32), cutoff_hz=400, order=8)) == 28
    with pytest.raises(ValueError, match="floats"):
        muffle(np.ones(100, np.int16), cutoff_hz=400, order=8)


def test_x2_filter_pads_its_ends_as_scipy_does_by_default() -> None:
    part = speech_like(0.125)  # 2,000 samples: the edges matter here
    sos = butter(8, 400, fs=RATE, output="sos")
    expected = sosfiltfilt(sos, part.astype(np.float64), padtype="odd", padlen=27)
    np.testing.assert_array_equal(muffle(part, cutoff_hz=400, order=8), expected)
    np.testing.assert_array_equal(expected, sosfiltfilt(sos, part.astype(np.float64)))


def test_reverse_windows_restarts_at_each_part_and_reverses_a_short_last_window() -> None:
    part = np.arange(10_000, dtype=np.float32)
    reversed_part = reverse_windows(part, window=4000)
    np.testing.assert_array_equal(reversed_part[:4000], np.arange(3999, -1, -1))
    np.testing.assert_array_equal(reversed_part[4000:8000], np.arange(7999, 3999, -1))
    np.testing.assert_array_equal(reversed_part[8000:], np.arange(9999, 7999, -1))
    np.testing.assert_array_equal(reverse_windows(reversed_part, window=4000), part)
    np.testing.assert_array_equal(reverse_windows(part[:100], window=4000), part[:100][::-1])
    assert reversed_part.dtype == np.float32
    with pytest.raises(ValueError, match="at least one sample"):
        reverse_windows(part, window=0)


def test_reverse_whole_plays_the_part_backwards() -> None:
    part = np.arange(10, dtype=np.float32)
    np.testing.assert_array_equal(reverse_whole(part), part[::-1])
    np.testing.assert_array_equal(reverse_whole(reverse_whole(part)), part)
    assert reverse_whole(part).flags.owndata


WINDOWS = [(0.5, 1.8), (2.5, 2.52), (3.0, 4.2)]  # 20,800, 320 and 19,200 samples


def test_views_share_their_length_and_parts() -> None:
    views = build_segment_views(speech_like(5.0), WINDOWS, SETTINGS)
    assert len(views.x0) == len(views.x2) == len(views.rev)
    assert views.part_starts == (0, 25_600, 30_720)
    assert views.part_ends == (20_800, 25_920, 49_920)
    assert views.x0.dtype == np.float32
    assert views.rev.dtype == np.float32
    assert views.x2.dtype == np.float64
    for clips in (
        {"x0": views.x0, "x2": views.x2, "rev": views.rev},
        {"x0": to_pcm16(views.x0), "x2": to_pcm16(views.x2), "rev": to_pcm16(views.rev)},
    ):
        check_segment_views(clips, views.part_starts, views.part_ends, gap=4800)


def test_x0_of_the_views_is_the_original_clip() -> None:
    audio = speech_like(5.0)
    views = build_segment_views(audio, WINDOWS, SETTINGS)
    np.testing.assert_array_equal(views.x0, cut_original(audio, WINDOWS, fade=FADE))


def test_each_rev_part_is_its_raw_part_reversed_by_windows_then_faded() -> None:
    audio = speech_like(5.0)
    views = build_segment_views(audio, WINDOWS, SETTINGS)
    n = round(FADE * RATE)
    for raw, start, end in zip(
        cut_parts(audio, WINDOWS), views.part_starts, views.part_ends, strict=True
    ):
        part = views.rev[start:end]
        np.testing.assert_array_equal(part, fade_part(reverse_windows(raw, window=4000), fade=FADE))
        if len(raw) > 2 * n:
            np.testing.assert_array_equal(part[n:-n], reverse_windows(raw, window=4000)[n:-n])


def test_x0s_fade_zone_appears_unfaded_in_rev() -> None:
    audio = speech_like(5.0)
    views = build_segment_views(audio, WINDOWS, SETTINGS)
    raw = cut_parts(audio, WINDOWS)[0]  # 20,800 samples, at least 4,160
    n = round(FADE * RATE)
    rev = views.rev[: len(raw)]
    np.testing.assert_array_equal(rev[3999 - np.arange(n)], raw[:n])
    assert not np.array_equal(views.x0[1:n], raw[1:n])


def test_each_x2_part_is_the_gain_times_the_faded_filtered_x0_part() -> None:
    audio = speech_like(5.0)
    views = build_segment_views(audio, WINDOWS, SETTINGS)
    assert not views.x2_peak_limited
    for start, end in zip(views.part_starts, views.part_ends, strict=True):
        x0_part = views.x0[start:end]
        expected = views.x2_gain * fade_part(muffle(x0_part, cutoff_hz=400, order=8), fade=FADE)
        np.testing.assert_allclose(views.x2[start:end], expected, rtol=1e-12, atol=1e-15)
        assert views.x2[start] == 0.0
        assert views.x2[end - 1] == 0.0
    assert clip_rms(views.x2) == pytest.approx(clip_rms(views.x0), rel=1e-12)


def test_building_the_views_changes_no_input() -> None:
    audio = speech_like(5.0)
    copy = audio.copy()
    windows = list(WINDOWS)
    build_segment_views(audio, windows, SETTINGS)
    np.testing.assert_array_equal(audio, copy)
    assert windows == WINDOWS


def test_a_loud_x2_clip_is_scaled_down_to_the_peak_limit_and_stays_below_its_rms() -> None:
    # A loud 100 Hz square wave: the filter keeps a near-sine whose RMS-matched peak is ~1.3.
    square = (0.9 * np.sign(sine(100, 3.0) + 1e-9)).astype(np.float32)
    views = build_segment_views(square, [(0.5, 2.5)], SETTINGS)
    assert views.x2_peak_limited
    assert clip_peak(views.x2) == pytest.approx(0.99, abs=1e-12)
    assert clip_rms(views.x2) < clip_rms(views.x0)
    for start, end in zip(views.part_starts, views.part_ends, strict=True):
        expected = views.x2_gain * fade_part(
            muffle(views.x0[start:end], cutoff_hz=400, order=8), fade=FADE
        )
        np.testing.assert_allclose(views.x2[start:end], expected, rtol=1e-12, atol=1e-15)


def test_a_silent_x0_stops_the_build_of_x2() -> None:
    with pytest.raises(ValueError, match="x0 is silent"):
        build_segment_views(np.zeros(5 * RATE, np.float32), WINDOWS, SETTINGS)


def test_check_segment_views_catches_a_moved_part_bound() -> None:
    views = build_segment_views(speech_like(5.0), WINDOWS, SETTINGS)
    clips = {"x0": views.x0, "rev": views.rev}
    moved = list(views.part_starts)
    moved[1] += 1
    with pytest.raises(ValueError, match="gap of 4801"):
        check_segment_views(clips, moved, views.part_ends, gap=4800)
    with pytest.raises(ValueError, match="do not span"):
        check_segment_views(clips, views.part_starts, [*views.part_ends[:-1], 1], gap=4800)


@pytest.mark.parametrize("view", ["x0", "x2", "rev"])
@pytest.mark.parametrize("pcm16", [False, True])
@pytest.mark.parametrize(
    ("place", "message"),
    [
        (lambda starts, ends: starts[0], "a part does not start and end at 0"),
        (lambda starts, ends: starts[1], "a part does not start and end at 0"),
        (lambda starts, ends: ends[1] - 1, "a part does not start and end at 0"),
        (lambda starts, ends: ends[2] - 1, "a part does not start and end at 0"),
        (lambda starts, ends: ends[0], "a gap is not silent"),  # first gap sample
        (lambda starts, ends: starts[1] - 1, "a gap is not silent"),  # last gap sample
        (lambda starts, ends: ends[0] + 2400, "a gap is not silent"),
    ],
)
def test_check_segment_views_catches_a_sample_planted_at_an_edge_or_in_a_gap(
    view: str, pcm16: bool, place: object, message: str
) -> None:
    views = build_segment_views(speech_like(5.0), WINDOWS, SETTINGS)
    clips = {"x0": views.x0, "x2": views.x2, "rev": views.rev}
    if pcm16:
        clips = {name: to_pcm16(clip) for name, clip in clips.items()}
    clips[view] = clips[view].copy()
    clips[view][place(views.part_starts, views.part_ends)] = 1 if pcm16 else 1e-6  # type: ignore[operator]
    with pytest.raises(ValueError, match=f"{view}: {message}"):
        check_segment_views(clips, views.part_starts, views.part_ends, gap=4800)


def test_check_segment_views_catches_views_of_different_lengths() -> None:
    views = build_segment_views(speech_like(5.0), WINDOWS, SETTINGS)
    with pytest.raises(ValueError, match="differ in length"):
        check_segment_views(
            {"x0": views.x0, "rev": views.rev[:-1]}, views.part_starts, views.part_ends, gap=4800
        )


def test_set_rms_reaches_the_target_and_refuses_silence_and_nan() -> None:
    clip = speech_like(1.0)
    scaled, gain = set_rms(clip, 0.05)
    assert scaled.dtype == np.float64
    assert clip_rms(scaled) == pytest.approx(0.05, rel=1e-12)
    assert gain == pytest.approx(0.05 / clip_rms(clip))
    with pytest.raises(ValueError, match="silent"):
        set_rms(np.zeros(100), 0.05)
    with pytest.raises(ValueError, match="NaN"):
        set_rms(np.array([0.1, np.nan]), 0.05)
    with pytest.raises(ValueError, match="positive"):
        set_rms(clip, 0.0)


def test_the_peak_guard_acts_only_above_its_limit() -> None:
    clip = np.array([0.0, 0.99, -0.5])
    same, factor, limited = limit_peak(clip, 0.99)
    assert same is clip
    assert (factor, limited) == (1.0, False)
    loud = np.array([0.0, -1.2, 0.6])
    scaled, factor, limited = limit_peak(loud, 0.99)
    assert limited
    assert factor == pytest.approx(0.99 / 1.2)
    assert clip_peak(scaled) == pytest.approx(0.99)
    with pytest.raises(ValueError, match="NaN"):
        limit_peak(np.array([np.nan]), 0.99)


def test_levels_are_measured_in_float64_over_the_whole_clip() -> None:
    samples = np.array([-32768, 0, 0, 16384], dtype=np.int16)
    values = samples.astype(np.float64) / 32768
    assert clip_peak(values) == 1.0
    assert clip_rms(values) == pytest.approx(np.sqrt((1 + 0.25) / 4))
    assert clip_rms(np.full(4, 0.5, np.float32)) == 0.5
    with pytest.raises(ValueError, match="empty"):
        clip_rms(np.zeros(0))
    with pytest.raises(ValueError, match="empty"):
        clip_peak(np.zeros(0))


def test_to_pcm16_gives_back_the_numbers_of_a_16_bit_recording() -> None:
    numbers = np.array([-32768, -1, 0, 1, 32767, 12345], dtype=np.int16)
    np.testing.assert_array_equal(to_pcm16((numbers / 32768).astype(np.float32)), numbers)
    halves = np.array([0.5, 1.5, -0.5, 2.5]) / 32768
    np.testing.assert_array_equal(to_pcm16(halves), [0, 2, 0, 2])  # halfway: to even


@pytest.mark.parametrize(
    ("clip", "message"),
    [
        (np.array([0.1, np.nan]), "NaN"),
        (np.array([np.inf]), "NaN"),
        (np.array([1.0]), "outside the 16-bit range"),
        (np.array([-1.0001]), "outside the 16-bit range"),
        (np.array([1], dtype=np.int16), "floats"),
    ],
)
def test_to_pcm16_refuses_instead_of_clipping(clip: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        to_pcm16(clip)


def test_stored_windows_reads_the_windows_of_a_segment_row() -> None:
    row = SimpleNamespace(window_starts=np.array([1.0, 3.0]), window_ends=np.array([2.0, 4.5]))
    assert stored_windows(row) == [(1.0, 2.0), (3.0, 4.5)]
    assert all(type(value) is float for window in stored_windows(row) for value in window)


def test_view_settings_come_from_the_data_config() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert ViewSettings.from_config(config) == SETTINGS


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"flac_bits": 24}, "16-bit"),
        ({"rev": {"window_ms": 250.03}}, "whole number of samples"),
        ({"peak_limit": 1.2}, "peak limit"),
        ({"x2": {"cutoff_hz": 9000, "butterworth_order": 8}}, "cutoff"),
    ],
)
def test_view_settings_refuse_other_files_and_impossible_values(change: dict, message: str) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["views"] = {**config["views"], **change}
    with pytest.raises(ValueError, match=message):
        ViewSettings.from_config(config)


def test_data_config_pins_the_view_and_sample_settings() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["views"] == {
        "peak_limit": 0.99,
        "flac_bits": 16,
        "x1": {
            "voices": {"female": ["af_heart", "af_bella"], "male": ["am_michael", "am_fenrir"]},
            "language": "a",
            "speed": 1.0,
            "seed": 106,
            "kokoro_rate_hz": 24000,
            "soxr_quality": "HQ",
            "trim_db": -40,
            "trim_margin_ms": 10,
            "rms": 0.05,
            "threads": 4,
            "test_render": {"segments": 20, "seed": 106, "repeats": 2, "heard": 10},
        },
        "x2": {"cutoff_hz": 400, "butterworth_order": 8},
        "rev": {"window_ms": 250},
    }
    assert config["word_check"]["sample"] == {"seed": 105, "segments_per_participant": 5}


def test_data_config_pins_every_setting_of_the_word_check() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    expected = {
        "sample": {"seed": 105, "segments_per_participant": 5},
        "transcribed_views": ["x0", "x1", "x2", "rev"],
        "partition_round": 1,
        "whisper": {
            "compute_type": "float16",
            "language": "en",
            "task": "transcribe",
            "beam_size": 5,
            "patience": 1.0,
            "length_penalty": 1.0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "temperature": 0.0,
            "compression_ratio_threshold": None,
            "log_prob_threshold": None,
            "no_speech_threshold": None,
            "hallucination_silence_threshold": None,
            "condition_on_previous_text": False,
            "initial_prompt": None,
            "prefix": None,
            "hotwords": None,
            "suppress_blank": True,
            "suppress_tokens": [-1],
            "without_timestamps": False,
            "max_initial_timestamp": 1.0,
            "word_timestamps": False,
            "multilingual": False,
            "vad_filter": False,
            "max_new_tokens": None,
            "chunk_length": None,
            "clip_timestamps": "0",
        },
        "wav2vec2": {"dtype": "float32", "decoding": "greedy"},
        "rules": {
            "rev_min_wer": 0.80,
            "rev_max_leakage_auroc": 0.60,
            "x0_max_wer": 0.40,
            "x0_leakage_interval": 0.95,
            "x1_max_wer": 0.15,
            "x1_max_icc": 0.30,
            "probe_margin_auroc": 0.05,
        },
        "leakage": {
            "ngram_range": [1, 2],
            "min_df": 2,
            "token_pattern": r"\S+",
            "logistic_c": 1.0,
            "max_iter": 5000,
        },
        "bootstrap": {"resamples": 10000, "seed": 105},
        "voice": {
            "measures": [
                "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
                "loudness_sma3_stddevNorm",
                "VoicedSegmentsPerSec",
            ],
            "level_rms": 0.05,
            "min_part_samples": 960,
            "probe_logistic_c": 0.1,
            "probe_max_iter": 5000,
        },
        "recheck_seed": 107,
        "rerun_time_factor": 1.5,
        "public_guard": {"run": "derived/check_view_tools/run-12529760", "max_wer": 0.10},
    }
    assert config["word_check"] == expected
    # The same types too: 1.0 is not 1, "0" is not 0 and False is not 0.
    assert json.dumps(config["word_check"], sort_keys=True) == json.dumps(expected, sort_keys=True)


def test_data_config_pins_each_built_view_and_the_sample_by_a_file_hash() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    runs = config["view_runs"]
    assert set(runs) == {"x0", "x1", "x2", "rev"}
    for view, pin in runs.items():
        assert set(pin) == {"run", "sha256"}
        assert pin["run"].startswith(f"views/{view}/run-")
        assert len(pin["sha256"]) == 64
        assert set(pin["sha256"]) <= set("0123456789abcdef")
    jobs = {view: pin["run"].rsplit("-", 1)[1] for view, pin in runs.items()}
    assert len({jobs["x0"], jobs["x2"], jobs["rev"]}) == 1  # one build job (build_views)
    assert jobs["x1"] != jobs["x0"]  # its own build (build_revoiced and its merge)
    sample = config["wordcheck_sample"]
    assert sample["run"].startswith("derived/wordcheck_sample/run-")
    assert len(sample["sha256"]) == 64
