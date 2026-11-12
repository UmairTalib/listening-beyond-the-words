import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from listening_beyond_words.segments import build_segments, segment_neighbours
from listening_beyond_words.views import cut_original, part_windows

RATE = 16_000
MARGIN = 0.1  # the pilot's settings, configs/pilot/data.yaml
FADE = 0.01


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
    assert set(cut) == {"margin_ms", "fade_ms"}
    assert 0 < cut["fade_ms"] < cut["margin_ms"]
