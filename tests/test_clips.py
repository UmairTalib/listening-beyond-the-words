import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from listening_beyond_words.clips import (
    LISTENING_PEAK,
    LISTENING_RMS,
    format_windows,
    listening_copy,
    listening_gain,
    load_segments,
    read_key,
)
from listening_beyond_words.runinfo import sha256_file
from listening_beyond_words.settings import Workspace

RATE = 16_000
CUT = {"guard_ms": 200, "offsets_s": {9001: 3.5}}  # session keys become strings in JSON
X0 = {"margin_ms": 100, "fade_ms": 10, "wall_guard_ms": 20}


def pinned_segments(root: Path) -> dict:
    """A workspace with a segments run and the data config that pins it."""
    run = root / "derived" / "segments" / "run-1"
    run.mkdir(parents=True)
    table = pd.DataFrame(
        {"segment": ["9002_000", "9001_001", "9001_000"], "session": [9002, 9001, 9001]}
    )
    table.to_parquet(run / "segments.parquet", index=False)
    built = {
        "inputs": {"word_times": {"sha256": "w" * 64}, "participant_lines": {"sha256": "l" * 64}},
        "settings": {"daicwoz_cut": CUT, "x0_cut": X0},
    }
    (run / "run.json").write_text(json.dumps(built), encoding="utf-8")
    return {
        "segments": {
            "run": "derived/segments/run-1",
            "sha256": sha256_file(run / "segments.parquet"),
        },
        "word_times": {"sha256": "w" * 64},
        "participant_lines": {"sha256": "l" * 64},
        "daicwoz_cut": CUT,
        "x0_cut": X0,
    }


def test_load_segments_returns_the_pinned_table_sorted_by_segment(tmp_path: Path) -> None:
    config = pinned_segments(tmp_path)
    table = load_segments(Workspace(tmp_path), config)
    assert table["segment"].tolist() == ["9001_000", "9001_001", "9002_000"]


def test_load_segments_refuses_a_changed_file(tmp_path: Path) -> None:
    config = pinned_segments(tmp_path)
    path = tmp_path / config["segments"]["run"] / "segments.parquet"
    pd.DataFrame({"segment": ["9001_000"], "session": [9001]}).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="does not match"):
        load_segments(Workspace(tmp_path), config)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"x0_cut": {**X0, "fade_ms": 20}}, "other x0_cut settings"),
        ({"daicwoz_cut": {**CUT, "guard_ms": 500}}, "other daicwoz_cut settings"),
        ({"word_times": {"sha256": "x" * 64}}, "pinned word_times"),
        ({"participant_lines": {"sha256": "x" * 64}}, "pinned participant_lines"),
    ],
)
def test_load_segments_refuses_other_inputs_or_settings(
    tmp_path: Path, change: dict, message: str
) -> None:
    config = pinned_segments(tmp_path)
    with pytest.raises(ValueError, match=message):
        load_segments(Workspace(tmp_path), {**config, **change})


def test_read_key_parses_the_windows_and_checks_the_hash(tmp_path: Path) -> None:
    windows = [[(1.0, 2.0)], [(10.25, 11.5), (12.0001, 13.9999)]]
    key = pd.DataFrame(
        {
            "clip": ["01.wav", "02.wav"],
            "segment": ["9001_000", "9002_013"],
            "windows": [format_windows(w) for w in windows],
        }
    )
    path = tmp_path / "key.tsv"
    key.to_csv(path, sep="\t", index=False)
    read = read_key(path, sha256_file(path))
    assert read["segment"].tolist() == ["9001_000", "9002_013"]
    assert read["session"].tolist() == [9001, 9002]
    assert read["windows"].tolist() == windows
    assert read["clip"].tolist() == ["01.wav", "02.wav"]
    with pytest.raises(ValueError, match="does not match"):
        read_key(path, "0" * 64)


def test_format_windows_writes_seconds_to_a_tenth_of_a_millisecond() -> None:
    assert format_windows([(0.9, 2.03), (4.123449, 5.0)]) == "0.9000-2.0300; 4.1234-5.0000"


def test_listening_gain_brings_a_quiet_clip_to_the_common_loudness() -> None:
    clip = np.full(1000, 0.01, dtype=np.float32)
    clip[::2] *= -1
    assert listening_gain(clip) == pytest.approx(LISTENING_RMS / 0.01)


def test_listening_gain_stops_short_of_clipping() -> None:
    clip = np.zeros(1000, dtype=np.float32)
    clip[0] = 0.5  # RMS far below 0.1, but the peak limits the gain
    assert listening_gain(clip) == pytest.approx(LISTENING_PEAK / 0.5)
    assert listening_gain(np.zeros(10, np.float32)) == 1.0


def test_listening_copy_is_the_float32_clip_times_its_gain(tmp_path: Path) -> None:
    rng = np.random.default_rng(9001)
    clip = rng.uniform(-0.05, 0.05, RATE).astype(np.float32)
    copy, gain = listening_copy(clip)
    assert copy.dtype == np.float32
    np.testing.assert_array_equal(copy, clip * np.float32(gain))
    sf.write(tmp_path / "01.wav", copy, RATE, subtype="PCM_16")
    assert sf.info(tmp_path / "01.wav").subtype == "PCM_16"
