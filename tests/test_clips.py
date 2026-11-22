import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import listening_beyond_words.clips as clips
from listening_beyond_words.clips import (
    LISTENING_PEAK,
    LISTENING_RMS,
    MANIFEST_COLUMNS,
    audio_sha256,
    format_windows,
    listening_copy,
    listening_gain,
    listening_pcm16,
    load_segments,
    read_key,
    read_session,
    write_view,
)
from listening_beyond_words.runinfo import sha256_file
from listening_beyond_words.settings import Workspace
from listening_beyond_words.views import cut_original, to_pcm16

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


def recording(root: Path, session: int, samples: np.ndarray, **options: object) -> Path:
    """Write a synthetic recording where the session guard expects it (pilot ID, temp folder)."""
    folder = root / "edaic" / "sessions" / f"{session}_P"
    folder.mkdir(parents=True)
    path = folder / f"{session}_AUDIO.wav"
    sf.write(path, samples, **{"samplerate": RATE, "subtype": "PCM_16", **options})
    return path


def test_read_session_reads_16_bit_audio_as_n_over_32768(tmp_path: Path) -> None:
    numbers = np.array([-32768, -1, 0, 1, 32767] * 100, dtype=np.int16)
    recording(tmp_path, 300, numbers)
    audio = read_session(tmp_path / "edaic" / "sessions", 300)
    assert audio.dtype == np.float32
    np.testing.assert_array_equal(audio, numbers / 32768)
    np.testing.assert_array_equal(to_pcm16(audio), numbers)


@pytest.mark.parametrize("session", [600, 718, 9001])
def test_read_session_refuses_every_session_outside_the_pilot(tmp_path: Path, session: int) -> None:
    with pytest.raises(ValueError, match="not a pilot session"):
        read_session(tmp_path / "edaic" / "sessions", session)


@pytest.mark.parametrize(
    ("samples", "options", "message"),
    [
        (np.zeros((1000, 2), np.int16), {}, "2 channel(s)"),
        (np.zeros(1000, np.int16), {"samplerate": 44_100}, "44100 Hz"),
        (np.zeros(1000, np.int32), {"subtype": "PCM_24"}, "PCM_24"),
    ],
)
def test_read_session_refuses_stereo_other_rates_and_other_bit_depths(
    tmp_path: Path, samples: np.ndarray, options: dict, message: str
) -> None:
    recording(tmp_path, 300, samples, **options)
    with pytest.raises(ValueError, match="expected 16000 Hz mono 16-bit PCM") as error:
        read_session(tmp_path / "edaic" / "sessions", 300)
    assert message in str(error.value)


def test_write_view_stores_the_samples_exactly_and_returns_the_manifest_values(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(9001)
    samples = rng.integers(-32768, 32768, 16_000).astype(np.int16)
    samples[10] = -32768
    values = write_view(tmp_path / "a.flac", samples)
    back, rate = sf.read(tmp_path / "a.flac", dtype="int16")
    assert rate == RATE
    np.testing.assert_array_equal(back, samples)
    assert sf.info(tmp_path / "a.flac").subtype == "PCM_16"
    assert values["samples"] == 16_000
    assert values["peak"] == 1.0
    assert values["rms"] == pytest.approx(np.sqrt(np.mean((samples / 32768.0) ** 2)), rel=1e-12)
    assert values["sha256"] == sha256_file(tmp_path / "a.flac")
    assert values["audio_sha256"] == hashlib.sha256(samples.astype("<i2").tobytes()).hexdigest()
    assert values["audio_sha256"] == audio_sha256(back)
    assert set(values) < set(MANIFEST_COLUMNS)
    # The same samples give the same file, byte for byte.
    assert write_view(tmp_path / "b.flac", samples)["sha256"] == values["sha256"]


def test_write_view_never_overwrites_a_file(tmp_path: Path) -> None:
    samples = np.ones(100, np.int16)
    write_view(tmp_path / "a.flac", samples)
    with pytest.raises(ValueError, match="exists already"):
        write_view(tmp_path / "a.flac", samples)


def test_write_view_refuses_a_file_that_reads_back_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read = sf.read

    def changed(*args: object, **kwargs: object) -> tuple[np.ndarray, int]:
        samples, rate = read(*args, **kwargs)
        samples[5] += 1
        return samples, rate

    monkeypatch.setattr(clips.sf, "read", changed)
    with pytest.raises(ValueError, match="does not read back as written"):
        write_view(tmp_path / "a.flac", np.zeros(100, np.int16))


def test_write_view_refuses_floats(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="int16"):
        write_view(tmp_path / "a.flac", np.zeros(100, np.float32))


def frozen_listening_copy(clip: np.ndarray) -> np.ndarray:
    """The listening copy as scripts/listening_check.py made it at commit 2645d45."""
    rms = float(np.sqrt(np.mean(np.square(clip, dtype=np.float64))))
    peak = float(np.max(np.abs(clip)))
    gain = 1.0 if rms == 0.0 or peak == 0.0 else min(0.1 / rms, 0.99 / peak)
    return clip * np.float32(gain)


def listening_clips() -> list[np.ndarray]:
    """A quiet clip, whose gain the RMS sets, and one whose gain the peak limits."""
    rng = np.random.default_rng(9001)
    audio = (np.rint(rng.normal(0, 0.03, 5 * RATE) * 32768) / 32768).astype(np.float32)
    quiet = cut_original(audio, [(0.5, 2.0), (3.0, 4.5)], fade=0.01)
    peaky = quiet.copy()
    peaky[RATE] = 0.9
    return [quiet, peaky]


@pytest.mark.parametrize("which", [0, 1])
def test_listening_copy_is_unchanged_from_the_listening_checks(which: int) -> None:
    clip = listening_clips()[which]
    copy, gain = listening_copy(clip)
    np.testing.assert_array_equal(copy, frozen_listening_copy(clip))
    assert np.float32(gain) * clip[RATE] == frozen_listening_copy(clip)[RATE]
    if which == 1:
        assert gain == 0.99 / float(np.float32(0.9))  # the peak limits the gain, as before


@pytest.mark.parametrize("which", [0, 1])
def test_listening_pcm16_recreates_a_listening_file_bit_for_bit(tmp_path: Path, which: int) -> None:
    clip = listening_clips()[which]
    # The heard file, written as scripts/listening_check.py wrote it at commit 2645d45:
    sf.write(tmp_path / "01.wav", frozen_listening_copy(clip), RATE, subtype="PCM_16")
    heard, _ = sf.read(tmp_path / "01.wav", dtype="int16")
    np.testing.assert_array_equal(listening_pcm16(clip), heard)
    # The WAV writer rounds down, so rounding to nearest would not match it.
    assert not np.array_equal(to_pcm16(frozen_listening_copy(clip)), heard)
