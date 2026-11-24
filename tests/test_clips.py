import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import listening_beyond_words.clips as clips
from listening_beyond_words.clips import (
    GAP_SAMPLES,
    LISTENING_PEAK,
    LISTENING_RMS,
    MANIFEST_COLUMNS,
    audio_sha256,
    format_windows,
    listening_copy,
    listening_gain,
    listening_pcm16,
    load_pinned_partitions,
    load_segments,
    read_key,
    read_session,
    read_view_clip,
    write_view,
)
from listening_beyond_words.runinfo import StepError, sha256_file
from listening_beyond_words.settings import Workspace
from listening_beyond_words.views import cut_original, fade_part, join_parts, to_pcm16
from x1_fakes import X1_LABELS

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


def test_load_cut_words_reads_the_words_of_the_pinned_segments_run(x1_workspace) -> None:
    words = clips.load_cut_words(Workspace(x1_workspace.root), x1_workspace.config)
    expected = x1_workspace.words.drop(columns=["word", "score"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(words, expected)
    assert "word" not in words.columns


def test_load_cut_words_refuses_a_changed_file_or_a_run_that_does_not_list_the_pin(
    x1_workspace,
) -> None:
    workspace, config = Workspace(x1_workspace.root), x1_workspace.config
    run = x1_workspace.root / "derived" / "segments" / "run-1"
    built = json.loads((run / "run.json").read_text(encoding="utf-8"))
    for change in (
        {"status": "failed"},
        {"outputs": {**built["outputs"], "segments.parquet": "x"}},
    ):
        (run / "run.json").write_text(json.dumps({**built, **change}), encoding="utf-8")
        with pytest.raises(ValueError, match="does not list the pinned segments"):
            clips.load_cut_words(workspace, config)
    outputs = {"segments.parquet": built["outputs"]["segments.parquet"]}
    (run / "run.json").write_text(json.dumps({**built, "outputs": outputs}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"no SHA-256 for words\.parquet"):
        clips.load_cut_words(workspace, config)
    (run / "run.json").write_text(json.dumps(built), encoding="utf-8")
    words = pd.read_parquet(run / "words.parquet")
    words.iloc[:-1].to_parquet(run / "words.parquet", index=False)
    with pytest.raises(ValueError, match="does not match"):
        clips.load_cut_words(workspace, config)


def test_read_lines_gives_the_text_of_each_line_in_the_aligners_order(tmp_path: Path) -> None:
    folder = tmp_path / "300_P"
    folder.mkdir()
    (folder / "300_Transcript.csv").write_text(
        'Start_Time,End_Time,Text,Confidence\n9.0,10.0,"second, line",0.9\n1.0,2.0,first,0.8\n',
        encoding="utf-8",
    )
    assert clips.read_lines(tmp_path, 300) == ["first", "second, line"]


@pytest.mark.parametrize("session", [9001, 600, 342])
def test_read_lines_refuses_every_session_outside_the_pilot(tmp_path: Path, session: int) -> None:
    with pytest.raises(ValueError, match="not a pilot session"):
        clips.read_lines(tmp_path, session)


def test_the_pinned_readers_return_labels_sample_and_manifest(x1_workspace) -> None:
    workspace, config = Workspace(x1_workspace.root), x1_workspace.config
    labels = clips.load_pinned_labels(workspace, config)
    assert list(labels.columns) == ["label", "gender"]
    assert labels.loc[301, "label"] == 1
    assert labels.loc[300, "gender"] == "female"
    sample = clips.load_pinned_sample(workspace, config)
    assert sample["segment"].tolist() == ["300_001", "301_000"]
    manifest = clips.load_view_manifest(workspace, config, "x0")
    assert list(manifest.columns) == list(MANIFEST_COLUMNS)
    assert len(manifest) == len(x1_workspace.segments)


@pytest.mark.parametrize(
    ("reader", "file"),
    [
        (lambda w, c: clips.load_pinned_labels(w, c), "derived/labels/run-1/labels.parquet"),
        (
            lambda w, c: clips.load_pinned_sample(w, c),
            "derived/wordcheck_sample/run-1/sample.parquet",
        ),
        (lambda w, c: clips.load_view_manifest(w, c, "x0"), "views/x0/run-1/manifest.parquet"),
    ],
)
def test_the_pinned_readers_refuse_a_changed_file(x1_workspace, reader, file: str) -> None:
    path = x1_workspace.root / file
    table = pd.read_parquet(path)
    table.iloc[:-1].to_parquet(path)
    with pytest.raises(ValueError, match="does not match"):
        reader(Workspace(x1_workspace.root), x1_workspace.config)


def partitions_table() -> pd.DataFrame:
    """Partitions of the five labelled synthetic participants (300-304), one per fold."""
    return pd.DataFrame(
        [
            {"session": session, "round": number, "fold": (k + number - 1) % 5 + 1}
            for number in (1, 2)
            for k, session in enumerate(sorted(X1_LABELS))
        ]
    )


def pin_partitions(
    workspace: Any, table: pd.DataFrame | None = None, built: dict | None = None
) -> dict:
    """Write a partitions run made from the pinned labels; return the config that pins it."""
    run = workspace.root / "derived" / "partitions" / "run-1"
    run.mkdir(parents=True, exist_ok=True)
    (partitions_table() if table is None else table).to_parquet(
        run / "partitions.parquet", index=False
    )
    if built is None:
        labels = workspace.config["labels"]
        built = {
            "step": "build_partitions",
            "status": "done",
            "inputs": {"labels": {"file": labels["run"], "sha256": labels["sha256"]}},
            "settings": {"seeds": [101, 102], "folds": 5, "stratified_by": ["label", "gender"]},
        }
    (run / "run.json").write_text(json.dumps(built), encoding="utf-8")
    pin = {"run": "derived/partitions/run-1", "sha256": sha256_file(run / "partitions.parquet")}
    return {**workspace.config, "partitions": pin}


def test_load_pinned_partitions_gives_each_labelled_participant_one_fold_per_round(
    x1_workspace,
) -> None:
    config = pin_partitions(x1_workspace)
    partitions = load_pinned_partitions(Workspace(x1_workspace.root), config)
    assert list(partitions.columns) == ["session", "round", "fold"]
    assert partitions["round"].tolist() == [1] * 5 + [2] * 5
    for number in (1, 2):
        part = partitions[partitions["round"] == number]
        assert part["session"].tolist() == sorted(X1_LABELS)
        assert sorted(part["fold"]) == [1, 2, 3, 4, 5]
    assert partitions.set_index(["round", "session"]).loc[(2, 300), "fold"] == 2


def test_load_pinned_partitions_refuses_a_changed_file(x1_workspace) -> None:
    config = pin_partitions(x1_workspace)
    path = x1_workspace.root / "derived" / "partitions" / "run-1" / "partitions.parquet"
    table = pd.read_parquet(path)
    table.loc[0, "fold"] = table.loc[1, "fold"]
    table.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="does not match"):
        load_pinned_partitions(Workspace(x1_workspace.root), config)


@pytest.mark.parametrize(
    "inputs", [{"labels": {"sha256": "x" * 64}}, {"labels": {}}, {}], ids=["other", "none", "no"]
)
def test_load_pinned_partitions_refuses_partitions_made_from_other_labels(
    x1_workspace, inputs: dict
) -> None:
    config = pin_partitions(x1_workspace, built={"step": "build_partitions", "inputs": inputs})
    with pytest.raises(ValueError, match="not made from the pinned labels"):
        load_pinned_partitions(Workspace(x1_workspace.root), config)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda t: t.drop(index=7), "round 2: 1 labelled participants missing, 0 extra rows"),
        (
            lambda t: pd.concat([t, t.iloc[[1]].assign(fold=4)]),
            "round 1: 0 labelled participants missing, 1 extra rows",
        ),
        (
            lambda t: pd.concat([t, pd.DataFrame([{"session": 9001, "round": 1, "fold": 5}])]),
            "round 1: .*, 1 participants without a label",
        ),
        (lambda t: pd.concat([t, t.iloc[:5].assign(round=3)]), r"rounds \[1, 2, 3\], not \[1, 2\]"),
        (
            lambda t: t.assign(fold=t["fold"].replace(5, 6)),
            r"round 1 holds folds \[1, 2, 3, 4, 6\]",
        ),
        (lambda t: t.drop(columns="fold"), "do not have the columns"),
    ],
    ids=["missing", "twice", "unlabelled", "round 3", "fold 6", "no folds"],
)
def test_load_pinned_partitions_refuses_a_participant_missing_from_a_round_or_in_it_twice(
    x1_workspace, change: Callable[[pd.DataFrame], pd.DataFrame], message: str
) -> None:
    config = pin_partitions(x1_workspace, change(partitions_table()))
    with pytest.raises(ValueError, match=message):
        load_pinned_partitions(Workspace(x1_workspace.root), config)


SEGMENT = "300_001"
CLIP_FILE = f"300/{SEGMENT}.flac"


def three_parts() -> tuple[np.ndarray, list[int], list[int]]:
    """A synthetic clip as a view stores it: three faded noise parts, 300 ms gaps, int16."""
    rng = np.random.default_rng(9001)
    parts = [
        fade_part(rng.uniform(-0.3, 0.3, length).astype(np.float32), fade=0.01)
        for length in (8000, 3000, 12000)
    ]
    clip, starts, ends = join_parts(parts)
    return to_pcm16(clip), starts, ends


def store(run_dir: Path, samples: np.ndarray, rate: int = RATE, subtype: str = "PCM_16") -> None:
    """Write the clip's file where its manifest row says it is, replacing any earlier file."""
    path = run_dir / CLIP_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, rate, subtype=subtype, format="FLAC")


def manifest_row(
    run_dir: Path, clip: np.ndarray, starts: list[int], ends: list[int], **changes: Any
) -> Any:
    """The clip's manifest row as a step reads it: written to Parquet and read back."""
    row = {
        "segment": SEGMENT,
        "session": 300,
        "file": CLIP_FILE,
        "samples": len(clip),
        "rms": 0.1,
        "peak": 0.3,
        "sha256": sha256_file(run_dir / CLIP_FILE),
        "audio_sha256": audio_sha256(clip),
        "part_starts": starts,
        "part_ends": ends,
        "peak_limited": False,
        "gain_db": 0.0,
        **changes,
    }
    table = pd.DataFrame([row], columns=list(MANIFEST_COLUMNS))
    table.to_parquet(run_dir / "manifest.parquet", index=False)
    return next(pd.read_parquet(run_dir / "manifest.parquet").itertuples(index=False))


def test_read_view_clip_gives_the_stored_samples_and_part_bounds(tmp_path: Path) -> None:
    run_dir = tmp_path / "views" / "x1" / "run-1"
    samples, starts, ends = three_parts()
    (run_dir / "300").mkdir(parents=True)
    values = write_view(run_dir / CLIP_FILE, samples)  # as the builds write every clip
    row = manifest_row(run_dir, samples, starts, ends, **values)
    assert isinstance(row.part_starts, np.ndarray)  # Parquet gives the bounds back as arrays
    clip = read_view_clip(run_dir, row)
    stored, _ = sf.read(run_dir / CLIP_FILE, dtype="int16")
    assert clip.samples.dtype == np.int16
    np.testing.assert_array_equal(clip.samples, stored)
    np.testing.assert_array_equal(clip.samples, samples)
    assert clip.part_starts == (0, 8000 + GAP_SAMPLES, 11000 + 2 * GAP_SAMPLES)
    assert clip.part_ends == tuple(ends)
    assert all(type(bound) is int for bound in clip.part_starts + clip.part_ends)
    assert clip.audio.dtype == np.float32
    np.testing.assert_array_equal(clip.audio, samples / 32768)
    np.testing.assert_array_equal(to_pcm16(clip.audio), samples)


def missing(run_dir: Path, samples: np.ndarray, starts: list[int], ends: list[int]) -> Any:
    row = manifest_row(run_dir, samples, starts, ends)
    (run_dir / CLIP_FILE).unlink()
    return row


def another_rate(run_dir: Path, samples: np.ndarray, starts: list[int], ends: list[int]) -> Any:
    store(run_dir, samples, rate=8000)
    return manifest_row(run_dir, samples, starts, ends)


def two_channels(run_dir: Path, samples: np.ndarray, starts: list[int], ends: list[int]) -> Any:
    store(run_dir, np.stack([samples, samples], axis=1))
    return manifest_row(run_dir, samples, starts, ends)


def other_bits(run_dir: Path, samples: np.ndarray, starts: list[int], ends: list[int]) -> Any:
    store(run_dir, samples.astype(np.int32) << 16, subtype="PCM_24")
    return manifest_row(run_dir, samples, starts, ends)


def sound_in_a_gap(run_dir: Path, samples: np.ndarray, starts: list[int], ends: list[int]) -> Any:
    planted = samples.copy()
    planted[ends[0] + GAP_SAMPLES // 2] = 1  # one step of 16 bits, in the middle of the gap
    store(run_dir, planted)
    return manifest_row(run_dir, planted, starts, ends)


@pytest.mark.parametrize(
    ("prepare", "message"),
    [
        (missing, "its file is missing"),
        (
            lambda d, s, a, e: manifest_row(d, s, a, e, sha256="0" * 64),
            "its file does not match the SHA-256 of its manifest row",
        ),
        (another_rate, r"its file is 8000 Hz, 1 channel\(s\), PCM_16; expected 16000 Hz mono"),
        (two_channels, r"16000 Hz, 2 channel\(s\), PCM_16; expected 16000 Hz mono 16-bit"),
        (other_bits, "PCM_24; expected 16000 Hz mono 16-bit"),
        (
            lambda d, s, a, e: manifest_row(d, s, a, e, samples=len(s) + 1),
            f"its file holds {8000 + 3000 + 12000 + 2 * 4800} samples, its manifest row",
        ),
        (
            lambda d, s, a, e: manifest_row(d, s, a, e, audio_sha256="0" * 64),
            "its samples do not match the SHA-256 of its manifest row",
        ),
        (sound_in_a_gap, r"do not fit its manifest row \(clip: a gap is not silent\)"),
        (
            lambda d, s, a, e: manifest_row(d, s, [a[0], a[1] + 1, a[2]], e),
            "a gap of 4801 samples, not 4800",
        ),
    ],
    ids=["missing", "file", "rate", "channels", "bits", "length", "samples", "gap", "bound"],
)
def test_read_view_clip_refuses_a_clip_that_differs_from_its_manifest_row_naming_the_segment(
    tmp_path: Path, prepare: Callable[..., Any], message: str
) -> None:
    run_dir = tmp_path / "views" / "x1" / "run-1"
    samples, starts, ends = three_parts()
    store(run_dir, samples)
    row = prepare(run_dir, samples, starts, ends)
    with pytest.raises(StepError, match=message) as error:
        read_view_clip(run_dir, row)
    text = str(error.value)
    assert text.startswith(f"segment {SEGMENT}: ")
    assert error.value.segment == SEGMENT
    assert "flac" not in text
    assert str(tmp_path) not in text
    assert error.value.__cause__ is None
    assert error.value.__context__ is None
