"""The clip build (scripts/build_views.py) end to end, on a synthetic workspace."""

import dataclasses
import importlib.util
import json
import os
import signal
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from listening_beyond_words.clips import MANIFEST_COLUMNS, format_windows
from listening_beyond_words.runinfo import sha256_file
from listening_beyond_words.views import (
    SegmentViews,
    ViewSettings,
    build_segment_views,
    cut_original,
    to_pcm16,
)

REPO = Path(__file__).parents[1]
RATE = 16_000
VIEWS = ("x0", "x2", "rev")
# Pilot IDs in a temporary folder: the session guard refuses every other ID. The first
# window starts at 0.50004 s, which the listening key stores as 0.5000: cutting x0 from
# the key's rounded windows instead of the stored ones would move it by one sample.
SEGMENTS = [
    {
        "segment": "300_000",
        "session": 300,
        "window_starts": [0.50004, 2.0],
        "window_ends": [1.5, 3.25],
    },
    {"segment": "300_001", "session": 300, "window_starts": [4.0], "window_ends": [6.0]},
    {
        "segment": "301_000",
        "session": 301,
        "window_starts": [1.0, 5.0, 7.5],
        "window_ends": [3.0, 6.0, 9.0],
    },
]


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_listening_copy(clip: np.ndarray) -> np.ndarray:
    """The listening copy as scripts/listening_check.py made it at commit 2645d45."""
    rms = float(np.sqrt(np.mean(np.square(clip, dtype=np.float64))))
    peak = float(np.max(np.abs(clip)))
    gain = 1.0 if rms == 0.0 or peak == 0.0 else min(0.1 / rms, 0.99 / peak)
    return clip * np.float32(gain)


def session_audio(root: Path, session: int) -> np.ndarray:
    audio, _ = sf.read(
        root / "edaic" / "sessions" / f"{session}_P" / f"{session}_AUDIO.wav", dtype="float32"
    )
    return audio


def windows_of(row: dict) -> list[tuple[float, float]]:
    return list(zip(row["window_starts"], row["window_ends"], strict=True))


@pytest.fixture
def build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, Path, dict]:
    """A workspace with two recordings, pinned segments, a listening check and the script."""
    root = tmp_path / "workspace"
    rng = np.random.default_rng(9001)
    for session in (300, 301):
        folder = root / "edaic" / "sessions" / f"{session}_P"
        folder.mkdir(parents=True)
        samples = np.clip(np.rint(rng.normal(0, 0.05, 20 * RATE) * 32768), -32768, 32767)
        sf.write(folder / f"{session}_AUDIO.wav", samples.astype(np.int16), RATE, "PCM_16")

    real = yaml.safe_load((REPO / "configs" / "pilot" / "data.yaml").read_text(encoding="utf-8"))
    segments_run = root / "derived" / "segments" / "run-1"
    segments_run.mkdir(parents=True)
    pd.DataFrame(SEGMENTS).to_parquet(segments_run / "segments.parquet", index=False)
    built = {
        "inputs": {"word_times": {"sha256": "w" * 64}, "participant_lines": {"sha256": "l" * 64}},
        "settings": {"daicwoz_cut": real["daicwoz_cut"], "x0_cut": real["x0_cut"]},
    }
    (segments_run / "run.json").write_text(json.dumps(built), encoding="utf-8")
    segments_sha = sha256_file(segments_run / "segments.parquet")

    # The listening check of these segments, written as scripts/listening_check.py writes it.
    check = root / "derived" / "listening_check" / "run-2"
    (check / "clips").mkdir(parents=True)
    key = []
    for number, row in enumerate([SEGMENTS[2], SEGMENTS[0]], start=1):
        clip = cut_original(session_audio(root, row["session"]), windows_of(row), fade=0.01)
        copy = frozen_listening_copy(clip)
        sf.write(check / "clips" / f"{number:02d}.wav", copy, RATE, subtype="PCM_16")
        key.append(
            {
                "clip": f"{number:02d}.wav",
                "segment": row["segment"],
                "windows": format_windows(windows_of(row)),
            }
        )
    pd.DataFrame(key).to_csv(check / "key.tsv", sep="\t", index=False)
    heard = {
        "status": "done",
        "inputs": {"segments": {"file": "derived/segments/run-1", "sha256": segments_sha}},
        "settings": {
            "seed": 102,
            "x0_cut": real["x0_cut"],
            "gap_seconds": 0.3,
            "listening_rms": 0.1,
            "listening_peak": 0.99,
        },
    }
    (check / "run.json").write_text(json.dumps(heard), encoding="utf-8")
    # An earlier check, made from other segments.
    earlier = root / "derived" / "listening_check" / "run-0"
    earlier.mkdir(parents=True)
    (earlier / "key.tsv").write_text("clip\tsegment\twindows\n", encoding="utf-8")
    other = {"status": "done", "inputs": {"segments": {"sha256": "o" * 64}}, "settings": {}}
    (earlier / "run.json").write_text(json.dumps(other), encoding="utf-8")

    config = {
        **real,
        "segments": {"run": "derived/segments/run-1", "sha256": segments_sha},
        "word_times": {"run": "derived/word_times/run-0", "sha256": "w" * 64},
        "participant_lines": {"run": "derived/participant_lines/run-0", "sha256": "l" * 64},
        "listening_checks": [
            {"run": "derived/listening_check/run-0", "sha256": sha256_file(earlier / "key.tsv")},
            {"run": "derived/listening_check/run-2", "sha256": sha256_file(check / "key.tsv")},
        ],
    }
    config_path = tmp_path / "data.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    module = load_script("build_views")
    monkeypatch.setattr(module, "CONFIG", config_path)
    monkeypatch.setattr(module, "stop_on_sigterm", lambda: None)
    monkeypatch.setenv("LBW_WORKSPACE", str(root))
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    return module, root, config


def test_build_writes_every_view_with_its_manifest(
    build: tuple[ModuleType, Path, dict],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, root, config = build
    monkeypatch.setattr(module, "LONG_SECONDS", 3.0)  # so that one clip counts as long
    module.main()
    out = capsys.readouterr().out
    assert (
        "x0 check: the listening copies of 2 of 2 clips (listening check with seed 102)"
        " are identical to the heard files" in out
    )
    settings = ViewSettings.from_config(config)
    for view in VIEWS:
        run = root / "views" / view / "run-4242"
        info = json.loads((run / "run.json").read_text(encoding="utf-8"))
        assert info["status"] == "done"
        assert info["view"] == view
        assert info["inputs"]["x0_check"]["identical"] == 2
        manifest = pd.read_parquet(run / "manifest.parquet")
        assert list(manifest.columns) == list(MANIFEST_COLUMNS)
        assert manifest["segment"].tolist() == ["300_000", "300_001", "301_000"]
        assert info["output"]["sha256"] == sha256_file(run / "manifest.parquet")
        assert f"{view} sha256: {info['output']['sha256']}" in out
        assert f"{view} run: views/{view}/run-4242" in out
        rms, peaks, gains, limited = [], [], [], []
        for row, segment in zip(manifest.itertuples(), SEGMENTS, strict=True):
            samples, rate = sf.read(run / row.file, dtype="int16")
            assert rate == RATE
            assert len(samples) == row.samples
            assert sha256_file(run / row.file) == row.sha256
            views = build_segment_views(
                session_audio(root, segment["session"]), windows_of(segment), settings
            )
            expected = {"x0": views.x0, "x2": views.x2, "rev": views.rev}[view]
            np.testing.assert_array_equal(samples, to_pcm16(expected))
            assert list(row.part_starts) == list(views.part_starts)
            assert list(row.part_ends) == list(views.part_ends)
            values = samples.astype(np.float64) / 32768
            rms.append(np.sqrt(np.mean(values**2)))
            peaks.append(np.max(np.abs(values)))
            if view == "x2":
                gains.append(20 * np.log10(views.x2_gain))
                limited.append(views.x2_peak_limited)
            else:
                gains.append(0.0)
                limited.append(False)
            assert row.gain_db == pytest.approx(gains[-1])
            assert row.peak_limited == limited[-1]
            assert row.rms == pytest.approx(rms[-1])
            assert row.peak == peaks[-1]
        counts = info["counts"]
        seconds = [2.54996, 2.0, 5.1]  # parts plus gaps of each segment
        assert counts["files"] == 3
        assert counts["sessions"] == 2
        assert counts["clips_over_30_s"] == 1
        assert counts["longest_seconds"] == pytest.approx(5.1)
        assert counts["hours"] == pytest.approx(sum(seconds) / 3600, abs=1e-7)
        assert counts["median_rms"] == pytest.approx(np.median(rms))
        assert counts["max_peak"] == pytest.approx(max(peaks))
        assert counts["peak_limited"] == sum(limited)
        assert counts["gain_db"] == pytest.approx(
            {"min": min(gains), "median": float(np.median(gains)), "max": max(gains)}
        )
        assert counts["storage_bytes"] == sum(
            (run / name).stat().st_size for name in manifest["file"]
        )


def test_peak_limited_x2_clips_are_flagged_and_counted(
    build: tuple[ModuleType, Path, dict],
) -> None:
    module, root, config = build
    config["views"] = {**config["views"], "peak_limit": 0.1}  # below every x2 peak here
    module.CONFIG.write_text(yaml.safe_dump(config), encoding="utf-8")
    module.main()
    settings = ViewSettings.from_config(config)
    for view in VIEWS:
        run = root / "views" / view / "run-4242"
        info = json.loads((run / "run.json").read_text(encoding="utf-8"))
        manifest = pd.read_parquet(run / "manifest.parquet")
        assert info["counts"]["peak_limited"] == (3 if view == "x2" else 0)
        assert manifest["peak_limited"].tolist() == [view == "x2"] * 3
    run = root / "views" / "x2" / "run-4242"
    manifest = pd.read_parquet(run / "manifest.parquet")
    for row, segment in zip(manifest.itertuples(), SEGMENTS, strict=True):
        views = build_segment_views(
            session_audio(root, segment["session"]), windows_of(segment), settings
        )
        assert views.x2_peak_limited
        assert row.gain_db == pytest.approx(20 * np.log10(views.x2_gain))
        assert row.peak == pytest.approx(0.1, abs=1 / 32768)


def test_build_stops_before_writing_when_x0_differs_from_a_heard_clip(
    build: tuple[ModuleType, Path, dict], capsys: pytest.CaptureFixture[str]
) -> None:
    module, root, _ = build
    heard = root / "derived" / "listening_check" / "run-2" / "clips" / "02.wav"
    samples, _ = sf.read(heard, dtype="int16")
    samples[RATE] += 1
    sf.write(heard, samples, RATE, subtype="PCM_16")
    with pytest.raises(SystemExit, match="x0 check failed"):
        module.main()
    assert "the listening copies of 1 of 2 clips" in capsys.readouterr().out
    assert not (root / "views").exists()


def test_build_needs_exactly_one_listening_check_of_the_pinned_segments(
    build: tuple[ModuleType, Path, dict],
) -> None:
    module, root, config = build
    path = root / "derived" / "listening_check" / "run-0" / "run.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info["inputs"]["segments"]["sha256"] = config["segments"]["sha256"]
    path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="found 2"):
        module.main()
    assert not (root / "views").exists()


def heard_run(root: Path) -> Path:
    return root / "derived" / "listening_check" / "run-2"


def change_heard_settings(root: Path, **changes: Any) -> None:
    path = heard_run(root) / "run.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info.update(changes.pop("top", {}))
    info["settings"].update(changes)
    path.write_text(json.dumps(info), encoding="utf-8")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"top": {"status": "failed"}}, "did not finish"),
        ({"x0_cut": {"margin_ms": 100, "fade_ms": 20, "wall_guard_ms": 20}}, "other x0 settings"),
        ({"gap_seconds": 0.25}, "other x0 settings"),
        ({"listening_rms": 0.2}, "another listening level"),
        ({"listening_peak": 0.9}, "another listening level"),
    ],
)
def test_build_refuses_a_listening_check_with_other_settings(
    build: tuple[ModuleType, Path, dict], changes: dict, message: str
) -> None:
    module, root, _ = build
    change_heard_settings(root, **changes)
    with pytest.raises(ValueError, match=message):
        module.main()
    assert not (root / "views").exists()


@pytest.mark.parametrize(("rate", "subtype"), [(44_100, "PCM_16"), (RATE, "PCM_24")])
def test_build_stops_when_a_heard_file_has_another_format(
    build: tuple[ModuleType, Path, dict], rate: int, subtype: str
) -> None:
    module, root, _ = build
    heard = heard_run(root) / "clips" / "02.wav"
    samples, _ = sf.read(heard, dtype="int16")
    sf.write(heard, samples, rate, subtype=subtype)
    with pytest.raises(SystemExit, match="1 identical, of 2"):
        module.main()
    assert not (root / "views").exists()


def test_build_stops_when_a_heard_clip_has_other_windows(
    build: tuple[ModuleType, Path, dict],
) -> None:
    module, root, config = build
    key = heard_run(root) / "key.tsv"
    text = key.read_text(encoding="utf-8")
    assert text.count("0.5000-1.5000") == 1
    key.write_text(text.replace("0.5000-1.5000", "0.5001-1.5000"), encoding="utf-8")
    config["listening_checks"][1]["sha256"] = sha256_file(key)  # pinned again
    module.CONFIG.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(SystemExit, match="1 with the same windows"):
        module.main()
    assert not (root / "views").exists()


def test_a_sound_in_a_gap_of_a_built_view_stops_the_build(
    build: tuple[ModuleType, Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, root, _ = build
    build_views = module.build_segment_views

    def with_a_sound_in_rev(*args: Any, **kwargs: Any) -> SegmentViews:
        views = build_views(*args, **kwargs)
        if len(views.part_starts) < 2:
            return views
        rev = views.rev.copy()
        rev[views.part_ends[0]] = 0.01
        return dataclasses.replace(views, rev=rev)

    monkeypatch.setattr(module, "build_segment_views", with_a_sound_in_rev)
    with pytest.raises(ValueError, match="rev: a gap is not silent"):
        module.main()
    for view in VIEWS:
        info = json.loads((root / "views" / view / "run-4242" / "run.json").read_text("utf-8"))
        assert info["status"] == "failed"


def test_a_job_that_slurm_stops_leaves_failed_runs(
    build: tuple[ModuleType, Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, root, _ = build
    calls = []
    write_view = module.write_view

    def stopped(*args: object, **kwargs: object) -> object:
        calls.append(1)
        if len(calls) == 4:
            raise SystemExit("stopped by signal 15")
        return write_view(*args, **kwargs)

    monkeypatch.setattr(module, "write_view", stopped)
    with pytest.raises(SystemExit, match="signal 15"):
        module.main()
    for view in VIEWS:
        info = json.loads((root / "views" / view / "run-4242" / "run.json").read_text("utf-8"))
        assert info["status"] == "failed"
        assert "finished_utc" in info


def test_a_rerun_never_writes_into_an_earlier_run(build: tuple[ModuleType, Path, dict]) -> None:
    module, root, _ = build
    (root / "views" / "x2" / "run-4242").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        module.main()


def test_build_refuses_to_run_outside_a_job(
    build: tuple[ModuleType, Path, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _, _ = build
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(SystemExit, match="submit this as a job"):
        module.main()


def test_sigterm_ends_the_job_through_the_error_path() -> None:
    module = load_script("build_views")
    previous = signal.getsignal(signal.SIGTERM)
    try:
        module.stop_on_sigterm()
        with pytest.raises(SystemExit, match="stopped by signal 15"):
            os.kill(os.getpid(), signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, previous)
