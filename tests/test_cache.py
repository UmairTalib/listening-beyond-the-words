"""The stored encoder outputs (cache.py): files, their checks, the check's sample and the pins."""

import hashlib
import io
import json
import math
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from listening_beyond_words import cache, voxtral
from listening_beyond_words.cache import (
    CACHE_COLUMNS,
    NUMERICS,
    CacheSettings,
    check_sample,
    features_sha256,
    from_bytes,
    load_cache_manifest,
    pinned_cache,
    pinned_clips,
    read_cache_run,
    read_file,
    relative_run,
    to_bytes,
    write_file,
)
from listening_beyond_words.clips import MANIFEST_COLUMNS
from listening_beyond_words.models import ModelPin
from listening_beyond_words.runinfo import StepError, item_seed, sha256_file
from listening_beyond_words.settings import Workspace
from voxtral_fakes import (
    ANSWER_IDS,
    WIDTH,
    FakeProcessor,
    cached_encoder,
    same_bytes,
    tiny_model,
    tone,
)

ROOT = Path(__file__).resolve().parents[1]
FRAMES = voxtral.FRAMES_PER_CHUNK
VOXTRAL = ModelPin("voxtral", "mistralai/Voxtral-Mini-3B-2507", "3" * 40)
SETTINGS = {"cpu_threads": 4, "verify": {"segments_per_view": 2, "seed": 108, "lm_watch_clips": 2}}
NUMERIC_RECORD = {
    **NUMERICS,
    "torch_threads": 4,
    "omp_num_threads": "4",
    "cublas_workspace_config": ":4096:8",
}
SEGMENTS = ("300_000", "300_001", "301_000", "301_001", "302_000")
LONG = "301_001"  # a clip of 37.5 s: two chunks, not among the 2 drawn
CACHE_RUN = "cache/x0/run-9"
CHECK_RUN = "derived/cache_check/run-10"


def encoder_output(n_chunks: int = 1, seed: int = 0, width: int = WIDTH) -> Any:
    """A random stand-in for an encoder output, shape (C, 1500, W), fp16."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(n_chunks, FRAMES, width, generator=generator) * 10).half()


def npy(array: np.ndarray, allow_pickle: bool = False) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=allow_pickle)
    return buffer.getvalue()


# Settings --------------------------------------------------------------------------------------


def test_cache_settings_read_the_pinned_block_of_the_data_config() -> None:
    config = yaml.safe_load((ROOT / "configs" / "pilot" / "data.yaml").read_text(encoding="utf-8"))
    assert CacheSettings.from_config(config) == CacheSettings(
        cpu_threads=4, segments_per_view=25, seed=108, lm_watch_clips=25
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda block: block.pop("cpu_threads"), "encoder_cache: missing keys ['cpu_threads']"),
        (
            lambda block: block.update(batch=1),
            "encoder_cache: missing keys [], unknown keys ['batch']",
        ),
        (lambda block: block["verify"].pop("seed"), "encoder_cache.verify: missing keys ['seed']"),
        (lambda block: block.update(cpu_threads=0), "encoder_cache.cpu_threads must be a whole"),
        (lambda block: block.update(cpu_threads=True), "encoder_cache.cpu_threads must be a whole"),
        (lambda block: block.update(cpu_threads=4.0), "encoder_cache.cpu_threads must be a whole"),
        (
            lambda block: block["verify"].update(segments_per_view="25"),
            "encoder_cache.verify.segments_per_view must be a whole",
        ),
        (
            lambda block: block.update(verify=[25, 108, 25]),
            "encoder_cache.verify must be a mapping",
        ),
    ],
)
def test_cache_settings_refuse_a_missing_unknown_or_wrong_value(
    change: Callable[[dict[str, Any]], Any], message: str
) -> None:
    block = json.loads(json.dumps(SETTINGS))
    change(block)
    with pytest.raises(StepError, match=f"^{re.escape(message)}"):
        CacheSettings.from_config({"encoder_cache": block})


def test_cache_settings_refuse_a_config_without_the_block() -> None:
    with pytest.raises(StepError, match="encoder_cache must be a mapping"):
        CacheSettings.from_config({})


# Files -----------------------------------------------------------------------------------------


BYTES_CODE = """
import hashlib, sys
import torch
from listening_beyond_words.cache import to_bytes
values = (torch.arange(2 * 1500 * 64) % 2003 - 1001).to(torch.float32) / 8
sys.stdout.write(hashlib.sha256(to_bytes(values.reshape(2, 1500, 64).half())).hexdigest())
"""


def test_one_output_gives_the_same_bytes_in_another_process() -> None:
    import torch

    values = (torch.arange(2 * 1500 * 64) % 2003 - 1001).to(torch.float32) / 8
    mine = hashlib.sha256(to_bytes(values.reshape(2, 1500, 64).half())).hexdigest()
    result = subprocess.run(
        [sys.executable, "-c", BYTES_CODE], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout == mine


@pytest.mark.parametrize("n_chunks", [1, 2])
def test_the_round_trip_gives_back_every_bit(n_chunks: int) -> None:
    import torch

    original = encoder_output(n_chunks)
    special = [0.0, -0.0, 65_504.0, -65_504.0, 64_512.0, 6.0e-8, -6.0e-8, math.inf, -math.inf]
    original[0, 0, : len(special)] = torch.tensor(special, dtype=torch.float16)
    original[-1, -1, -1] = math.nan
    data = to_bytes(original)
    back = from_bytes(data, WIDTH)
    assert same_bytes(back, original)  # -0, the infinities and the NaN too
    assert back.dtype == torch.float16
    assert back.device.type == "cpu"
    assert data.startswith(b"\x93NUMPY")
    assert data == npy(original.numpy())  # NumPy's own format, nothing added


def test_to_bytes_writes_the_values_whatever_the_layout_in_memory() -> None:
    wide = encoder_output(1, width=2 * WIDTH)
    spread = wide[:, :, ::2]  # not contiguous
    assert not spread.is_contiguous()
    assert to_bytes(spread) == to_bytes(spread.contiguous())


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: encoder_output().float(), "stored in fp16, not torch.float32"),
        (lambda: encoder_output().bfloat16(), "stored in fp16, not torch.bfloat16"),
        (lambda: encoder_output()[0], "has shape"),
        (lambda: encoder_output()[:, :-1], "has shape"),
        (lambda: encoder_output()[:0], "has shape"),
    ],
)
def test_to_bytes_refuses_another_format_or_shape(make: Callable[[], Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        to_bytes(make())


@pytest.mark.parametrize(
    ("data", "width", "message"),
    [
        (lambda: npy(np.zeros((1, FRAMES, WIDTH), np.float32)), WIDTH, "not fp16"),
        (lambda: to_bytes(encoder_output()), 2 * WIDTH, "not \\(C, 1500, 128\\)"),
        (lambda: npy(np.zeros((1, FRAMES - 1, WIDTH), np.float16)), WIDTH, "has shape"),
        (lambda: npy(np.zeros((FRAMES, WIDTH), np.float16)), WIDTH, "has shape"),
        (lambda: npy(np.zeros((0, FRAMES, WIDTH), np.float16)), WIDTH, "has shape"),
        (
            lambda: npy(np.asfortranarray(np.zeros((2, FRAMES, WIDTH), np.float16))),
            WIDTH,
            "not in C order",
        ),
        (lambda: to_bytes(encoder_output()) + b"\0", WIDTH, "1 bytes follow the array"),
        (lambda: npy(np.array([object()], dtype=object), True), WIDTH, "allow_pickle"),
        (lambda: b"not a file of NumPy's", WIDTH, "magic string"),
        (lambda: b"", WIDTH, "EOF"),
        (lambda: to_bytes(encoder_output())[:-2], WIDTH, "EOF"),
    ],
)
def test_from_bytes_refuses_anything_but_one_fp16_output_of_the_width(
    data: Callable[[], bytes], width: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        from_bytes(data(), width)


def test_features_sha256_fingerprints_the_float32_bytes() -> None:
    import torch

    features = torch.linspace(-1.5, 2.5, 2 * 128 * 3000).reshape(2, 128, 3000)
    expected = hashlib.sha256(features.numpy().tobytes()).hexdigest()
    assert features_sha256(features) == expected
    assert features_sha256(features.transpose(1, 2).contiguous().transpose(1, 2)) == expected
    changed = features.clone()
    changed[1, 127, 2999] = torch.nextafter(changed[1, 127, 2999], torch.tensor(9.0))
    assert features_sha256(changed) != expected
    for wrong in (features.half(), features.double()):
        with pytest.raises(ValueError, match="log-mel features are float32"):
            features_sha256(wrong)


def test_write_file_writes_the_bytes_under_the_name_and_nothing_else(tmp_path: Path) -> None:
    path = tmp_path / "300_000.npy"
    write_file(path, b"some bytes")
    assert path.read_bytes() == b"some bytes"
    assert [file.name for file in tmp_path.iterdir()] == ["300_000.npy"]


def test_write_file_refuses_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "300_000.npy"
    path.write_bytes(b"first")
    with pytest.raises(FileExistsError, match="never overwritten"):
        write_file(path, b"second")
    assert path.read_bytes() == b"first"
    assert [file.name for file in tmp_path.iterdir()] == ["300_000.npy"]


def test_write_file_leaves_no_temporary_file_after_a_planted_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(_descriptor: int) -> None:
        raise OSError("the disk is full")

    monkeypatch.setattr(cache.os, "fsync", failing)
    with pytest.raises(OSError, match="the disk is full"):
        write_file(tmp_path / "300_000.npy", b"some bytes")
    assert list(tmp_path.iterdir()) == []


def test_write_file_never_removes_a_temporary_file_that_it_did_not_make(tmp_path: Path) -> None:
    stale = tmp_path / "300_000.npy.tmp"
    stale.write_bytes(b"left by someone else")
    with pytest.raises(FileExistsError):
        write_file(tmp_path / "300_000.npy", b"some bytes")
    assert stale.read_bytes() == b"left by someone else"
    assert not (tmp_path / "300_000.npy").exists()


def stored(folder: Path, data: bytes, chunks: int, segment: str = "300_000") -> tuple[Path, Any]:
    """A cache file and the manifest row that names it."""
    path = folder / f"{segment}.npy"
    path.write_bytes(data)
    row = SimpleNamespace(segment=segment, sha256=hashlib.sha256(data).hexdigest(), chunks=chunks)
    return path, row


@pytest.mark.parametrize("n_chunks", [1, 2])
def test_read_file_gives_back_the_stored_output(tmp_path: Path, n_chunks: int) -> None:
    original = encoder_output(n_chunks, seed=n_chunks)
    path, row = stored(tmp_path, to_bytes(original), n_chunks)
    assert same_bytes(read_file(path, row, WIDTH), original)


def nan_output() -> bytes:
    output = encoder_output()
    output[0, 9, 9] = math.nan
    return to_bytes(output)


def inf_output() -> bytes:
    output = encoder_output()
    output[0, 9, 9] = -math.inf
    return to_bytes(output)


@pytest.mark.parametrize(
    ("data", "chunks", "width", "change", "message"),
    [
        (lambda: to_bytes(encoder_output()), 1, WIDTH, "delete", "its cache file is missing"),
        (
            lambda: to_bytes(encoder_output()),
            1,
            WIDTH,
            "sha256",
            "its cache file does not match the SHA-256 of its manifest row",
        ),
        (
            lambda: npy(np.zeros((1, FRAMES, WIDTH), np.float32)),
            1,
            WIDTH,
            None,
            "its cache file is not an encoder output of width 64 (the array is <f4, not fp16",
        ),
        (
            lambda: to_bytes(encoder_output()),
            1,
            2 * WIDTH,
            None,
            "its cache file is not an encoder output of width 128 (the array has shape",
        ),
        (
            lambda: to_bytes(encoder_output()),
            2,
            WIDTH,
            None,
            "its cache file holds 1 chunks, its manifest row 2",
        ),
        (nan_output, 1, WIDTH, None, "its cache file holds a value that is not finite"),
        (inf_output, 1, WIDTH, None, "its cache file holds a value that is not finite"),
    ],
)
def test_read_file_refuses_a_file_that_does_not_fit_its_row_naming_the_segment(
    tmp_path: Path,
    data: Callable[[], bytes],
    chunks: int,
    width: int,
    change: str | None,
    message: str,
) -> None:
    path, row = stored(tmp_path, data(), chunks)
    if change == "delete":
        path.unlink()
    elif change == "sha256":
        row.sha256 = "0" * 64
    with pytest.raises(StepError) as raised:
        read_file(path, row, width)
    assert str(raised.value).startswith(f"segment 300_000: {message}")
    assert raised.value.segment == "300_000"
    assert raised.value.__context__ is None  # raised outside the reader's except clause


@pytest.fixture(scope="module")
def tiny() -> Any:
    return tiny_model(half=True)


@pytest.mark.parametrize("seconds", [1.0, 14.0, 37.4])
def test_z_from_a_cache_file_equals_z_from_the_audio_bit_for_bit(
    tmp_path: Path, tiny: Any, seconds: float
) -> None:
    processor = FakeProcessor()
    audio = tone(seconds, seed=5)
    clip = voxtral.clip_inputs(processor, audio)
    from_audio = voxtral.yes_no_score(tiny, clip, ANSWER_IDS)
    data = to_bytes(voxtral.encoder_output(tiny, clip))
    path = tmp_path / "300_000.npy"
    write_file(path, data)
    count = voxtral.chunks(audio.size)
    row = SimpleNamespace(segment="300_000", sha256=hashlib.sha256(data).hexdigest(), chunks=count)
    from_file = read_file(path, row, WIDTH)
    with cached_encoder(tiny):
        z = voxtral.cached_score(
            tiny, voxtral.prompt_inputs(processor, count), from_file, ANSWER_IDS
        )
    assert math.isfinite(from_audio)
    assert z.hex() == from_audio.hex()


# The check's sample ----------------------------------------------------------------------------


def sample_manifest(count: int = 30, long: tuple[str, ...] = ("301_007",)) -> pd.DataFrame:
    segments = [f"{300 + index // 10}_{index % 10:03d}" for index in range(count)]
    return pd.DataFrame({"segment": segments, "chunks": [2 if s in long else 1 for s in segments]})


def test_check_sample_draws_from_the_views_own_stream_and_adds_every_long_clip() -> None:
    manifest = sample_manifest()
    sample = check_sample(manifest, "x0", 5, 108)
    segments = sorted(manifest["segment"])
    rng = np.random.default_rng(item_seed(108, "cache", "x0"))
    drawn = {segments[int(index)] for index in rng.choice(30, size=5, replace=False)}
    assert sample == tuple(sorted(drawn | {"301_007"}))
    assert len(sample) in (5, 6)
    assert check_sample(manifest.iloc[::-1], "x0", 5, 108) == sample  # the row order is not read
    assert check_sample(manifest, "x1", 5, 108) != sample  # a stream of its own per view
    assert check_sample(manifest, "x0", 5, 109) != sample
    assert check_sample(manifest, "x0", 50, 108) == tuple(segments)  # all, if fewer than asked


def test_check_sample_keeps_every_clip_of_two_chunks_however_small_the_draw() -> None:
    long = ("300_003", "302_009")
    sample = check_sample(sample_manifest(long=long), "rev", 1, 108)
    assert set(long) <= set(sample)
    assert len(sample) in (2, 3)


# Run folders -----------------------------------------------------------------------------------


def test_relative_run_spells_a_run_folder_one_way(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path.resolve())
    for spelling in ("cache/x0/run-9", "cache/x0/run-9/", "cache/./x0/../x0/run-9"):
        assert relative_run(workspace, spelling) == "cache/x0/run-9"
    for outside in ("/cache/x0/run-9", "../cache/x0/run-9", ".", ""):
        with pytest.raises(StepError, match="is not a folder inside the workspace"):
            relative_run(workspace, outside)


# Pinned runs: a synthetic cache run, its check and the pins --------------------------------------


@dataclass
class Pinned:
    """A pinned view, a cache run of it, its check and the pins, written as the job writes them."""

    workspace: Workspace
    config: dict[str, Any]

    @property
    def cache(self) -> Path:
        return self.workspace.root / CACHE_RUN

    @property
    def check(self) -> Path:
        return self.workspace.root / CHECK_RUN

    def manifest(self) -> pd.DataFrame:
        return pd.read_parquet(self.cache / "manifest.parquet")

    def edit(self, folder: Path, change: Callable[[dict[str, Any]], Any]) -> None:
        info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        change(info)
        (folder / "run.json").write_text(json.dumps(info), encoding="utf-8")

    def repin(self, manifest: pd.DataFrame) -> None:
        """Write another manifest and pin it in full: run.json, the check and the config."""
        manifest.to_parquet(self.cache / "manifest.parquet", index=False)
        sha256 = sha256_file(self.cache / "manifest.parquet")
        size = len(check_sample(manifest, "x0", 2, 108))

        def checked(info: dict[str, Any]) -> None:
            info["results"]["runs"][CACHE_RUN].update(
                manifest_sha256=sha256, segments=size, features_equal=size, outputs_equal=size
            )

        self.edit(self.cache, lambda info: info["outputs"].update({"manifest.parquet": sha256}))
        self.edit(self.check, checked)
        self.config["cache_runs"]["x0"]["sha256"] = sha256


def make_pinned(root: Path) -> Pinned:
    root.mkdir(parents=True)
    workspace = Workspace(root.resolve())
    clips = []
    for index, segment in enumerate(SEGMENTS):
        samples = 600_000 if segment == LONG else 16_000 + 100 * index
        clips.append(
            {
                "segment": segment,
                "session": int(segment[:3]),
                "file": f"{segment[:3]}/{segment}.flac",
                "samples": samples,
                "rms": 0.1,
                "peak": 0.5,
                "sha256": f"{index:064x}",
                "audio_sha256": f"{index + 100:064x}",
                "part_starts": [0],
                "part_ends": [samples],
                "peak_limited": False,
                "gain_db": 0.0,
            }
        )
    view_run = root / "views" / "x0" / "run-1"
    view_run.mkdir(parents=True)
    pd.DataFrame(clips, columns=list(MANIFEST_COLUMNS)).to_parquet(
        view_run / "manifest.parquet", index=False
    )
    view_pin = {"run": "views/x0/run-1", "sha256": sha256_file(view_run / "manifest.parquet")}
    rows = [
        {
            "segment": clip["segment"],
            "session": clip["session"],
            "file": f"{clip['session']}/{clip['segment']}.npy",
            "chunks": voxtral.chunks(clip["samples"]),
            "samples": clip["samples"],
            "sha256": f"{index + 200:064x}",
            "audio_sha256": clip["audio_sha256"],
            "features_sha256": f"{index + 300:064x}",
            "max_abs": 12.5,
        }
        for index, clip in enumerate(clips)
    ]
    cache_dir = root / CACHE_RUN
    cache_dir.mkdir(parents=True)
    manifest = pd.DataFrame(rows, columns=list(CACHE_COLUMNS))
    manifest.to_parquet(cache_dir / "manifest.parquet", index=False)
    sha256 = sha256_file(cache_dir / "manifest.parquet")
    built = {
        "step": "build_caches",
        "mode": {"name": "view", "view": "x0"},
        "status": "done",
        "inputs": {
            "view": view_pin,
            "voxtral": {"repo_id": VOXTRAL.repo_id, "revision": VOXTRAL.revision, "blobs": {}},
        },
        "settings": {"encoder_cache": SETTINGS, "numerics": NUMERIC_RECORD, "width": WIDTH},
        "outputs": {"manifest.parquet": sha256},
    }
    (cache_dir / "run.json").write_text(json.dumps(built), encoding="utf-8")
    size = len(check_sample(manifest, "x0", 2, 108))
    check_dir = root / CHECK_RUN
    check_dir.mkdir(parents=True)
    check = {
        "step": "build_caches",
        "mode": {"name": "verify", "runs": [CACHE_RUN]},
        "status": "done",
        "settings": {"encoder_cache": SETTINGS, "numerics": NUMERIC_RECORD, "width": WIDTH},
        "results": {
            "runs": {
                CACHE_RUN: {
                    "view": "x0",
                    "manifest_sha256": sha256,
                    "segments": size,
                    "features_equal": size,
                    "outputs_equal": size,
                    "lm_watch": {"clips": 2, "parts": 8, "flagged": [], "largest": []},
                }
            }
        },
    }
    (check_dir / "run.json").write_text(json.dumps(check), encoding="utf-8")
    config = {
        "view_runs": {"x0": view_pin},
        "encoder_cache": json.loads(json.dumps(SETTINGS)),
        "cache_runs": {"x0": {"run": CACHE_RUN, "sha256": sha256, "verified_by": CHECK_RUN}},
    }
    return Pinned(workspace, config)


@pytest.fixture
def pinned(tmp_path: Path) -> Pinned:
    return make_pinned(tmp_path / "workspace")


def test_read_cache_run_reads_a_run_that_fits_the_pins(pinned: Pinned) -> None:
    run = read_cache_run(pinned.workspace, pinned.config, CACHE_RUN + "/", VOXTRAL)
    assert run.run == CACHE_RUN
    assert run.folder == pinned.cache
    assert run.view == "x0"
    assert run.manifest_sha256 == pinned.config["cache_runs"]["x0"]["sha256"]
    assert run.manifest.equals(pinned.manifest())
    assert run.clips.equals(pinned_clips(pinned.workspace, pinned.config, "x0"))
    assert run.info["settings"]["width"] == WIDTH


def test_load_cache_manifest_gives_the_pinned_manifest_by_segment(pinned: Pinned) -> None:
    shuffled = pinned.manifest().iloc[[3, 0, 4, 2, 1]].reset_index(drop=True)
    pinned.repin(shuffled)
    loaded = load_cache_manifest(pinned.workspace, pinned.config, "x0", VOXTRAL)
    assert list(loaded.columns) == list(CACHE_COLUMNS)
    assert list(loaded["segment"]) == sorted(SEGMENTS)
    assert loaded.equals(shuffled.sort_values("segment", ignore_index=True))
    assert loaded.loc[loaded["segment"] == LONG, "chunks"].item() == 2


def test_pinned_cache_gives_the_run_its_manifest_by_segment_and_its_check(
    pinned: Pinned,
) -> None:
    shuffled = pinned.manifest().iloc[[3, 0, 4, 2, 1]].reset_index(drop=True)
    pinned.repin(shuffled)
    cache = pinned_cache(pinned.workspace, pinned.config, "x0", VOXTRAL)
    assert (cache.view, cache.check, cache.lm_flagged) == ("x0", CHECK_RUN, ())
    assert cache.pin == pinned.config["cache_runs"]["x0"]
    assert cache.pin is not pinned.config["cache_runs"]["x0"]  # a copy: the job records it
    assert (cache.run.run, cache.run.folder) == (CACHE_RUN, pinned.cache)
    assert cache.run.manifest.equals(shuffled)  # as stored
    assert cache.manifest.equals(shuffled.sort_values("segment", ignore_index=True))


def test_pinned_cache_passes_on_the_language_model_parts_that_its_check_flagged(
    pinned: Pinned,
) -> None:
    # A flag in the language model leaves the cache standing: the readers decide.
    flagged = ["model.language_model.layers.1/output", "model.language_model.norm/output"]
    pinned.edit(
        pinned.check,
        lambda info: info["results"]["runs"][CACHE_RUN]["lm_watch"].update(flagged=flagged),
    )
    cache = pinned_cache(pinned.workspace, pinned.config, "x0", VOXTRAL)
    assert cache.lm_flagged == tuple(flagged)
    assert load_cache_manifest(pinned.workspace, pinned.config, "x0", VOXTRAL).equals(
        cache.manifest
    )


def change_the_manifest(pinned: Pinned) -> None:
    pinned.manifest().iloc[::-1].to_parquet(pinned.cache / "manifest.parquet", index=False)


def drop_a_segment(pinned: Pinned) -> None:
    pinned.repin(pinned.manifest().iloc[1:])


def add_a_segment(pinned: Pinned) -> None:
    manifest = pinned.manifest()
    extra = manifest.iloc[[0]].assign(segment="303_000", session=303, file="303/303_000.npy")
    pinned.repin(pd.concat([manifest, extra], ignore_index=True))


def repeat_a_segment(pinned: Pinned) -> None:
    manifest = pinned.manifest()
    pinned.repin(pd.concat([manifest, manifest.iloc[[1]]], ignore_index=True))


def change_an_audio_sha256(pinned: Pinned) -> None:
    pinned.repin(
        pinned.manifest().assign(audio_sha256=lambda table: table["audio_sha256"].str[::-1])
    )


def give_a_short_clip_two_chunks(pinned: Pinned) -> None:
    manifest = pinned.manifest()
    manifest.loc[manifest["segment"] == "300_000", "chunks"] = 2
    pinned.repin(manifest)


def move_a_row_to_another_session(pinned: Pinned) -> None:
    manifest = pinned.manifest()
    manifest.loc[manifest["segment"] == "300_000", "session"] = 301
    pinned.repin(manifest)


def lengthen_a_row_within_its_chunk(pinned: Pinned) -> None:
    manifest = pinned.manifest()
    manifest.loc[manifest["segment"] == "300_000", "samples"] += 1  # still one chunk
    pinned.repin(manifest)


def other_numerics(info: dict[str, Any]) -> None:
    info["settings"]["numerics"]["fp16_reduced_precision_reduction"] = True


def drop_a_numerics_flag(info: dict[str, Any]) -> None:
    del info["settings"]["numerics"]["matmul_tf32"]


def cache_another_view(pinned: Pinned) -> None:
    pinned.config["view_runs"]["x2"] = dict(pinned.config["view_runs"]["x0"])
    pinned.edit(pinned.cache, lambda info: info["mode"].update(view="x2"))


def change_the_view_manifest(pinned: Pinned) -> None:
    path = pinned.workspace.root / "views" / "x0" / "run-1" / "manifest.parquet"
    pd.read_parquet(path).iloc[::-1].to_parquet(path, index=False)


CACHE_CHANGES: dict[str, tuple[Callable[[Pinned], None], str]] = {
    "a changed manifest": (
        change_the_manifest,
        "the pinned cache run of x0: its manifest.parquet does not match",
    ),
    "a run not done": (
        lambda p: p.edit(p.cache, lambda info: info.update(status="failed")),
        f"{CACHE_RUN} did not finish: its run.json does not say done",
    ),
    "another step": (
        lambda p: p.edit(p.cache, lambda info: info.update(step="check_views")),
        f"{CACHE_RUN} is not a run of build_caches.py --view",
    ),
    "another mode": (
        lambda p: p.edit(p.cache, lambda info: info.update(mode={"name": "verify", "runs": []})),
        f"{CACHE_RUN} is not a run of build_caches.py --view",
    ),
    "another view run": (
        lambda p: p.edit(p.cache, lambda info: info["inputs"]["view"].update(run="views/x0/run-0")),
        f"{CACHE_RUN} was not built from the pinned clips of x0 (view_runs)",
    ),
    "another view manifest": (
        lambda p: p.edit(p.cache, lambda info: info["inputs"]["view"].update(sha256="0" * 64)),
        f"{CACHE_RUN} was not built from the pinned clips of x0 (view_runs)",
    ),
    "another revision": (
        lambda p: p.edit(p.cache, lambda info: info["inputs"]["voxtral"].update(revision="4" * 40)),
        f"{CACHE_RUN} was built with another Voxtral than models.yaml pins",
    ),
    "another setting": (
        lambda p: p.config["encoder_cache"]["verify"].update(lm_watch_clips=24),
        f"{CACHE_RUN} was built with other encoder_cache settings",
    ),
    "other numerics": (
        lambda p: p.edit(p.cache, other_numerics),
        f"{CACHE_RUN} was built with other numeric settings than voxtral.set_numerics sets with"
        " 4 threads",
    ),
    "other threads": (
        lambda p: p.edit(
            p.cache, lambda info: info["settings"]["numerics"].update(torch_threads=8)
        ),
        f"{CACHE_RUN} was built with other numeric settings",
    ),
    "a numerics record without a flag": (
        lambda p: p.edit(p.cache, drop_a_numerics_flag),
        f"{CACHE_RUN} was built with other numeric settings",
    ),
    "a manifest that run.json does not list": (
        lambda p: p.edit(
            p.cache, lambda info: info["outputs"].update({"manifest.parquet": "0" * 64})
        ),
        f"{CACHE_RUN}: its manifest.parquet does not match its run.json",
    ),
    "a missing segment": (
        drop_a_segment,
        f"{CACHE_RUN}: its manifest lacks 1 segments of the clips of x0 and holds 0 others",
    ),
    "an extra segment": (
        add_a_segment,
        f"{CACHE_RUN}: its manifest lacks 0 segments of the clips of x0 and holds 1 others",
    ),
    "a segment twice": (repeat_a_segment, f"{CACHE_RUN}: its manifest holds 1 segments twice"),
    "another audio_sha256": (
        change_an_audio_sha256,
        f"{CACHE_RUN}: 5 rows differ from the clips of x0 in session, length or audio_sha256",
    ),
    "another session": (
        move_a_row_to_another_session,
        f"{CACHE_RUN}: 1 rows differ from the clips of x0 in session, length or audio_sha256",
    ),
    "another length": (
        lengthen_a_row_within_its_chunk,
        f"{CACHE_RUN}: 1 rows differ from the clips of x0 in session, length or audio_sha256",
    ),
    "another number of chunks": (
        give_a_short_clip_two_chunks,
        f"{CACHE_RUN}: 1 rows give another number of chunks than their length",
    ),
    "a run of another view": (cache_another_view, "the cache run pinned for x0 caches x2"),
    "a view manifest that changed": (
        change_the_view_manifest,
        "the manifest of the pinned clips of x0 does not match its pin",
    ),
}


@pytest.mark.parametrize("change", list(CACHE_CHANGES))
def test_load_cache_manifest_refuses_a_cache_run_that_does_not_fit(
    pinned: Pinned, change: str
) -> None:
    plant, message = CACHE_CHANGES[change]
    plant(pinned)
    with pytest.raises(StepError, match=f"^{re.escape(message)}"):
        load_cache_manifest(pinned.workspace, pinned.config, "x0", VOXTRAL)


CHECK_CHANGES: dict[str, tuple[Callable[[Pinned], None], str]] = {
    "no pin": (
        lambda p: p.config.pop("cache_runs"),
        "no cache run of x0 is pinned (cache_runs)",
    ),
    "a pin without its check": (
        lambda p: p.config["cache_runs"]["x0"].pop("verified_by"),
        "cache_runs.x0 needs exactly run, sha256 and verified_by",
    ),
    "a missing check run": (
        lambda p: p.config["cache_runs"]["x0"].update(verified_by="derived/cache_check/run-11"),
        "derived/cache_check/run-11, the check of the pinned cache run of x0, has no run.json",
    ),
    "a failed check run": (
        lambda p: p.edit(p.check, lambda info: info.update(status="failed")),
        f"{CHECK_RUN} did not finish: its run.json does not say done",
    ),
    "a check run of another step": (
        lambda p: p.edit(p.check, lambda info: info.update(step="check_views")),
        f"{CHECK_RUN} is not a run of build_caches.py --verify",
    ),
    "a cache run named as its own check": (
        lambda p: p.config["cache_runs"]["x0"].update(verified_by=CACHE_RUN),
        f"{CACHE_RUN} is not a run of build_caches.py --verify",
    ),
    "a check of other runs": (
        lambda p: p.edit(p.check, lambda info: info["mode"].update(runs=["cache/x0/run-8"])),
        f"{CHECK_RUN} did not check {CACHE_RUN}",
    ),
    "a check of another manifest": (
        lambda p: p.edit(
            p.check,
            lambda info: info["results"]["runs"][CACHE_RUN].update(manifest_sha256="0" * 64),
        ),
        f"{CHECK_RUN} checked another manifest of {CACHE_RUN}",
    ),
    "a check with other settings": (
        lambda p: p.edit(
            p.check, lambda info: info["settings"]["encoder_cache"]["verify"].update(seed=109)
        ),
        f"{CHECK_RUN} ran with other encoder_cache settings",
    ),
    "a check with other numerics": (
        lambda p: p.edit(
            p.check, lambda info: info["settings"]["numerics"].update(omp_num_threads="8")
        ),
        f"{CHECK_RUN} ran with other numeric settings than {CACHE_RUN}",
    ),
    "a check that found an output not equal": (
        lambda p: p.edit(
            p.check, lambda info: info["results"]["runs"][CACHE_RUN].update(outputs_equal=2)
        ),
        f"{CHECK_RUN} found the features of 3 and the encoder outputs of 2 of 3 segments equal,"
        " not those of all 3 of its sample",
    ),
    "a check without its language-model watch": (
        lambda p: p.edit(p.check, lambda info: info["results"]["runs"][CACHE_RUN].pop("lm_watch")),
        f"{CHECK_RUN} holds no record of its language-model watch on {CACHE_RUN}",
    ),
    "a check whose watch lists no flags": (
        lambda p: p.edit(
            p.check, lambda info: info["results"]["runs"][CACHE_RUN]["lm_watch"].pop("flagged")
        ),
        f"{CHECK_RUN} holds no record of its language-model watch on {CACHE_RUN}",
    ),
    "a check of a smaller sample": (
        lambda p: p.edit(
            p.check,
            lambda info: info["results"]["runs"][CACHE_RUN].update(
                segments=2, features_equal=2, outputs_equal=2
            ),
        ),
        f"{CHECK_RUN} found the features of 2 and the encoder outputs of 2 of 2 segments equal,"
        " not those of all 3 of its sample",
    ),
}


@pytest.mark.parametrize("change", list(CHECK_CHANGES))
def test_load_cache_manifest_refuses_a_missing_failed_or_other_check(
    pinned: Pinned, change: str
) -> None:
    plant, message = CHECK_CHANGES[change]
    plant(pinned)
    with pytest.raises(StepError, match=f"^{re.escape(message)}"):
        load_cache_manifest(pinned.workspace, pinned.config, "x0", VOXTRAL)


def test_the_sample_of_the_synthetic_check_holds_the_long_clip() -> None:
    # The refusals above count on it: 2 drawn plus the long clip, 3 segments in all.
    manifest = pd.DataFrame(
        {"segment": list(SEGMENTS), "chunks": [2 if s == LONG else 1 for s in SEGMENTS]}
    )
    sample = check_sample(manifest, "x0", 2, 108)
    assert LONG in sample
    assert len(sample) == 3


@pytest.mark.parametrize(
    "other",
    [
        ModelPin("voxtral", VOXTRAL.repo_id, "5" * 40),
        ModelPin("voxtral", "someone/Voxtral-Mini-3B-2507", VOXTRAL.revision),
    ],
    ids=["another revision", "another repo"],
)
def test_load_cache_manifest_refuses_another_voxtral_pin(pinned: Pinned, other: ModelPin) -> None:
    with pytest.raises(
        StepError, match=re.escape("was built with another Voxtral than models.yaml pins")
    ):
        load_cache_manifest(pinned.workspace, pinned.config, "x0", other)
