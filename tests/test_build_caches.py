"""The encoder cache job (scripts/build_caches.py), end to end, on the CPU.

The word check's synthetic workspace (wordcheck_fakes), cut to four participants and with
one clip over 30 s added to x0, x2 and rev, and a tiny random Voxtral saved as the pinned
snapshot (voxtral_fakes), loaded on the CPU by the job's own loader with a stand-in for the
processor. The language model's calls are logged; in --view, it raises if it ever runs.
"""

import contextlib
import io
import json
import math
import re
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from listening_beyond_words import voxtral
from listening_beyond_words.cache import (
    CACHE_COLUMNS,
    NUMERICS,
    CacheSettings,
    check_sample,
    features_sha256,
    from_bytes,
    load_cache_manifest,
    read_file,
)
from listening_beyond_words.clips import MANIFEST_COLUMNS, read_view_clip, write_view
from listening_beyond_words.models import load_pins, local_snapshot
from listening_beyond_words.runinfo import STOPPED, item_seed, sha256_file
from listening_beyond_words.settings import Workspace
from voxtral_fakes import (
    WIDTH,
    FakeProcessor,
    patch_processor,
    same_bytes,
    save_checkpoint,
    tiny_model,
)
from wordcheck_fakes import (
    GAP,
    WordcheckWorkspace,
    assert_only_aggregates_leave,
    clip_samples,
    info_of,
    make_wordcheck_workspace,
    write_pinned,
)
from x1_fakes import REPO, load_script

KEPT_SESSIONS = (300, 301, 302, 303)  # 19 segments: enough to draw from, quick to encode
LONG_SEGMENT = "301_002"  # its x0, x2 and rev clips last over 37 s: two chunks
LONG_SAMPLES = 598_400
LONG_VIEWS = ("x0", "x2", "rev")
FAST_CHECK = {"segments_per_view": 5, "seed": 108, "lm_watch_clips": 3}  # the config: 25, 108, 25
X0_JOB, REV_JOB, CHECK_JOB = "800", "802", "801"
ENCODER = tuple(f"model.audio_tower.layers.{index}" for index in range(2))
LANGUAGE_MODEL = tuple(f"model.language_model.layers.{index}" for index in range(2))
NORM = "model.audio_tower.layer_norm"
# Every step that the watches read, in order: a convolution, a norm and the adapter have
# their output only.
ENCODER_STEPS = ("model.audio_tower.conv1", "model.audio_tower.conv2", *ENCODER, NORM)
LANGUAGE_MODEL_STEPS = ("model.multi_modal_projector", *LANGUAGE_MODEL, "model.language_model.norm")
OUTPUT_ONLY = (*ENCODER_STEPS[:2], NORM, LANGUAGE_MODEL_STEPS[0], LANGUAGE_MODEL_STEPS[-1])
VOXTRAL = load_pins(REPO / "configs" / "pilot" / "models.yaml")["voxtral"]
GPU = {
    "gpu": "Synthetic GPU",
    "gpu_capability": "7.0",
    "gpu_memory_gb": 16.0,
    "cuda": "12.6",
    "cudnn": 91002,
    "driver": "550.00",
}
NUMERIC_RECORD = {
    **NUMERICS,
    "torch_threads": 4,
    "omp_num_threads": "4",
    "cublas_workspace_config": ":4096:8",
}


# The workspace ---------------------------------------------------------------------------------


def trim(workspace: WordcheckWorkspace, sessions: tuple[int, ...]) -> None:
    """Keep only the clips of some participants in every view, pinned again."""
    for pin in workspace.config["view_runs"].values():
        run = workspace.root / pin["run"]
        manifest = pd.read_parquet(run / "manifest.parquet")
        kept = manifest[manifest["session"].isin(sessions)].reset_index(drop=True)
        pin["sha256"] = write_pinned(run, kept, "manifest.parquet")
    workspace.write_config()


def lengthen(
    workspace: WordcheckWorkspace, segment: str, views: tuple[str, ...], samples: int
) -> None:
    """Replace one segment's clips in some views by clips over 30 s with the same parts."""
    for number, view in enumerate(views):
        pin = workspace.config["view_runs"][view]
        run = workspace.root / pin["run"]
        rows = pd.read_parquet(run / "manifest.parquet").to_dict("records")
        (row,) = [row for row in rows if row["segment"] == segment]
        parts = len(row["part_starts"])
        rng = np.random.default_rng([8, number])
        clip, starts, ends = clip_samples(
            rng, [0] * parts, (samples - GAP * (parts - 1)) // parts, 0
        )
        path = run / row["file"]
        path.unlink()
        row.update(write_view(path, clip), part_starts=starts, part_ends=ends)
        pin["sha256"] = write_pinned(
            run, pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS)), "manifest.parquet"
        )
    workspace.write_config()


def add_snapshot(workspace: WordcheckWorkspace) -> Path:
    """The tiny Voxtral in bf16 as the pinned snapshot, laid out as the Hugging Face cache does."""
    repo = workspace.root / "hf" / "hub" / ("models--" + VOXTRAL.repo_id.replace("/", "--"))
    saved = save_checkpoint(tiny_model(), repo / "saved")
    snapshot = repo / "snapshots" / VOXTRAL.revision
    snapshot.mkdir(parents=True)
    (repo / "blobs").mkdir()
    for file in sorted(saved.iterdir()):
        blob = sha256_file(file)
        file.rename(repo / "blobs" / blob)
        (snapshot / file.name).symlink_to(Path("..") / ".." / "blobs" / blob)
    saved.rmdir()
    return snapshot


def make_cache_workspace(root: Path, sessions: tuple[int, ...] | None) -> WordcheckWorkspace:
    workspace = make_wordcheck_workspace(root)
    if sessions is not None:
        trim(workspace, sessions)
    lengthen(workspace, LONG_SEGMENT, LONG_VIEWS, LONG_SAMPLES)
    add_snapshot(workspace)
    return workspace


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> WordcheckWorkspace:
    workspace = make_cache_workspace(
        tmp_path_factory.mktemp("template") / "workspace", KEPT_SESSIONS
    )
    workspace.config["encoder_cache"]["verify"] = dict(FAST_CHECK)
    workspace.write_config()
    return workspace


@pytest.fixture
def workspace(
    template: WordcheckWorkspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> WordcheckWorkspace:
    copy = template.copy_to(tmp_path)
    monkeypatch.setenv("LBW_WORKSPACE", str(copy.root))
    return copy


@pytest.fixture(scope="module")
def small_template(
    template: WordcheckWorkspace, tmp_path_factory: pytest.TempPathFactory
) -> WordcheckWorkspace:
    """One participant's 5 clips, the long one among them: for the planted faults."""
    small = template.copy_to(tmp_path_factory.mktemp("small"))
    trim(small, (int(LONG_SEGMENT[:3]),))
    return small


@pytest.fixture
def small(
    small_template: WordcheckWorkspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> WordcheckWorkspace:
    copy = small_template.copy_to(tmp_path)
    monkeypatch.setenv("LBW_WORKSPACE", str(copy.root))
    return copy


def cache_dir(workspace: WordcheckWorkspace, view: str = "x0", job: str = X0_JOB) -> Path:
    return workspace.root / "cache" / view / f"run-{job}"


def check_dir(workspace: WordcheckWorkspace, job: str = CHECK_JOB) -> Path:
    return workspace.root / "derived" / "cache_check" / f"run-{job}"


def manifest_of(workspace: WordcheckWorkspace, view: str) -> pd.DataFrame:
    return pd.read_parquet(
        workspace.root / workspace.config["view_runs"][view]["run"] / "manifest.parquet"
    )


# Running the job -------------------------------------------------------------------------------


@contextlib.contextmanager
def kept_numerics() -> Iterator[None]:
    """Put PyTorch's numeric settings back after a job, since set_numerics sets them."""
    import torch

    matmul = torch.backends.cuda.matmul
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        matmul.allow_fp16_reduced_precision_reduction,
        matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_num_threads(),
        matmul.allow_fp16_accumulation,
    )
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
        torch.backends.cudnn.deterministic = saved[2]
        torch.backends.cudnn.benchmark = saved[3]
        matmul.allow_fp16_reduced_precision_reduction = saved[4]
        matmul.allow_tf32 = saved[5]
        torch.backends.cudnn.allow_tf32 = saved[6]
        torch.set_num_threads(saved[7])
        matmul.allow_fp16_accumulation = saved[8]


@pytest.fixture(autouse=True)
def numerics_put_back() -> Iterator[None]:
    with kept_numerics():
        yield


@dataclass
class Model:
    """The job's model loader: the tiny snapshot on the CPU, with its calls logged.

    Attributes:
        language_model_raises: The language model and the output layer raise if they run.
        plant: Changes the loaded model, to plant a fault.
        calls: "encoder", "language model" and "output layer", one per call, in order;
            "load" once per load.
    """

    language_model_raises: bool = False
    plant: Callable[[Any], None] | None = None
    calls: list[str] = field(default_factory=list)

    def load(self, snapshot: Path, device: str = "cuda") -> voxtral.LoadedModel:
        assert device == "cuda"  # the job asks for the GPU; the tests run on the CPU
        self.calls.append("load")
        loaded = voxtral.load_model(snapshot, device="cpu")
        model = loaded.model
        model.model.audio_tower.register_forward_hook(lambda *_: self.calls.append("encoder"))
        model.model.language_model.register_forward_pre_hook(self.language_model)
        model.lm_head.register_forward_pre_hook(self.output_layer)
        if self.plant is not None:
            self.plant(model)
        return loaded

    def language_model(self, *_: Any) -> None:
        self.calls.append("language model")
        if self.language_model_raises:
            raise AssertionError("the language model ran")

    def output_layer(self, *_: Any) -> None:
        self.calls.append("output layer")
        raise AssertionError("the output layer ran")


def patch_job(
    module: ModuleType, workspace: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, model: Model
) -> None:
    """Point the job at the workspace, a clean checkout, a GPU, the tiny model and its processor."""
    monkeypatch.setattr(sys, "unraisablehook", sys.unraisablehook)  # main() replaces it
    monkeypatch.setattr(module, "CONFIG", workspace.config_path)
    monkeypatch.setattr(module, "MODELS", REPO / "configs" / "pilot" / "models.yaml")
    monkeypatch.setattr(module, "git_state", lambda: {"commit": "c" * 40, "dirty": False})
    monkeypatch.setattr(module, "stop_on_sigterm", lambda: None)
    monkeypatch.setattr(module, "gpu_facts", lambda: dict(GPU))
    monkeypatch.setattr(module, "load_model", model.load)
    patch_processor(monkeypatch)
    monkeypatch.setenv("LBW_WORKSPACE", str(workspace.root))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # As scripts/slurm/common.sh sets them in a job of four CPUs.
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def run_job(
    workspace: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    *args: str,
    job: str,
    model: Model | None = None,
    module: ModuleType | None = None,
) -> None:
    """Run the job on the synthetic workspace."""
    script = module or load_script("build_caches")
    patch_job(script, workspace, monkeypatch, model or Model())
    monkeypatch.setenv("SLURM_JOB_ID", job)
    monkeypatch.setattr(sys, "argv", ["build_caches.py", *args])
    script.main()


def never_scored(calls: list[str]) -> Callable[[ModuleType, pytest.MonkeyPatch], None]:
    """Make both scoring functions log their call and raise, wherever the job would find them."""

    def prepare(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("yes_no_score", "cached_score"):

            def refuse(*_args: Any, _name: str = name, **_kwargs: Any) -> Any:
                calls.append(_name)
                raise AssertionError(f"{_name} ran")

            monkeypatch.setattr(voxtral, name, refuse)
            monkeypatch.setattr(module, name, refuse, raising=False)

    return prepare


@dataclass
class Finished:
    """A run of the job that ended done, and what it printed.

    Attributes:
        reads: The clips that the job read, as (view, segment), when they were logged.
    """

    workspace: WordcheckWorkspace
    run: Path
    out: str
    err: str
    model: Model
    reads: list[tuple[str, str]] = field(default_factory=list)

    @property
    def info(self) -> dict:
        return info_of(self.run)


def logged_reads(seen: list[tuple[str, str]]) -> Callable[[ModuleType, pytest.MonkeyPatch], None]:
    """Log each clip that the job reads, as (view, segment)."""

    def prepare(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        real = module.read_view_clip

        def logged(folder: Path, row: Any) -> Any:
            seen.append((Path(folder).parts[-2], str(row.segment)))  # views/<view>/run-<job>
            return real(folder, row)

        monkeypatch.setattr(module, "read_view_clip", logged)

    return prepare


def finish(
    workspace: WordcheckWorkspace,
    run: Path,
    *args: str,
    job: str,
    model: Model | None = None,
    prepare: Callable[[ModuleType, pytest.MonkeyPatch], None] | None = None,
) -> Finished:
    """Run the job to its end in a patch context of its own, for fixtures shared by a module."""
    out, err = io.StringIO(), io.StringIO()
    used = model or Model()
    with pytest.MonkeyPatch.context() as monkeypatch, kept_numerics():
        module = load_script("build_caches")
        if prepare is not None:
            prepare(module, monkeypatch)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            run_job(workspace, monkeypatch, *args, job=job, model=used, module=module)
    return Finished(workspace, run, out.getvalue(), err.getvalue(), used)


@pytest.fixture(scope="module")
def built(
    template: WordcheckWorkspace, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Finished]:
    """--view x0 and --view rev, with a language model that raises if it runs and a progress
    line every 5 clips."""
    workspace = template.copy_to(tmp_path_factory.mktemp("built"))
    runs = {}
    for view, job in (("x0", X0_JOB), ("rev", REV_JOB)):
        model = Model(language_model_raises=True)
        scored = never_scored(model.calls)

        def prepare(
            module: ModuleType, monkeypatch: pytest.MonkeyPatch, _scored: Any = scored
        ) -> None:
            _scored(module, monkeypatch)
            monkeypatch.setattr(module, "PROGRESS_EVERY", 5)

        runs[view] = finish(
            workspace,
            cache_dir(workspace, view, job),
            "--view",
            view,
            job=job,
            model=model,
            prepare=prepare,
        )
    return runs


@pytest.fixture(scope="module")
def checked(built: dict[str, Finished], tmp_path_factory: pytest.TempPathFactory) -> Finished:
    """--verify of both cache runs, with every call of the encoder and language model logged,
    and every clip that it reads."""
    workspace = built["x0"].workspace.copy_to(tmp_path_factory.mktemp("checked"))
    runs = [f"cache/x0/run-{X0_JOB}", f"cache/rev/run-{REV_JOB}/"]  # either spelling
    seen: list[tuple[str, str]] = []
    finished = finish(
        workspace,
        check_dir(workspace),
        "--verify",
        *runs,
        job=CHECK_JOB,
        prepare=logged_reads(seen),
    )
    finished.reads = seen
    return finished


def kept_segments(workspace: WordcheckWorkspace) -> list[str]:
    return sorted(str(segment) for segment in manifest_of(workspace, "x0")["segment"])


def sample_of(workspace: WordcheckWorkspace, view: str, run: Path) -> tuple[str, ...]:
    manifest = pd.read_parquet(run / "manifest.parquet")
    return check_sample(manifest, view, FAST_CHECK["segments_per_view"], FAST_CHECK["seed"])


# --view ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("view", ["x0", "rev"])
def test_the_view_mode_stores_one_file_per_clip_and_ends_done(
    built: dict[str, Finished], view: str
) -> None:
    finished = built[view]
    workspace, run, info = finished.workspace, finished.run, finished.info
    assert info["status"] == "done"
    assert info["step"] == "build_caches"
    assert info["mode"] == {"name": "view", "view": view}
    segments = kept_segments(workspace)
    files = sorted(path.relative_to(run).as_posix() for path in run.rglob("*.npy"))
    assert files == sorted(f"{segment[:3]}/{segment}.npy" for segment in segments)
    assert sorted(path.name for path in run.iterdir() if path.is_file()) == [
        "manifest.parquet",
        "run.json",
    ]
    assert not list(run.rglob("*.tmp"))
    manifest = pd.read_parquet(run / "manifest.parquet")
    assert tuple(manifest.columns) == CACHE_COLUMNS
    assert sorted(manifest["segment"]) == segments
    assert info["outputs"] == {"manifest.parquet": sha256_file(run / "manifest.parquet")}
    clips = manifest_of(workspace, view).set_index("segment")
    for row in manifest.itertuples(index=False):
        data = (run / row.file).read_bytes()
        assert sha256_file(run / row.file) == row.sha256
        output = from_bytes(data, WIDTH)
        assert tuple(output.shape) == (row.chunks, voxtral.FRAMES_PER_CHUNK, WIDTH)
        assert row.chunks == (2 if row.segment == LONG_SEGMENT else 1)
        assert row.chunks == voxtral.chunks(row.samples)
        assert row.samples == clips.at[row.segment, "samples"]
        assert row.audio_sha256 == clips.at[row.segment, "audio_sha256"]
        assert row.session == clips.at[row.segment, "session"]
        assert row.max_abs == float(output.abs().max())
    out = finished.out
    assert f"run: cache/{view}/run-{X0_JOB if view == 'x0' else REV_JOB}\n" in out
    assert f"manifest.parquet sha256: {info['outputs']['manifest.parquet']}\n" in out
    assert re.search(r"^done in \d+\.\d min$", out, flags=re.MULTILINE)
    assert "FAILED" not in out


def test_the_cache_holds_the_encoder_output_of_each_clip_and_its_features_fingerprint(
    built: dict[str, Finished], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, run = built["x0"].workspace, built["x0"].run
    patch_processor(monkeypatch)
    loaded = voxtral.load_model(local_snapshot(VOXTRAL, workspace.root / "hf"), device="cpu")
    voxtral.set_numerics(4)  # as the job; numerics_put_back restores them
    processor = FakeProcessor()
    folder = workspace.root / workspace.config["view_runs"]["x0"]["run"]
    clips = {str(row.segment): row for row in manifest_of(workspace, "x0").itertuples(index=False)}
    manifest = pd.read_parquet(run / "manifest.parquet")
    for row in manifest.itertuples(index=False):
        clip = read_view_clip(folder, clips[row.segment])
        inputs = voxtral.clip_inputs(processor, clip.audio)
        assert features_sha256(inputs["input_features"]) == row.features_sha256
        stored = read_file(run / row.file, row, WIDTH)
        assert same_bytes(stored, voxtral.encoder_output(loaded.model, inputs)), row.segment
    # Each clip has its own output: nothing was written to another clip's row.
    assert manifest["sha256"].nunique() == len(manifest)


def test_the_view_mode_never_runs_the_language_model_or_scores(built: dict[str, Finished]) -> None:
    for view, finished in built.items():
        clips = len(kept_segments(finished.workspace))
        assert finished.model.calls == ["load", *["encoder"] * clips], view


def test_run_json_records_the_inputs_settings_and_aggregates_of_the_view_mode(
    built: dict[str, Finished],
) -> None:
    finished = built["x0"]
    workspace, run, info = finished.workspace, finished.run, finished.info
    assert info["settings"] == {
        "encoder_cache": workspace.config["encoder_cache"],
        "numerics": NUMERIC_RECORD,
        "width": WIDTH,
    }
    assert info["inputs"]["view"] == workspace.config["view_runs"]["x0"]
    model = info["inputs"]["voxtral"]
    assert (model["repo_id"], model["revision"]) == (VOXTRAL.repo_id, VOXTRAL.revision)
    assert {"config.json", "model.safetensors"} <= set(model["blobs"])
    assert info["config"] == {
        "file": str(workspace.config_path),
        "sha256": sha256_file(workspace.config_path),
    }
    models = REPO / "configs" / "pilot" / "models.yaml"
    assert info["models_config"]["sha256"] == sha256_file(models)
    assert info["slurm"]["job_id"] == X0_JOB
    assert info["slurm"]["cpus_per_task"] == "4"
    assert info["slurm"]["cpu"] == info["environment"]["cpu"]
    assert info["seeds"] == {}
    environment = info["environment"]
    assert {key: environment[key] for key in GPU} == GPU
    for package in ("torch", "transformers", "mistral-common", "numpy"):
        assert package in environment["packages"], package
    assert environment["packages"]["torch"]
    results = info["results"]
    clips = len(kept_segments(workspace))
    sizes = sum(path.stat().st_size for path in run.rglob("*.npy"))
    assert results["clips"] == clips
    assert results["chunks"] == clips + 1
    assert results["two_chunk_clips"] == 1
    assert results["bytes"] == sizes
    assert results["non_finite_outputs"] == 0
    manifest = pd.read_parquet(run / "manifest.parquet")
    assert results["max_abs"] == manifest["max_abs"].max()
    assert results["audio_hours"] == round(int(manifest["samples"].sum()) / 16_000 / 3600, 4)
    assert results["largest_weight"] > 0
    assert set(results["seconds"]) == {"features", "encoder", "serialise", "write", "read_back"}
    watch = results["encoder_watch"]
    assert list(watch["layers"]) == list(ENCODER_STEPS)
    assert watch["parts"] == 9  # the convolutions and the norm, and 2 layers with 3 parts each
    assert watch["flagged"] == []
    for name, parts in watch["layers"].items():
        assert tuple(parts) == (("output",) if name in OUTPUT_ONLY else voxtral.PARTS)
        for numbers in parts.values():
            assert numbers["calls"] == clips
            assert numbers["at_clamp"] == numbers["non_finite"] == 0
            assert 0 < numbers["max_abs"] < voxtral.CLAMP_ABS
    maxima = sorted(
        (numbers["max_abs"] for parts in watch["layers"].values() for numbers in parts.values()),
        reverse=True,
    )
    assert [value for _, value in watch["largest"]] == maxima[:5]


def test_the_view_mode_prints_its_aggregates_one_per_line(built: dict[str, Finished]) -> None:
    out = built["x0"].out
    clips = len(kept_segments(built["x0"].workspace))
    assert out.startswith(f"encoder caches of x0 (views/x0/run-1); job {X0_JOB} on ")
    assert "\nnumerics: deterministic_algorithms True, " in out
    assert re.search(r"^model: mistralai/Voxtral-Mini-3B-2507 at \w{8}, loaded in ", out, re.M)
    progress = re.findall(
        r"^clips: (\d+) of (\d+) encoded, [\d.]+ per second; projected end", out, re.M
    )
    assert progress == [(str(number), str(clips)) for number in (5, 10, 15)]
    assert f"\nclips: {clips} encoded (1 of two chunks), {clips + 1} chunks, " in out
    assert "\noutputs: largest |value| " in out
    assert "; 0 with a value that is not finite\n" in out
    assert "\nencoder watch: 0 of 9 parts flagged (a value at the clamp, |x| >= 64,512," in out
    assert f"\nread back: {clips} files match their manifest rows" in out


def test_nothing_per_clip_leaves_the_view_mode(built: dict[str, Finished]) -> None:
    for finished in built.values():
        assert_only_aggregates_leave(finished.run / "run.json", finished.out + finished.err)


# --verify --------------------------------------------------------------------------------------


def test_the_check_finds_every_sampled_feature_and_output_equal(checked: Finished) -> None:
    info, workspace = checked.info, checked.workspace
    runs = [f"cache/x0/run-{X0_JOB}", f"cache/rev/run-{REV_JOB}"]
    assert info["status"] == "done"
    assert info["mode"] == {"name": "verify", "runs": runs}
    assert sorted(path.name for path in checked.run.iterdir()) == ["run.json"]
    for view, name in zip(("x0", "rev"), runs, strict=True):
        cache = workspace.root / name
        sample = sample_of(workspace, view, cache)
        assert LONG_SEGMENT in sample
        result = info["results"]["runs"][name]
        assert {key: result[key] for key in ("view", "segments", "drawn", "two_chunk")} == {
            "view": view,
            "segments": len(sample),
            "drawn": 5,
            "two_chunk": 1,
        }
        assert result["manifest_sha256"] == sha256_file(cache / "manifest.parquet")
        assert result["features_equal"] == result["outputs_equal"] == len(sample)
        assert result["differ"] == 0
        assert result["largest_difference_fp16_steps"] is None
        assert result["same_cpu"]
        assert result["same_gpu"]
        n = len(sample)
        assert (
            f"{name} ({view}): features equal {n} of {n}; encoder outputs equal {n} of {n} (5 drawn"
            " with seed 108, 1 of two chunks); built with the same CPU model and GPU as this job\n"
        ) in checked.out
        assert info["inputs"]["runs"][name] == {
            "view": view,
            "manifest_sha256": result["manifest_sha256"],
            "run_json_sha256": sha256_file(cache / "run.json"),
            "view_run": workspace.config["view_runs"][view],
        }
    assert info["seeds"]["check_sample"]["seed"] == 108
    assert info["seeds"]["check_sample"]["item_seeds"] == {
        view: item_seed(108, "cache", view) for view in ("rev", "x0")
    }
    assert info["seeds"]["check_sample"]["streams"] == {
        view: f'numpy.random.default_rng(runinfo.item_seed(108, "cache", "{view}"))'
        for view in ("rev", "x0")
    }
    assert info["settings"]["numerics"] == NUMERIC_RECORD
    assert info["settings"]["encoder_cache"] == workspace.config["encoder_cache"]
    assert f"run: derived/cache_check/run-{CHECK_JOB}\n" in checked.out


def test_the_check_computes_exactly_its_sample_of_each_run_again(checked: Finished) -> None:
    for view, name in zip(("x0", "rev"), checked.info["mode"]["runs"], strict=True):
        manifest = pd.read_parquet(checked.workspace.root / name / "manifest.parquet")
        segments = sorted(manifest["segment"])
        rng = np.random.default_rng(item_seed(108, "cache", view))
        drawn = {segments[int(index)] for index in rng.choice(len(segments), 5, replace=False)}
        long = set(manifest.loc[manifest["chunks"] > 1, "segment"])
        assert sorted(s for v, s in checked.reads if v == view) == sorted(drawn | long), view


def test_the_check_watches_the_language_model_on_the_longest_clips_without_its_output_layer(
    checked: Finished,
) -> None:
    for name in checked.info["mode"]["runs"]:
        watch = checked.info["results"]["runs"][name]["lm_watch"]
        assert watch["clips"] == 3
        assert watch["two_chunk_clips"] == 1  # the longest clip is the two-chunk one
        assert watch["parts"] == 8  # the adapter, 2 layers with 3 parts each, the final norm
        assert watch["flagged"] == []
        assert list(watch["layers"]) == list(LANGUAGE_MODEL_STEPS)  # no output layer: no logits
        for parts in watch["layers"].values():
            assert {numbers["calls"] for numbers in parts.values()} == {3}
        manifest = pd.read_parquet(checked.workspace.root / name / "manifest.parquet")
        third = sorted(manifest["samples"], reverse=True)[2]
        assert watch["shortest_seconds"] == round(third / 16_000, 3)
        assert (
            f"{name} ({manifest_view(name)}): the language model without its output layer on the 3"
            " longest clips (1 of two chunks, the shortest"
        ) in checked.out


def manifest_view(name: str) -> str:
    return name.split("/")[1]


def test_the_checks_encoder_part_never_calls_the_language_model(checked: Finished) -> None:
    encoded = sum(
        checked.info["results"]["runs"][name]["segments"] for name in checked.info["mode"]["runs"]
    )
    # Every encoder call comes first; the language model then runs 3 times per cache run,
    # and the output layer never.
    assert checked.model.calls == ["load", *["encoder"] * encoded, *["language model"] * 6]


def test_nothing_per_clip_leaves_the_check(checked: Finished) -> None:
    assert_only_aggregates_leave(checked.run / "run.json", checked.out + checked.err)


@pytest.mark.parametrize(
    "results",
    [
        {"first_bad": "300_001"},
        {"note": "300_001 z=1.234"},
        {"segments": ["300_000", "300_001"]},
        {"pair": [300, 5]},
    ],
    ids=["a value", "a string", "a short list", "a participant"],
)
def test_the_leak_check_finds_an_id_anywhere_in_run_json_but_its_error(
    tmp_path: Path, results: dict[str, Any]
) -> None:
    path = tmp_path / "run.json"
    error = {"type": "StepError", "segment": "300_001", "message": "segment 300_001: a fault"}
    path.write_text(json.dumps({"status": "failed", "error": error, "results": {}}), "utf-8")
    assert_only_aggregates_leave(path, "FAILED: StepError in segment 300_001: a fault\n")
    path.write_text(json.dumps({"status": "failed", "error": error, "results": results}), "utf-8")
    with pytest.raises(AssertionError):
        assert_only_aggregates_leave(path, "")


def test_a_checked_run_passes_load_cache_manifest(checked: Finished) -> None:
    workspace = checked.workspace
    config = json.loads(json.dumps(workspace.config))
    config["cache_runs"] = {}
    for view, job in (("x0", X0_JOB), ("rev", REV_JOB)):
        run = f"cache/{view}/run-{job}"
        config["cache_runs"][view] = {
            "run": run,
            "sha256": sha256_file(workspace.root / run / "manifest.parquet"),
            "verified_by": f"derived/cache_check/run-{CHECK_JOB}",
        }
        loaded = load_cache_manifest(Workspace(workspace.root.resolve()), config, view, VOXTRAL)
        stored = pd.read_parquet(workspace.root / run / "manifest.parquet")
        assert loaded.equals(stored.sort_values("segment", ignore_index=True))


# Faults that end a run failed --------------------------------------------------------------------


def plant_in_output(module: Any, call: int, index: tuple[int, ...], value: float) -> Any:
    """Write one value into a module's output on one of its calls."""
    calls = [0]

    def hook(_module: Any, _args: Any, output: Any) -> Any:
        calls[0] += 1
        if calls[0] != call:
            return None
        planted = output.clone()
        planted[index] = value
        return planted

    return module.register_forward_hook(hook)


def failed_view_run(
    workspace: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    model: Model,
    module: ModuleType | None = None,
) -> tuple[dict, str]:
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, "--view", "x0", job=X0_JOB, model=model, module=module)
    out, err = capsys.readouterr()
    run = cache_dir(workspace)
    assert_only_aggregates_leave(run / "run.json", out + err)
    assert not (run / "manifest.parquet").exists()
    return info_of(run), out


@pytest.mark.parametrize(
    ("site", "flagged"),
    [
        ("after the encoder", []),
        ("in the final norm", [f"{NORM}/output"]),
        ("in the last layer", [f"{ENCODER[1]}/output", f"{NORM}/output"]),
    ],
)
def test_a_planted_nan_output_ends_the_run_failed_with_its_counts_in_run_json(
    small: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    site: str,
    flagged: list[str],
) -> None:
    # After the encoder, where no watched part sees it, the output alone shows the NaN; in
    # the encoder, the watch flags it where it starts, and in each step after it.
    module = load_script("build_caches")
    if site == "after the encoder":
        real = module.encoder_output
        calls = [0]

        def planted(model: Any, inputs: Any) -> Any:
            output = real(model, inputs)
            calls[0] += 1
            if calls[0] == 3:
                output = output.clone()
                output[0, 10, 5] = math.nan
            return output

        monkeypatch.setattr(module, "encoder_output", planted)

    def plant(tiny: Any) -> None:
        encoder = tiny.model.audio_tower
        if site != "after the encoder":
            target = encoder.layer_norm if site == "in the final norm" else encoder.layers[1]
            plant_in_output(target, 3, (0, 10, 5), math.nan)

    model = Model(language_model_raises=True, plant=plant)
    info, out = failed_view_run(small, monkeypatch, capsys, model, module)
    assert info["status"] == "failed"
    message = (
        f"the encoder watch flagged {len(flagged)} parts and 1 outputs hold a value that is not"
        " finite"
    )
    assert info["error"]["message"].startswith(message)
    assert f"FAILED: StepError: {message}" in out
    results = info["results"]
    assert results["non_finite_outputs"] == 1
    assert results["clips"] == len(kept_segments(small))  # every clip, then the verdict
    watch = results["encoder_watch"]
    assert watch["flagged"] == flagged
    assert watch["layers"][ENCODER[1]]["output"]["non_finite"] == int(site == "in the last layer")
    assert ("\nencoder watch, flagged: " + ", ".join(flagged) + "\n" in out) == bool(flagged)
    assert "; 1 with a value that is not finite\n" in out


def test_a_value_at_the_clamp_ends_the_run_failed_though_every_output_is_finite(
    small: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    model = Model(
        language_model_raises=True,
        plant=lambda tiny: plant_in_output(
            tiny.model.audio_tower.layers[1].fc2, 2, (0, 7, 3), math.inf
        ),
    )
    info, _ = failed_view_run(small, monkeypatch, capsys, model)
    assert info["status"] == "failed"
    results = info["results"]
    assert results["non_finite_outputs"] == 0  # the layer clamps the inf to 64,512
    watch = results["encoder_watch"]
    assert watch["flagged"] == [f"{ENCODER[1]}/output", f"{ENCODER[1]}/feed_forward"]
    assert watch["layers"][ENCODER[1]]["output"]["at_clamp"] == 1
    assert info["error"]["message"].startswith("the encoder watch flagged 2 parts and 0 outputs")


def nudge_the_long_clip(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Store one value of the two-chunk clip's output one fp16 step too high."""
    real = module.encoder_output

    def nudged(model: Any, inputs: Any) -> Any:
        output = real(model, inputs)
        if output.shape[0] == 2:
            output = output.clone()
            value = np.float16(output[1, 7, 3].item())
            output[1, 7, 3] = float(np.nextafter(value, np.float16(np.inf)))
        return output

    monkeypatch.setattr(module, "encoder_output", nudged)


def build_then_check(
    workspace: WordcheckWorkspace,
    capsys: pytest.CaptureFixture,
    plant_build: Callable[[ModuleType, pytest.MonkeyPatch], None] | None,
    plant_check: Callable[[ModuleType, pytest.MonkeyPatch], None] | None,
) -> tuple[dict, str, Model]:
    """A --view run of x0 (done), then a --verify run of it that fails."""
    finish(workspace, cache_dir(workspace), "--view", "x0", job=X0_JOB, prepare=plant_build)
    model = Model()
    with pytest.MonkeyPatch.context() as monkeypatch:
        module = load_script("build_caches")
        if plant_check is not None:
            plant_check(module, monkeypatch)
        capsys.readouterr()
        with pytest.raises(SystemExit):
            run_job(
                workspace,
                monkeypatch,
                "--verify",
                f"cache/x0/run-{X0_JOB}",
                job=CHECK_JOB,
                model=model,
                module=module,
            )
    out, err = capsys.readouterr()
    assert_only_aggregates_leave(check_dir(workspace) / "run.json", out + err)
    return info_of(check_dir(workspace)), out, model


def test_the_check_fails_on_a_cache_with_one_wrong_output(
    small: WordcheckWorkspace, capsys: pytest.CaptureFixture
) -> None:
    info, out, model = build_then_check(small, capsys, nudge_the_long_clip, None)
    name = f"cache/x0/run-{X0_JOB}"
    n = len(sample_of(small, "x0", small.root / name))
    result = info["results"]["runs"][name]
    assert (result["features_equal"], result["outputs_equal"], result["differ"]) == (n, n - 1, 1)
    assert result["largest_difference_fp16_steps"] == 1
    assert info["status"] == "failed"
    assert info["error"]["message"].startswith(
        f"1 of {n} sampled segments differ from their caches"
    )
    assert (
        f"features equal {n} of {n}; encoder outputs equal {n - 1} of {n}; largest difference 1 fp16 steps"
        in out
    )
    assert "lm_watch" not in result
    assert "language model" not in model.calls  # the run stops before the language model


def shift_the_long_clips_features(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Change one log-mel value of the two-chunk clip, as another CPU type might."""
    real = module.clip_inputs

    def shifted(processor: Any, audio: Any, *args: Any) -> Any:
        inputs = real(processor, audio, *args)
        if inputs["input_features"].shape[0] == 2:
            inputs["input_features"][0, 0, 0] += 0.5
        return inputs

    monkeypatch.setattr(module, "clip_inputs", shifted)


def test_the_check_tells_a_difference_in_the_features_from_one_in_the_encoder(
    small: WordcheckWorkspace, capsys: pytest.CaptureFixture
) -> None:
    info, out, _ = build_then_check(small, capsys, None, shift_the_long_clips_features)
    name = f"cache/x0/run-{X0_JOB}"
    n = len(sample_of(small, "x0", small.root / name))
    result = info["results"]["runs"][name]
    assert (result["features_equal"], result["outputs_equal"], result["differ"]) == (
        n - 1,
        n - 1,
        1,
    )
    assert result["largest_difference_fp16_steps"] > 0
    assert f"features equal {n - 1} of {n}; encoder outputs equal {n - 1} of {n}" in out


def zero_at(sign: float) -> Callable[[ModuleType, pytest.MonkeyPatch], None]:
    """Give one value of the two-chunk clip's output as +0 or -0, as two kernels might."""

    def prepare(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        real = module.encoder_output

        def zeroed(model: Any, inputs: Any) -> Any:
            output = real(model, inputs)
            if output.shape[0] == 2:
                output = output.clone()
                output[1, 7, 3] = math.copysign(0.0, sign)
            return output

        monkeypatch.setattr(module, "encoder_output", zeroed)

    return prepare


def test_the_check_compares_bytes_so_a_zero_of_the_other_sign_differs(
    small: WordcheckWorkspace, capsys: pytest.CaptureFixture
) -> None:
    info, _, _ = build_then_check(small, capsys, zero_at(1.0), zero_at(-1.0))
    name = f"cache/x0/run-{X0_JOB}"
    n = len(sample_of(small, "x0", small.root / name))
    result = info["results"]["runs"][name]
    assert (result["features_equal"], result["outputs_equal"], result["differ"]) == (n, n - 1, 1)
    assert result["largest_difference_fp16_steps"] == 0  # the same value, other bits
    assert info["status"] == "failed"


def same_output_other_features(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the first clip that the check reads another features fingerprint; its output stays."""
    real = module.features_sha256
    calls = [0]

    def fingerprint(features: Any) -> str:
        calls[0] += 1
        return "0" * 64 if calls[0] == 1 else real(features)

    monkeypatch.setattr(module, "features_sha256", fingerprint)


def test_a_difference_in_the_features_alone_fails_the_check(
    small: WordcheckWorkspace, capsys: pytest.CaptureFixture
) -> None:
    info, out, model = build_then_check(small, capsys, None, same_output_other_features)
    name = f"cache/x0/run-{X0_JOB}"
    n = len(sample_of(small, "x0", small.root / name))
    result = info["results"]["runs"][name]
    assert (result["features_equal"], result["outputs_equal"], result["differ"]) == (n - 1, n, 1)
    assert info["status"] == "failed"
    assert info["error"]["message"].startswith(f"1 of {n} sampled segments differ")
    assert f"features equal {n - 1} of {n}; encoder outputs equal {n} of {n} (" in out
    assert "language model" not in model.calls


def test_the_check_records_a_cache_run_built_on_another_cpu(small: WordcheckWorkspace) -> None:
    finish(small, cache_dir(small), "--view", "x0", job=X0_JOB)
    info = info_of(cache_dir(small))
    info["environment"]["cpu"] = "Another CPU"
    (cache_dir(small) / "run.json").write_text(json.dumps(info), encoding="utf-8")
    name = f"cache/x0/run-{X0_JOB}"
    checked = finish(small, check_dir(small), "--verify", name, job=CHECK_JOB)
    result = checked.info["results"]["runs"][name]
    assert checked.info["status"] == "done"  # the bits decide, not the machine
    assert (result["same_cpu"], result["same_gpu"]) == (False, True)
    assert "built with CPU Another CPU, GPU Synthetic GPU, unlike this job\n" in checked.out


def test_fp16_steps_counts_the_fp16_numbers_between_two_values_in_the_order_of_values() -> None:
    import torch

    module = load_script("build_caches")

    def steps(first: list[float], second: list[float]) -> int | None:
        return module.fp16_steps(
            torch.tensor(first, dtype=torch.float16), torch.tensor(second, dtype=torch.float16)
        )

    one_up = float(np.nextafter(np.float16(1.0), np.float16(2.0)))
    tiny = float(np.float16(6.0e-8))  # the smallest fp16 number above 0
    assert steps([0.0, 1.0], [-0.0, 1.0]) == 0  # +0 and -0 are the same value
    assert steps([1.0, 5.0], [one_up, 5.0]) == 1
    assert steps([-1.0], [1.0]) == 2 * 15_360  # 1.0 is the 15,360th fp16 number above 0
    assert steps([-tiny], [tiny]) == 2
    assert steps([1.0, 2.0], [1.0]) is None


def test_longest_clips_puts_the_longest_first_and_breaks_ties_by_segment() -> None:
    module = load_script("build_caches")
    manifest = pd.DataFrame(
        {"segment": ["300_003", "300_002", "300_001", "300_000"], "samples": [10, 20, 20, 5]}
    )
    assert module.longest_clips(manifest, 2) == ("300_001", "300_002")
    assert module.longest_clips(manifest, 9) == ("300_001", "300_002", "300_003", "300_000")


def test_max_abs_is_the_largest_size_of_an_output_even_when_it_is_negative(
    small: WordcheckWorkspace,
) -> None:
    def negate(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        real = module.encoder_output
        monkeypatch.setattr(module, "encoder_output", lambda model, inputs: -real(model, inputs))

    finished = finish(small, cache_dir(small), "--view", "x0", job=X0_JOB, prepare=negate)
    manifest = pd.read_parquet(finished.run / "manifest.parquet")
    for row in manifest.itertuples(index=False):
        output = from_bytes((finished.run / row.file).read_bytes(), WIDTH)
        assert float(output.max()) < row.max_abs == float(output.abs().max())
    assert finished.info["results"]["max_abs"] == manifest["max_abs"].max()


def test_the_encoders_time_is_its_own_apart_from_the_bytes_and_the_write(
    small: WordcheckWorkspace,
) -> None:
    def slow(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        encoder, serialise = module.encoder_output, module.to_bytes

        def encoded(model: Any, inputs: Any) -> Any:
            time.sleep(0.1)
            return encoder(model, inputs)

        def serialised(output: Any) -> bytes:
            time.sleep(0.3)
            return serialise(output)

        monkeypatch.setattr(module, "encoder_output", encoded)
        monkeypatch.setattr(module, "to_bytes", serialised)

    finished = finish(small, cache_dir(small), "--view", "x0", job=X0_JOB, prepare=slow)
    results = finished.info["results"]
    clips, seconds = results["clips"], results["seconds"]
    assert 0.1 * clips <= seconds["encoder"] < 0.3 * clips <= seconds["serialise"]
    assert results["encoder_seconds_per_chunk"] < 0.3 * clips / results["chunks"]


def test_a_view_without_clips_is_refused_before_the_model_loads(
    small: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    trim(small, ())  # every view pinned again without a clip
    model = Model()
    with pytest.raises(SystemExit):
        run_job(small, monkeypatch, "--view", "x0", job=X0_JOB, model=model)
    assert "FAILED: StepError: the pinned clips of x0 hold no clip" in capsys.readouterr().out
    assert info_of(cache_dir(small))["status"] == "failed"
    assert model.calls == []


def test_a_processor_that_makes_another_number_of_chunks_stops_the_run(
    small: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("build_caches")
    monkeypatch.setattr(module, "chunks", lambda _samples: 1)  # as if 30 s held any clip
    with pytest.raises(SystemExit):
        run_job(small, monkeypatch, "--view", "x0", job=X0_JOB, module=module)
    info = info_of(cache_dir(small))
    assert info["error"]["segment"] == LONG_SEGMENT
    assert info["error"]["message"] == (
        f"segment {LONG_SEGMENT}: the processor made 2 chunks of a clip of"
        f" {manifest_of(small, 'x0').set_index('segment').at[LONG_SEGMENT, 'samples']} samples,"
        " not 1"
    )
    assert f"FAILED: StepError in segment {LONG_SEGMENT}: " in capsys.readouterr().out


def test_a_job_that_slurm_stops_is_left_failed_naming_its_segment(
    small: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("build_caches")
    real = module.read_view_clip
    calls: list[int] = []

    def stopped(*args: Any) -> Any:  # what stop_on_sigterm's handler raises
        calls.append(1)
        if len(calls) == 3:
            raise SystemExit(STOPPED)
        return real(*args)

    monkeypatch.setattr(module, "read_view_clip", stopped)
    with pytest.raises(SystemExit):
        run_job(small, monkeypatch, "--view", "x0", job=X0_JOB, module=module)
    segment = str(manifest_of(small, "x0")["segment"].iloc[2])
    assert f"FAILED: SystemExit in segment {segment}: {STOPPED}" in capsys.readouterr().out
    run = cache_dir(small)
    info = info_of(run)
    assert (info["status"], info["error"]["message"]) == ("failed", STOPPED)
    assert info["error"]["segment"] == segment
    assert (run / "errors.log").is_file()
    assert not (run / "manifest.parquet").exists()


def test_a_cache_file_that_changes_before_its_read_back_fails_the_run_naming_its_segment(
    small: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("build_caches")
    real = module.write_file
    first: list[Path] = []

    def written(path: Path, data: bytes) -> None:
        real(path, data)
        if not first:
            first.append(path)
            path.write_bytes(data[:-1] + b"\0")  # as if the disk had changed a byte

    monkeypatch.setattr(module, "write_file", written)
    with pytest.raises(SystemExit):
        run_job(small, monkeypatch, "--view", "x0", job=X0_JOB, module=module)
    segment = first[0].stem
    info = info_of(cache_dir(small))
    assert info["error"] == {
        "type": "StepError",
        "segment": segment,
        "part": None,
        "message": f"segment {segment}: its cache file does not match the SHA-256 of its"
        " manifest row",
    }
    assert f"FAILED: StepError in segment {segment}: segment {segment}:" in capsys.readouterr().out
    assert not (cache_dir(small) / "manifest.parquet").exists()


def test_the_read_back_reads_every_file_to_the_last(
    small: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("build_caches")
    real = module.write_file
    total = len(manifest_of(small, "x0"))
    written: list[Path] = []

    def write(path: Path, data: bytes) -> None:
        real(path, data)
        written.append(path)
        if len(written) == total:  # the last file, changed after it was written
            path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))

    monkeypatch.setattr(module, "write_file", write)
    with pytest.raises(SystemExit):
        run_job(small, monkeypatch, "--view", "x0", job=X0_JOB, module=module)
    info = info_of(cache_dir(small))
    assert info["error"]["segment"] == written[-1].stem
    assert f"FAILED: StepError in segment {written[-1].stem}: " in capsys.readouterr().out
    assert not (cache_dir(small) / "manifest.parquet").exists()


# Refusals ----------------------------------------------------------------------------------------


MODES = {
    "view": (("--view", "x0"), cache_dir),
    "verify": (("--verify", f"cache/x0/run-{X0_JOB}"), check_dir),
}


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("job", "submit this as a job"),
        ("offline", "HF_HUB_OFFLINE is not 1"),
        ("dirty", "the checkout has uncommitted changes"),
        ("gpu", "no CUDA GPU"),
    ],
)
def test_both_modes_refuse_to_run_outside_a_clean_offline_gpu_job(
    workspace: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    mode: str,
    change: str,
    message: str,
) -> None:
    args, folder = MODES[mode]
    module = load_script("build_caches")
    model = Model()
    patch_job(module, workspace, monkeypatch, model)
    monkeypatch.setenv("SLURM_JOB_ID", X0_JOB if mode == "view" else CHECK_JOB)
    monkeypatch.setattr(sys, "argv", ["build_caches.py", *args])
    if change == "job":
        monkeypatch.delenv("SLURM_JOB_ID")
    elif change == "offline":
        monkeypatch.delenv("HF_HUB_OFFLINE")
    elif change == "dirty":
        monkeypatch.setattr(module, "git_state", lambda: {"commit": "c" * 40, "dirty": True})
    else:
        monkeypatch.setattr(module, "gpu_facts", lambda: {"gpu": None, "cuda": None})
    with pytest.raises(SystemExit):
        module.main()
    assert f"FAILED: StepError: {message}" in capsys.readouterr().out
    assert not folder(workspace).exists()
    assert model.calls == []


@pytest.mark.parametrize("mode", list(MODES))
def test_a_second_run_with_the_same_job_id_never_writes_into_the_first(
    workspace: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    mode: str,
) -> None:
    args, folder = MODES[mode]
    first = folder(workspace)
    first.mkdir(parents=True)
    (first / "run.json").write_text('{"status": "done"}\n', encoding="utf-8")
    model = Model()
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, *args, job=first.name.removeprefix("run-"), model=model)
    assert "FAILED: FileExistsError" in capsys.readouterr().out
    assert (first / "run.json").read_text(encoding="utf-8") == '{"status": "done"}\n'
    assert sorted(path.name for path in first.iterdir()) == ["run.json"]
    assert model.calls == []


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--view"],
        ["--view", "x0", "rev"],
        ["--view", "--verify"],
        ["--verify"],
        ["--verify", "cache/x0/run-800", "cache/x0/run-800"],
        ["--verify", "--view", "x0"],
        ["--views", "x0"],
        ["x0"],
    ],
)
def test_the_job_refuses_any_other_command_line(
    workspace: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    args: list[str],
) -> None:
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, *args, job=X0_JOB)
    assert "FAILED: StepError: usage: build_caches.py --view <view> | --verify" in (
        capsys.readouterr().out
    )
    assert not (workspace.root / "cache").exists()
    assert not (workspace.root / "derived" / "cache_check").exists()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--view", "x9"], "--view names a view without a pin in view_runs: 'x9'"),
        (["--verify", "../elsewhere"], "../elsewhere is not a folder inside the workspace"),
        (
            ["--verify", f"cache/x0/run-{X0_JOB}", f"cache/x0/../x0/run-{X0_JOB}"],
            "--verify names one cache run twice",
        ),
    ],
)
def test_the_job_refuses_a_view_without_a_pin_and_a_run_outside_the_workspace(
    workspace: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    args: list[str],
    message: str,
) -> None:
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, *args, job=CHECK_JOB)
    assert f"FAILED: StepError: {message}" in capsys.readouterr().out
    assert not (workspace.root / "cache").exists()
    assert not check_dir(workspace).exists()


def fail_the_cache_run(run: Path) -> None:
    info = json.loads((run / "run.json").read_text(encoding="utf-8"))
    info["status"] = "failed"
    (run / "run.json").write_text(json.dumps(info), encoding="utf-8")


def other_threads_in_the_cache_run(run: Path) -> None:
    info = json.loads((run / "run.json").read_text(encoding="utf-8"))
    info["settings"]["numerics"]["omp_num_threads"] = "8"
    (run / "run.json").write_text(json.dumps(info), encoding="utf-8")


@pytest.mark.parametrize(
    ("change", "target", "message"),
    [
        (fail_the_cache_run, f"cache/x0/run-{X0_JOB}", f"cache/x0/run-{X0_JOB} did not finish"),
        (
            None,
            f"cache/x0/run-{X0_JOB}/0",
            f"no cache run in cache/x0/run-{X0_JOB}/0: it has no run.json",
        ),
        (
            None,
            "views/x0/run-1",
            "views/x0/run-1 is not a run of build_caches.py --view",
        ),
        (
            other_threads_in_the_cache_run,
            f"cache/x0/run-{X0_JOB}",
            f"cache/x0/run-{X0_JOB} was built with other numeric settings than this job's",
        ),
    ],
)
def test_the_check_refuses_a_cache_run_it_cannot_compare_before_loading_the_model(
    built: dict[str, Finished],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    change: Callable[[Path], None] | None,
    target: str,
    message: str,
) -> None:
    workspace = built["x0"].workspace.copy_to(tmp_path)
    if change is not None:
        change(workspace.root / target)
    if target == "views/x0/run-1":
        (workspace.root / target / "run.json").write_text('{"step": "build_views"}', "utf-8")
    model = Model()
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, "--verify", target, job=CHECK_JOB, model=model)
    assert f"FAILED: StepError: {message}" in capsys.readouterr().out
    info = info_of(check_dir(workspace))
    assert info["status"] == "failed"
    assert info["error"]["message"].startswith(message)
    assert model.calls == []  # refused before the model loads


def test_the_check_refuses_any_named_cache_run_of_other_numerics_before_loading_the_model(
    built: dict[str, Finished],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    workspace = built["x0"].workspace.copy_to(tmp_path)
    other_threads_in_the_cache_run(cache_dir(workspace, "rev", REV_JOB))  # the second one
    model = Model()
    runs = [f"cache/x0/run-{X0_JOB}", f"cache/rev/run-{REV_JOB}"]
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, "--verify", *runs, job=CHECK_JOB, model=model)
    message = f"cache/rev/run-{REV_JOB} was built with other numeric settings than this job's"
    assert f"FAILED: StepError: {message}" in capsys.readouterr().out
    assert model.calls == []


def test_the_check_refuses_a_cache_run_of_another_width_than_the_model(
    built: dict[str, Finished],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    workspace = built["x0"].workspace.copy_to(tmp_path)
    run = cache_dir(workspace)
    info = info_of(run)
    info["settings"]["width"] = 2 * WIDTH
    (run / "run.json").write_text(json.dumps(info), encoding="utf-8")
    model = Model()
    with pytest.raises(SystemExit):
        run_job(
            workspace, monkeypatch, "--verify", f"cache/x0/run-{X0_JOB}", job=CHECK_JOB, model=model
        )
    message = f"cache/x0/run-{X0_JOB} holds outputs of width {2 * WIDTH}, not {WIDTH}"
    assert f"FAILED: StepError: {message}" in capsys.readouterr().out
    assert info_of(check_dir(workspace))["error"]["message"] == message
    assert model.calls == ["load"]  # refused before any clip is encoded


# The job file ------------------------------------------------------------------------------------


def test_the_job_file_asks_for_one_gpu_and_the_configs_threads_and_passes_its_arguments() -> None:
    text = (REPO / "scripts" / "slurm" / "build_caches.sbatch").read_text(encoding="utf-8")
    for line in (
        "#SBATCH --job-name=lbw-build-caches",
        "#SBATCH --partition=gpu",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=32G",
        "#SBATCH --time=02:00:00",
    ):
        assert f"{line}\n" in text, line
    # common.sh sets OMP_NUM_THREADS to the job's CPUs: the threads of encoder_cache.
    config = yaml.safe_load((REPO / "configs" / "pilot" / "data.yaml").read_text(encoding="utf-8"))
    cpus = re.search(r"^#SBATCH --cpus-per-task=(\d+)$", text, flags=re.MULTILINE)
    assert cpus is not None
    assert int(cpus.group(1)) == CacheSettings.from_config(config).cpu_threads
    assert "set -euo pipefail\nsource scripts/slurm/common.sh\n" in text
    assert "nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv\n" in (
        text
    )
    assert text.endswith('python scripts/build_caches.py "$@"\n')
    sbatch = [line for line in text.splitlines() if line.startswith("#SBATCH")]
    first_command = text.splitlines().index("set -euo pipefail")
    assert all(text.splitlines().index(line) < first_command for line in sbatch)


# The check's sizes of the config -----------------------------------------------------------------


@pytest.mark.slow
def test_both_modes_run_with_the_check_sizes_of_the_config(tmp_path: Path) -> None:
    workspace = make_cache_workspace(tmp_path / "workspace", None)  # every segment
    real = yaml.safe_load((REPO / "configs" / "pilot" / "data.yaml").read_text(encoding="utf-8"))
    assert workspace.config["encoder_cache"] == real["encoder_cache"]
    for view, job in (("x0", "900"), ("rev", "901")):
        built = finish(workspace, cache_dir(workspace, view, job), "--view", view, job=job)
        assert built.info["status"] == "done"
    names = ["cache/x0/run-900", "cache/rev/run-901"]
    checked = finish(workspace, check_dir(workspace, "902"), "--verify", *names, job="902")
    assert checked.info["status"] == "done"
    config = json.loads(json.dumps(workspace.config))
    config["cache_runs"] = {}
    for name in names:
        view = manifest_view(name)
        manifest = pd.read_parquet(workspace.root / name / "manifest.parquet")
        sample = check_sample(manifest, view, 25, 108)
        result = checked.info["results"]["runs"][name]
        assert result["segments"] == len(sample) >= 25
        assert result["features_equal"] == result["outputs_equal"] == len(sample)
        assert result["lm_watch"]["clips"] == 25
        config["cache_runs"][view] = {
            "run": name,
            "sha256": sha256_file(workspace.root / name / "manifest.parquet"),
            "verified_by": "derived/cache_check/run-902",
        }
        loaded = load_cache_manifest(Workspace(workspace.root.resolve()), config, view, VOXTRAL)
        assert len(loaded) == len(manifest)
