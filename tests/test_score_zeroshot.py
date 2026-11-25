"""The zero-shot scoring job (scripts/score_zeroshot.py), end to end, on the CPU.

The word check's synthetic workspace (wordcheck_fakes), cut to four participants, with one
clip over 30 s in x0, x2 and rev, a pinned word-check run of its own and no label file. A
tiny random Voxtral saved as the pinned snapshot (voxtral_fakes) is loaded on the CPU by
the jobs' own loader, with a stand-in for the processor. The caches of the four clip
versions are built and checked by the cache job itself (scripts/build_caches.py) and
pinned, so each cache differs per segment and view, and the model's z is a function of the
cached output alone: the tests compute it again from each file. The rules of the training
code count as dated, with their repeatability mode set: the repository's train.yaml, its
precision.deterministic set to warn_only as the commit of the timing run would set it.
"""

import contextlib
import hashlib
import io
import json
import math
import re
import shutil
import sys
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
from listening_beyond_words.cache import NUMERICS, CacheSettings, read_file
from listening_beyond_words.clips import MANIFEST_COLUMNS, write_view
from listening_beyond_words.models import load_pins, local_snapshot
from listening_beyond_words.runinfo import STOPPED, item_seed, sha256_file
from listening_beyond_words.settings import Workspace
from listening_beyond_words.wordcheck import FAIL, PASS, WAITS
from listening_beyond_words.zeroshot import SCORE_COLUMNS, check_sample, load_zeroshot_scores
from voxtral_fakes import (
    ANSWER_IDS,
    WIDTH,
    FakeProcessor,
    patch_processor,
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

KEPT_SESSIONS = (300, 301, 302, 303)  # 19 segments: enough to draw from, quick to score
LONG_SEGMENT = "301_002"  # its x0, x2 and rev clips last over 37 s: two chunks
LONG_SAMPLES = 598_400
LONG_VIEWS = ("x0", "x2", "rev")
VIEWS = ("x0", "x1", "rev")  # the scored views
CACHE_JOBS = {"x0": "800", "x1": "801", "x2": "802", "rev": "803"}  # every version is cached
CHECK_JOB = "804"
CHECK_RUN = f"derived/cache_check/run-{CHECK_JOB}"
JOB = "900"
STAMP = "20261201-120000"
WORDCHECK_RUN = "derived/wordcheck/run-700"
FAST_CACHE_CHECK = {"segments_per_view": 5, "seed": 108, "lm_watch_clips": 3}  # config: 25, 108, 25
FAST_ZEROSHOT = {  # configs/pilot/zeroshot.yaml checks 20 segments per view
    "extends": "train.yaml",
    "arm": "zeroshot",
    "views": ["x0", "x1", "rev"],
    "check": {"segments_per_view": 3, "seed": 108},
}
LAYERS = tuple(f"model.language_model.layers.{index}" for index in range(2))
# The watched steps from the adapter to the logits; all but the layers have their output only.
LANGUAGE_MODEL = ("model.multi_modal_projector", *LAYERS, "model.language_model.norm", "lm_head")
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
PLANTED = 12.345678
TRAINING = REPO / "configs" / "pilot" / "train.yaml"
MODE = "warn_only"  # a repeatability mode as the commit of the timing run sets it; not pending


# The workspace ---------------------------------------------------------------------------------


def trim(workspace: WordcheckWorkspace, sessions: tuple[int, ...]) -> None:
    """Keep only the clips of some participants in every view, pinned again."""
    for pin in workspace.config["view_runs"].values():
        run = workspace.root / pin["run"]
        manifest = pd.read_parquet(run / "manifest.parquet")
        kept = manifest[manifest["session"].isin(sessions)].reset_index(drop=True)
        pin["sha256"] = write_pinned(run, kept, "manifest.parquet")
    workspace.write_config()


def lengthen(workspace: WordcheckWorkspace) -> None:
    """Replace the long segment's clips in x0, x2 and rev by clips over 30 s with its parts."""
    for number, view in enumerate(LONG_VIEWS):
        pin = workspace.config["view_runs"][view]
        run = workspace.root / pin["run"]
        rows = pd.read_parquet(run / "manifest.parquet").to_dict("records")
        (row,) = [row for row in rows if row["segment"] == LONG_SEGMENT]
        parts = len(row["part_starts"])
        rng = np.random.default_rng([8, number])
        clip, starts, ends = clip_samples(
            rng, [0] * parts, (LONG_SAMPLES - GAP * (parts - 1)) // parts, 0
        )
        path = run / row["file"]
        path.unlink()
        row.update(write_view(path, clip), part_starts=starts, part_ends=ends)
        pin["sha256"] = write_pinned(
            run, pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS)), "manifest.parquet"
        )
    workspace.write_config()


def add_snapshot(workspace: WordcheckWorkspace) -> None:
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


def pin_wordcheck_run(
    workspace: WordcheckWorkspace,
    verdicts: dict[str, str | None] | None = None,
    views: dict[str, dict[str, str]] | None = None,
) -> None:
    """A finished word-check run of the default mode, written as check_views writes it, pinned."""
    folder = workspace.root / WORDCHECK_RUN
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    table = pd.DataFrame({"segment": ["300_000"], "view": ["rev"], "words": [40]})
    sha256 = write_pinned(folder, table, "transcripts.parquet")
    given = {"f4": PASS, "x0_control": PASS, "x1_words": PASS, "x1_voice": FAIL, **(verdicts or {})}
    pins = workspace.config["view_runs"]
    info = {
        "step": "check_views",
        "mode": {"name": "sample"},
        "status": "done",
        "inputs": {
            "views": views
            or {view: {"file": pin["run"], "sha256": pin["sha256"]} for view, pin in pins.items()}
        },
        "verdicts": {key: {"verdict": value} for key, value in given.items() if value is not None},
        "outputs": {"transcripts.parquet": sha256},
    }
    (folder / "run.json").write_text(json.dumps(info), encoding="utf-8")
    workspace.config["wordcheck_run"] = {"run": WORDCHECK_RUN, "sha256": sha256}
    workspace.write_config()


def make_base(root: Path, sessions: tuple[int, ...]) -> WordcheckWorkspace:
    """The workspace without caches: clips, the snapshot and the word-check run; no label file."""
    workspace = make_wordcheck_workspace(root)
    trim(workspace, sessions)
    lengthen(workspace)
    add_snapshot(workspace)
    for name in ("labels", "partitions"):  # the job reads no label
        shutil.rmtree(root / "derived" / name)
    workspace.config["encoder_cache"]["verify"] = dict(FAST_CACHE_CHECK)
    workspace.write_config()
    pin_wordcheck_run(workspace)
    return workspace


# Running the jobs ------------------------------------------------------------------------------


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
    """The jobs' model loader: the tiny snapshot on the CPU, with its calls logged.

    Attributes:
        plant: Changes the loaded model, to plant a fault.
        calls: "load" once per load; then "encoder", "language model" and "output layer",
            one per call, in order.
    """

    plant: Callable[[Any], None] | None = None
    calls: list[str] = field(default_factory=list)

    def load(self, snapshot: Path, device: str = "cuda") -> voxtral.LoadedModel:
        assert device == "cuda"  # the jobs ask for the GPU; the tests run on the CPU
        self.calls.append("load")
        loaded = voxtral.load_model(snapshot, device="cpu")
        model = loaded.model
        model.model.audio_tower.register_forward_hook(lambda *_: self.log("encoder"))
        model.model.language_model.register_forward_pre_hook(lambda *_: self.log("language model"))
        model.lm_head.register_forward_pre_hook(lambda *_: self.log("output layer"))
        if self.plant is not None:
            self.plant(model)
        return loaded

    def log(self, name: str) -> None:
        self.calls.append(name)


def zeroshot_path(workspace: WordcheckWorkspace) -> Path:
    return workspace.config_path.parent / "zeroshot.yaml"


def training_path(workspace: WordcheckWorkspace) -> Path:
    return workspace.config_path.parent / "train.yaml"


def write_training(workspace: WordcheckWorkspace, mode: str = MODE) -> None:
    """The repository's train.yaml beside the workspace's configs, with a repeatability mode."""
    config = yaml.safe_load(TRAINING.read_text(encoding="utf-8"))
    config["precision"]["deterministic"] = mode
    training_path(workspace).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def patch_job(
    module: ModuleType,
    workspace: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    model: Model,
    zeroshot: dict[str, Any] | None = None,
) -> None:
    """Point a job at the workspace, a clean checkout, a GPU, the tiny model and its processor."""
    monkeypatch.setattr(sys, "unraisablehook", sys.unraisablehook)  # main() replaces it
    monkeypatch.setattr(module, "CONFIG", workspace.config_path)
    monkeypatch.setattr(module, "MODELS", REPO / "configs" / "pilot" / "models.yaml")
    if hasattr(module, "ZEROSHOT"):
        path = zeroshot_path(workspace)
        path.write_text(yaml.safe_dump(zeroshot or FAST_ZEROSHOT), encoding="utf-8")
        monkeypatch.setattr(module, "ZEROSHOT", path)
        monkeypatch.setattr(module, "run_name", lambda: f"pilot-zeroshot-{STAMP}")
    if hasattr(module, "TRAINING"):  # the rules of the training code, dated, their mode set
        write_training(workspace)
        monkeypatch.setattr(module, "TRAINING", training_path(workspace))
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


def build_caches(workspace: WordcheckWorkspace) -> None:
    """Cache the four versions with the cache job, check them with it, and pin them."""
    calls = [(("--view", view), job) for view, job in CACHE_JOBS.items()]
    runs = [f"cache/{view}/run-{job}" for view, job in CACHE_JOBS.items()]
    calls.append((("--verify", *runs), CHECK_JOB))
    for args, job in calls:
        out = io.StringIO()
        with pytest.MonkeyPatch.context() as monkeypatch, kept_numerics():
            module = load_script("build_caches")
            patch_job(module, workspace, monkeypatch, Model())
            monkeypatch.setenv("SLURM_JOB_ID", job)
            monkeypatch.setattr(sys, "argv", ["build_caches.py", *args])
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                try:
                    module.main()
                except SystemExit as stopped:
                    raise AssertionError(out.getvalue()) from stopped
    workspace.config["cache_runs"] = {
        view: {
            "run": run,
            "sha256": sha256_file(workspace.root / run / "manifest.parquet"),
            "verified_by": CHECK_RUN,
        }
        for view, run in zip(CACHE_JOBS, runs, strict=True)
    }
    workspace.write_config()


def run_dir(workspace: WordcheckWorkspace) -> Path:
    return workspace.root / "runs" / "pilot" / f"pilot-zeroshot-{STAMP}"


def run_job(
    workspace: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    *args: str,
    model: Model | None = None,
    module: ModuleType | None = None,
    zeroshot: dict[str, Any] | None = None,
) -> Path:
    """Run the zero-shot job on the synthetic workspace; return its run folder."""
    script = module or load_script("score_zeroshot")
    patch_job(script, workspace, monkeypatch, model or Model(), zeroshot)
    monkeypatch.setenv("SLURM_JOB_ID", JOB)
    monkeypatch.setattr(sys, "argv", ["score_zeroshot.py", *args])
    script.main()
    return run_dir(workspace)


@dataclass
class Finished:
    """A run of the job that ended, and what it printed.

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


def finish(
    workspace: WordcheckWorkspace,
    model: Model | None = None,
    prepare: Callable[[ModuleType, pytest.MonkeyPatch], None] | None = None,
) -> Finished:
    """Run the job to its end in a patch context of its own, for fixtures shared by a module."""
    out, err = io.StringIO(), io.StringIO()
    used = model or Model()
    with pytest.MonkeyPatch.context() as monkeypatch, kept_numerics():
        module = load_script("score_zeroshot")
        if prepare is not None:
            prepare(module, monkeypatch)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            run_job(workspace, monkeypatch, model=used, module=module)
    return Finished(workspace, run_dir(workspace), out.getvalue(), err.getvalue(), used)


def failed(
    workspace: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    *args: str,
    model: Model | None = None,
    module: ModuleType | None = None,
) -> tuple[dict, str]:
    """Run the job, which must fail, leaving no score; return its run.json and its output."""
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, *args, model=model, module=module)
    out, err = capsys.readouterr()
    run = run_dir(workspace)
    assert not (run / "scores.parquet").exists()
    assert_only_aggregates_leave(run / "run.json", out + err)
    info = info_of(run)
    assert info["status"] == "failed"
    assert "outputs" not in info
    return info, out


# The workspaces --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> WordcheckWorkspace:
    """Four participants' clips, the snapshot and the word-check run; no cache yet."""
    return make_base(tmp_path_factory.mktemp("base") / "workspace", KEPT_SESSIONS)


@pytest.fixture(scope="module")
def template(
    base: WordcheckWorkspace, tmp_path_factory: pytest.TempPathFactory
) -> WordcheckWorkspace:
    """The base with the caches of x0, x1 and rev built, checked and pinned."""
    workspace = base.copy_to(tmp_path_factory.mktemp("template"))
    build_caches(workspace)
    return workspace


@pytest.fixture(scope="module")
def small_template(
    base: WordcheckWorkspace, tmp_path_factory: pytest.TempPathFactory
) -> WordcheckWorkspace:
    """One participant's clips, the long one among them, cached and pinned: for planted faults."""
    workspace = base.copy_to(tmp_path_factory.mktemp("small"))
    trim(workspace, (int(LONG_SEGMENT[:3]),))
    pin_wordcheck_run(workspace)  # it read the clips as they are pinned now
    build_caches(workspace)
    return workspace


@pytest.fixture
def uncached(base: WordcheckWorkspace, tmp_path: Path) -> WordcheckWorkspace:
    return base.copy_to(tmp_path)


@pytest.fixture
def workspace(template: WordcheckWorkspace, tmp_path: Path) -> WordcheckWorkspace:
    return template.copy_to(tmp_path)


@pytest.fixture
def small(small_template: WordcheckWorkspace, tmp_path: Path) -> WordcheckWorkspace:
    return small_template.copy_to(tmp_path)


def logged_reads(seen: list[tuple[str, str]]) -> Callable[[ModuleType, pytest.MonkeyPatch], None]:
    """Log each clip that the job reads, as (view, segment)."""

    def prepare(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        real = module.read_view_clip

        def logged(folder: Path, row: Any) -> Any:
            seen.append((Path(folder).parts[-2], str(row.segment)))  # views/<view>/run-<job>
            return real(folder, row)

        monkeypatch.setattr(module, "read_view_clip", logged)

    return prepare


@pytest.fixture(scope="module")
def scored(template: WordcheckWorkspace, tmp_path_factory: pytest.TempPathFactory) -> Finished:
    """The job, run to its end on the four participants, with a progress line every 10 clips
    and every clip that it reads logged."""
    seen: list[tuple[str, str]] = []

    def prepare(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(module, "PROGRESS_EVERY", 10)
        logged_reads(seen)(module, monkeypatch)

    finished = finish(template.copy_to(tmp_path_factory.mktemp("scored")), prepare=prepare)
    finished.reads = seen
    return finished


def cache_manifest(workspace: WordcheckWorkspace, view: str) -> pd.DataFrame:
    run = workspace.config["cache_runs"][view]["run"]
    return pd.read_parquet(workspace.root / run / "manifest.parquet").sort_values(
        "segment", ignore_index=True
    )


def samples_of(workspace: WordcheckWorkspace) -> dict[str, tuple[str, ...]]:
    return {view: check_sample(cache_manifest(workspace, view), view, 3, 108) for view in VIEWS}


def checked(workspace: WordcheckWorkspace) -> int:
    """The number of segments of the check, over all views."""
    return sum(len(sample) for sample in samples_of(workspace).values())


def segment_count(workspace: WordcheckWorkspace) -> int:
    return sum(len(cache_manifest(workspace, view)) for view in VIEWS)


# A finished run --------------------------------------------------------------------------------


def test_the_job_scores_every_segment_of_each_view_and_ends_done(scored: Finished) -> None:
    workspace, run, info = scored.workspace, scored.run, scored.info
    assert info["status"] == "done"
    assert (info["step"], info["arm"], info["run_id"]) == (
        "score_zeroshot",
        "zeroshot",
        f"pilot-zeroshot-{STAMP}",
    )
    assert sorted(path.name for path in run.iterdir()) == ["run.json", "scores.parquet"]
    scores = pd.read_parquet(run / "scores.parquet")
    assert tuple(scores.columns) == SCORE_COLUMNS
    assert str(scores["z"].dtype) == "float64"
    assert list(scores["view"].unique()) == list(VIEWS)
    for view in VIEWS:
        rows = scores[scores["view"] == view]
        manifest = cache_manifest(workspace, view)
        assert list(rows["segment"]) == list(manifest["segment"])  # by segment
        assert list(rows["session"]) == list(manifest["session"])
    assert np.isfinite(scores["z"]).all()
    assert info["outputs"] == {"scores.parquet": sha256_file(run / "scores.parquet")}
    out = scored.out
    assert f"run: runs/pilot/pilot-zeroshot-{STAMP}\n" in out
    assert f"scores.parquet sha256: {info['outputs']['scores.parquet']}\n" in out
    assert re.search(r"^done in \d+\.\d min$", out, flags=re.MULTILINE)
    assert "FAILED" not in out


def test_every_z_is_the_score_of_its_own_segments_cache_in_its_own_view(
    scored: Finished, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = scored.workspace
    patch_processor(monkeypatch)
    loaded = voxtral.load_model(local_snapshot(VOXTRAL, workspace.root / "hf"), device="cpu")
    voxtral.set_numerics(4)  # as the job; numerics_put_back restores them
    processor = FakeProcessor()
    prompts = {count: voxtral.prompt_inputs(processor, count) for count in (1, 2)}
    scores = pd.read_parquet(scored.run / "scores.parquet")
    voxtral.use_cached_encoder(loaded.model)
    files = []
    for view in VIEWS:
        folder = workspace.root / workspace.config["cache_runs"][view]["run"]
        stored = {row.segment: row for row in cache_manifest(workspace, view).itertuples()}
        for row in scores[scores["view"] == view].itertuples(index=False):
            cached = stored[row.segment]
            files.append(cached.sha256)
            output = read_file(folder / cached.file, cached, WIDTH)
            z = voxtral.cached_score(loaded.model, prompts[cached.chunks], output, ANSWER_IDS)
            assert z.hex() == float(row.z).hex(), (view, row.segment)
    # Every cache differs, and so does nearly every z (z is the difference of two fp16 logits,
    # so a few coincide): a row of another segment or view would show.
    assert len(set(files)) == len(scores)
    assert scores["z"].map(float.hex).nunique() >= 0.9 * len(scores)


def test_the_check_scores_its_sample_from_the_audio_and_from_the_cache_and_finds_them_equal(
    scored: Finished,
) -> None:
    workspace, info = scored.workspace, scored.info
    samples = samples_of(workspace)
    check = info["results"]["check"]["views"]
    for view, sample in samples.items():
        n = len(sample)
        two = 0 if view == "x1" else 1
        assert (LONG_SEGMENT in sample) == bool(two)
        assert check[view] == {
            "segments": n,
            "drawn": 3,
            "two_chunk": two,
            "z_equal": n,
            "non_finite": 0,
            "features_equal": n,
            "token_ids_equal": n,
        }
        assert (
            f"check {view}: z from the cache equals z from the audio in {n} of {n} segments (3"
            f" drawn with seed 108, {two} of two chunks); 0 with a z that is not finite; log-mel"
            f" features equal {n} of {n}, token IDs equal {n} of {n}\n"
        ) in scored.out
    # The check's rows of scores.parquet hold the z that the audio gave.
    assert (
        f"the check sample's {sum(len(s) for s in samples.values())} rows equal their z from the"
        " audio"
    ) in scored.out


def test_the_check_scores_exactly_its_sample_of_each_view_from_the_audio(
    scored: Finished,
) -> None:
    for view in VIEWS:
        manifest = cache_manifest(scored.workspace, view)
        segments = sorted(manifest["segment"])
        rng = np.random.default_rng(item_seed(108, "zeroshot", view))
        drawn = {segments[int(index)] for index in rng.choice(len(segments), 3, replace=False)}
        long = set(manifest.loc[manifest["chunks"] > 1, "segment"])
        assert sorted(s for v, s in scored.reads if v == view) == sorted(drawn | long), view
    assert {view for view, _ in scored.reads} == set(VIEWS)  # x2 is never scored


def test_the_encoder_runs_only_on_the_check_sample_and_every_other_z_comes_from_the_cache(
    scored: Finished,
) -> None:
    n = checked(scored.workspace)
    total = segment_count(scored.workspace)
    assert scored.model.calls == [
        "load",
        *["encoder", "language model", "output layer"] * n,  # the check, from the audio
        *["language model", "output layer"] * n,  # the check, from the cache
        *["language model", "output layer"] * total,  # every segment of each view
    ]


def test_run_json_records_the_configs_seeds_inputs_and_settings(scored: Finished) -> None:
    workspace, info = scored.workspace, scored.info
    assert info["config"] == {
        "file": str(workspace.config_path),
        "sha256": sha256_file(workspace.config_path),
    }
    models = REPO / "configs" / "pilot" / "models.yaml"
    assert info["models_config"] == {"file": str(models), "sha256": sha256_file(models)}
    path = zeroshot_path(workspace)
    assert info["zeroshot_config"] == {"file": str(path), "sha256": sha256_file(path)}
    settings = info["settings"]
    assert settings["zeroshot"] == FAST_ZEROSHOT
    assert settings["encoder_cache"] == workspace.config["encoder_cache"]
    assert settings["numerics"] == NUMERIC_RECORD
    assert settings["width"] == WIDTH
    processor = FakeProcessor()
    prompt = settings["prompt"]
    assert prompt["text"] == voxtral.PROMPT
    assert (prompt["answers"], prompt["answer_ids"]) == (["Yes", "No"], list(ANSWER_IDS))
    for count, tokens in (("1", 397), ("2", 772)):
        ids = voxtral.prompt_inputs(processor, int(count))["input_ids"]
        expected = hashlib.sha256(ids.numpy().astype("<i8").tobytes()).hexdigest()
        assert prompt["token_ids"][count] == {"tokens": tokens, "sha256": expected}
    assert info["seeds"]["check_sample"]["seed"] == 108
    assert info["seeds"]["check_sample"]["item_seeds"] == {
        view: item_seed(108, "zeroshot", view) for view in VIEWS
    }
    assert set(info["seeds"]["check_sample"]["streams"]) == set(VIEWS)
    inputs = info["inputs"]
    assert inputs["caches"] == {view: workspace.config["cache_runs"][view] for view in VIEWS}
    assert inputs["cache_checks"] == {
        view: {"run": CHECK_RUN, "lm_flagged": []} for view in ("x0", "x1", "x2", "rev")
    }
    assert inputs["views"] == {view: workspace.config["view_runs"][view] for view in VIEWS}
    model = inputs["voxtral"]
    assert (model["repo_id"], model["revision"]) == (VOXTRAL.repo_id, VOXTRAL.revision)
    assert {"config.json", "model.safetensors"} <= set(model["blobs"])
    record = workspace.root / WORDCHECK_RUN
    assert inputs["wordcheck_run"] == {
        "run": WORDCHECK_RUN,
        "sha256": workspace.config["wordcheck_run"]["sha256"],
        "run_json_sha256": sha256_file(record / "run.json"),
    }
    assert info["embargo"] == {
        "views": {
            view: {
                "file": workspace.config["view_runs"][view]["run"],
                "sha256": workspace.config["view_runs"][view]["sha256"],
            }
            for view in ("x1", "rev")
        },
        "verdicts": {"f4": PASS, "x1_words": PASS, "x1_voice": FAIL},
    }
    assert info["slurm"]["job_id"] == JOB
    assert info["slurm"]["cpus_per_task"] == "4"
    assert {key: info["environment"][key] for key in GPU} == GPU
    assert info["environment"]["packages"]["torch"]


def test_run_json_and_the_output_hold_label_free_aggregates_only(scored: Finished) -> None:
    info, out = scored.info, scored.out
    scores = pd.read_parquet(scored.run / "scores.parquet")
    for view in VIEWS:
        z = scores.loc[scores["view"] == view, "z"].to_numpy()
        numbers = info["results"]["z"][view]
        assert numbers == {
            "count": len(z),
            "all_finite": True,
            "mean": float(z.mean()),
            "sd": float(z.std()),
            "p5": float(np.percentile(z, 5)),
            "p50": float(np.percentile(z, 50)),
            "p95": float(np.percentile(z, 95)),
            "share_above_0": float((z > 0).mean()),
        }
        assert (
            f"z of {view}: {len(z)} scores, all finite; mean {numbers['mean']:.4f}, SD"
            f" {numbers['sd']:.4f}; percentiles 5, 50 and 95: {numbers['p5']:.4f},"
            f" {numbers['p50']:.4f}, {numbers['p95']:.4f}; share above 0"
            f" {numbers['share_above_0']:.4f}\n"
        ) in out
    assert_only_aggregates_leave(scored.run / "run.json", out + scored.err)


def test_the_share_above_0_leaves_out_a_z_of_exactly_0() -> None:
    module = load_script("score_zeroshot")
    numbers = module.label_free(np.array([-1.0, 0.0, -0.0, 0.5, 2.0]))
    assert numbers["share_above_0"] == 0.4
    assert numbers["count"] == 5


def test_the_language_model_and_its_logits_are_watched_on_every_scored_clip(
    scored: Finished,
) -> None:
    results = scored.info["results"]
    watch = results["lm_watch"]
    calls = 2 * checked(scored.workspace) + segment_count(scored.workspace)
    assert list(watch["layers"]) == list(LANGUAGE_MODEL)
    # The adapter, 2 layers with their attention and feed-forward parts, the final norm and
    # the logits.
    assert watch["parts"] == 9
    assert watch["flagged"] == []
    assert results["non_finite_z"] == 0
    for name, parts in watch["layers"].items():
        assert tuple(parts) == (voxtral.PARTS if name in LAYERS else ("output",))
        for numbers in parts.values():
            assert numbers["calls"] == calls
            assert numbers["at_clamp"] == numbers["non_finite"] == 0
    assert "\nlanguage-model watch after the check: 0 of 9 parts flagged" in scored.out
    assert "\nlanguage-model watch after every score: 0 of 9 parts flagged" in scored.out


def test_the_job_prints_its_steps_one_per_line(scored: Finished) -> None:
    out = scored.out
    assert out.startswith(f"zero-shot scores of x0, x1, rev; job {JOB} on ")
    assert "\nnumerics: deterministic_algorithms True, " in out
    assert (
        f"\nembargo: the checks of x1 are finished ({WORDCHECK_RUN}: F4 PASS, x1 words PASS, x1"
        " voice FAIL), on the pinned clips of x1 and rev\n"
    ) in out
    assert (
        f"\ncache checks: {CHECK_RUN} watched the language model on the longest clips of x0, x1,"
        " x2, rev: 0 parts flagged\n"
    ) in out
    for view in VIEWS:
        job = CACHE_JOBS[view]
        clips = len(cache_manifest(scored.workspace, view))
        two = 0 if view == "x1" else 1
        assert (
            f"\ncache {view}: cache/{view}/run-{job}, {clips} clips ({two} of two chunks), checked"
            f" by {CHECK_RUN}\n"
        ) in out
    assert re.search(r"^model: mistralai/Voxtral-Mini-3B-2507 at \w{8}, loaded in ", out, re.M)
    assert "; Yes 60, No 61\n" in out
    assert "\nprompt: 1 chunk(s): 397 token IDs, SHA-256 " in out
    assert re.search(r"^scores: \d+ clips from the cache in ", out, re.M)


def test_the_progress_lines_give_counts_and_times_only(scored: Finished) -> None:
    total = segment_count(scored.workspace)
    found = re.findall(
        r"^scores: (\d+) of (\d+), [\d.]+ per second; projected end \d\d:\d\d UTC$",
        scored.out,
        flags=re.MULTILINE,
    )
    assert found == [(str(done), str(total)) for done in range(10, total, 10)]


def test_no_label_file_lies_in_the_workspace_the_job_ran_in(scored: Finished) -> None:
    root = scored.workspace.root
    assert scored.info["status"] == "done"
    assert not list(root.rglob("labels*"))
    assert not list(root.rglob("partitions*"))


def test_the_finished_run_once_pinned_passes_load_zeroshot_scores(scored: Finished) -> None:
    workspace = scored.workspace
    config = json.loads(json.dumps(workspace.config))
    config["zeroshot_scores"] = {
        "run": f"runs/pilot/pilot-zeroshot-{STAMP}",
        "sha256": sha256_file(scored.run / "scores.parquet"),
    }
    loaded = load_zeroshot_scores(Workspace(workspace.root.resolve()), config)
    assert loaded.equals(pd.read_parquet(scored.run / "scores.parquet"))


# What may leave OMNI ---------------------------------------------------------------------------


def plant_z(
    segment: str, view: str, value: float
) -> Callable[[ModuleType, pytest.MonkeyPatch], None]:
    """Make the job's cached_score give one value for one segment of one view."""

    def prepare(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        real_read, real_score = module.read_file, module.cached_score
        last: dict[str, Path] = {}

        def read(path: Path, row: Any, width: int) -> Any:
            last["path"] = path
            return real_read(path, row, width)

        def score(*args: Any) -> float:
            z = real_score(*args)
            path = last["path"]
            return value if (path.stem, path.parents[2].name) == (segment, view) else z

        monkeypatch.setattr(module, "read_file", read)
        monkeypatch.setattr(module, "cached_score", score)

    return prepare


def test_a_planted_z_reaches_no_stream_and_no_run_json(
    template: WordcheckWorkspace, tmp_path: Path
) -> None:
    workspace = template.copy_to(tmp_path)
    sample = samples_of(workspace)["x0"]
    segment = next(s for s in cache_manifest(workspace, "x0")["segment"] if s not in sample)
    finished = finish(workspace, prepare=plant_z(segment, "x0", PLANTED))
    assert finished.info["status"] == "done"
    scores = pd.read_parquet(finished.run / "scores.parquet")
    row = scores[(scores["view"] == "x0") & (scores["segment"] == segment)]
    assert row["z"].tolist() == [PLANTED]  # the plant took
    shown = finished.out + finished.err + (finished.run / "run.json").read_text(encoding="utf-8")
    for spelling in ("12.345678", "12.34568", "12.3457", PLANTED.hex()):
        assert spelling not in shown, spelling
    assert_only_aggregates_leave(finished.run / "run.json", finished.out + finished.err)


def test_a_planted_z_of_a_checked_segment_reaches_no_stream_and_no_run_json(
    small: WordcheckWorkspace,
) -> None:
    segment = samples_of(small)["x0"][0]  # the first clip that the check scores

    def prepare(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        plant_z(segment, "x0", PLANTED)(module, monkeypatch)  # from the cache, every time
        replace_z(module, monkeypatch, "yes_no_score", lambda call, z: PLANTED if call == 1 else z)

    finished = finish(small, prepare=prepare)
    assert finished.info["status"] == "done"  # both paths agree: the check passes
    scores = pd.read_parquet(finished.run / "scores.parquet")
    assert scores.loc[scores["segment"] == segment, "z"].iloc[0] == PLANTED  # the plant took
    shown = finished.out + finished.err + (finished.run / "run.json").read_text(encoding="utf-8")
    for spelling in ("12.345678", "12.34568", "12.3457", PLANTED.hex()):
        assert spelling not in shown, spelling
    assert_only_aggregates_leave(finished.run / "run.json", finished.out + finished.err)


def test_a_check_that_fails_names_no_z_on_its_failed_line_or_in_run_json(
    small: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("score_zeroshot")
    replace_z(module, monkeypatch, "yes_no_score", lambda call, z: PLANTED if call == 1 else z)
    info, out = failed(small, monkeypatch, capsys, module=module)
    assert info["error"]["message"].startswith("the check found z from the cache unequal")
    shown = out + json.dumps(info)
    for spelling in ("12.345678", "12.34568", "12.3457", PLANTED.hex()):
        assert spelling not in shown, spelling


# Faults that end a run failed, with no score -----------------------------------------------------


def test_a_planted_check_mismatch_fails_the_job_before_any_other_score(
    small: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("score_zeroshot")
    real = module.cached_score
    calls: list[int] = []

    def nudged(*args: Any) -> float:  # the first z from the cache, one step off
        z = real(*args)
        calls.append(1)
        return math.nextafter(z, math.inf) if len(calls) == 1 else z

    monkeypatch.setattr(module, "cached_score", nudged)
    model = Model()
    info, out = failed(small, monkeypatch, capsys, model=model, module=module)
    n = checked(small)
    message = (
        f"the check found z from the cache unequal to z from the audio, or not finite, in 1 of {n}"
        " segments"
    )
    assert info["error"]["message"].startswith(message)
    assert f"FAILED: StepError: {message}" in out
    check = info["results"]["check"]["views"]
    assert sum(result["segments"] - result["z_equal"] for result in check.values()) == 1
    assert check["x0"]["z_equal"] == check["x0"]["segments"] - 1
    assert check["x0"]["features_equal"] == check["x0"]["segments"]
    assert len(calls) == n  # the check only: nothing else was scored
    assert "lm_watch" not in info["results"]
    assert model.calls.count("language model") == 2 * n


def plant_in_output(module: Any, when: Callable[[int], bool], value: float) -> None:
    """Write one value into a module's output at the last position, on the calls ``when`` picks.

    Only the last position reaches the logits (logits_to_keep=1).
    """
    calls = [0]

    def hook(_module: Any, _args: Any, output: Any) -> Any:
        calls[0] += 1
        if not when(calls[0]):
            return None
        planted = output.clone()
        planted[0, -1, 5] = value
        return planted

    module.register_forward_hook(hook)


@pytest.mark.parametrize(
    ("value", "flagged", "non_finite"),
    [
        (
            math.inf,
            [
                "model.language_model.layers.1/output",
                "model.language_model.norm/output",
                "lm_head/output",
            ],
            1,
        ),
        (64_512.0, ["model.language_model.layers.1/output"], 0),
    ],
)
def test_a_planted_overflow_in_the_language_model_fails_the_job_and_writes_no_score(
    small: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    value: float,
    flagged: list[str],
    non_finite: int,
) -> None:
    after = 2 * checked(small) + 4  # the fourth clip after the check

    def plant(tiny: Any) -> None:
        plant_in_output(tiny.model.language_model.layers[1], lambda call: call == after, value)

    model = Model(plant=plant)
    info, out = failed(small, monkeypatch, capsys, model=model)
    message = (
        f"the language-model watch flagged {len(flagged)} parts and {non_finite} scores are not"
        " finite: no score is written"
    )
    assert info["error"]["message"].startswith(message)
    assert f"FAILED: StepError: {message}" in out
    results = info["results"]
    assert results["lm_watch"]["flagged"] == flagged
    assert results["non_finite_z"] == non_finite
    assert results["scoring"]["clips"] == segment_count(small)  # every clip, then the verdict
    layer = results["lm_watch"]["layers"]["model.language_model.layers.1"]["output"]
    assert (layer["non_finite"], layer["at_clamp"]) == ((1, 0) if non_finite else (0, 1))
    assert "z" not in results  # no number of the scores
    assert "\nlanguage-model watch, flagged: " + ", ".join(flagged) + "\n" in out


def test_a_flag_in_the_check_stops_the_job_before_the_rest_is_scored(
    small: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    # The same value in every call: z from the audio and from the cache still agree.
    def plant(tiny: Any) -> None:
        plant_in_output(tiny.model.language_model.layers[1], lambda _call: True, 64_512.0)

    model = Model(plant=plant)
    info, out = failed(small, monkeypatch, capsys, model=model)
    n = checked(small)
    check = info["results"]["check"]["views"]
    assert all(result["z_equal"] == result["segments"] for result in check.values())
    assert info["results"]["lm_watch"]["flagged"] == ["model.language_model.layers.1/output"]
    assert info["error"]["message"].startswith("the language-model watch flagged 1 parts")
    assert "\nlanguage-model watch after the check: 1 of 9 parts flagged" in out
    assert "scoring" not in info["results"]
    assert model.calls.count("language model") == 2 * n


def replace_z(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    change: Callable[[int, float], float],
) -> None:
    """Let one of the job's scoring functions give change(call number, z) instead of z."""
    real = getattr(module, name)
    calls = [0]

    def replaced(*args: Any) -> float:
        calls[0] += 1
        return change(calls[0], real(*args))

    monkeypatch.setattr(module, name, replaced)


def test_a_zero_of_the_other_sign_fails_the_check_which_compares_bits(
    small: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("score_zeroshot")
    replace_z(module, monkeypatch, "yes_no_score", lambda call, z: 0.0 if call == 1 else z)
    replace_z(module, monkeypatch, "cached_score", lambda call, z: -0.0 if call == 1 else z)
    info, _ = failed(small, monkeypatch, capsys, module=module)
    check = info["results"]["check"]["views"]["x0"]
    assert (check["z_equal"], check["non_finite"]) == (check["segments"] - 1, 0)


def test_a_z_that_is_not_finite_fails_the_check_though_both_paths_give_it(
    small: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("score_zeroshot")
    for name in ("yes_no_score", "cached_score"):  # the same NaN, bit for bit, on both paths
        replace_z(module, monkeypatch, name, lambda call, z: math.nan if call == 1 else z)
    info, _ = failed(small, monkeypatch, capsys, module=module)
    check = info["results"]["check"]["views"]["x0"]
    assert (check["z_equal"], check["non_finite"]) == (check["segments"] - 1, 1)
    assert info["error"]["message"].startswith(
        "the check found z from the cache unequal to z from the audio, or not finite, in 1 of"
        f" {checked(small)} segments"
    )


def test_a_z_that_is_not_finite_fails_the_job_without_a_flag_and_writes_no_score(
    small: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("score_zeroshot")
    second = checked(small) + 2  # the second clip after the check
    replace_z(
        module, monkeypatch, "cached_score", lambda call, z: math.inf if call == second else z
    )
    info, out = failed(small, monkeypatch, capsys, module=module)
    message = (
        "the language-model watch flagged 0 parts and 1 scores are not finite: no score is written"
    )
    assert info["error"]["message"].startswith(message)
    assert f"FAILED: StepError: {message}" in out
    assert info["results"]["lm_watch"]["flagged"] == []
    assert info["results"]["non_finite_z"] == 1
    assert "z" not in info["results"]


def test_a_row_of_the_check_sample_scored_otherwise_later_fails_the_job_and_writes_no_score(
    small: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("score_zeroshot")
    n = checked(small)
    first = samples_of(small)["x0"][0]
    # Its call in the full pass, which scores x0 first, by segment.
    call = n + list(cache_manifest(small, "x0")["segment"]).index(first) + 1
    replace_z(
        module,
        monkeypatch,
        "cached_score",
        lambda number, z: math.nextafter(z, math.inf) if number == call else z,
    )
    info, out = failed(small, monkeypatch, capsys, module=module)
    message = (
        f"1 of the check sample's {n} rows differ from their z from the audio: no score is written"
    )
    assert info["error"]["message"] == message
    assert f"FAILED: StepError: {message}" in out
    check = info["results"]["check"]["views"]
    assert all(result["z_equal"] == result["segments"] for result in check.values())


def test_every_row_of_the_check_sample_is_compared_with_its_z_from_the_audio(
    small: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("score_zeroshot")
    n = checked(small)
    view = VIEWS[-1]
    last = samples_of(small)[view][-1]  # the last row of the check sample that is compared
    # Its call in the full pass: after the check and every segment of the views before it.
    before = sum(len(cache_manifest(small, other)) for other in VIEWS[:-1])
    call = n + before + list(cache_manifest(small, view)["segment"]).index(last) + 1
    replace_z(
        module,
        monkeypatch,
        "cached_score",
        lambda number, z: math.nextafter(z, math.inf) if number == call else z,
    )
    info, _ = failed(small, monkeypatch, capsys, module=module)
    assert info["error"]["message"] == (
        f"1 of the check sample's {n} rows differ from their z from the audio: no score is written"
    )


def test_a_prompt_unlike_the_clips_token_ids_shows_in_the_check_and_fails_it(
    small: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("score_zeroshot")
    real = module.prompt_inputs

    def other(processor: Any, count: int) -> dict[str, Any]:  # its last prompt token changed
        prompt = real(processor, count)
        ids = prompt["input_ids"].clone()
        ids[0, -2] = ids[0, -3]
        return {"input_ids": ids, "attention_mask": prompt["attention_mask"]}

    monkeypatch.setattr(module, "prompt_inputs", other)
    info, _ = failed(small, monkeypatch, capsys, module=module)
    for result in info["results"]["check"]["views"].values():
        assert result["token_ids_equal"] == 0
        assert result["features_equal"] == result["segments"]
    assert info["error"]["message"].startswith(
        "the check found z from the cache unequal to z from the audio"
    )


def test_a_scores_file_that_reads_back_otherwise_fails_the_job(
    small: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("score_zeroshot")
    real = module.pd.read_parquet

    def changed(path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:  # as a faulty disk might
        table = real(path, *args, **kwargs)
        if Path(path).name == "scores.parquet":
            table.loc[0, "z"] = math.nextafter(float(table.loc[0, "z"]), math.inf)
        return table

    monkeypatch.setattr(module.pd, "read_parquet", changed)
    with pytest.raises(SystemExit):
        run_job(small, monkeypatch, module=module)
    out, err = capsys.readouterr()
    run = run_dir(small)
    assert_only_aggregates_leave(run / "run.json", out + err)
    info = info_of(run)
    assert (info["status"], info["error"]["message"]) == (
        "failed",
        "scores.parquet does not read back as written",
    )
    assert "outputs" not in info  # never pinned: load_zeroshot_scores wants a run that is done
    assert "z" not in info["results"]


def test_a_changed_cache_file_fails_the_job_naming_its_segment(
    small: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    manifest = cache_manifest(small, "x1")
    segment = str(manifest["segment"].iloc[-1])
    path = small.root / small.config["cache_runs"]["x1"]["run"] / manifest["file"].iloc[-1]
    data = path.read_bytes()
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))  # as if the disk had changed a byte
    info, out = failed(small, monkeypatch, capsys)
    assert info["error"] == {
        "type": "StepError",
        "segment": segment,
        "part": None,
        "message": f"segment {segment}: its cache file does not match the SHA-256 of its"
        " manifest row",
    }
    assert f"FAILED: StepError in segment {segment}: segment {segment}:" in out


def test_a_job_that_slurm_stops_is_left_failed_naming_its_segment(
    small: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("score_zeroshot")
    real = module.read_view_clip
    calls: list[Any] = []

    def stopped(folder: Path, row: Any) -> Any:  # what stop_on_sigterm's handler raises
        calls.append(row.segment)
        if len(calls) == 2:
            raise SystemExit(STOPPED)
        return real(folder, row)

    monkeypatch.setattr(module, "read_view_clip", stopped)
    info, out = failed(small, monkeypatch, capsys, module=module)
    assert f"FAILED: SystemExit in segment {calls[-1]}: {STOPPED}" in out
    assert (info["error"]["message"], info["error"]["segment"]) == (STOPPED, calls[-1])
    assert (run_dir(small) / "errors.log").is_file()


# The embargo ------------------------------------------------------------------------------------


def edit_wordcheck_run(
    change: Callable[[dict[str, Any]], Any],
) -> Callable[[WordcheckWorkspace], None]:
    def plant(workspace: WordcheckWorkspace) -> None:
        path = workspace.root / WORDCHECK_RUN / "run.json"
        info = json.loads(path.read_text(encoding="utf-8"))
        change(info)
        path.write_text(json.dumps(info), encoding="utf-8")

    return plant


def unpin_wordcheck_run(workspace: WordcheckWorkspace) -> None:
    workspace.config.pop("wordcheck_run")
    workspace.write_config()


def change_the_transcripts(workspace: WordcheckWorkspace) -> None:
    table = pd.DataFrame({"segment": ["300_001"], "view": ["rev"], "words": [40]})
    table.to_parquet(workspace.root / WORDCHECK_RUN / "transcripts.parquet", index=False)


def other_clips_of(view: str) -> Callable[[WordcheckWorkspace], None]:
    def plant(workspace: WordcheckWorkspace) -> None:
        views = {
            name: {"file": pin["run"], "sha256": pin["sha256"]}
            for name, pin in workspace.config["view_runs"].items()
        }
        views[view]["sha256"] = "0" * 64
        pin_wordcheck_run(workspace, views=views)

    return plant


def verdict(**given: str | None) -> Callable[[WordcheckWorkspace], None]:
    return lambda workspace: pin_wordcheck_run(workspace, verdicts=dict(given))


def pin_outside_the_workspace(workspace: WordcheckWorkspace) -> None:
    workspace.config["wordcheck_run"]["run"] = "../" + WORDCHECK_RUN
    workspace.write_config()


RUN_NAME = WORDCHECK_RUN
EMBARGO: dict[str, tuple[Callable[[WordcheckWorkspace], None], str]] = {
    "the pin missing": (
        unpin_wordcheck_run,
        "no word-check run is pinned (wordcheck_run): no test-fold score is computed until the"
        " checks of x1 are finished",
    ),
    "a changed file": (
        change_the_transcripts,
        f"the pinned word-check run {RUN_NAME}: its transcripts.parquet does not match the pin"
        " (wordcheck_run)",
    ),
    "a run not done": (
        edit_wordcheck_run(lambda info: info.update(status="failed")),
        f"{RUN_NAME} did not finish: its run.json does not say done",
    ),
    "the backup mode": (
        edit_wordcheck_run(lambda info: info.update(mode={"name": "views", "views": ["rev"]})),
        f"{RUN_NAME} is not a run of the word check in its default mode",
    ),
    "the rerun mode": (
        edit_wordcheck_run(
            lambda info: info.update(mode={"name": "rerun", "first_run": "derived/wordcheck/run-1"})
        ),
        f"{RUN_NAME} is not a run of the word check in its default mode",
    ),
    "another step": (
        edit_wordcheck_run(lambda info: info.update(step="build_caches")),
        f"{RUN_NAME} is not a run of the word check in its default mode",
    ),
    "no run.json": (
        lambda workspace: (workspace.root / WORDCHECK_RUN / "run.json").unlink(),
        f"{RUN_NAME} is not a run of the word check in its default mode",
    ),
    "a run.json of other transcripts": (
        edit_wordcheck_run(lambda info: info["outputs"].update({"transcripts.parquet": "0" * 64})),
        f"{RUN_NAME}: its run.json lists another transcripts.parquet than the pin",
    ),
    "other clips of x1": (
        other_clips_of("x1"),
        f"{RUN_NAME} checked other clips of x1 than the pinned ones (view_runs.x1), from which"
        " the caches are built",
    ),
    "other clips of rev": (
        other_clips_of("rev"),
        f"{RUN_NAME} checked other clips of rev than the pinned ones (view_runs.rev)",
    ),
    "x1 words FAIL": (
        verdict(x1_words=FAIL),
        f"{RUN_NAME}: the words of x1 say FAIL, not PASS; the checks of x1 are finished only"
        " after the re-check",
    ),
    "no x1 words verdict": (
        verdict(x1_words=None),
        f"{RUN_NAME}: the words of x1 say None, not PASS",
    ),
    "no x1 voice verdict": (
        verdict(x1_voice=None),
        f"{RUN_NAME} holds no verdict on the voice of x1 (None)",
    ),
    "no F4 verdict": (verdict(f4=None), f"{RUN_NAME}: check F4 says None, not PASS"),
    "F4 waits for the rerun": (
        verdict(f4=WAITS),
        f"{RUN_NAME}: check F4 says WAITS FOR THE RERUN, not PASS",
    ),
    "F4 FAIL": (verdict(f4=FAIL), f"{RUN_NAME}: check F4 says FAIL, not PASS"),
    "a pin outside the workspace": (
        pin_outside_the_workspace,
        f"../{WORDCHECK_RUN} is not a folder inside the workspace",
    ),
}


@pytest.mark.parametrize("change", list(EMBARGO))
def test_the_embargo_refuses_unless_the_checks_of_x1_are_finished(
    uncached: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    change: str,
) -> None:
    plant, message = EMBARGO[change]
    plant(uncached)
    model = Model()
    info, out = failed(uncached, monkeypatch, capsys, model=model)
    assert f"FAILED: StepError: {message}" in out
    assert info["error"]["message"].startswith(message)
    assert model.calls == []  # refused before the model loads
    assert "embargo" not in info


@pytest.mark.parametrize("voice", [PASS, FAIL])
def test_the_embargo_lifts_with_x1_words_pass_f4_pass_and_either_voice_verdict(
    uncached: WordcheckWorkspace, voice: str
) -> None:
    pin_wordcheck_run(uncached, verdicts={"x1_voice": voice})
    module = load_script("score_zeroshot")
    workspace = Workspace(uncached.root.resolve())
    reading = module.embargo(workspace, uncached.config)
    assert reading["verdicts"] == {"f4": PASS, "x1_words": PASS, "x1_voice": voice}
    assert reading["run"] == WORDCHECK_RUN
    assert reading["sha256"] == uncached.config["wordcheck_run"]["sha256"]


def test_a_lifted_embargo_still_needs_the_pinned_caches(
    uncached: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    model = Model()
    info, out = failed(uncached, monkeypatch, capsys, model=model)
    assert "FAILED: StepError: no cache run of x0 is pinned (cache_runs)" in out
    assert info["embargo"]["verdicts"]["x1_voice"] == FAIL
    assert model.calls == []


# Refusals of the caches and the machine ----------------------------------------------------------


@pytest.mark.parametrize("view", ["x0", "x2"])
def test_the_job_refuses_while_a_cache_check_flagged_the_language_model(
    workspace: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    view: str,
) -> None:
    # x2 is never scored, but its check watched the language model before any score.
    path = workspace.root / CHECK_RUN / "run.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    flagged = ["model.language_model.layers.1/output"]
    info["results"]["runs"][f"cache/{view}/run-{CACHE_JOBS[view]}"]["lm_watch"]["flagged"] = flagged
    path.write_text(json.dumps(info), encoding="utf-8")
    model = Model()
    record, out = failed(workspace, monkeypatch, capsys, model=model)
    message = (
        f"the checks of the caches flagged the language model on the longest clips of {view}: no"
        " score is computed until the fallback of the pilot protocol (the flagged layers in fp32)"
        " is in place"
    )
    assert record["error"]["message"] == message
    assert f"FAILED: StepError: {message}" in out
    assert f"\ncache check of {view}, flagged: model.language_model.layers.1/output\n" in out
    assert record["inputs"]["cache_checks"][view] == {"run": CHECK_RUN, "lm_flagged": flagged}
    assert model.calls == []  # refused before the model loads


def test_the_job_refuses_without_the_checked_cache_of_x2(
    workspace: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    workspace.config["cache_runs"].pop("x2")
    workspace.write_config()
    model = Model()
    info, out = failed(workspace, monkeypatch, capsys, model=model)
    assert "FAILED: StepError: no cache run of x2 is pinned (cache_runs)" in out
    assert info["embargo"]["verdicts"]["f4"] == PASS  # the embargo lifted; the caches did not
    assert model.calls == []


def test_the_job_refuses_the_cache_of_any_scored_view_built_with_other_numeric_settings(
    workspace: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    # x1's cache and a check of its own, both from a job of other CPUs: they agree with each
    # other, but not with this job, which reads x0 first.
    run = f"cache/x1/run-{CACHE_JOBS['x1']}"
    other = "derived/cache_check/run-805"
    shutil.copytree(workspace.root / CHECK_RUN, workspace.root / other)
    for folder in (run, other):
        path = workspace.root / folder / "run.json"
        info = json.loads(path.read_text(encoding="utf-8"))
        info["settings"]["numerics"]["omp_num_threads"] = "8"
        path.write_text(json.dumps(info), encoding="utf-8")
    workspace.config["cache_runs"]["x1"]["verified_by"] = other
    workspace.write_config()
    model = Model()
    record, out = failed(workspace, monkeypatch, capsys, model=model)
    message = f"{run} was built with other numeric settings than this job's"
    assert record["error"]["message"].startswith(message)
    assert f"FAILED: StepError: {message}" in out
    assert record["inputs"]["cache_checks"]["x1"] == {"run": other, "lm_flagged": []}
    assert model.calls == []


def test_the_job_refuses_caches_built_with_other_numeric_settings(
    workspace: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("score_zeroshot")
    model = Model()
    patch_job(module, workspace, monkeypatch, model)
    monkeypatch.setenv("SLURM_JOB_ID", JOB)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")  # a job of other CPUs than the cache jobs
    monkeypatch.setattr(sys, "argv", ["score_zeroshot.py"])
    with pytest.raises(SystemExit):
        module.main()
    message = (
        f"cache/x0/run-{CACHE_JOBS['x0']} was built with other numeric settings than this job's"
    )
    assert f"FAILED: StepError: {message}" in capsys.readouterr().out
    assert model.calls == []


def test_the_job_refuses_caches_of_another_width_than_the_model(
    workspace: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    run = workspace.root / workspace.config["cache_runs"]["rev"]["run"]
    info = info_of(run)
    info["settings"]["width"] = 2 * WIDTH
    (run / "run.json").write_text(json.dumps(info), encoding="utf-8")
    model = Model()
    record, _ = failed(workspace, monkeypatch, capsys, model=model)
    message = f"cache/rev/run-{CACHE_JOBS['rev']} holds outputs of width {2 * WIDTH}, not {WIDTH}"
    assert record["error"]["message"] == message
    assert model.calls == ["load"]  # refused before any clip is scored


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("job", "submit this as a job"),
        ("offline", "HF_HUB_OFFLINE is not 1"),
        ("dirty", "the checkout has uncommitted changes"),
        ("gpu", "no CUDA GPU"),
        ("argument", "usage: score_zeroshot.py (no arguments"),
        ("training", "the rules of the training code are not dated yet ("),
        ("pending", "the repeatability mode of training is not set yet ("),
    ],
)
def test_the_job_refuses_to_run_outside_a_clean_offline_gpu_job(
    workspace: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    change: str,
    message: str,
) -> None:
    module = load_script("score_zeroshot")
    model = Model()
    patch_job(module, workspace, monkeypatch, model)
    monkeypatch.setenv("SLURM_JOB_ID", JOB)
    monkeypatch.setattr(sys, "argv", ["score_zeroshot.py"])
    if change == "job":
        monkeypatch.delenv("SLURM_JOB_ID")
    elif change == "offline":
        monkeypatch.delenv("HF_HUB_OFFLINE")
    elif change == "dirty":
        monkeypatch.setattr(module, "git_state", lambda: {"commit": "c" * 40, "dirty": True})
    elif change == "gpu":
        monkeypatch.setattr(module, "gpu_facts", lambda: {"gpu": None, "cuda": None})
    elif change == "training":  # step 8's first commit has not dated them
        monkeypatch.setattr(module, "TRAINING", workspace.root / "no" / "train.yaml")
    elif change == "pending":  # the commit of the timing run has not set the mode yet
        write_training(workspace, "pending")
    else:
        monkeypatch.setattr(sys, "argv", ["score_zeroshot.py", "--view", "x0"])
    with pytest.raises(SystemExit):
        module.main()
    assert f"FAILED: StepError: {message}" in capsys.readouterr().out
    assert not (workspace.root / "runs").exists()
    assert model.calls == []


@pytest.mark.parametrize("mode", ["strict", "warn_only"])
def test_either_mode_that_the_timing_run_can_set_lets_the_job_past_its_guards(
    uncached: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    mode: str,
) -> None:
    module = load_script("score_zeroshot")
    model = Model()
    patch_job(module, uncached, monkeypatch, model)
    write_training(uncached, mode)
    monkeypatch.setenv("SLURM_JOB_ID", JOB)
    monkeypatch.setattr(sys, "argv", ["score_zeroshot.py"])
    with pytest.raises(SystemExit):
        module.main()
    # It made its run folder and stopped only at the caches, which this workspace lacks.
    assert "FAILED: StepError: no cache run of x0 is pinned (cache_runs)" in capsys.readouterr().out
    assert info_of(run_dir(uncached))["status"] == "failed"
    assert model.calls == []


def test_the_job_never_writes_into_an_existing_run_folder(
    workspace: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    first = run_dir(workspace)
    first.mkdir(parents=True)
    (first / "run.json").write_text('{"status": "done"}\n', encoding="utf-8")
    model = Model()
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, model=model)
    assert "FAILED: FileExistsError" in capsys.readouterr().out
    assert (first / "run.json").read_text(encoding="utf-8") == '{"status": "done"}\n'
    assert sorted(path.name for path in first.iterdir()) == ["run.json"]
    assert model.calls == []


def test_the_job_waits_for_the_training_settings_of_the_repository() -> None:
    # Step 8's first commit creates the file when it dates the rules of the training code.
    training = load_script("score_zeroshot").TRAINING
    assert training.as_posix() == "configs/pilot/train.yaml"
    assert not training.is_absolute()


def test_the_run_is_named_by_study_arm_and_utc_time_in_any_time_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time
    from datetime import UTC, datetime

    module = load_script("score_zeroshot")
    monkeypatch.setenv("TZ", "Asia/Kathmandu")  # UTC+05:45
    time.tzset()
    try:
        before = datetime.now(UTC).replace(microsecond=0)
        name = module.run_name()
        after = datetime.now(UTC)
    finally:
        monkeypatch.undo()
        time.tzset()
    assert re.fullmatch(r"pilot-zeroshot-\d{8}-\d{6}", name)
    stamp = datetime.strptime(name, "pilot-zeroshot-%Y%m%d-%H%M%S").replace(tzinfo=UTC)
    assert before <= stamp <= after


# The job file ------------------------------------------------------------------------------------


def test_the_job_file_asks_for_one_gpu_and_the_cpus_of_the_cache_jobs() -> None:
    text = (REPO / "scripts" / "slurm" / "score_zeroshot.sbatch").read_text(encoding="utf-8")
    for line in (
        "#SBATCH --job-name=lbw-score-zeroshot",
        "#SBATCH --partition=gpu",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=32G",
        "#SBATCH --time=02:00:00",
    ):
        assert f"{line}\n" in text, line
    # common.sh sets OMP_NUM_THREADS to the job's CPUs, which the numeric settings record:
    # they must be those of the cache jobs, whose caches the job reads.
    config = yaml.safe_load((REPO / "configs" / "pilot" / "data.yaml").read_text(encoding="utf-8"))
    caches = (REPO / "scripts" / "slurm" / "build_caches.sbatch").read_text(encoding="utf-8")
    cpus = re.compile(r"^#SBATCH --cpus-per-task=(\d+)$", flags=re.MULTILINE)
    mine, theirs = cpus.search(text), cpus.search(caches)
    assert mine is not None
    assert theirs is not None
    assert (
        int(mine.group(1)) == int(theirs.group(1)) == CacheSettings.from_config(config).cpu_threads
    )
    assert "set -euo pipefail\nsource scripts/slurm/common.sh\n" in text
    assert "nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv\n" in (
        text
    )
    assert text.endswith('python scripts/score_zeroshot.py "$@"\n')
    sbatch = [line for line in text.splitlines() if line.startswith("#SBATCH")]
    first_command = text.splitlines().index("set -euo pipefail")
    assert all(text.splitlines().index(line) < first_command for line in sbatch)
