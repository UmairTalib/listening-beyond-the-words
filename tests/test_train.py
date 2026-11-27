"""The training job (scripts/train.py), end to end, on the CPU.

A training workspace: the word check's synthetic workspace (wordcheck_fakes) with its labels
and partitions kept, the pilot IDs 300 to 319, five participants in each stratum of label and
gender, so that each fold holds one of each. Every participant has 8 to 12 segments, 300
included, so that every training fold gives at least 16 positive and 16 negative whole bags per
epoch (the participant with 3 segments is a case of tests/test_bags.py). One clip in x0, x2 and
rev is over 30 s for a training participant of round 1, fold 1 and for a test participant of it.
The caches of the four clip versions are built and checked by the cache job
(scripts/build_caches.py) and the zero-shot scores by the zero-shot job
(scripts/score_zeroshot.py), both on the tiny random Voxtral (voxtral_fakes) loaded on the CPU,
and pinned. The arm files are the repository's, and so is their train.yaml, with 2 epochs (the
gain-alive rule then reads epochs 1 and 2), 3 timed steps per mode and the repeatability mode
warn_only. The job trains and scores on the CPU through its own loaders, with the stand-in for
the processor. The GPU test at the end runs the job's hand-over on the real model on OMNI, with
synthetic encoder outputs.
"""

import contextlib
import dataclasses
import hashlib
import io
import itertools
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
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
import torch
import yaml

import wordcheck_fakes
from listening_beyond_words import evaluation, scoring, training, voxtral
from listening_beyond_words.bags import BagSampler, BagSettings, stream_seed, training_people
from listening_beyond_words.cache import NUMERICS
from listening_beyond_words.clips import MANIFEST_COLUMNS, load_pinned_labels, write_view
from listening_beyond_words.config import load_arm
from listening_beyond_words.loss import STEP_MEASURES
from listening_beyond_words.models import load_pins
from listening_beyond_words.runinfo import item_seed, sha256_file
from listening_beyond_words.settings import Workspace
from listening_beyond_words.wordcheck import FAIL, PASS
from listening_beyond_words.zeroshot import SCORE_COLUMNS, load_zeroshot_scores
from voxtral_fakes import patch_processor, save_checkpoint, tiny_model
from wordcheck_fakes import (
    GAP,
    WordcheckWorkspace,
    assert_only_aggregates_leave,
    clip_samples,
    info_of,
    make_wordcheck_workspace,
    write_pinned,
)
from x1_fakes import REPO, SECRET, X1_TOKENS, load_script

CONFIGS = REPO / "configs" / "pilot"
VOXTRAL = load_pins(CONFIGS / "models.yaml")["voxtral"]
# Over 30 s in x0, x2 and rev: two chunks. 303 trains in round 1, fold 1, 301 is tested there.
LONG_SEGMENTS = ("303_002", "301_002")
LONG_SAMPLES = 598_400
LONG_VIEWS = ("x0", "x2", "rev")
CACHE_JOBS = {"x0": "800", "x1": "801", "x2": "802", "rev": "803"}
CHECK_JOB = "804"
CHECK_RUN = f"derived/cache_check/run-{CHECK_JOB}"
ZEROSHOT_JOB = "900"
ZEROSHOT_RUN = "runs/pilot/pilot-zeroshot-20261201-120000"
WORDCHECK_RUN = "derived/wordcheck/run-700"
JOB = "950"
STAMP = "20261202-120000"
FAST_CACHE_CHECK = {"segments_per_view": 3, "seed": 108, "lm_watch_clips": 2}
FAST_ZEROSHOT = {
    "extends": "train.yaml",
    "arm": "zeroshot",
    "views": ["x0", "x1", "rev"],
    "check": {"segments_per_view": 2, "seed": 108},
}
EPOCHS = 2
STEPS_PER_MODE = 3
MODE = "warn_only"  # as the commit that records the timing run may set it
GPU = {
    "gpu": "Synthetic GPU",
    "gpu_capability": "7.0",
    "gpu_memory_gb": 16.0,
    "cuda": "12.6",
    "cudnn": 91002,
    "driver": "550.00",
}
CLEAN = {
    "commit": "c" * 40,
    "dirty": False,
    "describe": "cccccccc",
    "tags": [],
    "release_tags": [],
}
RELEASED = {**CLEAN, "describe": "v0.1.0", "tags": ["v0.1.0"], "release_tags": ["v0.1.0"]}
PLANTED = 12.345678
LAYERS = tuple(f"model.language_model.layers.{index}" for index in range(2))


# The training workspace ------------------------------------------------------------------------


def training_lines(session: int) -> list[str]:
    """Transcript lines as wordcheck_fakes makes them, 8 to 10 per participant: 8 to 12 segments."""
    rng = np.random.default_rng(session)
    label, _ = wordcheck_fakes.label_of(session)
    lines = []
    for _ in range(8 + session % 3):
        size = int(rng.integers(40, 47))
        tokens = [str(t) for t in rng.choice(np.array(X1_TOKENS, dtype=object), size=size)]
        tokens[int(rng.integers(size))] = SECRET
        for _ in range(2):
            tokens[int(rng.integers(size))] = wordcheck_fakes.LABEL_WORDS[label]
        lines.append(" ".join(tokens))
    return lines


def lengthen(workspace: WordcheckWorkspace, segments: tuple[str, ...]) -> None:
    """Replace some segments' clips in x0, x2 and rev by clips over 30 s with their parts."""
    for number, view in enumerate(LONG_VIEWS):
        pin = workspace.config["view_runs"][view]
        run = workspace.root / pin["run"]
        rows = pd.read_parquet(run / "manifest.parquet").to_dict("records")
        for row in rows:
            if row["segment"] not in segments:
                continue
            parts = len(row["part_starts"])
            rng = np.random.default_rng([8, number, int(row["session"])])
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


def pin_wordcheck_run(workspace: WordcheckWorkspace) -> None:
    """A finished word-check run of the default mode that lifts the embargo, pinned."""
    folder = workspace.root / WORDCHECK_RUN
    folder.mkdir(parents=True)
    table = pd.DataFrame({"segment": ["300_000"], "view": ["rev"], "words": [40]})
    sha256 = write_pinned(folder, table, "transcripts.parquet")
    verdicts = {"f4": PASS, "x0_control": PASS, "x1_words": PASS, "x1_voice": FAIL}
    pins = workspace.config["view_runs"]
    info = {
        "step": "check_views",
        "mode": {"name": "sample"},
        "status": "done",
        "inputs": {
            "views": {
                view: {"file": pin["run"], "sha256": pin["sha256"]} for view, pin in pins.items()
            }
        },
        "verdicts": {key: {"verdict": value} for key, value in verdicts.items()},
        "outputs": {"transcripts.parquet": sha256},
    }
    (folder / "run.json").write_text(json.dumps(info), encoding="utf-8")
    workspace.config["wordcheck_run"] = {"run": WORDCHECK_RUN, "sha256": sha256}
    workspace.write_config()


def configs_of(workspace: WordcheckWorkspace) -> Path:
    """The folder of the arm files and their train.yaml, beside the workspace's data.yaml."""
    return workspace.config_path.parent / "configs"


def write_training(workspace: WordcheckWorkspace, mode: str = MODE, **changes: Any) -> None:
    """The repository's train.yaml and arm files, with the tests' epochs, timing and mode."""
    folder = configs_of(workspace)
    folder.mkdir(exist_ok=True)
    config = yaml.safe_load((CONFIGS / "train.yaml").read_text(encoding="utf-8"))
    config["epochs"] = EPOCHS
    config["rules"]["alive"]["epochs"] = [1, 2]
    config["timing"]["steps_per_mode"] = STEPS_PER_MODE
    config["precision"]["deterministic"] = mode
    for path, value in changes.items():
        *blocks, key = path.split("__")
        target = config
        for block in blocks:
            target = target[block]
        target[key] = value
    (folder / "train.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    for arm in ("b0", "b1", "cpg"):
        shutil.copyfile(CONFIGS / f"{arm}.yaml", folder / f"{arm}.yaml")


def one_epoch(workspace: WordcheckWorkspace) -> None:
    """Train one epoch, for a test whose run only needs to reach its scores or its bags."""
    write_training(workspace, epochs=1, rules__alive__epochs=[1])


def copy_workspace(workspace: WordcheckWorkspace, folder: Path) -> WordcheckWorkspace:
    """A copy of the workspace and its configs in a new folder.

    The cache files are hard links, which no test changes in place; everything else is copied.
    """

    def copy(source: str, target: str) -> None:
        if source.endswith(".npy"):
            os.link(source, target)
        else:
            shutil.copy2(source, target)

    root = folder / "workspace"
    shutil.copytree(workspace.root, root, symlinks=True, copy_function=copy)
    copied = WordcheckWorkspace(
        root,
        json.loads(json.dumps(workspace.config)),
        folder / "data.yaml",
        workspace.segments,
        workspace.sample,
        workspace.audio,
        workspace.text_of,
    )
    copied.write_config()
    shutil.copytree(configs_of(workspace), configs_of(copied))
    return copied


# Running the jobs ------------------------------------------------------------------------------


@contextlib.contextmanager
def kept_state() -> Iterator[None]:
    """Put back PyTorch's numeric settings and every random state that a job sets."""
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
    python, numpy = random.getstate(), np.random.get_state()  # noqa: NPY002
    try:
        with torch.random.fork_rng(devices=[]):
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
        random.setstate(python)
        np.random.set_state(numpy)  # noqa: NPY002


@pytest.fixture(autouse=True)
def state_put_back() -> Iterator[None]:
    with kept_state():
        yield


def load_on_cpu(snapshot: Path, device: str = "cuda") -> voxtral.LoadedModel:
    """The jobs' model loader: the tiny snapshot, on the CPU."""
    assert device == "cuda"  # the jobs ask for the GPU; the tests run on the CPU
    return voxtral.load_model(snapshot, device="cpu")


def patch_machine(module: ModuleType, monkeypatch: pytest.MonkeyPatch, git: dict[str, Any]) -> None:
    """A clean offline Slurm job of four CPUs on a GPU node, for any of the jobs."""
    monkeypatch.setattr(sys, "unraisablehook", sys.unraisablehook)  # main() replaces it
    monkeypatch.setattr(module, "git_state", lambda: dict(git))
    monkeypatch.setattr(module, "stop_on_sigterm", lambda: None)
    monkeypatch.setattr(module, "gpu_facts", lambda: dict(GPU))
    patch_processor(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # As scripts/slurm/common.sh sets them in a job of four CPUs.
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def run_step_job(workspace: WordcheckWorkspace, name: str, job: str, *args: str) -> str:
    """Run the cache job or the zero-shot job on the workspace; it must end done."""
    out = io.StringIO()
    with pytest.MonkeyPatch.context() as monkeypatch, kept_state():
        module = load_script(name)
        patch_machine(module, monkeypatch, CLEAN)
        monkeypatch.setattr(module, "CONFIG", workspace.config_path)
        monkeypatch.setattr(module, "MODELS", CONFIGS / "models.yaml")
        monkeypatch.setattr(module, "load_model", load_on_cpu)
        if name == "score_zeroshot":
            path = configs_of(workspace) / "zeroshot.yaml"  # beside the train.yaml it extends
            path.write_text(yaml.safe_dump(FAST_ZEROSHOT), encoding="utf-8")
            monkeypatch.setattr(module, "ZEROSHOT", path)
            monkeypatch.setattr(module, "TRAINING", configs_of(workspace) / "train.yaml")
            monkeypatch.setattr(module, "run_name", lambda: Path(ZEROSHOT_RUN).name)
        monkeypatch.setenv("LBW_WORKSPACE", str(workspace.root))
        monkeypatch.setenv("SLURM_JOB_ID", job)
        monkeypatch.setattr(sys, "argv", [f"{name}.py", *args])
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            try:
                module.main()
            except SystemExit as stopped:
                raise AssertionError(out.getvalue()) from stopped
    return out.getvalue()


def build_caches(workspace: WordcheckWorkspace) -> None:
    """Cache the four versions with the cache job, check them with it, and pin them."""
    runs = [f"cache/{view}/run-{job}" for view, job in CACHE_JOBS.items()]
    for view, job in CACHE_JOBS.items():
        run_step_job(workspace, "build_caches", job, "--view", view)
    run_step_job(workspace, "build_caches", CHECK_JOB, "--verify", *runs)
    workspace.config["cache_runs"] = {
        view: {
            "run": run,
            "sha256": sha256_file(workspace.root / run / "manifest.parquet"),
            "verified_by": CHECK_RUN,
        }
        for view, run in zip(CACHE_JOBS, runs, strict=True)
    }
    workspace.write_config()


def score_zeroshot(workspace: WordcheckWorkspace) -> None:
    """Score x0, x1 and rev with the zero-shot job, and pin its scores."""
    run_step_job(workspace, "score_zeroshot", ZEROSHOT_JOB)
    workspace.config["zeroshot_scores"] = {
        "run": ZEROSHOT_RUN,
        "sha256": sha256_file(workspace.root / ZEROSHOT_RUN / "scores.parquet"),
    }
    workspace.write_config()


def make_training_workspace(folder: Path) -> WordcheckWorkspace:
    """The training workspace: clips, labels, partitions, caches and zero-shot scores, pinned."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(wordcheck_fakes, "wordcheck_lines", training_lines)
        workspace = make_wordcheck_workspace(folder / "workspace")
    lengthen(workspace, LONG_SEGMENTS)
    add_snapshot(workspace)
    workspace.config["encoder_cache"]["verify"] = dict(FAST_CACHE_CHECK)
    workspace.write_config()
    pin_wordcheck_run(workspace)
    write_training(workspace)
    build_caches(workspace)
    score_zeroshot(workspace)
    return workspace


def run_dir(workspace: WordcheckWorkspace, arm: str, round_number: int = 1, fold: int = 1) -> Path:
    return workspace.root / "runs" / "pilot" / f"pilot-{arm}-r{round_number}-f{fold}-{STAMP}"


def timing_dir(workspace: WordcheckWorkspace, job: str = JOB) -> Path:
    return workspace.root / "derived" / "train_timing" / f"run-{job}"


def patch_train(
    module: ModuleType,
    workspace: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    git: dict[str, Any] | None = None,
) -> None:
    """Point the train job at the workspace, a clean checkout, a GPU and the tiny model."""
    patch_machine(module, monkeypatch, git or CLEAN)
    monkeypatch.setattr(module, "CONFIG", workspace.config_path)
    monkeypatch.setattr(module, "MODELS", CONFIGS / "models.yaml")
    monkeypatch.setattr(module, "CONFIGS", configs_of(workspace))  # the arm files of the tests
    monkeypatch.setattr(
        module,
        "run_name",
        lambda arm, round_number, fold: f"pilot-{arm}-r{round_number}-f{fold}-{STAMP}",
    )
    real_build, real_load = training.build_trainable, training.load_for_scoring

    def build(snapshot: Path, lora: Any, seed: int, device: str = "cuda") -> Any:
        assert device == "cuda"  # the job asks for the GPU; the tests train on the CPU
        return real_build(snapshot, lora, seed, device="cpu")

    def load(snapshot: Path, folder: Path, device: str = "cuda") -> Any:
        assert device == "cuda"
        return real_load(snapshot, folder, device="cpu")

    monkeypatch.setattr(module, "build_trainable", build)
    monkeypatch.setattr(module, "load_for_scoring", load)
    monkeypatch.setenv("LBW_WORKSPACE", str(workspace.root))


def train_args(
    workspace: WordcheckWorkspace, arm: str, round_number: int = 1, fold: int = 1, *more: str
) -> list[str]:
    config = configs_of(workspace) / f"{arm}.yaml"
    return [
        "train.py",
        "--config",
        str(config),
        "--round",
        str(round_number),
        "--fold",
        str(fold),
        *more,
    ]


def run_train(
    workspace: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    module: ModuleType | None = None,
    git: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run the train job on the workspace, with any other environment variables given."""
    script = module or load_script("train")
    patch_train(script, workspace, monkeypatch, git)
    monkeypatch.setenv("SLURM_JOB_ID", JOB)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "argv", args)
    script.main()


@dataclass
class Finished:
    """A run of the train job that ended, and what it wrote to its two streams."""

    workspace: WordcheckWorkspace
    run: Path
    out: str
    err: str
    ended: str  # "done", or the failure's type
    calls: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def info(self) -> dict:
        return info_of(self.run)

    @property
    def streams(self) -> str:
        return self.out + self.err


def finish(
    workspace: WordcheckWorkspace,
    arm: str,
    round_number: int = 1,
    fold: int = 1,
    *more: str,
    git: dict[str, Any] | None = None,
    prepare: Callable[[ModuleType, pytest.MonkeyPatch, dict[str, list[Any]]], None] | None = None,
    env: dict[str, str] | None = None,
) -> Finished:
    """Run the train job to its end in a patch context of its own, whether it fails or not."""
    out, err = io.StringIO(), io.StringIO()
    calls: dict[str, list[Any]] = {}
    ended = "done"
    with pytest.MonkeyPatch.context() as monkeypatch, kept_state():
        module = load_script("train")
        if prepare is not None:
            prepare(module, monkeypatch, calls)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                run_train(
                    workspace,
                    monkeypatch,
                    train_args(workspace, arm, round_number, fold, *more),
                    module=module,
                    git=git,
                    env=env,
                )
            except SystemExit:
                ended = "failed"
    timing = "--timing" in more
    folder = timing_dir(workspace) if timing else run_dir(workspace, arm, round_number, fold)
    return Finished(workspace, folder, out.getvalue(), err.getvalue(), ended, calls)


def metric_lines(run: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run / "metrics.jsonl").read_text("utf-8").splitlines()]


def cache_manifest(workspace: WordcheckWorkspace, view: str) -> pd.DataFrame:
    run = workspace.config["cache_runs"][view]["run"]
    return pd.read_parquet(workspace.root / run / "manifest.parquet").sort_values(
        "segment", ignore_index=True
    )


def fold_sessions(workspace: WordcheckWorkspace, round_number: int, fold: int) -> set[int]:
    pin = workspace.config["partitions"]
    partitions = pd.read_parquet(workspace.root / pin["run"] / "partitions.parquet")
    chosen = partitions[(partitions["round"] == round_number) & (partitions["fold"] == fold)]
    return {int(session) for session in chosen["session"]}


def round_sessions(workspace: WordcheckWorkspace, round_number: int) -> set[int]:
    return set().union(*(fold_sessions(workspace, round_number, fold) for fold in range(1, 6)))


def logged(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, calls: dict[str, list[Any]]
) -> None:
    """Log the bags of every step and the arguments of the scoring, as the job passes them."""
    real_step, real_score = module.train_step, module.score_views

    def step(*args: Any) -> Any:  # train_step's arguments, by position as the job gives them
        bags, number, epoch = args[4], args[8], args[9]
        calls.setdefault("steps", []).append(
            (number, epoch, [(bag.session, bag.label, bag.segments) for bag in bags])
        )
        return real_step(*args)

    def score(*args: Any, **kwargs: Any) -> Any:
        calls.setdefault("scoring", []).append((args, kwargs))
        return real_score(*args, **kwargs)

    monkeypatch.setattr(module, "train_step", step)
    monkeypatch.setattr(module, "score_views", score)


def plant_in_output(module: Any, when: Callable[[int], bool], value: float) -> None:
    """Write one value into a module's output at its last position, on the calls ``when`` picks."""
    calls = [0]

    def hook(_module: Any, _args: Any, output: Any) -> Any:
        calls[0] += 1
        if not when(calls[0]):
            return None
        planted = output.clone()
        planted[0, -1, 5] = value
        return planted

    module.register_forward_hook(hook)


def second_layer(model: Any) -> Any:
    return model.get_base_model().model.language_model.layers[1]


# The records of the runs of CPG's round 1, fold 1 --------------------------------------------------

# A run's rule_readings that name no run where the readings were made, and no reason it cannot tell.
UNPLACED = {"made_on": None, "f5_read_on": None, "cannot_tell": None}


def cpg_record(
    finished: str,
    *,
    status: str = "done",
    reads_rules: bool = False,
    reads_f5: bool = False,
    segments: int = 16,
    f5_passes: bool | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """A run.json of a run of CPG's round 1, fold 1, as far as the readings look at it."""
    f5: dict[str, Any] = {"training_hours": 3.0}
    if f5_passes is not None:  # the run read F5
        f5 |= {"fourth_epoch": False, "hours": 3.0, "max_hours": 4.0, "passes": f5_passes}
    return {
        "status": status,
        "finished_utc": f"2026-11-26T{finished}+00:00",
        "allow_dirty": allow_dirty,
        "results": {
            "sampler": {"max_segments_per_epoch": segments},
            "rule_readings": {"reads_f2": True, "reads_rules": reads_rules, "reads_f5": reads_f5},
            "rules": {"f5": f5},
        },
    }


def write_record(root: Path, stamp: str, record: dict[str, Any] | str | None) -> str:
    """A run folder of CPG's round 1, fold 1 with this run.json (a text as it is; None: none).

    Returns the folder's name relative to the workspace, as run.json names it.
    """
    folder = root / "runs" / "pilot" / f"pilot-cpg-r1-f1-{stamp}"
    folder.mkdir(parents=True)
    if isinstance(record, str):
        (folder / "run.json").write_text(record, encoding="utf-8")
    elif record is not None:
        (folder / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return folder.relative_to(root).as_posix()


def readings_at(
    module: ModuleType,
    root: Path,
    *,
    segments: int = 16,
    dirty: bool = False,
    released: bool = False,
) -> dict[str, Any]:
    """The readings of a new run of CPG's round 1, fold 1 in the workspace, from the records."""
    earlier = module.earlier_readings(Workspace(root))
    return module.rule_readings(
        "cpg", 1, 1, earlier, dirty=dirty, released=released, segments=segments
    )


def cpg_metric_lines() -> list[dict[str, Any]]:
    """A CPG run's metrics.jsonl: 3 epochs of 4 steps, every other step's bags active.

    So the gain-alive share of epochs 2 and 3 is 0.5 (F2 passes, m stays 0.3), and the CE
    part is flat (the 4-epoch rule does not fire).
    """
    lines = []
    for step in range(12):
        line: dict[str, Any] = {
            "step": step,
            "epoch": step // 4 + 1,
            "seconds": 1.0,
            "lr": 1.0e-4,
            "adapter_lr": 2.0e-5,
            "loss": 1.2,
            "ce_x0": 0.7,
            "ce_x1": 0.6,
            "ce_x2": 0.3,
            "ce_part": 1.0,
            "hinge": 0.2,
            "lambda_g": 1.0,
            "scale": 65_536.0,
            "next_scale": 65_536.0,
            "skipped": False,
            "grad_norm": 2.0,
            "clipped": False,
            "calls": 14,
        }
        line |= dict.fromkeys(STEP_MEASURES, 0.5)
        line |= {f"bags_{group}": 1 for group in training.GROUPS}
        line |= {f"active_{group}": step % 2 for group in training.GROUPS}
        line |= {"sigma0_at_floor": 0, "sigma1_at_floor": 0}
        lines.append(line)
    return lines


def finish_cpg_run(
    root: Path,
    stamp: str,
    *,
    fold: int = 1,
    segments: int = 16,
    hours: float = 3.0,
    git: dict[str, Any] | None = None,
    allow_dirty: bool = False,
) -> tuple[dict[str, Any], str]:
    """Stage 8 of a CPG run of round 1 in the workspace, on planted metrics (cpg_metric_lines).

    The run trained 3 epochs (configs/pilot/cpg.yaml) with ``segments`` per participant and
    epoch, for ``hours``. Returns its run.json and its log lines.
    """
    module = load_script("train")
    folder = root / "runs" / "pilot" / f"pilot-cpg-r1-f{fold}-{stamp}"
    folder.mkdir(parents=True)
    lines = cpg_metric_lines()
    text = "".join(json.dumps(line) + "\n" for line in lines)
    (folder / "metrics.jsonl").write_text(text, encoding="utf-8")
    info: dict[str, Any] = {
        "status": "running",
        "git": dict(git or CLEAN),
        "allow_dirty": allow_dirty,
        "results": {"hours": {"training": hours}, "sampler": {"max_segments_per_epoch": segments}},
        "outputs": {},
    }
    args = module.Arguments(CONFIGS / "cpg.yaml", 1, fold, False, allow_dirty, False)
    arm = load_arm(CONFIGS / "cpg.yaml")
    run = module.Run(args, JOB, arm, Workspace(root), {}, VOXTRAL, folder, info, time.monotonic())
    run.save()  # the run.json of a running run, which never counts itself
    log = io.StringIO()
    handler = logging.StreamHandler(log)
    module.logger.addHandler(handler)
    module.logger.setLevel(logging.INFO)
    try:
        module.finish(run, lines)
    finally:
        module.logger.removeHandler(handler)
    return info_of(folder), log.getvalue()


# The workspace and the finished runs --------------------------------------------------------------


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> WordcheckWorkspace:
    """The training workspace, with its caches and zero-shot scores built and pinned."""
    return make_training_workspace(tmp_path_factory.mktemp("built"))


@pytest.fixture
def workspace(built: WordcheckWorkspace, tmp_path: Path) -> WordcheckWorkspace:
    return copy_workspace(built, tmp_path)


@pytest.fixture(scope="module")
def finished(
    built: WordcheckWorkspace, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Finished]:
    """B0, B1 and CPG of round 1, fold 1, each run to its end on its own copy of the workspace.

    B1's round 1 is no first run: it needs a checkout at a release tag.
    """
    return {
        arm: finish(
            copy_workspace(built, tmp_path_factory.mktemp(arm)),
            arm,
            git=RELEASED if arm == "b1" else CLEAN,
            prepare=logged,
        )
        for arm in ("b0", "b1", "cpg")
    }


@pytest.fixture(scope="module")
def timed(built: WordcheckWorkspace, tmp_path_factory: pytest.TempPathFactory) -> Finished:
    """The timing run, in a workspace without zero-shot scores, its mode still pending."""
    workspace = copy_workspace(built, tmp_path_factory.mktemp("timed"))
    shutil.rmtree(workspace.root / "runs")
    workspace.config.pop("zeroshot_scores")
    workspace.write_config()
    write_training(workspace, mode="pending")
    return finish(workspace, "cpg", 1, 1, "--timing", prepare=logged)


# A finished run -----------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("arm", ["b0", "b1", "cpg"])
def test_each_arm_trains_two_epochs_then_scores_its_test_fold_and_ends_done(
    finished: dict[str, Finished], arm: str
) -> None:
    run = finished[arm]
    assert run.ended == "done", run.out
    info = run.info
    assert (info["status"], info["step"], info["arm"], info["round"], info["fold"]) == (
        "done",
        "train",
        arm,
        1,
        1,
    )
    assert info["timing"] is False
    assert info["run_id"] == f"pilot-{arm}-r1-f1-{STAMP}"
    assert sorted(path.name for path in run.run.iterdir()) == [
        "adapter",
        "metrics.jsonl",
        "run.json",
        "run.log",
        "scores.parquet",
    ]
    assert sorted(path.name for path in (run.run / "adapter").iterdir()) == [
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    assert info["outputs"] == {
        "adapter/adapter_model.safetensors": sha256_file(
            run.run / "adapter" / "adapter_model.safetensors"
        ),
        "scores.parquet": sha256_file(run.run / "scores.parquet"),
        "metrics.jsonl": sha256_file(run.run / "metrics.jsonl"),
    }
    assert info["model"]["adapter_sha256"] == info["outputs"]["adapter/adapter_model.safetensors"]
    steps = info["results"]["steps"]
    assert (steps["epochs"], steps["per_epoch"], steps["total"], steps["warmup"]) == (2, 8, 16, 1)
    assert steps.get("lambda_g_ramp") == (None if arm == "b0" else 4)
    lines = metric_lines(run.run)
    assert [line["step"] for line in lines] == list(range(16))  # one line per step
    assert [line["epoch"] for line in lines] == [1] * 8 + [2] * 8
    assert info["results"]["training"]["steps_done"] == 16
    assert ("hinge" in lines[0]) == (arm != "b0")  # B0 logs no measure of the gain
    assert info["results"]["sampler"]["max_segments_per_epoch"] == 16
    assert re.search(
        r"done in \d+\.\d min; trained with at most 16 segments per participant per epoch",
        run.err,
    )
    assert "FAILED" not in run.out


@pytest.mark.slow
@pytest.mark.parametrize("arm", ["b0", "b1", "cpg"])
def test_each_run_scores_every_test_segment_of_x0_and_rev_once_and_no_training_person(
    finished: dict[str, Finished], arm: str
) -> None:
    run = finished[arm]
    scores = pd.read_parquet(run.run / "scores.parquet")
    assert tuple(scores.columns) == SCORE_COLUMNS
    assert str(scores["z"].dtype) == "float64"
    assert list(scores["view"].unique()) == ["x0", "rev"]
    test = fold_sessions(run.workspace, 1, 1)
    for view in ("x0", "rev"):
        rows = scores[scores["view"] == view]
        manifest = cache_manifest(run.workspace, view)
        expected = manifest[manifest["session"].isin(test)]
        assert list(rows["segment"]) == list(expected["segment"])  # once each, by segment
        assert list(rows["session"]) == list(expected["session"])
    assert set(scores["session"]) == test
    assert np.isfinite(scores["z"]).all()
    assert {LONG_SEGMENTS[1]} <= set(scores["segment"])  # a two-chunk clip of the test fold
    z = info_of(run.run)["results"]["z"]
    for view in ("x0", "rev"):
        values = scores.loc[scores["view"] == view, "z"].to_numpy()
        assert z[view]["count"] == len(values)
        assert z[view]["mean"] == float(values.mean())
        assert z[view]["all_finite"] is True


@pytest.mark.slow
def test_the_three_arms_of_a_fold_run_train_the_same_bags_from_the_same_start(
    finished: dict[str, Finished], built: WordcheckWorkspace
) -> None:
    steps = {arm: run.calls["steps"] for arm, run in finished.items()}
    assert steps["b0"] == steps["b1"] == steps["cpg"]
    assert [number for number, _, _ in steps["cpg"]] == list(range(16))
    starts = {run.info["model"]["trainable"]["start_sha256"] for run in finished.values()}
    assert len(starts) == 1  # the same LoRA values, from the fold-run's torch seed
    seeds = [run.info["seeds"] for run in finished.values()]
    assert seeds[0] == seeds[1] == seeds[2]
    training_people = round_sessions(built, 1) - fold_sessions(built, 1, 1)
    labels = pd.read_parquet(built.root / built.config["labels"]["run"] / "labels.parquet")
    manifest = cache_manifest(built, "x0")
    for _, _, bags in steps["cpg"]:
        assert [label for _, label, _ in bags] == [1, 0, 1, 0]  # 2 + 2, positive first
        for session, label, segments in bags:
            assert session in training_people
            assert label == int(labels.loc[session, "label"])
            assert len(set(segments)) == 4
            assert set(manifest.loc[manifest["segment"].isin(segments), "session"]) == {session}
    trained = {session for _, _, bags in steps["cpg"] for session, _, _ in bags}
    assert trained == training_people
    assert LONG_SEGMENTS[0] in {s for _, _, bags in steps["cpg"] for _, _, seg in bags for s in seg}


@pytest.mark.slow
def test_run_json_records_the_config_seeds_inputs_numerics_and_model(
    finished: dict[str, Finished],
) -> None:
    run = finished["cpg"]
    info, workspace = run.info, run.workspace
    folder = configs_of(workspace)
    arm = load_arm(folder / "cpg.yaml")
    assert info["config"] == {
        "file": str(folder / "cpg.yaml"),
        "sha256": sha256_file(folder / "cpg.yaml"),
        "extends": {
            "file": str(folder / "train.yaml"),
            "sha256": sha256_file(folder / "train.yaml"),
        },
        "resolved": arm.resolved,
        "resolved_sha256": arm.sha256,
    }
    assert info["data_config"] == {
        "file": str(workspace.config_path),
        "sha256": sha256_file(workspace.config_path),
    }
    assert info["settings"]["bags"] == workspace.config["bags"]
    assert info["git"] == CLEAN
    assert (info["allow_dirty"], info["release"]) == (
        False,
        "no v0.1.x tag yet: one of the first runs (B0's round 1, CPG's round 1, fold 1)",
    )
    assert info["seeds"] == {
        "round": 101,
        "everything": item_seed(101, "torch", 1),
        "torch": {
            "stream": 'runinfo.item_seed(101, "torch", 1)',
            "seed": item_seed(101, "torch", 1),
        },
        "bags": {"stream": 'runinfo.item_seed(101, "bags", 1)', "seed": item_seed(101, "bags", 1)},
    }
    inputs = info["inputs"]
    assert inputs["labels"] == workspace.config["labels"]
    assert inputs["partitions"] == workspace.config["partitions"]
    assert inputs["caches"] == workspace.config["cache_runs"]
    assert inputs["cache_checks"] == {
        view: {"run": CHECK_RUN, "lm_flagged": []} for view in ("x0", "x1", "x2", "rev")
    }
    assert inputs["zeroshot_scores"] == workspace.config["zeroshot_scores"]
    assert inputs["wordcheck_run"]["run"] == WORDCHECK_RUN
    assert info["embargo"]["verdicts"] == {"f4": PASS, "x1_words": PASS, "x1_voice": FAIL}
    assert (inputs["voxtral"]["repo_id"], inputs["voxtral"]["revision"]) == (
        VOXTRAL.repo_id,
        VOXTRAL.revision,
    )
    numerics = info["settings"]["numerics"]
    cached = {
        **NUMERICS,
        "torch_threads": 4,
        "omp_num_threads": "4",
        "cublas_workspace_config": ":4096:8",
    }
    assert numerics["caches"] == numerics["training"] == numerics["scoring"] == cached
    assert numerics["attention"]["implementation"]
    model = info["model"]
    trainable = model["trainable"]
    assert (trainable["lora_modules"], trainable["layers"]) == (7 * len(LAYERS), len(LAYERS))
    assert "base_model_name_or_path" not in trainable["peft_config"]
    assert trainable["torch_seed"] == item_seed(101, "torch", 1)
    assert model["optimiser"]["total_steps"] == 16
    assert model["optimiser"]["foreach"] is True
    assert info["environment"]["packages"]["peft"]
    assert {key: info["environment"][key] for key in GPU} == GPU
    assert info["results"]["released"] == {"allocated_bytes": None, "reserved_bytes": None}
    assert info["results"]["hours"]["training"] > 0
    assert info["results"]["hours"]["scoring"] > 0


@pytest.mark.slow
def test_the_8_segment_rerun_of_cpg_round_1_fold_1_reads_f5_alone_and_the_first_readings_stand(
    workspace: WordcheckWorkspace,
) -> None:
    # The first CPG run of round 1, fold 1 that finished read the two rules and F5, which
    # failed at 16 segments: its rerun at 8 segments reads F5 alone, on its own hours, and
    # leaves the readings of m and of the epochs standing. A run that read the rules earlier
    # with --allow-dirty and a failed run do not count.
    root = workspace.root
    first = write_record(
        root, "20261126-100000", cpg_record("11:00:00", reads_rules=True, f5_passes=False)
    )
    write_record(
        root, "20261126-090000", cpg_record("10:00:00", reads_rules=True, allow_dirty=True)
    )
    write_record(root, "20261126-080000", {"status": "failed"})
    workspace.config["bags"]["max_segments_per_epoch"] = 8  # the fallback of check F5
    workspace.write_config()
    run = finish(workspace, "cpg", prepare=logged)
    assert run.ended == "done", run.out
    assert run.info["results"]["sampler"]["max_segments_per_epoch"] == 8
    assert run.info["results"]["rule_readings"] == {
        "reads_f2": True,
        "reads_rules": False,
        "reads_f5": True,
        **UNPLACED,
        "made_on": first,
    }
    rules = run.info["results"]["rules"]
    assert "f2_passes" in rules["alive"]  # F2 reads every CPG run
    assert "margin_if_read" not in rules["alive"]
    assert "fires" not in rules["falling"]
    f5 = rules["f5"]
    assert f5["fourth_epoch"] is False
    assert f5["hours"] == f5["training_hours"] == run.info["results"]["hours"]["training"]
    assert f"; the gain-alive rule was read once, on {first}, and that reading stands" in run.err
    assert f"(the rule was read once, on {first}, and that reading stands)" in run.err
    assert (
        f" h of training (this run's own hours, at 8 segments after F5 failed at 16 on {first}),"
        " at most 4 h: " in run.err
    )
    assert "the rule would set m to" not in run.err
    assert "fold 1 only" not in run.err  # the rules and F5 read this fold-run


@pytest.mark.slow
def test_the_rules_are_read_on_the_metrics_of_the_run(finished: dict[str, Finished]) -> None:
    for arm, run in finished.items():  # each of round 1, fold 1
        info = run.info
        readings = {"reads_f2": arm == "cpg", "reads_rules": arm == "cpg", "reads_f5": False}
        assert info["results"]["rule_readings"] == {**readings, **UNPLACED}
        measures = training.rule_measures(
            metric_lines(run.run),
            load_arm(configs_of(run.workspace) / f"{arm}.yaml").shared,
            info["results"]["hours"]["training"] if arm != "b0" else None,
            **readings,
        )
        assert info["results"]["rules"] == json.loads(json.dumps(measures))
        assert (measures["alive"] is None) == (arm == "b0")
        assert (measures["f5"] is None) == (arm == "b0")  # F5 times a three-view run
        assert f"F1: {'passes' if measures['f1']['passes'] else 'FAILS'};" in run.err
        assert f"4-epoch rule on epoch {EPOCHS}:" in run.err
        if arm == "cpg":  # CPG's round 1, fold 1: F2, the two rules and F5 read it
            assert re.search(r"F2 (passes|FAILS) on this CPG run", run.err)
            assert "; the rule would set m to " in run.err
            assert "fires" in measures["falling"]
            assert re.search(r"F5: \d+\.\d{3} h of training.*: (passes|FAILS)", run.err)
        else:  # no verdict of F2, the two rules or F5 on B0 and B1
            assert not re.search(r"F2 (passes|FAILS)", run.err)
            assert "the rule would set m" not in run.err
            assert "fires" not in measures["falling"]
            assert "(the rule reads CPG's round 1, fold 1 only)" in run.err
            assert "F5: " not in run.err
        if arm == "b1":
            assert measures["f5"] == {"training_hours": info["results"]["hours"]["training"]}
            assert "(check F5 reads CPG's round 1, fold 1 only)" in run.err
            assert "(check F2 reads CPG runs only)" in run.err
        log = (run.run / "run.log").read_text("utf-8").splitlines()
        assert log
        assert all(line in run.err for line in log)  # run.log holds the job's log lines


@pytest.mark.slow
def test_each_metrics_line_holds_its_arms_keys_its_loss_and_the_statistics_of_the_step_before(
    finished: dict[str, Finished],
) -> None:
    base = set(training.X0_MEASURES) | {
        "step",
        "epoch",
        "seconds",
        "lr",
        "adapter_lr",
        "loss",
        "ce_x0",
        "ce_part",
        "scale",
        "next_scale",
        "skipped",
        "grad_norm",
        "clipped",
        "calls",
    }
    gain_keys = base | {"ce_x1", "ce_x2", "hinge", "lambda_g", *STEP_MEASURES}
    for arm, run in finished.items():
        lines = metric_lines(run.run)
        for line in lines:  # exactly the logged keys: no number per bag or clip
            assert set(line) == (base if arm == "b0" else gain_keys), arm
        if arm == "b0":
            continue
        start = run.info["results"]["start_stats"]
        for line in lines:  # the loss holds the gain at the step's λG
            expected = line["ce_part"] + line["lambda_g"] * line["hinge"]
            assert line["loss"] == pytest.approx(expected, rel=1e-6, abs=1e-9)
        ramp = [0.0, 0.25, 0.5, 0.75, 1.0] if arm == "cpg" else [0.0] * 5  # T 16: a ramp of 4
        assert [line["lambda_g"] for line in lines[:5]] == ramp
        assert all(line["hinge"] > 0 for line in lines)
        for view, mean in (("0", "z0_bar_mean"), ("1", "z1_bar_mean")):
            assert lines[0][f"mu{view}"] == start[f"x{view}"]["mu"]  # started from the zero-shot
            for before, after in itertools.pairwise(lines):  # handed on, step by step
                assert after[f"mu{view}"] == pytest.approx(
                    0.9 * before[f"mu{view}"] + 0.1 * before[mean], rel=1e-12, abs=1e-15
                )
            assert len({line[f"mu{view}"] for line in lines}) > 1


@pytest.mark.slow
def test_the_round_2_run_trains_round_2s_other_folds_and_scores_its_own_fold(
    built: WordcheckWorkspace, tmp_path: Path
) -> None:
    workspace = copy_workspace(built, tmp_path)
    one_epoch(workspace)  # every training participant gives a bag that trains in epoch 1
    run = finish(workspace, "b0", 2, 2, git=RELEASED, prepare=logged)
    assert run.ended == "done", run.out
    assert run.info["release"] == "at the release tag v0.1.0"
    trained = {session for _, _, bags in run.calls["steps"] for session, _, _ in bags}
    assert trained == round_sessions(workspace, 2) - fold_sessions(workspace, 2, 2)
    scored = pd.read_parquet(run.run / "scores.parquet")
    assert set(scored["session"]) == fold_sessions(workspace, 2, 2)
    assert run.info["seeds"]["bags"]["seed"] == stream_seed(102, 2)


@pytest.mark.slow
def test_the_running_statistics_start_from_the_zero_shot_bag_means_of_the_first_8_steps(
    workspace: WordcheckWorkspace,
) -> None:
    # Plant zero-shot scores whose bag means differ by view and segment, pinned again.
    folder = workspace.root / ZEROSHOT_RUN
    scores = pd.read_parquet(folder / "scores.parquet")
    offset = {"x0": 0.0, "x1": 0.37, "rev": -0.5}
    scores["z"] = [
        offset[view] + 0.013 * (int(segment[:3]) - 300) + 0.11 * int(segment[4:])
        for view, segment in zip(scores["view"], scores["segment"], strict=True)
    ]
    scores.to_parquet(folder / "scores.parquet", index=False)
    info = json.loads((folder / "run.json").read_text("utf-8"))
    info["outputs"]["scores.parquet"] = sha256_file(folder / "scores.parquet")
    (folder / "run.json").write_text(json.dumps(info), "utf-8")
    workspace.config["zeroshot_scores"]["sha256"] = info["outputs"]["scores.parquet"]
    workspace.write_config()

    def stop_after_the_start(
        module: ModuleType, monkeypatch: pytest.MonkeyPatch, calls: dict[str, list[Any]]
    ) -> None:
        def stop(*_args: Any) -> None:
            raise module.StepError("stopped by the test once the statistics started")

        monkeypatch.setattr(module, "make_optimiser", stop)

    run = finish(workspace, "cpg", prepare=stop_after_the_start)
    assert run.info["error"]["message"] == "stopped by the test once the statistics started"
    arm = load_arm(configs_of(workspace) / "cpg.yaml")
    people = training_people(
        pd.read_parquet(
            workspace.root / workspace.config["partitions"]["run"] / "partitions.parquet"
        ),
        1,
        1,
    )
    labels = pd.read_parquet(workspace.root / workspace.config["labels"]["run"] / "labels.parquet")
    manifest = cache_manifest(workspace, "x0")
    sampler = BagSampler(
        manifest.loc[manifest["session"].isin(people), ["segment", "session"]],
        {session: int(labels.loc[session, "label"]) for session in people},
        BagSettings.from_config(workspace.config),
        arm.shared.sampler,
        stream_seed(101, 1),
    )
    bags = sampler.start_bags(8)
    assert len(bags) == 32
    z = {(row.view, row.segment): row.z for row in scores.itertuples(index=False)}
    recorded = run.info["results"]["start_stats"]
    assert recorded["source"] == "the zero-shot scores of the first 8 steps' bags"
    for view in ("x0", "x1"):
        means = np.array(
            [np.mean([z[(view, segment)] for segment in bag.segments]) for bag in bags]
        )
        assert recorded[view]["mu"] == pytest.approx(float(means.mean()), rel=1e-12)
        assert recorded[view]["raw_sigma"] == pytest.approx(float(means.std()), rel=1e-9)
        assert recorded[view]["floor"] == pytest.approx(max(0.25 * float(means.std()), 0.1))


# Faults in training and in scoring ---------------------------------------------------------------


def plant_in_training(
    value: float, call: int
) -> Callable[[ModuleType, pytest.MonkeyPatch, dict[str, list[Any]]], None]:
    """Plant a value in the second layer's output on one forward call of the training model."""

    def prepare(
        module: ModuleType, monkeypatch: pytest.MonkeyPatch, calls: dict[str, list[Any]]
    ) -> None:
        real = training.build_trainable

        def build(*args: Any, **kwargs: Any) -> Any:
            trainable = real(*args, **kwargs)  # the hook goes in before the job's watch
            plant_in_output(second_layer(trainable.model), lambda number: number == call, value)
            return trainable

        monkeypatch.setattr(training, "build_trainable", build)

    return prepare


@pytest.mark.slow
@pytest.mark.parametrize(
    ("value", "message"),
    [
        (64_512.0, "the overflow watch flagged 1 ("),
        (math.nan, "scores are not finite; check F1 fails"),
    ],
)
def test_a_value_planted_in_training_ends_the_run_failed_at_its_step_with_no_weights(
    workspace: WordcheckWorkspace, value: float, message: str
) -> None:
    run = finish(workspace, "b0", prepare=plant_in_training(value, call=11))
    assert run.ended == "failed"
    info = run.info
    assert info["status"] == "failed"
    done = info["results"]["training"]["steps_done"]
    assert 1 <= done < 16
    assert info["error"]["message"].startswith(f"step {done}")
    assert message in info["error"]["message"]
    assert f"FAILED: StepError: step {done}" in run.out
    assert len(metric_lines(run.run)) == done  # the steps before it, logged
    flagged = info["results"]["lm_watch_training"]["flagged"]
    assert "model.language_model.layers.1/output" in flagged
    assert not (run.run / "adapter").exists()
    assert not (run.run / "scores.parquet").exists()
    assert_only_aggregates_leave(run.run / "run.json", run.streams)


def plant_in_scoring(
    value: float, call: int
) -> Callable[[ModuleType, pytest.MonkeyPatch, dict[str, list[Any]]], None]:
    """Plant a value in the second layer's output on one call of the scoring model."""

    def prepare(
        module: ModuleType, monkeypatch: pytest.MonkeyPatch, calls: dict[str, list[Any]]
    ) -> None:
        real = training.load_for_scoring

        def load(*args: Any, **kwargs: Any) -> Any:
            loaded = real(*args, **kwargs)  # the hook goes in before the scoring watch
            plant_in_output(second_layer(loaded.model), lambda number: number == call, value)
            return loaded

        monkeypatch.setattr(training, "load_for_scoring", load)

    return prepare


@pytest.mark.slow
@pytest.mark.parametrize(("value", "non_finite"), [(64_512.0, False), (math.nan, True)])
def test_a_value_planted_while_the_trained_model_scores_writes_no_score(
    workspace: WordcheckWorkspace, value: float, non_finite: bool
) -> None:
    one_epoch(workspace)
    run = finish(workspace, "b0", prepare=plant_in_scoring(value, call=5))
    assert run.ended == "failed"
    info = run.info
    assert info["error"]["message"].startswith("the scoring watch flagged ")
    assert "no score is written; the run goes again from the start" in info["error"]["message"]
    results = info["results"]
    assert "model.language_model.layers.1/output" in results["lm_watch_scoring"]["flagged"]
    assert (results["non_finite_z"] > 0) == non_finite
    assert (run.run / "adapter" / "adapter_model.safetensors").is_file()  # trained and saved
    assert not (run.run / "scores.parquet").exists()
    assert "z" not in results
    assert_only_aggregates_leave(run.run / "run.json", run.streams)


@pytest.mark.slow
def test_an_embargo_that_changed_since_the_start_stops_the_run_before_any_score(
    workspace: WordcheckWorkspace,
) -> None:
    def prepare(
        module: ModuleType, monkeypatch: pytest.MonkeyPatch, calls: dict[str, list[Any]]
    ) -> None:
        real = module.embargo

        def changed(*args: Any) -> Any:
            reading = real(*args)
            calls.setdefault("embargo", []).append(1)
            if len(calls["embargo"]) > 1:  # the second reading, before the scores
                reading = json.loads(json.dumps(reading))
                reading["verdicts"]["x1_voice"] = PASS
            return reading

        monkeypatch.setattr(module, "embargo", changed)

    one_epoch(workspace)
    run = finish(workspace, "b0", prepare=prepare)
    assert run.ended == "failed"
    message = "the embargo's reading changed since the run started: no score"
    assert run.info["error"]["message"] == message
    assert f"FAILED: StepError: {message}" in run.out
    assert len(run.calls["embargo"]) == 2  # read at the start and again before scoring
    assert (run.run / "adapter" / "adapter_model.safetensors").is_file()  # trained and saved
    assert not (run.run / "scores.parquet").exists()
    assert "z" not in run.info["results"]


@pytest.mark.slow
def test_the_job_refuses_caches_of_the_four_versions_that_hold_different_segments(
    workspace: WordcheckWorkspace,
) -> None:
    def prepare(
        module: ModuleType, monkeypatch: pytest.MonkeyPatch, calls: dict[str, list[Any]]
    ) -> None:
        real = module.pinned_cache

        def short(*args: Any) -> Any:
            cache = real(*args)
            if cache.view != "rev":
                return cache
            return dataclasses.replace(cache, manifest=cache.manifest.iloc[1:])  # one row less

        def build(*_args: Any, **_kwargs: Any) -> None:
            calls.setdefault("build", []).append(1)
            raise AssertionError("the model was built")

        monkeypatch.setattr(module, "pinned_cache", short)
        monkeypatch.setattr(training, "build_trainable", build)

    run = finish(workspace, "b0", prepare=prepare)
    message = "the pinned caches of the four clip versions hold different segments"
    assert run.ended == "failed"
    assert run.info["error"]["message"] == message
    assert f"FAILED: StepError: {message}" in run.out
    assert "build" not in run.calls  # refused before the model is built


# What may leave OMNI ------------------------------------------------------------------------------

# A whole number on its own: not part of a word, a decimal, a time's milliseconds or a
# number with thousands separators; one followed by a unit is a time or a size.
WHOLE_NUMBER = re.compile(r"(?<![\w.,])(\d+)(?![\w.]|,\d)")
UNIT = re.compile(r"\s?(?:s|ms|min|h|GB|GiB|MiB|bytes|%)(?!\w)")


def session_ids_in(text: str, sessions: set[int]) -> list[str]:
    """The places where one of the workspace's session IDs stands as a whole number.

    Lines that start with ``FAILED:`` may name a segment (engineering conventions,
    section 10) and are left out.
    """
    found = []
    for line in text.splitlines():
        if line.startswith("FAILED: "):
            continue
        for match in WHOLE_NUMBER.finditer(line):
            if int(match.group(1)) in sessions and not UNIT.match(line, match.end()):
                found.append(line[max(0, match.start() - 40) : match.end() + 10])
    return found


def run_texts(run: Finished) -> str:
    """What a run shows: its streams, run.log, metrics.jsonl and run.json without its error."""
    info = {key: value for key, value in run.info.items() if key != "error"}
    texts = [run.streams, json.dumps(info, indent=1)]
    for name in ("run.log", "metrics.jsonl"):
        if (run.run / name).is_file():
            texts.append((run.run / name).read_text("utf-8"))
    return "\n".join(texts)


@pytest.mark.slow
@pytest.mark.parametrize("arm", ["b0", "b1", "cpg"])
def test_only_aggregates_leave_a_run_and_its_metrics_hold_no_id(
    finished: dict[str, Finished], arm: str
) -> None:
    run = finished[arm]
    assert_only_aggregates_leave(run.run / "run.json", run.streams)
    metrics = (run.run / "metrics.jsonl").read_text("utf-8")
    assert not wordcheck_fakes.SEGMENT_ID.search(metrics)
    assert not wordcheck_fakes.PARTICIPANT_PAIR.search(metrics)
    for line in metric_lines(run.run):
        assert not {"session", "segment", "segments", "bag", "bags"} & set(line)
        assert all(not isinstance(value, list | dict) for value in line.values())
    text = (run.run / "run.json").read_text("utf-8")
    for root in {str(run.workspace.root), str(run.workspace.root.resolve())}:
        assert root not in text  # no workspace path, the snapshot's included
    # No session ID of the workspace as a whole number anywhere, however it is worded.
    assert session_ids_in(run_texts(run), round_sessions(run.workspace, 1)) == []


@pytest.mark.slow
def test_the_check_of_what_leaves_fails_on_an_id_planted_on_a_stream(
    finished: dict[str, Finished],
) -> None:
    run = finished["cpg"]
    assert_only_aggregates_leave(run.run / "run.json", run.streams)  # passes as it is
    for planted in (
        f"bag of {LONG_SEGMENTS[0]}\n",
        "loss by participant {303: 0.71}\n",
        "smoke bag: 4 segments of participant 303\n",
    ):
        with pytest.raises(AssertionError):
            assert_only_aggregates_leave(run.run / "run.json", run.streams + planted)
    sessions = round_sessions(run.workspace, 1)
    assert sorted(sessions) == list(range(300, 320))
    assert session_ids_in(run_texts(run), sessions) == []  # passes as it is
    shown = "sampler: 12 training participants from 300 (6 positive, 6 negative)\n"
    assert session_ids_in(run_texts(run) + shown, sessions)  # a bare ID, worded any way
    for kept in ("took 300 s\n", "read 0.305 s\n", "1,303 bytes\n", "FAILED: StepError: 303\n"):
        assert session_ids_in(kept, sessions) == []


@pytest.mark.slow
def test_a_planted_test_z_reaches_no_stream_no_log_and_no_run_json(
    workspace: WordcheckWorkspace,
) -> None:
    test = sorted(fold_sessions(workspace, 1, 1))
    segment = f"{test[0]}_003"

    def prepare(
        module: ModuleType, monkeypatch: pytest.MonkeyPatch, calls: dict[str, list[Any]]
    ) -> None:
        real_read, real_score = scoring.read_file, scoring.cached_score
        last: dict[str, Path] = {}

        def read(path: Path, row: Any, width: int) -> Any:
            last["path"] = path
            return real_read(path, row, width)

        def score(*args: Any) -> float:
            z = real_score(*args)
            path = last["path"]  # cache/<view>/run-<job>/<session>/<segment>.npy
            return PLANTED if (path.stem, path.parents[2].name) == (segment, "x0") else z

        monkeypatch.setattr(scoring, "read_file", read)
        monkeypatch.setattr(scoring, "cached_score", score)

    one_epoch(workspace)
    run = finish(workspace, "b0", prepare=prepare)
    assert run.ended == "done", run.out
    scores = pd.read_parquet(run.run / "scores.parquet")
    row = scores[(scores["view"] == "x0") & (scores["segment"] == segment)]
    assert row["z"].tolist() == [PLANTED]  # the plant took
    shown = (
        run.streams
        + (run.run / "run.json").read_text("utf-8")
        + (run.run / "run.log").read_text("utf-8")
        + (run.run / "metrics.jsonl").read_text("utf-8")
    )
    for spelling in ("12.345678", "12.34568", "12.3457", PLANTED.hex()):
        assert spelling not in shown, spelling
    assert_only_aggregates_leave(run.run / "run.json", run.streams)


@pytest.mark.slow
def test_the_scoring_gets_the_test_segments_and_no_label(finished: dict[str, Finished]) -> None:
    for run in finished.values():
        ((args, kwargs),) = run.calls["scoring"]  # one call, once the weights are saved
        test = fold_sessions(run.workspace, 1, 1)
        segments = kwargs["segments"]
        assert list(segments) == ["x0", "rev"]
        for view, chosen in segments.items():
            manifest = cache_manifest(run.workspace, view)
            assert list(chosen) == list(manifest.loc[manifest["session"].isin(test), "segment"])
        caches = args[1]
        assert list(caches) == ["x0", "rev"]
        for value in [*args, *kwargs.values()]:
            assert not (isinstance(value, pd.DataFrame) and "label" in value.columns)
            assert not (isinstance(value, dict) and "label" in value)
        for cache in caches.values():
            assert "label" not in cache.manifest.columns


@pytest.mark.slow
def test_the_analysis_job_reads_each_finished_run_as_a_kept_run_with_its_test_folds_scores(
    finished: dict[str, Finished],
) -> None:
    """The contract between what the train job writes and what the analysis job reads.

    Each finished run, pinned by its files, counts as a kept run (B0 and CPG of round 1 as
    first runs at their commit, B1 at the release tag of its clone), and its scores hold
    exactly its test fold's segments of x0 and rev, as the zero-shot scores name them.
    """
    evaluate = load_script("evaluate")
    counts_by = {
        "b0": "the first runs' commit ccccccc",
        "b1": "the newest release tag of its clone, v0.1.0",
        "cpg": "the first runs' commit ccccccc",
    }
    for arm, run in finished.items():
        assert run.ended == "done", run.out
        workspace = Workspace(run.workspace.root)
        config = run.workspace.config
        pin = evaluation.RunPin(
            run.run.relative_to(workspace.root).as_posix(),
            sha256_file(run.run / "scores.parquet"),
            sha256_file(run.run / "run.json"),
        )
        kept = evaluate.pinned_run(workspace, (arm, 1, 1), pin, config, CLEAN["commit"])
        assert kept.counts_by == counts_by[arm]
        # The earlier of two runs of a fold-run is told by the time the run finished.
        assert evaluation.run_order(kept.record, kept.run) == (kept.record["finished_utc"], pin.run)
        zeroshot = load_zeroshot_scores(workspace, config)
        test = fold_sessions(run.workspace, 1, 1)
        segments = {}
        for view in ("x0", "rev"):
            rows = zeroshot[zeroshot["view"] == view]
            segments[view] = {
                str(segment): int(session)
                for segment, session in zip(rows["segment"], rows["session"], strict=True)
                if int(session) in test
            }
        labelled = {int(session) for session in load_pinned_labels(workspace, config).index}
        scores = evaluate.read_scores(workspace, kept, segments, labelled)
        assert list(scores) == ["x0", "rev"]
        for values in scores.values():
            assert list(values.index) == sorted(test)
        if arm == "cpg":  # check F2 reads every CPG run, F5 the first of round 1, fold 1
            reading = evaluation.f2_reading({kept.run: kept.record})
            assert reading["pooled"]["bags"] == kept.record["results"]["rules"]["alive"]["bags"]
            assert evaluate.f5_reading({("cpg", 1, 1): kept})["read_by"] == "reads_rules"


# The timing run --------------------------------------------------------------------------------


@pytest.mark.slow
def test_the_timing_run_needs_no_zero_shot_score_keeps_no_weights_and_writes_no_score(
    timed: Finished,
) -> None:
    assert timed.ended == "done", timed.out
    info = timed.info
    assert (info["status"], info["arm"], info["round"], info["fold"], info["timing"]) == (
        "done",
        "cpg",
        1,
        1,
        True,
    )
    assert info["run_id"] == f"run-{JOB}"
    assert sorted(path.name for path in timed.run.iterdir()) == ["run.json"]
    assert not (timed.workspace.root / "runs").exists()  # no run folder, weights or scores
    assert "zeroshot_scores" not in info["inputs"]
    assert "embargo" not in info  # it scores nothing
    assert "scoring" not in timed.calls
    assert info["config"]["resolved"]["precision"]["deterministic"] == "pending"
    assert_only_aggregates_leave(timed.run / "run.json", timed.streams)


@pytest.mark.slow
def test_the_timing_run_checks_the_smoke_bags_gradients_at_loss_scale_1(timed: Finished) -> None:
    results = timed.info["results"]
    smoke = results["smoke_bag"]
    assert smoke["segments"] == 4
    assert smoke["two_chunk"] is True  # the longest x0 clip of a training participant
    assert smoke["views"] == ["x1", "x0", "x2"]
    assert smoke["calls"] == 5  # x1 in one call; x0 and x2 with the two-chunk clip in its own
    assert (smoke["lambda_g"], smoke["loss_scale"]) == (1.0, 1.0)
    assert smoke["gain_term"] is True  # the gain's backward pass ran too
    assert smoke["with_gradient"] == smoke["trained_tensors"] == 4 * 7 + 2
    assert smoke["all_finite"] is True
    assert 0 < smoke["largest_norm"] <= smoke["joint_norm"]
    assert results["start_stats"]["source"] == "the stand-in start of the timing run"
    assert (results["start_stats"]["x0"]["mu"], results["start_stats"]["x0"]["sigma"]) == (0.0, 1.0)
    assert "smoke bag: 4 segments of one training participant" in timed.err
    assert "(gain term in the loss)" in timed.err
    # The smoke bag is one participant's: only its aggregates leave, never whose it is.
    assert set(smoke) == {
        "segments",
        "two_chunk",
        "views",
        "calls",
        "lambda_g",
        "gain_term",
        "loss_scale",
        "trained_tensors",
        "with_gradient",
        "all_finite",
        "largest_norm",
        "joint_norm",
    }
    workspace = timed.workspace
    people = training_people(
        pd.read_parquet(
            workspace.root / workspace.config["partitions"]["run"] / "partitions.parquet"
        ),
        1,
        1,
    )
    labels = pd.read_parquet(workspace.root / workspace.config["labels"]["run"] / "labels.parquet")
    module = load_script("train")
    bag = module.choose_smoke_bag(
        cache_manifest(workspace, "x0"),
        people,
        {session: int(labels.loc[session, "label"]) for session in people},
        4,
    )
    assert bag.segments[0] == LONG_SEGMENTS[0]  # the two-chunk clip of a training participant
    assert session_ids_in(run_texts(timed), {bag.session}) == []
    assert session_ids_in(run_texts(timed), round_sessions(workspace, 1)) == []


@pytest.mark.slow
def test_the_timing_run_times_the_first_steps_in_each_mode_and_projects_check_f5(
    timed: Finished,
) -> None:
    info = timed.info
    results = info["results"]
    steps = [number for number, _, _ in timed.calls["steps"]]
    assert steps == list(range(2 * STEPS_PER_MODE))  # the sampler's first steps, in order
    modes = results["modes"]
    for mode in ("warn_only", "strict"):
        assert modes[mode]["steps"] == STEPS_PER_MODE
        assert modes[mode]["outcome"] == "ran"
        assert modes[mode]["median_seconds"] > 0
    numerics = info["settings"]["numerics"]
    assert numerics["warn_only"]["deterministic_warn_only"] is True
    assert numerics["strict"]["deterministic_warn_only"] is False
    assert results["strict_runs"] is True
    ratio = modes["strict"]["median_seconds"] / modes["warn_only"]["median_seconds"]
    assert results["mode_by_rule"] == ("strict" if ratio <= 1.1 else "warn_only")
    projection = results["projection"]
    assert (projection["total_steps"], projection["epochs"], projection["max_hours"]) == (
        16,
        2,
        4.0,
    )
    for mode, hours in projection["hours"].items():
        expected = (results["load_seconds_from_start"] + 16 * modes[mode]["median_seconds"]) / 3600
        assert hours == pytest.approx(expected, abs=0.1 / 3600)
    assert projection["hours_in_mode"] == projection["hours"][results["mode_by_rule"]]
    assert projection["fallback_applies"] is False
    assert "the 8-segment fallback does not apply" in timed.err


def test_the_smoke_bag_is_the_longest_x0_clip_of_a_training_participant_with_4_segments() -> None:
    module = load_script("train")
    manifest = pd.DataFrame(
        {
            "segment": ["9001_000", "9001_001", "9001_002"]
            + [f"9002_{index:03d}" for index in range(6)]
            + ["9003_000", "9003_001", "9003_002", "9003_003"],
            "session": [9001] * 3 + [9002] * 6 + [9003] * 4,
            # 9001 holds the longest clip but has 3 segments; 9003 is not a training person.
            "samples": [900, 100, 100, 300, 500, 200, 500, 100, 400, 990, 10, 10, 10],
        }
    )
    bag = module.choose_smoke_bag(manifest, (9001, 9002), {9001: 1, 9002: 0}, 4)
    # Longest first; of two clips of 500 samples the lower segment ID goes first.
    assert (bag.session, bag.label, bag.segments) == (
        9002,
        0,
        ("9002_001", "9002_003", "9002_005", "9002_000"),
    )
    with pytest.raises(module.StepError, match="no training participant has the 4 segments"):
        module.choose_smoke_bag(manifest, (9001,), {9001: 1}, 4)
    # A participant with exactly 4 segments may give the smoke bag: "at least 4".
    four = manifest[manifest["session"] != 9002].copy()
    four.loc[four["segment"] == "9001_002", "session"] = 9004  # 9001 keeps 2 segments
    four = pd.concat(
        [
            four,
            pd.DataFrame(
                {
                    "segment": ["9005_000", "9005_001", "9005_002", "9005_003"],
                    "session": [9005] * 4,
                    "samples": [700, 20, 30, 40],
                }
            ),
        ],
        ignore_index=True,
    )
    bag = module.choose_smoke_bag(four, (9001, 9004, 9005), {9001: 1, 9004: 0, 9005: 1}, 4)
    assert (bag.session, bag.segments) == (9005, ("9005_000", "9005_003", "9005_002", "9005_001"))


@pytest.mark.parametrize(
    ("strict_runs", "strict", "warn", "mode"),
    [
        (True, 11.0, 10.0, "strict"),  # exactly 10% longer: at most 10%
        (True, 11.001, 10.0, "warn_only"),  # just over
        (True, 9.0, 10.0, "strict"),
        (True, 22.0, 20.0, "strict"),  # compared as exact fractions: 11/10 is no exact float
        (False, None, 10.0, "warn_only"),  # strict mode raised
        (False, 9.0, 10.0, "warn_only"),
    ],
)
def test_the_timing_runs_rule_gives_strict_mode_only_if_it_runs_at_most_10_percent_slower(
    strict_runs: bool, strict: float | None, warn: float, mode: str
) -> None:
    module = load_script("train")
    assert module.repeatability_mode(strict_runs, strict, warn, 0.1) == mode


def test_the_rules_read_every_cpg_run_for_check_f2_and_cpg_round_1_fold_1_for_the_rest() -> None:
    module = load_script("train")
    none = {"reads_f2": False, "reads_rules": False, "reads_f5": False, **UNPLACED}
    f2_only = {**none, "reads_f2": True}
    # A run of CPG's round 1, fold 1 with no record of another: F2, the two rules and F5.
    assert module.rule_readings("cpg", 1, 1, module.Earlier()) == {**f2_only, "reads_rules": True}
    with pytest.raises(ValueError, match="needs the record of the others"):
        module.rule_readings("cpg", 1, 1)
    # Other fold-runs never read the rules or F5, whatever the records of round 1, fold 1 hold.
    earlier = module.Earlier(made_on="runs/pilot/a", rules_segments=16, rules_f5_passes=False)
    for round_number, fold in ((1, 2), (2, 1)):
        assert module.rule_readings("cpg", round_number, fold) == f2_only
        assert module.rule_readings("cpg", round_number, fold, earlier, segments=8) == f2_only
    for arm in ("b0", "b1"):
        for round_number, fold in ((1, 1), (2, 3)):
            assert module.rule_readings(arm, round_number, fold, earlier, segments=8) == none


def test_the_rules_run_is_the_done_clean_run_that_read_them_not_merely_the_first_to_finish(
    tmp_path: Path,
) -> None:
    module = load_script("train")
    root = tmp_path / "workspace"
    workspace = Workspace(root)
    this = root / write_record(root, "20261127-080000", cpg_record("08:00:00", status="running"))
    assert module.earlier_readings(workspace, this) == module.Earlier()
    write_record(root, "20261126-090000", cpg_record("10:00:00", status="failed", reads_rules=True))
    write_record(
        root, "20261126-091000", cpg_record("10:01:00", reads_rules=True, allow_dirty=True)
    )
    write_record(root, "20261126-095000", None)  # no run.json
    for name in ("pilot-cpg-r1-f2", "pilot-cpg-r2-f1", "pilot-b1-r1-f1"):  # other fold-runs
        folder = root / "runs" / "pilot" / f"{name}-20261126-090000"
        folder.mkdir()
        record = cpg_record("09:00:00", reads_rules=True, reads_f5=True)
        (folder / "run.json").write_text(json.dumps(record), encoding="utf-8")
    # Done first and clean, but it read no rule (as a run at a release tag would).
    write_record(root, "20261126-100000", cpg_record("10:30:00"))
    assert module.earlier_readings(workspace, this) == module.Earlier()
    late = write_record(root, "20261126-110000", cpg_record("13:00:00", reads_rules=True))
    first = write_record(
        root,
        "20261126-120000",
        cpg_record("12:00:00", reads_rules=True, segments=8, f5_passes=True),
    )
    expected = module.Earlier(made_on=first, rules_segments=8, rules_f5_passes=True)
    assert module.earlier_readings(workspace, this) == expected  # the first to finish
    assert module.earlier_readings(workspace, root / first).made_on == late  # never itself
    # Of two that finished in the same second, the first by name.
    tie = write_record(root, "20261126-115000", cpg_record("12:00:00", reads_rules=True))
    assert module.earlier_readings(workspace, this).made_on == tie


def test_the_rules_and_f5_are_read_once_and_f5_alone_again_only_on_the_8_segment_rerun(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    logs = []
    # The first run that finishes, at 16 segments: the two rules and F5, which fails.
    info, log = finish_cpg_run(root, "20261126-100000", hours=4.5)
    logs.append(log)
    first = "runs/pilot/pilot-cpg-r1-f1-20261126-100000"
    assert info["status"] == "done"
    reads = {"reads_f2": True, "reads_rules": False, "reads_f5": False}
    assert info["results"]["rule_readings"] == {**reads, "reads_rules": True, **UNPLACED}
    rules = info["results"]["rules"]
    assert (rules["alive"]["f2_passes"], rules["alive"]["margin_if_read"]) == (True, 0.3)
    assert rules["falling"]["fires"] is False
    assert rules["f5"] == {
        "training_hours": 4.5,
        "fourth_epoch": False,
        "hours": 4.5,
        "max_hours": 4.0,
        "passes": False,
    }
    assert ": F2 passes on this CPG run; the rule would set m to 0.3 (now 0.3)" in log
    assert "): does not fire" in log
    assert "F5: 4.500 h of training, at most 4 h: FAILS" in log
    # The redo at 16 segments, as after m or the epochs changed: no verdict of either.
    info, log = finish_cpg_run(root, "20261126-120000", hours=3.5)
    logs.append(log)
    later = {**reads, **UNPLACED, "made_on": first}
    assert info["results"]["rule_readings"] == later
    rules = info["results"]["rules"]
    assert rules["alive"]["f2_passes"] is True  # F2 reads every CPG run
    assert "margin_if_read" not in rules["alive"]
    assert "fires" not in rules["falling"]
    assert rules["f5"] == {"training_hours": 3.5}  # a diagnostic
    assert f"; the gain-alive rule was read once, on {first}, and that reading stands" in log
    assert f"(the rule was read once, on {first}, and that reading stands)" in log
    assert f"3.500 h (no F5 verdict here: F5 was read on {first})" in log
    # The rerun at 8 segments after F5 failed at 16: F5 alone, on its own hours.
    info, log = finish_cpg_run(root, "20261126-140000", segments=8, hours=3.5)
    logs.append(log)
    rerun = "runs/pilot/pilot-cpg-r1-f1-20261126-140000"
    assert info["results"]["rule_readings"] == {**later, "reads_f5": True}
    rules = info["results"]["rules"]
    assert "margin_if_read" not in rules["alive"]
    assert "fires" not in rules["falling"]
    assert rules["f5"] == {
        "training_hours": 3.5,
        "fourth_epoch": False,
        "hours": 3.5,
        "max_hours": 4.0,
        "passes": True,
    }
    assert (
        "F5: 3.500 h of training (this run's own hours, at 8 segments after F5 failed at 16 on"
        f" {first}), at most 4 h: passes" in log
    )
    assert f"the gain-alive rule was read once, on {first}" in log
    # A second run at 8 segments: no F5 verdict, and both readings named.
    info, log = finish_cpg_run(root, "20261126-160000", segments=8, hours=3.9)
    logs.append(log)
    assert info["results"]["rule_readings"] == {**later, "f5_read_on": rerun}
    assert info["results"]["rules"]["f5"] == {"training_hours": 3.9}
    assert (
        f"3.900 h (no F5 verdict here: F5 was read on {first} and, at 8 segments, on {rerun})"
        in log
    )
    for log in logs:  # every run is CPG's round 1, fold 1
        assert "fold 1 only" not in log
        assert "only)" not in log


@pytest.mark.parametrize(
    ("rules_segments", "f5_passes", "segments", "reads_f5"),
    [
        (16, False, 8, True),  # the rerun at 8 segments after F5 failed at 16
        (16, False, 16, False),  # F5 failed, but this run has the rules run's 16 segments
        (16, True, 16, False),  # F5 passed: no later run reads it
        (16, True, 8, False),
        (8, False, 8, False),  # the projection set 8 and F5 failed there: no fallback is left
        (8, True, 8, False),
    ],
)
@pytest.mark.parametrize("released", [False, True])  # the rerun at 8 usually follows the freeze
def test_check_f5_alone_reads_only_the_first_run_at_8_segments_after_f5_failed_at_16(
    tmp_path: Path,
    rules_segments: int,
    f5_passes: bool,
    segments: int,
    reads_f5: bool,
    released: bool,
) -> None:
    module = load_script("train")
    root = tmp_path / "workspace"
    record = cpg_record("11:00:00", reads_rules=True, segments=rules_segments, f5_passes=f5_passes)
    first = write_record(root, "20261126-100000", record)
    reads = {"reads_f2": True, "reads_rules": False, "reads_f5": reads_f5}
    expected = {**reads, **UNPLACED, "made_on": first}
    assert readings_at(module, root, segments=segments, released=released) == expected
    # A failed run, or one with --allow-dirty, that read F5 alone does not count ...
    failed = cpg_record("13:00:00", status="failed", segments=8, reads_f5=True)
    write_record(root, "20261126-120000", failed)
    dirty = cpg_record("13:10:00", segments=8, reads_f5=True, allow_dirty=True)
    write_record(root, "20261126-121000", dirty)
    assert readings_at(module, root, segments=segments, released=released) == expected
    # ... but once a run has read F5 alone, no later run does.
    f5_run = write_record(
        root, "20261126-130000", cpg_record("14:00:00", segments=8, reads_f5=True)
    )
    assert readings_at(module, root, segments=segments, released=released) == {
        **expected,
        "reads_f5": False,
        "f5_read_on": f5_run,
    }


def test_a_run_that_finished_unable_to_tell_leaves_every_later_run_unable_to_tell(
    tmp_path: Path,
) -> None:
    # A run that finished while another run's record could not be read made no reading; it
    # may have been the run due to read, so a later run, once the record is repaired, must
    # not read in its place.
    module = load_script("train")
    root = tmp_path / "workspace"
    undecided = cpg_record("11:00:00")
    undecided["results"]["rule_readings"]["cannot_tell"] = "the run.json of X cannot be read"
    first = write_record(root, "20261126-100000", undecided)
    write_record(root, "20261126-090000", cpg_record("09:30:00", status="failed"))
    reason = f"{first} finished unable to tell where the readings were made"
    unread = {"reads_f2": True, "reads_rules": False, "reads_f5": False}
    assert readings_at(module, root) == {**unread, **UNPLACED, "cannot_tell": reason}
    # The same once a rules run exists: the undecided run may have been the 8-segment rerun.
    rules = cpg_record("08:00:00", reads_rules=True, f5_passes=False)
    write_record(root, "20261126-070000", rules)
    assert readings_at(module, root, segments=8) == {**unread, **UNPLACED, "cannot_tell": reason}
    # A dirty run that finished unable to tell does not count.
    clean = tmp_path / "clean"
    dirty = cpg_record("11:00:00", allow_dirty=True)
    dirty["results"]["rule_readings"]["cannot_tell"] = "the run.json of X cannot be read"
    write_record(clean, "20261126-100000", dirty)
    assert readings_at(module, clean)["reads_rules"] is True


def test_a_dirty_run_is_never_the_rules_run_and_makes_no_reading(tmp_path: Path) -> None:
    module = load_script("train")
    root = tmp_path / "workspace"
    write_record(
        root, "20261126-090000", cpg_record("10:00:00", reads_rules=True, allow_dirty=True)
    )
    reads = {"reads_f2": True, "reads_rules": False, "reads_f5": False}
    # The dirty run's readings do not count: a clean run reads the rules, a dirty one none.
    assert readings_at(module, root) == {**reads, "reads_rules": True, **UNPLACED}
    assert readings_at(module, root, dirty=True) == {**reads, **UNPLACED}
    # After F5 failed at 16 on the rules run, a dirty run at 8 reads no F5 either.
    rules = cpg_record("11:00:00", reads_rules=True, f5_passes=False)
    first = write_record(root, "20261126-100000", rules)
    clean = readings_at(module, root, segments=8)
    assert clean == {**reads, "reads_f5": True, **UNPLACED, "made_on": first}
    assert readings_at(module, root, segments=8, dirty=True) == {**clean, "reads_f5": False}
    # A dirty run with no rules run found says why it reads nothing.
    info, log = finish_cpg_run(tmp_path / "other", "20261126-100000", allow_dirty=True)
    assert info["results"]["rule_readings"] == {**reads, **UNPLACED}
    assert "margin_if_read" not in info["results"]["rules"]["alive"]
    assert info["results"]["rules"]["f5"] == {"training_hours": 3.0}
    reason = "a run with --allow-dirty makes no reading"
    assert (
        f"; the gain-alive rule is not read here, and no run that read it is found: {reason}" in log
    )
    assert f"(no F5 verdict here, and no run that read F5 is found: {reason})" in log
    assert "fold 1 only" not in log


def test_a_run_at_a_release_tag_makes_no_first_reading_and_a_later_clean_run_does(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    info, log = finish_cpg_run(root, "20261126-090000", git=RELEASED)
    reads = {"reads_f2": True, "reads_rules": False, "reads_f5": False}
    assert info["results"]["rule_readings"] == {**reads, **UNPLACED}
    rules = info["results"]["rules"]
    assert "margin_if_read" not in rules["alive"]
    assert "fires" not in rules["falling"]
    assert rules["f5"] == {"training_hours": 3.0}
    reason = "a run at a release tag makes no first reading, as the settings froze after it"
    assert (
        f"; the gain-alive rule is not read here, and no run that read it is found: {reason}" in log
    )
    assert f"(the rule is not read here, and no run that read it is found: {reason})" in log
    assert f"(no F5 verdict here, and no run that read F5 is found: {reason})" in log
    assert "fold 1 only" not in log
    # The first clean run before the freeze reads them, though the run above finished first.
    info, _ = finish_cpg_run(root, "20261126-100000")
    assert info["results"]["rule_readings"] == {**reads, "reads_rules": True, **UNPLACED}
    made_on = readings_at(load_script("train"), root)["made_on"]
    assert made_on == "runs/pilot/pilot-cpg-r1-f1-20261126-100000"


@pytest.mark.parametrize("text", ["{not json", "[1, 2]", ""])
def test_an_unreadable_run_json_leaves_a_run_of_cpg_round_1_fold_1_unable_to_tell_and_unread(
    tmp_path: Path, text: str
) -> None:
    root = tmp_path / "workspace"
    broken = write_record(root, "20261126-090000", text)
    reason = f"the run.json of {broken} cannot be read"
    reads = {"reads_f2": True, "reads_rules": False, "reads_f5": False}
    unread = {**reads, **UNPLACED, "cannot_tell": reason}
    # No rules run is found, yet the run reads no rule: the record it cannot read may hold one.
    info, log = finish_cpg_run(root, "20261126-100000")
    assert info["status"] == "done"  # the run's own record stands
    assert info["results"]["rule_readings"] == unread
    rules = info["results"]["rules"]
    assert "margin_if_read" not in rules["alive"]
    assert "fires" not in rules["falling"]
    assert rules["f5"] == {"training_hours": 3.0}
    cannot = f"is not read here: this run cannot tell which run read it, as {reason}"
    assert f"; the gain-alive rule {cannot}" in log
    assert f"(the rule {cannot})" in log
    assert f"(no F5 verdict here: this run cannot tell which run read F5, as {reason})" in log
    assert "fold 1 only" not in log
    # Nor F5 alone, at 8 segments after the rules run's F5 failed at 16.
    rules_run = cpg_record("09:30:00", reads_rules=True, f5_passes=False)
    write_record(root, "20261126-080000", rules_run)
    info, _ = finish_cpg_run(root, "20261126-110000", segments=8)
    assert info["results"]["rule_readings"] == unread
    assert info["results"]["rules"]["f5"] == {"training_hours": 3.0}
    assert (root / broken / "run.json").read_text(encoding="utf-8") == text


def test_cpg_runs_of_other_folds_get_check_f2_alone_and_say_the_rules_read_fold_1_only(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    write_record(root, "20261126-100000", cpg_record("11:00:00", reads_rules=True, f5_passes=False))
    write_record(root, "20261126-110000", "{not json")  # read by no other fold-run
    info, log = finish_cpg_run(root, "20261126-120000", fold=2, segments=8, git=RELEASED)
    assert info["results"]["rule_readings"] == {
        "reads_f2": True,
        "reads_rules": False,
        "reads_f5": False,
        **UNPLACED,
    }
    assert info["results"]["rules"]["alive"]["f2_passes"] is True
    assert "(the rule reads CPG's round 1, fold 1 only)" in log
    assert "3.000 h (check F5 reads CPG's round 1, fold 1 only)" in log
    assert "was read once" not in log


def test_the_check_of_what_leaves_sees_a_named_id_on_a_stream_and_an_id_value_in_run_json(
    tmp_path: Path,
) -> None:
    run_json = tmp_path / "run.json"
    kept = {"results": {"smoke_bag": {"segments": 4, "all_finite": True}}, "sessions": 20}
    run_json.write_text(json.dumps(kept), encoding="utf-8")
    assert_only_aggregates_leave(run_json, "smoke bag: 4 segments of one training participant\n")
    for planted in ("smoke bag: 4 segments of participant 303\n", "sampler: session 311 left\n"):
        with pytest.raises(AssertionError):
            assert_only_aggregates_leave(run_json, planted)
    for planted_json in (
        {"results": {"smoke_bag": {"session": 303}}},
        {"results": {"people": {"sessions": [301, 302]}}},
        {"results": {"first_participant": "305"}},
    ):
        run_json.write_text(json.dumps(planted_json), encoding="utf-8")
        with pytest.raises(AssertionError):
            assert_only_aggregates_leave(run_json, "")
    run_json.write_text(json.dumps({"error": {"session": 303}}), encoding="utf-8")
    assert_only_aggregates_leave(run_json, "FAILED: StepError: session 303\n")  # a failure's line


# Refusals --------------------------------------------------------------------------------------


def patch_guards(module: ModuleType, monkeypatch: pytest.MonkeyPatch, git: dict[str, Any]) -> None:
    """The job in the empty test workspace with the repository's configs: guards only."""
    patch_machine(module, monkeypatch, git)
    monkeypatch.setattr(module, "CONFIG", CONFIGS / "data.yaml")
    monkeypatch.setattr(module, "MODELS", CONFIGS / "models.yaml")
    monkeypatch.setattr(module, "CONFIGS", CONFIGS)
    monkeypatch.setattr(
        module,
        "run_name",
        lambda arm, round_number, fold: f"pilot-{arm}-r{round_number}-f{fold}-{STAMP}",
    )
    monkeypatch.setenv("SLURM_JOB_ID", JOB)


def refused(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    args: list[str],
) -> str:
    """Run the job, which must fail; return its FAILED line."""
    monkeypatch.setattr(sys, "argv", ["train.py", *args])
    with pytest.raises(SystemExit):
        module.main()
    out = capsys.readouterr().out
    (line,) = [line for line in out.splitlines() if line.startswith("FAILED: ")]
    return line


def repository_args(arm: str, round_number: int = 1, fold: int = 1, *more: str) -> list[str]:
    return [
        "--config",
        str(CONFIGS / f"{arm}.yaml"),
        "--round",
        str(round_number),
        "--fold",
        str(fold),
        *more,
    ]


@pytest.mark.parametrize(
    ("change", "args", "message"),
    [
        ("job", repository_args("b0"), "submit this as a job"),
        ("offline", repository_args("b0"), "HF_HUB_OFFLINE is not 1"),
        ("dirty", repository_args("b0"), "the checkout has uncommitted changes"),
        ("gpu", repository_args("b0"), "no CUDA GPU"),
        (None, ["--config", str(CONFIGS / "b0.yaml"), "--round", "1"], "usage: train.py"),
        (None, repository_args("b0", 3), "usage: train.py --config"),
        (None, repository_args("b0", 1, 6), "(a round has folds 1 to 5, not 6)"),
        (None, [*repository_args("b0"), "--epochs", "4"], "usage: train.py"),
        (None, repository_args("zeroshot"), "zeroshot.yaml: arm must be one of"),
        (
            None,
            repository_args("b0", 1, 1, "--timing"),
            "the timing run of check F5 is CPG's round 1, fold 1",
        ),
        (
            None,
            repository_args("cpg", 1, 2, "--timing"),
            "the timing run of check F5 is CPG's round 1, fold 1",
        ),
    ],
)
def test_the_job_refuses_to_run_outside_a_clean_offline_gpu_job_or_with_a_wrong_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    change: str | None,
    args: list[str],
    message: str,
) -> None:
    module = load_script("train")
    patch_guards(module, monkeypatch, CLEAN)
    if change == "job":
        monkeypatch.delenv("SLURM_JOB_ID")
    elif change == "offline":
        monkeypatch.delenv("HF_HUB_OFFLINE")
    elif change == "dirty":
        monkeypatch.setattr(module, "git_state", lambda: {**CLEAN, "dirty": True})
    elif change == "gpu":
        monkeypatch.setattr(module, "gpu_facts", lambda: {"gpu": None, "cuda": None})
    line = refused(module, monkeypatch, capsys, args)
    assert line.startswith("FAILED: StepError: ")
    assert message in line
    assert not (Path(os.environ["LBW_WORKSPACE"]) / "runs").exists()  # refused before its folder


@pytest.mark.parametrize("git", [CLEAN, RELEASED])
def test_the_job_refuses_an_arm_file_outside_the_checkouts_configs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    tmp_path: Path,
    git: dict[str, Any],
) -> None:
    # An arm file with a train.yaml beside it, outside the checkout: its settings (here a
    # set mode and another margin) would escape the clean-checkout and release-tag guards.
    outside = tmp_path / "configs"
    outside.mkdir()
    config = yaml.safe_load((CONFIGS / "train.yaml").read_text(encoding="utf-8"))
    config["precision"]["deterministic"] = "warn_only"
    config["gain"]["margin"] = 0.5
    (outside / "train.yaml").write_text(yaml.safe_dump(config, sort_keys=False), "utf-8")
    shutil.copyfile(CONFIGS / "cpg.yaml", outside / "cpg.yaml")
    assert load_arm(outside / "cpg.yaml").shared.gain.margin == 0.5  # a valid arm file
    module = load_script("train")
    patch_guards(module, monkeypatch, git)
    args = ["--config", str(outside / "cpg.yaml"), "--round", "1", "--fold", "1"]
    line = refused(module, monkeypatch, capsys, args)
    assert line.startswith(f"FAILED: StepError: the arm file must be {CONFIGS}/<arm>.yaml in")
    assert not (Path(os.environ["LBW_WORKSPACE"]) / "runs").exists()  # refused before its folder
    # A symbolic link in the checkout's folder that points outside it is refused too.
    monkeypatch.setattr(module, "CONFIGS", tmp_path / "checkout")
    (tmp_path / "checkout").mkdir()
    (tmp_path / "checkout" / "cpg.yaml").symlink_to(outside / "cpg.yaml")
    args = ["--config", str(tmp_path / "checkout" / "cpg.yaml"), "--round", "1", "--fold", "1"]
    assert "the arm file must be" in refused(module, monkeypatch, capsys, args)


OLDER = {**RELEASED, "describe": "v0.1.0-3-gcccccccc", "tags": []}  # a commit off the tag
OTHER_TAG = {**RELEASED, "describe": "v0.2.0", "tags": ["v0.2.0"]}
# Two releases, the checkout at the older one or at the newer one; release numbers order
# as numbers (v0.1.10 after v0.1.2), not as text.
TWO_RELEASES = ["v0.1.0", "v0.1.1"]
AT_OLDER_RELEASE = {
    **RELEASED,
    "describe": "v0.1.0",
    "tags": ["v0.1.0"],
    "release_tags": TWO_RELEASES,
}
AT_NEWER_RELEASE = {
    **RELEASED,
    "describe": "v0.1.1",
    "tags": ["v0.1.1"],
    "release_tags": TWO_RELEASES,
}
TENTH = ["v0.1.10", "v0.1.2"]
AT_V0_1_2 = {**RELEASED, "describe": "v0.1.2", "tags": ["v0.1.2"], "release_tags": TENTH}
AT_V0_1_10 = {**RELEASED, "describe": "v0.1.10", "tags": ["v0.1.10"], "release_tags": TENTH}


@pytest.mark.parametrize(
    ("arm", "round_number", "fold", "more", "git", "allowed"),
    [
        ("b0", 1, 1, (), CLEAN, True),
        ("b0", 1, 4, (), CLEAN, True),  # B0's round 1: every fold
        ("cpg", 1, 1, (), CLEAN, True),
        ("cpg", 1, 1, ("--timing",), CLEAN, True),  # the timing run counts as CPG r1f1
        ("cpg", 1, 2, (), CLEAN, False),
        ("b1", 1, 1, (), CLEAN, False),
        ("b0", 2, 1, (), CLEAN, False),
        ("b1", 2, 5, (), RELEASED, True),
        ("b0", 1, 1, (), RELEASED, True),
        ("b0", 1, 1, (), OLDER, False),  # a rerun off the tag once one exists
        ("cpg", 1, 1, ("--timing",), OLDER, False),
        ("cpg", 1, 3, (), OTHER_TAG, False),  # a tag, but not a v0.1.x one
        ("b1", 2, 5, (), AT_OLDER_RELEASE, False),  # v0.1.1 may hold a fallback: use it
        ("b0", 1, 1, (), AT_OLDER_RELEASE, False),
        ("b1", 2, 5, (), AT_NEWER_RELEASE, True),
        ("cpg", 2, 3, (), AT_V0_1_2, False),  # v0.1.10 is the newer of the two
        ("cpg", 2, 3, (), AT_V0_1_10, True),
    ],
)
def test_until_a_release_tag_exists_only_the_first_runs_may_run_and_then_only_from_the_tag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    arm: str,
    round_number: int,
    fold: int,
    more: tuple[str, ...],
    git: dict[str, Any],
    allowed: bool,
) -> None:
    module = load_script("train")
    patch_guards(module, monkeypatch, git)
    line = refused(module, monkeypatch, capsys, repository_args(arm, round_number, fold, *more))
    if allowed:  # past the guard; the empty test workspace has no pinned labels
        assert line.startswith(
            "FAILED: StepError: the pinned labels: labels.parquet does not match"
        )
    elif len(git["release_tags"]) > 1:
        newest = "v0.1.10" if "v0.1.10" in git["release_tags"] else "v0.1.1"
        assert line.startswith(f"FAILED: StepError: the checkout is at {git['tags'][0]}, not at")
        assert f"the newest release tag {newest}: every run after a release uses it" in line
        assert f"(git checkout {newest} on OMNI)" in line
    elif git["release_tags"]:
        assert line.startswith("FAILED: StepError: the checkout is not at a v0.1.* tag (")
        assert "after the settings freeze every run comes from a checkout at the newest" in line
    else:
        assert line.startswith("FAILED: StepError: no v0.1.* tag exists yet: before the settings")
        assert f"not {arm} round {round_number}, fold {fold}" in line


def test_the_release_guard_names_the_newest_release_and_refuses_a_tag_it_cannot_order() -> None:
    module = load_script("train")
    allowed = module.release_guard(AT_V0_1_10, "b1", 2, 5)
    assert allowed == "at the release tag v0.1.10"
    both = {**AT_NEWER_RELEASE, "tags": TWO_RELEASES}  # both releases on one commit
    assert module.release_guard(both, "b1", 2, 5) == "at the release tag v0.1.1"
    odd = {**RELEASED, "release_tags": ["v0.1.0", "v0.1.x"]}
    with pytest.raises(
        module.StepError, match=re.escape("the tag v0.1.x matches v0.1.* but is no")
    ):
        module.release_guard(odd, "b1", 2, 5)


def test_a_dirty_checkout_runs_only_with_allow_dirty_which_run_json_records(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("train")
    patch_guards(module, monkeypatch, {**CLEAN, "dirty": True})
    line = refused(module, monkeypatch, capsys, repository_args("b0", 1, 1, "--allow-dirty"))
    assert line.startswith("FAILED: StepError: the pinned labels")  # past the guard
    folder = Path(os.environ["LBW_WORKSPACE"]) / "runs" / "pilot" / f"pilot-b0-r1-f1-{STAMP}"
    info = info_of(folder)
    assert (info["allow_dirty"], info["git"]["dirty"], info["status"]) == (True, True, "failed")


def test_a_run_of_cpg_round_1_fold_1_does_not_start_while_another_ones_run_json_is_unreadable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("train")
    patch_guards(module, monkeypatch, CLEAN)
    root = Path(os.environ["LBW_WORKSPACE"])
    done = write_record(root, "20261126-080000", cpg_record("09:00:00", reads_rules=True))
    kept = (root / done / "run.json").read_text(encoding="utf-8")
    broken = write_record(root, "20261126-090000", "{not json")
    line = refused(module, monkeypatch, capsys, repository_args("cpg"))
    assert line.startswith(
        f"FAILED: StepError: the run.json of {broken} cannot be read: a run of CPG's round 1,"
        " fold 1 could not tell which run read the two allowed rules and check F5, so it does"
        " not start"
    )
    # Refused before its folder, before any GPU work; the records stay as they are.
    folders = sorted(path.name for path in (root / "runs" / "pilot").iterdir())
    assert folders == [Path(done).name, Path(broken).name]
    assert (root / done / "run.json").read_text(encoding="utf-8") == kept
    assert (root / broken / "run.json").read_text(encoding="utf-8") == "{not json"
    # A run that reads no such record gets past it: B0's round 1, the timing run, and CPG's
    # other folds; the empty test workspace then has no pinned labels.
    for git, args in (
        (CLEAN, repository_args("b0")),
        (CLEAN, repository_args("cpg", 1, 1, "--timing")),
        (RELEASED, repository_args("cpg", 1, 2)),
    ):
        patch_guards(module, monkeypatch, git)
        line = refused(module, monkeypatch, capsys, args)
        assert line.startswith("FAILED: StepError: the pinned labels"), line


def test_the_job_never_writes_into_an_existing_run_folder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("train")
    patch_guards(module, monkeypatch, CLEAN)
    first = Path(os.environ["LBW_WORKSPACE"]) / "runs" / "pilot" / f"pilot-b0-r1-f1-{STAMP}"
    first.mkdir(parents=True)
    (first / "run.json").write_text('{"status": "done"}\n', encoding="utf-8")
    line = refused(module, monkeypatch, capsys, repository_args("b0"))
    assert line.startswith("FAILED: FileExistsError")
    assert sorted(path.name for path in first.iterdir()) == ["run.json"]
    assert (first / "run.json").read_text(encoding="utf-8") == '{"status": "done"}\n'


def change_partitions(workspace: WordcheckWorkspace) -> None:
    path = workspace.root / workspace.config["partitions"]["run"] / "partitions.parquet"
    table = pd.read_parquet(path)
    table.loc[0, "fold"] = 1 + table.loc[0, "fold"] % 5  # one participant moved to another fold
    table.to_parquet(path, index=False)


def change_labels(workspace: WordcheckWorkspace) -> None:
    path = workspace.root / workspace.config["labels"]["run"] / "labels.parquet"
    table = pd.read_parquet(path)
    table.loc[table.index[0], "label"] = 1 - table.loc[table.index[0], "label"]
    table.to_parquet(path)


def flag_cache(view: str) -> Callable[[WordcheckWorkspace], None]:
    def plant(workspace: WordcheckWorkspace) -> None:
        path = workspace.root / CHECK_RUN / "run.json"
        info = json.loads(path.read_text("utf-8"))
        flagged = ["model.language_model.layers.1/output"]
        info["results"]["runs"][f"cache/{view}/run-{CACHE_JOBS[view]}"]["lm_watch"]["flagged"] = (
            flagged
        )
        path.write_text(json.dumps(info), "utf-8")

    return plant


def fail_the_words_of_x1(workspace: WordcheckWorkspace) -> None:
    path = workspace.root / WORDCHECK_RUN / "run.json"
    info = json.loads(path.read_text("utf-8"))
    info["verdicts"]["x1_words"]["verdict"] = FAIL
    path.write_text(json.dumps(info), "utf-8")


def unpin_zeroshot(workspace: WordcheckWorkspace) -> None:
    workspace.config.pop("zeroshot_scores")
    workspace.write_config()


REFUSALS: dict[str, tuple[str, Callable[[WordcheckWorkspace], None], str]] = {
    "changed partitions": (
        "b0",
        change_partitions,
        "the pinned partitions: partitions.parquet does not match its SHA-256 (partitions):"
        " training stops",
    ),
    "changed labels": (
        "b0",
        change_labels,
        "the pinned labels: labels.parquet does not match its SHA-256 (labels)",
    ),
    "a flagged cache of x2": (
        "b0",
        flag_cache("x2"),
        "the checks of the caches flagged the language model on the longest clips of x2:"
        " training waits for the fallback",
    ),
    "a flagged cache of rev": (
        "cpg",
        flag_cache("rev"),
        "the checks of the caches flagged the language model on the longest clips of rev",
    ),
    "the embargo": ("b0", fail_the_words_of_x1, f"{WORDCHECK_RUN}: the words of x1 say FAIL"),
    "no zero-shot pin for CPG": (
        "cpg",
        unpin_zeroshot,
        "no zero-shot scores are pinned: zeroshot_scores needs exactly run and sha256",
    ),
    "no zero-shot pin for B1": (
        "b1",
        unpin_zeroshot,
        "no zero-shot scores are pinned: zeroshot_scores needs exactly run and sha256",
    ),
    "pending": (
        "b0",
        lambda workspace: write_training(workspace, mode="pending"),
        "training runs in mode 'strict' or 'warn_only', not 'pending': the timing run sets it",
    ),
}


@pytest.mark.slow
@pytest.mark.parametrize("change", list(REFUSALS))
def test_the_job_refuses_to_train_unless_its_pins_hold(
    workspace: WordcheckWorkspace, change: str
) -> None:
    arm, plant, message = REFUSALS[change]
    plant(workspace)

    def never_trains(
        module: ModuleType, monkeypatch: pytest.MonkeyPatch, calls: dict[str, list[Any]]
    ) -> None:
        def build(*_args: Any, **_kwargs: Any) -> None:
            calls.setdefault("build", []).append(1)
            raise AssertionError("the model was built")

        monkeypatch.setattr(training, "build_trainable", build)

    run = finish(workspace, arm, git=RELEASED if arm == "b1" else CLEAN, prepare=never_trains)
    assert run.ended == "failed"
    assert f"FAILED: StepError: {message}" in run.out
    assert run.info["status"] == "failed"
    assert run.info["error"]["message"].startswith(message)
    assert "build" not in run.calls  # refused before the model is built
    assert_only_aggregates_leave(run.run / "run.json", run.streams)


@pytest.mark.slow
def test_the_job_refuses_other_numeric_settings_than_the_caches(
    workspace: WordcheckWorkspace,
) -> None:
    # A job of other CPUs than the cache jobs: OMP_NUM_THREADS follows --cpus-per-task.
    run = finish(workspace, "b0", env={"OMP_NUM_THREADS": "8", "SLURM_CPUS_PER_TASK": "8"})
    message = (
        f"cache/x0/run-{CACHE_JOBS['x0']} was built with other numeric settings than this job's"
    )
    assert run.ended == "failed"
    assert run.info["error"]["message"].startswith(message)
    assert f"FAILED: StepError: {message}" in run.out
    assert "training" not in run.info["settings"]["numerics"]
    assert "model" not in run.info  # refused before the model is built


# The job file ------------------------------------------------------------------------------------


def test_the_job_file_asks_for_one_gpu_the_cpus_of_the_cache_jobs_and_one_fold_per_task() -> None:
    text = (REPO / "scripts" / "slurm" / "train.sbatch").read_text(encoding="utf-8")
    for line in (
        "#SBATCH --job-name=lbw-train",
        "#SBATCH --partition=gpu",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=48G",
        "#SBATCH --time=02:30:00",
        "#SBATCH --array=0-4",
    ):
        assert f"{line}\n" in text, line
    config = yaml.safe_load((CONFIGS / "data.yaml").read_text(encoding="utf-8"))
    assert config["encoder_cache"]["cpu_threads"] == 4
    assert "set -euo pipefail\nsource scripts/slurm/common.sh\n" in text
    assert 'config="${1:?usage: train.sbatch <config> <round> [--timing]}"\n' in text
    assert 'round="${2:?usage: train.sbatch <config> <round> [--timing]}"\nshift 2\n' in text
    assert 'task="${SLURM_ARRAY_TASK_ID:?submit this job as an array}"\n' in text
    assert text.endswith(
        'python scripts/train.py --config "$config" --round "$round" --fold $((task + 1)) "$@"\n'
    )
    assert "git checkout v0.1.x" in text
    sbatch = [line for line in text.splitlines() if line.startswith("#SBATCH")]
    first_command = text.splitlines().index("set -euo pipefail")
    assert all(text.splitlines().index(line) < first_command for line in sbatch)
    # Task t trains fold t + 1: the job file's own command line, run by the shell with echo.
    command = text.splitlines()[-1].replace("python ", "echo ", 1)
    for task in range(5):
        shown = subprocess.run(
            ["bash", "-c", f"config=c.yaml; round=2; task={task}; set -- --timing; {command}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert shown == f"scripts/train.py --config c.yaml --round 2 --fold {task + 1} --timing\n"


def test_the_run_is_named_by_study_arm_round_fold_and_utc_time() -> None:
    from datetime import UTC, datetime

    module = load_script("train")
    before = datetime.now(UTC).replace(microsecond=0)
    name = module.run_name("cpg", 2, 5)
    after = datetime.now(UTC)
    assert re.fullmatch(r"pilot-cpg-r2-f5-\d{8}-\d{6}", name)
    stamp = datetime.strptime(name[len("pilot-cpg-r2-f5-") :], "%Y%m%d-%H%M%S").replace(tzinfo=UTC)
    assert before <= stamp <= after


# GPU test on OMNI: the job's hand-over on the real model, synthetic encoder outputs only ------


REAL_WIDTH = 1280  # the real encoder's width
GPU_BAGS = (  # 2 + 2 bags of fake participants; 9001_003 has two chunks in x0 and x2
    (9001, 1),
    (9002, 0),
    (9003, 1),
    (9004, 0),
)


def synthetic_cache(folder: Path, view: str, segments: list[str]) -> Any:
    """A cache of random fp16 encoder outputs, laid out and listed as a pinned cache run."""
    from listening_beyond_words.cache import (
        CACHE_COLUMNS,
        CacheRun,
        PinnedCache,
        to_bytes,
        write_file,
    )

    rows = []
    for segment in segments:
        session = int(segment.split("_")[0])
        chunks = 2 if segment == "9001_003" and view != "x1" else 1
        generator = torch.Generator().manual_seed(item_seed(5, view, segment))
        output = torch.randn(chunks, 1500, REAL_WIDTH, generator=generator).half()
        data = to_bytes(output)
        path = folder / view / str(session) / f"{segment}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_file(path, data)
        rows.append(
            {
                "segment": segment,
                "session": session,
                "file": f"{session}/{segment}.npy",
                "chunks": chunks,
                "samples": 480_000 * chunks,
                "sha256": hashlib.sha256(data).hexdigest(),
                "audio_sha256": "0" * 64,
                "features_sha256": "0" * 64,
                "max_abs": float(output.abs().max()),
            }
        )
    manifest = pd.DataFrame(rows, columns=list(CACHE_COLUMNS))
    run = CacheRun(f"cache/{view}/run-0", folder / view, view, {}, manifest, "0" * 64, manifest)
    return PinnedCache(view, {}, run, manifest, "derived/cache_check/run-0", ())


@pytest.mark.gpu
def test_the_jobs_hand_over_from_training_to_scoring_fits_on_the_gpu(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from listening_beyond_words.bags import Bag
    from listening_beyond_words.models import local_snapshot
    from listening_beyond_words.runinfo import Progress

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        pytest.skip("HF_HOME is not set; run the GPU tests in scripts/slurm/smoke_test.sbatch")
    module = load_script("train")
    snapshot = local_snapshot(VOXTRAL, Path(hf_home))
    arm = load_arm(CONFIGS / "cpg.yaml")
    threads = yaml.safe_load((CONFIGS / "data.yaml").read_text("utf-8"))["encoder_cache"][
        "cpu_threads"
    ]
    bags = tuple(
        Bag(session, label, tuple(f"{session}_{index:03d}" for index in range(4)))
        for session, label in GPU_BAGS
    )
    segments = [segment for bag in bags for segment in bag.segments]
    caches = {
        view: synthetic_cache(tmp_path / "cache", view, segments)
        for view in ("x0", "x1", "x2", "rev")
    }

    def show(line: str) -> None:
        with capsys.disabled():
            sys.stdout.write(line + "\n")

    def gib(value: float) -> str:
        return f"{value / 2**30:.2f} GiB"

    voxtral.set_numerics(threads)  # as the job sets them first; state_put_back restores them
    torch.cuda.reset_peak_memory_stats()
    trainable = training.build_trainable(snapshot, arm.shared.lora, 11)
    built = torch.cuda.max_memory_allocated()
    optimiser = training.make_optimiser(trainable, arm.shared, 10)
    watch = voxtral.watch_layers(voxtral.language_model_layers(trainable.model, logits=True))
    prompts = {count: voxtral.prompt_inputs(trainable.processor, count) for count in (1, 2)}
    answer_ids = trainable.answer_ids
    read = module.Reader(caches, REAL_WIDTH)
    stats = training.stand_in_stats(arm.shared)
    records = []
    for step in range(2):
        record = training.train_step(
            trainable, optimiser, watch, arm, bags, read, prompts, stats, step, 1
        )
        stats = record.stats
        records.append(record)
    trained = torch.cuda.max_memory_allocated()
    training.save_adapter(trainable.model, tmp_path / "adapter")
    released = training.release_training(trainable, optimiser, watch)
    del trainable, optimiser, watch
    torch.cuda.reset_peak_memory_stats()
    voxtral.set_numerics(threads)
    test = {
        view: [segment for segment in segments if segment.startswith(("9002", "9003"))]
        for view in ("x0", "rev")
    }
    table, summary = module.score_fold(
        snapshot,
        tmp_path / "adapter",
        {view: caches[view] for view in ("x0", "rev")},
        test,
        prompts,
        answer_ids,
        REAL_WIDTH,
        Progress(),
    )
    scored = torch.cuda.max_memory_allocated()
    finite = int(np.isfinite(table["z"]).sum())
    show(
        f"hand-over: peak {gib(built)} once built, {gib(trained)} over 2 CPG steps"
        f" ({sum(r.skipped for r in records)} skipped, {sum(r.calls for r in records)} calls);"
        f" after the release {released['allocated_bytes']:,} bytes allocated,"
        f" {released['reserved_bytes']:,} reserved; scoring peak {gib(scored)}; {finite} of"
        f" {len(table)} z finite on random encoder outputs, watch {len(summary['flagged'])} of"
        f" {summary['parts']} parts flagged"
    )
    # What this test judges: the hand-over fits, and every test clip gets its z.
    assert list(table["segment"]) == test["x0"] + test["rev"]
    assert list(table["view"]) == ["x0"] * 8 + ["rev"] * 8
