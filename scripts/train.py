"""Train one arm on one fold-run from the encoder caches, then score its test fold (GPU job).

Submit from an OMNI login node, one array of 5 fold-runs per arm and round (array task t
trains fold t + 1; after the settings freeze CPG's round 1 takes --array=1-4, since fold 1
has trained: scripts/slurm/train.sbatch):
    scripts/slurm/submit.sh --job-name=lbw-train-<arm> train configs/pilot/<arm>.yaml <round>
and, before any trained run, the timing run of check F5 (CPG, round 1, fold 1), held until
the CPU and GPU tests pass:
    scripts/slurm/submit.sh --job-name=lbw-train-cpg --array=0 --time=01:00:00 \\
        --dependency=afterok:<CPU-test job>:<GPU-test job> \\
        train configs/pilot/cpg.yaml 1 --timing
The job file passes --config, --round and --fold:
    train.py --config configs/pilot/<arm>.yaml --round R --fold F
        [--timing] [--allow-dirty] [--verbose]
--allow-dirty, for debugging only, lets a checkout with uncommitted changes run and is
recorded; pilot runs never use it. --verbose logs one line per step as well.

Pilot protocol, section "Training"; the settings are in the arm file (configs/pilot/b0.yaml,
b1.yaml or cpg.yaml) with configs/pilot/train.yaml, which it extends (config.load_arm), in
configs/pilot/data.yaml (bags, encoder_cache and the pins) and the pinned model in
configs/pilot/models.yaml. A run is defined by its arm file, its round and its fold: no
option changes a setting, and the arm file must be one of configs/pilot/ in the checkout,
so that the clean-checkout and release-tag guards cover the settings it runs with.

Stages, in one process:
  1. Guards: a Slurm job, offline, a clean checkout, a GPU; the arm file in
     configs/pilot/ of the checkout, with train.yaml; --timing only for CPG on round 1,
     fold 1; the release tag: until a v0.1.x tag exists in the clone, only B0's round 1
     and CPG's round 1, fold 1 may run (the timing run counts as the latter); once one
     exists, every run must come from a checkout at the newest v0.1.x tag, which holds
     any fallback released after the freeze (pilot protocol, section "Training", bullets
     "Overflow in training" and "Order, release tag and fallbacks"); a run of CPG's round
     1, fold 1 (the timing run excepted) refused while the run.json of another run of it
     cannot be read, since it could not tell which run read the rules and F5 (stage 8).
     Then the run folder and run.json (running).
  2. The pins: the labels and partitions, checked by their SHA-256; the caches of all four
     clip versions with their checks (cache.pinned_cache), refused while any check flagged
     the language model, since the fallback (the flagged layers in fp32) is not in place;
     for B1 and CPG the zero-shot scores (zeroshot.load_zeroshot_scores), whose x0 and x1
     start the running statistics; and the embargo (scoring.embargo), so that a run that
     could not score stops before it trains.
  3. The numeric settings of the caches (voxtral.set_numerics, with
     encoder_cache.cpu_threads threads), which must equal those that each cache recorded,
     so that a job of other CPUs stops here; then training's repeatability mode
     (precision.deterministic: strict or warn_only; pending is refused).
  4. The seeds of the fold-run (runinfo.seed_everything; training.torch_seed before LoRA
     is added; bags.stream_seed for the bags), the same for every arm; the model that
     trains (training.build_trainable) and the prompts of each chunk count; the bag
     sampler of the round's other four folds (bags.BagSampler), T with its warm-up and
     λG ramp; the running statistics, started from the zero-shot bag means of the first
     8 steps' bags (training.start_stats); AdamW, its schedule and the loss scaler; the
     overflow watch on the adapter, the language model and the logits.
  5. Training: every step of every epoch (training.train_step), one line per step in
     metrics.jsonl and one log line per epoch. A z or a loss that is not finite, a flagged
     part or a loss scale below 1 ends the run failed at that step (check F1).
  6. The weights saved (training.save_adapter: LoRA and the adapter copy in fp32), which
     ends the training part that check F5 times; the training model released and the GPU
     checked (training.release_training).
  7. The numeric settings of the caches again, compared; the embargo again. The scoring
     model built from the saved file (training.load_for_scoring) scores every segment of
     the test fold on x0 and rev, from the caches, one clip per call, by the zero-shot
     job's scoring code (scoring.score_views) with the watch on. A flagged part or a z
     that is not finite ends the run failed with no score written: it goes again from the
     start. scores.parquet must read back bit for bit.
  8. The checks and rules read on metrics.jsonl (training.rule_measures): F1's counts, the
     gain-alive share, the 4-epoch rule's halves and, for B1 and CPG, the training hours,
     with a verdict only where the protocol reads one (rule_readings): F1 on every run, F2
     on every CPG run; the gain-alive rule, the 4-epoch rule and F5 once, on the first CPG
     run of round 1, fold 1 that finishes, if it ran without --allow-dirty and before the
     freeze (at no v0.1.x tag); and F5 alone again, on its own hours, only on its rerun at
     8 segments after F5 failed there at 16. Which run made each reading comes from the
     run.json of the other runs of CPG's round 1, fold 1 (earlier_readings); every other
     run of it gets no verdict of the rules or of F5 and names the runs that made them, or
     says why it made none or cannot tell (a run.json that cannot be read). run.json
     (done).

--timing (check F5): stages 1 to 4 without the zero-shot scores and the embargo, pending
allowed, and μ and σ from the stand-in start (timing.stand_in_start). Then the smoke bag:
the training participant with at least 4 segments who holds the longest x0 clip, with its
three next-longest segments, goes forward and backward in every view of the arm at loss
scale 1.0, with the watch on and no update (a gradient that is not finite fails check
F1). Then the sampler's first timing.steps_per_mode steps in warn-only mode and as many
next ones in strict mode, an error there being strict mode's outcome. The projection of a
three-view run is the load time plus T times the median seconds of the timed steps after
the first (training.f5_projection), in each mode; precision.strict_max_extra_time_share
gives the mode. The run keeps no weights and writes no score.

Writes $LBW_WORKSPACE/runs/pilot/pilot-<arm>-r<round>-f<fold>-<YYYYmmdd-HHMMSS>/ (UTC) with
  adapter/        adapter_model.safetensors (LoRA and the adapter copy, in fp32),
                  adapter_config.json and PEFT's README.md
  metrics.jsonl   one line per step (training.StepRecord.line): the loss and its parts, the
                  rates, λG, the loss scale, the gradient norm, μ and σ before the update
                  and the logged measures as counts and means over the step's bags; no ID
  run.log         the job's log lines, as on stderr
  scores.parquet  one z per segment of the test fold and scored view: segment, session,
                  view and z (zeroshot.SCORE_COLUMNS)
  run.json        commit and tags, the resolved config and its SHA-256, the pins, the
                  seeds and their streams, the numeric settings of training and of
                  scoring, the model and its trained numbers, the optimiser, the
                  environment, and aggregate results: T, the sampler's counts, the start of
                  the running statistics, the measures per epoch, the checks and rules,
                  which verdicts the run reads and where they were read
                  (results.rule_readings), the hours and peak GPU memory of training and of
                  scoring, the watches and the label-free numbers of the scores
and, with --timing, only $LBW_WORKSPACE/derived/train_timing/run-<job id>/run.json. A job
that fails also leaves errors.log there: traceback frames and error types, never a message.
The job logs aggregate numbers only; a failure may name its segment on its one FAILED line
and in run.json's error. metrics.jsonl and scores.parquet stay in the workspace on OMNI.
"""

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import re
import socket
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pandas as pd
import torch
import yaml

from listening_beyond_words.bags import STREAM as BAG_STREAM
from listening_beyond_words.bags import (
    Bag,
    BagSampler,
    BagSettings,
    stream_seed,
    training_people,
)
from listening_beyond_words.cache import CacheSettings, PinnedCache, pinned_cache, read_file
from listening_beyond_words.clips import load_pinned_labels, load_pinned_partitions
from listening_beyond_words.config import (
    CLIP_VERSIONS,
    STRICT,
    WARN_ONLY,
    ArmConfig,
    load_arm,
    resolve_config,
)
from listening_beyond_words.loss import ce_on_z, gain, share_steps, soft_hinge
from listening_beyond_words.models import ModelPin, load_pins, local_snapshot, snapshot_record
from listening_beyond_words.partitions import N_FOLDS
from listening_beyond_words.runinfo import (
    RELEASE_TAGS,
    Progress,
    StepError,
    environment,
    git_state,
    gpu_facts,
    peak_memory_gb,
    quiet_unraisable,
    record_failure,
    seed_everything,
    sha256_file,
    stop_on_sigterm,
    utc_now,
    write_json,
)
from listening_beyond_words.scoring import embargo, label_free, score_views, watch_summary
from listening_beyond_words.settings import Workspace, get_workspace
from listening_beyond_words.training import (
    ADAPTER_FILE,
    PACKAGES,
    TORCH_STREAM,
    Optimiser,
    Trainable,
    ViewStats,
    build_trainable,
    f5_projection,
    load_for_scoring,
    make_optimiser,
    release_training,
    rule_measures,
    save_adapter,
    stand_in_stats,
    start_stats,
    torch_seed,
    train_step,
    training_numerics,
)
from listening_beyond_words.voxtral import (
    ANSWERS,
    CLAMP_ABS,
    PROMPT,
    LayerWatch,
    answer_token_ids,
    cached_z,
    language_model_layers,
    prompt_inputs,
    set_numerics,
    watch_layers,
)
from listening_beyond_words.zeroshot import SCORE_COLUMNS, SCORES, load_zeroshot_scores

CONFIG = Path("configs/pilot/data.yaml")
MODELS = Path("configs/pilot/models.yaml")
# The folder of the arm files and their train.yaml, in the checkout: a run reads its
# settings from there only, so that the clean-checkout and release-tag guards cover them.
CONFIGS = Path("configs/pilot")
STEP = "train"
STUDY = "pilot"
USAGE = (
    "usage: train.py --config configs/pilot/<arm>.yaml --round <round> --fold <fold>"
    " [--timing] [--allow-dirty] [--verbose]"
)
# The first runs, which may run before the settings freeze (pilot protocol, section
# "Training", bullet "Order, release tag and fallbacks"): B0's round 1, and CPG's round 1,
# fold 1, whose training logs decide the two allowed rules. The timing run is the latter.
FIRST_RUNS = (("b0", 1, None), ("cpg", 1, 1))
# CPG's round 1, fold 1: the training logs of its first run that finishes decide the
# gain-alive rule, the 4-epoch rule and check F5, and those of its rerun at 8 segments after
# F5 failed there at 16 decide F5 alone (pilot protocol, section "Training", bullets "F2 and
# the two allowed rules" and "Check F5"); the timing run is this fold-run too.
RULES_FOLD_RUN = ("cpg", 1, 1)
TIMING_RUN = RULES_FOLD_RUN
# Check F5's fallback: a participant offers at most 16 segments per epoch, and every arm
# trains with 8 once F5 fails at 16 (bags.max_segments_per_epoch; bullet "Check F5").
FULL_SEGMENTS = 16
FALLBACK_SEGMENTS = 8
# The verdicts a run may read beside check F1 (training.rule_measures).
READS = ("reads_f2", "reads_rules", "reads_f5")
# A release tag: v0.1.<number>, which the release guard orders by its number.
RELEASE = re.compile(r"v0\.1\.(0|[1-9][0-9]*)")
TIMING = "train_timing"  # the timing run writes derived/train_timing/run-<job id>/run.json only
METRICS = "metrics.jsonl"
RUN_LOG = "run.log"
ADAPTER = "adapter"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

logger = logging.getLogger(__name__)


# The machine and the record ------------------------------------------------------------------


def run_name(arm: str, round_number: int, fold: int) -> str:
    """The run's name: study, arm, round, fold and the UTC time (conventions, section 4)."""
    return f"{STUDY}-{arm}-r{round_number}-f{fold}-{datetime.now(UTC):%Y%m%d-%H%M%S}"


def ids_sha256(ids: Any) -> str:
    """The SHA-256 of token IDs as little-endian 64-bit integers."""
    return hashlib.sha256(ids.detach().cpu().numpy().astype("<i8").tobytes()).hexdigest()


def synchronise() -> None:
    """Wait until the GPU has run every kernel queued so far: a time taken next includes them."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory() -> None:
    """Start PyTorch's count of the most GPU memory held anew, for the next stage."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class _Parser(argparse.ArgumentParser):
    """argparse, except that a wrong command line is a refusal with the usage."""

    def error(self, message: str) -> NoReturn:
        raise StepError(f"{USAGE} ({message})")


@dataclass(frozen=True)
class Arguments:
    """The command line: the arm file and the fold-run, and the job's own options."""

    config: Path
    round: int
    fold: int
    timing: bool
    allow_dirty: bool
    verbose: bool


def parse_arguments(argv: Sequence[str]) -> Arguments:
    """The command line; anything else is refused with the usage."""
    parser = _Parser(prog="train.py", add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    given = parser.parse_args(list(argv))
    if given.round not in (1, 2):
        raise StepError(f"{USAGE} (the pilot has rounds 1 and 2, not {given.round})")
    if not 1 <= given.fold <= N_FOLDS:
        raise StepError(f"{USAGE} (a round has folds 1 to {N_FOLDS}, not {given.fold})")
    return Arguments(
        given.config, given.round, given.fold, given.timing, given.allow_dirty, given.verbose
    )


def release_guard(git: Mapping[str, Any], arm: str, round_number: int, fold: int) -> str:
    """Refuse a run that the release tags do not allow; return what allowed it.

    Pilot protocol, section "Training", bullet "Order, release tag and fallbacks": until a
    v0.1.x tag exists in the clone, only B0's round 1 and CPG's round 1, fold 1 run (the
    timing run among them); once one exists, every run, these included, comes from a
    checkout at a v0.1.x tag, so a checkout of an older commit is refused too. The tag must
    be the newest v0.1.x, ordered by its number: a fallback used after the freeze is
    released as the next v0.1.x, and every later run uses it (bullet "Overflow in
    training"), so a checkout of an older release is refused as well.
    """
    released = [str(tag) for tag in git.get("release_tags") or []]
    if not released:
        if any(
            (arm, round_number) == (first_arm, first_round) and first_fold in (None, fold)
            for first_arm, first_round, first_fold in FIRST_RUNS
        ):
            return "no v0.1.x tag yet: one of the first runs (B0's round 1, CPG's round 1, fold 1)"
        raise StepError(
            f"no {RELEASE_TAGS} tag exists yet: before the settings freeze only B0's round 1"
            " and CPG's round 1, fold 1 may run, not"
            f" {arm} round {round_number}, fold {fold} (pilot protocol, section"
            ' "Training", bullet "Order, release tag and fallbacks")'
        )
    numbers: dict[str, int] = {}
    for tag in released:
        found = RELEASE.fullmatch(tag)
        if found is None:
            raise StepError(
                f"the tag {tag} matches {RELEASE_TAGS} but is no release v0.1.<number>: the"
                " release guard cannot order it"
            )
        numbers[tag] = int(found.group(1))
    newest = max(released, key=numbers.__getitem__)
    at_head = set(map(str, git.get("tags") or [])) & set(released)
    if not at_head:
        raise StepError(
            f"the checkout is not at a {RELEASE_TAGS} tag ({git.get('describe')}), though"
            f" {', '.join(released)} exists: after the settings freeze every run comes from a"
            f" checkout at the newest release tag (git checkout {newest} on OMNI)"
        )
    if newest not in at_head:
        raise StepError(
            f"the checkout is at {', '.join(sorted(at_head, key=numbers.__getitem__))}, not at"
            f" the newest release tag {newest}: every run after a release uses it, with any"
            f" fallback released after the freeze (git checkout {newest} on OMNI)"
        )
    return f"at the release tag {newest}"


def check_config_folder(config: Path) -> None:
    """Refuse an arm file outside the checkout's ``CONFIGS`` folder.

    A run is defined by its committed arm file, its round and its fold (engineering
    conventions, section 5): an arm file elsewhere, with a train.yaml beside it, would
    escape the clean-checkout and release-tag guards, which see the checkout only.
    """
    if config.resolve().parent != CONFIGS.resolve():
        raise StepError(
            f"the arm file must be {CONFIGS}/<arm>.yaml in the checkout, not {config}: a run"
            " trains with the committed settings only"
        )


def set_up_logging() -> None:
    """Log at INFO to stderr, the Slurm log (engineering conventions, section 4)."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)


def log_to_file(path: Path) -> None:
    """Also write the log lines to run.log in the run folder."""
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)


# Stage 1: the run ---------------------------------------------------------------------------


@dataclass
class Run:
    """The run being made: what it is, where it writes, what it records."""

    args: Arguments
    job: str
    arm: ArmConfig
    workspace: Workspace
    config: dict[str, Any]
    voxtral: ModelPin
    run_dir: Path
    info: dict[str, Any]
    started: float

    def save(self) -> None:
        """Write run.json as it stands."""
        write_json(self.run_dir / "run.json", self.info)

    @property
    def relative(self) -> Path:
        """The run folder, relative to the workspace."""
        return self.run_dir.relative_to(self.workspace.root)

    @property
    def results(self) -> dict[str, Any]:
        """The results recorded so far."""
        return self.info["results"]


def start(state: Progress, started: float) -> Run:
    """Stage 1: refuse to run outside a clean offline GPU job or off its release; the folder."""
    stop_on_sigterm()
    # Before transformers loads, which reads it for its own progress bars too.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    args = parse_arguments(sys.argv[1:])
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise StepError("submit this as a job: scripts/slurm/submit.sh train <config> <round>")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise StepError(
            "HF_HUB_OFFLINE is not 1: the job must run offline (scripts/slurm/common.sh)"
        )
    git = git_state()
    if git["dirty"] and not args.allow_dirty:
        raise StepError("the checkout has uncommitted changes; commit them on the Mac and pull")
    gpu = gpu_facts()
    if not gpu.get("gpu"):
        raise StepError("no CUDA GPU: submit this as a GPU job (scripts/slurm/train.sbatch)")
    check_config_folder(args.config)
    arm = load_arm(args.config)
    resolved = resolve_config(args.config)
    if resolved.sha256 != arm.sha256:  # the files changed between the two reads
        raise StepError(f"{args.config.name} changed while it was read")
    if args.timing and (arm.arm, args.round, args.fold) != TIMING_RUN:
        raise StepError(
            "the timing run of check F5 is CPG's round 1, fold 1 (pilot protocol, section"
            f' "Training", bullet "Check F5"), not {arm.arm} round {args.round}, fold'
            f" {args.fold}"
        )
    allowed = release_guard(git, arm.arm, args.round, args.fold)
    workspace = get_workspace()
    if not args.timing and (arm.arm, args.round, args.fold) == RULES_FOLD_RUN:
        unreadable = earlier_readings(workspace).unreadable
        if unreadable:  # stage 8 could not tell which run read the rules and F5
            raise StepError(
                f"the run.json of {', '.join(unreadable)} cannot be read: a run of CPG's round"
                " 1, fold 1 could not tell which run read the two allowed rules and check F5,"
                ' so it does not start (pilot protocol, section "Training", bullets "F2 and the'
                ' two allowed rules" and "Check F5")'
            )
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    voxtral = load_pins(MODELS)[arm.shared.model]
    if args.timing:
        run_dir = workspace.derived / TIMING / f"run-{job}"
    else:
        run_dir = workspace.runs / STUDY / run_name(arm.arm, args.round, args.fold)
    run_dir.mkdir(parents=True, exist_ok=False)
    state.run_dir = run_dir
    if not args.timing:  # the timing run writes its run.json only
        log_to_file(run_dir / RUN_LOG)
    machine = environment(PACKAGES)
    info: dict[str, Any] = {
        "run_id": run_dir.name,
        "step": STEP,
        "arm": arm.arm,
        "round": args.round,
        "fold": args.fold,
        "timing": args.timing,
        "status": "running",
        "started_utc": utc_now(),
        "git": git,
        "allow_dirty": args.allow_dirty,
        "release": allowed,
        "config": {
            "file": str(args.config),
            "sha256": resolved.file_sha256,
            "extends": {"file": str(resolved.shared_file), "sha256": resolved.shared_sha256},
            "resolved": arm.resolved,
            "resolved_sha256": arm.sha256,
        },
        "data_config": {"file": str(CONFIG), "sha256": sha256_file(CONFIG)},
        "models_config": {"file": str(MODELS), "sha256": sha256_file(MODELS)},
        "slurm": {
            "job_id": job,
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "node": socket.gethostname(),
            "cpu": machine["cpu"],
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
        "settings": {
            "bags": config.get("bags"),
            "encoder_cache": config.get("encoder_cache"),
            "numerics": {},
        },
        "seeds": {},
        "inputs": {},
        "environment": {
            **machine,
            **gpu,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        },
        "results": {},
    }
    state.info = info
    run = Run(args, job, arm, workspace, config, voxtral, run_dir, info, started)
    run.save()
    what = "the timing run of check F5 for" if args.timing else "training"
    logger.info(
        f"{what} {arm.arm} (views {', '.join(arm.views)}), round {args.round}, fold"
        f" {args.fold}; job {job} on {socket.gethostname()}, {gpu['gpu']}; CPU {machine['cpu']}"
    )
    logger.info(f"release: {allowed}; commit {git['commit']} ({git.get('describe')})")
    return run


# Stage 2: the pins ----------------------------------------------------------------------------


@dataclass
class Pins:
    """What the run reads from the pins: participants, caches and, for B1 and CPG, z."""

    labels: pd.DataFrame
    partitions: pd.DataFrame
    caches: dict[str, PinnedCache]
    zeroshot: pd.DataFrame | None
    embargo: dict[str, Any] | None


def read_people(run: Run) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The pinned labels and partitions, checked by their SHA-256: a mismatch stops training."""
    for name, file in (("labels", "labels.parquet"), ("partitions", "partitions.parquet")):
        pin = run.config.get(name)
        if not isinstance(pin, Mapping) or not pin.get("run") or not pin.get("sha256"):
            raise StepError(f"no {name} are pinned ({name})")
        path = None
        with contextlib.suppress(ValueError):  # a pin outside the workspace: refused below
            path = run.workspace.resolve(pin["run"]) / file
        if path is None or not path.is_file() or sha256_file(path) != pin["sha256"]:
            raise StepError(
                f"the pinned {name}: {file} does not match its SHA-256 ({name}): training"
                ' stops (pilot protocol, section "Training", bullet "Who trains, and seeds")'
            )
    try:
        labels = load_pinned_labels(run.workspace, run.config)
        partitions = load_pinned_partitions(run.workspace, run.config)
    except ValueError as error:
        # Only the checks of clips.py raise a plain ValueError, with a message of rules and
        # counts; a reader's own error is named by its type.
        reason = str(error) if type(error) is ValueError else type(error).__name__
        raise StepError(f"the pinned labels and partitions do not hold: {reason}") from None
    return labels, partitions


def read_pins(run: Run) -> Pins:
    """Stage 2: labels, partitions, the four caches with their checks, z and the embargo."""
    labels, partitions = read_people(run)
    run.info["inputs"].update(
        labels=dict(run.config["labels"]), partitions=dict(run.config["partitions"])
    )
    caches = {
        view: pinned_cache(run.workspace, run.config, view, run.voxtral) for view in CLIP_VERSIONS
    }
    run.info["inputs"]["caches"] = {view: cache.pin for view, cache in caches.items()}
    run.info["inputs"]["cache_checks"] = {
        view: {"run": cache.check, "lm_flagged": list(cache.lm_flagged)}
        for view, cache in caches.items()
    }
    run.info["inputs"]["views"] = {
        view: dict(cache.run.info["inputs"]["view"]) for view, cache in caches.items()
    }
    run.save()
    flagged = {view: cache.lm_flagged for view, cache in caches.items() if cache.lm_flagged}
    checks = sorted({cache.check for cache in caches.values()})
    logger.info(
        f"cache checks: {', '.join(checks)} watched the language model on the longest clips of"
        f" {', '.join(CLIP_VERSIONS)}: {sum(map(len, flagged.values()))} parts flagged"
    )
    if flagged:
        raise StepError(
            "the checks of the caches flagged the language model on the longest clips of"
            f" {', '.join(flagged)}: training waits for the fallback of the pilot protocol (the"
            ' flagged layers in fp32; section "Training", bullet "Overflow in training")'
        )
    segments = {view: set(map(str, cache.manifest["segment"])) for view, cache in caches.items()}
    if any(found != segments["x0"] for found in segments.values()):
        raise StepError("the pinned caches of the four clip versions hold different segments")
    for view, cache in caches.items():
        logger.info(
            f"cache {view}: {cache.run.run}, {len(cache.manifest):,} clips"
            f" ({int((cache.manifest['chunks'] > 1).sum())} of two chunks)"
        )
    zeroshot = None
    reading = None
    if not run.args.timing:
        if run.arm.lambda_g is not None:  # B1 and CPG start μ and σ from them
            zeroshot = load_zeroshot_scores(run.workspace, run.config)
            run.info["inputs"]["zeroshot_scores"] = dict(run.config["zeroshot_scores"])
            logger.info(f"zero-shot scores: {run.config['zeroshot_scores']['run']}")
        reading = embargo(run.workspace, run.config)
        run.info["inputs"]["wordcheck_run"] = {
            key: reading[key] for key in ("run", "sha256", "run_json_sha256")
        }
        run.info["embargo"] = {"views": reading["views"], "verdicts": reading["verdicts"]}
        verdicts = reading["verdicts"]
        logger.info(
            f"embargo: the checks of x1 are finished ({reading['run']}: F4 {verdicts['f4']}, x1"
            f" words {verdicts['x1_words']}, x1 voice {verdicts['x1_voice']})"
        )
    run.save()
    return Pins(labels, partitions, caches, zeroshot, reading)


# Stage 3: the numeric settings ----------------------------------------------------------------


def caches_numerics(run: Run, caches: Mapping[str, PinnedCache]) -> dict[str, object]:
    """Set the numeric settings of the caches and stop unless every cache recorded them."""
    threads = CacheSettings.from_config(run.config).cpu_threads
    numerics = set_numerics(threads)
    for cache in caches.values():
        if cache.run.info["settings"]["numerics"] != numerics:
            raise StepError(
                f"{cache.run.run} was built with other numeric settings than this job's; run"
                " the job with its job file, whose CPUs are those of the cache jobs"
                " (scripts/slurm/train.sbatch)"
            )
    return numerics


def attention_record(model: Any) -> dict[str, Any]:
    """The attention kernel that the language model uses and PyTorch's choices for it."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return {
        "implementation": getattr(base.config, "_attn_implementation", None),
        "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp": torch.backends.cuda.math_sdp_enabled(),
    }


# Stage 4: the model, the sampler and the statistics ------------------------------------------


class Reader:
    """Reads a segment's stored encoder output of a view, checked, and counts the time it takes."""

    def __init__(self, caches: Mapping[str, PinnedCache], width: int) -> None:
        self.rows = {
            view: {str(row.segment): row for row in cache.manifest.itertuples(index=False)}
            for view, cache in caches.items()
        }
        self.folders = {view: Path(cache.run.folder) for view, cache in caches.items()}
        self.width = width
        self.seconds = 0.0

    def __call__(self, view: str, segment: str) -> torch.Tensor:
        clock = time.monotonic()
        row = self.rows[view][segment]
        output = read_file(self.folders[view] / str(row.file), row, self.width)
        self.seconds += time.monotonic() - clock
        return output


@dataclass
class Setup:
    """What stage 4 builds: the model, its optimiser and watch, the bags and the statistics."""

    trainable: Trainable
    optimiser: Optimiser
    watch: LayerWatch
    sampler: BagSampler
    people: tuple[int, ...]
    labels: dict[int, int]
    read: Reader
    prompts: dict[int, dict[str, Any]]
    answer_ids: tuple[int, int]
    width: int
    stats: ViewStats | None
    total_steps: int


def stats_record(stats: ViewStats) -> dict[str, Any]:
    """μ, σ and the floor of both views' running statistics, for run.json."""
    return {
        view: {
            "mu": values.mu,
            "sigma": values.sigma,
            "raw_sigma": values.raw_sigma,
            "floor": values.floor,
            "at_floor": values.at_floor,
            "mean_square": values.second,
        }
        for view, values in (("x0", stats.x0), ("x1", stats.x1))
    }


def prompts_of(
    run: Run, caches: Mapping[str, PinnedCache], processor: Any, answer_ids: tuple[int, int]
) -> dict[int, dict[str, Any]]:
    """The token IDs and mask of a clip of each number of chunks that the caches hold."""
    counts = sorted({int(count) for cache in caches.values() for count in cache.manifest["chunks"]})
    prompts = {count: prompt_inputs(processor, count) for count in counts}
    run.info["settings"]["prompt"] = {
        "text": PROMPT,
        "answers": list(ANSWERS),
        "answer_ids": list(answer_ids),
        "token_ids": {
            str(count): {
                "tokens": int(prompt["input_ids"].shape[1]),
                "sha256": ids_sha256(prompt["input_ids"]),
            }
            for count, prompt in prompts.items()
        },
    }
    return prompts


def set_up(run: Run, pins: Pins) -> Setup:
    """Stage 4: the seeds, the model, the prompts, the sampler, T, the statistics, AdamW."""
    shared, args = run.arm.shared, run.args
    round_seed = shared.seeds.of(args.round)
    seed = torch_seed(round_seed, args.fold)
    bag_seed = stream_seed(round_seed, args.fold)
    run.info["seeds"] = {
        "round": round_seed,
        "everything": seed,
        "torch": {
            "stream": f'runinfo.item_seed({round_seed}, "{TORCH_STREAM}", {args.fold})',
            "seed": seed,
        },
        "bags": {
            "stream": f'runinfo.item_seed({round_seed}, "{BAG_STREAM}", {args.fold})',
            "seed": bag_seed,
        },
    }
    seed_everything(seed)
    snapshot = local_snapshot(run.voxtral, run.workspace.hf_home)
    run.info["inputs"]["voxtral"] = snapshot_record(run.voxtral, snapshot)
    clock = time.monotonic()
    trainable = build_trainable(snapshot, shared.lora, seed)
    seconds = time.monotonic() - clock
    model = trainable.model.get_base_model()
    width = int(model.config.audio_config.hidden_size)
    run.info["settings"]["width"] = width
    for cache in pins.caches.values():
        if cache.run.info["settings"].get("width") != width:
            raise StepError(
                f"{cache.run.run} holds outputs of width {cache.run.info['settings'].get('width')},"
                f" not {width}"
            )
    answer_ids = trainable.answer_ids
    prompts = prompts_of(run, pins.caches, trainable.processor, answer_ids)
    record = dict(trainable.record)
    run.info["model"] = {"trainable": record}
    run.results["build_seconds"] = round(seconds, 1)
    run.save()
    logger.info(
        f"model: {run.voxtral.repo_id} at {run.voxtral.revision[:8]}, built in {seconds:.0f} s;"
        f" LoRA on {record['lora_modules']} modules of {record['layers']} layers,"
        f" {record['trained_numbers']:,} trained numbers in {record['trained_tensors']}"
        f" tensors; largest |weight| {record['largest_weight']:.6g}; encoder width {width};"
        f" {ANSWERS[0]} {answer_ids[0]}, {ANSWERS[1]} {answer_ids[1]}"
    )

    people = training_people(pins.partitions, args.round, args.fold)
    labels = {session: int(pins.labels.loc[session, "label"]) for session in people}
    manifest = pins.caches["x0"].manifest
    segments = manifest.loc[manifest["session"].isin(people), ["segment", "session"]]
    bag_settings = BagSettings.from_config(run.config)
    sampler = BagSampler(segments, labels, bag_settings, shared.sampler, bag_seed)
    total = sampler.total_steps(shared.epochs)
    counts = sampler.counts()
    warmup = share_steps(shared.optimiser.warmup_share, total)
    steps: dict[str, Any] = {
        "total": total,
        "per_epoch": sampler.steps_per_epoch,
        "epochs": shared.epochs,
        "warmup": warmup,
    }
    if run.arm.lambda_g is not None:
        steps["lambda_g_ramp"] = share_steps(shared.gain.ramp_share, total)
    run.results["steps"] = steps
    run.results["sampler"] = {
        **counts,
        "max_segments_per_epoch": bag_settings.max_segments_per_epoch,
    }
    ramp = f", λG ramp {steps['lambda_g_ramp']}" if "lambda_g_ramp" in steps else ""
    logger.info(
        f"sampler: {len(people)} training participants ({counts['positive_people']} positive,"
        f" {counts['negative_people']} negative; {counts['positive_people_without_a_bag']} and"
        f" {counts['negative_people_without_a_bag']} without a bag); bags per epoch"
        f" {counts['positive_bags_per_epoch']} positive and {counts['negative_bags_per_epoch']}"
        f" negative; {sampler.steps_per_epoch} steps per epoch, T {total}, warm-up {warmup}"
        f"{ramp}; at most {bag_settings.max_segments_per_epoch} segments per participant per"
        " epoch"
    )

    stats = None
    if run.arm.lambda_g is not None:
        if args.timing:
            stats = stand_in_stats(shared)
            source = "the stand-in start of the timing run"
        else:
            if pins.zeroshot is None:  # stage 2 reads them for B1 and CPG
                raise StepError("the running statistics start from the zero-shot scores")
            start_bags = sampler.start_bags(shared.gain.start_steps)
            stats = start_stats(start_bags, pins.zeroshot, shared.gain)
            source = f"the zero-shot scores of the first {shared.gain.start_steps} steps' bags"
        run.results["start_stats"] = {"source": source, **stats_record(stats)}
        logger.info(
            f"running statistics from {source}: μ0 {stats.x0.mu:.4f}, σ0 {stats.x0.sigma:.4f}"
            f" (floor {stats.x0.floor:.4f}); μ1 {stats.x1.mu:.4f}, σ1 {stats.x1.sigma:.4f}"
            f" (floor {stats.x1.floor:.4f})"
        )
    optimiser = make_optimiser(trainable, shared, total)
    run.info["model"]["optimiser"] = optimiser.record
    watch = watch_layers(language_model_layers(trainable.model, logits=True))
    run.save()
    return Setup(
        trainable=trainable,
        optimiser=optimiser,
        watch=watch,
        sampler=sampler,
        people=people,
        labels=labels,
        read=Reader(pins.caches, width),
        prompts=prompts,
        answer_ids=answer_ids,
        width=width,
        stats=stats,
        total_steps=total,
    )


# Stage 5: training ----------------------------------------------------------------------------


def epoch_line(arm: ArmConfig, epoch: int, measures: Mapping[str, Any]) -> str:
    """One epoch's log line: its steps and minutes, the loss parts and, with a gain, its shares."""
    reading = measures["epochs"][str(epoch)]
    f1 = measures["f1"]
    line = (
        f"epoch {epoch} of {arm.shared.epochs}: {reading['steps']} steps"
        f" ({reading['skipped']} skipped), {reading['minutes']:.1f} min; loss"
        f" {reading['loss']:.4f}, CE part {reading['ce_part']:.4f}"
    )
    if reading.get("hinge") is not None:
        share = reading["active_share"]["all"]
        line += (
            f", hinge {reading['hinge']:.4f}; active share"
            f" {'none' if share is None else f'{share:.3f}'} ({reading['active_bags']} of"
            f" {reading['bags']} bags); σ0 at its floor in"
            f" {reading['sigma0_at_floor_share']:.2f} of the steps"
        )
    return line + f"; lowest loss scale {f1['lowest_scale']:g}"


def train(run: Run, setup: Setup) -> list[dict[str, Any]]:
    """Stage 5: every step of every epoch, one line per step in metrics.jsonl."""
    arm = run.arm
    lines: list[dict[str, Any]] = []
    plans: dict[str, Any] = {}
    step = 0
    clock = time.monotonic()
    try:
        with (run.run_dir / METRICS).open("x", encoding="utf-8") as metrics:
            for epoch in range(1, arm.shared.epochs + 1):
                plan = setup.sampler.epoch(epoch)
                plans[str(epoch)] = plan.counts()
                first = len(lines)
                for bags in plan.steps:
                    record = train_step(
                        setup.trainable,
                        setup.optimiser,
                        setup.watch,
                        arm,
                        bags,
                        setup.read,
                        setup.prompts,
                        setup.stats,
                        step,
                        epoch,
                    )
                    setup.stats = record.stats
                    line = record.line()
                    metrics.write(json.dumps(line, allow_nan=False) + "\n")
                    metrics.flush()
                    lines.append(line)
                    step += 1
                    logger.debug(
                        f"step {record.step}: loss {record.loss:.4f}, scale {record.scale:g},"
                        f" skipped {record.skipped}, {record.seconds:.2f} s"
                    )
                logger.info(epoch_line(arm, epoch, rule_measures(lines[first:], arm.shared)))
    finally:  # also when a step stops the run: the counts so far go into run.json
        run.results["training"] = {
            "steps_done": step,
            "minutes": round((time.monotonic() - clock) / 60, 2),
            "cache_read_seconds": round(setup.read.seconds, 1),
            "epoch_plans": plans,
        }
        run.results["lm_watch_training"] = watch_summary(setup.watch)
        run.save()
    if step != setup.total_steps:
        raise StepError(f"the run made {step} steps, not its T of {setup.total_steps}")
    return lines


# Stages 6 and 7: the hand-over and the scores ---------------------------------------------------


def save_and_release(run: Run, setup: Setup) -> None:
    """Stage 6: the weights saved, the training part timed, the training model released."""
    folder = run.run_dir / ADAPTER
    sha256 = save_adapter(setup.trainable.model, folder)
    synchronise()
    hours = (time.monotonic() - run.started) / 3600
    peak = peak_memory_gb()
    released = release_training(setup.trainable, setup.optimiser, setup.watch)
    reset_peak_memory()
    run.info["model"]["adapter_sha256"] = sha256
    run.results["hours"] = {"training": hours}
    run.results["peak_gpu_memory_gb"] = {"training": peak}
    run.results["released"] = released
    run.info["outputs"] = {f"{ADAPTER}/{ADAPTER_FILE}": sha256}
    run.save()
    left = released["allocated_bytes"]
    logger.info(
        f"saved: {ADAPTER}/{ADAPTER_FILE}, SHA-256 {sha256}; training {hours:.3f} h from the"
        f" job's start; peak GPU memory {peak} GB; released, with"
        f" {'no GPU' if left is None else f'{left:,} bytes left allocated on the GPU'}"
    )


def score_fold(
    snapshot: Path,
    folder: Path,
    caches: Mapping[str, PinnedCache],
    segments: Mapping[str, Sequence[str]],
    prompts: Mapping[int, Mapping[str, Any]],
    answer_ids: tuple[int, int],
    width: int,
    state: Progress,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score the test fold with the saved weights; return the scores and the watch's summary.

    The scoring model is built from the saved file alone (training.load_for_scoring): a
    fresh model without the encoder, LoRA and the adapter cast to fp16, in evaluation
    mode. Every segment of each view is scored one clip per call by the zero-shot job's
    scoring code (scoring.score_views), with the watch on the adapter, the language model
    and the logits. Nothing is written here, and no label is passed on.
    """
    loaded = load_for_scoring(snapshot, folder)
    if answer_token_ids(loaded.processor.tokenizer) != tuple(answer_ids):
        raise StepError("the scoring model's Yes and No are other tokens than in training")
    with watch_layers(language_model_layers(loaded.model, logits=True)) as watch:
        table = score_views(
            loaded.model, caches, prompts, answer_ids, width, segments=segments, state=state
        )
    return table, watch_summary(watch)


def score(run: Run, pins: Pins, setup: Setup, state: Progress) -> None:
    """Stage 7: the caches' numeric settings and the embargo again, then the test fold's scores."""
    clock = time.monotonic()
    numerics = caches_numerics(run, pins.caches)  # back from training's mode, compared
    run.info["settings"]["numerics"]["scoring"] = numerics
    reading = embargo(run.workspace, run.config)
    if pins.embargo is None or reading != pins.embargo:
        raise StepError("the embargo's reading changed since the run started: no score")
    views = run.arm.shared.scoring.views
    partitions = pins.partitions
    test = set(
        partitions.loc[
            (partitions["round"] == run.args.round) & (partitions["fold"] == run.args.fold),
            "session",
        ].astype(int)
    )
    if test & set(setup.people):
        raise StepError("a participant of the test fold is also a training participant")
    segments = {}
    for view in views:
        manifest = pins.caches[view].manifest
        segments[view] = sorted(map(str, manifest.loc[manifest["session"].isin(test), "segment"]))
    snapshot = local_snapshot(run.voxtral, run.workspace.hf_home)
    caches = {view: pins.caches[view] for view in views}
    clips = sum(map(len, segments.values()))
    logger.info(
        f"scoring: numerics back to the caches' settings; the embargo holds; the saved weights"
        f" score {len(test)} participants of fold {run.args.fold}, {clips:,} clips of"
        f" {', '.join(views)}, one clip per call"
    )
    table, summary = score_fold(
        snapshot,
        run.run_dir / ADAPTER,
        caches,
        segments,
        setup.prompts,
        setup.answer_ids,
        setup.width,
        state,
    )
    non_finite = int((~np.isfinite(table["z"].to_numpy())).sum())
    results = run.results
    results["lm_watch_scoring"] = summary
    results["non_finite_z"] = non_finite
    run.save()
    largest = "; ".join(f"{name} {value:.6g}" for name, value in summary["largest"])
    logger.info(
        f"scoring watch: {len(summary['flagged'])} of {summary['parts']} parts flagged (a value"
        f" at the clamp, |x| >= {CLAMP_ABS:,}, or not finite; the logits included); largest"
        f" |value|: {largest or 'none'}"
    )
    if summary["flagged"] or non_finite:
        raise StepError(
            f"the scoring watch flagged {len(summary['flagged'])} parts and {non_finite} scores"
            " are not finite: no score is written; the run goes again from the start (pilot"
            ' protocol, section "Training", bullet "Scoring")'
        )
    path = run.run_dir / SCORES
    table.to_parquet(path, index=False)
    sha256 = sha256_file(path)
    back = pd.read_parquet(path)
    if (
        tuple(back.columns) != SCORE_COLUMNS
        or not back.equals(table)
        or [z.hex() for z in back["z"]] != [z.hex() for z in table["z"]]
    ):
        raise StepError(f"{SCORES} does not read back as written")
    synchronise()
    hours = (time.monotonic() - clock) / 3600
    results["hours"]["scoring"] = hours
    results["peak_gpu_memory_gb"]["scoring"] = peak_memory_gb()
    results["z"] = {
        view: label_free(table.loc[table["view"] == view, "z"].to_numpy()) for view in views
    }
    run.info["outputs"][SCORES] = sha256
    run.save()
    logger.info(f"{SCORES}: {len(back):,} rows read back as written; scoring {hours:.3f} h")
    for view, numbers in results["z"].items():
        logger.info(
            f"z of {view}: {numbers['count']:,} scores, all finite; mean {numbers['mean']:.4f},"
            f" SD {numbers['sd']:.4f}; percentiles 5, 50 and 95: {numbers['p5']:.4f},"
            f" {numbers['p50']:.4f}, {numbers['p95']:.4f}; share above 0"
            f" {numbers['share_above_0']:.4f}"
        )


# Stage 8: the checks and rules ------------------------------------------------------------------


def read_metrics(path: Path, lines: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """metrics.jsonl as written, checked against the lines the steps gave."""
    back = [json.loads(text) for text in path.read_text(encoding="utf-8").splitlines()]
    if back != list(lines):
        raise StepError(f"{METRICS} does not read back as written")
    return back


def nested(record: Any, *keys: str) -> Any:
    """The value under ``keys`` in nested JSON objects, or None where a level is missing."""
    for key in keys:
        if not isinstance(record, Mapping):
            return None
        record = record.get(key)
    return record


def at_release(git: Mapping[str, Any]) -> bool:
    """Whether the checkout is at a v0.1.x tag: the settings froze before it."""
    released = {str(tag) for tag in git.get("release_tags") or []}
    return bool(released & {str(tag) for tag in git.get("tags") or []})


@dataclass(frozen=True)
class Earlier:
    """What the other runs of CPG's round 1, fold 1 record of the readings (earlier_readings).

    Attributes:
        made_on: The rules run, which read the two allowed rules and check F5, relative to
            the workspace; None if no run did.
        rules_segments: The most segments per participant and epoch that the rules run
            trained with (its results.sampler.max_segments_per_epoch).
        rules_f5_passes: Whether check F5 passed on the rules run (its
            results.rules.f5.passes).
        f5_read_on: The run that read check F5 alone, relative to the workspace; None if no
            run did.
        unreadable: The run folders, relative to the workspace, whose run.json exists but
            cannot be read.
        undecided: The done runs, not started with --allow-dirty and relative to the
            workspace, that finished unable to tell where the readings were made (their
            results.rule_readings.cannot_tell is set): any of them may have been the run
            due to read, so no later run can tell either.
    """

    made_on: str | None = None
    rules_segments: Any = None
    rules_f5_passes: Any = None
    f5_read_on: str | None = None
    unreadable: tuple[str, ...] = ()
    undecided: tuple[str, ...] = ()


def earlier_readings(workspace: Workspace, run_dir: Path | None = None) -> Earlier:
    """Where the rules and check F5 were read, from the other runs of CPG's round 1, fold 1.

    The candidates are the folders pilot-cpg-r1-f1-* in runs/pilot, other than
    ``run_dir``, that hold a run.json. A run.json that cannot be read as a JSON object makes
    its folder unreadable: it may be the run that made a reading, so no run can tell where
    the readings were made. A readable candidate counts only if it is done and was not
    started with --allow-dirty (its allow_dirty is not true). The rules run is the one whose
    results.rule_readings.reads_rules is true, not merely the first to finish, and the F5
    run the one whose reads_f5 is true; of several, the one that finished first
    (finished_utc, then the name). A counted run whose results.rule_readings.cannot_tell is
    set is undecided: it may have been the run due to read.

    Args:
        workspace: The workspace, whose runs/pilot holds the run folders.
        run_dir: This run's folder, which never counts; None before it exists (stage 1).

    Returns:
        The rules run with its segments and its verdict of F5, the F5 run, the unreadable
        folders and the undecided runs.
    """
    arm, round_number, fold = RULES_FOLD_RUN
    rules: list[tuple[str, str, dict[str, Any]]] = []
    f5: list[tuple[str, str]] = []
    unreadable: list[str] = []
    undecided: list[str] = []
    pattern = f"{STUDY}-{arm}-r{round_number}-f{fold}-*"
    for folder in sorted((workspace.runs / STUDY).glob(pattern)):
        record = folder / "run.json"
        if run_dir is not None and folder.resolve() == run_dir.resolve():
            continue
        if not record.exists():
            continue
        name = folder.relative_to(workspace.root).as_posix()
        try:
            info = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            info = None
        if not isinstance(info, dict):
            unreadable.append(name)
            continue
        if info.get("status") != "done" or info.get("allow_dirty") is True:
            continue
        finished = str(info.get("finished_utc") or "")
        if nested(info, "results", "rule_readings", "cannot_tell"):
            undecided.append(name)
        if nested(info, "results", "rule_readings", "reads_rules") is True:
            rules.append((finished, name, info))
        if nested(info, "results", "rule_readings", "reads_f5") is True:
            f5.append((finished, name))
    if not rules:
        return Earlier(
            f5_read_on=min(f5)[1] if f5 else None,
            unreadable=tuple(unreadable),
            undecided=tuple(undecided),
        )
    _, made_on, info = min(rules, key=lambda run: run[:2])
    return Earlier(
        made_on=made_on,
        rules_segments=nested(info, "results", "sampler", "max_segments_per_epoch"),
        rules_f5_passes=nested(info, "results", "rules", "f5", "passes"),
        f5_read_on=min(f5)[1] if f5 else None,
        unreadable=tuple(unreadable),
        undecided=tuple(undecided),
    )


def rule_readings(
    arm: str,
    round_number: int,
    fold: int,
    earlier: Earlier | None = None,
    *,
    dirty: bool = False,
    released: bool = False,
    segments: int | None = None,
) -> dict[str, Any]:
    """Which verdicts the protocol reads on a run beside check F1, and where they were read.

    Check F2 reads every CPG run. The gain-alive rule, the 4-epoch rule and check F5 are
    read once, on the first CPG run of round 1, fold 1 that finishes, and F5 alone again
    only on its rerun at 8 segments after F5 failed there at 16; the readings of m and of
    the epochs stand (pilot protocol, section "Training", bullets "F2 and the two allowed
    rules", "Check F5" and "Order, release tag and fallbacks"). So a run of CPG's round 1,
    fold 1 reads the rules (``reads_rules``) only if no rules run exists, it was not
    started with --allow-dirty and its checkout is at no v0.1.x tag, since the settings
    freeze after the readings. It reads F5 alone (``reads_f5``, on its own hours, never
    times 4/3) only if the rules run trained with 16 segments and F5 failed there, this
    run trains with 8, no run read F5 alone before and this run was not started with
    --allow-dirty, whose readings no later run would see. While a run.json of another run
    of it cannot be read, or another run of it finished unable to tell, it reads neither. Any other run of it gets no verdict of the
    rules or of F5, and its hours stay a diagnostic.

    Args:
        arm: The run's arm.
        round_number: Its round.
        fold: Its fold.
        earlier: What the other runs of CPG's round 1, fold 1 record
            (``earlier_readings``); needed for that fold-run only.
        dirty: Whether this run was started with --allow-dirty.
        released: Whether its checkout is at a v0.1.x tag (``at_release``).
        segments: The most segments per participant and epoch that it trains with.

    Returns:
        results.rule_readings: the flags ``reads_f2``, ``reads_rules`` and ``reads_f5``;
        ``made_on``, the rules run, and ``f5_read_on``, the run that read F5 alone, each
        relative to the workspace, or None if there is none or the run cannot tell, and
        always for another fold-run; and ``cannot_tell``, the reason the run cannot tell
        where the readings were made, or None.

    Raises:
        ValueError: If a run of CPG's round 1, fold 1 comes without ``earlier``.
    """
    readings: dict[str, Any] = {
        "reads_f2": arm == RULES_FOLD_RUN[0],
        "reads_rules": False,
        "reads_f5": False,
        "made_on": None,
        "f5_read_on": None,
        "cannot_tell": None,
    }
    if (arm, round_number, fold) != RULES_FOLD_RUN:
        return readings
    if earlier is None:
        raise ValueError("a run of CPG's round 1, fold 1 needs the record of the others")
    if earlier.unreadable:
        readings["cannot_tell"] = f"the run.json of {', '.join(earlier.unreadable)} cannot be read"
        return readings
    if earlier.undecided:
        readings["cannot_tell"] = (
            f"{', '.join(earlier.undecided)} finished unable to tell where the readings were made"
        )
        return readings
    readings.update(made_on=earlier.made_on, f5_read_on=earlier.f5_read_on)
    if dirty:
        return readings
    if earlier.made_on is None:
        readings["reads_rules"] = not released
    else:
        readings["reads_f5"] = (
            earlier.rules_segments == FULL_SEGMENTS
            and earlier.rules_f5_passes is False
            and segments == FALLBACK_SEGMENTS
            and earlier.f5_read_on is None
        )
    return readings


def rules_lines(
    measures: Mapping[str, Any],
    readings: Mapping[str, Any],
    *,
    rules_fold_run: bool = False,
    withheld: str | None = None,
) -> list[str]:
    """The log lines of check F1, the gain-alive share, the 4-epoch rule and the hours of F5.

    A verdict appears only where the run's measures hold one (``rule_readings``). A run of
    CPG's round 1, fold 1 (``rules_fold_run``) without them says where the rules and F5
    were read (``made_on``, and ``f5_read_on`` for F5 alone), or why it cannot tell
    (``cannot_tell``); if no run read them, why this one did not (``withheld``). Only the
    other runs say that the rules and F5 read CPG's round 1, fold 1 only.
    """
    made_on, f5_read_on = readings.get("made_on"), readings.get("f5_read_on")
    cannot_tell = readings.get("cannot_tell")
    if cannot_tell is not None:
        unread = f"is not read here: this run cannot tell which run read it, as {cannot_tell}"
    elif made_on is not None:
        unread = f"was read once, on {made_on}, and that reading stands"
    else:
        unread = f"is not read here, and no run that read it is found: {withheld}"
    f1 = measures["f1"]
    norm = f1["largest_grad_norm"]
    clipped = f1["clipped_share"]
    found = [
        f"F1: {'passes' if f1['passes'] else 'FAILS'}; {f1['skipped_steps']} of"
        f" {measures['steps']} steps skipped (longest run {f1['longest_skipped_run']}), lowest"
        f" loss scale {f1['lowest_scale']:g} (at least {f1['min_scale']:g}),"
        f" {f1['non_finite_losses']} losses not finite, largest gradient norm"
        f" {'none' if norm is None else f'{norm:.4g}'}, share of clipped steps"
        f" {'none' if clipped is None else f'{clipped:.3f}'}"
    ]
    alive = measures["alive"]
    if alive is not None:
        epochs = ", ".join(map(str, alive["epochs"]))
        if alive["share"] is None:
            line = f"gain-alive share over epochs {epochs}: no bag"
        else:
            line = (
                f"gain-alive share over epochs {epochs}: {alive['share']:.4f} ({alive['active']}"
                f" of {alive['bags']} bags)"
            )
        if alive.get("f2_passes") is not None:
            line += f": F2 {'passes' if alive['f2_passes'] else 'FAILS'} on this CPG run"
        if "margin_if_read" in alive:  # the rules run
            if alive["margin_if_read"] is not None:
                line += (
                    f"; the rule would set m to {alive['margin_if_read']} (now {alive['margin']})"
                )
        elif rules_fold_run:
            line += f"; the gain-alive rule {unread}"
        if "f2_passes" not in alive:
            line += " (check F2 reads CPG runs only)"
        found.append(line)
    falling = measures["falling"]
    if "fires" in falling:
        read = f": {'fires' if falling['fires'] else 'does not fire'}"
    elif rules_fold_run:
        read = f" (the rule {unread})"
    else:
        read = " (the rule reads CPG's round 1, fold 1 only)"
    if "ce_part" in falling:
        share = falling["drop_share"]
        halves = falling["ce_part"]
        found.append(
            f"4-epoch rule on epoch {falling['epoch']}: the CE part's mean over halves of"
            f" {falling['half_steps']} steps, {halves['first_half']:.4f} then"
            f" {halves['second_half']:.4f}, a drop of {falling['drop']:.4g}"
            f" ({'none' if share is None else f'{share:.2%}'} of the first half; standard error"
            f" of the difference {falling['standard_error']:.4g}){read}"
        )
    else:
        found.append(
            f"4-epoch rule on epoch {falling['epoch']}: halves of {falling['half_steps']} steps,"
            f" too few to read{read}"
        )
    f5 = measures["f5"]
    if f5 is not None and "passes" in f5:
        added = f" (4/3 of {f5['training_hours']:.3f} h)" if f5["fourth_epoch"] else ""
        if readings.get("reads_f5"):
            added = (
                f" (this run's own hours, at {FALLBACK_SEGMENTS} segments after F5 failed at"
                f" {FULL_SEGMENTS} on {made_on})"
            )
        found.append(
            f"F5: {f5['hours']:.3f} h of training{added}, at most {f5['max_hours']:g} h:"
            f" {'passes' if f5['passes'] else 'FAILS'}"
        )
    elif f5 is not None:
        if not rules_fold_run:
            note = "check F5 reads CPG's round 1, fold 1 only"
        elif cannot_tell is not None:
            note = f"no F5 verdict here: this run cannot tell which run read F5, as {cannot_tell}"
        elif made_on is not None:
            note = f"no F5 verdict here: F5 was read on {made_on}"
            if f5_read_on is not None:
                note += f" and, at {FALLBACK_SEGMENTS} segments, on {f5_read_on}"
        else:
            note = f"no F5 verdict here, and no run that read F5 is found: {withheld}"
        found.append(f"training part of this three-view run: {f5['training_hours']:.3f} h ({note})")
    return found


def finish(run: Run, lines: Sequence[Mapping[str, Any]]) -> None:
    """Stage 8: the checks and rules read on metrics.jsonl; run.json (done)."""
    path = run.run_dir / METRICS
    back = read_metrics(path, lines)
    three_views = len(run.arm.views) == 3  # check F5 times a three-view run (B1 or CPG)
    hours = run.results["hours"]["training"] if three_views else None
    args = run.args
    rules_fold_run = (run.arm.arm, args.round, args.fold) == RULES_FOLD_RUN
    released = at_release(run.info["git"])
    readings = rule_readings(
        run.arm.arm,
        args.round,
        args.fold,
        earlier_readings(run.workspace, run.run_dir) if rules_fold_run else None,
        dirty=args.allow_dirty,
        released=released,
        segments=run.results["sampler"]["max_segments_per_epoch"],
    )
    measures = rule_measures(back, run.arm.shared, hours, **{key: readings[key] for key in READS})
    run.results["rule_readings"] = readings
    run.results["rules"] = measures
    run.info["outputs"][METRICS] = sha256_file(path)
    minutes = (time.monotonic() - run.started) / 60
    run.info.update(status="done", finished_utc=utc_now(), minutes=round(minutes, 2))
    run.save()
    withheld = None  # why a run of CPG's round 1, fold 1 made no reading where none was made
    if args.allow_dirty:
        withheld = "a run with --allow-dirty makes no reading"
    elif released:
        withheld = "a run at a release tag makes no first reading, as the settings froze after it"
    for line in rules_lines(measures, readings, rules_fold_run=rules_fold_run, withheld=withheld):
        logger.info(line)
    cap = run.results["sampler"]["max_segments_per_epoch"]
    logger.info(f"run: {run.relative}")
    logger.info(
        f"done in {minutes:.1f} min; trained with at most {cap} segments per participant per"
        " epoch (bags.max_segments_per_epoch)"
    )


# The timing run of check F5 ----------------------------------------------------------------


def choose_smoke_bag(
    manifest: pd.DataFrame, people: Sequence[int], labels: Mapping[int, int], size: int
) -> Bag:
    """The smoke bag: the longest x0 clip of a training participant and its three next-longest.

    Among the training participants with at least ``size`` segments, the one who holds the
    longest x0 clip gives it and its next-longest segments, longest first (ties by segment
    ID), so that the backward pass meets the largest call of the run before any trained
    run does (pilot protocol, section "Training", bullet "Check F5").
    """
    rows = manifest.loc[manifest["session"].isin(people), ["segment", "session", "samples"]]
    counts = rows["session"].value_counts()
    eligible = rows[rows["session"].map(counts) >= size]
    if eligible.empty:
        raise StepError(f"no training participant has the {size} segments of a smoke bag")
    ordered = eligible.sort_values(["samples", "segment"], ascending=[False, True], kind="stable")
    session = int(ordered["session"].iloc[0])
    chosen = ordered[ordered["session"] == session].head(size)
    return Bag(session, int(labels[session]), tuple(map(str, chosen["segment"])))


def smoke_bag_pass(run: Run, setup: Setup, bag: Bag) -> dict[str, Any]:
    """The smoke bag forward and backward in every view, at loss scale 1.0, with no update.

    The loss is that of one bag (training.train_step), with λG at its peak so that the
    gain's backward pass runs too; each view's call per chunk count runs under fp16
    autocast and its backward pass with the watch paused. Then every gradient must be
    finite (pilot protocol, section "Training", bullet "Check F1"); the gradients are
    dropped again.
    """
    arm, model = run.arm, setup.trainable.model
    shared = arm.shared
    weights = {"x0": 1.0, "x1": shared.loss.x1_weight, "x2": shared.loss.x2_weight}
    device = str(next(model.parameters()).device.type)
    model.zero_grad(set_to_none=True)
    calls = 0
    two_chunk = False
    gain_term = False  # whether the gain term entered the loss, so that its backward ran
    z1_bar = None
    for view in arm.views:
        outputs = [setup.read(view, segment) for segment in bag.segments]
        two_chunk = two_chunk or any(int(output.shape[0]) > 1 for output in outputs)
        groups: dict[int, list[int]] = {}
        for index, output in enumerate(outputs):
            groups.setdefault(int(output.shape[0]), []).append(index)
        found: dict[int, torch.Tensor] = {}
        for count in sorted(groups):  # one call per chunk count, nothing padded
            with torch.autocast(device, dtype=torch.float16):
                z = cached_z(
                    model,
                    setup.prompts[count],
                    [outputs[i] for i in groups[count]],
                    setup.answer_ids,
                )
            calls += 1
            for position, index in enumerate(groups[count]):
                found[index] = z[position]
        scores = torch.stack([found[index] for index in range(len(outputs))])
        if not bool(torch.isfinite(scores).all()):
            raise StepError(f"the smoke bag's z on {view} are not all finite: check F1 fails")
        term = weights[view] * ce_on_z(scores, bag.label)
        if view == "x1":
            z1_bar = scores.detach().mean()
        elif view == "x0" and setup.stats is not None and arm.lambda_g:
            if z1_bar is None:
                raise StepError("the smoke bag's gain needs x1 before x0")
            g = gain(
                scores.mean().reshape(1),
                z1_bar.reshape(1),
                [bag.label],
                setup.stats.x0,
                setup.stats.x1,
            )
            term = term + arm.lambda_g * soft_hinge(g, shared.gain.margin, shared.gain.softness)[0]
            gain_term = True
        if not bool(torch.isfinite(term)):
            raise StepError(f"the smoke bag's loss on {view} is not finite: check F1 fails")
        with setup.watch.pause():
            term.backward()
    flagged = setup.watch.flagged()
    if flagged:
        raise StepError(
            f"the overflow watch flagged {len(flagged)} parts in the smoke bag: the timing run"
            " goes again with the pre-set fallback, the flagged layers in fp32 (pilot protocol,"
            ' section "Training", bullet "Overflow in training")'
        )
    grads = [value.grad for value in model.parameters() if value.requires_grad]
    present = [grad for grad in grads if grad is not None]
    finite = all(bool(torch.isfinite(grad).all()) for grad in present)
    norms = [float(torch.linalg.vector_norm(grad.float())) for grad in present]
    model.zero_grad(set_to_none=True)
    return {
        "segments": len(bag.segments),
        "two_chunk": two_chunk,
        "views": list(arm.views),
        "calls": calls,
        "lambda_g": arm.lambda_g,
        "gain_term": gain_term,
        "loss_scale": 1.0,
        "trained_tensors": len(grads),
        "with_gradient": len(present),
        "all_finite": finite,
        "largest_norm": max(norms) if finite and norms else None,
        "joint_norm": math.sqrt(sum(norm * norm for norm in norms)) if finite and norms else None,
    }


def seconds_text(value: float | None) -> str:
    """Seconds for a log line, or none."""
    return "none" if value is None else f"{value:.3f} s"


def time_mode(
    run: Run, setup: Setup, mode: str, steps: Sequence[tuple[int, tuple[Bag, ...]]], first: int
) -> tuple[dict[str, Any], list[float]]:
    """Time the given steps in one repeatability mode; an error is strict mode's outcome.

    Returns:
        The mode's reading for run.json (aggregates only) and the steps' seconds.
    """
    threads = CacheSettings.from_config(run.config).cpu_threads
    run.info["settings"]["numerics"][mode] = training_numerics(threads, mode)
    seconds: list[float] = []
    reads: list[float] = []
    records = []
    outcome = "ran"
    try:
        for number, (epoch, bags) in enumerate(steps):
            before = setup.read.seconds
            record = train_step(
                setup.trainable,
                setup.optimiser,
                setup.watch,
                run.arm,
                bags,
                setup.read,
                setup.prompts,
                setup.stats,
                first + number,
                epoch,
            )
            setup.stats = record.stats
            records.append(record)
            seconds.append(record.seconds)
            reads.append(setup.read.seconds - before)
    except RuntimeError as error:  # PyTorch's refusal of an operation with no repeatable version
        if mode != STRICT:
            raise
        lines = str(error).strip().splitlines()
        outcome = f"raised {type(error).__name__}: {lines[0][:300] if lines else ''}"
    left_out = run.arm.shared.timing.first_steps_left_out
    timed, read = seconds[left_out:], reads[left_out:]
    reading = {
        "steps": len(seconds),
        "outcome": outcome,
        "median_seconds": statistics.median(timed) if timed else None,
        "largest_seconds": max(timed) if timed else None,
        "median_read_seconds": statistics.median(read) if read else None,
        "skipped": sum(record.skipped for record in records),
        "lowest_scale": min(
            (min(record.scale, record.next_scale) for record in records), default=None
        ),
    }
    return reading, seconds


def repeatability_mode(
    strict_runs: bool,
    strict_median: float | None,
    warn_median: float | None,
    extra_share: float,
) -> str:
    """Return the repeatability mode that the timing run's rule gives.

    Strict mode if it ran without an error and its median step takes at most
    ``extra_share`` (``precision.strict_max_extra_time_share``: 10%) longer than in
    warn-only mode, otherwise warn-only (pilot protocol, section "Training", bullet "Number
    formats and repeatability"). The times are compared as exact fractions, so a median
    exactly 10% longer still gives strict mode.
    """
    if not strict_runs or strict_median is None or warn_median is None:
        return WARN_ONLY
    limit = 1 + Fraction(repr(extra_share))
    return STRICT if Fraction(strict_median) <= limit * Fraction(warn_median) else WARN_ONLY


def timing(run: Run, pins: Pins, setup: Setup) -> None:
    """The timing run of check F5: the smoke bag, then the sampler's first steps in each mode."""
    shared = run.arm.shared
    settings = shared.timing
    synchronise()
    load_seconds = time.monotonic() - run.started  # the job's start until the inputs are loaded
    threads = CacheSettings.from_config(run.config).cpu_threads
    bag = choose_smoke_bag(
        pins.caches["x0"].manifest,
        setup.people,
        setup.labels,
        BagSettings.from_config(run.config).segments_per_bag,
    )
    run.info["settings"]["numerics"]["smoke_bag"] = training_numerics(threads, WARN_ONLY)
    smoke = smoke_bag_pass(run, setup, bag)
    run.results["smoke_bag"] = smoke
    run.save()
    logger.info(
        f"smoke bag: {smoke['segments']} segments of one training participant, the longest x0"
        f" clip among them ({'with' if smoke['two_chunk'] else 'without'} a two-chunk clip),"
        f" views {', '.join(smoke['views'])}, {smoke['calls']} calls, λG {smoke['lambda_g']}"
        f" (gain term {'in' if smoke['gain_term'] else 'not in'} the loss),"
        f" loss scale 1.0, no update: {smoke['with_gradient']} of {smoke['trained_tensors']}"
        f" trained tensors with a gradient, all finite {smoke['all_finite']}; largest norm"
        f" {smoke['largest_norm']}, joint norm {smoke['joint_norm']}"
    )
    if not smoke["all_finite"]:
        raise StepError(
            "the smoke bag's gradients at loss scale 1.0 are not all finite: check F1 fails; a"
            " fault to find and fix before any training run"
        )
    count = settings.steps_per_mode
    planned = [
        (epoch, bags)
        for epoch in range(1, shared.epochs + 1)
        for bags in setup.sampler.epoch(epoch).steps
    ]
    if len(planned) < 2 * count:
        raise StepError(f"the run has {len(planned)} steps, fewer than the {2 * count} timed")
    total = setup.total_steps
    left_out = settings.first_steps_left_out
    modes: dict[str, Any] = {}
    projection: dict[str, float] = {}
    for number, mode in enumerate((WARN_ONLY, STRICT)):
        first = number * count
        reading, seconds = time_mode(run, setup, mode, planned[first : first + count], first)
        if reading["outcome"] == "ran":
            projection[mode] = f5_projection(load_seconds, seconds, total, left_out)
        modes[mode] = reading
        run.results["modes"] = modes
        run.save()
        logger.info(
            f"{mode}: {reading['steps']} steps, {reading['outcome']}; median of steps"
            f" {left_out + 1} to {reading['steps']} {seconds_text(reading['median_seconds'])},"
            f" largest {seconds_text(reading['largest_seconds'])}, cache reads"
            f" {seconds_text(reading['median_read_seconds'])} per step; {reading['skipped']}"
            f" skipped, lowest loss scale {reading['lowest_scale']}"
        )
    warn, strict = modes[WARN_ONLY], modes[STRICT]
    strict_runs = strict["outcome"] == "ran"
    mode_by_rule = repeatability_mode(
        strict_runs,
        strict["median_seconds"],
        warn["median_seconds"],
        shared.precision.strict_max_extra_time_share,
    )
    hours = projection[mode_by_rule]
    fallback = hours > shared.rules.f5_max_hours
    synchronise()
    run.results.update(
        load_seconds_from_start=round(load_seconds, 1),
        strict_runs=strict_runs,
        mode_by_rule=mode_by_rule,
        projection={
            "total_steps": total,
            "epochs": shared.epochs,
            "hours": projection,
            "hours_in_mode": hours,
            "max_hours": shared.rules.f5_max_hours,
            "fallback_applies": fallback,
        },
        peak_gpu_memory_gb=peak_memory_gb(),
        cache_read_seconds=round(setup.read.seconds, 2),
        lm_watch=watch_summary(setup.watch),
    )
    minutes = (time.monotonic() - run.started) / 60
    run.info.update(status="done", finished_utc=utc_now(), minutes=round(minutes, 2))
    run.save()
    logger.info(
        f"the mode by the rule (strict if it runs and its median step takes at most"
        f" {shared.precision.strict_max_extra_time_share:.0%} longer than in warn_only):"
        f" {mode_by_rule}"
    )
    projected = "; ".join(f"{value:.3f} h in {mode}" for mode, value in projection.items())
    logger.info(
        f"projection of a three-view run (B1, CPG; T {total} for {shared.epochs} epochs, load"
        f" {load_seconds:.0f} s): {projected}; in {mode_by_rule} {hours:.3f} h against F5's"
        f" {shared.rules.f5_max_hours:g} h: the 8-segment fallback"
        f" {'applies' if fallback else 'does not apply'}"
    )
    summary = run.results["lm_watch"]
    largest = "; ".join(f"{name} {value:.6g}" for name, value in summary["largest"])
    logger.info(
        f"peak GPU memory {run.results['peak_gpu_memory_gb']} GB; watch:"
        f" {len(summary['flagged'])} of {summary['parts']} parts flagged; largest |value|:"
        f" {largest or 'none'}"
    )
    logger.info(f"run: {run.relative}")
    logger.info(f"done in {minutes:.1f} min; no weights kept, no score written")


# The job -------------------------------------------------------------------------------------


def set_numerics_and_mode(run: Run, pins: Pins) -> None:
    """Stage 3: the caches' numeric settings, compared; then training's repeatability mode.

    The mode is that of precision.deterministic, strict or warn_only; pending is refused
    (training.training_numerics), except by the timing run, which times both modes.
    """
    numerics = caches_numerics(run, pins.caches)
    recorded = run.info["settings"]["numerics"]
    recorded["caches"] = numerics
    logger.info("numerics: " + ", ".join(f"{key} {value}" for key, value in numerics.items()))
    if not run.args.timing:
        threads = CacheSettings.from_config(run.config).cpu_threads
        recorded["training"] = training_numerics(threads, run.arm.shared.precision.deterministic)
        logger.info(f"repeatability mode of training: {run.arm.shared.precision.deterministic}")
    run.save()


def train_job(state: Progress, started: float) -> None:
    """The job: stages 1 to 8, or the timing run."""
    run = start(state, started)
    pins = read_pins(run)  # 2
    set_numerics_and_mode(run, pins)  # 3
    setup = set_up(run, pins)  # 4
    run.info["settings"]["numerics"]["attention"] = attention_record(setup.trainable.model)
    if run.args.timing:
        timing(run, pins, setup)
        return
    lines = train(run, setup)  # 5
    save_and_release(run, setup)  # 6
    score(run, pins, setup, state)  # 7
    finish(run, lines)  # 8


def main() -> None:
    started = time.monotonic()  # the job's start, from which check F5 times training
    os.umask(0o077)
    sys.unraisablehook = quiet_unraisable  # a finaliser's message could hold anything
    set_up_logging()
    progress = Progress()
    try:
        try:
            train_job(progress, started)
            return
        except BaseException as caught:
            error = caught  # reported below, outside this clause, so that nothing chains to it
        with contextlib.suppress(BaseException):  # nothing here may print the error's message
            failure = record_failure(error, progress)
            print(failure.line)
            if failure.frames is not None:  # no errors.log holds them: frames and types only
                print(failure.frames, end="", file=sys.stderr)
        raise SystemExit(1)
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    main()
