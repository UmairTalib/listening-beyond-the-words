"""Probe Voxtral's frozen audio encoder for the voice and the label (CPU job).

Submit from an OMNI login node, from a checkout at the newest v0.1.x tag with its environment
built at that tag (git checkout v0.1.x, then bash scripts/setup_omni_env.sh), once:
    scripts/slurm/submit.sh probe_encoder
The job takes no option: everything it reads is pinned.

docs/pilot_diagnostics.md, section "Pooled-encoder probe", fixes the probe, and
configs/pilot/probe.yaml holds its settings (probe.ProbeSettings). The labels, partitions,
word times, x0 clips and encoder caches are pinned in configs/pilot/data.yaml, which also
holds the opensmile settings of the voice measures, and the Voxtral that built the caches
in configs/pilot/models.yaml. The probe is descriptive: no rule of the pilot reads it.

Stages, in one process:
  1. Guards: a Slurm job; a clean checkout at the newest v0.1.x tag of the clone; the
     installed code of that tag's version (runinfo.release_guard); probe.yaml read, a key
     given twice refused (config.read_settings), every value checked. Then the run folder
     and run.json (running).
  2. The pinned labels with their PHQ-8 totals, before anything else of the workspace is
     read: every participant one of the 189 pilot sessions and every label a total of 10
     or more (evaluation.checked_labels); then the pinned partitions of both rounds.
  3. The bootstrap's draws (evaluation.draw_counts), by the pilot analysis's rule from the
     probe's own stream of seed 111; they need only the labels and the partitions. run.json
     records their SHA-256 and the redraws.
  4. The other pins, before any clip or encoder output is read: the word times
     (clips.load_cut_words, segments.segment_parts), the x0 clips' manifest
     (cache.pinned_clips) and the cache runs of x0, x2 and rev with their checks
     (cache.pinned_cache). Every segment must have its 40 words and its parts, the same
     parts in all three views, and every cache the same width.
  5. The targets of every segment, before any encoder output is read or any probe fitted:
     each x0 clip read against its manifest row (clips.read_view_clip) and measured
     (probe.segment_targets): pitch variability and the 88 eGeMAPS measures with opensmile,
     speech rate and pause ratio from the word times, loudness from the stored samples.
  6. The encoder outputs of x0, x2 and rev (cache.read_file), each pooled over the vectors
     that start inside the segment's parts (probe.kept_vectors, probe.pooled).
  7. The probes on round 1's five folds and their controls (probe.ridge_folds,
     probe.logistic_folds): per view, a ridge regression of each target, pooled and within
     gender, and a logistic regression of the label; gender alone, length alone and, for
     the label, the 88 eGeMAPS measures of x0. Every number on each resample of the draws,
     with the fitted probes fixed, and the paired differences.
  8. probe.csv (reports.write_table), the reading and its limits printed, then run.json
     (done).

Writes $LBW_WORKSPACE/derived/probe/run-<job id>/ with
  probe.csv  per view, outcome, model (the encoder, a control or a difference) and kind of
             R²: the five fold values, their mean, the 80% and 95% percentile intervals,
             what was left out, and the penalty chosen in each fold with how often it lay
             at a grid end or on a plateau, the grid values skipped and the inner fits that
             did not converge; rev's rows say post hoc
  run.json   commit and tags, the release and the installed version, the configs and
             their SHA-256, the pins, the draws' seed, stream, redraws and SHA-256, the
             environment and the Slurm job, and aggregate results: the counts of
             participants, segments, parts and what was left out, the encoder's width and
             the kept vectors, the time of each stage, the reading and the table's SHA-256
Features, targets and predictions stay in memory: no file holds a number per segment or
participant. A cell that cannot exist reads "not applicable". A job that fails also leaves
errors.log there: traceback frames and error types, never a message. The job prints and
records aggregate numbers only: no participant ID and no number per participant or
segment. A refusal names a rule, a file and counts; a clip or cache file that does not
match its manifest row is named by its segment, on the one failure line.
"""

import contextlib
import hashlib
import os
import socket
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from listening_beyond_words.cache import PinnedCache, pinned_cache, pinned_clips, read_file
from listening_beyond_words.clips import (
    load_cut_words,
    load_pinned_labels,
    load_pinned_partitions,
    read_view_clip,
)
from listening_beyond_words.config import read_settings
from listening_beyond_words.evaluation import checked_labels, draw_counts, draws_sha256, interval
from listening_beyond_words.labels import GENDERS, PHQ8_CUTOFF
from listening_beyond_words.models import ModelPin, load_pins
from listening_beyond_words.probe import (
    BOOTSTRAP_KEYS,
    KINDS,
    LABEL,
    PITCH,
    POST_HOC,
    TARGET_VIEW,
    TARGETS,
    VIEWS,
    Choice,
    FoldFit,
    ProbeSettings,
    TargetSettings,
    fold_r2,
    kept_vectors,
    logistic_folds,
    pooled,
    resampled_auroc,
    resampled_r2,
    ridge_folds,
    segment_targets,
)
from listening_beyond_words.reports import write_table
from listening_beyond_words.runinfo import (
    PACKAGE,
    Progress,
    StepError,
    environment,
    git_state,
    installed_version,
    item_seed,
    quiet_unraisable,
    record_failure,
    release_guard,
    sha256_file,
    stop_on_sigterm,
    utc_now,
    write_json,
)
from listening_beyond_words.segments import segment_parts
from listening_beyond_words.settings import Workspace, get_workspace
from listening_beyond_words.voicecheck import (
    FEMALE,
    FUNCTIONAL_NAMES,
    PITCH_WINDOW_SAMPLES,
    Smile,
)

CONFIG = Path("configs/pilot/data.yaml")
PROBE = Path("configs/pilot/probe.yaml")
MODELS = Path("configs/pilot/models.yaml")
STEP = "probe_encoder"
FOLDER = "probe"  # derived/probe/run-<job id>/
TABLE = "probe.csv"
PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "torch",
    "opensmile",
    "soundfile",
    "pyarrow",
    "pyyaml",
    PACKAGE,
)
USAGE = "usage: probe_encoder.py (no arguments)"
WHAT = "the pooled-encoder probe"
# The rule that the probe runs only from its release (runinfo.release_guard).
ORDER = 'pilot diagnostics, section "Order"'
NOT_APPLICABLE = "not applicable"
PROGRESS_EVERY = 500  # clips between two progress lines
# The models of the table, as its column "model" names them.
ENCODER = "encoder"
GENDER_ONLY = "gender_only"
LENGTH_ONLY = "length_only"
EGEMAPS = "egemaps88"
MINUS = "_minus_"  # a difference: the first minus the second
R2 = "r2"
AUROC = "auroc"
RIDGE = "ridge alpha"
LOGISTIC = "logistic C"
LEFT_OUT = ("segments_left_out", "parts_short", "parts_unmeasured", "means_filled")
# Printed with the table (pilot diagnostics, section "Pooled-encoder probe", bullet "Reading
# and limits"); the numbers for scale are cited there and not computed here.
READING = (
    "reading: timing and loudness are expected to be predicted well, so the reading rests on"
    " pitch variability within gender and on the label probe",
    "limits: pooled features give only a lower bound, because the language model reads the"
    " whole sequence; the planted-cue check that covers this is left for the thesis",
    "limits: padding and gaps are left out",
    "for scale, cited and not recomputed: probes on mean-pooled features of similar encoders"
    " reached 0.54 to 0.56 AUROC on DAIC-WOZ, against 0.75 for 119 voice features"
    " (Perez-Toro et al. 2026, https://arxiv.org/abs/2610.06465)",
    "for scale, cited and not recomputed: the arms' AUROCs in round 1, on the same folds"
    " (x0: zero-shot 0.7220, B0 0.6330, B1 0.6240, CPG 0.6791; rev: 0.5177, 0.5174, 0.5384,"
    " 0.5435; reports/pilot/summary.csv)",
    "post hoc: rev was added after the pilot's AUROCs were seen; its rows say so, and no rule"
    " reads them",
)


def utc_in(seconds: float) -> str:
    """The UTC time this many seconds from now, to the minute."""
    return (datetime.now(UTC) + timedelta(seconds=seconds)).strftime("%H:%M UTC")


def level_name(level: float) -> str:
    """A level as the columns name it: 80 for 0.8."""
    return f"{level * 100:g}"


def refused(what: str, error: Exception) -> StepError:
    """A reader's refusal as the job's: a plain ValueError keeps its message, others their type."""
    # The readers' own checks raise a plain ValueError, with a message of rules and counts;
    # any other error is named by its type.
    reason = str(error) if type(error) is ValueError else type(error).__name__
    return StepError(f"{what} do not hold: {reason}")


# Stage 1: the job ---------------------------------------------------------------------------------


@dataclass
class Job:
    """The probe being made: its settings, inputs, folder and record."""

    job: str
    workspace: Workspace
    config: dict[str, Any]
    settings: ProbeSettings
    targets: TargetSettings
    voxtral: ModelPin
    run_dir: Path
    info: dict[str, Any]
    started: float

    def save(self) -> None:
        """Write run.json as it stands."""
        write_json(self.run_dir / "run.json", self.info)

    @property
    def results(self) -> dict[str, Any]:
        """The results recorded so far."""
        return self.info["results"]

    @property
    def relative(self) -> str:
        """The run folder, relative to the workspace."""
        return self.run_dir.relative_to(self.workspace.root).as_posix()

    def stage(self, name: str, began: float) -> float:
        """Record how long a stage took; return its minutes."""
        minutes = (time.monotonic() - began) / 60
        self.results.setdefault("minutes", {})[name] = round(minutes, 2)
        self.save()
        return minutes


def target_settings(config: Mapping[str, Any], settings: ProbeSettings) -> TargetSettings:
    """The settings of the voice measures that the targets share with the voice checks."""
    try:
        voice = config["word_check"]["voice"]
        level_rms, min_samples = voice["level_rms"], voice["min_part_samples"]
        peak_limit = config["views"]["peak_limit"]
    except (KeyError, TypeError):
        raise StepError(
            "data.yaml holds no word_check.voice.level_rms, word_check.voice.min_part_samples"
            " or views.peak_limit, which the targets read"
        ) from None
    if (
        not isinstance(level_rms, float)
        or not 0.0 < level_rms < 1.0
        or not isinstance(peak_limit, float)
        or not 0.0 < peak_limit < 1.0
        or isinstance(min_samples, bool)
        or not isinstance(min_samples, int)
        or min_samples < PITCH_WINDOW_SAMPLES
    ):
        raise StepError(
            "data.yaml: word_check.voice.level_rms and views.peak_limit must lie between 0 and"
            f" 1, and word_check.voice.min_part_samples must be at least {PITCH_WINDOW_SAMPLES}"
        )
    return TargetSettings(settings.words, level_rms, peak_limit, min_samples)


def start(state: Progress) -> Job:
    """Stage 1: refuse to run outside a clean Slurm job at the newest release; the folder."""
    stop_on_sigterm()
    if sys.argv[1:]:
        raise StepError(USAGE)
    started = time.monotonic()
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise StepError("submit this as a job: scripts/slurm/submit.sh probe_encoder")
    git = git_state()
    if git["dirty"]:
        raise StepError(
            f"the checkout has uncommitted changes: {WHAT} runs only from a release ({ORDER})"
        )
    installed = installed_version()
    release = release_guard(git, installed, WHAT, ORDER)
    content, probe_sha256 = read_settings(PROBE)
    settings = ProbeSettings.from_config(content)
    data = CONFIG.read_bytes()
    config = yaml.safe_load(data.decode("utf-8"))
    measured = target_settings(config, settings)
    voxtral = load_pins(MODELS)["voxtral"]
    workspace = get_workspace()
    run_dir = workspace.derived / FOLDER / f"run-{job}"
    run_dir.mkdir(parents=True, exist_ok=False)
    state.run_dir = run_dir
    machine = environment(PACKAGES)
    info: dict[str, Any] = {
        "run_id": run_dir.name,
        "step": STEP,
        "status": "running",
        "started_utc": utc_now(),
        "git": git,
        "release": release,
        "installed_version": installed,
        "config": {"file": str(PROBE), "sha256": probe_sha256, "content": content},
        "data_config": {"file": str(CONFIG), "sha256": hashlib.sha256(data).hexdigest()},
        "models_config": {"file": str(MODELS), "sha256": sha256_file(MODELS)},
        "settings": {"targets": measured._asdict()},
        "slurm": {
            "job_id": job,
            "node": socket.gethostname(),
            "cpu": machine["cpu"],
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
        "seeds": {},
        "inputs": {},
        "environment": machine,
        "results": {},
        "outputs": {},
    }
    state.info = info
    run = Job(job, workspace, config, settings, measured, voxtral, run_dir, info, started)
    run.save()
    print(
        f"{WHAT} at the release tag {release} (installed version {installed}), commit"
        f" {str(git['commit'])[:12]}; job {job} on {socket.gethostname()}; CPU {machine['cpu']}"
    )
    return run


# Stage 2: the participants ------------------------------------------------------------------


@dataclass(frozen=True)
class People:
    """The pilot's participants: labels, gender and partitions, checked.

    Attributes:
        labels: Each participant's label, 0 or 1, by session ID in ascending order.
        female: Whether each participant is a woman, in the same order.
        partitions: One row per participant and round: ``session``, ``round``, ``fold``.
        folds: Each participant's test fold in the probes' round, in the same order.
    """

    labels: pd.Series
    female: np.ndarray
    partitions: pd.DataFrame
    folds: np.ndarray


def read_people(job: Job) -> People:
    """Stage 2: the labels with their totals, checked first; then the partitions."""
    for name in ("labels", "partitions"):
        pin = job.config.get(name)
        if not isinstance(pin, Mapping) or not pin.get("run") or not pin.get("sha256"):
            raise StepError(f"no {name} are pinned ({name} in data.yaml)")
    error: Exception | None = None
    try:
        table = load_pinned_labels(job.workspace, dict(job.config), phq8_total=True)
    except (ValueError, OSError, KeyError) as caught:
        error = caught
    if error is not None:  # raised here, so that nothing chains to the reader's error
        raise refused("the pinned labels", error)
    labels, _ = checked_labels(table)  # before anything else of the workspace is read
    genders = table["gender"].set_axis(table.index.astype(np.int64)).reindex(labels.index)
    unknown = int((~genders.isin(sorted(GENDERS))).sum())
    if unknown:
        raise StepError(f"{unknown} participants have a gender other than {sorted(GENDERS)}")
    try:
        partitions = load_pinned_partitions(job.workspace, dict(job.config))
    except (ValueError, OSError, KeyError) as caught:
        error = caught
    if error is not None:
        raise refused("the pinned partitions", error)
    chosen = partitions[partitions["round"] == job.settings.partition_round]
    if not len(chosen):
        raise StepError(
            f"the pinned partitions hold no round {job.settings.partition_round}"
            " (probe.yaml, partition_round)"
        )
    folds = chosen.set_index("session")["fold"].reindex(labels.index)
    people = People(
        labels,
        (genders == FEMALE).to_numpy(dtype=bool),
        partitions,
        folds.to_numpy(dtype=np.int64),
    )
    job.info["inputs"].update(
        labels=dict(job.config["labels"]), partitions=dict(job.config["partitions"])
    )
    label = labels.to_numpy(dtype=np.int64)
    positives = int(label.sum())
    record: dict[str, Any] = {
        "participants": len(label),
        "positives": positives,
        "negatives": len(label) - positives,
        "women": int(people.female.sum()),
        "folds": {
            f"f{fold}": {
                "positives": int(label[people.folds == fold].sum()),
                "negatives": int((label[people.folds == fold] == 0).sum()),
            }
            for fold in sorted(set(people.folds.tolist()))
        },
    }
    job.results["counts"] = record
    job.save()
    print(
        f"participants: {record['participants']} in all, {positives} positive and"
        f" {record['negatives']} negative, {record['women']} women; every label is a PHQ-8"
        f" total of {PHQ8_CUTOFF} or more"
    )
    shown = "; ".join(
        f"fold {key[1:]}: {value['positives']} positive and {value['negatives']} negative"
        for key, value in record["folds"].items()
    )
    print(f"round {job.settings.partition_round}, whose folds the probes use: {shown}")
    return people


# Stage 3: the draws -------------------------------------------------------------------------


def draw(job: Job, people: People) -> np.ndarray:
    """Stage 3: the bootstrap's draws, from the labels and the partitions only."""
    settings = job.settings
    try:
        counts, redraws = draw_counts(
            people.labels, people.partitions, settings.rng(), settings.resamples
        )
    except ValueError as error:
        raise StepError(f"the bootstrap cannot draw: {error}") from None
    digest = draws_sha256(counts)
    job.info["seeds"]["bootstrap"] = {
        "seed": settings.seed,
        "stream": settings.stream,
        "item_seed": item_seed(settings.seed, *BOOTSTRAP_KEYS),
        "resamples": settings.resamples,
        "redraws": redraws,
        "draws_sha256": digest,
    }
    job.save()
    positives = int(people.labels.sum())
    print(
        f"bootstrap: {settings.resamples} resamples of {positives} positive and"
        f" {len(people.labels) - positives} negative draws within the PHQ-8 classes, from"
        f" {settings.stream}; {redraws} resamples drawn again (a test fold without one"
        f" class); draws' SHA-256 {digest}"
    )
    # One column per participant in ascending session ID, as the labels (draw_counts).
    return counts.to_numpy(dtype=np.int64)


# Stage 4: the pins ----------------------------------------------------------------------------


@dataclass
class Inputs:
    """The pinned inputs of the targets and the pooling, checked against each other.

    Attributes:
        segments: Every segment of x0's clips, in ascending order.
        owners: Each segment's participant, as a place among the labelled participants.
        clips: x0's manifest rows by segment.
        words: Each segment's word times, per part in clip order: starts and ends.
        caches: The pinned cache of each view, in the order of ``VIEWS``.
        width: The encoder's width, from the cache runs' records.
    """

    segments: list[str]
    owners: np.ndarray
    clips: dict[str, Any]
    words: dict[str, list[tuple[np.ndarray, np.ndarray]]]
    caches: dict[str, PinnedCache]
    width: int


def part_words(
    words: pd.DataFrame, parts: pd.DataFrame
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Each segment's word times, part by part in clip order, each part's words in time order."""
    keys = ["session", "line", "word_index"]
    table = words.loc[:, [*keys, "start", "end"]].astype(
        {"session": np.int64, "line": np.int64, "word_index": np.int64}
    )
    indexed = table.set_index(keys)
    if indexed.index.has_duplicates:
        raise StepError("the pinned word times hold a word twice (session, line and index)")
    sizes = parts["n_words"].to_numpy(dtype=np.int64)
    within = np.arange(int(sizes.sum())) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    wanted = pd.MultiIndex.from_arrays(
        [
            np.repeat(parts["session"].to_numpy(dtype=np.int64), sizes),
            np.repeat(parts["line"].to_numpy(dtype=np.int64), sizes),
            np.repeat(parts["first"].to_numpy(dtype=np.int64), sizes) + within,
        ],
        names=keys,
    )
    times = indexed.reindex(wanted)
    if times.isna().to_numpy().any():
        raise StepError("the pinned word times lack a word of a segment's part")
    expanded = pd.DataFrame(
        {
            "segment": np.repeat(parts["segment"].astype(str).to_numpy(), sizes),
            "position": np.repeat(parts["position"].to_numpy(dtype=np.int64), sizes),
            "index": wanted.get_level_values("word_index").to_numpy(dtype=np.int64),
            "start": times["start"].to_numpy(dtype=np.float64),
            "end": times["end"].to_numpy(dtype=np.float64),
        }
    ).sort_values(["segment", "position", "start", "index"], kind="stable")
    found: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for (segment, _), part in expanded.groupby(["segment", "position"], sort=True):
        found.setdefault(str(segment), []).append(
            (part["start"].to_numpy(dtype=np.float64), part["end"].to_numpy(dtype=np.float64))
        )
    return found


def bounds_of(row: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """A manifest row's part bounds, as whole numbers."""
    starts = tuple(int(value) for value in row.part_starts)
    return starts, tuple(int(value) for value in row.part_ends)


def check_words(
    parts: pd.DataFrame, rows: Mapping[str, Any], segments: Sequence[str], words: int
) -> None:
    """Stop unless the word times give every segment of x0's clips, its parts and its words."""
    counted = parts.groupby("segment")["n_words"].agg(["size", "sum"])
    counted.index = counted.index.astype(str)
    owner = parts.drop_duplicates("segment").set_index("segment")["session"]
    owner.index = owner.index.astype(str)
    shared = [segment for segment in segments if segment in counted.index]
    differ = len(set(counted.index) ^ set(segments))
    wrong_parts = sum(
        int(counted.loc[segment, "size"]) != len(rows[segment].part_starts) for segment in shared
    )
    wrong_words = int((counted["sum"] != words).sum())
    wrong_owner = sum(int(owner[segment]) != int(rows[segment].session) for segment in shared)
    if differ or wrong_parts or wrong_words or wrong_owner:
        raise StepError(
            f"the pinned word times do not give the clips of {TARGET_VIEW}: {differ} segments"
            f" in only one of them, {wrong_parts} with another number of parts,"
            f" {wrong_words} without {words} words and {wrong_owner} of another participant"
        )


def check_caches(
    caches: Mapping[str, PinnedCache], rows: Mapping[str, Any], segments: Sequence[str]
) -> int:
    """Stop unless every view has x0's segments with x0's parts, and one encoder width."""
    widths = {
        view: (cache.run.info.get("settings") or {}).get("width") for view, cache in caches.items()
    }
    width = widths[TARGET_VIEW]
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width < 1
        or any(value != width for value in widths.values())
    ):
        raise StepError(f"the cache runs do not record one encoder width: {widths}")
    for view, cache in caches.items():
        own = {str(row.segment): row for row in cache.run.clips.itertuples(index=False)}
        missing = len(set(segments) - set(own))
        extra = len(set(own) - set(segments))
        moved = sum(
            bounds_of(own[segment]) != bounds_of(rows[segment])
            or int(own[segment].samples) != int(rows[segment].samples)
            for segment in segments
            if segment in own
        )
        if missing or extra or moved:
            raise StepError(
                f"the clips of {view} lack {missing} segments of {TARGET_VIEW}'s, hold {extra}"
                f" others, and {moved} differ from {TARGET_VIEW}'s in their parts or length:"
                " the probe keeps the same vectors in every view"
            )
    return width


def read_pins(job: Job, people: People) -> Inputs:
    """Stage 4: the word times, x0's clips and the caches, checked against each other."""
    workspace, config, settings = job.workspace, job.config, job.settings
    pin = config.get("segments")
    if not isinstance(pin, Mapping) or not pin.get("run") or not pin.get("sha256"):
        raise StepError("no segments are pinned (segments in data.yaml), whose word times it reads")
    error: Exception | None = None
    try:
        words = load_cut_words(workspace, dict(config))
        parts = segment_parts(words, settings.words)
    except (ValueError, OSError, KeyError) as caught:
        error = caught
    if error is not None:  # raised here, so that nothing chains to the reader's error
        raise refused("the pinned word times", error)
    clips = pinned_clips(workspace, config, TARGET_VIEW).sort_values("segment", ignore_index=True)
    segments = [str(segment) for segment in clips["segment"]]
    if len(set(segments)) != len(segments):
        raise StepError(f"the manifest of the clips of {TARGET_VIEW} holds a segment twice")
    sessions = [int(session) for session in clips["session"]]
    place = {int(session): index for index, session in enumerate(people.labels.index)}
    outside = sum(session not in place for session in sessions)
    if outside:
        raise StepError(
            f"the clips of {TARGET_VIEW} hold {outside} segments of sessions outside the"
            " pilot's labelled participants"
        )
    without = len(set(place) - set(sessions))
    if without:
        raise StepError(
            f"{without} labelled participants have no segment in the clips of {TARGET_VIEW}"
        )
    rows = {str(row.segment): row for row in clips.itertuples(index=False)}
    check_words(parts, rows, segments, settings.words)
    caches = {view: pinned_cache(workspace, config, view, job.voxtral) for view in VIEWS}
    width = check_caches(caches, rows, segments)
    inputs = Inputs(
        segments,
        np.array([place[session] for session in sessions], dtype=np.int64),
        rows,
        part_words(words, parts),
        caches,
        width,
    )
    job.info["inputs"].update(
        segments=dict(pin),
        clips={TARGET_VIEW: dict(config["view_runs"][TARGET_VIEW])},
        caches={view: dict(cache.pin) for view, cache in caches.items()},
    )
    two = int((caches[TARGET_VIEW].manifest["chunks"] > 1).sum())
    job.results["counts"].update(segments=len(segments), parts=len(parts), two_chunk_clips=two)
    job.results["width"] = width
    job.save()
    checks = ", ".join(sorted({cache.check for cache in caches.values()}))
    print(
        f"pins: {len(segments):,} segments of {len(place)} participants, each of"
        f" {settings.words} words, in {len(parts):,} parts; the caches of {', '.join(VIEWS)},"
        f" checked by {checks}, hold the same parts; clips of two chunks: {two}; encoder"
        f" width {width}"
    )
    return inputs


# Stage 5: the targets -------------------------------------------------------------------------


@dataclass
class Targets:
    """The targets of every segment, measured on x0, and what was left out of them.

    Attributes:
        values: Each target by name, one value per segment; NaN where it has none.
        measures: The 88 eGeMAPS measures of each segment, shape (segments, 88); NaN where
            no part has a value.
        short: Parts under the shortest length that opensmile measures, left out of pitch
            variability and the 88 measures.
        unmeasured: Parts that opensmile could not measure, left out of them too.
        overlaps: Pairs of consecutive words of a part that overlap: no gap.
    """

    values: dict[str, np.ndarray]
    measures: np.ndarray
    short: int
    unmeasured: int
    overlaps: int

    def left_out(self, name: str) -> int:
        """Segments without a value of a target."""
        return int(np.isnan(self.values[name]).sum())


def measure_targets(job: Job, inputs: Inputs, state: Progress) -> Targets:
    """Stage 5: every segment's targets on x0, before any encoder output is read."""
    began = time.monotonic()
    extractor = Smile.load()
    run_dir = job.workspace.resolve(job.config["view_runs"][TARGET_VIEW]["run"])
    total = len(inputs.segments)
    values = {name: np.full(total, np.nan) for name in TARGETS}
    measures = np.full((total, len(FUNCTIONAL_NAMES)), np.nan)
    short = unmeasured = overlaps = 0
    for index, segment in enumerate(inputs.segments):
        state.segment = segment
        clip = read_view_clip(run_dir, inputs.clips[segment])
        found = segment_targets(
            clip.samples,
            clip.part_starts,
            clip.part_ends,
            inputs.words[segment],
            extractor,
            job.targets,
            segment=segment,
        )
        for name in TARGETS:
            values[name][index] = found.values[name]
        measures[index] = found.measures.to_numpy(dtype=np.float64)
        short += found.short
        unmeasured += found.unmeasured
        overlaps += found.overlaps
        done = index + 1
        if done % PROGRESS_EVERY == 0 and done < total:
            elapsed = time.monotonic() - began
            print(
                f"targets: {done:,} of {total:,} segments measured; projected end"
                f" {utc_in(elapsed / done * (total - done))}"
            )
    state.segment = None
    del extractor
    targets = Targets(values, measures, short, unmeasured, overlaps)
    record = {
        "segments_left_out": {name: targets.left_out(name) for name in TARGETS},
        "parts_short": short,
        "parts_unmeasured": unmeasured,
        "word_overlaps": overlaps,
        "segments_without_any_measure": int(np.isnan(measures).all(axis=1).sum()),
    }
    job.results["targets"] = record
    minutes = job.stage("targets", began)
    print(
        f"targets: {total:,} segments of {TARGET_VIEW} measured in {minutes:.1f} min, on copies"
        f" at RMS {job.targets.level_rms}; parts left out of pitch variability and the 88"
        f" measures: {short} under {job.targets.min_samples} samples, {unmeasured} that"
        f" opensmile could not measure; consecutive words that overlap: {overlaps}"
    )
    print(
        "targets: segments without a value: "
        + ", ".join(f"{name} {count}" for name, count in record["segments_left_out"].items())
    )
    return targets


# Stage 6: the pooling ------------------------------------------------------------------------


def pool(job: Job, inputs: Inputs, state: Progress) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Stage 6: each view's encoder outputs, pooled over the vectors inside the parts."""
    began = time.monotonic()
    total = len(inputs.segments)
    features: dict[str, np.ndarray] = {}
    lengths: np.ndarray | None = None
    for view, cache in inputs.caches.items():
        rows = {str(row.segment): row for row in cache.manifest.itertuples(index=False)}
        matrix = np.empty((total, 2 * inputs.width))
        kept_counts = np.empty(total, dtype=np.int64)
        for index, segment in enumerate(inputs.segments):
            state.segment = segment
            row = rows[segment]
            output = read_file(Path(cache.run.folder) / str(row.file), row, inputs.width)
            starts, ends = bounds_of(inputs.clips[segment])
            kept = kept_vectors(starts, ends, int(row.chunks))
            matrix[index] = pooled(output.numpy(), kept)
            kept_counts[index] = len(kept)
        state.segment = None
        if lengths is None:
            lengths = kept_counts
        elif not np.array_equal(lengths, kept_counts):
            raise StepError(f"{view} keeps other vectors than {TARGET_VIEW}, with the same parts")
        features[view] = matrix
        print(f"pooling: {total:,} encoder outputs of {view} read and pooled")
    assert lengths is not None  # VIEWS is not empty
    job.results["kept_vectors"] = {
        "mean": float(lengths.mean()),
        "minimum": int(lengths.min()),
        "maximum": int(lengths.max()),
    }
    minutes = job.stage("pooling", began)
    print(
        f"pooling: {2 * inputs.width} features per segment and view in {minutes:.1f} min;"
        f" kept vectors per segment: mean {lengths.mean():.1f}, from {lengths.min()} to"
        f" {lengths.max()}"
    )
    return features, lengths


# Stage 7: the probes ---------------------------------------------------------------------------


@dataclass
class Row:
    """One row of the table: a probe, a control or a difference, with its numbers.

    Attributes:
        view: The view of the encoder features, a difference of two views, or "not
            applicable" for a control without them.
        outcome: A target or the label.
        model: ``ENCODER``, a control, or a probe minus a control.
        kind: The kind of R², or "not applicable" for the label.
        number: ``R2`` or ``AUROC``.
        folds: The value in each test fold, folds in ascending order.
        resampled: The mean over the folds in each resample, shape (R,).
        left: What was left out (``LEFT_OUT``).
        choices: The penalty chosen in each fold; None for a difference.
        penalty: What the penalty is (``RIDGE`` or ``LOGISTIC``), or "not applicable".
    """

    view: str
    outcome: str
    model: str
    kind: str
    number: str
    folds: list[float]
    resampled: np.ndarray
    left: dict[str, Any]
    choices: list[Choice] | None
    penalty: str

    @property
    def post_hoc(self) -> bool:
        """Whether a view of the row was added after the pilot's AUROCs were seen."""
        return any(part in POST_HOC for part in self.view.split(MINUS))


def r2_values(
    fits: Sequence[FoldFit], owners: np.ndarray, counts: np.ndarray
) -> tuple[list[float], np.ndarray]:
    """R² in each test fold, and its mean over the folds in each resample."""
    folds: list[float] = []
    summed = np.zeros(len(counts))
    for fit in fits:
        folds.append(float(fold_r2(fit.truth, fit.predictions)))
        members, local = np.unique(owners[fit.rows], return_inverse=True)
        summed += resampled_r2(fit.truth, fit.predictions, local, counts[:, members])
    return folds, summed / len(fits)


def auroc_values(fits: Sequence[FoldFit], counts: np.ndarray) -> tuple[list[float], np.ndarray]:
    """The AUROC in each test fold, and its mean over the folds in each resample."""
    folds: list[float] = []
    summed = np.zeros(len(counts))
    for fit in fits:
        once = np.ones((1, len(fit.rows)), dtype=np.int64)
        folds.append(float(resampled_auroc(fit.predictions, fit.truth, once)[0]))
        summed += resampled_auroc(fit.predictions, fit.truth, counts[:, fit.rows])
    return folds, summed / len(fits)


def difference(first: Row, second: Row, view: str, model: str) -> Row:
    """The paired difference of two rows, on the same draws: each fold and each resample.

    What was left out is, cell by cell, the larger count of the two rows, so that a row that
    left something out is never hidden by one that left nothing out; a cell is "not
    applicable" only where it is so in both rows.
    """
    left: dict[str, Any] = {}
    for key in LEFT_OUT:
        counts = [row.left[key] for row in (first, second) if row.left[key] != NOT_APPLICABLE]
        left[key] = max(counts) if counts else NOT_APPLICABLE
    return Row(
        view,
        first.outcome,
        model,
        first.kind,
        first.number,
        [a - b for a, b in zip(first.folds, second.folds, strict=True)],
        first.resampled - second.resampled,
        left,
        None,
        NOT_APPLICABLE,
    )


def participant_means(matrix: np.ndarray, owners: np.ndarray, count: int) -> np.ndarray:
    """Each participant's mean over their segments, NaN skipped; NaN if none has a value."""
    means = pd.DataFrame(matrix).groupby(owners, sort=True).mean()
    if list(means.index) != list(range(count)):
        raise StepError("a labelled participant has no segment to average")
    return means.to_numpy(dtype=np.float64)


def fit_probes(
    job: Job,
    people: People,
    inputs: Inputs,
    targets: Targets,
    features: Mapping[str, np.ndarray],
    lengths: np.ndarray,
    counts: np.ndarray,
) -> list[Row]:
    """Stage 7: the probes, their controls and every number on the draws."""
    began = time.monotonic()
    settings = job.settings
    alphas, cs, max_iter = settings.ridge_alphas, settings.logistic_cs, settings.max_iter
    owners = inputs.owners
    female, folds = people.female[owners], people.folds[owners]
    labels = people.labels.to_numpy(dtype=np.int64)
    participants = len(labels)
    unit = {"parts_short": NOT_APPLICABLE, "parts_unmeasured": NOT_APPLICABLE}
    left = {
        name: {
            "segments_left_out": targets.left_out(name),
            **(
                {"parts_short": targets.short, "parts_unmeasured": targets.unmeasured}
                if name == PITCH
                else unit
            ),
            "means_filled": NOT_APPLICABLE,
        }
        for name in TARGETS
    }
    nothing = {"segments_left_out": 0, **unit, "means_filled": NOT_APPLICABLE}

    def ridge_rows(view: str, model: str, matrix: np.ndarray) -> list[Row]:
        found = ridge_folds(matrix, targets.values, female, folds, alphas)
        rows = []
        for name in TARGETS:
            for kind in KINDS:
                values, resampled = r2_values(found[(name, kind)], owners, counts)
                choices = [fit.choice for fit in found[(name, kind)]]
                rows.append(
                    Row(view, name, model, kind, R2, values, resampled, left[name], choices, RIDGE)
                )
        return rows

    def label_row(
        view: str, model: str, matrix: np.ndarray, what: str, gone: dict[str, Any]
    ) -> Row:
        found = logistic_folds(matrix, labels, people.folds, cs, max_iter, what)
        values, resampled = auroc_values(found, counts)
        choices = [fit.choice for fit in found]
        return Row(
            view, LABEL, model, NOT_APPLICABLE, AUROC, values, resampled, gone, choices, LOGISTIC
        )

    probes: dict[str, list[Row]] = {}
    for view, matrix in features.items():
        clock = time.monotonic()
        means = participant_means(matrix, owners, participants)
        probes[view] = [
            *ridge_rows(view, ENCODER, matrix),
            label_row(view, ENCODER, means, f"the label probe on {view}", nothing),
        ]
        print(f"probes of {view}: fitted in {(time.monotonic() - clock) / 60:.1f} min")
    measures = participant_means(targets.measures, owners, participants)
    filled = {
        "values": int(np.isnan(measures).sum()),
        "participants": int(np.isnan(measures).any(axis=1).sum()),
    }
    egemaps_left = {
        "segments_left_out": int(np.isnan(targets.measures).all(axis=1).sum()),
        "parts_short": targets.short,
        "parts_unmeasured": targets.unmeasured,
        "means_filled": filled["values"],
    }
    gender = female.astype(np.float64)[:, None]
    length = np.column_stack([lengths, 1.0 / lengths]).astype(np.float64)
    controls = {
        GENDER_ONLY: [
            *ridge_rows(NOT_APPLICABLE, GENDER_ONLY, gender),
            label_row(
                NOT_APPLICABLE,
                GENDER_ONLY,
                people.female.astype(np.float64)[:, None],
                "the label's gender-only control",
                nothing,
            ),
        ],
        LENGTH_ONLY: ridge_rows(NOT_APPLICABLE, LENGTH_ONLY, length),
        EGEMAPS: [
            label_row(
                TARGET_VIEW,
                EGEMAPS,
                measures,
                "the label's control on the 88 measures",
                egemaps_left,
            )
        ],
    }
    job.results["egemaps_means_filled"] = filled
    print(
        f"controls: gender alone, length alone and the 88 eGeMAPS measures of {TARGET_VIEW};"
        f" missing participant means, each filled with the training fold's mean:"
        f" {filled['values']}, from participants: {filled['participants']}"
    )
    rows = [row for view in VIEWS for row in probes[view]]
    rows += [row for model in (GENDER_ONLY, LENGTH_ONLY, EGEMAPS) for row in controls[model]]
    for view in VIEWS:
        for probe in probes[view]:
            for model, control_rows in controls.items():
                for control in control_rows:
                    if (control.outcome, control.kind) == (probe.outcome, probe.kind):
                        rows.append(difference(probe, control, view, f"{ENCODER}{MINUS}{model}"))
    reference = {(row.outcome, row.kind): row for row in probes[TARGET_VIEW]}
    for view in VIEWS[1:]:
        for probe in probes[view]:
            rows.append(
                difference(
                    probe,
                    reference[(probe.outcome, probe.kind)],
                    f"{view}{MINUS}{TARGET_VIEW}",
                    ENCODER,
                )
            )
    minutes = job.stage("probes", began)
    print(f"probes, controls and their {len(counts):,} resamples: {minutes:.1f} min")
    return rows


# Stage 8: the table -----------------------------------------------------------------------------


def table_of(rows: Sequence[Row], people: People, settings: ProbeSettings) -> pd.DataFrame:
    """probe.csv: one row per view, outcome, model and kind; a cell that cannot exist says so."""
    names = sorted(set(people.folds.tolist()))
    records = []
    for row in rows:
        cells: dict[str, Any] = {
            "view": row.view,
            "outcome": row.outcome,
            "model": row.model,
            "kind": row.kind,
            "number": row.number,
            "post_hoc": "yes" if row.post_hoc else "no",
        }
        cells.update({f"fold_{fold}": value for fold, value in zip(names, row.folds, strict=True)})
        cells["mean"] = float(np.mean(row.folds))
        for level in settings.levels:
            low, high = interval(row.resampled, level)
            cells[f"low_{level_name(level)}"] = low
            cells[f"high_{level_name(level)}"] = high
        cells.update(row.left)
        cells["penalty"] = row.penalty
        choices = row.choices
        for place, fold in enumerate(names):
            cells[f"penalty_fold_{fold}"] = (
                NOT_APPLICABLE if choices is None else choices[place].value
            )
        for name, count in (
            ("grid_ends", lambda choice: choice.at_end),
            ("plateaus", lambda choice: choice.plateau),
            ("skipped", lambda choice: choice.skipped),
            ("inner_not_converged", lambda choice: choice.not_converged),
        ):
            cells[name] = (
                NOT_APPLICABLE if choices is None else int(sum(count(choice) for choice in choices))
            )
        records.append(cells)
    return pd.DataFrame(records)


def print_rows(table: pd.DataFrame, settings: ProbeSettings) -> None:
    """Each row's mean and intervals, at 4 decimals."""
    for row in table.itertuples(index=False):
        number = "AUROC" if row.number == AUROC else "R²"
        kind = "" if row.kind == NOT_APPLICABLE else f", {row.kind}"
        ends = "; ".join(
            f"{level_name(level)}% interval {getattr(row, f'low_{level_name(level)}'):.4f} to"
            f" {getattr(row, f'high_{level_name(level)}'):.4f}"
            for level in settings.levels
        )
        tag = " (post hoc)" if row.post_hoc == "yes" else ""
        print(
            f"{row.view}, {row.outcome}{kind}, {row.model}: {number} {row.mean:.4f} ({ends}){tag}"
        )


def probe_job(state: Progress) -> None:
    """The job: stages 1 to 8."""
    job = start(state)  # 1
    people = read_people(job)  # 2
    counts = draw(job, people)  # 3
    inputs = read_pins(job, people)  # 4
    targets = measure_targets(job, inputs, state)  # 5
    features, lengths = pool(job, inputs, state)  # 6
    rows = fit_probes(job, people, inputs, targets, features, lengths, counts)  # 7
    table = table_of(rows, people, job.settings)  # 8
    print_rows(table, job.settings)
    for line in READING:
        print(line)
    job.results["rows"] = len(table)
    job.results["reading"] = list(READING)
    job.info["outputs"][TABLE] = write_table(table, job.run_dir / TABLE)
    minutes = (time.monotonic() - job.started) / 60
    job.info.update(status="done", finished_utc=utc_now(), minutes=round(minutes, 2))
    job.save()
    print(f"table: {TABLE} ({len(table)} rows) in {job.relative}")
    print(f"done in {minutes:.1f} min")


def main() -> None:
    os.umask(0o077)
    sys.unraisablehook = quiet_unraisable  # a finaliser's message could hold anything
    progress = Progress()
    try:
        probe_job(progress)
        return
    except BaseException as caught:
        error = caught  # reported below, outside this clause, so that nothing chains to it
    with contextlib.suppress(BaseException):  # nothing here may print the error's message
        failure = record_failure(error, progress)
        print(failure.line)
        if failure.frames is not None:  # no errors.log holds them: frames and types only
            print(failure.frames, end="", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
