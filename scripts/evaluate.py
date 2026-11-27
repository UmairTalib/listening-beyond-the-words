"""Analyse the pilot's test scores: fold AUROCs, Δ, intervals, the checks and the rule (CPU job).

Submit from an OMNI login node, from a checkout at the newest v0.1.x tag with its environment
built at that tag (git checkout v0.1.x, then bash scripts/setup_omni_env.sh), once the kept
training runs are pinned in configs/pilot/evaluate.yaml (scripts/list_runs.py prints the
block), the trial first:
    scripts/slurm/submit.sh evaluate --trial
and, once the trial is read, the analysis itself, once:
    scripts/slurm/submit.sh evaluate
--trial names the inputs that the job reads, never a setting. The job has no --allow-dirty:
every run on OMNI reads real scores and labels, so debugging uses the synthetic tests.

docs/pilot_analysis.md fixes the analysis and fills only the points that the pilot protocol
leaves open. The settings are in configs/pilot/evaluate.yaml (evaluation.EvaluateSettings),
which also pins the kept training runs; the labels, partitions, zero-shot scores and the
word-check run are pinned in configs/pilot/data.yaml.

Stages, in one process:
  1. Guards: a Slurm job; a clean checkout at the newest v0.1.x tag of the clone; the
     installed code of that tag's version; evaluate.yaml read, a key given twice refused
     (config.read_settings), with at least one run pinned. Then the run folder and run.json
     (running).
  2. The pins of the kept runs, before any score is read: each run's run.json and
     scores.parquet by their SHA-256, its run.json listing that scores.parquet, and its
     record (evaluation.kept_run: done, not started with --allow-dirty, check F1 passing,
     the label and partition pins of data.yaml, from the first runs' commit or the newest
     v0.1.x tag of its clone). Each pinned run must be the earliest done run of its
     fold-run that counts (evaluation.run_order: the one that finished first), and a later
     one is reported; only the other runs' records are read for this. The pinned set must
     be one of the protocol's versions: all 30 runs, 25 (no B1 round 2), 20 (no B1) or 10
     (B0 and CPG round 1 only).
  3. The pinned labels with their PHQ-8 totals, every label checked to be a total of 10 or
     more, and the pinned partitions; then the bootstrap's draws (evaluation.draw_counts),
     which need nothing else, so they exist before any score is read; run.json records
     their SHA-256 and the redraws.
  4. The checks that read no score: F1 on every kept run (stage 2); F2 on every kept CPG
     run, with the pooled share of the summed counts (evaluation.f2_reading); F4 from the
     pinned word-check run (scoring.embargo); F5 as the first CPG run of round 1, fold 1
     read it. A record that does not say that F4 or F5 passed, or lacks its key, is
     refused, never read as a fail.
  5. The zero-shot scores (zeroshot.load_zeroshot_scores), which hold each segment of the
     pinned caches once, and B0's: each B0 run's scores.parquet must hold exactly its test
     fold's segments of x0 and rev, each z finite (read_scores). Their fold AUROCs and
     Spearman correlations (evaluation.point_values), and check F3: B0's mean AUROC on x0.
  6. Only if all five checks pass: B1's and CPG's scores, read as B0's; every number on
     every fold-run, its 80% and 95% intervals from the draws and, for a difference, its
     standard error; Δ and the decision rule (evaluation.outcome), with the protocol's
     decision for the outcome, word for word, the framing line and the noise figures. A
     failed F2 or F3 writes the checks and the zero-shot and B0 numbers, prints
     "F<n> FAIL: reported as a finding; the rule is not read" and stops: no Δ and no
     outcome are computed.
  7. The tables (reports.write_table), then run.json (done).

--trial (pilot analysis, section "Order"): stages 1 to 4, then only the zero-shot scores
and B0's round 1: their fold AUROCs, Spearman correlations and intervals, the zero-shot
arm's over the 10 fold-runs and B0's over round 1's five, with no difference, no F3 and no
rule. These numbers count as seen.

Writes $LBW_WORKSPACE/derived/evaluate/run-<job id>/ (trial-<job id>/ with --trial) with
  checks.csv       checks F1 to F5, F2 per kept CPG run and pooled (not in the trial)
  fold_values.csv  per round, fold and view: the positive and negative participants, the
                   segments, each arm's AUROC and Spearman correlation with the PHQ-8
                   totals, the three differences and the agreement of B0's and CPG's scores
  summary.csv      per number and view: the mean over its fold-runs, each round's mean, the
                   80% and 95% intervals and, for a difference, its standard error
  outcome.csv      Δ on x0 and rev, exact, the outcome, the protocol's decision and the
                   framing line (only when the rule is read)
  run.json         mode, commit and tags, the configs' SHA-256, the pins with each run's
                   commit and tags, the bootstrap's seed, stream, redraws and the SHA-256 of
                   its draws, the environment, and aggregate results: the version of the
                   pinned set, the counts, the checks and the numbers read, and the tables'
                   SHA-256
A cell with no value reads "not available (cut: <version>)" in a cut version, "not defined"
where a Spearman correlation cannot be computed, and "not read (trial: ...)" in the trial,
never blank. A job that fails also leaves errors.log there: traceback frames and error types,
never a message. The job prints and records aggregate numbers only: no participant ID and no
number per participant or segment. A refusal names a rule, a run and counts.
"""

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import socket
import sys
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pandas as pd
import yaml

from listening_beyond_words.clips import load_pinned_labels, load_pinned_partitions
from listening_beyond_words.config import read_settings
from listening_beyond_words.evaluation import (
    AGREEMENT,
    ARMS,
    AUROC,
    BOOTSTRAP_KEYS,
    DIFFERENCE,
    ROUNDS,
    RUN_NAME,
    RUNS_FOLDER,
    SPEARMAN,
    ZEROSHOT,
    EvaluateSettings,
    FoldRun,
    Number,
    Point,
    Resampled,
    RunPin,
    Version,
    draw_counts,
    draws_sha256,
    f2_reading,
    f3_reading,
    fold_runs,
    framing,
    interval,
    kept_run,
    outcome,
    person_scores,
    point_values,
    printed,
    resampled_means,
    run_order,
    standard_error,
)
from listening_beyond_words.labels import PHQ8_CUTOFF
from listening_beyond_words.reports import write_table
from listening_beyond_words.runinfo import (
    RELEASE_TAGS,
    Progress,
    StepError,
    environment,
    git_state,
    item_seed,
    newest_release,
    quiet_unraisable,
    record_failure,
    sha256_file,
    stop_on_sigterm,
    utc_now,
    write_json,
)
from listening_beyond_words.scoring import embargo
from listening_beyond_words.sessions import PILOT_IDS
from listening_beyond_words.settings import Workspace, get_workspace
from listening_beyond_words.wordcheck import FAIL, PASS
from listening_beyond_words.zeroshot import SCORE_COLUMNS, SCORES, load_zeroshot_scores

CONFIG = Path("configs/pilot/data.yaml")
EVALUATE = Path("configs/pilot/evaluate.yaml")
STEP = "evaluate"
FULL = "full"
TRIAL = "trial"
PACKAGE = "listening-beyond-words"  # the installed code, whose version must be the tag's
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "pyyaml", PACKAGE)
USAGE = "usage: evaluate.py [--trial]"
CHECKS_TABLE = "checks.csv"
FOLD_TABLE = "fold_values.csv"
SUMMARY_TABLE = "summary.csv"
OUTCOME_TABLE = "outcome.csv"
# The line that a failed F2 or F3 prints before the job stops (pilot analysis, section
# "Checks first").
FAIL_LINE = "F{} FAIL: reported as a finding; the rule is not read"
# The arms and rounds that the trial reads (pilot analysis, section "Order"): the zero-shot
# scores of every fold-run and B0's round 1.
TRIAL_READS = {ZEROSHOT: ROUNDS, "b0": (1,)}
TRIAL_CELL = "not read (trial: B0's round 1 only)"
NOT_DEFINED = "not defined"
SE_CELL = "not computed: differences only"  # pilot analysis, section "Intervals"
# F2's bounds, which each CPG run read from its train.yaml (rules.alive) and gave its verdict
# on (pilot protocol, section "Training", bullet "F2 and the two allowed rules").
F2_BOUND = "0.20 to 0.80, both included, as each run read them (train.yaml, rules.alive)"
F5_FOLD_RUN = ("cpg", 1, 1)  # check F5 is read on the first CPG run of round 1, fold 1
F5_READINGS = ("reads_rules", "reads_f5")  # the flags of the run that read F5 (train.py)
NAMES = {ZEROSHOT: "zero-shot", "b0": "B0", "b1": "B1", "cpg": "CPG"}


# The machine and the command line --------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    """argparse, except that a wrong command line is a refusal with the usage."""

    def error(self, message: str) -> NoReturn:
        raise StepError(f"{USAGE} ({message})")


def parse_arguments(argv: Sequence[str]) -> bool:
    """The command line: whether this is the trial. Any other option is refused."""
    parser = _Parser(prog="evaluate.py", add_help=False)
    parser.add_argument("--trial", action="store_true")
    return bool(parser.parse_args(list(argv)).trial)


def installed_version() -> str:
    """The version of the installed code, as the environment's metadata holds it."""
    try:
        return version(PACKAGE)
    except PackageNotFoundError:
        return "none (the package is not installed)"


def release_guard(git: Mapping[str, Any]) -> tuple[str, str]:
    """Refuse a checkout off the newest v0.1.x tag, or code installed at another version.

    Pilot analysis, section "Order": the analysis runs from the release that holds its code,
    and the job refuses any other checkout. Releases are ordered by their numbers
    (``runinfo.newest_release``), so v0.1.10 is newer than v0.1.2. The environment must hold
    the tag's own version, since the version line changes at each release and the
    environment is built again then (engineering conventions, section 2).

    Returns:
        The tag and the installed version.
    """
    released = [str(tag) for tag in git.get("release_tags") or []]
    if not released:
        raise StepError(
            f"no {RELEASE_TAGS} tag exists in the clone: the analysis runs only from the"
            ' release that holds its code (pilot analysis, section "Order")'
        )
    newest = newest_release(released)  # refuses a tag that it cannot order
    if newest not in {str(tag) for tag in git.get("tags") or []}:
        raise StepError(
            f"the checkout ({git.get('describe')}) is not at the newest release tag {newest}:"
            f" the analysis runs only from it (git checkout {newest} on OMNI, then bash"
            " scripts/setup_omni_env.sh)"
        )
    installed = installed_version()
    if installed != newest.removeprefix("v"):
        raise StepError(
            f"the environment holds version {installed} of the code, not"
            f" {newest.removeprefix('v')} of the tag {newest}: build it again at the tag (bash"
            " scripts/setup_omni_env.sh)"
        )
    return newest, installed


def exact(value: Fraction) -> dict[str, Any]:
    """An exact value for run.json: the fraction as text and the nearest float."""
    return {"exact": f"{value.numerator}/{value.denominator}", "value": float(value)}


def shown(value: Any) -> str:
    """A value for the log, at 4 decimals: an exact one rounded half to even; text as it is."""
    if isinstance(value, Fraction):
        return printed(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def cell(value: Any) -> Any:
    """A value for a table: a float at full precision, or text; never a blank."""
    if isinstance(value, Fraction):
        return float(value)
    return value


def fold_names(fold_runs: Sequence[tuple[int, int]]) -> str:
    """Fold-runs for a cell or a line, such as "round 1, fold 3; round 2, fold 1"."""
    return "; ".join(f"round {number}, fold {fold}" for number, fold in fold_runs)


def arm_names(arms: Sequence[str]) -> str:
    """Arms for a line: "B0 and CPG"."""
    names = [NAMES[arm] for arm in arms]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"


# Stage 1: the job ---------------------------------------------------------------------------------


@dataclass
class Job:
    """The analysis being made: its mode, settings, inputs, folder and record."""

    trial: bool
    job: str
    workspace: Workspace
    config: dict[str, Any]
    settings: EvaluateSettings
    version: Version
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


def start(state: Progress) -> Job:
    """Stage 1: refuse to run outside a clean Slurm job at the newest release; the folder."""
    stop_on_sigterm()
    trial = parse_arguments(sys.argv[1:])
    started = time.monotonic()
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise StepError("submit this as a job: scripts/slurm/submit.sh evaluate [--trial]")
    git = git_state()
    if git["dirty"]:
        raise StepError(
            "the checkout has uncommitted changes: the analysis runs only from a release, and"
            ' it has no --allow-dirty (pilot analysis, section "Order")'
        )
    release, installed = release_guard(git)
    content, evaluate_sha256 = read_settings(EVALUATE)
    settings = EvaluateSettings.from_config(content)
    pinned = settings.version
    if pinned is None:
        raise StepError(
            f"no training run is pinned in {EVALUATE.name} (runs): pin the kept runs first;"
            " scripts/list_runs.py prints the block"
        )
    data = CONFIG.read_bytes()
    config = yaml.safe_load(data.decode("utf-8"))
    workspace = get_workspace()
    run_dir = workspace.derived / STEP / f"{TRIAL if trial else 'run'}-{job}"
    run_dir.mkdir(parents=True, exist_ok=False)
    state.run_dir = run_dir
    machine = environment(PACKAGES)
    info: dict[str, Any] = {
        "run_id": run_dir.name,
        "step": STEP,
        "mode": TRIAL if trial else FULL,
        "status": "running",
        "started_utc": utc_now(),
        "git": git,
        "release": release,
        "installed_version": installed,
        "config": {"file": str(EVALUATE), "sha256": evaluate_sha256, "content": content},
        "data_config": {"file": str(CONFIG), "sha256": hashlib.sha256(data).hexdigest()},
        "slurm": {
            "job_id": job,
            "node": socket.gethostname(),
            "cpu": machine["cpu"],
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
        "seeds": {},
        "inputs": {},
        "environment": machine,
        "results": {"version": {"name": pinned.name, "runs": pinned.runs}},
        "outputs": {},
    }
    state.info = info
    run = Job(trial, job, workspace, config, settings, pinned, run_dir, info, started)
    run.save()
    what = "the trial of the analysis" if trial else "the analysis"
    print(
        f"{what} at the release tag {release} (installed version {installed}), commit"
        f" {str(git['commit'])[:12]}; job {job} on {socket.gethostname()}; CPU {machine['cpu']}"
    )
    print(f"pinned runs: {pinned.name} ({pinned.runs} training runs)")
    return run


# Stage 2: the pins ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Kept:
    """A pinned training run whose pin and record were checked: it counts in the analysis.

    Attributes:
        arm: Its arm.
        round_number: Its round.
        fold: Its fold.
        pin: Its pin in evaluate.yaml.
        record: Its run.json, as read.
        counts_by: What lets it count (``evaluation.kept_run``).
    """

    arm: str
    round_number: int
    fold: int
    pin: RunPin
    record: Mapping[str, Any]
    counts_by: str

    @property
    def run(self) -> str:
        """The run folder, relative to the workspace."""
        return self.pin.run


def pinned_run(
    workspace: Workspace,
    key: tuple[str, int, int],
    pin: RunPin,
    config: Mapping[str, Any],
    first_runs_commit: str,
) -> Kept:
    """Check one pin of evaluate.yaml and its run's record; return the run if it counts.

    The run's run.json and scores.parquet must have the SHA-256 that the pin gives, and the
    run.json must list that scores.parquet as its own; then ``evaluation.kept_run`` reads the
    record. No score is read here.

    Args:
        workspace: The workspace.
        key: The fold-run that the pin is for, as (arm, round, fold).
        pin: The pin.
        config: The data config, whose label and partition pins the run must have used.
        first_runs_commit: The first runs' commit (evaluate.yaml).

    Returns:
        The run, with its record and what lets it count.

    Raises:
        StepError: If a file does not match the pin, or the run does not count.
    """
    arm, round_number, fold = key
    folder = workspace.root / pin.run
    record_path = folder / "run.json"
    data = record_path.read_bytes() if record_path.is_file() else None
    if data is None or hashlib.sha256(data).hexdigest() != pin.record_sha256:
        raise StepError(f"{pin.run}: its run.json does not match its pin (record_sha256)")
    scores_path = folder / SCORES
    if not scores_path.is_file() or sha256_file(scores_path) != pin.scores_sha256:
        raise StepError(f"{pin.run}: its {SCORES} does not match its pin (scores_sha256)")
    try:
        record = json.loads(data.decode("utf-8"))
    except ValueError:
        raise StepError(f"{pin.run}: its run.json cannot be read as JSON") from None
    if not isinstance(record, dict):
        raise StepError(f"{pin.run}: its run.json is not a JSON object")
    outputs = record.get("outputs")
    if not isinstance(outputs, Mapping) or outputs.get(SCORES) != pin.scores_sha256:
        raise StepError(f"{pin.run}: its run.json lists another {SCORES} than its pin")
    counts_by = kept_run(record, pin.run, arm, round_number, fold, config, first_runs_commit)
    return Kept(arm, round_number, fold, pin, record, counts_by)


def counting_rivals(
    workspace: Workspace, run: Kept, config: Mapping[str, Any], first_runs_commit: str
) -> list[tuple[tuple[str, str], str]]:
    """The other done runs of a kept run's fold-run that count too, each with its order.

    Pilot analysis, section "Runs and pins", bullet "Kept runs": if two done runs share a
    fold-run, the earlier one counts and the other is reported. The candidates are the other
    folders runs/pilot/pilot-<arm>-r<round>-f<fold>-* of the fold-run. Only their run.json
    files are read, never a score: a folder whose record is missing or cannot be read as a
    JSON object, or whose run does not count (``evaluation.kept_run``), is no rival, as
    scripts/list_runs.py leaves it out.

    Args:
        workspace: The workspace.
        run: The kept run.
        config: The data config, whose label and partition pins a run must have used.
        first_runs_commit: The first runs' commit (evaluate.yaml).

    Returns:
        Each rival's order (``evaluation.run_order``) and its folder, relative to the
        workspace.

    Raises:
        StepError: If a rival counts but its record does not say when it finished.
    """
    found: list[tuple[tuple[str, str], str]] = []
    pattern = f"pilot-{run.arm}-r{run.round_number}-f{run.fold}-*"
    for folder in sorted((workspace.root / RUNS_FOLDER).glob(pattern)):
        name = folder.relative_to(workspace.root).as_posix()
        if name == run.run or not RUN_NAME.fullmatch(folder.name):
            continue
        try:
            record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # no record, or one that cannot be read: the run does not count
        if not isinstance(record, dict):
            continue
        try:
            kept_run(record, name, run.arm, run.round_number, run.fold, config, first_runs_commit)
        except StepError:
            continue  # it does not count, so it is no rival
        found.append((run_order(record, name), name))
    return found


def pin_runs(job: Job) -> dict[tuple[str, int, int], Kept]:
    """Stage 2: every pin of evaluate.yaml, its files by their SHA-256 and its record.

    A pinned run must also be the earliest done run of its fold-run that counts; the later
    ones are reported (``counting_rivals``).
    """
    kept: dict[tuple[str, int, int], Kept] = {}
    order = sorted(job.settings.runs, key=lambda key: (ARMS.index(key[0]), key[1], key[2]))
    for key in order:
        kept[key] = pinned_run(
            job.workspace, key, job.settings.runs[key], job.config, job.settings.first_runs_commit
        )
    later: dict[tuple[str, int, int], list[str]] = {}
    for key, run in kept.items():
        own = run_order(run.record, run.run)
        rivals = counting_rivals(job.workspace, run, job.config, job.settings.first_runs_commit)
        for rival, name in sorted(rivals):
            if rival < own:
                raise StepError(
                    f"{run.run}: {name}, a done run of {run.arm}, round {run.round_number}, fold"
                    f" {run.fold} that counts, finished earlier, and the earlier one counts"
                    ' (pilot analysis, section "Runs and pins"); scripts/list_runs.py pins it'
                )
            later.setdefault(key, []).append(name)
    inputs: dict[str, Any] = {}
    for (arm, round_number, fold), run in kept.items():
        git = run.record["git"]
        inputs.setdefault(arm, {}).setdefault(f"r{round_number}", {})[f"f{fold}"] = {
            "run": run.run,
            "scores_sha256": run.pin.scores_sha256,
            "record_sha256": run.pin.record_sha256,
            "commit": git.get("commit"),
            "tags": list(git.get("tags") or []),
            "counts_by": run.counts_by,
            "later_runs": later.get((arm, round_number, fold), []),
        }
    job.info["inputs"]["runs"] = inputs
    job.save()
    first = sum("first runs' commit" in run.counts_by for run in kept.values())
    print(
        f"pins: {len(kept)} kept runs match their pins and records: done, not started with"
        " --allow-dirty, F1 passing, the pinned labels and partitions, each the earliest"
        f" done run of its fold-run that counts; {first} from the first runs' commit"
        f" {job.settings.first_runs_commit[:7]}, {len(kept) - first} from the newest release"
        " tag of their clone"
    )
    for names in later.values():
        for name in names:
            print(f"pins: {name} is a later done run of a pinned fold-run and does not count")
    return kept


# Stage 3: the participants and the draws --------------------------------------------------------


@dataclass(frozen=True)
class People:
    """The pilot's participants: labels, PHQ-8 totals and partitions, checked.

    Attributes:
        labels: Each participant's label, 0 or 1, by session ID in ascending order.
        totals: Each participant's PHQ-8 total, in the same order.
        partitions: One row per participant and round: ``session``, ``round``, ``fold``.
    """

    labels: pd.Series
    totals: pd.Series
    partitions: pd.DataFrame

    def members(self) -> dict[tuple[int, int], list[int]]:
        """Each test fold's participants by (round, fold), in ascending session ID."""
        return {
            (int(number), int(fold)): sorted(int(session) for session in part["session"])
            for (number, fold), part in self.partitions.groupby(["round", "fold"], sort=True)
        }


def read_people(workspace: Workspace, config: Mapping[str, Any]) -> People:
    """The pinned labels with their PHQ-8 totals and the pinned partitions, checked.

    Every participant must be a pilot session, and every label must be 1 exactly where the
    PHQ-8 total is 10 or more (pilot protocol, section "Speech used", bullet "Labels"; pilot
    analysis, section "Data protection"). A refusal gives counts, never a participant's ID,
    label or total.

    Args:
        workspace: The workspace.
        config: The data config, with the ``labels`` and ``partitions`` pins.

    Returns:
        The participants.

    Raises:
        StepError: If a pin is missing or does not match, or a check fails.
    """
    for name in ("labels", "partitions"):
        pin = config.get(name)
        if not isinstance(pin, Mapping) or not pin.get("run") or not pin.get("sha256"):
            raise StepError(f"no {name} are pinned ({name} in data.yaml)")
    try:
        table = load_pinned_labels(workspace, dict(config), phq8_total=True)
        partitions = load_pinned_partitions(workspace, dict(config))
    except (ValueError, OSError) as error:
        # The readers' own checks raise a plain ValueError, with a message of rules and
        # counts; any other error is named by its type.
        reason = str(error) if type(error) is ValueError else type(error).__name__
        raise StepError(f"the pinned labels and partitions do not hold: {reason}") from None
    sessions = [int(session) for session in table.index]
    outside = len(set(sessions) - set(PILOT_IDS))
    if outside or len(set(sessions)) != len(sessions):
        raise StepError(
            f"the pinned labels hold {outside} participants outside the pilot's sessions, or a"
            " participant twice"
        )
    if not all(pd.api.types.is_integer_dtype(table[name]) for name in ("label", "phq8_total")):
        raise StepError(
            "the pinned labels and PHQ-8 totals are not stored as whole numbers, as the labels"
            " step writes them"
        )
    label = table["label"].to_numpy(dtype=np.int64)
    total = table["phq8_total"].to_numpy(dtype=np.int64)
    bad = int((~np.isin(label, (0, 1))).sum())
    if bad:
        raise StepError(f"{bad} labels are not 0 or 1")
    disagree = int((label != (total >= PHQ8_CUTOFF).astype(np.int64)).sum())
    if disagree:
        raise StepError(
            f"{disagree} labels disagree with their PHQ-8 totals: a participant is positive"
            f" exactly when the total is {PHQ8_CUTOFF} or more (pilot analysis, section"
            ' "Data protection")'
        )
    index = pd.Index(sessions, dtype=np.int64, name="session")
    labels = pd.Series(label, index=index, name="label").sort_index()
    totals = pd.Series(total, index=index, name="phq8_total").sort_index()
    return People(labels, totals, partitions)


def draw(job: Job, people: People) -> pd.DataFrame:
    """Stage 3: the participants, then the bootstrap's draws, before any score is read."""
    people_record = read_and_count(job, people)
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
    job.results["counts"] = people_record
    job.save()
    positives, negatives = people_record["positives"], people_record["negatives"]
    print(
        f"bootstrap: {settings.resamples} resamples of {positives} positive and {negatives}"
        f" negative draws within the PHQ-8 classes, from {settings.stream}; {redraws}"
        f" resamples drawn again (a test fold without one class); draws' SHA-256 {digest}"
    )
    return counts


def read_and_count(job: Job, people: People) -> dict[str, Any]:
    """The participants of each class, overall and in each test fold; printed per round."""
    inputs = job.info["inputs"]
    inputs.update(labels=dict(job.config["labels"]), partitions=dict(job.config["partitions"]))
    folds: dict[str, dict[str, Any]] = {}
    for (number, fold), sessions in people.members().items():
        positive = int(people.labels.reindex(sessions).sum())
        folds.setdefault(f"r{number}", {})[f"f{fold}"] = {
            "positives": positive,
            "negatives": len(sessions) - positive,
        }
    positives = int(people.labels.sum())
    record = {
        "participants": len(people.labels),
        "positives": positives,
        "negatives": len(people.labels) - positives,
        "folds": folds,
    }
    print(
        f"participants: {record['participants']} in all, {positives} positive and"
        f" {record['negatives']} negative; every label is a PHQ-8 total of {PHQ8_CUTOFF} or"
        " more"
    )
    for round_key, round_folds in folds.items():
        parts = "; ".join(
            f"fold {key[1:]}: {value['positives']} positive and {value['negatives']} negative"
            for key, value in round_folds.items()
        )
        print(f"round {round_key[1:]}: {parts}")
    return record


# Stage 4: the checks that read no score ---------------------------------------------------------


def f5_reading(kept: Mapping[tuple[str, int, int], Kept]) -> dict[str, Any]:
    """Check F5 as the first CPG run of round 1, fold 1 read it: its record must say pass.

    Pilot protocol, section "Training", bullet "Check F5": F5 is read once, on the training
    hours of the first CPG run of round 1, fold 1 that finishes (``reads_rules``), or alone
    again on its rerun at 8 segments (``reads_f5``). The pinned run of that fold-run must be
    the one that read it, and its record must say that F5 passed (pilot analysis, section
    "Checks first"); a record that does not, or lacks the key, is refused, never read as a
    fail.

    Returns:
        The run, the reading that read F5 there, its hours, the bound and the verdict.
    """
    run = kept.get(F5_FOLD_RUN)
    if run is None:
        raise StepError("no run of CPG's round 1, fold 1 is pinned: check F5 is read on it")
    readings = (run.record.get("results") or {}).get("rule_readings") or {}
    read_by = [name for name in F5_READINGS if readings.get(name) is True]
    if not read_by:
        raise StepError(
            f"{run.run}: its run.json does not say that it read check F5"
            " (results.rule_readings): F5 is read on the first CPG run of round 1, fold 1"
            " that finishes"
        )
    f5 = ((run.record.get("results") or {}).get("rules") or {}).get("f5")
    if not isinstance(f5, Mapping) or f5.get("passes") is not True:
        raise StepError(
            f"{run.run}: its run.json does not say that check F5 passed"
            " (results.rules.f5.passes): such a record is refused, never read as a fail"
        )
    hours, bound = f5.get("hours"), f5.get("max_hours")
    if not all(
        isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
        for value in (hours, bound)
    ):
        raise StepError(f"{run.run}: its run.json lacks the hours of check F5")
    return {
        "run": run.run,
        "read_by": read_by[0],
        "hours": hours,
        "max_hours": bound,
        "passes": True,
        "verdict": PASS,
    }


def read_checks(job: Job, kept: Mapping[tuple[str, int, int], Kept]) -> dict[str, Any]:
    """Stage 4: checks F1, F2, F4 and F5, from records only; no score is read."""
    checks: dict[str, Any] = {
        "f1": {"kept_runs": len(kept), "passes": True, "verdict": PASS},
    }
    print(f"F1: {PASS}: all {len(kept)} kept runs say that F1 passed (results.rules.f1.passes)")
    cpg = {run.run: dict(run.record) for run in kept.values() if run.arm == "cpg"}
    f2 = f2_reading(cpg)
    checks["f2"] = {**f2, "verdict": PASS if f2["passes"] else FAIL}
    shares = [reading["share"] for reading in f2["runs"].values()]
    failing = sum(not reading["passes"] for reading in f2["runs"].values())
    pooled = f2["pooled"]
    print(
        f"F2: {checks['f2']['verdict']}: {len(shares) - failing} of {len(shares)} kept CPG runs"
        f" have a gain-alive share from 0.20 to 0.80 in epochs 2 and 3, from {min(shares):.4f}"
        f" to {max(shares):.4f}; pooled {pooled['share']:.4f} ({pooled['active']} of"
        f" {pooled['bags']} bags), reported only"
    )
    reading = embargo(job.workspace, job.config)  # refuses unless F4 says PASS
    checks["f4"] = {
        "run": reading["run"],
        "transcripts_sha256": reading["sha256"],
        "run_json_sha256": reading["run_json_sha256"],
        "passes": True,
        "verdict": reading["verdicts"]["f4"],
    }
    job.info["inputs"]["wordcheck_run"] = dict(job.config["wordcheck_run"])
    print(
        f"F4: {checks['f4']['verdict']}: verdicts.f4 of the pinned word-check run {reading['run']}"
    )
    checks["f5"] = f5_reading(kept)
    f5 = checks["f5"]
    print(
        f"F5: {PASS}: {f5['hours']:.4f} hours of training on {f5['run']}, at most"
        f" {f5['max_hours']:g} hours ({f5['read_by']})"
    )
    job.results["checks"] = checks
    job.save()
    return checks


# Stage 5: the scores ------------------------------------------------------------------------------


def read_zeroshot(
    job: Job, people: People
) -> tuple[dict[str, pd.Series], dict[tuple[int, int], dict[str, dict[str, int]]]]:
    """The zero-shot scores and each test fold's segments of x0 and rev, as they hold them.

    The pinned zero-shot run scored each segment of the pinned caches once
    (``zeroshot.load_zeroshot_scores``), so its rows name every segment of a test fold with
    its participant. Every labelled participant must have scores, and no one else.

    Returns:
        Each participant's exact score on each view, by view; and each test fold's segments
        of each view with their participants, by (round, fold) and view.
    """
    table = load_zeroshot_scores(job.workspace, job.config)
    job.info["inputs"]["zeroshot_scores"] = dict(job.config["zeroshot_scores"])
    labelled = {int(session) for session in people.labels.index}
    scores: dict[str, pd.Series] = {}
    rows: dict[str, pd.DataFrame] = {}
    for view in job.settings.views:
        rows[view] = table.loc[table["view"] == view, ["segment", "session"]]
        found = {int(session) for session in rows[view]["session"]}
        if found != labelled:
            raise StepError(
                f"the zero-shot scores of {view} name {len(found - labelled)} participants"
                f" without a label and lack {len(labelled - found)} labelled ones"
            )
        scores[view] = person_scores(table, view)
    segments: dict[tuple[int, int], dict[str, dict[str, int]]] = {}
    for key, sessions in people.members().items():
        chosen = set(sessions)
        segments[key] = {
            view: {
                str(segment): int(session)
                for segment, session in zip(
                    rows[view]["segment"], rows[view]["session"], strict=True
                )
                if int(session) in chosen
            }
            for view in job.settings.views
        }
    record: dict[str, Any] = {}
    for (number, fold), views in segments.items():
        record.setdefault(f"r{number}", {})[f"f{fold}"] = {
            view: len(found) for view, found in views.items()
        }
    job.results["counts"]["segments"] = record
    job.save()
    totals = ", ".join(f"{view} {len(rows[view])}" for view in job.settings.views)
    print(f"zero-shot scores: {job.config['zeroshot_scores']['run']}; segments {totals}")
    return scores, segments


def read_scores(
    workspace: Workspace,
    run: Kept,
    segments: Mapping[str, Mapping[str, int]],
    labelled: Collection[int],
) -> dict[str, pd.Series]:
    """Read a kept run's scores: exactly its test fold's segments of x0 and rev, each z finite.

    The run scored its own test fold (pilot protocol, section "Training", bullet "Scoring"),
    so its scores.parquet must hold each segment of the test fold's participants on each
    test view once, with its participant, as the zero-shot scores hold them, and nothing
    else. The bytes are checked against the pin again as they are read. A refusal gives
    counts, never a segment, a participant or a z.

    Args:
        workspace: The workspace.
        run: The kept run.
        segments: The test fold's segments of each view with their participants.
        labelled: The labelled participants.

    Returns:
        Each participant's exact score on each view (``evaluation.person_scores``), by view.

    Raises:
        StepError: If the scores do not hold exactly this.
    """
    data = (workspace.root / run.run / SCORES).read_bytes()
    if hashlib.sha256(data).hexdigest() != run.pin.scores_sha256:
        raise StepError(f"{run.run}: its {SCORES} changed since its pin was checked")
    table = pd.read_parquet(io.BytesIO(data))
    if tuple(table.columns) != SCORE_COLUMNS:
        raise StepError(f"{run.run}: its {SCORES} does not have the columns {list(SCORE_COLUMNS)}")
    if not pd.api.types.is_float_dtype(table["z"]):
        raise StepError(f"{run.run}: the z of its {SCORES} are not floating-point numbers")
    if not pd.api.types.is_integer_dtype(table["session"]):
        raise StepError(f"{run.run}: the sessions of its {SCORES} are not whole numbers")
    views = list(segments)
    others = int((~table["view"].isin(views)).sum())
    if others:
        raise StepError(f"{run.run}: its {SCORES} holds {others} rows of views other than {views}")
    found: dict[str, pd.Series] = {}
    for view in views:
        rows = table.loc[table["view"] == view]
        wanted = segments[view]
        sessions = [int(session) for session in rows["session"]]
        outside = sum(session not in labelled for session in sessions)
        if outside:
            raise StepError(
                f"{run.run}: its scores of {view} hold {outside} rows of sessions outside the"
                " pilot's labelled participants"
            )
        fold = set(wanted.values())
        training = sum(session not in fold for session in sessions)
        if training:
            raise StepError(
                f"{run.run}: its scores of {view} hold {training} segments of participants"
                " outside its test fold"
            )
        names = [str(segment) for segment in rows["segment"]]
        twice = len(names) - len(set(names))
        if twice:
            raise StepError(f"{run.run}: its scores of {view} hold {twice} segments twice")
        unknown = sum(
            wanted.get(name) != session for name, session in zip(names, sessions, strict=True)
        )
        if unknown:
            raise StepError(
                f"{run.run}: its scores of {view} hold {unknown} segments that the pinned"
                " caches do not hold for their participant"
            )
        missing = len(set(wanted) - set(names))
        if missing:
            raise StepError(
                f"{run.run}: its scores of {view} lack {missing} segments of its test fold"
            )
        z = rows["z"].to_numpy(dtype=np.float64)
        non_finite = int((~np.isfinite(z)).sum())
        if non_finite:
            raise StepError(f"{run.run}: {non_finite} of its z on {view} are not finite")
        found[view] = person_scores(rows, view)
    return found


def read_arm(
    job: Job,
    kept: Mapping[tuple[str, int, int], Kept],
    arm: str,
    rounds: Collection[int],
    segments: Mapping[tuple[int, int], Mapping[str, Mapping[str, int]]],
    labelled: Collection[int],
) -> dict[tuple[str, str, int, int], pd.Series]:
    """Every kept run of one arm in some rounds: its participants' scores by fold-run."""
    found: dict[tuple[str, str, int, int], pd.Series] = {}
    chosen = sorted(key for key in kept if key[0] == arm and key[1] in rounds)
    for key in chosen:
        _, number, fold = key
        scores = read_scores(job.workspace, kept[key], segments[(number, fold)], labelled)
        for view, values in scores.items():
            found[(arm, view, number, fold)] = values
    job.results.setdefault("read", {})[arm] = {
        "runs": len(chosen),
        "rounds": sorted({key[1] for key in chosen}),
    }
    job.save()
    if not chosen:
        print(f"{NAMES[arm]}: no run of it is pinned in this version ({job.version.name})")
        return found
    print(
        f"{NAMES[arm]}: {len(chosen)} runs read, each with exactly its test fold's segments of"
        f" {' and '.join(job.settings.views)}, every z finite"
    )
    return found


def zeroshot_by_fold(
    scores: Mapping[str, pd.Series], people: People
) -> dict[tuple[str, str, int, int], pd.Series]:
    """The zero-shot arm's scores taken fold by fold: its one model scores every fold-run."""
    return {
        (ZEROSHOT, view, number, fold): values.reindex(sessions)
        for (number, fold), sessions in people.members().items()
        for view, values in scores.items()
    }


# Stage 6: the numbers ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One row of summary.csv: a number on a view, as the mean over some fold-runs.

    Attributes:
        number: The number and its view.
        rounds: The rounds whose fold-runs the mean covers; empty where the number cannot
            exist in this job.
        cells: The row's cells, by column, in the table's order.
    """

    number: Number
    rounds: tuple[int, ...]
    cells: dict[str, Any]


def level_name(level: float) -> str:
    """A level as the columns name it: 80 for 0.8."""
    return f"{level * 100:g}"


def summary_columns(settings: EvaluateSettings) -> list[str]:
    """The columns of summary.csv."""
    ends = [f"{end}_{level_name(level)}" for level in settings.levels for end in ("low", "high")]
    rounds = [f"round_{number}" for number in ROUNDS]
    return ["number", "view", "rounds", "fold_runs", "mean", *rounds, *ends, "se"]


def number_name(number: Number) -> str:
    """A number as the log names it, such as "AUROC of B0" or "CPG − B0"."""
    if number.kind == AUROC:
        return f"AUROC of {NAMES[number.arms[0]]}"
    if number.kind == SPEARMAN:
        return f"Spearman correlation of {NAMES[number.arms[0]]}'s scores with the PHQ-8 totals"
    if number.kind == DIFFERENCE:
        return f"{NAMES[number.arms[0]]} − {NAMES[number.arms[1]]}"
    return f"agreement of {arm_names(number.arms)}'s scores"


def wanted_numbers(arms: Sequence[str], settings: EvaluateSettings) -> list[Number]:
    """The numbers that the arms read give, in the tables' order (``point_values``'s)."""
    found = [
        Number(kind, (arm,), view)
        for kind in (AUROC, SPEARMAN)
        for arm in settings.arms
        if arm in arms
        for view in settings.views
    ]
    found += [
        Number(DIFFERENCE, pair, view)
        for pair in settings.differences
        if set(pair) <= set(arms)
        for view in settings.views
    ]
    if set(settings.agreement) <= set(arms):
        found += [Number(AGREEMENT, settings.agreement, view) for view in settings.views]
    return found


def missing(job: Job) -> str:
    """A cell whose value cannot exist in this job (pilot analysis, section "Runs and pins")."""
    return TRIAL_CELL if job.trial else f"not available (cut: {job.version.name})"


def rounds_of(point: Point | None) -> tuple[int, ...]:
    """The rounds in which a number has fold-runs."""
    return () if point is None else tuple(sorted({key[0] for key in point.values}))


def not_defined(fold_runs: Sequence[tuple[int, int]]) -> str:
    """A mean or interval that needs a Spearman correlation that is not defined, with its folds.

    Pilot analysis, section "The main numbers", bullet "A fold with equal scores".
    """
    return f"{NOT_DEFINED} ({fold_names(fold_runs)})"


def mean_cell(point: Point, rounds: Collection[int]) -> Any:
    """A number's mean over the fold-runs of some rounds, for a table."""
    value = point.mean(rounds)
    return not_defined(point.undefined(rounds)) if value is None else cell(value)


def mean_text(point: Point, rounds: Collection[int], bounds: Sequence[Fraction] = ()) -> str:
    """A number's mean over the fold-runs of some rounds, for the log, at 4 decimals."""
    value = point.mean(rounds)
    if value is None:
        return not_defined(point.undefined(rounds))
    if isinstance(value, Fraction):
        return printed(value, bounds)
    return f"{value:.4f}"


def exact_mean(point: Point, rounds: Collection[int] | None = None) -> Fraction:
    """A difference's exact mean, which the rule and its framing line compare."""
    value = point.mean(rounds)
    if not isinstance(value, Fraction):
        raise StepError("a difference of AUROCs is not exact: the rule compares exact values")
    return value


def summary_row(
    job: Job,
    number: Number,
    rounds: tuple[int, ...],
    point: Point | None,
    resampled: Resampled | None,
) -> Row:
    """A number's row: its mean over the rounds given, each round's mean, intervals and SE.

    Pilot analysis, section "Intervals": every mean over fold-runs gets its 80% and 95%
    percentile intervals, and each difference its bootstrap standard error; per-round values
    are point estimates only.
    """
    cells: dict[str, Any] = dict.fromkeys(summary_columns(job.settings), missing(job))
    cells.update(
        number=number.name,
        view=number.view,
        rounds=" and ".join(map(str, rounds)) if rounds else "none",
        fold_runs=0 if point is None else sum(key[0] in rounds for key in point.values),
    )
    if point is None or resampled is None or not rounds:
        return Row(number, (), cells)
    cells["mean"] = mean_cell(point, rounds)
    for number_round in rounds_of(point):
        cells[f"round_{number_round}"] = mean_cell(point, (number_round,))
    values = resampled.means(rounds)
    undefined = resampled.undefined(rounds)
    for level in job.settings.levels:
        name = level_name(level)
        ends = (not_defined(undefined),) * 2 if undefined else interval(values, level)
        cells[f"low_{name}"], cells[f"high_{name}"] = ends
    if number.kind != DIFFERENCE:
        cells["se"] = SE_CELL
    else:
        cells["se"] = not_defined(undefined) if undefined else standard_error(values)
    return Row(number, rounds, cells)


def summary_rows(
    job: Job,
    arms: Sequence[str],
    points: Mapping[Number, Point],
    resampled: Mapping[Number, Resampled],
) -> list[Row]:
    """Every number's row over the fold-runs that it has; Δ also over round 1 beside B1's.

    Pilot analysis, section "Runs and pins", bullet "Versions": without B1's round 2, B1's
    numbers and its differences use round 1's five fold-runs, and CPG − B0 on the same five
    fold-runs is shown beside them.
    """
    wanted = wanted_numbers(arms, job.settings)
    rows = [
        summary_row(
            job, number, rounds_of(points.get(number)), points.get(number), resampled.get(number)
        )
        for number in wanted
    ]
    b1_rounds = {
        rounds_of(points[number])
        for number in wanted
        if number.kind == DIFFERENCE and "b1" in number.arms and number in points
    }
    deltas = [
        number
        for number in wanted
        if number.kind == DIFFERENCE
        and number.arms == job.settings.delta
        and rounds_of(points.get(number)) == ROUNDS
    ]
    if b1_rounds == {(1,)} and deltas:
        place = max(index for index, row in enumerate(rows) if row.number in deltas) + 1
        rows[place:place] = [
            summary_row(job, number, (1,), points[number], resampled[number]) for number in deltas
        ]
    return rows


def fold_table(
    job: Job,
    arms: Sequence[str],
    people: People,
    segments: Mapping[tuple[int, int], Mapping[str, Mapping[str, int]]],
    points: Mapping[Number, Point],
) -> pd.DataFrame:
    """fold_values.csv: per round, fold and view, the counts and every number's value there.

    A value is a float at full precision; "not defined" where a Spearman correlation cannot
    be computed; the job's text for a value that cannot exist in it (``missing``).
    """
    absent = missing(job)
    wanted = {(number.name, number.view): number for number in wanted_numbers(arms, job.settings)}
    columns = list(dict.fromkeys(name for name, _ in wanted))
    rows = []
    for (number, fold), sessions in people.members().items():
        positive = int(people.labels.reindex(sessions).sum())
        for view in job.settings.views:
            row: dict[str, Any] = {
                "round": number,
                "fold": fold,
                "view": view,
                "positives": positive,
                "negatives": len(sessions) - positive,
                "segments": len(segments[(number, fold)][view]),
            }
            for name in columns:
                point = points.get(wanted[(name, view)])
                if point is None or (number, fold) not in point.values:
                    row[name] = absent
                    continue
                value = point.values[(number, fold)]
                row[name] = NOT_DEFINED if value is None else cell(value)
            rows.append(row)
    counts = ["round", "fold", "view", "positives", "negatives", "segments"]
    return pd.DataFrame(rows, columns=[*counts, *columns])


def delta_bounds(settings: EvaluateSettings, view: str) -> tuple[Fraction, ...]:
    """The bounds that Δ on a view is read against: both on x0, the green one on rev.

    Pilot protocol, section "Decision rule": Δ on x0 decides between "Green: beats B0",
    "Red: worse on normal audio" and the two outcomes between them, and Δ on rev only
    between those two.
    """
    if view == "x0":
        return settings.green_at_least, settings.red_at_most
    return (settings.green_at_least,)


def number_bounds(settings: EvaluateSettings, number: Number) -> tuple[Fraction, ...]:
    """The bounds that a number is compared with, so that a value near one prints unrounded.

    Pilot analysis, section "Decision rule": values are compared exactly and printed with 4
    decimals, and a value that rounds to a bound is also printed unrounded. Δ is read
    against the rule's bounds (``delta_bounds``); on the framing's view, B1 − B0 against
    the bound that it must reach and CPG − B1 against the bound that it must stay below.
    Other numbers are compared with no bound.
    """
    if number.kind != DIFFERENCE:
        return ()
    if number.arms == settings.delta:
        return delta_bounds(settings, number.view)
    if number.view != settings.framing_view:
        return ()
    if number.arms == ("b1", "b0"):
        return (settings.b1_gain_at_least,)
    if number.arms == ("cpg", "b1"):
        return (settings.cpg_over_b1_below,)
    return ()


def print_rows(job: Job, rows: Sequence[Row], points: Mapping[Number, Point]) -> None:
    """Each number with both intervals and its per-round values, at 4 decimals."""
    for row in rows:
        name, view = number_name(row.number), row.number.view
        point = points.get(row.number)
        if point is None or not row.rounds:
            print(f"{name} on {view}: {row.cells['mean']}")
            continue
        bounds = number_bounds(job.settings, row.number)
        if row.number.kind == DIFFERENCE and row.number.arms == job.settings.delta:
            name = f"Δ = {name}"
        intervals = []
        for level in job.settings.levels:
            low, high = (row.cells[f"{end}_{level_name(level)}"] for end in ("low", "high"))
            span = low if isinstance(low, str) else f"{low:.4f} to {high:.4f}"
            intervals.append(f"{level_name(level)}% {span}")
        rounds = [
            f"round {number} {mean_text(point, (number,), bounds)}"
            if number in rounds_of(point)
            else f"round {number} {row.cells[f'round_{number}']}"
            for number in ROUNDS
        ]
        se = row.cells["se"]
        extra = "" if row.number.kind != DIFFERENCE else f"; SE {shown(se)}"
        over = "" if row.rounds == ROUNDS else " over round 1's fold-runs"
        print(
            f"{name} on {view}{over}: {mean_text(point, row.rounds, bounds)}"
            f" ({'; '.join(intervals)}; {'; '.join(rounds)}{extra})"
        )


def row_record(row: Row, points: Mapping[Number, Point]) -> dict[str, Any]:
    """A summary row for run.json: its cells, with the exact mean beside a float one."""
    record = {key: value for key, value in row.cells.items() if key not in ("number", "view")}
    point = points.get(row.number)
    if point is not None and row.rounds:
        value = point.mean(row.rounds)
        if isinstance(value, Fraction):
            record["mean_exact"] = exact(value)["exact"]
    return record


def numbers_record(rows: Sequence[Row], points: Mapping[Number, Point]) -> dict[str, Any]:
    """results.numbers: each number by its name, its view and the rounds of its mean."""
    record: dict[str, Any] = {}
    for row in rows:
        by_view = record.setdefault(row.number.name, {}).setdefault(row.number.view, {})
        by_view[row.cells["rounds"]] = row_record(row, points)
    return record


def f3_check(job: Job, points: Mapping[Number, Point]) -> dict[str, Any]:
    """Check F3 on B0's AUROC on x0, compared exactly, printed with its per-round values."""
    reading = f3_reading(points[Number(AUROC, ("b0",), "x0")], job.settings)
    bound = reading["bound"]
    verdict = PASS if reading["passes"] else FAIL
    check = {
        "value": {**exact(reading["value"]), "printed": printed(reading["value"], (bound,))},
        "rounds": {str(number): exact(value) for number, value in reading["rounds"].items()},
        "bound": exact(bound),
        "passes": reading["passes"],
        "verdict": verdict,
    }
    rounds = "; ".join(
        f"round {number} {printed(value, (bound,))}" for number, value in reading["rounds"].items()
    )
    print(
        f"F3: {verdict}: B0's mean AUROC on x0 {printed(reading['value'], (bound,))} ({rounds}),"
        f" at least {printed(bound)}"
    )
    job.results["checks"]["f3"] = check
    job.save()
    return check


def checks_table(job: Job) -> pd.DataFrame:
    """checks.csv: F1 to F5, F2 per kept CPG run and pooled; F3 where it was read."""
    checks = job.results["checks"]
    f1, f2, f4, f5 = checks["f1"], checks["f2"], checks["f4"], checks["f5"]
    rows: list[dict[str, Any]] = [
        {
            "check": "F1",
            "scope": f"the {f1['kept_runs']} kept runs",
            "value": f1["kept_runs"],
            "detail": "kept runs whose record says that F1 passed (results.rules.f1.passes)",
            "bound": "every kept run passes",
            "verdict": f1["verdict"],
        },
        {
            "check": "F2",
            "scope": f"the {len(f2['runs'])} kept CPG runs",
            "value": sum(reading["passes"] for reading in f2["runs"].values()),
            "detail": "kept CPG runs whose gain-alive share passes F2",
            "bound": "every kept CPG run passes",
            "verdict": f2["verdict"],
        },
    ]
    for run, reading in f2["runs"].items():
        rows.append(
            {
                "check": "F2",
                "scope": run,
                "value": reading["share"],
                "detail": f"{reading['active']} of {reading['bags']} bags active in epochs 2 and 3",
                "bound": F2_BOUND,
                "verdict": PASS if reading["passes"] else FAIL,
            }
        )
    pooled = f2["pooled"]
    rows.append(
        {
            "check": "F2",
            "scope": f"pooled over the {len(f2['runs'])} kept CPG runs",
            "value": pooled["share"],
            "detail": f"{pooled['active']} of {pooled['bags']} bags, summed over the runs",
            "bound": "reported only",
            "verdict": "reported only",
        }
    )
    if "f3" in checks:
        f3 = checks["f3"]
        rounds = "; ".join(f"round {key} {value['exact']}" for key, value in f3["rounds"].items())
        rows.append(
            {
                "check": "F3",
                "scope": "B0's mean AUROC on x0 over its fold-runs",
                "value": f3["value"]["value"],
                "detail": f"exact {f3['value']['exact']}; {rounds}",
                "bound": f"at least {f3['bound']['value']:g}, compared exactly",
                "verdict": f3["verdict"],
            }
        )
    rows.append(
        {
            "check": "F4",
            "scope": f4["run"],
            "value": f4["verdict"],
            "detail": "verdicts.f4 of the pinned word-check run",
            "bound": PASS,
            "verdict": f4["verdict"],
        }
    )
    rows.append(
        {
            "check": "F5",
            "scope": f5["run"],
            "value": f5["hours"],
            "detail": f"hours of training as the run read them ({f5['read_by']})",
            "bound": f"at most {f5['max_hours']:g} hours",
            "verdict": f5["verdict"],
        }
    )
    return pd.DataFrame(rows, columns=["check", "scope", "value", "detail", "bound", "verdict"])


def read_rule(job: Job, points: Mapping[Number, Point]) -> dict[str, Any]:
    """Δ on x0 and rev, the decision rule, the protocol's decision and the framing line.

    Pilot protocol, section "Decision rule", and pilot analysis, section "Decision rule": the
    rule reads the point estimates of Δ, compared exactly; the framing line reads B1 − B0
    and CPG − B1 on its view, over the fold-runs that B1 has, and run.json records the two
    values exactly beside it.
    """
    settings = job.settings
    deltas = {view: points[Number(DIFFERENCE, settings.delta, view)] for view in settings.views}
    values = {view: exact_mean(point) for view, point in deltas.items()}
    result = outcome(values["x0"], values["rev"], settings)
    found = [
        points.get(Number(DIFFERENCE, pair, settings.framing_view))
        for pair in (("b1", "b0"), ("cpg", "b1"))
    ]
    b1_gain, cpg_over_b1 = (None if point is None else exact_mean(point) for point in found)
    line = framing(result.name, b1_gain, cpg_over_b1, settings)
    # The two values that the framing line compares, exactly, beside it.
    read: dict[str, Any] = {}
    for (name, bound), value, point in zip(
        (
            ("b1_minus_b0", settings.b1_gain_at_least),
            ("cpg_minus_b1", settings.cpg_over_b1_below),
        ),
        (b1_gain, cpg_over_b1),
        found,
        strict=True,
    ):
        read[name] = (
            missing(job)
            if value is None or point is None
            else {
                **exact(value),
                "printed": printed(value, (bound,)),
                "bound": exact(bound),
                "fold_runs": len(point.values),
            }
        )
    record = {
        "fold_runs": len(deltas["x0"].values),
        "delta": {
            view: {**exact(value), "printed": printed(value, delta_bounds(settings, view))}
            for view, value in values.items()
        },
        "outcome": result.name,
        "decision": result.decision,
        "framing": line,
        "framing_values": {"view": settings.framing_view, **read},
    }
    for view, value in values.items():
        print(
            f"rule: Δ on {view} {printed(value, delta_bounds(settings, view))}, the point"
            f" estimate over {record['fold_runs']} fold-runs"
        )
    print(f"outcome: {result.name}")
    print(f"decision: {result.decision}")
    print(line)
    job.results["outcome"] = record
    job.save()
    return record


def outcome_table(record: Mapping[str, Any]) -> pd.DataFrame:
    """outcome.csv: Δ on x0 and rev, exact, the outcome, its decision and the framing line."""
    delta = record["delta"]
    return pd.DataFrame(
        [
            {
                "fold_runs": record["fold_runs"],
                "delta_x0": delta["x0"]["value"],
                "delta_x0_exact": delta["x0"]["exact"],
                "delta_rev": delta["rev"]["value"],
                "delta_rev_exact": delta["rev"]["exact"],
                "outcome": record["outcome"],
                "decision": record["decision"],
                "framing": record["framing"],
            }
        ]
    )


def noise_figures(job: Job, rows: Sequence[Row], points: Mapping[Number, Point]) -> None:
    """The numbers for planning the main study (pilot analysis, section "Also reported").

    B0's AUROC on x0; Δ's bootstrap standard error; the difference between the two rounds'
    Δ, signed (round 1 minus round 2) and as its absolute value; and the agreement of B0's
    and CPG's scores, whose value per fold is in fold_values.csv.
    """
    settings = job.settings
    by_key = {(row.number, row.rounds): row for row in rows}
    b0 = Number(AUROC, ("b0",), "x0")
    record: dict[str, Any] = {
        "b0_auroc_x0": row_record(by_key[(b0, rounds_of(points[b0]))], points)
    }
    for view in settings.views:
        delta = Number(DIFFERENCE, settings.delta, view)
        agreement = Number(AGREEMENT, settings.agreement, view)
        point = points[delta]
        entry: dict[str, Any] = {"se": by_key[(delta, rounds_of(point))].cells["se"]}
        if rounds_of(point) == ROUNDS:
            gap = exact_mean(point, (1,)) - exact_mean(point, (2,))
            entry |= {"round_1_minus_round_2": exact(gap), "absolute": exact(abs(gap))}
            gap_text = f"{printed(gap)} (absolute {printed(abs(gap))})"
        else:
            entry["round_1_minus_round_2"] = gap_text = missing(job)
        agreed = by_key[(agreement, rounds_of(points[agreement]))]
        record[f"delta_{view}"] = entry
        record[f"agreement_{view}"] = row_record(agreed, points)
        print(
            f"noise on {view}: Δ's bootstrap SE {shown(entry['se'])}; Δ of round 1 minus Δ of"
            f" round 2 {gap_text}; mean agreement of B0's and CPG's scores"
            f" {mean_text(points[agreement], agreed.rounds)}"
        )
    job.results["noise"] = record
    job.save()


# The job --------------------------------------------------------------------------------------


def write_tables(job: Job, tables: Mapping[str, pd.DataFrame]) -> None:
    """Write the tables through the one writer that refuses identifiers; record their SHA-256."""
    for name, table in tables.items():
        job.info["outputs"][name] = write_table(table, job.run_dir / name)
    job.save()


def evaluate_job(state: Progress) -> None:
    """The job: stages 1 to 7, or the trial."""
    job = start(state)  # 1
    kept = pin_runs(job)  # 2
    people = read_people(job.workspace, job.config)  # 3
    counts = draw(job, people)
    checks = read_checks(job, kept)  # 4
    zeroshot, segments = read_zeroshot(job, people)  # 5
    labelled = {int(session) for session in people.labels.index}
    scores = zeroshot_by_fold(zeroshot, people)
    job.results.setdefault("read", {})[ZEROSHOT] = {"runs": 1, "rounds": list(ROUNDS)}
    b0_rounds = TRIAL_READS["b0"] if job.trial else ROUNDS
    scores |= read_arm(job, kept, "b0", b0_rounds, segments, labelled)
    arms: tuple[str, ...] = (ZEROSHOT, "b0")
    runs: list[FoldRun] = fold_runs(people.labels, people.totals, people.partitions, scores)
    points = point_values(runs, job.settings)
    failed = [] if checks["f2"]["passes"] else ["F2"]
    if not job.trial:
        if not f3_check(job, points)["passes"]:
            failed.append("F3")
        if not failed:  # 6: B1's and CPG's scores only once all five checks pass
            for arm in ("b1", "cpg"):
                scores |= read_arm(job, kept, arm, ROUNDS, segments, labelled)
            arms = ARMS
            runs = fold_runs(people.labels, people.totals, people.partitions, scores)
            points = point_values(runs, job.settings)
    resampled = resampled_means(counts, runs, job.settings)
    rows = summary_rows(job, arms, points, resampled)
    print_rows(job, rows, points)
    job.results["numbers"] = numbers_record(rows, points)
    job.save()
    tables: dict[str, pd.DataFrame] = {}
    if not job.trial:
        tables[CHECKS_TABLE] = checks_table(job)
    tables[FOLD_TABLE] = fold_table(job, arms, people, segments, points)
    tables[SUMMARY_TABLE] = pd.DataFrame(
        [row.cells for row in rows], columns=summary_columns(job.settings)
    )
    if arms == ARMS:
        record = read_rule(job, points)
        noise_figures(job, rows, points)
        tables[OUTCOME_TABLE] = outcome_table(record)
    elif job.trial:
        job.results["outcome"] = "not read: the trial computes no difference and no rule"
    else:
        job.results["outcome"] = f"not read: check {' and '.join(failed)} failed"
    write_tables(job, tables)  # 7
    for check in failed:
        print(FAIL_LINE.format(check.removeprefix("F")))
    minutes = (time.monotonic() - job.started) / 60
    job.info.update(status="done", finished_utc=utc_now(), minutes=round(minutes, 2))
    job.save()
    print(f"tables: {', '.join(tables)} in {job.relative}")
    print(f"done in {minutes:.1f} min")


def main() -> None:
    os.umask(0o077)
    sys.unraisablehook = quiet_unraisable  # a finaliser's message could hold anything
    progress = Progress()
    try:
        evaluate_job(progress)
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
