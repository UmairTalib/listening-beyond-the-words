"""Simulate the main study's power again with the pilot's numbers (CPU job).

Submit from an OMNI login node, from a checkout at the newest v0.1.x tag with its environment
built at that tag (git checkout v0.1.x, then bash scripts/setup_omni_env.sh), once:
    scripts/slurm/submit.sh simulate_power
The job takes no option: everything it reads is pinned.

docs/pilot_diagnostics.md, section "The main study's power", fixes the simulation, and
configs/pilot/power.yaml holds its settings (power.PowerSettings). The simulation reads no
participant data and no file of the workspace: its inputs are two published tables of the
pilot's analysis, reports/pilot/summary.csv and reports/pilot/fold_values.csv, read from the
checkout and pinned by their SHA-256 in power.yaml. It is descriptive: no rule of the pilot
reads it.

Stages, in one process with worker processes for the blocks of simulated studies:
  1. Guards: a Slurm job; a clean checkout at the newest v0.1.x tag of the clone; the
     installed code of that tag's version (runinfo.release_guard); power.yaml read, a key
     given twice refused (config.read_settings), every value checked. Then the run folder
     and run.json (running).
  2. The two tables, each checked against its pin before anything else is read, then the
     pilot's numbers that the simulation uses (power.read_pilot): B0's AUROC, the agreement,
     Δ with its rounds and bootstrap standard error, the ends of the 80% intervals, the arms'
     per-round AUROCs and the class counts per fold.
  3. The workers: one per CPU of the job, started by spawn, with one BLAS thread each, which
     each worker checks (power.Workers); run.json records the thread pools measured in one
     of them. Every block of studies has its own stream, so the tables do not depend on the
     number of workers.
  4. Check 1 (power.design_check): on the study design's assumptions, the model must
     reproduce its figures, or the job stops as a code fault. Check 2
     (power.resample_check): the decisions at the simulation's resamples against those at
     10,000, which sets the resamples of every table.
  5. The fits of every view (power.fit_views): the lowest ρ, r fitted at each ρ and in the
     sensitivity rows. A fit at a ρ point that misses its target stops the job here, before
     any study; a sensitivity row that misses keeps r where the model comes closest, is
     reported as not reached, and the job goes on.
  6. The study design's three scenarios, simulated for the main study and printed beside the
     design's own figures, not checked (power.design_outcomes).
  7. The simulated studies of every fitted scenario at every true gain
     (power.simulate_views).
  8. power.csv and power_fit.csv (reports.write_table), the power line and the limits
     printed, then run.json (done).

Writes $LBW_WORKSPACE/derived/power/run-<job id>/ with
  power.csv      per view, scenario, true gain and α: the power over new samples of people
                 with its Monte Carlo standard error and, on x0, the power after a Green pilot
                 with the number of Green pilots
  power_fit.csv  per view and scenario: B0's AUROC, ρ, r, what r is fitted to and whether it
                 met its target ("met" or "not reached", a sensitivity row only); the model's
                 agreement and mean bootstrap standard error beside the measured ones; the
                 main study's standard error; the share of the round difference; the
                 correlation of the pilot's and the main study's Δ; the true gain needed for
                 80% power at each α; and the arms' per-round AUROCs
  run.json       commit and tags, the release and the installed version, the config and its
                 SHA-256, the two tables with their SHA-256, the seed and its streams, the
                 workers with a worker's measured thread pools, the environment and the Slurm
                 job, and the results: both checks, every fit, the design's scenarios, the
                 power line, the limits and the time of each stage, and the tables' SHA-256
A job that fails also leaves errors.log there: traceback frames and error types, never a
message. The job prints and records aggregate numbers only.
"""

import contextlib
import hashlib
import io
import math
import os
import socket
import sys
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from listening_beyond_words.config import read_settings
from listening_beyond_words.power import (
    DESIGN,
    DESIGN_NAMES,
    GREEN_VIEW,
    NOT_REACHED,
    ONE_THREAD,
    Design,
    Outcome,
    PilotView,
    PowerSettings,
    Scenario,
    Workers,
    alpha_name,
    design_check,
    design_outcomes,
    fit_table,
    fit_views,
    gain_for_power,
    power_table,
    read_pilot,
    resample_check,
    simulate_views,
)
from listening_beyond_words.reports import write_table
from listening_beyond_words.runinfo import (
    PACKAGE,
    Progress,
    StepError,
    environment,
    git_state,
    installed_version,
    quiet_unraisable,
    record_failure,
    release_guard,
    stop_on_sigterm,
    utc_now,
    write_json,
)
from listening_beyond_words.settings import Workspace, get_workspace

POWER = Path("configs/pilot/power.yaml")
# The two published tables of the pilot's analysis, read from the checkout, never from the
# workspace, and pinned in power.yaml (inputs_sha256).
SUMMARY = Path("reports/pilot/summary.csv")
FOLDS = Path("reports/pilot/fold_values.csv")
STEP = "simulate_power"
FOLDER = "power"  # derived/power/run-<job id>/
POWER_TABLE = "power.csv"
FIT_TABLE = "power_fit.csv"
PACKAGES = ("numpy", "pandas", "scipy", "threadpoolctl", "pyyaml", PACKAGE)
USAGE = "usage: simulate_power.py (no arguments)"
WHAT = "the power simulation"
# The rule that the simulation runs only from its release (runinfo.release_guard).
ORDER = 'pilot diagnostics, section "Order"'
# The study design's figures for the main study, printed beside the model's, not checked
# (pilot diagnostics, section "The main study's power", bullet "Checks first").
DESIGN_FIGURES = (
    "the study design's figures for the main study, printed beside and not checked, because it"
    " does not name their test: at one-sided 0.05, 80% power at a true gain of about 0.06 with"
    " low noise to 0.10 with high noise, and 27% to 51% power at a true gain of 0.04"
)
DESIGN_GAIN = 0.04  # the true gain of the design's second figure
# Printed with the tables (pilot diagnostics, section "The main study's power", bullets
# "Outputs" and "Limits").
LIMITS = (
    "limits: normal scores, without segments or bags",
    "limits: the same ρ in both arms, and no training part shared by the paired arms (it"
    " cancels in Δ)",
    "limits: rev stands for one mask, not the mean over the main study's three",
    "limits: the study design's main study trains on about 128 people per fold, the pilot on"
    " about 151",
    "limits: the power at a true gain equal to the pilot's Δ is no forecast, because the rule"
    " selected that Δ",
    "note: at a true gain of 0 a rate can exceed α, because the participant bootstrap does not"
    " see training noise",
)


# Stage 1: the job ---------------------------------------------------------------------------


@dataclass
class Job:
    """The simulation being made: its settings, workers, folder and record."""

    job: str
    workspace: Workspace
    settings: PowerSettings
    workers: int
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


def worker_count() -> int:
    """One worker per CPU that Slurm gives the job."""
    text = os.environ.get("SLURM_CPUS_PER_TASK") or "1"
    try:
        count = int(text)
    except ValueError:
        count = 0
    if count < 1:
        raise StepError(f"SLURM_CPUS_PER_TASK must be a whole number of at least 1, not {text!r}")
    return count


def start(state: Progress) -> Job:
    """Stage 1: refuse to run outside a clean Slurm job at the newest release; the folder."""
    stop_on_sigterm()
    if sys.argv[1:]:
        raise StepError(USAGE)
    started = time.monotonic()
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise StepError("submit this as a job: scripts/slurm/submit.sh simulate_power")
    git = git_state()
    if git["dirty"]:
        raise StepError(
            f"the checkout has uncommitted changes: {WHAT} runs only from a release ({ORDER})"
        )
    installed = installed_version()
    release = release_guard(git, installed, WHAT, ORDER)
    content, power_sha256 = read_settings(POWER)
    settings = PowerSettings.from_config(content)
    workers = worker_count()
    workspace = get_workspace()
    run_dir = workspace.derived / FOLDER / f"run-{job}"
    run_dir.mkdir(parents=True, exist_ok=False)
    state.run_dir = run_dir
    machine = environment(PACKAGES)
    seed = settings.seed
    info: dict[str, Any] = {
        "run_id": run_dir.name,
        "step": STEP,
        "status": "running",
        "started_utc": utc_now(),
        "git": git,
        "release": release,
        "installed_version": installed,
        "config": {"file": str(POWER), "sha256": power_sha256, "content": content},
        "settings": {
            "alphas": list(settings.alphas),
            "gains": {
                "first": settings.gains[0],
                "last": settings.gains[-1],
                "points": len(settings.gains),
            },
        },
        "slurm": {
            "job_id": job,
            "node": socket.gethostname(),
            "cpu": machine["cpu"],
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
        "workers": {"count": workers},
        "seeds": {
            "seed": seed,
            "blocks": settings.blocks,
            "streams": {
                "design_check": f'runinfo.item_seed({seed}, "design_check", <scenario>, <block>)',
                "resample_check": f'runinfo.item_seed({seed}, "resample_check", "middle", <block>)',
                "design": f'runinfo.item_seed({seed}, "design", <scenario>, <block>)',
                "calibration": f'runinfo.item_seed({seed}, <view>, "calibration", <block>)',
                "studies": f"runinfo.item_seed({seed}, <view>, <scenario>, <block>)",
            },
        },
        "inputs": {},
        "environment": machine,
        "results": {},
        "outputs": {},
    }
    state.info = info
    run = Job(job, workspace, settings, workers, run_dir, info, started)
    run.save()
    print(
        f"{WHAT} at the release tag {release} (installed version {installed}), commit"
        f" {str(git['commit'])[:12]}; job {job} on {socket.gethostname()}; CPU {machine['cpu']};"
        f" {workers} workers"
    )
    return run


# Stage 2: the pilot's numbers ---------------------------------------------------------------


def pinned_table(path: Path, pin: str, name: str) -> pd.DataFrame:
    """A published table, checked against its pin before it is read; every cell as text."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise StepError(
            f"{path} does not exist: the simulation reads it from the checkout"
        ) from None
    found = hashlib.sha256(data).hexdigest()
    if found != pin:
        raise StepError(
            f"{path}: SHA-256 {found[:12]}... does not match its pin in power.yaml"
            f" (inputs_sha256.{name}, {pin[:12]}...)"
        )
    return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)


def read_inputs(job: Job) -> tuple[dict[str, PilotView], Design]:
    """Stage 2: both tables checked against their pins, then the numbers that set the model."""
    began = time.monotonic()
    settings = job.settings
    summary = pinned_table(SUMMARY, settings.summary_sha256, "summary")
    folds = pinned_table(FOLDS, settings.folds_sha256, "folds")
    job.info["inputs"] = {
        "summary": {"file": str(SUMMARY), "sha256": settings.summary_sha256},
        "folds": {"file": str(FOLDS), "sha256": settings.folds_sha256},
    }
    pilots, design = read_pilot(summary, folds, settings)
    positives, negatives = design.classes
    job.info["settings"]["design"] = {
        "positives": list(design.positives),
        "negatives": list(design.negatives),
        "pilot_rounds": design.pilot_rounds,
        "main_rounds": settings.rounds,
    }
    print(
        f"pins: {SUMMARY} and {FOLDS} match power.yaml; {positives} positive and {negatives}"
        f" negative people in {len(design.positives)} folds per round, the same in both rounds"
    )
    for pilot in pilots.values():
        print(
            f"{pilot.view}: B0's AUROC {pilot.b0_auroc:.4f} (80% interval {pilot.b0_ends[0]:.4f}"
            f" to {pilot.b0_ends[1]:.4f}); Δ {pilot.delta:.4f} (round 1 {pilot.delta_rounds[0]:.4f},"
            f" round 2 {pilot.delta_rounds[1]:.4f}; bootstrap SE {pilot.delta_se:.4f}); agreement"
            f" {pilot.agreement:.4f} (80% interval {pilot.agreement_ends[0]:.4f} to"
            f" {pilot.agreement_ends[1]:.4f})"
        )
    job.stage("inputs", began)
    return pilots, design


# Stages 3 to 8 ------------------------------------------------------------------------------


def gain_text(value: float | str) -> str:
    """A true gain for 80% power, at 4 decimals, or the text that it lies above the grid."""
    return value if isinstance(value, str) else f"{value:.4f}"


def gains_at(outcome: Outcome, settings: PowerSettings) -> dict[str, float | str]:
    """The true gain for 80% power at each α, by α."""
    power = outcome.power()
    return {
        alpha_name(alpha): gain_for_power(power[index], settings.gains)
        for index, alpha in enumerate(settings.alphas)
    }


def record_workers(job: Job, workers: Workers) -> None:
    """Stage 3: how the workers start, and the thread pools measured in one of them."""
    pools = workers.pools()  # a worker whose pools run more than one thread breaks the pool
    job.info["workers"].update(
        start=workers.start_method,
        environment=dict.fromkeys(ONE_THREAD, "1"),
        thread_pools=pools,
    )
    job.save()
    seen = "; ".join(f"{pool['internal_api']} with {pool['num_threads']}" for pool in pools)
    print(
        f"workers: {job.workers}, started by {workers.start_method}, each with one BLAS thread;"
        f" a worker's thread pools: {seen or 'none that threadpoolctl sees'}"
    )


def run_checks(job: Job, design: Design, workers: Workers) -> int:
    """Stage 4: the design check, which stops the job on a miss, then the resample check."""
    began = time.monotonic()
    settings = job.settings
    check = design_check(settings, design, workers)
    figures = {
        name: {"pilot_sd": sd, "design": expected}
        for name, sd, expected in zip(
            DESIGN_NAMES, check.pilot_sd, settings.design_pilot_sd, strict=True
        )
    }
    job.results["design_check"] = {
        "b0_auroc": settings.design_auroc,
        "pilots": settings.design_pilots,
        "pilot_delta_sd": figures,
        "one_arm_sd": {"middle": check.one_arm_sd, "design": settings.design_one_arm_sd},
        "max_gap": settings.design_max_gap,
        "passed": not check.misses,
    }
    job.save()
    parts = "; ".join(
        f"{name} {values['pilot_sd']:.4f} (design {values['design']})"
        for name, values in figures.items()
    )
    print(
        f"design check (B0's AUROC {settings.design_auroc}, {settings.design_pilots:,} pilots per"
        f" scenario): SD of the pilot's Δ: {parts}; one arm's one-round AUROC SD, middle"
        f" {check.one_arm_sd:.4f} (design {settings.design_one_arm_sd}); each within"
        f" {settings.design_max_gap}: {'pass' if not check.misses else 'MISS'}"
    )
    if check.misses:
        raise StepError(
            "the design check misses, a code fault that stops the job (pilot diagnostics,"
            ' section "The main study\'s power", bullet "Checks first"): ' + "; ".join(check.misses)
        )
    resampled = resample_check(settings, design, workers)
    changed = {
        f"{gain:g}": {
            alpha_name(alpha): float(resampled.changed[a, g])
            for a, alpha in enumerate(settings.alphas)
        }
        for g, gain in enumerate(settings.check_gains)
    }
    job.results["resample_check"] = {
        "studies": settings.check_studies,
        "resamples": [settings.resamples, settings.check_resamples],
        "compared": "the decision on the first resamples of each study against all of them",
        "changed": changed,
        "max_changed": settings.max_changed,
        "passed": resampled.passed,
        "resamples_used": resampled.resamples,
        "redraws": resampled.redraws,
    }
    largest = float(resampled.changed.max())
    print(
        f"resample check ({settings.check_studies:,} studies of the design's middle scenario at"
        f" each true gain): the largest share of decisions that differ between the first"
        f" {settings.resamples:,} of each study's {settings.check_resamples:,} resamples and all"
        f" of them is {largest:.4f} (at most {settings.max_changed}):"
        f" {'pass' if resampled.passed else 'fail'}; every table uses {resampled.resamples:,}"
        " resamples"
    )
    job.stage("checks", began)
    return resampled.resamples


def run_fits(
    job: Job, pilots: dict[str, PilotView], design: Design, workers: Workers, resamples: int
) -> list[Scenario]:
    """Stage 5: the fits of every view, before any study; a ρ point's miss stops the job."""
    began = time.monotonic()
    scenarios = fit_views(pilots, job.settings, design, workers, resamples, report=reported(job))
    job.stage("fits", began)
    return scenarios


def run_design(job: Job, design: Design, workers: Workers, resamples: int) -> None:
    """Stage 6: the main study in the design's scenarios, beside the design's own figures."""
    began = time.monotonic()
    settings = job.settings
    outcomes = design_outcomes(settings, design, workers, resamples)
    at = settings.gains.index(DESIGN_GAIN) if DESIGN_GAIN in settings.gains else None
    recorded = {}
    print(DESIGN_FIGURES)
    for outcome in outcomes:
        gains = gains_at(outcome, settings)
        power = outcome.power()
        at_gain = (
            {alpha_name(alpha): float(power[i, at]) for i, alpha in enumerate(settings.alphas)}
            if at is not None
            else {}
        )
        recorded[outcome.name] = {
            "r": outcome.model.r,
            "rho": outcome.model.rho,
            "gain_80": gains,
            f"power_at_{DESIGN_GAIN:g}": at_gain,
            "main_study_se": outcome.main_se,
            "redraws": outcome.redraws,
        }
        line = "; ".join(f"{gain_text(gain)} at {alpha}" for alpha, gain in gains.items())
        extra = (
            f"; power at a true gain of {DESIGN_GAIN:g}, at {alpha_name(settings.alpha)}:"
            f" {at_gain[alpha_name(settings.alpha)]:.4f}"
            if at_gain
            else ""
        )
        print(
            f"{DESIGN}, {outcome.name} noise (r {outcome.model.r}, ρ {outcome.model.rho}, B0's"
            f" AUROC {outcome.model.b0_auroc}): 80% power at a true gain of {line}{extra}; main"
            f" study's SE {outcome.main_se:.4f}"
        )
    job.results["design_main_study"] = recorded
    job.stage("design", began)


def run_studies(
    job: Job, scenarios: Sequence[Scenario], design: Design, workers: Workers, resamples: int
) -> list[Outcome]:
    """Stage 7: the simulated studies of every fitted scenario, at every true gain."""
    began = time.monotonic()
    outcomes = simulate_views(
        scenarios, job.settings, design, workers, resamples, report=reported(job)
    )
    job.stage("studies", began)
    return outcomes


def reported(job: Job) -> Callable[[Scenario | Outcome], None]:
    """Print and record each fitted scenario and each simulated one as it comes."""
    settings = job.settings
    views: dict[str, Any] = job.results.setdefault("views", {})

    def report(item: Scenario | Outcome) -> None:
        if isinstance(item, Scenario):
            entry = views.setdefault(item.view, {}).setdefault(item.name, {})
            entry.update(
                b0_auroc=item.model.b0_auroc,
                rho=item.model.rho,
                r=item.model.r,
                fitted_to=item.fitted_to,
                target=item.target,
                agreement=item.agreement,
                bootstrap_se=item.bootstrap_se,
                round_share=item.round_share,
                pilot_redraws=item.redraws,
                target_met=item.target_met,
            )
            print(
                f"{item.view}, {item.name}: r {item.model.r:.4f} at ρ {item.model.rho:.4f}, B0's"
                f" AUROC {item.model.b0_auroc:.4f}, fitted to {item.fitted_to} {item.target:.4f};"
                f" the model's agreement {item.agreement:.4f}, mean bootstrap SE"
                f" {item.bootstrap_se:.4f}, round-difference share {item.round_share:.4f}"
            )
            if not item.target_met:
                print(
                    f"{item.view}, {item.name}: {NOT_REACHED}: the model comes closest at this r,"
                    f" more than {settings.tolerance:g} from its target; the row is simulated"
                    " there and reported as not reached, and the job goes on (pilot diagnostics,"
                    ' section "The main study\'s power", bullet "Which number sets what")'
                )
        else:
            entry = views.setdefault(item.view, {}).setdefault(item.name, {})
            gains = gains_at(item, settings)
            entry.update(
                gain_80=gains,
                main_study_se=item.main_se,
                pilot_main_correlation=item.correlation
                if math.isfinite(item.correlation)
                else None,
                redraws=item.redraws,
            )
            green = ""
            if item.view == GREEN_VIEW:
                zero = settings.gains.index(0.0)
                entry["green_pilots_at_0"] = int(item.green[zero])
                green = f"; Green pilots at a true gain of 0: {int(item.green[zero]):,}"
            line = "; ".join(f"{gain_text(gain)} at {alpha}" for alpha, gain in gains.items())
            print(
                f"{item.view}, {item.name}: {item.studies:,} studies; 80% power at a true gain of"
                f" {line}; main study's SE {item.main_se:.4f}{green}"
            )
        job.save()

    return report


def power_lines(
    outcomes: Sequence[Outcome],
    settings: PowerSettings,
    not_reached: Collection[tuple[str, str]] = (),
) -> list[str]:
    """The power line: the range over all ρ points at each α, with the sensitivity rows beside.

    Pilot diagnostics, section "The main study's power", bullet "The power line": no ρ is
    chosen or dropped by its fit, which is only reported. A sensitivity row that did not meet
    its target (``not_reached``, by view and scenario) says so beside its value.
    """
    lines = []
    for view in settings.views:
        own = [outcome for outcome in outcomes if outcome.view == view]
        points = [outcome for outcome in own if outcome.name.startswith("rho_")]
        others = [outcome for outcome in own if not outcome.name.startswith("rho_")]
        for alpha in settings.alphas:
            name = alpha_name(alpha)
            found = [gains_at(outcome, settings)[name] for outcome in points]
            numbers = sorted(value for value in found if not isinstance(value, str))
            above = [value for value in found if isinstance(value, str)]
            ends = [gain_text(value) for value in numbers] + above
            span = ends[0] if ends[0] == ends[-1] else f"{ends[0]} to {ends[-1]}"
            beside = "; ".join(
                f"{outcome.name} {gain_text(gains_at(outcome, settings)[name])}"
                + (f" ({NOT_REACHED})" if (view, outcome.name) in not_reached else "")
                for outcome in others
            )
            lines.append(
                f"power line, {view}, one-sided {name}: 80% power at a true gain of {span} over"
                f" the {len(points)} ρ points (sensitivity rows: {beside})"
            )
    return lines


def power_job(state: Progress) -> None:
    """The job: stages 1 to 8."""
    job = start(state)  # 1
    pilots, design = read_inputs(job)  # 2
    settings = job.settings
    with Workers(job.workers) as workers:  # 3
        record_workers(job, workers)
        resamples = run_checks(job, design, workers)  # 4
        scenarios = run_fits(job, pilots, design, workers, resamples)  # 5
        run_design(job, design, workers, resamples)  # 6
        outcomes = run_studies(job, scenarios, design, workers, resamples)  # 7
    began = time.monotonic()  # 8
    tables = {
        POWER_TABLE: power_table(outcomes, settings),
        FIT_TABLE: fit_table(scenarios, outcomes, pilots, settings),
    }
    missed = {(item.view, item.name) for item in scenarios if not item.target_met}
    lines = power_lines(outcomes, settings, missed)
    for line in [*lines, *LIMITS]:
        print(line)
    job.results["power_line"] = {f"line_{index}": line for index, line in enumerate(lines, 1)}
    job.results["limits"] = list(LIMITS)
    job.results["rows"] = {name: len(table) for name, table in tables.items()}
    for name, table in tables.items():
        job.info["outputs"][name] = write_table(table, job.run_dir / name)
    job.stage("tables", began)
    minutes = (time.monotonic() - job.started) / 60
    job.info.update(status="done", finished_utc=utc_now(), minutes=round(minutes, 2))
    job.save()
    print(
        f"tables: {POWER_TABLE} ({len(tables[POWER_TABLE])} rows) and {FIT_TABLE}"
        f" ({len(tables[FIT_TABLE])} rows) in {job.relative}"
    )
    print(f"done in {minutes:.1f} min")


def main() -> None:
    os.umask(0o077)
    sys.unraisablehook = quiet_unraisable  # a finaliser's message could hold anything
    progress = Progress()
    try:
        power_job(progress)
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
