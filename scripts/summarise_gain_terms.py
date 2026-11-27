"""Summarise what training logged about the gain term, over the kept runs (login node).

Run on an OMNI login node, once, from a checkout at the newest v0.1.x tag with its environment
built at that tag (git checkout v0.1.x, then bash scripts/setup_omni_env.sh):
    "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/summarise_gain_terms.py
It takes no option and runs in seconds.

docs/pilot_diagnostics.md, section "Gain-term summary", fixes what it summarises: the
measures that training logged per epoch, over the 10 CPG runs, with the 10 B1 runs and B0's
10 runs beside (diagnostics.gain_term_table). No rule reads the summary. The kept runs are
those that configs/pilot/evaluate.yaml pins (evaluation.EvaluateSettings), and
configs/pilot/data.yaml gives the label and partition pins that a kept run must have used.

Stages, in one process:
  1. Guards: a clean checkout at the newest v0.1.x tag of the clone and the installed code
     of that tag's version (runinfo.release_guard); evaluate.yaml read, a key given twice
     refused (config.read_settings), with at least one run pinned. Then the run folder and
     run.json (running).
  2. The pins, before any record is read for the summary: each kept run's run.json by its
     SHA-256 alone (evaluation.pinned_record) and its record (evaluation.kept_run: done,
     not started with --allow-dirty, check F1 passing, the label and partition pins of
     data.yaml, from the first runs' commit or the newest v0.1.x tag of its clone). Only
     the run.json files are opened: no scores file, no step log (metrics.jsonl) and no label.
  3. The table, from each record's per-epoch measures (results.rules.epochs) only, written
     through reports.write_table; then run.json (done).

Writes $LBW_WORKSPACE/derived/gain_terms/run-<UTC time>/ with
  gain_terms.csv  per arm, epoch and measure: the number of runs with a value, the mean,
                  minimum and maximum over them, and, on the row of the active share over all
                  bags, the pooled share (the summed active bags over the summed bags)
  run.json        commit and tags, the release and the installed version, the configs'
                  SHA-256, each pin with its run's commit and what lets it count, the
                  environment, and aggregate results: the runs per arm, the epochs, the
                  pooled active shares and the table's SHA-256
A cell with no number reads as text, never blank. A run that fails leaves errors.log there:
traceback frames and error types, never a message. The script prints and records aggregate
numbers only: no participant ID and no number per participant or segment. A refusal names a
rule, a run and counts.
"""

import contextlib
import hashlib
import os
import socket
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from listening_beyond_words.config import read_settings
from listening_beyond_words.diagnostics import POOLED, gain_term_table
from listening_beyond_words.evaluation import (
    ARMS,
    EvaluateSettings,
    RunPin,
    kept_run,
    pinned_record,
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
    utc_stamp,
    write_json,
)
from listening_beyond_words.settings import Workspace, get_workspace

CONFIG = Path("configs/pilot/data.yaml")
EVALUATE = Path("configs/pilot/evaluate.yaml")
STEP = "summarise_gain_terms"
FOLDER = "gain_terms"  # derived/gain_terms/run-<UTC time>/
TABLE = "gain_terms.csv"
PACKAGES = ("numpy", "pandas", "pyyaml", PACKAGE)
USAGE = "usage: summarise_gain_terms.py (no arguments)"
WHAT = "the gain-term summary"
# The rule that the summary runs only from its release (runinfo.release_guard).
ORDER = 'pilot diagnostics, section "Order"'
NAMES = {"b0": "B0", "b1": "B1", "cpg": "CPG"}


@dataclass
class Summary:
    """The summary being made: its workspace, settings, folder and record."""

    workspace: Workspace
    config: dict[str, Any]
    settings: EvaluateSettings
    run_dir: Path
    info: dict[str, Any]
    started: float

    def save(self) -> None:
        """Write run.json as it stands."""
        write_json(self.run_dir / "run.json", self.info)

    @property
    def relative(self) -> str:
        """The run folder, relative to the workspace."""
        return self.run_dir.relative_to(self.workspace.root).as_posix()


def start(state: Progress) -> Summary:
    """Stage 1: refuse anything but a clean checkout at the newest release; the folder."""
    stop_on_sigterm()
    if sys.argv[1:]:
        raise StepError(USAGE)
    started = time.monotonic()
    git = git_state()
    if git["dirty"]:
        raise StepError(
            f"the checkout has uncommitted changes: {WHAT} runs only from a release ({ORDER})"
        )
    installed = installed_version()
    release = release_guard(git, installed, WHAT, ORDER)
    content, evaluate_sha256 = read_settings(EVALUATE)
    settings = EvaluateSettings.from_config(content)
    if settings.version is None:
        raise StepError(
            f"no training run is pinned in {EVALUATE.name} (runs): the summary reads the kept"
            " runs that it pins"
        )
    data = CONFIG.read_bytes()
    config = yaml.safe_load(data.decode("utf-8"))
    workspace = get_workspace()
    run_dir = workspace.derived / FOLDER / f"run-{utc_stamp()}"
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
        "config": {"file": str(EVALUATE), "sha256": evaluate_sha256},
        "data_config": {"file": str(CONFIG), "sha256": hashlib.sha256(data).hexdigest()},
        "host": {"node": socket.gethostname(), "cpu": machine["cpu"]},
        "inputs": {},
        "environment": machine,
        "results": {},
        "outputs": {},
    }
    state.info = info
    summary = Summary(workspace, config, settings, run_dir, info, started)
    summary.save()
    print(
        f"{WHAT} at the release tag {release} (installed version {installed}), commit"
        f" {str(git['commit'])[:12]}; on {socket.gethostname()}"
    )
    return summary


def read_records(summary: Summary) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Stage 2: each kept run's run.json by its pin alone, and its record; nothing else."""
    records: dict[tuple[str, int, int], dict[str, Any]] = {}
    inputs: dict[str, Any] = {}
    first_runs_commit = summary.settings.first_runs_commit
    pins: Mapping[tuple[str, int, int], RunPin] = summary.settings.runs
    for key in sorted(pins, key=lambda key: (ARMS.index(key[0]), key[1], key[2])):
        pin = pins[key]
        record = pinned_record(summary.workspace.root, pin)
        counts_by = kept_run(record, pin.run, *key, summary.config, first_runs_commit)
        records[key] = record
        arm, round_number, fold = key
        git = record["git"]
        inputs.setdefault(arm, {}).setdefault(f"r{round_number}", {})[f"f{fold}"] = {
            "run": pin.run,
            "record_sha256": pin.record_sha256,
            "commit": git.get("commit"),
            "tags": list(git.get("tags") or []),
            "counts_by": counts_by,
        }
    summary.info["inputs"]["runs"] = inputs
    summary.save()
    counts = {arm: sum(key[0] == arm for key in records) for arm in NAMES}
    shown = ", ".join(f"{count} of {NAMES[arm]}" for arm, count in counts.items() if count)
    print(
        f"pins: {len(records)} kept runs match their pins and records ({shown}); only their"
        " run.json files are read"
    )
    return records


def summarise(state: Progress) -> None:
    """The summary: stages 1 to 3."""
    summary = start(state)  # 1
    records = read_records(summary)  # 2
    table = gain_term_table(records)  # 3
    pooled: dict[str, dict[str, Any]] = {}  # only B1 and CPG have the active share's row
    for row in table[table["measure"] == POOLED].itertuples(index=False):
        pooled.setdefault(str(row.arm), {})[str(row.epoch)] = row.pooled_share
    summary.info["results"] = {
        "runs": {arm: sum(key[0] == arm for key in records) for arm in NAMES},
        "epochs": sorted({int(epoch) for epoch in table["epoch"]}),
        "rows": len(table),
        "pooled_active_share": pooled,
    }
    summary.save()
    for arm, shares in pooled.items():
        shown = ", ".join(
            f"epoch {epoch} {share:.4f}" if isinstance(share, float) else f"epoch {epoch} {share}"
            for epoch, share in shares.items()
        )
        print(f"pooled active share, {NAMES[arm]}: {shown}")
    summary.info["outputs"][TABLE] = write_table(table, summary.run_dir / TABLE)
    seconds = time.monotonic() - summary.started
    summary.info.update(status="done", finished_utc=utc_now(), seconds=round(seconds, 2))
    summary.save()
    print(f"table: {TABLE} ({len(table)} rows) in {summary.relative}")
    print(f"done in {seconds:.1f} s")


def main() -> None:
    os.umask(0o077)
    sys.unraisablehook = quiet_unraisable  # a finaliser's message could hold anything
    progress = Progress()
    try:
        summarise(progress)
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
