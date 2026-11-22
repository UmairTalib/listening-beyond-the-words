"""Build the re-voiced clip x1 of every segment, one array task per share of the sessions (CPU job).

Submit from an OMNI login node, once the check job has passed and its clips were heard:
    scripts/slurm/submit.sh [--time=HH:MM:SS] [--mem=<size>] build_revoiced derived/revoice_check/run-<job id>

Array task t of 0-19 (revoice.TASKS) renders the sessions that revoice.assign_tasks gives
it, largest first, so that the tasks take about equally long. Each task first refuses a
check run that is not done or that was made with another commit, config or pinned
inputs, and loads and checks Kokoro as the check job did. Then it renders each of its
segments with revoice.render_segment, the code path of the test render (pilot protocol,
section "Clip versions", bullet "Re-voiced clip"), and writes into
$LBW_WORKSPACE/views/x1/run-<array job id>/:
  <session>/<segment>.flac    the clip as 16 kHz mono 16-bit FLAC, read back after writing
  tasks/<t>/manifest.parquet  one row per clip of the task (clips.MANIFEST_COLUMNS)
  tasks/<t>/parts.parquet     one row per part: voice, seed, the text read, its phonemes and
                              how the render was trimmed
  tasks/<t>/errors.log        if the task fails: traceback frames and error types only (a
                              task that fails before this folder exists prints them to
                              stderr)
  tasks/<t>/run.json          commit, config, settings, inputs, check run, model, voices,
                              environment, Slurm, timings and counts
parts.parquet holds participant text: only the researcher opens it, on OMNI. If any task
fails, nothing is merged, and the whole array runs again into a new run folder. When
every task is done, scripts/merge_revoiced.py checks them and writes the run's manifest.
Prints aggregate numbers only: never text, never one segment's values.
"""

import contextlib
import json
import os
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from listening_beyond_words.clips import MANIFEST_COLUMNS, read_lines, write_view
from listening_beyond_words.models import load_pins
from listening_beyond_words.revoice import (
    TASKS,
    Kokoro,
    X1Inputs,
    X1Settings,
    load_inputs,
    render_segment,
    segment_words,
)
from listening_beyond_words.runinfo import (
    Progress,
    StepError,
    environment,
    git_state,
    quiet_unraisable,
    record_failure,
    sha256_file,
    stop_on_sigterm,
    utc_now,
    write_json,
)
from listening_beyond_words.settings import Workspace, get_workspace

CONFIG = Path("configs/pilot/data.yaml")
MODELS = Path("configs/pilot/models.yaml")
USAGE = "usage: scripts/slurm/submit.sh build_revoiced derived/revoice_check/run-<job id>"
PACKAGES = (
    "kokoro",
    "misaki",
    "spacy",
    "en-core-web-sm",
    "thinc",
    "phonemizer-fork",
    "espeakng-loader",
    "num2words",
    "torch",
    "transformers",
    "soxr",
    "soundfile",
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "pyyaml",
)


def check_run(
    workspace: Workspace, folder: str, git: dict[str, object], config_sha256: str, inputs: X1Inputs
) -> dict[str, Any]:
    """The check run that this build follows: done, and made from the same code and inputs."""
    run = workspace.resolve(folder)
    path = run / "run.json"
    if not path.is_file():
        raise StepError(f"no check run in {folder}")
    info = json.loads(path.read_text(encoding="utf-8"))
    if info.get("step") != "check_revoice" or info.get("status") != "done":
        raise StepError("the check run did not pass: its run.json does not say done")
    if info["git"]["commit"] != git["commit"] or info["git"]["dirty"]:
        raise StepError("the check run was made with another commit; run the check job again")
    if info["config"]["sha256"] != config_sha256:
        raise StepError("the check run was made with another config; run the check job again")
    if info["inputs"] != inputs.pins:
        raise StepError("the check run was made from other pinned inputs")
    test = run / "test.parquet"
    if sha256_file(test) != info["outputs"]["test.parquet"]:
        raise StepError("the check run's test.parquet does not match its run.json")
    return {
        "run": str(run.relative_to(workspace.root)),
        "run_json_sha256": sha256_file(path),
        "test_sha256": info["outputs"]["test.parquet"],
        "cpu": info.get("environment", {}).get("cpu"),
    }


def write_table(table: pd.DataFrame, path: Path) -> str:
    """Write a table as Parquet and return its SHA-256."""
    table.to_parquet(path, index=False)
    return sha256_file(path)


def build(progress: Progress) -> None:
    """The task: check, load, render its sessions, write its records."""
    stop_on_sigterm()
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    job = os.environ.get("SLURM_JOB_ID")
    if not (array_job and task_id and job):
        raise StepError(f"submit this as an array job: {USAGE}")
    if not task_id.isdigit() or not 0 <= int(task_id) < TASKS:
        raise StepError(f"array task {task_id} is not one of 0 to {TASKS - 1}")
    task = int(task_id)
    if len(sys.argv) != 2:
        raise StepError(USAGE)
    git = git_state()
    if git["dirty"]:
        raise StepError("the checkout has uncommitted changes; commit them on the Mac and pull")
    started = time.monotonic()
    workspace = get_workspace()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config_sha256 = sha256_file(CONFIG)
    settings = X1Settings.from_config(config)
    run_dir = workspace.views / "x1" / f"run-{array_job}"
    if (run_dir / "run.json").exists():
        raise StepError("this run was merged already; a new build needs a new array job")
    (run_dir / "tasks").mkdir(parents=True, exist_ok=True)  # the first task to start makes it
    task_dir = run_dir / "tasks" / str(task)
    task_dir.mkdir(exist_ok=False)  # a second attempt of this task never writes here
    progress.run_dir = task_dir
    info: dict[str, Any] = {
        "step": "build_revoiced",
        "status": "running",
        "started_utc": utc_now(),
        "task": task,
        "tasks": TASKS,
        "git": git,
        "config": {"file": str(CONFIG), "sha256": config_sha256},
        "settings": {
            "x1": config["views"]["x1"],
            "peak_limit": config["views"]["peak_limit"],
            "fade_ms": config["x0_cut"]["fade_ms"],
            "resolved": asdict(settings),
        },
        "slurm": {
            "array_job_id": array_job,
            "array_task_id": task,
            "job_id": job,
            "node": socket.gethostname(),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
    }
    progress.info = info
    write_json(task_dir / "run.json", info)
    inputs = load_inputs(workspace, config, settings)
    info["inputs"] = inputs.pins
    info["check_run"] = check_run(workspace, sys.argv[1], git, config_sha256, inputs)
    loading = time.monotonic()
    kokoro = Kokoro.load(workspace, load_pins(MODELS)["kokoro"], settings)
    load_seconds = time.monotonic() - loading
    info.update(model=kokoro.identity, environment=environment(PACKAGES))
    info["environment"]["libsndfile"] = sf.__libsndfile_version__
    write_json(task_dir / "run.json", info)

    mine = sorted(session for session, owner in inputs.tasks.items() if owner == task)
    texts = dict(zip(inputs.segments["segment"], inputs.segments["text"], strict=True))
    parts = {str(s): group for s, group in inputs.parts.groupby("segment", sort=True)}
    manifest: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    rendering = time.monotonic()
    for session in mine:
        lines = read_lines(workspace.edaic / "sessions", session)
        (run_dir / str(session)).mkdir(exist_ok=False)  # each session belongs to one task
        own = inputs.segments[inputs.segments["session"] == session]["segment"]
        for segment in sorted(map(str, own)):
            progress.segment = segment
            group = parts[segment]
            if " ".join(segment_words(group, lines)) != texts[segment]:
                raise StepError(
                    f"segment {segment}: the transcript lines do not give its words",
                    segment=segment,
                )
            rendered = render_segment(
                kokoro, group, lines, inputs.voices[session], segment, settings
            )
            name = f"{session}/{segment}.flac"
            values = write_view(run_dir / name, rendered.samples, settings.rate)
            row = {
                "segment": segment,
                "session": session,
                "file": name,
                **values,
                "part_starts": list(rendered.part_starts),
                "part_ends": list(rendered.part_ends),
                "peak_limited": rendered.peak_limited,
                "gain_db": 20 * float(np.log10(rendered.gain)),
            }
            manifest.append({column: row[column] for column in MANIFEST_COLUMNS})
            records.extend(rendered.parts)
    progress.segment = None
    render_seconds = time.monotonic() - rendering
    outputs = {
        "manifest.parquet": write_table(
            pd.DataFrame(manifest, columns=list(MANIFEST_COLUMNS)), task_dir / "manifest.parquet"
        ),
        "parts.parquet": write_table(pd.DataFrame(records), task_dir / "parts.parquet"),
    }
    seconds = sum(row["samples"] for row in manifest) / settings.rate
    words = sum(int(record["n_words"]) for record in records)
    counts = {
        "sessions": len(mine),
        "segments": len(manifest),
        "parts": len(records),
        "words": words,
        "hours": seconds / 3600,
        "peak_limited": sum(bool(row["peak_limited"]) for row in manifest),
        "load_seconds": round(load_seconds, 1),
        "render_seconds": round(render_seconds, 1),
        "seconds_per_word": render_seconds / words if words else None,
    }
    minutes = (time.monotonic() - started) / 60
    info.update(
        status="done",
        finished_utc=utc_now(),
        minutes=round(minutes, 2),
        counts=counts,
        outputs=outputs,
    )
    write_json(task_dir / "run.json", info)
    print(
        f"task {task} of {TASKS}: {counts['sessions']} sessions, {counts['segments']:,} segments,"
        f" {counts['parts']:,} parts, {counts['hours']:.2f} h of audio; peak-limited"
        f" {counts['peak_limited']}; Kokoro loaded in {load_seconds:.0f} s, rendered in"
        f" {render_seconds / 60:.1f} min"
    )
    print(f"task run: {task_dir.relative_to(workspace.root)}")
    print(f"done in {minutes:.1f} min")


def main() -> None:
    os.umask(0o077)
    sys.unraisablehook = quiet_unraisable  # a finaliser's message could hold a word
    progress = Progress()
    try:
        build(progress)
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
