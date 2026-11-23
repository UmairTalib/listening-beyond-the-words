"""Check the array tasks of an x1 build and merge them into one run (OMNI login node).

Usage, inside the clone, with no uncommitted changes, once every task has finished:
    "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/merge_revoiced.py views/x1/run-<array job id>

Every check runs before anything is written, and a refused merge writes nothing:
  - tasks 0-19 (revoice.TASKS) each say done, with one commit, config, settings, inputs,
    check run, model, voice files and package versions; node and CPU may differ;
  - the inputs are still the pinned ones, and every session lies in the task that
    revoice.assign_tasks gives it, and in no other;
  - the segments are exactly those of x0's pinned manifest, with as many parts each;
  - each participant has the voice that revoice.assign_voices gives them, and each part
    the seed runinfo.item_seed(106, segment, position);
  - the tasks read as many words, with as many numbers written out, as the dry run of
    their check run, whose run.json is unchanged since the build; both counts follow
    from the pinned stored words and num2words, so this catches a num2words that writes
    some number in another number of words, or counts edited in parts.parquet, not other
    words of the same count, nor other capitals or marks;
  - every file is read again: its SHA-256 and samples match its task's manifest, and it
    passes views.check_segment_views with its own part bounds; no FLAC file anywhere in
    the run is missing from the tasks' manifests;
  - every task comes from this run's array job, so a rerun of single tasks copied in from
    another run is refused: a failed build runs again as a whole.
Then it writes into the run folder manifest.parquet (one row per clip,
clips.MANIFEST_COLUMNS), parts.parquet (one row per part; it holds participant text, so
only the researcher opens it, on OMNI) and, last, run.json with status done, so a folder
with run.json is merged and is never merged again. Pin the run in configs/pilot/data.yaml
(view_runs.x1) by the SHA-256 it prints. Prints aggregate numbers only. A refusal prints
why; any other error prints its type, and its traceback frames and type names to stderr,
never its message.
"""

import contextlib
import json
import os
import resource
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from listening_beyond_words.clips import MANIFEST_COLUMNS, audio_sha256, load_view_manifest
from listening_beyond_words.revoice import (
    TASKS,
    X1Settings,
    assign_tasks,
    load_inputs,
)
from listening_beyond_words.runinfo import (
    Progress,
    StepError,
    environment,
    git_state,
    item_seed,
    quiet_unraisable,
    record_failure,
    sha256_file,
    utc_now,
    write_json,
)
from listening_beyond_words.settings import get_workspace
from listening_beyond_words.views import check_segment_views

CONFIG = Path("configs/pilot/data.yaml")
USAGE = "usage: python scripts/merge_revoiced.py views/x1/run-<array job id>"
SHARED = ("git", "config", "settings", "inputs", "check_run", "tasks")
LONG_SECONDS = 30.0  # the encoder's window; longer clips are handled in step 7
PACKAGES = ("numpy", "pandas", "pyarrow", "soundfile", "pyyaml")


def same(infos: dict[int, dict[str, Any]], name: str, value: Any) -> Any:
    """The one value that every task has for ``name``; refuse if they differ."""
    values = {json.dumps(value(info), sort_keys=True) for info in infos.values()}
    if len(values) != 1:
        raise StepError(f"the tasks differ in {name}")
    return value(next(iter(infos.values())))


def model_without_cpu(info: dict[str, Any]) -> dict[str, Any]:
    """The model's record of a task without what depends on the node's CPU."""
    return {key: value for key, value in info["model"].items() if key != "cpu_capability"}


def packages(info: dict[str, Any]) -> dict[str, Any]:
    """The task's environment without the node's CPU."""
    return {key: value for key, value in info["environment"].items() if key != "cpu"}


def merge(progress: Progress) -> None:
    """The merge: every check first, then the three files, run.json last."""
    # A crash must not dump process memory into the clone: no core files (the soft limit).
    resource.setrlimit(resource.RLIMIT_CORE, (0, resource.getrlimit(resource.RLIMIT_CORE)[1]))
    if len(sys.argv) != 2:
        raise StepError(USAGE)
    git = git_state()
    if git["dirty"]:
        raise StepError("the checkout has uncommitted changes; commit them on the Mac and pull")
    started = time.monotonic()
    workspace = get_workspace()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    settings = X1Settings.from_config(config)
    run_dir = workspace.resolve(sys.argv[1])
    if (run_dir / "run.json").exists():
        raise StepError("this run is merged already")
    tasks_dir = run_dir / "tasks"
    found = sorted(path.name for path in tasks_dir.iterdir()) if tasks_dir.is_dir() else []
    if sorted(found) != sorted(str(task) for task in range(TASKS)):
        missing = sorted(set(map(str, range(TASKS))) - set(found), key=int)
        raise StepError(
            f"expected the folders of tasks 0 to {TASKS - 1}; missing {missing or 'none'},"
            f" {len(set(found) - set(map(str, range(TASKS))))} others"
        )
    infos = {
        task: json.loads((tasks_dir / str(task) / "run.json").read_text(encoding="utf-8"))
        for task in range(TASKS)
    }
    unfinished = [task for task, info in infos.items() if info.get("status") != "done"]
    if unfinished:
        raise StepError(f"tasks not done: {unfinished}; run the whole array again")
    array = run_dir.name.removeprefix("run-")
    foreign = [t for t, info in infos.items() if info["slurm"]["array_job_id"] != array]
    if foreign:
        raise StepError(
            f"tasks {foreign} come from another array job than {array}; run the whole array again"
        )
    shared = {name: same(infos, name, lambda info, n=name: info.get(n)) for name in SHARED}
    model = same(infos, "model", model_without_cpu)
    same(infos, "environment", packages)
    if shared["git"]["dirty"]:
        raise StepError("the tasks ran from a checkout with uncommitted changes")
    inputs = load_inputs(workspace, config, settings)
    if shared["inputs"] != inputs.pins:
        raise StepError("the tasks were built from other inputs than the pinned ones")
    if shared["config"]["sha256"] != sha256_file(CONFIG):
        raise StepError("the tasks were built with another config")

    manifests, part_tables = [], []
    owner: dict[int, int] = {}
    for task, info in infos.items():
        folder = tasks_dir / str(task)
        for name in ("manifest.parquet", "parts.parquet"):
            if sha256_file(folder / name) != info["outputs"][name]:
                raise StepError(f"task {task}: {name} does not match its run.json")
        manifest = pd.read_parquet(folder / "manifest.parquet")
        if list(manifest.columns) != list(MANIFEST_COLUMNS):
            raise StepError(f"task {task}: its manifest has other columns")
        for session in set(manifest["session"].astype(int)):
            if session in owner:
                raise StepError(f"session {session} lies in tasks {owner[session]} and {task}")
            owner[session] = task
        manifests.append(manifest)
        part_tables.append(pd.read_parquet(folder / "parts.parquet"))
    counts_by_session = Counter(inputs.segments["session"].astype(int).tolist())
    if owner != assign_tasks(counts_by_session):
        raise StepError("the sessions do not lie in the tasks that assign_tasks gives them")
    manifest = pd.concat(manifests, ignore_index=True).sort_values("segment", ignore_index=True)
    parts = pd.concat(part_tables, ignore_index=True).sort_values(
        ["segment", "position"], ignore_index=True
    )

    x0 = load_view_manifest(workspace, config, "x0").sort_values("segment", ignore_index=True)
    if manifest["segment"].tolist() != x0["segment"].tolist():
        raise StepError(
            f"the tasks hold {len(manifest):,} segments; x0's pinned manifest {len(x0):,},"
            " and they must be the same"
        )
    ours = manifest["part_starts"].map(len).tolist()
    if ours != x0["part_starts"].map(len).tolist():
        raise StepError("a segment has another number of parts than in x0")
    per_segment = parts.groupby("segment").size()
    if per_segment.index.tolist() != manifest["segment"].tolist() or per_segment.tolist() != ours:
        raise StepError("parts.parquet does not hold one row per part")
    positions = parts.groupby("segment")["position"].agg(list)
    if any(list(map(int, p)) != list(range(len(p))) for p in positions):
        raise StepError("the parts of a segment are not numbered 0, 1, ... in clip order")
    voices = parts.groupby("session")["voice"].agg(set)
    if any(voices[session] != {inputs.voices[int(session)]} for session in voices.index):
        raise StepError("a participant does not have the voice that assign_voices gives them")
    seeds = [
        item_seed(settings.seed, str(segment), int(position))
        for segment, position in zip(parts["segment"], parts["position"], strict=True)
    ]
    if parts["seed"].astype(object).tolist() != seeds:
        raise StepError("a part does not have the seed of its segment and position")
    check_json = workspace.root / shared["check_run"]["run"] / "run.json"
    if sha256_file(check_json) != shared["check_run"]["run_json_sha256"]:
        raise StepError("the check run's run.json changed after the build")
    dry = json.loads(check_json.read_text(encoding="utf-8"))["dry_run"]
    read = {
        "words_read": int(parts["words_read"].sum()),
        "numbers_written_out": int(parts["numbers"].sum()),
    }
    if any(read[name] != dry.get(name) for name in read):
        raise StepError(
            f"the tasks read {read['words_read']:,} words with {read['numbers_written_out']:,}"
            f" numbers written out; the check run's dry run read {dry.get('words_read')} with"
            f" {dry.get('numbers_written_out')}"
        )

    files = set()
    for row in manifest.itertuples(index=False):
        path = run_dir / row.file
        files.add(path)
        if sha256_file(path) != row.sha256:
            raise StepError(f"segment {row.segment}: its file does not match its task's manifest")
        samples, rate = sf.read(path, dtype="int16")
        if (
            rate != settings.rate
            or len(samples) != row.samples
            or audio_sha256(samples) != row.audio_sha256
        ):
            raise StepError(f"segment {row.segment}: its samples do not match the manifest")
        check_segment_views(
            {"x1": samples},
            list(map(int, row.part_starts)),
            list(map(int, row.part_ends)),
            gap=settings.gap_samples,
        )
    stray = set(run_dir.rglob("*.flac")) - files
    if stray:
        raise StepError(f"{len(stray)} FLAC files in the run are in no task's manifest")

    test = summary(manifest, parts, inputs.labels, infos, owner, shared, run_dir, settings.rate)
    manifest.to_parquet(run_dir / "manifest.parquet", index=False)
    parts.to_parquet(run_dir / "parts.parquet", index=False)
    outputs = {name: sha256_file(run_dir / name) for name in ("manifest.parquet", "parts.parquet")}
    minutes = (time.monotonic() - started) / 60
    info = {
        "step": "merge_revoiced",
        "status": "done",
        "started_utc": infos[0]["started_utc"],
        "finished_utc": utc_now(),
        "git": git,
        "array_job_id": array,
        "tasks_git": shared["git"],
        "config": shared["config"],
        "settings": shared["settings"],
        "inputs": shared["inputs"],
        "check_run": shared["check_run"],
        "model": model,
        "environment": environment(PACKAGES),
        "tasks": {
            str(task): {
                "node": info["slurm"]["node"],
                "cpu": info["environment"].get("cpu"),
                "cpu_capability": info["model"].get("cpu_capability"),
                "job_id": info["slurm"]["job_id"],
                "minutes": info["minutes"],
                "segments": info["counts"]["segments"],
            }
            for task, info in infos.items()
        },
        "counts": test,
        "minutes": round(minutes, 2),
        "outputs": outputs,
        "output": {"file": "manifest.parquet", "sha256": outputs["manifest.parquet"]},
    }
    write_json(run_dir / "run.json", info)
    print(f"run: {run_dir.relative_to(workspace.root)}")
    print(f"sha256: {outputs['manifest.parquet']}")


def summary(
    manifest: pd.DataFrame,
    parts: pd.DataFrame,
    labels: pd.DataFrame,
    infos: dict[int, dict[str, Any]],
    owner: dict[int, int],
    shared: dict[str, Any],
    run_dir: Path,
    rate: int,
) -> dict[str, Any]:
    """Aggregate numbers of the merged run: printed and kept in its run.json."""
    seconds = manifest["samples"].to_numpy() / rate
    gains = manifest["gain_db"].to_numpy(dtype=float)
    storage = sum((run_dir / name).stat().st_size for name in manifest["file"])
    counts: dict[str, Any] = {
        "files": len(manifest),
        "sessions": int(manifest["session"].nunique()),
        "parts": len(parts),
        "hours": float(seconds.sum() / 3600),
        "clips_over_30_s": int((seconds > LONG_SECONDS).sum()),
        "longest_seconds": float(seconds.max()),
        "median_rms": float(manifest["rms"].median()),
        "max_peak": float(manifest["peak"].max()),
        "peak_limited": int(manifest["peak_limited"].sum()),
        "gain_db": {
            "min": float(gains.min()),
            "median": float(np.median(gains)),
            "max": float(gains.max()),
        },
        "storage_bytes": int(storage),
        "words_read": int(parts["words_read"].sum()),
        "numbers_written_out": int(parts["numbers"].sum()),
    }
    print(
        f"x1: {counts['files']:,} files from {counts['sessions']} sessions, {counts['parts']:,}"
        f" parts, {counts['hours']:.2f} h; over {LONG_SECONDS:.0f} s: {counts['clips_over_30_s']}"
        f" (longest {counts['longest_seconds']:.1f} s); median RMS {counts['median_rms']:.4f},"
        f" highest peak {counts['max_peak']:.4f}; peak-limited {counts['peak_limited']};"
        f" storage {storage / 1e9:.2f} GB"
    )
    print(
        f"x1 gain (dB): min {gains.min():+.2f}, median {np.median(gains):+.2f},"
        f" max {gains.max():+.2f}"
    )
    print(
        f"text: {counts['words_read']:,} words read, {counts['numbers_written_out']:,} numbers"
        " written out in words: the counts of the check run's dry run"
    )
    voice = parts.drop_duplicates("session").set_index("session")["voice"]
    table = labels.join(voice, how="inner").join(
        manifest.groupby("session").size().rename("segments"), how="inner"
    )
    grouped = table.groupby(["gender", "voice", "label"]).agg(
        participants=("segments", "size"), segments=("segments", "sum")
    )
    counts["voices"] = [
        {"gender": g, "voice": v, "label": int(lab), "participants": int(p), "segments": int(s)}
        for (g, v, lab), (p, s) in grouped.iterrows()
    ]
    print("gender, voice, PHQ-8 class: participants, segments")
    for row in counts["voices"]:
        print(
            f"  {row['gender']}, {row['voice']}, {'positive' if row['label'] else 'negative'}:"
            f" {row['participants']}, {row['segments']:,}"
        )
    cpus = Counter(str(info["environment"].get("cpu")) for info in infos.values())
    counts["cpus"] = dict(cpus)
    print("CPU models of the tasks: " + "; ".join(f"{cpu}: {n}" for cpu, n in cpus.items()))
    root = get_workspace().root
    test = pd.read_parquet(root / shared["check_run"]["run"] / "test.parquet")
    rendered = manifest.set_index("segment")
    equal: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for segment, session, sha in zip(
        test["segment"], test["session"], test["audio_sha256"], strict=True
    ):
        cpu = str(infos[owner[int(session)]]["environment"].get("cpu"))
        total[cpu] += 1
        equal[cpu] += int(rendered.loc[segment, "audio_sha256"] == sha)
    counts["test_render_equal"] = {cpu: [equal[cpu], total[cpu]] for cpu in total}
    print(
        f"test render (check job on {shared['check_run'].get('cpu')}) equal to the full render,"
        " by the build task's CPU: "
        + "; ".join(f"{cpu}: {equal[cpu]} of {total[cpu]}" for cpu in total)
        + " (reported only)"
    )
    return counts


def main() -> None:
    os.umask(0o077)
    sys.unraisablehook = quiet_unraisable  # a finaliser's message could hold a word
    progress = Progress()  # a merge writes nothing when it fails, not even errors.log
    try:
        merge(progress)
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
