"""Build the original, muffled and reversed clips (x0, x2, rev) of every segment (CPU job).

Submit from an OMNI login node:
    scripts/slurm/submit.sh build_views

Reads the pinned segments, the x0 cut settings (``x0_cut``), the view settings
(``views``) and the listening checks (``listening_checks``) from
configs/pilot/data.yaml.

First, before anything is written, it checks x0: it finds the one listening check that
was made from the pinned segments, builds x0 of its 20 clips with this code and
requires each one's listening copy (x0 scaled and written to 16-bit WAV exactly as the
listening check did) to equal the heard file, sample for sample.

Then it builds x0, x2 and rev of every segment, part by part (pilot protocol, section
"Clip versions"), session by session, and writes for each view
$LBW_WORKSPACE/views/<view>/run-<job id>/ with
  <session>/<segment>.flac  the clip as 16 kHz mono 16-bit FLAC, read back after writing
  manifest.parquet          one row per clip: its file, length, RMS, peak, SHA-256 and
                            parts (clips.MANIFEST_COLUMNS)
  run.json                  commit, config, inputs, settings, environment, counts and
                            the SHA-256 of the manifest, by which the view is pinned
A job that fails or that Slurm ends writes status "failed"; a rerun writes new folders.
Prints aggregate numbers only: never text, and never one segment's values.
"""

import json
import math
import os
import signal
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from listening_beyond_words.clips import (
    LISTENING_PEAK,
    LISTENING_RMS,
    MANIFEST_COLUMNS,
    format_windows,
    listening_pcm16,
    load_segments,
    read_key,
    read_session,
    write_view,
)
from listening_beyond_words.runinfo import (
    environment,
    git_state,
    sha256_file,
    utc_now,
    write_json,
)
from listening_beyond_words.segments import GAP_SECONDS
from listening_beyond_words.settings import Workspace, get_workspace
from listening_beyond_words.views import (
    ViewSettings,
    build_segment_views,
    check_segment_views,
    stored_windows,
    to_pcm16,
)

CONFIG = Path("configs/pilot/data.yaml")
VIEWS = ("x0", "x2", "rev")
LONG_SECONDS = 30.0  # the encoder's window; longer clips are handled in step 7
PACKAGES = ("numpy", "scipy", "soundfile", "pandas", "pyarrow", "pyyaml")


def stop_on_sigterm() -> None:
    """End the job through the error path when Slurm stops it, so that run.json says failed."""

    def handler(signum: int, _frame: FrameType | None) -> None:
        raise SystemExit(f"stopped by signal {signum}")

    signal.signal(signal.SIGTERM, handler)


def listening_check_of(workspace: Workspace, config: dict) -> tuple[dict, dict]:
    """The one listening check made from the pinned segments: its pin and its run.json."""
    matches = []
    for check in config.get("listening_checks") or []:
        run = workspace.resolve(check["run"]) / "run.json"
        info = json.loads(run.read_text(encoding="utf-8"))
        if info["inputs"]["segments"]["sha256"] == config["segments"]["sha256"]:
            matches.append((check, info))
    if len(matches) != 1:
        raise ValueError(
            f"expected one listening check made from the pinned segments, found {len(matches)}"
        )
    check, info = matches[0]
    if info.get("status") != "done":
        raise ValueError("the listening check of the pinned segments did not finish")
    used = info["settings"]
    if used["x0_cut"] != config["x0_cut"] or used["gap_seconds"] != GAP_SECONDS:
        raise ValueError("the listening check of the pinned segments used other x0 settings")
    if (used["listening_rms"], used["listening_peak"]) != (LISTENING_RMS, LISTENING_PEAK):
        raise ValueError("the listening check of the pinned segments used another listening level")
    return check, info


def check_x0(
    workspace: Workspace, config: dict, table: pd.DataFrame, settings: ViewSettings
) -> dict[str, Any]:
    """Build x0 of the heard clips with this code and compare them with the heard files."""
    check, info = listening_check_of(workspace, config)
    run = workspace.resolve(check["run"])
    key = read_key(run / "key.tsv", check["sha256"])
    rows = table.set_index("segment")
    found = same_windows = identical = 0
    for name, segment, windows in zip(key["clip"], key["segment"], key["windows"], strict=True):
        if segment not in rows.index:
            continue
        found += 1
        row = rows.loc[segment]
        stored = stored_windows(row)
        if format_windows(stored) != format_windows(windows):
            continue
        same_windows += 1
        audio = read_session(workspace.edaic / "sessions", int(row["session"]))
        x0 = build_segment_views(audio, stored, settings).x0
        heard_file = run / "clips" / name
        heard_info = sf.info(heard_file)
        heard, _ = sf.read(heard_file, dtype="int16")
        same_format = (heard_info.samplerate, heard_info.channels, heard_info.subtype) == (
            settings.rate,
            1,
            "PCM_16",
        )
        if same_format and np.array_equal(heard, listening_pcm16(x0, settings.rate)):
            identical += 1
    seed = info["settings"]["seed"]
    counts = {
        "clips": len(key),
        "in_table": found,
        "same_windows": same_windows,
        "identical": identical,
    }
    print(
        f"x0 check: the listening copies of {identical} of {len(key)} clips (listening check"
        f" with seed {seed}) are identical to the heard files, sample for sample"
    )
    if not identical == same_windows == found == len(key) > 0:
        raise SystemExit(
            f"x0 check failed: {found} clips in the segments table, {same_windows} with the"
            f" same windows, {identical} identical, of {len(key)}"
        )
    return {"run": check["run"], "key_sha256": check["sha256"], "seed": seed, **counts}


def manifest_row(
    segment: str, session: int, name: str, values: dict[str, Any], **extra: Any
) -> dict[str, Any]:
    """One row of a view's manifest, in the order of MANIFEST_COLUMNS."""
    row = {"segment": segment, "session": session, "file": name, **values, **extra}
    return {column: row[column] for column in MANIFEST_COLUMNS}


def summary(view: str, manifest: pd.DataFrame, run_dir: Path, rate: int) -> dict[str, Any]:
    """Aggregate numbers of one view, printed and kept in its run.json."""
    seconds = manifest["samples"].to_numpy() / rate
    storage = sum((run_dir / name).stat().st_size for name in manifest["file"])
    gains = manifest["gain_db"].to_numpy(dtype=float)
    counts = {
        "files": len(manifest),
        "sessions": int(manifest["session"].nunique()),
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
    }
    print(
        f"{view}: {counts['files']:,} files from {counts['sessions']} sessions,"
        f" {counts['hours']:.2f} h; over {LONG_SECONDS:.0f} s: {counts['clips_over_30_s']}"
        f" (longest {counts['longest_seconds']:.1f} s); median RMS {counts['median_rms']:.4f},"
        f" highest peak {counts['max_peak']:.4f}; peak-limited {counts['peak_limited']};"
        f" storage {storage / 1e9:.2f} GB"
    )
    if view == "x2":
        print(
            f"x2 gain after the filter (dB): min {gains.min():+.2f},"
            f" median {np.median(gains):+.2f}, max {gains.max():+.2f}"
        )
    return counts


def main() -> None:
    os.umask(0o077)
    stop_on_sigterm()
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        sys.exit("submit this as a job: scripts/slurm/submit.sh build_views")
    started = time.monotonic()
    workspace = get_workspace()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    settings = ViewSettings.from_config(config)
    table = load_segments(workspace, config)
    x0_check = check_x0(workspace, config, table, settings)

    run_dirs = {view: workspace.views / view / f"run-{job}" for view in VIEWS}
    for run_dir in run_dirs.values():
        run_dir.mkdir(parents=True, exist_ok=False)
    shared: dict[str, Any] = {
        "step": "build_views",
        "status": "running",
        "started_utc": utc_now(),
        "git": git_state(),
        "config": {"file": str(CONFIG), "sha256": sha256_file(CONFIG)},
        "inputs": {
            "segments": {"file": config["segments"]["run"], "sha256": config["segments"]["sha256"]},
            "x0_check": x0_check,
        },
        "settings": {
            "x0_cut": config["x0_cut"],
            "views": config["views"],
            "resolved": asdict(settings),
        },
        "environment": {**environment(PACKAGES), "libsndfile": sf.__libsndfile_version__},
        "slurm": {"job_id": job, "node": socket.gethostname()},
    }
    infos = {view: {**shared, "view": view} for view in VIEWS}
    for view in VIEWS:
        write_json(run_dirs[view] / "run.json", infos[view])
    try:
        rows: dict[str, list[dict[str, Any]]] = {view: [] for view in VIEWS}
        gap = round(settings.gap * settings.rate)
        for session, group in table.groupby("session", sort=True):
            session_id = int(session)
            audio = read_session(workspace.edaic / "sessions", session_id)
            for run_dir in run_dirs.values():
                (run_dir / str(session_id)).mkdir()
            for segment in group.itertuples(index=False):
                built = build_segment_views(audio, stored_windows(segment), settings)
                samples = {
                    "x0": to_pcm16(built.x0),
                    "x2": to_pcm16(built.x2),
                    "rev": to_pcm16(built.rev),
                }
                check_segment_views(samples, built.part_starts, built.part_ends, gap=gap)
                for view in VIEWS:
                    name = f"{session_id}/{segment.segment}.flac"
                    values = write_view(run_dirs[view] / name, samples[view], settings.rate)
                    x2 = view == "x2"
                    rows[view].append(
                        manifest_row(
                            str(segment.segment),
                            session_id,
                            name,
                            values,
                            part_starts=list(built.part_starts),
                            part_ends=list(built.part_ends),
                            peak_limited=built.x2_peak_limited if x2 else False,
                            gain_db=20 * math.log10(built.x2_gain) if x2 else 0.0,
                        )
                    )
        if any(len(rows[view]) != len(table) for view in VIEWS):
            raise ValueError("a view does not hold one clip per segment")
        print(
            f"every segment: the three views have the same length and parts, silent gaps and"
            f" parts that start and end at 0; every file read back as written"
            f" ({len(table):,} segments)"
        )
        for view in VIEWS:
            manifest = pd.DataFrame(rows[view], columns=list(MANIFEST_COLUMNS))
            manifest.to_parquet(run_dirs[view] / "manifest.parquet", index=False)
            sha256 = sha256_file(run_dirs[view] / "manifest.parquet")
            counts = summary(view, manifest, run_dirs[view], settings.rate)
            infos[view].update(
                counts=counts,
                outputs={"manifest.parquet": sha256},
                output={"file": "manifest.parquet", "sha256": sha256},
            )
        minutes = (time.monotonic() - started) / 60
        for view in VIEWS:
            infos[view].update(status="done", finished_utc=utc_now(), minutes=round(minutes, 2))
        for view in VIEWS:
            print(f"{view} run: {run_dirs[view].relative_to(workspace.root)}")
            print(f"{view} sha256: {infos[view]['output']['sha256']}")
        print(f"done in {minutes:.1f} min")
    except BaseException:
        for view in VIEWS:
            infos[view].update(status="failed", finished_utc=utc_now())
        raise
    finally:
        for view in VIEWS:
            write_json(run_dirs[view] / "run.json", infos[view])


if __name__ == "__main__":
    main()
