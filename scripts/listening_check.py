"""Prepare a blind listening check: 20 random segments, cut as original clips (OMNI login node).

Usage, inside the clone:
    "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/listening_check.py

Reads the pinned segments, the x0 cut settings, the listening settings
(``daicwoz_cut.listening``) and the checks already heard (``listening_checks``) from
configs/pilot/data.yaml. Draws 20 segments at random from those whose audio shares
nothing with a clip of an earlier check, cuts each one as its original clip (x0) from
the windows stored with the segments, exactly as x0 is cut, and writes
$LBW_WORKSPACE/derived/listening_check/run-<UTC time>/ with
  clips/01.wav ... 20.wav  the clips, numbered in the random order of the draw and
                           scaled to a common loudness for listening
  key.tsv                  which segment each clip is; open it only after listening
  run.json                 commit, inputs, settings and counts
Prints aggregate numbers only: never text, and never one segment's values.

The rule (pilot protocol, section "Interviewer detector", cut rule): the check uses
seed 102; if another voice than the participant's is heard in more than 2 of the 20
clips, the guard is widened once to 500 ms, the speech is rebuilt, and 20 new segments
are checked with seed 104. The script takes seed 104 exactly when the pinned segments
were built with the widened guard.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from listening_beyond_words.clips import (
    LISTENING_PEAK,
    LISTENING_RMS,
    format_windows,
    listening_copy,
    load_segments,
    read_key,
)
from listening_beyond_words.runinfo import git_state, utc_now, utc_stamp, write_json
from listening_beyond_words.segments import GAP_SECONDS
from listening_beyond_words.sessions import session_files
from listening_beyond_words.settings import Workspace, get_workspace
from listening_beyond_words.views import SAMPLING_RATE, Window, cut_original, windows_overlap

CONFIG = Path("configs/pilot/data.yaml")
CLIPS = 20


def heard_windows(workspace: Workspace, checks: list[dict]) -> dict[int, list[Window]]:
    """The audio windows of every clip of the earlier checks, by session."""
    heard: dict[int, list[Window]] = {}
    for check in checks:
        key = read_key(workspace.resolve(check["run"]) / "key.tsv", check["sha256"])
        for session, windows in zip(key["session"], key["windows"], strict=True):
            heard.setdefault(int(session), []).extend(windows)
    return heard


def margins(windows: list[Window], starts: list[float], ends: list[float]) -> list[float]:
    """Audio kept before and after the words of each part, in seconds."""
    return [
        side
        for (low, high), start, end in zip(windows, starts, ends, strict=True)
        for side in (start - low, high - end)
    ]


def main() -> None:
    os.umask(0o077)
    workspace = get_workspace()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cut, rule = config["x0_cut"], config["daicwoz_cut"]
    listening = rule["listening"]
    margin, fade = cut["margin_ms"] / 1000, cut["fade_ms"] / 1000
    checks = config.get("listening_checks") or []
    widened = rule["guard_ms"] == rule["offset_guard_ms"] == listening["fallback_guard_ms"]
    if widened and len(checks) < 2:
        raise ValueError(
            "the fallback check comes after the seed-102 check: list it in listening_checks"
        )
    seed = listening["fallback_seed"] if widened else listening["seed"]
    info: dict[str, object] = {
        "step": "listening_check",
        "status": "running",
        "started_utc": utc_now(),
        "git": git_state(),
        "inputs": {
            "segments": {"file": config["segments"]["run"], "sha256": config["segments"]["sha256"]},
            "heard": checks,
        },
        "settings": {
            "clips": CLIPS,
            "seed": seed,
            "fallback": widened,
            "x0_cut": cut,
            "gap_seconds": GAP_SECONDS,
            "listening_rms": LISTENING_RMS,
            "listening_peak": LISTENING_PEAK,
        },
    }
    run_dir = workspace.derived / "listening_check" / f"run-{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "run.json", info)
    try:
        table = load_segments(workspace, config)
        heard = heard_windows(workspace, checks)
        windows_of: list[list[Window]] = [
            list(zip(map(float, lows), map(float, highs), strict=True))
            for lows, highs in zip(table["window_starts"], table["window_ends"], strict=True)
        ]
        fresh = np.array(
            [
                not windows_overlap(windows, heard.get(int(session), []))
                for windows, session in zip(windows_of, table["session"], strict=True)
            ]
        )
        pool = np.flatnonzero(fresh)
        rng = np.random.default_rng(seed)
        chosen = pool[rng.choice(len(pool), size=CLIPS, replace=False)]
        clips_dir = run_dir / "clips"
        clips_dir.mkdir()
        rows: list[dict[str, object]] = []
        kept: list[float] = []
        for number, index in enumerate(chosen, start=1):
            row = table.iloc[index]
            audio_path, _ = session_files(workspace.edaic / "sessions", int(row["session"]))
            audio, rate = sf.read(audio_path, dtype="float32")
            if rate != SAMPLING_RATE or audio.ndim != 1:
                raise ValueError(
                    f"session {row['session']}: expected {SAMPLING_RATE} Hz mono audio"
                )
            windows = windows_of[index]
            clip = cut_original(audio, windows, fade=fade, rate=rate)
            copy, gain = listening_copy(clip)
            name = f"{number:02d}.wav"
            sf.write(clips_dir / name, copy, rate, subtype="PCM_16")
            kept += margins(windows, list(row["starts"]), list(row["ends"]))
            rows.append(
                {
                    "clip": name,
                    "segment": row["segment"],
                    "parts": len(windows),
                    "seconds": round(len(clip) / rate, 2),
                    "gain_db": round(20 * np.log10(gain), 1),
                    "windows": format_windows(windows),
                }
            )
        key = pd.DataFrame(rows)
        key.to_csv(run_dir / "key.tsv", sep="\t", index=False)

        seconds = key["seconds"].to_numpy(dtype=float)
        gains = key["gain_db"].to_numpy(dtype=float)
        sides = np.array(kept)
        full = int(np.sum(sides >= margin - 1e-9))
        counts = {
            "segments": len(table),
            "segments_sharing_audio_with_heard_clips": int((~fresh).sum()),
            "segments_in_pool": len(pool),
            "clips": len(key),
            "sessions": int(table.iloc[chosen]["session"].nunique()),
            "listening_seconds": round(float(seconds.sum()), 1),
            "part_edges": len(sides),
            "part_edges_with_full_margin": full,
        }
        print(
            f"clips {counts['clips']} from {counts['sessions']} sessions, drawn with seed {seed}"
            f" from {counts['segments_in_pool']} of {counts['segments']} segments"
            f" ({counts['segments_sharing_audio_with_heard_clips']} share audio with the"
            f" {sum(len(w) for w in heard.values())} windows of {len(checks)} earlier check(s))"
        )
        print(
            f"clip length (s): min {seconds.min():.1f}, median {np.median(seconds):.1f},"
            f" max {seconds.max():.1f}; listening time {seconds.sum() / 60:.1f} min"
        )
        print(
            f"parts per clip: median {key['parts'].median():.0f}, max {key['parts'].max()};"
            f" part edges with the full {margin} s margin: {full} of {len(sides)},"
            f" median margin {np.median(sides):.3f} s"
        )
        print(
            f"listening gain (dB): min {gains.min():+.1f}, median {np.median(gains):+.1f},"
            f" max {gains.max():+.1f}"
        )
        print(f"clips: {clips_dir.relative_to(workspace.root)}")
        print(
            f"key (open it only after listening): {(run_dir / 'key.tsv').relative_to(workspace.root)}"
        )
        info.update(status="done", finished_utc=utc_now(), counts=counts)
    except BaseException:
        info.update(status="failed", finished_utc=utc_now())
        raise
    finally:
        write_json(run_dir / "run.json", info)


if __name__ == "__main__":
    main()
