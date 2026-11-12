"""Prepare the listening check: 20 random segments, cut as original clips (OMNI login node).

Usage, inside the clone:
    "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/listening_check.py

Reads the pinned segments and word times and the x0 cut settings from
configs/pilot/data.yaml, draws 20 segments at random, cuts each one as its
original clip (x0) and writes $LBW_WORKSPACE/derived/listening_check/run-<UTC time>/ with
  clips/01.wav ... 20.wav  the clips, numbered in the random order of the draw and
                           scaled to a common loudness for listening
  key.tsv                  which segment each clip is; open it only after listening
  run.json                 commit, inputs, settings and counts
Prints aggregate numbers only: never text, and never one segment's values.

The rule (pilot protocol, section "Speech used"): if another voice than the
participant's is heard in more than two of the 20 clips, the interviewer filter
and the cut are reviewed before any training.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from listening_beyond_words.runinfo import (
    check_sha256,
    git_state,
    utc_now,
    utc_stamp,
    write_json,
)
from listening_beyond_words.segments import GAP_SECONDS, segment_neighbours
from listening_beyond_words.sessions import session_files
from listening_beyond_words.settings import Workspace, get_workspace
from listening_beyond_words.views import SAMPLING_RATE, cut_original, part_windows

CONFIG = Path("configs/pilot/data.yaml")
SEED = 101
CLIPS = 20
LISTENING_RMS = 0.1  # common loudness of the listening copies (-20 dBFS) ...
LISTENING_PEAK = 0.99  # ... unless that would clip; x0 itself keeps the recording's level


def load_segments(workspace: Workspace, config: dict) -> pd.DataFrame:
    """The pinned segments with the times of the kept words just outside each one."""
    segments_pin, words_pin = config["segments"], config["word_times"]
    segments_path = workspace.resolve(segments_pin["run"]) / "segments.parquet"
    words_path = workspace.resolve(words_pin["run"]) / "words.parquet"
    check_sha256(segments_path, segments_pin["sha256"])
    check_sha256(words_path, words_pin["sha256"])
    built_from = json.loads((segments_path.parent / "run.json").read_text(encoding="utf-8"))
    if built_from["inputs"]["word_times"]["sha256"] != words_pin["sha256"]:
        raise ValueError("the pinned segments were not built from the pinned word times")
    segments = pd.read_parquet(segments_path, columns=["segment", "session", "starts", "ends"])
    neighbours = segment_neighbours(pd.read_parquet(words_path))
    if sorted(neighbours["segment"]) != sorted(segments["segment"]):
        raise ValueError("the word times give other segments than the pinned segment file")
    table = segments.merge(neighbours, on="segment", how="inner", validate="one_to_one")
    return table.sort_values("segment", ignore_index=True)


def margins(
    windows: list[tuple[float, float]], starts: list[float], ends: list[float]
) -> list[float]:
    """Audio kept before and after the words of each part, in seconds."""
    return [
        side
        for (low, high), start, end in zip(windows, starts, ends, strict=True)
        for side in (start - low, high - end)
    ]


def listening_gain(clip: np.ndarray) -> float:
    """Gain that brings a clip to the common listening loudness without clipping."""
    rms = float(np.sqrt(np.mean(np.square(clip, dtype=np.float64))))
    peak = float(np.max(np.abs(clip)))
    if rms == 0.0 or peak == 0.0:
        return 1.0
    return min(LISTENING_RMS / rms, LISTENING_PEAK / peak)


def main() -> None:
    os.umask(0o077)
    workspace = get_workspace()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cut = config["x0_cut"]
    margin, fade = cut["margin_ms"] / 1000, cut["fade_ms"] / 1000
    info: dict[str, object] = {
        "step": "listening_check",
        "status": "running",
        "started_utc": utc_now(),
        "git": git_state(),
        "inputs": {
            name: {"file": config[name]["run"], "sha256": config[name]["sha256"]}
            for name in ("segments", "word_times")
        },
        "settings": {
            "clips": CLIPS,
            "seed": SEED,
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
        rng = np.random.default_rng(SEED)
        chosen = table.iloc[rng.choice(len(table), size=CLIPS, replace=False)]
        clips_dir = run_dir / "clips"
        clips_dir.mkdir()
        rows: list[dict[str, object]] = []
        kept: list[float] = []
        for number, row in enumerate(chosen.itertuples(index=False), start=1):
            audio_path, _ = session_files(workspace.edaic / "sessions", int(row.session))
            audio, rate = sf.read(audio_path, dtype="float32")
            if rate != SAMPLING_RATE or audio.ndim != 1:
                raise ValueError(f"session {row.session}: expected {SAMPLING_RATE} Hz mono audio")
            starts, ends = list(row.starts), list(row.ends)
            windows = part_windows(
                starts, ends, len(audio) / rate, margin=margin, before=row.before, after=row.after
            )
            clip = cut_original(audio, windows, fade=fade, rate=rate)
            gain = listening_gain(clip)
            name = f"{number:02d}.wav"
            sf.write(clips_dir / name, clip * np.float32(gain), rate, subtype="PCM_16")
            kept += margins(windows, starts, ends)
            rows.append(
                {
                    "clip": name,
                    "segment": row.segment,
                    "parts": len(windows),
                    "seconds": round(len(clip) / rate, 2),
                    "gain_db": round(20 * np.log10(gain), 1),
                    "windows": "; ".join(f"{low:.4f}-{high:.4f}" for low, high in windows),
                }
            )
        key = pd.DataFrame(rows)
        key.to_csv(run_dir / "key.tsv", sep="\t", index=False)

        seconds = key["seconds"].to_numpy(dtype=float)
        gains = key["gain_db"].to_numpy(dtype=float)
        sides = np.array(kept)
        full = int(np.sum(sides >= margin - 1e-9))
        counts = {
            "segments_in_pool": len(table),
            "clips": len(key),
            "sessions": int(chosen["session"].nunique()),
            "listening_seconds": round(float(seconds.sum()), 1),
            "part_edges": len(sides),
            "part_edges_with_full_margin": full,
        }
        print(
            f"clips {counts['clips']} from {counts['sessions']} sessions,"
            f" drawn from {counts['segments_in_pool']} segments with seed {SEED}"
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
