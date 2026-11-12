"""Cut the pilot's 40-word segments on an OMNI login node; print aggregate numbers only.

Usage, inside the clone:
    "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/build_segments.py

Reads the pinned word times from configs/pilot/data.yaml and writes
$LBW_WORKSPACE/derived/segments/run-<UTC time>/ with segments.parquet (one row per
segment: its audio parts, length and words) and run.json.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from listening_beyond_words.runinfo import (
    check_sha256,
    git_state,
    sha256_file,
    utc_now,
    utc_stamp,
    write_json,
)
from listening_beyond_words.segments import GAP_SECONDS, SEGMENT_WORDS, build_segments
from listening_beyond_words.sessions import PILOT_IDS
from listening_beyond_words.settings import get_workspace

CONFIG = Path("configs/pilot/data.yaml")


def main() -> None:
    os.umask(0o077)
    workspace = get_workspace()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    pinned = config["word_times"]
    words_path = workspace.resolve(pinned["run"]) / "words.parquet"
    check_sha256(words_path, pinned["sha256"])
    run_dir = workspace.derived / "segments" / f"run-{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = utc_now()

    segments = build_segments(pd.read_parquet(words_path))
    output = run_dir / "segments.parquet"
    segments.to_parquet(output, index=False)
    digest = sha256_file(output)

    per_person = segments.groupby("session").size().reindex(PILOT_IDS, fill_value=0)
    seconds = segments["seconds"].to_numpy()
    counts = {
        "segments": len(segments),
        "participants_with_segments": int((per_person > 0).sum()),
        "participants_without": int((per_person == 0).sum()),
        "segments_over_30_s": int((seconds > 30).sum()),
    }
    print(
        f"segments {counts['segments']} from {counts['participants_with_segments']} participants;"
        f" participants without a segment: {counts['participants_without']}"
    )
    print(
        f"segments per participant: min {per_person.min()}, median {int(per_person.median())},"
        f" max {per_person.max()}"
    )
    q = np.quantile(seconds, [0.05, 0.5, 0.95])
    print(
        f"segment length (s, audio plus gaps): 5% {q[0]:.1f}, median {q[1]:.1f}, 95% {q[2]:.1f},"
        f" max {seconds.max():.1f}; over 30 s: {counts['segments_over_30_s']}"
    )
    print(
        f"lines per segment: median {segments['parts'].median():.0f}, max {segments['parts'].max()}"
    )
    write_json(
        run_dir / "run.json",
        {
            "step": "build_segments",
            "status": "done",
            "started_utc": started,
            "finished_utc": utc_now(),
            "git": git_state(),
            "inputs": {"word_times": {"file": pinned["run"], "sha256": pinned["sha256"]}},
            "settings": {"words_per_segment": SEGMENT_WORDS, "gap_seconds": GAP_SECONDS},
            "counts": counts,
            "output": {"file": output.name, "sha256": digest},
        },
    )
    print(f"run: {run_dir.relative_to(workspace.root)}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
