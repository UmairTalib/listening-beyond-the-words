"""Draw the word-check sample: five segments per participant (OMNI login node).

Usage, inside the clone, with no uncommitted changes:
    "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/draw_wordcheck_sample.py

Reads the pinned segments and the sample settings (``word_check.sample``: seed 105, five
segments per participant) from configs/pilot/data.yaml. Draws the sample with
``wordcheck.draw_sample``: five segments per participant without replacement, or all of
a participant's segments if there are fewer (pilot protocol, section "Clip versions").
Writes $LBW_WORKSPACE/derived/wordcheck_sample/run-<UTC time>/ with
  sample.parquet  the drawn segments (segment and session), sorted by segment
  run.json        commit, config, inputs, seed, versions, counts and the SHA-256 of the
                  sample, by which it is pinned
Prints aggregate numbers only: never text, and never one segment's values.
"""

import os
import sys
from pathlib import Path

import numpy as np
import yaml

from listening_beyond_words.clips import load_segments
from listening_beyond_words.runinfo import (
    environment,
    git_state,
    sha256_file,
    utc_now,
    utc_stamp,
    write_json,
)
from listening_beyond_words.settings import get_workspace
from listening_beyond_words.wordcheck import draw_sample

CONFIG = Path("configs/pilot/data.yaml")
PACKAGES = ("numpy", "pandas", "pyarrow", "pyyaml")


def main() -> None:
    os.umask(0o077)
    git = git_state()
    if git["dirty"]:
        sys.exit("the checkout has uncommitted changes; commit them on the Mac and pull first")
    workspace = get_workspace()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    sample = config["word_check"]["sample"]
    seed, per = int(sample["seed"]), int(sample["segments_per_participant"])
    run_dir = workspace.derived / "wordcheck_sample" / f"run-{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    info: dict[str, object] = {
        "step": "draw_wordcheck_sample",
        "status": "running",
        "started_utc": utc_now(),
        "git": git,
        "config": {"file": str(CONFIG), "sha256": sha256_file(CONFIG)},
        "inputs": {
            "segments": {"file": config["segments"]["run"], "sha256": config["segments"]["sha256"]}
        },
        "seeds": {"sample": seed},
        "settings": {"word_check_sample": sample},
        "environment": environment(PACKAGES),
    }
    write_json(run_dir / "run.json", info)
    try:
        table = load_segments(workspace, config)
        drawn = draw_sample(table, np.random.default_rng(seed), per)
        drawn.to_parquet(run_dir / "sample.parquet", index=False)
        sha256 = sha256_file(run_dir / "sample.parquet")
        available = table.groupby("session").size()
        counts = {
            "segments_in_table": len(table),
            "participants": int(available.size),
            "segments": len(drawn),
            "participants_with_fewer_than_the_count": int((available < per).sum()),
            "participants_with_exactly_the_count": int((available == per).sum()),
            "segments_per_participant": {
                str(n): int(c) for n, c in drawn.groupby("session").size().value_counts().items()
            },
        }
        print(
            f"word-check sample: {counts['segments']} segments from {counts['participants']}"
            f" participants, drawn with seed {seed} from {counts['segments_in_table']:,} segments"
        )
        print(
            f"participants with fewer than {per} segments (all taken):"
            f" {counts['participants_with_fewer_than_the_count']}; with exactly {per}:"
            f" {counts['participants_with_exactly_the_count']}"
        )
        info.update(
            status="done",
            finished_utc=utc_now(),
            counts=counts,
            outputs={"sample.parquet": sha256},
            output={"file": "sample.parquet", "sha256": sha256},
        )
        print(f"run: {run_dir.relative_to(workspace.root)}")
        print(f"sha256: {sha256}")
    except BaseException:
        info.update(status="failed", finished_utc=utc_now())
        raise
    finally:
        write_json(run_dir / "run.json", info)


if __name__ == "__main__":
    main()
