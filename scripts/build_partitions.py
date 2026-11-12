"""Split the 189 pilot participants into folds on an OMNI login node; print counts only.

Usage, inside the clone:
    "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/build_partitions.py

Reads the pinned labels from configs/pilot/data.yaml, makes 2 rounds of 5 folds
(seeds 101 and 102), stratified by label and gender, and writes
$LBW_WORKSPACE/derived/partitions/run-<UTC time>/ with partitions.parquet and
run.json. Pin the run and its SHA-256 in configs/pilot/data.yaml before training.
"""

import os
from pathlib import Path

import pandas as pd
import yaml

from listening_beyond_words.partitions import N_FOLDS, SEEDS, make_partitions
from listening_beyond_words.runinfo import (
    check_sha256,
    git_state,
    sha256_file,
    utc_now,
    utc_stamp,
    write_json,
)
from listening_beyond_words.settings import get_workspace

CONFIG = Path("configs/pilot/data.yaml")


def main() -> None:
    os.umask(0o077)
    workspace = get_workspace()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    pinned = config["labels"]
    labels_path = workspace.resolve(pinned["run"]) / "labels.parquet"
    check_sha256(labels_path, pinned["sha256"])
    run_dir = workspace.derived / "partitions" / f"run-{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = utc_now()

    labels = pd.read_parquet(labels_path)
    partitions = make_partitions(labels)
    output = run_dir / "partitions.parquet"
    partitions.to_parquet(output, index=False)
    digest = sha256_file(output)

    table = partitions.join(labels[["label", "gender"]], on="session")
    summary = table.groupby(["round", "fold"]).agg(
        participants=("session", "size"),
        positive=("label", "sum"),
        female=("gender", lambda g: int((g == "female").sum())),
    )
    print(summary.to_string())
    for round_number, part in partitions.groupby("round"):
        if not part["session"].is_unique or len(part) != len(labels):
            raise SystemExit(f"round {round_number}: a participant is missing or appears twice")
        print(f"round {round_number}: every participant in exactly one fold")
    write_json(
        run_dir / "run.json",
        {
            "step": "build_partitions",
            "status": "done",
            "started_utc": started,
            "finished_utc": utc_now(),
            "git": git_state(),
            "inputs": {"labels": {"file": pinned["run"], "sha256": pinned["sha256"]}},
            "settings": {
                "seeds": list(SEEDS),
                "folds": N_FOLDS,
                "stratified_by": ["label", "gender"],
            },
            "output": {"file": output.name, "sha256": digest},
        },
    )
    print(f"run: {run_dir.relative_to(workspace.root)}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
