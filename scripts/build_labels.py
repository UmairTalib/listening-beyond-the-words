"""Build the pilot's labels table on an OMNI login node; print aggregate counts only.

Usage, inside the clone:
    "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/build_labels.py

Reads the E-DAIC label files, keeps only the 189 pilot sessions and writes
$LBW_WORKSPACE/derived/labels/run-<UTC time>/labels.parquet with run.json. Prints
counts over all sessions and the file's SHA-256, never a row. Pin the run folder
and the hash in configs/pilot/data.yaml.
"""

import os

from listening_beyond_words.labels import DETAILED_FILE, PHQ8_FILE, SPLIT_FILES, load_labels
from listening_beyond_words.runinfo import git_state, sha256_file, utc_now, utc_stamp, write_json
from listening_beyond_words.settings import get_workspace


def main() -> None:
    os.umask(0o077)
    workspace = get_workspace()
    labels_dir = workspace.edaic / "raw" / "labels"
    run_dir = workspace.derived / "labels" / f"run-{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    labels, report = load_labels(labels_dir)
    output = run_dir / "labels.parquet"
    labels.to_parquet(output)
    digest = sha256_file(output)
    inputs = [*SPLIT_FILES.values(), DETAILED_FILE, PHQ8_FILE]
    write_json(
        run_dir / "run.json",
        {
            "step": "build_labels",
            "status": "done",
            "started_utc": started,
            "finished_utc": utc_now(),
            "git": git_state(),
            "inputs": {name: sha256_file(labels_dir / name) for name in inputs},
            "counts": report,
            "output": {"file": output.name, "sha256": digest},
        },
    )
    for key, value in report.items():
        print(f"{key}: {value}")
    print(f"run: {run_dir.relative_to(workspace.root)}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
