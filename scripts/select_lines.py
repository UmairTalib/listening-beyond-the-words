"""Apply the pilot's interviewer filter to every transcript line on an OMNI login node.

Usage, inside the clone:
    "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/select_lines.py

Reads configs/pilot/data.yaml for the interviewer-filter run and the pinned labels,
decides for every transcript line of the 189 pilot sessions whether it is kept, and
writes $LBW_WORKSPACE/derived/participant_lines/run-<UTC time>/ with lines.parquet
and run.json. Then checks that the dropped share of words does not differ between
PHQ-8 classes or genders. Prints aggregate numbers only.
"""

import os
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import mannwhitneyu

from listening_beyond_words.interviewer import line_decisions
from listening_beyond_words.runinfo import (
    check_sha256,
    git_state,
    sha256_file,
    utc_now,
    utc_stamp,
    write_json,
)
from listening_beyond_words.sessions import PILOT_IDS, session_files
from listening_beyond_words.settings import get_workspace
from listening_beyond_words.transcripts import INTERVIEWER_PHRASES, normalise, read_transcript

CONFIG = Path("configs/pilot/data.yaml")
SEGMENT_WORDS = 40


def transcript_lines(sessions_dir: Path) -> pd.DataFrame:
    rows = []
    for session_id in PILOT_IDS:
        _, transcript = session_files(sessions_dir, session_id)
        frame = read_transcript(transcript)
        for index, (start, end, text) in enumerate(
            zip(frame["start"], frame["end"], frame["text"], strict=True)
        ):
            words = normalise(text)
            rows.append(
                {
                    "session": session_id,
                    "line": index,
                    "start": float(start),
                    "end": float(end),
                    "words": len(words.split()),
                    "phrase": any(f" {p} " in f" {words} " for p in INTERVIEWER_PHRASES),
                }
            )
    return pd.DataFrame(rows)


def compare(values: pd.Series, groups: pd.Series, name: str) -> float:
    first, second = sorted(groups.unique())
    a, b = values[groups == first], values[groups == second]
    p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    print(
        f"dropped share of words by {name}: {first} mean {a.mean():.3f} (n={len(a)}),"
        f" {second} mean {b.mean():.3f} (n={len(b)}); Mann-Whitney p = {p:.3f}"
    )
    return p


def main() -> None:
    os.umask(0o077)
    workspace = get_workspace()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    labels_cfg, filter_cfg = config["labels"], config["interviewer_filter"]
    if not labels_cfg.get("run") or not labels_cfg.get("sha256"):
        raise SystemExit(f"{CONFIG}: pin the labels run and its SHA-256 first")
    labels_path = workspace.resolve(labels_cfg["run"]) / "labels.parquet"
    check_sha256(labels_path, labels_cfg["sha256"])
    windows_path = workspace.resolve(filter_cfg["run"]) / "windows.parquet"
    drop_share = float(filter_cfg["drop_share"])
    run_dir = workspace.derived / "participant_lines" / f"run-{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = utc_now()

    lines = transcript_lines(workspace.edaic / "sessions")
    if not filter_cfg["drop_interviewer_phrases"]:
        lines["phrase"] = False
    decided = line_decisions(lines, pd.read_parquet(windows_path), drop_share=drop_share)
    output = run_dir / "lines.parquet"
    decided.to_parquet(output, index=False)
    digest = sha256_file(output)

    words = decided.groupby("session")["words"].sum()
    dropped = decided[decided["drop"]].groupby("session")["words"].sum()
    dropped = dropped.reindex(words.index, fill_value=0)
    share = (dropped / words).rename("dropped_share")
    kept = words - dropped
    by_voice = int((decided["share_above_0"] >= drop_share).sum())
    counts = {
        "lines": len(decided),
        "lines_dropped": int(decided["drop"].sum()),
        "lines_dropped_by_voice": by_voice,
        "lines_with_phrase": int(decided["phrase"].sum()),
        "words": int(words.sum()),
        "words_dropped": int(dropped.sum()),
        "segments_possible": int((kept // SEGMENT_WORDS).sum()),
        "participants_under_40_words": int((kept < SEGMENT_WORDS).sum()),
    }
    print(
        f"lines {counts['lines']}: dropped {counts['lines_dropped']}"
        f" ({counts['lines_dropped'] / counts['lines']:.3f}); by voice {by_voice},"
        f" with a phrase {counts['lines_with_phrase']}"
    )
    print(
        f"words {counts['words']}: dropped {counts['words_dropped']}"
        f" ({counts['words_dropped'] / counts['words']:.3f})"
    )
    print(
        f"40-word segments possible from the kept words: {counts['segments_possible']};"
        f" participants with fewer than 40 kept words: {counts['participants_under_40_words']}"
    )
    print(
        f"dropped share of words per participant: median {share.median():.3f},"
        f" 90% {share.quantile(0.9):.3f}, max {share.max():.3f}"
    )
    joined = pd.read_parquet(labels_path).join(share)
    p_label = compare(
        joined["dropped_share"], joined["label"].map({0: "negative", 1: "positive"}), "PHQ-8 class"
    )
    p_gender = compare(joined["dropped_share"], joined["gender"], "gender")
    verdict = "passed" if p_label >= 0.05 else "REVIEW the filter before any training"
    print(f"balance check, PHQ-8 class (p >= 0.05): {verdict}")

    write_json(
        run_dir / "run.json",
        {
            "step": "select_lines",
            "status": "done",
            "started_utc": started,
            "finished_utc": utc_now(),
            "git": git_state(),
            "config": {"file": str(CONFIG), "sha256": sha256_file(CONFIG), "content": config},
            "inputs": {
                "labels": {"file": labels_cfg["run"], "sha256": labels_cfg["sha256"]},
                "windows": {"file": filter_cfg["run"], "sha256": sha256_file(windows_path)},
            },
            "counts": counts,
            "balance": {"p_label": p_label, "p_gender": p_gender, "verdict": verdict},
            "output": {"file": output.name, "sha256": digest},
        },
    )
    print(f"run: {run_dir.relative_to(workspace.root)}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
