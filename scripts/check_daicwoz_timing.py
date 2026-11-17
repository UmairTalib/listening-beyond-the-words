"""Check how the DAIC-WOZ transcript times fit the E-DAIC audio, on an OMNI login node.

Usage, inside the clone:
    "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/check_daicwoz_timing.py

Reads the pinned DAIC-WOZ transcripts, word times and labels from configs/pilot/data.yaml.
For every pilot session it finds the shift of the transcript times that puts the most
aligned participant words inside the participant's turns (searched from -90 s to +90 s in
0.1 s steps), and measures how many words fall inside the interviewer's turns, close to
them, or inside the interruptions that the DAIC-WOZ documentation names. Writes
$LBW_WORKSPACE/derived/daicwoz_timing/run-<UTC time>/ with sessions.tsv and run.json.
Prints aggregate numbers and session IDs, never transcript text or words.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu

from listening_beyond_words.daicwoz import best_offset, inside, load_transcripts, read_turns
from listening_beyond_words.runinfo import (
    check_sha256,
    git_state,
    sha256_file,
    utc_now,
    utc_stamp,
    write_json,
)
from listening_beyond_words.sessions import PILOT_IDS
from listening_beyond_words.settings import get_workspace

CONFIG = Path("configs/pilot/data.yaml")
SHIFTS = np.round(np.arange(-90.0, 90.0 + 1e-9, 0.1), 1)
SHIFTED = 0.5  # seconds: a best shift larger than this marks a session as offset
NEAR = (0.2, 0.5)  # seconds around the interviewer's turns
AUTOMATIC_FROM = 364  # her lines above session 363 were written automatically (documentation)
INTERRUPTIONS = {373: (352.0, 420.0), 444: (286.0, 387.0)}  # documentation: 5:52-7:00, 4:46-6:27


def near(starts: np.ndarray, ends: np.ndarray, turns: pd.DataFrame, margin: float) -> np.ndarray:
    """Words that start, end or sit within ``margin`` seconds of any of the given turns."""
    low, high = turns["start"].to_numpy() - margin, turns["stop"].to_numpy() + margin
    mids = (starts + ends) / 2
    return inside(starts, low, high) | inside(ends, low, high) | inside(mids, low, high)


def overlap_seconds(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Seconds during which a turn of ``a`` and a turn of ``b`` run at the same time."""
    total = 0.0
    for start, stop in zip(a["start"], a["stop"], strict=True):
        hits = b[(b["stop"] > start) & (b["start"] < stop)]
        total += float((np.minimum(hits["stop"], stop) - np.maximum(hits["start"], start)).sum())
    return total


def check_session(session: int, data: bytes, words: pd.DataFrame) -> dict[str, object]:
    turns = read_turns(data)
    ellie = turns[turns["speaker"] == "ellie"]
    participant = turns[turns["speaker"] == "participant"]
    starts, ends = words["start"].to_numpy(), words["end"].to_numpy()
    mids = (starts + ends) / 2
    p_start, p_stop = participant["start"].to_numpy(), participant["stop"].to_numpy()
    shift, share_best = best_offset(mids, p_start, p_stop, SHIFTS)
    used = shift if abs(shift) > SHIFTED else 0.0
    moved = ellie.assign(start=ellie["start"] + used, stop=ellie["stop"] + used)
    row: dict[str, object] = {
        "session": session,
        "words": len(words),
        "ellie_turns": len(ellie),
        "ellie_seconds": round(float((ellie["stop"] - ellie["start"]).sum()), 1),
        "participant_turns": len(participant),
        "simultaneous_seconds": round(overlap_seconds(ellie, participant), 1),
        "best_shift_s": shift,
        "share_in_participant_at_0": float(inside(mids, p_start, p_stop).mean()),
        "share_in_participant_at_best": share_best,
        "shift_used_s": used,
        "share_in_ellie": float(
            inside(mids, moved["start"].to_numpy(), moved["stop"].to_numpy()).mean()
        ),
    }
    for margin in NEAR:
        row[f"share_near_ellie_{margin}"] = float(near(starts, ends, moved, margin).mean())
    window = INTERRUPTIONS.get(session)
    row["words_in_interruption"] = (
        int(((mids >= window[0]) & (mids <= window[1])).sum()) if window else 0
    )
    return row


def main() -> None:
    os.umask(0o077)
    workspace = get_workspace()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    transcripts = load_transcripts(workspace, config["daicwoz_transcripts"])
    words_pin, labels_pin = config["word_times"], config["labels"]
    words_path = workspace.resolve(words_pin["run"]) / "words.parquet"
    labels_path = workspace.resolve(labels_pin["run"]) / "labels.parquet"
    check_sha256(words_path, words_pin["sha256"])
    check_sha256(labels_path, labels_pin["sha256"])
    words = pd.read_parquet(words_path, columns=["session", "start", "end"])
    labels = pd.read_parquet(labels_path)
    run_dir = workspace.derived / "daicwoz_timing" / f"run-{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = utc_now()

    rows = [
        check_session(s, transcripts[s], words[words["session"] == s].sort_values("start"))
        for s in PILOT_IDS
    ]
    table = pd.DataFrame(rows).set_index("session").join(labels[["label", "gender"]])
    table.to_csv(run_dir / "sessions.tsv", sep="\t")

    def ids(frame: pd.DataFrame) -> str:
        return ", ".join(str(s) for s in frame.index) or "none"

    at_zero = table["share_in_participant_at_0"]
    print(f"sessions: {len(table)}; aligned words: {int(table['words'].sum()):,}")
    print(
        "share of aligned words inside the participant's turns, no shift:"
        f" median {at_zero.median():.3f}, 5% {at_zero.quantile(0.05):.3f}, min {at_zero.min():.3f}"
    )
    shifted = table[table["shift_used_s"] != 0]
    print(f"sessions whose best shift exceeds {SHIFTED} s: {len(shifted)}")
    for session, row in shifted.iterrows():
        print(
            f"  {session}: shift {row['best_shift_s']:+.1f} s,"
            f" share inside participant turns {row['share_in_participant_at_0']:.2f}"
            f" -> {row['share_in_participant_at_best']:.2f}"
        )
    in_ellie = (table["share_in_ellie"] * table["words"]).sum()
    print(
        f"words inside the interviewer's turns (shift applied where found): {int(in_ellie):,}"
        f" of {int(table['words'].sum()):,} ({in_ellie / table['words'].sum():.2%});"
        f" sessions above 2%: {ids(table[table['share_in_ellie'] > 0.02])}"
    )
    for margin in NEAR:
        share = (table[f"share_near_ellie_{margin}"] * table["words"]).sum() / table["words"].sum()
        print(f"words within {margin} s of the interviewer's turns: {share:.2%}")
    positive, negative = table[table["label"] == 1], table[table["label"] == 0]
    p = float(mannwhitneyu(positive["share_in_ellie"], negative["share_in_ellie"]).pvalue)
    print(
        f"share inside her turns by PHQ-8 class: positive {positive['share_in_ellie'].mean():.4f},"
        f" negative {negative['share_in_ellie'].mean():.4f}; Mann-Whitney p = {p:.3f}"
    )
    manual = table[table.index < AUTOMATIC_FROM]["share_in_ellie"]
    automatic = table[table.index >= AUTOMATIC_FROM]["share_in_ellie"]
    print(
        f"median share inside her turns: sessions 300-363 (lines typed) {manual.median():.4f},"
        f" 364-492 (lines written automatically) {automatic.median():.4f}"
    )
    print(
        f"turns of the interviewer and the participant that overlap in time:"
        f" {table['simultaneous_seconds'].sum() / 60:.1f} min in total;"
        f" interviewer speech {table['ellie_seconds'].sum() / 3600:.1f} h"
    )
    no_ellie = table[table["ellie_turns"] == 0]
    print(
        f"sessions without interviewer lines: {ids(no_ellie)};"
        f" share of words inside participant turns: "
        + ", ".join(f"{s} {v:.2f}" for s, v in no_ellie["share_in_participant_at_0"].items())
    )
    print(
        "words inside the documented interruptions: "
        + ", ".join(f"{s} {int(table.loc[s, 'words_in_interruption'])}" for s in INTERRUPTIONS)
    )
    write_json(
        run_dir / "run.json",
        {
            "step": "check_daicwoz_timing",
            "status": "done",
            "started_utc": started,
            "finished_utc": utc_now(),
            "git": git_state(),
            "inputs": {
                "daicwoz_transcripts": config["daicwoz_transcripts"],
                "word_times": {"file": words_pin["run"], "sha256": words_pin["sha256"]},
                "labels": {"file": labels_pin["run"], "sha256": labels_pin["sha256"]},
            },
            "settings": {
                "shifts_s": [float(SHIFTS[0]), float(SHIFTS[-1]), 0.1],
                "shifted_s": SHIFTED,
                "near_s": list(NEAR),
                "interruptions_s": {str(k): list(v) for k, v in INTERRUPTIONS.items()},
            },
            "output": {"file": "sessions.tsv", "sha256": sha256_file(run_dir / "sessions.tsv")},
        },
    )
    print(f"run: {run_dir.relative_to(workspace.root)}")


if __name__ == "__main__":
    main()
