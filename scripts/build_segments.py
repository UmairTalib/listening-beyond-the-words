"""Apply the cut rule and cut the pilot's 40-word segments on an OMNI login node.

Usage, inside the clone:
    "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/build_segments.py

Reads the pinned word times, line selection, labels and DAIC-WOZ transcripts, the cut rule
(``daicwoz_cut``) and the x0 cut settings from configs/pilot/data.yaml. Decides for every
aligned word whether the cut rule keeps it and which part it belongs to (pilot protocol,
section "Interviewer detector"), cuts each session's kept words into segments of exactly
40 words, and computes the x0 window of every part with the x0 cut settings, which checks
that every segment can be cut from its recording. Writes
$LBW_WORKSPACE/derived/segments/run-<UTC time>/ with
  segments.parquet  one row per segment: its parts, the words just outside it, the dropped
                    words between its parts, the interviewer turns next to its parts, the
                    x0 window of each part, its length and its words; the listening check
                    and x0 cut exactly these windows
  words.parquet     one row per aligned word: kept or not, its part, its conditions and the
                    cause of a drop (no text)
  zones.parquet     the interviewer's turns and the widened interruptions, in recording time
  participants.tsv  per participant: words at each stage, dropped shares and segments
  run.json          commit, inputs, settings, counts and the SHA-256 of every file
Prints aggregate numbers only: never text, and never one participant's values.
"""

import json
import os
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from scipy.stats import mannwhitneyu

from listening_beyond_words.cut import (
    CAUSES,
    FLAGS,
    CutRule,
    Turns,
    all_turns,
    cut_words,
    to_ms,
    zone_table,
)
from listening_beyond_words.daicwoz import load_transcripts
from listening_beyond_words.runinfo import (
    check_sha256,
    git_state,
    sha256_file,
    utc_now,
    utc_stamp,
    write_json,
)
from listening_beyond_words.segments import (
    GAP_SECONDS,
    SEGMENT_WORDS,
    build_segments,
    segment_neighbours,
    segment_walls,
)
from listening_beyond_words.sessions import PILOT_IDS, session_files
from listening_beyond_words.settings import Workspace, get_workspace
from listening_beyond_words.views import segment_windows

CONFIG = Path("configs/pilot/data.yaml")
BAG = 4  # segments per bag (pilot protocol, fixed settings)
EPOCH_SEGMENTS = 16  # segments per participant per epoch (pilot protocol, check F5)
LONG_SECONDS = 30.0


def pinned_file(workspace: Workspace, pin: dict, name: str) -> Path:
    """A file of a pinned run, checked against its SHA-256."""
    path = workspace.resolve(pin["run"]) / name
    check_sha256(path, pin["sha256"])
    return path


def compare(values: pd.Series, groups: pd.Series, name: str) -> float:
    """Two-sided Mann-Whitney test between the two groups; prints the group means."""
    first, second = sorted(groups.unique())
    a, b = values[groups == first], values[groups == second]
    p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    print(
        f"  by {name}: {first} mean {a.mean():.4f} (n={len(a)}),"
        f" {second} mean {b.mean():.4f} (n={len(b)}); Mann-Whitney p = {p:.3f}"
    )
    return p


def interview_edges(words: pd.DataFrame, turns: dict[int, Turns]) -> tuple[int, int, int]:
    """Words dropped only by the interview limits: before her first turn, after her last, sessions."""
    only = words[(words["cause"] == CAUSES["outside_interview"])]
    before = after = 0
    for session, group in only.groupby("session"):
        span = turns[int(session)].interview
        assert span is not None
        before += int((to_ms(group["start"].to_numpy()) < span[0]).sum())
        after += int((to_ms(group["end"].to_numpy()) > span[1]).sum())
    return before, after, int(only["session"].nunique())


def dropped_line_check(lines: pd.DataFrame, cut: pd.DataFrame) -> tuple[int, int]:
    """Margins past the halfway point to a line the interviewer filter dropped, and parts over one.

    Lines with broken timestamps are left out, because their times are not those of their words.
    """
    dropped = lines[lines["drop"] & ~lines["broken"]]
    spans = {
        int(cast(int, s)): (g["start"].to_numpy(dtype=float), g["end"].to_numpy(dtype=float))
        for s, g in dropped.groupby("session")
    }
    past = over = 0
    for session, low, high, start, end in cut[
        ["session", "low", "high", "start", "end"]
    ].itertuples(index=False):
        line_starts, line_ends = spans.get(int(session), (np.empty(0), np.empty(0)))
        over += int(((line_starts < end) & (line_ends > start)).any())
        earlier, later = line_ends[line_ends <= start], line_starts[line_starts >= end]
        past += int(len(earlier) > 0 and low < (earlier.max() + start) / 2 - 1e-9)
        past += int(len(later) > 0 and high > (end + later.min()) / 2 + 1e-9)
    return past, over


def main() -> None:
    os.umask(0o077)
    workspace = get_workspace()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    pins = {name: config[name] for name in ("word_times", "participant_lines", "labels")}
    rule = CutRule.from_config(config["daicwoz_cut"])
    x0 = config["x0_cut"]
    margin, wall_guard = x0["margin_ms"] / 1000, x0["wall_guard_ms"] / 1000
    run_dir = workspace.derived / "segments" / f"run-{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    info: dict[str, object] = {
        "step": "build_segments",
        "status": "running",
        "started_utc": utc_now(),
        "git": git_state(),
        "config": {"file": str(CONFIG), "sha256": sha256_file(CONFIG)},
        "inputs": {
            **{name: {"file": pin["run"], "sha256": pin["sha256"]} for name, pin in pins.items()},
            "daicwoz_transcripts": config["daicwoz_transcripts"],
        },
        "settings": {
            "words_per_segment": SEGMENT_WORDS,
            "gap_seconds": GAP_SECONDS,
            "daicwoz_cut": config["daicwoz_cut"],
            "x0_cut": x0,
        },
    }
    write_json(run_dir / "run.json", info)
    try:
        words_path = pinned_file(workspace, pins["word_times"], "words.parquet")
        aligned_from = json.loads((words_path.parent / "run.json").read_text(encoding="utf-8"))
        if aligned_from["inputs"]["lines"]["sha256"] != pins["participant_lines"]["sha256"]:
            raise ValueError(
                "the pinned word times were not aligned from the pinned line selection"
            )
        lines = pd.read_parquet(pinned_file(workspace, pins["participant_lines"], "lines.parquet"))
        labels = pd.read_parquet(pinned_file(workspace, pins["labels"], "labels.parquet"))
        transcripts = load_transcripts(workspace, config["daicwoz_transcripts"])
        turns = all_turns({s: transcripts[s] for s in PILOT_IDS}, rule)

        words = pd.read_parquet(words_path)
        decided = cut_words(words, turns)
        zones = zone_table(turns)
        segments = build_segments(decided)
        segments = segments.merge(segment_neighbours(decided), on="segment", validate="one_to_one")
        segments = segments.merge(
            segment_walls(segments, zones), on="segment", validate="one_to_one"
        )

        # The x0 window of every part: in order, inside the recording, at most halfway to
        # every other word and at least the wall guard away from every zone. The clips of
        # the listening check and of x0 are cut from exactly these windows.
        durations = {}
        for session in PILOT_IDS:
            audio_path, _ = session_files(workspace.edaic / "sessions", session)
            details = sf.info(audio_path)
            durations[session] = details.frames / details.samplerate
        edges, lows, highs = [], [], []
        for row in segments.itertuples(index=False):
            windows = segment_windows(
                row, durations[int(row.session)], margin=margin, wall_guard=wall_guard
            )
            lows.append([low for low, _ in windows])
            highs.append([high for _, high in windows])
            edges += [
                {"session": int(row.session), "low": low, "high": high, "start": s, "end": e}
                for (low, high), s, e in zip(windows, row.starts, row.ends, strict=True)
            ]
        cut = pd.DataFrame(edges)
        segments["window_starts"] = lows
        segments["window_ends"] = highs
        segments["clip_seconds"] = [
            round(
                sum(h - lo for lo, h in zip(low, high, strict=True)) + GAP_SECONDS * (len(low) - 1),
                3,
            )
            for low, high in zip(lows, highs, strict=True)
        ]

        outputs = {
            "segments.parquet": segments,
            "words.parquet": decided.drop(columns=["word", "score"], errors="ignore"),
            "zones.parquet": zones,
        }
        for name, table in outputs.items():
            table.to_parquet(run_dir / name, index=False)

        # Per participant: words at each stage of the pipeline.
        per = pd.DataFrame(index=pd.Index(PILOT_IDS, name="session"))
        per["transcript_words"] = lines.groupby("session")["words"].sum()
        per["filter_dropped"] = lines[lines["drop"]].groupby("session")["words"].sum()
        per["aligned"] = decided.groupby("session").size()
        per["kept"] = decided[decided["kept"]].groupby("session").size()
        for name in FLAGS:
            per[f"cause_{name}"] = (
                decided[decided["cause"] == CAUSES[name]].groupby("session").size()
            )
        per["segments"] = segments.groupby("session").size()
        per["parts_per_segment"] = segments.groupby("session")["parts"].mean()
        counts_columns = [c for c in per.columns if c != "parts_per_segment"]
        per[counts_columns] = per[counts_columns].fillna(0).astype(int)
        per["unaligned"] = per["transcript_words"] - per["filter_dropped"] - per["aligned"]
        per["dropped_share_total"] = 1 - per["kept"] / per["transcript_words"]
        per["dropped_share_cut"] = 1 - per["kept"] / per["aligned"]
        per = per.join(labels[["label", "gender"]])
        per.to_csv(run_dir / "participants.tsv", sep="\t")

        aligned, kept = int(per["aligned"].sum()), int(per["kept"].sum())
        print(
            f"aligned words {aligned:,} in {int((per['aligned'] > 0).sum())} sessions;"
            f" kept by the cut rule {kept:,}; dropped {aligned - kept:,} ({1 - kept / aligned:.2%})"
        )
        by_cause = {CAUSES[n]: int(per[f"cause_{n}"].sum()) for n in FLAGS}
        print(
            "dropped by cause (first that applies): "
            + ", ".join(f"{c} {n:,}" for c, n in by_cause.items())
        )
        print(
            "words with each condition (a word can have several): "
            + ", ".join(f"{CAUSES[n]} {int(decided[n].sum()):,}" for n in FLAGS)
        )
        before, after, edge_sessions = interview_edges(decided, turns)
        print(
            f"dropped only by the interview limits: {before} before her first turn,"
            f" {after} after her last turn, in {edge_sessions} sessions"
        )
        near_interruption = decided["session"].isin(list(rule.interruptions_ms))
        print(
            f"words near the widened interruptions, {len(rule.interruptions_ms)} sessions together:"
            f" {int(decided.loc[near_interruption, 'near_interruption'].sum())}"
        )

        print(
            f"dropped share of each participant's transcript words (filter, word times and cut):"
            f" median {per['dropped_share_total'].median():.3f},"
            f" 90% {per['dropped_share_total'].quantile(0.9):.3f}, max {per['dropped_share_total'].max():.3f}"
        )
        classes = per["label"].map({0: "negative", 1: "positive"})
        p_label = compare(per["dropped_share_total"], classes, "PHQ-8 class")
        p_gender = compare(per["dropped_share_total"], per["gender"], "gender")
        verdict = (
            "passed"
            if p_label >= 0.05
            else "FAILED: the rule is kept and the difference is reported by cause"
        )
        print(f"balance check, PHQ-8 class (p >= 0.05): {verdict}")
        print("share of aligned words dropped by the cut rule alone:")
        p_cut = compare(per["dropped_share_cut"], classes, "PHQ-8 class")
        print("dropped share of aligned words by cause and PHQ-8 class (means):")
        for n in FLAGS:
            share = per[f"cause_{n}"] / per["aligned"]
            print(
                f"  {CAUSES[n]}: negative {share[classes == 'negative'].mean():.4f},"
                f" positive {share[classes == 'positive'].mean():.4f}"
            )

        n_segments = per["segments"]
        print(
            f"segments {len(segments):,} from {int((n_segments > 0).sum())} participants;"
            f" participants without a segment: {int((n_segments == 0).sum())}"
        )
        print(
            f"segments per participant: min {n_segments.min()}, median {int(n_segments.median())},"
            f" max {n_segments.max()}; fewest kept words {per['kept'].min()}"
        )
        for limit in (BAG, EPOCH_SEGMENTS):
            under = n_segments < limit
            print(
                f"participants with fewer than {limit} segments: {int(under.sum())}"
                f" (negative {int((under & (classes == 'negative')).sum())},"
                f" positive {int((under & (classes == 'positive')).sum())})"
            )
        seconds = segments["seconds"].to_numpy()
        clip_seconds = segments["clip_seconds"].to_numpy()
        for name, values in (
            ("words plus gaps", seconds),
            ("x0 clip: words, margins and gaps", clip_seconds),
        ):
            q = np.quantile(values, [0.05, 0.5, 0.95])
            print(
                f"segment length (s, {name}): 5% {q[0]:.1f}, median {q[1]:.1f}, 95% {q[2]:.1f},"
                f" max {values.max():.1f}; over {LONG_SECONDS:.0f} s: {int((values > LONG_SECONDS).sum())}"
            )
        session_class = segments["session"].map(labels["label"])
        print(
            f"parts per segment: median {segments['parts'].median():.0f}, max {segments['parts'].max()};"
            f" mean negative {segments.loc[session_class == 0, 'parts'].mean():.2f},"
            f" positive {segments.loc[session_class == 1, 'parts'].mean():.2f}"
        )
        print("parts per segment, mean per participant:")
        has_segments = n_segments > 0
        p_parts = compare(
            per.loc[has_segments, "parts_per_segment"], classes[has_segments], "PHQ-8 class"
        )
        sides = np.concatenate([cut["start"] - cut["low"], cut["high"] - cut["end"]]).astype(float)
        full = int(np.sum(sides >= margin - 1e-9))
        print(
            f"part edges with the full {margin} s margin: {full:,} of {len(sides):,};"
            f" median margin {np.median(sides):.3f} s"
        )
        past, over = dropped_line_check(lines, cut)
        print(
            f"margins past the halfway point to a line dropped by the interviewer filter: {past};"
            f" parts that overlap such a line's times: {over}"
        )

        counts = {
            "aligned_words": aligned,
            "kept_words": kept,
            "dropped_by_cause": by_cause,
            "interview_limits_only": {"before": before, "after": after, "sessions": edge_sessions},
            "segments": len(segments),
            "participants_without_segments": int((n_segments == 0).sum()),
            "participants_under_4_segments": int((n_segments < BAG).sum()),
            "participants_under_16_segments": int((n_segments < EPOCH_SEGMENTS).sum()),
            "fewest_kept_words": int(per["kept"].min()),
            "segments_over_30_s": int((clip_seconds > LONG_SECONDS).sum()),
            "part_edges": len(sides),
            "part_edges_with_full_margin": full,
            "margins_past_a_dropped_line": past,
            "parts_over_a_dropped_line": over,
        }
        balance = {
            "p_label": p_label,
            "p_gender": p_gender,
            "p_label_cut_only": p_cut,
            "p_label_parts_per_segment": p_parts,
            "verdict": verdict,
        }
        written = {name: sha256_file(run_dir / name) for name in [*outputs, "participants.tsv"]}
        info.update(
            status="done",
            finished_utc=utc_now(),
            counts=counts,
            balance=balance,
            outputs=written,
            output={"file": "segments.parquet", "sha256": written["segments.parquet"]},
        )
        print(f"run: {run_dir.relative_to(workspace.root)}")
        print(f"sha256: {written['segments.parquet']}")
    except BaseException:
        info.update(status="failed", finished_utc=utc_now())
        raise
    finally:
        write_json(run_dir / "run.json", info)


if __name__ == "__main__":
    main()
