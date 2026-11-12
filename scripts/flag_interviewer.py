"""Score the speech of the 189 pilot sessions for the interviewer's voice (GPU job).

Submit from an OMNI login node: scripts/slurm/submit.sh flag_interviewer

For each pilot session it finds the speech (Silero), cuts short windows inside the
participant's transcript lines and inside the gaps between them, embeds them with
the pinned speaker-verification model and scores each window: above 0 means closer
to the interviewer's voice than to the participant's.

Writes $LBW_WORKSPACE/derived/interviewer/run-<job id>/ with
  windows.parquet  one row per window: session, kind, line, start, end, phrase_line, score
  listening/       35 short clips in random order for a blind listening check, and
                   listening.tsv, which says what each clip is (open it after listening)
  run.json         commit, model, settings, status and times
and prints aggregate numbers only: never text, and never one session's values.
"""

import json
import os
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import roc_auc_score

from listening_beyond_words import interviewer as iv
from listening_beyond_words.models import load_pins, local_snapshot
from listening_beyond_words.sessions import PILOT_IDS, session_files
from listening_beyond_words.settings import get_workspace
from listening_beyond_words.transcripts import INTERVIEWER_PHRASES, normalise, read_transcript

SEED = 101
LISTENING = {"mostly": 10, "partly": 10, "none": 10, "phrase": 5}
CLIP_SECONDS = 10.0


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_state() -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout

    return {"commit": git("rev-parse", "HEAD").strip(), "dirty": bool(git("status", "--porcelain"))}


def write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def quantiles(values: pd.Series) -> str:
    q = np.quantile(values.to_numpy(dtype=float), [0.1, 0.5, 0.9])
    return f"10% {q[0]:+.3f}, median {q[1]:+.3f}, 90% {q[2]:+.3f}"


def score_sessions(
    embedder: iv.VoiceEmbedder, sessions_dir: Path
) -> tuple[pd.DataFrame, dict[tuple[int, int], iv.Span]]:
    rows: list[dict[str, object]] = []
    chunks: list[np.ndarray] = []
    line_spans: dict[tuple[int, int], iv.Span] = {}
    for session_id in PILOT_IDS:
        audio_path, transcript_path = session_files(sessions_dir, session_id)
        audio, rate = sf.read(audio_path, dtype="float32")
        if rate != iv.SAMPLING_RATE or audio.ndim != 1:
            raise ValueError(f"session {session_id}: expected 16 kHz mono audio")
        frame = read_transcript(transcript_path)
        lines = list(zip(frame["start"].astype(float), frame["end"].astype(float), strict=True))
        phrase_lines = {
            index
            for index, text in enumerate(frame["text"])
            if any(f" {p} " in f" {normalise(text)} " for p in INTERVIEWER_PHRASES)
        }
        regions = iv.speech_regions(audio)
        spans: list[tuple[str, int, iv.Span]] = []
        for index, line in enumerate(lines):
            line_spans[(session_id, index)] = line
            for part in iv.intersect([line], regions):
                spans += [("line", index, window) for window in iv.windows(part)]
        for part in iv.intersect(iv.gaps(lines, len(audio) / rate), regions):
            spans += [("gap", -1, window) for window in iv.windows(part)]
        chunks.append(
            embedder.embed([audio[int(s * rate) : int(e * rate)] for _, _, (s, e) in spans])
        )
        rows += [
            {
                "session": session_id,
                "kind": kind,
                "line": index,
                "start": start,
                "end": end,
                "phrase_line": index in phrase_lines,
            }
            for kind, index, (start, end) in spans
        ]
    table = pd.DataFrame(rows)
    embeddings = np.concatenate(chunks)
    is_gap = (table["kind"] == "gap").to_numpy()
    interviewer = iv.global_profile(embeddings[is_gap])
    scores = np.full(len(table), np.nan)
    for positions in table.groupby("session").indices.values():
        lines_here, gaps_here = positions[~is_gap[positions]], positions[is_gap[positions]]
        if len(lines_here) == 0:
            continue
        profiles = iv.session_profiles(embeddings[lines_here], embeddings[gaps_here], interviewer)
        scores[positions] = iv.interviewer_scores(embeddings[positions], *profiles)
    table["score"] = scores
    return table, line_spans


def report(table: pd.DataFrame) -> pd.DataFrame:
    scored = table[table["score"].notna()]
    line_w = scored[scored["kind"] == "line"].assign(flag=lambda t: t["score"] > 0)
    gap_w = scored[scored["kind"] == "gap"]
    print(
        f"sessions {scored['session'].nunique()}, line windows {len(line_w)}, gap windows {len(gap_w)}"
    )
    print(
        f"AUROC of the score, gap windows against line windows: {roc_auc_score(scored['kind'] == 'gap', scored['score']):.3f}"
    )
    print(f"score of gap windows:  {quantiles(gap_w['score'])}")
    print(f"score of line windows: {quantiles(line_w['score'])}")
    print(
        f"share above 0: gap windows {np.mean(gap_w['score'] > 0):.3f}, line windows {line_w['flag'].mean():.3f}"
    )
    per_line = line_w.groupby(["session", "line"]).agg(
        share=("flag", "mean"), phrase=("phrase_line", "first")
    )
    print(
        f"lines with speech windows {len(per_line)}: mostly interviewer (>= 50% of windows)"
        f" {np.mean(per_line['share'] >= 0.5):.3f}, any window above 0 {np.mean(per_line['share'] > 0):.3f}"
    )
    controls = per_line[per_line["phrase"]]
    print(
        f"control lines with a known interviewer phrase {len(controls)}: mean share above 0"
        f" {controls['share'].mean():.3f}, mostly interviewer {np.mean(controls['share'] >= 0.5):.3f}"
    )
    print(
        f"all other lines: mean share above 0 {per_line.loc[~per_line['phrase'], 'share'].mean():.3f}"
    )
    flagged_seconds = sum(
        e - s
        for _, group in line_w[line_w["flag"]].groupby("session")
        for s, e in iv.merge(zip(group["start"], group["end"], strict=True))
    )
    line_seconds = sum(
        e - s
        for _, group in line_w.groupby("session")
        for s, e in iv.merge(zip(group["start"], group["end"], strict=True))
    )
    print(
        f"line speech time inside windows above 0: {flagged_seconds / 3600:.2f} h of {line_seconds / 3600:.2f} h"
    )
    by_session = line_w.groupby("session")["flag"].mean()
    print(
        f"share of line windows above 0 per session: min {by_session.min():.3f},"
        f" median {by_session.median():.3f}, max {by_session.max():.3f};"
        f" sessions above 5%: {int((by_session > 0.05).sum())}, above 20%: {int((by_session > 0.2).sum())}"
    )
    return per_line


def write_listening_set(
    run_dir: Path,
    sessions_dir: Path,
    table: pd.DataFrame,
    per_line: pd.DataFrame,
    line_spans: dict[tuple[int, int], iv.Span],
) -> int:
    rng = np.random.default_rng(SEED)
    groups = {
        "mostly": per_line.index[per_line["share"] >= 0.5],
        "partly": per_line.index[(per_line["share"] > 0) & (per_line["share"] < 0.5)],
        "none": per_line.index[per_line["share"] == 0],
        "phrase": per_line.index[per_line["phrase"]],
    }
    chosen: list[tuple[str, tuple[int, int]]] = []
    for group, size in LISTENING.items():
        keys = [key for key in groups[group] if key not in {k for _, k in chosen}]
        picks = rng.choice(len(keys), size=min(size, len(keys)), replace=False) if keys else []
        chosen += [(group, keys[int(pick)]) for pick in picks]
    folder = run_dir / "listening"
    folder.mkdir()
    rows = []
    for number, position in enumerate(rng.permutation(len(chosen)), start=1):
        group, (session_id, line) = chosen[position]
        line_start, line_end = line_spans[(session_id, line)]
        windows_here = table[(table["session"] == session_id) & (table["line"] == line)]
        above = windows_here[windows_here["score"] > 0].sort_values("start")
        centre = float(above["start"].iloc[0]) if len(above) else line_start
        clip_start = max(line_start, centre - 3.0)
        clip_end = min(line_end, clip_start + CLIP_SECONDS)
        audio_path, _ = session_files(sessions_dir, session_id)
        rate = sf.info(str(audio_path)).samplerate
        audio, _ = sf.read(
            audio_path, dtype="float32", start=int(clip_start * rate), stop=int(clip_end * rate)
        )
        name = f"{number:02d}.wav"
        sf.write(folder / name, audio, rate, subtype="PCM_16")
        rows.append(
            {
                "clip": name,
                "group": group,
                "session": session_id,
                "line": line,
                "start": round(clip_start, 2),
                "end": round(clip_end, 2),
                "share_above_0": round(float(per_line.loc[(session_id, line), "share"]), 3),
            }
        )
    pd.DataFrame(rows).to_csv(folder / "listening.tsv", sep="\t", index=False)
    return len(rows)


def main() -> None:
    os.umask(0o077)
    workspace = get_workspace()
    sessions_dir = workspace.edaic / "sessions"
    pin = load_pins(Path("configs/pilot/models.yaml"))["wavlm_sv"]
    run_dir = workspace.derived / "interviewer" / f"run-{os.environ.get('SLURM_JOB_ID', 'local')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    info: dict[str, object] = {
        "step": "flag_interviewer",
        "status": "running",
        "started_utc": now(),
        "git": git_state(),
        "model": {"repo_id": pin.repo_id, "revision": pin.revision},
        "settings": {
            "window_seconds": iv.WINDOW_SECONDS,
            "hop_seconds": iv.HOP_SECONDS,
            "min_window_seconds": iv.MIN_WINDOW_SECONDS,
            "min_gap_seconds": iv.MIN_GAP_SECONDS,
            "min_profile_windows": iv.MIN_PROFILE_WINDOWS,
            "vad": iv.VAD_OPTIONS,
            "threshold": 0.0,
            "seed": SEED,
        },
        "slurm": {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname()},
    }
    write_json(run_dir / "run.json", info)
    try:
        embedder = iv.VoiceEmbedder(local_snapshot(pin, Path(os.environ["HF_HOME"])))
        table, line_spans = score_sessions(embedder, sessions_dir)
        table.to_parquet(run_dir / "windows.parquet", index=False)
        per_line = report(table)
        clips = write_listening_set(run_dir, sessions_dir, table, per_line, line_spans)
        print(f"listening clips: {clips}, in {run_dir / 'listening'}")
        info.update(status="done", finished_utc=now(), windows=len(table))
    except BaseException:
        info.update(status="failed", finished_utc=now())
        raise
    finally:
        write_json(run_dir / "run.json", info)


if __name__ == "__main__":
    main()
