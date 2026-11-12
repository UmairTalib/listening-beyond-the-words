"""Test a finer interviewer check on the clips of a listening check (GPU job).

Submit from an OMNI login node:
    scripts/slurm/submit.sh check_ellie_chunks derived/listening_check/run-<UTC time>

The line filter judges speech in 1.5 s windows, and a stretch of speech shorter
than 0.75 s gets no window at all, so a short remark by the interviewer inside a
kept line is never judged. This check cuts the speech inside each part of the
listened clips into short chunks at small pauses and scores every chunk on its
own with the line filter's voice profiles: above 0 means closer to the
interviewer's voice than to the participant's. As a control, it scores the
chunks of the dropped lines that hold a known interviewer phrase, which are the
interviewer. The profiles come from the sessions of the listened clips only.

Writes $LBW_WORKSPACE/derived/ellie_chunks/run-<job id>/ with chunks.parquet (one
row per chunk: kind, clip, session, part, start, end, words, score) and run.json.
Prints numbers by clip number only: never text, IDs or labels.
"""

import os
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from listening_beyond_words import interviewer as iv
from listening_beyond_words.models import load_pins, local_snapshot
from listening_beyond_words.runinfo import check_sha256, git_state, utc_now, write_json
from listening_beyond_words.sessions import session_files
from listening_beyond_words.settings import get_workspace

CONFIG = Path("configs/pilot/data.yaml")
CHUNK_VAD = {
    "threshold": 0.5,
    "min_speech_duration_ms": 100,
    "min_silence_duration_ms": 100,
    "speech_pad_ms": 30,
}
# The x-vector head needs at least 15 frames (about 0.31 s) of audio; shorter chunks
# are only counted, not scored.
MIN_CHUNK_SECONDS = 0.4


def clip_of(audio: np.ndarray, rate: int, span: iv.Span) -> np.ndarray:
    return audio[int(span[0] * rate) : int(span[1] * rate)]


def word_count(words: pd.DataFrame, span: iv.Span) -> int:
    """Aligned words with at least half of their time inside the span."""
    inside = np.minimum(words["end"], span[1]) - np.maximum(words["start"], span[0])
    return int((inside >= 0.5 * (words["end"] - words["start"])).sum())


def position(chunk: iv.Span, part: iv.Span) -> str:
    middle = (chunk[0] + chunk[1]) / 2
    share = (middle - part[0]) / max(part[1] - part[0], 1e-9)
    return "start" if share < 0.2 else "end" if share > 0.8 else "middle"


def main() -> None:
    os.umask(0o077)
    listening = sys.argv[1] if len(sys.argv) > 1 else ""
    if not listening:
        raise SystemExit("usage: check_ellie_chunks.py derived/listening_check/run-<UTC time>")
    workspace = get_workspace()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    lines_path = workspace.resolve(config["participant_lines"]["run"]) / "lines.parquet"
    words_path = workspace.resolve(config["word_times"]["run"]) / "words.parquet"
    check_sha256(lines_path, config["participant_lines"]["sha256"])
    check_sha256(words_path, config["word_times"]["sha256"])
    listen_dir = workspace.resolve(listening)
    key = pd.read_csv(listen_dir / "key.tsv", sep="\t").merge(
        pd.read_csv(listen_dir / "answers.tsv", sep="\t", keep_default_na=False), on="clip"
    )
    pin = load_pins(Path("configs/pilot/models.yaml"))["wavlm_sv"]
    info: dict[str, object] = {
        "step": "check_ellie_chunks",
        "status": "running",
        "started_utc": utc_now(),
        "git": git_state(),
        "model": {"repo_id": pin.repo_id, "revision": pin.revision},
        "inputs": {
            "listening_check": listening,
            "participant_lines": config["participant_lines"],
            "word_times": config["word_times"],
        },
        "settings": {
            "chunk_vad": CHUNK_VAD,
            "min_chunk_seconds": MIN_CHUNK_SECONDS,
            "profile_vad": iv.VAD_OPTIONS,
            "window_seconds": iv.WINDOW_SECONDS,
            "hop_seconds": iv.HOP_SECONDS,
        },
        "slurm": {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname()},
    }
    run_dir = workspace.derived / "ellie_chunks" / f"run-{os.environ.get('SLURM_JOB_ID', 'local')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "run.json", info)
    try:
        embedder = iv.VoiceEmbedder(local_snapshot(pin, Path(os.environ["HF_HOME"])))
        lines = pd.read_parquet(
            lines_path, columns=["session", "line", "start", "end", "phrase", "drop"]
        )
        words = pd.read_parquet(words_path, columns=["session", "start", "end"])
        key["session"] = key["segment"].str.split("_").str[0].astype(int)
        rows: list[dict[str, object]] = []
        chunk_embeddings: list[np.ndarray] = []
        profiles_in: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for session_id in sorted(key["session"].unique()):
            audio_path, _ = session_files(workspace.edaic / "sessions", int(session_id))
            audio, rate = sf.read(audio_path, dtype="float32")
            if rate != iv.SAMPLING_RATE or audio.ndim != 1:
                raise ValueError(f"session {session_id}: expected 16 kHz mono audio")
            here = lines[lines["session"] == session_id]
            spans = list(zip(here["start"].astype(float), here["end"].astype(float), strict=True))
            regions = iv.speech_regions(audio)
            line_windows = [w for part in iv.intersect(spans, regions) for w in iv.windows(part)]
            gap_windows = [
                w
                for part in iv.intersect(iv.gaps(spans, len(audio) / rate), regions)
                for w in iv.windows(part)
            ]
            profiles_in[int(session_id)] = (
                embedder.embed([clip_of(audio, rate, w) for w in line_windows]),
                embedder.embed([clip_of(audio, rate, w) for w in gap_windows]),
            )
            fine = iv.speech_regions(audio, options=CHUNK_VAD)
            session_words = words[words["session"] == session_id]
            candidates: list[dict[str, object]] = []
            for clip in key[key["session"] == session_id].itertuples():
                parts = [tuple(map(float, p.split("-"))) for p in clip.windows.split("; ")]
                for number, part in enumerate(parts, start=1):
                    for chunk in iv.intersect([part], fine):
                        candidates.append(
                            {
                                "kind": "clip",
                                "clip": clip.clip[:2],
                                "heard": clip.another_voice,
                                "where": clip.where,
                                "part": f"{number}/{len(parts)}",
                                "position": position(chunk, part),
                                "start": chunk[0],
                                "end": chunk[1],
                            }
                        )
            dropped = here[here["phrase"] & here["drop"]]
            for start, end in zip(dropped["start"], dropped["end"], strict=True):
                for chunk in iv.intersect([(float(start), float(end))], fine):
                    candidates.append({"kind": "control", "start": chunk[0], "end": chunk[1]})
            for row in candidates:
                span = (float(str(row["start"])), float(str(row["end"])))
                row.update(session=int(session_id), words=word_count(session_words, span))
                row["seconds"] = span[1] - span[0]
                if row["seconds"] >= MIN_CHUNK_SECONDS:
                    chunk_embeddings.append(embedder.embed([clip_of(audio, rate, span)])[0])
                    row["embedded"] = len(chunk_embeddings) - 1
                else:
                    row["embedded"] = -1
            rows += candidates

        interviewer = iv.global_profile(np.concatenate([gap for _, gap in profiles_in.values()]))
        table = pd.DataFrame(rows)
        table["score"] = np.nan
        for session_id, (line_emb, gap_emb) in profiles_in.items():
            ellie, participant = iv.session_profiles(line_emb, gap_emb, interviewer)
            mask = (table["session"] == session_id) & (table["embedded"] >= 0)
            if mask.any():
                emb = np.stack([chunk_embeddings[i] for i in table.loc[mask, "embedded"]])
                table.loc[mask, "score"] = iv.interviewer_scores(emb, ellie, participant)
        table.drop(columns=["embedded"]).to_parquet(run_dir / "chunks.parquet", index=False)

        control = table[(table["kind"] == "control") & table["score"].notna()]
        clips = table[table["kind"] == "clip"]
        scored = clips[clips["score"].notna()]
        print(
            f"control (dropped interviewer-phrase lines): {len(control)} chunks,"
            f" share above 0: {np.mean(control['score'] > 0):.3f}"
        )
        print(
            f"clip chunks: {len(clips)}, scored {len(scored)}, too short {len(clips) - len(scored)};"
            f" share above 0: {np.mean(scored['score'] > 0):.3f}"
        )
        print("clip heard where  | chunks above 0 (without words) | max score | at")
        summary = []
        for clip in key.itertuples():
            group = scored[scored["clip"] == clip.clip[:2]]
            if group.empty:
                summary.append((clip.another_voice, False, False))
                print(
                    f"  {clip.clip[:2]}  {clip.another_voice:<4} {clip.where:<6} | no chunk long enough to score"
                )
                continue
            above = group[group["score"] > 0]
            top = group.loc[group["score"].idxmax()]
            summary.append((clip.another_voice, len(above) > 0, bool((group["words"] == 0).any())))
            print(
                f"  {clip.clip[:2]}  {clip.another_voice:<4} {clip.where:<6} | {len(above):2d} of {len(group):2d}"
                f" ({int((above['words'] == 0).sum())}) | {top['score']:+.3f} |"
                f" part {top['part']} {top['position']}, {int(top['words'])} words"
            )
        yes = [s for s in summary if s[0] == "yes"]
        no = [s for s in summary if s[0] == "no"]
        print(
            f"clips with a chunk above 0: heard yes {sum(s[1] for s in yes)} of {len(yes)},"
            f" heard no {sum(s[1] for s in no)} of {len(no)}"
        )
        print(
            f"clips with a chunk without aligned words: heard yes {sum(s[2] for s in yes)} of"
            f" {len(yes)}, heard no {sum(s[2] for s in no)} of {len(no)}"
        )
        info.update(status="done", finished_utc=utc_now(), chunks=len(table))
    except BaseException:
        info.update(status="failed", finished_utc=utc_now())
        raise
    finally:
        write_json(run_dir / "run.json", info)


if __name__ == "__main__":
    main()
