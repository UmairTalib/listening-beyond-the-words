"""Align the words of every kept participant line to the audio (GPU job).

Submit from an OMNI login node: scripts/slurm/submit.sh align_words

Reads the pinned line selection from configs/pilot/data.yaml, aligns each kept
line's words with the pinned wav2vec2 model and writes
$LBW_WORKSPACE/derived/word_times/run-<job id>/ with words.parquet (one row per
word: session, line, word_index, word, start, end, score), failures.parquet (lines
that could not be aligned) and run.json. Prints aggregate numbers only.
"""

import os
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from listening_beyond_words.alignment import WordAligner
from listening_beyond_words.models import load_pins, local_snapshot
from listening_beyond_words.runinfo import (
    check_sha256,
    git_state,
    sha256_file,
    utc_now,
    write_json,
)
from listening_beyond_words.sessions import session_files
from listening_beyond_words.settings import get_workspace
from listening_beyond_words.transcripts import normalise, read_transcript

CONFIG = Path("configs/pilot/data.yaml")
PAD_SECONDS = 0.2  # extra audio on each side of a line, at most half the gap to its neighbour


def windows_for(lines: pd.DataFrame, duration: float) -> pd.DataFrame:
    """Audio window of every kept line: its span, padded into half the gap to its neighbours."""
    timed = lines[~lines["broken"]].sort_values("start")
    starts, ends = timed["start"].to_numpy(), timed["end"].to_numpy()
    previous_end = np.concatenate([[0.0], ends[:-1]])
    next_start = np.concatenate([starts[1:], [duration]])
    pad_before = np.clip(np.minimum(PAD_SECONDS, (starts - previous_end) / 2), 0.0, None)
    pad_after = np.clip(np.minimum(PAD_SECONDS, (next_start - ends) / 2), 0.0, None)
    timed = timed.assign(
        from_s=np.clip(starts - pad_before, 0.0, duration),
        to_s=np.clip(ends + pad_after, 0.0, duration),
    )
    return timed[~timed["drop"]]


def main() -> None:
    os.umask(0o077)
    workspace = get_workspace()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    selection = config["participant_lines"]
    lines_path = workspace.resolve(selection["run"]) / "lines.parquet"
    check_sha256(lines_path, selection["sha256"])
    pin = load_pins(Path("configs/pilot/models.yaml"))["wav2vec2_960h"]
    run_dir = workspace.derived / "word_times" / f"run-{os.environ.get('SLURM_JOB_ID', 'local')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    info: dict[str, object] = {
        "step": "align_words",
        "status": "running",
        "started_utc": utc_now(),
        "git": git_state(),
        "model": {"repo_id": pin.repo_id, "revision": pin.revision},
        "inputs": {"lines": {"file": selection["run"], "sha256": selection["sha256"]}},
        "settings": {"pad_seconds": PAD_SECONDS},
        "slurm": {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname()},
    }
    write_json(run_dir / "run.json", info)
    try:
        aligner = WordAligner(local_snapshot(pin, Path(os.environ["HF_HOME"])))
        lines = pd.read_parquet(lines_path)
        word_rows: list[dict[str, object]] = []
        failures: list[dict[str, object]] = []
        for session_id, session_lines in lines.groupby("session"):
            audio_path, transcript_path = session_files(
                workspace.edaic / "sessions", int(session_id)
            )
            audio, rate = sf.read(audio_path, dtype="float32")
            texts = read_transcript(transcript_path)["text"]
            for row in windows_for(session_lines, len(audio) / rate).itertuples():
                words = normalise(texts.iloc[row.line]).split()
                clip = audio[int(row.from_s * rate) : int(row.to_s * rate)]
                times = aligner.align(clip, words) if words else None
                if times is None:
                    failures.append(
                        {"session": int(session_id), "line": row.line, "words": len(words)}
                    )
                    continue
                word_rows += [
                    {
                        "session": int(session_id),
                        "line": row.line,
                        "word_index": t.position,
                        "word": words[t.position],
                        "start": round(row.from_s + t.start, 3),
                        "end": round(row.from_s + t.end, 3),
                        "score": round(t.score, 4),
                    }
                    for t in times
                ]
        words_table = pd.DataFrame(word_rows)
        words_table.to_parquet(run_dir / "words.parquet", index=False)
        pd.DataFrame(failures, columns=["session", "line", "words"]).to_parquet(
            run_dir / "failures.parquet", index=False
        )
        digest = sha256_file(run_dir / "words.parquet")
        seconds = words_table["end"] - words_table["start"]
        counts = {
            "lines_aligned": int(words_table.groupby(["session", "line"]).ngroups),
            "lines_failed": len(failures),
            "words_failed": sum(int(str(f["words"])) for f in failures),
            "words": len(words_table),
            "sessions": int(words_table["session"].nunique()),
        }
        print(
            f"lines aligned {counts['lines_aligned']}, failed {counts['lines_failed']}"
            f" ({counts['words_failed']} words); words {counts['words']} in {counts['sessions']} sessions"
        )
        q = np.quantile(seconds, [0.05, 0.5, 0.95])
        print(
            f"word duration (s): 5% {q[0]:.2f}, median {q[1]:.2f}, 95% {q[2]:.2f}, max {seconds.max():.2f}"
        )
        s = np.quantile(words_table["score"], [0.05, 0.25, 0.5])
        print(
            f"alignment score: 5% {s[0]:.2f}, 25% {s[1]:.2f}, median {s[2]:.2f};"
            f" words below 0.2: {np.mean(words_table['score'] < 0.2):.3f}"
        )
        info.update(
            status="done",
            finished_utc=utc_now(),
            counts=counts,
            output={"file": "words.parquet", "sha256": digest},
        )
        print(f"run: {run_dir.relative_to(workspace.root)}")
        print(f"sha256: {digest}")
    except BaseException:
        info.update(status="failed", finished_utc=utc_now())
        raise
    finally:
        write_json(run_dir / "run.json", info)


if __name__ == "__main__":
    main()
