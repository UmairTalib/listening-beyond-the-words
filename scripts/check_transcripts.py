"""Check the E-DAIC transcripts of the 189 pilot sessions; print aggregate numbers only.

Usage, on an OMNI login node, inside the clone:
    "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/check_transcripts.py

Reads the transcript and the audio header of every pilot session and prints
counts and distributions over all sessions. It never prints transcript text, so
its output may leave OMNI. It writes nothing.
"""

from collections.abc import Sequence

import numpy as np
import soundfile as sf

from listening_beyond_words.sessions import PILOT_IDS, session_files
from listening_beyond_words.settings import get_workspace
from listening_beyond_words.transcripts import (
    INTERVIEWER_PHRASES,
    LOW_CONFIDENCE,
    TranscriptSummary,
    read_transcript,
    summarise,
)


def spread(values: Sequence[float]) -> str:
    v = np.asarray(values, dtype=float)
    return f"min {v.min():.2f}, median {np.median(v):.2f}, max {v.max():.2f}"


def main() -> None:
    sessions_dir = get_workspace().edaic / "sessions"
    summaries: list[TranscriptSummary] = []
    durations: list[float] = []
    for session_id in PILOT_IDS:
        audio, transcript = session_files(sessions_dir, session_id)
        durations.append(sf.info(str(audio)).duration)
        summaries.append(summarise(read_transcript(transcript)))
    pairs = list(zip(summaries, durations, strict=True))
    n = len(summaries)
    words = [s.words for s in summaries]
    gaps = np.concatenate([np.asarray(s.gaps) for s in summaries if s.gaps])
    conf = np.concatenate([np.asarray(s.confidences) for s in summaries if s.confidences])

    print(f"sessions: {n}, audio {sum(durations) / 3600:.1f} h")
    print("lines per session:", spread([s.lines for s in summaries]))
    print("words per session:", spread(words), f"| total {sum(words)}")
    print("40-word segments if every word were the participant's:", sum(w // 40 for w in words))
    print("share of the audio inside lines:", spread([s.speech_seconds / d for s, d in pairs]))
    print("first line starts at (s):", spread([s.first_start for s in summaries]))
    print(
        "sessions with lines past the end of the audio:", sum(s.last_end > d + 1 for s, d in pairs)
    )
    print(
        f"overlapping lines: {sum(s.overlaps for s in summaries)}"
        f" in {sum(s.overlaps > 0 for s in summaries)} sessions"
    )
    print(
        f"pauses between lines (s): median {np.median(gaps):.2f},"
        f" share over 2 s {np.mean(gaps > 2):.2f}, share over 10 s {np.mean(gaps > 10):.2f}"
    )
    print(
        f"confidence per line: mean {conf.mean():.2f}, median {np.median(conf):.2f},"
        f" 5th percentile {np.percentile(conf, 5):.2f},"
        f" share below {LOW_CONFIDENCE} {np.mean(conf < LOW_CONFIDENCE):.2f},"
        f" lines without a value {sum(s.missing_confidence for s in summaries)}"
    )
    session_conf = [float(np.mean(s.confidences)) for s in summaries if s.confidences]
    print("mean confidence per session:", spread(session_conf))
    print(
        f"fillers (um, uh, ...): {sum(s.fillers for s in summaries)} in total,"
        f" sessions with any {sum(s.fillers > 0 for s in summaries)}"
    )
    print("lines with a bracketed tag:", sum(s.tags for s in summaries))
    print(f"interviewer phrases: number of sessions (of {n}) that contain each")
    for phrase in INTERVIEWER_PHRASES:
        print(f"  {sum(phrase in s.interviewer_phrases for s in summaries):4d}  {phrase}")
    print(
        "sessions with 3 or more different interviewer phrases:",
        sum(len(s.interviewer_phrases) >= 3 for s in summaries),
    )


if __name__ == "__main__":
    main()
