"""Finding the interviewer's voice inside the participant's transcript lines.

The E-DAIC transcripts hold the participant's lines, but the virtual interviewer
(Ellie) sometimes slips into them. Ellie always has the same voice, so a
speaker-verification model can tell her apart. The speech between the
participant's lines, which is mostly Ellie, gives her voice profile, and the
participant's lines give theirs. Each short window of speech is then scored by
which of the two profiles it is closer to. ``line_decisions`` turns the scores
into the pilot's rule for which participant lines are kept.

Spans are ``(start, end)`` pairs in seconds.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

Span = tuple[float, float]

SAMPLING_RATE: Final = 16_000
WINDOW_SECONDS: Final = 1.5
HOP_SECONDS: Final = 0.75
MIN_WINDOW_SECONDS: Final = 0.75
MIN_GAP_SECONDS: Final = 0.5
MIN_PROFILE_WINDOWS: Final = 10
DROP_SHARE: Final = 0.5
UNCLEAR_MIN_SECONDS: Final = 1.5
VAD_OPTIONS: Final = {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 300,
    "speech_pad_ms": 100,
}


def merge(spans: Iterable[Span]) -> list[Span]:
    """Return the union of spans: sorted, with overlapping spans joined.

    Args:
        spans: Spans in any order; empty or reversed spans are dropped.

    Returns:
        Sorted spans that do not overlap.
    """
    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def gaps(lines: Iterable[Span], duration: float, min_length: float = MIN_GAP_SECONDS) -> list[Span]:
    """Return the parts of a session that no line covers.

    Args:
        lines: The transcript lines; they may overlap.
        duration: Length of the session audio in seconds.
        min_length: Shorter gaps are dropped.

    Returns:
        The gaps, sorted, including the time before the first line and after the last.
    """
    result: list[Span] = []
    cursor = 0.0
    for start, end in merge(lines):
        if start > cursor and start - cursor >= min_length:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if duration > cursor and duration - cursor >= min_length:
        result.append((cursor, duration))
    return result


def intersect(spans: Iterable[Span], regions: Iterable[Span]) -> list[Span]:
    """Return the parts of spans that lie inside regions.

    Args:
        spans: Spans in any order.
        regions: Regions in any order, for example the detected speech.

    Returns:
        The sorted, non-overlapping parts that both cover.
    """
    a, b = merge(spans), merge(regions)
    result: list[Span] = []
    i = j = 0
    while i < len(a) and j < len(b):
        start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if start < end:
            result.append((start, end))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return result


def windows(
    span: Span,
    length: float = WINDOW_SECONDS,
    hop: float = HOP_SECONDS,
    min_length: float = MIN_WINDOW_SECONDS,
) -> list[Span]:
    """Cut a span into windows of a fixed length.

    Args:
        span: The span to cut.
        length: Window length in seconds.
        hop: Step between window starts in seconds.
        min_length: A span shorter than this gives no window; a span between
            ``min_length`` and ``length`` gives one window that covers it.

    Returns:
        The windows. The last one ends at the span's end, so no tail is lost.
    """
    start, end = span
    if end - start < min_length:
        return []
    if end - start <= length:
        return [(start, end)]
    result: list[Span] = []
    position = start
    while position + length <= end + 1e-9:
        result.append((position, position + length))
        position += hop
    if result[-1][1] < end - 1e-9:
        result.append((end - length, end))
    return result


def unit(vectors: np.ndarray) -> np.ndarray:
    """Scale each row (or a single vector) to length 1.

    Args:
        vectors: One vector, or one vector per row.

    Returns:
        The vectors divided by their lengths.
    """
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def centroid(embeddings: np.ndarray) -> np.ndarray:
    """Return the unit-length mean direction of unit-length embeddings.

    Args:
        embeddings: One embedding per row.

    Returns:
        The voice profile.
    """
    return unit(unit(embeddings).mean(axis=0))


def _closer_half(embeddings: np.ndarray, reference: np.ndarray, nearest: bool) -> np.ndarray:
    similarity = unit(embeddings) @ reference
    median = np.median(similarity)
    return embeddings[similarity >= median] if nearest else embeddings[similarity <= median]


def global_profile(gap_embeddings: np.ndarray) -> np.ndarray:
    """Return the interviewer's voice profile over all sessions.

    The gaps between participant lines hold mostly the interviewer, but also some
    participant speech. The profile is the mean of the half of the gap windows
    that lies closest to the plain mean, so a minority of other voices barely moves it.

    Args:
        gap_embeddings: Embeddings of gap windows from all sessions.

    Returns:
        The profile, a unit vector.
    """
    return centroid(_closer_half(gap_embeddings, centroid(gap_embeddings), nearest=True))


def session_profiles(
    line_embeddings: np.ndarray,
    gap_embeddings: np.ndarray,
    interviewer: np.ndarray,
    min_windows: int = MIN_PROFILE_WINDOWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one session's interviewer and participant voice profiles.

    The interviewer profile is the mean of the session's gap windows closest to the
    global profile (or the global profile if the session has too few gap windows),
    which adapts it to the session's room and microphone. The participant profile is
    the mean of the half of the line windows that is least like that interviewer
    profile, so interviewer speech inside a minority of lines does not move it.

    Args:
        line_embeddings: Embeddings of the session's line windows.
        gap_embeddings: Embeddings of the session's gap windows.
        interviewer: The global interviewer profile.
        min_windows: Fewest gap windows for a session's own interviewer profile.

    Returns:
        The interviewer profile and the participant profile.
    """
    if len(gap_embeddings) >= min_windows:
        interviewer = centroid(_closer_half(gap_embeddings, interviewer, nearest=True))
    participant = centroid(_closer_half(line_embeddings, interviewer, nearest=False))
    return interviewer, participant


def interviewer_scores(
    embeddings: np.ndarray, interviewer: np.ndarray, participant: np.ndarray
) -> np.ndarray:
    """Score windows: above 0 means closer to the interviewer than to the participant.

    Args:
        embeddings: One embedding per window.
        interviewer: The session's interviewer profile.
        participant: The session's participant profile.

    Returns:
        Cosine similarity to the interviewer minus cosine similarity to the participant.
    """
    u = unit(embeddings)
    return np.asarray(u @ interviewer - u @ participant)


def line_decisions(
    lines: pd.DataFrame,
    windows: pd.DataFrame,
    drop_share: float = DROP_SHARE,
    unclear_min_seconds: float | None = UNCLEAR_MIN_SECONDS,
) -> pd.DataFrame:
    """Decide which participant lines the pilot keeps.

    A line is dropped if at least ``drop_share`` of its speech windows score above 0
    (closer to the interviewer than to the participant), if it contains a known
    interviewer phrase, or if it lasts at least ``unclear_min_seconds`` but has no
    speech window at all: no stretch of clearly detected speech, so neither the
    model nor a listener has judged it. Shorter lines without a window, such as a
    quick "yeah", are kept.

    Args:
        lines: One row per transcript line, with the columns ``session``, ``line``,
            ``start``, ``end`` and ``phrase`` (True if the line contains an
            interviewer phrase).
        windows: Scored windows as written by ``scripts/flag_interviewer.py``, with
            the columns ``session``, ``kind``, ``line`` and ``score``.
        drop_share: Share of a line's windows above 0 from which it is dropped.
        unclear_min_seconds: Length from which a line without a speech window is
            dropped; None keeps all such lines.

    Returns:
        ``lines`` with three more columns: ``share_above_0`` (NaN for a line without
        a speech window), ``unclear`` and ``drop``.
    """
    scored = windows[(windows["kind"] == "line") & windows["score"].notna()]
    share = (
        (scored["score"] > 0)
        .groupby([scored["session"], scored["line"]])
        .mean()
        .rename("share_above_0")
        .reset_index()
    )
    decided = lines.merge(share, on=["session", "line"], how="left", validate="one_to_one")
    long_enough = (decided["end"] - decided["start"]) >= (
        unclear_min_seconds if unclear_min_seconds is not None else float("inf")
    )
    decided["unclear"] = decided["share_above_0"].isna() & long_enough
    decided["drop"] = (
        (decided["share_above_0"] >= drop_share)
        | decided["phrase"].astype(bool)
        | decided["unclear"]
    )
    return decided


def speech_regions(audio: np.ndarray, sampling_rate: int = SAMPLING_RATE) -> list[Span]:
    """Find the speech in a recording with the Silero detector bundled with faster-whisper.

    Args:
        audio: Mono audio samples in [-1, 1].
        sampling_rate: Sampling rate of ``audio``; the detector expects 16 kHz.

    Returns:
        The speech regions.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    chunks = get_speech_timestamps(
        audio, vad_options=VadOptions(**VAD_OPTIONS), sampling_rate=sampling_rate
    )
    return [(c["start"] / sampling_rate, c["end"] / sampling_rate) for c in chunks]


class VoiceEmbedder:
    """Speaker-verification embeddings (x-vectors) from the pinned WavLM model.

    Args:
        snapshot: Folder of the pinned ``microsoft/wavlm-base-plus-sv`` snapshot.
        device: Where the model runs.
    """

    def __init__(self, snapshot: Path, device: str = "cuda") -> None:
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.device = device
        self.extractor: Any = AutoFeatureExtractor.from_pretrained(snapshot)
        model: Any = WavLMForXVector.from_pretrained(snapshot)
        self.model = model.to(device).eval()

    def embed(self, clips: Sequence[np.ndarray], batch_size: int = 64) -> np.ndarray:
        """Return one unit-length embedding per clip.

        Args:
            clips: Mono 16 kHz clips; shorter clips in a batch are padded and masked.
            batch_size: Clips per forward pass.

        Returns:
            An array with one row per clip.
        """
        import torch

        rows = []
        for first in range(0, len(clips), batch_size):
            batch = [
                np.asarray(clip, dtype=np.float32) for clip in clips[first : first + batch_size]
            ]
            inputs = self.extractor(
                batch, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.inference_mode():
                rows.append(self.model(**inputs).embeddings.float().cpu().numpy())
        if not rows:
            return np.zeros((0, self.model.config.xvector_output_dim), dtype=np.float32)
        return unit(np.concatenate(rows))
