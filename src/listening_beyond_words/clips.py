"""File helpers shared by the scripts that read the pinned segments and write clips.

The clip code itself (``views.py``) takes arrays. These few functions read and write
the files around it on OMNI: a session's recording (through the guard in
``sessions.py``), the pinned segments table, the key of a listening check and the
FLAC file of a view. They also make the listening copy of a clip, scaled to a common
loudness, exactly as the listening checks wrote it.
"""

import hashlib
import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import soundfile as sf

from listening_beyond_words.runinfo import check_sha256, sha256_file
from listening_beyond_words.sessions import session_files
from listening_beyond_words.settings import Workspace
from listening_beyond_words.views import PCM16_SCALE, SAMPLING_RATE, Window, clip_peak, clip_rms

LISTENING_RMS: Final = 0.1  # common loudness of the listening copies (-20 dBFS) ...
LISTENING_PEAK: Final = 0.99  # ... unless that would clip; x0 itself keeps the recording's level
# One row per stored clip of a view, in manifest.parquet. ``file`` is relative to the run
# folder; ``rms`` and ``peak`` are measured on the stored samples; a part runs from its
# start sample up to, not including, its end sample.
MANIFEST_COLUMNS: Final = (
    "segment",
    "session",
    "file",
    "samples",
    "rms",
    "peak",
    "sha256",
    "audio_sha256",
    "part_starts",
    "part_ends",
    "peak_limited",
    "gain_db",
)


def read_session(sessions_dir: Path, session_id: int) -> np.ndarray:
    """Read the recording of one pilot session as float32.

    Args:
        sessions_dir: Folder with the unpacked sessions, ``<workspace>/edaic/sessions``.
        session_id: A pilot session ID; every other ID is refused before any file opens.

    Returns:
        The recording as float32 in [-1, 1), shape (samples,): each 16-bit sample n as
        n / 32768.

    Raises:
        ValueError: If the ID is not a pilot session, or the file is not 16 kHz mono
            16-bit PCM.
    """
    audio_path, _ = session_files(sessions_dir, session_id)
    info = sf.info(audio_path)
    if (info.samplerate, info.channels, info.subtype) != (SAMPLING_RATE, 1, "PCM_16"):
        raise ValueError(
            f"session {session_id}: expected {SAMPLING_RATE} Hz mono 16-bit PCM, got"
            f" {info.samplerate} Hz, {info.channels} channel(s), {info.subtype}"
        )
    audio, _ = sf.read(audio_path, dtype="float32")
    return np.asarray(audio, dtype=np.float32)


def load_segments(workspace: Workspace, config: dict) -> pd.DataFrame:
    """The pinned segments, after checking that they come from the pinned inputs and settings."""
    pin = config["segments"]
    path = workspace.resolve(pin["run"]) / "segments.parquet"
    check_sha256(path, pin["sha256"])
    built = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
    for name in ("word_times", "participant_lines"):
        if built["inputs"][name]["sha256"] != config[name]["sha256"]:
            raise ValueError(f"the pinned segments were not built from the pinned {name}")
    for name in ("daicwoz_cut", "x0_cut"):
        # run.json is JSON, where the session keys of the settings became strings.
        if built["settings"][name] != json.loads(json.dumps(config[name])):
            raise ValueError(f"the pinned segments were built with other {name} settings")
    return pd.read_parquet(path).sort_values("segment", ignore_index=True)


def listening_gain(clip: np.ndarray) -> float:
    """Gain that brings a clip to the common listening loudness without clipping."""
    rms = float(np.sqrt(np.mean(np.square(clip, dtype=np.float64))))
    peak = float(np.max(np.abs(clip)))
    if rms == 0.0 or peak == 0.0:
        return 1.0
    return min(LISTENING_RMS / rms, LISTENING_PEAK / peak)


def listening_copy(clip: np.ndarray) -> tuple[np.ndarray, float]:
    """The listening copy of a float32 clip, as ``scripts/listening_check.py`` writes it.

    Args:
        clip: A clip as float32, such as an original clip from ``views.cut_original``.

    Returns:
        The clip times the listening gain, in float32, and the gain.
    """
    gain = listening_gain(clip)
    return clip * np.float32(gain), gain


def format_windows(windows: Sequence[Window]) -> str:
    """The windows of a clip as a listening key stores them: ``low-high`` in seconds, to 0.1 ms."""
    return "; ".join(f"{low:.4f}-{high:.4f}" for low, high in windows)


def read_key(path: Path, sha256: str) -> pd.DataFrame:
    """Read the key of a listening check, after checking it against its pinned SHA-256.

    Args:
        path: The check's ``key.tsv``.
        sha256: Its pinned SHA-256.

    Returns:
        One row per clip: the columns of the key, with ``windows`` as a list of
        (start, end) windows in seconds, and the column ``session``.

    Raises:
        ValueError: If the file does not match its SHA-256.
    """
    check_sha256(path, sha256)
    key = pd.read_csv(path, sep="\t", dtype={"clip": str, "segment": str, "windows": str})

    def parse(text: Any) -> list[Window]:
        windows = []
        for window in str(text).split(";"):
            low, high = (float(t) for t in window.strip().split("-"))
            windows.append((low, high))
        return windows

    key["windows"] = pd.Series(
        [parse(text) for text in key["windows"]], index=key.index, dtype=object
    )
    key["session"] = [int(segment.split("_")[0]) for segment in key["segment"]]
    return key


def listening_pcm16(clip: np.ndarray, rate: int = SAMPLING_RATE) -> np.ndarray:
    """The 16-bit samples of a clip's listening copy, as the listening check's WAV file holds them.

    The copy is written to a WAV file in memory exactly as ``scripts/listening_check.py``
    writes it, and read back. The WAV writer turns floats into 16-bit numbers in its own
    way (it rounds down), so this never uses ``views.to_pcm16``.

    Args:
        clip: The original clip (x0) as float32.
        rate: Sampling rate in Hz.

    Returns:
        The samples as int16.
    """
    copy, _ = listening_copy(clip)
    buffer = io.BytesIO()
    sf.write(buffer, copy, rate, subtype="PCM_16", format="WAV")
    buffer.seek(0)
    samples, _ = sf.read(buffer, dtype="int16")
    return np.asarray(samples, dtype=np.int16)


def audio_sha256(samples: np.ndarray) -> str:
    """The SHA-256 of 16-bit samples as little-endian bytes, independent of the file format."""
    return hashlib.sha256(samples.astype("<i2").tobytes()).hexdigest()


def write_view(path: Path, samples: np.ndarray, rate: int = SAMPLING_RATE) -> dict[str, Any]:
    """Write one clip of a view as a 16-bit FLAC file and check it by reading it back.

    Args:
        path: The new file; it must not exist yet.
        samples: The clip as int16, from ``views.to_pcm16``.
        rate: Sampling rate in Hz.

    Returns:
        The manifest values of the file: ``samples``, ``rms`` and ``peak`` (of the stored
        samples, in float64), ``sha256`` (of the file) and ``audio_sha256`` (of the samples).

    Raises:
        ValueError: If the samples are not one-dimensional int16, the file exists, or
            the file does not read back as 16-bit mono at ``rate`` with the same samples.
    """
    if samples.dtype != np.int16 or samples.ndim != 1:
        raise ValueError(f"expected int16 samples of shape (samples,), got {samples.dtype}")
    if path.exists():
        raise ValueError(f"{path.name} exists already; a view never overwrites a file")
    sf.write(path, samples, rate, subtype="PCM_16", format="FLAC")
    info = sf.info(path)
    back, _ = sf.read(path, dtype="int16")
    if (info.samplerate, info.channels, info.subtype) != (rate, 1, "PCM_16") or not (
        np.array_equal(back, samples)
    ):
        raise ValueError(f"{path.name} does not read back as written")
    values = samples.astype(np.float64) / PCM16_SCALE
    return {
        "samples": len(samples),
        "rms": clip_rms(values),
        "peak": clip_peak(values),
        "sha256": sha256_file(path),
        "audio_sha256": audio_sha256(samples),
    }
