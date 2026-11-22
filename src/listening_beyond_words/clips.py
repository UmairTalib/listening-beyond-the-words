"""File helpers shared by the scripts that read the pinned segments and write clips.

The clip code itself (``views.py``) takes arrays. These few functions read the files
around it on OMNI: the pinned segments table and the key of a listening check. They
also make the listening copy of a clip, scaled to a common loudness, as the listening
checks wrote it.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from listening_beyond_words.runinfo import check_sha256
from listening_beyond_words.settings import Workspace
from listening_beyond_words.views import Window

LISTENING_RMS: Final = 0.1  # common loudness of the listening copies (-20 dBFS) ...
LISTENING_PEAK: Final = 0.99  # ... unless that would clip; x0 itself keeps the recording's level


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
