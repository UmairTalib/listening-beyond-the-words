"""File helpers shared by the scripts that read the pinned segments and write clips.

The clip code itself (``views.py``) takes arrays. These few functions read and write
the files around it on OMNI: a session's recording and transcript lines (through the
guard in ``sessions.py``), the pinned segments, cut words, labels, partitions,
word-check sample and view manifests, the key of a listening check and the FLAC files
of a view, which are read back only after a check against their manifest row. They
also make the listening copy of a clip, scaled to a common loudness, exactly as the
listening checks wrote it.
"""

import hashlib
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import soundfile as sf

from listening_beyond_words.partitions import N_FOLDS, SEEDS
from listening_beyond_words.runinfo import StepError, check_sha256, sha256_file
from listening_beyond_words.segments import GAP_SECONDS
from listening_beyond_words.sessions import session_files
from listening_beyond_words.settings import Workspace
from listening_beyond_words.transcripts import read_transcript
from listening_beyond_words.views import (
    PCM16_SCALE,
    SAMPLING_RATE,
    Window,
    check_segment_views,
    clip_peak,
    clip_rms,
)

LISTENING_RMS: Final = 0.1  # common loudness of the listening copies (-20 dBFS) ...
LISTENING_PEAK: Final = 0.99  # ... unless that would clip; x0 itself keeps the recording's level
GAP_SAMPLES: Final = round(GAP_SECONDS * SAMPLING_RATE)  # the 300 ms of silence between two parts
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


def _pinned_file(workspace: Workspace, pin: dict, name: str) -> Path:
    """A file of a pinned run, after checking it against the pinned SHA-256."""
    path = workspace.resolve(pin["run"]) / name
    check_sha256(path, pin["sha256"])
    return path


def load_cut_words(workspace: Workspace, config: dict) -> pd.DataFrame:
    """The words table of the pinned segments run: every aligned word, kept or not, without text.

    The pin covers ``segments.parquet`` only. The run's ``run.json`` lists the SHA-256 of
    every file it wrote, so this checks that it lists the pinned one for the segments and
    that ``words.parquet`` matches the one it lists.

    Args:
        workspace: The workspace.
        config: The data config, with the ``segments`` pin.

    Returns:
        One row per aligned word: ``session``, ``line``, ``word_index``, ``start``,
        ``end``, ``kept``, ``part``, ``cause`` and the cut rule's flags.

    Raises:
        ValueError: If the run did not finish, its ``run.json`` lists another SHA-256 for
            the segments or none for the words, or the words file does not match.
    """
    pin = config["segments"]
    run = workspace.resolve(pin["run"])
    built = json.loads((run / "run.json").read_text(encoding="utf-8"))
    outputs = built.get("outputs") or {}
    if built.get("status") != "done" or outputs.get("segments.parquet") != pin["sha256"]:
        raise ValueError("the segments run's run.json does not list the pinned segments as done")
    if "words.parquet" not in outputs:
        raise ValueError("the segments run's run.json lists no SHA-256 for words.parquet")
    path = run / "words.parquet"
    check_sha256(path, outputs["words.parquet"])
    return pd.read_parquet(path)


def read_lines(sessions_dir: Path, session_id: int) -> list[str]:
    """Read the text of a pilot session's transcript lines, as the alignment numbered them.

    Args:
        sessions_dir: Folder with the unpacked sessions, ``<workspace>/edaic/sessions``.
        session_id: A pilot session ID; every other ID is refused before any file opens.

    Returns:
        The text of each line, in the order of ``transcripts.read_transcript`` (by start
        time), so that a word's ``line`` is its index here.

    Raises:
        ValueError: If the ID is not a pilot session or the file is not an E-DAIC transcript.
    """
    _, transcript_path = session_files(sessions_dir, session_id)
    return [str(text) for text in read_transcript(transcript_path)["text"]]


def load_pinned_labels(workspace: Workspace, config: dict) -> pd.DataFrame:
    """The pinned labels: PHQ-8 class and gender of each pilot participant.

    Args:
        workspace: The workspace.
        config: The data config, with the ``labels`` pin.

    Returns:
        One row per participant, indexed by session ID, with the columns ``label`` (1 if
        the PHQ-8 total is 10 or more) and ``gender``.

    Raises:
        ValueError: If the file does not match its SHA-256.
    """
    labels = pd.read_parquet(_pinned_file(workspace, config["labels"], "labels.parquet"))
    return labels[["label", "gender"]]


def load_pinned_sample(workspace: Workspace, config: dict) -> pd.DataFrame:
    """The pinned word-check sample: the segments that the word check transcribes.

    Args:
        workspace: The workspace.
        config: The data config, with the ``wordcheck_sample`` pin.

    Returns:
        One row per drawn segment, with the columns ``segment`` and ``session``.

    Raises:
        ValueError: If the file does not match its SHA-256.
    """
    return pd.read_parquet(_pinned_file(workspace, config["wordcheck_sample"], "sample.parquet"))


def load_pinned_partitions(workspace: Workspace, config: dict) -> pd.DataFrame:
    """The pinned partitions: the fold of each labelled participant in each round.

    The pin covers ``partitions.parquet``. Its run's ``run.json`` must name the pinned
    labels' SHA-256 as its input (as ``load_segments`` checks its inputs), the file must
    hold the rounds of ``partitions.SEEDS`` and in each round the folds 1 to
    ``partitions.N_FOLDS``, and each participant of the pinned labels must lie in
    exactly one fold of each round, with no other participant.

    Args:
        workspace: The workspace.
        config: The data config, with the ``partitions`` and ``labels`` pins.

    Returns:
        One row per participant and round, with the columns ``session``, ``round`` (1 for
        seed 101, 2 for seed 102) and ``fold`` (from 1), sorted by round and session.

    Raises:
        ValueError: If a file does not match its SHA-256, the partitions were made from
            other labels, a round or a fold is missing or extra, or a labelled
            participant is missing from a round or lies in it twice, or one without a
            label lies in it.
    """
    path = _pinned_file(workspace, config["partitions"], "partitions.parquet")
    built = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
    named = ((built.get("inputs") or {}).get("labels") or {}).get("sha256")
    if named != config["labels"]["sha256"]:
        raise ValueError("the pinned partitions were not made from the pinned labels")
    table = pd.read_parquet(path)
    columns = ["session", "round", "fold"]
    if not set(columns) <= set(table.columns):
        raise ValueError("the pinned partitions do not have the columns session, round and fold")
    rounds = set(range(1, len(SEEDS) + 1))
    found = {int(number) for number in table["round"]}
    if found != rounds:
        raise ValueError(f"the pinned partitions hold rounds {sorted(found)}, not {sorted(rounds)}")
    labelled = {int(session) for session in load_pinned_labels(workspace, config).index}
    for number, part in table.groupby("round", sort=True):
        sessions = [int(session) for session in part["session"]]
        missing = len(labelled - set(sessions))
        twice = len(sessions) - len(set(sessions))
        unlabelled = len(set(sessions) - labelled)
        if missing or twice or unlabelled:
            raise ValueError(
                f"round {number}: {missing} labelled participants missing, {twice} extra rows"
                f" of participants already in it, {unlabelled} participants without a label;"
                " each labelled participant must lie in exactly one fold"
            )
        folds = {int(fold) for fold in part["fold"]}
        if folds != set(range(1, N_FOLDS + 1)):
            raise ValueError(f"round {number} holds folds {sorted(folds)}, not 1 to {N_FOLDS}")
    return table[columns].sort_values(["round", "session"], ignore_index=True)


def load_view_manifest(workspace: Workspace, config: dict, view: str) -> pd.DataFrame:
    """The manifest of a pinned clip version (``view_runs``).

    Args:
        workspace: The workspace.
        config: The data config, with the ``view_runs`` pins.
        view: The clip version, such as ``x0``.

    Returns:
        One row per clip, with the columns of ``MANIFEST_COLUMNS``.

    Raises:
        ValueError: If the file does not match its SHA-256.
    """
    return pd.read_parquet(_pinned_file(workspace, config["view_runs"][view], "manifest.parquet"))


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


@dataclass(frozen=True, slots=True)
class ViewClip:
    """One stored clip of a view, read back and checked against its manifest row.

    Attributes:
        samples: The stored 16-bit samples, int16 of shape (samples,).
        part_starts: The first sample of each part.
        part_ends: The sample just after each part.
    """

    samples: np.ndarray
    part_starts: tuple[int, ...]
    part_ends: tuple[int, ...]

    @property
    def audio(self) -> np.ndarray:
        """The clip as both speech recognisers read it: float32, each sample n as n / 32768."""
        return (self.samples / PCM16_SCALE).astype(np.float32)


def read_view_clip(run_dir: Path, row: Any) -> ViewClip:
    """Read one stored clip of a view, after checking it against its manifest row.

    The checks are those with which the merge of x1 reads its files again
    (``scripts/merge_revoiced.py``): the file's SHA-256; 16 kHz, mono, 16-bit; the
    number of samples and their SHA-256; then ``views.check_segment_views`` on the
    clip's own part bounds, so that every gap is silent and every part starts and ends
    at exactly 0.

    Args:
        run_dir: The view's run folder, such as ``<workspace>/views/x1/run-12576406``.
        row: One row of its manifest (``MANIFEST_COLUMNS``), such as from
            ``manifest.itertuples(index=False)``; its fields are read by name.

    Returns:
        The stored samples and the part bounds.

    Raises:
        StepError: If the file is missing or any check fails. The message names the
            segment and the check, never a path or a sample value.
    """
    segment = str(row.segment)

    def refused(what: str) -> StepError:
        return StepError(f"segment {segment}: {what}", segment=segment)

    path = run_dir / str(row.file)
    if not path.is_file():
        raise refused("its file is missing")
    if sha256_file(path) != row.sha256:
        raise refused("its file does not match the SHA-256 of its manifest row")
    info = sf.info(path)
    if (info.samplerate, info.channels, info.subtype) != (SAMPLING_RATE, 1, "PCM_16"):
        raise refused(
            f"its file is {info.samplerate} Hz, {info.channels} channel(s), {info.subtype};"
            f" expected {SAMPLING_RATE} Hz mono 16-bit PCM"
        )
    samples, _ = sf.read(path, dtype="int16")
    if len(samples) != int(row.samples):
        raise refused(f"its file holds {len(samples)} samples, its manifest row {int(row.samples)}")
    if audio_sha256(samples) != row.audio_sha256:
        raise refused("its samples do not match the SHA-256 of its manifest row")
    starts = tuple(int(start) for start in row.part_starts)
    ends = tuple(int(end) for end in row.part_ends)
    reason = None
    try:
        check_segment_views({"clip": samples}, starts, ends, gap=GAP_SAMPLES)
    except ValueError as error:  # its messages name rules and sample counts only
        reason = str(error)
    if reason is not None:  # raised here, so that nothing chains to the check's error
        raise refused(f"its parts and gaps do not fit its manifest row ({reason})")
    return ViewClip(np.asarray(samples, dtype=np.int16), starts, ends)
