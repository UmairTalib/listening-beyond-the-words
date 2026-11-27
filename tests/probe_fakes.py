"""Stand-ins for the tests of the pooled-encoder probe: opensmile and a light workspace in code.

Two stand-ins for opensmile. ``PartExtractor`` gives each part the outputs set for its
length, as the voice check's tests do (tests/test_voicecheck.py), so that the tests of the
targets can count by hand (tests/test_probe.py). ``ProbeSmile`` gives values from a hash of
each stretch of audio, as the word check's stand-in does (tests/wordcheck_fakes.py), with
parts planted by their length: without a voiced frame, refused, or with a planted pitch
variability; the job's tests use it (tests/test_probe_encoder.py).

The workspace holds everything that the job reads, written in code, with no audio model:
the labels with their PHQ-8 totals; partitions of 2 rounds of 5 folds
(partitions.make_partitions, seeds 101 and 102); a segments run with its words table, whose
word times give every segment's parts; the manifests of x0, x2 and rev, with x0's clips
stored as 16-bit FLAC; and the cache runs of the three views, of width 4, with their check.
Synthetic data only, on the pilot IDs 300 to 339, which assert_only_aggregates_leave
watches. Planted: a participant whose parts have no voiced frame (UNVOICED_SESSION), so
that their segments have no pitch variability and their means of the voiced measures are
missing; a part under 960 samples (SHORT_SEGMENT) and a part that the stand-in refuses
(REFUSED_SEGMENT); a segment all of whose parts it refuses (UNMEASURED_SEGMENT), so that
the segment has none of the 88 measures, whereas the unvoiced participant's segments lack
some of them only; a segment whose pitch variability is PLANTED_TARGET; and a clip of two
chunks (LONG_SEGMENT).
"""

import contextlib
import copy
import hashlib
import io
import json
import shutil
import sys
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from listening_beyond_words.cache import CACHE_COLUMNS, NUMERICS, check_sample
from listening_beyond_words.clips import MANIFEST_COLUMNS, audio_sha256
from listening_beyond_words.partitions import make_partitions
from listening_beyond_words.views import SAMPLING_RATE
from listening_beyond_words.voicecheck import FUNCTIONAL_NAMES, PITCH_VARIABILITY
from listening_beyond_words.voxtral import FRAMES_PER_CHUNK, chunks
from x1_fakes import REPO, load_script

CONFIGS = REPO / "configs" / "pilot"
MODELS = CONFIGS / "models.yaml"
SESSIONS = tuple(range(300, 340))  # pilot IDs, 16 positive and 24 negative participants
VIEWS = ("x0", "x2", "rev")
WIDTH = 4  # the caches' width: 8 features per segment
JOB = "991"
CHECKOUT = {
    "commit": "e" * 40,
    "dirty": False,
    "describe": "v0.1.1",
    "tags": ["v0.1.1"],
    "release_tags": ["v0.1.0", "v0.1.1"],
}
INSTALLED = "0.1.1"
RESAMPLES = 200  # resamples of the fast tests; the real config has 10,000
GAP = 4_800  # the 300 ms of silence between two parts
WORD_SECONDS = (0.03, 0.08)  # the range of a word's length in the word times ...
WORD_GAP = (0.0, 0.03)  # ... and of the gap after it, within a part
LONG_WORD_SECONDS = (1.0, 1.4)  # in LONG_SEGMENT, whose clip is longer than 30 s
MARGIN_SECONDS = 0.1  # a part's clip reaches this far past its first and last word
SHORT_SAMPLES = 900  # SHORT_SEGMENT's third part: under the 960 samples that opensmile needs
UNVOICED_SESSION = 305
SHORT_SEGMENT = "301_000"
REFUSED_SEGMENT = "302_001"  # the stand-in refuses its second part
UNMEASURED_SEGMENT = "307_000"  # the stand-in refuses every part of it
PLANTED_SEGMENT = "303_000"  # every part of it has the planted pitch variability
LONG_SEGMENT = "304_001"
PLANTED_TARGET = 12.3456789
CACHE_JOBS = {"x0": "810", "x2": "811", "rev": "812"}
CACHE_CHECK = "derived/cache_check/run-813"
ENCODER_CACHE = {
    "cpu_threads": 4,
    "verify": {"segments_per_view": 2, "seed": 108, "lm_watch_clips": 2},
}


# The stand-ins for opensmile -------------------------------------------------------------------


@dataclass
class FakePart:
    """A part's outputs, as opensmile would give them."""

    functionals: pd.Series
    pitch: np.ndarray


def frames_of(samples: int) -> int:
    """The frames of opensmile's pitch track for a part of this many samples."""
    return (samples - 960) // 160 + 2


def fake_part(samples: int, voiced: float, *, value: float = 0.0) -> FakePart:
    """A part with ``voiced`` seconds of pitch in one stretch and every functional ``value``.

    The voiced frames come first; the track then ends unvoiced, and opensmile divides the
    one stretch by 2 frames fewer than the track has.
    """
    count = frames_of(samples)
    pitch = np.zeros(count)
    pitch[: round(voiced * 100)] = 30.0
    functionals = dict.fromkeys(FUNCTIONAL_NAMES, value)
    stretches = 1 if voiced > 0 else 0
    functionals["VoicedSegmentsPerSec"] = stretches / ((count - 2) * 0.01)
    functionals["MeanVoicedSegmentLengthSec"] = voiced
    functionals["StddevVoicedSegmentLengthSec"] = 0.0
    return FakePart(pd.Series(functionals, dtype=float), pitch)


class PartExtractor:
    """opensmile's stand-in for hand counts: each part's outputs are set by its length.

    A length in ``refused`` gets a pitch track one frame short, which
    ``voicecheck.part_measures`` refuses with a ``StepError``.
    """

    def __init__(self, parts: dict[int, FakePart], refused: Collection[int] = ()) -> None:
        self.parts = parts
        self.refused = set(refused)
        self.pieces: list[np.ndarray] = []

    def functionals(self, part: np.ndarray) -> pd.Series:
        self.pieces.append(part.copy())
        return self.parts[len(part)].functionals

    def pitch(self, part: np.ndarray) -> np.ndarray:
        track = self.parts[len(part)].pitch
        return track[:-1] if len(part) in self.refused else track


class ProbeSmile:
    """opensmile's stand-in for the job: values from a hash of each stretch of audio.

    The pitch track holds 1 or 2 voiced stretches in its first 70% of frames, which
    VoicedSegmentsPerSec counts, divided by 2 frames fewer than the track has. A part whose
    length is in ``unvoiced`` has no voiced frame; one in ``refused`` lacks the pitch
    variability among its functionals, which ``voicecheck.part_measures`` refuses; one in
    ``planted`` has the pitch variability PLANTED_TARGET.
    """

    def __init__(
        self,
        unvoiced: Collection[int] = (),
        refused: Collection[int] = (),
        planted: Collection[int] = (),
    ) -> None:
        self.unvoiced, self.refused, self.planted = set(unvoiced), set(refused), set(planted)
        self.calls = 0

    @staticmethod
    def digest(part: np.ndarray) -> bytes:
        return hashlib.sha256(np.asarray(part, dtype=np.float64).tobytes()).digest()

    def stretches(self, part: np.ndarray) -> int:
        return 0 if len(part) in self.unvoiced else 1 + self.digest(part)[8] % 2

    def functionals(self, part: np.ndarray) -> pd.Series:
        self.calls += 1
        rng = np.random.default_rng(int.from_bytes(self.digest(part)[:8], "big"))
        values = dict(
            zip(FUNCTIONAL_NAMES, rng.normal(1.0, 0.3, len(FUNCTIONAL_NAMES)), strict=True)
        )
        values["VoicedSegmentsPerSec"] = self.stretches(part) / ((frames_of(len(part)) - 2) * 0.01)
        values["MeanVoicedSegmentLengthSec"] = 0.15 + 0.1 * rng.random()
        values["StddevVoicedSegmentLengthSec"] = 0.05 * rng.random()
        values["equivalentSoundLevel_dBp"] = -30.0 + rng.normal()
        if len(part) in self.planted:
            values[PITCH_VARIABILITY] = PLANTED_TARGET
        if len(part) in self.refused:
            del values[PITCH_VARIABILITY]
        return pd.Series(values, dtype=float)

    def pitch(self, part: np.ndarray) -> np.ndarray:
        frames = frames_of(len(part))
        track = np.zeros(frames)
        if len(part) in self.unvoiced:
            return track
        start = 0
        voiced = np.arange(int(frames * 0.7))
        for run in np.array_split(voiced, self.stretches(part)):
            track[start : start + len(run)] = 30.0
            start += len(run) + 2
        return track


# The workspace ---------------------------------------------------------------------------------


def label_of(session: int) -> tuple[int, str, int]:
    """Label, gender and PHQ-8 total: 8 positive and 12 negative participants of each gender."""
    index = session - SESSIONS[0]
    label = int(index % 5 in (0, 2))
    gender = "female" if (index // 5) % 2 == 0 else "male"
    total = 10 + index % 15 if label else index % 10
    return label, gender, total


def segments_of(session: int) -> list[str]:
    """A participant's segments: 2 or 3."""
    count = 3 if (session - SESSIONS[0]) % 3 == 0 else 2
    return [f"{session}_{index:03d}" for index in range(count)]


def words_per_part(segment: str) -> list[int]:
    """How the 40 words of a segment fall into its parts."""
    return [20, 19, 1] if segment == SHORT_SEGMENT else [20, 20]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_parquet(path: Path, table: pd.DataFrame, index: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=index)
    return sha256_of(path)


def write_json(path: Path, data: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return sha256_of(path)


def part_audio(samples: int, level: float, rng: np.random.Generator) -> np.ndarray:
    """A part's 16-bit samples: noise with 10 ms fades, starting and ending at exactly 0."""
    audio = rng.normal(0.0, level, samples)
    fade = min(160, samples // 2)
    ramp = np.linspace(0.0, 1.0, fade)
    audio[:fade] *= ramp
    audio[samples - fade :] *= ramp[::-1]
    audio[0] = audio[-1] = 0.0
    return np.clip(np.rint(audio), -32_768, 32_767).astype(np.int16)


@dataclass
class ProbeWorkspace:
    """The probe's synthetic workspace, its data config and its probe.yaml.

    Attributes:
        root: The workspace root.
        config: The data config (data.yaml).
        config_path: Where the job reads it.
        probe: The probe's settings (probe.yaml), with fewer resamples.
        probe_path: Where the job reads them.
        labels: The labels, by session: phq8_total, label and gender.
        partitions: The partitions: session, round and fold.
        lengths: The lengths in samples of the planted parts, by kind: unvoiced, refused,
            planted and short.
    """

    root: Path
    config: dict[str, Any]
    config_path: Path
    probe: dict[str, Any]
    probe_path: Path
    labels: pd.DataFrame
    partitions: pd.DataFrame
    lengths: dict[str, list[int]] = field(default_factory=dict)

    def write_config(self) -> None:
        self.config_path.write_text(yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8")

    def write_probe(self) -> None:
        self.probe_path.write_text(yaml.safe_dump(self.probe, sort_keys=False), encoding="utf-8")

    def copy_to(self, folder: Path) -> "ProbeWorkspace":
        """A copy of the workspace and its two configs in a new folder."""
        root = folder / "workspace"
        shutil.copytree(self.root, root)
        copied = ProbeWorkspace(
            root,
            copy.deepcopy(self.config),
            folder / "data.yaml",
            copy.deepcopy(self.probe),
            folder / "probe.yaml",
            self.labels.copy(),
            self.partitions.copy(),
            copy.deepcopy(self.lengths),
        )
        copied.write_config()
        copied.write_probe()
        return copied

    def smile(self) -> ProbeSmile:
        """The stand-in for opensmile, with this workspace's planted parts."""
        return ProbeSmile(
            self.lengths["unvoiced"], self.lengths["refused"], self.lengths["planted"]
        )

    def manifest(self, view: str) -> pd.DataFrame:
        return pd.read_parquet(
            self.root / self.config["view_runs"][view]["run"] / "manifest.parquet"
        )

    def cache_manifest(self, view: str) -> pd.DataFrame:
        return pd.read_parquet(
            self.root / self.config["cache_runs"][view]["run"] / "manifest.parquet"
        )

    def repin(self, name: str, path: Path) -> None:
        """Pin a changed file of data.yaml again, so that only its content counts."""
        self.config[name]["sha256"] = sha256_of(path)
        self.write_config()


def words_and_clips(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, list[int]]]:
    """The words table, x0's clips (rows with their samples) and the planted parts' lengths."""
    words: list[dict[str, Any]] = []
    clips: list[dict[str, Any]] = []
    planted: dict[str, list[int]] = {"unvoiced": [], "refused": [], "planted": [], "short": []}
    used: set[int] = set()  # every part's length is unique, so the stand-in tells them apart
    for session in SESSIONS:
        clock, line = 1.0, 0
        for segment in segments_of(session):
            seconds = LONG_WORD_SECONDS if segment == LONG_SEGMENT else WORD_SECONDS
            level = rng.uniform(1_000.0, 6_000.0)  # the segment's loudness varies
            pieces: list[np.ndarray] = []
            starts: list[int] = []
            ends: list[int] = []
            position = 0
            for part, count in enumerate(words_per_part(segment)):
                first = end = clock
                for index in range(count):
                    end = clock + rng.uniform(*seconds)
                    words.append(
                        {"session": session, "line": line, "word_index": index, "start": clock,
                         "end": end, "kept": True, "part": line, "cause": None}
                    )  # fmt: skip
                    clock = end + rng.uniform(*WORD_GAP)
                words.append(  # a dropped word after the part, on its line
                    {"session": session, "line": line, "word_index": count, "start": clock,
                     "end": clock + 0.05, "kept": False, "part": -1, "cause": "a stand-in"}
                )  # fmt: skip
                clock += 2.0
                line += 1
                if segment == SHORT_SEGMENT and part == 2:
                    samples = SHORT_SAMPLES
                    planted["short"].append(samples)
                else:
                    samples = round((end - first + 2 * MARGIN_SECONDS) * SAMPLING_RATE)
                    while samples in used:
                        samples += 1
                used.add(samples)
                if session == UNVOICED_SESSION:
                    planted["unvoiced"].append(samples)
                if (segment == REFUSED_SEGMENT and part == 1) or segment == UNMEASURED_SEGMENT:
                    planted["refused"].append(samples)
                if segment == PLANTED_SEGMENT:
                    planted["planted"].append(samples)
                if part:
                    pieces.append(np.zeros(GAP, dtype=np.int16))
                    position += GAP
                starts.append(position)
                pieces.append(part_audio(samples, level, rng))
                position += samples
                ends.append(position)
            clips.append(
                {"segment": segment, "session": session, "samples": np.concatenate(pieces),
                 "part_starts": starts, "part_ends": ends}
            )  # fmt: skip
        for index in range(7):  # words left over at the session's end: not a segment
            words.append(
                {"session": session, "line": line, "word_index": index, "start": clock,
                 "end": clock + 0.05, "kept": True, "part": line, "cause": None}
            )  # fmt: skip
            clock += 0.06
    return pd.DataFrame(words), clips, planted


def view_manifest(root: Path, view: str, clips: list[dict[str, Any]]) -> pd.DataFrame:
    """A view's manifest; x0's clips are written as FLAC, the other views' files are never read."""
    rows = []
    for number, clip in enumerate(clips):
        samples = clip["samples"]
        file = f"{clip['session']}/{clip['segment']}.flac"
        values = samples.astype(np.float64) / 32_768
        if view == "x0":
            path = root / "views" / view / "run-1" / file
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, samples, SAMPLING_RATE, subtype="PCM_16", format="FLAC")
            sha256, audio = sha256_of(path), audio_sha256(samples)
        else:
            sha256 = f"{VIEWS.index(view)}{number:063x}"
            audio = f"{VIEWS.index(view) + 5}{number:063x}"
        rows.append(
            {
                "segment": clip["segment"],
                "session": clip["session"],
                "file": file,
                "samples": len(samples),
                "rms": float(np.sqrt(np.mean(values * values))),
                "peak": float(np.max(np.abs(values))),
                "sha256": sha256,
                "audio_sha256": audio,
                "part_starts": clip["part_starts"],
                "part_ends": clip["part_ends"],
                "peak_limited": False,
                "gain_db": 0.0,
            }
        )
    return pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))


def npy(output: np.ndarray) -> bytes:
    """An encoder output as the bytes of its .npy file, as cache.to_bytes writes them."""
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(output, dtype="<f2"), allow_pickle=False)
    return buffer.getvalue()


def cache_run(
    root: Path, view: str, clips: pd.DataFrame, view_pin: dict[str, str], seed: int
) -> str:
    """A cache run of a view: one random fp16 output per clip, its manifest and run.json."""
    folder = root / "cache" / view / f"run-{CACHE_JOBS[view]}"
    rng = np.random.default_rng(seed)
    rows = []
    for row in clips.itertuples(index=False):
        count = chunks(int(row.samples))
        output = rng.normal(0.0, 1.0, (count, FRAMES_PER_CHUNK, WIDTH)).astype(np.float16)
        data = npy(output)
        file = f"{row.session}/{row.segment}.npy"
        (folder / file).parent.mkdir(parents=True, exist_ok=True)
        (folder / file).write_bytes(data)
        rows.append(
            {
                "segment": row.segment,
                "session": int(row.session),
                "file": file,
                "chunks": count,
                "samples": int(row.samples),
                "sha256": hashlib.sha256(data).hexdigest(),
                "audio_sha256": row.audio_sha256,
                "features_sha256": "f" * 64,
                "max_abs": float(np.abs(output.astype(np.float32)).max()),
            }
        )
    manifest = pd.DataFrame(rows, columns=list(CACHE_COLUMNS))
    sha256 = write_parquet(folder / "manifest.parquet", manifest)
    pins = yaml.safe_load(MODELS.read_text(encoding="utf-8"))["voxtral"]
    write_json(
        folder / "run.json",
        {
            "step": "build_caches",
            "mode": {"name": "view", "view": view},
            "status": "done",
            "inputs": {
                "view": view_pin,
                "voxtral": {"repo_id": pins["repo_id"], "revision": pins["revision"], "blobs": {}},
            },
            "settings": {
                "encoder_cache": ENCODER_CACHE,
                "numerics": {**NUMERICS, "torch_threads": ENCODER_CACHE["cpu_threads"]},
                "width": WIDTH,
            },
            "outputs": {"manifest.parquet": sha256},
        },
    )
    return sha256


def cache_check(root: Path, manifests: dict[str, str]) -> None:
    """The check of the three cache runs: every sampled output equal, nothing flagged."""
    runs = {view: f"cache/{view}/run-{CACHE_JOBS[view]}" for view in VIEWS}
    results = {}
    for view, run in runs.items():
        manifest = pd.read_parquet(root / run / "manifest.parquet")
        size = len(
            check_sample(
                manifest,
                view,
                ENCODER_CACHE["verify"]["segments_per_view"],
                ENCODER_CACHE["verify"]["seed"],
            )
        )
        results[run] = {
            "view": view,
            "manifest_sha256": manifests[view],
            "segments": size,
            "features_equal": size,
            "outputs_equal": size,
            "lm_watch": {"clips": 2, "parts": 8, "flagged": [], "largest": []},
        }
    write_json(
        root / CACHE_CHECK / "run.json",
        {
            "step": "build_caches",
            "mode": {"name": "verify", "runs": list(runs.values())},
            "status": "done",
            "settings": {
                "encoder_cache": ENCODER_CACHE,
                "numerics": {**NUMERICS, "torch_threads": ENCODER_CACHE["cpu_threads"]},
            },
            "results": {"runs": results},
        },
    )


def make_probe_workspace(folder: Path) -> ProbeWorkspace:
    """Write every input of the probe job."""
    root = folder / "workspace"
    rng = np.random.default_rng(7)
    labels = pd.DataFrame(
        [label_of(session) for session in SESSIONS],
        columns=["label", "gender", "phq8_total"],
        index=pd.Index(list(SESSIONS), name="session"),
    )[["phq8_total", "label", "gender"]]
    labels_sha = write_parquet(
        root / "derived" / "labels" / "run-1" / "labels.parquet", labels, True
    )
    partitions = make_partitions(labels[["label", "gender"]])
    partitions_run = root / "derived" / "partitions" / "run-1"
    partitions_sha = write_parquet(partitions_run / "partitions.parquet", partitions)
    write_json(
        partitions_run / "run.json",
        {"status": "done", "inputs": {"labels": {"sha256": labels_sha}}},
    )
    words, clips, planted = words_and_clips(rng)
    segments_run = root / "derived" / "segments" / "run-1"
    segments = pd.DataFrame(
        {
            "segment": [clip["segment"] for clip in clips],
            "session": [clip["session"] for clip in clips],
        }
    )
    segments_sha = write_parquet(segments_run / "segments.parquet", segments)
    words_sha = write_parquet(segments_run / "words.parquet", words)
    write_json(
        segments_run / "run.json",
        {
            "status": "done",
            "outputs": {"segments.parquet": segments_sha, "words.parquet": words_sha},
        },
    )
    view_runs: dict[str, dict[str, str]] = {}
    cache_runs: dict[str, dict[str, str]] = {}
    manifests: dict[str, str] = {}
    for number, view in enumerate(VIEWS):
        manifest = view_manifest(root, view, clips)
        view_runs[view] = {
            "run": f"views/{view}/run-1",
            "sha256": write_parquet(root / "views" / view / "run-1" / "manifest.parquet", manifest),
        }
        manifests[view] = cache_run(root, view, manifest, view_runs[view], seed=20 + number)
        cache_runs[view] = {
            "run": f"cache/{view}/run-{CACHE_JOBS[view]}",
            "sha256": manifests[view],
            "verified_by": CACHE_CHECK,
        }
    cache_check(root, manifests)
    config = {
        "labels": {"run": "derived/labels/run-1", "sha256": labels_sha},
        "partitions": {"run": "derived/partitions/run-1", "sha256": partitions_sha},
        "segments": {"run": "derived/segments/run-1", "sha256": segments_sha},
        "views": {"peak_limit": 0.99, "flac_bits": 16},
        "word_check": {"voice": {"level_rms": 0.05, "min_part_samples": 960}},
        "view_runs": view_runs,
        "encoder_cache": ENCODER_CACHE,
        "cache_runs": cache_runs,
    }
    probe = yaml.safe_load((CONFIGS / "probe.yaml").read_text(encoding="utf-8"))
    probe["bootstrap"]["resamples"] = RESAMPLES
    workspace = ProbeWorkspace(
        root,
        config,
        folder / "data.yaml",
        probe,
        folder / "probe.yaml",
        labels,
        partitions,
        planted,
    )
    workspace.write_config()
    workspace.write_probe()
    return workspace


# Running the job ---------------------------------------------------------------------------------


@dataclass
class Ended:
    """A run of the probe job that ended, and what it wrote to its two streams."""

    out: str
    err: str
    ended: str  # "done" or "failed"
    folder: Path

    @property
    def info(self) -> dict[str, Any]:
        return json.loads((self.folder / "run.json").read_text(encoding="utf-8"))

    @property
    def streams(self) -> str:
        return self.out + self.err

    def table(self) -> pd.DataFrame:
        """probe.csv as written: every cell as text."""
        return pd.read_csv(self.folder / "probe.csv", dtype=str, keep_default_na=False)


def run_probe(
    workspace: ProbeWorkspace,
    *args: str,
    job: str = JOB,
    git: dict[str, Any] | None = None,
    installed: str = INSTALLED,
    env: dict[str, str | None] | None = None,
    prepare: Callable[[ModuleType, pytest.MonkeyPatch], None] | None = None,
    smile: ProbeSmile | None = None,
) -> Ended:
    """Run the probe job on the workspace in a patch context of its own; it may fail."""
    out, err = io.StringIO(), io.StringIO()
    ended = "done"
    extractor = smile or workspace.smile()
    with pytest.MonkeyPatch.context() as monkeypatch:
        module = load_script("probe_encoder")
        monkeypatch.setattr(sys, "unraisablehook", sys.unraisablehook)  # main() replaces it
        monkeypatch.setattr(module, "CONFIG", workspace.config_path)
        monkeypatch.setattr(module, "PROBE", workspace.probe_path)
        monkeypatch.setattr(module, "MODELS", MODELS)
        monkeypatch.setattr(module, "git_state", lambda: copy.deepcopy(git or CHECKOUT))
        monkeypatch.setattr(module, "installed_version", lambda: installed)
        monkeypatch.setattr(module, "stop_on_sigterm", lambda: None)
        monkeypatch.setattr(module, "Smile", SimpleNamespace(load=lambda: extractor))
        monkeypatch.setenv("LBW_WORKSPACE", str(workspace.root))
        monkeypatch.setenv("SLURM_JOB_ID", job)
        for name, value in (env or {}).items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        monkeypatch.setattr(sys, "argv", ["probe_encoder.py", *args])
        if prepare is not None:
            prepare(module, monkeypatch)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                module.main()
            except SystemExit:
                ended = "failed"
    folder = workspace.root / "derived" / "probe" / f"run-{job}"
    return Ended(out.getvalue(), err.getvalue(), ended, folder)
