"""Stand-ins for the tests of the word check (scripts/check_views.py).

A synthetic workspace with everything the job reads: transcript lines, a segments run,
labels, partitions, a word-check sample, FLAC clips of all four views with their
manifests, the public files of a tool check and model snapshots that hold no model.
Fake ASR systems hear a clip by its samples; a fake opensmile gives values from a hash of
each stretch of audio. Synthetic data only, on pilot session IDs, because transcript
lines are read through the session guard. The lines hold SECRET (x1_fakes), which stands
for participant text: only transcripts.parquet may hold it. The tests of every job that
reads participant clips reuse the workspace and the check of what may leave OMNI
(assert_only_aggregates_leave).
"""

import hashlib
import json
import re
import shutil
import sys
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from x1_fakes import REPO, SECRET, X1_TOKENS, load_script, x1_words

SESSIONS = tuple(range(300, 320))  # pilot IDs: read_lines refuses every other ID
FEW_SEGMENTS = 300  # one participant with only 3 segments, all of them sampled
LABEL_WORDS = ("cheerful", "gloomy")  # in the lines of negative and positive participants
PUBLIC_TEXT = (
    "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background."
)
PUBLIC_VOICES = (
    "af_heart",
    "af_bella",
    "af_sarah",
    "af_kore",
    "am_michael",
    "am_fenrir",
    "am_puck",
)
REV_TEXT = "thank you for watching"  # what the fakes hear in a reversed clip
VIEWS = ("x0", "x1", "x2", "rev")
GAP = 4800  # 300 ms at 16 kHz
JOB = "700"
RERUN_JOB = "701"
RESAMPLES = 200  # bootstrap resamples of the fast tests; the real config has 10,000
MODEL_FILES = {
    "faster_whisper": (
        "model.bin",
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "vocabulary.json",
    ),
    "wav2vec2_960h": ("config.json", "model.safetensors", "vocab.json"),
}


def label_of(session: int) -> tuple[int, str]:
    """Label and gender: five participants in each of the four strata."""
    index = session - SESSIONS[0]
    return index % 2, "female" if (index // 2) % 2 == 0 else "male"


def wordcheck_lines(session: int) -> list[str]:
    """Transcript lines of one session: x1's tokens, the secret word and its label word."""
    rng = np.random.default_rng(session)
    label, _ = label_of(session)
    count = 3 if session == FEW_SEGMENTS else 4 + session % 3
    lines = []
    for _ in range(count):
        size = int(rng.integers(40, 47))
        tokens = [str(t) for t in rng.choice(np.array(X1_TOKENS, dtype=object), size=size)]
        tokens[int(rng.integers(size))] = SECRET
        for _ in range(2):
            tokens[int(rng.integers(size))] = LABEL_WORDS[label]
        lines.append(" ".join(tokens))
    return lines


def audio_key(samples: np.ndarray) -> str:
    """What the fakes recognise a clip by: the SHA-256 of its 16-bit samples."""
    pcm = np.rint(np.asarray(samples, dtype=np.float64) * 32768).astype("<i2")
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def clip_samples(
    rng: np.random.Generator, words_per_part: list[int], base: int, per_word: int
) -> tuple[np.ndarray, list[int], list[int]]:
    """A clip of noisy parts that start and end at 0, joined by silent gaps of 300 ms."""
    pieces: list[np.ndarray] = []
    starts: list[int] = []
    ends: list[int] = []
    position = 0
    for index, count in enumerate(words_per_part):
        if index:
            pieces.append(np.zeros(GAP, dtype=np.int16))
            position += GAP
        length = base + per_word * count
        part = np.clip(np.rint(rng.normal(0.0, 0.08, length) * 32768), -32768, 32767)
        part[0] = part[-1] = 0
        pieces.append(part.astype(np.int16))
        starts.append(position)
        position += length
        ends.append(position)
    return np.concatenate(pieces), starts, ends


@dataclass
class WordcheckWorkspace:
    """A synthetic workspace for the word check, its data config and what the fakes hear."""

    root: Path
    config: dict
    config_path: Path
    segments: pd.DataFrame
    sample: pd.DataFrame
    audio: dict[str, tuple[str, str]]  # audio key -> (view or "guard", segment or voice)
    text_of: dict[str, str]  # segment -> its stored words

    def write_config(self) -> None:
        self.config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")

    def copy_to(self, folder: Path) -> "WordcheckWorkspace":
        """A copy of the workspace in a new folder, with its own config file."""
        root = folder / "workspace"
        shutil.copytree(self.root, root, symlinks=True)
        copy = WordcheckWorkspace(
            root,
            json.loads(json.dumps(self.config)),
            folder / "data.yaml",
            self.segments,
            self.sample,
            self.audio,
            self.text_of,
        )
        copy.write_config()
        return copy

    def run_dir(self, job: str = JOB) -> Path:
        return self.root / "derived" / "wordcheck" / f"run-{job}"


def write_pinned(run: Path, table: pd.DataFrame, name: str, index: bool = False) -> str:
    """Write one table of a pinned run and return its SHA-256."""
    from listening_beyond_words.runinfo import sha256_file

    run.mkdir(parents=True, exist_ok=True)
    table.to_parquet(run / name, index=index)
    return sha256_file(run / name)


def make_wordcheck_workspace(root: Path) -> WordcheckWorkspace:
    """Write every input of the word check into a new workspace folder."""
    from listening_beyond_words.clips import MANIFEST_COLUMNS, write_view
    from listening_beyond_words.models import load_pins
    from listening_beyond_words.partitions import make_partitions
    from listening_beyond_words.runinfo import sha256_file
    from listening_beyond_words.segments import build_segments, segment_parts
    from listening_beyond_words.wordcheck import draw_sample

    real = yaml.safe_load((REPO / "configs" / "pilot" / "data.yaml").read_text(encoding="utf-8"))
    lines = {session: wordcheck_lines(session) for session in SESSIONS}
    for session, texts in lines.items():
        folder = root / "edaic" / "sessions" / f"{session}_P"
        folder.mkdir(parents=True)
        pd.DataFrame(
            {
                "Start_Time": [5.0 + 30.0 * k for k in range(len(texts))],
                "End_Time": [25.0 + 30.0 * k for k in range(len(texts))],
                "Text": texts,
                "Confidence": 0.9,
            }
        ).to_csv(folder / f"{session}_Transcript.csv", index=False)
    (root / "edaic" / "manifest").mkdir(parents=True)
    (root / "edaic" / "manifest" / "sessions_sha256.txt").write_text("checksums\n", "utf-8")

    words = pd.concat([x1_words(s, texts) for s, texts in lines.items()], ignore_index=True)
    segments = build_segments(words)
    segments_run = root / "derived" / "segments" / "run-1"
    outputs = {
        "segments.parquet": write_pinned(segments_run, segments, "segments.parquet"),
        "words.parquet": write_pinned(
            segments_run, words.drop(columns=["word", "score"]), "words.parquet"
        ),
    }
    (segments_run / "run.json").write_text(
        json.dumps(
            {
                "status": "done",
                "inputs": {
                    "word_times": {"sha256": "w" * 64},
                    "participant_lines": {"sha256": "l" * 64},
                },
                "settings": {"daicwoz_cut": real["daicwoz_cut"], "x0_cut": real["x0_cut"]},
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )

    labels = pd.DataFrame(
        {
            "phq8_total": [5 + 10 * label_of(s)[0] for s in SESSIONS],
            "label": [label_of(s)[0] for s in SESSIONS],
            "gender": [label_of(s)[1] for s in SESSIONS],
        },
        index=pd.Index(list(SESSIONS), name="session"),
    )
    labels_sha = write_pinned(root / "derived" / "labels" / "run-1", labels, "labels.parquet", True)
    partitions_run = root / "derived" / "partitions" / "run-1"
    partitions_sha = write_pinned(
        partitions_run, make_partitions(labels[["label", "gender"]]), "partitions.parquet"
    )
    (partitions_run / "run.json").write_text(
        json.dumps({"status": "done", "inputs": {"labels": {"sha256": labels_sha}}}), "utf-8"
    )
    sample = draw_sample(segments, np.random.default_rng(105), 5)
    sample_sha = write_pinned(
        root / "derived" / "wordcheck_sample" / "run-1", sample, "sample.parquet"
    )

    parts = segment_parts(words)
    sizes = {str(s): group["n_words"].tolist() for s, group in parts.groupby("segment")}
    audio: dict[str, tuple[str, str]] = {}
    view_runs = {}
    for number, view in enumerate(VIEWS):
        run = root / "views" / view / "run-1"
        rows = []
        for row in segments.itertuples(index=False):
            segment, session = str(row.segment), int(row.session)
            rng = np.random.default_rng([number, session, int(row.index)])
            # x1 has its own part bounds; its parts have an odd length, the others an even one
            base, per_word = (2801, 170) if view == "x1" else (3000, 150)
            samples, starts, ends = clip_samples(rng, sizes[segment], base, per_word)
            (run / str(session)).mkdir(parents=True, exist_ok=True)
            values = write_view(run / str(session) / f"{segment}.flac", samples)
            rows.append(
                {
                    "segment": segment,
                    "session": session,
                    "file": f"{session}/{segment}.flac",
                    **values,
                    "part_starts": starts,
                    "part_ends": ends,
                    "peak_limited": False,
                    "gain_db": 0.0,
                }
            )
            audio[audio_key(samples / 32768)] = (view, segment)
        manifest = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
        view_runs[view] = {
            "run": f"views/{view}/run-1",
            "sha256": write_pinned(run, manifest, "manifest.parquet"),
        }

    guard = root / "derived" / "check_view_tools" / "run-1"
    (guard / "voices").mkdir(parents=True)
    guard_outputs = {}
    for number, voice in enumerate(PUBLIC_VOICES):
        noise = np.random.default_rng([9, number]).normal(0.0, 0.1, 16000)
        sf.write(guard / "voices" / f"{voice}.wav", noise, 16000, subtype="PCM_16")
        back, _ = sf.read(guard / "voices" / f"{voice}.wav", dtype="int16")
        audio[audio_key(back / 32768)] = ("guard", voice)
        guard_outputs[f"voices/{voice}.wav"] = sha256_file(guard / "voices" / f"{voice}.wav")
    (guard / "run.json").write_text(
        json.dumps(
            {
                "step": "check_view_tools",
                "status": "done",
                "verdict": "PASS",
                "settings": {"text": PUBLIC_TEXT, "voices": list(PUBLIC_VOICES)},
                "outputs": guard_outputs,
            }
        ),
        encoding="utf-8",
    )

    pins = load_pins(REPO / "configs" / "pilot" / "models.yaml")
    for name, files in MODEL_FILES.items():
        repo = root / "hf" / "hub" / ("models--" + pins[name].repo_id.replace("/", "--"))
        snapshot = repo / "snapshots" / pins[name].revision
        snapshot.mkdir(parents=True)
        (repo / "blobs").mkdir()
        for file in files:
            blob = hashlib.sha256(f"{name}/{file}".encode()).hexdigest()
            (repo / "blobs" / blob).write_text(file, encoding="utf-8")
            (snapshot / file).symlink_to(Path("..") / ".." / "blobs" / blob)

    word_check = json.loads(json.dumps(real["word_check"]))
    word_check["bootstrap"]["resamples"] = RESAMPLES
    word_check["public_guard"]["run"] = "derived/check_view_tools/run-1"
    config = {
        **real,
        "segments": {"run": "derived/segments/run-1", "sha256": outputs["segments.parquet"]},
        "word_times": {"run": "derived/word_times/run-0", "sha256": "w" * 64},
        "participant_lines": {"run": "derived/participant_lines/run-0", "sha256": "l" * 64},
        "labels": {"run": "derived/labels/run-1", "sha256": labels_sha},
        "partitions": {"run": "derived/partitions/run-1", "sha256": partitions_sha},
        "wordcheck_sample": {"run": "derived/wordcheck_sample/run-1", "sha256": sample_sha},
        "view_runs": view_runs,
        "word_check": word_check,
    }
    text_of = dict(zip(segments["segment"].astype(str), segments["text"], strict=True))
    workspace = WordcheckWorkspace(
        root, config, root.parent / "data.yaml", segments, sample, audio, text_of
    )
    workspace.write_config()
    return workspace


# The ASR systems and opensmile ---------------------------------------------------------------


@dataclass
class Ears:
    """What the fake ASR systems hear, and a log of what the job did with them.

    Attributes:
        workspace: The workspace whose clips they know.
        texts: Texts to hear instead, by (view, segment).
        guard_text: What they hear in the public files.
        guard_texts: What one ASR system hears in them instead, by its name.
        fail_on: (ASR system, view, segment) whose read raises an error holding the secret.
        events: ("load", name, models held after loading), ("read", name, view, segment)
            and ("released", name), in order.
    """

    workspace: WordcheckWorkspace
    texts: dict[tuple[str, str], str] = field(default_factory=dict)
    guard_text: str = PUBLIC_TEXT
    guard_texts: dict[str, str] = field(default_factory=dict)
    fail_on: tuple[str, str, str] | None = None
    events: list[tuple[Any, ...]] = field(default_factory=list)
    held: int = 0

    def hear(self, name: str, audio: np.ndarray) -> str:
        view, item = self.workspace.audio[audio_key(audio)]
        self.events.append(("read", name, view, item))
        if self.fail_on == (name, view, item):
            raise ValueError(f"cannot read {SECRET}")
        if view == "guard":
            return self.guard_texts.get(name, self.guard_text)
        if (view, item) in self.texts:
            return self.texts[(view, item)]
        return REV_TEXT if view == "rev" else self.workspace.text_of[item]

    def loaded(self, name: str, model: Any) -> Any:
        self.held += 1
        self.events.append(("load", name, self.held))
        weakref.finalize(model, self.released, name)
        return model

    def released(self, name: str) -> None:
        self.held -= 1
        self.events.append(("released", name))

    def reads(self, name: str, kind: str = "participant") -> list[tuple[str, str]]:
        """(view, segment) of every read by one ASR system, guard reads or participant reads."""
        return [
            (event[2], event[3])
            for event in self.events
            if event[0] == "read"
            and event[1] == name
            and (event[2] == "guard") == (kind == "guard")
        ]


@dataclass
class FakeOptions:
    """faster-whisper's TranscriptionOptions, as far as the job records them."""

    beam_size: int
    temperatures: list[float]
    heard: str  # "guard" or "participant": which clip the options were resolved on


class FakeWhisperModel:
    """faster-whisper's WhisperModel: one segment per clip, with the text the ears hear."""

    def __init__(self, ears: Ears) -> None:
        self.ears = ears

    def transcribe(self, audio: np.ndarray, **options: Any) -> tuple[Any, Any]:
        text = self.ears.hear("whisper", audio)
        kind = "guard" if self.ears.events[-1][2] == "guard" else "participant"
        segments = (
            [
                SimpleNamespace(
                    text=" " + text,
                    seek=0,
                    avg_logprob=-0.2,
                    no_speech_prob=0.1,
                    compression_ratio=1.3,
                )
            ]
            if text
            else []
        )
        info = SimpleNamespace(
            transcription_options=FakeOptions(
                options["beam_size"], [float(options["temperature"])], kind
            ),
            duration=len(audio) / 16000,
        )
        return iter(segments), info


class FakeWav2vec2:
    """wav2vec2 as the job uses it: capitals, frames and no special symbols."""

    name = "wav2vec2"

    def __init__(self, ears: Ears) -> None:
        self.ears = ears
        self.identity = {"dtype": "float32", "device": "cuda", "symbols": 32}

    def read(self, audio: np.ndarray) -> Any:
        from listening_beyond_words.recognisers import Reading, check_audio

        check_audio(audio)
        text = self.ears.hear("wav2vec2", audio)
        return Reading(
            text=" ".join(text.upper().split()),
            seconds=0.001,
            diagnostics={"frames": len(audio) // 320, "special_symbols": 0},
        )


class FakeSmile:
    """opensmile's stand-in: values from a hash of each stretch of audio, a pitch track by length.

    A stretch of odd length, which only x1's parts have, gets the same values everywhere,
    with one voiced stretch: x1 then follows nothing of x0, so its gate ICCs are 0 and its
    probe knows only the gender, and the voice rule passes whatever x0's values are. The
    pitch track holds the voiced stretches that VoicedSegmentsPerSec counts, in its first
    70% of frames, and opensmile divides the count by 2 frames fewer than the track has.
    """

    def __init__(self) -> None:
        from listening_beyond_words.voicecheck import FUNCTIONAL_NAMES

        self.names = FUNCTIONAL_NAMES

    @staticmethod
    def frames(part: np.ndarray) -> int:
        return (len(part) - 960) // 160 + 2

    @staticmethod
    def digest(part: np.ndarray) -> bytes:
        return hashlib.sha256(np.asarray(part, dtype=np.float64).tobytes()).digest()

    def stretches(self, part: np.ndarray) -> int:
        return 1 if len(part) % 2 == 1 else 1 + self.digest(part)[8] % 2

    def voiced_share(self, part: np.ndarray) -> float:
        """The share of the frames, from the first, that hold the voiced stretches."""
        return 0.7

    def functionals(self, part: np.ndarray) -> pd.Series:
        digest = self.digest(part)
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        fixed = len(part) % 2 == 1
        values = dict.fromkeys(self.names, 1.0)
        if not fixed:
            values.update(zip(self.names, rng.normal(1.0, 0.3, len(self.names)), strict=True))
        values["VoicedSegmentsPerSec"] = self.stretches(part) / ((self.frames(part) - 2) * 0.01)
        values["MeanVoicedSegmentLengthSec"] = 0.2 if fixed else 0.15 + 0.1 * rng.random()
        values["StddevVoicedSegmentLengthSec"] = 0.05 if fixed else 0.05 * rng.random()
        values["equivalentSoundLevel_dBp"] = -30.0 if fixed else -30.0 + rng.normal()
        return pd.Series(values, dtype=float)

    def pitch(self, part: np.ndarray) -> np.ndarray:
        frames = self.frames(part)
        track = np.zeros(frames)
        start = 0
        voiced = np.arange(int(frames * self.voiced_share(part)))
        for run in np.array_split(voiced, max(self.stretches(part), 1)):
            track[start : start + len(run)] = 30.0
            start += len(run) + 2
        return track


# Running the job ---------------------------------------------------------------------------------


def patch_job(
    module: ModuleType,
    workspace: WordcheckWorkspace,
    monkeypatch: Any,
    ears: Ears,
    commit: str = "c" * 40,
) -> None:
    """Point the job at the workspace, a clean checkout, a GPU and the stand-ins."""
    from listening_beyond_words.recognisers import Whisper

    monkeypatch.setattr(sys, "unraisablehook", sys.unraisablehook)  # main() replaces it
    monkeypatch.setattr(module, "CONFIG", workspace.config_path)
    monkeypatch.setattr(module, "MODELS", REPO / "configs" / "pilot" / "models.yaml")
    monkeypatch.setattr(module, "git_state", lambda: {"commit": commit, "dirty": False})
    monkeypatch.setattr(module, "stop_on_sigterm", lambda: None)
    monkeypatch.setattr(
        module,
        "gpu_facts",
        lambda: {"gpu": "Synthetic GPU", "cuda": "12.6", "cudnn": None, "torch_threads": 1},
    )
    identity = {"compute_type": "float16", "device": "cuda", "mel_bands": 128}
    monkeypatch.setattr(
        module,
        "load_whisper",
        lambda snapshot, settings: ears.loaded(
            "whisper", Whisper(FakeWhisperModel(ears), settings, identity)
        ),
    )
    monkeypatch.setattr(
        module, "load_wav2vec2", lambda snapshot: ears.loaded("wav2vec2", FakeWav2vec2(ears))
    )
    monkeypatch.setattr(module, "Smile", SimpleNamespace(load=FakeSmile))
    monkeypatch.setenv("LBW_WORKSPACE", str(workspace.root))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def run_job(
    workspace: WordcheckWorkspace,
    monkeypatch: Any,
    ears: Ears,
    *args: str,
    job: str = JOB,
    module: ModuleType | None = None,
    commit: str = "c" * 40,
) -> Path:
    """Run the word check on the synthetic workspace; return its run folder."""
    script = module or load_script("check_views")
    patch_job(script, workspace, monkeypatch, ears, commit)
    monkeypatch.setenv("SLURM_JOB_ID", job)
    monkeypatch.setattr(sys, "argv", ["check_views.py", *args])
    script.main()
    return workspace.run_dir(job)


def info_of(run: Path) -> dict:
    return json.loads((run / "run.json").read_text(encoding="utf-8"))


# What may leave OMNI ----------------------------------------------------------------------------

# What would show a number per segment or participant: a segment ID, or a participant ID
# as a key or paired with a value; and in run.json, a list as long as a fold of segments.
SEGMENT_ID = re.compile(r"\b3\d\d_\d{3}\b")
PARTICIPANT_PAIR = re.compile(r"[{\[,]\s*3\d\d\s*[:,]")
NUMBER_LIST = re.compile(r"[\[(](?:\s*-?[0-9][0-9.e+-]*\s*,){7,}")  # 8 numbers or more
ITEM_KEY = re.compile(r"3\d\d(_\d{3})?")
LONGEST_LIST = 7  # the public guard's voices; folds and intervals are shorter
LONG_LISTS = {("settings", "whisper_resolved", "suppress_tokens")}  # long with faster-whisper


def per_item_paths(node: Any, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Where run.json could hold a number per segment or participant: IDs as keys, long lists."""
    found: list[tuple[str, ...]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if ITEM_KEY.fullmatch(str(key)):
                found.append((*path, str(key)))
            found += per_item_paths(value, (*path, str(key)))
    elif isinstance(node, list):
        if len(node) > LONGEST_LIST and path not in LONG_LISTS:
            found.append(path)
        for value in node:
            found += per_item_paths(value, path)
    return found


def assert_only_aggregates_leave(run_json: Path, streams: str) -> None:
    """No ID on the streams (but on a failure line), and none in run.json (but in its error).

    Shared by the tests of every job that reads participant clips. A failure may name its
    segment on its one ``FAILED:`` line and in ``run.json``'s ``error``, on OMNI only.
    Elsewhere in ``run.json`` no ID may appear, as a key or as a value, and no list is
    long enough to hold a number per item; values keyed by a layer's name are aggregates.
    """
    shown = "\n".join(line for line in streams.splitlines() if not line.startswith("FAILED: "))
    assert not SEGMENT_ID.search(shown)
    assert not PARTICIPANT_PAIR.search(shown)
    assert not NUMBER_LIST.search(shown)
    info = json.loads(run_json.read_text(encoding="utf-8"))
    assert per_item_paths(info) == []
    rest = json.dumps({key: value for key, value in info.items() if key != "error"})
    assert not SEGMENT_ID.search(rest)
    assert not PARTICIPANT_PAIR.search(rest)
