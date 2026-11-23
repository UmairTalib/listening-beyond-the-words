"""Stand-ins for the tests of the re-voiced clip x1: a synthetic workspace and Kokoro's pipeline.

Synthetic data only. The lines hold the word SECRET, which stands for participant
text: no output that may leave OMNI may contain it.
"""

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).parents[1]
# A word that stands for participant text: no output that may leave OMNI may hold it.
SECRET = "SYNTHETICWORD"
# "1990", "9:45" and "'007" are written out in words ("nineteen ninety", "nine forty
# five", "seven"), and "3rd" mixes digits and letters. Each has as many stored words as
# the token it replaced, so that the draws keep their shape.
X1_TOKENS = (
    "well", "I'm", "1990", "the", "x-ray", "PTSD", "9:45", "don\u2019t", SECRET, "yeah",
    "'007", "LA", "okay.", "we'll", "rock'n'roll", "so,", "it's", "a", "3rd",
)  # fmt: skip
X1_LABELS = {  # session: (label, gender); 304 has no segments
    300: (0, "female"),
    301: (1, "male"),
    302: (1, "female"),
    303: (0, "male"),
    304: (0, "female"),
}
X1_LINE_TOKENS = {300: (40, 40, 45), 301: (50, 41), 302: (44, 44), 303: (60, 30)}


def x1_lines(session: int) -> list[str]:
    """The transcript lines of one synthetic session, each with the secret word in it."""
    rng = np.random.default_rng(session)
    lines = []
    for count in X1_LINE_TOKENS[session]:
        tokens = [str(t) for t in rng.choice(np.array(X1_TOKENS, dtype=object), size=count)]
        tokens[int(rng.integers(count))] = SECRET
        lines.append(" ".join(tokens))
    return lines


def x1_words(session: int, lines: list[str]) -> pd.DataFrame:
    """Aligned words of one session, cut as cut.py would: one word of the second line dropped."""
    from listening_beyond_words.segments import in_segment_order
    from listening_beyond_words.transcripts import normalise

    rows = []
    for line, text in enumerate(lines):
        for index, word in enumerate(normalise(text).split()):
            start = round(5.0 + 30.0 * line + 0.25 * index, 3)
            rows.append(
                {
                    "session": session,
                    "line": line,
                    "word_index": index,
                    "word": word,
                    "start": start,
                    "end": round(start + 0.2, 3),
                    "score": 0.9,
                }
            )
    words = in_segment_order(pd.DataFrame(rows)).reset_index(drop=True)
    dropped = (words["line"] == 1) & (words["word_index"] == 12)
    words["kept"] = ~dropped
    words["cause"] = np.where(dropped, "interviewer turn", None)
    for flag in ("near_interviewer", "outside_participant", "outside_interview"):
        words[flag] = dropped if flag == "near_interviewer" else False
    words["near_interruption"] = False
    new = (words["line"].diff() != 0) | (~words["kept"]).shift(1, fill_value=False)
    words["part"] = np.where(words["kept"], (new & words["kept"]).cumsum() - 1, -1)
    return words


@dataclass
class X1Workspace:
    """A synthetic workspace with everything x1 is built from, and its data config."""

    root: Path
    config: dict
    config_path: Path
    lines: dict[int, list[str]]
    segments: pd.DataFrame
    words: pd.DataFrame = field(repr=False)

    def write_config(self) -> None:
        self.config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")


def make_x1_workspace(root: Path) -> X1Workspace:
    """Write transcripts, a segments run, labels, a word-check sample and x0's manifest."""
    from listening_beyond_words.clips import MANIFEST_COLUMNS
    from listening_beyond_words.runinfo import sha256_file
    from listening_beyond_words.segments import build_segments

    real = yaml.safe_load((REPO / "configs" / "pilot" / "data.yaml").read_text(encoding="utf-8"))
    lines = {session: x1_lines(session) for session in X1_LINE_TOKENS}
    for session, texts in lines.items():
        folder = root / "edaic" / "sessions" / f"{session}_P"
        folder.mkdir(parents=True)
        table = pd.DataFrame(
            {
                "Start_Time": [5.0 + 30.0 * k for k in range(len(texts))],
                "End_Time": [25.0 + 30.0 * k for k in range(len(texts))],
                "Text": texts,
                "Confidence": 0.9,
            }
        )
        table.to_csv(folder / f"{session}_Transcript.csv", index=False)
    (root / "edaic" / "manifest").mkdir(parents=True)
    (root / "edaic" / "manifest" / "sessions_sha256.txt").write_text("checksums\n", "utf-8")

    words = pd.concat([x1_words(s, texts) for s, texts in lines.items()], ignore_index=True)
    segments = build_segments(words)
    run = root / "derived" / "segments" / "run-1"
    run.mkdir(parents=True)
    segments.to_parquet(run / "segments.parquet", index=False)
    words.drop(columns=["word", "score"]).to_parquet(run / "words.parquet", index=False)
    outputs = {name: sha256_file(run / name) for name in ("segments.parquet", "words.parquet")}
    built = {
        "status": "done",
        "inputs": {"word_times": {"sha256": "w" * 64}, "participant_lines": {"sha256": "l" * 64}},
        "settings": {"daicwoz_cut": real["daicwoz_cut"], "x0_cut": real["x0_cut"]},
        "outputs": outputs,
    }
    (run / "run.json").write_text(json.dumps(built), encoding="utf-8")

    labels_run = root / "derived" / "labels" / "run-1"
    labels_run.mkdir(parents=True)
    labels = pd.DataFrame(
        {
            "phq8_total": [5 + 10 * label for label, _ in X1_LABELS.values()],
            "label": [label for label, _ in X1_LABELS.values()],
            "gender": [gender for _, gender in X1_LABELS.values()],
        },
        index=pd.Index(list(X1_LABELS), name="session"),
    )
    labels.to_parquet(labels_run / "labels.parquet")

    sample_run = root / "derived" / "wordcheck_sample" / "run-1"
    sample_run.mkdir(parents=True)
    sample = segments[segments["segment"].isin(["300_001", "301_000"])][["segment", "session"]]
    sample.to_parquet(sample_run / "sample.parquet", index=False)

    x0_run = root / "views" / "x0" / "run-1"
    x0_run.mkdir(parents=True)
    x0 = pd.DataFrame(
        [
            {
                "segment": row.segment,
                "session": row.session,
                "file": f"{row.session}/{row.segment}.flac",
                "samples": 16000,
                "rms": 0.05,
                "peak": 0.5,
                "sha256": "0" * 64,
                "audio_sha256": "0" * 64,
                "part_starts": list(range(row.parts)),
                "part_ends": list(range(1, row.parts + 1)),
                "peak_limited": False,
                "gain_db": 0.0,
            }
            for row in segments.itertuples()
        ],
        columns=list(MANIFEST_COLUMNS),
    )
    x0.to_parquet(x0_run / "manifest.parquet", index=False)

    config = {
        **real,
        "segments": {"run": "derived/segments/run-1", "sha256": outputs["segments.parquet"]},
        "word_times": {"run": "derived/word_times/run-0", "sha256": "w" * 64},
        "participant_lines": {"run": "derived/participant_lines/run-0", "sha256": "l" * 64},
        "labels": {
            "run": "derived/labels/run-1",
            "sha256": sha256_file(labels_run / "labels.parquet"),
        },
        "wordcheck_sample": {
            "run": "derived/wordcheck_sample/run-1",
            "sha256": sha256_file(sample_run / "sample.parquet"),
        },
        "view_runs": {
            "x0": {"run": "views/x0/run-1", "sha256": sha256_file(x0_run / "manifest.parquet")}
        },
    }
    x1 = dict(config["views"]["x1"])
    x1["test_render"] = {"segments": 4, "seed": 106, "repeats": 2, "heard": 2}
    config["views"] = {**config["views"], "x1": x1}
    workspace = X1Workspace(root, config, root.parent / "data.yaml", lines, segments, words)
    workspace.write_config()
    return workspace


# Kokoro's phoneme characters, from the pinned config.json (f3ff3571): 114 in all.
KOKORO_VOCABULARY = frozenset(
    ' !"(),.:;?AIOQSTWYabcdefhijklmnopqrstuvwxyzæçðøŋœɐɑɒɔɕɖəɚɛɜɟɡɣɤɥɨɪɯɰɲɳɴɸɹɻɽɾʁʂʃʈʊʋʌʎʒʔʝ'
    "ʣʤʥʦʧʨʰʲˈˌː\u0303βθχᵊᵝᵻ—“”…→↓↗↘ꭧ"
)


def fake_phonemes(word: str) -> str:
    """Sounds for a written word, made of Kokoro's characters: its letters, digits as 'n'."""
    sounds = []
    for character in word.lower():
        if character.isdigit():
            sounds.append("n")
        elif character == "g":
            sounds.append("ɡ")
        elif character.isalpha():
            sounds.append(character)
    return "ˈ" + "".join(sounds)


def fake_token(
    text: str,
    whitespace: str,
    phonemes: str | None,
    *,
    own: int | None = 4,
    merged: int | None = None,
    tag: str = "NN",
) -> SimpleNamespace:
    """A token as misaki 0.9.4 makes it: the rating on the token or in its ``_``."""
    token = SimpleNamespace(
        text=text,
        whitespace=whitespace,
        phonemes=phonemes,
        tag=tag,
        _=SimpleNamespace(rating=merged),
    )
    if own is not None:
        token.rating = own
    return token


class FakePipeline:
    """Kokoro's pipeline for tests: misaki's tokens, and audio from PyTorch's global generator.

    Args:
        readings: Token specs of particular words, as lists of (phonemes, own rating,
            merged rating) or (phonemes, own rating, merged rating, tag, token text);
            every other word becomes one token from ``fake_phonemes``.
        results: How many results each call yields.
        phonemes: Phonemes each result claims instead of the front end's.
        printing: Whether the front end prints a line with the text, as misaki does.
        fail_on: A word whose call raises an error that holds the text.
        punctuation_digits: Whether a word without a letter is read as misaki reads a
            token that spaCy tagged as punctuation: one token tagged ".", no phonemes,
            misaki's rating 4 in its ``_``.
    """

    def __init__(
        self,
        readings: dict[str, list[tuple[Any, ...]]] | None = None,
        *,
        results: int = 1,
        phonemes: str | None = None,
        printing: bool = False,
        fail_on: str | None = None,
        punctuation_digits: bool = False,
    ) -> None:
        self.readings = readings or {}
        self.results = results
        self.phonemes = phonemes
        self.printing = printing
        self.fail_on = fail_on
        self.punctuation_digits = punctuation_digits
        self.calls: list[tuple[str, str, float]] = []
        self.texts: list[str] = []  # every text given to the front end, in order

    def g2p(self, text: str) -> tuple[str, list[SimpleNamespace]]:
        self.texts.append(text)
        return self.read(text)

    def read(self, text: str) -> tuple[str, list[SimpleNamespace]]:
        if self.printing:
            print("TODO:NUM", text)  # noqa: T201  (misaki prints like this)
        if self.fail_on is not None and self.fail_on in text.split():
            raise ValueError(f"cannot read {text}")
        words = text.split()
        tokens = []
        for k, word in enumerate(words):
            if self.punctuation_digits and not any(c.isalpha() for c in word):
                specs: list[tuple[Any, ...]] = [("", None, 4, ".", word)]
            else:
                specs = self.readings.get(word) or [(fake_phonemes(word), 4, None)]
            for j, (phonemes, own, merged, *named) in enumerate(specs):
                tag, piece = named or ("NN", f"{word}{j}")
                space = " " if j == len(specs) - 1 and k < len(words) - 1 else ""
                tokens.append(fake_token(piece, space, phonemes, own=own, merged=merged, tag=tag))
        return "".join((t.phonemes or "") + t.whitespace for t in tokens), tokens

    def __call__(self, text: str, voice: str, speed: float) -> Any:
        import torch

        self.calls.append((text, voice, speed))
        _, tokens = self.read(text)
        phonemes = "".join((t.phonemes or "") + (" " if t.whitespace else "") for t in tokens)
        for _ in range(self.results):
            lead, tail, body = 7200, 14400, 2400 * len(text.split())
            audio = torch.randn(lead + body + tail) * 1e-5  # quiet, but not 0 at the edges
            audio[lead : lead + body] = torch.randn(body) * 0.1
            yield SimpleNamespace(
                phonemes=self.phonemes if self.phonemes is not None else phonemes.strip(),
                audio=audio.float(),
                pred_dur=torch.tensor([12, 3, 4, 5, 30]),
            )


class LookAheadPipeline(FakePipeline):
    """A stand-in whose reading of a word depends on the next word, as misaki's can.

    Before a word without a letter, "the" gets other phonemes with the same tag, as
    misaki's look-ahead gives them, and "in" another tag and other phonemes, as spaCy's
    tagging can. Every word must be one token.
    """

    def read(self, text: str) -> tuple[str, list[SimpleNamespace]]:
        _, tokens = super().read(text)
        words = text.split()
        assert len(tokens) == len(words), "the look-ahead stand-in reads one token per word"
        for word, after, token in zip(words, [*words[1:], ""], tokens, strict=True):
            if after and not any(character.isalpha() for character in after):
                if word == "the":
                    token.phonemes = "ðˈi"
                elif word == "in":
                    token.phonemes, token.tag = "ˈɪn", "RP"
        return "".join((t.phonemes or "") + t.whitespace for t in tokens), tokens


def fake_kokoro(pipeline: FakePipeline | None = None, settings: Any = None) -> Any:
    """Kokoro with a stand-in pipeline and voice files, as ``Kokoro.load`` would give it."""
    from listening_beyond_words.revoice import Kokoro

    names = (
        settings.all_voices
        if settings is not None
        else ("af_heart", "af_bella", "am_michael", "am_fenrir")
    )
    identity = {
        "repo_id": "hexgrad/Kokoro-82M",
        "revision": "f" * 40,
        "weights_sha256": "a" * 64,
        "voices": dict.fromkeys(names, "b" * 64),
        "espeak_version": "1.52.0",
        "cpu_capability": "DEFAULT",
    }
    return Kokoro(
        pipeline or FakePipeline(),
        KOKORO_VOCABULARY,
        {name: Path(f"voices/{name}.pt") for name in names},
        1.0,
        identity,
    )


def load_script(name: str) -> ModuleType:
    """A step script from scripts/, loaded as a module."""
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_script(
    module: ModuleType,
    workspace: X1Workspace,
    monkeypatch: Any,
    pipeline: FakePipeline | None = None,
    commit: str = "c" * 40,
) -> FakePipeline:
    """Point a script at the synthetic workspace, a clean checkout and a stand-in for Kokoro."""
    from listening_beyond_words import revoice

    used = pipeline or FakePipeline()
    monkeypatch.setattr(sys, "unraisablehook", sys.unraisablehook)  # main() replaces it
    monkeypatch.setattr(module, "CONFIG", workspace.config_path)
    if hasattr(module, "MODELS"):
        monkeypatch.setattr(module, "MODELS", REPO / "configs" / "pilot" / "models.yaml")
    monkeypatch.setattr(module, "git_state", lambda: {"commit": commit, "dirty": False})
    if hasattr(module, "stop_on_sigterm"):
        monkeypatch.setattr(module, "stop_on_sigterm", lambda: None)
    monkeypatch.setattr(
        revoice.Kokoro, "load", classmethod(lambda cls, w, pin, s: fake_kokoro(used, s))
    )
    monkeypatch.setenv("LBW_WORKSPACE", str(workspace.root))
    return used


CHECK_JOB = "500"
ARRAY_JOB = "600"


def check_dir(workspace: X1Workspace, job: str = CHECK_JOB) -> Path:
    return workspace.root / "derived" / "revoice_check" / f"run-{job}"


def run_check(
    workspace: X1Workspace,
    monkeypatch: Any,
    pipeline: FakePipeline | None = None,
    job: str = CHECK_JOB,
) -> Path:
    """Run the check job on the synthetic workspace; return its run folder."""
    module = load_script("check_revoice")
    patch_script(module, workspace, monkeypatch, pipeline)
    monkeypatch.setenv("SLURM_JOB_ID", job)
    module.main()
    return check_dir(workspace, job)


def run_task(
    module: ModuleType,
    monkeypatch: Any,
    task: int,
    array: str = ARRAY_JOB,
    check: str = f"derived/revoice_check/run-{CHECK_JOB}",
) -> None:
    """Run one array task of the build, as Slurm would start it."""
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", array)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", str(task))
    monkeypatch.setenv("SLURM_JOB_ID", str(int(array) + 1 + task))
    monkeypatch.setattr(sys, "argv", ["build_revoiced.py", check])
    module.main()


def build_all(workspace: X1Workspace, monkeypatch: Any, array: str = ARRAY_JOB) -> Path:
    """Run the check job and every task of the build; return the build's run folder."""
    from listening_beyond_words.revoice import TASKS

    run_check(workspace, monkeypatch)
    module = load_script("build_revoiced")
    patch_script(module, workspace, monkeypatch)
    for task in range(TASKS):
        run_task(module, monkeypatch, task, array)
    return workspace.root / "views" / "x1" / f"run-{array}"


def outside_text(*paths: Path, streams: str = "") -> str:
    """Everything that may leave OMNI, in lower case: the printed streams and the given files."""
    texts = [streams]
    for path in paths:
        if path.is_file():
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(texts).lower()
