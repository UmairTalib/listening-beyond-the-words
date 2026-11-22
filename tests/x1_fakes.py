"""Stand-ins for the tests of the re-voiced clip x1: a synthetic workspace and Kokoro's pipeline.

Synthetic data only. The lines hold the word SECRET, which stands for participant
text: no output that may leave OMNI may contain it.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).parents[1]
# A word that stands for participant text: no output that may leave OMNI may hold it.
SECRET = "SYNTHETICWORD"
X1_TOKENS = (
    "well", "I'm", "fine", "the", "x-ray", "PTSD", "9:30", "don\u2019t", SECRET, "yeah",
    "really", "LA", "okay.", "we'll", "rock'n'roll", "so,", "it's", "a", "lot",
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
) -> SimpleNamespace:
    """A token as misaki 0.9.4 makes it: the rating on the token or in its ``_``."""
    token = SimpleNamespace(
        text=text,
        whitespace=whitespace,
        phonemes=phonemes,
        tag="NN",
        _=SimpleNamespace(rating=merged),
    )
    if own is not None:
        token.rating = own
    return token


class FakePipeline:
    """Kokoro's pipeline for tests: misaki's tokens, and audio from PyTorch's global generator.

    Args:
        readings: Token specs of particular words, as lists of (phonemes, own rating,
            merged rating); every other word becomes one token from ``fake_phonemes``.
        results: How many results each call yields.
        phonemes: Phonemes each result claims instead of the front end's.
        printing: Whether the front end prints a line with the text, as misaki does.
        fail_on: A word whose call raises an error that holds the text.
    """

    def __init__(
        self,
        readings: dict[str, list[tuple[str | None, int | None, int | None]]] | None = None,
        *,
        results: int = 1,
        phonemes: str | None = None,
        printing: bool = False,
        fail_on: str | None = None,
    ) -> None:
        self.readings = readings or {}
        self.results = results
        self.phonemes = phonemes
        self.printing = printing
        self.fail_on = fail_on
        self.calls: list[tuple[str, str, float]] = []

    def g2p(self, text: str) -> tuple[str, list[SimpleNamespace]]:
        if self.printing:
            print("TODO:NUM", text)  # noqa: T201  (misaki prints like this)
        if self.fail_on is not None and self.fail_on in text.split():
            raise ValueError(f"cannot read {text}")
        words = text.split()
        tokens = []
        for k, word in enumerate(words):
            specs = self.readings.get(word) or [(fake_phonemes(word), 4, None)]
            for j, (phonemes, own, merged) in enumerate(specs):
                space = " " if j == len(specs) - 1 and k < len(words) - 1 else ""
                tokens.append(fake_token(f"{word}{j}", space, phonemes, own=own, merged=merged))
        return "".join((t.phonemes or "") + t.whitespace for t in tokens), tokens

    def __call__(self, text: str, voice: str, speed: float) -> Any:
        import torch

        self.calls.append((text, voice, speed))
        _, tokens = self.g2p(text)
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
