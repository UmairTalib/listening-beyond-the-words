import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from listening_beyond_words import alignment as al
from listening_beyond_words.models import load_pins, local_snapshot

PINS = Path(__file__).resolve().parents[1] / "configs" / "pilot" / "models.yaml"
VOCAB = {
    "<pad>": 0,
    "|": 1,
    **{letter: 2 + i for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")},
}


@pytest.mark.parametrize(
    ("number", "words"),
    [
        (0, "zero"),
        (25, "twenty five"),
        (100, "one hundred"),
        (2010, "two thousand ten"),
        (1234567, "one two three four five six seven"),
    ],
)
def test_number_words_spells_whole_numbers(number: int, words: str) -> None:
    assert al.number_words(number) == words


def test_spoken_spells_out_the_digits_inside_a_word() -> None:
    assert al.spoken("3rd") == "three rd"
    assert al.spoken("90s") == "ninety s"
    assert al.spoken("hello") == "hello"


def test_targets_mark_which_word_each_token_belongs_to() -> None:
    ids, owners = al.targets(["ab", "c"], VOCAB)
    assert ids == [VOCAB["A"], VOCAB["B"], VOCAB["|"], VOCAB["C"]]
    assert owners == [0, 0, -1, 1]
    ids, owners = al.targets(["25"], VOCAB)
    assert ids.count(VOCAB["|"]) == 1  # "twenty five": one delimiter inside the word
    assert set(owners) == {0}


def synthetic_log_probs(path: list[str]) -> torch.Tensor:
    """Log-probabilities that make the given frame-by-frame path clearly the best."""
    probs = np.full((len(path), len(VOCAB)), 0.03 / (len(VOCAB) - 1))
    for frame, token in enumerate(path):
        probs[frame, VOCAB[token]] = 0.97
    return torch.tensor(np.log(probs), dtype=torch.float32)


def test_align_words_places_each_word_on_its_frames() -> None:
    path = ["<pad>", "<pad>", "A", "A", "B", "<pad>", "|", "<pad>", "C", "C", "<pad>", "<pad>"]
    times = al.align_words(synthetic_log_probs(path), ["ab", "c"], VOCAB, 0, 0.02)
    assert times is not None
    assert [t.position for t in times] == [0, 1]
    assert times[0].start == pytest.approx(0.04)
    assert times[0].end == pytest.approx(0.10)
    assert times[1].start == pytest.approx(0.16)
    assert times[1].end == pytest.approx(0.20)
    assert all(0.9 < t.score <= 1.0 for t in times)


def test_align_words_gives_up_when_the_audio_is_too_short() -> None:
    assert al.align_words(synthetic_log_probs(["A", "B"]), ["abc", "c"], VOCAB, 0, 0.02) is None


@pytest.fixture(scope="module")
def aligner() -> Any:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        pytest.skip("HF_HOME is not set; run the GPU tests in scripts/slurm/smoke_test.sbatch")
    return al.WordAligner(local_snapshot(load_pins(PINS)["wav2vec2_960h"], Path(hf_home)))


@pytest.mark.gpu
def test_word_aligner_returns_ordered_times_inside_the_clip(aligner: Any) -> None:
    rng = np.random.default_rng(9005)
    audio = (0.05 * rng.standard_normal(3 * al.SAMPLING_RATE)).astype(np.float32)
    times = aligner.align(audio, ["synthetic", "noise"])
    assert times is not None
    assert 0 <= times[0].start <= times[0].end <= times[1].start <= times[1].end <= 3.0
