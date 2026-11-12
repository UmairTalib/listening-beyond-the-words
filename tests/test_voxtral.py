import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from listening_beyond_words import voxtral
from listening_beyond_words.models import load_pins, local_snapshot

PINS = Path(__file__).resolve().parents[1] / "configs" / "pilot" / "models.yaml"


class FakeTokenizer:
    """Encodes each known text to a fixed list of token IDs."""

    def __init__(self, vocab: dict[str, list[int]]) -> None:
        self.vocab = vocab

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert not add_special_tokens
        return self.vocab[text]


def synthetic_clip(seed: int, seconds: float = 3.0) -> np.ndarray:
    """A tone plus noise at 16 kHz: synthetic audio without speech."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * voxtral.SAMPLING_RATE)) / voxtral.SAMPLING_RATE
    tone = 0.3 * np.sin(2 * np.pi * (150 + 50 * seed) * t)
    return (tone + 0.05 * rng.standard_normal(t.size)).astype(np.float32)


def test_answer_token_ids_returns_yes_then_no() -> None:
    assert voxtral.answer_token_ids(FakeTokenizer({"Yes": [7], "No": [9]})) == (7, 9)


@pytest.mark.parametrize("vocab", [{"Yes": [7, 8], "No": [9]}, {"Yes": [7], "No": []}])
def test_answer_token_ids_rejects_an_answer_that_is_not_one_token(
    vocab: dict[str, list[int]],
) -> None:
    with pytest.raises(ValueError, match="must be one token"):
        voxtral.answer_token_ids(FakeTokenizer(vocab))


def test_answer_token_ids_rejects_answers_that_share_a_token() -> None:
    with pytest.raises(ValueError, match="share the token"):
        voxtral.answer_token_ids(FakeTokenizer({"Yes": [7], "No": [7]}))


def test_clip_inputs_needs_mono_audio_at_16_khz() -> None:
    with pytest.raises(ValueError, match="mono"):
        voxtral.clip_inputs(None, np.zeros((2, 16_000), dtype=np.float32))
    with pytest.raises(ValueError, match="16000 Hz"):
        voxtral.clip_inputs(None, np.zeros(8_000, dtype=np.float32), sampling_rate=8_000)


@pytest.fixture(scope="module")
def model_and_processor() -> tuple[Any, Any]:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        pytest.skip("HF_HOME is not set; run the GPU tests in scripts/slurm/smoke_test.sbatch")
    snapshot = local_snapshot(load_pins(PINS)["voxtral"], Path(hf_home))
    return voxtral.load_model(snapshot)


@pytest.mark.gpu
def test_yes_and_no_are_single_tokens(model_and_processor: tuple[Any, Any]) -> None:
    _, processor = model_and_processor
    yes_id, no_id = voxtral.answer_token_ids(processor.tokenizer)
    assert processor.tokenizer.decode([yes_id]) == "Yes"
    assert processor.tokenizer.decode([no_id]) == "No"


@pytest.mark.gpu
def test_voxtral_loads_in_fp16_and_scores_clips(model_and_processor: tuple[Any, Any]) -> None:
    import torch

    model, processor = model_and_processor
    assert model.dtype == torch.float16
    assert model.device.type == "cuda"
    answer_ids = voxtral.answer_token_ids(processor.tokenizer)
    scores = [
        voxtral.yes_no_score(
            model, voxtral.clip_inputs(processor, synthetic_clip(seed)), answer_ids
        )
        for seed in (1, 2)
    ]
    assert all(math.isfinite(z) for z in scores)
    assert scores[0] != scores[1], "the score does not depend on the audio"
