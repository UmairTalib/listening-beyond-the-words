"""A tiny random Voxtral and a stand-in for its processor, for tests on the CPU.

The model (``tiny_model``) has Voxtral's layout at a tiny size: an encoder of 2 layers
of width 64 (``WIDTH``) with a feed-forward width of 256, so that the adapter's 4-by-4
grouping fits as it does in Voxtral (5,120 = 4 x 1,280); a language model of 2 layers; a
vocabulary of 64 tokens; and the audio token 24. Its weights are random, from a seed. In
fp16 it keeps the encoder's position weights in fp32, as ``from_pretrained`` does with
the real model.

The processor stand-in (``FakeProcessor``) builds real log-mel features, with
``WhisperFeatureExtractor(feature_size=128)`` and the processor's audio settings and
split into 30 s chunks (the 37.4 s clip gives (2, 128, 3000)), and fixed token IDs laid
out as Voxtral's processor lays them out: ``<s> [INST] [BEGIN_AUDIO]``, 375 audio tokens
per chunk, 18 prompt tokens, ``[/INST]``. A clip of up to 30 s gives 397 IDs and one of
37.4 s 772, as with the real processor. "Yes" and "No" are the tokens 60 and 61.

``save_checkpoint`` writes a model as the pinned snapshot stores Voxtral, in bf16, and
``patch_processor`` lets ``voxtral.load_model`` load such a folder with the stand-in.
Synthetic audio only (``tone``). torch and transformers are imported inside functions.
"""

import base64
import copy
import io
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
import soundfile as sf

from listening_beyond_words import voxtral

RATE = 16_000
WIDTH = 64  # the encoder's width W (1,280 in Voxtral)
VOCAB_SIZE = 64
AUDIO_TOKEN_ID = 24  # as in Voxtral
BOS_ID, INST_ID, INST_END_ID, BEGIN_AUDIO_ID = 1, 3, 4, 25  # <s>, [INST], [/INST], [BEGIN_AUDIO]
PROMPT_IDS = tuple(range(30, 48))  # 18 tokens, as the real tokenizer splits the prompt
YES_ID, NO_ID = 60, 61
ANSWER_IDS = (YES_ID, NO_ID)
CHECKPOINT = "model.safetensors"


def tone(seconds: float, seed: int = 1) -> np.ndarray:
    """A pulsing tone plus noise at 16 kHz: synthetic audio without speech.

    Args:
        seconds: Length; the clip has ``round(seconds * 16000)`` samples.
        seed: Sets the pitch and the noise.

    Returns:
        Mono float32 samples within [-1, 1].
    """
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * RATE)) / RATE
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t)
    wave = 0.3 * np.sin(2 * np.pi * (150 + 50 * seed) * t) * pulse
    return (wave + 0.05 * rng.standard_normal(t.size)).astype(np.float32)


def tiny_config() -> Any:
    """The configuration of the tiny Voxtral (a ``VoxtralConfig``)."""
    from transformers import VoxtralConfig

    return VoxtralConfig(
        audio_config={
            "model_type": "voxtral_encoder",
            "hidden_size": WIDTH,
            "intermediate_size": 4 * WIDTH,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_mel_bins": 128,
            "max_source_positions": voxtral.FRAMES_PER_CHUNK,
            "initializer_range": 0.2,
        },
        text_config={
            "model_type": "llama",
            "vocab_size": VOCAB_SIZE,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "max_position_embeddings": 1024,
            "initializer_range": 0.2,
        },
        audio_token_id=AUDIO_TOKEN_ID,
        projector_hidden_act="gelu",
    )


def tiny_model(seed: int = 0, half: bool = False) -> Any:
    """A tiny random Voxtral on the CPU, in evaluation mode.

    Args:
        seed: Sets the random weights; the global random state is left as it was.
        half: Cast to fp16, except the encoder's position weights, which stay fp32 as
            ``from_pretrained`` keeps them.

    Returns:
        A ``VoxtralForConditionalGeneration``.
    """
    import torch
    from transformers import VoxtralForConditionalGeneration

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = VoxtralForConditionalGeneration(tiny_config())
    if half:
        model.half()
        model.model.audio_tower.embed_positions.float()
    return model.eval()


class FakeTokenizer:
    """Encodes the two answers to their fixed IDs, as Voxtral's tokenizer does to its own."""

    words: ClassVar[dict[str, int]] = {"Yes": YES_ID, "No": NO_ID}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert not add_special_tokens
        return [self.words[text]]

    def decode(self, ids: list[int]) -> str:
        names = {index: word for word, index in self.words.items()}
        return "".join(names[int(index)] for index in ids)


class FakeProcessor:
    """Stands in for Voxtral's processor: real log-mel features and fixed token IDs.

    Args:
        audio_tokens_per_chunk: Audio tokens per chunk (375, as in Voxtral); another
            number plants a fault.
    """

    audio_token_id = AUDIO_TOKEN_ID

    def __init__(self, audio_tokens_per_chunk: int = voxtral.TOKENS_PER_CHUNK) -> None:
        from transformers import WhisperFeatureExtractor

        self.audio_tokens_per_chunk = audio_tokens_per_chunk
        self.feature_extractor = WhisperFeatureExtractor(feature_size=128)
        self.tokenizer = FakeTokenizer()

    def features(self, audio: np.ndarray) -> Any:
        """Log-mel features of one clip, as Voxtral's processor makes them: (C, 128, 3000)."""
        from transformers.models.voxtral.processing_voxtral import (
            VoxtralProcessor,
            VoxtralProcessorKwargs,
        )

        settings = dict(VoxtralProcessorKwargs._defaults["audio_kwargs"])
        positions = settings.pop("max_source_positions")
        # The processor's own split, called on a stand-in that holds only the extractor.
        holder = SimpleNamespace(feature_extractor=self.feature_extractor)
        return VoxtralProcessor._retrieve_input_features(
            holder, [audio], positions, return_tensors="pt", **settings
        )

    def token_ids(self, n_chunks: int) -> list[int]:
        """The token IDs of a clip of n chunks."""
        audio = [AUDIO_TOKEN_ID] * (self.audio_tokens_per_chunk * n_chunks)
        return [BOS_ID, INST_ID, BEGIN_AUDIO_ID, *audio, *PROMPT_IDS, INST_END_ID]

    def apply_chat_template(self, conversation: list[dict[str, Any]]) -> Any:
        """The model input of the one user turn that ``voxtral.clip_inputs`` builds."""
        from transformers import BatchFeature

        (turn,) = conversation
        assert turn["role"] == "user"
        sound, text = turn["content"]
        assert sound["type"] == "audio"
        assert text == {"type": "text", "text": voxtral.PROMPT}
        audio, rate = sf.read(io.BytesIO(base64.b64decode(sound["base64"])), dtype="float32")
        assert rate == RATE
        features = self.features(audio)
        ids = self.token_ids(int(features.shape[0]))
        data = {"input_ids": [ids], "attention_mask": [[1] * len(ids)], "input_features": features}
        return BatchFeature(data=data, tensor_type="pt")


def same_bytes(first: Any, second: Any) -> bool:
    """True if two tensors have the same type, shape and bytes.

    Unlike ``torch.equal``, +0 and -0 differ and a NaN equals the same NaN.
    """
    import torch

    if first.dtype != second.dtype or first.shape != second.shape:
        return False

    def raw(tensor: Any) -> Any:
        return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)

    return torch.equal(raw(first), raw(second))


@contextmanager
def cached_encoder(model: Any) -> Iterator[None]:
    """Run a block with the pass-through in place of the encoder, then put the encoder back."""
    encoder = voxtral.use_cached_encoder(model)
    try:
        yield
    finally:
        voxtral.restore_encoder(model, encoder)


def save_checkpoint(model: Any, folder: Path) -> Path:
    """Save a copy of a model in bf16, as the pinned snapshot stores Voxtral's weights.

    Args:
        model: A Voxtral, for example ``tiny_model()``; it is not changed.
        folder: Where to save; it gets ``config.json`` and ``model.safetensors``.

    Returns:
        The folder.
    """
    import torch

    copy.deepcopy(model).to(torch.bfloat16).save_pretrained(folder)
    return folder


def edit_checkpoint(folder: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    """Change the tensors of a saved checkpoint in place, to plant a fault.

    Args:
        folder: A folder written by ``save_checkpoint``.
        edit: Gets the tensors by their names in the file (for example
            ``multi_modal_projector.linear_1.weight``) and changes the dictionary.
    """
    from safetensors.torch import load_file, save_file

    path = folder / CHECKPOINT
    tensors = load_file(path)
    edit(tensors)
    save_file(tensors, path, metadata={"format": "pt"})


def patch_processor(monkeypatch: pytest.MonkeyPatch, processor: Any = None) -> list[dict[str, Any]]:
    """Make ``AutoProcessor.from_pretrained`` return a processor stand-in.

    A tiny checkpoint has no tokenizer files, so ``voxtral.load_model`` needs this to load
    it.

    Args:
        monkeypatch: The test's monkeypatch fixture; the patch ends with the test.
        processor: What to return; a new ``FakeProcessor`` if None.

    Returns:
        The calls' arguments, one dictionary per call: ``path`` and the keyword arguments.
    """
    import transformers

    stand_in = FakeProcessor() if processor is None else processor
    calls: list[dict[str, Any]] = []

    def from_pretrained(path: Any, **kwargs: Any) -> Any:
        calls.append({"path": Path(path), **kwargs})
        return stand_in

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", from_pretrained)
    return calls
