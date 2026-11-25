import ast
import copy
import io
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from listening_beyond_words import voxtral
from listening_beyond_words.models import load_pins, local_snapshot
from listening_beyond_words.runinfo import StepError, environment
from voxtral_fakes import (
    ANSWER_IDS,
    AUDIO_TOKEN_ID,
    RATE,
    VOCAB_SIZE,
    WIDTH,
    FakeProcessor,
    cached_encoder,
    edit_checkpoint,
    patch_processor,
    same_bytes,
    save_checkpoint,
    tiny_model,
    tone,
)

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "configs" / "pilot" / "models.yaml"
DATA = ROOT / "configs" / "pilot" / "data.yaml"
PROTOCOL = ROOT / "docs" / "pilot_protocol.md"
HEAVY = ("torch", "transformers")
SECONDS = (1.0, 14.0, 37.4)  # one chunk, one chunk, two chunks
ENCODER = tuple(f"model.audio_tower.layers.{index}" for index in range(2))
LANGUAGE_MODEL = tuple(f"model.language_model.layers.{index}" for index in range(2))
ADAPTER = "model.multi_modal_projector"
# Every watched step, in the order the watch lists it: the encoder, then the adapter to the
# logits. A convolution, a norm, the adapter and the output layer have their output only.
ENCODER_STEPS = ("model.audio_tower.conv1", "model.audio_tower.conv2", *ENCODER)
ENCODER_STEPS = (*ENCODER_STEPS, "model.audio_tower.layer_norm")
LANGUAGE_MODEL_STEPS = (ADAPTER, *LANGUAGE_MODEL, "model.language_model.norm")
OUTPUT_ONLY = (*ENCODER_STEPS[:2], ENCODER_STEPS[-1], ADAPTER, LANGUAGE_MODEL_STEPS[-1], "lm_head")
CLEAN = dict.fromkeys(voxtral.PARTS, (0, 0))


class FakeTokenizer:
    """Encodes each known text to a fixed list of token IDs."""

    def __init__(self, vocab: dict[str, list[int]]) -> None:
        self.vocab = vocab

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert not add_special_tokens
        return self.vocab[text]


@pytest.fixture(scope="module")
def processor() -> FakeProcessor:
    return FakeProcessor()


@pytest.fixture(scope="module")
def inputs(processor: FakeProcessor) -> dict[float, Any]:
    """The model input of a synthetic clip of each length in SECONDS."""
    return {seconds: voxtral.clip_inputs(processor, tone(seconds)) for seconds in SECONDS}


@pytest.fixture(scope="module")
def models() -> dict[str, Any]:
    """The tiny Voxtral in fp32 and in fp16 (same seed); tests put back what they change."""
    return {"fp32": tiny_model(), "fp16": tiny_model(half=True)}


@pytest.fixture
def restore_numerics() -> Iterator[None]:
    """Put PyTorch's numeric settings back after a test, since set_numerics sets them."""
    import torch

    matmul = torch.backends.cuda.matmul
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        matmul.allow_fp16_reduced_precision_reduction,
        matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_num_threads(),
        matmul.allow_fp16_accumulation,
    )
    yield
    torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
    torch.backends.cudnn.deterministic = saved[2]
    torch.backends.cudnn.benchmark = saved[3]
    matmul.allow_fp16_reduced_precision_reduction = saved[4]
    matmul.allow_tf32 = saved[5]
    torch.backends.cudnn.allow_tf32 = saved[6]
    torch.set_num_threads(saved[7])
    matmul.allow_fp16_accumulation = saved[8]


def npy_round_trip(tensor: Any) -> Any:
    """The tensor written as .npy and read back, as a cache file holds it."""
    import torch

    buffer = io.BytesIO()
    np.save(buffer, tensor.cpu().numpy(), allow_pickle=False)
    return torch.from_numpy(np.load(io.BytesIO(buffer.getvalue()), allow_pickle=False))


def three_scores(model: Any, processor: Any, audio: np.ndarray) -> tuple[float, float, float]:
    """z from the audio, from the encoder output, and from that output after a .npy file."""
    clip = voxtral.clip_inputs(processor, audio)
    from_audio = voxtral.yes_no_score(model, clip, ANSWER_IDS)
    output = voxtral.encoder_output(model, clip)
    stored = npy_round_trip(output)
    prompt = voxtral.prompt_inputs(processor, voxtral.chunks(audio.size))
    with cached_encoder(model):
        from_output = voxtral.cached_score(model, prompt, output, ANSWER_IDS)
        from_file = voxtral.cached_score(model, prompt, stored, ANSWER_IDS)
    return from_audio, from_output, from_file


def every_layer(model: Any) -> dict[str, Any]:
    return {**voxtral.encoder_layers(model), **voxtral.language_model_layers(model, logits=True)}


def plant(module: Any, index: tuple[int, ...], value: float) -> Any:
    """Write one value into a module's output (the first part of a tuple), before the watch."""

    def hook(_module: Any, _args: Any, output: Any) -> Any:
        first = (output[0] if isinstance(output, tuple) else output).clone()
        first[index] = value
        return (first, *output[1:]) if isinstance(output, tuple) else first

    return module.register_forward_hook(hook)


def watched_score(model: Any, clip: Any, planted: tuple[Any, tuple[int, ...], float]) -> Any:
    """z with one value planted, and the watch on every layer and the logits."""
    handle = plant(*planted)
    try:
        with voxtral.watch_layers(every_layer(model)) as watch:
            z = voxtral.yes_no_score(model, clip, ANSWER_IDS)
    finally:
        handle.remove()
    return z, watch


def counts(watch: voxtral.LayerWatch, layer: str) -> dict[str, tuple[Any, Any]]:
    return {
        part: (numbers["at_clamp"], numbers["non_finite"])
        for part, numbers in watch.record()[layer].items()
    }


# The prompt and the answers --------------------------------------------------------------


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


def test_the_prompt_is_the_protocols_character_for_character() -> None:
    rows = [
        line
        for line in PROTOCOL.read_text(encoding="utf-8").splitlines()
        if line.startswith("| Prompt |")
    ]
    assert rows == [f'| Prompt | "{voxtral.PROMPT}" |']


def test_clip_inputs_needs_mono_audio_at_16_khz() -> None:
    with pytest.raises(ValueError, match="mono"):
        voxtral.clip_inputs(None, np.zeros((2, 16_000), dtype=np.float32))
    with pytest.raises(ValueError, match="16000 Hz"):
        voxtral.clip_inputs(None, np.zeros(8_000, dtype=np.float32), sampling_rate=8_000)


# Chunks and prompts ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("samples", "expected"),
    [(1, 1), (479_999, 1), (480_000, 1), (480_001, 2), (598_400, 2)],
)
def test_chunks_counts_the_30_s_pieces_the_processor_makes(samples: int, expected: int) -> None:
    assert voxtral.chunks(samples) == expected


def test_chunks_refuses_a_clip_without_samples() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        voxtral.chunks(0)


def test_the_processor_stand_in_splits_a_long_clip_into_chunks_as_voxtral_does(
    processor: FakeProcessor, inputs: dict[float, Any]
) -> None:
    import torch

    assert tuple(inputs[14.0]["input_features"].shape) == (1, 128, 3000)
    assert tuple(inputs[37.4]["input_features"].shape) == (2, 128, 3000)
    assert inputs[37.4]["input_features"].dtype == torch.float32
    assert tuple(inputs[14.0]["input_ids"].shape) == (1, 397)
    assert tuple(inputs[37.4]["input_ids"].shape) == (1, 772)
    # The feature extractor's own defaults would cut the clip to its first 30 s.
    alone = processor.feature_extractor(tone(37.4), sampling_rate=RATE, return_tensors="pt")
    assert tuple(alone["input_features"].shape) == (1, 128, 3000)


@pytest.mark.parametrize(
    ("seconds", "n_chunks", "length"),
    [
        (1.0, 1, 397),
        (14.0, 1, 397),
        (29.99, 1, 397),
        (30.0, 1, 397),
        (30.01, 2, 772),
        (37.4, 2, 772),
    ],
)
def test_prompt_inputs_gives_the_ids_of_any_clip_of_its_chunk_count(
    processor: FakeProcessor, seconds: float, n_chunks: int, length: int
) -> None:
    import torch

    audio = tone(seconds, seed=2)
    assert voxtral.chunks(audio.size) == n_chunks
    clip = voxtral.clip_inputs(processor, audio)
    prompt = voxtral.prompt_inputs(processor, n_chunks)
    assert set(prompt) == {"input_ids", "attention_mask"}
    assert tuple(prompt["input_ids"].shape) == (1, length)
    assert torch.equal(prompt["input_ids"], clip["input_ids"])
    assert torch.equal(prompt["attention_mask"], clip["attention_mask"])


def test_prompt_inputs_refuses_a_count_of_audio_tokens_other_than_375_per_chunk() -> None:
    with pytest.raises(StepError, match="gave 748 audio tokens"):
        voxtral.prompt_inputs(FakeProcessor(audio_tokens_per_chunk=374), 2)
    with pytest.raises(ValueError, match="at least one chunk"):
        voxtral.prompt_inputs(FakeProcessor(), 0)


# Numeric settings --------------------------------------------------------------------------


def test_clamp_abs_is_the_encoders_clamp_as_fp16_stores_it(models: dict[str, Any]) -> None:
    import torch

    assert torch.tensor(65_504 - 1_000, dtype=torch.float16).item() == voxtral.CLAMP_ABS
    assert torch.finfo(torch.float16).max - 1_000 == 64_504
    # An encoder layer in fp16 clamps values that would overflow at exactly this size.
    hidden = torch.zeros(1, voxtral.FRAMES_PER_CHUNK, WIDTH, dtype=torch.float16)
    hidden[0, 3, 5], hidden[0, 4, 6] = 65_000.0, -65_000.0
    with torch.inference_mode():
        output = models["fp16"].model.audio_tower.layers[0](hidden, attention_mask=None)
    assert float(output.max()) == voxtral.CLAMP_ABS
    assert float(output.min()) == -voxtral.CLAMP_ABS


def test_set_numerics_sets_and_returns_each_flag(
    restore_numerics: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Every flag starts the other way round.
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(1)
    record = voxtral.set_numerics(2)
    assert record == {
        "deterministic_algorithms": True,
        "deterministic_warn_only": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "fp16_reduced_precision_reduction": False,
        "fp16_accumulation": False,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "torch_threads": 2,
        "omp_num_threads": "2",
        "cublas_workspace_config": ":4096:8",
    }
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert not torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    assert not torch.backends.cuda.matmul.allow_fp16_accumulation
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    assert torch.get_num_threads() == 2
    assert json.loads(json.dumps(record)) == record  # fits run.json


@pytest.mark.parametrize(
    ("threads", "error"), [(0, ValueError), (True, ValueError), (2.0, TypeError)]
)
def test_set_numerics_refuses_a_thread_count_that_is_not_a_whole_number_from_1(
    restore_numerics: None, threads: Any, error: type[Exception]
) -> None:
    with pytest.raises(error):
        voxtral.set_numerics(threads)


# The cache path ----------------------------------------------------------------------------


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
@pytest.mark.parametrize("seconds", SECONDS)
def test_z_from_the_cache_equals_z_from_the_audio_bit_for_bit(
    models: dict[str, Any], processor: FakeProcessor, precision: str, seconds: float
) -> None:
    from_audio, from_output, from_file = three_scores(models[precision], processor, tone(seconds))
    assert math.isfinite(from_audio)
    assert from_output.hex() == from_audio.hex()
    assert from_file.hex() == from_audio.hex()


def test_the_score_depends_on_the_clip(models: dict[str, Any], inputs: dict[float, Any]) -> None:
    scores = {voxtral.yes_no_score(models["fp32"], inputs[s], ANSWER_IDS) for s in SECONDS}
    assert len(scores) == len(SECONDS)


class FixedLogits:
    """Stands in for the model: its logits at the last position are fixed fp16 values."""

    def __init__(self, yes: float, no: float) -> None:
        import torch

        self.device = torch.device("cpu")
        self.model = SimpleNamespace(audio_tower=None)
        self.logits = torch.tensor([[[yes, no]]], dtype=torch.float16)

    def __call__(self, **kwargs: Any) -> Any:
        assert kwargs["logits_to_keep"] == 1
        return SimpleNamespace(logits=self.logits)


def test_z_is_the_difference_of_the_two_logits_in_float32() -> None:
    import torch

    model = FixedLogits(2048.0, -0.5)  # in fp16, 2048.5 would round to 2048
    clip = {"input_ids": torch.zeros(1, 3, dtype=torch.long)}
    assert voxtral.yes_no_score(model, clip, (0, 1)) == 2048.5
    assert voxtral.yes_no_score(model, clip, (1, 0)) == -2048.5


@pytest.mark.parametrize("seconds", [14.0, 37.4])
def test_encoder_output_is_the_encoder_output_inside_the_model(
    models: dict[str, Any], inputs: dict[float, Any], seconds: float
) -> None:
    import torch

    model = models["fp16"]
    clip = inputs[seconds]
    inside: list[Any] = []
    handle = model.model.audio_tower.register_forward_hook(
        lambda _module, _args, output: inside.append(output.last_hidden_state.clone())
    )
    try:
        voxtral.yes_no_score(model, clip, ANSWER_IDS)
    finally:
        handle.remove()
    output = voxtral.encoder_output(model, clip)
    n_chunks = voxtral.chunks(round(seconds * RATE))
    assert tuple(output.shape) == (n_chunks, voxtral.FRAMES_PER_CHUNK, WIDTH)
    assert output.dtype == torch.float16
    assert len(inside) == 1
    assert same_bytes(output, inside[0])


def test_the_adapter_reads_four_encoder_vectors_side_by_side_per_audio_token(
    models: dict[str, Any], inputs: dict[float, Any]
) -> None:
    model = models["fp32"]
    clip = inputs[37.4]
    output = voxtral.encoder_output(model, clip)
    rows: list[Any] = []
    handle = model.model.multi_modal_projector.register_forward_pre_hook(
        lambda _module, args: rows.append(args[0].clone())
    )
    try:
        voxtral.yes_no_score(model, clip, ANSWER_IDS)
    finally:
        handle.remove()
    (adapter_input,) = rows
    assert tuple(adapter_input.shape) == (2 * voxtral.TOKENS_PER_CHUNK, 4 * WIDTH)
    assert same_bytes(adapter_input[0], output[0, 0:4].reshape(-1))  # frames 0-3
    assert same_bytes(adapter_input[1], output[0, 4:8].reshape(-1))  # frames 4-7
    assert same_bytes(adapter_input[375], output[1, 0:4].reshape(-1))  # the second chunk


def test_lm_pass_runs_the_language_model_but_never_its_output_layer(
    models: dict[str, Any], processor: FakeProcessor, inputs: dict[float, Any]
) -> None:
    model = models["fp16"]
    output = voxtral.encoder_output(model, inputs[37.4])
    prompt = voxtral.prompt_inputs(processor, 2)
    logits: list[Any] = []
    handle = model.lm_head.register_forward_hook(lambda _module, _args, out: logits.append(out))
    try:
        with (
            cached_encoder(model),
            voxtral.watch_layers(voxtral.language_model_layers(model)) as watch,
        ):
            returned = voxtral.lm_pass(model, prompt, output)
    finally:
        handle.remove()
    assert returned is None
    assert logits == []
    assert list(watch.record()) == list(LANGUAGE_MODEL_STEPS)  # no output layer: no logits
    calls = {numbers["calls"] for parts in watch.record().values() for numbers in parts.values()}
    assert calls == {1}


def test_cached_score_and_lm_pass_refuse_the_real_encoder(
    models: dict[str, Any], processor: FakeProcessor, inputs: dict[float, Any]
) -> None:
    model = models["fp16"]
    output = voxtral.encoder_output(model, inputs[14.0])
    prompt = voxtral.prompt_inputs(processor, 1)
    with pytest.raises(StepError, match="call use_cached_encoder first"):
        voxtral.cached_score(model, prompt, output, ANSWER_IDS)
    with pytest.raises(StepError, match="call use_cached_encoder first"):
        voxtral.lm_pass(model, prompt, output)


def test_cached_score_refuses_a_stored_output_that_does_not_fit(
    models: dict[str, Any], processor: FakeProcessor, inputs: dict[float, Any]
) -> None:
    model = models["fp16"]
    one = voxtral.encoder_output(model, inputs[14.0])
    two = voxtral.encoder_output(model, inputs[37.4])
    prompt_one = voxtral.prompt_inputs(processor, 1)
    prompt_two = voxtral.prompt_inputs(processor, 2)
    with cached_encoder(model):
        with pytest.raises(StepError, match=re.escape("holds 375 audio tokens, but the stored")):
            voxtral.cached_score(model, prompt_one, two, ANSWER_IDS)
        with pytest.raises(StepError, match=re.escape("holds 750 audio tokens, but the stored")):
            voxtral.cached_score(model, prompt_two, one, ANSWER_IDS)
        with pytest.raises(StepError, match="has shape"):
            voxtral.cached_score(model, prompt_one, one[:, :, : WIDTH // 2], ANSWER_IDS)
        with pytest.raises(StepError, match="has shape"):
            voxtral.cached_score(model, prompt_one, one[:, :-1], ANSWER_IDS)
        with pytest.raises(StepError, match="has shape"):
            voxtral.cached_score(model, prompt_one, one[0], ANSWER_IDS)
        with pytest.raises(StepError, match="float32"):
            voxtral.cached_score(model, prompt_one, one.float(), ANSWER_IDS)
        with pytest.raises(StepError, match=re.escape("holds 375 audio tokens, but the stored")):
            voxtral.lm_pass(model, prompt_one, two)


def test_the_audio_paths_refuse_the_pass_through_and_the_swap_refuses_to_repeat(
    models: dict[str, Any], inputs: dict[float, Any]
) -> None:
    model = models["fp16"]
    clip = inputs[1.0]
    encoder = model.model.audio_tower
    with cached_encoder(model):
        with pytest.raises(StepError, match="pass-through is in place"):
            voxtral.encoder_output(model, clip)
        with pytest.raises(StepError, match="pass-through is in place"):
            voxtral.yes_no_score(model, clip, ANSWER_IDS)
        with pytest.raises(StepError, match="pass-through is in place"):
            voxtral.encoder_layers(model)
        with pytest.raises(StepError, match="in place already"):
            voxtral.use_cached_encoder(model)
        with pytest.raises(StepError, match="the encoder that use_cached_encoder returned"):
            voxtral.restore_encoder(model, model.model.audio_tower)
    with pytest.raises(StepError, match="holds its own encoder already"):
        voxtral.restore_encoder(model, encoder)
    assert model.model.audio_tower is encoder


def test_swapping_the_encoder_out_and_back_changes_no_parameter(
    models: dict[str, Any], processor: FakeProcessor, inputs: dict[float, Any]
) -> None:
    model = models["fp16"]
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    encoder = model.model.audio_tower
    output = voxtral.encoder_output(model, inputs[1.0])
    with cached_encoder(model):
        assert not any(
            name.startswith("model.audio_tower.") for name, _ in model.named_parameters()
        )
        assert not any(module.training for module in model.modules())
        voxtral.cached_score(model, voxtral.prompt_inputs(processor, 1), output, ANSWER_IDS)
    assert model.model.audio_tower is encoder
    after = model.state_dict()
    assert list(after) == list(before)
    assert all(same_bytes(after[name], before[name]) for name in before)


# The overflow watch ------------------------------------------------------------------------


def test_the_watch_reads_every_layer_and_part_without_changing_the_score(
    models: dict[str, Any], inputs: dict[float, Any]
) -> None:
    model = models["fp16"]
    clip = inputs[37.4]
    z = voxtral.yes_no_score(model, clip, ANSWER_IDS)
    with voxtral.watch_layers(every_layer(model)) as watch:
        watched = voxtral.yes_no_score(model, clip, ANSWER_IDS)
        voxtral.yes_no_score(model, clip, ANSWER_IDS)
    assert watched.hex() == z.hex()
    record = watch.record()
    assert list(record) == [*ENCODER_STEPS, *LANGUAGE_MODEL_STEPS, "lm_head"]
    for name, parts in record.items():
        assert tuple(parts) == (("output",) if name in OUTPUT_ONLY else voxtral.PARTS)
        for numbers in parts.values():
            assert numbers["calls"] == 2
            assert numbers["at_clamp"] == numbers["non_finite"] == 0
            assert 0 < numbers["max_abs"] < voxtral.CLAMP_ABS
    assert watch.flagged() == []
    maxima = sorted((n["max_abs"] for p in record.values() for n in p.values()), reverse=True)
    assert [value for _, value in watch.largest(3)] == maxima[:3]
    assert json.loads(json.dumps(record)) == record  # fits run.json
    # Leaving the block removed every hook; the counts stay.
    assert not any(module._forward_hooks for module in model.modules())
    voxtral.yes_no_score(model, clip, ANSWER_IDS)
    assert watch.record() == record


def test_the_watch_counts_a_planted_inf_in_its_encoder_part_and_the_clamp_it_becomes(
    models: dict[str, Any], inputs: dict[float, Any]
) -> None:
    model = models["fp16"]
    feed_forward = model.model.audio_tower.layers[1].fc2
    _, watch = watched_score(model, inputs[1.0], (feed_forward, (0, 7, 3), math.inf))
    assert counts(watch, ENCODER[0]) == CLEAN
    assert counts(watch, ENCODER[1]) == {
        "output": (1, 0),
        "attention": (0, 0),
        "feed_forward": (0, 1),
    }
    # The layer clamps the inf to 64,512: finite, but the watch still sees it.
    record = watch.record()[ENCODER[1]]
    assert record["output"]["max_abs"] == voxtral.CLAMP_ABS
    assert 0 < record["feed_forward"]["max_abs"] < voxtral.CLAMP_ABS  # the largest finite value
    assert watch.flagged() == [f"{ENCODER[1]}/output", f"{ENCODER[1]}/feed_forward"]


def test_the_watch_counts_a_planted_64512_in_its_language_model_part(
    models: dict[str, Any], inputs: dict[float, Any]
) -> None:
    model = models["fp16"]
    attention = model.model.language_model.layers[0].self_attn
    planted = (attention, (0, 100, 5), float(voxtral.CLAMP_ABS))
    _, watch = watched_score(model, inputs[1.0], planted)
    assert counts(watch, ENCODER[0]) == counts(watch, ENCODER[1]) == CLEAN
    assert counts(watch, LANGUAGE_MODEL[0])["attention"] == (1, 0)
    assert watch.record()[LANGUAGE_MODEL[0]]["attention"]["max_abs"] == voxtral.CLAMP_ABS
    assert f"{LANGUAGE_MODEL[0]}/attention" in watch.flagged()


def test_the_watch_counts_non_finite_logits_and_the_score_shows_them(
    models: dict[str, Any], inputs: dict[float, Any]
) -> None:
    model = models["fp16"]
    feed_forward = model.model.language_model.layers[1].mlp
    z, watch = watched_score(model, inputs[1.0], (feed_forward, (0, -1, 5), math.inf))
    assert math.isnan(z)
    assert counts(watch, LANGUAGE_MODEL[0]) == CLEAN
    assert counts(watch, LANGUAGE_MODEL[1]) == {
        "output": (0, 1),
        "attention": (0, 0),
        "feed_forward": (0, 1),
    }
    # The final norm turns the inf into one NaN, which reaches every logit.
    assert counts(watch, "model.language_model.norm") == {"output": (0, 1)}
    assert counts(watch, "lm_head") == {"output": (0, VOCAB_SIZE)}
    assert watch.record()["lm_head"]["output"]["max_abs"] == 0.0  # no finite logit
    assert watch.flagged() == [
        f"{LANGUAGE_MODEL[1]}/output",
        f"{LANGUAGE_MODEL[1]}/feed_forward",
        "model.language_model.norm/output",
        "lm_head/output",
    ]


@pytest.mark.parametrize(
    ("step", "index"),
    [
        ("model.audio_tower.conv1", (0, 5, 7)),
        ("model.audio_tower.layer_norm", (0, 9, 3)),
        (ADAPTER, (0, 5)),
    ],
)
def test_the_watch_names_an_overflow_where_it_starts(
    models: dict[str, Any], inputs: dict[float, Any], step: str, index: tuple[int, ...]
) -> None:
    # Each of these steps has no watched part of its own: its output is watched.
    model = models["fp16"]
    module = dict(model.named_modules())[step]
    _, watch = watched_score(model, inputs[1.0], (module, index, math.inf))
    assert watch.flagged()[0] == f"{step}/output"
    assert watch.record()[step]["output"]["non_finite"] == 1
    # Everything watched before it stayed clean.
    before = list(watch.record())[: list(watch.record()).index(step)]
    assert not any(name.split("/")[0] in before for name in watch.flagged())


def test_a_peft_model_scores_from_the_cache_as_from_the_audio_with_its_lora_layers(
    processor: FakeProcessor, inputs: dict[float, Any]
) -> None:
    import torch
    from peft import LoraConfig, get_peft_model

    base = tiny_model(half=True)
    clip = inputs[37.4]
    untrained = voxtral.yes_no_score(base, clip, ANSWER_IDS)
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=r"model\.language_model\.layers\.\d+\.self_attn\.(q|v)_proj",
        modules_to_save=["multi_modal_projector"],
    )
    model = get_peft_model(base, config).eval()
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:  # trained, as if: zero at the start
                parameter.normal_(0.0, 0.5, generator=generator)
    from_audio = voxtral.yes_no_score(model, clip, ANSWER_IDS)
    output = voxtral.encoder_output(model, clip)
    prompt = voxtral.prompt_inputs(processor, 2)
    encoder = voxtral.use_cached_encoder(model)
    try:
        with voxtral.watch_layers(voxtral.language_model_layers(model, logits=True)) as watch:
            from_cache = voxtral.cached_score(model, prompt, output, ANSWER_IDS)
        voxtral.lm_pass(model, prompt, output)
    finally:
        voxtral.restore_encoder(model, encoder)
    assert math.isfinite(from_audio)
    assert from_cache.hex() == from_audio.hex()
    assert from_audio != untrained  # the LoRA layers took part
    assert list(watch.record()) == [*LANGUAGE_MODEL_STEPS, "lm_head"]
    assert model.get_base_model().model.audio_tower is encoder


def test_the_watch_keeps_the_largest_finite_value_and_adds_up_its_counts_over_calls() -> None:
    import torch

    layer = torch.nn.Identity()
    with voxtral.watch_layers({"layer": layer}) as watch:
        layer(torch.tensor([3.0, 64_512.0, -math.inf], dtype=torch.float16))
        layer(torch.tensor([-2.0, math.nan], dtype=torch.float16))
        layer(torch.tensor([], dtype=torch.float16))
    assert watch.record() == {
        "layer": {"output": {"calls": 3, "max_abs": 64_512.0, "at_clamp": 1, "non_finite": 2}}
    }
    assert watch.flagged() == ["layer/output"]
    assert watch.largest() == [("layer/output", 64_512.0)]


def test_the_watch_also_reads_outside_inference_mode(models: dict[str, Any]) -> None:
    import torch

    layer = models["fp32"].model.language_model.layers[0].mlp
    with voxtral.watch_layers({"mlp": layer}) as watch:
        with torch.inference_mode():
            layer(torch.ones(1, 3, 64))
        with torch.no_grad():
            layer(torch.ones(1, 3, 64))
        layer(torch.ones(1, 3, 64, requires_grad=True)).sum().backward()
    assert watch.record()["mlp"]["output"]["calls"] == 3


# Loading -----------------------------------------------------------------------------------


def test_check_parameters_wants_fp16_with_fp32_position_weights_and_gives_the_largest(
    models: dict[str, Any],
) -> None:
    model = models["fp16"]
    largest = voxtral.check_parameters(model)
    assert largest == max(float(p.detach().abs().max()) for p in model.parameters())
    # .half() in memory puts the position weights in fp16 too, unlike from_pretrained.
    halved = copy.deepcopy(models["fp32"]).half()
    with pytest.raises(StepError, match=r"embed_positions\.weight \(torch\.float16\)"):
        voxtral.check_parameters(halved)
    with pytest.raises(StepError, match="not in fp16"):
        voxtral.check_parameters(models["fp32"])


def test_load_model_loads_a_bf16_snapshot_offline_in_fp16_with_fp32_position_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    models: dict[str, Any],
    inputs: dict[float, Any],
) -> None:
    import torch
    from transformers import VoxtralForConditionalGeneration

    original = models["fp32"]
    snapshot = save_checkpoint(original, tmp_path / "snapshot")
    processor_calls = patch_processor(monkeypatch)
    model_calls: list[dict[str, Any]] = []
    from_pretrained = VoxtralForConditionalGeneration.from_pretrained

    def spy(*args: Any, **kwargs: Any) -> Any:
        model_calls.append(kwargs)
        return from_pretrained(*args, **kwargs)

    monkeypatch.setattr(VoxtralForConditionalGeneration, "from_pretrained", spy)
    loaded = voxtral.load_model(snapshot, device="cpu")
    assert processor_calls == [{"path": snapshot, "local_files_only": True}]
    assert [call["local_files_only"] for call in model_calls] == [True]
    assert [call["dtype"] for call in model_calls] == [torch.float16]
    assert isinstance(loaded.processor, FakeProcessor)
    dtypes = {name: weight.dtype for name, weight in loaded.model.named_parameters()}
    assert dtypes.pop("model.audio_tower.embed_positions.weight") == torch.float32
    assert set(dtypes.values()) == {torch.float16}
    assert not any(module.training for module in loaded.model.modules())
    # Each weight is the original rounded to bf16 in the file, then cast to its format.
    weights = dict(original.named_parameters())
    for name, weight in loaded.model.named_parameters():
        assert same_bytes(weight, weights[name].detach().to(torch.bfloat16).to(weight.dtype))
    largest = max(float(weight.detach().abs().max()) for weight in loaded.model.parameters())
    assert loaded.largest_weight == largest
    assert math.isfinite(voxtral.yes_no_score(loaded.model, inputs[1.0], ANSWER_IDS))


def remove_a_weight(tensors: dict[str, Any]) -> None:
    del tensors["multi_modal_projector.linear_1.weight"]


def add_a_weight(tensors: dict[str, Any]) -> None:
    import torch

    tensors["multi_modal_projector.extra.weight"] = torch.zeros(2, dtype=torch.bfloat16)


def overflow_in_fp16(tensors: dict[str, Any]) -> None:
    import torch

    weight = tensors["multi_modal_projector.linear_2.weight"]
    weight[0, 0] = 1.0e5  # finite in bf16, but above fp16's largest value of 65,504
    assert bool(torch.isfinite(weight).all())


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (remove_a_weight, "missing keys: 1 (model.multi_modal_projector.linear_1.weight)"),
        (add_a_weight, "unexpected keys: 1 (model.multi_modal_projector.extra.weight)"),
        (overflow_in_fp16, "non-finite value: 1 (model.multi_modal_projector.linear_2.weight)"),
    ],
)
def test_load_model_refuses_a_missing_an_extra_and_a_non_finite_weight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    models: dict[str, Any],
    edit: Any,
    message: str,
) -> None:
    snapshot = save_checkpoint(models["fp32"], tmp_path / "snapshot")
    edit_checkpoint(snapshot, edit)
    patch_processor(monkeypatch)
    with pytest.raises(StepError, match=re.escape(message)):
        voxtral.load_model(snapshot, device="cpu")


# Imports -----------------------------------------------------------------------------------


def imported(node: ast.AST) -> list[str]:
    """The top-level packages that an import statement names."""
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module:
        return [node.module.split(".")[0]]
    return []


def test_voxtral_imports_torch_and_transformers_inside_functions_only() -> None:
    tree = ast.parse(Path(voxtral.__file__).read_text(encoding="utf-8"))
    assert {name for node in tree.body for name in imported(node)}.isdisjoint(HEAVY)
    code = (
        "import sys\n"
        "import listening_beyond_words.voxtral\n"
        f"print(' '.join(name for name in {HEAVY!r} if name in sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.split() == []


# GPU tests on OMNI: the real model, synthetic clips only -----------------------------------


def cpu_threads() -> int:
    return int(yaml.safe_load(DATA.read_text(encoding="utf-8"))["encoder_cache"]["cpu_threads"])


def show(capsys: pytest.CaptureFixture[str], line: str) -> None:
    """Write a line to the job's log, past pytest's capture."""
    with capsys.disabled():
        sys.stdout.write(line + "\n")


@pytest.fixture(scope="module")
def real() -> voxtral.LoadedModel:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        pytest.skip("HF_HOME is not set; run the GPU tests in scripts/slurm/smoke_test.sbatch")
    voxtral.set_numerics(cpu_threads())  # first, as every step-7 job does
    snapshot = local_snapshot(load_pins(PINS)["voxtral"], Path(hf_home))
    return voxtral.load_model(snapshot)


@pytest.mark.gpu
def test_yes_and_no_are_single_tokens(real: voxtral.LoadedModel) -> None:
    tokenizer = real.processor.tokenizer
    yes_id, no_id = voxtral.answer_token_ids(tokenizer)
    assert tokenizer.decode([yes_id]) == "Yes"
    assert tokenizer.decode([no_id]) == "No"


@pytest.mark.gpu
def test_voxtral_loads_in_fp16_with_fp32_position_weights_and_scores_clips(
    real: voxtral.LoadedModel,
) -> None:
    import torch

    model = real.model
    dtypes = {name: weight.dtype for name, weight in model.named_parameters()}
    assert dtypes.pop("model.audio_tower.embed_positions.weight") == torch.float32
    assert set(dtypes.values()) == {torch.float16}
    assert model.device.type == "cuda"
    assert 0 < real.largest_weight < 65_504
    answer_ids = voxtral.answer_token_ids(real.processor.tokenizer)
    scores = [
        voxtral.yes_no_score(
            model, voxtral.clip_inputs(real.processor, tone(3.0, seed)), answer_ids
        )
        for seed in (1, 2)
    ]
    assert all(math.isfinite(z) for z in scores)
    assert scores[0] != scores[1], "the score does not depend on the audio"


@pytest.mark.gpu
@pytest.mark.parametrize("seconds", [1.0, 14.0, 29.99, 30.0, 37.4])
def test_z_from_the_cache_equals_z_from_the_audio_bit_for_bit_on_the_gpu(
    real: voxtral.LoadedModel, seconds: float, capsys: pytest.CaptureFixture[str]
) -> None:
    model, processor = real.model, real.processor
    answer_ids = voxtral.answer_token_ids(processor.tokenizer)
    audio = tone(seconds)
    clip = voxtral.clip_inputs(processor, audio)
    with voxtral.watch_layers(every_layer(model)) as watch:
        watched = voxtral.yes_no_score(model, clip, answer_ids)
    from_audio = voxtral.yes_no_score(model, clip, answer_ids)
    output = voxtral.encoder_output(model, clip)
    stored = npy_round_trip(output)
    prompt = voxtral.prompt_inputs(processor, voxtral.chunks(audio.size))
    with cached_encoder(model):
        from_output = voxtral.cached_score(model, prompt, output, answer_ids)
        from_file = voxtral.cached_score(model, prompt, stored, answer_ids)
    show(
        capsys,
        f"{seconds:5.2f} s: z {from_audio:+.6f}, chunks {output.shape[0]}, flagged "
        f"{len(watch.flagged())}, largest {watch.largest(2)}",
    )
    assert math.isfinite(from_audio)
    assert watched.hex() == from_audio.hex()
    assert from_output.hex() == from_audio.hex()
    assert from_file.hex() == from_audio.hex()


@pytest.mark.gpu
def test_encoder_output_is_the_one_inside_the_model_and_repeats_its_bits(
    real: voxtral.LoadedModel,
) -> None:
    model = real.model
    clip = voxtral.clip_inputs(real.processor, tone(37.4))
    answer_ids = voxtral.answer_token_ids(real.processor.tokenizer)
    inside: list[Any] = []
    handle = model.model.audio_tower.register_forward_hook(
        lambda _module, _args, output: inside.append(output.last_hidden_state.clone())
    )
    try:
        voxtral.yes_no_score(model, clip, answer_ids)
    finally:
        handle.remove()
    first = voxtral.encoder_output(model, clip)
    second = voxtral.encoder_output(model, clip)
    assert tuple(first.shape) == (2, voxtral.FRAMES_PER_CHUNK, 1280)
    assert same_bytes(first, inside[0])
    assert same_bytes(first, second)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("seconds", "length"),
    [(1.0, 397), (14.0, 397), (29.99, 397), (30.0, 397), (30.01, 772), (37.4, 772)],
)
def test_prompt_inputs_gives_the_processors_own_ids(
    real: voxtral.LoadedModel, seconds: float, length: int
) -> None:
    import torch

    audio = tone(seconds, seed=3)
    clip = voxtral.clip_inputs(real.processor, audio)
    prompt = voxtral.prompt_inputs(real.processor, voxtral.chunks(audio.size))
    assert tuple(prompt["input_ids"].shape) == (1, length)
    assert torch.equal(prompt["input_ids"], clip["input_ids"])
    assert torch.equal(prompt["attention_mask"], clip["attention_mask"])


@pytest.mark.gpu
@pytest.mark.parametrize("seconds", [14.0, 37.4])
def test_the_processor_stand_in_builds_the_real_processors_features(
    real: voxtral.LoadedModel, seconds: float
) -> None:
    # The Mac tests rest on the stand-in; here it meets the real processor.
    audio = tone(seconds, seed=4)
    features = voxtral.clip_inputs(real.processor, audio)["input_features"]
    assert same_bytes(FakeProcessor().features(audio), features)
    assert real.processor.audio_token_id == real.model.config.audio_token_id == AUDIO_TOKEN_ID


@pytest.mark.gpu
def test_log_mel_features_repeat_at_the_jobs_thread_count(
    real: voxtral.LoadedModel, capsys: pytest.CaptureFixture[str]
) -> None:
    threads = cpu_threads()
    audio = tone(37.4)
    try:
        voxtral.set_numerics(threads)
        reference = voxtral.clip_inputs(real.processor, audio)["input_features"]
        again = voxtral.clip_inputs(real.processor, audio)["input_features"]
        others = {}
        for count in (1, 2, 8):
            voxtral.set_numerics(count)
            features = voxtral.clip_inputs(real.processor, audio)["input_features"]
            others[count] = same_bytes(features, reference)
    finally:
        voxtral.set_numerics(threads)
    # Only the jobs' own count must repeat; the others are printed for the record.
    show(
        capsys,
        f"log-mel features equal to {threads} threads: {others}; CPU {environment([])['cpu']}",
    )
    assert same_bytes(again, reference)
