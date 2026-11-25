"""Voxtral Mini 3B: loading in fp16, the yes/no score of a clip, and the cache path.

The model hears a clip followed by the fixed prompt. A clip's score is
z = logit("Yes") - logit("No") at the first answer position, so "Yes" and "No"
must each be a single token. A participant's score is the mean z over their
segments.

The processor pads a clip with silence to a whole number of 30 s chunks (``chunks``)
and turns each chunk into log-mel features of shape (128, 3000). The audio encoder turns
each chunk into 1,500 vectors of width W (``audio_config.hidden_size``: 1,280), and the
adapter reads them 4 by 4, one per audio token, so a clip of C chunks holds 375 * C
audio tokens. The encoder is frozen (pilot protocol, section "Encoder caches and
zero-shot scores"), so its output before the adapter is computed once per clip
(``encoder_output``) and stored. Scoring from the store swaps the encoder for a
pass-through (``use_cached_encoder``) and sends the stored output through the library's
own adapter, token placement and language model (``cached_score``). z from the cache
then equals z from the audio bit for bit, when both run with the same numeric settings
(``set_numerics``) on the same GPU and CPU type. ``watch_layers`` counts fp16 overflow
in the parts it watches: every step of the encoder (``encoder_layers``), and every step
from the adapter to the logits (``language_model_layers``).

Every function that takes the model also takes it inside a PEFT wrapper (step 8's
trained arms): it then works on the wrapped model, whose modules hold the LoRA layers and
the trained adapter (``get_base_model``).

torch and transformers are imported inside the functions that use them.
"""

import base64
import functools
import io
import operator
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import soundfile as sf

from listening_beyond_words.runinfo import StepError, deterministic_torch

if TYPE_CHECKING:
    import torch

PROMPT: Final = (
    "Based on this speech, does the speaker have elevated depressive symptoms? Answer Yes or No."
)
ANSWERS: Final = ("Yes", "No")
SAMPLING_RATE: Final = 16_000
CHUNK_SAMPLES: Final = 480_000  # 30 s at 16 kHz: the processor pads each clip to a multiple
FRAMES_PER_CHUNK: Final = 1_500  # encoder output vectors per chunk (max_source_positions)
TOKENS_PER_CHUNK: Final = 375  # audio tokens per chunk: the adapter reads the vectors 4 by 4
# Each encoder layer clamps its fp16 output at 65,504 - 1,000, which fp16 stores as 64,512
# (modeling_voxtral.py, VoxtralEncoderLayer); the language model has no clamp.
CLAMP_ABS: Final = 64_512
# The library keeps the encoder's position weights in fp32 in an fp16 model
# (_keep_in_fp32_modules_strict of VoxtralForConditionalGeneration).
FP32_WEIGHTS: Final = "embed_positions"
PARTS: Final = ("output", "attention", "feed_forward")
LOADING_PROBLEMS: Final = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")


def answer_token_ids(tokenizer: Any) -> tuple[int, int]:
    """Return the token IDs of "Yes" and "No".

    Args:
        tokenizer: The model's tokenizer; it needs ``encode(text, add_special_tokens=False)``.

    Returns:
        The IDs of "Yes" and "No", in this order.

    Raises:
        ValueError: If an answer is not exactly one token, or both answers share a token.
    """
    ids: list[int] = []
    for answer in ANSWERS:
        tokens = list(tokenizer.encode(answer, add_special_tokens=False))
        if len(tokens) != 1:
            raise ValueError(f"{answer!r} must be one token, but is {len(tokens)}: {tokens}")
        ids.append(int(tokens[0]))
    yes_id, no_id = ids
    if yes_id == no_id:
        raise ValueError(f"'Yes' and 'No' share the token {yes_id}")
    return yes_id, no_id


# Loading -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """Voxtral as ``load_model`` returns it.

    Attributes:
        model: ``VoxtralForConditionalGeneration`` in fp16, its position weights in fp32,
            in evaluation mode.
        processor: Its processor.
        largest_weight: The largest |value| of any parameter after the cast to fp16, for
            ``run.json``.
    """

    model: Any
    processor: Any
    largest_weight: float


def load_model(snapshot: Path, device: str = "cuda") -> LoadedModel:
    """Load the model in fp16 from a fetched snapshot, offline, and check what was loaded.

    The snapshot holds bf16 weights, which load as fp16. The load is refused if the
    loading report lists a missing, unexpected or mismatched weight, or if a parameter is
    not in the expected format or not finite after the cast (``check_parameters``).

    Args:
        snapshot: Folder of the pinned snapshot (see ``models.local_snapshot``).
        device: Where the model runs.

    Returns:
        The model in evaluation mode, its processor and the largest |weight|.

    Raises:
        StepError: If a check fails.
    """
    import torch
    from transformers import AutoProcessor, VoxtralForConditionalGeneration

    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    # Any: with output_loading_info, from_pretrained returns the model and its report.
    result: Any = VoxtralForConditionalGeneration.from_pretrained(
        snapshot, dtype=torch.float16, local_files_only=True, output_loading_info=True
    )
    loaded, report = result
    problems: list[str] = []
    for kind in LOADING_PROBLEMS:
        # A mismatched key comes with the two shapes; its name is enough here.
        names = sorted(str(key[0] if isinstance(key, tuple) else key) for key in report[kind])
        if names:
            problems.append(f"{kind.replace('_', ' ')}: {_listed(names)}")
    if problems:
        raise StepError("the snapshot does not fit the model: " + "; ".join(problems))
    model = loaded.to(device)
    model.eval()
    return LoadedModel(model=model, processor=processor, largest_weight=check_parameters(model))


def check_parameters(model: Any) -> float:
    """Stop unless every parameter is fp16, apart from the fp32 position weights, and finite.

    A bf16 weight above 65,504 in size becomes infinite when cast to fp16, so the check
    runs after loading. The encoder's position weights must stay fp32, as the library
    keeps them; an fp16 model made in memory with ``.half()`` has them in fp16 instead.

    Args:
        model: A loaded ``VoxtralForConditionalGeneration``.

    Returns:
        The largest |value| of any parameter.

    Raises:
        StepError: If a parameter is in another format or holds a non-finite value.
    """
    import torch

    wrong: list[str] = []
    non_finite: list[str] = []
    largest = 0.0
    for name, parameter in model.named_parameters():
        owner = name.rpartition(".")[0].rpartition(".")[2]
        expected = torch.float32 if owner == FP32_WEIGHTS else torch.float16
        if parameter.dtype != expected:
            wrong.append(f"{name} ({parameter.dtype})")
        values = parameter.detach()
        if not bool(torch.isfinite(values).all()):
            non_finite.append(name)
        elif values.numel():
            largest = max(largest, float(values.abs().max()))
    if wrong:
        raise StepError(
            "parameters not in fp16, or the position weights not in fp32: " + _listed(wrong)
        )
    if non_finite:
        raise StepError("parameters with a non-finite value: " + _listed(non_finite))
    return largest


def _listed(names: list[str]) -> str:
    """A count and the first three names, for a message."""
    return f"{len(names)} ({', '.join(names[:3])}{', ...' if len(names) > 3 else ''})"


# Numeric settings --------------------------------------------------------------------


def set_numerics(threads: int) -> dict[str, object]:
    """Fix the numeric settings of a step-7 job or GPU test and say what they are.

    The same kernels on the same inputs give the same bits only under the same settings:
    deterministic kernels without cuDNN autotuning (``runinfo.deterministic_torch``),
    fp16 matrix sums without either of cuBLAS's shortcuts (reduced-precision reductions,
    and products summed in fp16), TF32 off (the V100 has none; this guards against
    another GPU), and a fixed number of PyTorch threads, which build the log-mel features
    on the CPU. Call it first; the settings hold for the whole process.

    Args:
        threads: PyTorch's CPU threads (``encoder_cache.cpu_threads`` in
            ``configs/pilot/data.yaml``).

    Returns:
        The settings in force, read back from PyTorch, with ``OMP_NUM_THREADS`` and
        ``CUBLAS_WORKSPACE_CONFIG`` from the environment, for ``run.json``.

    Raises:
        TypeError: If ``threads`` is not a whole number.
        ValueError: If ``threads`` is below 1, or True or False.
    """
    import torch

    count = operator.index(threads)
    if isinstance(threads, bool) or count < 1:
        raise ValueError(f"threads must be a whole number of at least 1, not {threads!r}")
    deterministic_torch()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(count)
    matmul = torch.backends.cuda.matmul
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "fp16_reduced_precision_reduction": bool(matmul.allow_fp16_reduced_precision_reduction),
        "fp16_accumulation": bool(matmul.allow_fp16_accumulation),
        "matmul_tf32": bool(matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


# Input -------------------------------------------------------------------------------


def chunks(samples: int) -> int:
    """Return the number of 30 s chunks the processor makes of a clip.

    The processor pads every clip with zeros to a multiple of 480,000 samples.

    Args:
        samples: The clip's length in samples at 16 kHz.

    Returns:
        The number of chunks: 1 up to 30.00 s, 2 above, up to 60 s.

    Raises:
        ValueError: If the clip has no sample.
    """
    count = operator.index(samples)
    if count < 1:
        raise ValueError(f"a clip needs at least one sample, not {count}")
    return -(-count // CHUNK_SAMPLES)


def clip_inputs(processor: Any, audio: np.ndarray, sampling_rate: int = SAMPLING_RATE) -> Any:
    """Build the model input for one clip: the audio, then the prompt, in one user turn.

    Args:
        processor: The model's processor.
        audio: Mono audio samples in [-1, 1].
        sampling_rate: Sampling rate of ``audio``; the model expects 16 kHz.

    Returns:
        Token IDs and attention mask of shape (1, L) and log-mel features of shape
        (C, 128, 3000), as PyTorch tensors (a ``BatchFeature``); L is 397 for one chunk.

    Raises:
        ValueError: If the audio is not mono or not sampled at 16 kHz.
    """
    if audio.ndim != 1:
        raise ValueError(f"expected mono audio, got shape {audio.shape}")
    if sampling_rate != SAMPLING_RATE:
        raise ValueError(f"expected {SAMPLING_RATE} Hz audio, got {sampling_rate} Hz")
    # The processor takes audio as an encoded file; base64 keeps it in memory.
    buffer = io.BytesIO()
    sf.write(buffer, audio, sampling_rate, format="WAV", subtype="FLOAT")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "base64": audio_b64},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    return processor.apply_chat_template(conversation)


def prompt_inputs(processor: Any, n_chunks: int) -> dict[str, "torch.Tensor"]:
    """Return the token IDs and attention mask of any clip of n chunks.

    The IDs depend on the number of chunks only, never on the sound, so they are built
    from a silent clip. Build them once per chunk count and reuse them.

    Args:
        processor: The model's processor.
        n_chunks: The number of chunks (``chunks``).

    Returns:
        ``input_ids`` and ``attention_mask``, each of shape (1, L).

    Raises:
        ValueError: If ``n_chunks`` is below 1.
        StepError: If the audio tokens do not number 375 per chunk.
    """
    count = operator.index(n_chunks)
    if count < 1:
        raise ValueError(f"a clip has at least one chunk, not {count}")
    inputs = clip_inputs(processor, np.zeros(count * CHUNK_SAMPLES, dtype=np.float32))
    ids = inputs["input_ids"]
    if ids.shape[0] != 1:
        raise StepError(f"the processor gave {ids.shape[0]} rows of token IDs for one clip")
    found = int((ids == processor.audio_token_id).sum())
    if found != TOKENS_PER_CHUNK * count:
        raise StepError(
            f"a silent clip of {count * 30} s gave {found} audio tokens, "
            f"not {TOKENS_PER_CHUNK * count} ({count} x {TOKENS_PER_CHUNK})"
        )
    return {"input_ids": ids, "attention_mask": inputs["attention_mask"]}


# The encoder and its cache -----------------------------------------------------------


@functools.cache
def _pass_through() -> type:
    """The class that stands in for the encoder, made once, when torch is first needed."""
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    class CachedEncoder(torch.nn.Module):
        """Stands in for the frozen audio encoder: returns the stored output it is given.

        The model calls its encoder as ``audio_tower(input_features, return_dict=True)``
        and reads ``last_hidden_state``; the stored output goes in as ``input_features``.
        """

        def forward(self, input_features: Any, *args: Any, **kwargs: Any) -> Any:
            return BaseModelOutputWithPooling(last_hidden_state=input_features)

    return CachedEncoder


def _base(model: Any) -> Any:
    """The model itself, or the model inside a PEFT wrapper.

    A PEFT model holds the loaded model (``get_base_model``), whose modules carry the
    LoRA layers and the trained adapter; running that model runs them too.
    """
    unwrap = getattr(model, "get_base_model", None)
    return unwrap() if callable(unwrap) else model


def _cached(model: Any) -> bool:
    """True if the pass-through stands in for the model's encoder."""
    return isinstance(_base(model).model.audio_tower, _pass_through())


def encoder_output(model: Any, inputs: Mapping[str, Any]) -> "torch.Tensor":
    """Run the frozen audio encoder alone on one clip, as ``model(**inputs)`` runs it.

    Only the encoder runs, never the adapter or the language model.

    Args:
        model: The loaded model, or a PEFT model wrapping it, with its own encoder in
            place.
        inputs: The model input of one clip (``clip_inputs``); its log-mel features, of
            shape (C, 128, 3000), are read.

    Returns:
        The encoder's output before the adapter (``last_hidden_state``), shape
        (C, 1500, W), in the model's format (fp16), on the model's device.

    Raises:
        StepError: If the pass-through is in place or the output has another shape.
    """
    import torch

    if _cached(model):
        raise StepError("the encoder output needs the encoder, but the pass-through is in place")
    model = _base(model)
    features = inputs["input_features"].to(model.device)
    with torch.inference_mode():
        # As VoxtralModel.get_audio_features calls it.
        output = model.model.audio_tower(features, return_dict=True).last_hidden_state
    expected = (int(features.shape[0]), FRAMES_PER_CHUNK, model.config.audio_config.hidden_size)
    if tuple(output.shape) != expected:
        raise StepError(f"the encoder output has shape {tuple(output.shape)}, not {expected}")
    return output


def use_cached_encoder(model: Any) -> Any:
    """Swap the model's encoder for a pass-through that hands on a stored output.

    The model then takes a stored encoder output, of shape (C, 1500, W), where it took
    log-mel features (``cached_score``, ``lm_pass``). No parameter changes.

    Args:
        model: The loaded model, or a PEFT model wrapping it, with its own encoder in
            place.

    Returns:
        The encoder taken out, for ``restore_encoder``.

    Raises:
        StepError: If the pass-through is in place already.
    """
    if _cached(model):
        raise StepError("the pass-through is in place already")
    inner = _base(model).model
    encoder = inner.audio_tower
    inner.audio_tower = _pass_through()().eval()
    return encoder


def restore_encoder(model: Any, encoder: Any) -> None:
    """Put the encoder back that ``use_cached_encoder`` took out.

    Args:
        model: The model, with the pass-through in place.
        encoder: The encoder that ``use_cached_encoder`` returned.

    Raises:
        StepError: If the model holds an encoder, or ``encoder`` is a pass-through.
    """
    if not _cached(model):
        raise StepError("the model holds its own encoder already")
    if isinstance(encoder, _pass_through()):
        raise StepError("restore_encoder needs the encoder that use_cached_encoder returned")
    _base(model).model.audio_tower = encoder


def _check_cached_input(model: Any, prompt: Mapping[str, Any], output: "torch.Tensor") -> None:
    """Stop unless a stored encoder output can go through the model with this prompt."""
    if not _cached(model):
        raise StepError("the model's own encoder is in place; call use_cached_encoder first")
    model = _base(model)
    width = model.config.audio_config.hidden_size
    shape = tuple(output.shape)
    if len(shape) != 3 or shape[0] < 1 or shape[1:] != (FRAMES_PER_CHUNK, width):
        raise StepError(
            f"a stored encoder output has shape (C, {FRAMES_PER_CHUNK}, {width}), not {shape}"
        )
    adapter = model.model.multi_modal_projector.linear_1.weight.dtype
    if output.dtype != adapter:
        raise StepError(f"the stored encoder output is {output.dtype}, the adapter {adapter}")
    tokens = int((prompt["input_ids"] == model.config.audio_token_id).sum())
    if tokens != TOKENS_PER_CHUNK * shape[0]:
        raise StepError(
            f"the prompt holds {tokens} audio tokens, but the stored output fills "
            f"{TOKENS_PER_CHUNK * shape[0]} ({shape[0]} x {TOKENS_PER_CHUNK})"
        )


def _model_inputs(prompt: Mapping[str, Any], output: "torch.Tensor") -> dict[str, Any]:
    """The prompt's IDs and mask, with the stored output where the log-mel features go."""
    return {
        "input_ids": prompt["input_ids"],
        "attention_mask": prompt["attention_mask"],
        "input_features": output,
    }


# Scores ------------------------------------------------------------------------------


def _score(model: Any, inputs: Mapping[str, Any], answer_ids: tuple[int, int]) -> float:
    """Return z of one clip: the one forward pass that every score goes through."""
    import torch

    yes_id, no_id = answer_ids
    model = _base(model)
    moved = {name: value.to(model.device) for name, value in inputs.items()}
    with torch.inference_mode():
        output = model(**moved, logits_to_keep=1)
    logits = output.logits[0, -1].float()
    return float(logits[yes_id] - logits[no_id])


def yes_no_score(model: Any, inputs: Mapping[str, Any], answer_ids: tuple[int, int]) -> float:
    """Return z = logit("Yes") - logit("No") at the first answer position, from the audio.

    Args:
        model: The loaded model (see ``load_model``), or a PEFT model wrapping it, with its
            own encoder in place.
        inputs: The model input of one clip (see ``clip_inputs``).
        answer_ids: The token IDs of "Yes" and "No" (see ``answer_token_ids``).

    Returns:
        The score z, computed in float32.

    Raises:
        StepError: If the pass-through is in place.
    """
    if _cached(model):
        raise StepError("yes_no_score reads the audio, but the pass-through is in place")
    return _score(model, inputs, answer_ids)


def cached_score(
    model: Any,
    prompt: Mapping[str, Any],
    encoder_output: "torch.Tensor",
    answer_ids: tuple[int, int],
) -> float:
    """Return z of one clip from its stored encoder output.

    The output goes through the library's own adapter, token placement and language
    model, in the same forward pass as ``yes_no_score``, so both give the same bits.
    Every test-fold score of the pilot comes from here, one clip per call: the trained
    arms pass their PEFT model, whose LoRA layers and adapter then take part.

    Args:
        model: The loaded model, or a PEFT model wrapping it, with the pass-through in
            place (``use_cached_encoder``).
        prompt: The token IDs and mask of a clip of C chunks (``prompt_inputs``).
        encoder_output: The clip's stored encoder output, shape (C, 1500, W), in the
            adapter's format, fp16: an adapter kept in another format is cast to fp16
            before scoring.
        answer_ids: The token IDs of "Yes" and "No" (see ``answer_token_ids``).

    Returns:
        The score z, computed in float32.

    Raises:
        StepError: If the model's own encoder is in place, or the output's shape, format
            or chunk count does not fit the model and the prompt.
    """
    _check_cached_input(model, prompt, encoder_output)
    return _score(model, _model_inputs(prompt, encoder_output), answer_ids)


def lm_pass(model: Any, prompt: Mapping[str, Any], encoder_output: "torch.Tensor") -> None:
    """Send a stored encoder output through the model without its output layer.

    The input is that of ``cached_score``, but ``model.model`` stops before the output
    layer, so no logit and no score exist. It serves the overflow watch only
    (``watch_layers``).

    Args:
        model: The loaded model, or a PEFT model wrapping it, with the pass-through in
            place (``use_cached_encoder``).
        prompt: The token IDs and mask of a clip of C chunks (``prompt_inputs``).
        encoder_output: The clip's stored encoder output, shape (C, 1500, W).

    Raises:
        StepError: As ``cached_score``.
    """
    import torch

    _check_cached_input(model, prompt, encoder_output)
    model = _base(model)
    inputs = _model_inputs(prompt, encoder_output)
    moved = {name: value.to(model.device) for name, value in inputs.items()}
    with torch.inference_mode():
        model.model(**moved)


# The overflow watch ------------------------------------------------------------------


def encoder_layers(model: Any) -> dict[str, Any]:
    """Return every step of the audio encoder by name, in order, for ``watch_layers``.

    Its two convolutions, its layers (32 in Voxtral Mini 3B) and its final norm, whose
    output is the stored output: an overflow is then named where it starts.

    Args:
        model: The loaded model, or a PEFT model wrapping it, with its own encoder in
            place.

    Returns:
        The modules, named as in the model: ``model.audio_tower.conv1``,
        ``model.audio_tower.conv2``, ``model.audio_tower.layers.0`` and on, and
        ``model.audio_tower.layer_norm``.

    Raises:
        StepError: If the pass-through is in place.
    """
    if _cached(model):
        raise StepError("the encoder's layers are out: the pass-through is in place")
    model = _base(model)
    encoder = model.model.audio_tower
    return _named(model, [encoder.conv1, encoder.conv2, *encoder.layers, encoder.layer_norm])


def language_model_layers(model: Any, logits: bool = False) -> dict[str, Any]:
    """Return every step from the adapter to the logits by name, in order, for ``watch_layers``.

    The adapter, whose fp16 output the language model reads in place of the audio
    tokens; the language model's layers (30 in Voxtral Mini 3B); its final norm; and its
    output layer if asked for. An overflow is then named where it starts.

    Args:
        model: The loaded model, or a PEFT model wrapping it.
        logits: Also return the output layer, ``lm_head``, whose output is the logits.

    Returns:
        The modules, named as in the model: ``model.multi_modal_projector``,
        ``model.language_model.layers.0`` and on, ``model.language_model.norm``, and
        ``lm_head`` last if asked for.
    """
    model = _base(model)
    inner = model.model
    modules = [inner.multi_modal_projector, *inner.language_model.layers, inner.language_model.norm]
    return _named(model, [*modules, model.lm_head] if logits else modules)


def _named(model: Any, modules: list[Any]) -> dict[str, Any]:
    """The modules under their names in the model."""
    names = {id(module): name for name, module in model.named_modules()}
    return {names[id(module)]: module for module in modules}


def _parts(layer: Any) -> dict[str, Any]:
    """Return the parts of a layer that the watch reads.

    The output and, where the layer has them, its attention part and its feed-forward
    part: ``mlp`` in the language model; in the encoder ``fc2``, the feed-forward part's
    last step. A convolution, a norm and the adapter have their output only.
    """
    parts = {"output": layer}
    attention = getattr(layer, "self_attn", None)
    if attention is not None:
        parts["attention"] = attention
    feed_forward = getattr(layer, "mlp", None)
    if feed_forward is None:
        feed_forward = getattr(layer, "fc2", None)
    if feed_forward is not None:
        parts["feed_forward"] = feed_forward
    return parts


class _Tally:
    """What one watched part showed so far.

    The numbers stay on the model's device, so the watch never waits for the GPU;
    ``LayerWatch.record`` reads them.
    """

    __slots__ = ("at_clamp", "calls", "max_abs", "non_finite")

    def __init__(self) -> None:
        self.calls = 0
        self.max_abs: Any = None
        self.at_clamp: Any = None
        self.non_finite: Any = None

    def hook(self, _module: Any, _args: Any, output: Any) -> None:
        """Count one call's output. Read only: it returns nothing, so the output stays."""
        import torch

        values = output if isinstance(output, torch.Tensor) else None
        if isinstance(output, tuple | list) and output and isinstance(output[0], torch.Tensor):
            values = output[0]  # an attention part also returns its weights
        if values is None:
            raise TypeError(f"the watch cannot read an output of type {type(output).__name__}")
        self.calls += 1
        if values.numel() == 0:
            return
        with torch.no_grad():
            values = values.detach()
            finite = torch.isfinite(values)
            size = values.abs()
            largest = size.masked_fill(~finite, 0).amax().float()
            at_clamp = (finite & (size >= CLAMP_ABS)).sum()
            non_finite = (~finite).sum()
            if self.max_abs is None:
                self.max_abs, self.at_clamp, self.non_finite = largest, at_clamp, non_finite
            else:
                self.max_abs = torch.maximum(self.max_abs, largest)
                self.at_clamp = self.at_clamp + at_clamp
                self.non_finite = self.non_finite + non_finite

    def numbers(self) -> dict[str, int | float | None]:
        """The counts and the largest finite |value| (None before the first call)."""
        return {
            "calls": self.calls,
            "max_abs": None if self.max_abs is None else float(self.max_abs),
            "at_clamp": 0 if self.at_clamp is None else int(self.at_clamp),
            "non_finite": 0 if self.non_finite is None else int(self.non_finite),
        }


class LayerWatch:
    """Read-only forward hooks that count fp16 overflow, layer by layer.

    Each watched layer is read at its output and, where it has them, at its attention and
    feed-forward parts (``PARTS``). For each, over all calls, the watch keeps the largest
    finite |value|, the finite values at the encoder's clamp or above (|x| >= 64,512)
    and the non-finite values. Any such value flags the layer. Counts only: no value of a
    clip is kept. Build it with ``watch_layers``; leaving a ``with`` block removes it.

    Args:
        layers: The modules to watch, by name.
    """

    def __init__(self, layers: Mapping[str, Any]) -> None:
        self._tallies: dict[tuple[str, str], _Tally] = {}
        self._handles: list[Any] = []
        for name, layer in layers.items():
            for part, module in _parts(layer).items():
                tally = _Tally()
                self._tallies[(name, part)] = tally
                self._handles.append(module.register_forward_hook(tally.hook))

    def record(self) -> dict[str, dict[str, dict[str, int | float | None]]]:
        """Return, by layer name and part: calls, ``max_abs``, ``at_clamp`` and ``non_finite``."""
        record: dict[str, dict[str, dict[str, int | float | None]]] = {}
        for (name, part), tally in self._tallies.items():
            record.setdefault(name, {})[part] = tally.numbers()
        return record

    def flagged(self) -> list[str]:
        """Return the flagged parts, as ``<layer>/<part>``, in the order they are watched."""
        return [
            f"{name}/{part}"
            for (name, part), tally in self._tallies.items()
            if (numbers := tally.numbers())["at_clamp"] or numbers["non_finite"]
        ]

    def largest(self, n: int = 5) -> list[tuple[str, float]]:
        """Return the n largest finite |values|, as (``<layer>/<part>``, value), largest first."""
        values = [
            (f"{name}/{part}", maximum)
            for (name, part), tally in self._tallies.items()
            if (maximum := tally.numbers()["max_abs"]) is not None
        ]
        return sorted(values, key=lambda item: -item[1])[:n]

    def remove(self) -> None:
        """Remove every hook; the counts stay readable."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "LayerWatch":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.remove()


def watch_layers(layers: Mapping[str, Any]) -> LayerWatch:
    """Start watching layers for fp16 overflow.

    Watch the encoder while caches are built (``encoder_layers``), and the adapter and
    the language model while it runs (``language_model_layers``, with the logits when
    scores are computed). The hooks only read; a score is the same with the watch on.

    Args:
        layers: The modules to watch, by name.

    Returns:
        The watch; it counts until ``remove`` or the end of its ``with`` block.
    """
    return LayerWatch(layers)
