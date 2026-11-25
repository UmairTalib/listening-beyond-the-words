"""Training of B0, B1 and CPG from the cached encoder outputs, and the hand-over to scoring.

Pilot protocol, section "Training". The model that trains (``build_trainable``) is the
pinned Voxtral in fp16 with its encoder swapped for the pass-through and dropped, since
training reads the pinned caches. LoRA sits on the 7 projections of each layer of the
language model (210 places in Voxtral Mini 3B), picked by their full names, and the
adapter trains as a copy that PEFT makes, while the original stays frozen. Every trained
number is kept in fp32 (the master weights) while the forward pass computes in fp16 under
autocast, with gradient checkpointing on every layer. ``make_optimiser`` gives AdamW in two
groups, LoRA and the adapter at their own rates, the learning rate's linear warm-up and
half cosine over the run's T steps, and PyTorch's dynamic loss scaler.

``train_step`` makes one step from the bags of a step (2 positive, 2 negative), bag by bag
and, within a bag, view by view in the arm's order: ``x1``, whose bag mean the gain reads,
then ``x0``, then ``x2`` (B0 has ``x0`` only). A view of a bag takes one call per chunk count
(``voxtral.cached_z``): its one-chunk clips together and a two-chunk clip alone, nothing
padded. One backward pass follows all calls of a bag and view, so the gain acts on the mean
z of all of a bag's clips and each clip's CE counts one quarter. A bag's loss is CE(x0) +
0.25 CE(x1) + 0.5 CE(x2) + λG · the soft hinge of its gain; a step's loss is the mean of its
bags' losses, so each bag's part goes into its backward pass divided by the number of bags
and multiplied by the loss scale. B1 is CPG with λG = 0: it computes the gain and its running
statistics, which it logs, without training on them. The overflow watch reads every forward
call, is paused in the backward pass and is read once per step; a flag, or a z or a loss
that is not finite, stops the run before the update. Then the gradients are unscaled and
clipped to a joint norm of 1.0, and the scaler steps AdamW, unless a gradient overflowed:
then it skips the update and halves the scale. The schedule, λG and the running statistics
move on with every step, skipped or not, and a loss scale below 1 fails check F1.

After the last step, ``save_adapter`` writes LoRA and the adapter copy in fp32,
``release_training`` frees the training model and checks that nothing of it is left, and
``load_for_scoring`` builds the scoring model from the saved file: a fresh model without the
encoder, LoRA and the adapter cast to fp16, in evaluation mode, for the zero-shot job's
scoring code. ``rule_measures`` reads a run's lines of ``metrics.jsonl`` for checks F1, F2
and F5, the gain-alive rule and the 4-epoch rule, each verdict only on the runs that the
protocol reads it on; ``f5_projection`` is the timing run's projection of check F5. No
record holds a participant ID.

transformers and PEFT are imported inside the functions that use them.
"""

import gc
import hashlib
import json
import math
import statistics
import time
import warnings
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, cast

import numpy as np
import pandas as pd
import torch

from listening_beyond_words.bags import Bag
from listening_beyond_words.config import (
    STRICT,
    WARN_ONLY,
    AliveRule,
    ArmConfig,
    FallingRule,
    GainSettings,
    LoraSettings,
    TrainSettings,
)
from listening_beyond_words.loss import (
    RunningStats,
    ce_on_z,
    gain,
    lambda_g,
    share_steps,
    soft_hinge,
    step_measures,
)
from listening_beyond_words.runinfo import StepError, item_seed, sha256_file
from listening_beyond_words.voxtral import (
    LayerWatch,
    LoadedModel,
    answer_token_ids,
    cached_z,
    load_model,
    set_numerics,
    use_cached_encoder,
)

TORCH_STREAM: Final = "torch"  # torch.manual_seed(item_seed(<round seed>, TORCH_STREAM, <fold>))
# The projections of each layer of the language model that get LoRA (lora.targets).
PROJECTIONS: Final = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
ADAPTER_FILE: Final = "adapter_model.safetensors"  # LoRA and the adapter copy, in fp32
ADAPTER_CONFIG: Final = "adapter_config.json"
# Gradient checkpointing as transformers 5.22.2 does it by default, written out so that a
# changed default cannot change a run: the non-reentrant kind, on every layer.
CHECKPOINTING: Final[Mapping[str, Any]] = MappingProxyType({"use_reentrant": False})
# What may stay allocated on the GPU once the training model is released. The pilot
# protocol's "the GPU is checked to be empty" (section "Training", bullet "Scoring") is read
# as: nothing of the training model is left (a weak reference to it is dead) and at most
# this much is allocated. cuBLAS keeps a workspace of 32 MiB per stream
# (CUBLAS_WORKSPACE_CONFIG=:4096:8) that no release frees, while the training model's
# smallest part that could be left, its fp32 gradients, takes 209 MB. run.json records
# what stays allocated.
MAX_LEFT_BYTES: Final = 128 * 2**20
# The packages whose versions a training run records (engineering conventions, section 8).
PACKAGES: Final = (
    "torch",
    "transformers",
    "peft",
    "mistral-common",
    "safetensors",
    "tokenizers",
    "nvidia-cudnn-cu12",
    "nvidia-cublas-cu12",
    "numpy",
    "pandas",
    "pyarrow",
    "pyyaml",
)
# The four groups of a step's bags, by class and by whether the words were right; the
# counts of step_measures are named bags_<group> and active_<group>.
GROUPS: Final = (
    "positive_words_right",
    "positive_words_wrong",
    "negative_words_right",
    "negative_words_wrong",
)
# The measures that B0 logs, which has no gain: those of step_measures that need none.
X0_MEASURES: Final = ("z0_bar_mean", "z0_bar_mean_square", "z0_bar_abs_mean", "ce_push")
# PyTorch's warning when a schedule steps before its optimiser first did (torch.optim).
SCHEDULE_FIRST: Final = r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`"


def torch_seed(round_seed: int, fold: int) -> int:
    """Return the PyTorch seed of a fold-run: the same for every arm.

    It is set right before LoRA is added (``build_trainable``), so B0, B1 and CPG of a
    fold-run start from the same LoRA values (pilot protocol, section "Training", bullet
    "Who trains, and seeds").

    Args:
        round_seed: The round's seed (``seeds`` in ``train.yaml``).
        fold: The fold whose test people the run scores, from 1.

    Returns:
        ``item_seed(round_seed, "torch", fold)``.
    """
    return item_seed(round_seed, TORCH_STREAM, fold)


def _is_lora(name: str) -> bool:
    """True for a parameter of LoRA's two matrices."""
    return ".lora_A." in name or ".lora_B." in name


def _is_adapter(name: str) -> bool:
    """True for a parameter of the adapter's trained copy."""
    return ".modules_to_save." in name


def _trained(model: Any) -> list[tuple[str, Any]]:
    """The parameters that train, by name, in the model's order."""
    return [(name, value) for name, value in model.named_parameters() if value.requires_grad]


def _device_type(model: Any) -> str:
    """The type of the device the model's weights are on: ``cuda`` or ``cpu``."""
    return str(next(model.parameters()).device.type)


def _listed(names: Sequence[str]) -> str:
    """A count and the first three names, for a message."""
    return f"{len(names)} ({', '.join(names[:3])}{', ...' if len(names) > 3 else ''})"


# The model that trains ---------------------------------------------------------------------


@dataclass
class Trainable:
    """The model that trains, ready for its first step.

    Attributes:
        model: A PEFT model around the pinned Voxtral: fp16 frozen weights, LoRA and the
            adapter copy in fp32, the pass-through in place of the encoder, gradient
            checkpointing on every layer, in train mode, on its device.
            ``release_training`` sets it to None.
        processor: The model's processor, for the prompts (``voxtral.prompt_inputs``).
        answer_ids: The token IDs of "Yes" and "No".
        record: What a run records about it in ``run.json``, aggregates only: the LoRA
            modules and layers, the trained numbers, the SHA-256 of their start values, the
            PEFT config without the snapshot's path, the checkpointing, the seed and the
            largest frozen weight.
    """

    model: Any
    processor: Any
    answer_ids: tuple[int, int]
    record: dict[str, Any]


def peft_record(model: Any) -> dict[str, Any]:
    """Return a PEFT model's config for ``run.json``, without the path of the base model.

    PEFT keeps the snapshot's folder, which lies in the workspace, as
    ``base_model_name_or_path``; a run records the snapshot by its pin instead.

    Args:
        model: A PEFT model with the adapter ``default``.

    Returns:
        The config as plain JSON types.
    """
    config = dict(model.peft_config["default"].to_dict())
    config.pop("base_model_name_or_path", None)

    def plain(value: Any) -> Any:
        if isinstance(value, set | frozenset):
            return sorted(value)
        return str(value)

    return json.loads(json.dumps(config, default=plain, sort_keys=True))


def _values_sha256(trained: Sequence[tuple[str, Any]]) -> str:
    """The SHA-256 of trained tensors: each name, then its values' bytes in C order."""
    digest = hashlib.sha256()
    for name, value in trained:
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _check_layout(model: Any) -> dict[str, int]:
    """Stop unless LoRA sits on the 7 projections of every layer and the adapter is a copy.

    Returns:
        The LoRA modules and the layers of the language model.
    """
    from peft.tuners.lora import LoraLayer
    from peft.utils.other import ModulesToSaveWrapper

    base = model.get_base_model()
    layers = len(base.model.language_model.layers)
    wrapped = [name for name, module in base.named_modules() if isinstance(module, LoraLayer)]
    expected = {
        f"model.language_model.layers.{index}.{projection}"
        for index in range(layers)
        for projection in PROJECTIONS
    }
    if len(wrapped) != len(expected) or set(wrapped) != expected:
        outside = sorted(set(wrapped) - expected)
        raise StepError(
            f"LoRA sits on {len(wrapped)} modules, {len(outside)} of them elsewhere"
            f" ({_listed(outside)}), not on the {len(PROJECTIONS)} projections of each of the"
            f" language model's {layers} layers ({len(expected)})"
        )
    copies = [
        name for name, module in base.named_modules() if isinstance(module, ModulesToSaveWrapper)
    ]
    if copies != ["model.multi_modal_projector"]:
        raise StepError(f"the adapter must be the one trained copy, not {copies}")
    return {"lora_modules": len(wrapped), "layers": layers}


def build_trainable(
    snapshot: Path, lora: LoraSettings, seed: int, device: str = "cuda"
) -> Trainable:
    """Build the model that trains: the pinned Voxtral with LoRA and the adapter's copy.

    In this order (pilot protocol, section "Training", bullets "Arms and what trains" and
    "Number formats and repeatability"): the model loads on the CPU in fp16
    (``voxtral.load_model``); the pass-through takes the encoder's place and the encoder is
    dropped, since training reads the caches (1.27 GB less on the GPU); PyTorch is seeded,
    so the LoRA values depend on the seed alone; PEFT adds LoRA with the settings of
    ``lora`` and makes the adapter's trained copy; every trained tensor is cast to fp32
    before any backward pass, since PEFT leaves the adapter's copy in fp16; the layout and
    the formats are checked; gradient checkpointing goes on every layer; then train mode
    and the device.

    Args:
        snapshot: Folder of the pinned snapshot (see ``models.local_snapshot``).
        lora: LoRA's settings (``lora`` in ``train.yaml``).
        seed: The fold-run's PyTorch seed (``torch_seed``).
        device: Where the model trains.

    Returns:
        The model, its processor, the IDs of "Yes" and "No", and its record.

    Raises:
        StepError: If the snapshot does not load (``voxtral.load_model``), LoRA does not sit
            on the 7 projections of every layer of the language model, the adapter is not
            the one trained copy, a parameter that is neither of them trains, or a trained
            tensor is not fp32 or a frozen one not fp16.
    """
    from peft import LoraConfig, get_peft_model

    loaded = load_model(snapshot, device="cpu")
    use_cached_encoder(loaded.model)  # the encoder it hands back is dropped here
    answer_ids = answer_token_ids(loaded.processor.tokenizer)
    torch.manual_seed(seed)
    config = LoraConfig(
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias=cast(Literal["none", "all", "lora_only"], lora.bias),
        init_lora_weights=lora.init_lora_weights,
        use_rslora=lora.use_rslora,
        use_dora=lora.use_dora,
        target_modules=lora.targets,
        modules_to_save=[lora.trained_adapter],
    )
    model: Any = get_peft_model(loaded.model, config)
    for _, value in _trained(model):
        value.data = value.data.to(torch.float32)
    layout = _check_layout(model)
    trained = _trained(model)
    others = [name for name, _ in trained if not (_is_lora(name) or _is_adapter(name))]
    if others:
        raise StepError(
            f"parameters train that are neither LoRA nor the adapter: {_listed(others)}"
        )
    wrong = [name for name, value in trained if value.dtype != torch.float32]
    frozen = [
        name
        for name, value in model.named_parameters()
        if not value.requires_grad and value.dtype != torch.float16
    ]
    if wrong or frozen:
        raise StepError(
            f"trained tensors not in fp32: {_listed(wrong)}; frozen ones not in fp16:"
            f" {_listed(frozen)}"
        )
    lora_numbers = sum(value.numel() for name, value in trained if _is_lora(name))
    record = {
        **layout,
        "trained_tensors": len(trained),
        "trained_numbers": sum(value.numel() for _, value in trained),
        "lora_numbers": lora_numbers,
        "adapter_numbers": sum(value.numel() for name, value in trained if _is_adapter(name)),
        "start_sha256": _values_sha256(trained),
        "peft_config": peft_record(model),
        "checkpointing": {**CHECKPOINTING, "every_n_layers": 1},
        "torch_seed": seed,
        "largest_weight": float(loaded.largest_weight),
    }
    model.get_base_model().gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=dict(CHECKPOINTING)
    )
    model.train()
    model.to(device)
    return Trainable(model=model, processor=loaded.processor, answer_ids=answer_ids, record=record)


def training_numerics(threads: int, mode: str) -> dict[str, object]:
    """Fix the numeric settings of training and say what they are.

    Training uses the settings of the caches and the zero-shot job (``voxtral.set_numerics``),
    and in strict repeatable mode it also stops at any operation without a repeatable
    version (pilot protocol, section "Training", bullet "Number formats and repeatability").
    Scoring switches back with ``voxtral.set_numerics`` before it starts.

    Args:
        threads: PyTorch's CPU threads (``encoder_cache.cpu_threads`` in ``data.yaml``).
        mode: ``strict`` or ``warn_only``; the timing run times both.

    Returns:
        The settings in force, as ``voxtral.set_numerics`` records them.

    Raises:
        StepError: If the mode is neither, such as ``pending``.
    """
    if mode not in (STRICT, WARN_ONLY):
        raise StepError(
            f"training runs in mode {STRICT!r} or {WARN_ONLY!r}, not {mode!r}: the timing run"
            ' sets it (pilot protocol, section "Training", bullet "Number formats and'
            ' repeatability")'
        )
    numerics = set_numerics(threads)
    if mode == STRICT:
        torch.use_deterministic_algorithms(True, warn_only=False)
        numerics["deterministic_warn_only"] = torch.is_deterministic_algorithms_warn_only_enabled()
    return numerics


# The optimiser -----------------------------------------------------------------------------


@dataclass
class Optimiser:
    """AdamW, its schedule and the loss scaler of one run.

    Attributes:
        optimizer: AdamW, with LoRA in its first group and the adapter copy in its second.
            ``release_training`` sets it to None, as the two below.
        schedule: The learning rate's schedule: transformers' cosine schedule with warm-up.
        scaler: PyTorch's dynamic loss scaler.
        total_steps: T, the run's steps.
        warmup_steps: ⌈0.05·T⌉.
        record: The resolved settings for ``run.json``.
    """

    optimizer: Any
    schedule: Any
    scaler: Any
    total_steps: int
    warmup_steps: int
    record: dict[str, Any]


def make_optimiser(trainable: Trainable, settings: TrainSettings, total_steps: int) -> Optimiser:
    """Build AdamW, its schedule and the loss scaler for a run of T steps.

    AdamW in two groups, LoRA at ``learning_rate`` and the adapter at
    ``adapter_learning_rate``, with ``train.yaml``'s betas, eps, AMSGrad, weight decay on
    every trained tensor and its ``foreach`` implementation. The learning rate is 0 at step
    0, rises in a straight line over ⌈0.05·T⌉ steps and falls along half a cosine to 0, for
    both groups (``transformers.get_cosine_schedule_with_warmup``; pilot protocol, section
    "Training", bullet "Loss and schedules"). The scaler (``torch.amp.GradScaler``) takes
    ``precision.scaler``'s settings.

    Args:
        trainable: The model that trains (``build_trainable``).
        settings: The shared training settings.
        total_steps: T, all steps of the run (``bags.BagSampler.total_steps``).

    Returns:
        The optimiser, its schedule and scaler, T, the warm-up steps and the record.

    Raises:
        ValueError: If T is below 1.
        StepError: If LoRA or the adapter has no trained tensor.
    """
    from transformers import get_cosine_schedule_with_warmup

    if isinstance(total_steps, bool) or total_steps < 1:
        raise ValueError(f"a run has at least one step, not {total_steps!r}")
    trained = _trained(trainable.model)
    lora = [value for name, value in trained if _is_lora(name)]
    adapter = [value for name, value in trained if _is_adapter(name)]
    if not lora or not adapter:
        raise StepError(
            f"{len(lora)} LoRA and {len(adapter)} adapter tensors train: both need some"
        )
    chosen = settings.optimiser
    optimizer = torch.optim.AdamW(
        [
            {"params": lora, "lr": chosen.learning_rate},
            {"params": adapter, "lr": chosen.adapter_learning_rate},
        ],
        lr=chosen.learning_rate,
        betas=chosen.betas,
        eps=chosen.eps,
        weight_decay=chosen.weight_decay,
        amsgrad=chosen.amsgrad,
        foreach=chosen.foreach,
    )
    warmup = share_steps(chosen.warmup_share, total_steps) if chosen.warmup_share > 0 else 0
    schedule = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_steps
    )
    scaling = settings.precision.scaler
    scaler = torch.amp.GradScaler(
        _device_type(trainable.model),
        init_scale=scaling.init_scale,
        growth_factor=scaling.growth_factor,
        backoff_factor=scaling.backoff_factor,
        growth_interval=scaling.growth_interval_steps,
    )
    record = {
        "total_steps": total_steps,
        "warmup_steps": warmup,
        "groups": [
            {
                "name": name,
                "tensors": len(values),
                "numbers": sum(value.numel() for value in values),
                "learning_rate": rate,
            }
            for name, values, rate in (
                ("lora", lora, chosen.learning_rate),
                ("adapter", adapter, chosen.adapter_learning_rate),
            )
        ],
        "foreach": optimizer.defaults["foreach"],
        "fused": optimizer.defaults["fused"],
        "schedule": "transformers.get_cosine_schedule_with_warmup",
        "scaler_device": _device_type(trainable.model),
    }
    return Optimiser(optimizer, schedule, scaler, total_steps, warmup, record)


# The running statistics of the gain ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ViewStats:
    """The running statistics of the gain, of ``x0`` and of ``x1`` (B1 and CPG).

    Attributes:
        x0: μ0 and σ0.
        x1: μ1 and σ1.
    """

    x0: RunningStats
    x1: RunningStats

    def update(self, z0_bars: torch.Tensor, z1_bars: torch.Tensor) -> "ViewStats":
        """Return both views' statistics after one step, updated at the same point.

        Args:
            z0_bars: The step's bag means on ``x0``, shape (B,).
            z1_bars: The step's bag means on ``x1``, shape (B,).

        Returns:
            New statistics; these stay as they are.
        """
        return ViewStats(self.x0.update(z0_bars), self.x1.update(z1_bars))


def start_stats(bags: Sequence[Bag], scores: pd.DataFrame, settings: GainSettings) -> ViewStats:
    """Start the running statistics from the zero-shot scores of the start bags.

    Each start bag's mean of its segments' zero-shot z on ``x0`` and on ``x1`` (pilot
    protocol, section "Encoder caches and zero-shot scores", bullet "Start of the running
    statistics"): the 32 bags of the sampler's first 8 steps, training people only.

    Args:
        bags: The start bags (``bags.BagSampler.start_bags``).
        scores: The pinned zero-shot scores (``zeroshot.load_zeroshot_scores``): one row per
            segment and view, with ``segment``, ``view`` and ``z``.
        settings: The gain's settings (``gain`` in ``train.yaml``).

    Returns:
        The statistics of both views at the start of the run.

    Raises:
        StepError: If there is no start bag, or the scores lack a segment of one.
    """
    if not bags:
        raise StepError("the running statistics start from at least one bag")
    z = {
        (str(view), str(segment)): float(value)
        for segment, view, value in zip(scores["segment"], scores["view"], scores["z"], strict=True)
    }
    means: dict[str, list[float]] = {}
    for view in ("x0", "x1"):
        missing = sum((view, segment) not in z for bag in bags for segment in bag.segments)
        if missing:
            raise StepError(
                f"the zero-shot scores lack {missing} segments of the start bags on {view}"
            )
        means[view] = [
            float(np.mean([z[(view, segment)] for segment in bag.segments])) for bag in bags
        ]
    return ViewStats(
        RunningStats.start(means["x0"], settings), RunningStats.start(means["x1"], settings)
    )


def stand_in_stats(settings: TrainSettings) -> ViewStats:
    """Return the timing run's statistics: the stand-in μ and σ for both views.

    The timing run keeps no weights and writes no score, so it reads no zero-shot score
    (pilot protocol, section "Training", bullet "Check F5").

    Args:
        settings: The shared training settings (``timing.stand_in_start`` and ``gain``).

    Returns:
        The same statistics for ``x0`` and ``x1``.
    """
    start = settings.timing.stand_in_start
    stats = RunningStats.of(start.mu_logit, start.sigma_logit, settings.gain)
    return ViewStats(stats, stats)


# One step ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StepRecord:
    """What one step did, for its line of ``metrics.jsonl``: numbers only, no ID.

    Attributes:
        step: t, counted from 0 over the whole run.
        epoch: The epoch, from 1.
        seconds: The step's time, cache reads included.
        lr: LoRA's learning rate in this step.
        adapter_lr: The adapter's learning rate in this step.
        lambda_g: λG in this step; None for B0.
        loss: The step's loss before scaling: the mean of its bags' losses.
        ce_x0: The mean of the bags' CE(x0).
        ce_x1: The mean of the bags' CE(x1); None for B0.
        ce_x2: The mean of the bags' CE(x2); None for B0.
        ce_part: The mean of the bags' CE(x0) + 0.25 CE(x1) + 0.5 CE(x2), which the 4-epoch
            rule reads.
        hinge: The mean of the bags' soft hinge, before λG; None for B0.
        scale: The loss scale of this step.
        next_scale: The loss scale after the scaler's update.
        skipped: Whether the scaler skipped the update, after a gradient overflowed.
        grad_norm: The joint norm of all gradients after unscaling, before clipping; None
            when it is not finite, on a skipped step.
        clipped: Whether the clip shrank the gradients.
        calls: The model's forward calls.
        measures: The logged measures over the step's bags (``loss.step_measures``; for B0
            ``X0_MEASURES``); ``z0_over_z1`` may be None.
        stats: The running statistics after the step, for the next one; None for B0.
    """

    step: int
    epoch: int
    seconds: float
    lr: float
    adapter_lr: float
    lambda_g: float | None
    loss: float
    ce_x0: float
    ce_x1: float | None
    ce_x2: float | None
    ce_part: float
    hinge: float | None
    scale: float
    next_scale: float
    skipped: bool
    grad_norm: float | None
    clipped: bool
    calls: int
    measures: dict[str, int | float | None]
    stats: ViewStats | None

    def line(self) -> dict[str, Any]:
        """Return the step's line of ``metrics.jsonl``: JSON numbers, no ID, no per-item value.

        Returns:
            The line as plain JSON types: the step, its epoch and seconds, both learning
            rates, the loss and its CE parts, the loss scale before and after the step,
            whether it was skipped, the gradient norm, whether it was clipped, the calls and
            the logged measures. B1 and CPG add CE(x1), CE(x2), the hinge, λG and
            ``loss.STEP_MEASURES``; B0's line holds no key of the gain, only
            ``X0_MEASURES``. ``grad_norm`` is null on a step whose gradients overflowed, and
            ``z0_over_z1`` on a step whose mean |z̄1| is 0.
        """
        line: dict[str, Any] = {
            "step": self.step,
            "epoch": self.epoch,
            "seconds": self.seconds,
            "lr": self.lr,
            "adapter_lr": self.adapter_lr,
            "loss": self.loss,
            "ce_x0": self.ce_x0,
            "ce_part": self.ce_part,
        }
        optional = {
            "ce_x1": self.ce_x1,
            "ce_x2": self.ce_x2,
            "hinge": self.hinge,
            "lambda_g": self.lambda_g,
        }
        line |= {key: value for key, value in optional.items() if value is not None}
        line |= {
            "scale": self.scale,
            "next_scale": self.next_scale,
            "skipped": self.skipped,
            "grad_norm": self.grad_norm,
            "clipped": self.clipped,
            "calls": self.calls,
        }
        return line | self.measures


def _bag_z(
    model: Any,
    prompts: Mapping[int, Mapping[str, Any]],
    outputs: Sequence[torch.Tensor],
    answer_ids: tuple[int, int],
    device_type: str,
) -> tuple[torch.Tensor, int]:
    """The z of one bag's clips on one view, in the bag's order, and the calls it took.

    One call per chunk count, in rising order, under fp16 autocast: the one-chunk clips
    together and a two-chunk clip alone, so no clip is padded (pilot protocol, section
    "Training", bullet "Calls").
    """
    groups: dict[int, list[int]] = {}
    for index, output in enumerate(outputs):
        groups.setdefault(int(output.shape[0]) if output.ndim else 0, []).append(index)
    found: dict[int, torch.Tensor] = {}
    for count in sorted(groups):
        members = groups[count]
        if count not in prompts:
            raise StepError(f"no prompt of {count} chunks is given for the clips of a call")
        with torch.autocast(device_type, dtype=torch.float16):
            z = cached_z(model, prompts[count], [outputs[index] for index in members], answer_ids)
        for position, index in enumerate(members):
            found[index] = z[position]
    return torch.stack([found[index] for index in range(len(outputs))]), len(groups)


def _x0_measures(labels: Sequence[int], z0_bar: torch.Tensor, z0: torch.Tensor) -> dict[str, float]:
    """B0's logged measures, defined as in ``loss.step_measures``: no gain, so no statistics."""
    with torch.no_grad():
        bars = z0_bar.detach().to(torch.float64)
        clips = z0.detach().to(torch.float32)
        y = torch.tensor(list(labels), dtype=clips.dtype, device=clips.device)
        return {
            "z0_bar_mean": float(bars.mean()),
            "z0_bar_mean_square": float((bars * bars).mean()),
            "z0_bar_abs_mean": float(bars.abs().mean()),
            # Adding 0.0 turns the -0.0 of a zero push into 0.0.
            "ce_push": float((torch.sigmoid(clips) - y[:, None]).abs().mean()) + 0.0,
        }


def train_step(
    trainable: Trainable,
    optimiser: Optimiser,
    watch: LayerWatch,
    arm: ArmConfig,
    bags: Sequence[Bag],
    read: Callable[[str, str], torch.Tensor],
    prompts: Mapping[int, Mapping[str, Any]],
    stats: ViewStats | None,
    step: int,
    epoch: int,
) -> StepRecord:
    """Make one step: the bags' forward and backward passes, then one update.

    Bag by bag, view by view in the arm's order (pilot protocol, section "Training",
    bullets "Calls", "The gain term" and "Loss and schedules"): ``x1`` first, whose bag mean
    the gain reads without gradient, then ``x0`` with the gain, then ``x2``. A view of a bag
    takes one call per chunk count under fp16 autocast, then one backward pass of its part
    of the bag's loss, divided by the number of bags and scaled, with the watch paused. A
    bag's loss is CE(x0) + 0.25 CE(x1) + 0.5 CE(x2) + λG · soft hinge(G_b), each CE the mean
    over the bag's clips; the gain uses the running statistics from before the step, and
    while λG is 0 (B1, and CPG's step 0) it is computed but not trained on. Then the watch
    is read; the gradients are unscaled and clipped to a joint norm of
    ``grad_clip_norm``; the scaler steps AdamW or skips the update after an overflow, and
    updates its scale; the schedule moves on. The running statistics of both views update
    from the step's bag means, also on a skipped step.

    Args:
        trainable: The model that trains, in train mode.
        optimiser: Its optimiser, schedule and scaler (``make_optimiser``).
        watch: The overflow watch on the training model's language model and logits
            (``voxtral.watch_layers``).
        arm: The arm: its views, λG and the shared settings.
        bags: The step's bags, positive, negative, positive, negative.
        read: Gives a segment's stored encoder output of a view: ``read(view, segment)``,
            shape (C, 1500, W), fp16.
        prompts: The token IDs and mask of a clip of each chunk count, by chunk count.
        stats: The running statistics from before the step; None for B0.
        step: t, counted from 0 over the whole run, below T.
        epoch: The epoch, from 1, for the record.

    Returns:
        The step's record, with the statistics for the next step.

    Raises:
        ValueError: If t is not below T or not the schedule's step, the statistics do not
            fit the arm, there is no bag, or the arm's views lack x0.
        StepError: If the model is not in train mode; if a z or a part of a bag's loss is
            not finite (check F1), or the watch flagged a part, which stops the run before
            the update; or if the loss scale falls below ``rules.f1_min_scale`` (check F1),
            after the update. The messages name the step and the bag's place in it, never
            a participant.
    """
    clock = time.monotonic()
    total = optimiser.total_steps
    if isinstance(step, bool) or not 0 <= step < total:
        raise ValueError(f"the step must lie from 0 to {total - 1}, not {step!r}")
    if optimiser.schedule.last_epoch != step:
        raise ValueError(
            f"the schedule stands at step {optimiser.schedule.last_epoch}, not at {step}: steps"
            " run in order from 0, so that the learning rate and λG count alike"
        )
    if (arm.lambda_g is None) != (stats is None):
        raise ValueError(f"arm {arm.arm} takes running statistics only if it has a gain")
    if not bags:
        raise ValueError("a step needs at least one bag")
    if "x0" not in arm.views:
        raise ValueError(f"every arm trains on x0, not only on {list(arm.views)}")
    model = trainable.model
    if not model.training:
        raise StepError("the training model is not in train mode: checkpointing would be off")
    shared = arm.shared
    weights = {"x0": 1.0, "x1": shared.loss.x1_weight, "x2": shared.loss.x2_weight}
    settings = shared.gain
    weight_g = None
    if stats is not None and arm.lambda_g is not None:
        weight_g = lambda_g(step, total, settings.ramp_share, peak=arm.lambda_g)
    optimizer, scaler = optimiser.optimizer, optimiser.scaler
    rates = [float(group["lr"]) for group in optimizer.param_groups]
    scale = float(scaler.get_scale())
    optimizer.zero_grad(set_to_none=True)
    device = _device_type(model)
    count = len(bags)
    ce: dict[str, list[float]] = {view: [] for view in arm.views}
    hinges: list[float] = []
    losses: list[float] = []
    z0_bars: list[torch.Tensor] = []
    z1_bars: list[torch.Tensor] = []
    z0_rows: list[torch.Tensor] = []
    calls = 0
    for number, bag in enumerate(bags, start=1):
        z1_bar: torch.Tensor | None = None
        bag_loss = 0.0
        for view in arm.views:
            where = f"step {step}, bag {number} of {count}, {view}"
            outputs = [read(view, segment) for segment in bag.segments]
            z, made = _bag_z(model, prompts, outputs, trainable.answer_ids, device)
            calls += made
            bad = int((~torch.isfinite(z)).sum())
            if bad:
                raise StepError(
                    f"{where}: {bad} of {z.numel()} scores are not finite; check F1 fails"
                    f" (the watch flagged {len(watch.flagged())} parts so far)"
                )
            value = ce_on_z(z, bag.label)
            ce[view].append(float(value.detach()))
            term = weights[view] * value
            if view == "x1":
                z1_bar = z.detach().mean()
                z1_bars.append(z1_bar)
            elif view == "x0":
                z0_bar = z.mean()
                z0_bars.append(z0_bar.detach())
                z0_rows.append(z.detach())
                if stats is not None and weight_g is not None:
                    if z1_bar is None:
                        raise StepError(f"{where}: the gain needs x1 before x0")
                    g = gain(z0_bar.reshape(1), z1_bar.reshape(1), [bag.label], stats.x0, stats.x1)
                    hinge = soft_hinge(g, settings.margin, settings.softness)[0]
                    hinges.append(float(hinge.detach()))
                    if weight_g > 0.0:
                        term = term + weight_g * hinge
            if not bool(torch.isfinite(term)):
                raise StepError(f"{where}: the loss is not finite; check F1 fails")
            bag_loss += float(term.detach())
            with watch.pause():
                scaler.scale(term / count).backward()
        losses.append(bag_loss)
    flagged = watch.flagged()
    if flagged:
        raise StepError(
            f"step {step}: the overflow watch flagged {_listed(flagged)}: the run stops before"
            " the update, and every later run uses the pre-set fallback, the flagged layers in"
            ' fp32 (pilot protocol, section "Training", bullet "Overflow in training")'
        )
    scaler.unscale_(optimizer)
    clip = shared.optimiser.grad_clip_norm
    parameters = [value for group in optimizer.param_groups for value in group["params"]]
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, clip))
    scaler.step(optimizer)
    scaler.update()
    with warnings.catch_warnings():
        # The schedule moves on with a skipped step on purpose; when the scaler skips the
        # first one, PyTorch would warn that the schedule stepped before AdamW did.
        warnings.filterwarnings("ignore", message=SCHEDULE_FIRST, category=UserWarning)
        optimiser.schedule.step()
    next_scale = float(scaler.get_scale())
    labels = [bag.label for bag in bags]
    z0_bar_all, z0_all = torch.stack(z0_bars), torch.stack(z0_rows)
    if stats is None or weight_g is None:
        measures: dict[str, int | float | None] = dict(_x0_measures(labels, z0_bar_all, z0_all))
        after = None
    else:
        z1_bar_all = torch.stack(z1_bars)
        measures = step_measures(
            labels, z0_bar_all, z1_bar_all, z0_all, stats.x0, stats.x1, weight_g, settings
        )
        after = stats.update(z0_bar_all, z1_bar_all)

    def mean_of(view: str) -> float | None:
        return float(np.mean(ce[view])) if view in ce else None

    ce_part = float(
        np.mean([sum(weights[view] * ce[view][index] for view in ce) for index in range(count)])
    )
    record = StepRecord(
        step=step,
        epoch=epoch,
        seconds=time.monotonic() - clock,
        lr=rates[0],
        adapter_lr=rates[1],
        lambda_g=weight_g,
        loss=float(np.mean(losses)),
        ce_x0=float(np.mean(ce["x0"])),
        ce_x1=mean_of("x1"),
        ce_x2=mean_of("x2"),
        ce_part=ce_part,
        hinge=float(np.mean(hinges)) if hinges else None,
        scale=scale,
        next_scale=next_scale,
        skipped=next_scale < scale,
        grad_norm=norm if math.isfinite(norm) else None,
        clipped=math.isfinite(norm) and clip / (norm + 1e-6) < 1.0,
        calls=calls,
        measures=measures,
        stats=after,
    )
    floor = shared.rules.f1_min_scale
    if next_scale < floor:
        raise StepError(
            f"step {step}: the loss scale fell to {next_scale:g}, below {floor:g}, where even"
            " unscaled gradients overflow: check F1 fails (pilot protocol, section"
            ' "Training", bullet "Check F1")'
        )
    return record


# Saving, releasing and the scoring model -----------------------------------------------------


def save_adapter(model: Any, folder: Path) -> str:
    """Save LoRA and the adapter copy, in fp32, and return the SHA-256 of their file.

    PEFT's ``save_pretrained`` writes ``adapter_model.safetensors`` (the trained tensors in
    the format they have), ``adapter_config.json`` and a ``README.md``. The file's bytes
    depend on the values alone, so two saves of the same weights are equal.

    Args:
        model: The training model (``Trainable.model``).
        folder: The new folder, such as ``<run>/adapter``.

    Returns:
        The SHA-256 of ``adapter_model.safetensors``.

    Raises:
        StepError: If the folder exists already, or PEFT did not write the two files.
    """
    if folder.exists():
        raise StepError(f"{folder.name} exists already: a run never overwrites saved weights")
    model.save_pretrained(folder)
    path = folder / ADAPTER_FILE
    if not path.is_file() or not (folder / ADAPTER_CONFIG).is_file():
        raise StepError(f"the saved weights lack {ADAPTER_FILE} or {ADAPTER_CONFIG}")
    return sha256_file(path)


def release_training(
    trainable: Trainable, optimiser: Optimiser, watch: LayerWatch | None
) -> dict[str, int | None]:
    """Free the training model and check that nothing of it is left, before scoring loads.

    The two models would not fit on the GPU together (pilot protocol, section "Training",
    bullet "Scoring"). The watch's hooks are removed (its counts stay readable), and the
    model, the optimiser with its gradients' owners, the schedule and the scaler are
    dropped from their holders; the garbage collector then runs and PyTorch's cache is
    emptied. The caller keeps no other reference to them: the model must be gone, and at
    most ``MAX_LEFT_BYTES`` may stay allocated on the GPU.

    Args:
        trainable: The model that trained; its ``model`` becomes None.
        optimiser: Its optimiser; its ``optimizer``, ``schedule`` and ``scaler`` become
            None.
        watch: The training model's overflow watch, if one is on.

    Returns:
        ``allocated_bytes`` and ``reserved_bytes`` on the GPU after the release; None
        without a GPU.

    Raises:
        StepError: If the model was released already, something still holds it, or more
            than ``MAX_LEFT_BYTES`` stay allocated.
    """
    if trainable.model is None:
        raise StepError("the training model is released already")
    gone = weakref.ref(trainable.model)
    if watch is not None:
        watch.remove()
    trainable.model = None
    optimiser.optimizer = None
    optimiser.schedule = None
    optimiser.scaler = None
    gc.collect()
    if gone() is not None:
        raise StepError(
            "the training model is still held after its release: the scoring model would not"
            " fit beside it"
        )
    if not torch.cuda.is_available():
        return {"allocated_bytes": None, "reserved_bytes": None}
    torch.cuda.empty_cache()
    allocated = int(torch.cuda.memory_allocated())
    if allocated > MAX_LEFT_BYTES:
        raise StepError(
            f"{allocated / 2**20:.0f} MiB stay allocated on the GPU after the training model's"
            f" release, more than {MAX_LEFT_BYTES / 2**20:.0f} MiB"
        )
    return {"allocated_bytes": allocated, "reserved_bytes": int(torch.cuda.memory_reserved())}


def cast_for_scoring(model: Any) -> int:
    """Cast LoRA and the adapter copy to fp16, the format that training computes them in.

    Under fp16 autocast, training computes with fp16 copies of its fp32 master weights,
    so the scoring model holds them in fp16 too (pilot protocol, section "Training",
    bullet "Scoring"). The frozen weights stay as they are.

    Args:
        model: A PEFT model, the one that trained or one loaded from saved weights.

    Returns:
        The tensors cast.
    """
    cast = 0
    for name, value in model.named_parameters():
        if _is_lora(name) or _is_adapter(name):
            value.data = value.data.to(torch.float16)
            cast += 1
    return cast


def check_for_scoring(model: Any) -> None:
    """Stop unless a model is fit to score: every weight fp16, every module in eval mode.

    LoRA and the adapter score in fp16, the format that training computes them in, and the
    frozen weights load in fp16; in train mode LoRA's dropout would act.

    Args:
        model: The scoring model, a PEFT model.

    Raises:
        StepError: If LoRA does not sit on the 7 projections of every layer, the adapter is
            not its trained copy, a weight is not fp16, or a module is in train mode.
    """
    _check_layout(model)
    wrong = [name for name, value in model.named_parameters() if value.dtype != torch.float16]
    if wrong:
        trained = [name for name in wrong if _is_lora(name) or _is_adapter(name)]
        raise StepError(
            f"weights not in fp16: {_listed(wrong)}, {len(trained)} of them trained; the"
            " scoring model computes in fp16"
        )
    training = [name or "the model" for name, module in model.named_modules() if module.training]
    if training:
        raise StepError(f"modules in train mode: {_listed(training)}; scoring needs eval mode")


def load_for_scoring(snapshot: Path, folder: Path, device: str = "cuda") -> LoadedModel:
    """Build the scoring model from saved weights: a fresh model, LoRA and the adapter in fp16.

    Pilot protocol, section "Training", bullet "Scoring": the pinned Voxtral loads on the
    CPU, the pass-through takes the encoder's place and the encoder is dropped, PEFT loads
    the saved folder, LoRA and the adapter copy are cast to fp16 (``cast_for_scoring``), the
    model goes to evaluation mode and then to the device, and ``check_for_scoring`` checks
    it. What scores
    is thus the kept file. Call it after ``release_training``.

    Args:
        snapshot: Folder of the pinned snapshot (see ``models.local_snapshot``).
        folder: The saved weights (``save_adapter``).
        device: Where the model scores.

    Returns:
        The scoring model, its processor and the largest frozen weight, for
        ``voxtral.cached_score``.

    Raises:
        StepError: If the snapshot does not load, the folder lacks the saved files, or the
            model is not fit to score (``check_for_scoring``).
    """
    from peft import PeftModel

    if not (folder / ADAPTER_FILE).is_file() or not (folder / ADAPTER_CONFIG).is_file():
        raise StepError(f"{folder.name} holds no saved weights ({ADAPTER_FILE}, {ADAPTER_CONFIG})")
    loaded = load_model(snapshot, device="cpu")
    use_cached_encoder(loaded.model)  # the encoder it hands back is dropped here
    model = PeftModel.from_pretrained(loaded.model, folder)
    cast_for_scoring(model)
    model.eval()
    model.to(device)
    check_for_scoring(model)
    return LoadedModel(
        model=model, processor=loaded.processor, largest_weight=loaded.largest_weight
    )


# The checks and rules read on the training logs ----------------------------------------------


def _mean(lines: Sequence[Mapping[str, Any]], key: str) -> float | None:
    """The mean of a key over lines that hold it; None if none does."""
    values = [float(line[key]) for line in lines if line.get(key) is not None]
    return float(np.mean(values)) if values else None


def _share(part: int, whole: int) -> float | None:
    """A share of whole numbers; None over nothing, never NaN."""
    return part / whole if whole else None


def _spread(mean: float | None, mean_square: float | None) -> float | None:
    """The population SD from a mean and a mean square, as σ is computed: 0 if rounding errs."""
    if mean is None or mean_square is None:
        return None
    return math.sqrt(max(mean_square - mean * mean, 0.0))


def _epoch_measures(lines: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One epoch's measures: the loss parts, the shares and the statistics against the data."""
    measures: dict[str, Any] = {
        "steps": len(lines),
        "skipped": sum(bool(line["skipped"]) for line in lines),
        "minutes": sum(float(line["seconds"]) for line in lines) / 60,
    }
    for key in ("loss", "ce_x0", "ce_x1", "ce_x2", "ce_part", "hinge", "ce_push"):
        value = _mean(lines, key)
        if value is not None:
            measures[key] = value
    mean0, square0 = _mean(lines, "z0_bar_mean"), _mean(lines, "z0_bar_mean_square")
    measures["x0"] = {"measured_mean": mean0, "measured_spread": _spread(mean0, square0)}
    if "bags_positive_words_right" not in lines[0]:
        return measures  # B0: no gain
    bags = {group: sum(int(line[f"bags_{group}"]) for line in lines) for group in GROUPS}
    active = {group: sum(int(line[f"active_{group}"]) for line in lines) for group in GROUPS}

    def counted(groups: Sequence[str]) -> float | None:
        return _share(sum(active[group] for group in groups), sum(bags[group] for group in groups))

    wrong = [group for group in GROUPS if group.endswith("words_wrong")]
    measures["active_share"] = {
        "all": counted(GROUPS),
        "positive": counted([group for group in GROUPS if group.startswith("positive")]),
        "negative": counted([group for group in GROUPS if group.startswith("negative")]),
        "words_right": counted([group for group in GROUPS if group.endswith("words_right")]),
        "words_wrong": counted(wrong),
    }
    measures["bags"], measures["active_bags"] = sum(bags.values()), sum(active.values())
    # The clamp sets the reference to 0 on every bag whose words were wrong.
    measures["clamp_share"] = _share(sum(bags[group] for group in wrong), sum(bags.values()))
    mean1, square1 = _mean(lines, "z1_bar_mean"), _mean(lines, "z1_bar_mean_square")
    measures["x0"] |= {"mu": _mean(lines, "mu0"), "sigma": _mean(lines, "sigma0")}
    measures["x1"] = {
        "measured_mean": mean1,
        "measured_spread": _spread(mean1, square1),
        "mu": _mean(lines, "mu1"),
        "sigma": _mean(lines, "sigma1"),
    }
    for view in ("0", "1"):
        measures[f"sigma{view}_at_floor_share"] = _share(
            sum(int(line[f"sigma{view}_at_floor"]) for line in lines), len(lines)
        )
    size0 = sum(float(line["z0_bar_abs_mean"]) for line in lines)
    size1 = sum(float(line["z1_bar_abs_mean"]) for line in lines)
    measures["z0_over_z1"] = size0 / size1 if size1 else None
    for key in ("gain_push", "gain_offset"):
        measures[key] = _mean(lines, key)
    return measures


def _f1(lines: Sequence[Mapping[str, Any]], min_scale: float) -> dict[str, Any]:
    """Check F1's counts: skipped steps, the lowest scale, non-finite losses and the norms."""
    longest = run = 0
    for line in lines:
        run = run + 1 if line["skipped"] else 0
        longest = max(longest, run)
    lowest = min(min(float(line["scale"]), float(line["next_scale"])) for line in lines)
    keys = ("loss", "ce_x0", "ce_x1", "ce_x2", "ce_part", "hinge")
    non_finite = sum(
        any(line.get(key) is not None and not math.isfinite(float(line[key])) for key in keys)
        for line in lines
    )
    stepped = [line for line in lines if not line["skipped"]]
    norms = [float(line["grad_norm"]) for line in stepped if line["grad_norm"] is not None]
    return {
        "skipped_steps": sum(bool(line["skipped"]) for line in lines),
        "longest_skipped_run": longest,
        "lowest_scale": lowest,
        "min_scale": min_scale,
        "non_finite_losses": non_finite,
        "largest_grad_norm": max(norms) if norms else None,
        "clipped_share": _share(sum(bool(line["clipped"]) for line in stepped), len(stepped)),
        "passes": lowest >= min_scale and non_finite == 0,
    }


def _alive(
    lines: Sequence[Mapping[str, Any]],
    rule: AliveRule,
    margin: float,
    reads_f2: bool,
    reads_rules: bool,
) -> dict[str, Any]:
    """The gain-alive share over the rule's epochs together, with check F2 and the rule's margin.

    "Fewer than 20%" is a share below 0.20 and "more than 80%" one above 0.80, so a share
    of exactly 0.20 or 0.80 changes nothing and passes F2. The share is compared as a
    fraction, with no rounding. ``f2_passes`` is given only where F2 reads the run (a CPG
    run), and ``margin_if_read`` only where the gain-alive rule reads it (the first CPG run
    of round 1, fold 1 that finishes); elsewhere the share is a diagnostic.
    """
    chosen = [line for line in lines if int(line["epoch"]) in rule.epochs]
    bags = sum(int(line[f"bags_{group}"]) for line in chosen for group in GROUPS)
    active = sum(int(line[f"active_{group}"]) for line in chosen for group in GROUPS)
    reading: dict[str, Any] = {
        "epochs": list(rule.epochs),
        "bags": bags,
        "active": active,
        "margin": margin,
        "share": float(Fraction(active, bags)) if bags else None,
    }
    if not bags:
        verdicts: dict[str, Any] = {"f2_passes": None, "margin_if_read": None}
    else:
        share = Fraction(active, bags)
        below = share < Fraction(repr(rule.min_share))
        above = share > Fraction(repr(rule.max_share))
        new = rule.margin_if_below if below else rule.margin_if_above if above else margin
        verdicts = {"f2_passes": not (below or above), "margin_if_read": new}
    if reads_f2:
        reading["f2_passes"] = verdicts["f2_passes"]
    if reads_rules:
        reading["margin_if_read"] = verdicts["margin_if_read"]
    return reading


def _falling(lines: Sequence[Mapping[str, Any]], epoch: int, rule: FallingRule) -> dict[str, Any]:
    """The 4-epoch rule's measures on one epoch: is the CE part still clearly falling?

    The halves are the epoch's first and last ⌊n/2⌋ steps, the middle step left out when n
    is odd. The CE part falls clearly if the second half's mean lies below the first's by
    more than ``min_standard_errors`` standard errors of the difference, from the per-step
    values (sample variances), and by at least ``min_drop_share`` of the first half's mean
    (``fires``; ``rule_measures`` keeps it only where the rule reads the run). The total
    loss and the hinge are reported beside it.
    """
    steps = sorted(
        (line for line in lines if int(line["epoch"]) == epoch), key=lambda line: line["step"]
    )
    half = len(steps) // 2
    reading: dict[str, Any] = {"epoch": epoch, "steps": len(steps), "half_steps": half}
    if half < 2:
        return reading | {"fires": False}  # no spread from fewer than 2 steps per half
    first, second = steps[:half], steps[len(steps) - half :]
    values = [[float(line["ce_part"]) for line in part] for part in (first, second)]
    means = [statistics.fmean(part) for part in values]
    error = math.sqrt(sum(statistics.variance(part) / half for part in values))
    drop = means[0] - means[1]
    fires = drop > rule.min_standard_errors * error and drop >= rule.min_drop_share * means[0]
    reading |= {
        "ce_part": {"first_half": means[0], "second_half": means[1]},
        "drop": drop,
        "drop_share": drop / means[0] if means[0] else None,
        "standard_error": error,
        "fires": fires,
    }
    for key in ("loss", "hinge"):
        halves = [_mean(part, key) for part in (first, second)]
        if None not in halves:
            reading[key] = {"first_half": halves[0], "second_half": halves[1]}
    return reading


def rule_measures(
    metrics: Sequence[Mapping[str, Any]],
    settings: TrainSettings,
    training_hours: float | None = None,
    *,
    reads_f2: bool = False,
    reads_rules: bool = False,
    reads_f5: bool = False,
) -> dict[str, Any]:
    """Read a run's per-step metrics for the checks and the allowed rules.

    Pilot protocol, section "Training", bullets "Check F1", "F2 and the two allowed rules"
    and "Check F5". Every run gets the measures; a verdict is given only where the protocol
    reads the run: F1 on every run, F2 on every CPG run (``reads_f2``), the gain-alive rule,
    the 4-epoch rule and F5 once, on the first CPG run of round 1, fold 1 that finishes
    (``reads_rules``), and F5 alone again only on its rerun at 8 segments after F5 failed
    there at 16 (``reads_f5``), while the readings of m and of the epochs stand. Every other
    run, other runs of CPG's round 1, fold 1 included, gets the shares, the halves and the
    hours as diagnostics; the job decides which run reads what.

    - ``epochs``: per epoch, its steps, skipped steps and minutes, the means of the loss
      and its parts, and, for B1 and CPG, the active shares (all bags, by class and by
      whether the words were right), the clamp's share, how often σ sat at its floor, the
      measured mean and spread of the bag means beside the average μ and σ, |z0|/|z1| and
      the pushes.
    - ``f1``: skipped steps (total and longest run), the lowest loss scale, losses that are
      not finite, the largest gradient norm and the share of clipped steps; F1 passes if
      the scale never fell below ``rules.f1_min_scale`` and every loss was finite.
    - ``alive`` (B1 and CPG; None for B0): the share of bags with G_b < m over the epochs of
      ``rules.alive``; with ``reads_f2``, whether it passes F2 (0.20 to 0.80, both
      included); with ``reads_rules``, the margin that the gain-alive rule sets.
    - ``falling``: the 4-epoch rule's measures on the run's last epoch; with
      ``reads_rules``, whether it fires.
    - ``f5`` (when the training hours are given): the training hours; with
      ``reads_rules``, also F5's reading, the hours times ``epochs_if_falling`` /
      ``epochs`` (4/3) if the 4-epoch rule fires on a run of ``epochs``, against
      ``rules.f5_max_hours``; with ``reads_f5``, F5's reading on the run's own hours,
      never times 4/3, since the epochs are set by then.

    Args:
        metrics: The run's lines of ``metrics.jsonl`` (``StepRecord.line``), in step order.
        settings: The shared training settings of the run.
        training_hours: The run's training part, from the job's start to the saved weights.
        reads_f2: Whether check F2 reads this run: a CPG run.
        reads_rules: Whether the gain-alive rule, the 4-epoch rule and check F5 read this
            run: the first CPG run of round 1, fold 1 that finishes.
        reads_f5: Whether check F5 alone reads this run, on its own hours: the rerun of
            CPG's round 1, fold 1 at 8 segments after F5 failed at 16 on the first.

    Returns:
        The measures, as plain JSON types; a share over no bag is None, never NaN.

    Raises:
        ValueError: If there is no line, F2 or the rules would read a run that logs no
            gain, the rules or F5 alone would read a run that F2 does not read, both
            ``reads_rules`` and ``reads_f5`` are set, or ``reads_f5`` has no hours to read.
    """
    lines = list(metrics)
    if not lines:
        raise ValueError("a run's metrics need at least one step")
    gain_logged = "bags_positive_words_right" in lines[0]
    if (reads_f2 or reads_rules) and not gain_logged:
        raise ValueError("check F2 and the allowed rules read a CPG run, which logs the gain")
    if reads_rules and not reads_f2:
        raise ValueError("the allowed rules read CPG's round 1, fold 1, which check F2 reads too")
    if reads_f5 and not reads_f2:
        raise ValueError("check F5 reads a later run of CPG's round 1, fold 1, which F2 reads too")
    if reads_f5 and reads_rules:
        raise ValueError("a run is read either with the allowed rules or, later, by F5 alone")
    if reads_f5 and training_hours is None:
        raise ValueError("check F5 reads a run's training hours, and none were given")
    epochs = sorted({int(line["epoch"]) for line in lines})
    rules = settings.rules
    falling = _falling(lines, epochs[-1], rules.falling)
    fires = bool(falling.pop("fires"))
    if reads_rules:
        falling["fires"] = fires
    alive = None
    if gain_logged:
        alive = _alive(lines, rules.alive, settings.gain.margin, reads_f2, reads_rules)
    measures: dict[str, Any] = {
        "steps": len(lines),
        "epochs": {
            str(epoch): _epoch_measures([line for line in lines if int(line["epoch"]) == epoch])
            for epoch in epochs
        },
        "f1": _f1(lines, rules.f1_min_scale),
        "alive": alive,
        "falling": falling,
        "f5": None,
    }
    if training_hours is not None:
        measures["f5"] = {"training_hours": training_hours}
        if reads_rules:
            added = fires and settings.epochs < rules.falling.epochs_if_falling
            read = training_hours * rules.falling.epochs_if_falling / settings.epochs
            hours = read if added else training_hours
            measures["f5"] |= {
                "fourth_epoch": added,
                "hours": hours,
                "max_hours": rules.f5_max_hours,
                "passes": hours <= rules.f5_max_hours,
            }
        elif reads_f5:
            measures["f5"] |= {
                "fourth_epoch": False,
                "hours": training_hours,
                "max_hours": rules.f5_max_hours,
                "passes": training_hours <= rules.f5_max_hours,
            }
    return measures


def f5_projection(
    load_seconds: float, step_seconds: Sequence[float], total_steps: int, left_out: int
) -> float:
    """Return the timing run's projection of a three-view run's training hours (check F5).

    The load time plus T times the median seconds of the timed steps after the first
    ``left_out``, which warm up (pilot protocol, section "Training", bullet "Check F5"):
    with 10 timed steps, the median of the 2nd to the 10th. Above ``rules.f5_max_hours``
    the 8-segment fallback applies before B0's round 1.

    Args:
        load_seconds: From the job's start until the model and its inputs are loaded.
        step_seconds: The timed steps' seconds, in order.
        total_steps: T for 3 epochs, in the mode the timing sets.
        left_out: The first timed steps left out (``timing.first_steps_left_out``).

    Returns:
        The projected hours.

    Raises:
        ValueError: If no timed step is left, a time is negative or not finite, or T is
            below 1.
    """
    timed = [float(seconds) for seconds in step_seconds][left_out:]
    if not timed or isinstance(total_steps, bool) or total_steps < 1:
        raise ValueError(f"no timed step after the first {left_out}, or T of {total_steps!r}")
    if not all(math.isfinite(value) and value >= 0 for value in [load_seconds, *timed]):
        raise ValueError("times must be finite and at least 0")
    return (float(load_seconds) + total_steps * statistics.median(timed)) / 3600
