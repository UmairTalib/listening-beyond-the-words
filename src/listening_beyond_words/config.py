"""The training configs: the shared settings and the arm files that extend them.

``configs/pilot/train.yaml`` holds the settings that every arm shares (pilot protocol,
sections "Fixed settings" and "Training"). An arm file, ``configs/pilot/b0.yaml``,
``b1.yaml`` or ``cpg.yaml``, extends it with ``extends: train.yaml`` and adds only its own
keys: the arm, its views in the order in which a bag goes through the model, and λG, the
weight of the gain term (``lambda_g``; B0 has none). One level only: the shared file
extends nothing (engineering conventions, section 5). ``resolve_config`` reads a file
with the file that it extends; ``load_arm`` checks every key and value of an arm and
returns frozen dataclasses: ``ArmConfig``, with the shared settings in ``TrainSettings``.
``configs/pilot/zeroshot.yaml`` extends ``train.yaml`` too: the same resolver reads it,
and its own keys stay with ``zeroshot.ZeroshotSettings``.

A run records its resolved config, the file's own keys and the shared ones merged, with
the SHA-256 of its canonical JSON (``canonical_sha256``): it changes when a setting
changes, never with the order of the keys, the comments or the YAML style. A refusal is a
``StepError`` that names the file, the key and the rule, so a typo stops a run before it
starts. A key given twice in one mapping is refused too, where plain YAML readers keep
the last one. ``read_settings`` reads a settings file that extends nothing, such as
``configs/pilot/evaluate.yaml``, with the same refusals.
"""

import dataclasses
import hashlib
import itertools
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, NoReturn

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from listening_beyond_words.runinfo import StepError
from listening_beyond_words.voxtral import PROMPT

SHARED_FILE: Final = "train.yaml"  # the one file that an arm file extends, in its folder
EXTENDS: Final = "extends"
ARMS: Final = ("b0", "b1", "cpg")
# The views of each arm, in the order in which a bag goes through the model: x1 first,
# whose bag mean the gain reads, then x0 and x2 (pilot protocol, section "Training",
# bullets "Arms and what trains" and "Calls").
ARM_VIEWS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {"b0": ("x0",), "b1": ("x1", "x0", "x2"), "cpg": ("x1", "x0", "x2")}
)
CLIP_VERSIONS: Final = ("x0", "x1", "x2", "rev")
MODEL: Final = "voxtral"  # the pin of configs/pilot/models.yaml that the code trains
# precision.deterministic: pending until the commit that records the timing run sets
# strict or warn_only (pilot protocol, section "Training", bullet "Number formats and
# repeatability").
PENDING: Final = "pending"
STRICT: Final = "strict"
WARN_ONLY: Final = "warn_only"
MODES: Final = (PENDING, STRICT, WARN_ONLY)


# Reading the values ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Range:
    """The values that a float setting may take: finite, and within these bounds."""

    rule: str  # how a refusal names the range
    low: float = -math.inf
    high: float = math.inf
    low_open: bool = False
    high_open: bool = False

    def holds(self, value: float) -> bool:
        """Whether a float is finite and within the bounds."""
        above = value > self.low if self.low_open else value >= self.low
        below = value < self.high if self.high_open else value <= self.high
        return math.isfinite(value) and above and below


_FINITE = _Range("a finite float")
_POSITIVE = _Range("a float above 0", low=0.0, low_open=True)
_NON_NEGATIVE = _Range("a float of at least 0", low=0.0)
_SHARE = _Range("a float from 0 to 1", low=0.0, high=1.0)
_POSITIVE_SHARE = _Range("a float above 0 and at most 1", low=0.0, high=1.0, low_open=True)
_BELOW_ONE = _Range("a float from 0 to below 1", low=0.0, high=1.0, high_open=True)
_FRACTION = _Range("a float above 0 and below 1", 0.0, 1.0, low_open=True, high_open=True)
_ABOVE_ONE = _Range("a float above 1", low=1.0, low_open=True)
_ZERO = _Range("0.0, since B1 is CPG's code with λG = 0", low=0.0, high=0.0)


class _Block:
    """One mapping of a config file, refused unless it holds exactly the expected keys.

    Its readers check each value's type and range, and a refusal names the file and the
    key, such as "train.yaml: lora.rank must be a whole number of at least 1, not 0".
    """

    def __init__(self, value: Any, keys: tuple[str, ...], file: str, path: str = "") -> None:
        where = f"{file}: {path}" if path else file
        if not isinstance(value, Mapping):
            raise StepError(f"{where} must be a mapping, not {value!r}")
        missing = sorted(set(keys) - set(value))
        unknown = sorted(str(key) for key in set(value) - set(keys))
        if missing or unknown:
            raise StepError(f"{where}: missing keys {missing}, unknown keys {unknown}")
        self.values = value
        self.file = file
        self.prefix = f"{path}." if path else ""

    def name(self, key: str) -> str:
        """The key as a refusal names it: the file, then the key's path."""
        return f"{self.file}: {self.prefix}{key}"

    def refuse(self, key: str, rule: str) -> NoReturn:
        """Stop with the rule that the key's value breaks."""
        raise StepError(f"{self.name(key)} must be {rule}, not {self.values[key]!r}")

    def whole(self, key: str, minimum: int) -> int:
        """A whole number of at least ``minimum``; true and false are refused."""
        value = self.values[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            self.refuse(key, f"a whole number of at least {minimum}")
        return value

    def real(self, key: str, bounds: _Range) -> float:
        """A float within its bounds; a whole number is refused (write 1.0, not 1)."""
        value = self.values[key]
        if not isinstance(value, float) or not bounds.holds(value):
            self.refuse(key, bounds.rule)
        return value

    def flag(self, key: str) -> bool:
        """True or false."""
        value = self.values[key]
        if not isinstance(value, bool):
            self.refuse(key, "true or false")
        return value

    def choice(self, key: str, choices: tuple[str, ...]) -> str:
        """One of the values that the code implements."""
        value = self.values[key]
        if not isinstance(value, str) or value not in choices:
            self.refuse(key, repr(choices[0]) if len(choices) == 1 else f"one of {list(choices)}")
        return value

    def names(self, key: str, choices: tuple[str, ...]) -> tuple[str, ...]:
        """A list of different values from ``choices``, at least one."""
        value = self.values[key]
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) or item not in choices for item in value)
            or len(set(value)) != len(value)
        ):
            self.refuse(key, f"a list of different names from {list(choices)}")
        return tuple(value)

    def pattern(self, key: str) -> str:
        """A regular expression, as PEFT matches it against a module's full name."""
        value = self.values[key]
        valid = isinstance(value, str) and bool(value)
        if valid:
            try:
                re.compile(value)
            except re.error:
                valid = False
        if not valid:
            self.refuse(key, "a regular expression")
        return str(value)

    def block(self, key: str, keys: tuple[str, ...]) -> "_Block":
        """A nested mapping with exactly the expected keys."""
        return _Block(self.values[key], keys, self.file, f"{self.prefix}{key}")


def _keys(settings: type) -> tuple[str, ...]:
    """The keys of a block: the names of the fields of its dataclass, in their order."""
    return tuple(field.name for field in dataclasses.fields(settings))


# The shared settings: train.yaml ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SeedSettings:
    """The seed of each round (``seeds``): of its partitions and of its training.

    Attributes:
        round_1: The seed of round 1.
        round_2: The seed of round 2.
    """

    round_1: int
    round_2: int

    @classmethod
    def _from_block(cls, block: _Block) -> "SeedSettings":
        """Read and check the block."""
        return cls(round_1=block.whole("round_1", 0), round_2=block.whole("round_2", 0))

    def of(self, round_number: int) -> int:
        """Return the seed of a round.

        Args:
            round_number: 1 or 2.

        Returns:
            The round's seed.

        Raises:
            ValueError: If the round is not 1 or 2.
        """
        if isinstance(round_number, bool) or round_number not in (1, 2):
            raise ValueError(f"the pilot has rounds 1 and 2, not {round_number!r}")
        return self.round_1 if round_number == 1 else self.round_2


@dataclass(frozen=True, slots=True)
class LoraSettings:
    """LoRA and the trained adapter (``lora``; pilot protocol, fixed settings "LoRA").

    Attributes:
        rank: r.
        alpha: α; LoRA's output is scaled by α / r.
        dropout: LoRA's dropout, which acts only in training.
        bias: Which biases train: none.
        init_lora_weights: True: the second matrix starts at 0.
        use_rslora: False: the scale is α / r, not α / √r.
        use_dora: False: plain LoRA.
        trained_adapter: The module that trains as a copy and is saved with LoRA.
        targets: The modules that get LoRA, as a regular expression over full names.
    """

    rank: int
    alpha: int
    dropout: float
    bias: str
    init_lora_weights: bool
    use_rslora: bool
    use_dora: bool
    trained_adapter: str
    targets: str

    @classmethod
    def _from_block(cls, block: _Block) -> "LoraSettings":
        """Read and check the block."""
        return cls(
            rank=block.whole("rank", 1),
            alpha=block.whole("alpha", 1),
            dropout=block.real("dropout", _BELOW_ONE),
            bias=block.choice("bias", ("none",)),
            init_lora_weights=block.flag("init_lora_weights"),
            use_rslora=block.flag("use_rslora"),
            use_dora=block.flag("use_dora"),
            trained_adapter=block.choice("trained_adapter", ("multi_modal_projector",)),
            targets=block.pattern("targets"),
        )


@dataclass(frozen=True, slots=True)
class OptimiserSettings:
    """AdamW and its schedule (``optimiser``; pilot protocol, fixed settings "Optimiser").

    Attributes:
        learning_rate: LoRA's peak learning rate.
        adapter_learning_rate: The adapter's peak learning rate.
        betas: AdamW's two averaging factors.
        eps: AdamW's eps.
        amsgrad: Whether AdamW uses AMSGrad.
        foreach: Whether AdamW runs its ``foreach`` implementation.
        weight_decay: AdamW's weight decay, on every trained tensor.
        grad_clip_norm: The largest joint norm of all gradients, after unscaling.
        warmup_share: The share of a run's T steps, rounded up, over which the rate rises.
    """

    learning_rate: float
    adapter_learning_rate: float
    betas: tuple[float, float]
    eps: float
    amsgrad: bool
    foreach: bool
    weight_decay: float
    grad_clip_norm: float
    warmup_share: float

    @classmethod
    def _from_block(cls, block: _Block) -> "OptimiserSettings":
        """Read and check the block."""
        betas = block.values["betas"]
        if (
            not isinstance(betas, list)
            or len(betas) != 2
            or any(not isinstance(beta, float) or not _BELOW_ONE.holds(beta) for beta in betas)
        ):
            block.refuse("betas", "a list of 2 floats from 0 to below 1")
        return cls(
            learning_rate=block.real("learning_rate", _POSITIVE),
            adapter_learning_rate=block.real("adapter_learning_rate", _POSITIVE),
            betas=(betas[0], betas[1]),
            eps=block.real("eps", _POSITIVE),
            amsgrad=block.flag("amsgrad"),
            foreach=block.flag("foreach"),
            weight_decay=block.real("weight_decay", _NON_NEGATIVE),
            grad_clip_norm=block.real("grad_clip_norm", _POSITIVE),
            warmup_share=block.real("warmup_share", _SHARE),
        )


@dataclass(frozen=True, slots=True)
class ScalerSettings:
    """PyTorch's dynamic loss scaler (``precision.scaler``), with its defaults.

    Attributes:
        init_scale: The scale at step 0.
        growth_factor: The scale's factor after ``growth_interval_steps`` clean steps.
        backoff_factor: The scale's factor when a gradient overflows; the step is skipped.
        growth_interval_steps: Clean steps before the scale grows.
    """

    init_scale: float
    growth_factor: float
    backoff_factor: float
    growth_interval_steps: int

    @classmethod
    def _from_block(cls, block: _Block) -> "ScalerSettings":
        """Read and check the block."""
        return cls(
            init_scale=block.real("init_scale", _POSITIVE),
            growth_factor=block.real("growth_factor", _ABOVE_ONE),
            backoff_factor=block.real("backoff_factor", _FRACTION),
            growth_interval_steps=block.whole("growth_interval_steps", 1),
        )


@dataclass(frozen=True, slots=True)
class PrecisionSettings:
    """Number formats and repeatability (``precision``).

    Attributes:
        autocast: The format of the forward pass: fp16.
        masters: The format of every trained number: fp32.
        scaler: The loss scaler.
        checkpointing: Gradient checkpointing: on every layer.
        deterministic: ``pending`` until the timing run sets ``strict`` or ``warn_only``.
        strict_max_extra_time_share: How much longer strict mode's median three-view step
            may take than warn-only mode's, for strict mode to be chosen.
    """

    autocast: str
    masters: str
    scaler: ScalerSettings
    checkpointing: str
    deterministic: str
    strict_max_extra_time_share: float

    @classmethod
    def _from_block(cls, block: _Block) -> "PrecisionSettings":
        """Read and check the block."""
        return cls(
            autocast=block.choice("autocast", ("fp16",)),
            masters=block.choice("masters", ("fp32",)),
            scaler=ScalerSettings._from_block(block.block("scaler", _keys(ScalerSettings))),
            checkpointing=block.choice("checkpointing", ("every_layer",)),
            deterministic=block.choice("deterministic", MODES),
            strict_max_extra_time_share=block.real("strict_max_extra_time_share", _NON_NEGATIVE),
        )


@dataclass(frozen=True, slots=True)
class SamplerSettings:
    """The bag sampler (``sampler``); its bags' own settings are ``bags`` in ``data.yaml``.

    Attributes:
        positive_bags: Bags of PHQ-8-positive participants per step.
        negative_bags: Bags of PHQ-8-negative participants per step.
        epoch_end: When an epoch ends: when a class has fewer bags left than a step takes.
        draw: Which segments a participant offers: those used least often, ties at random.
        bag_choice: Which bags train: lowest trained share of their participant first.
        step_order: How the chosen bags fill the steps: shuffled within their class.
    """

    positive_bags: int
    negative_bags: int
    epoch_end: str
    draw: str
    bag_choice: str
    step_order: str

    @classmethod
    def _from_block(cls, block: _Block) -> "SamplerSettings":
        """Read and check the block."""
        return cls(
            positive_bags=block.whole("positive_bags", 1),
            negative_bags=block.whole("negative_bags", 1),
            epoch_end=block.choice("epoch_end", ("a_class_runs_out",)),
            draw=block.choice("draw", ("least_trained_first",)),
            bag_choice=block.choice("bag_choice", ("lowest_trained_share_first",)),
            step_order=block.choice("step_order", ("shuffled",)),
        )


@dataclass(frozen=True, slots=True)
class LossSettings:
    """The weights of the cross-entropy terms besides CE(x0), which weighs 1 (``loss``).

    Attributes:
        x1_weight: The weight of CE(x1).
        x2_weight: The weight of CE(x2).
    """

    x1_weight: float
    x2_weight: float

    @classmethod
    def _from_block(cls, block: _Block) -> "LossSettings":
        """Read and check the block."""
        return cls(
            x1_weight=block.real("x1_weight", _NON_NEGATIVE),
            x2_weight=block.real("x2_weight", _NON_NEGATIVE),
        )


@dataclass(frozen=True, slots=True)
class GainSettings:
    """The gain term and its running statistics (``gain``; fixed settings "Gain term").

    Attributes:
        margin: m.
        softness: s.
        ramp_share: λG rises over this share of a run's T steps, rounded up.
        momentum: Of the running averages of the bag means and of their squares.
        start_steps: The sampler's first steps whose bags start the averages.
        floor_share: σ never falls below this share of the start spread ...
        floor_logit: ... nor below this.
    """

    margin: float
    softness: float
    ramp_share: float
    momentum: float
    start_steps: int
    floor_share: float
    floor_logit: float

    @classmethod
    def _from_block(cls, block: _Block) -> "GainSettings":
        """Read and check the block."""
        return cls(
            margin=block.real("margin", _FINITE),
            softness=block.real("softness", _POSITIVE),
            ramp_share=block.real("ramp_share", _POSITIVE_SHARE),
            momentum=block.real("momentum", _BELOW_ONE),
            start_steps=block.whole("start_steps", 1),
            floor_share=block.real("floor_share", _NON_NEGATIVE),
            floor_logit=block.real("floor_logit", _POSITIVE),
        )


@dataclass(frozen=True, slots=True)
class AliveRule:
    """The gain-alive rule and check F2 (``rules.alive``).

    Attributes:
        epochs: The epochs whose bags the active share counts, together.
        min_share: Below this share, m becomes ``margin_if_below``.
        max_share: Above this share, m becomes ``margin_if_above``; F2 passes from
            ``min_share`` to ``max_share``, both included.
        margin_if_below: m after a share below ``min_share``.
        margin_if_above: m after a share above ``max_share``.
    """

    epochs: tuple[int, ...]
    min_share: float
    max_share: float
    margin_if_below: float
    margin_if_above: float

    @classmethod
    def _from_block(cls, block: _Block) -> "AliveRule":
        """Read and check the block."""
        epochs = block.values["epochs"]
        if (
            not isinstance(epochs, list)
            or not epochs
            or any(isinstance(epoch, bool) or not isinstance(epoch, int) for epoch in epochs)
            or epochs[0] < 1
            or any(later <= earlier for earlier, later in itertools.pairwise(epochs))
        ):
            block.refuse("epochs", "a list of rising whole numbers of at least 1")
        min_share = block.real("min_share", _SHARE)
        max_share = block.real("max_share", _SHARE)
        if min_share >= max_share:
            block.refuse("min_share", f"below max_share ({max_share})")
        return cls(
            epochs=tuple(epochs),
            min_share=min_share,
            max_share=max_share,
            margin_if_below=block.real("margin_if_below", _FINITE),
            margin_if_above=block.real("margin_if_above", _FINITE),
        )


@dataclass(frozen=True, slots=True)
class FallingRule:
    """The 4-epoch rule (``rules.falling``).

    Attributes:
        loss: What it reads: the CE part of the loss, without the gain term.
        min_drop_share: The second half of epoch 3 must lie below its first half by at
            least this share of the first half's mean ...
        min_standard_errors: ... and by more than this many standard errors.
        epochs_if_falling: The epochs of every arm if the rule fires.
    """

    loss: str
    min_drop_share: float
    min_standard_errors: float
    epochs_if_falling: int

    @classmethod
    def _from_block(cls, block: _Block) -> "FallingRule":
        """Read and check the block."""
        return cls(
            loss=block.choice("loss", ("ce_part",)),
            min_drop_share=block.real("min_drop_share", _SHARE),
            min_standard_errors=block.real("min_standard_errors", _NON_NEGATIVE),
            epochs_if_falling=block.whole("epochs_if_falling", 1),
        )


@dataclass(frozen=True, slots=True)
class RuleSettings:
    """The checks and allowed changes read on the training logs (``rules``).

    Attributes:
        f1_min_scale: Check F1 fails if the loss scale falls below this.
        alive: The gain-alive rule and check F2.
        falling: The 4-epoch rule.
        f5_max_hours: Check F5: the most GPU-hours of a three-view run's training.
    """

    f1_min_scale: float
    alive: AliveRule
    falling: FallingRule
    f5_max_hours: float

    @classmethod
    def _from_block(cls, block: _Block) -> "RuleSettings":
        """Read and check the block."""
        return cls(
            f1_min_scale=block.real("f1_min_scale", _POSITIVE),
            alive=AliveRule._from_block(block.block("alive", _keys(AliveRule))),
            falling=FallingRule._from_block(block.block("falling", _keys(FallingRule))),
            f5_max_hours=block.real("f5_max_hours", _POSITIVE),
        )


@dataclass(frozen=True, slots=True)
class ScoringSettings:
    """How a trained run scores its test fold (``scoring``).

    Attributes:
        views: The scored views.
        format: The format of LoRA and the adapter when they score: fp16.
    """

    views: tuple[str, ...]
    format: str

    @classmethod
    def _from_block(cls, block: _Block) -> "ScoringSettings":
        """Read and check the block."""
        return cls(
            views=block.names("views", CLIP_VERSIONS), format=block.choice("format", ("fp16",))
        )


@dataclass(frozen=True, slots=True)
class StandInStart:
    """μ and σ of the timing run, which reads no zero-shot score (``timing.stand_in_start``).

    Attributes:
        mu_logit: μ of both views.
        sigma_logit: σ of both views.
    """

    mu_logit: float
    sigma_logit: float

    @classmethod
    def _from_block(cls, block: _Block) -> "StandInStart":
        """Read and check the block."""
        return cls(
            mu_logit=block.real("mu_logit", _FINITE),
            sigma_logit=block.real("sigma_logit", _POSITIVE),
        )


@dataclass(frozen=True, slots=True)
class TimingSettings:
    """The timing run of check F5 (``timing``).

    Attributes:
        steps_per_mode: The sampler's first steps, timed in each repeatability mode.
        first_steps_left_out: The first timed steps of each mode, which warm up.
        stand_in_start: μ and σ of the timing run.
    """

    steps_per_mode: int
    first_steps_left_out: int
    stand_in_start: StandInStart

    @classmethod
    def _from_block(cls, block: _Block) -> "TimingSettings":
        """Read and check the block."""
        steps = block.whole("steps_per_mode", 1)
        left_out = block.whole("first_steps_left_out", 0)
        if left_out >= steps:
            block.refuse("first_steps_left_out", f"below steps_per_mode ({steps})")
        return cls(
            steps_per_mode=steps,
            first_steps_left_out=left_out,
            stand_in_start=StandInStart._from_block(
                block.block("stand_in_start", _keys(StandInStart))
            ),
        )


@dataclass(frozen=True, slots=True)
class TrainSettings:
    """The shared training settings of ``train.yaml``, one dataclass per block.

    Attributes:
        model: The pin of ``configs/pilot/models.yaml`` that trains: ``voxtral``.
        prompt: The prompt, which is the code's ``voxtral.PROMPT``.
        seeds: The seed of each round.
        epochs: Epochs per run.
        lora: LoRA and the trained adapter.
        optimiser: AdamW and its schedule.
        precision: Number formats and repeatability.
        sampler: The bag sampler.
        loss: The weights of CE(x1) and CE(x2).
        gain: The gain term and its running statistics.
        rules: The checks and allowed changes read on the training logs.
        scoring: How a trained run scores its test fold.
        timing: The timing run of check F5.
    """

    model: str
    prompt: str
    seeds: SeedSettings
    epochs: int
    lora: LoraSettings
    optimiser: OptimiserSettings
    precision: PrecisionSettings
    sampler: SamplerSettings
    loss: LossSettings
    gain: GainSettings
    rules: RuleSettings
    scoring: ScoringSettings
    timing: TimingSettings

    @classmethod
    def from_config(cls, config: Any) -> "TrainSettings":
        """Read the content of ``train.yaml`` and check every key and value.

        Args:
            config: The content of ``train.yaml``.

        Returns:
            The settings.

        Raises:
            StepError: If a key is missing or unknown, a block is not a mapping, a value
                has the wrong type, lies outside its range or is not one that the code
                implements, or the epochs of the gain-alive rule lie beyond ``epochs``.
                The message names ``train.yaml``, the key and the rule.
        """
        read = _Block(config, _keys(cls), SHARED_FILE)
        epochs = read.whole("epochs", 1)
        rules = RuleSettings._from_block(read.block("rules", _keys(RuleSettings)))
        if rules.alive.epochs[-1] > epochs:
            raise StepError(
                f"{SHARED_FILE}: rules.alive.epochs must lie within the {epochs} epochs, not"
                f" {list(rules.alive.epochs)}"
            )
        return cls(
            model=read.choice("model", (MODEL,)),
            prompt=read.choice("prompt", (PROMPT,)),
            seeds=SeedSettings._from_block(read.block("seeds", _keys(SeedSettings))),
            epochs=epochs,
            lora=LoraSettings._from_block(read.block("lora", _keys(LoraSettings))),
            optimiser=OptimiserSettings._from_block(
                read.block("optimiser", _keys(OptimiserSettings))
            ),
            precision=PrecisionSettings._from_block(
                read.block("precision", _keys(PrecisionSettings))
            ),
            sampler=SamplerSettings._from_block(read.block("sampler", _keys(SamplerSettings))),
            loss=LossSettings._from_block(read.block("loss", _keys(LossSettings))),
            gain=GainSettings._from_block(read.block("gain", _keys(GainSettings))),
            rules=rules,
            scoring=ScoringSettings._from_block(read.block("scoring", _keys(ScoringSettings))),
            timing=TimingSettings._from_block(read.block("timing", _keys(TimingSettings))),
        )


# Files that extend train.yaml ------------------------------------------------------------


class _UniqueKeyLoader(yaml.SafeLoader):
    """PyYAML's safe loader, except that a key given twice in one mapping is an error.

    YAML allows each key once per mapping, but PyYAML keeps the last of two silently, so a
    repeated setting would change a run without a word.
    """

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Any, Any]:
        """Build a mapping, refusing a key that it holds twice."""
        self.flatten_mapping(node)
        seen: list[Any] = []
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if key in seen:
                raise ConstructorError(
                    problem=f"the key {key!r} is given twice", problem_mark=key_node.start_mark
                )
            seen.append(key)
        return super().construct_mapping(node, deep=deep)


def _read(path: Path) -> tuple[dict[str, Any], str]:
    """A config file's content, a mapping of named keys, and the SHA-256 of its bytes."""
    if not path.is_file():
        raise StepError(f"{path.name} is missing ({path})")
    data = path.read_bytes()
    try:
        content = yaml.load(data.decode("utf-8"), Loader=_UniqueKeyLoader)
    except UnicodeDecodeError:
        raise StepError(f"{path.name} is not UTF-8 text") from None
    except yaml.YAMLError as error:
        problem = getattr(error, "problem", None) or "it cannot be parsed"
        mark = getattr(error, "problem_mark", None)
        line = f" (line {mark.line + 1})" if mark is not None else ""
        raise StepError(f"{path.name} is not valid YAML: {problem}{line}") from None
    if not isinstance(content, dict) or not all(isinstance(key, str) for key in content):
        raise StepError(f"{path.name} must hold a mapping of named settings")
    return content, hashlib.sha256(data).hexdigest()


def read_settings(path: Path) -> tuple[dict[str, Any], str]:
    """Read a settings file that extends no other, refusing a key given twice in a mapping.

    For files such as ``configs/pilot/evaluate.yaml``, whose reader checks every key and
    value itself: a key given twice, such as two pins of one fold-run, would otherwise keep
    the last of the two without a word, as PyYAML's own loader does.

    Args:
        path: The file.

    Returns:
        Its content, a mapping of named settings, and the SHA-256 of its bytes, as read.

    Raises:
        StepError: If the file is missing, is not UTF-8 text or valid YAML, holds a key twice
            in one mapping, or does not hold a mapping of named settings.
    """
    return _read(path)


def canonical_sha256(content: Mapping[str, Any]) -> str:
    """Return the SHA-256 of a resolved config as canonical JSON.

    Canonical JSON (engineering conventions, section 5): keys sorted at every level, no
    spaces, UTF-8, and no NaN or infinity. The same settings give the same SHA-256 in any
    order of keys, while 1 and 1.0, or true and 1, give different ones.

    Args:
        content: The resolved config: JSON types only.

    Returns:
        The SHA-256 as 64 hex characters.

    Raises:
        ValueError: If a value is NaN or infinite.
        TypeError: If a value is not of a JSON type.
    """
    text = json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """A config file read with the shared file that it extends.

    Attributes:
        file: The file.
        file_sha256: The SHA-256 of its bytes, as read.
        shared_file: The file that it extends: ``train.yaml`` in the same folder.
        shared_sha256: The SHA-256 of that file's bytes, as read.
        own: The file's content, ``extends`` included.
        shared: The shared file's content.
        resolved: The file's own keys without ``extends``, then the shared ones: the
            settings that define a run, as plain JSON types, for its ``run.json``.
        sha256: The SHA-256 of ``resolved`` as canonical JSON (``canonical_sha256``).
    """

    file: Path
    file_sha256: str
    shared_file: Path
    shared_sha256: str
    own: dict[str, Any]
    shared: dict[str, Any]
    resolved: dict[str, Any]
    sha256: str


def resolve_config(path: Path) -> ResolvedConfig:
    """Read a file that extends ``train.yaml``, with that file: one level of ``extends``.

    The file must say ``extends: train.yaml``, which names the file of that name in its own
    folder, and ``train.yaml`` must extend nothing (engineering conventions, section 5). The
    file adds only its own keys: a key of ``train.yaml`` is refused there, so that a shared
    change stays one edit and the arms stay paired. Only the keys and the YAML are checked
    here; ``load_arm`` and ``zeroshot.ZeroshotSettings`` check the values.

    Args:
        path: The file, such as ``configs/pilot/cpg.yaml``.

    Returns:
        Both files, their content and SHA-256s, the resolved config and its SHA-256.

    Raises:
        StepError: If a file is missing, not UTF-8, not valid YAML, gives a key twice in
            one mapping or holds no mapping of named keys; if the file extends no file or
            another file than ``train.yaml``; if ``train.yaml`` extends a file; if the file
            sets a key of ``train.yaml``; or if a value is not a JSON type (such as a date)
            or not finite.
    """
    own, own_sha256 = _read(path)
    if EXTENDS not in own:
        raise StepError(
            f"{path.name} extends no file: it must start from {SHARED_FILE} (engineering"
            " conventions, section 5)"
        )
    if own[EXTENDS] != SHARED_FILE:
        raise StepError(
            f"{path.name} must extend {SHARED_FILE!r}, the shared file in its folder, not"
            f" {own[EXTENDS]!r}"
        )
    shared_file = path.parent / SHARED_FILE
    shared, shared_sha256 = _read(shared_file)
    if EXTENDS in shared:
        raise StepError(
            f"{SHARED_FILE} extends {shared[EXTENDS]!r}: one level of extends only"
            " (engineering conventions, section 5)"
        )
    both = sorted(set(own) & set(shared))
    if both:
        raise StepError(
            f"{path.name} sets {both}, which {SHARED_FILE} sets too: a file that extends it"
            " adds only its own keys"
        )
    merged = {key: value for key, value in own.items() if key != EXTENDS} | shared
    try:
        sha256 = canonical_sha256(merged)
        resolved = json.loads(json.dumps(merged, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError):
        raise StepError(
            f"{path.name} with {SHARED_FILE} holds a value that JSON cannot hold, such as a"
            " date or a number that is not finite"
        ) from None
    return ResolvedConfig(
        file=path,
        file_sha256=own_sha256,
        shared_file=shared_file,
        shared_sha256=shared_sha256,
        own=own,
        shared=shared,
        resolved=resolved,
        sha256=sha256,
    )


# The arms ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArmConfig:
    """One arm of the pilot: its own keys, with the shared settings that it extends.

    Attributes:
        arm: ``b0``, ``b1`` or ``cpg``.
        views: The trained views, in the order in which a bag goes through the model
            (``ARM_VIEWS``).
        lambda_g: λG, the weight of the gain term: 0.0 for B1, which computes and logs
            the gain without training on it, above 0 for CPG; None for B0, which has no
            gain term.
        shared: The settings of ``train.yaml``.
        resolved: The arm file's keys without ``extends``, then those of ``train.yaml``,
            for ``run.json``.
        sha256: The SHA-256 of ``resolved`` as canonical JSON (``canonical_sha256``).
    """

    arm: str
    views: tuple[str, ...]
    lambda_g: float | None
    shared: TrainSettings
    resolved: dict[str, Any]
    sha256: str


def load_arm(path: Path) -> ArmConfig:
    """Read an arm file with the shared settings that it extends, every key and value checked.

    The arm file holds ``extends: train.yaml``, its arm (and is named after it, such as
    ``cpg.yaml``), its views, which must be its arm's (``ARM_VIEWS``), and, for B1 and
    CPG, ``lambda_g``: 0.0 for B1, a float above 0 for CPG. B0 has no gain term and no
    ``lambda_g``. ``train.yaml`` is read with ``TrainSettings.from_config``.

    Args:
        path: The arm file, such as ``configs/pilot/cpg.yaml``.

    Returns:
        The arm, its views and λG, the shared settings, the resolved config and its SHA-256.

    Raises:
        StepError: If ``resolve_config`` or ``TrainSettings.from_config`` refuses the files,
            if the arm is not one of ``ARMS`` or not the file's name, or if a key of the arm
            file is missing or unknown, its views are not its arm's, or ``lambda_g`` is not
            a float of its arm's range.
    """
    config = resolve_config(path)
    own = config.own
    arm = own.get("arm")
    if not isinstance(arm, str) or arm not in ARMS:
        raise StepError(f"{path.name}: arm must be one of {list(ARMS)}, not {arm!r}")
    if path.stem != arm:
        raise StepError(f"{path.name} holds arm {arm!r}: an arm file is named after its arm")
    keys = (EXTENDS, "arm", "views") if arm == "b0" else (EXTENDS, "arm", "views", "lambda_g")
    read = _Block(own, keys, path.name)
    views = ARM_VIEWS[arm]
    if read.values["views"] != list(views):
        read.refuse("views", f"{list(views)} for arm {arm}")
    lambda_g = None
    if arm == "b1":
        lambda_g = read.real("lambda_g", _ZERO)
    elif arm == "cpg":
        lambda_g = read.real("lambda_g", _POSITIVE)
    return ArmConfig(
        arm=arm,
        views=views,
        lambda_g=lambda_g,
        shared=TrainSettings.from_config(config.shared),
        resolved=config.resolved,
        sha256=config.sha256,
    )
