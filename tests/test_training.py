"""Training from the cached encoder outputs (training.py) on the tiny random Voxtral, on the CPU.

The tiny Voxtral of voxtral_fakes, saved as a snapshot and loaded by the training code's own
loaders with the stand-in for its processor. Synthetic encoder outputs only: random fp16
values of the encoder's shape, made per view and segment from a seed, for bags of fake
participants from 9001. One segment of the first bag has two chunks in x0 and x2, as the
pilot's long segment has. The forward passes run under fp16 autocast with LoRA and the
adapter in fp32, as on the GPU. The GPU tests at the end use the real model on OMNI, with
synthetic encoder outputs too.
"""

import contextlib
import copy
import dataclasses
import gc
import json
import math
import os
import re
import sys
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from listening_beyond_words import training, voxtral
from listening_beyond_words.bags import Bag, BagSampler, BagSettings
from listening_beyond_words.config import STRICT, WARN_ONLY, ArmConfig, load_arm
from listening_beyond_words.loss import STEP_MEASURES, RunningStats, gain, lambda_g
from listening_beyond_words.models import load_pins, local_snapshot
from listening_beyond_words.runinfo import StepError, environment, item_seed
from voxtral_fakes import (
    ANSWER_IDS,
    WIDTH,
    FakeProcessor,
    patch_processor,
    same_bytes,
    save_checkpoint,
    tiny_model,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "pilot"
LAYERS = 2  # of the tiny model's language model
LONG = "9001_003"  # the last segment of the first bag: two chunks in x0 and x2
BAGS = tuple(
    Bag(session, label, tuple(f"{session}_{index:03d}" for index in range(4)))
    for session, label in ((9001, 1), (9002, 0), (9003, 1), (9004, 0))
)
PAIR = BAGS[:2]  # one positive and one negative bag, the long clip among them: quicker steps
TOTAL = 20  # T of most tests: 1 warm-up step, a λG ramp of 4
CALLS = {"b0": 5, "b1": 14, "cpg": 14}  # one call per bag and view, the long clip's own one more
# A share of steps after which λG is 1 for T = 20.
RAMPED = 10
# The keys of a step's line of metrics.jsonl: every arm's, B0's, and those of B1 and CPG.
BASE_KEYS = frozenset(
    {
        "step",
        "epoch",
        "seconds",
        "lr",
        "adapter_lr",
        "loss",
        "ce_x0",
        "ce_part",
        "scale",
        "next_scale",
        "skipped",
        "grad_norm",
        "clipped",
        "calls",
    }
)
B0_KEYS = BASE_KEYS | set(training.X0_MEASURES)
GAIN_KEYS = BASE_KEYS | {"ce_x1", "ce_x2", "hinge", "lambda_g", *STEP_MEASURES}
# The readings of the first CPG run of round 1, fold 1: check F2, the two rules and F5.
FIRST_CPG = {"reads_f2": True, "reads_rules": True}


# Synthetic inputs ------------------------------------------------------------------------------


class Reader:
    """Synthetic encoder outputs, made on first read from a seed and kept.

    A clip of a segment in ``long`` has two chunks in x0 and x2, every other clip one. The
    reader remembers the view and segment it read last.
    """

    def __init__(self, long: frozenset[str] = frozenset({LONG}), size: float = 0.5) -> None:
        self.long = long
        self.size = size
        self.outputs: dict[tuple[str, str], torch.Tensor] = {}
        self.view = ""
        self.segment = ""

    def __call__(self, view: str, segment: str) -> torch.Tensor:
        self.view, self.segment = view, segment
        if (view, segment) not in self.outputs:
            chunks = 2 if segment in self.long and view != "x1" else 1
            generator = torch.Generator().manual_seed(item_seed(5, view, segment))
            values = torch.randn(chunks, voxtral.FRAMES_PER_CHUNK, WIDTH, generator=generator)
            self.outputs[(view, segment)] = (values * self.size).half()
        return self.outputs[(view, segment)]


@pytest.fixture(scope="module")
def arms() -> dict[str, ArmConfig]:
    return {name: load_arm(CONFIGS / f"{name}.yaml") for name in ("b0", "b1", "cpg")}


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny Voxtral saved as the pinned snapshot stores Voxtral: in bf16."""
    return save_checkpoint(tiny_model(), tmp_path_factory.mktemp("snapshot") / "voxtral")


@pytest.fixture
def snapshot(saved: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The tiny snapshot, which the training code loads with the processor's stand-in."""
    patch_processor(monkeypatch)
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    return saved


@pytest.fixture(scope="module")
def prompts() -> dict[int, dict[str, torch.Tensor]]:
    processor = FakeProcessor()
    return {count: voxtral.prompt_inputs(processor, count) for count in (1, 2)}


def stats_of(arm: ArmConfig) -> training.ViewStats | None:
    """Running statistics away from their floor, for the arms with a gain."""
    if arm.lambda_g is None:
        return None
    gain_settings = arm.shared.gain
    return training.ViewStats(
        RunningStats.of(0.1, 0.8, gain_settings), RunningStats.of(-0.2, 1.1, gain_settings)
    )


def tuned(
    arm: ArmConfig,
    *,
    init_scale: float | None = None,
    grad_clip_norm: float | None = None,
    dropout: float | None = None,
) -> ArmConfig:
    """The arm with one of its shared settings changed, for a test."""
    shared = arm.shared
    if init_scale is not None:
        scaler = dataclasses.replace(shared.precision.scaler, init_scale=init_scale)
        precision = dataclasses.replace(shared.precision, scaler=scaler)
        shared = dataclasses.replace(shared, precision=precision)
    if grad_clip_norm is not None:
        optimiser = dataclasses.replace(shared.optimiser, grad_clip_norm=grad_clip_norm)
        shared = dataclasses.replace(shared, optimiser=optimiser)
    if dropout is not None:
        shared = dataclasses.replace(shared, lora=dataclasses.replace(shared.lora, dropout=dropout))
    return dataclasses.replace(arm, shared=shared)


@pytest.fixture(autouse=True)
def own_random_state() -> Iterator[None]:
    """Put PyTorch's random state back after each test: build_trainable seeds it, as in a run."""
    with torch.random.fork_rng(devices=[]):
        yield


def build(snapshot: Path, arm: ArmConfig, seed: int = 11) -> training.Trainable:
    """The model that trains, on the CPU; LoRA's dropout then draws from the seeded stream."""
    return training.build_trainable(snapshot, arm.shared.lora, seed, device="cpu")


def nudge(model: Any, seed: int = 3, size: float = 0.05) -> None:
    """Set every LoRA B away from 0, as after some training, from a seed."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if ".lora_B." in name:
                value.copy_(torch.randn(value.shape, generator=generator) * size)


def trained_values(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if value.requires_grad
    }


def same_values(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> bool:
    return first.keys() == second.keys() and all(same_bytes(first[k], second[k]) for k in first)


@dataclass
class Run:
    """A run's model, optimiser, watch and statistics, stepped one step at a time."""

    arm: ArmConfig
    trainable: training.Trainable
    optimiser: training.Optimiser
    watch: voxtral.LayerWatch
    stats: training.ViewStats | None
    reader: Reader
    prompts: dict[int, dict[str, torch.Tensor]]

    def advance(self, steps: int) -> None:
        """Move the schedule on by some steps without training, as if they had run."""
        for _ in range(steps):
            self.optimiser.optimizer.step()  # no gradient yet: the weights stay
            self.optimiser.schedule.step()

    def step(self, t: int, bags: tuple[Bag, ...] = BAGS, epoch: int = 1) -> training.StepRecord:
        record = training.train_step(
            self.trainable,
            self.optimiser,
            self.watch,
            self.arm,
            bags,
            self.reader,
            self.prompts,
            self.stats,
            t,
            epoch,
        )
        self.stats = record.stats
        return record


def start(
    snapshot: Path,
    arm: ArmConfig,
    prompts: dict[int, dict[str, torch.Tensor]],
    total: int = TOTAL,
    seed: int = 11,
    nudged: bool = False,
    before_watch: Callable[[training.Trainable], None] | None = None,
) -> Run:
    """A run ready for its first step: the model, its optimiser and its watch."""
    trainable = build(snapshot, arm, seed)
    if nudged:
        nudge(trainable.model)
    if before_watch is not None:
        before_watch(trainable)  # a planted hook must run before the watch reads the output
    optimiser = training.make_optimiser(trainable, arm.shared, total)
    watch = voxtral.watch_layers(voxtral.language_model_layers(trainable.model, logits=True))
    return Run(arm, trainable, optimiser, watch, stats_of(arm), Reader(), prompts)


def record_scores(
    monkeypatch: pytest.MonkeyPatch, reader: Reader
) -> list[tuple[str, int, torch.Tensor]]:
    """Keep each bag and view's z, with its gradient once the backward pass has run."""
    seen: list[tuple[str, int, torch.Tensor]] = []
    real = training._bag_z

    def bag_z(*args: Any) -> tuple[torch.Tensor, int]:
        z, calls = real(*args)
        z.retain_grad()
        seen.append((reader.view, int(reader.segment.split("_")[0]), z))
        return z, calls

    monkeypatch.setattr(training, "_bag_z", bag_z)
    return seen


def one_clip_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send every clip through the model in a call of its own."""
    real = training._bag_z

    def single(model: Any, prompts: Any, outputs: Any, ids: Any, device: str) -> Any:
        scores = [real(model, prompts, [output], ids, device)[0][0] for output in outputs]
        return torch.stack(scores), len(outputs)

    monkeypatch.setattr(training, "_bag_z", single)


def show(capsys: pytest.CaptureFixture[str], line: str) -> None:
    """Write a line to the test log, past pytest's capture."""
    with capsys.disabled():
        sys.stdout.write(line + "\n")


@contextlib.contextmanager
def kept_numerics() -> Iterator[None]:
    """Put PyTorch's numeric settings back afterwards, since set_numerics sets them."""
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
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
        torch.backends.cudnn.deterministic = saved[2]
        torch.backends.cudnn.benchmark = saved[3]
        matmul.allow_fp16_reduced_precision_reduction = saved[4]
        matmul.allow_tf32 = saved[5]
        torch.backends.cudnn.allow_tf32 = saved[6]
        torch.set_num_threads(saved[7])
        matmul.allow_fp16_accumulation = saved[8]


def cpu_threads() -> int:
    data = yaml.safe_load((CONFIGS / "data.yaml").read_text(encoding="utf-8"))
    return int(data["encoder_cache"]["cpu_threads"])


# The model that trains -------------------------------------------------------------------------


def test_lora_sits_on_the_7_projections_of_every_language_model_layer_and_trains_in_fp32(
    snapshot: Path, arms: dict[str, ArmConfig]
) -> None:
    from peft.tuners.lora import LoraLayer

    trainable = build(snapshot, arms["cpg"])
    model = trainable.model
    base = model.get_base_model()
    wrapped = sorted(name for name, module in base.named_modules() if isinstance(module, LoraLayer))
    expected = sorted(
        f"model.language_model.layers.{index}.{projection}"
        for index in range(LAYERS)
        for projection in training.PROJECTIONS
    )
    assert wrapped == expected  # the language model's layers only: no encoder module
    trained = dict(training._trained(model))
    adapter = sorted(name for name in trained if ".modules_to_save." in name)
    assert len(adapter) == 2  # the copy's two layers; the original stays frozen
    assert len(trained) == 2 * len(wrapped) + len(adapter)  # LoRA's A and B, and the copy
    assert {value.dtype for value in trained.values()} == {torch.float32}
    frozen = {value.dtype for value in model.parameters() if not value.requires_grad}
    assert frozen == {torch.float16}
    assert not any(".audio_tower." in name for name, _ in model.named_parameters())  # dropped
    assert trainable.record["lora_modules"] == 7 * LAYERS == 14
    assert trainable.record["trained_numbers"] == sum(v.numel() for v in trained.values()) == 53_248
    assert trainable.record["checkpointing"] == {"use_reentrant": False, "every_n_layers": 1}
    assert all(layer.gradient_checkpointing for layer in base.model.language_model.layers)
    assert model.training
    assert trainable.answer_ids == ANSWER_IDS


def test_the_lora_start_values_follow_the_seed_alone_and_the_second_matrix_starts_at_0(
    snapshot: Path, arms: dict[str, ArmConfig]
) -> None:
    first, again = build(snapshot, arms["cpg"], seed=11), build(snapshot, arms["b0"], seed=11)
    other = build(snapshot, arms["cpg"], seed=12)
    assert first.record["start_sha256"] == again.record["start_sha256"]  # the same for every arm
    assert first.record["start_sha256"] != other.record["start_sha256"]
    assert same_values(trained_values(first.model), trained_values(again.model))
    for name, value in first.model.named_parameters():
        if ".lora_B." in name:
            assert not value.detach().any(), name  # training starts from the untrained model


def test_the_torch_seed_is_the_same_for_every_arm_and_differs_by_round_and_fold() -> None:
    assert training.torch_seed(101, 1) == item_seed(101, "torch", 1)
    seeds = {training.torch_seed(seed, fold) for seed in (101, 102) for fold in range(1, 6)}
    assert len(seeds) == 10


def test_the_record_holds_no_path_of_the_snapshot_folder(
    snapshot: Path, arms: dict[str, ArmConfig]
) -> None:
    trainable = build(snapshot, arms["cpg"])
    # PEFT keeps the snapshot's folder; the record must not.
    assert trainable.model.peft_config["default"].base_model_name_or_path == str(snapshot)
    text = json.dumps(trainable.record)
    assert str(snapshot.parent) not in text
    peft = trainable.record["peft_config"]
    assert "base_model_name_or_path" not in peft
    lora = arms["cpg"].shared.lora
    assert (peft["r"], peft["lora_alpha"], peft["lora_dropout"]) == (16, 32, 0.05)
    assert (peft["target_modules"], peft["modules_to_save"]) == (
        lora.targets,
        [lora.trained_adapter],
    )
    assert (peft["bias"], peft["init_lora_weights"], peft["use_rslora"]) == ("none", True, False)
    assert peft["use_dora"] is False


def test_the_packages_a_training_run_records_include_peft_at_its_pin() -> None:
    packages = environment(training.PACKAGES)["packages"]
    assert packages["peft"] == "0.21.2"
    assert packages["torch"]
    assert packages["transformers"] == "5.22.2"


# The optimiser and its schedule ----------------------------------------------------------------


def test_make_optimiser_gives_lora_and_the_adapter_their_rates_and_train_yamls_settings(
    snapshot: Path, arms: dict[str, ArmConfig]
) -> None:
    trainable = build(snapshot, arms["cpg"])
    optimiser = training.make_optimiser(trainable, arms["cpg"].shared, 252)
    groups = optimiser.optimizer.param_groups
    names = {id(value): name for name, value in trainable.model.named_parameters()}
    assert all(".lora_" in names[id(value)] for value in groups[0]["params"])
    assert all(".modules_to_save." in names[id(value)] for value in groups[1]["params"])
    assert len(groups[0]["params"]) + len(groups[1]["params"]) == 30
    assert [group["initial_lr"] for group in groups] == [1.0e-4, 2.0e-5]
    for group in groups:
        assert group["betas"] == (0.9, 0.999)
        assert (group["eps"], group["weight_decay"]) == (1.0e-8, 0.01)
        assert (group["amsgrad"], group["foreach"]) == (False, True)
    assert (optimiser.record["foreach"], optimiser.record["fused"]) == (True, None)
    assert (optimiser.total_steps, optimiser.warmup_steps) == (252, 13)
    scaler = optimiser.scaler
    assert scaler.get_scale() == 65_536.0
    assert (scaler.get_growth_factor(), scaler.get_backoff_factor()) == (2.0, 0.5)
    assert scaler.get_growth_interval() == 2000


def cosine(step: int, total: int, warmup: int) -> float:
    """The schedule's factor: a straight line from 0, then half a cosine to 0."""
    if step < warmup:
        return step / warmup
    return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / (total - warmup)))


@pytest.mark.parametrize(("total", "warmup"), [(252, 13), (100, 5), (7, 1)])
def test_each_groups_rate_follows_the_warm_up_and_the_half_cosine(
    snapshot: Path, arms: dict[str, ArmConfig], total: int, warmup: int
) -> None:
    trainable = build(snapshot, arms["b0"])
    optimiser = training.make_optimiser(trainable, arms["b0"].shared, total)
    assert optimiser.warmup_steps == warmup
    rates = []
    for _ in range(total):
        rates.append([group["lr"] for group in optimiser.optimizer.param_groups])
        optimiser.optimizer.step()  # no gradient: the weights stay
        optimiser.schedule.step()
    for step, (lora, adapter) in enumerate(rates):
        factor = cosine(step, total, warmup)
        assert lora == pytest.approx(1.0e-4 * factor, rel=1e-12, abs=1e-20)
        assert adapter == pytest.approx(2.0e-5 * factor, rel=1e-12, abs=1e-20)
    if total == 252:
        assert rates[0] == [0.0, 0.0]
        assert rates[12][0] == pytest.approx(9.23e-5, rel=1e-3)
        assert rates[13] == [1.0e-4, 2.0e-5]
        assert rates[251][0] == pytest.approx(4.32e-9, rel=1e-3)


def test_zero_gradients_at_a_rate_above_0_shrink_every_trained_tensor_by_the_weight_decay(
    snapshot: Path, arms: dict[str, ArmConfig]
) -> None:
    trainable = build(snapshot, arms["cpg"])
    nudge(trainable.model)
    optimiser = training.make_optimiser(trainable, arms["cpg"].shared, TOTAL)
    for _ in range(3):
        optimiser.optimizer.step()
        optimiser.schedule.step()
    rates = [group["lr"] for group in optimiser.optimizer.param_groups]
    assert min(rates) > 0
    before = trained_values(trainable.model)
    names = {id(value): name for name, value in trainable.model.named_parameters()}
    for group in optimiser.optimizer.param_groups:
        for value in group["params"]:
            value.grad = torch.zeros_like(value)
    optimiser.optimizer.step()
    for rate, group in zip(rates, optimiser.optimizer.param_groups, strict=True):
        for value in group["params"]:
            name = names[id(value)]
            expected = before[name] * (1.0 - rate * 0.01)
            torch.testing.assert_close(value.detach(), expected, rtol=2e-7, atol=0.0)
            if ".lora_B." in name or ".modules_to_save." in name:
                assert not torch.equal(value.detach(), before[name]), name


# One step --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["b0", "b1", "cpg"])
def test_a_step_of_each_arm_runs_and_gives_every_lora_b_and_the_adapter_a_gradient(
    snapshot: Path, arms: dict[str, ArmConfig], prompts: dict, name: str
) -> None:
    run = start(snapshot, arms[name], prompts)
    record = run.step(0)
    for parameter_name, value in run.trainable.model.named_parameters():
        if ".lora_B." in parameter_name or ".modules_to_save." in parameter_name:
            assert value.grad is not None, parameter_name
            assert value.grad.any(), parameter_name
    assert record.calls == CALLS[name]
    assert (record.lr, record.adapter_lr, record.skipped) == (0.0, 0.0, False)  # step 0
    line = record.line()
    json.dumps(line, allow_nan=False)  # finite JSON numbers only
    assert not any(isinstance(value, str) for value in line.values())  # no ID, no text
    # Exactly the logged keys: no number per bag or clip can join a line unnoticed.
    assert set(line) == (B0_KEYS if name == "b0" else GAIN_KEYS)
    if name == "b0":
        assert record.stats is None
        assert line["loss"] == line["ce_part"] == line["ce_x0"]
    else:
        assert line["lambda_g"] == 0.0  # step 0 of the ramp, and B1 always
        assert line["loss"] == pytest.approx(line["ce_part"], rel=1e-12)
        assert record.stats is not None
        assert record.stats != stats_of(arms[name])
        assert sum(int(line[f"bags_{group}"]) for group in training.GROUPS) == 4
        # |z0|/|z1| of the step: the mean |z̄0| over the mean |z̄1| of its bags.
        assert line["z0_over_z1"] == line["z0_bar_abs_mean"] / line["z1_bar_abs_mean"]


@pytest.mark.parametrize(("step", "weight"), [(RAMPED, 1.0), (2, 0.5)])  # ramped, half-way
def test_x1_gets_its_ce_gradient_alone_and_none_through_the_gain(
    snapshot: Path,
    arms: dict[str, ArmConfig],
    prompts: dict,
    monkeypatch: pytest.MonkeyPatch,
    step: int,
    weight: float,
) -> None:
    arm = tuned(arms["cpg"], init_scale=256.0)  # no overflow in the backward pass (see below)
    run = start(snapshot, arm, prompts, nudged=True)
    stats, settings = run.stats, arm.shared.gain
    assert stats is not None
    seen = record_scores(monkeypatch, run.reader)
    run.advance(step)
    record = run.step(step)
    assert record.lambda_g == weight  # λG ramps over 4 of the 20 steps
    assert not record.skipped
    labels = {bag.session: bag.label for bag in BAGS}
    per_bag = record.scale / len(BAGS)
    scores = {(view, session): z for view, session, z in seen}
    for bag in BAGS:
        y = bag.label
        z1, z0 = scores[("x1", bag.session)], scores[("x0", bag.session)]
        # x1: its own CE's gradient and nothing else, though the gain read its bag mean.
        ce1 = (torch.sigmoid(z1.detach()) - y) / 4  # dCE/dz of the bag's mean CE
        torch.testing.assert_close(z1.grad, per_bag * 0.25 * ce1, rtol=1e-5, atol=1e-6)
        # x0: its CE's gradient plus a quarter of λG times the gain's push on the bag mean.
        g = gain(
            z0.detach().mean().reshape(1), z1.detach().mean().reshape(1), [y], stats.x0, stats.x1
        )
        push = (
            -torch.sigmoid((settings.margin - g) / settings.softness) * (2 * y - 1) / stats.x0.sigma
        )
        ce0 = (torch.sigmoid(z0.detach()) - y) / 4
        expected = per_bag * (ce0 + weight * push / 4)
        torch.testing.assert_close(z0.grad, expected, rtol=1e-5, atol=1e-6)
        assert float(push.abs()) > 0
    assert set(labels.values()) == {0, 1}
    # The logged loss holds the gain at the same weight: CE part plus λG times the hinge.
    assert record.hinge is not None
    assert record.loss == pytest.approx(record.ce_part + weight * record.hinge, rel=1e-6)


def test_a_bag_with_the_two_chunk_clip_gives_the_gradients_of_one_clip_calls(
    snapshot: Path, arms: dict[str, ArmConfig], prompts: dict
) -> None:
    # A low start scale: on the tiny model with LoRA B set, the scaled fp16 backward of the
    # grouped calls overflows at 2^16, where the scaler would skip the step. No clip, so the
    # gradients compare as they are, and no dropout, whose masks differ between the calls.
    arm = tuned(arms["cpg"], init_scale=256.0, grad_clip_norm=1.0e30, dropout=0.0)
    grouped = start(snapshot, arm, prompts, nudged=True)
    grouped.advance(RAMPED)
    with pytest.MonkeyPatch.context() as patch:
        seen = record_scores(patch, grouped.reader)
        first = grouped.step(RAMPED)
    single = start(snapshot, arm, prompts, nudged=True)
    single.advance(RAMPED)
    with pytest.MonkeyPatch.context() as patch:
        one_clip_per_call(patch)
        second = single.step(RAMPED)
    assert (first.calls, second.calls) == (14, 48)
    assert not first.skipped
    assert not second.skipped
    assert first.loss == pytest.approx(second.loss, rel=1e-6)
    # The fp16 backward sums four clips' gradients in one product instead of four, so the
    # last bits differ: on the tiny model at most 8.2e-4 of each tensor's largest value.
    for name, value in training._trained(grouped.trainable.model):
        other = dict(training._trained(single.trainable.model))[name].grad
        assert other is not None
        assert value.grad is not None
        largest = float(other.abs().max())
        assert float((value.grad - other).abs().max()) <= 4e-3 * largest, name
    # The long clip's z gets the gain's share: a quarter of its push on the bag mean.
    _, _, z = next(item for item in seen if item[:2] == ("x0", 9001))
    assert z.numel() == 4
    _, _, z1 = next(item for item in seen if item[:2] == ("x1", 9001))
    settings, stats = arm.shared.gain, stats_of(arm)
    assert stats is not None
    g = gain(z.detach().mean().reshape(1), z1.detach().mean().reshape(1), [1], stats.x0, stats.x1)
    push = -torch.sigmoid((settings.margin - g) / settings.softness) / stats.x0.sigma  # y = 1
    expected = (torch.sigmoid(z.detach()[3]) - 1.0) / 4 + float(push[0]) / 4
    assert float(z.grad[3]) == pytest.approx(first.scale / 4 * float(expected), rel=1e-5)


@pytest.mark.parametrize(("name", "view", "weight"), [("b0", "x0", 1.0), ("cpg", "x2", 0.5)])
def test_the_long_clips_ce_counts_a_quarter_of_its_bags_ce(
    snapshot: Path,
    arms: dict[str, ArmConfig],
    prompts: dict,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    view: str,
    weight: float,
) -> None:
    run = start(snapshot, arms[name], prompts)
    seen = record_scores(monkeypatch, run.reader)
    run.advance(1)
    record = run.step(1)
    _, _, z = next(item for item in seen if item[:2] == (view, 9001))
    expected = record.scale / 4 * weight * (torch.sigmoid(z.detach()) - 1.0) / 4  # y = 1
    torch.testing.assert_close(z.grad, expected, rtol=1e-5, atol=1e-6)


def test_b1s_update_equals_an_update_without_the_gain_code_bit_for_bit(
    snapshot: Path, arms: dict[str, ArmConfig], prompts: dict
) -> None:
    b1 = tuned(arms["b1"], dropout=0.0)  # the two runs draw from one random stream here
    with_gain = start(snapshot, b1, prompts)
    plain = start(snapshot, dataclasses.replace(b1, lambda_g=None), prompts)
    assert plain.stats is None  # no gain: no statistics, no hinge
    for step in range(2):
        first, second = with_gain.step(step, bags=PAIR), plain.step(step, bags=PAIR)
        assert first.loss == second.loss
        assert first.hinge is not None
        assert second.hinge is None
    assert same_values(
        trained_values(with_gain.trainable.model), trained_values(plain.trainable.model)
    )
    assert not same_values(
        trained_values(with_gain.trainable.model), trained_values(build(snapshot, b1).model)
    )  # it trained


def test_two_runs_of_one_seed_give_the_same_weights_bit_for_bit_dropout_and_all(
    snapshot: Path, arms: dict[str, ArmConfig], prompts: dict
) -> None:
    weights = []
    for seed in (11, 11, 12):  # one after the other, as jobs would run them
        run = start(snapshot, arms["b0"], prompts, seed=seed)
        for step in range(2):
            run.step(step)
        weights.append(trained_values(run.trainable.model))
    assert same_values(weights[0], weights[1])
    assert not same_values(weights[0], weights[2])


def test_a_step_starts_from_cleared_gradients_whatever_the_model_holds_before_it(
    snapshot: Path, arms: dict[str, ArmConfig], prompts: dict
) -> None:
    b0 = tuned(arms["b0"], dropout=0.0)  # the two runs draw from one random stream here
    runs = [start(snapshot, b0, prompts) for _ in range(2)]
    records = []
    for number, run in enumerate(runs):
        run.step(0, bags=PAIR)  # step 0 leaves its gradients on the trained tensors
        if number == 1:
            for _, value in training._trained(run.trainable.model):
                value.grad = torch.ones_like(value)  # planted: stale gradients of any size
        records.append(run.step(1, bags=PAIR))
    first, second = records
    assert first.grad_norm is not None
    assert (first.loss, first.grad_norm, first.skipped) == (
        second.loss,
        second.grad_norm,
        second.skipped,
    )
    assert same_values(
        trained_values(runs[0].trainable.model), trained_values(runs[1].trainable.model)
    )


def test_a_planted_overflow_skips_the_update_but_moves_schedule_lambda_and_statistics_on(
    snapshot: Path, arms: dict[str, ArmConfig], prompts: dict
) -> None:
    arm = arms["cpg"]
    run = start(snapshot, arm, prompts)
    for step in range(2):  # step 0 starts AdamW's averages, at a rate of 0
        run.step(step, bags=PAIR)
    weights = trained_values(run.trainable.model)
    state = {
        id(value): {key: item.clone() for key, item in moments.items()}
        for value, moments in run.optimiser.optimizer.state.items()
    }
    stats = run.stats
    assert stats is not None
    run.optimiser.scaler.update(new_scale=torch.tensor(2.0**100))  # the fp16 backward overflows
    record = run.step(2, bags=PAIR)
    assert record.skipped
    assert record.grad_norm is None
    assert not record.clipped
    assert (record.scale, record.next_scale) == (2.0**100, 2.0**99)
    assert same_values(trained_values(run.trainable.model), weights)
    for value, moments in run.optimiser.optimizer.state.items():
        assert all(same_bytes(item, state[id(value)][key]) for key, item in moments.items())
    # λG and the schedule move on ...
    assert record.lambda_g == lambda_g(2, TOTAL, 0.2) == 0.5
    rates = [group["lr"] for group in run.optimiser.optimizer.param_groups]
    assert rates[0] == pytest.approx(1.0e-4 * cosine(3, TOTAL, 1), rel=1e-12)
    assert run.optimiser.schedule.last_epoch == 3
    # ... and both views' statistics update from the step's bag means.
    measures = record.measures
    after = record.stats
    assert after is not None
    assert after.x0.first == pytest.approx(
        0.9 * stats.x0.first + 0.1 * float(measures["z0_bar_mean"])
    )
    assert after.x1.second == pytest.approx(
        0.9 * stats.x1.second + 0.1 * float(measures["z1_bar_mean_square"])
    )
    following = run.step(3, bags=PAIR)
    assert following.lambda_g == 0.75
    assert following.lr == pytest.approx(1.0e-4 * cosine(3, TOTAL, 1), rel=1e-12)


def test_a_skipped_step_changes_no_later_bag(
    snapshot: Path, arms: dict[str, ArmConfig], prompts: dict
) -> None:
    people = range(9001, 9009)
    segments = pd.DataFrame(
        [(f"{person}_{index:03d}", person) for person in people for index in range(8)],
        columns=["segment", "session"],
    )
    labels = {person: int(person % 2) for person in people}
    settings = BagSettings(segments_per_bag=4, max_segments_per_epoch=16, repeat_segments=False)
    sampler_settings = arms["b0"].shared.sampler

    def sampler() -> BagSampler:
        return BagSampler(segments, labels, settings, sampler_settings, 4)

    used = sampler()
    plan = used.epoch(1)
    run = start(snapshot, arms["b0"], prompts)
    value = next(
        value for name, value in training._trained(run.trainable.model) if ".lora_B." in name
    )
    value.register_hook(lambda grad: torch.full_like(grad, math.inf))  # an overflow
    assert run.step(0, bags=plan.steps[0]).skipped
    assert used.epoch(2) == sampler().epoch(2)  # the sampler takes nothing from training


@pytest.mark.parametrize("plant", ["z", "loss"])
def test_a_planted_nan_score_or_loss_stops_the_step_before_any_update(
    snapshot: Path,
    arms: dict[str, ArmConfig],
    prompts: dict,
    monkeypatch: pytest.MonkeyPatch,
    plant: str,
) -> None:
    run = start(snapshot, arms["cpg"], prompts)
    run.step(0, bags=PAIR)
    weights = trained_values(run.trainable.model)
    calls = [0]
    if plant == "z":
        real = training._bag_z

        def planted(*args: Any) -> Any:
            z, made = real(*args)
            calls[0] += 1
            return (
                torch.where(torch.arange(z.numel()) == 0, math.nan, z) if calls[0] == 2 else z
            ), made

        monkeypatch.setattr(training, "_bag_z", planted)
        message = "step 1, bag 1 of 2, x0: 1 of 4 scores are not finite; check F1 fails"
    else:
        real_ce = training.ce_on_z

        def nan_ce(z: torch.Tensor, y: Any) -> torch.Tensor:
            calls[0] += 1
            value = real_ce(z, y)
            return value * math.nan if calls[0] == 5 else value

        monkeypatch.setattr(training, "ce_on_z", nan_ce)
        message = "step 1, bag 2 of 2, x0: the loss is not finite; check F1 fails"
    with pytest.raises(StepError, match=re.escape(message)):
        run.step(1, bags=PAIR)
    assert same_values(trained_values(run.trainable.model), weights)
    assert run.optimiser.schedule.last_epoch == 1  # the step never finished


class Plant:
    """Writes one value into the language model's last layer or its attention, on chosen calls.

    The value lands at the last position of the call's first row, the one position whose
    output reaches the logits. ``when`` gets whether the watch is paused, that is whether
    the call is the backward pass's recompute.
    """

    def __init__(self, when: Callable[[bool], bool], value: float, part: str = "") -> None:
        self.when = when
        self.value = value
        self.part = part  # "" for the layer's output, "self_attn" for its attention part
        self.fired = 0
        self.watch: voxtral.LayerWatch | None = None

    def attach(self, trainable: training.Trainable) -> None:
        layer = trainable.model.get_base_model().model.language_model.layers[1]
        (getattr(layer, self.part) if self.part else layer).register_forward_hook(self.hook)

    def hook(self, _module: Any, _args: Any, output: Any) -> Any:
        if self.watch is None or not self.when(self.watch.paused):
            return None
        self.fired += 1
        first = (output[0] if isinstance(output, tuple) else output).clone()
        first[0, -1, 5] = self.value
        return (first, *output[1:]) if isinstance(output, tuple) else first


@pytest.mark.parametrize(
    ("value", "message", "counts"),
    [
        (
            64_512.0,
            "step 1: the overflow watch flagged 1 (model.language_model.layers.1/output)",
            (1, 0),
        ),
        (math.nan, "step 1, bag 1 of 4, x0: 1 of 4 scores are not finite", (0, 1)),
    ],
)
def test_a_value_planted_in_a_layers_output_on_a_forward_call_stops_that_step(
    snapshot: Path,
    arms: dict[str, ArmConfig],
    prompts: dict,
    value: float,
    message: str,
    counts: tuple[int, int],
) -> None:
    forward = [0]

    def kth(paused: bool) -> bool:
        if paused:
            return False  # the recompute of the backward pass
        forward[0] += 1
        return forward[0] == CALLS["b0"] + 1  # step 1's first forward call: bag 1's x0

    plant = Plant(kth, value)
    run = start(snapshot, arms["b0"], prompts, before_watch=plant.attach)
    plant.watch = run.watch
    run.step(0)
    weights = trained_values(run.trainable.model)
    with pytest.raises(StepError, match=re.escape(message)):
        run.step(1)
    assert plant.fired == 1
    numbers = run.watch.record()["model.language_model.layers.1"]["output"]
    assert (numbers["at_clamp"], numbers["non_finite"]) == counts
    assert same_values(trained_values(run.trainable.model), weights)  # no update


def test_a_value_planted_only_in_the_recompute_of_the_backward_pass_flags_nothing(
    snapshot: Path, arms: dict[str, ArmConfig], prompts: dict
) -> None:
    # Of the watched parts, only the attention runs again in the recompute: checkpointing
    # stops recomputing a layer once it holds every value that the backward pass needs.
    plant = Plant(lambda paused: paused, 64_512.0, part="self_attn")
    run = start(snapshot, arms["cpg"], prompts, before_watch=plant.attach)
    plant.watch = run.watch
    run.step(0)
    assert plant.fired == CALLS["cpg"]  # the plant took, in every recompute ...
    assert run.watch.flagged() == []  # ... which the paused watch never reads


def test_with_checkpointing_every_watched_parts_calls_equal_the_forward_calls(
    snapshot: Path, arms: dict[str, ArmConfig], prompts: dict
) -> None:
    every: list[int] = []

    def count_every_call(trainable: training.Trainable) -> None:
        layer = trainable.model.get_base_model().model.language_model.layers[0]
        layer.self_attn.register_forward_hook(lambda *_: every.append(1))

    run = start(snapshot, arms["cpg"], prompts, before_watch=count_every_call)
    calls = run.step(0).calls
    assert calls == CALLS["cpg"]
    assert len(every) == 2 * calls  # each attention part ran again in the backward pass ...
    found = {
        numbers["calls"] for parts in run.watch.record().values() for numbers in parts.values()
    }
    assert found == {calls}  # ... and the watch counted the forward calls alone


def test_a_planted_large_gradient_is_clipped_to_a_joint_norm_of_1_after_unscaling(
    snapshot: Path, arms: dict[str, ArmConfig], prompts: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = training.ce_on_z
    monkeypatch.setattr(training, "ce_on_z", lambda z, y: real(z, y) * 50.0)
    run = start(snapshot, tuned(arms["b0"], init_scale=16.0), prompts)
    run.advance(1)
    record = run.step(1)
    assert not record.skipped
    assert record.grad_norm is not None
    assert record.grad_norm > 50.0
    assert record.clipped
    gradients = [value.grad for _, value in training._trained(run.trainable.model)]
    total = math.sqrt(
        sum(float((grad.double() ** 2).sum()) for grad in gradients if grad is not None)
    )
    assert total == pytest.approx(1.0, abs=1e-5)  # unscaled first: 1/16 if the clip came before


def test_the_loss_scale_falling_below_1_fails_check_f1(
    snapshot: Path, arms: dict[str, ArmConfig], prompts: dict
) -> None:
    run = start(snapshot, tuned(arms["b0"], init_scale=1.0), prompts)
    value = next(
        value for name, value in training._trained(run.trainable.model) if ".lora_B." in name
    )
    handle = value.register_hook(lambda grad: torch.full_like(grad, math.inf))  # an overflow
    with pytest.raises(StepError, match=r"step 0: the loss scale fell to 0\.5, below 1, where"):
        run.step(0)
    handle.remove()


def test_a_loss_scale_that_falls_to_exactly_1_passes_check_f1(
    snapshot: Path, arms: dict[str, ArmConfig], prompts: dict
) -> None:
    run = start(snapshot, tuned(arms["b0"], init_scale=2.0), prompts)
    value = next(
        value for name, value in training._trained(run.trainable.model) if ".lora_B." in name
    )
    handle = value.register_hook(lambda grad: torch.full_like(grad, math.inf))  # an overflow
    record = run.step(0)  # no error: "below 1" leaves a scale of 1 alone
    handle.remove()
    assert record.skipped
    assert (record.scale, record.next_scale) == (2.0, 1.0)
    assert run.step(1).next_scale == 1.0  # and the run goes on


def test_a_step_whose_words_wrong_group_is_empty_logs_counts_of_0_and_no_nan(
    snapshot: Path, arms: dict[str, ArmConfig], prompts: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = start(snapshot, arms["cpg"], prompts)
    real = training._bag_z
    labels = {bag.session: bag.label for bag in BAGS}

    def leaning(*args: Any) -> Any:  # x1 leans far to the right label in every bag
        z, made = real(*args)
        if run.reader.view != "x1":
            return z, made
        return z + (5.0 if labels[int(run.reader.segment[:4])] else -5.0), made

    monkeypatch.setattr(training, "_bag_z", leaning)
    run.advance(RAMPED)
    line = run.step(RAMPED).line()
    assert line["bags_positive_words_wrong"] == line["bags_negative_words_wrong"] == 0
    assert line["active_positive_words_wrong"] == line["active_negative_words_wrong"] == 0
    assert line["bags_positive_words_right"] + line["bags_negative_words_right"] == 4
    json.dumps(line, allow_nan=False)
    measures = training.rule_measures([line], arms["cpg"].shared)
    assert measures["epochs"]["1"]["active_share"]["words_wrong"] is None  # not NaN
    assert measures["epochs"]["1"]["clamp_share"] == 0.0


def test_train_step_refuses_a_model_in_eval_mode_and_statistics_that_do_not_fit_the_arm(
    snapshot: Path, arms: dict[str, ArmConfig], prompts: dict
) -> None:
    run = start(snapshot, arms["b0"], prompts)
    with pytest.raises(ValueError, match="takes running statistics only if it has a gain"):
        training.train_step(
            run.trainable,
            run.optimiser,
            run.watch,
            run.arm,
            BAGS,
            run.reader,
            prompts,
            stats_of(arms["cpg"]),
            0,
            1,
        )
    with pytest.raises(ValueError, match="the step must lie from 0 to 19"):
        run.step(TOTAL)
    with pytest.raises(ValueError, match="the schedule stands at step 0, not at 1"):
        run.step(1)
    run.trainable.model.eval()
    with pytest.raises(StepError, match="not in train mode"):
        run.step(0)


# The record of a step and the start of the statistics ------------------------------------------


def test_the_running_statistics_start_from_the_zero_shot_means_of_the_start_bags(
    arms: dict[str, ArmConfig],
) -> None:
    rng = np.random.default_rng(7)
    bags = [
        Bag(9001 + index, index % 2, tuple(f"{9001 + index}_{k:03d}" for k in range(4)))
        for index in range(32)
    ]
    rows = [
        (segment, bag.session, view, float(rng.normal(1.0 if view == "x0" else -0.5, 2.0)))
        for bag in bags
        for segment in bag.segments
        for view in ("x0", "x1", "rev")
    ]
    scores = pd.DataFrame(rows, columns=["segment", "session", "view", "z"])
    gain_settings = arms["cpg"].shared.gain
    stats = training.start_stats(bags, scores, gain_settings)
    for view, found in (("x0", stats.x0), ("x1", stats.x1)):
        z = scores[scores["view"] == view].set_index("segment")["z"]
        means = np.array([np.mean([z[segment] for segment in bag.segments]) for bag in bags])
        assert found.mu == pytest.approx(means.mean(), rel=1e-12)
        assert found.raw_sigma == pytest.approx(means.std(), rel=1e-9)  # the population SD
        assert found.floor == pytest.approx(max(0.25 * means.std(), 0.1), rel=1e-9)
    with pytest.raises(StepError, match="lack 1 segments of the start bags on x1"):
        training.start_stats(
            bags, scores.drop(index=scores.index[(scores["view"] == "x1")][:1]), gain_settings
        )
    stand_in = training.stand_in_stats(arms["cpg"].shared)
    assert stand_in.x0 == stand_in.x1
    assert (stand_in.x0.mu, stand_in.x0.sigma) == (0.0, 1.0)


# Saving, releasing and the scoring model -------------------------------------------------------


def scores_of(
    model: Any, reader: Reader, prompts: dict, segments: list[tuple[str, str]]
) -> list[float]:
    return [
        voxtral.cached_score(
            model, prompts[int(reader(view, segment).shape[0])], reader(view, segment), ANSWER_IDS
        )
        for view, segment in segments
    ]


CLIPS = [("x0", segment) for bag in BAGS for segment in bag.segments]  # the long clip among them


def test_the_training_forward_gives_cached_scores_z_bit_for_bit_at_one_clip_per_call(
    snapshot: Path, arms: dict[str, ArmConfig], prompts: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    trainable = build(snapshot, tuned(arms["cpg"], dropout=0.0))
    nudge(trainable.model)
    assert trainable.model.training  # train mode and checkpointing on; LoRA's dropout off
    scoring = copy.deepcopy(trainable.model)
    training.cast_for_scoring(scoring)
    scoring.eval()
    reader = Reader()
    expected = scores_of(scoring, reader, prompts, CLIPS)
    found = []
    for view, segment in CLIPS:
        output = reader(view, segment)
        with torch.autocast("cpu", dtype=torch.float16):
            z = voxtral.cached_z(
                trainable.model, prompts[int(output.shape[0])], [output], ANSWER_IDS
            )
        found.append(float(z.detach()[0]))
    assert [z.hex() for z in found] == [z.hex() for z in expected]  # the two-chunk clip too
    four = [reader("x0", segment) for segment in BAGS[1].segments]
    with torch.autocast("cpu", dtype=torch.float16):
        together = voxtral.cached_z(trainable.model, prompts[1], four, ANSWER_IDS).detach().tolist()
    gap = max(abs(a - b) for a, b in zip(together, expected[4:8], strict=True))
    show(capsys, f"training forward, 4 clips per call against one: largest |dz| {gap:.3g}")


def test_saved_reloaded_and_cast_to_fp16_the_weights_give_the_same_z_bit_for_bit(
    snapshot: Path, arms: dict[str, ArmConfig], prompts: dict, tmp_path: Path
) -> None:
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    run = start(snapshot, arms["cpg"], prompts)
    for step in range(2):  # step 1 moves LoRA and the adapter
        run.step(step, bags=PAIR)
    trained = {
        name: value.detach().clone()
        for name, value in get_peft_model_state_dict(run.trainable.model).items()
    }
    first = training.save_adapter(run.trainable.model, tmp_path / "adapter")
    second = training.save_adapter(run.trainable.model, tmp_path / "again")
    assert first == second  # two saves, the same bytes
    data = (tmp_path / "adapter" / training.ADAPTER_FILE).read_bytes()
    assert data == (tmp_path / "again" / training.ADAPTER_FILE).read_bytes()
    # The kept file holds the fp32 master weights of LoRA and the adapter, as trained.
    saved = load_file(tmp_path / "adapter" / training.ADAPTER_FILE)
    assert saved.keys() == trained.keys()
    assert {value.dtype for value in saved.values()} == {torch.float32}
    assert same_values(saved, trained)
    assert any(".lora_B." in name for name in saved)
    assert any("multi_modal_projector" in name for name in saved)
    with pytest.raises(StepError, match="exists already"):
        training.save_adapter(run.trainable.model, tmp_path / "adapter")
    scoring = copy.deepcopy(run.trainable.model)
    training.cast_for_scoring(scoring)
    scoring.eval()
    expected = scores_of(scoring, run.reader, prompts, CLIPS)
    del scoring
    training.release_training(run.trainable, run.optimiser, run.watch)
    loaded = training.load_for_scoring(snapshot, tmp_path / "adapter", device="cpu")
    assert [z.hex() for z in scores_of(loaded.model, run.reader, prompts, CLIPS)] == [
        z.hex() for z in expected
    ]
    assert len(set(expected)) == len(expected)


def test_the_training_model_is_gone_before_the_scoring_model_is_built(
    snapshot: Path, arms: dict[str, ArmConfig], prompts: dict, tmp_path: Path
) -> None:
    run = start(snapshot, arms["b0"], prompts)
    run.step(0)
    training.save_adapter(run.trainable.model, tmp_path / "adapter")
    alive = weakref.ref(run.trainable.model)
    released = training.release_training(run.trainable, run.optimiser, run.watch)
    assert alive() is None
    assert released == {"allocated_bytes": None, "reserved_bytes": None}  # no GPU here
    assert (run.trainable.model, run.optimiser.optimizer, run.optimiser.scaler) == (
        None,
        None,
        None,
    )
    assert run.watch.record()  # its counts stay readable
    loaded = training.load_for_scoring(snapshot, tmp_path / "adapter", device="cpu")
    training.check_for_scoring(loaded.model)
    with pytest.raises(StepError, match="released already"):
        training.release_training(run.trainable, run.optimiser, run.watch)


def test_the_release_refuses_while_another_reference_holds_the_training_model(
    snapshot: Path, arms: dict[str, ArmConfig], prompts: dict
) -> None:
    run = start(snapshot, arms["b0"], prompts)
    held = run.trainable.model
    with pytest.raises(StepError, match="still held after its release"):
        training.release_training(run.trainable, run.optimiser, run.watch)
    assert held is not None


def test_load_for_scoring_refuses_a_trained_weight_left_in_fp32(
    snapshot: Path,
    arms: dict[str, ArmConfig],
    prompts: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = start(snapshot, arms["b0"], prompts)
    training.save_adapter(run.trainable.model, tmp_path / "adapter")
    real = training.cast_for_scoring

    def all_but_one(model: Any) -> int:
        kept = next(value for name, value in model.named_parameters() if ".lora_A." in name)
        before = kept.data
        cast = real(model)
        kept.data = before  # left in fp32, as PEFT loads LoRA
        return cast - 1

    monkeypatch.setattr(training, "cast_for_scoring", all_but_one)
    with pytest.raises(
        StepError, match=r"weights not in fp16: 1 \(.*lora_A.*\), 1 of them trained"
    ):
        training.load_for_scoring(snapshot, tmp_path / "adapter", device="cpu")


def test_load_for_scoring_refuses_a_module_left_in_train_mode(
    snapshot: Path,
    arms: dict[str, ArmConfig],
    prompts: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from peft import PeftModel

    run = start(snapshot, arms["b0"], prompts)
    training.save_adapter(run.trainable.model, tmp_path / "adapter")
    real = PeftModel.from_pretrained

    def stuck(model: Any, folder: Any, **kwargs: Any) -> Any:
        loaded = real(model, folder, **kwargs)
        head = loaded.get_base_model().lm_head
        head.training = True
        head.train = lambda mode=True: head  # ignores eval()
        return loaded

    monkeypatch.setattr(PeftModel, "from_pretrained", stuck)
    with pytest.raises(StepError, match=r"modules in train mode: 1 \(base_model.model.lm_head\)"):
        training.load_for_scoring(snapshot, tmp_path / "adapter", device="cpu")


def test_load_for_scoring_refuses_a_folder_without_saved_weights(
    snapshot: Path, tmp_path: Path
) -> None:
    with pytest.raises(StepError, match="holds no saved weights"):
        training.load_for_scoring(snapshot, tmp_path, device="cpu")


def test_after_a_strict_training_phase_warn_only_is_back_when_scoring_starts() -> None:
    threads = cpu_threads()
    with kept_numerics():
        scoring = voxtral.set_numerics(threads)
        assert training.training_numerics(threads, WARN_ONLY) == scoring
        strict = training.training_numerics(threads, STRICT)
        assert strict == {**scoring, "deterministic_warn_only": False}
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert voxtral.set_numerics(threads) == scoring  # scoring switches back first
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        with pytest.raises(StepError, match="not 'pending'"):
            training.training_numerics(threads, "pending")


# The checks and rules read on the training logs ------------------------------------------------


def metric_line(step: int, epoch: int, **given: Any) -> dict[str, Any]:
    """A planted line of a CPG run's metrics.jsonl: 4 bags, none active, unless given."""
    line: dict[str, Any] = {
        "step": step,
        "epoch": epoch,
        "seconds": 10.0,
        "lr": 1.0e-4,
        "adapter_lr": 2.0e-5,
        "loss": 1.2,
        "ce_x0": 0.7,
        "ce_part": 1.0,
        "ce_x1": 0.6,
        "ce_x2": 0.3,
        "hinge": 0.2,
        "lambda_g": 1.0,
        "scale": 65_536.0,
        "next_scale": 65_536.0,
        "skipped": False,
        "grad_norm": 2.0,
        "clipped": True,
        "calls": 14,
    }
    line |= dict.fromkeys(STEP_MEASURES, 0.0)
    line |= {f"bags_{group}": 1 for group in training.GROUPS}
    line |= {f"active_{group}": 0 for group in training.GROUPS}
    line |= {"sigma0": 1.0, "sigma1": 1.0, "sigma0_at_floor": 0, "sigma1_at_floor": 0}
    line |= {"z1_bar_abs_mean": 1.0, "z0_bar_abs_mean": 2.0}
    return line | given


def active_lines(active: int, steps: int = 25) -> list[dict[str, Any]]:
    """Epochs 1 to 3 of 25 steps each; ``active`` of the 200 bags of epochs 2 and 3 active."""
    lines = []
    left = active
    for step in range(3 * steps):
        epoch = step // steps + 1
        line = metric_line(step, epoch)
        if epoch > 1:
            for group in training.GROUPS:
                if left:
                    line[f"active_{group}"] = 1
                    left -= 1
        lines.append(line)
    return lines


@pytest.mark.parametrize(
    ("active", "passes", "margin"),
    [(40, True, 0.3), (39, False, 0.5), (160, True, 0.3), (161, False, 0.1), (100, True, 0.3)],
)
def test_the_gain_alive_share_reads_its_bounds_as_fewer_and_more_than(
    arms: dict[str, ArmConfig], active: int, passes: bool, margin: float
) -> None:
    alive = training.rule_measures(active_lines(active), arms["cpg"].shared, **FIRST_CPG)["alive"]
    assert (alive["bags"], alive["active"]) == (200, active)
    assert alive["share"] == active / 200  # 40 of 200 is exactly 0.20: it changes nothing
    assert (alive["f2_passes"], alive["margin_if_read"]) == (passes, margin)


def test_the_rules_give_a_verdict_only_on_the_runs_that_the_protocol_reads(
    arms: dict[str, ArmConfig],
) -> None:
    # Every bag of epochs 2 and 3 active (F2 would fail), a CE part clearly falling in the
    # last epoch (the 4-epoch rule would fire) and 3.1 h of training (4.13 h with the 4/3).
    lines = active_lines(200, steps=10)
    for index, line in enumerate(lines[20:]):  # epoch 3: a fall of 10%, far above its noise
        line["ce_part"] = 1.0 - 0.02 * index + (0.001 if index % 2 else -0.001)
    settings = arms["cpg"].shared
    other = training.rule_measures(lines, settings, training_hours=3.1)  # B1, or B0's hours
    assert set(other["alive"]) == {"epochs", "bags", "active", "margin", "share"}
    assert other["alive"]["share"] == 1.0  # the share stays, as a diagnostic
    assert "fires" not in other["falling"]
    assert other["falling"]["drop"] > 0
    assert other["f5"] == {"training_hours": 3.1}
    cpg = training.rule_measures(lines, settings, training_hours=3.1, reads_f2=True)
    assert cpg["alive"]["f2_passes"] is False  # every CPG run: F2
    assert "margin_if_read" not in cpg["alive"]
    assert "fires" not in cpg["falling"]
    assert cpg["f5"] == {"training_hours": 3.1}
    first = training.rule_measures(lines, settings, training_hours=3.1, **FIRST_CPG)
    assert (first["alive"]["f2_passes"], first["alive"]["margin_if_read"]) == (False, 0.1)
    assert first["falling"]["fires"] is True
    assert first["f5"] == {
        "training_hours": 3.1,
        "fourth_epoch": True,
        "hours": pytest.approx(3.1 * 4 / 3),
        "max_hours": 4.0,
        "passes": False,
    }
    for measures in (other, cpg, first):
        assert measures["f1"]["passes"] is True  # F1 reads every run
    later = training.rule_measures(
        lines, settings, training_hours=3.1, reads_f2=True, reads_f5=True
    )
    assert "margin_if_read" not in later["alive"]  # the first run's readings stand
    assert "fires" not in later["falling"]
    assert later["f5"] == {  # F5 on the run's own hours: never times 4/3, the epochs are set
        "training_hours": 3.1,
        "fourth_epoch": False,
        "hours": 3.1,
        "max_hours": 4.0,
        "passes": True,
    }
    over = training.rule_measures(
        lines, settings, training_hours=4.01, reads_f2=True, reads_f5=True
    )
    assert over["f5"]["passes"] is False
    with pytest.raises(ValueError, match="which check F2 reads too"):
        training.rule_measures(lines, settings, reads_rules=True)
    with pytest.raises(ValueError, match="which F2 reads too"):
        training.rule_measures(lines, settings, training_hours=3.1, reads_f5=True)
    with pytest.raises(ValueError, match="either with the allowed rules or, later, by F5 alone"):
        training.rule_measures(lines, settings, training_hours=3.1, **FIRST_CPG, reads_f5=True)
    with pytest.raises(ValueError, match="none were given"):
        training.rule_measures(lines, settings, reads_f2=True, reads_f5=True)
    b0_lines = [{key: line[key] for key in B0_KEYS} for line in lines]
    with pytest.raises(ValueError, match="read a CPG run, which logs the gain"):
        training.rule_measures(b0_lines, arms["b0"].shared, reads_f2=True)


def falling_lines(
    first: list[float], second: list[float], middle: list[float] | None = None
) -> list[dict[str, Any]]:
    """Epoch 3's steps with these CE parts, after two epochs of one step each."""
    values = [*first, *(middle or []), *second]
    lines = [metric_line(0, 1), metric_line(1, 2)]
    lines += [metric_line(2 + index, 3, ce_part=value) for index, value in enumerate(values)]
    return lines


def around(mean: float, spread: float) -> list[float]:
    """Five values with this mean and a sample SD of ``spread``."""
    return [mean + spread, mean - spread, mean + spread, mean - spread, mean]


@pytest.mark.parametrize(
    ("second", "spread", "fires"),
    [
        (0.94, 0.001, True),  # a fall of 6% above 2 standard errors
        (0.951, 0.001, False),  # 4.9%, though far above 2 standard errors
        (0.95, 0.06, False),  # 5%, but within 2 standard errors
        (1.0, 0.0, False),  # flat
        (1.0, 0.05, False),  # flat with noise
    ],
)
def test_the_4_epoch_rule_fires_only_when_both_bounds_hold(
    arms: dict[str, ArmConfig], second: float, spread: float, fires: bool
) -> None:
    lines = falling_lines(around(1.0, spread), around(second, spread))
    falling = training.rule_measures(lines, arms["cpg"].shared, **FIRST_CPG)["falling"]
    assert falling["epoch"] == 3
    assert falling["half_steps"] == 5
    assert falling["fires"] is fires
    assert falling["ce_part"]["first_half"] == pytest.approx(1.0)
    assert falling["drop"] == pytest.approx(1.0 - second)
    assert falling["standard_error"] == pytest.approx(math.sqrt(2 * spread**2 / 5), abs=1e-12)


def test_an_odd_number_of_steps_leaves_the_middle_step_out_of_the_halves(
    arms: dict[str, ArmConfig],
) -> None:
    lines = falling_lines(around(1.0, 0.001), around(0.9, 0.001), middle=[100.0])
    falling = training.rule_measures(lines, arms["cpg"].shared, **FIRST_CPG)["falling"]
    assert (falling["steps"], falling["half_steps"]) == (11, 5)
    assert falling["ce_part"] == {
        "first_half": pytest.approx(1.0),
        "second_half": pytest.approx(0.9),
    }
    assert falling["fires"]
    assert falling["loss"] == {"first_half": 1.2, "second_half": 1.2}  # reported beside it


@pytest.mark.parametrize(("lowest", "passes"), [(0.5, False), (1.0, True)])
def test_check_f1_fails_once_the_scale_fell_below_1(
    arms: dict[str, ArmConfig], lowest: float, passes: bool
) -> None:
    lines = [
        metric_line(0, 1),
        metric_line(1, 1, skipped=True, grad_norm=None, clipped=False, next_scale=2.0),
    ]
    lines += [
        metric_line(
            2, 1, scale=2.0, next_scale=lowest, skipped=lowest < 2.0, grad_norm=None, clipped=False
        )
    ]
    lines += [metric_line(3, 1, scale=lowest, next_scale=lowest, grad_norm=0.5, clipped=False)]
    f1 = training.rule_measures(lines, arms["cpg"].shared)["f1"]
    assert (f1["lowest_scale"], f1["passes"]) == (lowest, passes)
    assert (f1["skipped_steps"], f1["longest_skipped_run"]) == (2, 2)
    assert (f1["largest_grad_norm"], f1["clipped_share"], f1["non_finite_losses"]) == (2.0, 0.5, 0)


def test_the_f5_projection_and_its_reading_lie_on_either_side_of_4_hours(
    arms: dict[str, ArmConfig],
) -> None:
    first = 99.0  # the first timed step warms up and is left out
    assert training.f5_projection(120.0, [first] + [10.0] * 9, 252, 1) == pytest.approx(2640 / 3600)
    assert training.f5_projection(
        120.0, [first, 9, 10, 11, 10, 12, 8, 10, 30, 1], 252, 1
    ) == pytest.approx(2640 / 3600)  # the median of the 2nd to 10th steps
    assert training.f5_projection(120.0, [first] + [55.0] * 9, 252, 1) < 4.0
    assert training.f5_projection(120.0, [first] + [57.0] * 9, 252, 1) > 4.0
    # The first timed step is left out, not the fastest: here the first is the slowest, and
    # the median of 1 to 9 is 5 (6 if the fastest were left out instead).
    assert training.f5_projection(
        120.0, [100.0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 252, 1
    ) == pytest.approx((120 + 252 * 5) / 3600)
    assert training.f5_projection(
        120.0, [1.0, 100, 2, 3, 4, 5, 6, 7, 8, 9], 252, 1
    ) == pytest.approx((120 + 252 * 6) / 3600)  # the fastest step first: left out as the first
    with pytest.raises(ValueError, match="no timed step"):
        training.f5_projection(120.0, [first], 252, 1)
    settings = arms["cpg"].shared
    falling = falling_lines(around(1.0, 0.001), around(0.9, 0.001))
    flat = falling_lines(around(1.0, 0.001), around(1.0, 0.001))
    for lines, hours, read, passes in (
        (falling, 2.9, 2.9 * 4 / 3, True),  # times 4/3 with a 4th epoch: 3.87 h
        (falling, 3.1, 3.1 * 4 / 3, False),  # 4.13 h
        (flat, 3.1, 3.1, True),
    ):
        f5 = training.rule_measures(lines, settings, training_hours=hours, **FIRST_CPG)["f5"]
        assert f5["hours"] == pytest.approx(read)
        assert (f5["passes"], f5["max_hours"]) == (passes, 4.0)
    assert training.rule_measures(flat, settings, **FIRST_CPG)["f5"] is None


def test_the_rule_measures_of_an_epoch_hold_its_shares_and_statistics_without_nan(
    arms: dict[str, ArmConfig],
) -> None:
    lines = [
        metric_line(
            0, 1, z0_bar_mean=1.0, z0_bar_mean_square=5.0, mu0=0.5, sigma0=1.5, sigma0_at_floor=1
        ),
        metric_line(1, 1, z0_bar_mean=3.0, z0_bar_mean_square=21.0, mu0=1.5, sigma0=2.5),
    ]
    lines[0] |= {"active_positive_words_right": 1, "bags_negative_words_wrong": 0}
    epoch = training.rule_measures(lines, arms["cpg"].shared)["epochs"]["1"]
    assert epoch["x0"] == {"measured_mean": 2.0, "measured_spread": 3.0, "mu": 1.0, "sigma": 2.0}
    # 7 bags: 2 in each group but 1 negative words-wrong; 1 active, positive and words right.
    assert epoch["active_share"] == {
        "all": 1 / 7,
        "positive": 1 / 4,
        "negative": 0.0,
        "words_right": 1 / 4,
        "words_wrong": 0.0,
    }
    assert epoch["clamp_share"] == 3 / 7  # the words-wrong bags, where the clamp acts
    assert epoch["sigma0_at_floor_share"] == 0.5
    assert epoch["z0_over_z1"] == 2.0
    assert epoch["minutes"] == pytest.approx(20 / 60)
    json.dumps(training.rule_measures(lines, arms["cpg"].shared), allow_nan=False)


def test_the_rule_measures_of_b0_hold_no_gain(arms: dict[str, ArmConfig]) -> None:
    lines = [
        {key: value for key, value in metric_line(step, 1).items() if key in B0_KEYS}
        for step in range(4)
    ]
    measures = training.rule_measures(lines, arms["b0"].shared)
    assert measures["alive"] is None
    assert "active_share" not in measures["epochs"]["1"]
    assert "fires" not in measures["falling"]  # the 4-epoch rule reads a CPG run
    assert measures["falling"]["half_steps"] == 2
    assert measures["f5"] is None


def test_b0s_logged_measures_are_its_bag_means_and_its_ce_push() -> None:
    labels = [1, 0, 1, 0]
    z0 = torch.tensor(
        [[0.4, 0.9, -0.2, 0.1], [1.2, 0.3, 0.7, -0.5], [-1.0, 0.2, 0.0, 0.6], [2.1, -0.4, 0.3, 0.8]]
    )
    bars = z0.mean(dim=1)  # 0.3, 0.425, -0.05 and 0.7: means of different sizes and signs
    measures = training._x0_measures(labels, bars, z0)
    assert set(measures) == set(training.X0_MEASURES)
    expected = bars.double()
    assert measures["z0_bar_mean"] == pytest.approx(float(expected.mean()), rel=1e-12)
    assert measures["z0_bar_mean_square"] == pytest.approx(float((expected**2).mean()), rel=1e-12)
    assert measures["z0_bar_mean_square"] > measures["z0_bar_mean"] ** 2  # not the squared mean
    assert measures["z0_bar_abs_mean"] == pytest.approx(0.36875, rel=1e-6)  # not |mean| 0.34375
    y = torch.tensor(labels, dtype=torch.float32)[:, None]
    push = float((torch.sigmoid(z0) - y).abs().mean())  # mean |sigmoid(z) - y| over the clips
    assert measures["ce_push"] == pytest.approx(push, rel=1e-6)
    flipped = float((torch.sigmoid(z0) - (1.0 - y)).abs().mean())
    assert abs(push - flipped) > 0.05  # the labels' side matters here


# GPU tests on OMNI: the real model, synthetic encoder outputs only ------------------------------


PINS = CONFIGS / "models.yaml"


@pytest.fixture(scope="module")
def real_snapshot() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        pytest.skip("HF_HOME is not set; run the GPU tests in scripts/slurm/smoke_test.sbatch")
    return local_snapshot(load_pins(PINS)["voxtral"], Path(hf_home))


@pytest.fixture
def gpu_numerics() -> Iterator[int]:
    """The numeric settings of the caches, set first as every job sets them."""
    threads = cpu_threads()
    with kept_numerics():
        voxtral.set_numerics(threads)
        yield threads


class RealReader(Reader):
    """Synthetic encoder outputs of the real model's width, made on first read."""

    def __call__(self, view: str, segment: str) -> torch.Tensor:
        self.view, self.segment = view, segment
        if (view, segment) not in self.outputs:
            chunks = 2 if segment in self.long and view != "x1" else 1
            generator = torch.Generator().manual_seed(item_seed(5, view, segment))
            values = torch.randn(chunks, voxtral.FRAMES_PER_CHUNK, 1280, generator=generator)
            self.outputs[(view, segment)] = values.half()
        return self.outputs[(view, segment)]


@contextlib.contextmanager
def on_gpu(arm: ArmConfig, snapshot: Path, total: int) -> Iterator[Run]:
    """A run of the real model on the GPU; whatever it holds is dropped at the end."""
    trainable = training.build_trainable(snapshot, arm.shared.lora, 11)
    optimiser = training.make_optimiser(trainable, arm.shared, total)
    watch = voxtral.watch_layers(voxtral.language_model_layers(trainable.model, logits=True))
    prompts = {count: voxtral.prompt_inputs(trainable.processor, count) for count in (1, 2)}
    run = Run(
        arm,
        trainable,
        optimiser,
        watch,
        training.stand_in_stats(arm.shared) if arm.lambda_g is not None else None,
        RealReader(),
        prompts,
    )
    try:
        yield run
    finally:
        watch.remove()
        trainable.model = None
        optimiser.optimizer = optimiser.schedule = optimiser.scaler = None
        gc.collect()
        torch.cuda.empty_cache()


def gib(value: float) -> str:
    return f"{value / 2**30:.2f} GiB"


@pytest.mark.gpu
def test_the_training_forward_gives_cached_scores_z_bit_for_bit_on_the_gpu(
    real_snapshot: Path,
    arms: dict[str, ArmConfig],
    gpu_numerics: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arm = tuned(arms["cpg"], dropout=0.0)
    with on_gpu(arm, real_snapshot, total=10) as run:
        # Step 1 has the full rate, but the scaler may skip it at its start scale of 2^16;
        # every LoRA B is then set away from 0, as in the CPU test, so that LoRA adds to z
        # in both paths and a difference in its number format shows.
        skipped = sum(run.step(step).skipped for step in range(2))
        nudge(run.trainable.model)
        lora_b = [
            value for name, value in run.trainable.model.named_parameters() if ".lora_B." in name
        ]
        assert lora_b
        assert all(bool(value.detach().ne(0).any()) for value in lora_b)
        answer_ids = run.trainable.answer_ids
        clips = [("x0", segment) for bag in BAGS for segment in bag.segments]
        found = []
        for view, segment in clips:
            output = run.reader(view, segment)
            with torch.autocast("cuda", dtype=torch.float16):
                z = voxtral.cached_z(
                    run.trainable.model, run.prompts[int(output.shape[0])], [output], answer_ids
                )
            found.append(float(z.detach()[0]))
        four = [run.reader("x0", segment) for segment in BAGS[1].segments]
        with torch.autocast("cuda", dtype=torch.float16):
            together = (
                voxtral.cached_z(run.trainable.model, run.prompts[1], four, answer_ids)
                .detach()
                .tolist()
            )
        training.cast_for_scoring(run.trainable.model)
        run.trainable.model.eval()
        expected = [
            voxtral.cached_score(
                run.trainable.model,
                run.prompts[int(run.reader(v, s).shape[0])],
                run.reader(v, s),
                answer_ids,
            )
            for v, s in clips
        ]
    gap = max(abs(a - b) for a, b in zip(together, expected[4:8], strict=True))
    equal = sum(a.hex() == b.hex() for a, b in zip(found, expected, strict=True))
    show(
        capsys,
        f"training forward against cached_score: {equal} of {len(clips)} equal at one clip"
        " per call (the two-chunk clip among them), LoRA B set after 2 steps"
        f" ({skipped} skipped); 4 clips per call: largest |dz| {gap:.3g}",
    )
    assert [z.hex() for z in found] == [z.hex() for z in expected]


@pytest.mark.gpu
def test_a_three_view_step_with_the_two_chunk_clip_fits_on_the_gpu(
    real_snapshot: Path,
    arms: dict[str, ArmConfig],
    gpu_numerics: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with on_gpu(arms["cpg"], real_snapshot, total=10) as run:
        torch.cuda.reset_peak_memory_stats()
        record = run.step(0)
        step_peak = torch.cuda.max_memory_allocated()
        show(
            capsys,
            "three-view CPG step, the bag of 3 one-chunk clips and the two-chunk clip: peak"
            f" {gib(step_peak)}, {record.calls} calls, {record.seconds:.1f} s, skipped"
            f" {record.skipped}, scale {record.next_scale:g}",
        )
        run.trainable.model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        sixteen = [run.reader("x0", f"{9101 + index}_000") for index in range(16)]
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                z = voxtral.cached_z(
                    run.trainable.model, run.prompts[1], sixteen, run.trainable.answer_ids
                )
            with run.watch.pause():
                z.float().mean().backward()
            outcome = f"peak {gib(torch.cuda.max_memory_allocated())}"
        except torch.cuda.OutOfMemoryError:
            outcome = "does not fit"
        show(capsys, f"one call of a view's 16 one-chunk clips, forward and backward: {outcome}")
        largest = "; ".join(f"{name} {value:.6g}" for name, value in run.watch.largest(5))
        show(capsys, f"watch: {len(run.watch.flagged())} parts flagged; largest |value|: {largest}")
    assert record.calls == 14


@pytest.mark.gpu
def test_ten_b0_steps_twice_in_each_repeatability_mode(
    real_snapshot: Path,
    arms: dict[str, ArmConfig],
    gpu_numerics: int,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from torch.profiler import ProfilerActivity, profile

    found: dict[str, list[Any]] = {STRICT: [], WARN_ONLY: []}
    medians: dict[str, list[float]] = {STRICT: [], WARN_ONLY: []}
    kernels: set[str] = set()
    implementation = "unknown"
    for number, mode in enumerate((STRICT, STRICT, WARN_ONLY, WARN_ONLY)):
        training.training_numerics(gpu_numerics, mode)
        try:
            with on_gpu(arms["b0"], real_snapshot, total=10) as run:
                seconds = [run.step(step).seconds for step in range(10)]
                medians[mode].append(float(np.median(seconds[1:])))  # the first warms up
                found[mode].append(
                    training.save_adapter(run.trainable.model, tmp_path / f"{mode}-{number}")
                )
                if not kernels:
                    output = run.reader("x0", "9002_000")
                    with (
                        profile(activities=[ProfilerActivity.CUDA]) as profiler,
                        torch.autocast("cuda", dtype=torch.float16),
                    ):
                        voxtral.cached_z(
                            run.trainable.model, run.prompts[1], [output], run.trainable.answer_ids
                        )
                    kernels = {
                        event.name
                        for event in profiler.events()
                        if re.search(
                            "attention|fmha|flash|efficient|sdpa", event.name, re.IGNORECASE
                        )
                    }
                    implementation = (
                        run.trainable.model.get_base_model().config._attn_implementation
                    )
        except RuntimeError as error:  # PyTorch's own message names the operation
            found[mode].append(f"raised: {str(error).splitlines()[0][:300]}")
        finally:
            voxtral.set_numerics(gpu_numerics)
    for mode in (STRICT, WARN_ONLY):
        runs = found[mode]
        equal = len(runs) == 2 and runs[0] == runs[1] and not str(runs[0]).startswith("raised")
        times = ", ".join(f"{median:.3f}" for median in medians[mode]) or "none"
        show(
            capsys,
            f"{mode}: {'; '.join(str(item)[:80] for item in runs)}; saved bytes equal"
            f" {equal}; median seconds of the 2nd to 10th step of each run: {times}",
        )
    show(
        capsys,
        f"attention: {implementation}; flash {torch.backends.cuda.flash_sdp_enabled()},"
        f" memory-efficient {torch.backends.cuda.mem_efficient_sdp_enabled()}, math"
        f" {torch.backends.cuda.math_sdp_enabled()}; kernels {sorted(kernels)}",
    )


@pytest.mark.gpu
def test_saved_reloaded_and_cast_to_fp16_the_weights_give_the_same_z_bit_for_bit_on_the_gpu(
    real_snapshot: Path,
    arms: dict[str, ArmConfig],
    gpu_numerics: int,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clips = [("x0", segment) for bag in BAGS for segment in bag.segments]
    with on_gpu(arms["cpg"], real_snapshot, total=10) as run:
        records = [run.step(step) for step in range(4)]
        training.save_adapter(run.trainable.model, tmp_path / "adapter")
        training.cast_for_scoring(run.trainable.model)
        run.trainable.model.eval()
        answer_ids = run.trainable.answer_ids
        prompts, reader = run.prompts, run.reader
        expected = [
            voxtral.cached_score(
                run.trainable.model, prompts[int(reader(v, s).shape[0])], reader(v, s), answer_ids
            )
            for v, s in clips
        ]
        released = training.release_training(run.trainable, run.optimiser, run.watch)
    loaded = training.load_for_scoring(real_snapshot, tmp_path / "adapter")
    try:
        again = [
            voxtral.cached_score(
                loaded.model, prompts[int(reader(v, s).shape[0])], reader(v, s), answer_ids
            )
            for v, s in clips
        ]
    finally:
        del loaded
        gc.collect()
        torch.cuda.empty_cache()
    skipped = sum(record.skipped for record in records)
    lowest = min(min(record.scale, record.next_scale) for record in records)
    show(
        capsys,
        f"4 CPG steps: {skipped} skipped, lowest scale {lowest:g}; after the release"
        f" {released['allocated_bytes']} bytes allocated; reloaded z equal"
        f" {sum(a.hex() == b.hex() for a, b in zip(again, expected, strict=True))} of"
        f" {len(clips)}",
    )
    assert [z.hex() for z in again] == [z.hex() for z in expected]
