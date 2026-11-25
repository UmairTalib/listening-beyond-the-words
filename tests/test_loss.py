"""The loss of the trained arms (loss.py): CE, the gain's numbers, the statistics and the measures.

The reference checks of the gain are in test_cpg_gain.py. Here: the gain in at least fp32
from fp16 scores, within a bound that scales with its inputs; the cross-entropy on z; the
soft hinge; λG's ramp and the shares of a run's steps; the running statistics, their start,
update and floor; and one step's logged counts and measures (pilot protocol, section
"Training", bullets "The gain term", "Loss and schedules" and "Logged measures").
"""

import dataclasses
import json
import math
from collections.abc import Callable

import numpy as np
import pytest
import torch

from listening_beyond_words.config import GainSettings
from listening_beyond_words.loss import (
    STEP_MEASURES,
    RunningStats,
    ce_on_z,
    gain,
    lambda_g,
    share_steps,
    soft_hinge,
    step_measures,
)

M, S = 0.3, 0.1
SETTINGS = GainSettings(
    margin=M,
    softness=S,
    ramp_share=0.2,
    momentum=0.9,
    start_steps=8,
    floor_share=0.25,
    floor_logit=0.1,
)
F64 = torch.float64
EPS32 = 2.0**-24  # fp32 rounds a value by at most this share of it


def floats(*values: float, dtype: torch.dtype = F64, grad: bool = False) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, requires_grad=grad)


def at_floor(mu: float) -> RunningStats:
    """Statistics whose spread (0.05) lies below their floor of 0.1."""
    return RunningStats(momentum=0.9, first=mu, second=mu * mu + 0.05**2, floor=0.1)


# The gain's number format ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scale", "floored"), [(2.0, False), (8.0, True), (15.0, True), (40.0, True)]
)
def test_the_gain_from_fp16_scores_is_fp32_within_a_bound_that_scales_with_its_inputs(
    scale: float, floored: bool
) -> None:
    rng = np.random.default_rng(int(scale))
    z0 = torch.tensor(rng.normal(0.0, scale, 20_000)).half()
    z1 = torch.tensor(rng.normal(0.0, scale, 20_000)).half()
    y = torch.tensor(rng.integers(0, 2, 20_000))
    mu0, mu1 = 0.37, -1.21
    if floored:
        stats0, stats1 = at_floor(mu0), at_floor(mu1)
        assert stats0.at_floor
        assert stats0.sigma == 0.1
    else:
        stats0, stats1 = RunningStats.of(mu0, 1.0, SETTINGS), RunningStats.of(mu1, 1.0, SETTINGS)
    g32 = gain(z0, z1, y, stats0, stats1)
    assert g32.dtype == torch.float32
    # A float64 reference from the same fp16 values.
    g64 = gain(z0.double(), z1.double(), y, stats0, stats1)
    assert g64.dtype == torch.float64
    error = (g32.double() - g64).abs()
    bound = (
        8
        * EPS32
        * (
            (z0.double().abs() + abs(mu0)) / stats0.sigma
            + (z1.double().abs() + abs(mu1)) / stats1.sigma
        )
    )
    assert bool((error <= bound).all())
    if floored:
        # A fixed tolerance would fail at the floor, where |G| reaches hundreds.
        assert float(error.max()) > 1e-6
        with pytest.raises(AssertionError):
            torch.testing.assert_close(g32.double(), g64, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("dtype0", "dtype1", "expected"),
    [
        (torch.float16, torch.float16, torch.float32),
        (torch.bfloat16, torch.float16, torch.float32),
        (torch.float16, torch.float32, torch.float32),
        (torch.float32, torch.float32, torch.float32),
        (torch.float64, torch.float64, torch.float64),
        (torch.float16, torch.float64, torch.float64),
    ],
)
def test_the_gain_is_computed_in_at_least_fp32(
    dtype0: torch.dtype, dtype1: torch.dtype, expected: torch.dtype
) -> None:
    stats = RunningStats.of(0.0, 1.0, SETTINGS)
    z0, z1 = floats(1.5, -2.0, dtype=dtype0), floats(0.5, 0.25, dtype=dtype1)
    assert gain(z0, z1, [1, 0], stats, stats).dtype == expected
    assert soft_hinge(z0, M, S).dtype == torch.promote_types(dtype0, torch.float32)
    assert ce_on_z(z0, 1).dtype == torch.promote_types(dtype0, torch.float32)


def test_fp16_scores_are_cast_before_any_arithmetic() -> None:
    # 65,504 is the largest fp16 value: in fp16, z̄0 − μ0 would overflow to inf.
    stats0 = RunningStats.of(-100.0, 1.0, SETTINGS)
    stats1 = RunningStats.of(0.0, 1.0, SETTINGS)
    g = gain(
        floats(65_504.0, dtype=torch.float16), floats(0.0, dtype=torch.float16), [1], stats0, stats1
    )
    assert float(g) == 65_604.0


def test_the_gain_has_one_value_per_bag_and_only_z0_bar_passes_a_gradient() -> None:
    stats0, stats1 = RunningStats.of(0.2, 1.5, SETTINGS), RunningStats.of(-0.1, 0.8, SETTINGS)
    z0_bar = floats(0.5, -1.0, 2.0, 0.1, dtype=torch.float32, grad=True)
    z1_bar = floats(1.0, 1.0, -0.5, 0.3, dtype=torch.float32, grad=True)
    g = gain(z0_bar, z1_bar, [1, 0, 1, 0], stats0, stats1)
    assert g.shape == (4,)
    g0, g1 = torch.autograd.grad(
        soft_hinge(g, M, S).sum(), (z0_bar, z1_bar), allow_unused=True, materialize_grads=True
    )
    assert bool((g0 != 0).all())
    assert torch.equal(g1, torch.zeros_like(g1))


@pytest.mark.parametrize(
    ("z0_bar", "z1_bar", "y", "message"),
    [
        (floats(0.1, 0.2), floats(0.1), [1, 0], "both have shape"),
        (floats(), floats(), [], "both have shape"),
        (floats(0.1, 0.2).reshape(1, 2), floats(0.1, 0.2).reshape(1, 2), [1, 0], "both have shape"),
        (torch.tensor([1, 2]), torch.tensor([1, 2]), [1, 0], "floating point"),
        (floats(0.1, 0.2), floats(0.1, 0.2), [1, 2], "0 or 1"),
        (floats(0.1, 0.2), floats(0.1, 0.2), [1], "one label per entry"),
    ],
)
def test_the_gain_refuses_inputs_that_do_not_fit(
    z0_bar: torch.Tensor, z1_bar: torch.Tensor, y: list[int], message: str
) -> None:
    stats = RunningStats.of(0.0, 1.0, SETTINGS)
    with pytest.raises(ValueError, match=message):
        gain(z0_bar, z1_bar, y, stats, stats)


# Cross-entropy and the soft hinge --------------------------------------------------------------


@pytest.mark.parametrize("y", [0, 1])
def test_ce_is_the_binary_cross_entropy_on_z_with_weight_1_averaged_over_a_bags_clips(
    y: int,
) -> None:
    z = floats(2.0, -1.0, 0.5, 3.0, grad=True)
    p = torch.sigmoid(z.detach())
    expected = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
    ce = ce_on_z(z, y)
    assert ce.shape == ()
    assert float(ce.detach()) == pytest.approx(float(expected), abs=1e-15)
    # Each of the 4 clips counts one quarter.
    (gradient,) = torch.autograd.grad(ce, z)
    assert torch.allclose(gradient, (p - y) / 4, rtol=0.0, atol=1e-15)
    # One label per clip gives the same.
    assert torch.equal(ce_on_z(z, torch.full((4,), y)), ce)


@pytest.mark.parametrize(
    ("z", "y", "message"),
    [
        (floats(), 1, "shape"),
        (floats(0.1, 0.2).reshape(2, 1), 1, "shape"),
        (torch.tensor([1, 2]), 1, "floating point"),
        (floats(0.1, 0.2), 2, "0 or 1"),
        (floats(0.1, 0.2), torch.tensor([1, 0, 1]), "one label per entry"),
    ],
)
def test_ce_refuses_inputs_that_do_not_fit(
    z: torch.Tensor, y: int | torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ce_on_z(z, y)


def test_the_soft_hinge_never_overflows() -> None:
    g = floats(-1.0e4, M, 1.0e4)
    hinge = soft_hinge(g, M, S)
    assert bool(torch.isfinite(hinge).all())
    assert float(hinge[0]) == pytest.approx(M + 1.0e4)
    assert float(hinge[1]) == pytest.approx(S * math.log(2.0))
    assert float(hinge[2]) == 0.0


@pytest.mark.parametrize("softness", [0.0, -0.1, math.inf, math.nan])
def test_the_soft_hinge_refuses_a_softness_that_is_not_above_0(softness: float) -> None:
    with pytest.raises(ValueError, match="softness"):
        soft_hinge(floats(0.1), M, softness)


# λG and the shares of a run's steps ------------------------------------------------------------


def test_lambda_g_ramps_up_over_the_first_20_percent_of_100_steps() -> None:
    assert lambda_g(0, 100, 0.2) == 0.0
    assert lambda_g(10, 100, 0.2) == 0.5
    assert lambda_g(19, 100, 0.2) == 0.95
    for step in range(20, 100):
        assert lambda_g(step, 100, 0.2) == 1.0


def test_lambda_g_ramps_up_over_51_of_252_steps() -> None:
    # T = 252 in an average fold: ⌈0.2 · 252⌉ = 51.
    for step in range(51):
        assert lambda_g(step, 252, 0.2) == step / 51
    for step in range(51, 252):
        assert lambda_g(step, 252, 0.2) == 1.0


def test_lambda_g_is_the_arms_peak_after_the_ramp() -> None:
    # B1 is CPG's code with λG = 0: it never trains on the gain.
    assert all(lambda_g(step, 252, 0.2, peak=0.0) == 0.0 for step in range(252))
    assert lambda_g(25, 100, 0.2, peak=0.5) == 0.5
    assert lambda_g(5, 100, 0.2, peak=2.0) == 0.5


@pytest.mark.parametrize(
    ("share", "total", "expected"),
    [
        (0.2, 252, 51),
        (0.05, 252, 13),
        (0.2, 100, 20),
        (0.05, 100, 5),
        (0.2, 1, 1),
        (1.0, 7, 7),
        (0.05, 336, 17),
    ],
)
def test_a_share_of_a_runs_steps_is_rounded_up_from_its_decimal(
    share: float, total: int, expected: int
) -> None:
    # 0.2 · 100 is 20, not 21 from 0.2's binary rounding.
    assert share_steps(share, total) == expected


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: lambda_g(-1, 100, 0.2), "step"),
        (lambda: lambda_g(100, 100, 0.2), "step"),
        (lambda: lambda_g(True, 100, 0.2), "step"),
        (lambda: lambda_g(0, 100, 0.2, peak=-1.0), "peak"),
        (lambda: lambda_g(0, 100, 0.2, peak=math.nan), "peak"),
        (lambda: share_steps(0.0, 100), "share"),
        (lambda: share_steps(1.5, 100), "share"),
        (lambda: share_steps(math.nan, 100), "share"),
        (lambda: share_steps(0.2, 0), "whole number"),
        (lambda: share_steps(0.2, True), "whole number"),
    ],
)
def test_the_schedule_refuses_values_outside_their_range(
    call: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        call()


# The running statistics ------------------------------------------------------------------------


def test_the_statistics_start_from_the_mean_and_population_sd_of_the_32_start_bags() -> None:
    means = np.random.default_rng(3).normal(0.4, 1.3, 32)
    stats = RunningStats.start(torch.tensor(means), SETTINGS)
    assert stats.mu == pytest.approx(float(np.mean(means)), rel=1e-15)
    assert stats.sigma == pytest.approx(float(np.std(means)), rel=1e-12)  # ddof 0: population
    assert stats.second == pytest.approx(float(np.mean(means**2)), rel=1e-15)
    assert stats.floor == max(0.25 * stats.raw_sigma, 0.1)
    assert stats.momentum == 0.9
    assert not stats.at_floor


def test_the_floor_is_a_quarter_of_the_start_spread_or_0_1() -> None:
    wide = RunningStats.start([-1.0, 3.0] * 16, SETTINGS)  # spread 2.0
    assert wide.sigma == 2.0
    assert wide.floor == 0.5
    narrow = RunningStats.start([0.9, 1.1] * 16, SETTINGS)  # spread 0.1
    assert narrow.floor == 0.1


@pytest.mark.parametrize(("mu", "sigma"), [(0.2, 1.3), (-0.4, 0.6), (0.0, 1.0)])
def test_statistics_of_a_given_mu_and_sigma_give_them_back(mu: float, sigma: float) -> None:
    stats = RunningStats.of(mu, sigma, SETTINGS)
    assert stats.mu == mu
    assert stats.sigma == pytest.approx(sigma, rel=1e-15)  # σ, not √σ: the average square holds σ²
    assert stats.raw_sigma == pytest.approx(sigma, rel=1e-15)
    assert stats.second == pytest.approx(mu * mu + sigma * sigma, rel=1e-15)
    assert stats.floor == max(0.25 * sigma, 0.1)  # σ is also the floor's start spread
    assert not stats.at_floor


def test_both_views_update_by_0_9_of_the_old_and_0_1_of_the_new_moments_without_a_graph() -> None:
    rng = np.random.default_rng(5)
    stats0 = RunningStats.start(rng.normal(0.4, 1.3, 32), SETTINGS)
    stats1 = RunningStats.start(rng.normal(-0.2, 0.9, 32), SETTINGS)
    weight = torch.tensor(2.0, requires_grad=True)
    z0_bar = weight * torch.tensor(rng.normal(0.0, 1.0, 4))  # part of a graph
    z1_bar = weight * torch.tensor(rng.normal(0.0, 1.0, 4))
    new0, new1 = stats0.update(z0_bar), stats1.update(z1_bar)
    for old, new, values in ((stats0, new0, z0_bar), (stats1, new1, z1_bar)):
        step = values.detach().numpy()
        first = 0.9 * old.first + 0.1 * float(step.mean())
        second = 0.9 * old.second + 0.1 * float((step**2).mean())
        assert new.first == pytest.approx(first, rel=1e-15, abs=1e-15)
        assert new.second == pytest.approx(second, rel=1e-15, abs=1e-15)
        assert (new.momentum, new.floor) == (old.momentum, old.floor)
        # No graph: the statistics are Python floats.
        assert all(type(getattr(new, field.name)) is float for field in dataclasses.fields(new))


def test_a_steps_gains_use_the_statistics_from_before_it() -> None:
    before0 = RunningStats.of(0.1, 1.2, SETTINGS)
    before1 = RunningStats.of(-0.3, 0.7, SETTINGS)
    z0_bar, z1_bar, y = floats(0.9, -0.4, 1.5, -2.0), floats(0.2, 0.8, -1.1, 0.4), [1, 0, 1, 0]
    gains = gain(z0_bar, z1_bar, y, before0, before1)
    after0, after1 = before0.update(z0_bar), before1.update(z1_bar)
    # An update returns new statistics; the old ones, and the gains made with them, stay.
    assert before0 == RunningStats.of(0.1, 1.2, SETTINGS)
    assert before1 == RunningStats.of(-0.3, 0.7, SETTINGS)
    assert torch.equal(gain(z0_bar, z1_bar, y, before0, before1), gains)
    assert not torch.equal(gain(z0_bar, z1_bar, y, after0, after1), gains)
    with pytest.raises(dataclasses.FrozenInstanceError):
        before0.first = 0.0  # type: ignore[misc]


def test_equal_bag_means_leave_sigma_finite_at_its_floor_of_0_1_and_the_gain_finite() -> None:
    stats = RunningStats.start([-7.3] * 32, SETTINGS)
    same = floats(-7.3, -7.3, -7.3, -7.3)
    for _ in range(31):  # the start and 30 updates
        assert math.isfinite(stats.sigma)
        assert stats.sigma == 0.1
        assert stats.at_floor
        assert bool(torch.isfinite(gain(same, same, [1, 0, 1, 0], stats, stats)).all())
        stats = stats.update(same)


def test_sigma_is_the_root_of_a_variance_taken_as_0_when_rounding_makes_it_negative() -> None:
    # The average square lies one rounding step below the squared average.
    stats = RunningStats(momentum=0.9, first=1.1, second=math.nextafter(1.1 * 1.1, 0.0), floor=0.1)
    assert stats.second - stats.first**2 < 0
    assert stats.raw_sigma == 0.0
    assert stats.sigma == 0.1
    assert stats.at_floor
    assert bool(torch.isfinite(gain(floats(1.1), floats(1.1), [1], stats, stats)).all())


def test_after_a_start_spread_of_2_equal_bag_means_stop_sigma_at_a_quarter_of_it() -> None:
    stats = RunningStats.start([-0.5] * 16 + [3.5] * 16, SETTINGS)  # mean 1.5, spread 2.0
    assert stats.sigma == 2.0
    for _ in range(20):
        stats = stats.update([1.5] * 4)
    assert stats.sigma > 0.5  # still falling: 2 · 0.9^10 = 0.70
    for _ in range(40):
        stats = stats.update([1.5] * 4)
    assert stats.raw_sigma < 0.5
    assert stats.sigma == 0.5  # σ_zs / 4, above 0.1
    assert stats.at_floor


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: RunningStats.start([], SETTINGS), "shape"),
        (lambda: RunningStats.start([0.1, math.nan], SETTINGS), "finite"),
        (lambda: RunningStats.start([[0.1, 0.2]], SETTINGS), "shape"),
        (lambda: RunningStats.of(0.0, 1.0, SETTINGS).update([math.inf]), "finite"),
        (lambda: RunningStats.of(0.0, 0.0, SETTINGS), "above 0"),
        (lambda: RunningStats.of(math.nan, 1.0, SETTINGS), "finite"),
        (lambda: RunningStats(momentum=1.0, first=0.0, second=1.0, floor=0.1), "momentum"),
        (lambda: RunningStats(momentum=0.9, first=0.0, second=-1.0, floor=0.1), "squares"),
        (lambda: RunningStats(momentum=0.9, first=0.0, second=1.0, floor=0.0), "floor"),
        (lambda: RunningStats(momentum=0.9, first=math.inf, second=1.0, floor=0.1), "finite"),
    ],
)
def test_the_statistics_refuse_values_that_are_missing_or_not_finite(
    call: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        call()


# One step's logged measures --------------------------------------------------------------------


def step(
    z0_bar: list[float],
    z1_bar: list[float],
    y: list[int],
    stats0: RunningStats,
    stats1: RunningStats,
    lambda_now: float = 1.0,
) -> dict[str, int | float | None]:
    """The measures of a step whose x0 clips all score their bag's mean."""
    z0 = floats(*z0_bar)
    return step_measures(
        y, z0, floats(*z1_bar), z0[:, None].expand(-1, 4), stats0, stats1, lambda_now, SETTINGS
    )


def test_a_bag_at_the_margin_is_not_active_and_one_just_below_it_is() -> None:
    unit = RunningStats.of(0.0, 1.0, SETTINGS)
    words = RunningStats.of(0.5, 1.0, SETTINGS)  # z̄1 = 0.5 gives ẑ1 = 0: a reference of 0
    for y in (1, 0):
        t = 2 * y - 1
        measures = step([t * M, t * (M - 1e-12)], [0.5, 0.5], [y, y], unit, words)
        name = "positive" if y == 1 else "negative"
        assert measures[f"bags_{name}_words_wrong"] == 2  # ẑ1 = 0 counts as wrong
        assert measures[f"bags_{name}_words_right"] == 0
        assert measures[f"active_{name}_words_wrong"] == 1  # G_b = m is not active


def test_the_counts_split_the_bags_by_class_and_by_words() -> None:
    unit = RunningStats.of(0.0, 1.0, SETTINGS)
    # Positive: words right and active; words right, not active; negative: words wrong and
    # active; words right and active.
    measures = step([0.5, 2.0, 0.5, -1.0], [0.8, 0.8, 0.4, -0.8], [1, 1, 0, 0], unit, unit)
    assert {
        key: value for key, value in measures.items() if key.startswith(("bags", "active"))
    } == {
        "bags_positive_words_right": 2,
        "bags_positive_words_wrong": 0,
        "bags_negative_words_right": 1,
        "bags_negative_words_wrong": 1,
        "active_positive_words_right": 1,
        "active_positive_words_wrong": 0,
        "active_negative_words_right": 1,
        "active_negative_words_wrong": 1,
    }


def test_a_step_without_a_words_wrong_bag_logs_counts_of_0_and_no_nan() -> None:
    unit = RunningStats.of(0.0, 1.0, SETTINGS)
    measures = step([0.5, 2.0, -0.5, -2.0], [0.8, 0.8, -0.8, -0.8], [1, 1, 0, 0], unit, unit)
    assert measures["bags_positive_words_wrong"] == measures["bags_negative_words_wrong"] == 0
    assert measures["active_positive_words_wrong"] == measures["active_negative_words_wrong"] == 0
    assert all(value is not None and math.isfinite(value) for value in measures.values())
    json.dumps(measures, allow_nan=False)


def test_the_measures_are_the_logged_ones_in_their_order_with_counts_as_whole_numbers() -> None:
    stats0, stats1 = at_floor(0.1), RunningStats.of(-0.2, 1.0, SETTINGS)
    measures = step([0.5, 2.0, 0.5, -1.0], [0.8, 0.8, 0.4, -0.8], [1, 1, 0, 0], stats0, stats1)
    assert tuple(measures) == STEP_MEASURES
    for key, value in measures.items():
        counted = key.startswith(("bags_", "active_")) or key.endswith("_at_floor")
        assert type(value) is (int if counted else float), key
    assert (measures["sigma0_at_floor"], measures["sigma1_at_floor"]) == (1, 0)
    assert (measures["mu0"], measures["sigma0"]) == (0.1, 0.1)
    assert (measures["mu1"], measures["sigma1"]) == (stats1.mu, stats1.sigma)


def test_the_pushes_and_the_offset_are_the_sizes_of_the_gradients() -> None:
    stats0 = RunningStats.of(0.2, 1.3, SETTINGS)
    stats1 = RunningStats.of(-0.1, 0.9, SETTINGS)
    y = [1, 0, 1, 0]
    z0 = floats(0.4, 0.9, -0.2, 0.1, 1.2, 0.3, 0.7, -0.5, -1.0, 0.2, 0.0, 0.6, 2.1, -0.4, 0.3, 0.8)
    z0 = z0.reshape(4, 4).requires_grad_(True)
    z0_bar = z0.mean(dim=1)
    z1_bar = floats(0.3, -0.6, -0.2, 0.5)
    lambda_now = 0.7
    measures = step_measures(y, z0_bar, z1_bar, z0, stats0, stats1, lambda_now, SETTINGS)
    # The gain's push and offset: the mean size and the signed mean of each bag's gradient.
    bag_means = z0_bar.detach().requires_grad_(True)
    term = lambda_now * soft_hinge(gain(bag_means, z1_bar, y, stats0, stats1), M, S).sum()
    (per_bag,) = torch.autograd.grad(term, bag_means)
    assert measures["gain_push"] == pytest.approx(float(per_bag.abs().mean()), abs=1e-15)
    assert measures["gain_offset"] == pytest.approx(float(per_bag.mean()), abs=1e-15)
    # CE's push: the mean |dCE/dz| of the x0 clips, each clip's CE counted in full.
    sizes = [
        torch.autograd.grad(ce_on_z(z0[bag], y[bag]), z0, retain_graph=True)[0][bag].abs() * 4
        for bag in range(4)
    ]
    assert measures["ce_push"] == pytest.approx(float(torch.cat(sizes).mean()), abs=1e-15)
    # The moments are those that the update reads.
    after = stats0.update(z0_bar)
    assert after.first == 0.9 * stats0.first + (1.0 - 0.9) * measures["z0_bar_mean"]
    assert after.second == 0.9 * stats0.second + (1.0 - 0.9) * measures["z0_bar_mean_square"]
    # The mean sizes of the bag means, and their quotient |z0|/|z1| for the step. The bag
    # means 0.3, 0.425, -0.05 and 0.7 differ in sign: their mean size is 0.36875, not 0.34375.
    assert measures["z0_bar_abs_mean"] == pytest.approx(float(z0_bar.detach().abs().mean()))
    assert measures["z0_bar_abs_mean"] == pytest.approx(0.36875)
    assert measures["z1_bar_abs_mean"] == pytest.approx(0.4)
    assert measures["z0_over_z1"] == measures["z0_bar_abs_mean"] / measures["z1_bar_abs_mean"]


def test_the_steps_z0_over_z1_is_null_when_every_bag_mean_of_x1_is_0() -> None:
    unit = RunningStats.of(0.0, 1.0, SETTINGS)
    measures = step([0.5, 2.0, 0.5, -1.0], [0.0, 0.0, 0.0, 0.0], [1, 1, 0, 0], unit, unit)
    assert measures["z1_bar_abs_mean"] == 0.0
    assert measures["z0_over_z1"] is None  # no ratio over 0, and no NaN in metrics.jsonl
    assert json.loads(json.dumps(measures, allow_nan=False))["z0_over_z1"] is None
    small = step([0.5, 2.0, 0.5, -1.0], [1e-320, 0.0, 0.0, 0.0], [1, 1, 0, 0], unit, unit)
    assert small["z0_over_z1"] is None  # a quotient too large for a float is not written


def test_b1_logs_the_gain_without_a_push() -> None:
    unit = RunningStats.of(0.0, 1.0, SETTINGS)
    measures = step([0.5, 2.0, 0.5, -1.0], [0.8, 0.8, 0.4, -0.8], [1, 1, 0, 0], unit, unit, 0.0)
    assert measures["gain_push"] == 0.0
    assert math.copysign(1.0, measures["gain_offset"]) == 1.0  # 0.0, not -0.0
    assert measures["active_negative_words_wrong"] == 1  # the gain itself is still measured


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"lambda_now": -0.1}, "λG"),
        ({"lambda_now": math.nan}, "λG"),
        ({"z0": floats(0.1, 0.2, 0.3, 0.4)}, "one row of clips per bag"),
        ({"z0": floats(0.1, 0.2, 0.3, 0.4).reshape(2, 2)}, "one row of clips per bag"),
        ({"z0": floats(0.1, math.nan, 0.3, 0.4).reshape(4, 1)}, "not finite"),
        ({"z0_bar": floats(0.1, math.inf, 0.3, 0.4)}, "not finite"),
        ({"y": [1, 0, 1, 3]}, "0 or 1"),
    ],
)
def test_the_measures_refuse_inputs_that_do_not_fit(change: dict, message: str) -> None:
    unit = RunningStats.of(0.0, 1.0, SETTINGS)
    inputs = {
        "y": [1, 0, 1, 0],
        "z0_bar": floats(0.1, 0.2, 0.3, 0.4),
        "z1_bar": floats(0.1, 0.2, 0.3, 0.4),
        "z0": floats(0.1, 0.2, 0.3, 0.4).reshape(4, 1),
        "stats0": unit,
        "stats1": unit,
        "lambda_now": 1.0,
        "settings": SETTINGS,
    } | change
    with pytest.raises(ValueError, match=message):
        step_measures(**inputs)
