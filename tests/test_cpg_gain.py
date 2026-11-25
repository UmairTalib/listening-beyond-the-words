"""The reference checks of the CPG gain, ported to PyTorch in float64 (loss.py).

The seven reference checks of the engineering conventions (section 10) run here with the
numbers that the reference computation printed for m = 0.3, s = 0.1 and λG = 1, from its
own random draws (``reference_draws``): check 4's mean pushes are those of its 4,000 bags.
Then the push on bags whose words are right, which is not 0, and the invariance of the gain
to an offset and a scale between the views: with exact statistics (the mean and spread of
all bags at once) and with running ones, which keep it only while both views follow the
same relation (pilot protocol, section "Training", bullets "The gain term" and "Logged
measures"). Inputs are float64, so the gain stays float64 and the checks hold to about
1e-12, where float64 rounds at about 1e-16 per operation.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from listening_beyond_words.config import GainSettings
from listening_beyond_words.loss import RunningStats, ce_on_z, gain, soft_hinge, step_measures

M, S, LAMBDA_G = 0.3, 0.1, 1.0
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
# z0 = k·z1 + c: no offset or scale, an offset with a scale above 1, and one below 1.
RELATIONS = [(1.0, 0.0), (3.0, -2.0), (0.3, 4.0)]


def sigmoid(x: float) -> float:
    return float(torch.sigmoid(torch.tensor(x, dtype=F64)))


def fixed(mu: float, sigma: float) -> RunningStats:
    """Statistics with a given μ and σ, far above the floor."""
    return RunningStats.of(mu, sigma, SETTINGS)


def as_tensor(values: Sequence[float] | np.ndarray, grad: bool = False) -> torch.Tensor:
    return torch.tensor(np.asarray(values, dtype=np.float64), dtype=F64, requires_grad=grad)


def gain_gradients(
    z0_bar: torch.Tensor,
    z1_bar: torch.Tensor,
    y: Sequence[int],
    stats0: RunningStats,
    stats1: RunningStats,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The gradients of the bags' summed gain terms, λG · soft hinge, on z̄0 and z̄1."""
    z0 = z0_bar.detach().clone().requires_grad_(True)
    z1 = z1_bar.detach().clone().requires_grad_(True)
    term = LAMBDA_G * soft_hinge(gain(z0, z1, y, stats0, stats1), M, S).sum()
    g0, g1 = torch.autograd.grad(term, (z0, z1), allow_unused=True, materialize_grads=True)
    return g0, g1


def pushes(
    z0_bar: torch.Tensor,
    z1_bar: torch.Tensor,
    y: Sequence[int],
    stats0: RunningStats,
    stats1: RunningStats,
) -> torch.Tensor:
    """Each bag's push, the size of its gradient on z̄0 in units of λG/σ0."""
    return gain_gradients(z0_bar, z1_bar, y, stats0, stats1)[0].abs() * stats0.sigma / LAMBDA_G


def signs(y: Sequence[int]) -> torch.Tensor:
    return 2.0 * as_tensor(y) - 1.0


@dataclass(frozen=True)
class ReferenceDraws:
    """The reference computation's draws from numpy.random.default_rng(0), in its order."""

    cases: list[tuple[int, float, float, tuple[float, float, float, float]]]  # check 2
    z1: np.ndarray  # check 4: 4,000 bags
    y: np.ndarray


@pytest.fixture(scope="module")
def reference_draws() -> ReferenceDraws:
    rng = np.random.default_rng(0)
    cases = []
    for _ in range(2000):
        y = int(rng.integers(0, 2))
        z0, z1 = rng.normal(0, 2, 2)
        stats = (rng.normal(0, 0.5), rng.uniform(0.3, 2), rng.normal(0, 0.5), rng.uniform(0.3, 2))
        cases.append((y, float(z0), float(z1), tuple(float(value) for value in stats)))
    z1_bags = rng.normal(0.4, 1.3, 4000)
    y_bags = rng.integers(0, 2, 4000)
    return ReferenceDraws(cases=cases, z1=z1_bags, y=y_bags)


# The seven reference checks --------------------------------------------------------------------


def test_check_1_the_softmax_of_the_two_answers_is_the_sigmoid_of_z() -> None:
    a, b = 2.3, -0.4  # logit("Yes"), logit("No")
    logits = torch.tensor([a, b], dtype=F64)
    p_yes = float(torch.softmax(logits, dim=0)[0])
    assert round(p_yes, 12) == 0.937026643943
    assert sigmoid(a - b) == pytest.approx(p_yes, abs=1e-15)
    z = (logits[0] - logits[1]).reshape(1)
    for y, answer in ((1, 0), (0, 1)):
        softmax_ce = -float(torch.log_softmax(logits, dim=0)[answer])
        assert float(ce_on_z(z, y)) == pytest.approx(softmax_ce, abs=1e-15)
    score = as_tensor([2.7], grad=True)
    (gradient,) = torch.autograd.grad(ce_on_z(score, 1), score)
    assert round(float(gradient), 6) == -0.062973
    assert float(gradient) == pytest.approx(sigmoid(2.7) - 1.0, abs=1e-15)


def test_check_2_the_gradient_on_z0_bar_is_the_formula_and_z1_bar_gets_none(
    reference_draws: ReferenceDraws,
) -> None:
    worst = 0.0
    for y, z0, z1, (mu0, sd0, mu1, sd1) in reference_draws.cases:
        stats0, stats1 = fixed(mu0, sd0), fixed(mu1, sd1)
        z0_bar, z1_bar = as_tensor([z0]), as_tensor([z1])
        g0, g1 = gain_gradients(z0_bar, z1_bar, [y], stats0, stats1)
        g = float(gain(z0_bar, z1_bar, [y], stats0, stats1))
        formula = -LAMBDA_G * sigmoid((M - g) / S) * (2 * y - 1) / stats0.sigma
        worst = max(worst, abs(float(g0) - formula))
        assert torch.equal(g1, torch.zeros_like(g1))
    assert len(reference_draws.cases) == 2000
    assert worst <= 1e-12  # the reference computation: 4.44e-16


def gain_without_stop_gradient(
    z0_bar: torch.Tensor, z1_bar: torch.Tensor, y: int, clamp: bool
) -> torch.Tensor:
    """The gain at unit statistics with ẑ1 left in the graph: the path that must not exist."""
    t = 2 * y - 1
    reference = torch.clamp(t * z1_bar, min=0.0) if clamp else t * z1_bar
    return t * z0_bar - reference


# y, z̄1, the words, and dL/dz̄1 without the stop-gradient: with the clamp and without it.
CHECK_3 = [
    (1, 0.8, "right", 1.000, 1.000),
    (1, -0.8, "wrong", 0.000, 0.001),
    (0, -0.8, "right", -1.000, -1.000),
    (0, 0.8, "wrong", -0.000, -0.047),
]


@pytest.mark.parametrize(("y", "z1", "words", "with_clamp", "without_clamp"), CHECK_3)
def test_check_3_without_the_stop_gradient_the_words_only_score_would_learn_the_wrong_label(
    y: int, z1: float, words: str, with_clamp: float, without_clamp: float
) -> None:
    t = 2 * y - 1
    assert (t * z1 > 0) == (words == "right")
    for clamp, expected in ((True, with_clamp), (False, without_clamp)):
        z0_bar, z1_bar = as_tensor([0.2]), as_tensor([z1], grad=True)
        term = LAMBDA_G * soft_hinge(gain_without_stop_gradient(z0_bar, z1_bar, y, clamp), M, S)
        (g1,) = torch.autograd.grad(term.sum(), z1_bar)
        assert round(float(g1), 3) == expected
        if abs(float(g1)) > 1e-12:
            # Descent moves z̄1 by −dL/dz̄1, away from the bag's label.
            assert float(g1) * t > 0
    # The implementation has no such path: z̄1 gets exactly 0 in all four cases.
    unit = fixed(0.0, 1.0)
    _, g1 = gain_gradients(as_tensor([0.2]), as_tensor([z1]), [y], unit, unit)
    assert torch.equal(g1, torch.zeros_like(g1))


@pytest.mark.parametrize(("k", "c"), RELATIONS)
def test_check_4_an_offset_and_a_scale_between_the_views_give_no_gain_where_the_words_are_right(
    reference_draws: ReferenceDraws, k: float, c: float
) -> None:
    z1 = as_tensor(reference_draws.z1)
    z0 = k * z1 + c
    y = [int(label) for label in reference_draws.y]
    # Exact statistics: the mean and the population SD of all 4,000 bag means.
    stats0, stats1 = RunningStats.start(z0, SETTINGS), RunningStats.start(z1, SETTINGS)
    g = gain(z0, z1, y, stats0, stats1)
    t = signs(y)
    right = t * (z1 - stats1.mu) / stats1.sigma > 0
    assert float(g[right].abs().max()) <= 1e-12  # the reference: 0.0, 4.4e-16 and 2.2e-15
    push = pushes(z0, z1, y, stats0, stats1)
    labels = torch.tensor(y)
    for label in (1, 0):
        mean_push = float(push[~right & (labels == label)].mean())
        assert round(mean_push, 3) == 0.996  # the reference, for both classes


@pytest.mark.parametrize(("k", "c"), RELATIONS)
def test_check_4_mirrored_bags_of_the_two_classes_get_the_same_push_where_the_words_are_wrong(
    k: float, c: float
) -> None:
    # A positive bag below the mean of x1 and a negative one as far above it: both wrong.
    rng = np.random.default_rng(4)
    a = np.abs(rng.normal(0.0, 1.3, 500)) + 0.01
    z1 = as_tensor(np.concatenate([0.4 - a, 0.4 + a]))
    z0 = k * z1 + c
    y = [1] * 500 + [0] * 500
    stats0, stats1 = RunningStats.start(z0, SETTINGS), RunningStats.start(z1, SETTINGS)
    z1_hat = (z1 - stats1.mu) / stats1.sigma
    assert bool((signs(y) * z1_hat < 0).all())
    # The push depends on |ẑ1| alone: G_b = −|ẑ1| on every words-wrong bag.
    g = gain(z0, z1, y, stats0, stats1)
    assert float((g + z1_hat.abs()).abs().max()) <= 1e-12
    push = pushes(z0, z1, y, stats0, stats1)
    # Equal within 1e-12, not bit for bit: the two classes round differently.
    assert float((push[:500] - push[500:]).abs().max()) <= 1e-12


def test_check_5_the_clamp_example() -> None:
    # y = 1, unit statistics: the words score −1.0, the real audio −0.7.
    unit = fixed(0.0, 1.0)
    z0_bar, z1_bar = as_tensor([-0.7]), as_tensor([-1.0])
    g = float(gain(z0_bar, z1_bar, [1], unit, unit))
    assert g == pytest.approx(-0.70, abs=1e-12)
    push = float(pushes(z0_bar, z1_bar, [1], unit, unit)[0])
    assert push == pytest.approx(sigmoid(10.0), abs=1e-12)
    assert round(push, 3) == 1.000
    # Without the clamp, the bag would meet the margin while its real audio is still wrong.
    unclamped = float(gain_without_stop_gradient(z0_bar, z1_bar, 1, clamp=False)[0])
    assert unclamped == pytest.approx(0.30, abs=1e-12)
    assert sigmoid((M - unclamped) / S) == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("softness", [0.1, 0.01])
def test_check_6_the_soft_hinge_stays_within_s_ln_2_of_the_hard_hinge(softness: float) -> None:
    g = torch.linspace(-2.0, 2.0, 4001, dtype=F64)
    gap = soft_hinge(g, M, softness) - torch.clamp(M - g, min=0.0)
    assert float(gap.max()) <= softness * math.log(2.0) + 1e-12
    assert float(gap.max()) == pytest.approx(softness * math.log(2.0), abs=1e-12)
    assert float(g[int(gap.argmax())]) == pytest.approx(M, abs=1e-9)  # largest at G_b = m
    assert float(gap.min()) >= -1e-15  # never below the hard hinge, beyond rounding


# z̄0 of a positive bag whose words-only reference is 0 (σ0 = 1, μ0 = 0), |dCE/dz| and the
# gain push, as the reference printed them.
CHECK_7 = [
    (0.0, 0.500, 0.953),
    (1.0, 0.269, 0.001),
    (2.0, 0.119, 0.000),
    (4.0, 0.018, 0.000),
    (8.0, 0.000, 0.000),
]


def test_check_7_the_ce_push_fades_gradually_while_the_gain_push_stays_until_the_margin() -> None:
    unit = fixed(0.0, 1.0)
    for z, ce_expected, gain_expected in CHECK_7:
        score = as_tensor([z], grad=True)
        (ce_gradient,) = torch.autograd.grad(ce_on_z(score, 1), score)
        assert round(abs(float(ce_gradient)), 3) == ce_expected
        assert abs(float(ce_gradient)) == pytest.approx(1.0 - sigmoid(z), abs=1e-15)
        push = float(pushes(as_tensor([z]), as_tensor([0.0]), [1], unit, unit)[0])
        assert round(push, 3) == gain_expected
        assert push == pytest.approx(sigmoid((M - z) / S), abs=1e-15)


# The push on bags whose words are right --------------------------------------------------------


def statistics(kind: str) -> tuple[RunningStats, RunningStats]:
    """Exact statistics of 4,000 bags, or running ones after a start and 30 steps."""
    rng = np.random.default_rng(7)
    if kind == "exact":
        z1 = rng.normal(0.4, 1.3, 4000)
        return RunningStats.start(3.0 * z1 - 2.0, SETTINGS), RunningStats.start(z1, SETTINGS)
    start = rng.normal(0.4, 1.3, 32)
    stats0 = RunningStats.start(0.7 * start + 1.0, SETTINGS)
    stats1 = RunningStats.start(start, SETTINGS)
    for _ in range(30):
        step = rng.normal(0.2, 1.1, 4)
        stats0, stats1 = stats0.update(0.5 * step + 1.5), stats1.update(step)
    return stats0, stats1


@pytest.mark.parametrize("kind", ["exact", "running"])
@pytest.mark.parametrize(
    ("offset", "expected", "rounded"),
    [(-0.3, sigmoid(3.0), 0.95), (0.0, 0.5, 0.50), (0.3, sigmoid(-3.0), 0.05)],
)
def test_the_push_on_a_bag_whose_words_are_right_is_not_0(
    kind: str, offset: float, expected: float, rounded: float
) -> None:
    # A bag whose words lean the right way by 0.8 σ1, with G_b at m − 0.3, m and m + 0.3.
    stats0, stats1 = statistics(kind)
    assert not stats0.at_floor
    assert not stats1.at_floor
    for y in (1, 0):
        t = 2 * y - 1
        z1_bar = as_tensor([stats1.mu + t * 0.8 * stats1.sigma])
        z0_bar = as_tensor([stats0.mu + t * (M + offset + 0.8) * stats0.sigma])
        assert float(gain(z0_bar, z1_bar, [y], stats0, stats1)) == pytest.approx(M + offset)
        push = float(pushes(z0_bar, z1_bar, [y], stats0, stats1)[0])
        assert push == pytest.approx(expected, abs=1e-12)
        assert round(push, 2) == rounded


# Invariance to an offset and a scale ---------------------------------------------------------


def mirrored(stats1: RunningStats, k: float, c: float) -> tuple[torch.Tensor, torch.Tensor, list]:
    """Words-wrong bags in mirrored pairs around μ1: positive below, negative as far above."""
    rng = np.random.default_rng(9)
    a = np.abs(rng.normal(0.0, 1.0, 200)) + 0.01
    z1 = as_tensor(np.concatenate([stats1.mu - a * stats1.sigma, stats1.mu + a * stats1.sigma]))
    return k * z1 + c, z1, [1] * 200 + [0] * 200


def running_statistics(
    start: tuple[float, float], steps: tuple[float, float]
) -> tuple[RunningStats, RunningStats]:
    """Statistics started from 32 bags with z0 = k·z1 + c, then 30 steps of another relation.

    Both views update at the same point, once per step, from the step's 4 bag means.
    """
    rng = np.random.default_rng(11)
    z1 = rng.normal(0.4, 1.3, 32)
    stats0 = RunningStats.start(start[0] * z1 + start[1], SETTINGS)
    stats1 = RunningStats.start(z1, SETTINGS)
    for _ in range(30):
        step = rng.normal(0.4, 1.3, 4)
        stats0, stats1 = stats0.update(steps[0] * step + steps[1]), stats1.update(step)
    return stats0, stats1


@pytest.mark.parametrize("constant", [-1.25, 3.5])
def test_a_constant_words_only_output_adds_no_class_dependent_push(constant: float) -> None:
    # Mirrored x0 bags: a positive bag above the mean of x0, a negative one as far below.
    rng = np.random.default_rng(13)
    a = np.abs(rng.normal(0.0, 1.0, 300)) + 0.01
    z0 = as_tensor(np.concatenate([0.5 + a, 0.5 - a]))
    y = [1] * 300 + [0] * 300
    z1 = as_tensor([constant] * 600)
    stats0, stats1 = RunningStats.start(z0, SETTINGS), RunningStats.start(z1, SETTINGS)
    assert stats1.at_floor  # x1 does not vary: σ1 sits at its floor and ẑ1 = 0
    g = gain(z0, z1, y, stats0, stats1)
    # The reference is 0 in every bag, so G_b = (2y − 1)·ẑ0, whatever the constant.
    assert torch.equal(g, signs(y) * (z0 - stats0.mu) / stats0.sigma)
    push = pushes(z0, z1, y, stats0, stats1)
    assert float((push[:300] - push[300:]).abs().max()) <= 1e-12
    other = as_tensor([constant + 1.0] * 600)
    assert torch.equal(push, pushes(z0, other, y, stats0, RunningStats.start(other, SETTINGS)))


@pytest.mark.parametrize(("k", "c"), RELATIONS)
def test_running_statistics_keep_the_invariance_while_both_views_follow_one_relation(
    k: float, c: float
) -> None:
    stats0, stats1 = running_statistics((k, c), (k, c))
    assert not stats0.at_floor
    assert not stats1.at_floor
    # Words-right bags: no gain.
    rng = np.random.default_rng(17)
    b = np.abs(rng.normal(0.0, 1.0, 200)) + 0.01
    z1 = as_tensor(np.concatenate([stats1.mu + b * stats1.sigma, stats1.mu - b * stats1.sigma]))
    y = [1] * 200 + [0] * 200
    assert float(gain(k * z1 + c, z1, y, stats0, stats1).abs().max()) <= 1e-12
    # Words-wrong bags: the same push for both classes, and no offset in their gradient.
    z0, z1, y = mirrored(stats1, k, c)
    push = pushes(z0, z1, y, stats0, stats1)
    assert float((push[:200] - push[200:]).abs().max()) <= 1e-12
    measures = step_measures(y, z0, z1, z0[:, None], stats0, stats1, LAMBDA_G, SETTINGS)
    assert abs(measures["gain_offset"]) <= 1e-12
    assert measures["gain_push"] > 0.9 / stats0.sigma


def test_a_start_from_another_relation_gives_a_class_dependent_push() -> None:
    # The statistics start from z0 = z1, while the bags follow z0 = 3·z1 − 2.
    stats0, stats1 = running_statistics((1.0, 0.0), (1.0, 0.0))
    z0, z1, y = mirrored(stats1, 3.0, -2.0)
    push = pushes(z0, z1, y, stats0, stats1)
    assert float((push[:200] - push[200:]).abs().max()) > 0.1
    measures = step_measures(y, z0, z1, z0[:, None], stats0, stats1, LAMBDA_G, SETTINGS)
    assert abs(measures["gain_offset"]) > 0.01


@pytest.mark.parametrize(("k", "c"), RELATIONS)
def test_the_unclamped_gain_is_0_on_every_bag_under_an_offset_and_a_scale(
    reference_draws: ReferenceDraws, k: float, c: float
) -> None:
    z1 = as_tensor(reference_draws.z1)
    z0 = k * z1 + c
    y = [int(label) for label in reference_draws.y]
    stats0, stats1 = RunningStats.start(z0, SETTINGS), RunningStats.start(z1, SETTINGS)
    t = signs(y)
    # The unclamped gain of the ablation, local to this test: no clamp on the reference.
    unclamped = t * (z0 - stats0.mu) / stats0.sigma - t * (z1 - stats1.mu) / stats1.sigma
    right = t * (z1 - stats1.mu) / stats1.sigma > 0
    assert bool(right.any())
    assert bool((~right).any())
    assert float(unclamped.abs().max()) <= 1e-12
