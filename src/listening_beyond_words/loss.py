"""The loss of the trained arms: cross-entropy on z, the CPG gain term and its running statistics.

The loss is written per bag (pilot protocol, fixed settings "Loss" and "Gain term", and
section "Training", bullets "Arms and what trains", "The gain term" and "Loss and
schedules"): L_b = CE(x0) + 0.25 CE(x1) + 0.5 CE(x2) + λG · s·log(1 + exp((m − G_b)/s)).
Each CE is the mean over one bag's clips of one view, with class weights 1 (``ce_on_z``);
B0 keeps CE(x0) only, and B1 is CPG with λG = 0. The gain of a bag (``gain``) is
G_b = (2y − 1)·ẑ0 − max(0, (2y − 1)·ẑ1), with ẑ = (z̄ − μ)/σ for each view, where y is 1
for a PHQ-8-positive participant, z̄0 and z̄1 are the bag's mean z on ``x0`` and ``x1``,
and μ and σ are the running mean and spread of that view's bag means (``RunningStats``).
ẑ1, μ and σ pass no gradient: μ and σ are Python floats, and ẑ1 is detached, so only z̄0
is pushed. The soft hinge (``soft_hinge``) is active on a bag when G_b < m. The gain is
computed in at least fp32: fp16 inputs are cast to fp32 before any arithmetic, and
float64 inputs stay float64. λG rises from 0 over the first 20% of a run's steps
(``lambda_g``). A step's loss is the mean of its bags' losses; the training loop forms
it.

μ and σ are running averages of the bag means and of their squares, with momentum 0.9,
updated once per step from its bag means, for both views at the same point. They start
from the zero-shot bag means of the 32 start bags. σ is the root of the average square
minus the squared average, taken as 0 if rounding makes that difference negative, and it
never falls below a quarter of the start spread nor below 0.1.

``step_measures`` gives one step's logged counts and measures (bullet "Logged measures"):
aggregates over the step's bags only, never an ID.
"""

import dataclasses
import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Final

import numpy as np
import torch
import torch.nn.functional as F

from listening_beyond_words.config import GainSettings

# One step's logged measures, in the order of ``step_measures``: the bags and the active
# bags (G_b < m) of each class and words group, whether σ sits at its floor, μ and σ
# before the update, the mean and the mean square of the step's bag means and their mean
# size, from which the measured mean and spread of an epoch are formed, the step's
# |z0|/|z1|, and the pushes of CE and of the gain with the gain's offset part.
STEP_MEASURES: Final = (
    "bags_positive_words_right",
    "bags_positive_words_wrong",
    "bags_negative_words_right",
    "bags_negative_words_wrong",
    "active_positive_words_right",
    "active_positive_words_wrong",
    "active_negative_words_right",
    "active_negative_words_wrong",
    "mu0",
    "sigma0",
    "sigma0_at_floor",
    "mu1",
    "sigma1",
    "sigma1_at_floor",
    "z0_bar_mean",
    "z0_bar_mean_square",
    "z1_bar_mean",
    "z1_bar_mean_square",
    "z0_bar_abs_mean",
    "z1_bar_abs_mean",
    "z0_over_z1",
    "ce_push",
    "gain_push",
    "gain_offset",
)

# Bag means as the running statistics take them: a tensor (detached here), an array or a
# sequence of floats, of shape (B,).
BagMeans = torch.Tensor | np.ndarray | Sequence[float]


def _at_least_fp32(*tensors: torch.Tensor) -> torch.dtype:
    """The dtype of the loss's arithmetic: fp32, or float64 if an input is float64."""
    dtype = torch.float32
    for tensor in tensors:
        if not tensor.is_floating_point():
            raise ValueError(f"scores must be floating point, not {tensor.dtype}")
        dtype = torch.promote_types(dtype, tensor.dtype)
    return dtype


def _labels(y: Any, like: torch.Tensor) -> torch.Tensor:
    """The labels as 0.0 and 1.0 in the dtype and on the device of ``like``, one per entry."""
    if isinstance(y, torch.Tensor):
        labels = y.detach()
    elif isinstance(y, numbers.Integral) and not isinstance(y, bool):
        labels = torch.full(like.shape, int(y))  # one label for the whole bag
    else:
        labels = torch.as_tensor(np.asarray(y))
    if labels.shape != like.shape:
        raise ValueError(
            f"one label per entry is needed: labels of shape {tuple(labels.shape)}, entries of"
            f" shape {tuple(like.shape)}"
        )
    # Checked before the cast, so that no label rounds to 0 or 1 in fp32.
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("labels must be 0 or 1 (1 for PHQ-8-positive)")
    return labels.to(device=like.device, dtype=like.dtype)


def ce_on_z(z: torch.Tensor, y: int | torch.Tensor) -> torch.Tensor:
    """Return the cross-entropy of one bag and view: the answer's CE from z, class weights 1.

    z = logit("Yes") − logit("No"), so the softmax over the two answers gives "Yes" the
    probability sigmoid(z), and the CE of the two-answer softmax is the binary
    cross-entropy on z. Class weights are 1, because every step holds 2 positive and 2
    negative bags (pilot protocol, section "Training", bullet "Arms and what trains"). The
    CE is the mean over the bag's clips, joined after their calls, so each of 4 clips
    counts one quarter (bullet "Calls"). It is computed in at least fp32.

    Args:
        z: The scores of one bag's clips on one view, shape (B,).
        y: The bag's label (1 if PHQ-8-positive, 0 if not), or one label per clip,
            shape (B,).

    Returns:
        The mean CE, a scalar tensor in at least fp32, with the graph of ``z``.

    Raises:
        ValueError: If z is not floating point or not of shape (B,) with B of at least 1,
            or a label is not 0 or 1.
    """
    dtype = _at_least_fp32(z)
    if z.ndim != 1 or z.numel() == 0:
        raise ValueError(f"z must have shape (B,) with B of at least 1, not {tuple(z.shape)}")
    scores = z.to(dtype)
    return F.binary_cross_entropy_with_logits(scores, _labels(y, scores))


def soft_hinge(g: torch.Tensor, margin: float, softness: float) -> torch.Tensor:
    """Return the soft hinge s·log(1 + exp((m − G_b)/s)) of each bag.

    It is computed as s·logaddexp(0, (m − G_b)/s), which never overflows, and its gradient
    on G_b is −sigmoid((m − G_b)/s) at every value (pilot protocol, section "Training",
    bullet "The gain term"). It is never more than s·ln 2 above the hard hinge
    max(0, m − G_b), most at G_b = m.

    Args:
        g: The gains G_b, shape (B,).
        margin: m.
        softness: s.

    Returns:
        The soft hinge of each bag, shape (B,), in at least fp32.

    Raises:
        ValueError: If g is not floating point, or s is not a finite number above 0.
    """
    if not (math.isfinite(softness) and softness > 0):
        raise ValueError(f"the softness s must be a finite number above 0, not {softness!r}")
    values = g.to(_at_least_fp32(g))
    pull = (margin - values) / softness
    return softness * torch.logaddexp(torch.zeros_like(pull), pull)


def _moments(values: BagMeans) -> tuple[float, float]:
    """The mean and the mean square of bag means, in float64, without any graph."""
    if isinstance(values, torch.Tensor):
        array = values.detach().to(device="cpu", dtype=torch.float64).numpy()
    else:
        array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"bag means must have shape (B,) with B of at least 1, not {array.shape}")
    if not bool(np.isfinite(array).all()):
        raise ValueError("every bag mean must be finite")
    return float(array.mean()), float((array * array).mean())


@dataclass(frozen=True, slots=True)
class RunningStats:
    """The running mean μ and spread σ of one view's bag means, without gradient.

    Two running averages, of the bag means (``first``) and of their squares (``second``),
    each keep ``momentum`` of their old value at every update and add the rest of the new
    one (pilot protocol, section "Training", bullet "The gain term"). μ is the first;
    σ is the root of the second minus the first squared, taken as 0 if rounding makes
    that difference negative, because its root would be NaN, which no floor catches; σ
    then never falls below ``floor``. An update returns new statistics, so the gains of a
    step, computed with the statistics from before it, never change.

    Attributes:
        momentum: The share of the old value that an update keeps: 0.9.
        first: The running average of the bag means: μ.
        second: The running average of their squares.
        floor: The lowest σ: max(a quarter of the start spread, 0.1).
    """

    momentum: float
    first: float
    second: float
    floor: float

    def __post_init__(self) -> None:
        values = (self.momentum, self.first, self.second, self.floor)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"running statistics must be finite, not {values!r}")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(f"the momentum must lie from 0 to below 1, not {self.momentum!r}")
        if self.second < 0.0:
            raise ValueError(f"an average of squares cannot be negative: {self.second!r}")
        if self.floor <= 0.0:
            raise ValueError(f"the floor of σ must lie above 0, not {self.floor!r}")

    @classmethod
    def start(cls, bag_means: BagMeans, settings: GainSettings) -> "RunningStats":
        """Start the statistics of one view from the zero-shot bag means of the start bags.

        The two averages start at the mean and the mean square of the bag means; the start
        spread σ_zs is the root of their difference, computed as σ is (the population SD),
        and the floor is max(``floor_share`` · σ_zs, ``floor_logit``) (pilot protocol,
        section "Encoder caches and zero-shot scores", bullet "Start of the running
        statistics", and section "Training", bullet "The gain term").

        Args:
            bag_means: The zero-shot bag means of the 32 start bags on one view, shape (B,).
            settings: The gain's settings (``gain`` in ``train.yaml``).

        Returns:
            The statistics at the start of the run.

        Raises:
            ValueError: If there is no bag mean, or one is not finite.
        """
        mean, mean_square = _moments(bag_means)
        spread = math.sqrt(max(mean_square - mean * mean, 0.0))
        return cls(
            momentum=settings.momentum,
            first=mean,
            second=mean_square,
            floor=max(settings.floor_share * spread, settings.floor_logit),
        )

    @classmethod
    def of(cls, mu: float, sigma: float, settings: GainSettings) -> "RunningStats":
        """Return statistics with a given μ and σ, such as the timing run's stand-in values.

        The averages are μ and μ² + σ², and σ also serves as the start spread of the floor.

        Args:
            mu: μ.
            sigma: σ, above 0.
            settings: The gain's settings (``gain`` in ``train.yaml``).

        Returns:
            The statistics.

        Raises:
            ValueError: If μ is not finite or σ is not a finite number above 0.
        """
        if not (math.isfinite(mu) and math.isfinite(sigma) and sigma > 0):
            raise ValueError(
                f"μ must be finite and σ a finite number above 0, not {mu!r}, {sigma!r}"
            )
        return cls(
            momentum=settings.momentum,
            first=float(mu),
            second=float(mu) * float(mu) + float(sigma) * float(sigma),
            floor=max(settings.floor_share * float(sigma), settings.floor_logit),
        )

    @property
    def mu(self) -> float:
        """μ: the running mean of the bag means."""
        return self.first

    @property
    def raw_sigma(self) -> float:
        """σ before its floor: the root of the average square minus the squared average, or 0."""
        return math.sqrt(max(self.second - self.first * self.first, 0.0))

    @property
    def sigma(self) -> float:
        """σ: the running spread of the bag means, never below the floor."""
        return max(self.raw_sigma, self.floor)

    @property
    def at_floor(self) -> bool:
        """Whether σ sits at its floor."""
        return self.raw_sigma <= self.floor

    def update(self, step_bag_means: BagMeans) -> "RunningStats":
        """Return the statistics after one step, from the step's bag means.

        Each average keeps ``momentum`` of its value and adds the rest of the step's mean
        of the bag means, or of their squares. The training loop updates both views at the
        same point, once per step, after its bags, also on a skipped step, whose forward
        values are finite (pilot protocol, section "Training", bullet "The gain term"). The
        values are read without their graph.

        Args:
            step_bag_means: The bag means of one step on this view, shape (B,).

        Returns:
            New statistics; these stay as they are.

        Raises:
            ValueError: If there is no bag mean, or one is not finite.
        """
        mean, mean_square = _moments(step_bag_means)
        keep, add = self.momentum, 1.0 - self.momentum
        return dataclasses.replace(
            self,
            first=keep * self.first + add * mean,
            second=keep * self.second + add * mean_square,
        )


def _gain_parts(
    z0_bar: torch.Tensor,
    z1_bar: torch.Tensor,
    y: torch.Tensor | Sequence[int],
    stats0: RunningStats,
    stats1: RunningStats,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """2y − 1, ẑ1 and G_b of each bag, in the gain's dtype; only G_b has z̄0's graph."""
    if z0_bar.ndim != 1 or z0_bar.numel() == 0 or z1_bar.shape != z0_bar.shape:
        raise ValueError(
            "z0_bar and z1_bar must both have shape (B,) with B of at least 1, not"
            f" {tuple(z0_bar.shape)} and {tuple(z1_bar.shape)}"
        )
    dtype = _at_least_fp32(z0_bar, z1_bar)
    # Cast before any arithmetic, so that no fp16 operation rounds the gain.
    z0 = z0_bar.to(dtype)
    z1 = z1_bar.detach().to(dtype)
    t = 2.0 * _labels(y, z0) - 1.0
    z0_hat = (z0 - stats0.mu) / stats0.sigma
    z1_hat = (z1 - stats1.mu) / stats1.sigma
    return t, z1_hat, t * z0_hat - torch.clamp(t * z1_hat, min=0.0)


def gain(
    z0_bar: torch.Tensor,
    z1_bar: torch.Tensor,
    y: torch.Tensor | Sequence[int],
    stats0: RunningStats,
    stats1: RunningStats,
) -> torch.Tensor:
    """Return the gain G_b of each bag.

    G_b = (2y − 1)·ẑ0 − max(0, (2y − 1)·ẑ1), with ẑ0 = (z̄0 − μ0)/σ0 and
    ẑ1 = (z̄1 − μ1)/σ1 (pilot protocol, section "Training", bullet "The gain term"). The
    second part is the reference that ``x0`` must beat; the clamp sets it to 0 when ``x1``
    leans the wrong way. ẑ1, μ and σ pass no gradient, so the gradient of
    λG · ``soft_hinge``(G_b) on z̄0 is −λG·sigmoid((m − G_b)/s)·(2y − 1)/σ0, and z̄1 gets
    none. Computed in at least fp32: fp16 inputs are cast to fp32 first, and float64
    inputs stay float64.

    Args:
        z0_bar: Each bag's mean z on ``x0``, shape (B,).
        z1_bar: Each bag's mean z on ``x1``, shape (B,); its graph is not used.
        y: Each bag's label, 1 if PHQ-8-positive and 0 if not, shape (B,).
        stats0: The running statistics of ``x0``, from before the step.
        stats1: The running statistics of ``x1``, from before the step.

    Returns:
        G_b, shape (B,), in at least fp32, with the graph of ``z0_bar`` only.

    Raises:
        ValueError: If the bag means are not floating point or not both of shape (B,)
            with B of at least 1, or a label is not 0 or 1.
    """
    return _gain_parts(z0_bar, z1_bar, y, stats0, stats1)[2]


def share_steps(share: float, total_steps: int) -> int:
    """Return the steps that a share of a run covers, rounded up: ⌈share·T⌉.

    The share is taken as the decimal that the config writes, so that 0.2 of 100 steps is
    20, never 21 from the rounding of 0.2 in binary. It gives the λG ramp (0.2) and the
    warm-up of the learning rate (0.05): 51 and 13 of 252 steps (pilot protocol, section
    "Training", bullet "Loss and schedules").

    Args:
        share: The share, above 0 and at most 1.
        total_steps: T, the run's steps, at least 1.

    Returns:
        The number of steps.

    Raises:
        ValueError: If the share or T is outside its range.
    """
    if not (math.isfinite(share) and 0.0 < share <= 1.0):
        raise ValueError(f"a share of steps must lie above 0 and at most 1, not {share!r}")
    if isinstance(total_steps, bool) or total_steps < 1:
        raise ValueError(f"a run's steps must be a whole number of at least 1, not {total_steps!r}")
    return math.ceil(Fraction(repr(float(share))) * total_steps)


def lambda_g(step: int, total_steps: int, share: float, peak: float = 1.0) -> float:
    """Return λG, the weight of the gain term, at a step: peak · min(1, t / ⌈share·T⌉).

    λG rises in a straight line from 0 at step 0 and reaches its peak after the first
    ⌈0.2·T⌉ steps (pilot protocol, section "Training", bullet "Loss and schedules"). T is
    all steps of the run, so a 4th epoch or the 8-segment fallback changes the ramp; a
    step that the loss scaler skips still counts. The peak is the arm's ``lambda_g``: 1.0
    for CPG and 0.0 for B1, which computes the gain without training on it.

    Args:
        step: t, counted from 0.
        total_steps: T.
        share: The share of T over which λG rises (``gain.ramp_share``: 0.2).
        peak: λG after the ramp.

    Returns:
        λG at the step.

    Raises:
        ValueError: If the step is not a whole number from 0 to T − 1, the share or T is
            outside its range, or the peak is not a finite number of at least 0.
    """
    ramp = share_steps(share, total_steps)
    if isinstance(step, bool) or not 0 <= step < total_steps:
        raise ValueError(
            f"the step must be a whole number from 0 to {total_steps - 1}, not {step!r}"
        )
    if not (math.isfinite(peak) and peak >= 0.0):
        raise ValueError(f"the peak of λG must be a finite number of at least 0, not {peak!r}")
    return peak * min(1.0, step / ramp)


def step_measures(
    y: torch.Tensor | Sequence[int],
    z0_bar: torch.Tensor,
    z1_bar: torch.Tensor,
    z0: torch.Tensor,
    stats0: RunningStats,
    stats1: RunningStats,
    lambda_now: float,
    settings: GainSettings,
) -> dict[str, int | float | None]:
    """Return the logged counts and measures of one step of B1 or CPG (``STEP_MEASURES``).

    They follow the pilot protocol, section "Training", bullet "Logged measures", and hold
    no participant ID. G_b and ẑ1 are computed as in the loss (``gain``), from the same bag
    means and the statistics from before the step's update.

    - Counts, from which the shares are formed per epoch: the bags and the active bags
      (G_b < m) of each class and words group. The words are right when (2y − 1)·ẑ1 > 0,
      the clamp's own test, so ẑ1 = 0 counts as wrong, and in a words-wrong bag the clamp
      sets the reference to 0. Whether σ0 and σ1 sit at their floor (1 or 0).
    - μ and σ of each view, before the update.
    - The mean and the mean square of the step's bag means of each view, as the update
      reads them, and their mean size |z̄|, from which an epoch's measured mean and spread
      (population SD) of its bag means and its |z0|/|z1| are formed. The step's own
      |z0|/|z1|, the mean |z̄0| over the mean |z̄1|, is None (null in ``metrics.jsonl``)
      when the mean |z̄1| is 0, so that no step writes a NaN.
    - The pushes, the sizes of the gradients: CE's as the mean |sigmoid(z) − y| over the
      step's ``x0`` clips, the gain's as the mean λG·sigmoid((m − G_b)/s)/σ0 over its bags.
      The gain's offset part is the signed mean of its gradient on z̄0,
      −λG·sigmoid((m − G_b)/s)·(2y − 1)/σ0, over the bags: pushes of equal size on the two
      classes cancel in it, while an offset does not.

    Args:
        y: Each bag's label, shape (B,).
        z0_bar: Each bag's mean z on ``x0``, as in the loss, shape (B,).
        z1_bar: Each bag's mean z on ``x1``, as in the loss, shape (B,).
        z0: The z of each bag's ``x0`` clips, one row per bag, shape (B, S).
        stats0: The running statistics of ``x0`` that the step's gains used.
        stats1: The running statistics of ``x1`` that the step's gains used.
        lambda_now: λG at the step (``lambda_g``).
        settings: The gain's settings (``gain`` in ``train.yaml``).

    Returns:
        The values of ``STEP_MEASURES``: counts as whole numbers, the rest as finite floats,
        apart from ``z0_over_z1``, which is None when the mean |z̄1| is 0.

    Raises:
        ValueError: If the inputs' shapes do not fit, a label is not 0 or 1, a value is not
            finite, or λG is not a finite number of at least 0.
    """
    if not (math.isfinite(lambda_now) and lambda_now >= 0.0):
        raise ValueError(f"λG must be a finite number of at least 0, not {lambda_now!r}")
    with torch.no_grad():
        t, z1_hat, g = _gain_parts(z0_bar.detach(), z1_bar.detach(), y, stats0, stats1)
        if z0.ndim != 2 or z0.shape[0] != g.shape[0] or z0.shape[1] == 0:
            raise ValueError(
                f"z0 must hold one row of clips per bag, shape ({g.shape[0]}, S), not"
                f" {tuple(z0.shape)}"
            )
        clips = z0.detach().to(_at_least_fp32(z0))
        if not (bool(torch.isfinite(g).all()) and bool(torch.isfinite(clips).all())):
            raise ValueError("a bag mean or a clip's z is not finite")
        positive = t > 0
        right = t * z1_hat > 0
        active = g < settings.margin
        size = lambda_now * torch.sigmoid((settings.margin - g) / settings.softness) / stats0.sigma
        labels = ((t + 1.0) / 2.0).to(clips.dtype)
        ce_push = float((torch.sigmoid(clips) - labels[:, None]).abs().mean())
        gain_push = float(size.mean())
        gain_offset = float((-size * t).mean())
    cells: dict[str, torch.Tensor] = {}
    for name, members in (("positive", positive), ("negative", ~positive)):
        for words, group in (("words_right", right), ("words_wrong", ~right)):
            cells[f"{name}_{words}"] = members & group
    measures: dict[str, int | float | None] = {
        f"bags_{key}": int(cell.sum()) for key, cell in cells.items()
    }
    measures |= {f"active_{key}": int((cell & active).sum()) for key, cell in cells.items()}
    mean0, square0 = _moments(z0_bar)
    mean1, square1 = _moments(z1_bar)
    size0 = float(z0_bar.detach().abs().to(torch.float64).mean())
    size1 = float(z1_bar.detach().abs().to(torch.float64).mean())
    ratio = size0 / size1 if size1 else None
    if ratio is not None and not math.isfinite(ratio):
        ratio = None  # a mean |z̄1| too small for the quotient: no number rather than inf
    measures |= {
        "mu0": stats0.mu,
        "sigma0": stats0.sigma,
        "sigma0_at_floor": int(stats0.at_floor),
        "mu1": stats1.mu,
        "sigma1": stats1.sigma,
        "sigma1_at_floor": int(stats1.at_floor),
        "z0_bar_mean": mean0,
        "z0_bar_mean_square": square0,
        "z1_bar_mean": mean1,
        "z1_bar_mean_square": square1,
        "z0_bar_abs_mean": size0,
        "z1_bar_abs_mean": size1,
        "z0_over_z1": ratio,
        # Adding 0.0 turns the -0.0 of a zero push into 0.0.
        "ce_push": ce_push + 0.0,
        "gain_push": gain_push + 0.0,
        "gain_offset": gain_offset + 0.0,
    }
    return {key: measures[key] for key in STEP_MEASURES}
