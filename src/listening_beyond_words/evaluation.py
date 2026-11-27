"""The pilot's analysis: AUROCs of participant scores and their percentile intervals.

Pilot protocol, design "Main numbers" and fixed settings "Intervals", and
``docs/pilot_analysis.md``, sections "The main numbers" and "Intervals". A fold's AUROC is
the share of its pairs of a PHQ-8-positive and a PHQ-8-negative participant in which the
positive one scores higher, ties counting one half (``pair_wins``). In a bootstrap resample
a participant drawn k times counts k times, so each pair counts as often as the product of
the two draw counts (``weighted_aurocs``), which equals the AUROC of the repeated rows. An
interval's ends are NumPy's linear percentiles of the resampled values (``interval``).

The word check (``wordcheck``) computes its leakage AUROCs and the interval of its x0
control with this code too, so both steps share one implementation of each.
"""

import numpy as np


def pair_wins(positives: np.ndarray, negatives: np.ndarray) -> np.ndarray:
    """Return how each pair of a positive and a negative participant compares: 1, 1/2 or 0.

    Ties count one half (pilot protocol, check F4; pilot analysis, section "The main
    numbers"). The scores may be any values that NumPy compares element by element, exact
    fractions in an object array included; the wins 0, 0.5 and 1 are exact floats.

    Args:
        positives: The scores of the fold's positive participants, of shape (P,).
        negatives: The scores of its negative participants, of shape (N,).

    Returns:
        1.0 where the positive participant scores above the negative one, 0.5 where the
        two tie and 0.0 below, of shape (P, N).
    """
    above = (positives[:, None] > negatives[None, :]).astype(np.float64)
    return above + 0.5 * (positives[:, None] == negatives[None, :])


def weighted_aurocs(
    wins: np.ndarray, positive_counts: np.ndarray, negative_counts: np.ndarray
) -> np.ndarray:
    """Return a fold's AUROC in each resample, each participant counted as often as drawn.

    A pair of a positive and a negative participant counts as often as the product of
    their draw counts, so the AUROC equals that of the resample's repeated rows, ties one
    half. The counts are whole numbers and the wins 0, 0.5 or 1, so every sum is exact in
    floating point and only the final division rounds.

    Args:
        wins: The fold's pair wins (``pair_wins``), of shape (P, N).
        positive_counts: How often each positive participant is drawn in each resample, of
            shape (R, P), in the order of the rows of ``wins``.
        negative_counts: How often each negative participant is drawn in each resample, of
            shape (R, N), in the order of the columns of ``wins``.

    Returns:
        The AUROCs, of shape (R,).

    Raises:
        ValueError: If the shapes do not fit together, or a resample draws no positive or
            no negative participant of the fold.
    """
    weight_pos = np.asarray(positive_counts).astype(np.float64)
    weight_neg = np.asarray(negative_counts).astype(np.float64)
    if (
        weight_pos.ndim != 2
        or weight_neg.ndim != 2
        or len(weight_pos) != len(weight_neg)
        or wins.shape != (weight_pos.shape[1], weight_neg.shape[1])
    ):
        raise ValueError(
            f"draw counts of shapes {weight_pos.shape} and {weight_neg.shape} do not fit pair"
            f" wins of shape {wins.shape}"
        )
    pairs = weight_pos.sum(axis=1) * weight_neg.sum(axis=1)
    if not pairs.all():
        raise ValueError("a resample draws no positive or no negative participant of the fold")
    return ((weight_pos @ wins) * weight_neg).sum(axis=1) / pairs


def interval(values: np.ndarray, level: float) -> tuple[float, float]:
    """Return the percentile interval of resampled values, with NumPy's linear percentiles.

    Pilot protocol, section "Clip versions", bullet "The x0 control", and pilot analysis,
    section "Intervals": the ends are the (1 − level) / 2 and (1 + level) / 2 quantiles of
    the values, interpolated linearly between neighbouring order statistics. So the 80%
    interval runs from the 10th to the 90th percentile, and the 95% interval from the 2.5th
    to the 97.5th.

    Args:
        values: One value per resample, of shape (R,).
        level: The interval's coverage, such as 0.8 or 0.95.

    Returns:
        The lower and upper end.

    Raises:
        ValueError: If the level does not lie strictly between 0 and 1.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"the interval's level must lie between 0 and 1, got {level}")
    low, high = np.quantile(values, [(1.0 - level) / 2.0, (1.0 + level) / 2.0], method="linear")
    return float(low), float(high)
