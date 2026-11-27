"""The pilot's analysis: exact fold AUROCs, the paired bootstrap, checks F2 and F3 and the rule.

Pilot protocol, design "Main numbers", fixed settings "Intervals", sections "Feasibility
checks", "Decision rule" and "If time runs short", and ``docs/pilot_analysis.md``, which
fills only the points that the protocol leaves open. The settings are in
``configs/pilot/evaluate.yaml`` (``EvaluateSettings``). This module holds:

- the check of the pinned labels against the pilot's sessions and their PHQ-8 totals
  (``checked_labels``);
- the kept runs: a run's record checked against its pin alone (``pinned_record``), the
  rules that let it count (``kept_run``), the earlier of two runs of one fold-run
  (``run_order``) and the versions that the pinned set may be (``VERSIONS``);
- the point estimates, exact: a participant's score is the mean z over all their segments
  of a view (``person_scores``), a fold's AUROC counts ties as one half (``fold_auroc``),
  and its Spearman correlation uses average ranks (``fold_spearman``); ``fold_runs``
  gathers each round's test folds with every arm's scores, and ``point_values`` gives
  every number in every fold-run, each AUROC and Spearman checked against scikit-learn and
  SciPy;
- the paired bootstrap: the draw counts within each PHQ-8 class (``draw_counts``,
  ``draws_sha256``), every number on each resample with each participant counted as often
  as drawn (``resampled_means``), percentile intervals (``interval``) and standard errors
  (``standard_error``);
- checks F2 and F3 and the decision rule with its framing line (``f2_reading``,
  ``f3_reading``, ``outcome``, ``framing``), and how a value near a bound is printed
  (``printed``).

A fold's AUROC is the share of its pairs of a PHQ-8-positive and a PHQ-8-negative
participant in which the positive one scores higher, ties counting one half
(``pair_wins``). In a bootstrap resample a participant drawn k times counts k times, so
each pair counts as often as the product of the two draw counts (``weighted_aurocs``),
which equals the AUROC of the repeated rows. An interval's ends are NumPy's linear
percentiles of the resampled values (``interval``). The word check (``wordcheck``)
computes its leakage AUROCs and the interval of its x0 control with these three too, so
both steps share one implementation of each.

Participant scores and fold AUROCs are exact fractions, so a value on a bound of a rule is
never a rounding artefact. Floating-point numbers are used only inside the bootstrap and
for the last step of a Spearman correlation, a square root. Nothing here prints, and a
refusal names rules, runs, folds and counts, never a participant's ID, score, label or
PHQ-8 total.
"""

import hashlib
import json
import math
import re
import warnings
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Context, Decimal
from fractions import Fraction
from numbers import Rational
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, NamedTuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from listening_beyond_words.labels import PHQ8_CUTOFF
from listening_beyond_words.partitions import N_FOLDS
from listening_beyond_words.runinfo import (
    RELEASE_TAGS,
    StepError,
    is_first_run,
    item_seed,
    newest_release,
)
from listening_beyond_words.sessions import PILOT_IDS

ZEROSHOT: Final = "zeroshot"
ARMS: Final = (ZEROSHOT, "b0", "b1", "cpg")  # pilot protocol, design "Arms"
TRAINED_ARMS: Final = ("b0", "b1", "cpg")
VIEWS: Final = ("x0", "rev")  # design "Test views": the two views that every arm scored
# First arm minus second, fold by fold: Δ = CPG − B0 (design "Main numbers"), then B1 − B0
# and CPG − B1 (design "Also reported").
DIFFERENCES: Final = (("cpg", "b0"), ("b1", "b0"), ("cpg", "b1"))
ROUNDS: Final = (1, 2)
BOOTSTRAP_KEYS: Final = ("evaluate", "bootstrap")  # the draws' stream: item_seed(seed, *keys)
TRAIN_STEP: Final = "train"  # the step that writes the training runs, scripts/train.py
# Where the training runs lie, relative to the workspace, and a training run's folder
# (engineering conventions, section 4); the zero-shot run's folder has no round and fold.
RUNS_FOLDER: Final = "runs/pilot"
RUN_NAME: Final = re.compile(r"pilot-(b0|b1|cpg)-r([0-9]+)-f([0-9]+)-[0-9]{8}-[0-9]{6}")
SHA256: Final = re.compile(r"[0-9a-f]{64}")  # a file's SHA-256 in lowercase hex, as pinned
# The point AUROCs and Spearman correlations must agree with scikit-learn and SciPy within
# this (pilot analysis, section "The main numbers", bullet "Exact values").
CROSS_CHECK: Final = 1e-12

# The kinds of number, each per view (``Number``).
AUROC: Final = "auroc"  # an arm's AUROC
SPEARMAN: Final = "spearman"  # an arm's Spearman correlation with the PHQ-8 totals
DIFFERENCE: Final = "difference"  # one arm's AUROC minus another's
AGREEMENT: Final = "agreement"  # the Spearman correlation of B0's and CPG's scores

# The outcomes of the decision rule and the protocol's decision for each, word for word
# (pilot protocol, section "Decision rule").
GREEN_BEATS: Final = "Green: beats B0"
GREEN_ROBUST: Final = "Green: robust without words"
AMBER: Final = "Amber: no visible effect"
RED: Final = "Red: worse on normal audio"
DECISIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        GREEN_BEATS: "Go",
        GREEN_ROBUST: "Go. The thesis leads with robustness.",
        AMBER: (
            "The researcher decides, using the diagnostics; the supervisor gets a one-line"
            ' report (decided with the supervisor until 2026-11-25, section "Training")'
        ),
        RED: "No-go for the recipe as it stands",
    }
)
# The framing line (pilot analysis, section "Decision rule", bullet "Where a gain comes
# from"), read only in "Green: beats B0".
FRAMING_EXTRA_CLIPS: Final = "framing: the gain comes mainly from the extra clips"
FRAMING_STAYS: Final = "framing: read; its condition does not hold, so the framing stays"
FRAMING_NOT_READ: Final = "framing: not read"
FRAMING_NO_B1: Final = "framing: not available (cut: no B1)"
# The view of the framing's two differences, B1 − B0 and CPG − B1: "both on x0", as Δ's
# reading of "Green: beats B0".
FRAMING_VIEW: Final = "x0"

_FILE: Final = "evaluate.yaml"
_TOP_KEYS: Final = (
    "arms",
    "views",
    "differences",
    "bootstrap",
    "checks",
    "rule",
    "framing",
    "first_runs_commit",
    "runs",
)
_PIN_KEYS: Final = ("run", "scores_sha256", "record_sha256")
_COMMIT: Final = re.compile(r"[0-9a-f]{40}")
_RUN: Final = re.compile(f"{RUNS_FOLDER}/{RUN_NAME.pattern}")
_ROUND_KEY: Final = re.compile(r"r([1-9][0-9]*)")
_FOLD_KEY: Final = re.compile(r"f([1-9][0-9]*)")
_UNROUNDED: Final = Context(prec=20)  # significant digits of a value printed unrounded


# The pairwise comparisons, shared with the word check ------------------------------------


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
    to the 97.5th. A value that is not finite stands for a number that is not defined in
    that resample, and such a number has no interval.

    Args:
        values: One value per resample, of shape (R,).
        level: The interval's coverage, such as 0.8 or 0.95.

    Returns:
        The lower and upper end.

    Raises:
        ValueError: If the level does not lie strictly between 0 and 1, there is no value,
            the values do not have one dimension, or a value is not finite.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"the interval's level must lie between 0 and 1, got {level}")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("an interval needs one value per resample, of shape (R,), and R > 0")
    if not np.isfinite(array).all():
        raise ValueError("a value is not finite: a number not defined in a resample has none")
    low, high = np.quantile(array, [(1.0 - level) / 2.0, (1.0 + level) / 2.0], method="linear")
    return float(low), float(high)


def standard_error(values: np.ndarray) -> float:
    """Return a number's bootstrap standard error: the SD of its resampled values, ddof 1.

    Pilot analysis, section "Intervals": each difference gets the standard deviation
    (ddof 1) of its 10,000 resampled values.

    Args:
        values: One value per resample, of shape (R,).

    Returns:
        The standard error.

    Raises:
        ValueError: If there are fewer than two values, they do not have one dimension, or
            a value is not finite.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError("a standard error needs one value per resample, of shape (R,), R > 1")
    if not np.isfinite(array).all():
        raise ValueError("a value is not finite: a number not defined in a resample has none")
    return float(np.std(array, ddof=1))


# Settings ----------------------------------------------------------------------------------


class Version(NamedTuple):
    """A set of training runs that the pilot may read (pilot protocol, "If time runs short").

    Attributes:
        name: How a cell that the version cannot fill names it: "not available (cut:
            <name>)" (pilot analysis, section "Runs and pins", bullet "Versions").
        runs: Its number of training runs.
        arm_rounds: The arms and rounds that it holds, each with all five fold-runs.
    """

    name: str
    runs: int
    arm_rounds: frozenset[tuple[str, int]]


# Cut in this order: B1's round 2, then B1, then round 2 of B0 and CPG.
VERSIONS: Final = (
    Version(
        "all 30 runs", 30, frozenset((arm, number) for arm in TRAINED_ARMS for number in ROUNDS)
    ),
    Version(
        "no B1 round 2", 25, frozenset({("b0", 1), ("b0", 2), ("b1", 1), ("cpg", 1), ("cpg", 2)})
    ),
    Version("no B1", 20, frozenset({("b0", 1), ("b0", 2), ("cpg", 1), ("cpg", 2)})),
    Version("B0 and CPG round 1 only", 10, frozenset({("b0", 1), ("cpg", 1)})),
)


@dataclass(frozen=True, slots=True)
class RunPin:
    """A kept training run as ``evaluate.yaml`` pins it (pilot analysis, "Runs and pins").

    Attributes:
        run: The run folder, relative to the workspace:
            ``runs/pilot/pilot-<arm>-r<round>-f<fold>-<YYYYmmdd-HHMMSS>``.
        scores_sha256: The SHA-256 of its ``scores.parquet``.
        record_sha256: The SHA-256 of its ``run.json``.
    """

    run: str
    scores_sha256: str
    record_sha256: str


def _block(value: Any, keys: Sequence[str], where: str) -> Mapping[str, Any]:
    """A config block, refused unless it is a mapping with exactly the expected keys."""
    if not isinstance(value, Mapping):
        raise StepError(f"{where} must be a mapping, not {type(value).__name__}")
    missing = sorted(set(keys) - set(value))
    unknown = sorted(str(key) for key in set(value) - set(keys))
    if missing or unknown:
        raise StepError(f"{where}: missing keys {missing}, unknown keys {unknown}")
    return value


def _whole(value: Any, minimum: int, where: str) -> int:
    """A whole number of at least ``minimum`` from the config; true and false are refused."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StepError(f"{where} must be a whole number of at least {minimum}, not {value!r}")
    return value


def _decimal(value: Any, where: str) -> Fraction:
    """A float of the config as the exact decimal that the file writes: 0.03 is 3/100.

    The float itself, 0.0299999999999999988897769753748..., lies just below 3/100, so a
    value between the two would reach it though not the bound written. The bounds of the
    checks and the rule are the decimals written, so they are read from the float's
    shortest text, as the training code reads the bounds of check F2.
    """
    if not isinstance(value, float) or not math.isfinite(value):
        raise StepError(f"{where} must be a finite float, such as 0.03, not {value!r}")
    return Fraction(repr(value))


def _run_pins(value: Any) -> Mapping[tuple[str, int, int], RunPin]:
    """The pins of the kept runs: arm, then r<round>, then f<fold>, then the pin itself."""
    where = f"{_FILE}: runs"
    if not isinstance(value, Mapping):
        raise StepError(f"{where} must be a mapping, not {type(value).__name__}")
    pins: dict[tuple[str, int, int], RunPin] = {}
    for arm, rounds in value.items():
        if arm not in TRAINED_ARMS:
            raise StepError(f"{where}: {arm!r} is not a trained arm ({', '.join(TRAINED_ARMS)})")
        if not isinstance(rounds, Mapping) or not rounds:
            raise StepError(f"{where}.{arm} must map r<round> to its folds")
        for round_key, folds in rounds.items():
            found = _ROUND_KEY.fullmatch(round_key) if isinstance(round_key, str) else None
            round_number = int(found.group(1)) if found else 0
            if round_number not in ROUNDS:
                raise StepError(f"{where}.{arm}: the rounds are r1 and r2, not {round_key!r}")
            if not isinstance(folds, Mapping) or not folds:
                raise StepError(f"{where}.{arm}.{round_key} must map f<fold> to a pin")
            for fold_key, pin in folds.items():
                found = _FOLD_KEY.fullmatch(fold_key) if isinstance(fold_key, str) else None
                fold = int(found.group(1)) if found else 0
                if not 1 <= fold <= N_FOLDS:
                    raise StepError(
                        f"{where}.{arm}.{round_key}: the folds are f1 to f{N_FOLDS}, not"
                        f" {fold_key!r}"
                    )
                place = f"{where}.{arm}.{round_key}.{fold_key}"
                block = _block(pin, _PIN_KEYS, place)
                run = block["run"]
                named = _RUN.fullmatch(run) if isinstance(run, str) else None
                if named is None or (
                    named.group(1),
                    int(named.group(2)),
                    int(named.group(3)),
                ) != (arm, round_number, fold):
                    raise StepError(
                        f"{place}: run must be the folder runs/pilot/pilot-{arm}-r{round_number}"
                        f"-f{fold}-<YYYYmmdd-HHMMSS> of this fold-run, not {run!r}"
                    )
                for key in ("scores_sha256", "record_sha256"):
                    if not isinstance(block[key], str) or not SHA256.fullmatch(block[key]):
                        raise StepError(f"{place}: {key} must be a SHA-256 in lowercase hex")
                pins[(arm, round_number, fold)] = RunPin(
                    run, block["scores_sha256"], block["record_sha256"]
                )
    return MappingProxyType(pins)


def pinned_version(runs: Collection[tuple[str, int, int]]) -> Version | None:
    """Return the version that a set of pinned fold-runs makes, or None if it is empty.

    Pilot protocol, section "If time runs short", and pilot analysis, section "Runs and
    pins", bullet "Versions": the set must be all 30 runs, 25 (no B1 round 2), 20 (no B1)
    or 10 (B0 and CPG, round 1 only), each arm and round with all five folds.

    Args:
        runs: The pinned fold-runs, as (arm, round, fold).

    Returns:
        The version, or None before any run is pinned.

    Raises:
        StepError: If the set is none of the versions.
    """
    if not runs:
        return None
    arm_rounds = {(arm, number) for arm, number, _ in runs}
    whole = all(
        {fold for arm, number, fold in runs if (arm, number) == pair} == set(range(1, N_FOLDS + 1))
        for pair in arm_rounds
    )
    for version in VERSIONS:
        if whole and len(set(runs)) == len(runs) and arm_rounds == version.arm_rounds:
            return version
    raise StepError(
        f"the {len(runs)} pinned runs are none of the pilot's versions: all 30 runs, 25 (no"
        " B1 round 2), 20 (no B1) or 10 (B0 and CPG, round 1 only), each arm and round with"
        f' all {N_FOLDS} folds (pilot protocol, section "If time runs short")'
    )


@dataclass(frozen=True, slots=True)
class EvaluateSettings:
    """The settings of the pilot's analysis, from ``configs/pilot/evaluate.yaml``.

    Each bound is the exact decimal that the file writes (0.03 is 3/100), so that the
    checks and the rule compare exactly (pilot analysis, section "Decision rule").

    Attributes:
        arms: The arms, zero-shot first: ``ARMS``.
        views: The test views: ``VIEWS``.
        differences: The differences of mean per-fold AUROC, first arm minus second, Δ
            first: ``DIFFERENCES``.
        resamples: Resamples of the bootstrap.
        seed: The run seed of the draws' own stream (``rng``).
        levels: The coverages of the percentile intervals: 0.8 and 0.95.
        f3_min_auroc: Check F3 passes if B0's mean AUROC on x0 is at least this.
        green_at_least: "Green: beats B0" if Δ on x0 is at least this; "Green: robust
            without words" if Δ on x0 lies strictly between the two bounds and Δ on rev is
            at least this.
        red_at_most: "Red: worse on normal audio" if Δ on x0 is at most this; "Amber: no
            visible effect" if Δ on x0 lies strictly between the two bounds and Δ on rev is
            below ``green_at_least``.
        framing_view: The view on which the framing line reads its two differences:
            ``FRAMING_VIEW``, x0.
        cpg_gain_at_least: CPG − B0 reaches this where the framing line is read: the bound
            of "Green: beats B0".
        b1_gain_at_least: The gain comes mainly from the extra clips if B1 − B0 reaches
            this ...
        cpg_over_b1_below: ... and CPG − B1 stays below this.
        first_runs_commit: The commit of B0's round 1 and of CPG's round 1, fold 1, which
            ran before the first release tag.
        runs: The kept training runs by (arm, round, fold); empty until they are pinned
            after training.
    """

    arms: tuple[str, ...]
    views: tuple[str, ...]
    differences: tuple[tuple[str, str], ...]
    resamples: int
    seed: int
    levels: tuple[float, ...]
    f3_min_auroc: Fraction
    green_at_least: Fraction
    red_at_most: Fraction
    framing_view: str
    cpg_gain_at_least: Fraction
    b1_gain_at_least: Fraction
    cpg_over_b1_below: Fraction
    first_runs_commit: str
    runs: Mapping[tuple[str, int, int], RunPin]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "EvaluateSettings":
        """Read the settings and check every key and value.

        The arms, views and differences are the ones that this code reads, in their order;
        the pinned runs must form one of the pilot's versions (``pinned_version``).

        Args:
            config: The content of ``configs/pilot/evaluate.yaml``.

        Returns:
            The settings.

        Raises:
            StepError: If a key is missing or unknown, a value has the wrong type or lies
                out of range, the red bound does not lie below the green one, the framing
                is not read on x0 at the bound of "Green: beats B0", or a pin or the pinned
                set is refused.
        """
        top = _block(config, _TOP_KEYS, _FILE)
        expected: dict[str, Any] = {
            "arms": list(ARMS),
            "views": list(VIEWS),
            "differences": [list(pair) for pair in DIFFERENCES],
        }
        for key, value in expected.items():
            if top[key] != value:
                raise StepError(
                    f"{_FILE}: {key} must be {value}, as the analysis reads them (pilot"
                    f' analysis, section "The main numbers"), not {top[key]!r}'
                )
        bootstrap = _block(top["bootstrap"], ("resamples", "seed", "levels"), f"{_FILE}: bootstrap")
        levels = bootstrap["levels"]
        if (
            not isinstance(levels, list)
            or not levels
            or any(not isinstance(level, float) or not 0.0 < level < 1.0 for level in levels)
            or len(set(levels)) != len(levels)
        ):
            raise StepError(
                f"{_FILE}: bootstrap.levels must be different floats between 0 and 1, not"
                f" {levels!r}"
            )
        checks = _block(top["checks"], ("f3_min_auroc",), f"{_FILE}: checks")
        f3_min_auroc = _decimal(checks["f3_min_auroc"], f"{_FILE}: checks.f3_min_auroc")
        if not 0 <= f3_min_auroc <= 1:
            raise StepError(f"{_FILE}: checks.f3_min_auroc must lie from 0 to 1, as an AUROC")
        rule = _block(top["rule"], ("green_at_least", "red_at_most"), f"{_FILE}: rule")
        green = _decimal(rule["green_at_least"], f"{_FILE}: rule.green_at_least")
        red = _decimal(rule["red_at_most"], f"{_FILE}: rule.red_at_most")
        if not red < green:
            raise StepError(f"{_FILE}: rule.red_at_most must lie below rule.green_at_least")
        framing = _block(
            top["framing"],
            ("view", "cpg_gain_at_least", "b1_gain_at_least", "cpg_over_b1_below"),
            f"{_FILE}: framing",
        )
        if framing["view"] != FRAMING_VIEW:
            raise StepError(
                f"{_FILE}: framing.view must be {FRAMING_VIEW}, where the framing line reads"
                ' both differences (pilot analysis, section "Decision rule"), not'
                f" {framing['view']!r}"
            )
        cpg_gain = _decimal(framing["cpg_gain_at_least"], f"{_FILE}: framing.cpg_gain_at_least")
        if cpg_gain != green:
            raise StepError(
                f"{_FILE}: framing.cpg_gain_at_least must equal rule.green_at_least: the"
                ' framing line is read only in "Green: beats B0" (pilot analysis, section'
                ' "Decision rule")'
            )
        commit = top["first_runs_commit"]
        if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
            raise StepError(
                f"{_FILE}: first_runs_commit must be a full commit hash, not {commit!r}"
            )
        runs = _run_pins(top["runs"])
        pinned_version(tuple(runs))
        return cls(
            arms=ARMS,
            views=VIEWS,
            differences=DIFFERENCES,
            resamples=_whole(bootstrap["resamples"], 2, f"{_FILE}: bootstrap.resamples"),
            seed=_whole(bootstrap["seed"], 0, f"{_FILE}: bootstrap.seed"),
            levels=tuple(levels),
            f3_min_auroc=f3_min_auroc,
            green_at_least=green,
            red_at_most=red,
            framing_view=str(framing["view"]),
            cpg_gain_at_least=cpg_gain,
            b1_gain_at_least=_decimal(
                framing["b1_gain_at_least"], f"{_FILE}: framing.b1_gain_at_least"
            ),
            cpg_over_b1_below=_decimal(
                framing["cpg_over_b1_below"], f"{_FILE}: framing.cpg_over_b1_below"
            ),
            first_runs_commit=commit,
            runs=runs,
        )

    @property
    def delta(self) -> tuple[str, str]:
        """Δ's two arms, first minus second: CPG and B0."""
        return self.differences[0]

    @property
    def agreement(self) -> tuple[str, str]:
        """The two arms whose scores' agreement is reported: B0 and CPG, Δ's arms."""
        return self.delta[1], self.delta[0]

    @property
    def version(self) -> Version | None:
        """The version of the pinned runs (``pinned_version``); None before the pins."""
        return pinned_version(tuple(self.runs))

    @property
    def stream(self) -> str:
        """The draws' stream as run.json names it."""
        keys = ", ".join(f'"{key}"' for key in BOOTSTRAP_KEYS)
        return f"runinfo.item_seed({self.seed}, {keys})"

    def rng(self) -> np.random.Generator:
        """The draws' own stream: ``item_seed(seed, "evaluate", "bootstrap")``.

        Pilot analysis, section "Intervals", bullet "Draws": seed 109, which no other step
        uses, through a stream of its own.
        """
        return np.random.default_rng(item_seed(self.seed, *BOOTSTRAP_KEYS))


# The labels --------------------------------------------------------------------------------


def checked_labels(table: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Check the pinned labels against the pilot's sessions and their PHQ-8 totals.

    Every participant must be a pilot session, given once, and every label must be 1
    exactly where the PHQ-8 total is 10 or more (pilot protocol, section "Speech used",
    bullet "Labels"; pilot analysis, section "Data protection"). The jobs that read the
    labels check them so before anything else. A refusal gives counts, never a
    participant's ID, label or total.

    Args:
        table: The pinned labels with their totals (``clips.load_pinned_labels`` with
            ``phq8_total=True``), indexed by session ID, with the columns ``label`` and
            ``phq8_total``.

    Returns:
        Each participant's label and PHQ-8 total, as whole numbers, by session ID in
        ascending order.

    Raises:
        StepError: If a participant lies outside the pilot's sessions or is given twice, a
            label or total is not stored as a whole number, a label is not 0 or 1, or a
            label disagrees with its total.
    """
    sessions = [int(session) for session in table.index]
    outside = len(set(sessions) - set(PILOT_IDS))
    if outside or len(set(sessions)) != len(sessions):
        raise StepError(
            f"the pinned labels hold {outside} participants outside the pilot's sessions, or a"
            " participant twice"
        )
    if not all(pd.api.types.is_integer_dtype(table[name]) for name in ("label", "phq8_total")):
        raise StepError(
            "the pinned labels and PHQ-8 totals are not stored as whole numbers, as the labels"
            " step writes them"
        )
    label = table["label"].to_numpy(dtype=np.int64)
    total = table["phq8_total"].to_numpy(dtype=np.int64)
    bad = int((~np.isin(label, (0, 1))).sum())
    if bad:
        raise StepError(f"{bad} labels are not 0 or 1")
    disagree = int((label != (total >= PHQ8_CUTOFF).astype(np.int64)).sum())
    if disagree:
        raise StepError(
            f"{disagree} labels disagree with their PHQ-8 totals: a participant is positive"
            f" exactly when the total is {PHQ8_CUTOFF} or more (pilot analysis, section"
            ' "Data protection")'
        )
    index = pd.Index(sessions, dtype=np.int64, name="session")
    labels = pd.Series(label, index=index, name="label").sort_index()
    totals = pd.Series(total, index=index, name="phq8_total").sort_index()
    return labels, totals


# Kept runs ---------------------------------------------------------------------------------


def pinned_record_data(root: Path, pin: RunPin) -> bytes:
    """Read a kept run's run.json as bytes, checked against its pin, and nothing else of the run.

    Pilot analysis, section "Runs and pins", bullet "Pins": the record must have the SHA-256
    that ``evaluate.yaml`` pins (``record_sha256``).

    Args:
        root: The workspace root, which the pinned run folder is relative to.
        pin: The run's pin.

    Returns:
        The run.json's bytes.

    Raises:
        StepError: If the run.json is missing or does not match its pin. The message names
            the rule and the run.
    """
    path = root / pin.run / "run.json"
    data = path.read_bytes() if path.is_file() else None
    if data is None or hashlib.sha256(data).hexdigest() != pin.record_sha256:
        raise StepError(f"{pin.run}: its run.json does not match its pin (record_sha256)")
    return data


def record_object(data: bytes, run: str) -> dict[str, Any]:
    """Read a run.json's bytes as a JSON object.

    Args:
        data: The run.json's bytes (``pinned_record_data``).
        run: The run folder, relative to the workspace, for the messages.

    Returns:
        The record.

    Raises:
        StepError: If the bytes cannot be read as JSON or are not a JSON object.
    """
    try:
        record = json.loads(data.decode("utf-8"))
    except ValueError:
        raise StepError(f"{run}: its run.json cannot be read as JSON") from None
    if not isinstance(record, dict):
        raise StepError(f"{run}: its run.json is not a JSON object")
    return record


def pinned_record(root: Path, pin: RunPin) -> dict[str, Any]:
    """Read a kept run's run.json, checked against its pin, and nothing else of the run.

    Pilot analysis, section "Runs and pins", bullet "Pins": the record must have the SHA-256
    that ``evaluate.yaml`` pins (``record_sha256``) and be a JSON object
    (``pinned_record_data``, then ``record_object``). This is the check of the record
    alone: no scores file is opened, so a step that reads only the records, such as the
    gain-term summary (pilot diagnostics, section "Gain-term summary", bullet "Records"),
    uses it with ``kept_run``. The analysis job checks the scores against their pin between
    the two steps, as it always has.

    Args:
        root: The workspace root, which the pinned run folder is relative to.
        pin: The run's pin.

    Returns:
        The record, a JSON object.

    Raises:
        StepError: If the run.json is missing or does not match its pin, or is not a JSON
            object. The message names the rule and the run.
    """
    return record_object(pinned_record_data(root, pin), pin.run)


def _nested(record: Any, *keys: str) -> Any:
    """The value under ``keys`` in nested JSON objects, or None where a level is missing."""
    for key in keys:
        if not isinstance(record, Mapping):
            return None
        record = record.get(key)
    return record


def _as_json(value: Any) -> str:
    """A value as canonical JSON text, so that a pin read from YAML compares with a record."""
    return json.dumps(value, sort_keys=True, default=str)


def kept_run(
    record: Mapping[str, Any],
    run: str,
    arm: str,
    round_number: int,
    fold: int,
    pins: Mapping[str, Any],
    first_runs_commit: str,
) -> str:
    """Refuse a training run that does not count in the analysis; return what lets it count.

    Pilot analysis, section "Runs and pins", bullet "Kept runs": a run counts if its record
    says done, it was not started with --allow-dirty (from a clean checkout), it passed
    check F1, it used the labels and partitions that ``data.yaml`` pins now, and it came
    from the first runs' commit (only B0's round 1 and CPG's round 1, fold 1:
    ``runinfo.FIRST_RUNS``) or from the newest ``v0.1.x`` tag of its clone, as its record
    shows.
    The timing run never counts, and the record must be the training job's own record of
    the fold-run that it is pinned for. A record that does not say that F1 passed, or that
    lacks a key, is refused, never read as a fail.

    Args:
        record: The run's ``run.json``, as read.
        run: Its folder, relative to the workspace, as pinned.
        arm: The arm of the fold-run that it is pinned for.
        round_number: Its round.
        fold: Its fold.
        pins: The data config, whose ``labels`` and ``partitions`` pins the run must have
            used.
        first_runs_commit: The commit of the first runs (``first_runs_commit`` in
            ``evaluate.yaml``).

    Returns:
        What lets the run count: the first runs' commit or the newest release tag.

    Raises:
        StepError: If the run does not count. The message names the rule and the run,
            never a score.
    """
    if not isinstance(record, Mapping):
        raise StepError(f"{run}: its run.json is not a JSON object")
    if record.get("step") != TRAIN_STEP:
        raise StepError(f"{run} is not a run of scripts/train.py")
    if record.get("timing") is not False:
        raise StepError(
            f"{run}: the timing run of check F5 never counts, nor a record that is silent"
        )
    if record.get("run_id") != PurePosixPath(run).name:
        raise StepError(f"{run}: its run.json names another run")
    if (record.get("arm"), record.get("round"), record.get("fold")) != (arm, round_number, fold):
        raise StepError(f"{run} is not a run of {arm}, round {round_number}, fold {fold}")
    if record.get("status") != "done":
        raise StepError(f"{run} did not finish: its run.json does not say done")
    git = record.get("git")
    if (
        record.get("allow_dirty") is not False
        or not isinstance(git, Mapping)
        or git.get("dirty") is not False
    ):
        raise StepError(
            f"{run} was started with --allow-dirty or from a checkout with uncommitted changes,"
            " or its run.json does not say: such a run never counts"
        )
    passes = _nested(record, "results", "rules", "f1", "passes")
    if passes is False:
        raise StepError(f"{run} fails check F1: a failing run cannot be pinned")
    if passes is not True:
        raise StepError(f"{run}: its run.json does not say that check F1 passed")
    for name in ("labels", "partitions"):
        current = pins.get(name)
        if not isinstance(current, Mapping) or not current.get("run") or not current.get("sha256"):
            raise StepError(f"data.yaml pins no {name} ({name}): no run can be kept")
        if _as_json(_nested(record, "inputs", name)) != _as_json(current):
            raise StepError(f"{run} was made from other {name} than data.yaml pins now")
    commit = git.get("commit")
    first = is_first_run(arm, round_number, fold)
    if first and commit == first_runs_commit:
        return f"the first runs' commit {first_runs_commit[:7]}"
    released = [str(tag) for tag in git.get("release_tags") or []]
    if not released:
        raise StepError(
            f"{run} comes from neither the first runs' commit {first_runs_commit[:7]} (B0's"
            f" round 1 and CPG's round 1, fold 1 only) nor a {RELEASE_TAGS} tag"
        )
    newest = newest_release(released)
    if newest not in {str(tag) for tag in git.get("tags") or []}:
        raise StepError(
            f"{run} does not come from the newest release tag of its clone, {newest}"
            + (" (nor is it one of the first runs at their commit)" if first else "")
        )
    return f"the newest release tag of its clone, {newest}"


def run_order(record: Mapping[str, Any], run: str) -> tuple[str, str]:
    """Return the key that orders the done runs of one fold-run: the smallest is the earliest.

    Pilot analysis, section "Runs and pins", bullet "Kept runs": if two done runs share a
    fold-run, the earlier one counts and the other is reported. The earlier run is the one
    that finished first, by the UTC time of its record (``finished_utc``, in one ISO 8601
    form, so that its text sorts as the time does), and of two that finished in the same
    second the one whose folder's name comes first, as the training job orders the runs of
    CPG's round 1, fold 1. Both the listing of the runs and the analysis job read it.

    Args:
        record: The run's ``run.json``, as read.
        run: Its folder, relative to the workspace.

    Returns:
        ``(finished_utc, run)``.

    Raises:
        StepError: If the record does not say when the run finished: such a run cannot be
            ordered, so it is never taken for the earlier or the later one.
    """
    finished = record.get("finished_utc")
    if not isinstance(finished, str) or not finished:
        raise StepError(
            f"{run}: its run.json does not say when it finished (finished_utc), so it cannot"
            " be told whether it is the earlier of two runs of its fold-run"
        )
    return finished, run


# Point estimates ---------------------------------------------------------------------------


def person_scores(table: pd.DataFrame, view: str) -> pd.Series:
    """Return each participant's exact score on a view: the mean z over all their segments.

    Pilot protocol, fixed settings "Score", and pilot analysis, section "The main numbers":
    a participant's score is the mean z over all their segments of the view, in one run's
    scores. Each z is a float, which is an exact fraction, and the mean is taken without
    rounding, so participants whose scores are equal tie exactly and no order between two
    participants comes from rounding.

    Args:
        table: A run's scores, one row per segment and view, with the columns
            ``segment``, ``session``, ``view`` and ``z`` (``zeroshot.SCORE_COLUMNS``).
        view: The view whose scores are averaged.

    Returns:
        Each participant's score as a ``Fraction``, indexed by session ID in ascending
        order.

    Raises:
        StepError: If the table lacks a column, holds no segment of the view or one twice,
            or a z of the view is not finite.
    """
    missing = sorted({"segment", "session", "view", "z"} - set(table.columns))
    if missing:
        raise StepError(f"the scores lack the columns {missing}")
    rows = table.loc[table["view"] == view]
    if rows.empty:
        raise StepError(f"the scores hold no segment of {view}")
    twice = int(rows["segment"].duplicated().sum())
    if twice:
        raise StepError(f"the scores hold {twice} segments of {view} twice")
    z = rows["z"].to_numpy(dtype=np.float64)
    non_finite = int((~np.isfinite(z)).sum())
    if non_finite:
        raise StepError(f"{non_finite} scores of {view} are not finite")
    sums: dict[int, Fraction] = {}
    counts: dict[int, int] = {}
    for session, value in zip(
        rows["session"].to_numpy(dtype=np.int64).tolist(), z.tolist(), strict=True
    ):
        sums[session] = sums.get(session, Fraction(0)) + Fraction(value)
        counts[session] = counts.get(session, 0) + 1
    sessions = sorted(sums)
    return pd.Series(
        [sums[session] / counts[session] for session in sessions],
        index=pd.Index(sessions, dtype=np.int64, name="session"),
        dtype=object,
        name="score",
    )


def _exact_values(values: Sequence[Any], what: str) -> None:
    """Refuse values that are neither fractions, whole numbers nor finite floats."""
    for value in values:
        exact = isinstance(value, Rational) and not isinstance(value, bool)
        if not exact and not (isinstance(value, float) and math.isfinite(value)):
            raise ValueError(f"{what}: a value is neither a fraction nor a finite float")


def _aligned(first: pd.Series, second: pd.Series, what: str) -> tuple[list[Any], list[Any]]:
    """Two series of the same participants as lists, in ascending order of session ID."""
    if (
        first.index.has_duplicates
        or second.index.has_duplicates
        or set(first.index) != set(second.index)
    ):
        raise ValueError(f"{what}: the two must name the same participants, once each")
    order = sorted(first.index)
    values = first.reindex(order).tolist(), second.reindex(order).tolist()
    for part in values:
        _exact_values(part, what)
    return values


def _dense_ranks(values: Sequence[Any]) -> np.ndarray:
    """Each value's place among the distinct values, the smallest 0: their order and ties, exactly."""
    place = {value: rank for rank, value in enumerate(sorted(set(values)))}
    return np.array([place[value] for value in values], dtype=np.int64)


def _doubled_ranks(dense: np.ndarray) -> list[int]:
    """Twice each value's average rank, from 1: twice the values below it, plus its ties, plus 1."""
    level = np.bincount(dense)
    below = np.cumsum(level) - level
    return [int(2 * below[rank] + level[rank] + 1) for rank in dense]


def fold_auroc(scores: pd.Series, labels: pd.Series) -> Fraction:
    """Return a fold's AUROC, exactly: ties count one half.

    Pilot analysis, section "The main numbers": the share of the fold's pairs of a positive
    and a negative participant in which the positive one scores higher, a tie counting one
    half. A fold whose participants all have the same score gets 1/2, since every pair ties.

    Args:
        scores: Each participant's score, exact (``person_scores``), by session ID.
        labels: Each participant's label, 0 or 1, by session ID.

    Returns:
        The AUROC as a fraction.

    Raises:
        ValueError: If the two do not name the same participants, a value is not exact, a
            label is not 0 or 1, or the fold lacks a class.
    """
    values, label = _aligned(scores, labels, "a fold's AUROC")
    classes = np.array(label, dtype=np.int64)
    if not np.isin(classes, (0, 1)).all():
        raise ValueError("a fold's AUROC: a label must be 0 or 1")
    positive = classes == 1
    if positive.all() or not positive.any():
        raise ValueError("a fold's AUROC needs a positive and a negative participant")
    ranks = _dense_ranks(values)
    wins = pair_wins(ranks[positive], ranks[~positive])
    # The wins are halves and their sum is far below 2**53, so it is exact.
    return Fraction(float(wins.sum())) / (int(positive.sum()) * int((~positive).sum()))


def fold_spearman(scores: pd.Series, other: pd.Series) -> float | None:
    """Return the Spearman correlation of two values per participant of a fold.

    Pilot analysis, section "Also reported": the correlation of the average ranks, tied
    values sharing their average rank. The ranks are counted exactly, and so are the sums
    of squares and of products, which are whole numbers once the ranks are doubled; only the
    last square root rounds. If either side does not vary, the correlation is not defined
    (section "The main numbers", bullet "A fold with equal scores").

    Args:
        scores: Each participant's score, exact (``person_scores``), by session ID.
        other: Each participant's other value, such as the PHQ-8 total or another arm's
            score, by session ID.

    Returns:
        The correlation, from -1 to 1, or None where it is not defined.

    Raises:
        ValueError: If the two do not name the same participants or a value is not exact.
    """
    first, second = _aligned(scores, other, "a fold's Spearman correlation")
    x = _doubled_ranks(_dense_ranks(first))
    y = _doubled_ranks(_dense_ranks(second))
    n = len(x)
    sum_x, sum_y = sum(x), sum(y)
    var_x = n * sum(value * value for value in x) - sum_x * sum_x
    var_y = n * sum(value * value for value in y) - sum_y * sum_y
    if var_x == 0 or var_y == 0:
        return None
    cov = n * sum(a * b for a, b in zip(x, y, strict=True)) - sum_x * sum_y
    return math.copysign(math.sqrt(float(Fraction(cov * cov, var_x * var_y))), cov)


@dataclass(frozen=True, eq=False)
class FoldRun:
    """One round's test fold with every arm's exact scores of its participants.

    A fold-run is one trained model and its own test fold (pilot analysis, section "The main
    numbers"); the zero-shot arm's one model scores every fold of both rounds.

    Attributes:
        round_number: The round, 1 or 2.
        fold: The fold, from 1.
        labels: Each participant's label, 0 or 1, by session ID in ascending order.
        totals: Each participant's PHQ-8 total, by session ID in the same order.
        scores: Each arm's exact scores (``person_scores``) on each view, by (arm, view),
            by session ID in the same order; an arm without a pinned run of the fold-run
            has none.
    """

    round_number: int
    fold: int
    labels: pd.Series
    totals: pd.Series
    scores: Mapping[tuple[str, str], pd.Series]

    @property
    def key(self) -> tuple[int, int]:
        """The fold-run as (round, fold)."""
        return self.round_number, self.fold


def fold_runs(
    labels: pd.Series,
    totals: pd.Series,
    partitions: pd.DataFrame,
    scores: Mapping[tuple[str, str, int, int], pd.Series],
) -> list[FoldRun]:
    """Gather each round's test folds with their participants' labels, totals and scores.

    Every label must be a PHQ-8 total of 10 or more (pilot analysis, section "Data
    protection"), every test fold must hold both classes, and each arm's scores of a
    fold-run must name exactly the participants of its test fold: a run scores its own
    test fold only, and the zero-shot arm's scores are taken fold by fold.

    Args:
        labels: Each participant's label, 0 or 1, by session ID.
        totals: Each participant's PHQ-8 total, by session ID.
        partitions: One row per participant and round, with the columns ``session``,
            ``round`` and ``fold`` (``clips.load_pinned_partitions``).
        scores: Each arm's exact scores (``person_scores``) of each view and fold-run, by
            (arm, view, round, fold).

    Returns:
        The fold-runs, by round and fold.

    Raises:
        StepError: If a label disagrees with its PHQ-8 total, the labels, totals and
            partitions do not name the same participants once in each round, a test fold
            lacks a class, a set of scores names a fold-run that does not exist or other
            participants than its test fold's, or a score is not exact.
    """
    label = pd.Series(labels.to_numpy(dtype=np.int64), index=labels.index.astype(np.int64))
    total = pd.Series(totals.to_numpy(dtype=np.int64), index=totals.index.astype(np.int64))
    if (
        label.index.has_duplicates
        or total.index.has_duplicates
        or set(label.index) != set(total.index)
    ):
        raise StepError("the labels and the PHQ-8 totals must name the same participants once each")
    disagree = int((label != (total.reindex(label.index) >= PHQ8_CUTOFF).astype(np.int64)).sum())
    if disagree:
        raise StepError(
            f"{disagree} labels disagree with their PHQ-8 totals: a participant is positive if"
            f" the total is {PHQ8_CUTOFF} or more"
        )
    people = set(label.index)
    found: list[FoldRun] = []
    used: set[tuple[str, str, int, int]] = set()
    rows = partitions.loc[:, ["session", "round", "fold"]].astype(np.int64)
    for number in sorted(set(rows["round"].tolist())):
        part = rows.loc[rows["round"] == number]
        sessions = part["session"].tolist()
        if len(sessions) != len(set(sessions)) or set(sessions) != people:
            raise StepError(
                f"round {number} of the partitions does not hold each labelled participant once"
            )
        for fold in sorted(set(part["fold"].tolist())):
            order = sorted(part.loc[part["fold"] == fold, "session"].tolist())
            fold_labels = label.reindex(order)
            if fold_labels.nunique() != 2:
                raise StepError(f"round {number}, fold {fold} lacks a class")
            run_scores: dict[tuple[str, str], pd.Series] = {}
            for key, values in scores.items():
                arm, view, score_round, score_fold = key
                if (score_round, score_fold) != (number, fold):
                    continue
                if values.index.has_duplicates or set(values.index) != set(order):
                    raise StepError(
                        f"the scores of {arm} on {view}, round {number}, fold {fold}, name"
                        f" {len(values)} participants, not the {len(order)} of the test fold"
                    )
                ordered = values.reindex(order)
                try:
                    _exact_values(ordered.tolist(), f"{arm} on {view}")
                except ValueError as error:
                    raise StepError(str(error)) from None
                run_scores[(arm, view)] = ordered
                used.add(key)
            found.append(
                FoldRun(
                    number, fold, fold_labels, total.reindex(order), MappingProxyType(run_scores)
                )
            )
    unknown = len(set(scores) - used)
    if unknown:
        raise StepError(f"{unknown} sets of scores name a fold-run that the partitions do not hold")
    return found


class Number(NamedTuple):
    """One number of the analysis on one view.

    Attributes:
        kind: ``AUROC``, an arm's AUROC; ``SPEARMAN``, an arm's Spearman correlation with
            the PHQ-8 totals; ``DIFFERENCE``, the first arm's AUROC minus the second's;
            ``AGREEMENT``, the Spearman correlation of the two arms' scores.
        arms: The arm, or the two arms.
        view: The view.
    """

    kind: str
    arms: tuple[str, ...]
    view: str

    @property
    def name(self) -> str:
        """The number's name in tables, such as ``auroc_b0`` or ``cpg_minus_b0``."""
        if self.kind == DIFFERENCE:
            return f"{self.arms[0]}_minus_{self.arms[1]}"
        return "_".join((self.kind, *self.arms))


@dataclass(frozen=True, eq=False)
class Point:
    """A number's point estimate in each fold-run where it exists.

    Attributes:
        values: The value of each fold-run, by (round, fold): an exact ``Fraction`` for an
            AUROC or a difference, a float for a Spearman correlation, None where it is
            not defined.
    """

    values: Mapping[tuple[int, int], Fraction | float | None]

    def _chosen(self, rounds: Collection[int] | None) -> list[tuple[tuple[int, int], Any]]:
        chosen = [
            (key, value)
            for key, value in sorted(self.values.items())
            if rounds is None or key[0] in rounds
        ]
        if not chosen:
            raise ValueError("the number has no fold-run in these rounds")
        return chosen

    def mean(self, rounds: Collection[int] | None = None) -> Fraction | float | None:
        """Return the mean over the fold-runs, or over those of some rounds.

        Args:
            rounds: The rounds whose fold-runs count; all if None.

        Returns:
            The exact mean of exact values, the mean of floats, or None if a fold-run's
            value is not defined (``undefined`` names it).

        Raises:
            ValueError: If the number has no fold-run in these rounds.
        """
        values = [value for _, value in self._chosen(rounds)]
        if any(value is None for value in values):
            return None
        if all(isinstance(value, Fraction) for value in values):
            return sum(values, Fraction(0)) / len(values)
        return math.fsum(float(value) for value in values) / len(values)

    def undefined(self, rounds: Collection[int] | None = None) -> tuple[tuple[int, int], ...]:
        """Return the fold-runs, of some rounds or all, whose value is not defined.

        Args:
            rounds: The rounds whose fold-runs count; all if None.

        Returns:
            The fold-runs as (round, fold).

        Raises:
            ValueError: If the number has no fold-run in these rounds.
        """
        return tuple(key for key, value in self._chosen(rounds) if value is None)


def _cross_check_auroc(exact: Fraction, scores: pd.Series, labels: pd.Series, what: str) -> None:
    """Stop unless scikit-learn's AUROC of the fold equals the exact one within CROSS_CHECK."""
    floats = [float(value) for value in scores.reindex(labels.index)]
    library = roc_auc_score(labels.to_numpy(dtype=np.int64), floats)
    if not abs(float(library) - float(exact)) <= CROSS_CHECK:
        raise StepError(
            f"{what}: the exact AUROC and scikit-learn's differ by more than {CROSS_CHECK:g}"
        )


def _cross_check_spearman(
    exact: float | None, first: pd.Series, second: pd.Series, what: str
) -> None:
    """Stop unless SciPy's Spearman correlation equals the exact one; two undefined are equal."""
    with warnings.catch_warnings():  # SciPy warns where a side does not vary
        warnings.simplefilter("ignore", stats.ConstantInputWarning)
        library = float(
            stats.spearmanr(
                [float(value) for value in first],
                [float(value) for value in second.reindex(first.index)],
            ).statistic
        )
    if exact is None and math.isnan(library):
        return
    if exact is None or not abs(library - exact) <= CROSS_CHECK:
        raise StepError(
            f"{what}: the Spearman correlation and SciPy's differ by more than {CROSS_CHECK:g},"
            " or only one of them is defined"
        )


def _numbers(found: Collection[Number], settings: EvaluateSettings) -> list[Number]:
    """The numbers found, in the tables' order: AUROCs, Spearman, differences, agreement."""
    wanted = [
        *(
            Number(kind, (arm,), view)
            for kind in (AUROC, SPEARMAN)
            for arm in settings.arms
            for view in settings.views
        ),
        *(
            Number(DIFFERENCE, pair, view)
            for pair in settings.differences
            for view in settings.views
        ),
        *(Number(AGREEMENT, settings.agreement, view) for view in settings.views),
    ]
    return [number for number in wanted if number in found]


def _check_arms(run: FoldRun, settings: EvaluateSettings) -> None:
    """Refuse scores of an arm or a view that the analysis does not read."""
    for arm, view in run.scores:
        if arm not in settings.arms or view not in settings.views:
            raise ValueError(f"scores of {arm} on {view}, which the analysis does not read")


def point_values(runs: Sequence[FoldRun], settings: EvaluateSettings) -> dict[Number, Point]:
    """Return every number's point estimate in every fold-run that has it.

    Pilot analysis, sections "The main numbers" and "Also reported": each arm's AUROC and
    its Spearman correlation with the PHQ-8 totals on each view and fold-run; each
    difference of AUROCs, taken fold by fold on the same participants where both arms have
    a run; and the Spearman correlation of B0's and CPG's scores in each fold-run. Each
    AUROC and Spearman correlation is computed again with scikit-learn and SciPy from the
    scores as floats, and the two must agree within ``CROSS_CHECK``, two undefined values
    counting as equal.

    Args:
        runs: The fold-runs (``fold_runs``), each with the scores of the arms read so far.
        settings: The analysis settings.

    Returns:
        Each number found, in the order of the tables: the AUROCs, the Spearman
        correlations, the differences and the agreement, each by arm or pair, then view.

    Raises:
        StepError: If a value and the libraries' disagree.
        ValueError: If a fold-run holds scores of an arm or a view that the analysis does
            not read.
    """
    found: dict[Number, dict[tuple[int, int], Fraction | float | None]] = {}
    for run in runs:
        _check_arms(run, settings)
        aurocs: dict[tuple[str, str], Fraction] = {}
        for (arm, view), scores in run.scores.items():
            what = f"{arm} on {view}, round {run.round_number}, fold {run.fold}"
            aurocs[(arm, view)] = fold_auroc(scores, run.labels)
            _cross_check_auroc(aurocs[(arm, view)], scores, run.labels, what)
            spearman = fold_spearman(scores, run.totals)
            _cross_check_spearman(spearman, scores, run.totals, what)
            found.setdefault(Number(AUROC, (arm,), view), {})[run.key] = aurocs[(arm, view)]
            found.setdefault(Number(SPEARMAN, (arm,), view), {})[run.key] = spearman
        for view in settings.views:
            for first, second in settings.differences:
                if (first, view) in aurocs and (second, view) in aurocs:
                    found.setdefault(Number(DIFFERENCE, (first, second), view), {})[run.key] = (
                        aurocs[(first, view)] - aurocs[(second, view)]
                    )
            one, two = settings.agreement
            if (one, view) in run.scores and (two, view) in run.scores:
                agreement = fold_spearman(run.scores[(one, view)], run.scores[(two, view)])
                _cross_check_spearman(
                    agreement,
                    run.scores[(one, view)],
                    run.scores[(two, view)],
                    f"{one} and {two} on {view}, round {run.round_number}, fold {run.fold}",
                )
                found.setdefault(Number(AGREEMENT, (one, two), view), {})[run.key] = agreement
    return {number: Point(MappingProxyType(found[number])) for number in _numbers(found, settings)}


# The paired bootstrap ----------------------------------------------------------------------


def draw_counts(
    labels: pd.Series, partitions: pd.DataFrame, rng: np.random.Generator, resamples: int
) -> tuple[pd.DataFrame, int]:
    """Draw the participants of every resample within each class, as counts by session.

    Pilot analysis, section "Intervals", bullet "Draws": in each resample, as many
    participants as each PHQ-8 class holds are drawn with replacement from all its
    participants, 57 positive and 132 negative draws in the pilot: the positives first,
    then the negatives, each class in ascending session ID, as the word check draws
    (``wordcheck.bootstrap_sets``). A participant drawn k times counts k times in their
    test fold of each round. A resample that leaves any test fold of any round without a
    positive or without a negative participant is drawn again and counted. One set of
    draws serves both rounds, every arm and both views, so every difference is paired; it
    reads only the labels and the partitions, so it is drawn before any score is read.

    Args:
        labels: Each participant's label, 0 or 1, by session ID.
        partitions: One row per participant and round, with the columns ``session``,
            ``round`` and ``fold`` (``clips.load_pinned_partitions``).
        rng: The draws' own stream (``EvaluateSettings.rng``).
        resamples: Resamples to keep.

    Returns:
        How often each participant is drawn in each kept resample, of shape
        (resamples, participants), one column per session ID in ascending order, never by
        table row; and the number of resamples drawn again.

    Raises:
        ValueError: If the count is below 1, a label is not 0 or 1, the labels and the
            partitions do not name the same participants once in each round, or a test
            fold lacks a class, which no resample could then fill.
    """
    if resamples < 1:
        raise ValueError(f"draw at least one resample, not {resamples}")
    if labels.index.has_duplicates or not np.isin(labels.to_numpy(), (0, 1)).all():
        raise ValueError("the labels must be 0 or 1, one per participant")
    sessions = sorted(int(session) for session in labels.index)
    label = labels.set_axis(labels.index.astype(np.int64)).reindex(sessions).to_numpy()
    rows = partitions.loc[:, ["session", "round", "fold"]].astype(np.int64)
    rounds: list[np.ndarray] = []  # each participant's test fold in each round, from 0
    for number in sorted(set(rows["round"].tolist())):
        part = rows.loc[rows["round"] == number]
        members = part["session"].tolist()
        if len(members) != len(set(members)) or set(members) != set(sessions):
            raise ValueError(
                f"round {number} of the partitions does not hold each participant once"
            )
        folds = sorted(set(part["fold"].tolist()))
        place = dict(zip(members, part["fold"].tolist(), strict=True))
        rounds.append(np.array([folds.index(place[session]) for session in sessions]))
    if not rounds:
        raise ValueError("the partitions hold no round")
    classes = []
    for value in (1, 0):
        members = np.flatnonzero(label == value)  # column positions, in ascending session ID
        if any(set(fold[members].tolist()) != set(fold.tolist()) for fold in rounds):
            raise ValueError("every test fold of every round must hold both classes")
        classes.append((members, [fold[members] for fold in rounds]))
    table = np.zeros((resamples, len(sessions)), dtype=np.int64)
    redraws = kept = 0
    while kept < resamples:
        drawn = [
            rng.choice(len(members), size=len(members), replace=True) for members, _ in classes
        ]
        if all(
            np.bincount(member_folds[choice], minlength=int(fold.max()) + 1).all()
            for (_, class_folds), choice in zip(classes, drawn, strict=True)
            for member_folds, fold in zip(class_folds, rounds, strict=True)
        ):
            for (members, _), choice in zip(classes, drawn, strict=True):
                table[kept, members] = np.bincount(choice, minlength=len(members))
            kept += 1
        else:
            redraws += 1
    counts = pd.DataFrame(table, columns=pd.Index(sessions, dtype=np.int64, name="session"))
    counts.index.name = "resample"
    return counts, redraws


def draws_sha256(counts: pd.DataFrame) -> str:
    """Return the SHA-256 of the draws, which run.json records in place of the draws.

    The bytes are the session IDs of the columns and then every count, row by row, each as
    a little-endian 64-bit integer.

    Args:
        counts: The draw counts (``draw_counts``).

    Returns:
        The SHA-256 in hex.
    """
    digest = hashlib.sha256()
    digest.update(
        np.ascontiguousarray(counts.columns.to_numpy(dtype=np.int64), dtype="<i8").tobytes()
    )
    digest.update(np.ascontiguousarray(counts.to_numpy(dtype=np.int64), dtype="<i8").tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class Resampled:
    """A number's value on every resample, in each fold-run where it exists.

    Attributes:
        fold_runs: The fold-runs of the columns of ``values``, as (round, fold), in order.
        values: The value of each fold-run on each resample, of shape (R, F); NaN where a
            Spearman correlation is not defined on that resample.
    """

    fold_runs: tuple[tuple[int, int], ...]
    values: np.ndarray

    def _columns(self, rounds: Collection[int] | None) -> list[int]:
        columns = [
            column
            for column, key in enumerate(self.fold_runs)
            if rounds is None or key[0] in rounds
        ]
        if not columns:
            raise ValueError("the number has no fold-run in these rounds")
        return columns

    def means(self, rounds: Collection[int] | None = None) -> np.ndarray:
        """Return the mean over the fold-runs, or those of some rounds, on each resample.

        Args:
            rounds: The rounds whose fold-runs count; all if None.

        Returns:
            One mean per resample, of shape (R,); NaN where a fold-run's value is not
            defined on that resample (``undefined`` names it).

        Raises:
            ValueError: If the number has no fold-run in these rounds.
        """
        return self.values[:, self._columns(rounds)].mean(axis=1)

    def undefined(self, rounds: Collection[int] | None = None) -> tuple[tuple[int, int], ...]:
        """Return the fold-runs, of some rounds or all, not defined on some resample.

        Args:
            rounds: The rounds whose fold-runs count; all if None.

        Returns:
            The fold-runs as (round, fold).

        Raises:
            ValueError: If the number has no fold-run in these rounds.
        """
        return tuple(
            self.fold_runs[column]
            for column in self._columns(rounds)
            if np.isnan(self.values[:, column]).any()
        )


def _resampled_ranks(drawn: np.ndarray, dense: np.ndarray) -> np.ndarray:
    """Twice each participant's average rank among the drawn rows of the fold, per resample.

    A participant's doubled average rank is one, plus twice the drawn rows below it, plus
    the drawn rows level with it, its own included: ``1 + drawn @ (2 * wins).T``, where
    ``wins`` compares every participant with every other (``pair_wins``). Every value is a
    whole number far below 2**53, so it is exact.
    """
    return 1.0 + drawn @ (2.0 * pair_wins(dense, dense)).T


def _weighted_correlation(weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The Pearson correlation of ranks over the repeated rows of each resample; NaN if one side is flat.

    Each participant counts as often as drawn. The sums of the doubled ranks, of their
    squares and of their products are whole numbers far below 2**53, so they and the
    variances and covariance built from them are exact; only the division and the square
    roots round.
    """
    total = weights.sum(axis=1)
    sum_x, sum_y = (weights * x).sum(axis=1), (weights * y).sum(axis=1)
    var_x = total * (weights * x * x).sum(axis=1) - sum_x * sum_x
    var_y = total * (weights * y * y).sum(axis=1) - sum_y * sum_y
    cov = total * (weights * x * y).sum(axis=1) - sum_x * sum_y
    found = np.full(len(weights), np.nan)
    defined = (var_x > 0) & (var_y > 0)
    # sqrt(v * v) is v exactly, so two sides that rank alike give exactly 1.
    found[defined] = np.clip(cov[defined] / np.sqrt(var_x[defined] * var_y[defined]), -1.0, 1.0)
    return found


def resampled_means(
    counts: pd.DataFrame, runs: Sequence[FoldRun], settings: EvaluateSettings
) -> dict[Number, Resampled]:
    """Return every number of ``point_values`` on each resample, fold-run by fold-run.

    Pilot analysis, section "Intervals", bullet "Recomputing": on each resample, every
    fold's AUROC and Spearman correlation is computed again on the participants drawn into
    that fold, each counted as often as drawn, which equals the value of the repeated rows:
    the AUROC with ties one half (``weighted_aurocs``), the Spearman correlation with tied
    rows sharing their average rank. A difference is taken fold by fold on the same
    resample and the means over fold-runs follow (``Resampled.means``), so every difference
    is paired. The scores enter only through their exact order, so a resample in which
    every participant is drawn once gives each point estimate.

    Args:
        counts: The draw counts (``draw_counts``), one column per session ID.
        runs: The fold-runs (``fold_runs``), each with the scores of the arms read so far.
        settings: The analysis settings.

    Returns:
        The same numbers as ``point_values``, in the same order.

    Raises:
        ValueError: If the counts are not whole numbers of at least 0 by session ID, lack
            a participant of a fold, or leave a fold without a class in a resample, or a
            fold-run holds scores of an arm or a view that the analysis does not read.
    """
    columns = counts.columns
    if columns.has_duplicates or not pd.api.types.is_integer_dtype(columns):
        raise ValueError("the draw counts must have one column per session ID")
    table = counts.to_numpy()
    if not pd.api.types.is_integer_dtype(table.dtype) or (table < 0).any():
        raise ValueError("the draw counts must be whole numbers of at least 0")
    position = {int(session): column for column, session in enumerate(columns)}
    found: dict[Number, dict[tuple[int, int], np.ndarray]] = {}
    for run in runs:
        _check_arms(run, settings)
        order = [int(session) for session in run.labels.index]
        if any(session not in position for session in order):
            raise ValueError(
                f"the draw counts lack participants of round {run.round_number}, fold {run.fold}"
            )
        drawn = table[:, [position[session] for session in order]].astype(np.float64)
        positive = run.labels.to_numpy(dtype=np.int64) == 1
        totals = _resampled_ranks(drawn, _dense_ranks(run.totals.tolist()))
        ranks: dict[tuple[str, str], np.ndarray] = {}
        aurocs: dict[tuple[str, str], np.ndarray] = {}
        for (arm, view), scores in run.scores.items():
            dense = _dense_ranks(scores.tolist())
            aurocs[(arm, view)] = weighted_aurocs(
                pair_wins(dense[positive], dense[~positive]),
                drawn[:, positive],
                drawn[:, ~positive],
            )
            ranks[(arm, view)] = _resampled_ranks(drawn, dense)
            found.setdefault(Number(AUROC, (arm,), view), {})[run.key] = aurocs[(arm, view)]
            found.setdefault(Number(SPEARMAN, (arm,), view), {})[run.key] = _weighted_correlation(
                drawn, ranks[(arm, view)], totals
            )
        for view in settings.views:
            for first, second in settings.differences:
                if (first, view) in aurocs and (second, view) in aurocs:
                    found.setdefault(Number(DIFFERENCE, (first, second), view), {})[run.key] = (
                        aurocs[(first, view)] - aurocs[(second, view)]
                    )
            one, two = settings.agreement
            if (one, view) in ranks and (two, view) in ranks:
                found.setdefault(Number(AGREEMENT, (one, two), view), {})[run.key] = (
                    _weighted_correlation(drawn, ranks[(one, view)], ranks[(two, view)])
                )
    resampled = {}
    for number in _numbers(found, settings):
        keys = tuple(sorted(found[number]))
        resampled[number] = Resampled(keys, np.column_stack([found[number][key] for key in keys]))
    return resampled


# Checks F2 and F3 and the decision rule ----------------------------------------------------


def f2_reading(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Read check F2 on the kept CPG runs: each run's verdict, and the pooled share.

    Pilot protocol, section "Training", bullet "F2 and the two allowed rules", and pilot
    analysis, section "Checks first": F2 passes if every CPG run's share of active bags in
    epochs 2 and 3 lies from 0.20 to 0.80, both included, as the run read it against the
    bounds of its own ``train.yaml`` (``results.rules.alive.f2_passes``). The pooled share,
    the summed active bags over the summed bags of the kept CPG runs, not a mean of shares,
    is reported only. A record must say whether F2 passed (true or false) on counts that
    give its share; a record that does not, or that lacks a key, is refused, never read as
    a fail.

    Args:
        records: The ``run.json`` of each kept CPG run, by its run folder.

    Returns:
        ``runs``: by run folder, its ``active`` and ``bags``, their ``share`` and whether
        it ``passes``; ``pooled``: the summed ``active`` and ``bags`` and their ``share``;
        ``passes``: whether every run passes.

    Raises:
        StepError: If there is no record, or a record does not say that F2 reads the run,
            lacks its counts or its verdict, or holds a share that its counts do not give.
    """
    if not records:
        raise StepError("check F2 reads the kept CPG runs, and there is none")
    runs: dict[str, dict[str, Any]] = {}
    for run, record in records.items():
        if _nested(record, "results", "rule_readings", "reads_f2") is not True:
            raise StepError(f"{run}: its run.json does not say that check F2 reads it")
        alive = _nested(record, "results", "rules", "alive")
        bags = _nested(alive, "bags")
        active = _nested(alive, "active")
        if (
            isinstance(bags, bool)
            or isinstance(active, bool)
            or not isinstance(bags, int)
            or not isinstance(active, int)
            or not 0 <= active <= bags
            or bags < 1
        ):
            raise StepError(f"{run}: its run.json lacks the bags and active bags of check F2")
        if _nested(alive, "share") != float(Fraction(active, bags)):
            raise StepError(f"{run}: the share of check F2 in its run.json is not its counts'")
        passes = _nested(alive, "f2_passes")
        if not isinstance(passes, bool):
            raise StepError(f"{run}: its run.json does not say whether check F2 passed")
        runs[run] = {"active": active, "bags": bags, "share": alive["share"], "passes": passes}
    active = sum(reading["active"] for reading in runs.values())
    bags = sum(reading["bags"] for reading in runs.values())
    return {
        "runs": runs,
        "pooled": {"active": active, "bags": bags, "share": float(Fraction(active, bags))},
        "passes": all(reading["passes"] for reading in runs.values()),
    }


def _exact(value: Any, what: str) -> Fraction:
    """An exact value for a check or the rule; a float, which may have rounded, is refused."""
    if isinstance(value, bool) or not isinstance(value, Rational):
        raise TypeError(f"{what} must be exact, a Fraction, not {type(value).__name__}")
    return Fraction(value)


def f3_reading(b0_x0: Point, settings: EvaluateSettings) -> dict[str, Any]:
    """Read check F3: B0's mean AUROC on x0 is at least ``f3_min_auroc``, compared exactly.

    Pilot protocol, section "Feasibility checks", F3, and pilot analysis, section "Decision
    rule": the mean over B0's fold-runs, compared unrounded with the bound, beside its
    per-round values.

    Args:
        b0_x0: B0's AUROC on x0 (``point_values``).
        settings: The analysis settings.

    Returns:
        ``value``, the mean over B0's fold-runs; ``rounds``, the mean of each round's
        fold-runs, by round; ``bound``; and ``passes``. The values are fractions.

    Raises:
        TypeError: If the AUROCs are not exact.
    """
    value = _exact(b0_x0.mean(), "B0's mean AUROC on x0")
    rounds = sorted({key[0] for key in b0_x0.values})
    return {
        "value": value,
        "rounds": {number: _exact(b0_x0.mean({number}), "B0's AUROC") for number in rounds},
        "bound": settings.f3_min_auroc,
        "passes": value >= settings.f3_min_auroc,
    }


class Outcome(NamedTuple):
    """The decision rule's outcome and the protocol's decision for it, word for word.

    Attributes:
        name: The outcome, such as "Green: beats B0".
        decision: Its decision in the protocol's table.
    """

    name: str
    decision: str


def outcome(delta_x0: Fraction, delta_rev: Fraction, settings: EvaluateSettings) -> Outcome:
    """Read the decision rule on the point estimates of Δ, compared exactly.

    Pilot protocol, section "Decision rule": "Green: beats B0" if Δ on x0 is at least +0.03;
    "Red: worse on normal audio" if it is at most −0.04; between the two, strictly, "Green:
    robust without words" if Δ on rev is at least +0.03 and "Amber: no visible effect" if
    not. Every case falls in exactly one. The intervals and per-round values are reported
    beside it, never read as a test.

    Args:
        delta_x0: Δ = CPG − B0 on x0, the mean of the per-fold differences, exact.
        delta_rev: Δ on rev, the same way.
        settings: The analysis settings, with the rule's bounds.

    Returns:
        The outcome and its decision.

    Raises:
        TypeError: If a Δ is not exact.
    """
    x0 = _exact(delta_x0, "Δ on x0")
    rev = _exact(delta_rev, "Δ on rev")
    if x0 >= settings.green_at_least:
        name = GREEN_BEATS
    elif x0 <= settings.red_at_most:
        name = RED
    elif rev >= settings.green_at_least:
        name = GREEN_ROBUST
    else:
        name = AMBER
    return Outcome(name, DECISIONS[name])


def framing(
    outcome_name: str,
    b1_minus_b0: Fraction | None,
    cpg_minus_b1: Fraction | None,
    settings: EvaluateSettings,
) -> str:
    """Read the framing line, which only "Green: beats B0" reads.

    Pilot analysis, section "Decision rule", bullet "Where a gain comes from": in "Green:
    beats B0", if B1 − B0 also reaches +0.03 and CPG − B1 stays below +0.02, both on x0, the
    gain comes mainly from the extra clips. It changes the framing, not the decision. In
    any other outcome the line is not read; without B1 it is not available.

    Args:
        outcome_name: The outcome (``outcome``).
        b1_minus_b0: B1 − B0 on the framing's view, exact; None without B1.
        cpg_minus_b1: CPG − B1 on the framing's view, exact; None without B1.
        settings: The analysis settings.

    Returns:
        The framing line: ``FRAMING_EXTRA_CLIPS``, ``FRAMING_STAYS``,
        ``FRAMING_NOT_READ`` or ``FRAMING_NO_B1``.

    Raises:
        TypeError: If a difference is not exact.
    """
    if outcome_name != GREEN_BEATS:
        return FRAMING_NOT_READ
    if b1_minus_b0 is None or cpg_minus_b1 is None:
        return FRAMING_NO_B1
    b1_gain = _exact(b1_minus_b0, "B1 − B0")
    cpg_over_b1 = _exact(cpg_minus_b1, "CPG − B1")
    if b1_gain >= settings.b1_gain_at_least and cpg_over_b1 < settings.cpg_over_b1_below:
        return FRAMING_EXTRA_CLIPS
    return FRAMING_STAYS


def printed(value: Fraction, bounds: Iterable[Fraction] = ()) -> str:
    """Print an exact value with 4 decimals, and unrounded too where it rounds to a bound.

    Pilot protocol, section "Clip versions", bullet "Reading the rules", which the pilot
    analysis applies to F3 and the decision rule (section "Decision rule"): values are
    compared unrounded and printed with 4 decimals, and a value that rounds to its bound is
    also printed unrounded, with 20 significant digits and as a fraction, so that it is
    never mistaken for the bound.

    Args:
        value: The exact value.
        bounds: The bounds that the value is read against.

    Returns:
        Such as "0.0312", or "0.0300 (unrounded 0.03 = 3/100)" for a value of exactly 3/100.

    Raises:
        TypeError: If the value or a bound is not exact.
    """
    exact = _exact(value, "a printed value")
    rounded = round(exact, 4)  # half to even, on the exact value
    text = f"{float(rounded):.4f}"
    if any(rounded == round(_exact(bound, "a bound"), 4) for bound in bounds):
        digits = _UNROUNDED.divide(Decimal(exact.numerator), Decimal(exact.denominator))
        text += f" (unrounded {digits} = {exact.numerator}/{exact.denominator})"
    return text
