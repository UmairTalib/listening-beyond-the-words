"""The main study's power, simulated again with the pilot's numbers.

``docs/pilot_diagnostics.md``, section "The main study's power", fixes the simulation, and
``configs/pilot/power.yaml`` holds its settings (``PowerSettings``). Power is the chance that
the main study's test says "CPG beats B0" when CPG is truly better by a given amount, the
true gain; at a true gain of 0 it is the false-positive rate. The simulation reads no
participant data: its inputs are two published tables of the pilot's analysis, read here
into the numbers that it uses (``read_pilot``).

The model is the study design's, as the page reads it. A participant's score in an arm is
the arm's class shift, set by its AUROC (``class_shift``), plus a stable part, plus run
noise that each fold-run draws anew. B0's and CPG's stable parts correlate at r, and ρ is
the correlation of an arm's scores between two runs, so within a class the two arms' scores
in one fold-run correlate at r·ρ. There are no segments or bags. This module holds:

- the people of a block of studies (``simulate_people``), their test folds in each round,
  stratified by label with the pilot's class counts per fold (``stratified_folds``), their
  scores in each fold-run, CPG's class shift set by B0's AUROC plus the true gain
  (``class_shifts``, ``arm_scores``), and the pair wins of each fold (``pair_wins``);
- the pilot analysis's draws, vectorised (``vector_draws``), Δ on each resample
  (``resampled_delta``), the test's decision at each α (``passes``), the Green call of the
  pilot inside each study, compared exactly (``green_pilot``), and the agreement of the two
  arms' scores within each fold (``fold_agreement``);
- one block of simulated pilots or studies from its own stream (``simulate_block``), run in
  worker processes started by spawn, with one BLAS thread each (``Workers``);
- the two checks that come first (``design_check``, ``resample_check``), the fits of r and
  of the lowest ρ (``calibrate_r``, ``lowest_rho``), the scenarios of every view, all
  fitted before any study is simulated (``fit_views``), their simulated studies
  (``simulate_views``), and the two tables (``power_table``, ``fit_table``).

Every count is exact: pair wins are doubled so that a tie counts 1, draw counts are whole
numbers, and every sum of their products is a whole number below 2**24, so it is exact in
float32 whatever the order of the sums; only the last divisions round, in float64. A block's
numbers therefore depend on its stream alone, never on the worker or the number of workers.
Nothing here prints; the job reports what the functions return.
"""

import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from multiprocessing import get_context
from statistics import NormalDist
from types import TracebackType
from typing import Any, Final, NamedTuple, Protocol

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from listening_beyond_words.runinfo import StepError, item_seed

VIEWS: Final = ("x0", "rev")  # the views of the main study's power, simulated apart
GREEN_VIEW: Final = "x0"  # the pilot's Green call is read on x0; rev is not conditioned
ARMS: Final = ("zeroshot", "b0", "b1", "cpg")  # the arms whose per-round AUROCs stand beside
TARGET_POWER: Final = 0.8  # the true gain needed for 80% power (bullet "Outputs")
AT: Final = "middle_rho"  # where the sensitivity rows sit
SENSITIVITY: Final = ("agreement_80", "b0_auroc_80", "se_matched")
DESIGN: Final = "design"  # the study design's scenarios, as a view of their own streams
DESIGN_NAMES: Final = ("low", "middle", "high")  # its three noise scenarios, in this order
LEVEL: Final = "80"  # the pilot analysis's 80% interval: its columns low_80 and high_80
SEARCH_STEPS: Final = 30  # the most halvings of a fit's bracket on [0, 1]
CHUNK: Final = (
    1024  # resamples summed at a time in the bootstrap, so that its products stay in the cache
)
CLOSE: Final = 0.1  # a fit stops once within this share of its tolerance
NOT_APPLICABLE: Final = "not applicable"
NO_GREEN: Final = "not defined (no Green pilot)"
# Whether r met its target, as power_fit.csv says it: only a sensitivity row can miss it.
MET: Final = "met"
NOT_REACHED: Final = "not reached"
# What r is fitted to, as power_fit.csv names it.
AGREEMENT: Final = "agreement"
BOOTSTRAP_SE: Final = "bootstrap_se"
_FILE: Final = "power.yaml"
_SOURCE: Final = 'pilot diagnostics, section "The main study\'s power"'
_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_TOP_KEYS: Final = (
    "inputs_sha256",
    "views",
    "main",
    "green",
    "simulation",
    "sensitivity",
    "resample_check",
    "design_check",
)
# The published tables' rows that the simulation reads: number, view and column.
_DELTA: Final = "cpg_minus_b0"
_AGREEMENT: Final = "agreement_b0_cpg"
_B0: Final = "auroc_b0"
_BLAS: list[Any] = []  # a worker's thread limit, kept alive while it works
# The variables that make a worker's BLAS and OpenMP libraries start with one thread: the
# workers are spawned with them, so that no library starts a thread per CPU of the job. The
# last is Apple's Accelerate, which threadpoolctl can neither see nor limit.
ONE_THREAD: Final = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
# What the record keeps of each thread pool that threadpoolctl sees in a worker.
POOL_KEYS: Final = ("user_api", "internal_api", "num_threads")


# Settings ----------------------------------------------------------------------------------


def _block(value: Any, keys: Sequence[str], where: str) -> Mapping[str, Any]:
    """A mapping with exactly these keys, or a refusal that names them."""
    if not isinstance(value, Mapping) or list(value) != list(keys):
        found = list(value) if isinstance(value, Mapping) else type(value).__name__
        raise StepError(f"{where} must hold exactly {list(keys)}, in this order, not {found}")
    return value


def _fixed(value: Any, expected: Any, where: str) -> None:
    """A setting that names what the simulation computes must be the value that it computes."""
    if value != expected or type(value) is not type(expected):
        raise StepError(
            f"{where} must be {expected!r}, as the simulation computes it ({_SOURCE}), not"
            f" {value!r}"
        )


def _whole(value: Any, minimum: int, where: str) -> int:
    """A whole number of at least ``minimum``; True and False are refused."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StepError(f"{where} must be a whole number of at least {minimum}, not {value!r}")
    return value


def _float(value: Any, low: float, high: float, where: str, *, closed: bool = False) -> float:
    """A float between two bounds: strictly, or with the bounds included if ``closed``."""
    inside = (
        isinstance(value, float)
        and math.isfinite(value)
        and (low <= value <= high if closed else low < value < high)
    )
    if not inside:
        span = f"from {low:g} to {high:g}" if closed else f"strictly between {low:g} and {high:g}"
        raise StepError(f"{where} must be a float {span}, not {value!r}")
    return float(value)


def _floats(
    value: Any, count: int | None, low: float, high: float, where: str
) -> tuple[float, ...]:
    """A list of floats from ``low`` (included) to ``high`` (excluded), of a given length."""
    if (
        not isinstance(value, list)
        or not value
        or (count is not None and len(value) != count)
        or any(
            not isinstance(item, float) or not math.isfinite(item) or not low <= item < high
            for item in value
        )
    ):
        size = f"{count} " if count is not None else ""
        raise StepError(
            f"{where} must be a list of {size}floats from {low:g} up to {high:g}, not {value!r}"
        )
    return tuple(value)


def _grid(value: Any, where: str) -> tuple[float, ...]:
    """The true gains: from ``first`` to ``last`` in equal steps, written as clean decimals."""
    block = _block(value, ("first", "last", "step"), where)
    first = _float(block["first"], 0.0, 1.0, f"{where}.first", closed=True)
    last = _float(block["last"], 0.0, 1.0, f"{where}.last", closed=True)
    step = _float(block["step"], 0.0, 1.0, f"{where}.step")
    points = (last - first) / step
    if first != 0.0 or last >= 1.0 or points < 1.0 or abs(points - round(points)) > 1e-9:
        raise StepError(
            f"{where} must run from 0.0 to below 1.0 in whole steps: the main study's standard"
            " error and the false-positive rate are read at a true gain of 0"
        )
    return tuple(round(first + index * step, 10) for index in range(round(points) + 1))


def _sha256(value: Any, where: str) -> str:
    """A pinned SHA-256: 64 lowercase hex characters."""
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise StepError(f"{where} must be a SHA-256 of 64 lowercase hex characters")
    return value


@dataclass(frozen=True, slots=True)
class PowerSettings:
    """The settings of the power simulation, from ``configs/pilot/power.yaml``.

    The settings that name what the simulation computes (its views and its sensitivity
    rows) must be the ones that this code implements, in their order; the numbers are
    checked for their type and range.

    Attributes:
        summary_sha256: The pin of the pilot's ``summary.csv``.
        folds_sha256: The pin of the pilot's ``fold_values.csv``.
        views: The views simulated, apart: ``VIEWS``.
        rounds: The main study's rounds.
        folds: Folds per round, as many as the pilot's, whose class counts it reuses.
        gains: The true gains, from 0 in equal steps.
        alpha: The one-sided level of the last of Holm's steps.
        holm_family: The hypotheses that share it under Holm's correction.
        green_auroc: Check F3's bound on B0's mean AUROC on ``x0``, as written.
        green_delta: The rule's bound on Δ on ``x0``, as written.
        studies: Simulated studies per point.
        blocks: The fixed blocks of a point's studies, each with its own stream.
        resamples: Resamples of the paired bootstrap per simulated study.
        rho_points: The points of ρ, from the lowest that the agreement allows to 1.
        calibration_pilots: Simulated pilots per fit of r.
        tolerance: How close a fit must come to its target.
        seed: The run seed of the simulation's streams.
        check_studies: Studies at each true gain of the resample check.
        check_gains: The resample check's true gains.
        check_resamples: The resamples that the check compares against.
        max_changed: The largest share of studies whose decision may change.
        fallback_resamples: The resamples of every table if the check fails.
        design_auroc: The study design's assumed AUROC of B0.
        design_scenarios: Its three scenarios (r, ρ): low, middle and high noise.
        design_pilots: Simulated pilots per scenario of the design check.
        design_pilot_sd: The standard deviation of the pilot's Δ in each scenario.
        design_one_arm_sd: One arm's one-round AUROC standard deviation, middle scenario.
        design_max_gap: How close the model must come to each of the design's figures.
    """

    summary_sha256: str
    folds_sha256: str
    views: tuple[str, ...]
    rounds: int
    folds: int
    gains: tuple[float, ...]
    alpha: float
    holm_family: int
    green_auroc: Fraction
    green_delta: Fraction
    studies: int
    blocks: int
    resamples: int
    rho_points: int
    calibration_pilots: int
    tolerance: float
    seed: int
    check_studies: int
    check_gains: tuple[float, ...]
    check_resamples: int
    max_changed: float
    fallback_resamples: int
    design_auroc: float
    design_scenarios: tuple[tuple[float, float], ...]
    design_pilots: int
    design_pilot_sd: tuple[float, ...]
    design_one_arm_sd: float
    design_max_gap: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "PowerSettings":
        """Read the settings and check every key and value.

        Args:
            config: The content of ``configs/pilot/power.yaml``.

        Returns:
            The settings.

        Raises:
            StepError: If a key is missing, unknown or out of order, a setting that names
                what the simulation computes is not the one that it computes, or a number
                has the wrong type or lies out of range.
        """
        top = _block(config, _TOP_KEYS, _FILE)
        pins = _block(top["inputs_sha256"], ("summary", "folds"), f"{_FILE}: inputs_sha256")
        _fixed(top["views"], list(VIEWS), f"{_FILE}: views")
        main = _block(
            top["main"], ("rounds", "folds", "gains", "alpha", "holm_family"), f"{_FILE}: main"
        )
        green = _block(top["green"], ("f3_min_auroc", "delta_x0_at_least"), f"{_FILE}: green")
        simulation = _block(
            top["simulation"],
            (
                "studies",
                "blocks",
                "resamples",
                "rho_points",
                "calibration_pilots",
                "tolerance",
                "seed",
            ),
            f"{_FILE}: simulation",
        )
        sensitivity = _block(top["sensitivity"], ("at", "rows"), f"{_FILE}: sensitivity")
        _fixed(sensitivity["at"], AT, f"{_FILE}: sensitivity.at")
        _fixed(sensitivity["rows"], list(SENSITIVITY), f"{_FILE}: sensitivity.rows")
        check = _block(
            top["resample_check"],
            ("studies", "gains", "resamples", "max_changed", "fallback_resamples"),
            f"{_FILE}: resample_check",
        )
        design = _block(
            top["design_check"],
            ("b0_auroc", "scenarios", "pilots", "pilot_sd", "one_arm_sd", "max_gap"),
            f"{_FILE}: design_check",
        )
        rho_points = _whole(simulation["rho_points"], 3, f"{_FILE}: simulation.rho_points")
        if rho_points % 2 == 0:
            raise StepError(
                f"{_FILE}: simulation.rho_points must be odd, so that the sensitivity rows have"
                f" a middle ρ, not {rho_points}"
            )
        resamples = _whole(simulation["resamples"], 2, f"{_FILE}: simulation.resamples")
        check_resamples = _whole(check["resamples"], 2, f"{_FILE}: resample_check.resamples")
        if check_resamples <= resamples:
            raise StepError(
                f"{_FILE}: resample_check.resamples must exceed simulation.resamples, whose"
                " decisions it checks"
            )
        scenarios = design["scenarios"]
        if (
            not isinstance(scenarios, list)
            or len(scenarios) != len(DESIGN_NAMES)
            or any(not isinstance(pair, list) or len(pair) != 2 for pair in scenarios)
        ):
            raise StepError(
                f"{_FILE}: design_check.scenarios must be the study design's three pairs"
                f" [r, ρ], not {scenarios!r}"
            )
        pairs = tuple(
            (
                _float(r, 0.0, 1.0, f"{_FILE}: design_check.scenarios r", closed=True),
                _float(rho, 0.0, 1.0, f"{_FILE}: design_check.scenarios ρ", closed=True),
            )
            for r, rho in scenarios
        )
        green_auroc = _float(green["f3_min_auroc"], 0.0, 1.0, f"{_FILE}: green.f3_min_auroc")
        green_delta = _float(
            green["delta_x0_at_least"], -1.0, 1.0, f"{_FILE}: green.delta_x0_at_least"
        )
        return cls(
            summary_sha256=_sha256(pins["summary"], f"{_FILE}: inputs_sha256.summary"),
            folds_sha256=_sha256(pins["folds"], f"{_FILE}: inputs_sha256.folds"),
            views=VIEWS,
            rounds=_whole(main["rounds"], 1, f"{_FILE}: main.rounds"),
            folds=_whole(main["folds"], 2, f"{_FILE}: main.folds"),
            gains=_grid(main["gains"], f"{_FILE}: main.gains"),
            alpha=_float(main["alpha"], 0.0, 1.0, f"{_FILE}: main.alpha"),
            holm_family=_whole(main["holm_family"], 1, f"{_FILE}: main.holm_family"),
            # The bounds are the decimals written: 0.03 is 3/100, not the float below it.
            green_auroc=Fraction(repr(green_auroc)),
            green_delta=Fraction(repr(green_delta)),
            studies=_whole(simulation["studies"], 2, f"{_FILE}: simulation.studies"),
            blocks=_whole(simulation["blocks"], 1, f"{_FILE}: simulation.blocks"),
            resamples=resamples,
            rho_points=rho_points,
            calibration_pilots=_whole(
                simulation["calibration_pilots"], 2, f"{_FILE}: simulation.calibration_pilots"
            ),
            tolerance=_float(simulation["tolerance"], 0.0, 1.0, f"{_FILE}: simulation.tolerance"),
            seed=_whole(simulation["seed"], 0, f"{_FILE}: simulation.seed"),
            check_studies=_whole(check["studies"], 1, f"{_FILE}: resample_check.studies"),
            check_gains=_floats(check["gains"], None, 0.0, 1.0, f"{_FILE}: resample_check.gains"),
            check_resamples=check_resamples,
            max_changed=_float(
                check["max_changed"], 0.0, 1.0, f"{_FILE}: resample_check.max_changed", closed=True
            ),
            fallback_resamples=_whole(
                check["fallback_resamples"], 2, f"{_FILE}: resample_check.fallback_resamples"
            ),
            design_auroc=_float(design["b0_auroc"], 0.0, 1.0, f"{_FILE}: design_check.b0_auroc"),
            design_scenarios=pairs,
            design_pilots=_whole(design["pilots"], 2, f"{_FILE}: design_check.pilots"),
            design_pilot_sd=_floats(
                design["pilot_sd"], len(DESIGN_NAMES), 0.0, 1.0, f"{_FILE}: design_check.pilot_sd"
            ),
            design_one_arm_sd=_float(
                design["one_arm_sd"], 0.0, 1.0, f"{_FILE}: design_check.one_arm_sd"
            ),
            design_max_gap=_float(
                design["max_gap"], 0.0, 1.0, f"{_FILE}: design_check.max_gap", closed=True
            ),
        )

    @property
    def alphas(self) -> tuple[float, ...]:
        """The one-sided levels, ascending: α/k, ..., α/2, α for Holm's k hypotheses.

        Pilot diagnostics, section "The main study's power", bullet "α": the smallest
        p-value is tested at α/3, the next at α/2 and the last at α, so the test on normal
        audio meets one of these levels, by its rank.
        """
        return tuple(self.alpha / rank for rank in range(self.holm_family, 0, -1))


# The pilot's numbers ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Design:
    """The pilot's design: its class counts per test fold, the same in every round.

    Participants 0 to P − 1 are the positive ones and P to N − 1 the negative ones; a study's
    rounds lay them out by fold (``stratified_folds``).

    Attributes:
        positives: Positive participants in each fold, in fold order.
        negatives: Negative participants in each fold.
        pilot_rounds: The pilot's rounds.
    """

    positives: tuple[int, ...]
    negatives: tuple[int, ...]
    pilot_rounds: int

    @property
    def classes(self) -> tuple[int, int]:
        """The positive and the negative participants: P and N − P."""
        return sum(self.positives), sum(self.negatives)

    @property
    def size(self) -> int:
        """All participants: N."""
        return sum(self.positives) + sum(self.negatives)

    @property
    def pairs(self) -> np.ndarray:
        """Each fold's pairs of a positive and a negative participant, of shape (F,)."""
        return np.array(self.positives, dtype=np.int64) * np.array(self.negatives, dtype=np.int64)

    def slices(self) -> list[tuple[slice, slice]]:
        """Where each fold's positives and negatives sit in a round's layout."""
        positive_edges = np.concatenate([[0], np.cumsum(self.positives)]).tolist()
        negative_edges = (
            sum(self.positives) + np.concatenate([[0], np.cumsum(self.negatives)])
        ).tolist()
        return [
            (
                slice(positive_edges[f], positive_edges[f + 1]),
                slice(negative_edges[f], negative_edges[f + 1]),
            )
            for f in range(len(self.positives))
        ]


@dataclass(frozen=True, slots=True)
class PilotView:
    """The pilot's numbers on one view, read from the published tables.

    Attributes:
        view: The view.
        b0_auroc: B0's AUROC, the mean of its 10 fold AUROCs.
        b0_ends: The ends of its 80% percentile interval.
        delta: Δ = CPG − B0.
        delta_rounds: Δ in each round.
        delta_se: Δ's bootstrap standard error.
        agreement: The agreement of B0's and CPG's scores (their Spearman correlation within
            each fold, averaged).
        agreement_ends: The ends of its 80% percentile interval.
        aurocs: Each arm's AUROC in each round, by arm (``ARMS``).
    """

    view: str
    b0_auroc: float
    b0_ends: tuple[float, float]
    delta: float
    delta_rounds: tuple[float, ...]
    delta_se: float
    agreement: float
    agreement_ends: tuple[float, float]
    aurocs: Mapping[str, tuple[float, ...]]

    @property
    def round_difference(self) -> float:
        """Δ in round 1 minus Δ in round 2."""
        return self.delta_rounds[0] - self.delta_rounds[1]


def _cell(rows: pd.DataFrame, number: str, view: str, column: str) -> float:
    """One number of summary.csv, which must be finite."""
    found = rows.loc[(rows["number"] == number) & (rows["view"] == view)]
    if len(found) != 1 or column not in rows.columns:
        raise StepError(
            f"summary.csv must hold one row of {number} on {view} with the column {column}"
        )
    text = str(found.iloc[0][column])
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        raise StepError(f"summary.csv: {number} on {view} holds no number in {column}")
    return value


def _counts(table: pd.DataFrame, column: str) -> np.ndarray:
    """A column of fold_values.csv as whole numbers of at least 1."""
    try:
        values = np.array([int(text) for text in table[column]], dtype=np.int64)
    except (KeyError, ValueError):
        raise StepError(f"fold_values.csv must hold whole numbers in {column}") from None
    if (values < 1).any():
        raise StepError(f"fold_values.csv: every fold must hold both classes ({column})")
    return values


def read_pilot(
    summary: pd.DataFrame, folds: pd.DataFrame, settings: PowerSettings
) -> tuple[dict[str, PilotView], Design]:
    """Read the pilot's numbers and design from its two published tables.

    Pilot diagnostics, section "The main study's power", bullet "Which number sets what":
    B0's AUROC, the agreement, Δ with its rounds and bootstrap standard error, the ends of
    the 80% intervals of the agreement and of B0's AUROC, and the arms' per-round AUROCs,
    from ``summary.csv``; the class counts of each test fold, from ``fold_values.csv``. The
    main study reuses the pilot's class counts per fold, so every round of the pilot must
    hold the same counts fold by fold, in both views, as the pilot's partitions do (their
    stratification allocates the same counts whatever the seed).

    Args:
        summary: ``summary.csv``, every cell as text.
        folds: ``fold_values.csv``, every cell as text.
        settings: The simulation's settings.

    Returns:
        The numbers of each view, and the design.

    Raises:
        StepError: If a number is missing or not finite, a fold lacks a class, the rounds or
            views hold other class counts, or the pilot has another number of folds per round
            than the main study.
    """
    for name, table, columns in (
        ("summary.csv", summary, ("number", "view")),
        ("fold_values.csv", folds, ("round", "fold", "view", "positives", "negatives")),
    ):
        if any(column not in table.columns for column in columns):
            raise StepError(f"{name} must hold the columns {list(columns)}")
    layouts = {}
    for view in settings.views:
        rows = folds.loc[folds["view"] == view]
        if rows.empty:
            raise StepError(f"fold_values.csv holds no fold of {view}")
        rounds = _counts(rows, "round")
        numbers = _counts(rows, "fold")
        count_rounds, count_folds = int(rounds.max()), int(numbers.max())
        keys = sorted(zip(rounds.tolist(), numbers.tolist(), strict=True))
        if keys != [(r, f) for r in range(1, count_rounds + 1) for f in range(1, count_folds + 1)]:
            raise StepError(f"fold_values.csv must hold each fold of each round of {view} once")
        positive, negative = _counts(rows, "positives"), _counts(rows, "negatives")
        by_fold = {
            (int(r), int(f)): (int(p), int(n))
            for r, f, p, n in zip(rounds, numbers, positive, negative, strict=True)
        }
        layouts[view] = [
            [by_fold[(r, f)] for f in range(1, count_folds + 1)] for r in range(1, count_rounds + 1)
        ]
    first = layouts[settings.views[0]]
    if any(layout != first for layout in layouts.values()) or any(
        layout != first[0] for layout in first
    ):
        raise StepError(
            "fold_values.csv: every round and view must hold the same class counts fold by fold,"
            " which the main study reuses"
        )
    if len(first[0]) != settings.folds:
        raise StepError(
            f"{_FILE}: main.folds is {settings.folds}, but the pilot's rounds hold"
            f" {len(first[0])} folds, whose class counts the main study reuses"
        )
    if len(first) != 2:
        raise StepError(
            f"fold_values.csv holds {len(first)} rounds, not the pilot's two, whose difference"
            " the simulation reads"
        )
    design = Design(
        positives=tuple(p for p, _ in first[0]),
        negatives=tuple(n for _, n in first[0]),
        pilot_rounds=len(first),
    )
    per_round = [f"round_{number}" for number in range(1, design.pilot_rounds + 1)]
    pilots = {}
    for view in settings.views:
        pilot = PilotView(
            view=view,
            b0_auroc=_cell(summary, _B0, view, "mean"),
            b0_ends=(
                _cell(summary, _B0, view, f"low_{LEVEL}"),
                _cell(summary, _B0, view, f"high_{LEVEL}"),
            ),
            delta=_cell(summary, _DELTA, view, "mean"),
            delta_rounds=tuple(_cell(summary, _DELTA, view, column) for column in per_round),
            delta_se=_cell(summary, _DELTA, view, "se"),
            agreement=_cell(summary, _AGREEMENT, view, "mean"),
            agreement_ends=(
                _cell(summary, _AGREEMENT, view, f"low_{LEVEL}"),
                _cell(summary, _AGREEMENT, view, f"high_{LEVEL}"),
            ),
            aurocs={
                arm: tuple(_cell(summary, f"auroc_{arm}", view, column) for column in per_round)
                for arm in ARMS
            },
        )
        for auroc in (pilot.b0_auroc, *pilot.b0_ends):
            if not 0.0 < auroc < 1.0 or not 0.0 < auroc + pilot.delta < 1.0:
                raise StepError(f"summary.csv: B0's AUROC on {view} and CPG's must lie in (0, 1)")
            if not 0.0 < auroc + settings.gains[-1] < 1.0:
                raise StepError(
                    f"summary.csv: B0's AUROC on {view} plus each true gain must lie below 1"
                )
        pilots[view] = pilot
    return pilots, design


# The model ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Model:
    """One scenario of the model.

    Attributes:
        b0_auroc: B0's AUROC; CPG's is B0's plus the true gain.
        r: The correlation of B0's and CPG's stable parts.
        rho: The correlation of an arm's scores between two runs: the stable part's share of
            a score's variance within a class.
    """

    b0_auroc: float
    r: float
    rho: float


def class_shift(auroc: float) -> float:
    """Return the positive class's shift that gives an AUROC: √2·Φ⁻¹(AUROC).

    With unit variance in each class, a positive participant's score lies above a negative
    one's with probability Φ(shift / √2), which is the AUROC.

    Args:
        auroc: The AUROC, strictly between 0 and 1.

    Returns:
        The shift, in standard deviations of a class.

    Raises:
        ValueError: If the AUROC does not lie strictly between 0 and 1.
    """
    if not 0.0 < auroc < 1.0:
        raise ValueError(f"an AUROC must lie strictly between 0 and 1, not {auroc}")
    return math.sqrt(2.0) * NormalDist().inv_cdf(auroc)


def class_shifts(model: Model, gains: Sequence[float]) -> tuple[float, np.ndarray]:
    """Return B0's class shift and CPG's at each true gain.

    CPG's AUROC is B0's plus the true gain, so its shift is that of the sum
    (``class_shift``), not B0's shift plus the gain. Both ``arm_scores`` and the blocks
    take their shifts from here.

    Args:
        model: The scenario.
        gains: The true gains.

    Returns:
        B0's shift, and CPG's at each true gain, of shape (G,).

    Raises:
        ValueError: If an AUROC does not lie strictly between 0 and 1.
    """
    b0 = class_shift(model.b0_auroc)
    return b0, np.array([class_shift(model.b0_auroc + gain) for gain in gains], dtype=np.float64)


class People(NamedTuple):
    """The stable parts of a block's participants, before r is set: (count, N), standard normal.

    Attributes:
        stable: B0's stable part.
        other: An independent part, which CPG's stable part mixes in: r·stable + √(1 − r²)·other.
    """

    stable: np.ndarray
    other: np.ndarray


def simulate_people(rng: np.random.Generator, count: int, size: int) -> People:
    """Draw the participants of a block of studies: new people in every study.

    Args:
        rng: The block's stream.
        count: Studies (or pilots) in the block.
        size: Participants per study, N.

    Returns:
        Their stable parts.
    """
    return People(rng.standard_normal((count, size)), rng.standard_normal((count, size)))


def run_bases(
    model: Model, people: People, noise_b0: np.ndarray, noise_cpg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return both arms' scores in one fold-run, before the class shift.

    B0's is √ρ·stable + √(1 − ρ)·noise; CPG's stable part is r·stable + √(1 − r²)·other,
    with its own noise. A score is its base plus the class shift, which only the positive
    participants carry (``arm_scores``).

    Args:
        model: The scenario.
        people: The participants' stable parts.
        noise_b0: B0's run noise in this fold-run, standard normal, of shape (count, N).
        noise_cpg: CPG's.

    Returns:
        B0's and CPG's bases, each of shape (count, N).
    """
    cpg_stable = model.r * people.stable + math.sqrt(1.0 - model.r**2) * people.other
    stays, goes = math.sqrt(model.rho), math.sqrt(1.0 - model.rho)
    return stays * people.stable + goes * noise_b0, stays * cpg_stable + goes * noise_cpg


def arm_scores(
    model: Model,
    people: People,
    noise_b0: np.ndarray,
    noise_cpg: np.ndarray,
    gain: float,
    positives: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return both arms' scores in one fold-run: the base plus the class shift.

    B0's class shift is set by its AUROC and CPG's by B0's AUROC plus the true gain
    (``class_shifts``); participants 0 to ``positives`` − 1 are the positive ones. The
    blocks add the same shifts to the same bases, fold by fold, at every true gain at once.

    Args:
        model: The scenario.
        people: The participants' stable parts.
        noise_b0: B0's run noise in this fold-run, of shape (count, N).
        noise_cpg: CPG's.
        gain: The true gain.
        positives: The positive participants, P.

    Returns:
        B0's and CPG's scores, each of shape (count, N).
    """
    b0, cpg = run_bases(model, people, noise_b0, noise_cpg)
    b0_shift, cpg_shift = class_shifts(model, (gain,))
    b0[:, :positives] += b0_shift
    cpg[:, :positives] += cpg_shift[0]
    return b0, cpg


def stratified_folds(rng: np.random.Generator, count: int, design: Design) -> np.ndarray:
    """Lay out each study's participants by test fold for one round, stratified by label.

    The positive participants come first, in a random order, then the negative ones; fold f
    holds the positions of ``design.slices()[f]``, so every fold has the pilot's class
    counts, and the round's assignment is a random one with those counts.

    Args:
        rng: The block's stream.
        count: Studies in the block.
        design: The class counts per fold.

    Returns:
        Each study's participants in fold order, of shape (count, N).
    """
    positives, negatives = design.classes
    first = rng.permuted(np.tile(np.arange(positives), (count, 1)), axis=1)
    second = rng.permuted(np.tile(np.arange(positives, positives + negatives), (count, 1)), axis=1)
    return np.concatenate([first, second], axis=1)


def pair_wins(positives: np.ndarray, negatives: np.ndarray) -> np.ndarray:
    """Return doubled pair wins: 2 where the positive participant scores above, 1 on a tie, 0 below.

    Doubled so that a tie, which counts one half (pilot analysis, section "The main
    numbers"), is a whole number.

    Args:
        positives: The positive participants' scores, of shape (..., P).
        negatives: The negative participants' scores, of shape (..., N), broadcastable.

    Returns:
        The doubled wins, of shape (..., P, N), as int8.
    """
    above = positives[..., :, None] > negatives[..., None, :]
    level = positives[..., :, None] == negatives[..., None, :]
    return (2 * above.astype(np.int8) + level.astype(np.int8)).astype(np.int8)


def fold_agreement(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the Spearman correlation of two arms' scores within a fold, per study.

    The pilot analysis's agreement: the Pearson correlation of the average ranks, tied
    values sharing their average rank. The doubled ranks are whole numbers, so their sums,
    squares and products are exact; only the division and the square root round.

    Args:
        first: One arm's scores of the fold's participants, of shape (count, m).
        second: The other arm's, in the same order.

    Returns:
        The correlations, of shape (count,).

    Raises:
        ValueError: If an arm's scores do not vary within a fold.
    """
    x = 2.0 * rankdata(first, axis=1)
    y = 2.0 * rankdata(second, axis=1)
    size = x.shape[1]
    sum_x, sum_y = x.sum(axis=1), y.sum(axis=1)
    var_x = size * (x * x).sum(axis=1) - sum_x * sum_x
    var_y = size * (y * y).sum(axis=1) - sum_y * sum_y
    if (var_x <= 0).any() or (var_y <= 0).any():
        raise ValueError("an arm's scores do not vary within a fold: no Spearman correlation")
    cov = size * (x * y).sum(axis=1) - sum_x * sum_y
    return np.clip(cov / np.sqrt(var_x * var_y), -1.0, 1.0)


# Draws and the test ------------------------------------------------------------------------


def vector_draws(
    rng: np.random.Generator,
    classes: tuple[int, int],
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    resamples: int,
) -> tuple[np.ndarray, int]:
    """Draw a study's resamples by the pilot analysis's rule, many at a time.

    Pilot analysis, section "Intervals", bullet "Draws": in each resample, as many
    participants as each class holds are drawn with replacement from all its participants,
    and a participant drawn k times counts k times in their test fold of each round. A
    resample that leaves any test fold of any round without a positive or without a
    negative participant is drawn again and counted. The resamples are drawn as a batch,
    the positives of every resample first, then the negatives, and the batch's valid
    resamples are kept in their order.

    Args:
        rng: The block's stream.
        classes: The positive participants, 0 to P − 1, and the negative ones, which follow.
        folds: Every test fold of every round: its positive and its negative participants.
        resamples: Resamples to keep.

    Returns:
        How often each participant is drawn in each kept resample, of shape
        (resamples, N), as int64; and the number of resamples drawn again.

    Raises:
        ValueError: If fewer than one resample is asked for, or a fold lacks a class, which no
            resample could then fill.
    """
    if resamples < 1:
        raise ValueError(f"draw at least one resample, not {resamples}")
    if any(len(positives) == 0 or len(negatives) == 0 for positives, negatives in folds):
        raise ValueError("every test fold of every round must hold both classes")
    # Each column marks the positive or the negative participants of one fold, so that one
    # product gives every fold's drawn participants of each class; the sums are whole numbers
    # far below 2**24, exact in float32.
    members = np.zeros((sum(classes), 2 * len(folds)), dtype=np.float32)
    for column, (positives, negatives) in enumerate(folds):
        members[positives, 2 * column] = 1.0
        members[negatives, 2 * column + 1] = 1.0
    kept: list[np.ndarray] = []
    have = redraws = 0
    while have < resamples:
        want = resamples - have
        drawn = []
        for size in classes:  # the positives of every resample, then the negatives
            picks = rng.integers(0, size, size=(want, size)) + np.arange(want)[:, None] * size
            drawn.append(np.bincount(picks.ravel(), minlength=want * size).reshape(want, size))
        counts = np.concatenate(drawn, axis=1)
        valid = ((counts.astype(np.float32) @ members) > 0).all(axis=1)
        kept.append(counts[valid])
        have += int(valid.sum())
        redraws += int((~valid).sum())
    return np.concatenate(kept, axis=0).astype(np.int64), redraws


class FoldWins(NamedTuple):
    """One test fold of one round of a study, as the bootstrap sees it.

    Attributes:
        positives: The fold's positive participants, of shape (p,).
        negatives: Its negative participants, of shape (q,).
        b0: B0's doubled pair wins, of shape (p, q).
        cpg: CPG's at each true gain, of shape (G, p, q).
    """

    positives: np.ndarray
    negatives: np.ndarray
    b0: np.ndarray
    cpg: np.ndarray


def resampled_delta(counts: np.ndarray, folds: Sequence[FoldWins]) -> np.ndarray:
    """Return Δ = CPG − B0 on each resample and at each true gain.

    Pilot analysis, section "Intervals", bullet "Recomputing": every fold's AUROC is
    computed again on the participants drawn into it, each counted as often as drawn, the
    difference is taken fold by fold and the differences are averaged. A pair counts as
    often as the product of its two draw counts, so a resample in which every participant
    is drawn once gives the point Δ. A fold's sum over its pairs is a whole number of at most
    twice the square of a resample's draws, so it is exact in float32 while that stays below
    2**24 (float64 otherwise), in any order of the sums; only the last divisions round, in
    float64. The resamples go ``CHUNK`` at a time, so that the products stay in the cache.

    Args:
        counts: The draw counts (``vector_draws``), whole numbers of shape (R, N).
        folds: The study's test folds, with their pair wins.

    Returns:
        Δ, of shape (R, G).

    Raises:
        ValueError: If no fold is given.
    """
    if not folds:
        raise ValueError("Δ needs at least one fold")
    table = np.asarray(counts)
    largest = int(table.sum(axis=1).max()) if table.size else 0
    exact: np.dtype[Any] = np.dtype(np.float32 if 2 * largest * largest < 2**24 else np.float64)
    gains = folds[0].cpg.shape[0]
    prepared = []
    for fold in folds:
        _, p, q = fold.cpg.shape
        difference = fold.cpg.astype(exact) - fold.b0.astype(exact)[None, :, :]
        # Laid out as (q, G·p): the negatives' counts times it give, for each true gain,
        # each positive participant's summed wins over the drawn negatives.
        stacked = np.ascontiguousarray(difference.transpose(2, 0, 1).reshape(q, gains * p))
        prepared.append((fold.positives, fold.negatives, stacked, p))
    total = np.zeros((len(table), gains))
    for start in range(0, len(table), CHUNK):
        weights = table[start : start + CHUNK].astype(exact)
        for positives, negatives, stacked, p in prepared:
            positive, negative = weights[:, positives], weights[:, negatives]
            per_positive = (negative @ stacked).reshape(-1, gains, p)
            wins = np.einsum("rgp,rp->rg", per_positive, positive)
            pairs = (
                2.0
                * positive.sum(axis=1, dtype=np.float64)
                * negative.sum(axis=1, dtype=np.float64)
            )
            total[start : start + CHUNK] += wins.astype(np.float64) / pairs[:, None]
    return total / len(folds)


def passes(resampled: np.ndarray, alphas: Sequence[float]) -> np.ndarray:
    """Return whether the test says "CPG beats B0" at each α and true gain.

    Pilot diagnostics, section "The main study's power", bullet "The main study": the test
    says "CPG beats B0" if the α-quantile of the resampled Δ, with NumPy's linear
    quantiles, lies above 0.

    Args:
        resampled: Δ on each resample, of shape (R, G).
        alphas: The one-sided levels.

    Returns:
        The decisions, of shape (len(alphas), G).
    """
    quantiles = np.quantile(resampled, list(alphas), axis=0, method="linear")
    return np.asarray(quantiles > 0.0)


def green_pilot(
    b0_wins: np.ndarray,
    cpg_wins: np.ndarray,
    pairs: np.ndarray,
    min_auroc: Fraction,
    delta_at_least: Fraction,
) -> np.ndarray:
    """Return whether each simulated pilot is Green, compared exactly.

    Pilot protocol, sections "Feasibility checks" and "Decision rule": Green if B0's mean
    AUROC on ``x0`` is at least the bound of check F3 and Δ on ``x0`` at least the rule's.
    Both means are exact fractions of the doubled wins over a common denominator, so a
    value that lies on a bound is never a rounding artefact.

    Args:
        b0_wins: B0's doubled pair wins summed over each fold-run, of shape (count, J).
        cpg_wins: CPG's, at each true gain, of shape (count, G, J).
        pairs: Each fold-run's pairs of a positive and a negative participant, of shape (J,).
        min_auroc: Check F3's bound, as written.
        delta_at_least: The rule's bound on Δ, as written.

    Returns:
        The calls, of shape (count, G).

    Raises:
        ValueError: If the counts are too large for exact whole-number arithmetic.
    """
    sizes = [2 * int(pair) for pair in pairs]
    common = math.lcm(*sizes)
    runs = len(sizes)
    largest = max(abs(min_auroc.numerator), min_auroc.denominator)
    largest = max(largest, abs(delta_at_least.numerator), delta_at_least.denominator)
    if runs * common * largest >= 2**62:
        raise ValueError("the folds' pair counts are too large for exact whole-number arithmetic")
    scale = np.array([common // size for size in sizes], dtype=np.int64)
    b0 = (np.asarray(b0_wins, dtype=np.int64) * scale).sum(axis=-1)
    delta = ((np.asarray(cpg_wins, dtype=np.int64) - b0_wins[:, None, :]) * scale).sum(axis=-1)
    # B0's mean AUROC is b0 / (runs · common), Δ is delta / (runs · common).
    learns = b0 * min_auroc.denominator >= min_auroc.numerator * runs * common
    beats = delta * delta_at_least.denominator >= delta_at_least.numerator * runs * common
    return np.asarray(learns[:, None] & beats)


def fold_mean(wins: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Return the mean over fold-runs of the AUROCs (or their differences) from doubled wins.

    Args:
        wins: Doubled wins summed over each fold-run, with the fold-runs on the last axis.
        pairs: Each fold-run's pairs, of shape (J,).

    Returns:
        The means, over the last axis.
    """
    return (np.asarray(wins, dtype=np.float64) / (2.0 * pairs)).mean(axis=-1)


# Blocks -------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Task:
    """One block of simulated pilots or studies.

    Attributes:
        seed: The block's own stream (``runinfo.item_seed``).
        count: The pilots or studies of the block.
        model: The scenario.
        gains: The true gains, which share every draw.
        design: The pilot's class counts per fold and rounds.
        main_rounds: The main study's rounds; 0 for pilots alone.
        resamples: The main study's resamples per study; for pilots alone, the resamples of
            each pilot's bootstrap standard error of Δ at the first gain (0: none).
        alphas: The one-sided levels of the main study's test.
        first_resamples: For the resample check: the decisions also on the first ones of the
            resamples (0: none).
        agreement: Whether to compute each pilot's agreement at the first gain.
    """

    seed: int
    count: int
    model: Model
    gains: tuple[float, ...]
    design: Design
    main_rounds: int = 0
    resamples: int = 0
    alphas: tuple[float, ...] = ()
    first_resamples: int = 0
    agreement: bool = False


class Block(NamedTuple):
    """What one block gives, per pilot or study.

    Attributes:
        pilot_b0: B0's doubled pair wins in each of the pilot's fold-runs, round by round, of
            shape (count, J).
        pilot_cpg: CPG's, at each true gain, of shape (count, G, J).
        agreement: Each pilot's agreement at the first gain, of shape (count,), or None.
        pilot_se: Each pilot's bootstrap standard error of Δ at the first gain, or None.
        main_delta: The main study's Δ at each true gain, of shape (count, G), or None.
        decisions: Whether its test says "CPG beats B0", of shape (count, A, G), or None.
        first_decisions: The same on the first resamples only, or None.
        redraws: The resamples drawn again in the block.
    """

    pilot_b0: np.ndarray
    pilot_cpg: np.ndarray
    agreement: np.ndarray | None
    pilot_se: np.ndarray | None
    main_delta: np.ndarray | None
    decisions: np.ndarray | None
    first_decisions: np.ndarray | None
    redraws: int


class _FoldRun(NamedTuple):
    """One test fold of one round for every study of a block."""

    positives: np.ndarray  # (count, p) participants
    negatives: np.ndarray  # (count, q)
    b0: np.ndarray  # (count, p, q) doubled pair wins
    cpg: np.ndarray  # (count, G, p, q)
    b0_scores: np.ndarray  # (count, p + q) scores at the fold's positions, positives first
    cpg_scores: np.ndarray  # (count, p + q), at the first gain


def _fold_runs(
    rng: np.random.Generator,
    task: Task,
    people: People,
    rounds: int,
) -> list[_FoldRun]:
    """The fold-runs of ``rounds`` new rounds: folds, run noise, scores and pair wins."""
    model, design = task.model, task.design
    count = task.count
    rows = np.arange(count)[:, None]
    b0_shift, cpg_shifts = class_shifts(model, task.gains)
    found = []
    for _ in range(rounds):
        order = stratified_folds(rng, count, design)
        noise_b0 = rng.standard_normal((count, design.size))
        noise_cpg = rng.standard_normal((count, design.size))
        b0_base, cpg_base = run_bases(model, people, noise_b0, noise_cpg)
        for positive_slice, negative_slice in design.slices():
            positives, negatives = order[:, positive_slice], order[:, negative_slice]
            # A score is its base plus the class shift, which only the positives carry.
            b0_positive = b0_base[rows, positives] + b0_shift
            b0_negative = b0_base[rows, negatives]
            cpg_positive = cpg_base[rows, positives][:, None, :] + cpg_shifts[None, :, None]
            cpg_negative = cpg_base[rows, negatives]
            found.append(
                _FoldRun(
                    positives=positives,
                    negatives=negatives,
                    b0=pair_wins(b0_positive, b0_negative),
                    cpg=pair_wins(cpg_positive, cpg_negative[:, None, :]),
                    b0_scores=np.concatenate([b0_positive, b0_negative], axis=1),
                    cpg_scores=np.concatenate([cpg_positive[:, 0, :], cpg_negative], axis=1),
                )
            )
    return found


def _study_folds(runs: Sequence[_FoldRun], study: int, gains: slice) -> list[FoldWins]:
    """One study's fold-runs as the bootstrap sees them."""
    return [
        FoldWins(run.positives[study], run.negatives[study], run.b0[study], run.cpg[study, gains])
        for run in runs
    ]


def simulate_block(task: Task) -> Block:
    """Simulate one block of pilots or studies from its own stream.

    Each study draws new participants with their stable parts (``simulate_people``), then
    the pilot's rounds and the main study's rounds, each with a new stratified assignment to
    folds and new run noise for each arm; every true gain shares these draws, as do both
    arms but for their own noise. A study's main test draws its resamples by the pilot
    analysis's rule over the main study's test folds (``vector_draws``), computes Δ on each
    resample at every true gain (``resampled_delta``) and decides at each α (``passes``).
    The stream is used in this order: participants, the pilot's rounds, the main study's
    rounds, then the resamples study by study.

    Args:
        task: The block.

    Returns:
        Its numbers, per pilot or study.
    """
    rng = np.random.default_rng(task.seed)
    design = task.design
    people = simulate_people(rng, task.count, design.size)
    pilot_runs = _fold_runs(rng, task, people, design.pilot_rounds)
    main_runs = _fold_runs(rng, task, people, task.main_rounds)
    pilot_b0 = np.stack([run.b0.sum(axis=(1, 2), dtype=np.int64) for run in pilot_runs], axis=-1)
    pilot_cpg = np.stack([run.cpg.sum(axis=(2, 3), dtype=np.int64) for run in pilot_runs], axis=-1)
    agreement = None
    if task.agreement:
        agreement = np.mean(
            [fold_agreement(run.b0_scores, run.cpg_scores) for run in pilot_runs], axis=0
        )
    redraws = 0
    pilot_se = None
    if not task.main_rounds and task.resamples:
        pilot_se = np.empty(task.count)
        for study in range(task.count):
            folds = _study_folds(pilot_runs, study, slice(0, 1))
            counts, redrawn = vector_draws(
                rng,
                design.classes,
                [(fold.positives, fold.negatives) for fold in folds],
                task.resamples,
            )
            redraws += redrawn
            pilot_se[study] = np.std(resampled_delta(counts, folds)[:, 0], ddof=1)
    main_delta = decisions = first_decisions = None
    if task.main_rounds:
        pairs = np.tile(design.pairs, task.main_rounds)
        main_b0 = np.stack([run.b0.sum(axis=(1, 2), dtype=np.int64) for run in main_runs], -1)
        main_cpg = np.stack([run.cpg.sum(axis=(2, 3), dtype=np.int64) for run in main_runs], -1)
        main_delta = fold_mean(main_cpg - main_b0[:, None, :], pairs)
        decisions = np.empty((task.count, len(task.alphas), len(task.gains)), dtype=bool)
        if task.first_resamples:
            first_decisions = np.empty_like(decisions)
        for study in range(task.count):
            folds = _study_folds(main_runs, study, slice(None))
            counts, redrawn = vector_draws(
                rng,
                design.classes,
                [(fold.positives, fold.negatives) for fold in folds],
                task.resamples,
            )
            redraws += redrawn
            delta = resampled_delta(counts, folds)
            decisions[study] = passes(delta, task.alphas)
            if first_decisions is not None:
                first_decisions[study] = passes(delta[: task.first_resamples], task.alphas)
    return Block(
        pilot_b0=pilot_b0,
        pilot_cpg=pilot_cpg,
        agreement=agreement,
        pilot_se=pilot_se,
        main_delta=main_delta,
        decisions=decisions,
        first_decisions=first_decisions,
        redraws=redraws,
    )


def thread_pools() -> list[dict[str, Any]]:
    """Return the BLAS and OpenMP thread pools of this process, as threadpoolctl sees them.

    threadpoolctl sees the libraries that it knows, such as OpenBLAS, MKL and OpenMP; it
    cannot see Apple's Accelerate, so on a Mac the list may be empty.

    Returns:
        Each pool's ``POOL_KEYS``: its API, its library and its number of threads.
    """
    from threadpoolctl import threadpool_info

    return [{key: pool.get(key) for key in POOL_KEYS} for pool in threadpool_info()]


def check_one_thread(pools: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a worker whose thread pools do not run one thread each.

    Args:
        pools: The worker's thread pools (``thread_pools``).

    Raises:
        RuntimeError: If a pool runs another number of threads than 1.
    """
    wide = [
        f"{pool.get('internal_api')} with {pool.get('num_threads')} threads"
        for pool in pools
        if pool.get("num_threads") != 1
    ]
    if wide:
        raise RuntimeError("a worker's thread pools must run one thread each: " + ", ".join(wide))


def _one_blas_thread() -> None:
    """A worker's start: one BLAS thread, so that W workers use W cores and no more; checked."""
    from threadpoolctl import threadpool_limits

    _BLAS.append(threadpool_limits(limits=1))
    check_one_thread(thread_pools())


class Runner(Protocol):
    """What runs the blocks: their results in the order of the tasks."""

    def map(self, tasks: Sequence[Task]) -> list[Block]:
        """Simulate every block and return the blocks in the order of the tasks."""
        ...


class Workers:
    """Worker processes started by spawn, each with one BLAS thread; results in task order.

    Each block depends on its own stream only, so the results are the same for any number
    of workers. Use it as a context manager, which stops the workers on leaving. While it
    lives, the variables ``ONE_THREAD`` read 1, so that every worker, which starts when it
    is first needed, starts its BLAS and OpenMP libraries with one thread; each worker then
    limits them to one thread with threadpoolctl and checks that they run one, or the pool
    breaks and the job stops. On leaving, the variables get their values back.

    Args:
        count: Worker processes, at least 1.

    Raises:
        ValueError: If the count is below 1.
    """

    def __init__(self, count: int) -> None:
        if count < 1:
            raise ValueError(f"start at least one worker, not {count}")
        self.count = count
        self._context = get_context("spawn")
        self._pool: ProcessPoolExecutor | None = None
        self._saved: dict[str, str | None] = {}

    @property
    def start_method(self) -> str:
        """How the workers are started: spawn."""
        return self._context.get_start_method()

    def __enter__(self) -> "Workers":
        self._saved = {name: os.environ.get(name) for name in ONE_THREAD}
        os.environ.update(dict.fromkeys(ONE_THREAD, "1"))
        try:
            self._pool = ProcessPoolExecutor(
                max_workers=self.count,
                mp_context=self._context,
                initializer=_one_blas_thread,
            )
        except BaseException:
            self._restore()
            raise
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        try:
            if self._pool is not None:
                self._pool.shutdown(wait=True, cancel_futures=True)
                self._pool = None
        finally:
            self._restore()

    def _restore(self) -> None:
        """Give the variables ``ONE_THREAD`` back the values they had before."""
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._saved = {}

    def pools(self) -> list[dict[str, Any]]:
        """Return the thread pools of one worker, measured there (``thread_pools``).

        Returns:
            Each pool's API, library and number of threads.

        Raises:
            RuntimeError: If the workers have not been started (outside the ``with`` block).
        """
        if self._pool is None:
            raise RuntimeError("start the workers first: use Workers as a context manager")
        return self._pool.submit(thread_pools).result()

    def map(self, tasks: Sequence[Task]) -> list[Block]:
        """Simulate every block in the workers.

        Args:
            tasks: The blocks.

        Returns:
            Their results, in the order of the tasks.

        Raises:
            RuntimeError: If the workers have not been started (outside the ``with`` block).
        """
        if self._pool is None:
            raise RuntimeError("start the workers first: use Workers as a context manager")
        return list(self._pool.map(simulate_block, tasks))


def block_sizes(count: int, blocks: int) -> list[int]:
    """Split items into at most ``blocks`` fixed blocks, the first ones larger by one.

    Args:
        count: The items, at least 1.
        blocks: The blocks wanted, at least 1.

    Returns:
        The items of each block; never a block of none.
    """
    used = min(blocks, count)
    base, extra = divmod(count, used)
    return [base + (1 if block < extra else 0) for block in range(used)]


def _tasks(settings: PowerSettings, keys: tuple[str, str], count: int, **fields: Any) -> list[Task]:
    """The blocks of ``count`` pilots or studies, each with its stream item_seed(seed, *keys, b)."""
    return [
        Task(seed=item_seed(settings.seed, *keys, block), count=size, **fields)
        for block, size in enumerate(block_sizes(count, settings.blocks))
    ]


def _joined(blocks: Sequence[Block], field: str) -> np.ndarray:
    """One field of every block, joined in block order."""
    parts = [getattr(block, field) for block in blocks]
    if any(part is None for part in parts):
        raise ValueError(f"a block holds no {field}")
    return np.concatenate(parts, axis=0)


# The checks that come first -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DesignCheck:
    """The model on the study design's assumptions, against the design's figures.

    Attributes:
        pilot_sd: The standard deviation of the simulated pilot's Δ in each scenario.
        one_arm_sd: One arm's one-round AUROC standard deviation in the middle scenario.
        misses: The figures that the model misses by more than the gap allowed, as text.
    """

    pilot_sd: tuple[float, ...]
    one_arm_sd: float
    misses: tuple[str, ...]


def design_check(settings: PowerSettings, design: Design, runner: Runner) -> DesignCheck:
    """Check the model against the study design's figures, on its assumptions only.

    Pilot diagnostics, section "The main study's power", bullet "Checks first", check 1: at
    B0's AUROC of 0.65 and a true gain of 0, in the design's three scenarios, the model
    must reproduce the standard deviation of the pilot's Δ and, in the middle scenario,
    one arm's one-round AUROC standard deviation (B0's mean AUROC over round 1's folds),
    each within the gap allowed. The pilots run in the pilot's design, with the streams
    item_seed(seed, "design_check", scenario, block).

    Args:
        settings: The simulation's settings.
        design: The pilot's design.
        runner: What runs the blocks.

    Returns:
        The model's figures and the ones that miss; the job stops on a miss, as a code fault.
    """
    pairs = np.tile(design.pairs, design.pilot_rounds)
    folds = len(design.positives)
    sds, misses = [], []
    one_arm = math.nan
    for name, (r, rho), expected in zip(
        DESIGN_NAMES, settings.design_scenarios, settings.design_pilot_sd, strict=True
    ):
        model = Model(settings.design_auroc, r, rho)
        blocks = runner.map(
            _tasks(
                settings,
                ("design_check", name),
                settings.design_pilots,
                model=model,
                gains=(0.0,),
                design=design,
            )
        )
        b0, cpg = _joined(blocks, "pilot_b0"), _joined(blocks, "pilot_cpg")[:, 0, :]
        sd = float(np.std(fold_mean(cpg - b0, pairs), ddof=1))
        sds.append(sd)
        if abs(sd - expected) > settings.design_max_gap:
            misses.append(f"the pilot's Δ in the {name} scenario: SD {sd:.4f}, not {expected}")
        if name == "middle":
            one_arm = float(np.std(fold_mean(b0[:, :folds], pairs[:folds]), ddof=1))
            if abs(one_arm - settings.design_one_arm_sd) > settings.design_max_gap:
                misses.append(
                    f"one arm's one-round AUROC in the middle scenario: SD {one_arm:.4f}, not"
                    f" {settings.design_one_arm_sd}"
                )
    return DesignCheck(tuple(sds), one_arm, tuple(misses))


@dataclass(frozen=True, slots=True)
class ResampleCheck:
    """The decisions at the simulation's resamples against those at the check's.

    Attributes:
        changed: The share of studies whose decision changes, of shape (A, G) for the check's
            gains.
        passed: Whether every share is at most the one allowed.
        resamples: The resamples per study that every table then uses.
        redraws: The resamples drawn again.
    """

    changed: np.ndarray
    passed: bool
    resamples: int
    redraws: int


def resample_check(settings: PowerSettings, design: Design, runner: Runner) -> ResampleCheck:
    """Check that the simulation's resamples decide as the check's larger number does.

    Pilot diagnostics, section "The main study's power", bullet "Checks first", check 2: on
    the check's studies of the study design's middle scenario, at each of its true gains,
    the decisions on the first resamples of the simulation's number and on all of the
    check's may differ in at most the share allowed at each α; otherwise every table uses
    the fallback. The streams are item_seed(seed, "resample_check", "middle", block).

    Args:
        settings: The simulation's settings.
        design: The pilot's design.
        runner: What runs the blocks.

    Returns:
        The shares, the verdict and the resamples that every table uses.
    """
    r, rho = settings.design_scenarios[DESIGN_NAMES.index("middle")]
    blocks = runner.map(
        _tasks(
            settings,
            ("resample_check", "middle"),
            settings.check_studies,
            model=Model(settings.design_auroc, r, rho),
            gains=settings.check_gains,
            design=design,
            main_rounds=settings.rounds,
            resamples=settings.check_resamples,
            alphas=settings.alphas,
            first_resamples=settings.resamples,
        )
    )
    changed = (_joined(blocks, "decisions") != _joined(blocks, "first_decisions")).mean(axis=0)
    passed = bool((changed <= settings.max_changed).all())
    return ResampleCheck(
        changed=changed,
        passed=passed,
        resamples=settings.resamples if passed else settings.fallback_resamples,
        redraws=sum(block.redraws for block in blocks),
    )


# Fits ---------------------------------------------------------------------------------------


class Fit(NamedTuple):
    """A parameter found by a fit, and the model's value there.

    Attributes:
        value: The parameter, in [0, 1].
        reached: The model's value at it.
        met: Whether the value meets the target within the tolerance; a sensitivity row
            that misses keeps its closest point, marked not met.
    """

    value: float
    reached: float
    met: bool = True


def _search(
    value_at: Callable[[float], float],
    target: float,
    tolerance: float,
    what: str,
    stop: bool = True,
) -> Fit:
    """Bisection on [0, 1] for a model value that rises or falls with its parameter.

    With ``stop`` False, a miss returns the closest point found, marked not met, instead of
    stopping the job.
    """
    low, high = 0.0, 1.0
    at_low, at_high = value_at(low), value_at(high)
    rising = at_high >= at_low
    lowest, highest = min(at_low, at_high), max(at_low, at_high)
    best = min(((low, at_low), (high, at_high)), key=lambda pair: abs(pair[1] - target))
    if not lowest - tolerance <= target <= highest + tolerance:
        if not stop:
            return Fit(*best, met=False)
        raise StepError(
            f"{what}: the model reaches {lowest:.4f} to {highest:.4f}, which misses the target"
            f" {target:.4f} by more than {tolerance:g}, so the job stops ({_SOURCE})"
        )
    for _ in range(SEARCH_STEPS):
        if abs(best[1] - target) <= CLOSE * tolerance:
            break
        middle = (low + high) / 2.0
        value = value_at(middle)
        if abs(value - target) < abs(best[1] - target):
            best = (middle, value)
        if (value < target) == rising:
            low = middle
        else:
            high = middle
    if abs(best[1] - target) > tolerance:
        if not stop:
            return Fit(*best, met=False)
        raise StepError(
            f"{what}: the closest fit reaches {best[1]:.6f}, which misses the target"
            f" {target:.6f} by more than {tolerance:g}, so the job stops ({_SOURCE})"
        )
    return Fit(*best)


def calibrate_r(
    value_at: Callable[[float], float],
    target: float,
    tolerance: float,
    what: str,
    stop: bool = True,
) -> Fit:
    """Find r in [0, 1] at which the model's value meets a measured one.

    Pilot diagnostics, section "The main study's power", bullet "Which number sets what":
    the model's mean agreement in the pilot's design must meet the measured one within the
    tolerance, or the job stops. A sensitivity row (the sensitivity row "se_matched" fits r
    to Δ's bootstrap standard error instead) that cannot meet its target within the
    tolerance is reported as not reached, with r where the model comes closest, and the job
    goes on: no decision reads these rows. The value must rise or fall with r, as both do.
    The fit halves its bracket until a point lies within a tenth of the tolerance
    (``CLOSE``), or ``SEARCH_STEPS`` times, and keeps the closest point.

    Args:
        value_at: The model's value at a given r, from fixed draws.
        target: The measured value.
        tolerance: How close the fit must come.
        what: What is fitted, for the refusal.
        stop: True for a fit at a ρ point, whose miss stops the job; False for a
            sensitivity row, whose miss returns the closest point, marked not met.

    Returns:
        r, the model's value there, and whether it meets the target.

    Raises:
        StepError: If ``stop`` is True and no r in [0, 1] meets the target within the
            tolerance.
    """
    return _search(value_at, target, tolerance, what, stop)


def lowest_rho(
    value_at: Callable[[float], float], target: float, tolerance: float, what: str
) -> Fit:
    """Find the smallest ρ that the measured agreement allows: the ρ at which r = 1 meets it.

    Pilot diagnostics, section "The main study's power", bullet "Which number sets what": ρ
    takes equal steps from the smallest value that the agreement allows (r = 1: all
    disagreement is run noise) to 1 (none is).

    Args:
        value_at: The model's agreement at a given ρ, with r = 1, from fixed draws.
        target: The measured agreement.
        tolerance: How close the fit must come.
        what: What is fitted, for the refusal.

    Returns:
        The lowest ρ and the model's agreement there.

    Raises:
        StepError: If no ρ in [0, 1] meets the agreement within the tolerance.
    """
    return _search(value_at, target, tolerance, what)


@dataclass(frozen=True, slots=True)
class Scenario:
    """A scenario of one view, fitted in the pilot's design.

    Attributes:
        view: The view.
        name: The scenario: rho_1 to rho_K, or a sensitivity row.
        model: B0's AUROC, r and ρ.
        fitted_to: What r is fitted to: the agreement, an end of its 80% interval, or Δ's
            bootstrap standard error.
        target: The value that r is fitted to.
        agreement: The model's mean agreement in the pilot's design.
        bootstrap_se: The model's mean bootstrap standard error of Δ there.
        round_share: The share of simulated pilots whose two rounds' Δ lie at least as far
            apart as the pilot's.
        redraws: The resamples drawn again for the bootstrap standard errors.
        target_met: Whether r meets its target within the tolerance; only a sensitivity
            row can miss it, and is then reported as not reached.
    """

    view: str
    name: str
    model: Model
    fitted_to: str
    target: float
    agreement: float
    bootstrap_se: float
    round_share: float
    redraws: int
    target_met: bool = True


class _Pilots:
    """The calibration pilots of one view: fixed streams, evaluated at any model."""

    def __init__(
        self, view: str, pilot: PilotView, settings: PowerSettings, design: Design, runner: Runner
    ) -> None:
        self.view, self.pilot, self.settings = view, pilot, settings
        self.design, self.runner = design, runner

    def run(self, model: Model, resamples: int) -> list[Block]:
        """The pilots at the pilot's Δ as the true gain; the bootstrap if resamples > 0."""
        return self.runner.map(
            _tasks(
                self.settings,
                (self.view, "calibration"),
                self.settings.calibration_pilots,
                model=model,
                gains=(self.pilot.delta,),
                design=self.design,
                resamples=resamples,
                agreement=True,
            )
        )

    def agreement(self, model: Model) -> float:
        """The model's mean agreement."""
        return float(np.mean(_joined(self.run(model, 0), "agreement")))

    def bootstrap_se(self, model: Model, resamples: int) -> float:
        """The model's mean bootstrap standard error of Δ."""
        return float(np.mean(_joined(self.run(model, resamples), "pilot_se")))

    def scenario(
        self,
        name: str,
        model: Model,
        fitted_to: str,
        target: float,
        resamples: int,
        target_met: bool = True,
    ) -> Scenario:
        """Every number of a fitted scenario in the pilot's design."""
        blocks = self.run(model, resamples)
        pairs = self.design.pairs
        folds = len(pairs)
        b0, cpg = _joined(blocks, "pilot_b0"), _joined(blocks, "pilot_cpg")[:, 0, :]
        rounds = [
            fold_mean(
                cpg[:, k * folds : (k + 1) * folds] - b0[:, k * folds : (k + 1) * folds], pairs
            )
            for k in range(2)
        ]
        apart = np.abs(rounds[0] - rounds[1]) >= abs(self.pilot.round_difference)
        return Scenario(
            view=self.view,
            name=name,
            model=model,
            fitted_to=fitted_to,
            target=target,
            agreement=float(np.mean(_joined(blocks, "agreement"))),
            bootstrap_se=float(np.mean(_joined(blocks, "pilot_se"))),
            round_share=float(np.mean(apart)),
            redraws=sum(block.redraws for block in blocks),
            target_met=target_met,
        )


def fit_view(
    pilot: PilotView, settings: PowerSettings, design: Design, runner: Runner, resamples: int
) -> list[Scenario]:
    """Fit the scenarios of one view in the pilot's design, at its B0 AUROC and Δ.

    Pilot diagnostics, section "The main study's power", bullet "Which number sets what":
    the lowest ρ is the one at which r = 1 meets the measured agreement; ρ then takes equal
    steps from it to 1, and r is fitted at each point to the agreement (at the lowest ρ, r is
    1). At the middle ρ come the sensitivity rows: r fitted to each end of the agreement's
    80% interval; B0's AUROC at each end of its 80% interval, with r fitted to the agreement;
    and r fitted to Δ's bootstrap standard error instead. Every fit uses the same
    calibration pilots, the streams item_seed(seed, view, "calibration", block). A fit at a
    ρ point that misses its target stops the job; a sensitivity row that misses keeps r
    where the model comes closest and is reported as not reached.

    Args:
        pilot: The pilot's numbers on the view.
        settings: The simulation's settings.
        design: The pilot's design.
        runner: What runs the blocks.
        resamples: The resamples of each pilot's bootstrap standard error.

    Returns:
        The scenarios: rho_1 to rho_K, then the sensitivity rows.

    Raises:
        StepError: If the lowest ρ or a fit at a ρ point misses its target by more than the
            tolerance.
    """
    view, tolerance = pilot.view, settings.tolerance
    pilots = _Pilots(view, pilot, settings, design, runner)
    b0 = pilot.b0_auroc

    def agreement_at(auroc: float, rho: float) -> Callable[[float], float]:
        """The model's agreement as r varies, at one B0 AUROC and ρ."""
        return lambda r: pilots.agreement(Model(auroc, r, rho))

    floor = lowest_rho(
        lambda rho: pilots.agreement(Model(b0, 1.0, rho)),
        pilot.agreement,
        tolerance,
        f"{view}: the lowest ρ for the agreement",
    )
    steps = settings.rho_points - 1
    rhos = [floor.value + (1.0 - floor.value) * k / steps for k in range(steps)] + [1.0]
    fits: list[tuple[str, Model, str, float, bool]] = [
        ("rho_1", Model(b0, 1.0, rhos[0]), AGREEMENT, pilot.agreement, True)
    ]
    for point, rho in enumerate(rhos[1:], start=2):
        what = f"{view}, rho_{point}: r for the agreement"
        fit = calibrate_r(agreement_at(b0, rho), pilot.agreement, tolerance, what)
        fits.append((f"rho_{point}", Model(b0, fit.value, rho), AGREEMENT, pilot.agreement, True))
    middle = rhos[steps // 2]
    for end, target in zip(("low", "high"), pilot.agreement_ends, strict=True):
        name = f"{AGREEMENT}_{end}_{LEVEL}"
        what = f"{view}, {name}: r for the agreement's {end} end"
        fit = calibrate_r(agreement_at(b0, middle), target, tolerance, what, stop=False)
        fits.append((name, Model(b0, fit.value, middle), name, target, fit.met))
    for end, auroc in zip(("low", "high"), pilot.b0_ends, strict=True):
        name = f"b0_auroc_{end}_{LEVEL}"
        what = f"{view}, {name}: r for the agreement"
        fit = calibrate_r(agreement_at(auroc, middle), pilot.agreement, tolerance, what, stop=False)
        fits.append((name, Model(auroc, fit.value, middle), AGREEMENT, pilot.agreement, fit.met))
    fit = calibrate_r(
        lambda r: pilots.bootstrap_se(Model(b0, r, middle), resamples),
        pilot.delta_se,
        tolerance,
        f"{view}, se_matched: r for Δ's bootstrap standard error",
        stop=False,
    )
    fits.append(("se_matched", Model(b0, fit.value, middle), BOOTSTRAP_SE, pilot.delta_se, fit.met))
    return [
        pilots.scenario(name, model, fitted_to, target, resamples, target_met=met)
        for name, model, fitted_to, target, met in fits
    ]


# The simulated studies ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Outcome:
    """The simulated studies of one view and scenario.

    Attributes:
        view: The view, or ``DESIGN`` for the study design's scenarios.
        name: The scenario.
        model: B0's AUROC, r and ρ.
        studies: The simulated studies.
        passes: How many say "CPG beats B0", of shape (A, G).
        green: How many pilots are Green, at each true gain, of shape (G,).
        green_passes: How many say "CPG beats B0" after a Green pilot, of shape (A, G).
        main_se: The standard deviation of the main study's Δ at a true gain of 0.
        correlation: The correlation of the pilot's and the main study's Δ there.
        redraws: The resamples drawn again.
    """

    view: str
    name: str
    model: Model
    studies: int
    passes: np.ndarray
    green: np.ndarray
    green_passes: np.ndarray
    main_se: float
    correlation: float
    redraws: int

    def power(self) -> np.ndarray:
        """The power over new samples of people, of shape (A, G)."""
        return self.passes / self.studies


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Pearson's correlation from sums, without BLAS, so that it never depends on threads."""
    a, b = first - first.mean(), second - second.mean()
    return float((a * b).sum() / math.sqrt(float((a * a).sum() * (b * b).sum())))


def simulate_scenario(
    view: str,
    name: str,
    model: Model,
    settings: PowerSettings,
    design: Design,
    runner: Runner,
    resamples: int,
) -> Outcome:
    """Simulate the studies of one scenario at every true gain, with the same draws for each.

    Pilot diagnostics, section "The main study's power", bullets "The main study", "The
    pilot inside each study" and "Grid": each study draws new people, runs the pilot on them
    (Green, compared exactly by ``green_pilot``) and the main study's rounds, and tests Δ by
    the paired participant bootstrap at each α. The streams are item_seed(seed, view,
    scenario, block).

    Args:
        view: The view, or ``DESIGN``.
        name: The scenario.
        model: B0's AUROC, r and ρ.
        settings: The simulation's settings.
        design: The pilot's design.
        runner: What runs the blocks.
        resamples: The resamples per study.

    Returns:
        The counts and the numbers at a true gain of 0; the correlation is NaN if either Δ
        never varies.
    """
    blocks = runner.map(
        _tasks(
            settings,
            (view, name),
            settings.studies,
            model=model,
            gains=settings.gains,
            design=design,
            main_rounds=settings.rounds,
            resamples=resamples,
            alphas=settings.alphas,
        )
    )
    decisions = _joined(blocks, "decisions")
    b0, cpg = _joined(blocks, "pilot_b0"), _joined(blocks, "pilot_cpg")
    pairs = np.tile(design.pairs, design.pilot_rounds)
    green = green_pilot(b0, cpg, pairs, settings.green_auroc, settings.green_delta)
    zero = settings.gains.index(0.0)
    main = _joined(blocks, "main_delta")[:, zero]
    pilot = fold_mean(cpg[:, zero, :] - b0, pairs)
    if np.ptp(main) == 0.0 or np.ptp(pilot) == 0.0:
        correlation = math.nan
    else:
        correlation = _correlation(pilot, main)
    return Outcome(
        view=view,
        name=name,
        model=model,
        studies=len(decisions),
        passes=decisions.sum(axis=0),
        green=green.sum(axis=0),
        green_passes=(decisions & green[:, None, :]).sum(axis=0),
        main_se=float(np.std(main, ddof=1)),
        correlation=correlation,
        redraws=sum(block.redraws for block in blocks),
    )


def fit_views(
    pilots: Mapping[str, PilotView],
    settings: PowerSettings,
    design: Design,
    runner: Runner,
    resamples: int,
    report: Callable[[Scenario], None] | None = None,
) -> list[Scenario]:
    """Fit the scenarios of every view, before any study is simulated.

    The fits at the ρ points can stop the job (``fit_view``), so every fit comes first: a
    fit that misses then stops the job before any study is simulated. Each block has its
    own stream, so this order changes no number.

    Args:
        pilots: The pilot's numbers by view.
        settings: The simulation's settings.
        design: The pilot's design.
        runner: What runs the blocks.
        resamples: The resamples of each calibration pilot's bootstrap standard error
            (``ResampleCheck.resamples``).
        report: Called with each scenario once fitted.

    Returns:
        The scenarios, view by view in the settings' order.

    Raises:
        StepError: If the lowest ρ or a fit at a ρ point misses its target by more than the
            tolerance.
    """
    scenarios = []
    for view in settings.views:
        fitted = fit_view(pilots[view], settings, design, runner, resamples)
        for scenario in fitted:
            if report is not None:
                report(scenario)
        scenarios += fitted
    return scenarios


def simulate_views(
    scenarios: Sequence[Scenario],
    settings: PowerSettings,
    design: Design,
    runner: Runner,
    resamples: int,
    report: Callable[[Outcome], None] | None = None,
) -> list[Outcome]:
    """Simulate the studies of every fitted scenario.

    Args:
        scenarios: The fitted scenarios (``fit_views``).
        settings: The simulation's settings.
        design: The pilot's design.
        runner: What runs the blocks.
        resamples: The resamples per study (``ResampleCheck.resamples``).
        report: Called with each outcome once simulated.

    Returns:
        The outcomes, in the order of the scenarios.
    """
    outcomes = []
    for scenario in scenarios:
        outcome = simulate_scenario(
            scenario.view, scenario.name, scenario.model, settings, design, runner, resamples
        )
        if report is not None:
            report(outcome)
        outcomes.append(outcome)
    return outcomes


def design_outcomes(
    settings: PowerSettings, design: Design, runner: Runner, resamples: int
) -> list[Outcome]:
    """Simulate the main study in the study design's three scenarios, at its B0 AUROC.

    Pilot diagnostics, section "The main study's power", bullet "Checks first": the study
    design's figures for the main study are printed beside these, not checked, because it
    does not name their test. The streams are item_seed(seed, "design", scenario, block).

    Args:
        settings: The simulation's settings.
        design: The pilot's design.
        runner: What runs the blocks.
        resamples: The resamples per study.

    Returns:
        One outcome per scenario: low, middle and high noise.
    """
    return [
        simulate_scenario(
            DESIGN, name, Model(settings.design_auroc, r, rho), settings, design, runner, resamples
        )
        for name, (r, rho) in zip(DESIGN_NAMES, settings.design_scenarios, strict=True)
    ]


# Tables -------------------------------------------------------------------------------------


def alpha_name(alpha: float) -> str:
    """An α as the columns name it: 0.01667, 0.025, 0.05."""
    return f"{alpha:.4g}"


def gain_for_power(power: np.ndarray, gains: Sequence[float]) -> float | str:
    """Return the true gain at which the power first reaches 80%, linear between grid points.

    Pilot diagnostics, section "The main study's power", bullet "Outputs".

    Args:
        power: The power at each true gain, of shape (G,).
        gains: The true gains, ascending.

    Returns:
        The gain, or "above" the last gain as text if the power never reaches 80%.
    """
    for index, value in enumerate(power):
        if value >= TARGET_POWER:
            if index == 0:
                return float(gains[0])
            low, high = float(power[index - 1]), float(value)
            step = gains[index] - gains[index - 1]
            return float(gains[index - 1] + (TARGET_POWER - low) / (high - low) * step)
    return f"above {gains[-1]:.2f}"


def _standard_error(share: float, count: int) -> float:
    """A share's Monte Carlo standard error: √(p(1 − p)/n)."""
    return math.sqrt(share * (1.0 - share) / count)


def power_table(outcomes: Sequence[Outcome], settings: PowerSettings) -> pd.DataFrame:
    """power.csv: per view, scenario, true gain and α, the power with its Monte Carlo SE.

    Pilot diagnostics, section "The main study's power", bullet "Outputs": the power over new
    samples of people and, on ``x0``, after a Green pilot, with the number of Green pilots;
    on ``rev``, which is not conditioned, those cells are "not applicable", and a gain with
    no Green pilot reads "not defined (no Green pilot)".

    Args:
        outcomes: The simulated studies of every view and scenario.
        settings: The simulation's settings.

    Returns:
        One row per view, scenario, true gain and α.
    """
    rows = []
    for outcome in outcomes:
        power = outcome.power()
        for gain_index, gain in enumerate(settings.gains):
            for alpha_index, alpha in enumerate(settings.alphas):
                share = float(power[alpha_index, gain_index])
                cells: dict[str, Any] = {
                    "view": outcome.view,
                    "scenario": outcome.name,
                    "b0_auroc": outcome.model.b0_auroc,
                    "rho": outcome.model.rho,
                    "r": outcome.model.r,
                    "gain": gain,
                    "alpha": alpha,
                    "studies": outcome.studies,
                    "power": share,
                    "power_se": _standard_error(share, outcome.studies),
                }
                if outcome.view != GREEN_VIEW:
                    green: Any = NOT_APPLICABLE
                    after: Any = NOT_APPLICABLE
                    after_se: Any = NOT_APPLICABLE
                elif not outcome.green[gain_index]:
                    green, after, after_se = 0, NO_GREEN, NO_GREEN
                else:
                    green = int(outcome.green[gain_index])
                    after = float(outcome.green_passes[alpha_index, gain_index] / green)
                    after_se = _standard_error(after, green)
                cells.update(
                    green_pilots=green, power_after_green=after, power_after_green_se=after_se
                )
                rows.append(cells)
    return pd.DataFrame(rows)


def fit_table(
    scenarios: Sequence[Scenario],
    outcomes: Sequence[Outcome],
    pilots: Mapping[str, PilotView],
    settings: PowerSettings,
) -> pd.DataFrame:
    """power_fit.csv: per view and scenario, the fit beside the measured numbers.

    Pilot diagnostics, section "The main study's power", bullet "Outputs": r; the model's
    agreement and mean bootstrap standard error beside the measured ones; the share of the
    round difference; the simulated main study's standard error (the standard deviation of
    its Δ at a true gain of 0) beside the pilot's measured one; the correlation of the
    pilot's and the main study's Δ; the true gain needed for 80% power at each α; and the
    arms' per-round AUROCs beside.

    Args:
        scenarios: The fitted scenarios.
        outcomes: Their simulated studies, in the same order.
        pilots: The pilot's numbers by view.
        settings: The simulation's settings.

    Returns:
        One row per view and scenario.

    Raises:
        ValueError: If the scenarios and the outcomes do not match.
    """
    if [(s.view, s.name) for s in scenarios] != [(o.view, o.name) for o in outcomes]:
        raise ValueError("every scenario needs its outcome, in the same order")
    rows = []
    for scenario, outcome in zip(scenarios, outcomes, strict=True):
        pilot = pilots[scenario.view]
        cells: dict[str, Any] = {
            "view": scenario.view,
            "scenario": scenario.name,
            "b0_auroc": scenario.model.b0_auroc,
            "rho": scenario.model.rho,
            "r": scenario.model.r,
            "fitted_to": scenario.fitted_to,
            "target": scenario.target,
            "target_met": MET if scenario.target_met else NOT_REACHED,
            "agreement_model": scenario.agreement,
            "agreement_measured": pilot.agreement,
            "bootstrap_se_model": scenario.bootstrap_se,
            "bootstrap_se_measured": pilot.delta_se,
            "main_study_se": outcome.main_se,
            "pilot_delta": pilot.delta,
            "round_difference_measured": pilot.round_difference,
            "round_difference_share": scenario.round_share,
            "pilot_main_correlation": (
                outcome.correlation if math.isfinite(outcome.correlation) else "not defined"
            ),
        }
        power = outcome.power()
        for index, alpha in enumerate(settings.alphas):
            cells[f"gain_80_at_{alpha_name(alpha)}"] = gain_for_power(power[index], settings.gains)
        for arm in ARMS:
            for number, value in enumerate(pilot.aurocs[arm], start=1):
                cells[f"auroc_{arm}_round_{number}"] = value
        rows.append(cells)
    return pd.DataFrame(rows)
