"""The gain-term summary: what training logged about the gain term, over the kept runs.

Pilot diagnostics, section "Gain-term summary" (``docs/pilot_diagnostics.md``): the measures
that training logged per epoch (pilot protocol, section "Training", bullet "Logged
measures"), as each kept run's record holds them (``results.rules.epochs``, written by
``training.rule_measures``), summarised over the 10 CPG runs, with the 10 B1 runs beside,
which compute the gain term with weight 0, and B0's 10 runs for the measures that B0 logs:
CE(x0), the CE push and the statistics of the x0 bag means. ``gain_term_table`` gives, per
arm, epoch and measure, the mean, the minimum and the maximum over the runs and the number
of runs with a value, and per epoch also the pooled active share: the summed active bags
over the summed bags, as for check F2 (pilot analysis, section "Checks first"). New here,
for x0 and x1: the gap between the measured mean of the bag means and the average μ, and
the ratio of their measured spread to the average σ. B1's gain push and its offset are 0 by
construction and are left out.

No rule reads the summary. It reads only the records' per-epoch measures, never a score, a
step log or a label, and it holds no participant ID and no number per participant or
segment. Nothing here prints.
"""

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final, NamedTuple

import pandas as pd

from listening_beyond_words.evaluation import TRAINED_ARMS
from listening_beyond_words.runinfo import StepError

GAIN_ARMS: Final = ("b1", "cpg")  # the arms that compute the gain term (B1 with λG = 0)
COLUMNS: Final = ("arm", "epoch", "measure", "runs", "mean", "minimum", "maximum", "pooled_share")
# A cell that cannot hold a number reads as text, never blank (reports.write_table).
NO_VALUE: Final = "not available (no run has a value)"
NO_BAG: Final = "not available (no bag)"
NOT_APPLICABLE: Final = "not applicable"
POOLED: Final = "active_share_all"  # the measure whose row carries the pooled share
LOGGED: Final = "logged"  # a measure as the record holds it
GAP: Final = "gap"  # the measured mean of the bag means minus the average μ
RATIO: Final = "ratio"  # the measured spread of the bag means over the average σ
_EPOCH: Final = re.compile(r"[1-9][0-9]*")  # an epoch's key in a record, from 1


class Measure(NamedTuple):
    """One measure of an epoch, as the summary reads it from a record.

    Attributes:
        name: Its name in the table.
        keys: Where it lies in an epoch's measures (``training.rule_measures``): the nested
            keys of the value, or for a gap or a ratio those of its view's block.
        arms: The arms whose rows it has.
        kind: ``LOGGED``, ``GAP`` or ``RATIO``.
    """

    name: str
    keys: tuple[str, ...]
    arms: tuple[str, ...]
    kind: str = LOGGED


def _view(view: str) -> tuple[Measure, ...]:
    """The statistics of a view's bag means: measured beside the average μ and σ."""
    arms = TRAINED_ARMS if view == "x0" else GAIN_ARMS
    return (
        Measure(f"{view}_measured_mean", (view, "measured_mean"), arms),
        Measure(f"{view}_mu", (view, "mu"), GAIN_ARMS),
        Measure(f"{view}_mean_minus_mu", (view,), GAIN_ARMS, GAP),
        Measure(f"{view}_measured_spread", (view, "measured_spread"), arms),
        Measure(f"{view}_sigma", (view, "sigma"), GAIN_ARMS),
        Measure(f"{view}_spread_over_sigma", (view,), GAIN_ARMS, RATIO),
    )


# The measures in the page's order (pilot diagnostics, section "Gain-term summary", bullets
# "What" and "Per epoch"): B0 logs no gain, so its rows are those of the measures that it
# logs, CE(x0), the statistics of the x0 bag means and the CE push; B1 computes the gain
# with λG = 0, so its gain push and offset are 0 by construction and left out.
MEASURES: Final = (
    Measure("loss", ("loss",), GAIN_ARMS),
    Measure("ce_x0", ("ce_x0",), TRAINED_ARMS),
    Measure("ce_x1", ("ce_x1",), GAIN_ARMS),
    Measure("ce_x2", ("ce_x2",), GAIN_ARMS),
    Measure("ce_part", ("ce_part",), GAIN_ARMS),
    Measure("hinge", ("hinge",), GAIN_ARMS),
    Measure(POOLED, ("active_share", "all"), GAIN_ARMS),
    Measure("active_share_positive", ("active_share", "positive"), GAIN_ARMS),
    Measure("active_share_negative", ("active_share", "negative"), GAIN_ARMS),
    Measure("active_share_words_right", ("active_share", "words_right"), GAIN_ARMS),
    Measure("active_share_words_wrong", ("active_share", "words_wrong"), GAIN_ARMS),
    Measure("clamp_share", ("clamp_share",), GAIN_ARMS),
    Measure("sigma0_at_floor_share", ("sigma0_at_floor_share",), GAIN_ARMS),
    Measure("sigma1_at_floor_share", ("sigma1_at_floor_share",), GAIN_ARMS),
    *_view("x0"),
    *_view("x1"),
    Measure("z0_over_z1", ("z0_over_z1",), GAIN_ARMS),
    Measure("ce_push", ("ce_push",), TRAINED_ARMS),
    Measure("gain_push", ("gain_push",), ("cpg",)),
    Measure("gain_offset", ("gain_offset",), ("cpg",)),
    Measure("skipped", ("skipped",), GAIN_ARMS),
)


def _epochs(record: Mapping[str, Any], where: str) -> dict[int, Mapping[str, Any]]:
    """A record's per-epoch measures by epoch, refused unless training wrote them so."""
    epochs: Any = record
    for key in ("results", "rules", "epochs"):
        epochs = epochs.get(key) if isinstance(epochs, Mapping) else None
    if not isinstance(epochs, Mapping) or not epochs:
        raise StepError(f"{where}: its record holds no per-epoch measures (results.rules.epochs)")
    found: dict[int, Mapping[str, Any]] = {}
    for key, measures in epochs.items():
        if (
            not isinstance(key, str)
            or not _EPOCH.fullmatch(key)
            or not isinstance(measures, Mapping)
        ):
            raise StepError(
                f"{where}: its per-epoch measures must be keyed by the epoch, from 1"
                " (results.rules.epochs)"
            )
        found[int(key)] = measures
    return found


def _lookup(measures: Mapping[str, Any], keys: Sequence[str], where: str) -> Any:
    """The value under nested keys of an epoch's measures; refused if a key is missing."""
    value: Any = measures
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise StepError(f"{where}: the record holds no {'.'.join(keys)}")
        value = value[key]
    return value


def _read(measures: Mapping[str, Any], keys: Sequence[str], where: str) -> int | float | None:
    """A logged value: a finite number, or None where training had none (a share over no bag)."""
    value = _lookup(measures, keys, where)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise StepError(f"{where}: {'.'.join(keys)} is not a finite number")
    return value


def _value(measures: Mapping[str, Any], measure: Measure, where: str) -> int | float | None:
    """One run's value of a measure in an epoch; None where it has none."""
    if measure.kind == LOGGED:
        return _read(measures, measure.keys, where)
    names = ("measured_mean", "mu") if measure.kind == GAP else ("measured_spread", "sigma")
    first, second = (_read(measures, (*measure.keys, name), where) for name in names)
    if first is None or second is None:
        return None
    if measure.kind == GAP:
        return first - second
    return first / second if second else None  # σ has a floor above 0, as the loss keeps it


def _count(measures: Mapping[str, Any], key: str, where: str) -> int:
    """A count of an epoch's bags: a whole number of at least 0."""
    value = _lookup(measures, (key,), where)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StepError(f"{where}: {key} is not a count of bags")
    return value


def _pooled(present: Sequence[tuple[str, Mapping[str, Any]]]) -> float | str:
    """The pooled active share of an epoch: the summed active bags over the summed bags."""
    bags = active = 0
    for where, measures in present:
        found_bags, found_active = (_count(measures, key, where) for key in ("bags", "active_bags"))
        if found_active > found_bags:
            raise StepError(f"{where}: more active bags than bags")
        bags += found_bags
        active += found_active
    return active / bags if bags else NO_BAG


def _row(
    arm: str, epoch: int, name: str, values: Sequence[int | float], pooled: float | str
) -> dict[str, Any]:
    """One row of the table: a measure's mean, minimum and maximum over the runs with a value.

    Every number is a float, counts of skipped steps too, so that a column reads the same
    whether or not a cell of it holds text.
    """
    found = [float(value) for value in values]
    return {
        "arm": arm,
        "epoch": epoch,
        "measure": name,
        "runs": len(found),
        "mean": math.fsum(found) / len(found) if found else NO_VALUE,
        "minimum": min(found) if found else NO_VALUE,
        "maximum": max(found) if found else NO_VALUE,
        "pooled_share": pooled,
    }


def gain_term_table(records: Mapping[tuple[str, int, int], Mapping[str, Any]]) -> pd.DataFrame:
    """Summarise the per-epoch measures of the kept runs, per arm, epoch and measure.

    Pilot diagnostics, section "Gain-term summary", bullets "Per epoch" and "Summary": for
    each arm (B0, B1 and CPG, those with records), each epoch that a run logged, and each
    measure that the arm logs (``MEASURES``), the mean, the minimum and the maximum over the
    runs that have a value, and their number; a run without a value, such as a share over
    no bag, is left out of that row. The row of the active share over all bags of B1 and
    CPG also carries the pooled share: the epoch's active bags summed over the runs, over
    their bags summed. Only ``results.rules.epochs`` of each record is read; the means are
    exactly rounded sums (``math.fsum``), so the table does not depend on the records'
    order.

    Args:
        records: The kept runs' records (their ``run.json``), by (arm, round, fold).

    Returns:
        One row per arm, epoch and measure, in that order, with the columns ``COLUMNS``. A
        cell without a number reads as text: ``NO_VALUE`` where no run has a value,
        ``NO_BAG`` for a pooled share over no bag, ``NOT_APPLICABLE`` for the pooled share of
        every other measure.

    Raises:
        StepError: If there is no record, an arm is no trained arm, or a record holds no
            per-epoch measures, lacks a measure that its arm logs, or holds a value that is
            no finite number or a count that is no count. The message names the run and
            the epoch, never a value.
    """
    if not records:
        raise StepError("no kept run to summarise")
    for arm, _, _ in records:
        if arm not in TRAINED_ARMS:
            raise StepError(f"{arm!r} is not a trained arm ({', '.join(TRAINED_ARMS)})")
    runs: dict[str, list[tuple[str, dict[int, Mapping[str, Any]]]]] = {
        arm: [] for arm in TRAINED_ARMS
    }
    for (arm, round_number, fold), record in sorted(
        records.items(), key=lambda item: (TRAINED_ARMS.index(item[0][0]), *item[0][1:])
    ):
        where = f"{arm}, round {round_number}, fold {fold}"
        runs[arm].append((where, _epochs(record, where)))
    rows: list[dict[str, Any]] = []
    for arm in TRAINED_ARMS:
        for epoch in sorted({epoch for _, epochs in runs[arm] for epoch in epochs}):
            present = [
                (f"{where}, epoch {epoch}", epochs[epoch])
                for where, epochs in runs[arm]
                if epoch in epochs
            ]
            pooled = _pooled(present) if arm in GAIN_ARMS else NOT_APPLICABLE
            for measure in MEASURES:
                if arm not in measure.arms:
                    continue
                found = (_value(measures, measure, where) for where, measures in present)
                values = [value for value in found if value is not None]
                share = pooled if measure.name == POOLED else NOT_APPLICABLE
                rows.append(_row(arm, epoch, measure.name, values, share))
    return pd.DataFrame(rows, columns=list(COLUMNS))
