"""Stand-ins for the tests of the power simulation (power.py and scripts/simulate_power.py).

Synthetic numbers only, never the published tables: the simulation's two inputs are written
in code with the layout of the pilot analysis's summary.csv and fold_values.csv, and the
model is checked on the study design's assumptions (B0's AUROC 0.65 and its three
scenarios). The class counts per fold are those that partitions.make_partitions allocates
to 57 positive and 132 negative participants stratified by label and gender: the same in
every round, whatever the seed. TINY is power.yaml with every count cut down, so that a whole
job takes seconds; its design check passes whatever the model gives (max_gap 1.0). Serial
runs the blocks in the test's own process, and Recording also keeps every task with its
block, so that a test can recompute a number from the blocks it came from.
"""

import copy
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from listening_beyond_words import power
from x1_fakes import REPO

CONFIGS = REPO / "configs" / "pilot"
DESIGN = power.Design(
    positives=(12, 12, 11, 11, 11), negatives=(27, 27, 26, 26, 26), pilot_rounds=2
)
SUMMARY_COLUMNS = [
    "number",
    "view",
    "rounds",
    "fold_runs",
    "mean",
    "round_1",
    "round_2",
    "low_80",
    "high_80",
    "low_95",
    "high_95",
    "se",
]
ARMS = ("zeroshot", "b0", "b1", "cpg")
FOLD_NUMBERS = [
    *(f"auroc_{arm}" for arm in ARMS),
    *(f"spearman_{arm}" for arm in ARMS),
    "cpg_minus_b0",
    "b1_minus_b0",
    "cpg_minus_b1",
    "agreement_b0_cpg",
]
FOLD_COLUMNS = ["round", "fold", "view", "positives", "negatives", "segments", *FOLD_NUMBERS]
# The synthetic pilot: what each view's numbers are, by the summary's number. Each tuple is
# (mean, round 1, round 2, low 80, high 80, SE); None writes the analysis's text instead.
NUMBERS: dict[str, dict[str, tuple[float | None, ...]]] = {
    "x0": {
        "auroc_zeroshot": (0.70, 0.71, 0.69, 0.66, 0.74, None),
        "auroc_b0": (0.66, 0.65, 0.67, 0.61, 0.71, None),
        "auroc_b1": (0.67, 0.66, 0.68, 0.62, 0.72, None),
        "auroc_cpg": (0.70, 0.68, 0.72, 0.65, 0.75, None),
        "cpg_minus_b0": (0.04, 0.03, 0.05, 0.02, 0.06, 0.016),
        "agreement_b0_cpg": (0.90, 0.89, 0.91, 0.88, 0.92, None),
    },
    "rev": {
        "auroc_zeroshot": (0.52, 0.53, 0.51, 0.47, 0.57, None),
        "auroc_b0": (0.55, 0.54, 0.56, 0.50, 0.60, None),
        "auroc_b1": (0.56, 0.55, 0.57, 0.51, 0.61, None),
        "auroc_cpg": (0.56, 0.55, 0.57, 0.51, 0.61, None),
        "cpg_minus_b0": (0.01, 0.0, 0.02, -0.01, 0.03, 0.022),
        "agreement_b0_cpg": (0.85, 0.84, 0.86, 0.83, 0.87, None),
    },
}
SE_CELL = "not computed: differences only"


def summary_table(
    numbers: dict[str, dict[str, tuple[float | None, ...]]] = NUMBERS,
) -> pd.DataFrame:
    """summary.csv's layout: every number of every view, the ones the simulation reads set."""
    rows = []
    for name in [*(f"auroc_{arm}" for arm in ARMS), *(f"spearman_{arm}" for arm in ARMS)]:
        for view in ("x0", "rev"):
            rows.append(summary_row(name, view, numbers[view].get(name)))
    for name in ("cpg_minus_b0", "b1_minus_b0", "cpg_minus_b1", "agreement_b0_cpg"):
        for view in ("x0", "rev"):
            rows.append(summary_row(name, view, numbers[view].get(name)))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def summary_row(name: str, view: str, values: tuple[float | None, ...] | None) -> dict[str, Any]:
    """One row of summary.csv; a number the simulation does not read gets made-up values."""
    mean, first, second, low, high, se = values or (0.3, 0.29, 0.31, 0.2, 0.4, None)
    return {
        "number": name,
        "view": view,
        "rounds": "1 and 2",
        "fold_runs": 10,
        "mean": mean,
        "round_1": first,
        "round_2": second,
        "low_80": low,
        "high_80": high,
        "low_95": low - 0.01 if isinstance(low, float) else low,
        "high_95": high + 0.01 if isinstance(high, float) else high,
        "se": SE_CELL if se is None else se,
    }


def fold_table(design: power.Design = DESIGN, rounds: int = 2) -> pd.DataFrame:
    """fold_values.csv's layout: the class counts of every fold, the same in every round."""
    rows = []
    for number in range(1, rounds + 1):
        for fold, (positives, negatives) in enumerate(
            zip(design.positives, design.negatives, strict=True), start=1
        ):
            for view in ("x0", "rev"):
                cells: dict[str, Any] = {
                    "round": number,
                    "fold": fold,
                    "view": view,
                    "positives": positives,
                    "negatives": negatives,
                    "segments": 31 * (positives + negatives),
                }
                cells.update(dict.fromkeys(FOLD_NUMBERS, 0.5))
                rows.append(cells)
    return pd.DataFrame(rows, columns=FOLD_COLUMNS)


def write_inputs(
    folder: Path, summary: pd.DataFrame | None = None, folds: pd.DataFrame | None = None
) -> tuple[Path, Path]:
    """Write the two tables as the analysis job writes its CSV files; return their paths."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = (folder / "summary.csv", folder / "fold_values.csv")
    for path, table in zip(
        paths,
        (summary_table() if summary is None else summary, fold_table() if folds is None else folds),
        strict=True,
    ):
        path.write_text(table.to_csv(index=False, lineterminator="\n"), encoding="utf-8")
    return paths


def sha256(path: Path) -> str:
    """A file's SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


TINY: dict[str, Any] = {
    "inputs_sha256": {"summary": "0" * 64, "folds": "0" * 64},
    "views": ["x0", "rev"],
    "main": {
        "rounds": 3,
        "folds": 5,
        "gains": {"first": 0.0, "last": 0.1, "step": 0.01},
        "alpha": 0.05,
        "holm_family": 3,
    },
    "green": {"f3_min_auroc": 0.6, "delta_x0_at_least": 0.03},
    "simulation": {
        "studies": 12,
        "blocks": 3,
        "resamples": 40,
        "rho_points": 5,
        "calibration_pilots": 24,
        "tolerance": 1.0e-3,
        "seed": 110,
    },
    "sensitivity": {"at": "middle_rho", "rows": ["agreement_80", "b0_auroc_80", "se_matched"]},
    "resample_check": {
        "studies": 6,
        "gains": [0.0, 0.06],
        "resamples": 120,
        "max_changed": 1.0,
        "fallback_resamples": 60,
    },
    "design_check": {
        "b0_auroc": 0.65,
        "scenarios": [[0.9, 0.85], [0.8, 0.75], [0.6, 0.65]],
        "pilots": 60,
        "pilot_sd": [0.026, 0.034, 0.041],
        "one_arm_sd": 0.044,
        "max_gap": 1.0,
    },
}


def tiny_config(
    summary: Path | None = None, folds: Path | None = None, **blocks: dict[str, Any]
) -> dict[str, Any]:
    """TINY with the tables' pins and any block's values changed."""
    config = copy.deepcopy(TINY)
    if summary is not None and folds is not None:
        config["inputs_sha256"] = {"summary": sha256(summary), "folds": sha256(folds)}
    for name, changes in blocks.items():
        config[name].update(changes)
    return config


def write_config(path: Path, config: dict[str, Any]) -> Path:
    """Write a power.yaml."""
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def real_config() -> dict[str, Any]:
    """The committed power.yaml."""
    return yaml.safe_load((CONFIGS / "power.yaml").read_text(encoding="utf-8"))


class Serial:
    """Runs the blocks in this process, in order: for fast tests (the job uses power.Workers)."""

    def map(self, tasks: Sequence[power.Task]) -> list[power.Block]:
        return [power.simulate_block(task) for task in tasks]


class Recording(Serial):
    """Runs the blocks as Serial does and keeps every call: its tasks and their blocks."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[power.Task], list[power.Block]]] = []

    def map(self, tasks: Sequence[power.Task]) -> list[power.Block]:
        blocks = super().map(tasks)
        self.calls.append((list(tasks), blocks))
        return blocks


def pilots_of(settings: power.PowerSettings) -> tuple[dict[str, power.PilotView], power.Design]:
    """The synthetic pilot's numbers and design, through the job's reader."""
    summary = summary_table().astype(str)
    folds = fold_table().astype(str)
    return power.read_pilot(summary, folds, settings)
