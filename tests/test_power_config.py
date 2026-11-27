"""The settings of the main study's power simulation, pinned value by value.

Every value of configs/pilot/power.yaml is pinned here with its type (8000 is not 8000.0) and
in its order, as docs/pilot_diagnostics.md fixed them on 2026-11-27, before any number of the
simulation existed. The two input tables are pinned by their SHA-256; this test hashes the
files and computes nothing from their content.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "pilot"
TABLES = {"summary": "reports/pilot/summary.csv", "folds": "reports/pilot/fold_values.csv"}
POWER = {
    "inputs_sha256": {
        "summary": "3038fb53727be4325267f0fdc2624f1a54297e1712aa57b177f6d83d2f878678",
        "folds": "212fe103e57c0a281b1491c40c38e800f5201ef6e6d684ec8b4b42df150839b2",
    },
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
        "studies": 8000,
        "blocks": 64,
        "resamples": 2000,
        "rho_points": 5,
        "calibration_pilots": 2000,
        "tolerance": 1.0e-3,
        "seed": 110,
    },
    "sensitivity": {"at": "middle_rho", "rows": ["agreement_80", "b0_auroc_80", "se_matched"]},
    "resample_check": {
        "studies": 400,
        "gains": [0.0, 0.06],
        "resamples": 10000,
        "max_changed": 0.01,
        "fallback_resamples": 10000,
    },
    "design_check": {
        "b0_auroc": 0.65,
        "scenarios": [[0.9, 0.85], [0.8, 0.75], [0.6, 0.65]],
        "pilots": 8000,
        "pilot_sd": [0.026, 0.034, 0.041],
        "one_arm_sd": 0.044,
        "max_gap": 3.0e-3,
    },
}


def load(name: str) -> dict[str, Any]:
    config = yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    return config


def seeds(value: Any, named: bool = False) -> set[int]:
    """Every whole number under a key that names a seed, in nested blocks and lists too."""
    if isinstance(value, dict):
        return {
            seed for key, item in value.items() for seed in seeds(item, named or "seed" in str(key))
        }
    if isinstance(value, list):
        return {seed for item in value for seed in seeds(item, named)}
    if named and isinstance(value, int) and not isinstance(value, bool):
        return {value}
    return set()


def test_power_config_pins_every_setting_with_its_type_in_this_order() -> None:
    config = load("power.yaml")
    assert config == POWER
    # JSON tells 1 from 1.0 and true from 1, and keeps the keys' order.
    assert json.dumps(config) == json.dumps(POWER)


def test_the_input_pins_are_the_sha256_of_the_published_tables() -> None:
    pins = load("power.yaml")["inputs_sha256"]
    assert list(pins) == list(TABLES)
    for name, table in TABLES.items():
        assert pins[name] == hashlib.sha256((ROOT / table).read_bytes()).hexdigest()


def test_the_simulations_seed_is_one_that_no_other_config_uses() -> None:
    others = set()
    for path in sorted(CONFIGS.glob("*.yaml")):
        if path.name != "power.yaml":
            others |= seeds(load(path.name))
    assert set(range(101, 110)) | {111} <= others  # the walk finds the seeds of the other steps
    assert load("power.yaml")["simulation"]["seed"] not in others


def test_the_green_call_and_the_views_are_those_of_the_pilot_analysis() -> None:
    config, evaluate = load("power.yaml"), load("evaluate.yaml")
    assert config["green"]["f3_min_auroc"] == evaluate["checks"]["f3_min_auroc"]
    assert config["green"]["delta_x0_at_least"] == evaluate["rule"]["green_at_least"]
    assert config["views"] == evaluate["views"]


def test_the_grid_the_checks_and_the_resamples_fit_together() -> None:
    config = load("power.yaml")
    gains = config["main"]["gains"]
    points = round((gains["last"] - gains["first"]) / gains["step"]) + 1
    assert points == 11  # 0.00 to 0.10 in steps of 0.01
    design = config["design_check"]
    assert len(design["scenarios"]) == len(design["pilot_sd"]) == 3
    assert all(0.0 < r <= 1.0 and 0.0 < rho <= 1.0 for r, rho in design["scenarios"])
    real_test = load("evaluate.yaml")["bootstrap"]["resamples"]
    check = config["resample_check"]
    assert check["resamples"] == check["fallback_resamples"] == real_test
    assert config["simulation"]["resamples"] < real_test
