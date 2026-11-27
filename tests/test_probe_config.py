"""The settings of the pooled-encoder probe, pinned value by value.

Every value of configs/pilot/probe.yaml is pinned here with its type (5000 is not 5000.0) and
in its order, as docs/pilot_diagnostics.md fixed them on 2026-11-27, before any number of the
probe existed.
"""

import json
import math
from pathlib import Path
from typing import Any

import yaml

from listening_beyond_words.segments import SEGMENT_WORDS

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "pilot"
PROBE = {
    "views": ["x0", "x2", "rev"],
    "partition_round": 1,
    "pooling": {"statistics": ["mean", "sd"], "kept": "inside_parts"},
    "targets": {
        "view": "x0",
        "words": 40,
        "pitch_variability": "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
        "timing": "word_times",
    },
    "controls": ["gender", "length", "egemaps88"],
    "penalty": {
        "ridge_alphas": [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0],
        "logistic_cs": [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0],
        "max_iter": 5000,
    },
    "bootstrap": {"resamples": 10000, "seed": 111, "levels": [0.8, 0.95]},
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


def test_probe_config_pins_every_setting_with_its_type_in_this_order() -> None:
    config = load("probe.yaml")
    assert config == PROBE
    # JSON tells 1 from 1.0 and true from 1, and keeps the keys' order.
    assert json.dumps(config) == json.dumps(PROBE)


def test_the_probes_seed_is_one_that_no_other_config_uses() -> None:
    others = set()
    for path in sorted(CONFIGS.glob("*.yaml")):
        if path.name != "probe.yaml":
            others |= seeds(load(path.name))
    assert set(range(101, 111)) <= others  # the walk finds the seeds of the other steps
    assert load("probe.yaml")["bootstrap"]["seed"] not in others


def test_the_draws_follow_the_pilot_analysis_and_the_views_are_cached_ones() -> None:
    config, evaluate, data = load("probe.yaml"), load("evaluate.yaml"), load("data.yaml")
    assert config["bootstrap"]["resamples"] == evaluate["bootstrap"]["resamples"]
    assert config["bootstrap"]["levels"] == evaluate["bootstrap"]["levels"]
    assert set(config["views"]) <= set(data["cache_runs"])
    assert config["views"][0] == config["targets"]["view"] == "x0"
    assert config["targets"]["words"] == SEGMENT_WORDS
    assert config["targets"]["pitch_variability"] in data["word_check"]["voice"]["measures"]
    assert config["partition_round"] == data["word_check"]["partition_round"]  # the voice probe's


def test_each_penalty_grid_has_one_value_per_decade_from_end_to_end() -> None:
    # Ridge α from 1 to 10⁶ and logistic C from 10⁻⁶ to 1 (pilot diagnostics, bullet "Penalty").
    penalty = load("probe.yaml")["penalty"]
    for grid, exponents in (
        (penalty["ridge_alphas"], range(0, 7)),
        (penalty["logistic_cs"], range(-6, 1)),
    ):
        assert grid == [float(f"1e{exponent}") for exponent in exponents]
        assert [round(math.log10(value)) for value in grid] == list(exponents)
