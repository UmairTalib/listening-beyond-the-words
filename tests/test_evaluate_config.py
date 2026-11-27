"""The settings of the pilot's analysis, pinned value by value.

Every value of configs/pilot/evaluate.yaml is pinned here with its type (3 is not 3.0, and
true is not 1) and in its order, as docs/pilot_analysis.md fixed them on 2026-11-26, before
any AUROC of any arm was computed. The pins of the kept training runs (runs) are added after
training, in the commit that pins them, which changes this test in the same commit.
"""

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "pilot"
EVALUATE = {
    "arms": ["zeroshot", "b0", "b1", "cpg"],
    "views": ["x0", "rev"],
    "differences": [["cpg", "b0"], ["b1", "b0"], ["cpg", "b1"]],
    "bootstrap": {"resamples": 10000, "seed": 109, "levels": [0.8, 0.95]},
    "checks": {"f3_min_auroc": 0.6},
    "rule": {"green_at_least": 0.03, "red_at_most": -0.04},
    "framing": {
        "view": "x0",
        "cpg_gain_at_least": 0.03,
        "b1_gain_at_least": 0.03,
        "cpg_over_b1_below": 0.02,
    },
    "first_runs_commit": "271cce6d31e6c30fd10353649b246893882aaea2",
    "runs": {},
}


def load(name: str) -> dict[str, Any]:
    config = yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    return config


def type_differences(found: Any, expected: Any, where: str = "") -> list[str]:
    """Every place where two values differ in type, keys and items included."""
    if type(found) is not type(expected):
        return [f"{where or 'top'}: {type(found).__name__}, not {type(expected).__name__}"]
    if isinstance(expected, dict):
        if list(found) != list(expected):
            return [f"{where or 'top'}: keys {list(found)}, not {list(expected)}"]
        return [
            difference
            for key in expected
            for difference in type_differences(found[key], expected[key], f"{where}.{key}")
        ]
    if isinstance(expected, list):
        if len(found) != len(expected):
            return [f"{where}: {len(found)} items, not {len(expected)}"]
        return [
            difference
            for index, (item, wanted) in enumerate(zip(found, expected, strict=True))
            for difference in type_differences(item, wanted, f"{where}[{index}]")
        ]
    return []


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


def test_evaluate_config_pins_every_setting_with_its_type_in_this_order() -> None:
    config = load("evaluate.yaml")
    assert config == EVALUATE
    assert type_differences(config, EVALUATE) == []
    assert json.dumps(config) == json.dumps(EVALUATE)


def test_the_bootstrap_seed_is_one_that_no_other_config_uses() -> None:
    others = set()
    for path in sorted(CONFIGS.glob("*.yaml")):
        if path.name != "evaluate.yaml":
            others |= seeds(load(path.name))
    assert {101, 102, 105, 106, 108} <= others  # the walk finds the seeds of the other steps
    assert load("evaluate.yaml")["bootstrap"]["seed"] not in others


def test_every_arm_has_its_config_and_scored_both_test_views() -> None:
    config = load("evaluate.yaml")
    for arm in config["arms"]:
        assert load(f"{arm}.yaml")["arm"] == arm
    assert load("train.yaml")["scoring"]["views"] == config["views"]  # B0, B1 and CPG
    assert set(config["views"]) <= set(load("zeroshot.yaml")["views"])


def test_each_difference_is_one_trained_arm_minus_another_with_delta_first() -> None:
    config = load("evaluate.yaml")
    trained = set(config["arms"]) - {"zeroshot"}
    assert config["differences"][0] == ["cpg", "b0"]  # Δ, the main number
    for first, second in config["differences"]:
        assert {first, second} <= trained
        assert first != second


def test_the_framing_is_read_on_x0_only_where_cpg_beats_b0() -> None:
    config = load("evaluate.yaml")
    rule, framing = config["rule"], config["framing"]
    assert rule["red_at_most"] < 0.0 < rule["green_at_least"]
    assert framing["view"] == "x0"
    assert framing["view"] in config["views"]
    assert framing["cpg_gain_at_least"] == rule["green_at_least"]  # "Green: beats B0"
