import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "configs" / "pilot" / "data.yaml"
ZEROSHOT = ROOT / "configs" / "pilot" / "zeroshot.yaml"


def load(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    return config


def seeds(node: Any, path: str = "") -> list[tuple[str, int]]:
    """Every value under a key that names a seed, with where it is."""
    found: list[tuple[str, int]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            where = f"{path}.{key}" if path else str(key)
            if "seed" in str(key) and isinstance(value, int):
                found.append((where, value))
            found.extend(seeds(value, where))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(seeds(value, f"{path}[{index}]"))
    return found


def test_data_config_pins_every_setting_of_the_encoder_cache() -> None:
    expected = {
        "cpu_threads": 4,
        "verify": {"segments_per_view": 25, "seed": 108, "lm_watch_clips": 25},
    }
    block = load(DATA)["encoder_cache"]
    assert block == expected
    # The same types too: 4 is not 4.0 and not "4".
    assert json.dumps(block, sort_keys=True) == json.dumps(expected, sort_keys=True)


def test_zeroshot_config_pins_the_arm_its_views_and_its_check() -> None:
    expected = {
        "arm": "zeroshot",
        "views": ["x0", "x1", "rev"],
        "check": {"segments_per_view": 20, "seed": 108},
    }
    config = load(ZEROSHOT)
    assert config == expected
    assert json.dumps(config, sort_keys=True) == json.dumps(expected, sort_keys=True)


def test_the_zero_shot_views_are_pinned_clip_versions_and_x2_is_not_scored() -> None:
    views = load(ZEROSHOT)["views"]
    assert set(views) <= set(load(DATA)["view_runs"])
    assert len(set(views)) == len(views)
    assert "x2" not in views  # x2 is watched in the check job, never scored


def test_seed_108_is_new_to_the_data_config() -> None:
    # 101-107 are taken: the partitions and listening checks (101, 102), the interviewer
    # detector (103, 104), the word check (105), x1 (106) and its re-check (107). The cache
    # check and the zero-shot check both draw from 108, each with a stream of its own.
    found = seeds(load(DATA))
    assert [where for where, value in found if value == 108] == ["encoder_cache.verify.seed"]
    assert {value for _, value in found} - {108} <= set(range(101, 108))


def test_no_cache_run_and_no_zero_shot_score_is_pinned_before_the_jobs_run() -> None:
    # The record commits pin them (cache_runs, zeroshot_scores) and replace this test with
    # one that checks their shape.
    config = load(DATA)
    assert "cache_runs" not in config
    assert "zeroshot_scores" not in config
