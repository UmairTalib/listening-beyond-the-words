"""The shared training settings and the arm files of the pilot, pinned value by value.

Every value of configs/pilot/train.yaml and of the arm files b0.yaml, b1.yaml and cpg.yaml
is pinned here with its type (4 is not 4.0, and true is not 1), as the pilot protocol fixed
them on 2026-11-25 (sections "Fixed settings" and "Training"). A setting that changes later,
such as the repeatability mode after the timing run, one of the two allowed changes or a
pre-set fallback, changes this test in the same dated commit.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from listening_beyond_words.models import load_pins
from listening_beyond_words.partitions import SEEDS
from listening_beyond_words.voxtral import PROMPT

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "pilot"
TARGETS = (
    r"model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(gate_proj|up_proj|down_proj))"
)
TRAIN = {
    "model": "voxtral",
    "prompt": (
        "Based on this speech, does the speaker have elevated depressive symptoms?"
        " Answer Yes or No."
    ),
    "seeds": {"round_1": 101, "round_2": 102},
    "epochs": 3,
    "lora": {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "bias": "none",
        "init_lora_weights": True,
        "use_rslora": False,
        "use_dora": False,
        "trained_adapter": "multi_modal_projector",
        "targets": TARGETS,
    },
    "optimiser": {
        "learning_rate": 1.0e-4,
        "adapter_learning_rate": 2.0e-5,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "amsgrad": False,
        "foreach": True,
        "weight_decay": 0.01,
        "grad_clip_norm": 1.0,
        "warmup_share": 0.05,
    },
    "precision": {
        "autocast": "fp16",
        "masters": "fp32",
        "scaler": {
            "init_scale": 65536.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval_steps": 2000,
        },
        "checkpointing": "every_layer",
        "deterministic": "pending",  # set by the commit that records the timing run
        "strict_max_extra_time_share": 0.1,
    },
    "sampler": {
        "positive_bags": 2,
        "negative_bags": 2,
        "epoch_end": "a_class_runs_out",
        "draw": "least_trained_first",
        "bag_choice": "lowest_trained_share_first",
        "step_order": "shuffled",
    },
    "loss": {"x1_weight": 0.25, "x2_weight": 0.5},
    "gain": {
        "margin": 0.3,
        "softness": 0.1,
        "ramp_share": 0.2,
        "momentum": 0.9,
        "start_steps": 8,
        "floor_share": 0.25,
        "floor_logit": 0.1,
    },
    "rules": {
        "f1_min_scale": 1.0,
        "alive": {
            "epochs": [2, 3],
            "min_share": 0.2,
            "max_share": 0.8,
            "margin_if_below": 0.5,
            "margin_if_above": 0.1,
        },
        "falling": {
            "loss": "ce_part",
            "min_drop_share": 0.05,
            "min_standard_errors": 2.0,
            "epochs_if_falling": 4,
        },
        "f5_max_hours": 4.0,
    },
    "scoring": {"views": ["x0", "rev"], "format": "fp16"},
    "timing": {
        "steps_per_mode": 10,
        "first_steps_left_out": 1,
        "stand_in_start": {"mu_logit": 0.0, "sigma_logit": 1.0},
    },
}
ARMS = {
    "b0": {"extends": "train.yaml", "arm": "b0", "views": ["x0"]},
    "b1": {"extends": "train.yaml", "arm": "b1", "views": ["x1", "x0", "x2"], "lambda_g": 0.0},
    "cpg": {"extends": "train.yaml", "arm": "cpg", "views": ["x1", "x0", "x2"], "lambda_g": 1.0},
}
PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


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


def test_train_config_pins_every_shared_setting_with_its_type_in_this_order() -> None:
    config = load("train.yaml")
    assert config == TRAIN
    assert type_differences(config, TRAIN) == []
    assert json.dumps(config) == json.dumps(TRAIN)


@pytest.mark.parametrize("arm", list(ARMS))
def test_each_arm_file_extends_train_yaml_with_its_arm_views_and_gain_weight(arm: str) -> None:
    config = load(f"{arm}.yaml")
    assert config == ARMS[arm]
    assert type_differences(config, ARMS[arm]) == []
    assert json.dumps(config) == json.dumps(ARMS[arm])


def test_the_arm_files_set_no_shared_setting_and_train_yaml_no_arm_key() -> None:
    shared = set(load("train.yaml"))
    for arm in ARMS:
        assert not set(load(f"{arm}.yaml")) & shared, arm
    assert not shared & {"extends", "arm", "views", "lambda_g"}


def test_b1_is_cpg_with_lambda_g_0_and_b0_has_no_gain() -> None:
    b1, cpg = load("b1.yaml"), load("cpg.yaml")
    assert {**b1, "arm": "cpg", "lambda_g": 1.0} == cpg
    assert "lambda_g" not in load("b0.yaml")


def test_the_prompt_is_the_one_that_the_code_scores_with() -> None:
    assert load("train.yaml")["prompt"] == PROMPT


def test_the_model_is_the_pinned_voxtral() -> None:
    pins = load_pins(CONFIGS / "models.yaml")
    assert pins[load("train.yaml")["model"]].repo_id == "mistralai/Voxtral-Mini-3B-2507"


def test_each_round_trains_with_the_seed_of_its_partitions() -> None:
    seeds = load("train.yaml")["seeds"]
    assert (seeds["round_1"], seeds["round_2"]) == SEEDS


def test_the_bags_stay_in_the_data_config_with_the_sampler_in_train_yaml() -> None:
    assert load("data.yaml")["bags"] == {
        "segments_per_bag": 4,
        "max_segments_per_epoch": 16,
        "repeat_segments": False,
    }
    assert "bags" not in load("train.yaml")


def test_the_scale_halved_16_times_reaches_the_floor_of_check_f1() -> None:
    # Pilot protocol, section "Training", bullet "Check F1": a run has far fewer than 2,000
    # steps, so the scale never grows back, and a 17th skipped step fails it.
    train = load("train.yaml")
    scaler = train["precision"]["scaler"]
    scale = scaler["init_scale"] * scaler["backoff_factor"] ** 16
    assert scale == train["rules"]["f1_min_scale"]
    assert scale * scaler["backoff_factor"] < train["rules"]["f1_min_scale"]


def test_lora_targets_the_seven_projections_of_each_language_model_layer_and_nothing_else() -> None:
    import torch
    from transformers import VoxtralForConditionalGeneration

    from voxtral_fakes import tiny_config

    config = tiny_config()
    config.text_config.num_hidden_layers = 30  # as in Voxtral Mini 3B
    with torch.device("meta"):  # names only: no weights are made
        model = VoxtralForConditionalGeneration(config)
    names = [name for name, _ in model.named_modules()]
    pattern = load("train.yaml")["lora"]["targets"]
    picked = [name for name in names if re.fullmatch(pattern, name)]  # as PEFT reads a string
    assert sorted(picked) == sorted(
        f"model.language_model.layers.{layer}.{part}" for layer in range(30) for part in PROJECTIONS
    )
    assert len(picked) == 210
    # The encoder has q, k and v projections of its own, which a list of names would pick.
    encoder = [name for name in names if name.startswith("model.audio_tower.layers.")]
    assert {name.rsplit(".", 1)[1] for name in encoder} >= {"q_proj", "k_proj", "v_proj"}
    assert not set(picked) & set(encoder)
