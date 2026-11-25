"""The training configs (config.py): the arm files, the shared settings that they extend, the hash.

Each refusal is planted once in a copy of the pilot's configs (train.yaml, the three arm
files and zeroshot.yaml) in a temporary folder, and the refusal's message must name the
file, the key and the rule.
"""

import dataclasses
import hashlib
import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from listening_beyond_words.config import (
    ARM_VIEWS,
    AliveRule,
    ArmConfig,
    FallingRule,
    GainSettings,
    LoraSettings,
    LossSettings,
    OptimiserSettings,
    PrecisionSettings,
    RuleSettings,
    SamplerSettings,
    ScalerSettings,
    ScoringSettings,
    SeedSettings,
    StandInStart,
    TimingSettings,
    TrainSettings,
    canonical_sha256,
    load_arm,
    resolve_config,
)
from listening_beyond_words.runinfo import StepError, sha256_file
from listening_beyond_words.voxtral import PROMPT
from listening_beyond_words.zeroshot import ZeroshotSettings

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "pilot"
FILES = ("train.yaml", "b0.yaml", "b1.yaml", "cpg.yaml", "zeroshot.yaml")
ARMS = ("b0", "b1", "cpg")


class Drop:
    """In a planted change: the key is removed."""

    def __repr__(self) -> str:
        return "DROP"


DROP = Drop()
TARGETS = (
    r"model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(gate_proj|up_proj|down_proj))"
)
# The settings of train.yaml as load_arm reads them, field by field (pilot protocol,
# sections "Fixed settings" and "Training").
SHARED = TrainSettings(
    model="voxtral",
    prompt=PROMPT,
    seeds=SeedSettings(round_1=101, round_2=102),
    epochs=3,
    lora=LoraSettings(
        rank=16,
        alpha=32,
        dropout=0.05,
        bias="none",
        init_lora_weights=True,
        use_rslora=False,
        use_dora=False,
        trained_adapter="multi_modal_projector",
        targets=TARGETS,
    ),
    optimiser=OptimiserSettings(
        learning_rate=1.0e-4,
        adapter_learning_rate=2.0e-5,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        amsgrad=False,
        foreach=True,
        weight_decay=0.01,
        grad_clip_norm=1.0,
        warmup_share=0.05,
    ),
    precision=PrecisionSettings(
        autocast="fp16",
        masters="fp32",
        scaler=ScalerSettings(
            init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval_steps=2000
        ),
        checkpointing="every_layer",
        deterministic="strict",  # by its rule, from the timing run (job 12618228)
        strict_max_extra_time_share=0.1,
    ),
    sampler=SamplerSettings(
        positive_bags=2,
        negative_bags=2,
        epoch_end="a_class_runs_out",
        draw="least_trained_first",
        bag_choice="lowest_trained_share_first",
        step_order="shuffled",
    ),
    loss=LossSettings(x1_weight=0.25, x2_weight=0.5),
    gain=GainSettings(
        margin=0.3,
        softness=0.1,
        ramp_share=0.2,
        momentum=0.9,
        start_steps=8,
        floor_share=0.25,
        floor_logit=0.1,
    ),
    rules=RuleSettings(
        f1_min_scale=1.0,
        alive=AliveRule(
            epochs=(2, 3), min_share=0.2, max_share=0.8, margin_if_below=0.5, margin_if_above=0.1
        ),
        falling=FallingRule(
            loss="ce_part", min_drop_share=0.05, min_standard_errors=2.0, epochs_if_falling=4
        ),
        f5_max_hours=4.0,
    ),
    scoring=ScoringSettings(views=("x0", "rev"), format="fp16"),
    timing=TimingSettings(
        steps_per_mode=10,
        first_steps_left_out=1,
        stand_in_start=StandInStart(mu_logit=0.0, sigma_logit=1.0),
    ),
)


def read(path: Path) -> dict[str, Any]:
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(content, dict)
    return content


def canonical(content: dict[str, Any]) -> str:
    """The SHA-256 of canonical JSON, as engineering conventions, section 5, sets it."""
    text = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """A copy of the pilot's training configs, as the repository holds them."""
    for name in FILES:
        shutil.copy(CONFIGS / name, tmp_path / name)
    return tmp_path


def write(folder: Path, name: str, content: Any) -> None:
    text = yaml.safe_dump(content, sort_keys=False, allow_unicode=True)
    (folder / name).write_text(text, encoding="utf-8")


def plant(folder: Path, name: str, path: str, value: Any) -> None:
    """Set one value of a config file by its dotted path, or remove the key (DROP)."""
    content = read(folder / name)
    *blocks, key = path.split(".")
    block = content
    for part in blocks:
        block = block[part]
    if value is DROP:
        del block[key]
    else:
        block[key] = value
    write(folder, name, content)


# The pilot's arms -------------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ARMS)
def test_load_arm_reads_each_arm_with_the_shared_settings_and_the_resolved_config(arm: str) -> None:
    path = CONFIGS / f"{arm}.yaml"
    config = load_arm(path)
    own = read(path)
    shared = read(CONFIGS / "train.yaml")
    assert isinstance(config, ArmConfig)
    assert config.arm == arm
    assert config.views == ARM_VIEWS[arm] == tuple(own["views"])
    assert config.lambda_g == {"b0": None, "b1": 0.0, "cpg": 1.0}[arm]
    assert config.shared == SHARED
    resolved = {key: value for key, value in own.items() if key != "extends"} | shared
    assert config.resolved == resolved
    assert list(config.resolved) == [*(key for key in own if key != "extends"), *shared]
    assert config.sha256 == canonical(resolved)


def test_the_arms_differ_only_in_their_own_keys() -> None:
    configs = [load_arm(CONFIGS / f"{arm}.yaml") for arm in ARMS]
    own = {"arm", "views", "lambda_g"}
    shared = [{k: v for k, v in c.resolved.items() if k not in own} for c in configs]
    assert shared[0] == shared[1] == shared[2] == read(CONFIGS / "train.yaml")
    assert len({config.sha256 for config in configs}) == 3
    assert "lambda_g" not in configs[0].resolved


def test_the_views_of_each_arm_go_x1_first_and_b0_trains_x0_only() -> None:
    assert dict(ARM_VIEWS) == {"b0": ("x0",), "b1": ("x1", "x0", "x2"), "cpg": ("x1", "x0", "x2")}


def test_each_round_has_its_seed() -> None:
    seeds = SHARED.seeds
    assert (seeds.of(1), seeds.of(2)) == (101, 102)
    for other in (0, 3, True):
        with pytest.raises(ValueError, match=r"^the pilot has rounds 1 and 2, not "):
            seeds.of(other)


def test_the_settings_are_frozen() -> None:
    config = load_arm(CONFIGS / "cpg.yaml")
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.lambda_g = 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.shared.gain.margin = 0.5


def test_every_repeatability_mode_reads_and_a_changed_mode_changes_the_hash(folder: Path) -> None:
    hashes = set()
    for mode in ("pending", "strict", "warn_only"):
        plant(folder, "train.yaml", "precision.deterministic", mode)
        config = load_arm(folder / "cpg.yaml")
        assert config.shared.precision.deterministic == mode
        hashes.add(config.sha256)
    assert len(hashes) == 3


# The resolver -------------------------------------------------------------------------------------


def test_resolve_config_reads_zeroshot_yaml_with_the_shared_settings() -> None:
    path = CONFIGS / "zeroshot.yaml"
    config = resolve_config(path)
    own, shared = read(path), read(CONFIGS / "train.yaml")
    assert (config.file, config.shared_file) == (path, CONFIGS / "train.yaml")
    assert (config.own, config.shared) == (own, shared)
    assert config.file_sha256 == sha256_file(path)
    assert config.shared_sha256 == sha256_file(CONFIGS / "train.yaml")
    resolved = {key: value for key, value in own.items() if key != "extends"} | shared
    assert config.resolved == resolved
    assert config.sha256 == canonical(resolved)
    # Its own keys stay with the zero-shot arm's settings.
    assert ZeroshotSettings.from_config(config.own).views == ("x0", "x1", "rev")


def test_the_resolved_config_is_a_copy(folder: Path) -> None:
    config = resolve_config(folder / "cpg.yaml")
    config.resolved["gain"]["margin"] = 0.5
    assert config.shared["gain"]["margin"] == 0.3
    assert load_arm(folder / "cpg.yaml").resolved["gain"]["margin"] == 0.3


# The hash -----------------------------------------------------------------------------------------


def reordered(content: Any) -> Any:
    """The same content with the keys of every mapping in reverse order."""
    if isinstance(content, dict):
        return {key: reordered(content[key]) for key in reversed(list(content))}
    return content


def test_the_sha256_does_not_change_when_the_keys_are_reordered(folder: Path) -> None:
    before = load_arm(folder / "cpg.yaml")
    for name in ("train.yaml", "cpg.yaml"):
        write(folder, name, reordered(read(folder / name)))
    assert next(iter(read(folder / "train.yaml"))) == "timing"  # the plant took
    after = load_arm(folder / "cpg.yaml")
    assert after.sha256 == before.sha256
    assert after.resolved == before.resolved
    assert after.shared == before.shared


def test_the_sha256_does_not_change_with_the_yaml_style_or_the_comments(folder: Path) -> None:
    before = load_arm(folder / "b1.yaml")
    text = (folder / "train.yaml").read_text(encoding="utf-8")
    plain = "\n".join(
        line.split(" # ")[0] for line in text.splitlines() if not line.startswith("#")
    )
    (folder / "train.yaml").write_text(plain + "\n", encoding="utf-8")
    flow = json.dumps(read(folder / "b1.yaml"))  # JSON is YAML in flow style
    (folder / "b1.yaml").write_text(flow, encoding="utf-8")
    assert "#" not in (folder / "train.yaml").read_text(encoding="utf-8")  # the plant took
    after = load_arm(folder / "b1.yaml")
    assert after.sha256 == before.sha256
    assert resolve_config(folder / "b1.yaml").file_sha256 != sha256_file(CONFIGS / "b1.yaml")


def test_a_changed_setting_changes_the_sha256(folder: Path) -> None:
    before = load_arm(folder / "cpg.yaml").sha256
    plant(folder, "train.yaml", "gain.margin", 0.5)
    assert load_arm(folder / "cpg.yaml").sha256 != before


def test_the_canonical_sha256_sorts_keys_keeps_types_and_writes_utf_8() -> None:
    assert canonical_sha256({"b": 1, "a": {"d": 2, "c": 3}}) == canonical_sha256(
        {"a": {"c": 3, "d": 2}, "b": 1}
    )
    text = '{"a":{"c":3,"d":2},"b":"λ"}'
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert canonical_sha256({"b": "λ", "a": {"d": 2, "c": 3}}) == expected
    assert canonical_sha256({"a": 1}) != canonical_sha256({"a": 1.0})
    assert canonical_sha256({"a": True}) != canonical_sha256({"a": 1})
    with pytest.raises(ValueError, match="not JSON compliant"):
        canonical_sha256({"a": float("nan")})


# Refusals: the files ------------------------------------------------------------------------------


def write_text(name: str, text: str) -> Callable[[Path], None]:
    def change(folder: Path) -> None:
        (folder / name).write_text(text, encoding="utf-8")

    return change


def append(name: str, text: str) -> Callable[[Path], None]:
    def change(folder: Path) -> None:
        with (folder / name).open("a", encoding="utf-8") as handle:
            handle.write(text)

    return change


def replace(name: str, old: str, new: str) -> Callable[[Path], None]:
    def change(folder: Path) -> None:
        text = (folder / name).read_text(encoding="utf-8")
        assert text.count(old) == 1
        (folder / name).write_text(text.replace(old, new), encoding="utf-8")

    return change


def setting(name: str, path: str, value: Any) -> Callable[[Path], None]:
    def change(folder: Path) -> None:
        plant(folder, name, path, value)

    return change


def remove(name: str) -> Callable[[Path], None]:
    def change(folder: Path) -> None:
        (folder / name).unlink()

    return change


def not_utf_8(folder: Path) -> None:
    (folder / "cpg.yaml").write_bytes(b"arm: \xff\n")


# (change, the arm file that load_arm reads, the message's start)
FILE_REFUSALS: dict[str, tuple[Callable[[Path], None], str, str]] = {
    "no arm file": (remove("b1.yaml"), "b1.yaml", "b1.yaml is missing ("),
    "no train.yaml": (remove("train.yaml"), "cpg.yaml", "train.yaml is missing ("),
    "not YAML": (write_text("cpg.yaml", "arm: [cpg\n"), "cpg.yaml", "cpg.yaml is not valid YAML: "),
    "not UTF-8": (not_utf_8, "cpg.yaml", "cpg.yaml is not UTF-8 text"),
    "a list": (
        write_text("cpg.yaml", "- extends: train.yaml\n"),
        "cpg.yaml",
        "cpg.yaml must hold a mapping of named settings",
    ),
    "a key that is a number": (
        append("cpg.yaml", "1: one\n"),
        "cpg.yaml",
        "cpg.yaml must hold a mapping of named settings",
    ),
    "an empty train.yaml": (
        write_text("train.yaml", ""),
        "cpg.yaml",
        "train.yaml must hold a mapping of named settings",
    ),
    "a key twice in the arm file": (
        append("cpg.yaml", "lambda_g: 0.5\n"),
        "cpg.yaml",
        "cpg.yaml is not valid YAML: the key 'lambda_g' is given twice (line 8)",
    ),
    "a key twice in a block of train.yaml": (
        replace("train.yaml", "  margin: 0.3 #", "  margin: 0.5\n  margin: 0.3 #"),
        "cpg.yaml",
        "train.yaml is not valid YAML: the key 'margin' is given twice",
    ),
    "no extends": (
        setting("cpg.yaml", "extends", DROP),
        "cpg.yaml",
        "cpg.yaml extends no file: it must start from train.yaml (engineering conventions,"
        " section 5)",
    ),
    "an extends naming another file": (
        setting("cpg.yaml", "extends", "b1.yaml"),
        "cpg.yaml",
        "cpg.yaml must extend 'train.yaml', the shared file in its folder, not 'b1.yaml'",
    ),
    "an extends naming a path": (
        setting("cpg.yaml", "extends", "../pilot/train.yaml"),
        "cpg.yaml",
        "cpg.yaml must extend 'train.yaml', the shared file in its folder, not"
        " '../pilot/train.yaml'",
    ),
    "an extends naming two files": (
        setting("b0.yaml", "extends", ["train.yaml"]),
        "b0.yaml",
        "b0.yaml must extend 'train.yaml', the shared file in its folder, not ['train.yaml']",
    ),
    "two levels of extends": (
        setting("train.yaml", "extends", "base.yaml"),
        "cpg.yaml",
        "train.yaml extends 'base.yaml': one level of extends only (engineering conventions,"
        " section 5)",
    ),
    "train.yaml extending itself": (
        setting("train.yaml", "extends", "train.yaml"),
        "b0.yaml",
        "train.yaml extends 'train.yaml': one level of extends only",
    ),
    "an arm file setting a shared key": (
        setting("cpg.yaml", "epochs", 4),
        "cpg.yaml",
        "cpg.yaml sets ['epochs'], which train.yaml sets too: a file that extends it adds only"
        " its own keys",
    ),
    "a date": (
        setting("cpg.yaml", "fixed", "2026-11-25"),
        "cpg.yaml",
        "cpg.yaml: missing keys [], unknown keys ['fixed']",
    ),
    "a date as YAML reads it": (
        append("cpg.yaml", "fixed: 2026-11-25\n"),
        "cpg.yaml",
        "cpg.yaml with train.yaml holds a value that JSON cannot hold, such as a date or a"
        " number that is not finite",
    ),
    "a value that is not a number": (
        replace("train.yaml", "  margin: 0.3 #", "  margin: .nan #"),
        "cpg.yaml",
        "cpg.yaml with train.yaml holds a value that JSON cannot hold",
    ),
}


@pytest.mark.parametrize("case", list(FILE_REFUSALS))
def test_load_arm_refuses_files_that_do_not_read_as_one_level_of_extends(
    folder: Path, case: str
) -> None:
    change, arm, message = FILE_REFUSALS[case]
    change(folder)
    with pytest.raises(StepError, match=f"^{re.escape(message)}"):
        load_arm(folder / arm)


def test_the_resolver_refuses_zeroshot_yaml_setting_a_shared_key(folder: Path) -> None:
    plant(folder, "zeroshot.yaml", "prompt", PROMPT)
    with pytest.raises(StepError, match=re.escape("zeroshot.yaml sets ['prompt'], which")):
        resolve_config(folder / "zeroshot.yaml")


# Refusals: the arm files --------------------------------------------------------------------------


# (arm file, key, planted value or DROP, the message)
ARM_REFUSALS = [
    ("cpg", "arm", DROP, "cpg.yaml: arm must be one of ['b0', 'b1', 'cpg'], not None"),
    ("cpg", "arm", "b2", "cpg.yaml: arm must be one of ['b0', 'b1', 'cpg'], not 'b2'"),
    ("cpg", "arm", "zeroshot", "cpg.yaml: arm must be one of ['b0', 'b1', 'cpg'], not 'zeroshot'"),
    ("cpg", "arm", "b1", "cpg.yaml holds arm 'b1': an arm file is named after its arm"),
    ("b0", "arm", "b1", "b0.yaml holds arm 'b1': an arm file is named after its arm"),
    ("cpg", "notes", "x", "cpg.yaml: missing keys [], unknown keys ['notes']"),
    ("cpg", "views", DROP, "cpg.yaml: missing keys ['views'], unknown keys []"),
    ("cpg", "lambda_g", DROP, "cpg.yaml: missing keys ['lambda_g'], unknown keys []"),
    ("b1", "lambda_g", DROP, "b1.yaml: missing keys ['lambda_g'], unknown keys []"),
    ("b0", "lambda_g", 0.0, "b0.yaml: missing keys [], unknown keys ['lambda_g']"),
    (
        "cpg",
        "views",
        ["x0", "x1", "x2"],
        "cpg.yaml: views must be ['x1', 'x0', 'x2'] for arm cpg, not ['x0', 'x1', 'x2']",
    ),
    ("b1", "views", ["x1", "x0"], "b1.yaml: views must be ['x1', 'x0', 'x2'] for arm b1, not"),
    ("b0", "views", ["x1", "x0", "x2"], "b0.yaml: views must be ['x0'] for arm b0, not"),
    ("b0", "views", ["x0", "rev"], "b0.yaml: views must be ['x0'] for arm b0, not"),
    ("b0", "views", "x0", "b0.yaml: views must be ['x0'] for arm b0, not 'x0'"),
    ("b1", "lambda_g", 1.0, "b1.yaml: lambda_g must be 0.0, since B1 is CPG's code with λG = 0"),
    ("b1", "lambda_g", 0, "b1.yaml: lambda_g must be 0.0, since B1 is CPG's code with λG = 0"),
    ("b1", "lambda_g", -0.5, "b1.yaml: lambda_g must be 0.0"),
    ("cpg", "lambda_g", 0.0, "cpg.yaml: lambda_g must be a float above 0, not 0.0"),
    ("cpg", "lambda_g", -1.0, "cpg.yaml: lambda_g must be a float above 0, not -1.0"),
    ("cpg", "lambda_g", 1, "cpg.yaml: lambda_g must be a float above 0, not 1"),
    ("cpg", "lambda_g", True, "cpg.yaml: lambda_g must be a float above 0, not True"),
    ("cpg", "lambda_g", "1.0", "cpg.yaml: lambda_g must be a float above 0, not '1.0'"),
    ("cpg", "lambda_g", float("inf"), "cpg.yaml with train.yaml holds a value that JSON"),
]


@pytest.mark.parametrize(
    ("arm", "key", "value", "message"),
    ARM_REFUSALS,
    ids=[f"{arm}.{key}={value!r}" for arm, key, value, _ in ARM_REFUSALS],
)
def test_load_arm_refuses_an_arm_file_that_breaks_its_arm(
    folder: Path, arm: str, key: str, value: Any, message: str
) -> None:
    plant(folder, f"{arm}.yaml", key, value)
    with pytest.raises(StepError, match=f"^{re.escape(message)}"):
        load_arm(folder / f"{arm}.yaml")


# Refusals: the shared settings --------------------------------------------------------------------


def rule(path: str, value: Any, what: str) -> tuple[str, Any, str]:
    """A planted value of train.yaml and the message that refuses it."""
    return (path, value, f"train.yaml: {path} must be {what}, not {value!r}")


WHOLE = "a whole number of at least"
VIEW_NAMES = "a list of different names from ['x0', 'x1', 'x2', 'rev']"
# (dotted key of train.yaml, planted value or DROP, the message)
SHARED_REFUSALS = [
    ("bags", {"segments_per_bag": 4}, "train.yaml: missing keys [], unknown keys ['bags']"),
    ("timing", DROP, "train.yaml: missing keys ['timing'], unknown keys []"),
    ("lora", [16, 32], "train.yaml: lora must be a mapping, not [16, 32]"),
    ("lora", None, "train.yaml: lora must be a mapping, not None"),
    ("lora.rnak", 16, "train.yaml: lora: missing keys [], unknown keys ['rnak']"),
    (
        "precision.scaler.growth_factor",
        DROP,
        "train.yaml: precision.scaler: missing keys ['growth_factor'], unknown keys []",
    ),
    ("rules.alive", "2-3", "train.yaml: rules.alive must be a mapping, not '2-3'"),
    rule("model", "llama", "'voxtral'"),
    rule("prompt", "Answer Yes or No.", repr(PROMPT)),
    rule("seeds.round_1", -1, f"{WHOLE} 0"),
    rule("seeds.round_2", 102.0, f"{WHOLE} 0"),
    rule("seeds.round_1", True, f"{WHOLE} 0"),
    rule("epochs", 0, f"{WHOLE} 1"),
    rule("epochs", 3.0, f"{WHOLE} 1"),
    rule("lora.rank", 0, f"{WHOLE} 1"),
    rule("lora.alpha", 0, f"{WHOLE} 1"),
    rule("lora.alpha", 32.0, f"{WHOLE} 1"),
    rule("lora.dropout", 1.0, "a float from 0 to below 1"),
    rule("lora.dropout", -0.05, "a float from 0 to below 1"),
    rule("lora.dropout", 0, "a float from 0 to below 1"),
    rule("lora.bias", "all", "'none'"),
    rule("lora.init_lora_weights", "true", "true or false"),
    rule("lora.use_rslora", 0, "true or false"),
    rule("lora.use_dora", None, "true or false"),
    rule("lora.trained_adapter", "audio_tower", "'multi_modal_projector'"),
    rule("lora.targets", "(q_proj", "a regular expression"),
    rule("lora.targets", "", "a regular expression"),
    rule("lora.targets", ["q_proj", "v_proj"], "a regular expression"),
    rule("optimiser.learning_rate", 0.0, "a float above 0"),
    rule("optimiser.learning_rate", "1e-4", "a float above 0"),  # what PyYAML makes of 1e-4
    rule("optimiser.adapter_learning_rate", -2.0e-5, "a float above 0"),
    rule("optimiser.betas", [0.9], "a list of 2 floats from 0 to below 1"),
    rule("optimiser.betas", [0.9, 1.0], "a list of 2 floats from 0 to below 1"),
    rule("optimiser.betas", [0.9, 0.999, 0.9], "a list of 2 floats from 0 to below 1"),
    rule("optimiser.betas", [0.9, "0.999"], "a list of 2 floats from 0 to below 1"),
    rule("optimiser.betas", 0.9, "a list of 2 floats from 0 to below 1"),
    rule("optimiser.eps", 0.0, "a float above 0"),
    rule("optimiser.amsgrad", "false", "true or false"),
    rule("optimiser.foreach", 1, "true or false"),
    rule("optimiser.weight_decay", -0.01, "a float of at least 0"),
    rule("optimiser.grad_clip_norm", 0.0, "a float above 0"),
    rule("optimiser.warmup_share", 1.5, "a float from 0 to 1"),
    rule("optimiser.warmup_share", -0.05, "a float from 0 to 1"),
    rule("precision.autocast", "bf16", "'fp16'"),
    rule("precision.masters", "fp16", "'fp32'"),
    rule("precision.scaler.init_scale", 0.0, "a float above 0"),
    rule("precision.scaler.init_scale", 65536, "a float above 0"),
    rule("precision.scaler.growth_factor", 1.0, "a float above 1"),
    rule("precision.scaler.backoff_factor", 1.0, "a float above 0 and below 1"),
    rule("precision.scaler.backoff_factor", 0.0, "a float above 0 and below 1"),
    rule("precision.scaler.growth_interval_steps", 0, f"{WHOLE} 1"),
    rule("precision.checkpointing", "none", "'every_layer'"),
    rule("precision.deterministic", "on", "one of ['pending', 'strict', 'warn_only']"),
    rule("precision.deterministic", True, "one of ['pending', 'strict', 'warn_only']"),
    rule("precision.deterministic", "Strict", "one of ['pending', 'strict', 'warn_only']"),
    rule("precision.strict_max_extra_time_share", -0.1, "a float of at least 0"),
    rule("sampler.positive_bags", 0, f"{WHOLE} 1"),
    rule("sampler.negative_bags", 0, f"{WHOLE} 1"),
    rule("sampler.epoch_end", "all_bags_used", "'a_class_runs_out'"),
    rule("sampler.draw", "random", "'least_trained_first'"),
    rule("sampler.bag_choice", "random", "'lowest_trained_share_first'"),
    rule("sampler.step_order", "by_share", "'shuffled'"),
    rule("loss.x1_weight", -0.25, "a float of at least 0"),
    rule("loss.x2_weight", 1, "a float of at least 0"),
    rule("gain.margin", "0.3", "a finite float"),
    rule("gain.margin", None, "a finite float"),
    rule("gain.softness", 0.0, "a float above 0"),
    rule("gain.ramp_share", 0.0, "a float above 0 and at most 1"),
    rule("gain.ramp_share", 1.1, "a float above 0 and at most 1"),
    rule("gain.momentum", 1.0, "a float from 0 to below 1"),
    rule("gain.start_steps", 0, f"{WHOLE} 1"),
    rule("gain.floor_share", -0.25, "a float of at least 0"),
    rule("gain.floor_logit", 0.0, "a float above 0"),
    rule("rules.f1_min_scale", 0.0, "a float above 0"),
    rule("rules.alive.epochs", [3, 2], "a list of rising whole numbers of at least 1"),
    rule("rules.alive.epochs", [2, 2], "a list of rising whole numbers of at least 1"),
    rule("rules.alive.epochs", [0, 1], "a list of rising whole numbers of at least 1"),
    rule("rules.alive.epochs", [], "a list of rising whole numbers of at least 1"),
    rule("rules.alive.epochs", [2.0, 3.0], "a list of rising whole numbers of at least 1"),
    rule("rules.alive.epochs", [True, 3], "a list of rising whole numbers of at least 1"),
    rule("rules.alive.epochs", "2, 3", "a list of rising whole numbers of at least 1"),
    (
        "rules.alive.epochs",
        [2, 4],
        "train.yaml: rules.alive.epochs must lie within the 3 epochs, not [2, 4]",
    ),
    rule("rules.alive.min_share", -0.2, "a float from 0 to 1"),
    rule("rules.alive.max_share", 1.2, "a float from 0 to 1"),
    rule("rules.alive.min_share", 0.8, "below max_share (0.8)"),
    rule("rules.alive.min_share", 0.9, "below max_share (0.8)"),
    rule("rules.alive.margin_if_below", "0.5", "a finite float"),
    rule("rules.alive.margin_if_above", None, "a finite float"),
    rule("rules.falling.loss", "total", "'ce_part'"),
    rule("rules.falling.min_drop_share", 1.5, "a float from 0 to 1"),
    rule("rules.falling.min_drop_share", -0.05, "a float from 0 to 1"),
    rule("rules.falling.min_standard_errors", -2.0, "a float of at least 0"),
    rule("rules.falling.min_standard_errors", 2, "a float of at least 0"),
    rule("rules.falling.epochs_if_falling", 0, f"{WHOLE} 1"),
    rule("rules.f5_max_hours", 0.0, "a float above 0"),
    rule("rules.f5_max_hours", 4, "a float above 0"),
    rule("scoring.views", [], VIEW_NAMES),
    rule("scoring.views", ["x0", "x0"], VIEW_NAMES),
    rule("scoring.views", ["x0", "x3"], VIEW_NAMES),
    rule("scoring.views", "x0", VIEW_NAMES),
    rule("scoring.format", "fp32", "'fp16'"),
    rule("timing.steps_per_mode", 0, f"{WHOLE} 1"),
    rule("timing.first_steps_left_out", -1, f"{WHOLE} 0"),
    rule("timing.first_steps_left_out", 10, "below steps_per_mode (10)"),
    rule("timing.stand_in_start.mu_logit", None, "a finite float"),
    rule("timing.stand_in_start.sigma_logit", 0.0, "a float above 0"),
]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    SHARED_REFUSALS,
    ids=[f"{path}={value!r}" for path, value, _ in SHARED_REFUSALS],
)
def test_load_arm_refuses_a_shared_setting_of_the_wrong_key_type_or_range(
    folder: Path, path: str, value: Any, message: str
) -> None:
    plant(folder, "train.yaml", path, value)
    with pytest.raises(StepError) as raised:
        load_arm(folder / "cpg.yaml")
    assert str(raised.value) == message


@pytest.mark.parametrize("arm", ["b0", "b1"])
def test_every_arm_reads_the_shared_settings_with_the_same_checks(folder: Path, arm: str) -> None:
    plant(folder, "train.yaml", "gain.softness", 0.0)
    with pytest.raises(StepError, match=r"^train\.yaml: gain\.softness must be a float above 0"):
        load_arm(folder / f"{arm}.yaml")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("lora.dropout", 0.0),
        ("optimiser.warmup_share", 0.0),
        ("optimiser.weight_decay", 0.0),
        ("optimiser.betas", [0.0, 0.0]),
        ("gain.margin", -1.5),
        ("gain.momentum", 0.0),
        ("gain.ramp_share", 1.0),
        ("gain.floor_share", 0.0),
        ("loss.x1_weight", 0.0),
        ("rules.alive.min_share", 0.0),
        ("rules.alive.max_share", 1.0),
        ("rules.alive.epochs", [3]),
        ("rules.falling.min_standard_errors", 0.0),
        ("precision.strict_max_extra_time_share", 0.0),
        ("seeds.round_1", 0),
        ("timing.first_steps_left_out", 0),
        ("timing.first_steps_left_out", 9),
        ("timing.stand_in_start.mu_logit", -2.5),
        ("scoring.views", ["rev", "x1", "x2", "x0"]),
    ],
)
def test_load_arm_keeps_a_value_at_the_edge_of_its_range(
    folder: Path, path: str, value: Any
) -> None:
    plant(folder, "train.yaml", path, value)
    shared = load_arm(folder / "cpg.yaml").shared
    *blocks, key = path.split(".")
    found: Any = shared
    for part in [*blocks, key]:
        found = getattr(found, part)
    assert found == (tuple(value) if isinstance(value, list) else value)
