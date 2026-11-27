"""The settings of the pilot's analysis, pinned value by value.

Every value of configs/pilot/evaluate.yaml is pinned here with its type (3 is not 3.0, and
true is not 1) and in its order, as docs/pilot_analysis.md fixed them on 2026-11-26, before
any AUROC of any arm was computed. The pins of the 30 kept training runs (runs) were added on
2026-11-27, after training, in the commit that pinned them, which changed this test with them.
"""

import json
from pathlib import Path
from typing import Any

import yaml

from listening_beyond_words.evaluation import EvaluateSettings, pinned_version

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "pilot"
RUNS = {
    "b0": {
        "r1": {
            "f1": {
                "run": "runs/pilot/pilot-b0-r1-f1-20261126-055122",
                "scores_sha256": "1164e967937aa633de8e90eaebbac4aea0515b9f70ab1f414bd4ff85d47fddf2",
                "record_sha256": "e80e44c6dce8cc9f905c55d087fe7cca880113391c3137f26ab8fedfd979c8ac",
            },
            "f2": {
                "run": "runs/pilot/pilot-b0-r1-f2-20261126-055122",
                "scores_sha256": "af25715b68e2451ce816deabd36a05c2c8f5e33d2d0d4df814ae6eed9a452ba9",
                "record_sha256": "223890dbbc66b93ed9c1c90e1569bbb10cbdc8ed7b2c4b29099d4620a502cd72",
            },
            "f3": {
                "run": "runs/pilot/pilot-b0-r1-f3-20261126-055122",
                "scores_sha256": "6bf23c81504445747087bc7bdd5749702e42f1010bac324582a0b36b9745d426",
                "record_sha256": "6e33cb9efeb6bff267112c868c7539c2598b941000985ed1d7125cfb6b98606b",
            },
            "f4": {
                "run": "runs/pilot/pilot-b0-r1-f4-20261126-055122",
                "scores_sha256": "0d5266836ca2318dcf9fa4e68c044045af9fe62016699ab2ceb27b12c7327bb9",
                "record_sha256": "40933f128a9564db4ff8cd02c99fc16c8a2910a619620ee64f99b58dbb601fc6",
            },
            "f5": {
                "run": "runs/pilot/pilot-b0-r1-f5-20261126-055122",
                "scores_sha256": "8ef3fdc45eb0acd0af08f13f60eb744a969d97bbe62f0ec840b5c880d89b026d",
                "record_sha256": "2598b84a9fc673bffff375d3a386abd978a586baf1f961a2668d645f2d9afa5f",
            },
        },
        "r2": {
            "f1": {
                "run": "runs/pilot/pilot-b0-r2-f1-20261126-131819",
                "scores_sha256": "ddb96d13d36613ab82c354a43741312a364092359eb3ed2c2f5d7aa7324b7c07",
                "record_sha256": "7def13bcd46e6f88b16fc40ae44028ccfb41d85b45f7d07589e0e58845f89bbb",
            },
            "f2": {
                "run": "runs/pilot/pilot-b0-r2-f2-20261126-133232",
                "scores_sha256": "b493c29899376fbe23c75ff5b6156d85ea2c6a9324b812ae7fa39564eb90505b",
                "record_sha256": "e94aa891f5a889a5371559aa5d224da490a2af6eee89f176b91366ea00634b79",
            },
            "f3": {
                "run": "runs/pilot/pilot-b0-r2-f3-20261126-133456",
                "scores_sha256": "31ee8286aa3c0309f7f668132a1af01e844a9832e622aa11286a4e3ad5254904",
                "record_sha256": "61e5ef6e598a2a3fb9a60b49335e6a298c69241ea50607394aeae8c56356ca88",
            },
            "f4": {
                "run": "runs/pilot/pilot-b0-r2-f4-20261126-134625",
                "scores_sha256": "fd13916bf36ffdae08c1db3ffd9e1d8f495b7240f6104fc0db468ec8a9b3b111",
                "record_sha256": "4d35e008ad645dbec9a34ce5db91e06e4ab3c5ee556f5fc739146f16f1af28e8",
            },
            "f5": {
                "run": "runs/pilot/pilot-b0-r2-f5-20261126-140114",
                "scores_sha256": "fd54f78afadb084e296fe92ed779e631815f8e734bd4dfbc5232202537545ff0",
                "record_sha256": "b8873e3514be36ac96618e4f5f65b46c62ac055692574761675220b5c99c534d",
            },
        },
    },
    "b1": {
        "r1": {
            "f1": {
                "run": "runs/pilot/pilot-b1-r1-f1-20261126-143323",
                "scores_sha256": "2ded8e0914a858466f99aeebcf01e77dca48a29b2a5703d6d1c0e5d222962c62",
                "record_sha256": "c11b791181d85ee765cba0482c605087dfa6065c1a2f56813bb7b2382977def2",
            },
            "f2": {
                "run": "runs/pilot/pilot-b1-r1-f2-20261126-143323",
                "scores_sha256": "8dd836efcfcc205dcb0f30c315604bdeaac1ee2a6e67b2e6c8b59bb9e4e525ae",
                "record_sha256": "3e6788d934118623b6f661e431b73fe210fbd781a6f1af36c0c3182fb26743f9",
            },
            "f3": {
                "run": "runs/pilot/pilot-b1-r1-f3-20261126-154329",
                "scores_sha256": "332b31ab66d122673cad3668886bba1701e06cfa7affd5782a3842cefcee847d",
                "record_sha256": "c71667c04a5b9ffb16e70e5869023f4d60fbc9738d55cbcda16712ec2811565b",
            },
            "f4": {
                "run": "runs/pilot/pilot-b1-r1-f4-20261126-154329",
                "scores_sha256": "a57ae78419d1a829d84137930c8628c19fb9491879dbf4b3c2f6e525fdfa4045",
                "record_sha256": "f7618860b1e573a71fd96dc9618013816be1b534233fcfc149149428414186b9",
            },
            "f5": {
                "run": "runs/pilot/pilot-b1-r1-f5-20261126-154451",
                "scores_sha256": "d2548f698f5f7d086580e8b7740589a7b586ae160eb4e8483c5217a44fb2f359",
                "record_sha256": "63ccde4556f4720acacd720cf82deb40f2eac5e8debcc28b585244d926b6eec0",
            },
        },
        "r2": {
            "f1": {
                "run": "runs/pilot/pilot-b1-r2-f1-20261126-165420",
                "scores_sha256": "0ed071dcae60af5018c5af5b764ad32f20798866202cbbba06d114abec7f42db",
                "record_sha256": "9a8b7abd0ad5fe042ff4044bef1ec41bb53593a102b7fc8dba3476b88ce83370",
            },
            "f2": {
                "run": "runs/pilot/pilot-b1-r2-f2-20261126-165654",
                "scores_sha256": "f95ef0c8b898365a13f5fd04a769b555491b8158ceae4a9854c4cdd14b0bc5b8",
                "record_sha256": "1d040fba6678a8026846b3eda907d7f67bab35a9078853e8e7563d8f02ae2272",
            },
            "f3": {
                "run": "runs/pilot/pilot-b1-r2-f3-20261126-165654",
                "scores_sha256": "5e78397cdec6d2a85cef268013d42a1f1a3e91abee3ecbfac1dc59a0f4c19881",
                "record_sha256": "6dda7da75eea9ab91a8966725e3cd01b587b3a389727ff9da4f347a0bc40df8e",
            },
            "f4": {
                "run": "runs/pilot/pilot-b1-r2-f4-20261126-165420",
                "scores_sha256": "f67f7bf2cb84cb3074d44bca3b7c61e1098df62f33c6b627acb87fd547f53243",
                "record_sha256": "984afb4bdf6a5a6f3a51398e9f9c00698226e67cce7a83e13ec1f747de535acb",
            },
            "f5": {
                "run": "runs/pilot/pilot-b1-r2-f5-20261126-165420",
                "scores_sha256": "53c8f1b7746a494c74852fd5880d8fc23d2b8273a22f1e225a9a1c9809287e51",
                "record_sha256": "6be83f4a981c1b4fcda3f02b06324c8fe3ce37e0e28884ffa7affa61cb9e0fe7",
            },
        },
    },
    "cpg": {
        "r1": {
            "f1": {
                "run": "runs/pilot/pilot-cpg-r1-f1-20261126-055122",
                "scores_sha256": "2ff398a575dd8417422228ffd244d86ada0a39d28006b0fbbbd108f1a68c72c1",
                "record_sha256": "0bf9bed4b8dba4f538769071ca954ae3e580c3a5e3728dd5e54ab0572a33d5cc",
            },
            "f2": {
                "run": "runs/pilot/pilot-cpg-r1-f2-20261126-130259",
                "scores_sha256": "6e208dc3d91d0c06378b9b57519cb08713d771b196a52b804c983ba45ba7e137",
                "record_sha256": "093907edda63b245183e654946eadebe673795a954167adc31f88489237bd17f",
            },
            "f3": {
                "run": "runs/pilot/pilot-cpg-r1-f3-20261126-130251",
                "scores_sha256": "a28e061ac2138fae51eb71045001d9eb81bd250dfa48011691380e7fa3aa4464",
                "record_sha256": "7d801c1a9d3adf31a3d270ad9db5011a027a9b89878b3513c38f5e3fa7dc4100",
            },
            "f4": {
                "run": "runs/pilot/pilot-cpg-r1-f4-20261126-130251",
                "scores_sha256": "fa262244f9d03205ace3ce4962a22d6aee8591a406acc7523fd9b5da5eaac875",
                "record_sha256": "4d18d33d61b4f16a3ffc4fd70f849be51ea38c242b09f2cb8b3c7e68dfc978da",
            },
            "f5": {
                "run": "runs/pilot/pilot-cpg-r1-f5-20261126-130251",
                "scores_sha256": "309a0025f4c1a73b2d904fe4880832e9dff9d30967a1c341d31bb11887ea5708",
                "record_sha256": "8f4b7603d7be3b1c06e1b4c4261cf0316e13caab429246ed8d1be12b5fd135f4",
            },
        },
        "r2": {
            "f1": {
                "run": "runs/pilot/pilot-cpg-r2-f1-20261126-143323",
                "scores_sha256": "66df8907d67ba840a48251efce05f32b59fb59875428c0a38131e1b829896678",
                "record_sha256": "e9fa0f9400f4a973cdd4445bf5e38cfcf2647bb02fc21cc671daffd2b39ce295",
            },
            "f2": {
                "run": "runs/pilot/pilot-cpg-r2-f2-20261126-143323",
                "scores_sha256": "26adad620b316cd67cef6ba83b641e566c236513fd36c80ecd8704174c9890a3",
                "record_sha256": "925b8fd173200bd173e3aeae5f707cadffb03a1c7c90197d1a87c7e23132371f",
            },
            "f3": {
                "run": "runs/pilot/pilot-cpg-r2-f3-20261126-143323",
                "scores_sha256": "1a932c9455090aa173ae4cfe3ed2d346c45adf4ded695a4758471c051dc4e6f4",
                "record_sha256": "93e26753f0b76864d8ba42493e9b89b33aea8d73be064f0d43459047a2f0ae95",
            },
            "f4": {
                "run": "runs/pilot/pilot-cpg-r2-f4-20261126-143323",
                "scores_sha256": "6399cb898b2245293c506daaa9bb1270ca0c11d0156cb99c58a24c2b7e581b3b",
                "record_sha256": "69bdb6201b31f8b82a6a42a18870794704b5ecb078e34c8a49139d27ec588558",
            },
            "f5": {
                "run": "runs/pilot/pilot-cpg-r2-f5-20261126-143323",
                "scores_sha256": "1ae5dfeff86b911b08bd37064a55d2a062f1b99a99fe9e2503130c1c24590e21",
                "record_sha256": "f4ea5371c4f48c5efccef15ceba5e24f58c291ee59f1da77b4d0cdf5be000e41",
            },
        },
    },
}
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
    "runs": RUNS,
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


def test_the_pinned_runs_are_all_30_runs_as_the_job_reads_them() -> None:
    settings = EvaluateSettings.from_config(load("evaluate.yaml"))
    assert len(settings.runs) == 30
    version = pinned_version(tuple(settings.runs))
    assert version is not None
    assert version.runs == 30
