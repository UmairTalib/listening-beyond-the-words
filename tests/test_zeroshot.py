"""The zero-shot scores (zeroshot.py): the settings, the check's sample and the pinned scores."""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from listening_beyond_words import cache
from listening_beyond_words.cache import CACHE_COLUMNS
from listening_beyond_words.runinfo import StepError, item_seed, sha256_file
from listening_beyond_words.settings import Workspace
from listening_beyond_words.zeroshot import (
    SCORE_COLUMNS,
    VIEWS,
    ZeroshotSettings,
    check_sample,
    load_zeroshot_scores,
)

ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = ("300_000", "300_001", "301_000", "301_001", "302_000")
LONG = "301_001"  # two chunks in x0 and rev, as the 37.4 s clip
RUN = "runs/pilot/pilot-zeroshot-20261201-120000"
CHECK_RUN = "derived/cache_check/run-10"
SETTINGS = {
    "extends": "train.yaml",
    "arm": "zeroshot",
    "views": ["x0", "x1", "rev"],
    "check": {"segments_per_view": 20, "seed": 108},
}


# Settings --------------------------------------------------------------------------------------


def test_the_settings_read_the_pinned_zeroshot_config() -> None:
    config = yaml.safe_load((ROOT / "configs" / "pilot" / "zeroshot.yaml").read_text("utf-8"))
    assert ZeroshotSettings.from_config(config) == ZeroshotSettings(
        views=("x0", "x1", "rev"), segments_per_view=20, seed=108
    )
    assert tuple(config["views"]) == VIEWS


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c.pop("check"), "zeroshot.yaml: missing keys ['check'], unknown keys []"),
        (lambda c: c.pop("extends"), "zeroshot.yaml: missing keys ['extends'], unknown keys []"),
        (
            lambda c: c.update(extends="other.yaml"),
            "zeroshot.yaml must extend 'train.yaml', the shared training settings in its"
            " folder (engineering conventions, section 5), not 'other.yaml'",
        ),
        (
            lambda c: c.update(extends="../pilot/train.yaml"),
            "zeroshot.yaml must extend 'train.yaml', the shared training settings",
        ),
        (lambda c: c.update(extends=None), "zeroshot.yaml must extend 'train.yaml'"),
        (lambda c: c.update(notes="x"), "zeroshot.yaml: missing keys [], unknown keys ['notes']"),
        (lambda c: c.update(arm="b0"), "zeroshot.yaml: the arm must be 'zeroshot', not 'b0'"),
        (lambda c: c.update(views=["x0", "x1", "x2", "rev"]), "zeroshot.yaml: the views must be"),
        (lambda c: c.update(views=["x0", "rev", "x1"]), "zeroshot.yaml: the views must be"),
        (lambda c: c.update(views="x0 x1 rev"), "zeroshot.yaml: the views must be"),
        (lambda c: c.update(check=[20, 108]), "zeroshot.yaml check must be a mapping"),
        (lambda c: c["check"].pop("seed"), "zeroshot.yaml check: missing keys ['seed']"),
        (
            lambda c: c["check"].update(segments_per_view=0),
            "zeroshot.yaml: check.segments_per_view must be a whole",
        ),
        (
            lambda c: c["check"].update(segments_per_view=True),
            "zeroshot.yaml: check.segments_per_view must be a whole",
        ),
        (
            lambda c: c["check"].update(seed=108.0),
            "zeroshot.yaml: check.seed must be a whole number",
        ),
        (
            lambda c: c["check"].update(seed="108"),
            "zeroshot.yaml: check.seed must be a whole number",
        ),
    ],
)
def test_the_settings_refuse_a_missing_unknown_or_wrong_value(
    change: Callable[[dict[str, Any]], Any], message: str
) -> None:
    config = json.loads(json.dumps(SETTINGS))
    change(config)
    with pytest.raises(StepError, match=f"^{re.escape(message)}"):
        ZeroshotSettings.from_config(config)


# The check's sample ----------------------------------------------------------------------------


def sample_manifest(count: int = 30, long: tuple[str, ...] = ("301_007",)) -> pd.DataFrame:
    segments = [f"{300 + index // 10}_{index % 10:03d}" for index in range(count)]
    return pd.DataFrame({"segment": segments, "chunks": [2 if s in long else 1 for s in segments]})


def test_check_sample_draws_from_its_own_stream_per_view_and_adds_every_long_clip() -> None:
    manifest = sample_manifest()
    sample = check_sample(manifest, "x0", 5, 108)
    segments = sorted(manifest["segment"])
    rng = np.random.default_rng(item_seed(108, "zeroshot", "x0"))
    drawn = {segments[int(index)] for index in rng.choice(30, size=5, replace=False)}
    assert sample == tuple(sorted(drawn | {"301_007"}))
    assert check_sample(manifest.iloc[::-1], "x0", 5, 108) == sample  # the row order is not read
    assert check_sample(manifest, "rev", 5, 108) != sample  # a stream of its own per view
    assert check_sample(manifest, "x0", 5, 109) != sample
    assert check_sample(manifest, "x0", 50, 108) == tuple(segments)  # all, if fewer than asked


def test_the_zero_shot_check_draws_apart_from_the_cache_check() -> None:
    # Both draw with seed 108, each from a stream of its own (pilot protocol: 20 segments per
    # view for the zero-shot check, 25 for the cache check).
    manifest = sample_manifest(count=200, long=())
    for view in VIEWS:
        mine = check_sample(manifest, view, 20, 108)
        theirs = cache.check_sample(manifest, view, 20, 108)
        assert len(mine) == len(theirs) == 20
        assert mine != theirs


def test_check_sample_keeps_every_clip_of_two_chunks_however_small_the_draw() -> None:
    long = ("300_003", "302_009")
    sample = check_sample(sample_manifest(long=long), "rev", 1, 108)
    assert set(long) <= set(sample)
    assert len(sample) in (2, 3)


# The pinned scores -----------------------------------------------------------------------------


@dataclass
class Pinned:
    """Cache manifests of the three views, a zero-shot run and their pins, as the jobs write them."""

    workspace: Workspace
    config: dict[str, Any]

    @property
    def run(self) -> Path:
        return self.workspace.root / RUN

    def scores(self) -> pd.DataFrame:
        return pd.read_parquet(self.run / "scores.parquet")

    def edit(self, change: Callable[[dict[str, Any]], Any]) -> None:
        info = json.loads((self.run / "run.json").read_text(encoding="utf-8"))
        change(info)
        (self.run / "run.json").write_text(json.dumps(info), encoding="utf-8")

    def repin(self, table: pd.DataFrame) -> None:
        """Write other scores and pin them in full: run.json and the config."""
        table.to_parquet(self.run / "scores.parquet", index=False)
        sha256 = sha256_file(self.run / "scores.parquet")
        self.edit(lambda info: info["outputs"].update({"scores.parquet": sha256}))
        self.config["zeroshot_scores"]["sha256"] = sha256


def z_of(view: str, segment: str) -> float:
    """A distinct score per view and segment."""
    return round(VIEWS.index(view) - 1.0 + int(segment[:3]) / 1000 + int(segment[4:]) / 100, 6)


def make_pinned(root: Path) -> Pinned:
    root.mkdir(parents=True)
    workspace = Workspace(root.resolve())
    pins = {}
    rows = []
    for number, view in enumerate(VIEWS):
        run = f"cache/{view}/run-{number + 1}"
        manifest = pd.DataFrame(
            [
                {
                    "segment": segment,
                    "session": int(segment[:3]),
                    "file": f"{segment[:3]}/{segment}.npy",
                    "chunks": 2 if segment == LONG and view != "x1" else 1,
                    "samples": 598_400 if segment == LONG and view != "x1" else 160_000,
                    "sha256": f"{index:064x}",
                    "audio_sha256": f"{index + 100:064x}",
                    "features_sha256": f"{index + 200:064x}",
                    "max_abs": 9.5,
                }
                for index, segment in enumerate(SEGMENTS)
            ],
            columns=list(CACHE_COLUMNS),
        )
        (root / run).mkdir(parents=True)
        manifest.to_parquet(root / run / "manifest.parquet", index=False)
        pins[view] = {
            "run": run,
            "sha256": sha256_file(root / run / "manifest.parquet"),
            "verified_by": CHECK_RUN,
        }
        rows += [
            {"segment": s, "session": int(s[:3]), "view": view, "z": z_of(view, s)}
            for s in SEGMENTS
        ]
    folder = root / RUN
    folder.mkdir(parents=True)
    pd.DataFrame(rows, columns=list(SCORE_COLUMNS)).to_parquet(
        folder / "scores.parquet", index=False
    )
    sha256 = sha256_file(folder / "scores.parquet")
    info = {
        "run_id": folder.name,
        "step": "score_zeroshot",
        "arm": "zeroshot",
        "status": "done",
        "settings": {"zeroshot": SETTINGS},
        "inputs": {"caches": json.loads(json.dumps(pins))},
        "outputs": {"scores.parquet": sha256},
    }
    (folder / "run.json").write_text(json.dumps(info), encoding="utf-8")
    config = {"cache_runs": pins, "zeroshot_scores": {"run": RUN, "sha256": sha256}}
    return Pinned(workspace, config)


@pytest.fixture
def pinned(tmp_path: Path) -> Pinned:
    return make_pinned(tmp_path / "workspace")


def test_load_zeroshot_scores_gives_one_z_per_segment_and_view_by_view_and_segment(
    pinned: Pinned,
) -> None:
    pinned.repin(pinned.scores().iloc[::-1].reset_index(drop=True))  # the file's order is not kept
    loaded = load_zeroshot_scores(pinned.workspace, pinned.config)
    assert tuple(loaded.columns) == SCORE_COLUMNS
    assert list(loaded["view"]) == [view for view in VIEWS for _ in SEGMENTS]
    assert list(loaded["segment"]) == list(SEGMENTS) * len(VIEWS)
    for row in loaded.itertuples(index=False):
        assert row.z == z_of(row.view, row.segment)
        assert row.session == int(row.segment[:3])
    assert list(loaded.index) == list(range(len(loaded)))


def test_load_zeroshot_scores_accepts_any_spelling_of_the_pinned_run(pinned: Pinned) -> None:
    pinned.config["zeroshot_scores"]["run"] = RUN + "/"
    assert len(load_zeroshot_scores(pinned.workspace, pinned.config)) == 15


def drop_a_segment(pinned: Pinned) -> None:
    scores = pinned.scores()
    pinned.repin(scores[~((scores["view"] == "x1") & (scores["segment"] == "301_000"))])


def add_a_segment(pinned: Pinned) -> None:
    scores = pinned.scores()
    extra = scores.iloc[[0]].assign(segment="303_000", session=303)
    pinned.repin(pd.concat([scores, extra], ignore_index=True))


def repeat_a_segment(pinned: Pinned) -> None:
    scores = pinned.scores()
    pinned.repin(pd.concat([scores, scores.iloc[[6]]], ignore_index=True))


def plant(value: float) -> Callable[[Pinned], None]:
    def change(pinned: Pinned) -> None:
        scores = pinned.scores()
        scores.loc[(scores["view"] == "rev") & (scores["segment"] == LONG), "z"] = value
        pinned.repin(scores)

    return change


def move_a_segment_to_another_session(pinned: Pinned) -> None:
    scores = pinned.scores()
    scores.loc[scores["segment"] == "302_000", "session"] = 301
    pinned.repin(scores)


def score_x2_too(pinned: Pinned) -> None:
    scores = pinned.scores()
    pinned.repin(pd.concat([scores, scores[scores["view"] == "x0"].assign(view="x2")]))


def rename_a_column(pinned: Pinned) -> None:
    pinned.repin(pinned.scores().rename(columns={"z": "score"}))


def store_z_as_text(pinned: Pinned) -> None:
    pinned.repin(pinned.scores().assign(z=lambda table: table["z"].map(str)))


def outdate_a_cache_pin(pinned: Pinned) -> None:
    pinned.config["cache_runs"]["x1"]["verified_by"] = "derived/cache_check/run-11"


def change_a_cache_manifest(pinned: Pinned) -> None:
    path = pinned.workspace.root / pinned.config["cache_runs"]["rev"]["run"] / "manifest.parquet"
    pd.read_parquet(path).iloc[::-1].to_parquet(path, index=False)


CHANGES: dict[str, tuple[Callable[[Pinned], None], str]] = {
    "no pin": (
        lambda p: p.config.pop("zeroshot_scores"),
        "no zero-shot scores are pinned: zeroshot_scores needs exactly run and sha256",
    ),
    "a pin with another key": (
        lambda p: p.config["zeroshot_scores"].update(verified_by=CHECK_RUN),
        "no zero-shot scores are pinned: zeroshot_scores needs exactly run and sha256",
    ),
    "a pin outside the workspace": (
        lambda p: p.config["zeroshot_scores"].update(run="../" + RUN),
        f"../{RUN} is not a folder inside the workspace",
    ),
    "a changed file": (
        lambda p: p.scores().iloc[::-1].to_parquet(p.run / "scores.parquet", index=False),
        f"the pinned zero-shot scores: {RUN}/scores.parquet does not match the pin",
    ),
    "a missing file": (
        lambda p: (p.run / "scores.parquet").unlink(),
        f"the pinned zero-shot scores: {RUN}/scores.parquet does not match the pin",
    ),
    "a run not done": (
        lambda p: p.edit(lambda info: info.update(status="failed")),
        f"{RUN} did not finish: its run.json does not say done",
    ),
    "a run of another step": (
        lambda p: p.edit(lambda info: info.update(step="build_caches")),
        f"{RUN} is not a run of scripts/score_zeroshot.py",
    ),
    "a run without run.json": (
        lambda p: (p.run / "run.json").unlink(),
        f"{RUN} is not a run of scripts/score_zeroshot.py",
    ),
    "a run.json that lists another file": (
        lambda p: p.edit(lambda info: info["outputs"].update({"scores.parquet": "0" * 64})),
        f"{RUN}: its run.json lists another scores.parquet than the pin",
    ),
    "other views": (
        lambda p: p.edit(lambda info: info["settings"]["zeroshot"].update(views=["x0", "rev"])),
        f"{RUN} scored the views ['x0', 'rev'], not ['x0', 'x1', 'rev']",
    ),
    "an outdated cache pin": (
        outdate_a_cache_pin,
        f"{RUN} was scored from other cache runs than cache_runs pins now (x1)",
    ),
    "a cache pin gone": (
        lambda p: p.config["cache_runs"].pop("rev"),
        f"{RUN} was scored from other cache runs than cache_runs pins now (rev)",
    ),
    "a recorded cache of another view": (
        lambda p: p.edit(
            lambda info: info["inputs"]["caches"].update(x2=info["inputs"]["caches"]["x0"])
        ),
        f"{RUN} was scored from other cache runs than cache_runs pins now (other views)",
    ),
    "a changed cache manifest": (
        change_a_cache_manifest,
        "the pinned cache run of rev: its manifest.parquet does not match",
    ),
    "a missing segment": (
        drop_a_segment,
        f"{RUN}: scores.parquet lacks 1 segments of the cache of x1 and holds 0 others",
    ),
    "an extra segment": (
        add_a_segment,
        f"{RUN}: scores.parquet lacks 0 segments of the cache of x0 and holds 1 others",
    ),
    "a segment twice": (repeat_a_segment, f"{RUN}: scores.parquet holds 1 segments of x1 twice"),
    "another session": (
        move_a_segment_to_another_session,
        f"{RUN}: 1 rows of x0 name another session than the cache",
    ),
    "a NaN": (plant(float("nan")), f"{RUN}: 1 scores of rev are not finite"),
    "an infinity": (plant(float("-inf")), f"{RUN}: 1 scores of rev are not finite"),
    "rows of x2": (score_x2_too, f"{RUN}: its scores.parquet holds rows of other views: ['x2']"),
    "another column": (
        rename_a_column,
        f"{RUN}: its scores.parquet does not have the columns of SCORE_COLUMNS",
    ),
    "z as text": (
        store_z_as_text,
        f"{RUN}: the z of its scores.parquet are not floating-point numbers",
    ),
}


@pytest.mark.parametrize("change", list(CHANGES))
def test_load_zeroshot_scores_refuses_scores_that_do_not_fit_the_pins(
    pinned: Pinned, change: str
) -> None:
    plant_change, message = CHANGES[change]
    plant_change(pinned)
    with pytest.raises(StepError, match=f"^{re.escape(message)}"):
        load_zeroshot_scores(pinned.workspace, pinned.config)


@pytest.mark.parametrize("change", list(CHANGES))
def test_no_refusal_names_a_score(pinned: Pinned, change: str) -> None:
    pinned.repin(pinned.scores().assign(z=12.345678))  # every score the same planted value
    CHANGES[change][0](pinned)
    with pytest.raises(StepError) as raised:
        load_zeroshot_scores(pinned.workspace, pinned.config)
    for spelling in ("12.34", (12.345678).hex()):
        assert spelling not in str(raised.value), spelling
