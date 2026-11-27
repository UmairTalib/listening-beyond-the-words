"""The analysis of the pilot's test scores, on synthetic data only.

Fold AUROCs and Spearman correlations are checked against counts over all pairs and against
scikit-learn and SciPy, the paired bootstrap against the repeated rows that its counts stand
for, and the checks, the rule and its framing line at their exact bounds (pilot protocol,
sections "Feasibility checks" and "Decision rule"; docs/pilot_analysis.md). Fake
participant IDs start at 9001.
"""

import copy
import itertools
import math
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import stats
from sklearn.metrics import roc_auc_score

from listening_beyond_words import evaluation, training
from listening_beyond_words.config import load_arm
from listening_beyond_words.evaluation import (
    AGREEMENT,
    AMBER,
    ARMS,
    AUROC,
    DECISIONS,
    DIFFERENCE,
    DIFFERENCES,
    FRAMING_EXTRA_CLIPS,
    FRAMING_NO_B1,
    FRAMING_NOT_READ,
    FRAMING_STAYS,
    GREEN_BEATS,
    GREEN_ROBUST,
    RED,
    SPEARMAN,
    VERSIONS,
    VIEWS,
    EvaluateSettings,
    FoldRun,
    Number,
    Point,
    draw_counts,
    draws_sha256,
    f2_reading,
    f3_reading,
    fold_auroc,
    fold_runs,
    fold_spearman,
    framing,
    interval,
    kept_run,
    outcome,
    person_scores,
    pinned_version,
    point_values,
    printed,
    resampled_means,
    run_order,
    standard_error,
    weighted_aurocs,
)
from listening_beyond_words.loss import STEP_MEASURES
from listening_beyond_words.partitions import make_partitions
from listening_beyond_words.runinfo import StepError, item_seed

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "pilot"
CONFIG = yaml.safe_load((CONFIGS / "evaluate.yaml").read_text(encoding="utf-8"))
SETTINGS = EvaluateSettings.from_config(CONFIG)
FIRST = "271cce6d31e6c30fd10353649b246893882aaea2"
TINY = Fraction(1, 10**12)
SHIFTS = {"zeroshot": 0.3, "b0": 0.8, "b1": 0.9, "cpg": 1.0}


# Synthetic participants and scores ------------------------------------------------------------


def design(
    positives: int = 15, negatives: int = 35, seed: int = 0
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Labels, PHQ-8 totals and 2 rounds of 5 folds, stratified by label and gender."""
    rng = np.random.default_rng(seed)
    ids = pd.Index(np.arange(9001, 9001 + positives + negatives), name="session")
    totals = pd.Series(
        np.r_[rng.integers(10, 25, positives), rng.integers(0, 10, negatives)], index=ids
    )
    labels = (totals >= 10).astype(np.int64)
    gender = pd.Series(np.where(np.arange(len(ids)) % 2, "female", "male"), index=ids)
    partitions = make_partitions(pd.DataFrame({"label": labels, "gender": gender}))
    return labels, totals, partitions


def scores_table(
    sessions: list[int], labels: pd.Series, rng: np.random.Generator, shift: float
) -> pd.DataFrame:
    """A run's scores: 1 to 5 segments per participant and view, z on a grid of 1/4 (ties)."""
    rows = []
    for session in sessions:
        for index in range(int(rng.integers(1, 6))):
            for view in VIEWS:
                z = float(np.round(4 * (rng.normal() + shift * labels[session])) / 4)
                rows.append((f"{session}_{index:03d}", int(session), view, z))
    return pd.DataFrame(rows, columns=["segment", "session", "view", "z"])


def fold_members(partitions: pd.DataFrame) -> dict[tuple[int, int], list[int]]:
    return {
        (int(number), int(fold)): sorted(int(s) for s in part["session"])
        for (number, fold), part in partitions.groupby(["round", "fold"])
    }


def arm_scores(
    labels: pd.Series, partitions: pd.DataFrame, seed: int = 1, arms: tuple[str, ...] = ARMS
) -> dict[tuple[str, str, int, int], pd.Series]:
    """Every arm's exact scores by fold-run; the zero-shot arm's one model scores every fold."""
    rng = np.random.default_rng(seed)
    whole = {}
    if "zeroshot" in arms:
        table = scores_table(sorted(labels.index), labels, rng, SHIFTS["zeroshot"])
        whole = {view: person_scores(table, view) for view in VIEWS}
    found = {}
    for (number, fold), sessions in fold_members(partitions).items():
        for arm in arms:
            if arm == "zeroshot":
                for view in VIEWS:
                    found[(arm, view, number, fold)] = whole[view].reindex(sessions)
                continue
            table = scores_table(sessions, labels, rng, SHIFTS[arm])
            for view in VIEWS:
                found[(arm, view, number, fold)] = person_scores(table, view)
    return found


def every_run(seed: int = 0) -> tuple[pd.Series, pd.Series, pd.DataFrame, list[FoldRun]]:
    labels, totals, partitions = design(seed=seed)
    runs = fold_runs(labels, totals, partitions, arm_scores(labels, partitions, seed=seed + 1))
    return labels, totals, partitions, runs


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", stats.ConstantInputWarning)
        return float(stats.spearmanr(first, second).statistic)


def library_values(run: FoldRun, drawn: pd.Series) -> dict[Number, float]:
    """Each number of a fold-run from scikit-learn and SciPy on the repeated rows of a draw."""
    sessions = list(run.labels.index)
    times = [int(drawn[session]) for session in sessions]

    def rows(values: pd.Series) -> np.ndarray:
        return np.repeat([float(value) for value in values.reindex(sessions)], times)

    labels = rows(run.labels)
    totals = rows(run.totals)
    found: dict[Number, float] = {}
    for (arm, view), scores in run.scores.items():
        found[Number(AUROC, (arm,), view)] = float(roc_auc_score(labels, rows(scores)))
        found[Number(SPEARMAN, (arm,), view)] = spearman(rows(scores), totals)
    for view in VIEWS:
        for first, second in DIFFERENCES:
            if (first, view) in run.scores and (second, view) in run.scores:
                found[Number(DIFFERENCE, (first, second), view)] = (
                    found[Number(AUROC, (first,), view)] - found[Number(AUROC, (second,), view)]
                )
        if ("b0", view) in run.scores and ("cpg", view) in run.scores:
            found[Number(AGREEMENT, ("b0", "cpg"), view)] = spearman(
                rows(run.scores[("b0", view)]), rows(run.scores[("cpg", view)])
            )
    return found


def ones(labels: pd.Series, rows: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        np.ones((rows, len(labels)), dtype=np.int64),
        columns=pd.Index(sorted(labels.index), name="session"),
    )


def same(found: float, expected: float | None) -> bool:
    """Equal within 1e-12; two values that are not defined (None or NaN) count as equal."""
    if expected is None or (isinstance(expected, float) and math.isnan(expected)):
        return math.isnan(found)
    return abs(found - float(expected)) <= 1e-12


# Settings -------------------------------------------------------------------------------------


def test_the_settings_of_evaluate_yaml_load_with_exact_decimal_bounds() -> None:
    settings = SETTINGS
    assert (settings.arms, settings.views, settings.differences) == (ARMS, VIEWS, DIFFERENCES)
    assert (settings.resamples, settings.seed, settings.levels) == (10000, 109, (0.8, 0.95))
    assert settings.f3_min_auroc == Fraction(3, 5)
    assert (settings.green_at_least, settings.red_at_most) == (Fraction(3, 100), Fraction(-4, 100))
    # The float 0.03 lies below 3/100: a Δ of exactly 3/100 must reach the written bound.
    assert Fraction(0.03) < settings.green_at_least
    assert settings.framing_view == "x0"
    assert settings.cpg_gain_at_least == settings.green_at_least
    assert (settings.b1_gain_at_least, settings.cpg_over_b1_below) == (
        Fraction(3, 100),
        Fraction(2, 100),
    )
    assert settings.first_runs_commit == FIRST
    assert len(settings.runs) == 30  # the kept training runs, pinned on 2026-11-27
    assert settings.version is not None
    assert settings.version.runs == 30
    assert settings.delta == ("cpg", "b0")
    assert settings.agreement == ("b0", "cpg")


def test_the_draws_have_a_stream_of_their_own() -> None:
    assert SETTINGS.stream == 'runinfo.item_seed(109, "evaluate", "bootstrap")'
    assert item_seed(109, "evaluate", "bootstrap") == 6394762830805724214
    own = SETTINGS.rng().integers(0, 2**32, size=5)
    recorded = np.random.default_rng(6394762830805724214).integers(0, 2**32, size=5)
    np.testing.assert_array_equal(own, recorded)
    plain = np.random.default_rng(109).integers(0, 2**32, size=5)
    word_check = np.random.default_rng(item_seed(105, "wordcheck", "bootstrap"))
    assert not np.array_equal(own, plain)
    assert not np.array_equal(own, word_check.integers(0, 2**32, size=5))


def changed(config: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    """A copy of a config with one value changed, or removed where the value is ``...``."""
    found = copy.deepcopy(config)
    *parents, last = path.split(".")
    block = found
    for key in parents:
        block = block[key]
    if value is ...:
        del block[last]
    else:
        block[last] = value
    return found


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("arms", ..., "missing keys"),
        ("extra", 1, "unknown keys"),
        ("arms", ["zeroshot", "b0", "cpg", "b1"], "arms must be"),
        ("views", ["x0", "x1"], "views must be"),
        ("differences", [["b0", "cpg"], ["b1", "b0"], ["cpg", "b1"]], "differences must be"),
        ("bootstrap.resamples", 1, "at least 2"),
        ("bootstrap.resamples", True, "whole number"),
        ("bootstrap.seed", -1, "at least 0"),
        ("bootstrap.levels", [1.0], "between 0 and 1"),
        ("bootstrap.levels", [0.8, 0.8], "different floats"),
        ("bootstrap.levels", [], "different floats"),
        ("bootstrap.other", 1, "unknown keys"),
        ("checks.f3_min_auroc", 1, "finite float"),
        ("checks.f3_min_auroc", 1.5, "from 0 to 1"),
        ("checks.f3_min_auroc", float("nan"), "finite float"),
        ("rule.green_at_least", "0.03", "finite float"),
        ("rule.red_at_most", 0.03, "below rule.green_at_least"),
        ("framing.view", "x1", "framing.view"),
        ("framing.view", "rev", "framing.view must be x0"),
        ("framing.cpg_gain_at_least", 0.02, "must equal rule.green_at_least"),
        ("framing.b1_gain_at_least", None, "finite float"),
        ("first_runs_commit", "271cce6", "full commit hash"),
        ("runs", [], "runs must be a mapping"),
        ("checks", 0.6, "must be a mapping"),
    ],
)
def test_the_settings_refuse_unknown_and_missing_keys_and_bad_values(
    path: str, value: Any, match: str
) -> None:
    with pytest.raises(StepError, match=match):
        EvaluateSettings.from_config(changed(CONFIG, path, value))


def pin_block(
    arm_rounds: frozenset[tuple[str, int]], skip: tuple[tuple[str, int, int], ...] = ()
) -> dict[str, Any]:
    """The runs block that pins every fold-run of these arms and rounds, but those skipped."""
    runs: dict[str, Any] = {}
    for arm, number in sorted(arm_rounds):
        for fold in range(1, 6):
            if (arm, number, fold) not in skip:
                runs.setdefault(arm, {}).setdefault(f"r{number}", {})[f"f{fold}"] = {
                    "run": f"runs/pilot/pilot-{arm}-r{number}-f{fold}-20261127-12000{fold}",
                    "scores_sha256": "c" * 64,
                    "record_sha256": "d" * 64,
                }
    return runs


@pytest.mark.parametrize("version", VERSIONS)
def test_each_version_of_the_pinned_runs_is_recognised(version: evaluation.Version) -> None:
    settings = EvaluateSettings.from_config(changed(CONFIG, "runs", pin_block(version.arm_rounds)))
    assert settings.version == version
    assert len(settings.runs) == version.runs
    pin = settings.runs[("cpg", 1, 1)]
    assert pin.run == "runs/pilot/pilot-cpg-r1-f1-20261127-120001"
    assert (pin.scores_sha256, pin.record_sha256) == ("c" * 64, "d" * 64)


def test_a_pinned_set_that_is_no_version_is_refused() -> None:
    full = VERSIONS[0].arm_rounds
    for runs in (
        pin_block(full, skip=(("b1", 2, 3),)),  # a fold-run missing
        pin_block(frozenset({("b0", 1), ("b0", 2), ("cpg", 1)})),  # CPG's round 2 cut first
        pin_block(frozenset({("b0", 1), ("b1", 1), ("cpg", 1)})),
    ):
        with pytest.raises(StepError, match="none of the pilot's versions"):
            EvaluateSettings.from_config(changed(CONFIG, "runs", runs))
    with pytest.raises(StepError, match="none of the pilot's versions"):
        pinned_version([("b0", 1, fold) for fold in range(1, 6)] * 2)
    assert pinned_version([]) is None


def test_the_versions_are_the_protocols_four_and_no_other_set_of_whole_arm_rounds() -> None:
    # Pilot protocol, section "If time runs short": cut B1's round 2, then B1, then round 2
    # of B0 and CPG. Written out here, not read from VERSIONS.
    assert [(version.name, version.runs, set(version.arm_rounds)) for version in VERSIONS] == [
        ("all 30 runs", 30, {("b0", 1), ("b0", 2), ("b1", 1), ("b1", 2), ("cpg", 1), ("cpg", 2)}),
        ("no B1 round 2", 25, {("b0", 1), ("b0", 2), ("b1", 1), ("cpg", 1), ("cpg", 2)}),
        ("no B1", 20, {("b0", 1), ("b0", 2), ("cpg", 1), ("cpg", 2)}),
        ("B0 and CPG round 1 only", 10, {("b0", 1), ("cpg", 1)}),
    ]
    arm_rounds = [(arm, number) for arm in ("b0", "b1", "cpg") for number in (1, 2)]
    accepted = []
    for size in range(1, len(arm_rounds) + 1):
        for chosen in itertools.combinations(arm_rounds, size):
            runs = [(arm, number, fold) for arm, number in chosen for fold in range(1, 6)]
            try:
                version = pinned_version(runs)
            except StepError:
                continue
            assert version is not None
            assert version.runs == len(runs)
            accepted.append(version.name)
    # Of the 63 sets of whole arm-rounds that are not empty, exactly the four versions.
    assert accepted == ["B0 and CPG round 1 only", "no B1", "no B1 round 2", "all 30 runs"]


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda runs: runs.update(zeroshot=runs.pop("b0")), "not a trained arm"),
        (lambda runs: runs["b0"].update(r3=runs["b0"].pop("r1")), "the rounds are r1 and r2"),
        (lambda runs: runs["b0"]["r1"].update(f6=runs["b0"]["r1"].pop("f1")), "the folds are"),
        (lambda runs: runs["b0"]["r1"]["f1"].update(run="runs/pilot/x"), "run must be the folder"),
        (
            lambda runs: runs["b0"]["r1"]["f1"].update(
                run="runs/pilot/pilot-b0-r1-f2-20261127-120001"
            ),
            "run must be the folder",
        ),
        (lambda runs: runs["b0"]["r1"]["f1"].update(scores_sha256="C" * 64), "SHA-256"),
        (lambda runs: runs["b0"]["r1"]["f1"].update(extra=1), "unknown keys"),
        (lambda runs: runs["b0"]["r1"]["f1"].pop("record_sha256"), "missing keys"),
        (lambda runs: runs["b0"].update(r1={}), "must map f<fold>"),
        (lambda runs: runs.update(b0={}), "must map r<round> to its folds"),
    ],
)
def test_a_pin_must_name_its_own_fold_run_and_two_hashes(edit: Any, match: str) -> None:
    runs = pin_block(VERSIONS[0].arm_rounds)
    edit(runs)
    with pytest.raises(StepError, match=match):
        EvaluateSettings.from_config(changed(CONFIG, "runs", runs))


# Kept runs ------------------------------------------------------------------------------------

PINS = {
    "labels": {"run": "derived/labels/run-20261112T103229Z", "sha256": "a" * 64},
    "partitions": {"run": "derived/partitions/run-20261112T111941Z", "sha256": "b" * 64},
}


def train_record(
    arm: str = "b0",
    round_number: int = 1,
    fold: int = 1,
    commit: str = FIRST,
    tags: tuple[str, ...] = (),
    release_tags: tuple[str, ...] = (),
) -> dict[str, Any]:
    """The fields of a done training run's run.json that the analysis reads."""
    return {
        "run_id": f"pilot-{arm}-r{round_number}-f{fold}-20261126-101112",
        "step": "train",
        "arm": arm,
        "round": round_number,
        "fold": fold,
        "timing": False,
        "status": "done",
        "allow_dirty": False,
        "git": {
            "commit": commit,
            "dirty": False,
            "describe": "v0.1.0" if tags else commit[:7],
            "tags": list(tags),
            "release_tags": list(release_tags),
        },
        "inputs": copy.deepcopy(PINS),
        "results": {"rules": {"f1": {"passes": True}}},
    }


def keep(record: dict[str, Any], arm: str = "b0", round_number: int = 1, fold: int = 1) -> str:
    run = f"runs/pilot/pilot-{arm}-r{round_number}-f{fold}-20261126-101112"
    return kept_run(record, run, arm, round_number, fold, PINS, FIRST)


def test_the_first_runs_count_at_their_commit_and_every_run_at_its_clones_newest_tag() -> None:
    assert keep(train_record()) == "the first runs' commit 271cce6"
    assert keep(train_record(fold=4), fold=4) == "the first runs' commit 271cce6"
    assert keep(train_record("cpg"), "cpg") == "the first runs' commit 271cce6"
    released = train_record("b1", 2, 3, "e" * 40, ("v0.1.0",), ("v0.1.0",))
    assert keep(released, "b1", 2, 3) == "the newest release tag of its clone, v0.1.0"
    # Releases are ordered by number: v0.1.10 is newer than v0.1.2.
    tenth = train_record("b1", 2, 3, "e" * 40, ("v0.1.10",), ("v0.1.10", "v0.1.2"))
    assert keep(tenth, "b1", 2, 3) == "the newest release tag of its clone, v0.1.10"
    # A rerun of a first run after the freeze comes from the newest tag.
    rerun = train_record("cpg", 1, 1, "e" * 40, ("v0.1.1",), ("v0.1.0", "v0.1.1"))
    assert keep(rerun, "cpg") == "the newest release tag of its clone, v0.1.1"


def edited(record: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    return changed(record, path, value)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("step", "check_views", "not a run of scripts/train.py"),
        ("timing", True, "timing run"),
        ("timing", ..., "timing run"),
        ("run_id", "pilot-b0-r1-f2-20261126-101112", "names another run"),
        ("arm", "cpg", "not a run of b0, round 1, fold 1"),
        ("round", 2, "not a run of b0, round 1, fold 1"),
        ("fold", 2, "not a run of b0, round 1, fold 1"),
        ("status", "running", "did not finish"),
        ("status", "failed", "did not finish"),
        ("allow_dirty", True, "--allow-dirty"),
        ("allow_dirty", ..., "--allow-dirty"),
        ("git.dirty", True, "uncommitted changes"),
        ("git", ..., "uncommitted changes"),
        ("results.rules.f1.passes", False, "fails check F1"),
        ("results.rules.f1.passes", ..., "does not say that check F1 passed"),
        ("results.rules.f1.passes", "yes", "does not say that check F1 passed"),
        ("results.rules", None, "does not say that check F1 passed"),
        ("inputs.labels.sha256", "f" * 64, "other labels"),
        ("inputs.partitions", ..., "other partitions"),
        ("git.commit", "e" * 40, "neither the first runs' commit"),
    ],
)
def test_a_run_that_breaks_a_rule_of_the_kept_runs_is_refused(
    path: str, value: Any, match: str
) -> None:
    with pytest.raises(StepError, match=match):
        keep(edited(train_record(), path, value))


def test_a_run_is_refused_while_data_yaml_pins_no_labels_or_partitions() -> None:
    for name in ("labels", "partitions"):
        record = train_record()
        del record["inputs"][name]
        pins = {key: value for key, value in PINS.items() if key != name}
        with pytest.raises(StepError, match=f"data.yaml pins no {name}"):
            kept_run(record, "runs/pilot/pilot-b0-r1-f1-20261126-101112", "b0", 1, 1, pins, FIRST)


def test_only_b0s_round_1_and_cpgs_first_fold_run_count_at_the_first_commit() -> None:
    for arm, number, fold in (("b0", 2, 1), ("cpg", 1, 2), ("b1", 1, 1), ("cpg", 2, 1)):
        with pytest.raises(StepError, match="neither the first runs' commit"):
            keep(train_record(arm, number, fold), arm, number, fold)


def test_a_run_from_an_older_release_or_off_any_tag_is_refused() -> None:
    older = train_record("b1", 2, 3, "e" * 40, ("v0.1.0",), ("v0.1.0", "v0.1.2"))
    with pytest.raises(StepError, match=r"newest release tag of its clone, v0\.1\.2"):
        keep(older, "b1", 2, 3)
    untagged = train_record("b1", 2, 3, "e" * 40, (), ("v0.1.0",))
    with pytest.raises(StepError, match="newest release tag"):
        keep(untagged, "b1", 2, 3)
    first_untagged = train_record("b0", 1, 2, "e" * 40, (), ("v0.1.0",))
    with pytest.raises(StepError, match="nor is it one of the first runs"):
        keep(first_untagged, "b0", 1, 2)
    unordered = train_record("b1", 2, 3, "e" * 40, ("v0.1.x",), ("v0.1.x",))
    with pytest.raises(StepError, match="cannot order it"):
        keep(unordered, "b1", 2, 3)
    with pytest.raises(StepError, match="not a JSON object"):
        kept_run([], "runs/pilot/x", "b0", 1, 1, PINS, FIRST)


def test_of_two_runs_of_a_fold_run_the_one_that_finished_first_is_the_earlier() -> None:
    # Pilot analysis, section "Runs and pins": the earlier one counts.
    run = "runs/pilot/pilot-b1-r2-f4-20261127-000001"
    first = run_order({"finished_utc": "2026-11-27T11:00:00+00:00"}, run)
    started_first = "runs/pilot/pilot-b1-r2-f4-20261126-000001"
    later = run_order({"finished_utc": "2026-11-28T09:00:00+00:00"}, started_first)
    assert first == ("2026-11-27T11:00:00+00:00", run)
    assert first < later  # it finished first, though the other started first
    same_second = "runs/pilot/pilot-b1-r2-f4-20261127-000002"
    assert first < run_order({"finished_utc": "2026-11-27T11:00:00+00:00"}, same_second)
    for record in ({}, {"finished_utc": None}, {"finished_utc": ""}, {"finished_utc": 3}):
        with pytest.raises(StepError, match=r"^runs/pilot/x: its run.json does not say when"):
            run_order(record, "runs/pilot/x")


# Person scores, fold AUROCs and Spearman correlations ---------------------------------------


def test_a_persons_score_is_the_exact_mean_of_all_their_segments() -> None:
    values = [0.1] * 16 + [3.4]
    table = pd.DataFrame(
        {
            "segment": [f"9001_{index:03d}" for index in range(17)] + ["9002_000"],
            "session": [9001] * 17 + [9002],
            "view": ["x0"] * 18,
            "z": [*values, 0.5],
        }
    )
    scores = person_scores(table, "x0")
    assert list(scores.index) == [9001, 9002]
    assert scores[9001] == sum(map(Fraction, values)) / 17  # every segment, not the first 16
    assert scores[9001] != sum(map(Fraction, values[:16])) / 16
    assert scores[9002] == Fraction(1, 2)
    assert all(isinstance(value, Fraction) for value in scores)
    reordered = person_scores(table.iloc[::-1], "x0")
    assert reordered.equals(scores)  # the order of the rows does not matter
    with pytest.raises(StepError, match="no segment of rev"):
        person_scores(table, "rev")


def test_person_scores_refuse_a_repeated_segment_a_score_that_is_not_finite_and_a_column() -> None:
    table = pd.DataFrame(
        {"segment": ["9001_000", "9001_001"], "session": [9001, 9001], "view": "x0", "z": 0.5}
    )
    with pytest.raises(StepError, match="1 segments of x0 twice"):
        person_scores(table.assign(segment="9001_000"), "x0")
    for bad in (math.nan, math.inf):
        with pytest.raises(StepError, match="1 scores of x0 are not finite"):
            person_scores(table.assign(z=[0.5, bad]), "x0")
    with pytest.raises(StepError, match="lack the columns"):
        person_scores(table.drop(columns="z"), "x0")


def test_the_fold_auroc_counts_every_pair_with_ties_one_half() -> None:
    sessions = pd.Index(range(9001, 9009))
    labels = pd.Series([1, 1, 1, 0, 0, 0, 0, 0], index=sessions)
    scores = pd.Series(
        [Fraction(v) for v in ("1/2", "1/4", "-1", "1/4", "1/4", "-3", "1/2", "7")],
        index=sessions,
        dtype=object,
    )
    wins = sum(
        Fraction(1) if p > n else Fraction(1, 2) if p == n else Fraction(0)
        for p in scores[labels == 1]
        for n in scores[labels == 0]
    )
    assert fold_auroc(scores, labels) == wins / 15
    assert fold_auroc(scores, labels) == Fraction(13, 30)
    as_floats = [float(value) for value in scores]
    assert float(fold_auroc(scores, labels)) == pytest.approx(roc_auc_score(labels, as_floats))
    shuffled = scores.iloc[[3, 0, 7, 1, 6, 2, 5, 4]]
    assert fold_auroc(shuffled, labels) == Fraction(13, 30)
    with pytest.raises(ValueError, match="a positive and a negative"):
        fold_auroc(scores, labels.clip(upper=0))
    with pytest.raises(ValueError, match="same participants"):
        fold_auroc(scores.iloc[1:], labels)
    with pytest.raises(ValueError, match="neither a fraction"):
        fold_auroc(scores.replace({Fraction(7): math.nan}), labels)
    with pytest.raises(ValueError, match="a label must be 0 or 1"):
        fold_auroc(scores, labels.replace({0: 2}))


def test_the_spearman_correlation_equals_scipys_with_ties() -> None:
    rng = np.random.default_rng(3)
    for _ in range(50):
        size = int(rng.integers(3, 30))
        first = rng.integers(0, 5, size)
        second = rng.integers(0, 4, size)
        index = pd.Index(range(9001, 9001 + size))
        found = fold_spearman(
            pd.Series([Fraction(int(v), 3) for v in first], index=index, dtype=object),
            pd.Series(second, index=index),
        )
        expected = spearman(first, second)
        assert same(
            math.nan if found is None else found, None if math.isnan(expected) else expected
        )
    flat = pd.Series([Fraction(1, 3)] * 4, index=range(9001, 9005), dtype=object)
    assert fold_spearman(flat, pd.Series([1, 2, 3, 4], index=range(9001, 9005))) is None
    assert (
        fold_spearman(pd.Series([1, 2, 3], index=[1, 2, 3]), pd.Series([3, 2, 1], index=[1, 2, 3]))
        == -1.0
    )


def test_a_monotone_rescale_of_one_fold_models_scores_changes_no_fold_auroc() -> None:
    labels, totals, partitions = design()
    scores = arm_scores(labels, partitions)
    points = point_values(fold_runs(labels, totals, partitions, scores), SETTINGS)
    rescaled = dict(scores)
    for view in VIEWS:  # one fold model, CPG's of round 2, fold 3: z to 4z - 3, then s to s³ + s
        rescaled[("cpg", view, 2, 3)] = scores[("cpg", view, 2, 3)].map(
            lambda s: (4 * s - 3) ** 3 + (4 * s - 3)
        )
    again = point_values(fold_runs(labels, totals, partitions, rescaled), SETTINGS)
    assert list(again) == list(points)
    for number, point in points.items():
        assert dict(again[number].values) == dict(point.values), number
    assert dict(rescaled[("cpg", "x0", 2, 3)]) != dict(scores[("cpg", "x0", 2, 3)])


def test_a_constant_fold_has_an_auroc_of_one_half_and_no_spearman() -> None:
    labels, totals, partitions = design()
    scores = arm_scores(labels, partitions)
    sessions = list(scores[("b0", "x0", 1, 2)].index)
    scores[("b0", "x0", 1, 2)] = pd.Series([Fraction(5, 4)] * len(sessions), index=sessions)
    runs = fold_runs(labels, totals, partitions, scores)
    points = point_values(runs, SETTINGS)
    auroc = points[Number(AUROC, ("b0",), "x0")]
    assert auroc.values[(1, 2)] == Fraction(1, 2)
    for number in (Number(SPEARMAN, ("b0",), "x0"), Number(AGREEMENT, ("b0", "cpg"), "x0")):
        assert points[number].values[(1, 2)] is None
        assert points[number].mean() is None
        assert points[number].undefined() == ((1, 2),)
        assert points[number].mean({2}) is not None  # round 2's mean needs no fold of round 1
        assert points[number].undefined({2}) == ()
    counts, _ = draw_counts(labels, partitions, np.random.default_rng(4), 50)
    resampled = resampled_means(counts, runs, SETTINGS)
    column = resampled[Number(AUROC, ("b0",), "x0")].fold_runs.index((1, 2))
    assert (resampled[Number(AUROC, ("b0",), "x0")].values[:, column] == 0.5).all()
    flat = resampled[Number(SPEARMAN, ("b0",), "x0")]
    assert np.isnan(flat.values[:, column]).all()
    assert np.isnan(flat.means()).all()
    assert flat.undefined() == ((1, 2),)
    assert np.isfinite(flat.means({2})).all()
    with pytest.raises(ValueError, match="not finite"):
        interval(flat.means(), 0.8)
    with pytest.raises(ValueError, match="not finite"):
        standard_error(flat.means())


# Fold-runs and point estimates ------------------------------------------------------------------


def test_fold_runs_refuse_a_label_that_disagrees_with_its_total() -> None:
    labels, totals, partitions = design()
    wrong = totals.copy()
    wrong[9003] = 9  # a positive participant with a total below 10
    with pytest.raises(StepError, match=r"^1 labels disagree with their PHQ-8 totals") as refusal:
        fold_runs(labels, wrong, partitions, {})
    assert "9003" not in str(refusal.value)
    with pytest.raises(StepError, match="same participants"):
        fold_runs(labels, totals.iloc[1:], partitions, {})


def test_fold_runs_refuse_scores_of_other_participants_and_of_fold_runs_that_do_not_exist() -> None:
    labels, totals, partitions = design()
    scores = arm_scores(labels, partitions, arms=("b0",))
    runs = fold_runs(labels, totals, partitions, scores)
    assert [run.key for run in runs] == [(r, f) for r in (1, 2) for f in range(1, 6)]
    assert all(list(run.labels.index) == sorted(run.labels.index) for run in runs)
    first = dict(scores)
    first[("b0", "x0", 1, 1)] = first[("b0", "x0", 1, 1)].iloc[1:]
    with pytest.raises(StepError, match=r"round 1, fold 1, name \d+ participants, not the \d+"):
        fold_runs(labels, totals, partitions, first)
    other = dict(scores)
    other[("b0", "x0", 1, 1)] = scores[("b0", "x0", 1, 2)]
    with pytest.raises(StepError, match="not the"):
        fold_runs(labels, totals, partitions, other)
    with pytest.raises(StepError, match="1 sets of scores name a fold-run"):
        fold_runs(
            labels, totals, partitions, {**scores, ("b0", "x0", 3, 1): scores[("b0", "x0", 1, 1)]}
        )
    floats = dict(scores)
    floats[("b0", "x0", 1, 1)] = scores[("b0", "x0", 1, 1)].map(lambda _: math.nan)
    with pytest.raises(StepError, match="neither a fraction nor a finite float"):
        fold_runs(labels, totals, partitions, floats)
    one_class = partitions.copy()
    one_class.loc[(one_class["round"] == 1) & (one_class["session"] <= 9015), "fold"] = 1
    with pytest.raises(StepError, match="lacks a class"):
        fold_runs(labels, totals, one_class, {})
    with pytest.raises(StepError, match="round 1 of the partitions does not hold each"):
        fold_runs(labels, totals, partitions.iloc[1:], {})


def test_each_fold_run_holds_its_participants_labels_and_phq8_totals() -> None:
    # The Spearman correlations read the totals, which vary within each class, not the labels.
    labels, totals, partitions = design()
    runs = fold_runs(labels, totals, partitions, {})
    assert len(runs) == 10
    for run in runs:
        chosen = partitions[
            (partitions["round"] == run.round_number) & (partitions["fold"] == run.fold)
        ]
        members = sorted(int(session) for session in chosen["session"])
        assert list(run.labels.index) == list(run.totals.index) == members
        assert run.labels.tolist() == labels.reindex(members).tolist()
        assert run.totals.tolist() == totals.reindex(members).tolist()
        assert run.totals.nunique() > 2  # not the labels


def test_delta_is_the_mean_of_the_per_fold_differences() -> None:
    labels, _, partitions, runs = every_run()
    points = point_values(runs, SETTINGS)
    for view in VIEWS:
        delta = points[Number(DIFFERENCE, ("cpg", "b0"), view)]
        cpg, b0 = points[Number(AUROC, ("cpg",), view)], points[Number(AUROC, ("b0",), view)]
        assert len(delta.values) == 10
        for key, value in delta.values.items():
            assert value == cpg.values[key] - b0.values[key]
        assert delta.mean() == sum(delta.values.values(), Fraction(0)) / 10
        assert delta.mean() == cpg.mean() - b0.mean()  # the same 10 fold-runs
        assert delta.mean({1}) == sum(delta.values[(1, f)] for f in range(1, 6)) / 5
    counts, _ = draw_counts(labels, partitions, np.random.default_rng(5), 200)
    resampled = resampled_means(counts, runs, SETTINGS)
    for view in VIEWS:
        delta = resampled[Number(DIFFERENCE, ("cpg", "b0"), view)]
        cpg = resampled[Number(AUROC, ("cpg",), view)]
        b0 = resampled[Number(AUROC, ("b0",), view)]
        np.testing.assert_array_equal(delta.values, cpg.values - b0.values)
        np.testing.assert_allclose(delta.means(), cpg.means() - b0.means(), atol=1e-15)
    assert list(resampled) == list(points)
    number = Number(DIFFERENCE, ("cpg", "b0"), "x0")
    for found in (points[number].mean, points[number].undefined, resampled[number].means):
        with pytest.raises(ValueError, match="no fold-run in these rounds"):
            found({3})


def test_the_numbers_come_in_the_order_of_the_tables() -> None:
    _, _, _, runs = every_run()
    names = [(number.name, number.view) for number in point_values(runs, SETTINGS)]
    assert names[:2] == [("auroc_zeroshot", "x0"), ("auroc_zeroshot", "rev")]
    assert names[8] == ("spearman_zeroshot", "x0")
    assert names[16:22] == [
        ("cpg_minus_b0", "x0"),
        ("cpg_minus_b0", "rev"),
        ("b1_minus_b0", "x0"),
        ("b1_minus_b0", "rev"),
        ("cpg_minus_b1", "x0"),
        ("cpg_minus_b1", "rev"),
    ]
    assert names[22:] == [("agreement_b0_cpg", "x0"), ("agreement_b0_cpg", "rev")]


def test_without_b1_or_cpg_only_the_numbers_of_the_arms_read_appear() -> None:
    labels, totals, partitions = design()
    scores = arm_scores(labels, partitions, arms=("zeroshot", "b0"))
    runs = fold_runs(labels, totals, partitions, scores)
    points = point_values(runs, SETTINGS)
    assert {number.kind for number in points} == {AUROC, SPEARMAN}
    assert {number.arms for number in points} == {("zeroshot",), ("b0",)}
    counts, _ = draw_counts(labels, partitions, np.random.default_rng(6), 20)
    assert list(resampled_means(counts, runs, SETTINGS)) == list(points)
    # B1's round 1 alone (no B1 round 2): its differences use round 1's five fold-runs.
    round_1 = {
        key: value
        for key, value in arm_scores(labels, partitions, seed=7).items()
        if key[0] != "b1" or key[2] == 1
    }
    points = point_values(fold_runs(labels, totals, partitions, round_1), SETTINGS)
    b1_gain = points[Number(DIFFERENCE, ("b1", "b0"), "x0")]
    assert sorted(b1_gain.values) == [(1, fold) for fold in range(1, 6)]
    assert len(points[Number(DIFFERENCE, ("cpg", "b0"), "x0")].values) == 10
    with pytest.raises(ValueError, match="no fold-run in these rounds"):
        b1_gain.mean({2})


def test_point_values_refuse_an_arm_that_the_analysis_does_not_read() -> None:
    labels, totals, partitions = design()
    scores = arm_scores(labels, partitions, arms=("b0",))
    scores = {("b2", *key[1:]): value for key, value in scores.items()}
    runs = fold_runs(labels, totals, partitions, scores)
    with pytest.raises(ValueError, match="b2 on x0"):
        point_values(runs, SETTINGS)


def test_each_point_value_is_checked_against_scikit_learn_and_scipy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels, totals, partitions = design()
    runs = fold_runs(labels, totals, partitions, arm_scores(labels, partitions, arms=("b0",)))
    original_auroc = evaluation.roc_auc_score
    monkeypatch.setattr(evaluation, "roc_auc_score", lambda y, s: original_auroc(y, s) + 1e-11)
    with pytest.raises(StepError, match="exact AUROC and scikit-learn's differ"):
        point_values(runs, SETTINGS)
    monkeypatch.setattr(evaluation, "roc_auc_score", lambda y, s: original_auroc(y, s) + 1e-13)
    point_values(runs, SETTINGS)  # within 1e-12
    monkeypatch.setattr(evaluation, "roc_auc_score", original_auroc)

    class Undefined:
        statistic = math.nan

    monkeypatch.setattr(evaluation.stats, "spearmanr", lambda first, second: Undefined())
    with pytest.raises(StepError, match="only one of them is defined"):
        point_values(runs, SETTINGS)


# The paired bootstrap ---------------------------------------------------------------------------


def test_every_draw_keeps_the_class_sizes() -> None:
    labels, _, partitions = design()
    counts, _ = draw_counts(labels, partitions, np.random.default_rng(7), 300)
    assert counts.shape == (300, 50)
    assert list(counts.columns) == sorted(labels.index)  # keyed by session, in ascending order
    positives = [session for session in counts.columns if labels[session] == 1]
    negatives = [session for session in counts.columns if labels[session] == 0]
    assert (counts[positives].sum(axis=1) == 15).all()
    assert (counts[negatives].sum(axis=1) == 35).all()
    assert counts.to_numpy().dtype == np.int64


def planted_partitions() -> tuple[pd.Series, pd.DataFrame]:
    """30 participants, 6 positive; round 1, fold 1 holds one positive, every other fold more."""
    ids = list(range(9001, 9031))
    labels = pd.Series([1] * 6 + [0] * 24, index=pd.Index(ids, name="session"))
    positive_folds = {1: [1, 2, 3, 4, 5, 2], 2: [1, 2, 3, 4, 5, 1]}
    rows = [
        (session, number, positive_folds[number][index] if index < 6 else index % 5 + 1)
        for number in (1, 2)
        for index, session in enumerate(ids)
    ]
    return labels, pd.DataFrame(rows, columns=["session", "round", "fold"])


def test_a_fold_with_one_positive_forces_counted_redraws_and_fills_every_fold() -> None:
    labels, planted = planted_partitions()
    counts, redraws = draw_counts(labels, planted, np.random.default_rng(8), 2000)
    # A resample misses fold 1's one positive with a chance of (5/6)^6, about a third, and
    # round 2's two folds of one positive each as well.
    assert redraws > 1000
    members = fold_members(planted)
    assert len(members) == 10
    for sessions in members.values():
        drawn = counts[sessions]
        assert (drawn[[s for s in sessions if labels[s] == 1]].sum(axis=1) > 0).all()
        assert (drawn[[s for s in sessions if labels[s] == 0]].sum(axis=1) > 0).all()
    assert (counts.sum(axis=1) == 30).all()


def test_the_draws_take_the_positives_then_the_negatives_each_in_ascending_session_id() -> None:
    # Pilot analysis, section "Intervals", bullet "Draws", drawn again by hand: each class
    # in ascending session ID, the positives first, a resample that leaves a test fold of a
    # round without a class drawn again and counted.
    labels, _, partitions = design()
    counts, redraws = draw_counts(labels, partitions, np.random.default_rng(21), 20)
    fold_of = {(int(row.session), int(row.round)): int(row.fold) for row in partitions.itertuples()}
    positives = sorted(int(session) for session in labels.index if labels[session] == 1)
    negatives = sorted(int(session) for session in labels.index if labels[session] == 0)
    rng = np.random.default_rng(21)
    kept: list[list[int]] = []
    again = 0
    while len(kept) < 20:
        drawn = [
            [group[index] for index in rng.choice(len(group), size=len(group), replace=True)]
            for group in (positives, negatives)
        ]
        if all(
            {fold_of[(session, number)] for session in group} == set(range(1, 6))
            for group in drawn
            for number in (1, 2)
        ):
            kept.append(drawn[0] + drawn[1])
        else:
            again += 1
    assert again > 0  # the check covers the redraws too
    assert redraws == again
    for row, sessions in enumerate(kept):
        expected = [sessions.count(session) for session in counts.columns]
        assert counts.iloc[row].tolist() == expected, row


def test_the_same_seed_gives_the_same_draws_and_another_seed_others() -> None:
    labels, _, partitions = design()
    first, _ = draw_counts(labels, partitions, np.random.default_rng(item_seed(109, "e")), 100)
    again, _ = draw_counts(labels, partitions, np.random.default_rng(item_seed(109, "e")), 100)
    other, _ = draw_counts(labels, partitions, np.random.default_rng(item_seed(110, "e")), 100)
    assert draws_sha256(first) == draws_sha256(again)
    assert draws_sha256(first) != draws_sha256(other)
    assert len(draws_sha256(first)) == 64
    shuffled = labels.iloc[::-1]  # the order of the rows does not change the draws
    reordered, _ = draw_counts(
        shuffled, partitions.iloc[::-1], np.random.default_rng(item_seed(109, "e")), 100
    )
    assert draws_sha256(reordered) == draws_sha256(first)
    relabelled = first.set_axis([*first.columns[:-1], 9999], axis=1)
    assert draws_sha256(relabelled) != draws_sha256(first)  # the hash covers the session IDs


def test_draw_counts_refuse_inputs_that_no_draw_could_serve() -> None:
    labels, _, partitions = design()
    rng = np.random.default_rng(9)
    with pytest.raises(ValueError, match="at least one resample"):
        draw_counts(labels, partitions, rng, 0)
    with pytest.raises(ValueError, match="0 or 1"):
        draw_counts(labels.replace({1: 2}), partitions, rng, 5)
    with pytest.raises(ValueError, match="does not hold each participant once"):
        draw_counts(labels, partitions.iloc[1:], rng, 5)
    with pytest.raises(ValueError, match="the partitions hold no round"):
        draw_counts(labels, partitions.iloc[:0], rng, 5)
    one_class = partitions.copy()
    one_class.loc[(one_class["round"] == 2) & (one_class["session"] <= 9015), "fold"] = 1
    with pytest.raises(ValueError, match="both classes"):
        draw_counts(labels, one_class, rng, 5)


def test_with_every_count_one_the_resamples_give_every_exact_point_estimate() -> None:
    labels, _, _, runs = every_run(seed=10)
    points = point_values(runs, SETTINGS)
    resampled = resampled_means(ones(labels), runs, SETTINGS)
    assert list(resampled) == list(points)
    assert len(points) == 24  # 4 AUROCs, 4 Spearman, 3 differences and the agreement, by view
    for number, point in points.items():
        found = resampled[number]
        assert found.fold_runs == tuple(sorted(point.values))
        for column, key in enumerate(found.fold_runs):
            for row in range(2):
                assert same(found.values[row, column], point.values[key]), (number, key)
        assert same(found.means()[0], point.mean())
        assert same(found.means({1})[0], point.mean({1}))


def test_an_arm_against_itself_gives_delta_zero_on_every_draw() -> None:
    labels, totals, partitions = design(seed=11)
    scores = arm_scores(labels, partitions, seed=12)
    for (arm, view, number, fold), values in list(scores.items()):
        if arm == "b0":
            scores[("cpg", view, number, fold)] = values
    runs = fold_runs(labels, totals, partitions, scores)
    delta = point_values(runs, SETTINGS)[Number(DIFFERENCE, ("cpg", "b0"), "x0")]
    assert set(delta.values.values()) == {Fraction(0)}
    counts, _ = draw_counts(labels, partitions, np.random.default_rng(13), 500)
    resampled = resampled_means(counts, runs, SETTINGS)
    by_key = {run.key: run for run in runs}
    flat = 0
    for view in VIEWS:
        assert (resampled[Number(DIFFERENCE, ("cpg", "b0"), view)].values == 0.0).all()
        agreement = resampled[Number(AGREEMENT, ("b0", "cpg"), view)]
        defined = np.isfinite(agreement.values)
        assert (agreement.values[defined] == 1.0).all()  # two sides that rank alike give 1
        # Not defined only where every participant drawn into the fold has the same score.
        for row, column in zip(*np.nonzero(~defined), strict=True):
            run = by_key[agreement.fold_runs[column]]
            drawn = counts.loc[row, list(run.labels.index)]
            assert run.scores[("b0", view)][drawn[drawn > 0].index].nunique() == 1
            flat += 1
    assert flat < 50


def test_with_round_2s_partition_equal_to_round_1s_every_draw_gives_equal_fold_values() -> None:
    labels, totals, partitions = design(seed=14)
    same_rounds = partitions.copy()
    first = partitions.loc[partitions["round"] == 1].set_index("session")["fold"]
    same_rounds.loc[same_rounds["round"] == 2, "fold"] = (
        same_rounds.loc[same_rounds["round"] == 2, "session"].map(first).to_numpy()
    )
    scores = arm_scores(labels, same_rounds, seed=15)
    for (arm, view, number, fold), values in list(scores.items()):
        if number == 1:
            scores[(arm, view, 2, fold)] = values  # the same model scores both rounds
    runs = fold_runs(labels, totals, same_rounds, scores)
    counts, _ = draw_counts(labels, same_rounds, np.random.default_rng(16), 300)
    for number, found in resampled_means(counts, runs, SETTINGS).items():
        for fold in range(1, 6):
            one = found.values[:, found.fold_runs.index((1, fold))]
            two = found.values[:, found.fold_runs.index((2, fold))]
            np.testing.assert_array_equal(one, two, err_msg=str(number))
    # On different partitions the two rounds differ: the check above is not empty.
    labels, _, partitions, runs = every_run(seed=14)
    counts, _ = draw_counts(labels, partitions, np.random.default_rng(16), 300)
    b0 = resampled_means(counts, runs, SETTINGS)[Number(AUROC, ("b0",), "x0")]
    assert not np.array_equal(b0.values[:, 0], b0.values[:, 5])


def test_a_person_drawn_three_times_changes_exactly_the_two_fold_runs_that_hold_them() -> None:
    labels, _, _, runs = every_run(seed=17)
    person = 9004
    counts = ones(labels)
    counts.loc[1, person] = 3
    resampled = resampled_means(counts, runs, SETTINGS)
    holding = {run.key for run in runs if person in run.labels.index}
    assert len(holding) == 2
    assert {key[0] for key in holding} == {1, 2}
    changed_runs = set()
    for run in runs:
        expected = library_values(run, counts.loc[1])
        for number, found in resampled.items():
            if run.key not in found.fold_runs:
                continue
            column = found.fold_runs.index(run.key)
            before, after = found.values[0, column], found.values[1, column]
            if run.key in holding:
                assert same(after, expected[number]), (number, run.key)
                if not (before == after or (math.isnan(before) and math.isnan(after))):
                    changed_runs.add(run.key)
            else:
                assert before == after or (math.isnan(before) and math.isnan(after))
    assert changed_runs == holding


def test_the_resampled_values_equal_those_of_the_repeated_rows() -> None:
    labels, _, partitions, runs = every_run(seed=18)
    counts, _ = draw_counts(labels, partitions, np.random.default_rng(19), 40)
    resampled = resampled_means(counts, runs, SETTINGS)
    for row in range(0, 40, 8):
        for run in runs:
            expected = library_values(run, counts.loc[row])
            for number, found in resampled.items():
                column = found.fold_runs.index(run.key)
                assert same(found.values[row, column], expected[number]), (number, run.key, row)


def test_resampled_means_refuse_counts_that_are_not_whole_numbers_by_session() -> None:
    labels, _, _, runs = every_run()
    counts = ones(labels)
    with pytest.raises(ValueError, match="whole numbers"):
        resampled_means(counts.astype(float), runs, SETTINGS)
    with pytest.raises(ValueError, match="whole numbers"):
        resampled_means(counts.replace({1: -1}), runs, SETTINGS)
    with pytest.raises(ValueError, match="lack participants of round 1"):
        resampled_means(counts.drop(columns=9001), runs, SETTINGS)
    with pytest.raises(ValueError, match="one column per session ID"):
        resampled_means(counts.set_axis([str(c) for c in counts.columns], axis=1), runs, SETTINGS)
    with pytest.raises(ValueError, match="no positive or no negative"):
        resampled_means(counts * 0, runs, SETTINGS)


def test_weighted_aurocs_refuse_counts_that_do_not_fit_the_pair_wins() -> None:
    wins = evaluation.pair_wins(np.array([1.0, 2.0]), np.array([0.5, 1.5, 3.0]))
    assert weighted_aurocs(wins, np.ones((4, 2)), np.ones((4, 3))).tolist() == [3 / 6] * 4
    for positive, negative in (((4, 3), (4, 3)), ((4, 2), (4, 2)), ((4, 2), (3, 3)), ((2,), (3,))):
        with pytest.raises(ValueError, match="do not fit pair wins"):
            weighted_aurocs(wins, np.ones(positive), np.ones(negative))


# Intervals and standard errors --------------------------------------------------------------


def test_interval_ends_are_numpys_linear_percentiles() -> None:
    values = np.random.default_rng(20).normal(size=10_000)
    for level, percentiles in ((0.8, [10, 90]), (0.95, [2.5, 97.5])):
        # The ends sit at (1 - level) / 2 and (1 + level) / 2, computed in floating point as
        # the word check computes them, which is the 10th and 90th or the 2.5th and 97.5th
        # percentile up to the last bits.
        expected = np.quantile(values, [(1 - level) / 2, (1 + level) / 2], method="linear")
        assert interval(values, level) == (float(expected[0]), float(expected[1]))
        named = np.percentile(values, percentiles, method="linear")
        assert interval(values, level) == pytest.approx(tuple(named), abs=1e-12)
    # (1 - 0.8) / 2 is 0.09999999999999998 in floating point, so the end lies a hair below 1.
    assert interval(np.arange(11.0), 0.8) == pytest.approx((1.0, 9.0), abs=1e-12)
    assert interval(np.arange(5.0), 0.5) == (1.0, 3.0)
    for level in (0.0, 1.0):
        with pytest.raises(ValueError, match="between 0 and 1"):
            interval(values, level)
    with pytest.raises(ValueError, match="one value per resample"):
        interval(np.array([]), 0.8)


def test_the_standard_error_uses_ddof_1() -> None:
    values = np.array([0.1, 0.4, 0.2, 0.7])
    assert standard_error(values) == pytest.approx(float(np.std(values, ddof=1)), abs=0)
    assert standard_error(values) != pytest.approx(float(np.std(values)))
    assert standard_error(np.array([1.0, 3.0])) == pytest.approx(math.sqrt(2.0))
    with pytest.raises(ValueError, match="R > 1"):
        standard_error(np.array([0.5]))


# Checks F2 and F3 -------------------------------------------------------------------------------


def metric_line(step: int, epoch: int) -> dict[str, Any]:
    """A planted line of a CPG run's metrics.jsonl: 4 bags, one per group, none active."""
    line: dict[str, Any] = {
        "step": step,
        "epoch": epoch,
        "seconds": 10.0,
        "loss": 1.2,
        "ce_x0": 0.7,
        "ce_part": 1.0,
        "ce_x1": 0.6,
        "ce_x2": 0.3,
        "hinge": 0.2,
        "lambda_g": 1.0,
        "scale": 65_536.0,
        "next_scale": 65_536.0,
        "skipped": False,
        "grad_norm": 2.0,
        "clipped": False,
    }
    line |= dict.fromkeys(STEP_MEASURES, 0.0)
    line |= {f"bags_{group}": 1 for group in training.GROUPS}
    line |= {f"active_{group}": 0 for group in training.GROUPS}
    line |= {"sigma0": 1.0, "sigma1": 1.0, "sigma0_at_floor": 0, "sigma1_at_floor": 0}
    return line | {"z1_bar_abs_mean": 1.0, "z0_bar_abs_mean": 2.0}


def cpg_record(active: int, steps: int) -> dict[str, Any]:
    """A CPG run's record of check F2, from training.rule_measures as the train job writes it."""
    lines = []
    left = active
    for step in range(3 * steps):
        line = metric_line(step, step // steps + 1)
        for group in training.GROUPS:
            if line["epoch"] > 1 and left:
                line[f"active_{group}"] = 1
                left -= 1
        lines.append(line)
    settings = load_arm(CONFIGS / "cpg.yaml").shared
    measures = training.rule_measures(lines, settings, reads_f2=True)
    return {"results": {"rules": measures, "rule_readings": {"reads_f2": True}}}


def test_f2_pools_the_summed_counts_not_the_shares() -> None:
    records = {"runs/pilot/run-a": cpg_record(40, 5), "runs/pilot/run-b": cpg_record(80, 20)}
    reading = f2_reading(records)
    assert reading["runs"]["runs/pilot/run-a"] == {
        "active": 40,
        "bags": 40,
        "share": 1.0,
        "passes": False,
    }
    assert reading["runs"]["runs/pilot/run-b"]["share"] == 0.5
    assert reading["pooled"] == {"active": 120, "bags": 200, "share": 0.6}
    assert reading["pooled"]["share"] != (1.0 + 0.5) / 2
    assert reading["passes"] is False
    bounds = f2_reading(
        {"runs/pilot/run-c": cpg_record(32, 20), "runs/pilot/run-d": cpg_record(128, 20)}
    )
    assert [run["share"] for run in bounds["runs"].values()] == [0.2, 0.8]
    assert bounds["passes"] is True  # 0.20 and 0.80 are inside, both included


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("results.rule_readings.reads_f2", False, "does not say that check F2 reads it"),
        ("results.rules.alive.f2_passes", ..., "does not say whether check F2 passed"),
        ("results.rules.alive.f2_passes", None, "does not say whether check F2 passed"),
        ("results.rules.alive.bags", 0, "lacks the bags"),
        ("results.rules.alive.active", 201, "lacks the bags"),
        ("results.rules.alive.active", True, "lacks the bags"),
        ("results.rules.alive", None, "lacks the bags"),
        ("results.rules.alive.share", 0.4, "is not its counts'"),
    ],
)
def test_f2_refuses_a_record_that_does_not_say_never_reading_it_as_a_fail(
    path: str, value: Any, match: str
) -> None:
    record = changed(cpg_record(100, 25), path, value)
    with pytest.raises(StepError, match=match):
        f2_reading({"runs/pilot/run-a": record})
    with pytest.raises(StepError, match="there is none"):
        f2_reading({})


def b0_point(mean: Fraction) -> Point:
    """B0's AUROC on x0 in 10 fold-runs, with the given mean and other values per round."""
    values = {(1, fold): mean + Fraction(fold - 3, 100) for fold in range(1, 6)}
    values |= {(2, fold): mean - Fraction(fold - 3, 50) for fold in range(1, 6)}
    return Point(values)


def test_f3_passes_at_exactly_three_fifths_and_fails_just_below() -> None:
    reading = f3_reading(b0_point(Fraction(3, 5)), SETTINGS)
    assert reading == {
        "value": Fraction(3, 5),
        "rounds": {1: Fraction(3, 5), 2: Fraction(3, 5)},
        "bound": Fraction(3, 5),
        "passes": True,
    }
    assert f3_reading(b0_point(Fraction(3, 5) - TINY), SETTINGS)["passes"] is False
    round_1 = Point({(1, fold): Fraction(7, 10) for fold in range(1, 6)})
    assert f3_reading(round_1, SETTINGS)["rounds"] == {1: Fraction(7, 10)}
    with pytest.raises(TypeError, match="must be exact"):
        f3_reading(Point({(1, 1): 0.6}), SETTINGS)


# The decision rule and its framing line ---------------------------------------------------------


@pytest.mark.parametrize(
    ("x0", "rev", "expected"),
    [
        (Fraction(3, 100), Fraction(-1), GREEN_BEATS),
        (Fraction(3, 100) - TINY, Fraction(-1), AMBER),
        (Fraction(3, 100) - TINY, Fraction(3, 100), GREEN_ROBUST),
        (Fraction(-4, 100), Fraction(1), RED),
        (Fraction(-4, 100) + TINY, Fraction(1), GREEN_ROBUST),
        (Fraction(-4, 100) + TINY, Fraction(3, 100) - TINY, AMBER),
        (Fraction(0), Fraction(3, 100), GREEN_ROBUST),
        (Fraction(0), Fraction(3, 100) - TINY, AMBER),
        (Fraction(1, 2), Fraction(-1, 2), GREEN_BEATS),
        (Fraction(-1, 2), Fraction(1, 2), RED),
        (Fraction(3, 100), Fraction(3, 100), GREEN_BEATS),
    ],
)
def test_the_rule_at_its_bounds(x0: Fraction, rev: Fraction, expected: str) -> None:
    found = outcome(x0, rev, SETTINGS)
    assert found == (expected, DECISIONS[expected])


def test_the_rule_reads_exact_values_only() -> None:
    with pytest.raises(TypeError, match="Δ on x0 must be exact"):
        outcome(0.03, Fraction(0), SETTINGS)
    with pytest.raises(TypeError, match="Δ on rev must be exact"):
        outcome(Fraction(0), np.float64(0.03), SETTINGS)
    assert outcome(0, 0, SETTINGS).name == AMBER  # whole numbers are exact


def test_the_outcomes_and_decisions_are_the_protocols_word_for_word() -> None:
    protocol = (ROOT / "docs" / "pilot_protocol.md").read_text(encoding="utf-8")
    table = protocol.split("## Decision rule", 1)[1].split("## ", 1)[0]
    rows = [line for line in table.splitlines() if line.startswith("| ") and "---" not in line]
    found = {
        cells[0]: cells[2]
        for cells in ([cell.strip() for cell in row.strip("|").split("|")] for row in rows[1:])
    }
    assert dict(DECISIONS) == found
    assert list(DECISIONS) == [GREEN_BEATS, GREEN_ROBUST, AMBER, RED]


@pytest.mark.parametrize(
    ("b1_gain", "cpg_over_b1", "expected"),
    [
        (Fraction(3, 100), Fraction(2, 100) - TINY, FRAMING_EXTRA_CLIPS),
        (Fraction(3, 100) - TINY, Fraction(0), FRAMING_STAYS),
        (Fraction(1, 10), Fraction(2, 100), FRAMING_STAYS),
        (Fraction(1, 10), Fraction(2, 100) - TINY, FRAMING_EXTRA_CLIPS),
        (Fraction(1, 10), Fraction(-1, 10), FRAMING_EXTRA_CLIPS),
    ],
)
def test_the_framing_line_at_its_bounds_in_green_beats_b0(
    b1_gain: Fraction, cpg_over_b1: Fraction, expected: str
) -> None:
    name = outcome(Fraction(3, 100), Fraction(0), SETTINGS).name  # CPG − B0 exactly at +0.03
    assert framing(name, b1_gain, cpg_over_b1, SETTINGS) == expected


def test_the_framing_line_is_not_read_outside_green_beats_b0() -> None:
    holds = (Fraction(1, 10), Fraction(0))  # B1's condition holds
    assert framing(GREEN_BEATS, *holds, SETTINGS) == FRAMING_EXTRA_CLIPS
    for x0, rev in (
        (Fraction(3, 100) - TINY, Fraction(3, 100)),  # robust: CPG − B0 just below +0.03
        (Fraction(0), Fraction(0)),  # Amber
        (Fraction(-4, 100), Fraction(1)),  # Red
    ):
        name = outcome(x0, rev, SETTINGS).name
        assert name != GREEN_BEATS
        assert framing(name, *holds, SETTINGS) == FRAMING_NOT_READ
        assert framing(name, None, None, SETTINGS) == FRAMING_NOT_READ
    assert framing(GREEN_BEATS, None, None, SETTINGS) == FRAMING_NO_B1
    assert FRAMING_NO_B1.endswith("not available (cut: no B1)")
    with pytest.raises(TypeError, match="must be exact"):
        framing(GREEN_BEATS, 0.1, Fraction(0), SETTINGS)


def test_a_value_is_printed_with_4_decimals_and_unrounded_where_it_rounds_to_a_bound() -> None:
    bounds = (SETTINGS.green_at_least, SETTINGS.red_at_most)
    assert printed(Fraction(312, 10000), bounds) == "0.0312"
    assert printed(Fraction(3, 100), bounds) == "0.0300 (unrounded 0.03 = 3/100)"
    assert printed(Fraction(299999, 10**7), bounds) == (
        "0.0300 (unrounded 0.0299999 = 299999/10000000)"
    )
    assert printed(Fraction(-4, 100), bounds) == "-0.0400 (unrounded -0.04 = -1/25)"
    assert printed(Fraction(1, 3)) == "0.3333"
    # Rounded on the exact value, half to even: the float of 0.03005 lies above it.
    assert f"{0.03005:.4f}" == "0.0301"
    assert printed(Fraction(601, 20000), bounds) == "0.0300 (unrounded 0.03005 = 601/20000)"
    assert printed(Fraction(3, 5) - TINY, [SETTINGS.f3_min_auroc]) == (
        "0.6000 (unrounded 0.599999999999 = 599999999999/1000000000000)"
    )
    with pytest.raises(TypeError, match="must be exact"):
        printed(0.03)
