"""The gain-term summary (diagnostics.gain_term_table), on synthetic training logs only.

Each run's record comes from training.rule_measures on synthetic metrics lines, as the train
job writes it (and as tests/evaluate_fakes.py builds them), with values chosen so that every
mean, minimum, maximum and pooled share can be counted by hand (docs/pilot_diagnostics.md,
section "Gain-term summary"). The runs differ in their values, their active bags and their
steps per epoch, so that the pooled share is not the mean of the shares.
"""

import copy
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from listening_beyond_words import training
from listening_beyond_words.config import load_arm
from listening_beyond_words.diagnostics import (
    COLUMNS,
    MEASURES,
    NO_BAG,
    NO_VALUE,
    NOT_APPLICABLE,
    gain_term_table,
)
from listening_beyond_words.evaluation import RunPin, kept_run, pinned_record
from listening_beyond_words.loss import STEP_MEASURES
from listening_beyond_words.reports import write_table
from listening_beyond_words.runinfo import StepError

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "pilot"
FIRST = "271cce6d31e6c30fd10353649b246893882aaea2"  # the first runs' commit (evaluate.yaml)
# CPG's three runs: run k has 4, 4 and 8 steps per epoch; run 1 skips the first step of epoch 3.
CPG_STEPS = (4, 4, 8)
SKIPPED = (1, 3, 0)  # (run, epoch, step of the epoch)
B0_MEASURES = {"ce_x0", "x0_measured_mean", "x0_measured_spread", "ce_push"}


def metric_lines(arm: str, k: int, steps: int, wrong_bags: int = 1) -> list[dict[str, Any]]:
    """Run k's metrics.jsonl: 3 epochs of ``steps`` steps, every value a dyadic fraction.

    Each step has one bag in each class and words group (``wrong_bags`` in each words-wrong
    group). Active in each step: the positive words-right bag; the negative words-wrong bag
    on even steps of the epoch; the negative words-right bag on the epoch's first k steps.
    """
    gain = arm != "b0"
    lines: list[dict[str, Any]] = []
    for epoch in (1, 2, 3):
        for index in range(steps):
            skipped = (k, epoch, index) == SKIPPED and arm == "cpg"
            z0 = 1.0 + 0.25 * k
            line: dict[str, Any] = {
                "step": (epoch - 1) * steps + index,
                "epoch": epoch,
                "seconds": 30.0,
                "lr": 1.0e-4,
                "adapter_lr": 2.0e-5,
                "loss": 1.0 + 0.25 * k,
                "ce_x0": 0.5 + 0.125 * k,
                "ce_part": 0.75 + 0.125 * k,
            }
            if gain:
                lambda_g = 1.0 if arm == "cpg" else 0.0
                line |= {"ce_x1": 0.375, "ce_x2": 0.25, "hinge": 0.0625 * (k + 1)}
                line |= {"lambda_g": lambda_g}
            line |= {
                "scale": 65_536.0,
                "next_scale": 32_768.0 if skipped else 65_536.0,
                "skipped": skipped,
                "grad_norm": None if skipped else 2.0,
                "clipped": False,
                "calls": 14 if gain else 4,
            }
            if gain:
                line |= dict.fromkeys(STEP_MEASURES, 0.0)
                line |= {
                    "bags_positive_words_right": 1,
                    "bags_positive_words_wrong": wrong_bags,
                    "bags_negative_words_right": 1,
                    "bags_negative_words_wrong": wrong_bags,
                    "active_positive_words_right": 1,
                    "active_positive_words_wrong": 0,
                    "active_negative_words_right": int(index < k),
                    "active_negative_words_wrong": int(index % 2 == 0 and wrong_bags > 0),
                    "mu0": 0.5 + 0.125 * k,
                    "sigma0": 1.0 + 0.5 * k,
                    "sigma0_at_floor": int(k == 2 and index == 0),
                    "mu1": 0.25,
                    "sigma1": 0.5,
                    "sigma1_at_floor": 0,
                    "z0_bar_mean": z0,
                    "z0_bar_mean_square": z0 * z0 + 0.25,
                    "z1_bar_mean": 0.5,
                    "z1_bar_mean_square": 0.3125,
                    "z0_bar_abs_mean": z0,
                    "z1_bar_abs_mean": 0.5,
                    "z0_over_z1": z0 / 0.5,
                    "ce_push": 0.25 + 0.0625 * k,
                    "gain_push": lambda_g * 0.125 * (k + 1),
                    "gain_offset": lambda_g * -0.0625 * k,
                }
            else:
                line |= {
                    "z0_bar_mean": z0,
                    "z0_bar_mean_square": z0 * z0 + 0.25,
                    "z0_bar_abs_mean": z0,
                    "ce_push": 0.25 + 0.0625 * k,
                }
            lines.append(line)
    return lines


def rules(arm: str, k: int, steps: int = 4, wrong_bags: int = 1) -> dict[str, Any]:
    """results.rules of a run, as the train job's stage 8 writes it."""
    settings = load_arm(CONFIGS / f"{arm}.yaml").shared
    return training.rule_measures(metric_lines(arm, k, steps, wrong_bags), settings, None)


def records() -> dict[tuple[str, int, int], dict[str, Any]]:
    """CPG's three runs, B1's and B0's two, each record holding only results.rules."""
    found: dict[tuple[str, int, int], dict[str, Any]] = {}
    for k, steps in enumerate(CPG_STEPS):
        found[("cpg", 1, k + 1)] = {"results": {"rules": rules("cpg", k, steps)}}
    for arm in ("b1", "b0"):
        for k in (0, 1):
            found[(arm, 2, k + 1)] = {"results": {"rules": rules(arm, k)}}
    return found


def row(table: pd.DataFrame, arm: str, epoch: int, measure: str) -> dict[str, Any]:
    chosen = table[
        (table["arm"] == arm) & (table["epoch"] == epoch) & (table["measure"] == measure)
    ]
    assert len(chosen) == 1
    return chosen.iloc[0].to_dict()


def summary_of(values: list[Fraction]) -> tuple[int, float, float, float]:
    """The hand count of a row: runs, mean, minimum and maximum, from exact fractions."""
    return (
        len(values),
        float(sum(values) / len(values)),
        float(min(values)),
        float(max(values)),
    )


def test_each_mean_minimum_maximum_and_pooled_share_equals_a_hand_count() -> None:
    table = gain_term_table(records())
    assert list(table.columns) == list(COLUMNS)
    third = Fraction(1, 3)
    # Per epoch, CPG's runs 0, 1 and 2 (4, 4 and 8 steps): the active bags are the positive
    # words-right bag of every step, the negative words-wrong bag of every other step and
    # the negative words-right bag of the first k steps, out of 4 bags per step.
    shares = [Fraction(4 + 0 + 2, 16), Fraction(4 + 1 + 2, 16), Fraction(8 + 2 + 4, 32)]
    expected = {
        "loss": [Fraction(1), Fraction(5, 4), Fraction(3, 2)],
        "ce_x0": [Fraction(1, 2), Fraction(5, 8), Fraction(3, 4)],
        "ce_x1": [Fraction(3, 8)] * 3,
        "ce_x2": [Fraction(1, 4)] * 3,
        "ce_part": [Fraction(3, 4), Fraction(7, 8), Fraction(1)],
        "hinge": [Fraction(1, 16), Fraction(2, 16), Fraction(3, 16)],
        "active_share_all": shares,
        "active_share_positive": [Fraction(1, 2)] * 3,
        "active_share_negative": [Fraction(2, 8), Fraction(3, 8), Fraction(6, 16)],
        "active_share_words_right": [Fraction(4, 8), Fraction(5, 8), Fraction(10, 16)],
        "active_share_words_wrong": [Fraction(2, 8), Fraction(2, 8), Fraction(4, 16)],
        "clamp_share": [Fraction(1, 2)] * 3,
        "sigma0_at_floor_share": [Fraction(0), Fraction(0), Fraction(1, 8)],
        "sigma1_at_floor_share": [Fraction(0)] * 3,
        "x0_measured_mean": [Fraction(1), Fraction(5, 4), Fraction(3, 2)],
        "x0_mu": [Fraction(1, 2), Fraction(5, 8), Fraction(3, 4)],
        "x0_mean_minus_mu": [Fraction(1, 2), Fraction(5, 8), Fraction(3, 4)],
        "x0_measured_spread": [Fraction(1, 2)] * 3,
        "x0_sigma": [Fraction(1), Fraction(3, 2), Fraction(2)],
        "x1_measured_mean": [Fraction(1, 2)] * 3,
        "x1_mu": [Fraction(1, 4)] * 3,
        "x1_mean_minus_mu": [Fraction(1, 4)] * 3,
        "x1_measured_spread": [Fraction(1, 4)] * 3,
        "x1_sigma": [Fraction(1, 2)] * 3,
        "x1_spread_over_sigma": [Fraction(1, 2)] * 3,
        "z0_over_z1": [Fraction(2), Fraction(5, 2), Fraction(3)],
        "ce_push": [Fraction(1, 4), Fraction(5, 16), Fraction(6, 16)],
        "gain_push": [Fraction(1, 8), Fraction(2, 8), Fraction(3, 8)],
        "gain_offset": [Fraction(0), Fraction(-1, 16), Fraction(-2, 16)],
    }
    # Every measure of CPG is counted here, but the ratio of x0 and the skipped steps below.
    assert set(expected) | {"x0_spread_over_sigma", "skipped"} == {m.name for m in MEASURES}
    for epoch in (1, 2, 3):
        for measure, values in expected.items():
            found = row(table, "cpg", epoch, measure)
            hand = summary_of(values)
            assert (found["runs"], found["mean"], found["minimum"], found["maximum"]) == hand
    # The ratio spread / σ of x0: 1/2, 1/3 and 1/4, whose mean is 13/36.
    ratio = row(table, "cpg", 2, "x0_spread_over_sigma")
    assert ratio["mean"] == pytest.approx(13 / 36, rel=1e-15)
    assert (ratio["minimum"], ratio["maximum"]) == (0.25, 0.5)
    # The pooled share: the summed active bags over the summed bags, not the mean of shares.
    for epoch in (1, 2, 3):
        pooled = row(table, "cpg", epoch, "active_share_all")
        assert pooled["pooled_share"] == float(Fraction(6 + 7 + 14, 16 + 16 + 32)) == 0.421875
        assert pooled["mean"] == float(sum(shares) / 3) != pooled["pooled_share"]
    # Skipped steps: only run 1's first step of epoch 3.
    assert summary_of([Fraction(0)] * 3) == tuple(
        row(table, "cpg", 2, "skipped")[key] for key in ("runs", "mean", "minimum", "maximum")
    )
    skipped = row(table, "cpg", 3, "skipped")
    assert (skipped["runs"], skipped["mean"], skipped["maximum"]) == (3, float(third), 1.0)
    # B1 (runs 0 and 1, 4 steps): the same hand counts for its own measures.
    b1 = row(table, "b1", 1, "active_share_all")
    assert (b1["runs"], b1["mean"], b1["pooled_share"]) == (2, (0.375 + 0.4375) / 2, 13 / 32)
    assert row(table, "b1", 1, "hinge")["maximum"] == 0.125
    # B0: CE(x0), the x0 bag means and the CE push of runs 0 and 1.
    assert row(table, "b0", 3, "ce_x0")["mean"] == (0.5 + 0.625) / 2
    assert row(table, "b0", 3, "x0_measured_spread")["maximum"] == 0.5
    assert row(table, "b0", 3, "ce_push")["minimum"] == 0.25
    # Only the active share over all bags carries a pooled share, and only for B1 and CPG.
    others = table[table["measure"] != "active_share_all"]
    assert set(others["pooled_share"]) == {NOT_APPLICABLE}
    assert set(table.loc[table["arm"] == "b0", "pooled_share"]) == {NOT_APPLICABLE}


def test_b1_has_no_gain_push_and_b0_only_the_measures_it_logs() -> None:
    table = gain_term_table(records())
    every = [measure.name for measure in MEASURES]
    expected = {
        "cpg": every,
        "b1": [name for name in every if name not in {"gain_push", "gain_offset"}],
        "b0": [name for name in every if name in B0_MEASURES],
    }
    for arm, names in expected.items():
        for epoch in (1, 2, 3):
            chosen = table[(table["arm"] == arm) & (table["epoch"] == epoch)]
            assert chosen["measure"].tolist() == names
    assert table["arm"].tolist() == sorted(table["arm"], key=["b0", "b1", "cpg"].index)
    assert len(table) == 3 * (len(every) + len(every) - 2 + len(B0_MEASURES))
    # B1's records do hold a gain push, 0 by construction; it is left out.
    b1 = records()[("b1", 2, 1)]["results"]["rules"]["epochs"]["1"]
    assert b1["gain_push"] == 0.0


def test_a_run_without_a_value_is_left_out_of_its_row_and_a_row_without_any_says_so() -> None:
    found = records()
    # CPG's run 3 has no words-wrong bag at all, so its words-wrong share is None.
    found[("cpg", 1, 3)] = {"results": {"rules": rules("cpg", 2, 8, wrong_bags=0)}}
    table = gain_term_table(found)
    words_wrong = row(table, "cpg", 2, "active_share_words_wrong")
    assert (words_wrong["runs"], words_wrong["mean"]) == (2, 0.25)
    assert row(table, "cpg", 2, "loss")["runs"] == 3
    # With no words-wrong bag in any run, the row has no number.
    none = {
        key: {"results": {"rules": rules(key[0], key[2] - 1, 4, wrong_bags=0)}}
        for key in (("b1", 2, 1), ("b1", 2, 2))
    }
    table = gain_term_table(none)
    empty = row(table, "b1", 1, "active_share_words_wrong")
    assert empty["runs"] == 0
    assert (empty["mean"], empty["minimum"], empty["maximum"]) == (NO_VALUE,) * 3
    assert row(table, "b1", 1, "clamp_share")["mean"] == 0.0
    # A pooled share over no bag says so too.
    no_bags = copy.deepcopy(none)
    for record in no_bags.values():
        for epoch in record["results"]["rules"]["epochs"].values():
            epoch["bags"] = epoch["active_bags"] = 0
    assert row(gain_term_table(no_bags), "b1", 2, "active_share_all")["pooled_share"] == NO_BAG


def test_only_the_epoch_measures_are_read_in_any_order_of_the_records() -> None:
    found = records()
    full = {
        key: {
            "run_id": "anything",
            "results": {"rules": record["results"]["rules"], "sampler": {"steps": 12}},
            "outputs": {"scores.parquet": "not read"},
        }
        for key, record in found.items()
    }
    table = gain_term_table(found)
    assert gain_term_table(full).equals(table)
    assert gain_term_table(dict(reversed(list(found.items())))).equals(table)
    # Nothing outside results.rules.epochs is read: the other rules may go.
    bare = {
        key: {"results": {"rules": {"epochs": record["results"]["rules"]["epochs"]}}}
        for key, record in found.items()
    }
    assert gain_term_table(bare).equals(table)


def changed(path: tuple[str, ...], value: Any) -> dict[tuple[str, int, int], dict[str, Any]]:
    """The records with one value of CPG's first run replaced, or removed for Ellipsis."""
    found = records()
    node: Any = found[("cpg", 1, 1)]
    for key in path[:-1]:
        node = node[key]
    if value is ...:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return found


EPOCH = ("results", "rules", "epochs", "2")


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("results", "rules", "epochs"),
            ...,
            "cpg, round 1, fold 1: its record holds no per-epoch",
        ),
        (("results", "rules", "epochs"), {}, "its record holds no per-epoch measures"),
        (("results",), None, "its record holds no per-epoch measures"),
        ((*EPOCH[:-1], "0"), {}, "must be keyed by the epoch, from 1"),
        ((*EPOCH[:-1], "x"), {}, "must be keyed by the epoch, from 1"),
        ((*EPOCH[:-1], "4"), [], "must be keyed by the epoch, from 1"),
        ((*EPOCH, "hinge"), ..., "cpg, round 1, fold 1, epoch 2: the record holds no hinge"),
        ((*EPOCH, "x1", "sigma"), ..., "epoch 2: the record holds no x1.sigma"),
        ((*EPOCH, "active_share"), None, "epoch 2: the record holds no active_share.all"),
        ((*EPOCH, "gain_push"), math.nan, "epoch 2: gain_push is not a finite number"),
        ((*EPOCH, "loss"), math.inf, "epoch 2: loss is not a finite number"),
        ((*EPOCH, "skipped"), True, "epoch 2: skipped is not a finite number"),
        ((*EPOCH, "clamp_share"), "0.5", "epoch 2: clamp_share is not a finite number"),
        ((*EPOCH, "x0", "mu"), "high", "epoch 2: x0.mu is not a finite number"),
        ((*EPOCH, "bags"), 2.0, "epoch 2: bags is not a count of bags"),
        ((*EPOCH, "active_bags"), -1, "epoch 2: active_bags is not a count of bags"),
        ((*EPOCH, "active_bags"), 1000, "epoch 2: more active bags than bags"),
    ],
)
def test_a_record_that_lacks_a_measure_or_holds_no_number_is_refused(
    path: tuple[str, ...], value: Any, message: str
) -> None:
    with pytest.raises(StepError, match=message.replace("(", r"\(").replace(".", r"\.")):
        gain_term_table(changed(path, value))


def test_no_record_or_an_arm_that_did_not_train_is_refused() -> None:
    with pytest.raises(StepError, match="no kept run to summarise"):
        gain_term_table({})
    found = records()
    found[("zeroshot", 1, 1)] = found.pop(("b0", 2, 1))
    with pytest.raises(StepError, match=r"'zeroshot' is not a trained arm \(b0, b1, cpg\)"):
        gain_term_table(found)


def test_a_record_that_does_not_match_its_pin_is_refused_and_no_scores_file_is_opened(
    tmp_path: Path,
) -> None:
    # The records reach the summary through the check of the record alone
    # (evaluation.pinned_record) and evaluation.kept_run, as scripts/summarise_gain_terms.py
    # reads them: the scores file beside each record cannot even be opened.
    pins = {
        "labels": {"run": "derived/labels/run-1", "sha256": "a" * 64},
        "partitions": {"run": "derived/partitions/run-1", "sha256": "b" * 64},
    }
    kept: dict[tuple[str, int, int], dict[str, Any]] = {}
    for fold in (1, 2):
        run = f"runs/pilot/pilot-b0-r1-f{fold}-20261126-10111{fold}"
        record = {
            "run_id": Path(run).name,
            "step": "train",
            "arm": "b0",
            "round": 1,
            "fold": fold,
            "timing": False,
            "status": "done",
            "allow_dirty": False,
            "git": {"commit": FIRST, "dirty": False, "tags": [], "release_tags": []},
            "inputs": copy.deepcopy(pins),
            "results": {"rules": rules("b0", fold - 1)},
        }
        folder = tmp_path / run
        folder.mkdir(parents=True)
        data = json.dumps(record).encode("utf-8")
        (folder / "run.json").write_bytes(data)
        (folder / "scores.parquet").write_bytes(b"never opened")
        (folder / "scores.parquet").chmod(0)
        pin = RunPin(run, "0" * 64, hashlib.sha256(data).hexdigest())
        read = pinned_record(tmp_path, pin)
        assert kept_run(read, run, "b0", 1, fold, pins, FIRST) == "the first runs' commit 271cce6"
        kept[("b0", 1, fold)] = read
        with pytest.raises(StepError, match=r"its run\.json does not match its pin"):
            pinned_record(tmp_path, RunPin(run, "0" * 64, "f" * 64))
    table = gain_term_table(kept)
    assert set(table["measure"]) == B0_MEASURES
    for path in tmp_path.rglob("scores.parquet"):
        path.chmod(0o600)


def test_the_table_passes_the_writer_of_the_tables_of_results_at_full_precision(
    tmp_path: Path,
) -> None:
    table = gain_term_table(records())
    digest = write_table(table, tmp_path / "gain_terms.csv")
    assert digest == hashlib.sha256((tmp_path / "gain_terms.csv").read_bytes()).hexdigest()
    read = pd.read_csv(tmp_path / "gain_terms.csv", keep_default_na=False)
    assert list(read.columns) == list(COLUMNS)
    assert read["runs"].dtype == read["epoch"].dtype == np.int64
    for name in ("mean", "minimum", "maximum"):
        assert read[name].tolist() == table[name].tolist()  # every row of these has a value
