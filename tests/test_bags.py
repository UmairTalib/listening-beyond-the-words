"""The bag sampler (bags.py): whole bags, 2 + 2 per step, least used first, chosen by share.

Planted folds of fake participants (IDs from 9001), whose segments are named after their
session, check the rules of the pilot protocol, section "Training", bullets "Bags and
epochs" and "Who trains, and seeds": the bags, the steps, the end of an epoch, which
segments a participant offers and which bags train, that leftovers wait, that the order of
the steps does not follow earlier use, the training people, the streams and the start bags.
"""

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from listening_beyond_words.bags import (
    IMPLEMENTED,
    NEGATIVE,
    POSITIVE,
    STREAM,
    Bag,
    BagSampler,
    BagSettings,
    EpochPlan,
    stream_seed,
    training_people,
)
from listening_beyond_words.config import SamplerSettings, load_arm
from listening_beyond_words.partitions import make_partitions
from listening_beyond_words.runinfo import StepError, item_seed

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "pilot"
SAMPLER = SamplerSettings(
    positive_bags=2,
    negative_bags=2,
    epoch_end="a_class_runs_out",
    draw="least_trained_first",
    bag_choice="lowest_trained_share_first",
    step_order="shuffled",
)
BAGS = BagSettings(segments_per_bag=4, max_segments_per_epoch=16, repeat_segments=False)
FALLBACK = BagSettings(segments_per_bag=4, max_segments_per_epoch=8, repeat_segments=False)
SEED = stream_seed(101, 1)
# Segment counts of a planted fold: 28 positive bags and 61 negative ones per epoch, so 14
# steps, and 33 negative bags left over in each epoch, as in the pilot's folds, where the
# negative bags left over outnumber the trained ones. One participant of each class has 3.
POSITIVES = (3, 5, 8, 12, 16, 22, 30, 41, 9, 17)
NEGATIVES = (4, 7, 10, 13, 16, 19, 25, 33, 48, 60, 86, 3, 6, 11, 14, 20, 27, 35, 50, 103)


def fold(
    positives: Sequence[int], negatives: Sequence[int], first: int = 9001
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Segments and labels of planted participants: the counts of positive, then negative ones."""
    rows, labels = [], {}
    for offset, count in enumerate((*positives, *negatives)):
        session = first + offset
        labels[session] = POSITIVE if offset < len(positives) else NEGATIVE
        rows += [
            {"segment": f"{session}_{index:03d}", "session": session} for index in range(count)
        ]
    return pd.DataFrame(rows), labels


def sampler_of(
    positives: Sequence[int] = POSITIVES,
    negatives: Sequence[int] = NEGATIVES,
    bags: BagSettings = BAGS,
    seed: int = SEED,
) -> BagSampler:
    segments, labels = fold(positives, negatives)
    return BagSampler(segments, labels, bags, SAMPLER, seed)


def offered_bags(plan: EpochPlan) -> list[Bag]:
    return [*plan.bags(), *plan.unused]


def history(sampler: BagSampler, epochs: int) -> list[tuple[EpochPlan, Counter, Counter, Counter]]:
    """Each epoch's plan with the uses of the segments and the participants' trained and
    offered bags from the epochs before it."""
    uses: Counter = Counter()
    trained: Counter = Counter()
    offered: Counter = Counter()
    result = []
    for index in range(1, epochs + 1):
        plan = sampler.epoch(index)
        result.append((plan, uses.copy(), trained.copy(), offered.copy()))
        for bag in plan.bags():
            trained[bag.session] += 1
            uses.update(bag.segments)
        for bag in offered_bags(plan):
            offered[bag.session] += 1
    return result


def digest(sampler: BagSampler, epochs: int = 3) -> str:
    plans = [
        [[[bag.session, bag.label, list(bag.segments)] for bag in step] for step in plan.steps]
        + [[bag.session, bag.label, list(bag.segments)] for bag in plan.unused]
        for plan in (sampler.epoch(index) for index in range(1, epochs + 1))
    ]
    return hashlib.sha256(json.dumps(plans).encode()).hexdigest()


# Bags and steps --------------------------------------------------------------------------------


def test_every_bag_holds_4_different_segments_of_one_participant_with_its_label() -> None:
    segments, labels = fold(POSITIVES, NEGATIVES)
    sampler = BagSampler(segments, labels, BAGS, SAMPLER, SEED)
    owner = dict(zip(segments["segment"], segments["session"], strict=True))
    for index in (1, 2, 3):
        for bag in offered_bags(sampler.epoch(index)):
            assert len(bag.segments) == 4
            assert len(set(bag.segments)) == 4
            assert {owner[segment] for segment in bag.segments} == {bag.session}
            assert bag.label == labels[bag.session]


def test_no_segment_is_offered_twice_in_an_epoch() -> None:
    sampler = sampler_of()
    for index in (1, 2, 3):
        segments = [
            segment for bag in offered_bags(sampler.epoch(index)) for segment in bag.segments
        ]
        assert len(segments) == len(set(segments))


@pytest.mark.parametrize("bags", [BAGS, FALLBACK])
def test_a_participant_offers_its_whole_bags_of_at_most_16_segments_or_8_under_the_fallback(
    bags: BagSettings,
) -> None:
    segments, _ = fold(POSITIVES, NEGATIVES)
    counts = segments.groupby("session").size()
    sampler = sampler_of(bags=bags)
    most = bags.max_segments_per_epoch
    for index in (1, 2, 3):
        offered = Counter()
        for bag in offered_bags(sampler.epoch(index)):
            offered[bag.session] += len(bag.segments)
        for session, count in counts.items():
            assert offered[session] == 4 * (min(count, most) // 4) <= most


def test_a_participant_with_3_segments_gives_no_bag() -> None:
    sampler = sampler_of()
    few = {9001, 9001 + len(POSITIVES) + NEGATIVES.index(3)}  # one of each class
    for index in (1, 2, 3):
        assert not few & {bag.session for bag in offered_bags(sampler.epoch(index))}
    counts = sampler.counts()
    assert counts["positive_people_without_a_bag"] == counts["negative_people_without_a_bag"] == 1
    assert (counts["positive_people"], counts["negative_people"]) == (10, 20)


def test_every_step_takes_2_positive_and_2_negative_bags_in_turn() -> None:
    sampler = sampler_of()
    for index in (1, 2, 3):
        plan = sampler.epoch(index)
        assert plan.index == index
        for step in plan.steps:
            assert tuple(bag.label for bag in step) == (1, 0, 1, 0)


def test_two_bags_of_one_participant_may_share_a_step() -> None:
    # Two positive participants of 4 bags each fill 4 steps.
    shared = 0
    for seed in range(10):
        plan = sampler_of((16, 16), (16,) * 4, seed=seed).epoch(1)
        shared += sum(
            len({bag.session for bag in step if bag.label == POSITIVE}) == 1 for step in plan.steps
        )
    assert shared > 0


# The end of an epoch ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("positives", "negatives", "steps", "unused"),
    [
        ((4,) * 10, (4,) * 25, 5, (0, 15)),  # the positive bags run out, as in every pilot fold
        ((4,) * 11, (4,) * 25, 5, (1, 15)),  # one positive bag is left over: fewer than 2
        ((16,) * 7 + (4, 4), (8,) * 12 + (4,), 12, (6, 1)),  # 30 and 25 bags
        ((16,) * 8, (4,) * 9, 4, (24, 1)),  # the negative bags run out
        ((16, 16), (5, 6, 7, 3, 4), 2, (4, 0)),  # four negative bags: 2 steps
    ],
)
def test_the_epoch_ends_when_a_class_has_fewer_than_2_bags_left(
    positives: tuple[int, ...], negatives: tuple[int, ...], steps: int, unused: tuple[int, int]
) -> None:
    sampler = sampler_of(positives, negatives)
    positive_bags = sum(min(count, 16) // 4 for count in positives)
    negative_bags = sum(min(count, 16) // 4 for count in negatives)
    assert steps == min(positive_bags // 2, negative_bags // 2)
    assert sampler.steps_per_epoch == steps
    for index in (1, 2, 3):
        assert sampler.epoch(index).counts() == {
            "steps": steps,
            "positive_offered": positive_bags,
            "negative_offered": negative_bags,
            "positive_unused": unused[0],
            "negative_unused": unused[1],
        }
    assert sampler.total_steps(3) == 3 * steps


@pytest.mark.parametrize(("positives", "negatives"), [((4,), (4,) * 9), ((4,) * 6, (5, 3))])
def test_a_fold_that_cannot_fill_one_step_is_refused(
    positives: tuple[int, ...], negatives: tuple[int, ...]
) -> None:
    with pytest.raises(StepError, match="fewer than one step's 2 \\+ 2"):
        sampler_of(positives, negatives)


def test_every_epoch_has_the_same_steps_known_before_training() -> None:
    sampler = sampler_of()
    assert sampler.steps_per_epoch == 14
    assert sampler.total_steps(3) == 42
    assert sampler.total_steps(4) == 56
    assert [len(sampler.epoch(index).steps) for index in (1, 2, 3, 4)] == [14] * 4
    counts = sampler.counts()
    assert (counts["positive_bags_per_epoch"], counts["negative_bags_per_epoch"]) == (28, 61)
    assert counts["steps_per_epoch"] == 14


def test_the_fallback_of_8_segments_changes_the_steps_of_an_epoch() -> None:
    sampler = sampler_of(bags=FALLBACK)
    # 17 positive bags at 8 segments, against 28 at 16.
    assert sampler.steps_per_epoch == sum(min(count, 8) // 4 for count in POSITIVES) // 2 == 8


# Which segments and bags train -----------------------------------------------------------------


def test_no_offered_segment_was_used_more_often_than_one_left_out_of_the_same_participant() -> None:
    segments, _ = fold(POSITIVES, NEGATIVES)
    by_session = segments.groupby("session")["segment"].apply(list).to_dict()
    for seed in range(10):
        for plan, uses, _, _ in history(sampler_of(seed=seed), 4):
            offered = {segment for bag in offered_bags(plan) for segment in bag.segments}
            for own in by_session.values():
                inside = [uses[segment] for segment in own if segment in offered]
                outside = [uses[segment] for segment in own if segment not in offered]
                if inside and outside:
                    assert max(inside) <= min(outside)


def test_each_class_trains_the_bags_of_the_participants_with_the_lowest_trained_share() -> None:
    for seed in range(10):
        for plan, _, trained, offered in history(sampler_of(seed=seed), 4):

            def share(bag: Bag, trained: Counter = trained, offered: Counter = offered) -> float:
                return trained[bag.session] / offered[bag.session] if offered[bag.session] else 0.0

            for label in (POSITIVE, NEGATIVE):
                chosen = [share(bag) for bag in plan.bags() if bag.label == label]
                left = [share(bag) for bag in plan.unused if bag.label == label]
                if left:
                    assert max(chosen) <= min(left)


def test_ties_of_the_trained_share_are_drawn_at_random_not_by_session() -> None:
    # In epoch 1 every share is 0: which 28 of the 61 negative bags train is a draw, so the
    # trained participants change with the seed, and a one-bag participant trains for some
    # seeds and is left out for others, the lowest session IDs as much as the highest.
    segments, labels = fold(POSITIVES, NEGATIVES)
    counts = segments.groupby("session").size()
    single = {
        session
        for session, label in labels.items()
        if label == NEGATIVE and 4 <= counts[session] < 8  # one bag
    }
    assert len(single) >= 3
    found = []
    for seed in range(20):
        trained = Counter(
            bag.session for bag in sampler_of(seed=seed).epoch(1).bags() if bag.label == NEGATIVE
        )
        found.append(frozenset(trained.items()))
    assert len(set(found)) > 1
    for session in single:
        times = sum(session in dict(seen) for seen in found)
        assert 0 < times < len(found), session


def test_ties_of_segment_use_are_drawn_at_random_not_by_segment() -> None:
    # 9008 has 41 segments and offers 16 in epoch 1, where none was used: which 16 is a
    # draw, so over 20 seeds far more than the first 16 of its interview are offered.
    segments, _ = fold(POSITIVES, NEGATIVES)
    assert (segments["session"] == 9008).sum() == 41
    seen: set[str] = set()
    for seed in range(20):
        plan = sampler_of(seed=seed).epoch(1)
        offered = [bag for bag in offered_bags(plan) if bag.session == 9008]
        assert sum(len(bag.segments) for bag in offered) == 16
        seen.update(segment for bag in offered for segment in bag.segments)
    assert len(seen) > 30


def test_a_participants_offered_segments_are_shuffled_before_they_are_cut_into_bags() -> None:
    # Every offered bag trains in epoch 1 (8 + 8 bags, 4 steps). 9003 has 20 segments and
    # leaves 4 unused; in epoch 2 those 4 come first by use, and the shuffle spreads them
    # over its bags, so they rarely form a bag of their own.
    together = 0
    for seed in range(10):
        sampler = sampler_of((16, 16), (20, 16), seed=seed)
        first = sampler.epoch(1)
        assert not first.unused
        used = {segment for bag in first.bags() if bag.session == 9003 for segment in bag.segments}
        left = {f"9003_{index:03d}" for index in range(20)} - used
        assert len(left) == 4
        bags = [set(bag.segments) for bag in offered_bags(sampler.epoch(2)) if bag.session == 9003]
        assert left <= set().union(*bags)  # the least used come first
        together += left in bags
    assert together < 10


def test_a_participant_whose_bags_were_all_left_out_trains_in_the_next_epoch() -> None:
    # 24 negative bags train in each epoch and 6 are left over, so all bags of those left
    # out rank first in the next epoch, at a share of 0.
    positives = (16,) * 5 + (8, 8)
    negatives = (4, 5, 6, 7, 8, 9, 12, 13, 16, 18, 22, 30, 3)
    left_out = 0
    for seed in range(50):
        sampler = sampler_of(positives, negatives, seed=seed)
        first, second = sampler.epoch(1), sampler.epoch(2)
        trained = {bag.session for bag in first.bags()}
        missed = {bag.session for bag in first.unused} - trained
        left_out += len(missed)
        assert missed <= {bag.session for bag in second.bags()}
    assert left_out > 0


def test_participants_of_one_bag_left_out_train_in_the_next_epoch_epoch_after_epoch() -> None:
    # 20 of 22 one-bag negative participants train in each epoch; the 2 left out have the
    # lowest share afterwards.
    negatives = (4, 5, 6, 7) * 5 + (4, 5)
    for seed in range(20):
        sampler = sampler_of((16,) * 5, negatives, seed=seed)
        for index in range(1, 6):
            missed = {bag.session for bag in sampler.epoch(index).unused}
            assert len(missed) == 2
            assert missed <= {bag.session for bag in sampler.epoch(index + 1).bags()}


def test_the_two_halves_of_an_epoch_hold_negative_bags_of_the_same_earlier_use() -> None:
    # Earlier use: how often a bag's segments were placed in a step in earlier epochs. Over
    # the first 1,000 seeds, epoch 3's second half minus its first half lies within 3
    # standard errors of 0. (The first 200 seeds alone lie at 3.07, a chance of about 0.2%;
    # 4,000 seeds lie at 0.98.)
    seeds = range(1000)
    differences = []
    for seed in seeds:
        plan, uses, _, _ = history(sampler_of(seed=seed), 3)[2]
        half = len(plan.steps) // 2

        def mean_use(steps: Sequence[tuple[Bag, ...]], uses: Counter = uses) -> float:
            return float(
                np.mean(
                    [
                        sum(uses[segment] for segment in bag.segments)
                        for step in steps
                        for bag in step
                        if bag.label == NEGATIVE
                    ]
                )
            )

        differences.append(mean_use(plan.steps[-half:]) - mean_use(plan.steps[:half]))
    assert np.mean(differences) != 0.0
    assert abs(np.mean(differences)) <= 3 * np.std(differences, ddof=1) / np.sqrt(len(seeds))


def test_a_fourth_epoch_keeps_the_bags_of_the_first_three() -> None:
    three, four = sampler_of(), sampler_of()
    fourth = four.epoch(4)  # made first: it makes epochs 1 to 3 on the way
    assert [four.epoch(index) for index in (1, 2, 3)] == [three.epoch(index) for index in (1, 2, 3)]
    assert fourth == three.epoch(4)


def test_the_bags_do_not_depend_on_the_order_of_the_rows_or_the_type_of_the_labels() -> None:
    segments, labels = fold(POSITIVES, NEGATIVES)
    shuffled = segments.sample(frac=1.0, random_state=0)
    series = pd.Series(labels).sample(frac=1.0, random_state=1)
    assert digest(BagSampler(shuffled, series, BAGS, SAMPLER, SEED)) == digest(sampler_of())


# Start bags, people and streams ----------------------------------------------------------------


def test_the_start_bags_are_the_bags_of_the_first_8_steps_16_of_each_class() -> None:
    sampler = sampler_of((16,) * 10, (16,) * 20)
    start = sampler.start_bags(8)
    assert start == [bag for step in sampler.epoch(1).steps[:8] for bag in step]
    assert Counter(bag.label for bag in start) == {POSITIVE: 16, NEGATIVE: 16}
    with pytest.raises(StepError, match="epoch 1 has 20 steps, fewer than the 21"):
        sampler.start_bags(21)


def test_training_people_are_those_of_the_rounds_other_folds() -> None:
    sessions = range(9001, 9041)
    labels = pd.DataFrame(
        {"label": [int(s % 3 == 0) for s in sessions], "gender": [s % 2 for s in sessions]},
        index=list(sessions),
    )
    partitions = make_partitions(labels)
    for round_number in (1, 2):
        for test_fold in range(1, 6):
            people = training_people(partitions, round_number, test_fold)
            in_round = partitions[partitions["round"] == round_number]
            tested = set(in_round.loc[in_round["fold"] == test_fold, "session"])
            assert people == tuple(sorted(set(sessions) - tested))
    assert training_people(partitions, 1, 2) != training_people(partitions, 2, 2)


@pytest.mark.parametrize(
    ("round_number", "fold_number"), [(3, 1), (0, 1), (1, 6), (1, 0), (True, 1), (1, True)]
)
def test_training_people_refuse_a_round_or_fold_that_does_not_exist(
    round_number: int, fold_number: int
) -> None:
    labels = pd.DataFrame(
        {"label": [0, 1] * 10, "gender": [0, 0, 1, 1] * 5}, index=range(9001, 9021)
    )
    with pytest.raises(ValueError, match=r"no round|holds folds"):
        training_people(make_partitions(labels), round_number, fold_number)


def test_only_training_people_get_bags() -> None:
    segments, labels = fold(POSITIVES, NEGATIVES)
    table = pd.DataFrame(
        {"label": labels.values(), "gender": [s % 2 for s in labels]}, index=list(labels)
    )
    partitions = make_partitions(table)
    people = training_people(partitions, 1, 3)
    mine = segments[segments["session"].isin(people)]
    sampler = BagSampler(mine, {s: labels[s] for s in people}, BAGS, SAMPLER, stream_seed(101, 3))
    for index in (1, 2, 3):
        assert {bag.session for bag in offered_bags(sampler.epoch(index))} <= set(people)
    # A test-fold participant's segment is refused, not trained.
    with pytest.raises(StepError, match="no training label"):
        BagSampler(segments, {s: labels[s] for s in people}, BAGS, SAMPLER, stream_seed(101, 3))


def test_the_stream_of_a_fold_run_is_its_rounds_seed_and_its_fold() -> None:
    assert STREAM == "bags"
    assert stream_seed(101, 1) == item_seed(101, "bags", 1)
    seeds = {stream_seed(round_seed, fold) for round_seed in (101, 102) for fold in range(1, 6)}
    assert len(seeds) == 10


def test_another_round_or_fold_gives_other_bags() -> None:
    same = digest(sampler_of(seed=stream_seed(101, 1)))
    assert digest(sampler_of(seed=stream_seed(101, 1))) == same
    assert digest(sampler_of(seed=stream_seed(102, 1))) != same
    assert digest(sampler_of(seed=stream_seed(101, 2))) != same


def test_the_three_arms_share_one_sampler() -> None:
    settings = {
        arm: load_arm(CONFIGS / f"{arm}.yaml").shared.sampler for arm in ("b0", "b1", "cpg")
    }
    assert settings["b0"] == settings["b1"] == settings["cpg"] == SAMPLER
    segments, labels = fold(POSITIVES, NEGATIVES)
    digests = {
        digest(BagSampler(segments, labels, BAGS, sampler, SEED)) for sampler in settings.values()
    }
    assert len(digests) == 1


DIGEST_CODE = """
import hashlib, json, sys
import pandas as pd
from listening_beyond_words.bags import BagSampler, BagSettings, stream_seed
from listening_beyond_words.config import SamplerSettings

counts = {9001 + i: 3 + (i * 7) % 29 for i in range(30)}
rows = [{"segment": f"{s}_{k:03d}", "session": s} for s, n in counts.items() for k in range(n)]
labels = {s: int(i % 3 == 0) for i, s in enumerate(counts)}
sampler = BagSampler(
    pd.DataFrame(rows),
    labels,
    BagSettings(4, 16, False),
    SamplerSettings(
        2, 2, "a_class_runs_out", "least_trained_first", "lowest_trained_share_first", "shuffled"
    ),
    stream_seed(101, 1),
)
plans = [
    [[[b.session, b.label, list(b.segments)] for b in step] for step in sampler.epoch(e).steps]
    for e in (1, 2, 3)
]
sys.stdout.write(hashlib.sha256(json.dumps(plans).encode()).hexdigest())
"""


def test_the_same_seed_gives_the_same_bags_in_another_process() -> None:
    mine = io.StringIO()
    with contextlib.redirect_stdout(mine):
        exec(DIGEST_CODE, {})
    for hash_seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", DIGEST_CODE],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert result.stdout == mine.getvalue()


# Settings and refusals -------------------------------------------------------------------------


def test_the_bag_settings_come_from_the_data_config() -> None:
    config = yaml.safe_load((CONFIGS / "data.yaml").read_text(encoding="utf-8"))
    assert BagSettings.from_config(config) == BAGS


@pytest.mark.parametrize(
    ("bags", "message"),
    [
        (None, "bags must be a mapping"),
        (
            {"segments_per_bag": 4, "max_segments_per_epoch": 16},
            "missing keys \\['repeat_segments'\\]",
        ),
        (
            {"segments_per_bag": 4, "max_segments_per_epoch": 16, "repeat_segments": False, "x": 1},
            "unknown keys \\['x'\\]",
        ),
        (
            {"segments_per_bag": 0, "max_segments_per_epoch": 16, "repeat_segments": False},
            "at least 1",
        ),
        (
            {"segments_per_bag": 4.0, "max_segments_per_epoch": 16, "repeat_segments": False},
            "whole number",
        ),
        (
            {"segments_per_bag": True, "max_segments_per_epoch": 16, "repeat_segments": False},
            "whole number",
        ),
        (
            {"segments_per_bag": 4, "max_segments_per_epoch": 3, "repeat_segments": False},
            "at least 4",
        ),
        (
            {"segments_per_bag": 4, "max_segments_per_epoch": 16, "repeat_segments": True},
            "must be false",
        ),
        (
            {"segments_per_bag": 4, "max_segments_per_epoch": 16, "repeat_segments": 0},
            "must be false",
        ),
    ],
)
def test_the_bag_settings_refuse_what_the_sampler_does_not_do(bags: object, message: str) -> None:
    with pytest.raises(StepError, match=message):
        BagSettings.from_config({"bags": bags})


def test_the_sampler_refuses_segments_and_labels_that_break_its_rules() -> None:
    segments, labels = fold(POSITIVES, NEGATIVES)
    twice = pd.concat([segments, segments.iloc[[5]]])
    with pytest.raises(StepError, match="listed twice"):
        BagSampler(twice, labels, BAGS, SAMPLER, SEED)
    without_9001 = {session: label for session, label in labels.items() if session != 9001}
    with pytest.raises(StepError, match="segment 9001_000 belongs to session 9001"):
        BagSampler(segments, without_9001, BAGS, SAMPLER, SEED)
    for label in (2, -1, True, 1.0, "1"):
        with pytest.raises(StepError, match="the label must be 1 or 0"):
            BagSampler(segments, {**labels, 9001: label}, BAGS, SAMPLER, SEED)
    with pytest.raises(StepError, match="columns segment and session"):
        BagSampler(segments.rename(columns={"session": "participant"}), labels, BAGS, SAMPLER, SEED)


@pytest.mark.parametrize("option", sorted(IMPLEMENTED))
def test_the_sampler_refuses_an_option_that_it_does_not_implement(option: str) -> None:
    segments, labels = fold(POSITIVES, NEGATIVES)
    settings = {
        "positive_bags": 2,
        "negative_bags": 2,
        **IMPLEMENTED,
        option: "something_else",
    }
    with pytest.raises(ValueError, match=f"implements {option}"):
        BagSampler(segments, labels, BAGS, SamplerSettings(**settings), SEED)


def test_the_sampler_refuses_repeated_segments_a_negative_seed_and_epochs_from_0() -> None:
    segments, labels = fold(POSITIVES, NEGATIVES)
    with pytest.raises(ValueError, match="never repeats"):
        BagSampler(segments, labels, BagSettings(4, 16, True), SAMPLER, SEED)
    with pytest.raises(ValueError, match="seed"):
        BagSampler(segments, labels, BAGS, SAMPLER, -1)
    sampler = BagSampler(segments, labels, BAGS, SAMPLER, SEED)
    with pytest.raises(ValueError, match="counted from 1"):
        sampler.epoch(0)
    with pytest.raises(ValueError, match="at least 1 step"):
        sampler.start_bags(0)
    with pytest.raises(ValueError, match="at least 1 epoch"):
        sampler.total_steps(0)
