"""The main study's power simulation (power.py), on synthetic numbers only.

The model, the draws, the test's decision, the Green call, the fits and the tables are
checked against hand values, NumPy, SciPy and scikit-learn, on synthetic tables
(power_fakes) and the study design's assumptions; never on the published tables, which only
the job reads, after the release (docs/pilot_diagnostics.md, section "Order"). A block is
also rebuilt from its own stream, and the numbers of a fit and of a scenario's studies are
recomputed from the blocks that they came from (power_fakes.Recording). The slow test
checks the model against the study design's figures at full size.
"""

import os
import re
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

import power_fakes as fakes
from listening_beyond_words import power
from listening_beyond_words.reports import write_table
from listening_beyond_words.runinfo import StepError, item_seed

ALPHAS = (0.05 / 3, 0.025, 0.05)
GAINS = tuple(round(0.01 * k, 10) for k in range(11))


def settings_of(config: dict | None = None) -> power.PowerSettings:
    return power.PowerSettings.from_config(config or fakes.tiny_config())


# Settings and inputs --------------------------------------------------------------------------


def test_the_settings_of_power_yaml_are_read_with_every_alpha_and_gain() -> None:
    settings = power.PowerSettings.from_config(fakes.real_config())
    assert settings.alphas == (0.05 / 3, 0.025, 0.05)
    assert settings.gains == GAINS
    assert settings.gains[3] == 0.03  # clean decimals, not 0.030000000000000002
    assert (settings.green_auroc, settings.green_delta) == (Fraction(3, 5), Fraction(3, 100))
    assert (settings.studies, settings.blocks, settings.resamples) == (8000, 64, 2000)
    assert (settings.rho_points, settings.calibration_pilots, settings.seed) == (5, 2000, 110)
    assert settings.design_scenarios == ((0.9, 0.85), (0.8, 0.75), (0.6, 0.65))
    assert (settings.check_resamples, settings.fallback_resamples) == (10000, 10000)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c.pop("green"), "power.yaml must hold exactly"),
        (lambda c: c.update(views=["rev", "x0"]), "power.yaml: views must be ['x0', 'rev']"),
        (
            lambda c: c["sensitivity"].update(rows=["agreement_80"]),
            "power.yaml: sensitivity.rows must be",
        ),
        (lambda c: c["simulation"].update(rho_points=4), "simulation.rho_points must be odd"),
        (lambda c: c["simulation"].update(studies=8000.0), "simulation.studies must be a whole"),
        (
            lambda c: c["resample_check"].update(resamples=40),
            "resample_check.resamples must exceed simulation.resamples",
        ),
        (lambda c: c["main"]["gains"].update(first=0.01), "main.gains must run from 0.0"),
        (lambda c: c["main"]["gains"].update(step=0.03), "main.gains must run from 0.0"),
        (lambda c: c["inputs_sha256"].update(summary="ABC"), "inputs_sha256.summary must be"),
        (
            lambda c: c["design_check"].update(scenarios=[[0.9, 0.85]]),
            "design_check.scenarios must be the study design's three pairs",
        ),
        (lambda c: c["green"].update(f3_min_auroc=1), "green.f3_min_auroc must be a float"),
    ],
)
def test_a_setting_out_of_its_type_range_or_order_is_refused(change, message: str) -> None:
    config = fakes.tiny_config()
    change(config)
    with pytest.raises(StepError, match=re.escape(message)):
        power.PowerSettings.from_config(config)


def test_the_pilots_numbers_and_design_are_read_from_the_two_tables() -> None:
    pilots, design = fakes.pilots_of(settings_of())
    assert design == fakes.DESIGN
    assert design.classes == (57, 132)
    x0 = pilots["x0"]
    assert (x0.b0_auroc, x0.b0_ends) == (0.66, (0.61, 0.71))
    assert (x0.delta, x0.delta_rounds, x0.delta_se) == (0.04, (0.03, 0.05), 0.016)
    assert (x0.agreement, x0.agreement_ends) == (0.9, (0.88, 0.92))
    assert x0.round_difference == pytest.approx(-0.02)
    assert x0.aurocs == {
        "zeroshot": (0.71, 0.69),
        "b0": (0.65, 0.67),
        "b1": (0.66, 0.68),
        "cpg": (0.68, 0.72),
    }
    assert pilots["rev"].agreement == 0.85


def test_a_table_without_a_number_or_with_other_class_counts_is_refused() -> None:
    settings = settings_of()
    summary, folds = fakes.summary_table().astype(str), fakes.fold_table().astype(str)
    missing = summary.loc[~((summary["number"] == "agreement_b0_cpg") & (summary["view"] == "rev"))]
    with pytest.raises(StepError, match="one row of agreement_b0_cpg on rev"):
        power.read_pilot(missing, folds, settings)
    text = summary.copy()
    text.loc[(text["number"] == "cpg_minus_b0") & (text["view"] == "x0"), "se"] = "not defined"
    with pytest.raises(StepError, match="cpg_minus_b0 on x0 holds no number in se"):
        power.read_pilot(text, folds, settings)
    for where, column, value in (
        ((folds["round"] == "2") & (folds["fold"] == "1"), "positives", "13"),
        ((folds["view"] == "rev") & (folds["fold"] == "3"), "negatives", "25"),
    ):
        other = folds.copy()
        other.loc[where, column] = value
        with pytest.raises(StepError, match="the same class counts fold by fold"):
            power.read_pilot(summary, other, settings)
    one_round = folds.loc[folds["round"] == "1"]
    with pytest.raises(StepError, match="holds 1 rounds, not the pilot's two"):
        power.read_pilot(summary, one_round, settings)
    six = settings_of(fakes.tiny_config(main={"folds": 6}))
    with pytest.raises(StepError, match=r"main\.folds is 6, but the pilot's rounds hold 5 folds"):
        power.read_pilot(summary, folds, six)


# The model ----------------------------------------------------------------------------------


def test_the_model_gives_its_auroc_and_the_within_class_agreement_r_times_rho() -> None:
    rng = np.random.default_rng(7)
    positives = 150_000
    people = power.simulate_people(rng, 1, 2 * positives)
    model = power.Model(b0_auroc=0.65, r=0.8, rho=0.75)
    noise = [rng.standard_normal((1, 2 * positives)) for _ in range(4)]
    b0, cpg = power.arm_scores(model, people, noise[0], noise[1], 0.05, positives)
    again, _ = power.arm_scores(model, people, noise[2], noise[3], 0.05, positives)
    labels = np.r_[np.ones(positives), np.zeros(positives)]
    assert roc_auc_score(labels, b0[0]) == pytest.approx(0.65, abs=0.004)
    assert roc_auc_score(labels, cpg[0]) == pytest.approx(0.70, abs=0.004)
    for part in (slice(0, positives), slice(positives, None)):  # within each class
        assert np.corrcoef(b0[0, part], cpg[0, part])[0, 1] == pytest.approx(0.6, abs=0.01)
        assert np.corrcoef(b0[0, part], again[0, part])[0, 1] == pytest.approx(0.75, abs=0.01)
        assert np.std(b0[0, part]) == pytest.approx(1.0, abs=0.01)
    assert power.class_shift(0.5) == 0.0
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        power.class_shift(1.0)
    # CPG's shift is that of B0's AUROC plus the gain, not B0's shift plus the gain.
    b0_shift, cpg_shifts = power.class_shifts(model, (0.0, 0.05))
    assert b0_shift == cpg_shifts[0] == power.class_shift(0.65)
    assert cpg_shifts[1] == power.class_shift(0.65 + 0.05)
    assert cpg_shifts[1] - b0_shift == pytest.approx(0.1967, abs=1e-4)  # not the gain, 0.05


def test_the_blocks_pilots_reach_b0s_auroc_and_cpg_b0s_auroc_plus_the_true_gain() -> None:
    task = power.Task(
        seed=item_seed(110, "test", "shifts"),
        count=4000,
        model=power.Model(0.65, 0.8, 0.75),
        gains=(0.0, 0.05),
        design=fakes.DESIGN,
    )
    block = power.simulate_block(task)
    pairs = np.tile(fakes.DESIGN.pairs, fakes.DESIGN.pilot_rounds)
    # A fold's AUROC estimates the model's without bias, so over 4,000 pilots the mean is
    # within 0.003 (about 5 standard errors) of the AUROC that the scenario sets.
    assert power.fold_mean(block.pilot_b0, pairs).mean() == pytest.approx(0.65, abs=0.003)
    delta = power.fold_mean(block.pilot_cpg - block.pilot_b0[:, None, :], pairs)
    assert delta[:, 0].mean() == pytest.approx(0.0, abs=0.003)
    assert delta[:, 1].mean() == pytest.approx(0.05, abs=0.003)


def test_each_rounds_folds_hold_the_pilots_class_counts_and_every_participant_once() -> None:
    rng = np.random.default_rng(3)
    order = power.stratified_folds(rng, 50, fakes.DESIGN)
    assert order.shape == (50, 189)
    for row in order:
        assert sorted(row[:57].tolist()) == list(range(57))
        assert sorted(row[57:].tolist()) == list(range(57, 189))
    slices = fakes.DESIGN.slices()
    assert [(s.stop - s.start, t.stop - t.start) for s, t in slices] == list(
        zip(fakes.DESIGN.positives, fakes.DESIGN.negatives, strict=True)
    )
    first = np.concatenate([order[:, s] for s, _ in slices], axis=1)
    assert (first < 57).all()
    # Random assignments: participant 0, a positive one, and participant 57, a negative one,
    # each land in every fold over 50 studies.
    folds_of_zero = {next(f for f, (s, _) in enumerate(slices) if 0 in row[s]) for row in order}
    assert folds_of_zero == set(range(5))
    folds_of_57 = {next(f for f, (_, t) in enumerate(slices) if 57 in row[t]) for row in order}
    assert folds_of_57 == set(range(5))


def test_the_agreement_is_the_spearman_correlation_within_a_fold() -> None:
    rng = np.random.default_rng(5)
    first = np.round(rng.standard_normal((20, 38)), 1)  # ties on a grid of 0.1
    second = first + rng.standard_normal((20, 38))
    found = power.fold_agreement(first, second)
    for row in range(20):
        x, y = rankdata(first[row]), rankdata(second[row])
        assert found[row] == pytest.approx(np.corrcoef(x, y)[0, 1], abs=1e-12)
    assert (power.fold_agreement(first, first) == 1.0).all()
    with pytest.raises(ValueError, match="do not vary"):
        power.fold_agreement(np.ones((1, 5)), first[:1, :5])


# Draws and the test ------------------------------------------------------------------------


def test_the_draws_keep_the_class_sizes_and_redraw_a_fold_without_a_class() -> None:
    classes = (10, 20)
    # Round 1: a fold with a single positive participant, so that many resamples miss it.
    folds = [
        (np.array([0]), np.arange(10, 15)),
        (np.arange(1, 10), np.arange(15, 30)),
        (np.arange(0, 5), np.arange(10, 20)),  # round 2
        (np.arange(5, 10), np.arange(20, 30)),
    ]
    counts, redraws = power.vector_draws(np.random.default_rng(11), classes, folds, 500)
    assert counts.shape == (500, 30)
    assert counts.dtype == np.int64
    assert (counts[:, :10].sum(axis=1) == 10).all()
    assert (counts[:, 10:].sum(axis=1) == 20).all()
    for positives, negatives in folds:
        assert (counts[:, positives].sum(axis=1) > 0).all()
        assert (counts[:, negatives].sum(axis=1) > 0).all()
    assert redraws > 0.9**10 * 500 * 0.8  # about 35% of the resamples miss participant 0
    # The kept resamples are the valid ones of the stream's first batch, in their order.
    rng = np.random.default_rng(11)
    batch = np.concatenate(
        [
            np.stack(
                [np.bincount(row, minlength=size) for row in rng.integers(0, size, (500, size))]
            )
            for size in classes
        ],
        axis=1,
    )
    valid = np.ones(500, dtype=bool)
    for positives, negatives in folds:
        valid &= (batch[:, positives].sum(axis=1) > 0) & (batch[:, negatives].sum(axis=1) > 0)
    assert (counts[: valid.sum()] == batch[valid]).all()
    with pytest.raises(ValueError, match="must hold both classes"):
        power.vector_draws(
            np.random.default_rng(1), classes, [(np.arange(0), np.arange(10, 12))], 5
        )


def test_without_any_redraw_each_participant_is_drawn_once_on_average() -> None:
    folds = [(np.arange(0, 57), np.arange(57, 189))]
    counts, redraws = power.vector_draws(np.random.default_rng(2), (57, 132), folds, 4000)
    assert redraws == 0
    assert counts.mean(axis=0) == pytest.approx(np.ones(189), abs=0.06)


def fold_of(
    rng: np.random.Generator, people: np.ndarray, gains: int
) -> tuple[power.FoldWins, np.ndarray, np.ndarray]:
    """A fold with scores on a grid (ties included): its wins and both arms' scores."""
    positives, negatives = people
    b0 = np.round(rng.standard_normal(len(positives) + len(negatives)) * 2) / 2
    cpg = np.round(rng.standard_normal((gains, len(positives) + len(negatives))) * 2) / 2
    p = len(positives)
    wins = power.FoldWins(
        positives,
        negatives,
        power.pair_wins(b0[:p], b0[p:]),
        np.stack([power.pair_wins(row[:p], row[p:]) for row in cpg]),
    )
    return wins, b0, cpg


def test_every_count_1_gives_the_point_delta_and_a_participant_drawn_k_times_counts_k_times() -> (
    None
):
    rng = np.random.default_rng(13)
    layout = [
        (np.array([0, 1, 2]), np.array([6, 7, 8, 9])),
        (np.array([3, 4, 5]), np.array([10, 11, 12, 13, 14])),
    ]
    folds, scores = [], []
    for people in layout:
        wins, b0, cpg = fold_of(rng, people, gains=2)
        folds.append(wins)
        scores.append((b0, cpg))
    point = np.zeros(2)
    for (positives, negatives), (b0, cpg) in zip(layout, scores, strict=True):
        labels = np.r_[np.ones(len(positives)), np.zeros(len(negatives))]
        point += [roc_auc_score(labels, row) - roc_auc_score(labels, b0) for row in cpg]
    point /= len(layout)
    ones = np.ones((1, 15), dtype=np.int64)
    assert power.resampled_delta(ones, folds)[0] == pytest.approx(point, abs=1e-12)
    # Participant 1 drawn 3 times and participant 7 twice: the AUROCs of the repeated rows.
    counts = np.ones((1, 15), dtype=np.int64)
    counts[0, 1], counts[0, 7] = 3, 2
    repeated = np.zeros(2)
    for (positives, negatives), (b0, cpg) in zip(layout, scores, strict=True):
        people = np.r_[positives, negatives]
        times = counts[0, people]
        labels = np.repeat(np.r_[np.ones(len(positives)), np.zeros(len(negatives))], times)
        repeated += [
            roc_auc_score(labels, np.repeat(row, times))
            - roc_auc_score(labels, np.repeat(b0, times))
            for row in cpg
        ]
    assert power.resampled_delta(counts, folds)[0] == pytest.approx(repeated / 2, abs=1e-12)


def float64_delta(counts: np.ndarray, folds: list[power.FoldWins]) -> np.ndarray:
    """Δ on each resample, written out in float64 without chunks: the reference."""
    weights = counts.astype(np.float64)
    total = np.zeros((len(weights), folds[0].cpg.shape[0]))
    for fold in folds:
        positive, negative = weights[:, fold.positives], weights[:, fold.negatives]
        for gain, cpg in enumerate(fold.cpg):
            difference = cpg.astype(np.float64) - fold.b0
            wins = np.einsum("rp,pq,rq->r", positive, difference, negative)
            total[:, gain] += wins / (2.0 * positive.sum(axis=1) * negative.sum(axis=1))
    return total / len(folds)


def test_the_resampled_delta_is_exact_in_float32_and_in_chunks() -> None:
    task = power.Task(
        seed=item_seed(110, "test", "exact"),
        count=2,
        model=power.Model(0.62, 0.7, 0.8),
        gains=(0.0, 0.04, 0.08),
        design=fakes.DESIGN,
        main_rounds=3,
        resamples=10,
        alphas=ALPHAS,
    )
    rng = np.random.default_rng(task.seed)
    people = power.simulate_people(rng, task.count, fakes.DESIGN.size)
    runs = power._fold_runs(rng, task, people, 3)
    folds = power._study_folds(runs, 1, slice(None))
    members = [(fold.positives, fold.negatives) for fold in folds]
    # More resamples than a chunk, so that the chunks' seams are crossed.
    counts, _ = power.vector_draws(rng, fakes.DESIGN.classes, members, power.CHUNK + 300)
    assert np.array_equal(power.resampled_delta(counts, folds), float64_delta(counts, folds))
    # Counts so large that float32 could not hold the sums exactly take float64.
    large = counts[:20] * 400
    assert np.array_equal(power.resampled_delta(large, folds), float64_delta(large, folds))
    assert power.resampled_delta(np.full((1, 189), 400), folds) == pytest.approx(
        power.resampled_delta(np.ones((1, 189), dtype=np.int64), folds), abs=1e-15
    )


def test_the_decision_at_each_alpha_is_the_linear_quantile_above_0() -> None:
    values = np.arange(-1, 60) / 1000  # 61 resamples: -0.001 to 0.059
    planted = np.column_stack([values, values - 0.002, values + 0.001])
    found = power.passes(planted, ALPHAS)
    expected = np.quantile(planted, list(ALPHAS), axis=0, method="linear") > 0
    assert (found == expected).all()
    # At 0.05/3 the quantile is exactly 0 (not above); at 0.025 it is 0.0005; at 0.05, 0.002.
    assert found[:, 0].tolist() == [False, True, True]
    assert found[:, 1].tolist() == [False, False, False]
    assert found[:, 2].tolist() == [True, True, True]
    assert not power.passes(np.zeros((40, 3)), ALPHAS).any()


def test_the_green_call_holds_at_its_bounds_exactly() -> None:
    pairs = np.array([50, 50, 50])  # three fold-runs; doubled wins are 100·AUROC
    b0 = np.array(
        [
            [60, 60, 60],  # B0 at exactly 0.60
            [61, 59, 60],  # 0.61, 0.59 and 0.60: exactly 0.60 on average
            [59, 60, 60],  # just below 0.60
        ]
    )
    # CPG at each "gain": Δ exactly 0.03, just below (0.0267), and exactly 0.03 again.
    cpg = np.stack([b0 + np.array(step) for step in ([3, 3, 3], [3, 3, 2], [4, 3, 2])], axis=1)
    green = power.green_pilot(b0, cpg, pairs, Fraction(3, 5), Fraction(3, 100))
    assert green.tolist() == [[True, False, True], [True, False, True], [False, False, False]]
    # Unequal folds: 30/(2·25) = 0.6 and 66/(2·55) = 0.6.
    unequal = power.green_pilot(
        np.array([[30, 66]]),
        np.array([[[33, 69]]]),
        np.array([25, 55]),
        Fraction(3, 5),
        Fraction(3, 100),
    )
    # Δ = (3/50 + 3/110) / 2 = 0.0436 >= 0.03.
    assert unequal.tolist() == [[True]]
    with pytest.raises(ValueError, match="too large for exact"):
        power.green_pilot(b0, cpg, np.array([2**50, 3, 5]), Fraction(3, 5), Fraction(3, 100))


def test_a_mean_over_folds_is_the_mean_of_the_per_fold_values_not_a_pooled_ratio() -> None:
    # Fold AUROCs 4/(2·2) = 1.0 and 8/(2·8) = 0.5: their mean is 0.75; pooled, 12/20 = 0.6.
    assert power.fold_mean(np.array([[4, 8]]), np.array([2, 8])).tolist() == [0.75]


def test_r_and_rho_of_1_at_a_true_gain_of_0_give_a_delta_of_0_and_no_pass() -> None:
    task = power.Task(
        seed=item_seed(110, "test", "identical"),
        count=4,
        model=power.Model(0.65, 1.0, 1.0),
        gains=(0.0,),
        design=fakes.DESIGN,
        main_rounds=3,
        resamples=200,
        alphas=ALPHAS,
    )
    block = power.simulate_block(task)
    assert (block.pilot_cpg[:, 0, :] == block.pilot_b0).all()
    assert (block.main_delta == 0.0).all()
    assert not block.decisions.any()


def test_every_true_gain_shares_the_draws_of_a_block() -> None:
    fields = {
        "seed": item_seed(110, "test", "shared"),
        "count": 3,
        "model": power.Model(0.6, 0.8, 0.75),
        "design": fakes.DESIGN,
        "main_rounds": 3,
        "resamples": 100,
        "alphas": ALPHAS,
    }
    grid = power.simulate_block(power.Task(gains=(0.0, 0.05, 0.1), **fields))
    alone = power.simulate_block(power.Task(gains=(0.05,), **fields))
    assert (grid.pilot_b0 == alone.pilot_b0).all()
    assert (grid.pilot_cpg[:, 1] == alone.pilot_cpg[:, 0]).all()
    assert (grid.main_delta[:, 1] == alone.main_delta[:, 0]).all()
    assert (grid.decisions[:, :, 1] == alone.decisions[:, :, 0]).all()
    # The same draws make CPG's wins and the decisions rise with the true gain.
    assert (np.diff(grid.pilot_cpg, axis=1) >= 0).all()
    assert (np.diff(grid.decisions.astype(int), axis=2) >= 0).all()


def weighted_auroc(
    positive: np.ndarray,
    negative: np.ndarray,
    times_positive: np.ndarray,
    times_negative: np.ndarray,
) -> float:
    """An AUROC with each participant counted as often as drawn and ties one half: the reference."""
    above = (positive[:, None] > negative[None, :]) + 0.5 * (positive[:, None] == negative[None, :])
    return float(
        times_positive @ above @ times_negative / (times_positive.sum() * times_negative.sum())
    )


def rebuilt_rounds(task: power.Task) -> tuple[np.random.Generator, list[tuple[np.ndarray, list]]]:
    """A block's rounds rebuilt from its stream: each round's layout and each gain's scores.

    The stream is used in simulate_block's documented order: the participants, then the
    pilot's rounds and the main study's, each a stratified layout and both arms' run noise.
    The scores come from arm_scores, one gain at a time, not from the blocks' own code.
    """
    design = fakes.DESIGN
    rng = np.random.default_rng(task.seed)
    people = power.simulate_people(rng, task.count, design.size)
    rounds = []
    for _ in range(design.pilot_rounds + task.main_rounds):
        order = power.stratified_folds(rng, task.count, design)
        noise = (
            rng.standard_normal((task.count, design.size)),
            rng.standard_normal((task.count, design.size)),
        )
        scores = [
            power.arm_scores(task.model, people, *noise, gain, design.classes[0])
            for gain in task.gains
        ]
        rounds.append((order, scores))
    return rng, rounds


def fold_runs_of(rounds: list[tuple[np.ndarray, list]], study: int) -> list[tuple[np.ndarray, ...]]:
    """One study's fold-runs: positives, negatives, B0's scores and CPG's at each gain."""
    runs = []
    for order, scores in rounds:
        for positive, negative in fakes.DESIGN.slices():
            people = (order[study, positive], order[study, negative])
            runs.append(
                (
                    *people,
                    np.concatenate([scores[0][0][study, part] for part in people]),
                    np.stack(
                        [np.concatenate([cpg[study, part] for part in people]) for _, cpg in scores]
                    ),
                )
            )
    return runs


def resampled_reference(runs: list[tuple[np.ndarray, ...]], counts: np.ndarray) -> np.ndarray:
    """Δ on each resample and at each gain, from repeated participants: shape (R, G)."""
    values = np.zeros((len(counts), runs[0][3].shape[0]))
    for positives, negatives, b0, cpg in runs:
        p = len(positives)
        for row, times in enumerate(counts):
            weights = times[positives].astype(float), times[negatives].astype(float)
            base = weighted_auroc(b0[:p], b0[p:], *weights)
            values[row] += [weighted_auroc(arm[:p], arm[p:], *weights) - base for arm in cpg]
    return values / len(runs)


def test_a_block_rebuilt_from_its_stream_gives_its_agreement_standard_error_and_main_study() -> (
    None
):
    model, gains, pilot_runs = power.Model(0.62, 0.8, 0.75), (0.0, 0.05), 10
    pilots = power.Task(
        seed=item_seed(110, "test", "rebuilt", "pilots"),
        count=2,
        model=model,
        gains=gains,
        design=fakes.DESIGN,
        resamples=30,
        agreement=True,
    )
    block = power.simulate_block(pilots)
    rng, rounds = rebuilt_rounds(pilots)
    for study in range(pilots.count):
        runs = fold_runs_of(rounds, study)
        labels = [np.r_[np.ones(len(p)), np.zeros(len(n))] for p, n, _, _ in runs]
        # Each fold-run's doubled wins are twice its AUROC times its pairs.
        for run, (positives, negatives, b0, cpg), label in zip(
            range(pilot_runs), runs, labels, strict=True
        ):
            pairs = len(positives) * len(negatives)
            assert block.pilot_b0[study, run] == round(2 * pairs * roc_auc_score(label, b0))
            assert block.pilot_cpg[study, :, run].tolist() == [
                round(2 * pairs * roc_auc_score(label, arm)) for arm in cpg
            ]
        # The agreement: the Spearman correlation of B0's and CPG's scores within each of the
        # pilot's 10 fold-runs, at the first gain, averaged.
        within = [spearmanr(b0, cpg[0]).statistic for _, _, b0, cpg in runs]
        assert block.agreement[study] == pytest.approx(np.mean(within), abs=1e-12)
        # The bootstrap standard error of Δ at the first gain, on the pilot's 10 fold-runs,
        # from the stream's next draws: the SD (ddof 1) over the resamples.
        members = [(positives, negatives) for positives, negatives, _, _ in runs]
        counts, _ = power.vector_draws(rng, fakes.DESIGN.classes, members, pilots.resamples)
        expected = np.std(resampled_reference(runs, counts)[:, 0], ddof=1)
        assert block.pilot_se[study] == pytest.approx(expected, abs=1e-12)
    studies = power.Task(
        seed=item_seed(110, "test", "rebuilt", "studies"),
        count=2,
        model=model,
        gains=gains,
        design=fakes.DESIGN,
        main_rounds=3,
        resamples=30,
        alphas=ALPHAS,
    )
    block = power.simulate_block(studies)
    rng, rounds = rebuilt_rounds(studies)
    skipped = 0
    for study in range(studies.count):
        main = fold_runs_of(rounds, study)[pilot_runs:]  # the main study's 15 fold-runs
        assert len(main) == 15
        # Δ: the mean of the 15 per-fold differences, at each gain.
        point = resampled_reference(main, np.ones((1, fakes.DESIGN.size), dtype=np.int64))[0]
        assert block.main_delta[study] == pytest.approx(point, abs=1e-12)
        # The decisions: the linear α-quantile of the resampled Δ above 0.
        members = [(positives, negatives) for positives, negatives, _, _ in main]
        counts, _ = power.vector_draws(rng, fakes.DESIGN.classes, members, studies.resamples)
        quantiles = np.quantile(
            resampled_reference(main, counts), list(ALPHAS), axis=0, method="linear"
        )
        clear = np.abs(quantiles) > 1e-9  # a quantile at 0 itself could round either way
        skipped += int((~clear).sum())
        assert (block.decisions[study][clear] == (quantiles > 0)[clear]).all()
    assert skipped <= 1


# Fits and checks ----------------------------------------------------------------------------


def test_a_calibration_meets_its_target_within_the_tolerance_or_stops() -> None:
    fit = power.calibrate_r(lambda r: 0.5 + 0.3 * r, 0.62, 1e-3, "rising")
    assert abs(fit.reached - 0.62) <= 1e-4  # within a tenth of the tolerance
    assert fit.value == pytest.approx(0.4, abs=1e-4 / 0.3)
    fit = power.calibrate_r(lambda r: 0.04 - 0.03 * r, 0.016, 1e-3, "falling")
    assert abs(fit.reached - 0.016) <= 1e-4
    assert fit.value == pytest.approx(0.8, abs=1e-4 / 0.03)
    # A tight tolerance takes nearly every halving and keeps the closest point.
    assert power.calibrate_r(lambda r: r, 1 / 3, 1e-8, "tight").value == pytest.approx(
        1 / 3, abs=1e-8
    )
    with pytest.raises(
        StepError, match=r"rising: the model reaches 0.5000 to 0.8000, which misses"
    ):
        power.calibrate_r(lambda r: 0.5 + 0.3 * r, 0.95, 1e-3, "rising")
    # A step that jumps over the target cannot meet it within the tolerance either.
    with pytest.raises(StepError, match="the closest fit reaches"):
        power.calibrate_r(lambda r: 0.5 if r < 0.3 else 0.8, 0.62, 1e-3, "jump")
    # Near the bound: a fit within the tolerance is kept, one that misses by 1.5 times it is not.
    near = power.calibrate_r(lambda r: 0.6195 if r < 0.5 else 0.6215, 0.62, 1e-3, "near")
    assert near == (0.0, 0.6195, True)
    with pytest.raises(
        StepError,
        match=r"near: the closest fit reaches 0\.618500, which misses the target 0\.620000 by"
        r" more than 0\.001",
    ):
        power.calibrate_r(lambda r: 0.6185 if r < 0.5 else 0.6215, 0.62, 1e-3, "near")
    floor = power.lowest_rho(lambda rho: rho**2, 0.81, 1e-3, "rho")
    assert abs(floor.reached - 0.81) <= 1e-4


def test_a_sensitivity_fit_that_misses_keeps_its_closest_point_marked_not_met() -> None:
    # stop=False, the sensitivity rows' fit: where a fit at a ρ point stops the job (the test
    # above), this one returns the closest point it found, marked not met.
    above = power.calibrate_r(lambda r: 0.5 + 0.25 * r, 0.95, 1e-3, "rising", stop=False)
    assert above == (1.0, 0.75, False)  # out of reach above: the end r = 1
    below = power.calibrate_r(lambda r: 0.5 + 0.25 * r, 0.2, 1e-3, "rising", stop=False)
    assert below == (0.0, 0.5, False)  # out of reach below: the end r = 0
    jump = power.calibrate_r(lambda r: 0.5 if r < 0.3 else 0.8, 0.62, 1e-3, "jump", stop=False)
    assert jump == (0.0, 0.5, False)  # in range, but a step jumps over the target
    near = power.calibrate_r(
        lambda r: 0.6195 if r < 0.5 else 0.6215, 0.62, 1e-3, "near", stop=False
    )
    assert near == (0.0, 0.6195, True)  # within the tolerance: met
    met = power.calibrate_r(lambda r: 0.5 + 0.3 * r, 0.62, 1e-3, "rising", stop=False)
    assert met.met
    assert abs(met.reached - 0.62) <= 1e-4


def round_deltas(blocks: list[power.Block], pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each pilot's Δ in round 1 and in round 2, at the first gain, from the blocks."""
    b0 = np.concatenate([block.pilot_b0 for block in blocks]).astype(float)
    cpg = np.concatenate([block.pilot_cpg[:, 0, :] for block in blocks]).astype(float)
    per_fold = (cpg - b0) / (2.0 * np.tile(pairs, 2))
    folds = len(pairs)
    return per_fold[:, :folds].mean(axis=1), per_fold[:, folds:].mean(axis=1)


def test_only_the_sensitivity_rows_go_on_after_a_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    # The lowest ρ and r at every ρ point stop the job on a miss; the five sensitivity rows
    # keep their closest point instead (pilot diagnostics, bullet "Which number sets what").
    settings = settings_of()
    pilots, design = fakes.pilots_of(settings)
    real = power.calibrate_r
    stops: dict[str, bool] = {}

    def watched(
        value_at: Any, target: float, tolerance: float, what: str, stop: bool = True
    ) -> power.Fit:
        stops[what.split(": ")[0]] = stop
        return real(value_at, target, tolerance, what, stop)

    monkeypatch.setattr(power, "calibrate_r", watched)
    power.fit_view(pilots["x0"], settings, design, fakes.Recording(), settings.resamples)
    assert stops == {
        "x0, rho_2": True,
        "x0, rho_3": True,
        "x0, rho_4": True,
        "x0, rho_5": True,
        "x0, agreement_low_80": False,
        "x0, agreement_high_80": False,
        "x0, b0_auroc_low_80": False,
        "x0, b0_auroc_high_80": False,
        "x0, se_matched": False,
    }


def test_the_fits_of_a_view_meet_their_targets_on_equal_steps_of_rho() -> None:
    settings = settings_of()
    pilots, design = fakes.pilots_of(settings)
    runner = fakes.Recording()
    pilot = pilots["x0"]
    scenarios = power.fit_view(pilot, settings, design, runner, settings.resamples)
    names = [scenario.name for scenario in scenarios]
    assert names == [
        "rho_1",
        "rho_2",
        "rho_3",
        "rho_4",
        "rho_5",
        "agreement_low_80",
        "agreement_high_80",
        "b0_auroc_low_80",
        "b0_auroc_high_80",
        "se_matched",
    ]
    rhos = [scenario.model.rho for scenario in scenarios[:5]]
    assert rhos[-1] == 1.0
    assert np.diff(rhos) == pytest.approx([(1.0 - rhos[0]) / 4] * 4)
    assert scenarios[0].model.r == 1.0
    middle = rhos[2]
    assert {scenario.model.rho for scenario in scenarios[5:]} == {middle}
    for scenario in scenarios:
        reached = (
            scenario.bootstrap_se if scenario.fitted_to == "bootstrap_se" else scenario.agreement
        )
        assert abs(reached - scenario.target) <= settings.tolerance, scenario.name
        assert 0.0 <= scenario.model.r <= 1.0
        assert 0.0 <= scenario.round_share <= 1.0
    assert [scenario.target for scenario in scenarios[5:7]] == [0.88, 0.92]
    assert [scenario.model.b0_auroc for scenario in scenarios[7:9]] == [0.61, 0.71]
    assert (scenarios[-1].fitted_to, scenarios[-1].target) == ("bootstrap_se", 0.016)
    # Every fit runs the same calibration pilots, at the pilot's Δ as the true gain, never 0.
    seeds = [item_seed(110, "x0", "calibration", block) for block in range(settings.blocks)]
    for tasks, _ in runner.calls:
        assert [task.seed for task in tasks] == seeds
        assert {task.gains for task in tasks} == {(pilot.delta,)}
        assert sum(task.count for task in tasks) == settings.calibration_pilots
        assert all(task.agreement and not task.main_rounds for task in tasks)
    # Each scenario's numbers come from its own pilots, the last call of the fit for each.
    for scenario, (tasks, blocks) in zip(scenarios, runner.calls[-len(scenarios) :], strict=True):
        assert {(task.model, task.resamples) for task in tasks} == {
            (scenario.model, settings.resamples)
        }
        agreement = np.concatenate([block.agreement for block in blocks])
        assert scenario.agreement == pytest.approx(agreement.mean(), abs=1e-15)
        standard_errors = np.concatenate([block.pilot_se for block in blocks])
        assert scenario.bootstrap_se == pytest.approx(standard_errors.mean(), abs=1e-15)
        first, second = round_deltas(blocks, design.pairs)
        apart = np.abs(first - second) >= abs(pilot.round_difference)
        assert scenario.round_share == pytest.approx(apart.mean(), abs=1e-12)
    # x0's round difference is negative, so the share compares with its size: below 1.
    assert pilot.round_difference < 0.0
    assert all(scenario.round_share < 1.0 for scenario in scenarios)


def test_the_design_check_reports_every_figure_that_misses() -> None:
    small = settings_of(fakes.tiny_config(design_check={"pilots": 400, "max_gap": 0.0}))
    check = power.design_check(small, fakes.DESIGN, fakes.Serial())
    assert len(check.pilot_sd) == 3
    assert len(check.misses) == 4  # the three SDs and the one-arm SD, none exactly the design's
    assert check.misses[0].startswith("the pilot's Δ in the low scenario: SD")
    loose = settings_of(fakes.tiny_config(design_check={"pilots": 400, "max_gap": 0.01}))
    assert power.design_check(loose, fakes.DESIGN, fakes.Serial()).misses == ()


def test_the_resample_check_falls_back_when_decisions_change() -> None:
    config = fakes.tiny_config(
        simulation={"resamples": 20},
        resample_check={"studies": 30, "resamples": 400, "max_changed": 0.0},
    )
    check = power.resample_check(settings_of(config), fakes.DESIGN, fakes.Serial())
    assert check.changed.shape == (3, 2)
    assert check.changed.max() > 0.0
    assert (not check.passed, check.resamples) == (True, 60)
    config["resample_check"]["max_changed"] = 1.0
    check = power.resample_check(settings_of(config), fakes.DESIGN, fakes.Serial())
    assert (check.passed, check.resamples) == (True, 20)


class Changed:
    """A stand-in runner for the resample check: in the first block, the first ``changed``
    studies decide otherwise on the first resamples, at every α and true gain."""

    def __init__(self, changed: int) -> None:
        self.changed = changed

    def map(self, tasks: list[power.Task]) -> list[power.Block]:
        blocks = []
        for number, task in enumerate(tasks):
            decisions = np.zeros((task.count, len(task.alphas), len(task.gains)), dtype=bool)
            first = decisions.copy()
            if number == 0:
                first[: self.changed] = True
            runs = 2 * len(task.design.positives)
            blocks.append(
                power.Block(
                    pilot_b0=np.zeros((task.count, runs), dtype=np.int64),
                    pilot_cpg=np.zeros((task.count, len(task.gains), runs), dtype=np.int64),
                    agreement=None,
                    pilot_se=None,
                    main_delta=np.zeros((task.count, len(task.gains))),
                    decisions=decisions,
                    first_decisions=first,
                    redraws=0,
                )
            )
        return blocks


def test_the_resample_check_passes_at_exactly_its_bound_and_falls_back_above_it() -> None:
    # 100 studies at each true gain: one changed decision is a share of exactly 0.01.
    config = fakes.tiny_config(resample_check={"studies": 100, "max_changed": 0.01})
    settings = settings_of(config)
    at = power.resample_check(settings, fakes.DESIGN, Changed(1))
    assert (at.changed == 0.01).all()
    assert (at.passed, at.resamples) == (True, settings.resamples)
    above = power.resample_check(settings, fakes.DESIGN, Changed(2))
    assert (above.changed == 0.02).all()
    assert (above.passed, above.resamples) == (False, settings.fallback_resamples)
    # The real check: 4 changed decisions of 400 studies are exactly the bound, 0.01.
    assert np.mean(np.arange(400) < 4) == 0.01


# Tables -------------------------------------------------------------------------------------


def test_the_gain_for_80_percent_power_is_linear_between_grid_points_or_above_the_grid() -> None:
    gains = (0.0, 0.05, 0.1)
    assert power.gain_for_power(np.array([0.1, 0.5, 0.9]), gains) == pytest.approx(0.0875)
    assert power.gain_for_power(np.array([0.1, 0.8, 0.9]), gains) == 0.05
    assert power.gain_for_power(np.array([0.85, 0.9, 0.95]), gains) == 0.0
    assert power.gain_for_power(np.array([0.1, 0.5, 0.79]), gains) == "above 0.10"


def view_outcomes(settings: power.PowerSettings, runner) -> tuple[list, list, dict]:
    pilots, design = fakes.pilots_of(settings)
    scenarios = power.fit_views(pilots, settings, design, runner, settings.resamples)
    outcomes = power.simulate_views(scenarios, settings, design, runner, settings.resamples)
    return scenarios, outcomes, pilots


@pytest.fixture(scope="module")
def serial() -> tuple[list, list, dict]:
    """The tiny config's scenarios, outcomes and pilot from seed 110, in this process."""
    return view_outcomes(settings_of(), fakes.Serial())


def exact_green(b0: np.ndarray, cpg: np.ndarray, pairs: np.ndarray) -> bool:
    """The Green call in fractions: B0's mean AUROC at least 3/5 and Δ at least 3/100."""
    aurocs = [Fraction(int(wins), 2 * int(count)) for wins, count in zip(b0, pairs, strict=True)]
    deltas = [
        Fraction(int(arm) - int(wins), 2 * int(count))
        for arm, wins, count in zip(cpg, b0, pairs, strict=True)
    ]
    return sum(aurocs) / len(aurocs) >= Fraction(3, 5) and sum(deltas) / len(deltas) >= Fraction(
        3, 100
    )


def test_a_scenarios_studies_are_counted_from_their_own_blocks_at_each_true_gain() -> None:
    settings = settings_of()
    runner = fakes.Recording()
    model = power.Model(0.64, 0.8, 0.75)
    outcome = power.simulate_scenario(
        "x0", "rho_3", model, settings, fakes.DESIGN, runner, settings.resamples
    )
    [(tasks, blocks)] = runner.calls
    # Each block draws from its own stream, item_seed(seed, view, scenario, block).
    seeds = [item_seed(110, "x0", "rho_3", block) for block in range(settings.blocks)]
    assert [task.seed for task in tasks] == seeds
    assert len(set(seeds)) == settings.blocks
    assert not np.array_equal(blocks[0].pilot_b0, blocks[1].pilot_b0)
    assert [task.count for task in tasks] == power.block_sizes(settings.studies, settings.blocks)
    for task in tasks:
        assert (task.model, task.gains, task.main_rounds) == (model, settings.gains, 3)
        assert (task.resamples, task.alphas) == (settings.resamples, settings.alphas)
    b0 = np.concatenate([block.pilot_b0 for block in blocks])
    cpg = np.concatenate([block.pilot_cpg for block in blocks])
    main = np.concatenate([block.main_delta for block in blocks])
    decisions = np.concatenate([block.decisions for block in blocks])
    pairs = np.tile(fakes.DESIGN.pairs, fakes.DESIGN.pilot_rounds)
    gains = range(len(settings.gains))
    green = np.array(
        [[exact_green(b0[s], cpg[s, g], pairs) for g in gains] for s in range(len(b0))]
    )
    assert outcome.studies == settings.studies == 12
    assert outcome.passes.tolist() == decisions.sum(axis=0).tolist()
    assert outcome.green.tolist() == green.sum(axis=0).tolist()
    assert 0 < outcome.green.sum() < green.size  # some pilots are Green, and some are not
    for alpha in range(len(settings.alphas)):
        for gain in gains:
            both = decisions[:, alpha, gain] & green[:, gain]
            assert outcome.green_passes[alpha, gain] == both.sum()
    # The main study's SE and the pilot-to-main correlation, read at a true gain of 0.
    zero = settings.gains.index(0.0)
    assert outcome.main_se == pytest.approx(np.std(main[:, zero], ddof=1), abs=1e-15)
    assert outcome.main_se != pytest.approx(np.std(main[:, -1], ddof=1), abs=1e-6)
    pilot = np.mean((cpg[:, zero, :] - b0) / (2.0 * pairs), axis=1)
    assert outcome.correlation == pytest.approx(np.corrcoef(pilot, main[:, zero])[0, 1], abs=1e-12)


def test_the_power_after_a_green_pilot_is_a_rate_among_the_green_pilots_only() -> None:
    settings = settings_of(
        fakes.tiny_config(main={"gains": {"first": 0.0, "last": 0.01, "step": 0.01}})
    )
    model = power.Model(0.65, 0.9, 0.5)
    counts = {
        "studies": 10,
        "passes": np.array([[2, 3], [3, 4], [5, 6]]),
        "green": np.array([4, 0]),
        "green_passes": np.array([[1, 0], [1, 0], [1, 0]]),
        "main_se": 0.03,
        "correlation": 0.5,
        "redraws": 0,
    }
    x0 = power.Outcome(view="x0", name="rho_1", model=model, **counts)
    rev = power.Outcome(view="rev", name="rho_1", model=model, **counts)
    table = power.power_table([x0, rev], settings)
    rows = table.loc[table["view"] == "x0"]
    assert rows["power"].tolist() == [0.2, 0.3, 0.5, 0.3, 0.4, 0.6]
    assert rows["power_se"].tolist() == pytest.approx(
        [np.sqrt(p * (1 - p) / 10) for p in (0.2, 0.3, 0.5, 0.3, 0.4, 0.6)]
    )
    at_zero = rows.loc[rows["gain"] == 0.0]
    assert at_zero["green_pilots"].tolist() == [4, 4, 4]
    assert at_zero["power_after_green"].tolist() == [0.25, 0.25, 0.25]  # 1 of 4, not of 10
    assert at_zero["power_after_green_se"].tolist() == pytest.approx([np.sqrt(0.25 * 0.75 / 4)] * 3)
    none = rows.loc[rows["gain"] == 0.01]
    assert none["green_pilots"].tolist() == [0, 0, 0]
    assert set(none["power_after_green"]) == set(none["power_after_green_se"]) == {power.NO_GREEN}
    others = table.loc[table["view"] == "rev"]
    for column in ("green_pilots", "power_after_green", "power_after_green_se"):
        assert set(others[column]) == {power.NOT_APPLICABLE}


def test_the_two_tables_hold_every_view_scenario_gain_and_alpha(tmp_path, serial) -> None:
    settings = settings_of()
    scenarios, outcomes, pilots = serial
    table = power.power_table(outcomes, settings)
    assert list(table.columns) == [
        "view",
        "scenario",
        "b0_auroc",
        "rho",
        "r",
        "gain",
        "alpha",
        "studies",
        "power",
        "power_se",
        "green_pilots",
        "power_after_green",
        "power_after_green_se",
    ]
    assert len(table) == 2 * 10 * 11 * 3
    rev = table.loc[table["view"] == "rev"]
    assert set(rev["power_after_green"]) == {"not applicable"}
    x0 = table.loc[table["view"] == "x0"]
    assert set(x0["studies"]) == {12}
    row = x0.iloc[5]
    assert row["power_se"] == pytest.approx(np.sqrt(row["power"] * (1 - row["power"]) / 12))
    assert all(
        isinstance(value, float) or value == "not defined (no Green pilot)"
        for value in x0["power_after_green"]
    )
    assert any(isinstance(value, float) for value in x0["power_after_green"])
    fits = power.fit_table(scenarios, outcomes, pilots, settings)
    assert len(fits) == 20
    assert list(fits.columns)[:9] == [
        "view",
        "scenario",
        "b0_auroc",
        "rho",
        "r",
        "fitted_to",
        "target",
        "target_met",
        "agreement_model",
    ]
    assert set(fits["target_met"]) == {"met"}  # the fakes' targets are all within reach
    assert {"gain_80_at_0.01667", "gain_80_at_0.025", "gain_80_at_0.05"} <= set(fits.columns)
    assert fits.loc[0, "auroc_cpg_round_2"] == 0.72
    assert fits.loc[0, "bootstrap_se_measured"] == 0.016
    assert fits.loc[0, "round_difference_measured"] == pytest.approx(-0.02)
    write_table(table, tmp_path / "power.csv")
    write_table(fits, tmp_path / "power_fit.csv")


def table_bytes(
    settings: power.PowerSettings, scenarios: list, outcomes: list, pilots: dict, folder
) -> bytes:
    folder.mkdir()
    write_table(power.power_table(outcomes, settings), folder / "power.csv")
    write_table(power.fit_table(scenarios, outcomes, pilots, settings), folder / "power_fit.csv")
    return (folder / "power.csv").read_bytes() + (folder / "power_fit.csv").read_bytes()


def thread_variables() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in power.ONE_THREAD}


def test_one_and_three_workers_and_the_same_seed_twice_give_byte_identical_tables(
    tmp_path, serial
) -> None:
    settings = settings_of()
    expected = table_bytes(settings, *serial, tmp_path / "serial")
    before = thread_variables()
    for count in (1, 3):
        with power.Workers(count) as workers:
            assert workers.start_method == "spawn"
            # The workers start with one BLAS and OpenMP thread; each checks it, and every
            # pool that threadpoolctl sees in one of them runs one thread (none on a Mac,
            # whose Accelerate it cannot see).
            assert thread_variables() == dict.fromkeys(power.ONE_THREAD, "1")
            assert all(pool["num_threads"] == 1 for pool in workers.pools())
            assert workers.map([]) == []
            found = table_bytes(
                settings, *view_outcomes(settings, workers), tmp_path / f"workers_{count}"
            )
        assert found == expected
        assert thread_variables() == before  # the variables get their values back
    # Another seed: other studies, so other tables.
    other = settings_of(fakes.tiny_config(simulation={"seed": 210}))
    scenarios, outcomes, _ = serial
    moved = power.simulate_views(scenarios, other, fakes.DESIGN, fakes.Serial(), other.resamples)
    assert not power.power_table(moved, other).equals(power.power_table(outcomes, settings))


def test_the_workers_run_only_inside_their_block() -> None:
    workers = power.Workers(1)
    with pytest.raises(RuntimeError, match="use Workers as a context manager"):
        workers.map([])
    with pytest.raises(RuntimeError, match="use Workers as a context manager"):
        workers.pools()
    with pytest.raises(ValueError, match="at least one worker"):
        power.Workers(0)


def test_a_worker_whose_thread_pools_run_more_than_one_thread_is_refused() -> None:
    power.check_one_thread([])  # no pool that threadpoolctl sees, as with Apple's Accelerate
    one = {"user_api": "blas", "internal_api": "openblas", "num_threads": 1}
    power.check_one_thread([one, {**one, "user_api": "openmp", "internal_api": "openmp"}])
    with pytest.raises(RuntimeError, match="one thread each: openblas with 16 threads"):
        power.check_one_thread([one, {**one, "num_threads": 16}])


def test_a_workers_start_sets_one_thread_and_refuses_a_wider_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The start-up function itself: it asks threadpoolctl for one thread, keeps the limit
    # alive and checks the pools that the worker then has.
    import threadpoolctl

    asked: list[int] = []

    def limits(limits: int) -> str:
        asked.append(limits)
        return "the limit"

    kept: list[object] = []
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", limits)
    monkeypatch.setattr(power, "_BLAS", kept)
    one = {"user_api": "blas", "internal_api": "openblas", "num_threads": 1}
    monkeypatch.setattr(power, "thread_pools", lambda: [one])
    power._one_blas_thread()
    assert (asked, kept) == ([1], ["the limit"])
    monkeypatch.setattr(power, "thread_pools", lambda: [{**one, "num_threads": 16}])
    with pytest.raises(RuntimeError, match="one thread each: openblas with 16 threads"):
        power._one_blas_thread()


def test_blocks_split_the_items_in_fixed_sizes() -> None:
    assert power.block_sizes(8000, 64) == [125] * 64
    assert power.block_sizes(2000, 64) == [32] * 16 + [31] * 48
    assert power.block_sizes(3, 64) == [1, 1, 1]


@pytest.mark.slow
def test_the_model_reproduces_the_study_designs_figures() -> None:
    settings = power.PowerSettings.from_config(fakes.real_config())
    check = power.design_check(settings, fakes.DESIGN, fakes.Serial())
    assert check.misses == ()
    for found, expected in zip(check.pilot_sd, (0.026, 0.034, 0.041), strict=True):
        assert abs(found - expected) <= 0.003
    assert abs(check.one_arm_sd - 0.044) <= 0.003
