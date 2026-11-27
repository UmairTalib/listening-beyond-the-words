"""The pooled-encoder probe's library (probe.py): pooling, targets, fits, choices and numbers.

Synthetic data only: random encoder outputs, word times and clips written by hand, a stand-in
for opensmile that gives each part the outputs set for its length (tests/probe_fakes.py),
and features and targets drawn from fixed seeds. Each rule of docs/pilot_diagnostics.md,
section "Pooled-encoder probe", is checked against a hand count, NumPy or scikit-learn.
"""

import copy
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from listening_beyond_words import probe
from listening_beyond_words.evaluation import draw_counts, interval
from listening_beyond_words.partitions import make_partitions
from listening_beyond_words.probe import (
    KINDS,
    LOUDNESS,
    PAUSES,
    PITCH,
    POOLED,
    RATE,
    WITHIN,
    ConvergenceError,
    ProbeSettings,
    TargetSettings,
    fit_logistic,
    fit_ridge,
    fold_r2,
    inner_choice,
    kept_vectors,
    logistic_folds,
    loudness,
    pooled,
    resampled_auroc,
    resampled_r2,
    ridge_folds,
    segment_targets,
    timing_targets,
    training_fill,
    within_gender,
)
from listening_beyond_words.runinfo import StepError, item_seed
from listening_beyond_words.views import PCM16_SCALE, SAMPLING_RATE
from listening_beyond_words.voicecheck import (
    FUNCTIONAL_NAMES,
    PITCH_VARIABILITY,
    level_copy,
    part_measures,
    segment_measures,
)
from probe_fakes import PartExtractor, fake_part

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs" / "pilot" / "probe.yaml").read_text(encoding="utf-8"))
SETTINGS = TargetSettings(words=40, level_rms=0.05, peak_limit=0.99, min_samples=960)
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0)
CS = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
GAP = 4_800


def output(chunks: int = 1, seed: int = 0, width: int = 4) -> np.ndarray:
    """A random stand-in for an encoder output, shape (C, 1500, W), fp16."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, (chunks, 1500, width)).astype(np.float16)


# Pooling ---------------------------------------------------------------------------------------


def test_the_kept_vectors_are_those_that_start_inside_a_part() -> None:
    # Vector f starts at sample 320·f: 0, 320, 640 and 960 lie in [0, 1000); 6080 (f = 19)
    # to 8960 (f = 28) lie in [5800, 9000). A part that starts on a vector's first sample
    # keeps it, and one that ends on it does not.
    assert kept_vectors([0, 5800], [1000, 9000], 1).tolist() == [0, 1, 2, 3, *range(19, 29)]
    assert kept_vectors([320], [640], 1).tolist() == [1]
    assert kept_vectors([321], [961], 1).tolist() == [2, 3]
    assert kept_vectors([0], [480_000], 1).tolist() == list(range(1500))


def test_a_clip_over_30_s_keeps_the_vectors_of_both_its_chunks() -> None:
    # Chunk 0's vectors 1469 to 1499 start at 470,080 to 479,680; chunk 1's vectors 0 to 31
    # start at 480,000 + 320·f, up to 489,920: places 1500 to 1531 of the flattened order.
    kept = kept_vectors([470_000], [490_000], 2)
    assert kept.tolist() == list(range(1469, 1532))
    first, second = output(2, seed=1)[0], output(2, seed=1)[1]
    expected = np.concatenate([first[1469:], second[:32]]).astype(np.float64)
    features = pooled(output(2, seed=1), kept)
    np.testing.assert_array_equal(features[:4], expected.mean(axis=0))
    np.testing.assert_array_equal(features[4:], expected.std(axis=0))


@pytest.mark.parametrize(
    ("starts", "ends", "chunks"),
    [
        ([], [], 1),
        ([0, 100], [200], 1),
        ([500], [400], 1),
        ([0, 300], [400, 500], 1),  # overlapping parts
        ([0], [480_001], 1),  # past the chunk
        ([-1], [10], 1),
        ([10], [20], 1),  # no vector starts inside
    ],
)
def test_kept_vectors_refuse_parts_out_of_order_or_past_the_chunks(
    starts: list[int], ends: list[int], chunks: int
) -> None:
    with pytest.raises(ValueError, match=r"part|vector"):
        kept_vectors(starts, ends, chunks)


def test_a_value_planted_in_the_padding_or_a_gap_changes_nothing() -> None:
    starts, ends = [0, 9_600 + GAP], [9_600, 9_600 + GAP + 16_000]
    kept = kept_vectors(starts, ends, 1)
    clean = output(seed=2)
    before = pooled(clean, kept)
    gap_vector = (9_600 + 320) // 320  # starts inside the gap
    padding_vector = 1_400  # starts after the clip's last part
    assert gap_vector not in kept
    assert padding_vector not in kept
    for place in (gap_vector, padding_vector):
        planted = clean.copy()
        planted[0, place] = 1_000.0
        np.testing.assert_array_equal(pooled(planted, kept), before)
    control = clean.copy()
    control[0, kept[3]] = 1_000.0  # a kept vector does change the features
    assert not np.array_equal(pooled(control, kept), before)


def test_pooled_mean_and_sd_equal_numpys_in_float64() -> None:
    clip = output(seed=3, width=6)
    kept = kept_vectors([0, 20_000], [12_000, 31_000], 1)
    features = pooled(clip, kept)
    vectors = clip[0, kept].astype(np.float64)
    assert features.dtype == np.float64
    assert features.shape == (12,)
    np.testing.assert_array_equal(features[:6], np.mean(vectors, axis=0))
    np.testing.assert_array_equal(features[6:], np.std(vectors, axis=0, ddof=0))


def test_pooled_refuses_another_shape_or_vectors_outside_the_output() -> None:
    with pytest.raises(ValueError, match="encoder output has shape"):
        pooled(np.zeros((1, 1499, 4), dtype=np.float16), np.array([0]))
    with pytest.raises(ValueError, match="do not lie inside"):
        pooled(output(), np.array([1500]))
    with pytest.raises(ValueError, match="do not lie inside"):
        pooled(output(), np.array([], dtype=np.int64))


# Targets ---------------------------------------------------------------------------------------


def clip_of(levels: list[tuple[int, int]]) -> tuple[np.ndarray, list[int], list[int]]:
    """A stored clip of parts of constant 16-bit values (length, value), joined by gaps."""
    pieces, starts, ends, position = [], [], [], 0
    for index, (length, value) in enumerate(levels):
        if index:
            pieces.append(np.zeros(GAP, dtype=np.int16))
            position += GAP
        starts.append(position)
        pieces.append(np.full(length, value, dtype=np.int16))
        position += length
        ends.append(position)
    return np.concatenate(pieces), starts, ends


def words_of(first: float, count: int, length: float, step: float) -> tuple[np.ndarray, np.ndarray]:
    """A part's word times: ``count`` words of ``length`` seconds, one every ``step`` seconds."""
    starts = first + step * np.arange(count)
    return starts, starts + length


TWO_PARTS = [words_of(10.0, 20, 0.2, 0.3), words_of(30.0, 20, 0.25, 0.25)]


def test_the_four_targets_equal_hand_values() -> None:
    samples, starts, ends = clip_of([(24_000, 1_000), (16_000, 2_000)])
    extractor = PartExtractor(
        {24_000: fake_part(24_000, 1.0, value=0.10), 16_000: fake_part(16_000, 0.5, value=0.40)}
    )
    found = segment_targets(samples, starts, ends, TWO_PARTS, extractor, SETTINGS)
    # Pitch variability by voiced time: (1.0 · 0.10 + 0.5 · 0.40) / 1.5.
    assert found.values[PITCH] == pytest.approx(0.2, abs=1e-12)
    # Part spans 0.3 · 19 + 0.2 = 5.9 s and 0.25 · 20 = 5.0 s; gaps 19 · 0.1 = 1.9 s and 0.
    assert found.values[RATE] == pytest.approx(40 / 10.9, abs=1e-12)
    assert found.values[PAUSES] == pytest.approx(1.9 / 10.9, abs=1e-12)
    # The stored samples of the parts, without the gap: 24,000 of 1,000 and 16,000 of 2,000.
    power = (24_000 * (1_000 / 32_768) ** 2 + 16_000 * (2_000 / 32_768) ** 2) / 40_000
    assert found.values[LOUDNESS] == pytest.approx(10 * math.log10(power), abs=1e-12)
    assert (found.short, found.unmeasured, found.overlaps) == (0, 0, 0)
    # Each part went to opensmile on its own, from the copy at an RMS of 0.05.
    assert sorted(len(piece) for piece in extractor.pieces) == [16_000, 24_000]
    copy_, _ = level_copy(samples.astype(np.float64) / PCM16_SCALE, 0.05, 0.99)
    np.testing.assert_array_equal(extractor.pieces[0], copy_[starts[0] : ends[0]])


def test_the_88_measures_are_the_voice_checks_segment_values() -> None:
    samples, starts, ends = clip_of([(24_000, 1_000), (16_000, 2_000)])
    parts = {
        24_000: fake_part(24_000, 1.0, value=0.10),
        16_000: fake_part(16_000, 0.5, value=0.40),
    }
    found = segment_targets(samples, starts, ends, TWO_PARTS, PartExtractor(parts), SETTINGS)
    copy_, _ = level_copy(samples.astype(np.float64) / PCM16_SCALE, 0.05, 0.99)
    table = part_measures(copy_, starts, ends, PartExtractor(parts), 960)
    pd.testing.assert_series_equal(found.measures, segment_measures(table))


def test_a_part_that_opensmile_cannot_measure_is_left_out_alone_and_counted() -> None:
    samples, starts, ends = clip_of([(24_000, 1_000), (20_000, 1_500), (16_000, 2_000)])
    parts = {
        24_000: fake_part(24_000, 1.0, value=0.10),
        20_000: fake_part(20_000, 1.0, value=50.0),  # refused: its pitch track is one short
        16_000: fake_part(16_000, 0.5, value=0.40),
    }
    words = [words_of(10.0, 14, 0.2, 0.3), words_of(20.0, 13, 0.2, 0.3), words_of(30, 13, 0.2, 0.3)]
    found = segment_targets(
        samples, starts, ends, words, PartExtractor(parts, refused=[20_000]), SETTINGS
    )
    assert (found.short, found.unmeasured) == (0, 1)
    assert found.values[PITCH] == pytest.approx(0.2, abs=1e-12)  # the other two parts
    # Loudness and timing keep the refused part.
    power = (
        24_000 * (1_000 / 32_768) ** 2
        + 20_000 * (1_500 / 32_768) ** 2
        + 16_000 * (2_000 / 32_768) ** 2
    ) / 60_000
    assert found.values[LOUDNESS] == pytest.approx(10 * math.log10(power), abs=1e-12)
    assert found.values[RATE] == pytest.approx(40 / (3 * 0.2 + 0.3 * 37), abs=1e-12)
    # Without the planted refusal, that part counts.
    whole = segment_targets(samples, starts, ends, words, PartExtractor(parts), SETTINGS)
    assert whole.unmeasured == 0
    assert whole.values[PITCH] != pytest.approx(0.2)


def test_a_part_under_960_samples_is_left_out_of_the_voice_measures_only() -> None:
    samples, starts, ends = clip_of([(24_000, 1_000), (959, 3_000)])
    extractor = PartExtractor({24_000: fake_part(24_000, 1.0, value=0.10)})
    words = [words_of(10.0, 39, 0.2, 0.3), words_of(30.0, 1, 0.05, 0.05)]
    found = segment_targets(samples, starts, ends, words, extractor, SETTINGS)
    assert (found.short, found.unmeasured) == (1, 0)
    assert [len(piece) for piece in extractor.pieces] == [24_000]  # never given to opensmile
    assert found.values[PITCH] == pytest.approx(0.10, abs=1e-12)
    power = (24_000 * (1_000 / 32_768) ** 2 + 959 * (3_000 / 32_768) ** 2) / 24_959
    assert found.values[LOUDNESS] == pytest.approx(10 * math.log10(power), abs=1e-12)
    assert found.values[RATE] == pytest.approx(40 / (0.3 * 38 + 0.2 + 0.05), abs=1e-12)


def test_a_segment_without_voiced_time_has_no_pitch_variability_but_its_other_targets() -> None:
    samples, starts, ends = clip_of([(24_000, 1_000), (16_000, 2_000)])
    extractor = PartExtractor(
        {24_000: fake_part(24_000, 0.0, value=0.3), 16_000: fake_part(16_000, 0.0, value=0.3)}
    )
    found = segment_targets(samples, starts, ends, TWO_PARTS, extractor, SETTINGS)
    assert math.isnan(found.values[PITCH])
    assert np.isnan(found.measures[PITCH_VARIABILITY])
    assert not any(math.isnan(found.values[name]) for name in (RATE, PAUSES, LOUDNESS))


def test_no_measured_part_leaves_every_voice_measure_without_a_value() -> None:
    samples, starts, ends = clip_of([(900, 1_000), (959, 2_000)])
    words = [words_of(10.0, 20, 0.02, 0.03), words_of(30.0, 20, 0.02, 0.03)]
    found = segment_targets(samples, starts, ends, words, PartExtractor({}), SETTINGS)
    assert found.short == 2
    assert found.measures.isna().all()
    assert list(found.measures.index) == list(FUNCTIONAL_NAMES)
    assert math.isnan(found.values[PITCH])


def test_words_that_overlap_leave_no_gap_and_are_counted() -> None:
    starts = np.array([0.0, 0.15, 0.5])
    ends = np.array([0.2, 0.4, 0.6])  # the second word starts before the first ends
    timing = timing_targets([(starts, ends)], 3)
    assert timing.overlaps == 1
    assert timing.rate == pytest.approx(3 / 0.6, abs=1e-12)
    assert timing.pauses == pytest.approx(0.1 / 0.6, abs=1e-12)


def test_the_timing_refuses_parts_that_do_not_hold_the_segments_words() -> None:
    with pytest.raises(ValueError, match="hold 39 words, not 40"):
        timing_targets([words_of(0.0, 39, 0.1, 0.2)], 40)
    with pytest.raises(ValueError, match="start and end times"):
        timing_targets([(np.array([]), np.array([]))], 0)
    samples, starts, ends = clip_of([(24_000, 1_000)])
    with pytest.raises(ValueError, match="the word times give 2 parts, the clip 1"):
        segment_targets(samples, starts, ends, TWO_PARTS, PartExtractor({}), SETTINGS)


def test_the_loudness_of_silent_parts_has_no_value() -> None:
    assert math.isnan(loudness(np.zeros(2_000, dtype=np.int16), [0, 1_500], [1_000, 2_000]))
    with pytest.raises(ValueError, match="outside the clip"):
        loudness(np.ones(100, dtype=np.int16), [0], [101])
    with pytest.raises(ValueError, match="int16"):
        loudness(np.ones(100), [0], [10])


# Fits ---------------------------------------------------------------------------------------


def test_within_gender_takes_each_genders_training_mean_off_every_row() -> None:
    values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 40.0])
    female = np.array([True, True, True, False, False, False, True, False])
    train = np.array([True, True, False, True, True, False, False, False])
    centred = within_gender(values, female, train)
    # Women's training mean 1.5, men's 15, from the training rows only.
    np.testing.assert_array_equal(centred, values - np.where(female, 1.5, 15.0))
    changed = values.copy()
    changed[~train] = 1_000.0  # the test rows never shape the means
    np.testing.assert_array_equal(within_gender(changed, female, train)[train], centred[train])
    with pytest.raises(ValueError, match="no training row"):
        within_gender(values, female, train & ~female)


def data(seed: int, rows: int = 160, features: int = 5) -> dict[str, Any]:
    """Segment rows of 40 participants in 5 folds: features, a target with signal, gender."""
    rng = np.random.default_rng(seed)
    owners = np.repeat(np.arange(40), rows // 40)
    folds = owners % 5 + 1
    female = owners % 2 == 0
    matrix = rng.normal(size=(rows, features)) + 0.5 * female[:, None]
    target = matrix @ rng.normal(size=features) + rng.normal(size=rows) + 2.0 * female
    return {"matrix": matrix, "target": target, "female": female, "folds": folds}


def test_fit_ridge_equals_scikit_learns_ridge_on_the_standardised_training_rows() -> None:
    rng = np.random.default_rng(4)
    matrix = rng.normal(3.0, 2.0, (60, 5))
    targets = np.column_stack([matrix @ rng.normal(size=5), rng.normal(size=60)])
    new = rng.normal(3.0, 2.0, (9, 5))
    fit = fit_ridge(matrix, targets, ALPHAS)
    scaler = StandardScaler().fit(matrix)
    predicted = fit.predict(new)
    assert predicted.shape == (len(ALPHAS), 9, 2)
    for place, alpha in enumerate(ALPHAS):
        for column in range(2):
            model = Ridge(alpha=alpha).fit(scaler.transform(matrix), targets[:, column])
            np.testing.assert_allclose(
                predicted[place, :, column], model.predict(scaler.transform(new)), rtol=1e-9
            )
            np.testing.assert_allclose(
                fit.coefficients(place)[:, column], model.coef_, rtol=1e-8, atol=1e-12
            )


def test_the_scaler_sees_the_training_rows_only() -> None:
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(40, 3))
    fit = fit_ridge(matrix[:30], rng.normal(size=(30, 1)), ALPHAS)
    np.testing.assert_array_equal(fit.scaler.mean_, matrix[:30].mean(axis=0))
    np.testing.assert_array_equal(fit.scaler.scale_, matrix[:30].std(axis=0))
    labels = np.array([0, 1] * 15)
    model = fit_logistic(matrix[:30], labels, 1.0, 5000)
    np.testing.assert_array_equal(model.scaler.mean_, matrix[:30].mean(axis=0))


@pytest.mark.parametrize(
    ("scores", "stronger_last", "expected"),
    [
        # The best mean wins.
        ([[0.1, 0.3, 0.2], [0.1, 0.5, 0.2]], True, (1, False, False, 0, 0)),
        # A tie goes to the stronger penalty: the larger α ...
        ([[0.3, 0.3, 0.1], [0.3, 0.3, 0.1]], True, (1, False, False, 0, 0)),
        # ... or the smaller C, here at the grid's end: a plateau, not a grid end.
        ([[0.3, 0.3, 0.1], [0.3, 0.3, 0.1]], False, (0, False, True, 0, 0)),
        # A grid end counts only on a strict win over its neighbour.
        ([[0.5, 0.3, 0.1]], True, (0, True, False, 0, 0)),
        ([[0.1, 0.3, 0.5]], True, (2, True, False, 0, 0)),
        ([[0.1, 0.5, 0.5]], True, (2, False, True, 0, 0)),
        # An inner fit that did not converge skips its grid value, even where another inner
        # fold scores it best, and both are counted: the grid values skipped and the inner
        # fits that did not converge. The neighbour is the next value with a score.
        ([[0.1, math.nan, 0.2], [0.1, 0.9, 0.2]], True, (2, True, False, 1, 1)),
        ([[0.1, math.nan, 0.2], [0.1, math.nan, 0.2]], True, (2, True, False, 1, 2)),
        ([[0.5, math.nan, 0.5]], True, (2, False, True, 1, 1)),
        ([[math.nan, math.nan, 0.4], [0.3, math.nan, 0.4]], True, (2, True, False, 2, 3)),
    ],
)
def test_the_inner_choice_picks_the_best_ties_to_the_stronger_and_counts_ends_and_skips(
    scores: list[list[float]], stronger_last: bool, expected: tuple[int, bool, bool, int, int]
) -> None:
    grid = [1.0, 10.0, 100.0]
    choice = inner_choice(np.array(scores), grid, stronger_last)
    found = (choice.place, choice.at_end, choice.plateau, choice.skipped, choice.not_converged)
    assert found == expected
    assert choice.value == grid[expected[0]]


def test_the_inner_choice_refuses_a_grid_without_a_score_or_of_another_shape() -> None:
    with pytest.raises(StepError, match="no penalty of the grid has a converged fit"):
        inner_choice(np.full((4, 3), math.nan), [1.0, 2.0, 3.0], True)
    with pytest.raises(ValueError, match="expected scores of shape"):
        inner_choice(np.zeros((4, 2)), [1.0, 2.0, 3.0], True)


def test_the_inner_choice_of_the_ridge_is_its_inner_folds_best_mean_r2() -> None:
    given = data(6)
    found = ridge_folds(
        given["matrix"], {"t": given["target"]}, given["female"], given["folds"], ALPHAS
    )
    matrix, target, female, folds = (given[key] for key in ("matrix", "target", "female", "folds"))
    for kind in KINDS:
        for fit in found[("t", kind)]:
            train = folds != fit.fold
            inner = sorted(set(folds[train].tolist()))
            scores = np.empty((len(inner), len(ALPHAS)))
            for place, held in enumerate(inner):
                fitted, checked = train & (folds != held), folds == held
                values = target if kind == POOLED else within_gender(target, female, fitted)
                predicted = fit_ridge(matrix[fitted], values[fitted][:, None], ALPHAS).predict(
                    matrix[checked]
                )
                for column in range(len(ALPHAS)):
                    scores[place, column] = r2_score(values[checked], predicted[column, :, 0])
            means = scores.mean(axis=0)
            assert fit.choice.place == int(np.argmax(means))
            assert len(set(np.round(means, 12))) == len(ALPHAS)  # no tie here


def test_a_tie_of_every_penalty_goes_to_the_strongest_in_both_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every inner score the same: every grid value ties, so each fold takes the strongest
    # penalty, the largest α and the smallest C, at the grid's end, as a plateau.
    def level(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        return np.zeros(np.broadcast_shapes(np.shape(targets), np.shape(predictions))[:-1])

    monkeypatch.setattr(probe, "fold_r2", level)
    monkeypatch.setattr(probe, "_auroc", lambda scores, labels: 0.5)
    given = data(13)
    matrix, target, female, folds = (given[key] for key in ("matrix", "target", "female", "folds"))
    found = ridge_folds(matrix, {"t": target}, female, folds, ALPHAS)
    labels = np.array([0, 1] * 20)
    participants = logistic_folds(matrix[:40], labels, np.arange(40) % 5 + 1, CS, 5000, "a probe")
    for fits, strongest in (
        (found[("t", POOLED)], len(ALPHAS) - 1),
        (found[("t", WITHIN)], len(ALPHAS) - 1),
        (participants, 0),
    ):
        assert len(fits) == 5
        for fit in fits:
            assert (fit.choice.place, fit.choice.at_end, fit.choice.plateau) == (
                strongest,
                False,
                True,
            )
    assert {fit.choice.value for fit in participants} == {1.0e-6}
    assert {fit.choice.value for fit in found[("t", POOLED)]} == {1.0e6}


def test_changing_the_outer_test_fold_changes_neither_the_choice_nor_the_weights() -> None:
    given = data(7)
    matrix, target, female, folds = (given[key] for key in ("matrix", "target", "female", "folds"))
    before = ridge_folds(matrix, {"t": target}, female, folds, ALPHAS)
    test = folds == 3
    rng = np.random.default_rng(8)
    changed_matrix, changed_target = matrix.copy(), target.copy()
    changed_matrix[test] = rng.normal(50.0, 30.0, changed_matrix[test].shape)
    changed_target[test] = rng.normal(-80.0, 9.0, int(test.sum()))
    after = ridge_folds(changed_matrix, {"t": changed_target}, female, folds, ALPHAS)
    train = ~test
    for kind in KINDS:
        old, new = before[("t", kind)][2], after[("t", kind)][2]
        assert new.fold == old.fold == 3
        assert new.choice == old.choice
        # The weights come from the training rows alone: refitted there, they predict the
        # changed test rows exactly as the probe did.
        values = target if kind == POOLED else within_gender(target, female, train)
        refit = fit_ridge(matrix[train], values[train][:, None], ALPHAS)
        expected = refit.predict(changed_matrix[test])[new.choice.place, :, 0]
        np.testing.assert_allclose(new.predictions, expected, rtol=1e-10, atol=1e-12)
    # The same holds for the logistic probe of participants.
    rng = np.random.default_rng(9)
    labels = np.array([0, 1] * 20)
    participant_folds = np.arange(40) % 5 + 1
    means = rng.normal(size=(40, 4)) + 0.8 * labels[:, None]
    first = logistic_folds(means, labels, participant_folds, CS, 5000, "a probe")
    moved = means.copy()
    moved[participant_folds == 2] = rng.normal(5.0, 4.0, (8, 4))
    second = logistic_folds(moved, labels, participant_folds, CS, 5000, "a probe")
    assert second[1].choice == first[1].choice
    keep = participant_folds != 2
    model = fit_logistic(means[keep], labels[keep], second[1].choice.value, 5000)
    np.testing.assert_allclose(
        second[1].predictions, model.scores(moved[~keep]), rtol=1e-10, atol=1e-12
    )


def test_a_segment_without_a_value_is_left_out_of_that_target_only() -> None:
    given = data(10)
    matrix, target, female, folds = (given[key] for key in ("matrix", "target", "female", "folds"))
    gapped = target * 0.5 + 1.0
    gapped[[3, 50, 51, 140]] = np.nan
    both = ridge_folds(matrix, {"full": target, "gapped": gapped}, female, folds, ALPHAS)
    alone = ridge_folds(matrix, {"full": target}, female, folds, ALPHAS)
    rows = np.isfinite(gapped)
    subset = ridge_folds(matrix[rows], {"gapped": gapped[rows]}, female[rows], folds[rows], ALPHAS)
    for kind in KINDS:
        for left, right in zip(both[("full", kind)], alone[("full", kind)], strict=True):
            np.testing.assert_array_equal(left.predictions, right.predictions)
        for left, right in zip(both[("gapped", kind)], subset[("gapped", kind)], strict=True):
            assert left.choice == right.choice
            np.testing.assert_array_equal(left.rows, np.flatnonzero(rows)[right.rows])
            np.testing.assert_allclose(left.predictions, right.predictions, rtol=1e-12)
        assert sum(len(fit.rows) for fit in both[("gapped", kind)]) == 156


def test_a_logistic_fit_that_does_not_converge_is_refused() -> None:
    rng = np.random.default_rng(11)
    labels = np.array([0, 1] * 20)
    matrix = rng.normal(size=(40, 6)) + 3.0 * labels[:, None]
    with pytest.raises(ConvergenceError, match="did not converge in 1 iterations"):
        fit_logistic(matrix, labels, 1.0, 1)
    assert isinstance(fit_logistic(matrix, labels, 1.0, 5000), probe.LogisticFit)


def test_an_inner_fit_that_does_not_converge_skips_its_grid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = probe.fit_logistic
    calls: list[float] = []

    def stalls_at_c(features: np.ndarray, labels: np.ndarray, c: float, max_iter: int) -> Any:
        calls.append(c)
        if c == 1.0e-1 and len(features) < 32:  # inner fits only: 24 of 40 participants
            raise ConvergenceError("planted")
        return real(features, labels, c, max_iter)

    monkeypatch.setattr(probe, "fit_logistic", stalls_at_c)
    rng = np.random.default_rng(12)
    labels = np.array([0, 1] * 20)
    folds = np.arange(40) % 5 + 1
    matrix = rng.normal(size=(40, 4)) + 0.6 * labels[:, None]
    found = logistic_folds(matrix, labels, folds, CS, 5000, "a probe")
    assert [fit.choice.skipped for fit in found] == [1] * 5
    assert [fit.choice.not_converged for fit in found] == [4] * 5  # each inner fold's fit
    assert all(fit.choice.value != 1.0e-1 for fit in found)
    assert len(calls) == 5 * (4 * len(CS) + 1)


def test_a_final_fit_that_does_not_converge_stops_the_job(monkeypatch: pytest.MonkeyPatch) -> None:
    real = probe.fit_logistic

    def stalls_when_final(features: np.ndarray, labels: np.ndarray, c: float, max_iter: int) -> Any:
        if len(features) == 32:  # a final fit: all four training folds of 8 participants
            raise ConvergenceError("planted")
        return real(features, labels, c, max_iter)

    monkeypatch.setattr(probe, "fit_logistic", stalls_when_final)
    labels = np.array([0, 1] * 20)
    matrix = np.random.default_rng(13).normal(size=(40, 4))
    with pytest.raises(StepError, match=r"fold 1: the final fit of a probe did not converge"):
        logistic_folds(matrix, labels, np.arange(40) % 5 + 1, CS, 5000, "a probe")


def test_a_missing_value_takes_its_columns_mean_over_the_fits_training_rows() -> None:
    matrix = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 8.0], [100.0, np.nan], [7.0, 1.0]])
    fit = np.array([True, True, True, False, False])
    train, rest = training_fill(matrix, fit, ~fit)
    np.testing.assert_array_equal(train, [[1.0, 6.0], [3.0, 4.0], [2.0, 8.0]])
    np.testing.assert_array_equal(rest, [[100.0, 6.0], [7.0, 1.0]])
    with pytest.raises(StepError, match="1 measures have no value among a fit's training"):
        training_fill(matrix, np.array([True, False, False, True, False]), ~fit)


# Numbers ----------------------------------------------------------------------------------------


def test_fold_r2_equals_scikit_learns_against_the_folds_own_mean() -> None:
    rng = np.random.default_rng(14)
    truth = rng.normal(5.0, 2.0, 50)
    guess = truth + rng.normal(0.0, 1.5, 50)
    assert float(fold_r2(truth, guess)) == pytest.approx(r2_score(truth, guess), abs=1e-12)
    many = fold_r2(np.stack([truth, -truth]), np.stack([guess, guess]))
    assert many.shape == (2,)
    assert many[1] == pytest.approx(r2_score(-truth, guess), abs=1e-12)
    with pytest.raises(ValueError, match="does not vary"):
        fold_r2(np.ones(5), np.zeros(5))


def test_with_every_count_one_the_resampled_r2_and_auroc_are_the_point_values() -> None:
    rng = np.random.default_rng(15)
    owners = np.repeat(np.arange(8), [3, 1, 4, 2, 5, 2, 3, 1])
    truth = rng.normal(-25.0, 4.0, len(owners))
    guess = truth + rng.normal(0.0, 3.0, len(owners))
    once = np.ones((1, 8), dtype=np.int64)
    assert resampled_r2(truth, guess, owners, once)[0] == pytest.approx(
        float(fold_r2(truth, guess)), abs=1e-12
    )
    scores = np.array([0.3, -1.2, 0.3, 2.0, 0.7, -0.4, 0.3, 1.1])  # ties count one half
    labels = np.array([1, 0, 0, 1, 1, 0, 1, 0])
    assert resampled_auroc(scores, labels, once)[0] == pytest.approx(
        roc_auc_score(labels, scores), abs=1e-12
    )


def test_a_participant_drawn_k_times_counts_as_k_copies_of_all_their_segments() -> None:
    rng = np.random.default_rng(16)
    owners = np.repeat(np.arange(5), [3, 2, 4, 1, 2])
    truth = rng.normal(size=len(owners))
    guess = truth + rng.normal(0.0, 0.5, len(owners))
    counts = np.array([[2, 0, 1, 3, 1], [0, 1, 4, 0, 2]])
    found = resampled_r2(truth, guess, owners, counts)
    for row, draw in enumerate(counts):
        repeated = np.concatenate(
            [np.flatnonzero(owners == who) for who in range(5) for _ in range(draw[who])]
        )
        assert found[row] == pytest.approx(r2_score(truth[repeated], guess[repeated]), abs=1e-12)
    scores = rng.normal(size=5)
    labels = np.array([1, 0, 1, 0, 0])
    aurocs = resampled_auroc(scores, labels, counts)
    for row, draw in enumerate(counts):
        repeated = np.repeat(np.arange(5), draw)
        assert aurocs[row] == pytest.approx(
            roc_auc_score(labels[repeated], scores[repeated]), abs=1e-12
        )


def test_a_resample_without_a_segment_of_the_fold_or_its_variance_is_refused() -> None:
    owners = np.array([0, 0, 1])
    with pytest.raises(ValueError, match="draws no segment of the fold"):
        resampled_r2(np.array([1.0, 2.0, 3.0]), np.zeros(3), owners, np.array([[0, 0]]))
    with pytest.raises(ValueError, match="only segments of one value"):
        resampled_r2(np.array([1.0, 2.0, 2.0]), np.zeros(3), owners, np.array([[0, 2]]))


def labels_and_partitions() -> tuple[pd.Series, pd.DataFrame]:
    sessions = list(range(300, 340))
    table = pd.DataFrame(
        {
            "label": [int(index % 5 in (0, 2)) for index in range(40)],
            "gender": ["female" if (index // 5) % 2 == 0 else "male" for index in range(40)],
        },
        index=pd.Index(sessions, name="session"),
    )
    return table["label"], make_partitions(table)


def test_the_draws_are_the_pilot_analysis_draws_from_seed_111s_own_stream() -> None:
    settings = ProbeSettings.from_config(CONFIG)
    assert settings.seed == 111
    assert settings.stream == 'runinfo.item_seed(111, "probe", "bootstrap")'
    labels, partitions = labels_and_partitions()
    found, _ = draw_counts(labels, partitions, settings.rng(), 50)
    stream = np.random.default_rng(item_seed(111, "probe", "bootstrap"))
    expected, _ = draw_counts(labels, partitions, stream, 50)
    pd.testing.assert_frame_equal(found, expected)
    analysis = np.random.default_rng(item_seed(109, "evaluate", "bootstrap"))
    assert not found.equals(draw_counts(labels, partitions, analysis, 50)[0])


def test_the_intervals_are_numpys_linear_percentiles() -> None:
    values = np.random.default_rng(17).normal(size=1_001)
    for level in ProbeSettings.from_config(CONFIG).levels:
        expected = np.quantile(values, [(1 - level) / 2, (1 + level) / 2], method="linear")
        assert interval(values, level) == (float(expected[0]), float(expected[1]))


# Settings -----------------------------------------------------------------------------------------


def test_the_settings_are_read_from_the_probes_config() -> None:
    settings = ProbeSettings.from_config(CONFIG)
    assert settings.views == ("x0", "x2", "rev")
    assert settings.partition_round == 1
    assert settings.words == 40
    assert settings.ridge_alphas == ALPHAS
    assert settings.logistic_cs == CS
    assert (settings.max_iter, settings.resamples, settings.levels) == (5000, 10000, (0.8, 0.95))


def changed(change: Any) -> dict[str, Any]:
    config = copy.deepcopy(CONFIG)
    change(config)
    return config


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c.update(views=["x0", "x2"]), "probe.yaml: views must be ['x0', 'x2', 'rev']"),
        (lambda c: c["pooling"].update(kept="all"), "pooling.kept must be 'inside_parts'"),
        (lambda c: c["pooling"].update(statistics=["mean"]), "pooling.statistics must be"),
        (lambda c: c["targets"].update(view="x2"), "targets.view must be 'x0'"),
        (lambda c: c["targets"].update(pitch_variability="F0"), "targets.pitch_variability"),
        (lambda c: c["targets"].update(timing="parts"), "targets.timing must be 'word_times'"),
        (lambda c: c["targets"].update(words=40.0), "targets.words must be a whole number"),
        (lambda c: c.update(controls=["gender"]), "controls must be"),
        (lambda c: c["penalty"].update(ridge_alphas=[1.0, 1.0]), "penalty.ridge_alphas"),
        (lambda c: c["penalty"].update(logistic_cs=[1.0, 0.1]), "penalty.logistic_cs"),
        (lambda c: c["penalty"].update(logistic_cs=[0.0, 0.1]), "penalty.logistic_cs"),
        (lambda c: c["penalty"].update(max_iter=0), "penalty.max_iter must be a whole number"),
        (lambda c: c["bootstrap"].update(resamples=1), "bootstrap.resamples must be"),
        (lambda c: c["bootstrap"].update(seed=True), "bootstrap.seed must be a whole number"),
        (lambda c: c["bootstrap"].update(levels=[0.8, 0.8]), "bootstrap.levels must be"),
        (lambda c: c.update(partition_round="1"), "partition_round must be a whole number"),
        (lambda c: c.update(extra=1), "probe.yaml must hold exactly"),
        (lambda c: c.pop("bootstrap"), "probe.yaml must hold exactly"),
        (lambda c: c["penalty"].pop("max_iter"), "probe.yaml: penalty must hold exactly"),
    ],
)
def test_the_settings_refuse_what_the_probe_does_not_compute(change: Any, message: str) -> None:
    with pytest.raises(StepError, match=re.escape(message)):
        ProbeSettings.from_config(changed(change))


# opensmile itself (OMNI only) ----------------------------------------------------------------


def tone(samples: int, pitch_hz: float, vibrato: float, amplitude: float = 0.2) -> np.ndarray:
    """A harmonic tone with 5 ms ramps, synthetic, as the voice check's tests make it."""
    t = np.arange(samples) / SAMPLING_RATE
    frequency = pitch_hz * (1 + vibrato * np.sin(2 * np.pi * 4 * t))
    phase = 2 * np.pi * np.cumsum(frequency) / SAMPLING_RATE
    wave = sum(np.sin(k * phase) / k for k in range(1, 13))
    wave = wave / np.max(np.abs(wave)) * amplitude
    ramp = min(80, samples // 2)
    window = np.ones(samples)
    window[:ramp] = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp) / ramp)
    window[samples - ramp :] = window[:ramp][::-1]
    return wave * window


@pytest.mark.slow
def test_the_targets_through_the_real_opensmile_on_synthetic_tones() -> None:
    pytest.importorskip("opensmile")
    from listening_beyond_words.voicecheck import Smile

    pieces = [tone(12_000, 150.0, 0.05), tone(10_000, 210.0, 0.10), tone(900, 180.0, 0.0)]
    stored = [np.rint(piece * PCM16_SCALE).astype(np.int16) for piece in pieces]
    samples, starts, ends, position = [], [], [], 0
    for index, piece in enumerate(stored):
        if index:
            samples.append(np.zeros(GAP, dtype=np.int16))
            position += GAP
        starts.append(position)
        samples.append(piece)
        position += len(piece)
        ends.append(position)
    clip = np.concatenate(samples)
    words = [
        words_of(1.0, 20, 0.02, 0.03),
        words_of(5.0, 19, 0.03, 0.03),
        words_of(9.0, 1, 0.05, 1),
    ]
    found = segment_targets(clip, starts, ends, words, Smile.load(), SETTINGS)
    assert (found.short, found.unmeasured) == (1, 0)
    assert found.values[PITCH] > 0  # the vibrato varies the pitch
    assert found.values[RATE] == pytest.approx(40 / (0.03 * 19 + 0.02 + 0.03 * 18 + 0.03 + 0.05))
    joined = np.concatenate(stored).astype(np.float64) / PCM16_SCALE
    assert found.values[LOUDNESS] == pytest.approx(
        20 * math.log10(float(np.sqrt(np.mean(joined * joined)))), abs=1e-12
    )
    assert found.measures.notna().sum() >= 80
    for name, value in found.values.items():
        assert math.isfinite(value), name
