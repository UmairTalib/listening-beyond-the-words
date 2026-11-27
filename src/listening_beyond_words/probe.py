"""The pooled-encoder probe: does Voxtral's frozen audio encoder pass on the voice?

``docs/pilot_diagnostics.md``, section "Pooled-encoder probe", fixes the probe, and
``configs/pilot/probe.yaml`` holds its settings (``ProbeSettings``). A probe is a simple
model trained on a network's inner features, to test whether some information can be read
from them; it changes nothing in Voxtral. This module holds:

- the pooling: the vectors of a cached clip that start inside one of its parts
  (``kept_vectors``), and the mean and the population standard deviation of each of the
  encoder's numbers over them, in float64 (``pooled``);
- the targets of a segment, measured on ``x0`` (``segment_targets``): pitch variability by
  the rules of the voice checks, with one ``voicecheck.part_measures`` call per part on the
  copy at x1's level; speech rate and pause ratio from the word times
  (``timing_targets``); and the loudness of the stored samples (``loudness``);
- the probes, each standardised on its own training rows: ridge regressions of the
  targets (``fit_ridge``) and logistic regressions of the label (``fit_logistic``), with
  the penalty chosen by inner cross-validation (``inner_choice``) inside each outer
  training set (``ridge_folds``, ``logistic_folds``), the targets also taken within gender
  (``within_gender``), and a participant's missing value filled with the training rows'
  mean (``training_fill``);
- the numbers: R² in a test fold (``fold_r2``) and in each resample of the participants
  (``resampled_r2``), and the AUROC in each resample (``resampled_auroc``), with the pilot
  analysis's pair wins (``evaluation.pair_wins``, ``evaluation.weighted_aurocs``); the
  draws and intervals are the pilot analysis's too (``evaluation.draw_counts``,
  ``evaluation.interval``).

Every fit sees only its own training rows: the inner folds of an outer training set are its
four round-1 folds, each left out once, so no participant is split, and the outer test fold
shapes neither the penalty nor the weights. Nothing here prints. A refusal names rules,
folds and counts, never a participant's ID, target, prediction, label or PHQ-8 total.
"""

import itertools
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, NamedTuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from listening_beyond_words.evaluation import pair_wins, weighted_aurocs
from listening_beyond_words.runinfo import StepError, item_seed
from listening_beyond_words.views import PCM16_SCALE
from listening_beyond_words.voicecheck import (
    FUNCTIONAL_NAMES,
    PITCH_VARIABILITY,
    Extractor,
    level_copy,
    part_measures,
    segment_measures,
)
from listening_beyond_words.voxtral import CHUNK_SAMPLES, FRAMES_PER_CHUNK

# One encoder vector per 20 ms: vector f of chunk c starts at sample 480,000·c + 320·f.
VECTOR_SAMPLES: Final = CHUNK_SAMPLES // FRAMES_PER_CHUNK
VIEWS: Final = ("x0", "x2", "rev")  # the cached views that the probe reads, in this order
POST_HOC: Final = frozenset({"rev"})  # added after the pilot's AUROCs were seen
TARGET_VIEW: Final = "x0"  # every target is measured on x0 and used for every view
STATISTICS: Final = ("mean", "sd")  # pooled over the kept vectors
KEPT: Final = "inside_parts"  # the vectors that start inside one of the segment's parts
TIMING: Final = "word_times"  # speech rate and pause ratio come from the pinned word times
CONTROLS: Final = ("gender", "length", "egemaps88")
PITCH: Final = "pitch_variability"
RATE: Final = "speech_rate"
PAUSES: Final = "pause_ratio"
LOUDNESS: Final = "loudness"
TARGETS: Final = (PITCH, RATE, PAUSES, LOUDNESS)
LABEL: Final = "label"
POOLED: Final = "pooled"
WITHIN: Final = "within_gender"
KINDS: Final = (POOLED, WITHIN)  # the two kinds of R²
BOOTSTRAP_KEYS: Final = ("probe", "bootstrap")  # the draws' stream: item_seed(seed, *keys)
_FILE: Final = "probe.yaml"
_SOURCE: Final = 'pilot diagnostics, section "Pooled-encoder probe"'
_TOP_KEYS: Final = (
    "views",
    "partition_round",
    "pooling",
    "targets",
    "controls",
    "penalty",
    "bootstrap",
)


# Settings ----------------------------------------------------------------------------------


def _block(value: Any, keys: Sequence[str], where: str) -> Mapping[str, Any]:
    """A mapping with exactly these keys, or a refusal that names them."""
    if not isinstance(value, Mapping) or list(value) != list(keys):
        found = list(value) if isinstance(value, Mapping) else type(value).__name__
        raise StepError(f"{where} must hold exactly {list(keys)}, in this order, not {found}")
    return value


def _fixed(value: Any, expected: Any, where: str) -> None:
    """A setting that names what the probe computes must be the value that it computes."""
    if value != expected or type(value) is not type(expected):
        raise StepError(
            f"{where} must be {expected!r}, as the probe computes it ({_SOURCE}), not {value!r}"
        )


def _whole(value: Any, minimum: int, where: str) -> int:
    """A whole number of at least ``minimum``; True and False are refused."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StepError(f"{where} must be a whole number of at least {minimum}, not {value!r}")
    return value


def _grid(value: Any, where: str) -> tuple[float, ...]:
    """A penalty grid: positive floats in strictly ascending order."""
    if (
        not isinstance(value, list)
        or len(value) < 2
        or any(not isinstance(item, float) or not 0.0 < item < math.inf for item in value)
        or any(low >= high for low, high in itertools.pairwise(value))
    ):
        raise StepError(
            f"{where} must be at least two positive floats in ascending order, not {value!r}"
        )
    return tuple(value)


@dataclass(frozen=True, slots=True)
class ProbeSettings:
    """The settings of the pooled-encoder probe, from ``configs/pilot/probe.yaml``.

    The settings that name what the probe computes (its views, pooling, targets and
    controls) must be the ones that this code implements, in their order; the numbers are
    checked for their type and range.

    Attributes:
        views: The cached views whose encoder outputs are probed: ``VIEWS``.
        partition_round: The round of the pinned partitions whose five folds the probes use.
        words: The words of each segment, the numerator of the speech rate.
        ridge_alphas: The ridge penalties of the inner choice, weakest first.
        logistic_cs: The logistic regression's inverse penalties, strongest first.
        max_iter: The most iterations of lbfgs in a logistic fit.
        resamples: Resamples of the bootstrap.
        seed: The run seed of the draws' own stream (``rng``).
        levels: The coverages of the percentile intervals: 0.8 and 0.95.
    """

    views: tuple[str, ...]
    partition_round: int
    words: int
    ridge_alphas: tuple[float, ...]
    logistic_cs: tuple[float, ...]
    max_iter: int
    resamples: int
    seed: int
    levels: tuple[float, ...]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ProbeSettings":
        """Read the settings and check every key and value.

        Args:
            config: The content of ``configs/pilot/probe.yaml``.

        Returns:
            The settings.

        Raises:
            StepError: If a key is missing, unknown or out of order, a setting that names
                what the probe computes is not the one that it computes, or a number has
                the wrong type or lies out of range.
        """
        top = _block(config, _TOP_KEYS, _FILE)
        _fixed(top["views"], list(VIEWS), f"{_FILE}: views")
        pooling = _block(top["pooling"], ("statistics", "kept"), f"{_FILE}: pooling")
        _fixed(pooling["statistics"], list(STATISTICS), f"{_FILE}: pooling.statistics")
        _fixed(pooling["kept"], KEPT, f"{_FILE}: pooling.kept")
        targets = _block(
            top["targets"], ("view", "words", "pitch_variability", "timing"), f"{_FILE}: targets"
        )
        _fixed(targets["view"], TARGET_VIEW, f"{_FILE}: targets.view")
        _fixed(
            targets["pitch_variability"], PITCH_VARIABILITY, f"{_FILE}: targets.pitch_variability"
        )
        _fixed(targets["timing"], TIMING, f"{_FILE}: targets.timing")
        _fixed(top["controls"], list(CONTROLS), f"{_FILE}: controls")
        penalty = _block(
            top["penalty"], ("ridge_alphas", "logistic_cs", "max_iter"), f"{_FILE}: penalty"
        )
        bootstrap = _block(top["bootstrap"], ("resamples", "seed", "levels"), f"{_FILE}: bootstrap")
        levels = bootstrap["levels"]
        if (
            not isinstance(levels, list)
            or not levels
            or any(not isinstance(level, float) or not 0.0 < level < 1.0 for level in levels)
            or len(set(levels)) != len(levels)
        ):
            raise StepError(
                f"{_FILE}: bootstrap.levels must be different floats between 0 and 1, not"
                f" {levels!r}"
            )
        return cls(
            views=VIEWS,
            partition_round=_whole(top["partition_round"], 1, f"{_FILE}: partition_round"),
            words=_whole(targets["words"], 1, f"{_FILE}: targets.words"),
            ridge_alphas=_grid(penalty["ridge_alphas"], f"{_FILE}: penalty.ridge_alphas"),
            logistic_cs=_grid(penalty["logistic_cs"], f"{_FILE}: penalty.logistic_cs"),
            max_iter=_whole(penalty["max_iter"], 1, f"{_FILE}: penalty.max_iter"),
            resamples=_whole(bootstrap["resamples"], 2, f"{_FILE}: bootstrap.resamples"),
            seed=_whole(bootstrap["seed"], 0, f"{_FILE}: bootstrap.seed"),
            levels=tuple(levels),
        )

    @property
    def stream(self) -> str:
        """The draws' stream as run.json names it."""
        keys = ", ".join(f'"{key}"' for key in BOOTSTRAP_KEYS)
        return f"runinfo.item_seed({self.seed}, {keys})"

    def rng(self) -> np.random.Generator:
        """The draws' own stream: ``item_seed(seed, "probe", "bootstrap")``.

        Pilot diagnostics, section "Pooled-encoder probe", bullet "Intervals": seed 111, a
        seed that no other step uses, through a stream of its own.
        """
        return np.random.default_rng(item_seed(self.seed, *BOOTSTRAP_KEYS))


# Pooling -----------------------------------------------------------------------------------


def kept_vectors(starts: Sequence[int], ends: Sequence[int], chunks: int) -> np.ndarray:
    """Return the encoder vectors of a cached clip that start inside one of its parts.

    A cache file holds every vector of every 30 s chunk, the padding included: 1,500 per
    chunk, one per 20 ms. Vector f of chunk c covers the 20 ms that start at sample
    480,000·c + 320·f, which is 320 times its place in the flattened (chunks, 1,500) order.
    A vector is kept if that sample lies in a part, from its start sample up to, not
    including, its end sample, so the padding and the 300 ms gaps between the parts drop
    out. A clip over 30 s keeps vectors of both its chunks.

    Args:
        starts: The first sample of each part, in clip order.
        ends: The sample just after each part.
        chunks: The clip's number of chunks (``voxtral.chunks``).

    Returns:
        The kept vectors' places in the flattened (chunks, 1,500) order, ascending, int64.

    Raises:
        ValueError: If there is no part, the parts are out of order, overlap or reach past
            the chunks, or no vector starts inside a part.
    """
    first = np.asarray(starts, dtype=np.int64)
    last = np.asarray(ends, dtype=np.int64)
    if chunks < 1 or first.ndim != 1 or first.shape != last.shape or not len(first):
        raise ValueError("expected at least one part, one end per start, and a chunk")
    edges = np.column_stack([first, last]).ravel()
    if first[0] < 0 or np.any(np.diff(edges) < 0) or np.any(last <= first):
        raise ValueError("the parts are out of order, overlap or are empty")
    if last[-1] > chunks * CHUNK_SAMPLES:
        raise ValueError(f"a part reaches past the {chunks} chunks of the clip")
    places = np.arange(chunks * FRAMES_PER_CHUNK, dtype=np.int64)
    onsets = places * VECTOR_SAMPLES
    part = np.searchsorted(first, onsets, side="right") - 1
    inside = (part >= 0) & (onsets < last[np.clip(part, 0, None)])
    kept = places[inside]
    if not len(kept):
        raise ValueError("no encoder vector starts inside a part of the clip")
    return kept


def pooled(output: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """Pool one clip's encoder output over its kept vectors: the mean and population SD.

    Args:
        output: The stored encoder output, shape (chunks, 1,500, W), such as fp16.
        kept: The kept vectors' places in the flattened (chunks, 1,500) order
            (``kept_vectors``).

    Returns:
        The mean of each of the W numbers over the kept vectors, then their population
        standard deviation (ddof 0), computed in float64: shape (2W,).

    Raises:
        ValueError: If the output does not have the shape of an encoder output, or a kept
            place lies outside it.
    """
    array = np.asarray(output)
    if array.ndim != 3 or array.shape[1] != FRAMES_PER_CHUNK or not array.shape[2]:
        raise ValueError(
            f"an encoder output has shape (C, {FRAMES_PER_CHUNK}, W), not {array.shape}"
        )
    places = np.asarray(kept, dtype=np.int64)
    flat = array.reshape(-1, array.shape[2])
    if places.ndim != 1 or not len(places) or places.min() < 0 or places.max() >= len(flat):
        raise ValueError("the kept vectors do not lie inside the encoder output")
    vectors = flat[places].astype(np.float64)
    return np.concatenate([vectors.mean(axis=0), vectors.std(axis=0)])


# Targets -----------------------------------------------------------------------------------


class Timing(NamedTuple):
    """A segment's timing targets from its word times.

    Attributes:
        rate: Words per second over the summed span of the parts; NaN if the span is 0.
        pauses: The summed gaps between consecutive words within the parts, over the same
            span; NaN if the span is 0.
        overlaps: Consecutive words of a part that overlap; such a pair has no gap.
    """

    rate: float
    pauses: float
    overlaps: int


def timing_targets(parts: Sequence[tuple[np.ndarray, np.ndarray]], words: int) -> Timing:
    """Return a segment's speech rate and pause ratio from the word times of its parts.

    Each part spans from its first word's start to its last word's end, so the parts'
    margins and the gaps between the parts stay out. The speech rate is the segment's words
    over the summed span of its parts; the pause ratio is the summed gaps between
    consecutive words within each part over the same span. Every gap counts, with no
    shortest pause; two words that overlap leave no gap and are counted.

    Args:
        parts: For each part, in clip order, its words' start and end times in seconds, in
            time order.
        words: The segment's number of words (40).

    Returns:
        The speech rate, the pause ratio and the count of overlapping pairs.

    Raises:
        ValueError: If a part has no word, its times are not paired, or the parts do not
            hold ``words`` words in all.
    """
    span = gaps = 0.0
    overlaps = count = 0
    for starts, ends in parts:
        first = np.asarray(starts, dtype=np.float64)
        last = np.asarray(ends, dtype=np.float64)
        if first.ndim != 1 or first.shape != last.shape or not len(first):
            raise ValueError("each part needs its words' start and end times")
        span += float(last[-1] - first[0])
        between = first[1:] - last[:-1]
        gaps += float(np.clip(between, 0.0, None).sum())
        overlaps += int((between < 0).sum())
        count += len(first)
    if count != words:
        raise ValueError(f"the parts hold {count} words, not {words}")
    if not span > 0:
        return Timing(math.nan, math.nan, overlaps)
    return Timing(words / span, gaps / span, overlaps)


def loudness(samples: np.ndarray, starts: Sequence[int], ends: Sequence[int]) -> float:
    """Return the level of a clip's parts as stored: 20·log10 of their RMS, in dBFS.

    The RMS is taken over the stored samples of the parts together, without the gaps, each
    16-bit sample n read as n / 32768: the level that the encoder hears.

    Args:
        samples: The stored clip, int16 of shape (samples,).
        starts: The first sample of each part.
        ends: The sample just after each part.

    Returns:
        The level in dBFS; NaN if the parts are silent.

    Raises:
        ValueError: If the clip is not 16-bit samples or a part lies outside it.
    """
    if samples.dtype != np.int16 or samples.ndim != 1:
        raise ValueError(f"expected int16 samples of shape (samples,), got {samples.dtype}")
    pieces = []
    for start, end in zip(starts, ends, strict=True):
        if not 0 <= int(start) < int(end) <= len(samples):
            raise ValueError(f"a part from sample {start} to {end} lies outside the clip")
        pieces.append(samples[int(start) : int(end)].astype(np.float64) / PCM16_SCALE)
    if not pieces:
        raise ValueError("a clip needs at least one part")
    joined = np.concatenate(pieces)
    rms = math.sqrt(float(np.mean(joined * joined)))
    return 20.0 * math.log10(rms) if rms > 0 else math.nan


class TargetSettings(NamedTuple):
    """How the targets of a segment are measured.

    Attributes:
        words: The segment's words, the numerator of the speech rate (``ProbeSettings``).
        level_rms: The RMS of the copy that opensmile measures (data.yaml,
            ``word_check.voice.level_rms``).
        peak_limit: The peak guard of that copy (data.yaml, ``views.peak_limit``).
        min_samples: Parts with fewer samples get no voice measures (data.yaml,
            ``word_check.voice.min_part_samples``).
    """

    words: int
    level_rms: float
    peak_limit: float
    min_samples: int


class SegmentTargets(NamedTuple):
    """One segment's targets, measured on x0, and what was left out of them.

    Attributes:
        values: The four targets by name (``TARGETS``); NaN where the segment has none.
        measures: The 88 eGeMAPS functionals of the segment (``voicecheck.segment_measures``
            over its measured parts); NaN where no part has a value.
        short: Parts under ``min_samples`` samples: no voice measure.
        unmeasured: Parts that opensmile could not measure (a ``StepError`` of
            ``voicecheck.part_measures``): no voice measure.
        overlaps: Pairs of consecutive words that overlap: no gap (``timing_targets``).
    """

    values: dict[str, float]
    measures: pd.Series
    short: int
    unmeasured: int
    overlaps: int


def segment_targets(
    samples: np.ndarray,
    starts: Sequence[int],
    ends: Sequence[int],
    words: Sequence[tuple[np.ndarray, np.ndarray]],
    extractor: Extractor,
    settings: TargetSettings,
    *,
    segment: str | None = None,
) -> SegmentTargets:
    """Measure one segment's targets on its stored x0 clip and its word times.

    Pilot diagnostics, section "Pooled-encoder probe", bullet "Targets":

    - pitch variability (``F0semitoneFrom27.5Hz_sma3nz_stddevNorm``) by the rules of the
      voice checks: on a copy of the whole clip at an RMS of 0.05, gaps included, then the
      peak guard (``voicecheck.level_copy``); each part measured on its own, with one
      ``voicecheck.part_measures`` call per part, so that one bad part cannot stop its clip;
      the parts weighted by voiced time (``voicecheck.segment_measures``, which also gives
      the 88 measures of the eGeMAPS control). A part under ``min_samples`` samples, or
      one that opensmile cannot measure, is left out of these and counted by kind;
    - speech rate and pause ratio from the word times (``timing_targets``);
    - loudness of the stored samples of the parts (``loudness``).

    A target without a value is NaN; the caller leaves the segment out of it and counts it.

    Args:
        samples: The stored x0 clip, int16 of shape (samples,).
        starts: The first sample of each part, from x0's manifest.
        ends: The sample just after each part.
        words: For each part, in clip order, its words' start and end times in seconds, in
            time order.
        extractor: opensmile (``voicecheck.Smile``), or a stand-in.
        settings: How the targets are measured.
        segment: The segment ID, for the messages of errors.

    Returns:
        The targets, the 88 measures and the counts of what was left out.

    Raises:
        ValueError: If the clip is not 16-bit samples, the parts do not fit it or the word
            times, or the clip is silent.
    """
    if len(words) != len(starts):
        raise ValueError(f"the word times give {len(words)} parts, the clip {len(starts)}")
    audio = samples.astype(np.float64) / PCM16_SCALE
    copy, _ = level_copy(audio, settings.level_rms, settings.peak_limit)
    tables = []
    short = unmeasured = 0
    for position, (start, end) in enumerate(zip(starts, ends, strict=True)):
        try:
            table = part_measures(
                copy, [int(start)], [int(end)], extractor, settings.min_samples, segment=segment
            )
        except StepError:  # opensmile's output does not hold: this part only is left out
            unmeasured += 1
            continue
        if not bool(table["measured"].iloc[0]):
            short += 1
            continue
        table["part"] = position
        tables.append(table)
    if tables:
        measures = segment_measures(pd.concat(tables, ignore_index=True))
    else:
        measures = pd.Series(math.nan, index=list(FUNCTIONAL_NAMES), dtype=float)
    timing = timing_targets(words, settings.words)
    values = {
        PITCH: float(measures[PITCH_VARIABILITY]),
        RATE: timing.rate,
        PAUSES: timing.pauses,
        LOUDNESS: loudness(samples, starts, ends),
    }
    return SegmentTargets(values, measures, short, unmeasured, timing.overlaps)


# Fits --------------------------------------------------------------------------------------


def within_gender(values: np.ndarray, female: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Take each gender's mean over the training rows off the target, on every row.

    Pilot diagnostics, section "Pooled-encoder probe", bullet "Numbers": the R² within
    gender reads the target less the mean of the row's gender over the training rows, on
    the training and the test rows alike; the test rows never shape the means.

    Args:
        values: The target, one value per row, shape (n,).
        female: Whether each row is a woman's, shape (n,).
        train: Whether each row is a training row, shape (n,).

    Returns:
        The target within gender, shape (n,), float64.

    Raises:
        ValueError: If the shapes differ, a value is not finite, or a gender has no
            training row.
    """
    target = np.asarray(values, dtype=np.float64)
    women = np.asarray(female, dtype=bool)
    fit = np.asarray(train, dtype=bool)
    if target.ndim != 1 or women.shape != target.shape or fit.shape != target.shape:
        raise ValueError("expected one target, gender and training flag per row")
    if not np.isfinite(target).all():
        raise ValueError("a target within gender needs a finite value on every row")
    centred = target.copy()
    for flag in (True, False):
        rows = women == flag
        if not (rows & fit).any():
            raise ValueError("a gender has no training row, so no training mean")
        centred[rows] = target[rows] - target[rows & fit].mean()
    return centred


@dataclass(frozen=True, eq=False)
class RidgeFit:
    """Ridge regressions of several targets at every penalty of a grid, from one decomposition.

    The features are standardised with the training rows' mean and standard deviation
    (scikit-learn's ``StandardScaler``), and each target gets its training mean as its
    intercept. With the eigendecomposition V diag(λ) Vᵀ of the standardised training
    rows' cross-product XᵀX, the weights at penalty α are V diag(1 / (λ + α)) Vᵀ Xᵀ (y − ȳ),
    as scikit-learn's ``Ridge`` gives them, for every α and target at the cost of one
    decomposition.

    Attributes:
        alphas: The grid of penalties.
        scaler: The standardisation, fitted on the training rows only.
        intercepts: Each target's mean over the training rows, shape (T,).
        vectors: The eigenvectors V, shape (p, p).
        values: The eigenvalues λ, shape (p,).
        projected: Vᵀ Xᵀ (Y − ȳ), shape (p, T).
    """

    alphas: tuple[float, ...]
    scaler: StandardScaler
    intercepts: np.ndarray
    vectors: np.ndarray
    values: np.ndarray
    projected: np.ndarray

    def coefficients(self, alpha: int) -> np.ndarray:
        """Return the weights at the grid's penalty ``alpha`` (its place), shape (p, T)."""
        shrink = 1.0 / (self.values + self.alphas[alpha])
        return np.asarray(self.vectors @ (shrink[:, None] * self.projected))

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict every target at every penalty for new rows.

        Args:
            features: The rows to predict, shape (m, p).

        Returns:
            The predictions, shape (penalties, m, T).
        """
        rotated = np.asarray(self.scaler.transform(np.asarray(features, dtype=np.float64)))
        rotated = rotated @ self.vectors
        return np.stack(
            [
                self.intercepts + (rotated / (self.values + alpha)) @ self.projected
                for alpha in self.alphas
            ]
        )


def fit_ridge(features: np.ndarray, targets: np.ndarray, alphas: Sequence[float]) -> RidgeFit:
    """Fit ridge regressions of each target at every penalty, on the training rows given.

    Args:
        features: The training rows' features, shape (n, p).
        targets: Their targets, shape (n, T), every value finite.
        alphas: The penalties, each above 0.

    Returns:
        The fits, which predict every target at every penalty (``RidgeFit.predict``).

    Raises:
        ValueError: If the shapes do not fit, a value is not finite, or a penalty is not
            above 0.
    """
    matrix = np.asarray(features, dtype=np.float64)
    values = np.asarray(targets, dtype=np.float64)
    if matrix.ndim != 2 or values.ndim != 2 or len(matrix) != len(values) or len(matrix) < 2:
        raise ValueError("expected at least two training rows of features and targets")
    if not (np.isfinite(matrix).all() and np.isfinite(values).all()):
        raise ValueError("a ridge fit needs finite features and targets")
    if not alphas or any(not alpha > 0 for alpha in alphas):
        raise ValueError("every ridge penalty must be above 0")
    scaler = StandardScaler().fit(matrix)
    standard = np.asarray(scaler.transform(matrix))
    intercepts = values.mean(axis=0)
    eigenvalues, vectors = np.linalg.eigh(standard.T @ standard)
    projected = vectors.T @ (standard.T @ (values - intercepts))
    return RidgeFit(
        tuple(float(alpha) for alpha in alphas), scaler, intercepts, vectors, eigenvalues, projected
    )


class ConvergenceError(StepError):
    """A logistic fit that did not converge within its iterations."""


@dataclass(frozen=True, eq=False)
class LogisticFit:
    """A logistic regression of the label, on features standardised on its training rows.

    Attributes:
        scaler: The standardisation, fitted on the training rows only.
        model: scikit-learn's ``LogisticRegression`` (L2, lbfgs), fitted on them.
    """

    scaler: StandardScaler
    model: LogisticRegression

    def scores(self, features: np.ndarray) -> np.ndarray:
        """Return the log-odds of new rows, shape (m,)."""
        standard = self.scaler.transform(np.asarray(features, dtype=np.float64))
        return np.asarray(self.model.decision_function(standard), dtype=np.float64)


def fit_logistic(features: np.ndarray, labels: np.ndarray, c: float, max_iter: int) -> LogisticFit:
    """Fit a logistic regression of the label (L2, lbfgs) on the training rows given.

    Args:
        features: The training rows' features, shape (n, p), every value finite.
        labels: Their labels, 0 or 1, both present.
        c: The inverse strength of the penalty.
        max_iter: The most iterations of lbfgs.

    Returns:
        The fit.

    Raises:
        ConvergenceError: If lbfgs does not converge within ``max_iter`` iterations.
        ValueError: If a value is not finite or a class is missing.
    """
    matrix = np.asarray(features, dtype=np.float64)
    label = np.asarray(labels, dtype=np.int64)
    if not np.isfinite(matrix).all():
        raise ValueError("a logistic fit needs finite features")
    if set(label.tolist()) != {0, 1}:
        raise ValueError("a logistic fit needs both classes among its training rows")
    scaler = StandardScaler().fit(matrix)
    model = LogisticRegression(C=c, solver="lbfgs", max_iter=max_iter)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(scaler.transform(matrix), label)
    except ConvergenceWarning:
        raise ConvergenceError(
            f"a logistic fit at C {c:g} did not converge in {max_iter} iterations"
        ) from None
    return LogisticFit(scaler, model)


class Choice(NamedTuple):
    """The penalty chosen by inner cross-validation in one outer training set.

    Attributes:
        place: The chosen value's place in the grid.
        value: The chosen value.
        at_end: It lies at an end of the grid and beats its neighbour strictly.
        plateau: It lies at an end of the grid and ties with its neighbour.
        skipped: Grid values left out because an inner fit did not converge.
        not_converged: Inner fits that did not converge, each of which skips its grid value.
    """

    place: int
    value: float
    at_end: bool
    plateau: bool
    skipped: int
    not_converged: int


def inner_choice(scores: np.ndarray, grid: Sequence[float], stronger_last: bool) -> Choice:
    """Choose a penalty from its inner cross-validation scores.

    Pilot diagnostics, section "Pooled-encoder probe", bullet "Penalty": the value with the
    best mean score over the inner folds wins, and a tie goes to the stronger penalty. An
    inner fit that did not converge (NaN) skips its grid value; both the inner fits that did
    not converge and the grid values skipped are counted. A choice at a grid end counts as
    one only if it beats its neighbour, the nearest value inwards that has a score,
    strictly; a tie there is a plateau. A choice at a grid end with no scored neighbour
    counts as one.

    Args:
        scores: Each inner fold's score at each grid value, shape (inner folds, grid
            values); NaN where the inner fit did not converge.
        grid: The grid, in the config's order.
        stronger_last: Whether the penalty grows along the grid: True for ridge's α, False
            for logistic regression's C.

    Returns:
        The choice.

    Raises:
        ValueError: If the shapes do not fit the grid.
        StepError: If no grid value has a score in every inner fold.
    """
    array = np.asarray(scores, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or array.shape[1] != len(grid):
        raise ValueError(f"expected scores of shape (inner folds, {len(grid)}), not {array.shape}")
    usable = np.asarray(np.isfinite(array).all(axis=0), dtype=bool)
    if not usable.any():
        raise StepError("no penalty of the grid has a converged fit in every inner fold")
    means = np.full(len(grid), -math.inf)
    means[usable] = array[:, usable].mean(axis=0)
    tied = np.flatnonzero(usable & (means == means[usable].max()))
    chosen = int(tied[-1] if stronger_last else tied[0])
    last = len(grid) - 1
    at_end = plateau = False
    if last and chosen in (0, last):
        inward = range(1, len(grid)) if chosen == 0 else range(last - 1, -1, -1)
        neighbour = next((place for place in inward if bool(usable[place])), None)
        if neighbour is None or means[chosen] > means[neighbour]:
            at_end = True
        else:  # the chosen value is the best, so it ties with its neighbour
            plateau = True
    skipped = int((~usable).sum())
    not_converged = int((~np.isfinite(array)).sum())
    return Choice(chosen, float(grid[chosen]), at_end, plateau, skipped, not_converged)


def fold_r2(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Return R² in a test fold, against that fold's own mean of the target.

    R² = 1 − Σ (y − ŷ)² / Σ (y − ȳ)², with ȳ the mean of y over the fold: 1 is perfect, 0
    is no better than the fold's own mean, and it can be negative. The rows lie along the
    last axis, so several targets and penalties go at once.

    Args:
        targets: The target on the fold's rows, shape (..., m).
        predictions: The predictions, broadcastable with ``targets``, shape (..., m).

    Returns:
        R², one per leading index; a 0-d array for one target.

    Raises:
        ValueError: If the target does not vary in the fold.
    """
    truth = np.asarray(targets, dtype=np.float64)
    guess = np.asarray(predictions, dtype=np.float64)
    total = np.square(truth - truth.mean(axis=-1, keepdims=True)).sum(axis=-1)
    if not np.all(total > 0):
        raise ValueError("the target does not vary in the fold, so R² is not defined")
    return np.asarray(1.0 - np.square(truth - guess).sum(axis=-1) / total)


def resampled_r2(
    targets: np.ndarray, predictions: np.ndarray, owners: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """Return a test fold's R² in each resample, each participant counted as often as drawn.

    A participant drawn k times counts k times, with all their segments, so the R² is that
    of the resample's repeated rows, against their own mean (pilot diagnostics, section
    "Pooled-encoder probe", bullet "Intervals"). The fitted probe stays fixed. The sums run
    over participants: each one's segments, summed target, summed squares and summed
    squared errors, weighted by the draw counts, with the target first shifted by its
    fold mean, which leaves R² unchanged.

    Args:
        targets: The target on the fold's segments, shape (m,).
        predictions: The probe's predictions for them, shape (m,).
        owners: Each segment's participant, as a column of ``counts``, shape (m,).
        counts: How often each of the fold's participants is drawn in each resample, shape
            (R, P).

    Returns:
        R² in each resample, shape (R,).

    Raises:
        ValueError: If the shapes do not fit, or a resample draws no segment of the fold or
            only segments of one value.
    """
    truth = np.asarray(targets, dtype=np.float64)
    guess = np.asarray(predictions, dtype=np.float64)
    place = np.asarray(owners, dtype=np.int64)
    weights = np.asarray(counts, dtype=np.float64)
    if (
        truth.ndim != 1
        or guess.shape != truth.shape
        or place.shape != truth.shape
        or weights.ndim != 2
        or (len(place) and (place.min() < 0 or place.max() >= weights.shape[1]))
    ):
        raise ValueError("expected one prediction and one participant per segment of the fold")
    centre = truth.mean() if len(truth) else 0.0
    shifted = truth - centre
    size = weights.shape[1]
    rows = weights @ np.bincount(place, minlength=size).astype(np.float64)
    summed = weights @ np.bincount(place, weights=shifted, minlength=size)
    squares = weights @ np.bincount(place, weights=shifted * shifted, minlength=size)
    errors = weights @ np.bincount(place, weights=np.square(truth - guess), minlength=size)
    if not np.all(rows > 0):
        raise ValueError("a resample draws no segment of the fold")
    total = squares - summed * summed / rows
    if not np.all(total > 0):
        raise ValueError("a resample draws only segments of one value, so R² is not defined")
    return np.asarray(1.0 - errors / total)


def resampled_auroc(scores: np.ndarray, labels: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Return a test fold's AUROC in each resample, each participant counted as often as drawn.

    Ties count one half (pilot analysis, section "The main numbers"); a pair of a positive
    and a negative participant counts as often as the product of their draw counts
    (``evaluation.weighted_aurocs``). With every count 1 it is the fold's AUROC.

    Args:
        scores: The probe's score of each of the fold's participants, shape (P,).
        labels: Their labels, 0 or 1, shape (P,).
        counts: How often each of them is drawn in each resample, shape (R, P).

    Returns:
        The AUROC in each resample, shape (R,).

    Raises:
        ValueError: If the shapes do not fit, or a resample draws no positive or no negative
            participant of the fold.
    """
    values = np.asarray(scores, dtype=np.float64)
    label = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(counts)
    if values.ndim != 1 or label.shape != values.shape or weights.ndim != 2:
        raise ValueError("expected one score and one label per participant of the fold")
    if weights.shape[1] != len(values):
        raise ValueError("expected one draw count per participant of the fold")
    positive, negative = label == 1, label == 0
    wins = pair_wins(values[positive], values[negative])
    return weighted_aurocs(wins, weights[:, positive], weights[:, negative])


# Cross-validation -----------------------------------------------------------------------


class FoldFit(NamedTuple):
    """One outer fold of a probe: its test rows, their true values and the probe's output.

    Attributes:
        fold: The outer test fold, as the partitions number it.
        rows: The test rows, as places in the probe's inputs, ascending.
        truth: The target on these rows (taken within gender for that kind), or the labels.
        predictions: The probe's prediction (ridge) or log-odds (logistic) for each test row.
        choice: The penalty that inner cross-validation chose on the training rows.
    """

    fold: int
    rows: np.ndarray
    truth: np.ndarray
    predictions: np.ndarray
    choice: Choice


def _with_kinds(
    values: np.ndarray, female: np.ndarray, fit: np.ndarray, other: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The targets pooled, then within gender with the fit rows' means: fit rows and others."""
    within = [within_gender(values[:, column], female, fit) for column in range(values.shape[1])]
    both = np.concatenate([values, np.column_stack(within)], axis=1)
    return both[fit], both[other]


def ridge_folds(
    features: np.ndarray,
    targets: Mapping[str, np.ndarray],
    female: np.ndarray,
    folds: np.ndarray,
    alphas: Sequence[float],
) -> dict[tuple[str, str], list[FoldFit]]:
    """Cross-validate ridge probes of the targets, pooled and within gender.

    For each outer test fold, the penalty is chosen by inner cross-validation over the
    other folds, each left out once (``inner_choice``, ties to the larger α), on the mean
    inner R²; the probe is then fitted on all the training rows at that penalty and
    predicts the test rows. Every fit is standardised on its own training rows
    (``fit_ridge``), and within gender each fit takes its own training rows' gender means
    off the target, on its training rows and on the rows it predicts (``within_gender``).
    A segment without a value of a target (NaN) is left out of that target; targets with
    the same rows share every fit's decomposition.

    Args:
        features: One row per segment, shape (n, p).
        targets: Each target by name, one value per segment, NaN where it has none.
        female: Whether each segment is a woman's, shape (n,).
        folds: Each segment's test fold, its participant's, shape (n,).
        alphas: The ridge penalties, weakest first.

    Returns:
        For each target and kind (``POOLED``, ``WITHIN``), its folds in ascending order.

    Raises:
        ValueError: If the shapes do not fit, a feature is not finite, a training set lacks
            a gender, or a target does not vary in a fold.
    """
    matrix = np.asarray(features, dtype=np.float64)
    women = np.asarray(female, dtype=bool)
    fold_of = np.asarray(folds, dtype=np.int64)
    if matrix.ndim != 2 or women.shape != (len(matrix),) or fold_of.shape != (len(matrix),):
        raise ValueError("expected one gender and one fold per row of features")
    groups: dict[bytes, list[str]] = {}
    for name, value in targets.items():
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (len(matrix),):
            raise ValueError(f"expected one value of {name} per row of features")
        groups.setdefault(np.isfinite(array).tobytes(), []).append(name)
    found: dict[tuple[str, str], list[FoldFit]] = {
        (name, kind): [] for name in targets for kind in KINDS
    }
    for key, names in groups.items():
        rows = np.flatnonzero(np.frombuffer(key, dtype=bool))
        own = matrix[rows]
        gender, fold = women[rows], fold_of[rows]
        values = np.column_stack(
            [np.asarray(targets[name], dtype=np.float64)[rows] for name in names]
        )
        columns = [(name, kind) for kind in KINDS for name in names]
        for outer in sorted(set(fold.tolist())):
            test = fold == outer
            train = ~test
            inner = sorted(set(fold[train].tolist()))
            scores = np.empty((len(columns), len(inner), len(alphas)))
            for place, held in enumerate(inner):
                fit, check = train & (fold != held), fold == held
                fitted, checked = _with_kinds(values, gender, fit, check)
                predicted = fit_ridge(own[fit], fitted, alphas).predict(own[check])
                scores[:, place, :] = fold_r2(checked.T, predicted.transpose(0, 2, 1)).T
            fitted, tested = _with_kinds(values, gender, train, test)
            final = fit_ridge(own[train], fitted, alphas).predict(own[test])
            for column, name_kind in enumerate(columns):
                choice = inner_choice(scores[column], alphas, stronger_last=True)
                found[name_kind].append(
                    FoldFit(
                        outer,
                        rows[test],
                        tested[:, column],
                        final[choice.place, :, column],
                        choice,
                    )
                )
    return found


def training_fill(
    matrix: np.ndarray, fit: np.ndarray, other: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fill each missing value with its column's mean over a fit's training rows.

    Pilot diagnostics, section "Pooled-encoder probe", bullet "Controls": a participant's
    missing mean of one of the 88 measures gets the training fold's mean; every fit takes
    the mean of its own training rows, so the rows it predicts never shape it.

    Args:
        matrix: One row per participant, NaN where a value is missing, shape (n, p).
        fit: The fit's training rows, shape (n,).
        other: The rows it predicts, shape (n,).

    Returns:
        The training rows and the other rows, each missing value filled.

    Raises:
        StepError: If a column has no value among the training rows.
    """
    train, rest = matrix[fit], matrix[other]
    if not (np.isnan(train).any() or np.isnan(rest).any()):
        return train, rest
    empty = int(np.isnan(train).all(axis=0).sum())
    if empty:
        raise StepError(f"{empty} measures have no value among a fit's training participants")
    means = np.nanmean(train, axis=0)
    return np.where(np.isnan(train), means, train), np.where(np.isnan(rest), means, rest)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """A fold's AUROC, ties one half: ``resampled_auroc`` with every count 1."""
    return float(resampled_auroc(scores, labels, np.ones((1, len(scores)), dtype=np.int64))[0])


def logistic_folds(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    cs: Sequence[float],
    max_iter: int,
    what: str,
) -> list[FoldFit]:
    """Cross-validate a logistic probe of the label on participants.

    For each outer test fold, C is chosen by inner cross-validation over the other folds,
    each left out once (``inner_choice``, ties to the smaller C), on the mean inner fold
    AUROC; an inner fit that does not converge skips its grid value, and the choice counts
    both. The probe is then fitted on all the training participants at that C. Every fit
    is standardised on its own training rows and fills a missing value with their mean
    (``training_fill``).

    Args:
        features: One row per participant, NaN where a value is missing, shape (P, p).
        labels: Each participant's label, 0 or 1, shape (P,).
        folds: Each participant's test fold, shape (P,).
        cs: The inverse penalties, strongest first.
        max_iter: The most iterations of lbfgs.
        what: The probe, for the message of a final fit that does not converge.

    Returns:
        The folds in ascending order; the truth is the labels.

    Raises:
        StepError: If a final fit does not converge, which stops the job, or no C
            converges in every inner fold.
        ValueError: If the shapes do not fit or a fold lacks a class.
    """
    matrix = np.asarray(features, dtype=np.float64)
    label = np.asarray(labels, dtype=np.int64)
    fold_of = np.asarray(folds, dtype=np.int64)
    if matrix.ndim != 2 or label.shape != (len(matrix),) or fold_of.shape != (len(matrix),):
        raise ValueError("expected one label and one fold per row of features")
    found = []
    for outer in sorted(set(fold_of.tolist())):
        test = fold_of == outer
        train = ~test
        inner = sorted(set(fold_of[train].tolist()))
        scores = np.full((len(inner), len(cs)), np.nan)
        for place, held in enumerate(inner):
            fit, check = train & (fold_of != held), fold_of == held
            fitted, checked = training_fill(matrix, fit, check)
            for column, c in enumerate(cs):
                try:
                    model = fit_logistic(fitted, label[fit], c, max_iter)
                except ConvergenceError:
                    continue
                scores[place, column] = _auroc(model.scores(checked), label[check])
        choice = inner_choice(scores, cs, stronger_last=False)
        fitted, tested = training_fill(matrix, train, test)
        try:
            model = fit_logistic(fitted, label[train], choice.value, max_iter)
        except ConvergenceError:
            raise StepError(
                f"fold {outer}: the final fit of {what} did not converge in {max_iter}"
                f" iterations at C {choice.value:g}, so the job stops ({_SOURCE}, bullet"
                ' "Penalty")'
            ) from None
        found.append(
            FoldFit(outer, np.flatnonzero(test), label[test], model.scores(tested), choice)
        )
    return found
