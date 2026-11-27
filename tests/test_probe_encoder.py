"""The pooled-encoder probe job (scripts/probe_encoder.py), end to end.

It runs on the light synthetic workspace of probe_fakes, with the labels and their PHQ-8
totals, the partitions, the word times, x0's clips and the caches of x0, x2 and rev written
in code, and opensmile's stand-in, in a clean checkout at the newest release tag. The fold
values and intervals of a ridge row, a length-only row and a label row, and the intervals of
two differences, are checked against an independent fit and bootstrap from the raw files and
the recorded draws; the table's counts and its pairing, its layout, every guard and pin
refusal, the order of the reads, what was left out and what may leave OMNI are checked one
by one (docs/pilot_diagnostics.md, sections "Pooled-encoder probe" and "Data protection").
"""

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import probe_fakes as fakes
from listening_beyond_words.evaluation import draw_counts, draws_sha256
from listening_beyond_words.probe import PITCH, Choice, FoldFit, ProbeSettings
from listening_beyond_words.runinfo import STOPPED, item_seed
from listening_beyond_words.voicecheck import weight_column
from probe_fakes import (
    CHECKOUT,
    JOB,
    MODELS,
    PLANTED_TARGET,
    RESAMPLES,
    SESSIONS,
    ProbeWorkspace,
    make_probe_workspace,
    run_probe,
)
from wordcheck_fakes import assert_only_aggregates_leave
from x1_fakes import REPO, load_script

ORDER = '(pilot diagnostics, section "Order")'
TARGETS = ("pitch_variability", "speech_rate", "pause_ratio", "loudness")
KINDS = ("pooled", "within_gender")
VIEWS = ("x0", "x2", "rev")
NA = "not applicable"
COLUMNS = [
    "view",
    "outcome",
    "model",
    "kind",
    "number",
    "post_hoc",
    *(f"fold_{fold}" for fold in range(1, 6)),
    "mean",
    "low_80",
    "high_80",
    "low_95",
    "high_95",
    "segments_left_out",
    "parts_short",
    "parts_unmeasured",
    "means_filled",
    "penalty",
    *(f"penalty_fold_{fold}" for fold in range(1, 6)),
    "grid_ends",
    "plateaus",
    "skipped",
    "inner_not_converged",
]
COUNTED = ["grid_ends", "plateaus", "skipped", "inner_not_converged"]
PLANTED_PREDICTION = 98.7654321


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> ProbeWorkspace:
    """The light workspace, with every input written and pinned."""
    return make_probe_workspace(tmp_path_factory.mktemp("built"))


@pytest.fixture
def workspace(built: ProbeWorkspace, tmp_path: Path) -> ProbeWorkspace:
    return built.copy_to(tmp_path)


@pytest.fixture(scope="module")
def done(built: ProbeWorkspace, tmp_path_factory: pytest.TempPathFactory) -> fakes.Ended:
    """One run of the job on a copy of the workspace, which ends done."""
    ended = run_probe(built.copy_to(tmp_path_factory.mktemp("done")))
    assert ended.ended == "done", ended.out
    return ended


def failure(ended: fakes.Ended) -> str:
    """The job's FAILED line."""
    lines = [line for line in ended.out.splitlines() if line.startswith("FAILED: ")]
    assert ended.ended == "failed", ended.out
    assert len(lines) == 1, ended.out
    return lines[0]


def expected_rows() -> list[tuple[str, str, str, str]]:
    """The table's rows in order: view, outcome, model and kind."""
    outcomes = [(name, kind) for name in TARGETS for kind in KINDS] + [("label", NA)]
    rows = [(view, outcome, "encoder", kind) for view in VIEWS for outcome, kind in outcomes]
    rows += [(NA, outcome, "gender_only", kind) for outcome, kind in outcomes]
    rows += [(NA, name, "length_only", kind) for name in TARGETS for kind in KINDS]
    rows += [("x0", "label", "egemaps88", NA)]
    for view in VIEWS:
        for outcome, kind in outcomes:
            controls = ["gender_only", "length_only"] if outcome != "label" else ["gender_only"]
            controls += ["egemaps88"] if outcome == "label" else []
            rows += [(view, outcome, f"encoder_minus_{model}", kind) for model in controls]
    for view in VIEWS[1:]:
        rows += [(f"{view}_minus_x0", outcome, "encoder", kind) for outcome, kind in outcomes]
    return rows


# The job ---------------------------------------------------------------------------------------


def test_the_probe_reads_its_inputs_and_writes_its_table_and_record(
    done: fakes.Ended, built: ProbeWorkspace
) -> None:
    assert sorted(path.name for path in done.folder.iterdir()) == ["probe.csv", "run.json"]
    info = done.info
    assert (info["status"], info["step"], info["run_id"]) == ("done", "probe_encoder", f"run-{JOB}")
    assert (info["release"], info["installed_version"], info["git"]) == (
        "v0.1.1",
        "0.1.1",
        CHECKOUT,
    )
    assert info["config"]["sha256"] == hashlib.sha256(built.probe_path.read_bytes()).hexdigest()
    assert info["config"]["content"] == built.probe
    assert info["models_config"]["sha256"] == hashlib.sha256(MODELS.read_bytes()).hexdigest()
    assert info["slurm"]["job_id"] == JOB
    assert info["settings"]["targets"] == {
        "words": 40,
        "level_rms": 0.05,
        "peak_limit": 0.99,
        "min_samples": 960,
    }
    for name in ("labels", "partitions", "segments"):
        assert info["inputs"][name] == built.config[name]
    assert info["inputs"]["clips"] == {"x0": built.config["view_runs"]["x0"]}
    assert info["inputs"]["caches"] == built.config["cache_runs"]
    # The draws: the pilot analysis's rule from seed 111's own stream, recorded by SHA-256.
    labels = built.labels["label"].astype(np.int64)
    stream = np.random.default_rng(item_seed(111, "probe", "bootstrap"))
    counts, redraws = draw_counts(labels, built.partitions, stream, RESAMPLES)
    assert info["seeds"]["bootstrap"] == {
        "seed": 111,
        "stream": 'runinfo.item_seed(111, "probe", "bootstrap")',
        "item_seed": item_seed(111, "probe", "bootstrap"),
        "resamples": RESAMPLES,
        "redraws": redraws,
        "draws_sha256": draws_sha256(counts),
    }
    results = info["results"]
    assert {key: results["counts"][key] for key in ("participants", "positives", "negatives")} == {
        "participants": 40,
        "positives": 16,
        "negatives": 24,
    }
    segments = sum(len(fakes.segments_of(session)) for session in SESSIONS)
    assert (results["counts"]["segments"], results["counts"]["two_chunk_clips"]) == (segments, 1)
    assert results["width"] == fakes.WIDTH
    assert results["rows"] == len(expected_rows())
    assert info["outputs"] == {
        "probe.csv": hashlib.sha256((done.folder / "probe.csv").read_bytes()).hexdigest()
    }
    assert set(results["minutes"]) == {"targets", "pooling", "probes"}
    assert results["reading"][0].startswith("reading: timing and loudness are expected")
    for line in results["reading"]:
        assert f"{line}\n" in done.out
    assert (
        f"table: probe.csv ({len(expected_rows())} rows) in derived/probe/run-{JOB}\n" in done.out
    )


def test_the_table_has_one_row_per_view_outcome_model_and_kind(done: fakes.Ended) -> None:
    table = done.table()
    assert list(table.columns) == COLUMNS
    assert list(table[["view", "outcome", "model", "kind"]].itertuples(index=False, name=None)) == (
        expected_rows()
    )
    assert (table["number"] == np.where(table["outcome"] == "label", "auroc", "r2")).all()
    # rev was added post hoc: its rows say so, the differences with it included.
    post_hoc = table["view"].isin(["rev", "rev_minus_x0"])
    assert (table["post_hoc"] == np.where(post_hoc, "yes", "no")).all()
    folds = table[[f"fold_{fold}" for fold in range(1, 6)]].astype(float)
    np.testing.assert_allclose(table["mean"].astype(float), folds.mean(axis=1), rtol=1e-12)
    ends = table[["low_95", "low_80", "high_80", "high_95"]].astype(float).to_numpy()
    assert (np.diff(ends, axis=1) >= 0).all()
    differences = table["model"].str.contains("_minus_") | table["view"].str.contains("_minus_")
    penalties = ["penalty", *(f"penalty_fold_{fold}" for fold in range(1, 6))]
    for column in [*penalties, *COUNTED]:
        assert (table.loc[differences, column] == NA).all(), column
        assert (table.loc[~differences, column] != NA).all(), column
    alphas = {str(value) for value in ProbeSettings.from_config(fakes_config()).ridge_alphas}
    ridge = ~differences & (table["outcome"] != "label")
    assert (table.loc[ridge, "penalty"] == "ridge alpha").all()
    assert set(table.loc[ridge, "penalty_fold_1"]) <= alphas
    assert (table.loc[~differences & ~ridge, "penalty"] == "logistic C").all()
    # A difference row is its two rows' difference, fold by fold.
    by_key = {tuple(row[:4]): row for row in table.itertuples(index=False, name=None)}
    for view in VIEWS:
        probe = by_key[(view, "speech_rate", "encoder", "pooled")]
        control = by_key[(NA, "speech_rate", "length_only", "pooled")]
        both = by_key[(view, "speech_rate", "encoder_minus_length_only", "pooled")]
        np.testing.assert_allclose(
            np.array(both[6:11], dtype=float),
            np.array(probe[6:11], dtype=float) - np.array(control[6:11], dtype=float),
            rtol=1e-12,
        )
    rev = by_key[("rev_minus_x0", "label", "encoder", NA)]
    rev_probe, x0_probe = (
        by_key[("rev", "label", "encoder", NA)],
        by_key[("x0", "label", "encoder", NA)],
    )
    np.testing.assert_allclose(
        np.array(rev[6:11], dtype=float),
        np.array(rev_probe[6:11], dtype=float) - np.array(x0_probe[6:11], dtype=float),
        rtol=1e-12,
    )


def fakes_config() -> dict[str, Any]:
    return yaml.safe_load((REPO / "configs" / "pilot" / "probe.yaml").read_text(encoding="utf-8"))


def test_what_was_left_out_is_counted_by_kind(done: fakes.Ended) -> None:
    table = done.table()
    unvoiced = len(fakes.segments_of(fakes.UNVOICED_SESSION))
    # The refused part of REFUSED_SEGMENT, and both parts of UNMEASURED_SEGMENT.
    refused = 1 + len(fakes.words_per_part(fakes.UNMEASURED_SEGMENT))
    targets = done.info["results"]["targets"]
    assert targets["segments_left_out"] == {
        "pitch_variability": unvoiced + 1,
        "speech_rate": 0,
        "pause_ratio": 0,
        "loudness": 0,
    }
    assert (targets["parts_short"], targets["parts_unmeasured"]) == (1, refused)
    # The unvoiced participant's segments lack some of the 88 measures, UNMEASURED_SEGMENT
    # all of them: only that one is left out of the measures.
    assert targets["segments_without_any_measure"] == 1
    pitch = table["outcome"] == "pitch_variability"
    label = table["outcome"] == "label"
    assert (table.loc[pitch, "segments_left_out"] == str(unvoiced + 1)).all()
    assert (table.loc[pitch, "parts_short"] == "1").all()
    assert (table.loc[pitch, "parts_unmeasured"] == str(refused)).all()
    timing = ~pitch & ~label
    assert (table.loc[timing, "segments_left_out"] == "0").all()
    assert (table.loc[timing, ["parts_short", "parts_unmeasured"]] == NA).all(axis=None)
    assert (table.loc[~label, "means_filled"] == NA).all()
    assert (
        "parts left out of pitch variability and the 88 measures: 1 under 960 samples,"
        f" {refused} that opensmile could not measure" in done.out
    )
    # The label: the encoder and gender leave nothing out; the 88 measures leave out the
    # segments with none of them, the parts by kind and the filled means. A difference row
    # shows, cell by cell, the larger count of its two rows, so the 88 measures' count shows.
    left = ["segments_left_out", "parts_short", "parts_unmeasured", "means_filled"]
    rows = {tuple(row[:3]): row for row in table[["view", "outcome", "model", *left]].to_numpy()}
    filled = str(done.info["results"]["egemaps_means_filled"]["values"])
    assert list(rows[("x0", "label", "egemaps88")][3:]) == ["1", "1", str(refused), filled]
    for view in VIEWS:
        assert list(rows[(view, "label", "encoder")][3:]) == ["0", NA, NA, NA]
        assert list(rows[(view, "label", "encoder_minus_egemaps88")][3:]) == list(
            rows[("x0", "label", "egemaps88")][3:]
        )
        assert list(rows[(view, "label", "encoder_minus_gender_only")][3:]) == ["0", NA, NA, NA]


def test_a_participant_without_one_of_the_88_means_gets_the_training_mean_counted(
    done: fakes.Ended,
) -> None:
    # The planted participant has no voiced frame: no value for the measures taken on voiced
    # frames or weighted by the voiced stretches, so each of those means is filled.
    missing = [
        name
        for name in fakes.FUNCTIONAL_NAMES
        if weight_column(name) in ("voiced_seconds", "stretches")
    ]
    assert done.info["results"]["egemaps_means_filled"] == {
        "values": len(missing),
        "participants": 1,
    }
    table = done.table()
    egemaps = table["model"].str.endswith("egemaps88")
    assert (table.loc[egemaps, "means_filled"] == str(len(missing))).all()
    assert (table.loc[~egemaps, "means_filled"] == NA).all()
    assert f"filled with the training fold's mean: {len(missing)}, from participants: 1" in done.out


def test_two_runs_write_byte_identical_tables(workspace: ProbeWorkspace, done: fakes.Ended) -> None:
    again = run_probe(workspace, job="992")
    assert again.ended == "done", again.out
    assert (again.folder / "probe.csv").read_bytes() == (done.folder / "probe.csv").read_bytes()


# An independent computation --------------------------------------------------------------------


def raw_inputs(workspace: ProbeWorkspace) -> dict[str, Any]:
    """The segments, their participants, folds, kept vectors and pooled features, from the files."""
    clips = workspace.manifest("x0").sort_values("segment", ignore_index=True)
    round_one = workspace.partitions[workspace.partitions["round"] == 1].set_index("session")[
        "fold"
    ]
    features: dict[str, np.ndarray] = {}
    lengths: dict[str, list[int]] = {}
    for view in VIEWS:
        cache = workspace.cache_manifest(view).set_index("segment")
        folder = workspace.root / workspace.config["cache_runs"][view]["run"]
        rows = []
        for row in clips.itertuples(index=False):
            output = np.load(folder / cache.loc[row.segment, "file"]).astype(np.float64)
            onsets = 320 * np.arange(output.shape[0] * 1500)
            inside = np.zeros(len(onsets), dtype=bool)
            for start, end in zip(row.part_starts, row.part_ends, strict=True):
                inside |= (onsets >= start) & (onsets < end)
            kept = output.reshape(-1, output.shape[2])[inside]
            rows.append(np.concatenate([kept.mean(axis=0), kept.std(axis=0)]))
            lengths.setdefault(view, []).append(int(inside.sum()))
        features[view] = np.array(rows)
    return {
        "segments": list(clips["segment"]),
        "sessions": clips["session"].to_numpy(dtype=np.int64),
        "folds": round_one.reindex(clips["session"]).to_numpy(dtype=np.int64),
        "features": features,
        "lengths": np.array(lengths["x0"], dtype=np.int64),
    }


def speech_rates(workspace: ProbeWorkspace) -> dict[str, float]:
    """Each segment's speech rate from the raw word times: 40 kept words, spans of their parts."""
    run = workspace.root / workspace.config["segments"]["run"]
    words = pd.read_parquet(run / "words.parquet")
    kept = words[words["kept"]].sort_values(["session", "start", "word_index", "line"])
    rates = {}
    for session, group in kept.groupby("session"):
        for index in range(len(group) // 40):
            chunk = group.iloc[40 * index : 40 * index + 40]
            span = sum(part["end"].max() - part["start"].min() for _, part in chunk.groupby("part"))
            rates[f"{session}_{index:03d}"] = 40 / span
    return rates


@pytest.fixture(scope="module")
def raw(built: ProbeWorkspace) -> dict[str, Any]:
    """The workspace's raw inputs, read without the job's code."""
    found = raw_inputs(built)
    rates = speech_rates(built)
    found["speech_rate"] = np.array([rates[segment] for segment in found["segments"]])
    return found


@pytest.fixture(scope="module")
def draws(built: ProbeWorkspace) -> pd.DataFrame:
    """The recorded draws: each participant's count in each resample, by session."""
    labels = built.labels["label"].astype(np.int64)
    stream = np.random.default_rng(item_seed(111, "probe", "bootstrap"))
    counts, _ = draw_counts(labels, built.partitions, stream, RESAMPLES)
    return counts


def row_of(table: pd.DataFrame, view: str, outcome: str, model: str, kind: str) -> pd.Series:
    """The one row of the table with this view, outcome, model and kind."""
    found = table[
        (table["view"] == view)
        & (table["outcome"] == outcome)
        & (table["model"] == model)
        & (table["kind"] == kind)
    ]
    assert len(found) == 1
    return found.iloc[0]


def ridge_predictions(
    raw: dict[str, Any], features: np.ndarray, target: np.ndarray, row: pd.Series
) -> np.ndarray:
    """An independent ridge fit at each fold's chosen α, whose R² must be the row's fold value."""
    folds = raw["folds"]
    predictions = np.empty(len(target))
    for fold in range(1, 6):
        train, test = folds != fold, folds == fold
        scaler = StandardScaler().fit(features[train])
        model = Ridge(alpha=float(row[f"penalty_fold_{fold}"])).fit(
            scaler.transform(features[train]), target[train]
        )
        predictions[test] = model.predict(scaler.transform(features[test]))
        assert float(row[f"fold_{fold}"]) == pytest.approx(
            r2_score(target[test], predictions[test]), abs=1e-9
        )
    return predictions


def resampled_r2s(
    raw: dict[str, Any], target: np.ndarray, predictions: np.ndarray, draws: pd.DataFrame
) -> np.ndarray:
    """Each recorded draw's mean R² over the folds, against each fold's own mean.

    Each participant is repeated as often as drawn, with all their segments.
    """
    folds, sessions = raw["folds"], raw["sessions"]
    values = []
    for _, draw in draws.iterrows():
        r2s = []
        for fold in range(1, 6):
            chosen = np.concatenate(
                [
                    np.repeat(np.flatnonzero(sessions == session), int(draw[session]))
                    for session in sorted(set(sessions[folds == fold].tolist()))
                ]
            )
            r2s.append(r2_score(target[chosen], predictions[chosen]))
        values.append(np.mean(r2s))
    return np.array(values)


def label_aurocs(
    built: ProbeWorkspace, raw: dict[str, Any], view: str, row: pd.Series, draws: pd.DataFrame
) -> np.ndarray:
    """An independent logistic probe of the label on participant means, and its draws.

    The fit at each fold's chosen C must give the row's fold AUROCs. Returned: each recorded
    draw's mean AUROC over the folds, each participant repeated as often as drawn.
    """
    sessions = np.array(sorted(SESSIONS))
    features = raw["features"][view]
    means = np.array([features[raw["sessions"] == session].mean(axis=0) for session in sessions])
    labels = built.labels["label"].reindex(sessions).to_numpy(dtype=np.int64)
    folds = built.partitions[built.partitions["round"] == 1].set_index("session")["fold"]
    folds = folds.reindex(sessions).to_numpy(dtype=np.int64)
    scores = np.empty(len(sessions))
    for fold in range(1, 6):
        train, test = folds != fold, folds == fold
        scaler = StandardScaler().fit(means[train])
        model = LogisticRegression(C=float(row[f"penalty_fold_{fold}"]), max_iter=5000)
        model.fit(scaler.transform(means[train]), labels[train])
        scores[test] = model.decision_function(scaler.transform(means[test]))
        assert float(row[f"fold_{fold}"]) == pytest.approx(
            roc_auc_score(labels[test], scores[test]), abs=1e-9
        )
    values = []
    for _, draw in draws.iterrows():
        times = draw.reindex(sessions).to_numpy(dtype=np.int64)
        aurocs = [
            roc_auc_score(
                np.repeat(labels[folds == fold], times[folds == fold]),
                np.repeat(scores[folds == fold], times[folds == fold]),
            )
            for fold in range(1, 6)
        ]
        values.append(np.mean(aurocs))
    return np.array(values)


LEVELS = {80: (0.1, 0.9), 95: (0.025, 0.975)}


def assert_intervals(row: pd.Series, values: np.ndarray) -> None:
    """The row's 80% and 95% ends are the linear percentiles of the resampled values."""
    for level, ends in LEVELS.items():
        expected = np.quantile(values, ends, method="linear")
        assert float(row[f"low_{level}"]) == pytest.approx(expected[0], abs=1e-9)
        assert float(row[f"high_{level}"]) == pytest.approx(expected[1], abs=1e-9)


def test_a_ridge_row_equals_an_independent_fit_and_bootstrap_of_the_recorded_draws(
    done: fakes.Ended, raw: dict[str, Any], draws: pd.DataFrame
) -> None:
    row = row_of(done.table(), "x2", "speech_rate", "encoder", "pooled")
    predictions = ridge_predictions(raw, raw["features"]["x2"], raw["speech_rate"], row)
    assert_intervals(row, resampled_r2s(raw, raw["speech_rate"], predictions, draws))


def test_the_length_only_control_is_a_fit_on_the_kept_vectors_and_their_inverse(
    done: fakes.Ended, raw: dict[str, Any], draws: pd.DataFrame
) -> None:
    row = row_of(done.table(), NA, "speech_rate", "length_only", "pooled")
    lengths = raw["lengths"].astype(np.float64)
    assert lengths.min() > 0
    assert len(set(lengths.tolist())) > 2
    features = np.column_stack([lengths, 1.0 / lengths])
    predictions = ridge_predictions(raw, features, raw["speech_rate"], row)
    assert_intervals(row, resampled_r2s(raw, raw["speech_rate"], predictions, draws))


def test_a_label_row_equals_an_independent_fit_and_bootstrap_on_participant_means(
    built: ProbeWorkspace, done: fakes.Ended, raw: dict[str, Any], draws: pd.DataFrame
) -> None:
    row = row_of(done.table(), "x0", "label", "encoder", NA)
    assert_intervals(row, label_aurocs(built, raw, "x0", row, draws))


def test_every_difference_is_paired_on_the_same_recorded_draws(
    built: ProbeWorkspace, done: fakes.Ended, raw: dict[str, Any], draws: pd.DataFrame
) -> None:
    table = done.table()
    # A difference of AUROCs: the label probe on x2 minus the one on x0, draw by draw.
    x0 = label_aurocs(built, raw, "x0", row_of(table, "x0", "label", "encoder", NA), draws)
    x2 = label_aurocs(built, raw, "x2", row_of(table, "x2", "label", "encoder", NA), draws)
    # A difference of R²: the probe of x2's speech rate minus its length-only control.
    target = raw["speech_rate"]
    probe_row = row_of(table, "x2", "speech_rate", "encoder", "pooled")
    control_row = row_of(table, NA, "speech_rate", "length_only", "pooled")
    lengths = raw["lengths"].astype(np.float64)
    probe = resampled_r2s(
        raw, target, ridge_predictions(raw, raw["features"]["x2"], target, probe_row), draws
    )
    control = resampled_r2s(
        raw,
        target,
        ridge_predictions(raw, np.column_stack([lengths, 1.0 / lengths]), target, control_row),
        draws,
    )
    levels = [end for ends in LEVELS.values() for end in ends]
    for (view, outcome, model, kind), first, second in (
        (("x2_minus_x0", "label", "encoder", NA), x2, x0),
        (("x2", "speech_rate", "encoder_minus_length_only", "pooled"), probe, control),
    ):
        assert_intervals(row_of(table, view, outcome, model, kind), first - second)
        # The pairing shows: the differences of the sorted draws, or of the two rows' ends,
        # would give other ends.
        paired = np.quantile(first - second, levels)
        assert not np.allclose(paired, np.quantile(np.sort(first) - np.sort(second), levels))
        assert not np.allclose(paired, np.quantile(first, levels) - np.quantile(second, levels))


def test_the_table_counts_each_choice_and_a_difference_pairs_resamples_and_left_out_counts() -> (
    None
):
    module = load_script("probe_encoder")
    settings = ProbeSettings.from_config(fakes_config())
    people = SimpleNamespace(folds=np.repeat(np.arange(1, 6), 3))
    choices = [
        Choice(6, 1.0e6, True, False, 0, 0),  # a grid end, won strictly
        Choice(0, 1.0, False, True, 0, 0),  # a plateau at the other end
        Choice(3, 1.0e3, False, False, 2, 5),  # two grid values skipped, five fits failed
        Choice(6, 1.0e6, True, False, 0, 0),
        Choice(2, 1.0e2, False, False, 1, 1),
    ]
    nothing = {"segments_left_out": 0, "parts_short": NA, "parts_unmeasured": NA}
    probe = module.Row(
        "x0", "label", "encoder", NA, "auroc", [0.6, 0.7, 0.5, 0.6, 0.8],
        np.array([0.0, 1.0, 2.0, 3.0]), {**nothing, "means_filled": NA}, choices, "logistic C",
    )  # fmt: skip
    control = module.Row(
        "x0", "label", "egemaps88", NA, "auroc", [0.5, 0.6, 0.6, 0.5, 0.5],
        np.array([3.0, 0.0, 1.0, 2.0]),
        {"segments_left_out": 4, "parts_short": 1, "parts_unmeasured": 2, "means_filled": 9},
        choices, "logistic C",
    )  # fmt: skip
    both = module.difference(probe, control, "x0", "encoder_minus_egemaps88")
    # Paired: resample by resample, not the sorted resamples of each row.
    assert both.resampled.tolist() == [-3.0, 1.0, 1.0, 1.0]
    assert both.folds == pytest.approx([0.1, 0.1, -0.1, 0.1, 0.3])
    # What was left out: the larger count of the two rows, cell by cell.
    assert both.left == control.left
    assert module.difference(probe, probe, "x0", "x").left == probe.left
    table = module.table_of([probe, control, both], people, settings)
    assert table.loc[0, COUNTED].tolist() == [2, 1, 2 + 1, 5 + 1]
    assert table.loc[0, [f"penalty_fold_{fold}" for fold in range(1, 6)]].tolist() == [
        1.0e6,
        1.0,
        1.0e3,
        1.0e6,
        1.0e2,
    ]
    for level, ends in LEVELS.items():
        expected = np.quantile([-3.0, 1.0, 1.0, 1.0], ends, method="linear")
        ends_found = table.loc[2, [f"low_{level}", f"high_{level}"]].tolist()
        assert ends_found == pytest.approx(expected.tolist(), abs=1e-12)
    assert table.loc[2, "low_80"] != table.loc[0, "low_80"] - table.loc[1, "low_80"]
    assert table.loc[2, [*COUNTED, "penalty"]].tolist() == [NA] * 5
    assert table.loc[2, "segments_left_out"] == 4


# Guards and pins --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {"env": {"SLURM_JOB_ID": None}},
            "submit this as a job: scripts/slurm/submit.sh probe_encoder",
        ),
        (
            {"git": {**CHECKOUT, "dirty": True}},
            f"the checkout has uncommitted changes: the pooled-encoder probe runs only from a release {ORDER}",
        ),
        (
            {"git": {**CHECKOUT, "tags": [], "release_tags": []}},
            "no v0.1.* tag exists in the clone: the pooled-encoder probe runs only from the"
            f" release that holds its code {ORDER}",
        ),
        (
            {"git": {**CHECKOUT, "tags": ["v0.1.0"], "describe": "v0.1.0"}},
            "the checkout (v0.1.0) is not at the newest release tag v0.1.1: the pooled-encoder"
            " probe runs only from it",
        ),
        (
            {"git": {**CHECKOUT, "release_tags": ["v0.1.0", "v0.1.1", "v0.1.x"]}},
            "the tag v0.1.x matches v0.1.* but is no release",
        ),
        ({"installed": "0.1.0"}, "the environment holds version 0.1.0 of the code, not 0.1.1"),
        ({"args": ("--trial",)}, "usage: probe_encoder.py (no arguments)"),
    ],
)
def test_the_job_refuses_outside_a_clean_job_at_the_newest_release(
    workspace: ProbeWorkspace, change: dict[str, Any], message: str
) -> None:
    ended = run_probe(
        workspace,
        *change.get("args", ()),
        git=change.get("git"),
        installed=change.get("installed", fakes.INSTALLED),
        env=change.get("env"),
    )
    assert message in failure(ended)
    assert not (workspace.root / "derived" / "probe").exists()


def test_the_job_refuses_settings_that_it_does_not_compute(workspace: ProbeWorkspace) -> None:
    workspace.probe["views"] = ["x0", "x2"]
    workspace.write_probe()
    assert "probe.yaml: views must be ['x0', 'x2', 'rev']" in failure(run_probe(workspace))
    workspace.probe["views"] = ["x0", "x2", "rev"]
    text = workspace.probe_path.read_text(encoding="utf-8")
    workspace.probe_path.write_text(text + "views: [x0, x2, rev]\n", encoding="utf-8")
    assert "is given twice" in failure(run_probe(workspace))


def rewrite(path: Path, change: Any) -> None:
    """Change a parquet file in place."""
    table = pd.read_parquet(path)
    change(table).to_parquet(path, index=path.name == "labels.parquet")


PIN_CHANGES = {
    "labels": (
        lambda w: w.root / "derived" / "labels" / "run-1" / "labels.parquet",
        lambda table: table.iloc[:-1],
        "the pinned labels do not hold: labels.parquet: SHA-256",
    ),
    "partitions": (
        lambda w: w.root / "derived" / "partitions" / "run-1" / "partitions.parquet",
        lambda table: table.iloc[:-1],
        "the pinned partitions do not hold: partitions.parquet: SHA-256",
    ),
    "word times": (
        lambda w: w.root / "derived" / "segments" / "run-1" / "words.parquet",
        lambda table: table.assign(end=table["end"] + 0.001),
        "the pinned word times do not hold: words.parquet: SHA-256",
    ),
    "x0's manifest": (
        lambda w: w.root / "views" / "x0" / "run-1" / "manifest.parquet",
        lambda table: table.iloc[:-1],
        "the manifest of the pinned clips of x0 does not match its pin",
    ),
    "x2's manifest": (
        lambda w: w.root / "views" / "x2" / "run-1" / "manifest.parquet",
        lambda table: table.iloc[:-1],
        "the manifest of the pinned clips of x2 does not match its pin",
    ),
    "rev's manifest": (
        lambda w: w.root / "views" / "rev" / "run-1" / "manifest.parquet",
        lambda table: table.iloc[:-1],
        "the manifest of the pinned clips of rev does not match its pin",
    ),
    **{
        f"{view}'s cache manifest": (
            lambda w, view=view: w.root / w.config["cache_runs"][view]["run"] / "manifest.parquet",
            lambda table: table.iloc[:-1],
            f"the pinned cache run of {view}: its manifest.parquet does not match",
        )
        for view in VIEWS
    },
}


@pytest.mark.parametrize("name", list(PIN_CHANGES))
def test_the_job_refuses_a_pinned_file_that_does_not_match_before_it_reads_a_clip(
    workspace: ProbeWorkspace, name: str
) -> None:
    where, change, message = PIN_CHANGES[name]
    rewrite(where(workspace), change)
    ended = run_probe(workspace)
    assert message in failure(ended)
    assert "targets:" not in ended.out  # no clip was read
    assert not (ended.folder / "probe.csv").exists()


def test_the_job_refuses_word_times_of_other_segments_than_the_pinned_ones(
    workspace: ProbeWorkspace,
) -> None:
    workspace.config["segments"]["sha256"] = "0" * 64
    workspace.write_config()
    ended = run_probe(workspace)
    assert (
        "the pinned word times do not hold: the segments run's run.json does not list the pinned"
        " segments as done" in failure(ended)
    )
    assert "targets:" not in ended.out


def test_the_job_refuses_a_cache_without_its_check_or_from_another_voxtral(
    workspace: ProbeWorkspace, tmp_path: Path
) -> None:
    check = workspace.root / fakes.CACHE_CHECK / "run.json"
    record = check.read_text(encoding="utf-8")
    check.unlink()
    assert "the check of the pinned cache run of x0, has no run.json" in failure(
        run_probe(workspace)
    )
    check.write_text(record, encoding="utf-8")
    models = yaml.safe_load(MODELS.read_text(encoding="utf-8"))
    models["voxtral"]["revision"] = "0" * 40
    other = tmp_path / "models.yaml"
    other.write_text(yaml.safe_dump(models), encoding="utf-8")

    def another_voxtral(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(module, "MODELS", other)

    ended = run_probe(workspace, job="993", prepare=another_voxtral)
    assert "was built with another Voxtral than models.yaml pins" in failure(ended)
    assert "targets:" not in ended.out


def test_the_job_refuses_parts_that_differ_between_the_views(workspace: ProbeWorkspace) -> None:
    path = workspace.root / "views" / "rev" / "run-1" / "manifest.parquet"
    table = pd.read_parquet(path)
    table.at[3, "part_ends"] = [
        int(table.at[3, "part_ends"][0]) - 320,
        *table.at[3, "part_ends"][1:],
    ]
    table.to_parquet(path, index=False)
    workspace.config["view_runs"]["rev"]["sha256"] = fakes.sha256_of(path)
    workspace.write_config()
    cache = workspace.root / workspace.config["cache_runs"]["rev"]["run"] / "run.json"
    info = json.loads(cache.read_text(encoding="utf-8"))
    info["inputs"]["view"]["sha256"] = workspace.config["view_runs"]["rev"]["sha256"]
    cache.write_text(json.dumps(info), encoding="utf-8")
    ended = run_probe(workspace)
    assert (
        "the clips of rev lack 0 segments of x0's, hold 0 others, and 1 differ from x0's in their"
        " parts or length" in failure(ended)
    )
    assert "targets:" not in ended.out


def test_the_job_refuses_word_times_that_do_not_give_x0s_segments(
    workspace: ProbeWorkspace,
) -> None:
    run = workspace.root / workspace.config["segments"]["run"]
    words = pd.read_parquet(run / "words.parquet")
    words = words[~((words["session"] == 339) & words["kept"])].reset_index(drop=True)
    words.to_parquet(run / "words.parquet", index=False)
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    record["outputs"]["words.parquet"] = fakes.sha256_of(run / "words.parquet")
    (run / "run.json").write_text(json.dumps(record), encoding="utf-8")
    ended = run_probe(workspace)
    missing = len(fakes.segments_of(339))
    assert f"the pinned word times do not give the clips of x0: {missing} segments in only one" in (
        failure(ended)
    )


@pytest.mark.parametrize("which", ["clip", "cache"])
def test_a_clip_or_cache_file_that_does_not_match_its_row_stops_the_job_naming_its_segment(
    workspace: ProbeWorkspace, which: str
) -> None:
    segment = "317_001"
    if which == "clip":
        path = workspace.root / "views" / "x0" / "run-1" / "317" / f"{segment}.flac"
        message = f"segment {segment}: its file does not match the SHA-256 of its manifest row"
    else:
        path = (
            workspace.root / workspace.config["cache_runs"]["x2"]["run"] / "317" / f"{segment}.npy"
        )
        message = (
            f"segment {segment}: its cache file does not match the SHA-256 of its manifest row"
        )
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    ended = run_probe(workspace)
    line = failure(ended)
    assert line.startswith(f"FAILED: StepError in segment {segment}: {message}")
    assert ended.info["error"]["segment"] == segment
    assert not (ended.folder / "probe.csv").exists()
    assert_only_aggregates_leave(ended.folder / "run.json", ended.streams)


def closed_but(workspace: ProbeWorkspace, keep: Path) -> list[Path]:
    """Close every file of the workspace but one: a read of any other fails at once."""
    closed = [path for path in workspace.root.rglob("*") if path.is_file() and path != keep]
    for path in closed:
        path.chmod(0)
    return closed


def relabel(workspace: ProbeWorkspace, change: Any) -> Path:
    """New pinned labels, changed in code; the partitions are left as they are."""
    labels = workspace.labels.copy()
    changed = change(labels)
    path = workspace.root / "derived" / "labels" / "run-1" / "labels.parquet"
    (labels if changed is None else changed).to_parquet(path)
    workspace.repin("labels", path)
    return path


@pytest.mark.parametrize(
    ("case", "message"),
    [
        (
            342,
            "the pinned labels hold 1 participants outside the pilot's sessions, or a"
            " participant twice",
        ),
        (
            600,
            "the pinned labels hold 1 participants outside the pilot's sessions, or a"
            " participant twice",
        ),
        (
            "a total below 10",
            "1 labels disagree with their PHQ-8 totals: a participant is positive exactly when"
            ' the total is 10 or more (pilot analysis, section "Data protection")',
        ),
    ],
)
def test_the_job_refuses_labels_before_anything_else_is_read(
    workspace: ProbeWorkspace, case: Any, message: str
) -> None:
    if isinstance(case, int):  # 342 never existed; 600 to 718 are the external test

        def more(labels: pd.DataFrame) -> pd.DataFrame:
            row = pd.DataFrame(
                {"phq8_total": [3], "label": [0], "gender": ["female"]},
                index=pd.Index([case], name="session"),
            )
            return pd.concat([labels, row.astype(labels.dtypes)])

        path = relabel(workspace, more)
    else:
        assert workspace.labels.loc[300, "label"] == 1  # with a total of 10

        def below(labels: pd.DataFrame) -> None:
            labels.loc[300, "phq8_total"] = 9

        path = relabel(workspace, below)
    closed = closed_but(workspace, path)
    try:
        ended = run_probe(workspace)
    finally:
        for file in closed:
            file.chmod(0o600)
    line = failure(ended)
    shown, log = line.split("; traceback frames in ")
    # The whole message, so that nothing is added to it: counts only, no ID, label or total.
    assert shown == f"FAILED: StepError: {message}"
    assert log == str(ended.folder / "errors.log")
    assert not re.search(r"\b(3[0-9][0-9]|600)\b", shown)
    assert "bootstrap:" not in ended.out  # before the draws, and before anything else
    # The control: with the labels as they were, the job gets past them and stops at the
    # first other file it opens, which is closed.
    workspace.labels.to_parquet(path)
    workspace.repin("labels", path)
    closed = closed_but(workspace, path)
    try:
        control = run_probe(workspace, job="994")
    finally:
        for file in closed:
            file.chmod(0o600)
    assert failure(control).startswith("FAILED: StepError: the pinned partitions do not hold:")


def test_the_job_refuses_clips_of_a_session_outside_the_labelled_participants(
    workspace: ProbeWorkspace,
) -> None:
    path = workspace.root / "views" / "x0" / "run-1" / "manifest.parquet"
    table = pd.read_parquet(path)
    table.loc[0, "session"] = 600
    table.to_parquet(path, index=False)
    workspace.config["view_runs"]["x0"]["sha256"] = fakes.sha256_of(path)
    workspace.write_config()
    ended = run_probe(workspace)
    assert (
        "the clips of x0 hold 1 segments of sessions outside the pilot's labelled participants"
        in failure(ended)
    )
    assert "targets:" not in ended.out


def test_a_job_that_slurm_stops_is_left_failed_with_no_table(workspace: ProbeWorkspace) -> None:
    def prepare(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        def stopped(*args: Any) -> Any:  # what the handler raises
            raise SystemExit(STOPPED)

        monkeypatch.setattr(module, "pool", stopped)

    ended = run_probe(workspace, prepare=prepare)
    assert f"FAILED: SystemExit: {STOPPED}" in failure(ended)
    assert (ended.info["status"], ended.info["error"]["message"]) == ("failed", STOPPED)
    assert (ended.folder / "errors.log").is_file()
    assert not (ended.folder / "probe.csv").exists()


# What may leave OMNI ----------------------------------------------------------------------------


def outputs_of(ended: fakes.Ended) -> list[str]:
    """Everything that the job wrote or printed."""
    texts = [ended.streams]
    if ended.folder.exists():
        texts += [path.read_text(encoding="utf-8") for path in sorted(ended.folder.iterdir())]
    return texts


def plant_a_prediction(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """One participant's log-odds in the label probe on x0 becomes the planted value."""
    real = module.logistic_folds

    def planted(*args: Any) -> list[FoldFit]:
        found = real(*args)
        if args[-1] == "the label probe on x0":
            first = found[0]
            predictions = first.predictions.copy()
            predictions[0] = PLANTED_PREDICTION
            found[0] = first._replace(predictions=predictions)
        return found

    monkeypatch.setattr(module, "logistic_folds", planted)


def leaks(texts: list[str]) -> list[str]:
    """What of the planted target and prediction, as written or rounded, or a segment ID shows."""
    found = []
    for text in texts:
        for planted in (PLANTED_TARGET, PLANTED_PREDICTION):
            found += [shown for shown in (f"{planted:.3f}"[:-1], f"{planted:.3f}") if shown in text]
        found += re.findall(r"\b3\d\d_\d{3}\b", text)
    return found


@pytest.mark.parametrize("case", ["done", "a refusal"])
def test_only_aggregates_leave_the_job(
    workspace: ProbeWorkspace, done: fakes.Ended, case: str
) -> None:
    if case == "a refusal":
        path = workspace.root / workspace.config["cache_runs"]["rev"]["run"] / "manifest.parquet"
        rewrite(path, lambda table: table.iloc[:-1])
    measured: dict[str, float] = {}

    def prepare(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        plant_a_prediction(module, monkeypatch)
        # A progress line after every clip, the planted segment's included, so that a value
        # shown on that line would be found.
        monkeypatch.setattr(module, "PROGRESS_EVERY", 1)
        real = module.segment_targets

        def watched(*args: Any, **kwargs: Any) -> Any:
            found = real(*args, **kwargs)
            measured[kwargs["segment"]] = found.values[PITCH]
            return found

        monkeypatch.setattr(module, "segment_targets", watched)

    ended = run_probe(workspace, prepare=prepare)
    assert ended.ended == ("failed" if case == "a refusal" else "done"), ended.out
    assert_only_aggregates_leave(ended.folder / "run.json", ended.streams)
    assert leaks(outputs_of(ended)) == []
    if case == "done":
        # The plants reached the job: the planted target was measured, the progress line
        # printed, and the planted prediction moved the x0 label probe's first fold.
        assert measured[fakes.PLANTED_SEGMENT] == pytest.approx(PLANTED_TARGET)
        assert "targets: 10 of " in ended.out
        first = row_of(ended.table(), "x0", "label", "encoder", NA)["fold_1"]
        assert first != row_of(done.table(), "x0", "label", "encoder", NA)["fold_1"]
        table = pd.read_csv(ended.folder / "probe.csv", dtype=str, keep_default_na=False)
        assert not {"session", "segment", "pid", "segment_id"} & set(table.columns)
        for value in table.to_numpy().ravel():
            try:
                number = float(value)
            except ValueError:
                continue
            assert not (number.is_integer() and int(number) in SESSIONS)
        # The control: the same search finds a planted value or a segment ID that is shown.
        for shown in (
            f"AUROC {PLANTED_PREDICTION:.4f}",
            f"R² {PLANTED_TARGET:.3f}",
            "303_000",
            f"targets: 8 of 94 segments measured (pitch {PLANTED_TARGET:.4f})",
        ):
            assert leaks([*outputs_of(ended), shown]) != []
    else:
        assert measured == {}  # refused before any clip was read


def test_the_leak_check_fails_on_a_planted_id(done: fakes.Ended, tmp_path: Path) -> None:
    assert_only_aggregates_leave(done.folder / "run.json", done.streams)
    with pytest.raises(AssertionError):
        assert_only_aggregates_leave(done.folder / "run.json", done.streams + "participant 312\n")
    info = done.info
    info["results"]["counts"]["session"] = 312
    planted = tmp_path / "run.json"
    planted.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_only_aggregates_leave(planted, done.streams)


# The job file ------------------------------------------------------------------------------------


def test_the_job_file_asks_for_two_cpus_and_16_gb_of_the_short_partition_for_four_hours() -> None:
    text = (REPO / "scripts" / "slurm" / "probe_encoder.sbatch").read_text(encoding="utf-8")
    sbatch = [line for line in text.splitlines() if line.startswith("#SBATCH")]
    assert sbatch == [
        "#SBATCH --job-name=lbw-probe-encoder",
        "#SBATCH --partition=short",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=2",
        "#SBATCH --mem=16G",
        "#SBATCH --time=04:00:00",
    ]
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail\nsource scripts/slurm/common.sh\n" in text
    assert text.endswith('python scripts/probe_encoder.py "$@"\n')
    first_command = text.splitlines().index("set -euo pipefail")
    assert all(text.splitlines().index(line) < first_command for line in sbatch)
    assert "gres" not in text  # a CPU job
