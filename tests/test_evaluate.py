"""The analysis job (scripts/evaluate.py), end to end.

It runs on the light synthetic workspace of evaluate_fakes, with the 30 training runs, the
zero-shot scores, the labels with their PHQ-8 totals and the partitions written in code, in
a clean checkout at the newest release tag. The numbers are checked against an independent
count over all pairs of the participants' exact mean scores, and every interval and
standard error against an independent bootstrap of the recorded draws; the gating, the
trial, every version of the protocol, every guard and refusal, the tables and what may leave
OMNI are checked one by one (docs/pilot_analysis.md; pilot protocol, sections "Feasibility
checks", "Decision rule" and "If time runs short"). The listing of the runs has its own
tests (tests/test_list_runs.py).
"""

import hashlib
import itertools
import json
import math
import re
import shutil
import warnings
from dataclasses import dataclass
from fractions import Fraction
from importlib.metadata import version as package_version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import stats
from sklearn.metrics import roc_auc_score

import evaluate_fakes as fakes
from evaluate_fakes import (
    CHECKOUT,
    JOB,
    PLANTED_RUN,
    PLANTED_SESSION,
    PLANTED_Z,
    RESAMPLES,
    SESSIONS,
    EvaluateWorkspace,
    make_evaluate_workspace,
    run_job,
    run_name,
    version_runs,
)
from listening_beyond_words.evaluation import (
    ARMS,
    DIFFERENCE,
    FRAMING_EXTRA_CLIPS,
    FRAMING_NO_B1,
    FRAMING_NOT_READ,
    FRAMING_STAYS,
    GREEN_BEATS,
    VERSIONS,
    EvaluateSettings,
    Number,
    Point,
    draw_counts,
    draws_sha256,
)
from listening_beyond_words.runinfo import STOPPED, item_seed
from wordcheck_fakes import NAMED_ID, PARTICIPANT_PAIR, assert_only_aggregates_leave
from x1_fakes import REPO, load_script

TABLES = ("checks.csv", "fold_values.csv", "summary.csv", "outcome.csv")
COUNTS = ["round", "fold", "view", "positives", "negatives", "segments"]
VIEWS = ("x0", "rev")
DIFFERENCES = (("cpg", "b0"), ("b1", "b0"), ("cpg", "b1"))
FAIL_LINE = "{} FAIL: reported as a finding; the rule is not read"
SEGMENT_ID = re.compile(r"\b3\d\d_\d{3}\b")


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> EvaluateWorkspace:
    """The light workspace, with all 30 runs pinned."""
    return make_evaluate_workspace(tmp_path_factory.mktemp("built"))


@pytest.fixture
def workspace(built: EvaluateWorkspace, tmp_path: Path) -> EvaluateWorkspace:
    return built.copy_to(tmp_path)


def failure(ended: fakes.Ended) -> str:
    """The job's FAILED line."""
    lines = [line for line in ended.out.splitlines() if line.startswith("FAILED: ")]
    assert ended.ended == "failed", ended.out
    assert len(lines) == 1, ended.out
    return lines[0]


# An independent count -------------------------------------------------------------------------


def exact_means(table: pd.DataFrame, view: str) -> dict[int, Fraction]:
    """Each participant's mean z on a view, as an exact fraction, from a scores table."""
    rows = table[table["view"] == view]
    sums: dict[int, list[Fraction]] = {}
    for session, z in zip(rows["session"], rows["z"], strict=True):
        sums.setdefault(int(session), []).append(Fraction(float(z)))
    return {session: sum(values, Fraction(0)) / len(values) for session, values in sums.items()}


def pair_count(scores: dict[int, Fraction], labels: dict[int, int]) -> Fraction:
    """A fold's AUROC by counting every pair of a positive and a negative participant."""
    positives = [score for session, score in scores.items() if labels[session] == 1]
    negatives = [score for session, score in scores.items() if labels[session] == 0]
    wins = sum(
        Fraction(1) if a > b else Fraction(1, 2) if a == b else Fraction(0)
        for a in positives
        for b in negatives
    )
    return wins / (len(positives) * len(negatives))


def counted_aurocs(
    workspace: EvaluateWorkspace, runs: list[tuple[str, int, int]]
) -> dict[tuple[str, str, int, int], Fraction]:
    """Every arm's AUROC in every fold-run that it has, counted over all pairs."""
    labels = {int(session): int(label) for session, label in workspace.labels["label"].items()}
    zeroshot = pd.read_parquet(workspace.root / fakes.ZEROSHOT_RUN / "scores.parquet")
    found = {}
    for number, fold in itertools.product((1, 2), range(1, 6)):
        members = set(workspace.fold_sessions(number, fold))
        for view in VIEWS:
            means = exact_means(zeroshot, view)
            found[("zeroshot", view, number, fold)] = pair_count(
                {session: means[session] for session in members}, labels
            )
    for key in runs:
        table = workspace.scores(key)
        for view in VIEWS:
            found[(key[0], view, key[1], key[2])] = pair_count(exact_means(table, view), labels)
    return found


def mean(values: list[Fraction]) -> Fraction:
    return sum(values, Fraction(0)) / len(values)


# An independent bootstrap: the draws by plain loops, every number on the repeated rows --------


def doubled_ranks(values: list[Any], counts: np.ndarray) -> np.ndarray:
    """Twice each participant's average rank among a resample's drawn rows, of shape (R, n)."""
    levels = sorted(set(values))
    place = [levels.index(value) for value in values]
    by_level = np.zeros((len(counts), len(levels)), dtype=np.int64)
    for column, level in enumerate(place):
        by_level[:, level] += counts[:, column]
    below = np.cumsum(by_level, axis=1) - by_level
    return 2 * below[:, place] + by_level[:, place] + 1


def rank_sum_aurocs(scores: list[Any], labels: list[int], counts: np.ndarray) -> np.ndarray:
    """A fold's AUROC in each resample from the rank sum of its positive rows (Mann-Whitney)."""
    twice = doubled_ranks(scores, counts)
    positive = np.array(labels) == 1
    n_pos, n_neg = counts[:, positive].sum(axis=1), counts[:, ~positive].sum(axis=1)
    twice_u = (counts[:, positive] * twice[:, positive]).sum(axis=1) - n_pos * (n_pos + 1)
    return twice_u / (2.0 * n_pos * n_neg)


def repeated_spearman(first: list[Any], second: list[Any], counts: np.ndarray) -> np.ndarray:
    """The Spearman correlation of each resample's repeated rows; NaN where a side is flat."""
    x, y = doubled_ranks(first, counts), doubled_ranks(second, counts)
    rows = counts.sum(axis=1)
    sum_x, sum_y = (counts * x).sum(axis=1), (counts * y).sum(axis=1)
    var_x = rows * (counts * x * x).sum(axis=1) - sum_x * sum_x
    var_y = rows * (counts * y * y).sum(axis=1) - sum_y * sum_y
    cov = rows * (counts * x * y).sum(axis=1) - sum_x * sum_y
    found = np.full(len(counts), np.nan)
    flat = (var_x == 0) | (var_y == 0)
    found[~flat] = cov[~flat] / np.sqrt(var_x[~flat].astype(float) * var_y[~flat].astype(float))
    return found


@dataclass
class Recount:
    """Every number of the job's tables, computed again from the workspace's own files.

    The draws follow the page with plain loops: from item_seed(109, "evaluate",
    "bootstrap"), the positives, then the negatives, each class in ascending session ID,
    with replacement; a resample that leaves a test fold of a round without a drawn positive
    or negative is drawn again. A fold's AUROC comes from the rank sum of its positive rows
    and its Spearman correlation from the average ranks of the repeated rows, each anchored
    to scikit-learn and SciPy on the first resamples.
    """

    labels: dict[int, int]
    totals: dict[int, int]
    members: dict[tuple[int, int], list[int]]
    scores: dict[tuple[str, str, int, int], dict[int, Fraction]]
    sessions: list[int]
    counts: np.ndarray
    redraws: int

    @property
    def digest(self) -> str:
        """The draws' SHA-256: the session IDs, then every count, as little-endian int64."""
        data = np.array(self.sessions, dtype="<i8").tobytes() + self.counts.astype("<i8").tobytes()
        return hashlib.sha256(data).hexdigest()

    def has(self, name: str, view: str, key: tuple[int, int]) -> bool:
        """Whether the job read the arms that a number needs in a fold-run."""
        return all((arm, view, *key) in self.scores for arm in arms_of(name))

    def keys(self, name: str, view: str) -> list[tuple[int, int]]:
        return [key for key in sorted(self.members) if self.has(name, view, key)]

    def ordered(self, arm: str, view: str, key: tuple[int, int]) -> list[Fraction]:
        found = self.scores[(arm, view, *key)]
        return [found[session] for session in self.members[key]]

    def point(self, name: str, view: str, key: tuple[int, int]) -> Fraction | float | None:
        """A number's value in a fold-run: exact for an AUROC, SciPy's for a Spearman."""
        members = self.members[key]
        if name.startswith(("auroc_", "spearman_")):
            kind, arm = name.split("_", 1)
            if kind == "auroc":
                return pair_count(self.scores[(arm, view, *key)], self.labels)
            other = [self.totals[session] for session in members]
            return scipy_spearman(self.ordered(arm, view, key), other)
        if name.startswith("agreement_"):
            return scipy_spearman(self.ordered("b0", view, key), self.ordered("cpg", view, key))
        first, second = name.split("_minus_")
        auroc = [pair_count(self.scores[(arm, view, *key)], self.labels) for arm in (first, second)]
        return auroc[0] - auroc[1]

    def resampled(self, name: str, view: str, key: tuple[int, int]) -> np.ndarray:
        """A number's value in a fold-run on every resample, from the counts drawn."""
        members = self.members[key]
        counts = self.counts[:, [self.sessions.index(session) for session in members]]
        labels = [self.labels[session] for session in members]
        if name.startswith("auroc_"):
            return rank_sum_aurocs(self.ordered(name[6:], view, key), labels, counts)
        if name.startswith("spearman_"):
            totals = [self.totals[session] for session in members]
            return repeated_spearman(self.ordered(name[9:], view, key), totals, counts)
        if name.startswith("agreement_"):
            pair = (self.ordered("b0", view, key), self.ordered("cpg", view, key))
            return repeated_spearman(*pair, counts)
        first, second = name.split("_minus_")
        found = [
            rank_sum_aurocs(self.ordered(arm, view, key), labels, counts) for arm in (first, second)
        ]
        return found[0] - found[1]

    def anchor(self, rows: int = 3) -> None:
        """Check the rank sums and repeated ranks against the libraries on some resamples."""
        for key, members in self.members.items():
            for view in VIEWS:
                for row in range(rows):
                    times = [int(self.counts[row, self.sessions.index(s)]) for s in members]
                    labels = np.repeat([self.labels[s] for s in members], times)
                    totals = np.repeat([self.totals[s] for s in members], times)
                    for arm in ARMS:
                        if (arm, view, *key) not in self.scores:
                            continue
                        scores = np.repeat([float(z) for z in self.ordered(arm, view, key)], times)
                        auroc = self.resampled(f"auroc_{arm}", view, key)[row]
                        assert auroc == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)
                        rho = self.resampled(f"spearman_{arm}", view, key)[row]
                        assert same_or_undefined(rho, scipy_spearman(scores, totals))


def arms_of(name: str) -> tuple[str, ...]:
    """The arms that a number of the tables reads, by its name."""
    if name.startswith(("auroc_", "spearman_")):
        return (name.split("_", 1)[1],)
    if name.startswith("agreement_"):
        return ("b0", "cpg")
    first, second = name.split("_minus_")
    return (first, second)


def scipy_spearman(first: Any, second: Any) -> float | None:
    """SciPy's Spearman correlation of two lists of values; None where it is not defined."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", stats.ConstantInputWarning)
        found = stats.spearmanr([float(v) for v in first], [float(v) for v in second])
    return None if math.isnan(float(found.statistic)) else float(found.statistic)


def same_or_undefined(found: float, expected: float | None) -> bool:
    if expected is None:
        return math.isnan(found)
    return abs(found - expected) <= 1e-12


def recount(workspace: EvaluateWorkspace, runs: list[tuple[str, int, int]]) -> Recount:
    """The independent count of a job that read the zero-shot scores and these runs."""
    labels = {int(session): int(label) for session, label in workspace.labels["label"].items()}
    totals = {int(s): int(total) for s, total in workspace.labels["phq8_total"].items()}
    members = {
        (number, fold): workspace.fold_sessions(number, fold)
        for number, fold in itertools.product((1, 2), range(1, 6))
    }
    zeroshot = pd.read_parquet(workspace.root / fakes.ZEROSHOT_RUN / "scores.parquet")
    scores: dict[tuple[str, str, int, int], dict[int, Fraction]] = {}
    for view in VIEWS:
        means = exact_means(zeroshot, view)
        for key, sessions in members.items():
            scores[("zeroshot", view, *key)] = {session: means[session] for session in sessions}
    for arm, number, fold in runs:
        table = workspace.scores((arm, number, fold))
        for view in VIEWS:
            scores[(arm, view, number, fold)] = exact_means(table, view)
    sessions = sorted(labels)
    fold_of = {
        (int(row.session), int(row.round)): int(row.fold)
        for row in workspace.partitions.itertuples()
    }
    groups = [[session for session in sessions if labels[session] == value] for value in (1, 0)]
    rng = np.random.default_rng(item_seed(109, "evaluate", "bootstrap"))
    counts = np.zeros((RESAMPLES, len(sessions)), dtype=np.int64)
    kept = redraws = 0
    while kept < RESAMPLES:
        drawn = [
            [group[index] for index in rng.choice(len(group), size=len(group), replace=True)]
            for group in groups
        ]
        if not all(
            {fold_of[(session, number)] for session in group} == set(range(1, 6))
            for group in drawn
            for number in (1, 2)
        ):
            redraws += 1
            continue
        for session in drawn[0] + drawn[1]:
            counts[kept, sessions.index(session)] += 1
        kept += 1
    return Recount(labels, totals, members, scores, sessions, counts, redraws)


def protocol_decisions() -> dict[str, str]:
    """The outcomes and the decision for each, from the protocol's table, word for word."""
    protocol = (REPO / "docs" / "pilot_protocol.md").read_text(encoding="utf-8")
    table = protocol.split("## Decision rule", 1)[1].split("## ", 1)[0]
    rows = [line for line in table.splitlines() if line.startswith("| ") and "---" not in line]
    return {
        cells[0]: cells[2]
        for cells in ([cell.strip() for cell in row.strip("|").split("|")] for row in rows[1:])
    }


def read_by_rule(delta_x0: Fraction, delta_rev: Fraction) -> str:
    """The protocol's rule, written out again: +0.03 and -0.04 on x0, +0.03 on rev."""
    if delta_x0 >= Fraction(3, 100):
        return "Green: beats B0"
    if delta_x0 <= Fraction(-4, 100):
        return "Red: worse on normal audio"
    if delta_rev >= Fraction(3, 100):
        return "Green: robust without words"
    return "Amber: no visible effect"


# The analysis of all 30 runs --------------------------------------------------------------------


def test_the_analysis_reads_all_30_runs_and_writes_four_tables_and_its_record(
    workspace: EvaluateWorkspace,
) -> None:
    ended = run_job(workspace)
    assert ended.ended == "done", ended.out
    assert ended.folder == workspace.root / "derived" / "evaluate" / f"run-{JOB}"
    assert sorted(path.name for path in ended.folder.iterdir()) == sorted([*TABLES, "run.json"])
    info = ended.info
    assert (info["step"], info["mode"], info["status"], info["run_id"]) == (
        "evaluate",
        "full",
        "done",
        f"run-{JOB}",
    )
    assert (info["release"], info["installed_version"], info["git"]) == (
        "v0.1.1",
        "0.1.1",
        CHECKOUT,
    )
    for name in TABLES:
        data = (ended.folder / name).read_bytes()
        assert info["outputs"][name] == hashlib.sha256(data).hexdigest()
    assert (
        info["config"]["sha256"] == hashlib.sha256(workspace.evaluate_path.read_bytes()).hexdigest()
    )
    assert (
        info["data_config"]["sha256"]
        == hashlib.sha256(workspace.config_path.read_bytes()).hexdigest()
    )
    assert info["config"]["content"] == yaml.safe_load(workspace.evaluate_path.read_text("utf-8"))
    assert set(info["environment"]["packages"]) >= {
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "pyarrow",
    }
    assert info["slurm"]["job_id"] == JOB
    assert info["results"]["version"] == {"name": "all 30 runs", "runs": 30}
    # The draws need only the labels and the partitions: the same draws again, from seed 109.
    seeds = info["seeds"]["bootstrap"]
    rng = np.random.default_rng(item_seed(109, "evaluate", "bootstrap"))
    counts, redraws = draw_counts(workspace.labels["label"], workspace.partitions, rng, RESAMPLES)
    assert seeds == {
        "seed": 109,
        "stream": 'runinfo.item_seed(109, "evaluate", "bootstrap")',
        "item_seed": item_seed(109, "evaluate", "bootstrap"),
        "resamples": RESAMPLES,
        "redraws": redraws,
        "draws_sha256": draws_sha256(counts),
    }
    # Each pin with its commit and tags, and what lets the run count.
    pins = info["inputs"]["runs"]
    assert sum(len(folds) for rounds in pins.values() for folds in rounds.values()) == 30
    first = pins["cpg"]["r1"]["f1"]
    assert (first["commit"], first["tags"]) == (fakes.FIRST, [])
    assert first["counts_by"] == "the first runs' commit 271cce6"
    later = pins["b1"]["r2"]["f5"]
    assert (later["commit"], later["tags"]) == (fakes.RELEASED, ["v0.1.0"])
    assert later["counts_by"] == "the newest release tag of its clone, v0.1.0"
    assert info["inputs"]["labels"] == workspace.config["labels"]
    assert info["inputs"]["zeroshot_scores"] == workspace.config["zeroshot_scores"]
    checks = info["results"]["checks"]
    assert checks["f1"] == {"kept_runs": 30, "passes": True, "verdict": "PASS"}
    assert checks["f2"]["pooled"] == {"active": 160, "bags": 320, "share": 0.5}
    assert checks["f2"]["verdict"] == "PASS"
    assert (checks["f4"]["run"], checks["f4"]["verdict"]) == (fakes.WORDCHECK_RUN, "PASS")
    assert checks["f5"] == {
        "run": run_name("cpg", 1, 1),
        "read_by": "reads_rules",
        "hours": fakes.TRAINING_HOURS,
        "max_hours": 4.0,
        "passes": True,
        "verdict": "PASS",
    }
    assert info["results"]["read"] == {
        "zeroshot": {"runs": 1, "rounds": [1, 2]},
        "b0": {"runs": 10, "rounds": [1, 2]},
        "b1": {"runs": 10, "rounds": [1, 2]},
        "cpg": {"runs": 10, "rounds": [1, 2]},
    }
    assert info["results"]["counts"]["participants"] == len(SESSIONS)
    assert "done in" in ended.out


def test_every_number_matches_an_independent_count_over_all_pairs(
    workspace: EvaluateWorkspace,
) -> None:
    ended = run_job(workspace)
    assert ended.ended == "done", ended.out
    found = counted_aurocs(workspace, version_runs("all 30 runs"))
    folds = ended.table("fold_values.csv")
    assert len(folds) == 20
    assert list(folds.columns) == [
        *COUNTS,
        *(f"auroc_{arm}" for arm in ARMS),
        *(f"spearman_{arm}" for arm in ARMS),
        "cpg_minus_b0",
        "b1_minus_b0",
        "cpg_minus_b1",
        "agreement_b0_cpg",
    ]
    for row in folds.itertuples(index=False):
        members = workspace.fold_sessions(row.round, row.fold)
        positives = int(workspace.labels.loc[members, "label"].sum())
        assert (row.positives, row.negatives) == (positives, len(members) - positives)
        segments = sum(len(fakes.segments_of(session)) for session in members)
        assert row.segments == segments
        for arm in ARMS:
            assert getattr(row, f"auroc_{arm}") == float(
                found[(arm, row.view, row.round, row.fold)]
            )
        for first, second in DIFFERENCES:
            difference = (
                found[(first, row.view, row.round, row.fold)]
                - found[(second, row.view, row.round, row.fold)]
            )
            assert getattr(row, f"{first}_minus_{second}") == float(difference)
    summary = ended.table("summary.csv").set_index(["number", "view", "rounds"])
    for arm, view in itertools.product(ARMS, VIEWS):
        values = [found[(arm, view, number, fold)] for number in (1, 2) for fold in range(1, 6)]
        row = summary.loc[(f"auroc_{arm}", view, "1 and 2")]
        assert row["fold_runs"] == 10
        assert row["mean"] == float(mean(values))
        assert row["round_1"] == float(mean(values[:5]))
        assert row["round_2"] == float(mean(values[5:]))
        assert (
            float(row["low_95"])
            <= float(row["low_80"])
            <= float(row["high_80"])
            <= float(row["high_95"])
        )
    deltas = {
        view: mean(
            [
                found[("cpg", view, number, fold)] - found[("b0", view, number, fold)]
                for number in (1, 2)
                for fold in range(1, 6)
            ]
        )
        for view in VIEWS
    }
    outcome = ended.table("outcome.csv")
    assert len(outcome) == 1
    row = outcome.iloc[0]
    assert Fraction(row["delta_x0_exact"]) == deltas["x0"]
    assert Fraction(row["delta_rev_exact"]) == deltas["rev"]
    assert row["delta_x0"] == float(deltas["x0"])
    assert row["delta_rev"] == float(deltas["rev"])
    assert row["fold_runs"] == 10
    # The two values that the framing line compares, exactly, beside it in run.json.
    framed = ended.info["results"]["outcome"]["framing_values"]
    assert framed["view"] == "x0"
    for (first, second), bound in (((("b1", "b0")), "3/100"), ((("cpg", "b1")), "1/50")):
        differences = [
            found[(first, "x0", number, fold)] - found[(second, "x0", number, fold)]
            for number in (1, 2)
            for fold in range(1, 6)
        ]
        value = framed[f"{first}_minus_{second}"]
        assert Fraction(value["exact"]) == mean(differences)
        assert (value["bound"]["exact"], value["fold_runs"]) == (bound, 10)
    expected = read_by_rule(deltas["x0"], deltas["rev"])
    assert row["outcome"] == expected
    assert row["decision"] == protocol_decisions()[expected]
    assert f"outcome: {expected}\n" in ended.out
    assert f"decision: {protocol_decisions()[expected]}\n" in ended.out
    assert f"{row['framing']}\n" in ended.out
    # Δ's standard error is the SD (ddof 1) of its resampled values; its row says so.
    delta = summary.loc[("cpg_minus_b0", "x0", "1 and 2")]
    assert float(delta["se"]) > 0
    assert summary.loc[("auroc_b0", "x0", "1 and 2")]["se"] == "not computed: differences only"
    # F3: B0's mean AUROC on x0, exact, with its per-round values.
    b0 = [found[("b0", "x0", number, fold)] for number in (1, 2) for fold in range(1, 6)]
    f3 = ended.info["results"]["checks"]["f3"]
    assert Fraction(f3["value"]["exact"]) == mean(b0)
    assert {key: Fraction(value["exact"]) for key, value in f3["rounds"].items()} == {
        "1": mean(b0[:5]),
        "2": mean(b0[5:]),
    }
    assert f3["verdict"] == ("PASS" if mean(b0) >= Fraction(3, 5) else "FAIL") == "PASS"


def test_the_noise_figures_give_deltas_se_and_the_difference_of_its_rounds(
    workspace: EvaluateWorkspace,
) -> None:
    ended = run_job(workspace)
    found = counted_aurocs(workspace, version_runs("all 30 runs"))
    noise = ended.info["results"]["noise"]
    summary = ended.table("summary.csv").set_index(["number", "view", "rounds"])
    for view in VIEWS:
        differences = [
            found[("cpg", view, number, fold)] - found[("b0", view, number, fold)]
            for number in (1, 2)
            for fold in range(1, 6)
        ]
        gap = mean(differences[:5]) - mean(differences[5:])
        entry = noise[f"delta_{view}"]
        assert Fraction(entry["round_1_minus_round_2"]["exact"]) == gap
        assert Fraction(entry["absolute"]["exact"]) == abs(gap)
        assert entry["se"] == float(summary.loc[("cpg_minus_b0", view, "1 and 2")]["se"])
        agreement = summary.loc[("agreement_b0_cpg", view, "1 and 2")]
        assert noise[f"agreement_{view}"]["mean"] == float(agreement["mean"])
    assert (
        noise["b0_auroc_x0"]["mean_exact"]
        == ended.info["results"]["checks"]["f3"]["value"]["exact"]
    )
    assert "noise on x0: Δ's bootstrap SE " in ended.out


def test_two_runs_write_byte_identical_tables(workspace: EvaluateWorkspace) -> None:
    first = run_job(workspace, job="991")
    second = run_job(workspace, job="992")
    assert first.ended == second.ended == "done"
    for name in TABLES:
        assert (first.folder / name).read_bytes() == (second.folder / name).read_bytes()
    assert first.info["seeds"] == second.info["seeds"]


def absent_cell(value: Any) -> bool:
    """A cell of a value that the job could not have: the text of a cut or of the trial."""
    return isinstance(value, str) and value.startswith(("not available (cut: ", "not read (trial"))


@pytest.mark.parametrize("version", ["all 30 runs", "no B1 round 2", "trial"])
def test_every_cell_matches_an_independent_bootstrap_of_the_recorded_draws(
    workspace: EvaluateWorkspace, version: str
) -> None:
    trial = version == "trial"
    runs = version_runs("all 30 runs" if trial else version)
    workspace.pin(runs)
    ended = run_job(workspace, *(("--trial",) if trial else ()))
    assert ended.ended == "done", ended.out
    if trial:  # the trial reads B0's round 1 only, besides the zero-shot scores
        runs = [("b0", 1, fold) for fold in range(1, 6)]
    found = recount(workspace, runs)
    found.anchor()
    # The draws of the recorded stream, drawn again with plain loops.
    seeds = ended.info["seeds"]["bootstrap"]
    assert (seeds["draws_sha256"], seeds["redraws"]) == (found.digest, found.redraws)
    assert found.redraws > 0  # the redraws are part of what is checked
    # Each fold-run's point values: Spearman against the PHQ-8 totals, as SciPy computes it.
    folds = ended.table("fold_values.csv")
    names = [name for name in folds.columns if name not in COUNTS]
    for row in folds.to_dict("records"):
        key = (int(row["round"]), int(row["fold"]))
        for name in names:
            if not found.has(name, row["view"], key):
                assert absent_cell(row[name]), (name, key)
                continue
            expected = found.point(name, row["view"], key)
            if expected is None:
                assert row[name] == "not defined"
            elif isinstance(expected, Fraction):  # a cut leaves text in the column
                assert float(row[name]) == float(expected), (name, key)
            else:
                assert float(row[name]) == pytest.approx(expected, abs=1e-12), (name, key)
    # Every mean, interval end and standard error, from the recorded draws.
    summary = ended.table("summary.csv")
    checked = 0
    for row in summary.to_dict("records"):
        if row["rounds"] == "none":
            continue
        rounds = [int(number) for number in str(row["rounds"]).split(" and ")]
        keys = [key for key in found.keys(row["number"], row["view"]) if key[0] in rounds]
        assert row["fold_runs"] == len(keys)
        # The mean over the row's fold-runs, and each round's mean over the number's own.
        every = found.keys(row["number"], row["view"])
        for cell, chosen in (
            (row["mean"], keys),
            *((row[f"round_{n}"], [key for key in every if key[0] == n]) for n in (1, 2)),
        ):
            points = [found.point(row["number"], row["view"], key) for key in chosen]
            if not points:
                assert absent_cell(cell)
            elif None in points:
                assert str(cell).startswith("not defined (")
            else:
                expected = math.fsum(map(float, points)) / len(points)
                assert float(cell) == pytest.approx(expected, abs=1e-12)
        resampled = [found.resampled(row["number"], row["view"], key) for key in keys]
        undefined = [
            key for key, values in zip(keys, resampled, strict=True) if np.isnan(values).any()
        ]
        if undefined:  # a Spearman correlation not defined in some resample: no interval
            text = "not defined (" + "; ".join(f"round {r}, fold {f}" for r, f in undefined) + ")"
            ends = [row[f"{end}_{level}"] for level in (80, 95) for end in ("low", "high")]
            assert ends == [text] * 4
            assert row["se"] == "not computed: differences only"
            checked += 1
            continue
        values = np.column_stack(resampled).mean(axis=1)
        for level, percentiles in ((80, (10, 90)), (95, (2.5, 97.5))):
            low, high = np.percentile(values, percentiles, method="linear")
            assert float(row[f"low_{level}"]) == pytest.approx(low, abs=1e-12), row
            assert float(row[f"high_{level}"]) == pytest.approx(high, abs=1e-12), row
        if "_minus_" in row["number"]:
            assert float(row["se"]) == pytest.approx(float(np.std(values, ddof=1)), abs=1e-12)
        else:
            assert row["se"] == "not computed: differences only"
        checked += 1
    assert checked == {"all 30 runs": 24, "no B1 round 2": 26, "trial": 8}[version]


def test_the_framings_two_values_print_unrounded_where_they_round_to_their_bounds(
    capsys: pytest.CaptureFixture,
) -> None:
    # Pilot analysis, section "Decision rule": a value that rounds to a bound is also printed
    # unrounded. B1 - B0 is read against +0.03 and CPG - B1 against +0.02, on x0 only; Δ
    # against both bounds of the rule on x0 and against the green one on rev.
    module = load_script("evaluate")
    config = yaml.safe_load((REPO / "configs" / "pilot" / "evaluate.yaml").read_text("utf-8"))
    job = SimpleNamespace(settings=EvaluateSettings.from_config(config))
    cells = {"low_80": 0.0, "high_80": 0.1, "low_95": -0.1, "high_95": 0.2, "se": 0.01}
    near = {
        ("b1", "b0"): Fraction(749, 25000),  # 0.02996, just below +0.03
        ("cpg", "b1"): Fraction(4999, 250000),  # 0.019996, just below +0.02
        ("cpg", "b0"): Fraction(-999, 25000),  # -0.03996, just above -0.04
    }
    for pair, value in near.items():
        for view in VIEWS:
            number = Number(DIFFERENCE, pair, view)
            point = Point({(n, fold): value for n in (1, 2) for fold in range(1, 6)})
            module.print_rows(job, [module.Row(number, (1, 2), cells)], {number: point})
    out = capsys.readouterr().out
    assert "B1 − B0 on x0: 0.0300 (unrounded 0.02996 = 749/25000) (80% " in out
    assert "round 1 0.0300 (unrounded 0.02996 = 749/25000)" in out
    assert "CPG − B1 on x0: 0.0200 (unrounded 0.019996 = 4999/250000) (80% " in out
    assert "Δ = CPG − B0 on x0: -0.0400 (unrounded -0.03996 = -999/25000) (80% " in out
    # On rev only Δ's green bound is read; -0.04 is no bound there, and the framing's are not.
    assert "B1 − B0 on rev: 0.0300 (80% " in out
    assert "CPG − B1 on rev: 0.0200 (80% " in out
    assert "Δ = CPG − B0 on rev: -0.0400 (80% " in out


# Versions, gating and the trial ---------------------------------------------------------------


@pytest.mark.parametrize("version", [version.name for version in VERSIONS])
def test_every_version_runs_and_its_tables_say_what_was_cut(
    workspace: EvaluateWorkspace, version: str
) -> None:
    runs = version_runs(version)
    workspace.pin(runs)
    ended = run_job(workspace)
    assert ended.ended == "done", ended.out
    info = ended.info
    (cut,) = [item for item in VERSIONS if item.name == version]
    assert info["results"]["version"] == {"name": version, "runs": len(runs)}
    absent = f"not available (cut: {version})"
    folds = ended.table("fold_values.csv")
    assert list(folds.columns)[: len(COUNTS) + 4] == [*COUNTS, *(f"auroc_{arm}" for arm in ARMS)]
    for row in folds.itertuples(index=False):

        def has(arm: str, number: int = row.round) -> bool:
            return arm == "zeroshot" or (arm, number) in cut.arm_rounds

        for arm in ARMS:
            value = getattr(row, f"auroc_{arm}")
            if has(arm):
                assert value != absent
                assert 0 <= float(value) <= 1, (arm, row.round, value)
            else:
                assert value == absent
                assert getattr(row, f"spearman_{arm}") == absent
        for first, second in DIFFERENCES:
            value = getattr(row, f"{first}_minus_{second}")
            assert (value == absent) == (not (has(first) and has(second)))
        assert (row.agreement_b0_cpg == absent) == (not (has("b0") and has("cpg")))
    summary = ended.table("summary.csv")
    cells = summary.drop(columns=["number", "view", "rounds", "fold_runs"]).to_numpy().ravel()
    assert "" not in set(map(str, cells))  # no cell is blank
    rows = summary.set_index(["number", "view", "rounds"])
    zeroshot = rows.loc[("auroc_zeroshot", "x0", "1 and 2")]
    assert zeroshot["fold_runs"] == 10
    assert zeroshot["round_1"] != absent  # zero-shot's round-1 mean beside its 10-fold mean
    b1 = summary[(summary["number"] == "auroc_b1") & (summary["view"] == "x0")]
    if ("b1", 1) not in cut.arm_rounds:
        assert b1["rounds"].tolist() == ["none"]
        assert set(b1.iloc[0][["mean", "round_1", "round_2", "low_80", "high_95", "se"]]) == {
            absent
        }
        assert f"B1: no run of it is pinned in this version ({version})\n" in ended.out
        assert info["results"]["read"]["b1"] == {"runs": 0, "rounds": []}
    elif ("b1", 2) not in cut.arm_rounds:
        # Without B1's round 2, B1's numbers use round 1, with CPG - B0 on the same fold-runs.
        assert b1["rounds"].tolist() == ["1"]
        assert b1.iloc[0]["fold_runs"] == 5
        assert b1.iloc[0]["round_2"] == absent
        beside = summary[summary["number"] == "cpg_minus_b0"]
        assert beside[["view", "rounds"]].values.tolist() == [
            ["x0", "1 and 2"],
            ["rev", "1 and 2"],
            ["x0", "1"],
            ["rev", "1"],
        ]
        assert beside.iloc[2]["mean"] == beside.iloc[0]["round_1"]
    outcome = ended.table("outcome.csv").iloc[0]
    rounds = sorted({number for arm, number in cut.arm_rounds if arm == "cpg"})
    assert outcome["fold_runs"] == 5 * len(rounds)
    found = counted_aurocs(workspace, runs)
    delta = mean(
        [
            found[("cpg", "x0", number, fold)] - found[("b0", "x0", number, fold)]
            for number in rounds
            for fold in range(1, 6)
        ]
    )
    assert Fraction(outcome["delta_x0_exact"]) == delta
    assert outcome["delta_x0"] == float(delta)
    delta_rev = mean(
        [
            found[("cpg", "rev", number, fold)] - found[("b0", "rev", number, fold)]
            for number in rounds
            for fold in range(1, 6)
        ]
    )
    assert (Fraction(outcome["delta_rev_exact"]), outcome["delta_rev"]) == (
        delta_rev,
        float(delta_rev),
    )
    if ("b1", 1) not in cut.arm_rounds:
        expected = FRAMING_NO_B1 if outcome["outcome"] == GREEN_BEATS else FRAMING_NOT_READ
        assert outcome["framing"] == expected
    # The difference between the two rounds' Δ exists only where Δ has both rounds.
    gap = info["results"]["noise"]["delta_x0"]["round_1_minus_round_2"]
    if rounds == [1]:
        assert gap == absent
    else:
        by_round = [
            mean(
                [
                    found[("cpg", "x0", number, f)] - found[("b0", "x0", number, f)]
                    for f in range(1, 6)
                ]
            )
            for number in (1, 2)
        ]
        assert Fraction(gap["exact"]) == by_round[0] - by_round[1]


@pytest.mark.parametrize(
    ("perfect", "version", "expected"),
    [
        (("cpg",), "all 30 runs", FRAMING_STAYS),  # B1 - B0 is below +0.03
        (("b1", "cpg"), "all 30 runs", FRAMING_EXTRA_CLIPS),
        (("b1", "cpg"), "no B1 round 2", FRAMING_STAYS),  # B1's numbers: round 1 only
        (("cpg",), "no B1", FRAMING_NO_B1),
    ],
)
def test_the_framing_line_is_read_in_green_beats_b0_from_b1s_differences(
    workspace: EvaluateWorkspace, perfect: tuple[str, ...], version: str, expected: str
) -> None:
    # The scores of these arms follow the label exactly, so CPG's Δ on x0 reaches +0.03
    # ("Green: beats B0"), and B1's gain on B0 depends on the fold-runs that B1 has.
    runs = version_runs(version)
    labels = workspace.labels["label"]
    for key in runs:
        if key[0] in perfect:
            table = workspace.scores(key)
            table["z"] = [float(labels[session]) for session in table["session"]]
            workspace.write_scores(key, table)
    workspace.pin(runs)
    ended = run_job(workspace)
    assert ended.ended == "done", ended.out
    outcome = ended.table("outcome.csv").iloc[0]
    assert outcome["outcome"] == GREEN_BEATS
    assert outcome["framing"] == expected
    assert f"{expected}\n" in ended.out
    b1 = [(number, fold) for arm, number, fold in runs if arm == "b1"]
    if b1:  # the job's reading, counted again over B1's fold-runs
        found = counted_aurocs(workspace, runs)
        b1_gain = mean([found[("b1", "x0", *key)] - found[("b0", "x0", *key)] for key in b1])
        cpg_over_b1 = mean([found[("cpg", "x0", *key)] - found[("b1", "x0", *key)] for key in b1])
        clips = b1_gain >= Fraction(3, 100) and cpg_over_b1 < Fraction(2, 100)
        assert expected == (FRAMING_EXTRA_CLIPS if clips else FRAMING_STAYS)


def no_b1_or_cpg(calls: list[str]) -> Any:
    """A stand-in for the scores reader that refuses B1's and CPG's runs and logs the others."""

    def prepare(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        real = module.read_scores

        def read(workspace: Any, run: Any, *args: Any) -> Any:
            calls.append(run.run)
            if run.arm in ("b1", "cpg"):
                raise AssertionError(f"the job opened {run.run}")
            return real(workspace, run, *args)

        monkeypatch.setattr(module, "read_scores", read)

    return prepare


def fail_f2(workspace: EvaluateWorkspace) -> None:
    """One CPG run whose hinge was never active in epochs 2 and 3: its share is 0."""
    key = ("cpg", 2, 4)
    workspace.change_record(
        key, lambda record: record["results"].update(fakes.rule_record(*key, active=False))
    )


def fail_f3(workspace: EvaluateWorkspace) -> None:
    """B0's scores lean to the wrong label, so its mean AUROC on x0 lies far below 0.60."""
    for number, fold in itertools.product((1, 2), range(1, 6)):
        key = ("b0", number, fold)
        table = workspace.scores(key)
        labels = workspace.labels["label"]
        table["z"] = [-float(labels[session]) for session in table["session"]]
        workspace.write_scores(key, table)


@pytest.mark.parametrize("check", ["F2", "F3"])
def test_a_failed_f2_or_f3_writes_the_checks_and_the_zero_shot_and_b0_rows_then_stops(
    workspace: EvaluateWorkspace, check: str
) -> None:
    (fail_f2 if check == "F2" else fail_f3)(workspace)
    calls: list[str] = []
    ended = run_job(workspace, prepare=no_b1_or_cpg(calls))
    assert ended.ended == "done", ended.out
    assert calls == [run_name("b0", number, fold) for number in (1, 2) for fold in range(1, 6)]
    assert ended.out.splitlines()[-3] == FAIL_LINE.format(check)
    assert sorted(path.name for path in ended.folder.iterdir()) == [
        "checks.csv",
        "fold_values.csv",
        "run.json",
        "summary.csv",
    ]
    folds = ended.table("fold_values.csv")
    assert list(folds.columns) == [
        *COUNTS,
        "auroc_zeroshot",
        "auroc_b0",
        "spearman_zeroshot",
        "spearman_b0",
    ]
    summary = ended.table("summary.csv")
    assert set(summary["number"]) == {
        "auroc_zeroshot",
        "auroc_b0",
        "spearman_zeroshot",
        "spearman_b0",
    }
    checks = ended.table("checks.csv").set_index(["check", "scope"])
    assert checks.loc[("F3", "B0's mean AUROC on x0 over its fold-runs"), "verdict"] == (
        "FAIL" if check == "F3" else "PASS"
    )
    f2 = checks.loc["F2", "verdict"]
    assert f2["the 10 kept CPG runs"] == ("FAIL" if check == "F2" else "PASS")
    assert f2[run_name("cpg", 2, 4)] == ("FAIL" if check == "F2" else "PASS")
    assert f2["pooled over the 10 kept CPG runs"] == "reported only"
    # No Δ, no other difference and no outcome exist anywhere.
    texts = [ended.streams, (ended.folder / "run.json").read_text("utf-8")]
    texts += [
        (ended.folder / name).read_text("utf-8") for name in ("fold_values.csv", "summary.csv")
    ]
    for text in texts:
        assert "minus" not in text
        assert "Δ" not in text
        assert "outcome:" not in text
    results = ended.info["results"]
    assert "noise" not in results
    assert results["outcome"] == f"not read: check {check} failed"
    assert set(results["read"]) == {"zeroshot", "b0"}


def test_the_trial_opens_no_scores_but_the_zero_shot_and_b0_round_1(
    workspace: EvaluateWorkspace,
) -> None:
    calls: list[str] = []
    ended = run_job(workspace, "--trial", prepare=no_b1_or_cpg(calls))
    assert ended.ended == "done", ended.out
    assert calls == [run_name("b0", 1, fold) for fold in range(1, 6)]
    assert ended.folder == workspace.root / "derived" / "evaluate" / f"trial-{JOB}"
    assert sorted(path.name for path in ended.folder.iterdir()) == [
        "fold_values.csv",
        "run.json",
        "summary.csv",
    ]
    info = ended.info
    assert (info["mode"], info["status"]) == ("trial", "done")
    assert set(info["results"]["checks"]) == {"f1", "f2", "f4", "f5"}  # no F3 and no rule
    assert info["results"]["outcome"] == "not read: the trial computes no difference and no rule"
    assert info["results"]["read"]["b0"] == {"runs": 5, "rounds": [1]}
    summary = ended.table("summary.csv").set_index(["number", "view"])
    assert set(summary.index.get_level_values(0)) == {
        "auroc_zeroshot",
        "auroc_b0",
        "spearman_zeroshot",
        "spearman_b0",
    }
    b0 = summary.loc[("auroc_b0", "x0")]
    assert (b0["rounds"], b0["fold_runs"]) == ("1", 5)
    assert b0["round_2"] == "not read (trial: B0's round 1 only)"
    found = counted_aurocs(workspace, [("b0", 1, fold) for fold in range(1, 6)])
    assert b0["mean"] == float(mean([found[("b0", "x0", 1, fold)] for fold in range(1, 6)]))
    zeroshot = summary.loc[("auroc_zeroshot", "x0")]
    assert (zeroshot["rounds"], zeroshot["fold_runs"]) == ("1 and 2", 10)
    folds = ended.table("fold_values.csv")
    assert set(folds.loc[folds["round"] == 2, "auroc_b0"]) == {
        "not read (trial: B0's round 1 only)"
    }
    assert "F3" not in ended.out
    assert "Δ" not in ended.out
    assert "FAIL" not in ended.out


def test_a_trial_with_a_failed_f2_says_so_and_still_writes_its_tables(
    workspace: EvaluateWorkspace,
) -> None:
    # The researcher decides what follows a failed F2 from the training logs, before the
    # trial (pilot analysis, section "Checks first"); the trial reads what a stop would write.
    fail_f2(workspace)
    calls: list[str] = []
    ended = run_job(workspace, "--trial", prepare=no_b1_or_cpg(calls))
    assert ended.ended == "done", ended.out
    assert calls == [run_name("b0", 1, fold) for fold in range(1, 6)]
    assert FAIL_LINE.format("F2") in ended.out.splitlines()
    assert ended.info["results"]["checks"]["f2"]["verdict"] == "FAIL"
    assert sorted(path.name for path in ended.folder.iterdir()) == [
        "fold_values.csv",
        "run.json",
        "summary.csv",
    ]


# Guards ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"env": {"SLURM_JOB_ID": None}}, "submit this as a job: scripts/slurm/submit.sh evaluate"),
        ({"git": {**CHECKOUT, "dirty": True}}, "the checkout has uncommitted changes"),
        (
            {"git": {**CHECKOUT, "tags": [], "release_tags": []}},
            "no v0.1.* tag exists in the clone",
        ),
        (
            {"git": {**CHECKOUT, "tags": ["v0.1.0"], "describe": "v0.1.0"}},
            "the checkout (v0.1.0) is not at the newest release tag v0.1.1",
        ),
        (
            {"git": {**CHECKOUT, "tags": [], "describe": "v0.1.1-2-gabc"}},
            "is not at the newest release tag v0.1.1",
        ),
        (
            {"git": {**CHECKOUT, "release_tags": ["v0.1.0", "v0.1.1", "v0.1.x"]}},
            "the tag v0.1.x matches v0.1.* but is no release",
        ),
        (
            {
                "git": {
                    **CHECKOUT,
                    "tags": ["v0.1.2"],
                    "describe": "v0.1.2",
                    "release_tags": ["v0.1.0", "v0.1.10", "v0.1.2"],
                },
                "installed": "0.1.2",
            },
            "the checkout (v0.1.2) is not at the newest release tag v0.1.10",  # not by its text
        ),
        ({"installed": "0.1.0"}, "the environment holds version 0.1.0 of the code, not 0.1.1"),
        ({"installed": "0.1.1.dev0"}, "the environment holds version 0.1.1.dev0"),
        ({"args": ("--allow-dirty",)}, "usage: evaluate.py [--trial]"),
    ],
)
def test_the_job_refuses_outside_a_clean_job_at_the_newest_release(
    workspace: EvaluateWorkspace, change: dict[str, Any], message: str
) -> None:
    ended = run_job(
        workspace,
        *change.get("args", ()),
        git=change.get("git"),
        installed=change.get("installed", fakes.INSTALLED),
        env=change.get("env"),
    )
    assert message in failure(ended)
    assert not (workspace.root / "derived" / "evaluate").exists()


def test_the_newest_release_is_the_one_with_the_highest_number(
    workspace: EvaluateWorkspace,
) -> None:
    git = {
        **CHECKOUT,
        "tags": ["v0.1.10"],
        "describe": "v0.1.10",
        "release_tags": ["v0.1.0", "v0.1.10", "v0.1.2"],
    }
    ended = run_job(workspace, "--trial", git=git, installed="0.1.10")
    assert ended.ended == "done", ended.out
    assert (ended.info["release"], ended.info["installed_version"]) == ("v0.1.10", "0.1.10")


def test_the_installed_version_is_read_from_the_packages_metadata() -> None:
    module = load_script("evaluate")
    assert module.installed_version() == package_version("listening-beyond-words")
    assert not module.installed_version().startswith("none")


def test_the_job_listens_for_slurms_stop_before_its_first_refusal(
    workspace: EvaluateWorkspace,
) -> None:
    calls: list[str] = []

    def prepare(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(module, "stop_on_sigterm", lambda: calls.append("listening"))

    line = failure(run_job(workspace, env={"SLURM_JOB_ID": None}, prepare=prepare))
    assert calls == ["listening"]  # even before the first refusal
    assert "submit this as a job" in line


def test_a_job_that_slurm_stops_is_left_failed_with_no_table(workspace: EvaluateWorkspace) -> None:
    calls: list[str] = []

    def prepare(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        real = module.read_scores

        def stopped(where: Any, run: Any, *args: Any) -> Any:  # what the handler raises
            calls.append(run.run)
            if len(calls) == 3:
                raise SystemExit(STOPPED)
            return real(where, run, *args)

        monkeypatch.setattr(module, "read_scores", stopped)

    ended = run_job(workspace, prepare=prepare)
    assert f"FAILED: SystemExit: {STOPPED}" in failure(ended)
    info = ended.info
    assert (info["status"], info["error"]["message"]) == ("failed", STOPPED)
    assert (ended.folder / "errors.log").is_file()
    assert not list(ended.folder.glob("*.csv"))


def test_the_job_refuses_while_no_run_is_pinned(workspace: EvaluateWorkspace) -> None:
    workspace.evaluate["runs"] = {}
    workspace.write_evaluate()
    assert "no training run is pinned in evaluate.yaml" in failure(run_job(workspace))


def test_the_job_refuses_a_set_that_is_no_version_and_two_pins_of_one_fold_run(
    workspace: EvaluateWorkspace,
) -> None:
    workspace.pin(version_runs("all 30 runs")[:-1])
    assert "the 29 pinned runs are none of the pilot's versions" in failure(run_job(workspace))
    workspace.pin(version_runs("all 30 runs"))
    text = workspace.evaluate_path.read_text(encoding="utf-8")
    pin = workspace.evaluate["runs"]["b0"]["r1"]["f2"]
    twice = text.replace(
        "    r1:\n      f1:\n",
        f"    r1:\n      f2:\n        run: {pin['run']}\n"
        f"        scores_sha256: {pin['scores_sha256']}\n"
        f"        record_sha256: {pin['record_sha256']}\n      f1:\n",
        1,
    )
    assert twice != text
    workspace.write_evaluate(twice)
    assert "evaluate.yaml is not valid YAML: the key 'f2' is given twice" in failure(
        run_job(workspace)
    )


# The pins and the records ---------------------------------------------------------------------

KEY = ("b0", 1, 2)


def set_path(record: dict[str, Any], path: str, value: Any) -> None:
    *keys, last = path.split(".")
    for key in keys:
        record = record[key]
    record[last] = value


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("scores_sha256", "its scores.parquet does not match its pin (scores_sha256)"),
        ("record_sha256", "its run.json does not match its pin (record_sha256)"),
    ],
)
def test_the_job_refuses_a_file_that_does_not_match_its_pin(
    workspace: EvaluateWorkspace, change: str, message: str
) -> None:
    workspace.evaluate["runs"]["b0"]["r1"]["f2"][change] = "0" * 64
    workspace.write_evaluate()
    line = failure(run_job(workspace))
    assert f"{run_name(*KEY)}: {message}" in line


def test_the_job_refuses_a_record_that_lists_other_scores_or_scores_changed_after_the_pins(
    workspace: EvaluateWorkspace,
) -> None:
    workspace.change_record(
        KEY, lambda record: record["outputs"].update({"scores.parquet": "1" * 64})
    )
    assert "its run.json lists another scores.parquet than its pin" in failure(run_job(workspace))
    table = workspace.scores(KEY)
    workspace.write_scores(KEY, table)  # listed and pinned again
    table.loc[0, "z"] = table.loc[0, "z"] + 0.125
    table.to_parquet(workspace.folder(KEY) / "scores.parquet", index=False)  # after the pins
    assert "its scores.parquet does not match its pin (scores_sha256)" in failure(
        run_job(workspace, job="991")
    )


def test_the_job_refuses_scores_that_change_after_the_pin_stage(
    workspace: EvaluateWorkspace,
) -> None:
    # The scores are hashed against the pin again as they are read.
    def prepare(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        real = module.pin_runs

        def pin_then_change(job: Any) -> Any:
            kept = real(job)
            table = workspace.scores(KEY)
            table.loc[0, "z"] = table.loc[0, "z"] + 0.125
            table.to_parquet(workspace.folder(KEY) / "scores.parquet", index=False)
            return kept

        monkeypatch.setattr(module, "pin_runs", pin_then_change)

    line = failure(run_job(workspace, prepare=prepare))
    assert f"{run_name(*KEY)}: its scores.parquet changed since its pin was checked" in line


def rival(workspace: EvaluateWorkspace, key: tuple[str, int, int], day: str, **change: Any) -> str:
    """Another done run of a fold-run, a copy of the pinned one started on another day."""
    first = workspace.folder(key)
    folder = first.parent / first.name.replace("20261127", day)
    shutil.copytree(first, folder)
    record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    record.update(run_id=folder.name, **change)
    (folder / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return folder.relative_to(workspace.root).as_posix()


def test_the_job_keeps_the_earlier_of_two_done_runs_of_a_fold_run_and_reports_the_later(
    workspace: EvaluateWorkspace,
) -> None:
    # Pilot analysis, section "Runs and pins": the earlier one counts, the other is reported.
    key = ("b1", 2, 4)
    later = rival(workspace, key, "20261128", finished_utc="2026-11-28T11:00:00+00:00")
    # A run that does not count is no rival, though it finished first.
    rival(workspace, key, "20261125", finished_utc="2026-11-25T11:00:00+00:00", allow_dirty=True)
    ended = run_job(workspace, "--trial")
    assert ended.ended == "done", ended.out
    assert ended.info["inputs"]["runs"]["b1"]["r2"]["f4"]["later_runs"] == [later]
    assert ended.info["inputs"]["runs"]["b1"]["r2"]["f5"]["later_runs"] == []
    assert f"pins: {later} is a later done run of a pinned fold-run and does not count\n" in (
        ended.out
    )
    # The later run pinned by hand instead: refused, naming the earlier one.
    pin = workspace.evaluate["runs"]["b1"]["r2"]["f4"]
    pin["run"] = later
    pin["record_sha256"] = hashlib.sha256(
        (workspace.root / later / "run.json").read_bytes()
    ).hexdigest()
    workspace.write_evaluate()
    line = failure(run_job(workspace, "--trial", job="991"))
    assert (
        f"{later}: {run_name(*key)}, a done run of b1, round 2, fold 4 that counts, finished"
        " earlier, and the earlier one counts" in line
    )


def test_the_job_refuses_a_pinned_run_that_does_not_say_when_it_finished(
    workspace: EvaluateWorkspace,
) -> None:
    workspace.change_record(KEY, lambda record: record.pop("finished_utc"))
    line = failure(run_job(workspace, "--trial"))
    assert f"{run_name(*KEY)}: its run.json does not say when it finished (finished_utc)" in line


def test_the_job_refuses_a_rival_that_counts_but_does_not_say_when_it_finished(
    workspace: EvaluateWorkspace,
) -> None:
    # Neither run can be told to be the earlier one (scripts/list_runs.py then pins neither).
    key = ("b1", 2, 4)
    other = rival(workspace, key, "20261128")
    path = workspace.root / other / "run.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    del record["finished_utc"]
    path.write_text(json.dumps(record), encoding="utf-8")
    line = failure(run_job(workspace, "--trial"))
    assert f"{other}: its run.json does not say when it finished (finished_utc)" in line


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{not json", "its run.json cannot be read as JSON"),
        ("[1, 2]", "its run.json is not a JSON object"),
    ],
)
def test_the_job_refuses_a_pinned_record_that_is_no_json_object(
    workspace: EvaluateWorkspace, text: str, message: str
) -> None:
    (workspace.folder(KEY) / "run.json").write_text(text, encoding="utf-8")
    workspace.pin(workspace.pinned())  # the pin matches the file: only the content is wrong
    assert f"{run_name(*KEY)}: {message}" in failure(run_job(workspace))


def test_the_job_checks_the_scores_pin_before_it_reads_the_record(
    workspace: EvaluateWorkspace,
) -> None:
    # Two faults at once: the record's pin holds but its content is no JSON, and the scores
    # do not match their pin. The pins are checked first, the record's, then the scores'.
    (workspace.folder(KEY) / "run.json").write_text("{not json", encoding="utf-8")
    workspace.pin(workspace.pinned())
    workspace.evaluate["runs"]["b0"]["r1"]["f2"]["scores_sha256"] = "0" * 64
    workspace.write_evaluate()
    line = failure(run_job(workspace))
    assert f"{run_name(*KEY)}: its scores.parquet does not match its pin (scores_sha256)" in line


def test_the_job_refuses_zero_shot_scores_of_a_participant_without_a_label(
    workspace: EvaluateWorkspace,
) -> None:
    # Session 340 is a pilot session with segments in the caches, but no label.
    config = workspace.config
    for pin in config["cache_runs"].values():
        path = workspace.root / pin["run"] / "manifest.parquet"
        manifest = pd.read_parquet(path)
        manifest = pd.concat(
            [manifest, pd.DataFrame({"segment": ["340_000"], "session": [340]})],
            ignore_index=True,
        )
        manifest.to_parquet(path, index=False)
        pin["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    run = workspace.root / fakes.ZEROSHOT_RUN
    table = pd.read_parquet(run / "scores.parquet")
    extra = pd.DataFrame(
        [("340_000", 340, view, 0.5) for view in fakes.SCORED], columns=table.columns
    )
    pd.concat([table, extra], ignore_index=True).to_parquet(run / "scores.parquet", index=False)
    scores_sha256 = hashlib.sha256((run / "scores.parquet").read_bytes()).hexdigest()
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    record["inputs"]["caches"] = config["cache_runs"]
    record["outputs"]["scores.parquet"] = scores_sha256
    (run / "run.json").write_text(json.dumps(record), encoding="utf-8")
    config["zeroshot_scores"]["sha256"] = scores_sha256
    workspace.write_config()
    line = failure(run_job(workspace))
    assert (
        "the zero-shot scores of x0 name 1 participants without a label and lack 0 labelled ones"
        in line
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("status", "running", "did not finish: its run.json does not say done"),
        ("allow_dirty", True, "was started with --allow-dirty"),
        ("results.rules.f1.passes", False, "fails check F1: a failing run cannot be pinned"),
        ("results.rules.f1", {}, "its run.json does not say that check F1 passed"),
        ("git.commit", "f" * 40, "comes from neither the first runs' commit 271cce6"),
        ("fold", 3, "is not a run of b0, round 1, fold 2"),
        ("timing", True, "the timing run of check F5 never counts"),
    ],
)
def test_the_job_refuses_a_run_that_does_not_count(
    workspace: EvaluateWorkspace, path: str, value: Any, message: str
) -> None:
    workspace.change_record(KEY, lambda record: set_path(record, path, value))
    assert message in failure(run_job(workspace))


def test_the_job_refuses_a_run_off_the_newest_release_tag_of_its_clone(
    workspace: EvaluateWorkspace,
) -> None:
    key = ("b1", 2, 2)

    def older(record: dict[str, Any]) -> None:
        record["git"]["release_tags"] = ["v0.1.0", "v0.1.1"]  # v0.1.1 came out before the run

    workspace.change_record(key, older)
    line = failure(run_job(workspace))
    assert (
        f"{run_name(*key)} does not come from the newest release tag of its clone, v0.1.1" in line
    )


@pytest.mark.parametrize("name", ["labels", "partitions"])
def test_the_job_refuses_a_run_made_from_other_labels_or_partitions(
    workspace: EvaluateWorkspace, name: str
) -> None:
    workspace.change_record(KEY, lambda record: set_path(record, f"inputs.{name}.sha256", "9" * 64))
    assert f"{run_name(*KEY)} was made from other {name} than data.yaml pins now" in failure(
        run_job(workspace)
    )


@pytest.mark.parametrize(
    ("file", "message"),
    [
        (
            "derived/labels/run-1/labels.parquet",
            "the pinned labels and partitions do not hold: labels.parquet: SHA-256",
        ),
        (
            "derived/partitions/run-1/partitions.parquet",
            "the pinned labels and partitions do not hold: partitions.parquet: SHA-256",
        ),
        (
            f"{fakes.ZEROSHOT_RUN}/scores.parquet",
            f"the pinned zero-shot scores: {fakes.ZEROSHOT_RUN}/scores.parquet does not match the pin",
        ),
        (
            f"{fakes.WORDCHECK_RUN}/transcripts.parquet",
            "its transcripts.parquet does not match the pin",
        ),
    ],
)
def test_the_job_refuses_a_pinned_input_of_data_yaml_that_does_not_match_its_pin(
    workspace: EvaluateWorkspace, file: str, message: str
) -> None:
    path = workspace.root / file
    table = pd.read_parquet(path)
    table.iloc[:-1].to_parquet(path)
    assert message in failure(run_job(workspace))


def relabel(workspace: EvaluateWorkspace, change: Any, partitions: Any = None) -> None:
    """New pinned labels (and partitions), with every run's record made from them."""
    labels = workspace.labels.copy()
    changed = change(labels)  # in place, or a new table
    labels = labels if changed is None else changed
    path = workspace.root / "derived" / "labels" / "run-1" / "labels.parquet"
    labels.to_parquet(path)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    workspace.config["labels"]["sha256"] = sha256
    built = workspace.root / "derived" / "partitions" / "run-1"
    if partitions is not None:
        partitions(workspace.partitions).to_parquet(built / "partitions.parquet", index=False)
        data = (built / "partitions.parquet").read_bytes()
        workspace.config["partitions"]["sha256"] = hashlib.sha256(data).hexdigest()
    workspace.write_config()
    (built / "run.json").write_text(
        json.dumps({"status": "done", "inputs": {"labels": {"sha256": sha256}}})
    )
    for key in workspace.pinned():
        record = workspace.record(key)
        record["inputs"]["labels"] = dict(workspace.config["labels"])
        record["inputs"]["partitions"] = dict(workspace.config["partitions"])
        (workspace.folder(key) / "run.json").write_text(json.dumps(record), encoding="utf-8")
    workspace.pin(workspace.pinned())


def test_the_job_refuses_a_label_that_disagrees_with_its_phq8_total(
    workspace: EvaluateWorkspace,
) -> None:
    assert workspace.labels.loc[300, "label"] == 1  # with a total of 10

    def below(labels: pd.DataFrame) -> None:
        labels.loc[300, "phq8_total"] = 9

    relabel(workspace, below)
    line = failure(run_job(workspace))
    # Counts only: neither the participant nor the label or total.
    assert line.startswith(
        "FAILED: StepError: 1 labels disagree with their PHQ-8 totals: a participant is"
        " positive exactly when the total is 10 or more (pilot analysis, section"
        ' "Data protection"); traceback frames in '
    )
    assert "300" not in line.split("; traceback frames in ")[0]
    assert not list((workspace.root / "derived" / "evaluate" / f"run-{JOB}").glob("*.csv"))


def add_participant(workspace: EvaluateWorkspace, session: int) -> None:
    """One participant more, in the pinned labels and in a test fold of each round."""
    row = pd.DataFrame(
        {"phq8_total": [3], "label": [0], "gender": ["female"]},
        index=pd.Index([session], name="session"),
    )

    def more(labels: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([labels, row.astype(labels.dtypes)])

    def placed(partitions: pd.DataFrame) -> pd.DataFrame:
        extra = pd.DataFrame({"session": [session] * 2, "round": [1, 2], "fold": [1, 1]})
        return pd.concat([partitions, extra.astype(partitions.dtypes)], ignore_index=True)

    relabel(workspace, more, placed)


def test_the_job_refuses_partitions_with_a_test_fold_of_one_class_before_any_draw(
    workspace: EvaluateWorkspace,
) -> None:
    def one_class(partitions: pd.DataFrame) -> pd.DataFrame:
        moved = partitions.copy()
        positive = moved["session"].map(workspace.labels["label"]) == 1
        moved.loc[positive & (moved["round"] == 1) & (moved["fold"] == 1), "fold"] = 2
        return moved

    relabel(workspace, lambda labels: None, one_class)
    ended = run_job(workspace)
    assert failure(ended).startswith(
        "FAILED: StepError: the bootstrap cannot draw: every test fold of every round must hold"
        " both classes"
    )
    assert "bootstrap:" not in ended.out
    assert not list(ended.folder.glob("*.csv"))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        (342, "the pinned labels hold 1 participants outside the pilot's sessions"),
        (600, "the pinned labels hold 1 participants outside the pilot's sessions"),
        ("a label of 2", "1 labels are not 0 or 1"),
        ("a total of 10.5", "the pinned labels and PHQ-8 totals are not stored as whole numbers"),
    ],
)
def test_the_job_refuses_labels_outside_the_pilot_or_not_whole_before_any_draw(
    workspace: EvaluateWorkspace, case: Any, message: str
) -> None:
    if isinstance(case, int):  # 342 never existed; 600 to 718 are the external test
        add_participant(workspace, case)
    elif case == "a label of 2":

        def two(labels: pd.DataFrame) -> None:
            labels.loc[301, "label"] = 2  # a negative participant, total 1

        relabel(workspace, two)
    else:

        def half(labels: pd.DataFrame) -> None:
            labels["phq8_total"] = labels["phq8_total"].astype(np.float64)
            labels.loc[300, "phq8_total"] = 10.5

        relabel(workspace, half)
    ended = run_job(workspace)
    line = failure(ended)
    assert line.startswith(f"FAILED: StepError: {message}")
    assert "bootstrap:" not in ended.out  # refused before the draws
    shown = line.split("; traceback frames in ")[0]
    assert not re.search(r"\b(30\d|3[1-3]\d|342|600)\b", shown)  # counts only, never an ID
    assert not list(ended.folder.glob("*.csv"))


# The scores of a run ----------------------------------------------------------------------------


def other_fold_segment(workspace: EvaluateWorkspace) -> tuple[str, int]:
    """A segment of a participant outside KEY's test fold: one of its training participants."""
    members = set(workspace.fold_sessions(KEY[1], KEY[2]))
    session = next(session for session in SESSIONS if session not in members)
    return fakes.segments_of(session)[0], session


def add_row(table: pd.DataFrame, segment: str, session: int) -> pd.DataFrame:
    row = pd.DataFrame([(segment, session, "x0", 0.5)], columns=table.columns)
    return pd.concat([table, row], ignore_index=True)


SCORE_CHANGES = {
    "a training participant's segment": (
        lambda workspace, table: add_row(table, *other_fold_segment(workspace)),
        "its scores of x0 hold 1 segments of participants outside its test fold",
    ),
    "a missing segment": (
        lambda workspace, table: table.drop(index=0).reset_index(drop=True),
        "its scores of x0 lack 1 segments of its test fold",
    ),
    "a z not finite": (
        lambda workspace, table: table.assign(z=[np.nan, *table["z"].iloc[1:]]),
        "1 of its z on x0 are not finite",
    ),
    "a session outside the pilot's labelled participants": (
        lambda workspace, table: add_row(table, "600_000", 600),
        "its scores of x0 hold 1 rows of sessions outside the pilot's labelled participants",
    ),
    "a segment of another participant": (
        lambda workspace, table: table.assign(
            session=[int(table["session"].iloc[-1]), *table["session"].iloc[1:]]
        ),
        "hold 1 segments that the pinned caches do not hold for their participant",
    ),
    "a segment twice": (
        lambda workspace, table: pd.concat([table, table.iloc[[0]]], ignore_index=True),
        "its scores of x0 hold 1 segments twice",
    ),
    "a view that is no test view": (
        lambda workspace, table: table.assign(view=["x1", *table["view"].iloc[1:]]),
        "holds 1 rows of views other than ['x0', 'rev']",
    ),
    "other columns": (
        lambda workspace, table: table.rename(columns={"z": "score"}),
        "its scores.parquet does not have the columns ['segment', 'session', 'view', 'z']",
    ),
    "z that are not floats": (
        lambda workspace, table: table.assign(z=table["z"].round().astype(np.int64)),
        "the z of its scores.parquet are not floating-point numbers",
    ),
    "sessions that are not whole numbers": (
        lambda workspace, table: table.assign(session=table["session"].astype(np.float64)),
        "the sessions of its scores.parquet are not whole numbers",
    ),
}


@pytest.mark.parametrize("case", list(SCORE_CHANGES))
def test_the_job_refuses_scores_that_are_not_exactly_the_test_fold(
    workspace: EvaluateWorkspace, case: str
) -> None:
    change, message = SCORE_CHANGES[case]
    table = workspace.scores(KEY)
    planted = [PLANTED_Z + index / 64 for index in range(len(table))]
    table["z"] = planted  # each z distinctive
    edited = change(workspace, table)
    workspace.write_scores(KEY, edited)
    line = failure(run_job(workspace))
    assert line.startswith(f"FAILED: StepError: {run_name(*KEY)}: ")
    assert message in line
    shown = line.split("; traceback frames in ")[0]
    assert "12." not in shown  # no z of the run, however rounded
    if "z" in edited and pd.api.types.is_float_dtype(edited["z"]):
        for z in edited["z"].dropna():
            assert repr(float(z)) not in shown
            assert f"{z:.4f}" not in shown
    for z in planted:  # nor as the exact fraction that the job computes with
        assert str(Fraction(z).numerator) not in shown


# Checks F4 and F5 -------------------------------------------------------------------------------


def change_wordcheck(workspace: EvaluateWorkspace, change: Any) -> None:
    path = workspace.root / fakes.WORDCHECK_RUN / "run.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    change(record)
    path.write_text(json.dumps(record), encoding="utf-8")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda record: record["verdicts"].update(f4={"verdict": "FAIL"}), "check F4 says FAIL"),
        (lambda record: record["verdicts"].pop("f4"), "check F4 says None"),
    ],
)
def test_the_job_refuses_an_f4_record_that_does_not_say_pass(
    workspace: EvaluateWorkspace, change: Any, message: str
) -> None:
    change_wordcheck(workspace, change)
    line = failure(run_job(workspace))
    assert f"{fakes.WORDCHECK_RUN}: {message}, not PASS" in line


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("results.rules.f5.passes", False, "does not say that check F5 passed"),
        ("results.rules.f5", {"training_hours": 1.0}, "does not say that check F5 passed"),
        ("results.rules.f5.hours", None, "lacks the hours of check F5"),
        ("results.rule_readings.reads_rules", False, "does not say that it read check F5"),
    ],
)
def test_the_job_refuses_an_f5_record_that_does_not_say_pass(
    workspace: EvaluateWorkspace, path: str, value: Any, message: str
) -> None:
    workspace.change_record(("cpg", 1, 1), lambda record: set_path(record, path, value))
    line = failure(run_job(workspace))
    assert f"{run_name('cpg', 1, 1)}: its run.json {message}" in line


def test_an_f2_record_that_lacks_its_verdict_is_refused_not_read_as_a_fail(
    workspace: EvaluateWorkspace,
) -> None:
    key = ("cpg", 2, 1)
    workspace.change_record(
        key, lambda record: record["results"]["rules"]["alive"].pop("f2_passes")
    )
    line = failure(run_job(workspace))
    assert f"{run_name(*key)}: its run.json does not say whether check F2 passed" in line


# What leaves OMNI ---------------------------------------------------------------------------------


def assert_no_participant_in_table(path: Path) -> None:
    """No cell of a table names a participant of the workspace, as a number or in its text."""
    text = path.read_text(encoding="utf-8")
    assert not NAMED_ID.search(text), path.name
    assert not PARTICIPANT_PAIR.search(text), path.name
    assert not SEGMENT_ID.search(text), path.name
    for value in pd.read_csv(path, dtype=str, keep_default_na=False).to_numpy().ravel():
        try:
            number = float(value)
        except ValueError:
            continue
        assert not (number.is_integer() and int(number) in SESSIONS), path.name


def outputs_of(ended: fakes.Ended) -> list[str]:
    """Everything that the job wrote or printed."""
    texts = [ended.streams]
    if ended.folder.exists():
        texts += [path.read_text(encoding="utf-8") for path in sorted(ended.folder.iterdir())]
    return texts


@pytest.mark.parametrize("case", ["full", "trial", "F2 fails", "a refusal"])
def test_only_aggregates_leave_the_job(workspace: EvaluateWorkspace, case: str) -> None:
    # One participant's zero-shot score on x0 is the planted value, and so is one
    # participant's in PLANTED_RUN: no score of a participant may leave either.
    zeroshot = pd.read_parquet(workspace.root / fakes.ZEROSHOT_RUN / "scores.parquet")
    assert exact_means(zeroshot, "x0")[PLANTED_SESSION] == Fraction(PLANTED_Z)
    first = workspace.fold_sessions(*PLANTED_RUN[1:])[0]
    assert exact_means(workspace.scores(PLANTED_RUN), "x0")[first] == Fraction(PLANTED_Z)
    args: tuple[str, ...] = ()
    if case == "trial":
        args = ("--trial",)
    elif case == "F2 fails":
        fail_f2(workspace)
    elif case == "a refusal":
        table = workspace.scores(PLANTED_RUN)
        table.loc[1, "z"] = np.inf
        workspace.write_scores(PLANTED_RUN, table)
    ended = run_job(workspace, *args)
    assert ended.ended == ("failed" if case == "a refusal" else "done"), ended.out
    assert_only_aggregates_leave(ended.folder / "run.json", ended.streams)
    for text in outputs_of(ended):
        # The planted z, as written, rounded or as the exact fraction that the job computes
        # with, and no segment ID, refusals included.
        assert "12.345" not in text
        assert "12.346" not in text
        assert str(Fraction(PLANTED_Z).numerator) not in text
        assert not SEGMENT_ID.search(text)
    for name in TABLES:
        if (ended.folder / name).exists():
            table = ended.table(name)
            assert not {"session", "segment", "pid", "segment_id"} & set(table.columns)
            assert len(table) < len(SESSIONS)  # no table holds a row per participant
            assert_no_participant_in_table(ended.folder / name)
    if case == "a refusal":
        line = failure(ended)
        assert line.startswith(
            f"FAILED: StepError: {run_name(*PLANTED_RUN)}: 1 of its z on x0 are not finite;"
            " traceback frames in "
        )
        assert (
            ended.info["error"]["message"]
            == line.split("FAILED: StepError: ")[1].split("; traceback frames in ")[0]
        )


def test_the_leak_check_fails_on_a_planted_id(workspace: EvaluateWorkspace) -> None:
    ended = run_job(workspace)
    assert_only_aggregates_leave(ended.folder / "run.json", ended.streams)
    for name in TABLES:  # the checks of the tables, on tables with an ID planted
        assert_no_participant_in_table(ended.folder / name)
        for planted in ("312", "participant 312", "312.0"):
            table = ended.table(name).astype(str)
            table.iloc[0, -1] = planted
            table.to_csv(ended.folder / f"planted-{name}", index=False)
            with pytest.raises(AssertionError):
                assert_no_participant_in_table(ended.folder / f"planted-{name}")
            (ended.folder / f"planted-{name}").unlink()
    with pytest.raises(AssertionError):
        assert_only_aggregates_leave(ended.folder / "run.json", ended.streams + "participant 312\n")
    info = ended.info
    info["results"]["counts"]["session"] = 312
    (ended.folder / "run.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_only_aggregates_leave(ended.folder / "run.json", ended.streams)


# The job file ------------------------------------------------------------------------------------


def test_the_job_file_asks_for_two_cpus_of_the_short_partition_for_30_minutes() -> None:
    text = (REPO / "scripts" / "slurm" / "evaluate.sbatch").read_text(encoding="utf-8")
    sbatch = [line for line in text.splitlines() if line.startswith("#SBATCH")]
    assert sbatch == [
        "#SBATCH --job-name=lbw-evaluate",
        "#SBATCH --partition=short",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=2",
        "#SBATCH --mem=8G",
        "#SBATCH --time=00:30:00",
    ]
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail\nsource scripts/slurm/common.sh\n" in text
    assert text.endswith('python scripts/evaluate.py "$@"\n')
    first_command = text.splitlines().index("set -euo pipefail")
    assert all(text.splitlines().index(line) < first_command for line in sbatch)
    assert "gres" not in text  # a CPU job
