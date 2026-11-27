"""Stand-ins for the tests of the analysis job (scripts/evaluate.py): a light workspace in code.

Everything that the job reads, written in code, with no audio and no model: the labels with
their PHQ-8 totals; partitions of 2 rounds of 5 folds (partitions.make_partitions, seeds 101
and 102); the manifests of the pinned caches of x0, x1 and rev, which hold only what the
zero-shot reader looks at (segment and session); a zero-shot run with the fields that
zeroshot.load_zeroshot_scores reads; a word-check run with the fields that scoring.embargo
reads; and the 30 training runs, each with its test fold's scores of x0 and rev and a
run.json whose rule records come from training.rule_measures on synthetic metrics lines, as
the train job writes them. B0's round 1 and CPG's round 1, fold 1 come from the first runs'
commit, every other run from the release tag v0.1.0 of its clone; the job runs at v0.1.1.

Synthetic data only, on the pilot IDs 300 to 339, which assert_only_aggregates_leave
watches. Each participant has 2 or 4 segments, and every z lies on a grid of 1/8, so a
participant's mean is exact in floating point and ties are frequent, except the planted z
(PLANTED_Z): every segment of x0 of one participant in the zero-shot run (PLANTED_SESSION)
and of the first participant of one training run's test fold (PLANTED_RUN), so that a
participant's score is the planted value too. It must appear in no output.
"""

import contextlib
import copy
import hashlib
import io
import json
import shutil
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from listening_beyond_words import training
from listening_beyond_words.config import load_arm
from listening_beyond_words.evaluation import ARMS, ROUNDS, TRAINED_ARMS, VERSIONS
from listening_beyond_words.loss import STEP_MEASURES
from listening_beyond_words.partitions import make_partitions
from listening_beyond_words.wordcheck import FAIL, PASS
from listening_beyond_words.zeroshot import SCORE_COLUMNS
from x1_fakes import REPO, load_script

CONFIGS = REPO / "configs" / "pilot"
SESSIONS = tuple(range(300, 340))  # pilot IDs, 16 positive and 24 negative participants
SCORED = ("x0", "x1", "rev")  # the zero-shot run's views
TEST_VIEWS = ("x0", "rev")  # every trained run's
JOB = "990"
FIRST = yaml.safe_load((CONFIGS / "evaluate.yaml").read_text(encoding="utf-8"))["first_runs_commit"]
RELEASED = "d" * 40  # the commit of the release v0.1.0, from which the later runs come
CHECKOUT = {
    "commit": "e" * 40,
    "dirty": False,
    "describe": "v0.1.1",
    "tags": ["v0.1.1"],
    "release_tags": ["v0.1.0", "v0.1.1"],
}
INSTALLED = "0.1.1"
RESAMPLES = 200  # resamples of the fast tests; the real config has 10,000
# How far each arm's z lean to the label: B0 learns (F3 passes) and CPG beats B0 on x0.
SHIFTS = {"zeroshot": 0.4, "b0": 1.0, "b1": 1.1, "cpg": 1.6}
PLANTED_Z = 12.345678
PLANTED_RUN = ("cpg", 2, 3)
PLANTED_SESSION = 305  # its zero-shot score on x0 is PLANTED_Z
ZEROSHOT_RUN = "runs/pilot/pilot-zeroshot-20261126-090000"
WORDCHECK_RUN = "derived/wordcheck/run-700"
CACHE_JOBS = {"x0": "800", "x1": "801", "rev": "803"}
CACHE_CHECK = "derived/cache_check/run-804"
TRAINING_HOURS = 1.0785  # the first CPG run's training part, which check F5 reads


def label_of(session: int) -> tuple[int, str, int]:
    """Label, gender and PHQ-8 total: 8 positive and 12 negative participants of each gender."""
    index = session - SESSIONS[0]
    label = int(index % 5 in (0, 2))
    gender = "female" if (index // 5) % 2 == 0 else "male"
    total = 10 + index % 15 if label else index % 10
    return label, gender, total


def segments_of(session: int) -> list[str]:
    """A participant's segments: 2 or 4, so that a mean of z on a grid of 1/8 stays exact."""
    count = 4 if (session - SESSIONS[0]) % 3 == 0 else 2
    return [f"{session}_{index:03d}" for index in range(count)]


def on_grid(value: float) -> float:
    """A z on a grid of 1/8, so that sums and means of 2 or 4 of them are exact."""
    return float(np.round(8.0 * value) / 8.0)


def run_name(arm: str, round_number: int, fold: int) -> str:
    """A training run's folder, relative to the workspace, as the train job names it."""
    day = "20261126" if first_run(arm, round_number, fold) else "20261127"
    order = ARMS.index(arm) * 100 + round_number * 10 + fold
    return f"runs/pilot/pilot-{arm}-r{round_number}-f{fold}-{day}-{order:06d}"


def first_run(arm: str, round_number: int, fold: int) -> bool:
    """B0's round 1 and CPG's round 1, fold 1: the runs from before the settings froze."""
    return (arm, round_number) == ("b0", 1) or (arm, round_number, fold) == ("cpg", 1, 1)


def version_runs(name: str) -> list[tuple[str, int, int]]:
    """The fold-runs of one of the protocol's versions, by its name."""
    (version,) = [version for version in VERSIONS if version.name == name]
    return sorted((arm, number, fold) for arm, number in version.arm_rounds for fold in range(1, 6))


def metric_lines(gain: bool, active: bool = True) -> list[dict[str, Any]]:
    """A run's metrics.jsonl: 3 epochs of 4 steps; with the gain, every other step's bags active.

    So a CPG run's gain-alive share over epochs 2 and 3 is 0.5 (F2 passes), or 0 when none is
    active (F2 fails); the CE part is flat, so the 4-epoch rule does not fire.
    """
    lines = []
    for step in range(12):
        line: dict[str, Any] = {
            "step": step,
            "epoch": step // 4 + 1,
            "seconds": 60.0,
            "lr": 1.0e-4,
            "adapter_lr": 2.0e-5,
            "loss": 1.2,
            "ce_x0": 0.7,
            "ce_part": 1.0,
        }
        if gain:
            line |= {"ce_x1": 0.6, "ce_x2": 0.3, "hinge": 0.2, "lambda_g": 1.0}
        line |= {
            "scale": 65_536.0,
            "next_scale": 65_536.0,
            "skipped": False,
            "grad_norm": 2.0,
            "clipped": False,
            "calls": 14 if gain else 4,
        }
        if gain:
            line |= dict.fromkeys(STEP_MEASURES, 0.5)
            line |= {f"bags_{group}": 1 for group in training.GROUPS}
            line |= {f"active_{group}": int(active and step % 2 == 1) for group in training.GROUPS}
            line |= {"sigma0_at_floor": 0, "sigma1_at_floor": 0}
        else:
            line |= dict.fromkeys(training.X0_MEASURES, 0.5)
        lines.append(line)
    return lines


def rule_record(arm: str, round_number: int, fold: int, active: bool = True) -> dict[str, Any]:
    """results.rules and results.rule_readings, as the train job's stage 8 writes them."""
    first_cpg = (arm, round_number, fold) == ("cpg", 1, 1)
    readings = {
        "reads_f2": arm == "cpg",
        "reads_rules": first_cpg,
        "reads_f5": False,
        "made_on": None,
        "f5_read_on": None,
        "cannot_tell": None,
    }
    settings = load_arm(CONFIGS / f"{arm}.yaml").shared
    hours = None if arm == "b0" else (TRAINING_HOURS if first_cpg else 1.05)
    measures = training.rule_measures(
        metric_lines(arm != "b0", active),
        settings,
        hours,
        reads_f2=readings["reads_f2"],
        reads_rules=readings["reads_rules"],
    )
    return {"rules": measures, "rule_readings": readings}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class EvaluateWorkspace:
    """The analysis job's synthetic workspace, its data config and its evaluate.yaml.

    Attributes:
        root: The workspace root.
        config: The data config (data.yaml).
        config_path: Where the job reads it.
        evaluate: The analysis settings (evaluate.yaml), with the pins of the kept runs.
        evaluate_path: Where the job reads them.
        labels: The labels, by session: phq8_total, label and gender.
        partitions: The partitions: session, round and fold.
    """

    root: Path
    config: dict[str, Any]
    config_path: Path
    evaluate: dict[str, Any]
    evaluate_path: Path
    labels: pd.DataFrame
    partitions: pd.DataFrame

    def write_config(self) -> None:
        self.config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")

    def write_evaluate(self, text: str | None = None) -> None:
        """Write evaluate.yaml from its content, or as the text given."""
        if text is None:
            text = yaml.safe_dump(self.evaluate, sort_keys=False)
        self.evaluate_path.write_text(text, encoding="utf-8")

    def copy_to(self, folder: Path) -> "EvaluateWorkspace":
        """A copy of the workspace and its two configs in a new folder."""
        root = folder / "workspace"
        shutil.copytree(self.root, root)
        copied = EvaluateWorkspace(
            root,
            copy.deepcopy(self.config),
            folder / "data.yaml",
            copy.deepcopy(self.evaluate),
            folder / "evaluate.yaml",
            self.labels.copy(),
            self.partitions.copy(),
        )
        copied.write_config()
        copied.write_evaluate()
        return copied

    def fold_sessions(self, round_number: int, fold: int) -> list[int]:
        chosen = self.partitions[
            (self.partitions["round"] == round_number) & (self.partitions["fold"] == fold)
        ]
        return sorted(int(session) for session in chosen["session"])

    def folder(self, key: tuple[str, int, int]) -> Path:
        return self.root / run_name(*key)

    def record(self, key: tuple[str, int, int]) -> dict[str, Any]:
        return json.loads((self.folder(key) / "run.json").read_text(encoding="utf-8"))

    def scores(self, key: tuple[str, int, int]) -> pd.DataFrame:
        return pd.read_parquet(self.folder(key) / "scores.parquet")

    def pin(self, runs: Iterable[tuple[str, int, int]]) -> None:
        """Pin these runs in evaluate.yaml by their folders and files, as list_runs prints them."""
        pins: dict[str, Any] = {}
        for arm, round_number, fold in sorted(runs, key=lambda key: (ARMS.index(key[0]), *key[1:])):
            folder = self.folder((arm, round_number, fold))
            pins.setdefault(arm, {}).setdefault(f"r{round_number}", {})[f"f{fold}"] = {
                "run": run_name(arm, round_number, fold),
                "scores_sha256": sha256_of(folder / "scores.parquet"),
                "record_sha256": sha256_of(folder / "run.json"),
            }
        self.evaluate["runs"] = pins
        self.write_evaluate()

    def pinned(self) -> list[tuple[str, int, int]]:
        """The fold-runs pinned now."""
        return sorted(
            (arm, int(round_key[1:]), int(fold_key[1:]))
            for arm, rounds in self.evaluate["runs"].items()
            for round_key, folds in rounds.items()
            for fold_key in folds
        )

    def write_record(self, key: tuple[str, int, int], record: dict[str, Any]) -> None:
        """Write a run's run.json and pin it again, so that only the change counts."""
        (self.folder(key) / "run.json").write_text(json.dumps(record, indent=2), "utf-8")
        self.pin(self.pinned())

    def write_scores(self, key: tuple[str, int, int], table: pd.DataFrame) -> None:
        """Write a run's scores, list them in its run.json and pin both again."""
        path = self.folder(key) / "scores.parquet"
        table.to_parquet(path, index=False)
        record = self.record(key)
        record["outputs"]["scores.parquet"] = sha256_of(path)
        self.write_record(key, record)

    def change_record(self, key: tuple[str, int, int], change: Callable[[dict], None]) -> None:
        record = self.record(key)
        change(record)
        self.write_record(key, record)


def write_parquet(path: Path, table: pd.DataFrame, index: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=index)
    return sha256_of(path)


def write_json(path: Path, data: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return sha256_of(path)


def run_scores(
    workspace_labels: pd.DataFrame, sessions: Iterable[int], arm: str, seed: list[int]
) -> pd.DataFrame:
    """A trained run's scores of its test fold: every segment of x0 and rev once, z on the grid."""
    rng = np.random.default_rng(seed)
    rows = []
    for view in TEST_VIEWS:
        for session in sessions:
            label = int(workspace_labels.loc[session, "label"])
            for segment in segments_of(session):
                z = on_grid(rng.normal() + SHIFTS[arm] * label)
                rows.append((segment, int(session), view, z))
    return pd.DataFrame(rows, columns=list(SCORE_COLUMNS))


def train_record(
    workspace: EvaluateWorkspace, key: tuple[str, int, int], scores_sha256: str
) -> dict[str, Any]:
    """A done training run's run.json, with the fields that the analysis reads."""
    arm, round_number, fold = key
    first = first_run(*key)
    git = {
        "commit": FIRST if first else RELEASED,
        "dirty": False,
        "describe": FIRST[:7] if first else "v0.1.0",
        "tags": [] if first else ["v0.1.0"],
        "release_tags": [] if first else ["v0.1.0"],
    }
    day = "26" if first else "27"
    return {
        "run_id": Path(run_name(*key)).name,
        "step": "train",
        "arm": arm,
        "round": round_number,
        "fold": fold,
        "timing": False,
        "status": "done",
        "started_utc": f"2026-11-{day}T10:00:00+00:00",
        "finished_utc": f"2026-11-{day}T11:00:00+00:00",
        "git": git,
        "allow_dirty": False,
        "release": "a stand-in",
        "inputs": {
            "labels": dict(workspace.config["labels"]),
            "partitions": dict(workspace.config["partitions"]),
        },
        "results": {"sampler": {"max_segments_per_epoch": 16}, **rule_record(*key)},
        "outputs": {"scores.parquet": scores_sha256},
    }


def make_evaluate_workspace(folder: Path) -> EvaluateWorkspace:
    """Write every input of the analysis job, with all 30 runs pinned."""
    root = folder / "workspace"
    labels = pd.DataFrame(
        [label_of(session) for session in SESSIONS],
        columns=["label", "gender", "phq8_total"],
        index=pd.Index(list(SESSIONS), name="session"),
    )[["phq8_total", "label", "gender"]]
    labels_sha = write_parquet(
        root / "derived" / "labels" / "run-1" / "labels.parquet", labels, True
    )
    partitions = make_partitions(labels[["label", "gender"]])
    partitions_run = root / "derived" / "partitions" / "run-1"
    partitions_sha = write_parquet(partitions_run / "partitions.parquet", partitions)
    write_json(
        partitions_run / "run.json",
        {"status": "done", "inputs": {"labels": {"sha256": labels_sha}}},
    )

    manifest = pd.DataFrame(
        [(segment, session) for session in SESSIONS for segment in segments_of(session)],
        columns=["segment", "session"],
    )
    cache_runs = {
        view: {
            "run": f"cache/{view}/run-{job}",
            "sha256": write_parquet(
                root / "cache" / view / f"run-{job}" / "manifest.parquet", manifest
            ),
            "verified_by": CACHE_CHECK,
        }
        for view, job in CACHE_JOBS.items()
    }
    rng = np.random.default_rng(0)
    zeroshot = pd.DataFrame(
        [
            (
                segment,
                session,
                view,
                on_grid(rng.normal() + SHIFTS["zeroshot"] * label_of(session)[0]),
            )
            for view in SCORED
            for session in SESSIONS
            for segment in segments_of(session)
        ],
        columns=list(SCORE_COLUMNS),
    )
    planted = (zeroshot["session"] == PLANTED_SESSION) & (zeroshot["view"] == "x0")
    zeroshot.loc[planted, "z"] = PLANTED_Z
    zeroshot_sha = write_parquet(root / ZEROSHOT_RUN / "scores.parquet", zeroshot)
    write_json(
        root / ZEROSHOT_RUN / "run.json",
        {
            "step": "score_zeroshot",
            "status": "done",
            "settings": {"zeroshot": {"views": list(SCORED)}},
            "inputs": {"caches": cache_runs},
            "outputs": {"scores.parquet": zeroshot_sha},
        },
    )

    view_runs = {
        view: {"run": f"views/{view}/run-1", "sha256": str(number) * 64}
        for number, view in enumerate(("x0", "x1", "x2", "rev"), start=1)
    }
    transcripts = pd.DataFrame({"segment": ["300_000"], "view": ["rev"], "words": [40]})
    transcripts_sha = write_parquet(root / WORDCHECK_RUN / "transcripts.parquet", transcripts)
    write_json(
        root / WORDCHECK_RUN / "run.json",
        {
            "step": "check_views",
            "mode": {"name": "sample"},
            "status": "done",
            "inputs": {
                "views": {
                    view: {"file": pin["run"], "sha256": pin["sha256"]}
                    for view, pin in view_runs.items()
                }
            },
            "verdicts": {
                key: {"verdict": value}
                for key, value in {
                    "f4": PASS,
                    "x0_control": PASS,
                    "x1_words": PASS,
                    "x1_voice": FAIL,
                }.items()
            },
            "outputs": {"transcripts.parquet": transcripts_sha},
        },
    )
    config = {
        "labels": {"run": "derived/labels/run-1", "sha256": labels_sha},
        "partitions": {"run": "derived/partitions/run-1", "sha256": partitions_sha},
        "view_runs": view_runs,
        "wordcheck_run": {"run": WORDCHECK_RUN, "sha256": transcripts_sha},
        "cache_runs": cache_runs,
        "zeroshot_scores": {"run": ZEROSHOT_RUN, "sha256": zeroshot_sha},
    }
    evaluate = yaml.safe_load((CONFIGS / "evaluate.yaml").read_text(encoding="utf-8"))
    evaluate["bootstrap"]["resamples"] = RESAMPLES
    workspace = EvaluateWorkspace(
        root, config, folder / "data.yaml", evaluate, folder / "evaluate.yaml", labels, partitions
    )
    workspace.write_config()
    for arm in TRAINED_ARMS:
        for round_number in ROUNDS:
            for fold in range(1, 6):
                key = (arm, round_number, fold)
                seed = [ARMS.index(arm), round_number, fold]
                table = run_scores(labels, workspace.fold_sessions(round_number, fold), arm, seed)
                if key == PLANTED_RUN:  # every x0 segment of the fold's first participant
                    first = (table["session"] == table.loc[0, "session"]) & (table["view"] == "x0")
                    table.loc[first, "z"] = PLANTED_Z
                scores_sha = write_parquet(workspace.folder(key) / "scores.parquet", table)
                write_json(
                    workspace.folder(key) / "run.json", train_record(workspace, key, scores_sha)
                )
    workspace.pin(version_runs(VERSIONS[0].name))
    return workspace


# Running the job ---------------------------------------------------------------------------------


@dataclass
class Ended:
    """A run of the analysis job that ended, and what it wrote to its two streams."""

    workspace: EvaluateWorkspace
    out: str
    err: str
    ended: str  # "done" or "failed"
    folder: Path

    @property
    def info(self) -> dict[str, Any]:
        return json.loads((self.folder / "run.json").read_text(encoding="utf-8"))

    @property
    def streams(self) -> str:
        return self.out + self.err

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.folder / name, keep_default_na=False)


def patch_job(
    module: ModuleType,
    workspace: EvaluateWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    git: dict[str, Any] | None = None,
    installed: str = INSTALLED,
) -> None:
    """Point the job at the workspace, a clean checkout at v0.1.1 and code of that version."""
    monkeypatch.setattr(sys, "unraisablehook", sys.unraisablehook)  # main() replaces it
    monkeypatch.setattr(module, "CONFIG", workspace.config_path)
    monkeypatch.setattr(module, "EVALUATE", workspace.evaluate_path)
    monkeypatch.setattr(module, "git_state", lambda: copy.deepcopy(git or CHECKOUT))
    monkeypatch.setattr(module, "installed_version", lambda: installed)
    monkeypatch.setattr(module, "stop_on_sigterm", lambda: None)
    monkeypatch.setenv("LBW_WORKSPACE", str(workspace.root))


def run_job(
    workspace: EvaluateWorkspace,
    *args: str,
    job: str = JOB,
    git: dict[str, Any] | None = None,
    installed: str = INSTALLED,
    env: dict[str, str | None] | None = None,
    prepare: Callable[[ModuleType, pytest.MonkeyPatch], None] | None = None,
) -> Ended:
    """Run the analysis job on the workspace in a patch context of its own; it may fail."""
    out, err = io.StringIO(), io.StringIO()
    ended = "done"
    with pytest.MonkeyPatch.context() as monkeypatch:
        module = load_script("evaluate")
        patch_job(module, workspace, monkeypatch, git, installed)
        monkeypatch.setenv("SLURM_JOB_ID", job)
        for name, value in (env or {}).items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        monkeypatch.setattr(sys, "argv", ["evaluate.py", *args])
        if prepare is not None:
            prepare(module, monkeypatch)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                module.main()
            except SystemExit:
                ended = "failed"
    mode = "trial" if "--trial" in args else "run"
    folder = workspace.root / "derived" / "evaluate" / f"{mode}-{job}"
    return Ended(workspace, out.getvalue(), err.getvalue(), ended, folder)
