"""The power simulation job (scripts/simulate_power.py), on synthetic tables in a fake checkout.

The job runs on power_fakes' synthetic tables with the layout of the pilot analysis's
summary.csv and fold_values.csv, pinned in a tiny power.yaml, in a clean checkout at the
newest release tag, as a Slurm job of 2 CPUs. Its tables must be the library's tables of the
same inputs; every guard and pin refusal, the files it reads (the checkout's tables, never
the workspace), the stop of the design check, the stop of a fit that misses before any
study, the resample check's fallback, the workers' measured threads, what may leave OMNI
and the job file are checked one by one (docs/pilot_diagnostics.md, section "The main
study's power"). Never the published tables: only the job reads them, after the release.
"""

import contextlib
import copy
import hashlib
import io
import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

import power_fakes as fakes
from listening_beyond_words import power
from listening_beyond_words.reports import write_table
from listening_beyond_words.runinfo import StepError
from wordcheck_fakes import assert_only_aggregates_leave
from x1_fakes import REPO, load_script

JOB = "992"
CHECKOUT = {
    "commit": "e" * 40,
    "dirty": False,
    "describe": "v0.1.1",
    "tags": ["v0.1.1"],
    "release_tags": ["v0.1.0", "v0.1.1"],
}
INSTALLED = "0.1.1"
ORDER = '(pilot diagnostics, section "Order")'
DECOYS = ("reports/pilot/summary.csv", "derived/labels/run-1/labels.parquet", "derived/x.csv")


@dataclass
class Inputs:
    """A fake checkout with the two tables and power.yaml, and an empty workspace."""

    checkout: Path
    workspace: Path
    summary: Path
    folds: Path
    config: Path


def make_inputs(
    folder: Path, table: pd.DataFrame | None = None, **blocks: dict[str, Any]
) -> Inputs:
    checkout = folder / "checkout"
    summary, folds = fakes.write_inputs(checkout / "reports" / "pilot", summary=table)
    config = checkout / "configs" / "pilot" / "power.yaml"
    config.parent.mkdir(parents=True)
    fakes.write_config(config, fakes.tiny_config(summary, folds, **blocks))
    workspace = folder / "workspace"
    workspace.mkdir()
    return Inputs(checkout, workspace, summary, folds, config)


@dataclass
class Ended:
    """A run of the job that ended, and what it wrote to its two streams."""

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


def run_power(
    inputs: Inputs,
    *args: str,
    job: str = JOB,
    git: dict[str, Any] | None = None,
    installed: str = INSTALLED,
    env: dict[str, str | None] | None = None,
    prepare: Callable[[ModuleType, pytest.MonkeyPatch], None] | None = None,
) -> Ended:
    """Run the job on the inputs in a patch context of its own; it may fail."""
    out, err = io.StringIO(), io.StringIO()
    ended = "done"
    with pytest.MonkeyPatch.context() as monkeypatch:
        module = load_script("simulate_power")
        monkeypatch.setattr(sys, "unraisablehook", sys.unraisablehook)  # main() replaces it
        monkeypatch.setattr(module, "POWER", inputs.config)
        monkeypatch.setattr(module, "SUMMARY", inputs.summary)
        monkeypatch.setattr(module, "FOLDS", inputs.folds)
        monkeypatch.setattr(module, "git_state", lambda: copy.deepcopy(git or CHECKOUT))
        monkeypatch.setattr(module, "installed_version", lambda: installed)
        monkeypatch.setattr(module, "stop_on_sigterm", lambda: None)
        monkeypatch.setenv("LBW_WORKSPACE", str(inputs.workspace))
        monkeypatch.setenv("SLURM_JOB_ID", job)
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
        for name, value in (env or {}).items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        monkeypatch.setattr(sys, "argv", ["simulate_power.py", *args])
        if prepare is not None:
            prepare(module, monkeypatch)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                module.main()
            except SystemExit:
                ended = "failed"
    folder = inputs.workspace / "derived" / "power" / f"run-{job}"
    return Ended(out.getvalue(), err.getvalue(), ended, folder)


def failure(ended: Ended) -> str:
    """The job's FAILED line."""
    lines = [line for line in ended.out.splitlines() if line.startswith("FAILED: ")]
    assert ended.ended == "failed", ended.out
    assert len(lines) == 1, ended.out
    return lines[0]


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> tuple[Inputs, Ended]:
    """One finished run, with decoy files in the workspace that nobody may read."""
    inputs = make_inputs(tmp_path_factory.mktemp("finished"))
    decoys = [inputs.workspace / name for name in DECOYS]
    for path in decoys:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("decoy\n", encoding="utf-8")
        path.chmod(0)
    try:
        ended = run_power(inputs)
    finally:
        for path in decoys:
            path.chmod(0o600)
    assert ended.ended == "done", ended.out + ended.err
    return inputs, ended


def expected_tables(inputs: Inputs, resamples: int, folder: Path) -> dict[str, bytes]:
    """The library's tables of the same inputs, computed here without the job."""
    config = fakes.tiny_config(inputs.summary, inputs.folds)
    settings = power.PowerSettings.from_config(config)
    summary = pd.read_csv(inputs.summary, dtype=str, keep_default_na=False)
    folds = pd.read_csv(inputs.folds, dtype=str, keep_default_na=False)
    pilots, design = power.read_pilot(summary, folds, settings)
    scenarios = power.fit_views(pilots, settings, design, fakes.Serial(), resamples)
    outcomes = power.simulate_views(scenarios, settings, design, fakes.Serial(), resamples)
    folder.mkdir()
    tables = {
        "power.csv": power.power_table(outcomes, settings),
        "power_fit.csv": power.fit_table(scenarios, outcomes, pilots, settings),
    }
    for name, table in tables.items():
        write_table(table, folder / name)
    return {name: (folder / name).read_bytes() for name in tables}


def test_the_job_writes_the_librarys_tables_and_its_record(
    finished: tuple[Inputs, Ended], tmp_path: Path
) -> None:
    inputs, ended = finished
    info = ended.info
    resamples = info["results"]["resample_check"]["resamples_used"]
    assert resamples == 40  # the check passes (max_changed 1.0): the simulation's resamples
    expected = expected_tables(inputs, resamples, tmp_path / "expected")
    assert sorted(path.name for path in ended.folder.iterdir()) == [
        "power.csv",
        "power_fit.csv",
        "run.json",
    ]
    for name, data in expected.items():
        assert (ended.folder / name).read_bytes() == data
        assert info["outputs"][name] == hashlib.sha256(data).hexdigest()
    assert (info["status"], info["step"], info["run_id"]) == (
        "done",
        "simulate_power",
        f"run-{JOB}",
    )
    assert (info["release"], info["installed_version"], info["git"]) == (
        "v0.1.1",
        "0.1.1",
        CHECKOUT,
    )
    assert info["config"]["sha256"] == hashlib.sha256(inputs.config.read_bytes()).hexdigest()
    assert info["inputs"] == {
        "summary": {"file": str(inputs.summary), "sha256": fakes.sha256(inputs.summary)},
        "folds": {"file": str(inputs.folds), "sha256": fakes.sha256(inputs.folds)},
    }
    workers = info["workers"]
    assert list(workers) == ["count", "start", "environment", "thread_pools"]
    assert (workers["count"], workers["start"]) == (2, "spawn")
    assert workers["environment"] == {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    # Measured in a worker: every pool that threadpoolctl sees runs one thread (on a Mac it
    # sees none, as it cannot see Apple's Accelerate).
    assert all(pool["num_threads"] == 1 for pool in workers["thread_pools"])
    assert "workers: 2, started by spawn, each with one BLAS thread" in ended.out
    assert info["seeds"]["seed"] == 110
    assert info["seeds"]["streams"]["studies"] == (
        "runinfo.item_seed(110, <view>, <scenario>, <block>)"
    )
    assert info["settings"]["design"] == {
        "positives": [12, 12, 11, 11, 11],
        "negatives": [27, 27, 26, 26, 26],
        "pilot_rounds": 2,
        "main_rounds": 3,
    }
    results = info["results"]
    assert results["design_check"]["passed"] is True
    assert set(results["design_check"]["pilot_delta_sd"]) == {"low", "middle", "high"}
    assert results["resample_check"]["passed"] is True
    assert set(results["design_main_study"]) == {"low", "middle", "high"}
    assert set(results["views"]) == {"x0", "rev"}
    x0 = results["views"]["x0"]
    assert list(x0) == [
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
    assert x0["rho_1"]["r"] == 1.0
    assert "green_pilots_at_0" in x0["rho_1"]
    assert "green_pilots_at_0" not in results["views"]["rev"]["rho_1"]
    assert len(results["power_line"]) == 6  # two views at three levels
    assert results["limits"] == list(load_script("simulate_power").LIMITS)
    assert results["rows"] == {"power.csv": 660, "power_fit.csv": 20}
    assert list(results["minutes"]) == ["inputs", "checks", "fits", "design", "studies", "tables"]
    out = ended.out
    assert f"pins: {inputs.summary} and {inputs.folds} match power.yaml; 57 positive and 132" in out
    assert "design check (B0's AUROC 0.65, 60 pilots per scenario)" in out
    assert "every table uses 40 resamples" in out
    assert "the study design's figures for the main study, printed beside and not checked" in out
    assert "power line, x0, one-sided 0.01667: 80% power at a true gain of " in out
    assert "limits: rev stands for one mask" in out
    assert (
        f"tables: power.csv (660 rows) and power_fit.csv (20 rows) in derived/power/run-{JOB}"
        in out
    )


def test_the_job_reads_its_tables_from_the_checkout_and_no_file_of_the_workspace(
    finished: tuple[Inputs, Ended], tmp_path: Path
) -> None:
    # The finished run had every decoy of the workspace unreadable, a summary.csv among them.
    inputs, ended = finished
    assert ended.ended == "done"
    assert ended.info["inputs"]["summary"]["file"] == str(inputs.summary)
    # The control: the job pointed at the workspace's unreadable copy stops.
    other = make_inputs(tmp_path)
    decoy = other.workspace / "reports" / "pilot" / "summary.csv"
    decoy.parent.mkdir(parents=True)
    shutil.copy(other.summary, decoy)
    decoy.chmod(0)
    try:
        control = run_power(
            other, prepare=lambda module, patch: patch.setattr(module, "SUMMARY", decoy)
        )
    finally:
        decoy.chmod(0o600)
    assert failure(control).startswith("FAILED: PermissionError")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {"env": {"SLURM_JOB_ID": None}},
            "submit this as a job: scripts/slurm/submit.sh simulate_power",
        ),
        (
            {"git": {**CHECKOUT, "dirty": True}},
            f"the checkout has uncommitted changes: the power simulation runs only from a release {ORDER}",
        ),
        (
            {"git": {**CHECKOUT, "tags": [], "release_tags": []}},
            "no v0.1.* tag exists in the clone: the power simulation runs only from the release"
            f" that holds its code {ORDER}",
        ),
        (
            {"git": {**CHECKOUT, "tags": ["v0.1.0"], "describe": "v0.1.0"}},
            "the checkout (v0.1.0) is not at the newest release tag v0.1.1: the power simulation"
            " runs only from it",
        ),
        (
            {"git": {**CHECKOUT, "release_tags": ["v0.1.0", "v0.1.1", "v0.1.x"]}},
            "the tag v0.1.x matches v0.1.* but is no release",
        ),
        ({"installed": "0.1.0"}, "the environment holds version 0.1.0 of the code, not 0.1.1"),
        ({"args": ("--trial",)}, "usage: simulate_power.py (no arguments)"),
        (
            {"env": {"SLURM_CPUS_PER_TASK": "0"}},
            "SLURM_CPUS_PER_TASK must be a whole number of at least 1, not '0'",
        ),
    ],
)
def test_the_job_refuses_anything_but_a_clean_slurm_job_at_the_newest_release(
    tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    inputs = make_inputs(tmp_path)
    ended = run_power(
        inputs,
        *change.get("args", ()),
        git=change.get("git"),
        installed=change.get("installed", INSTALLED),
        env=change.get("env"),
    )
    assert message in failure(ended)
    assert not (inputs.workspace / "derived" / "power").exists()


@pytest.mark.parametrize("table", ["summary", "folds"])
def test_a_table_that_does_not_match_its_pin_is_refused_before_anything_else(
    tmp_path: Path, table: str
) -> None:
    inputs = make_inputs(tmp_path)
    path = inputs.summary if table == "summary" else inputs.folds
    path.write_bytes(path.read_bytes() + b"\n")  # the same table, but other bytes
    ended = run_power(inputs)
    assert f"{path}: SHA-256 " in failure(ended)
    assert f"does not match its pin in power.yaml (inputs_sha256.{table}," in failure(ended)
    info = ended.info
    assert info["status"] == "failed"
    assert info["error"]["type"] == "StepError"
    assert info["results"] == {}
    assert sorted(item.name for item in ended.folder.iterdir()) == ["errors.log", "run.json"]
    assert "design check" not in ended.out


def test_a_missing_table_is_named_and_a_key_given_twice_refused(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    inputs.folds.unlink()
    assert f"{inputs.folds} does not exist: the simulation reads it from the checkout" in failure(
        run_power(inputs, job="994")
    )
    text = inputs.config.read_text(encoding="utf-8")
    inputs.config.write_text(
        text.replace("views:", "views: [x0, rev]\nviews:", 1), encoding="utf-8"
    )
    assert "views" in failure(run_power(inputs, job="995"))


def test_a_design_check_that_misses_stops_the_job_as_a_code_fault(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, design_check={"max_gap": 0.0})
    ended = run_power(inputs)
    line = failure(ended)
    assert "the design check misses, a code fault that stops the job" in line
    assert "the pilot's Δ in the low scenario: SD" in line
    info = ended.info
    assert info["status"] == "failed"
    assert info["results"]["design_check"]["passed"] is False
    assert "resample_check" not in info["results"]
    assert not (ended.folder / "power.csv").exists()


def test_a_fit_that_misses_stops_the_job_before_the_designs_scenarios_and_any_study(
    tmp_path: Path,
) -> None:
    # An agreement on x0 that the model cannot reach: its first fit, the lowest ρ, misses.
    numbers = copy.deepcopy(fakes.NUMBERS)
    numbers["x0"]["agreement_b0_cpg"] = (0.9999, 0.9999, 0.9999, 0.9998, 0.99995, None)
    inputs = make_inputs(tmp_path, fakes.summary_table(numbers))
    ended = run_power(inputs)
    line = failure(ended)
    assert "x0: the lowest ρ for the agreement: the model reaches" in line
    assert "which misses the target 0.9999 by more than 0.001, so the job stops" in line
    results = ended.info["results"]
    assert list(results["minutes"]) == ["inputs", "checks"]  # stage 5 stopped
    assert results["views"] == {}
    assert "design_main_study" not in results
    assert "the study design's figures for the main study" not in ended.out
    assert sorted(path.name for path in ended.folder.iterdir()) == ["errors.log", "run.json"]


def test_a_fit_that_misses_on_the_last_view_stops_the_job_before_any_study(
    tmp_path: Path,
) -> None:
    # Every fit of every view runs before any study: a miss on rev, the last view, stops the
    # job after x0's fits and before any of x0's studies.
    numbers = copy.deepcopy(fakes.NUMBERS)
    numbers["rev"]["agreement_b0_cpg"] = (-0.5, -0.51, -0.49, -0.52, -0.48, None)
    inputs = make_inputs(tmp_path, fakes.summary_table(numbers))
    ended = run_power(inputs)
    line = failure(ended)
    assert "rev: the lowest ρ for the agreement: the model reaches" in line
    assert "which misses the target -0.5000 by more than 0.001, so the job stops" in line
    results = ended.info["results"]
    assert list(results["minutes"]) == ["inputs", "checks"]  # stage 5 stopped
    assert set(results["views"]) == {"x0"}
    assert "gain_80" not in str(results["views"])
    assert "design_main_study" not in results
    assert sorted(path.name for path in ended.folder.iterdir()) == ["errors.log", "run.json"]


def test_a_sensitivity_row_that_misses_its_target_is_reported_as_not_reached(
    tmp_path: Path,
) -> None:
    # The low end of x0's agreement interval out of the model's reach: that row keeps r where
    # the model comes closest, is marked not reached, and the job goes on.
    numbers = copy.deepcopy(fakes.NUMBERS)
    mean, first, second, _, high, se = numbers["x0"]["agreement_b0_cpg"]
    numbers["x0"]["agreement_b0_cpg"] = (mean, first, second, -0.5, high, se)
    inputs = make_inputs(tmp_path, fakes.summary_table(numbers))
    ended = run_power(inputs)
    assert ended.ended == "done", ended.out + ended.err
    fits = pd.read_csv(ended.folder / "power_fit.csv", keep_default_na=False)
    missed = fits[fits["target_met"] == "not reached"]
    assert list(zip(missed["view"], missed["scenario"], strict=True)) == [
        ("x0", "agreement_low_80")
    ]
    assert set(fits["target_met"]) == {"met", "not reached"}
    assert "x0, agreement_low_80: not reached: the model comes closest at this r" in ended.out
    views = ended.info["results"]["views"]
    assert views["x0"]["agreement_low_80"]["target_met"] is False
    assert views["x0"]["agreement_high_80"]["target_met"] is True
    assert "gain_80" in views["x0"]["agreement_low_80"]  # simulated all the same
    lines = [line for line in ended.out.splitlines() if line.startswith("power line, ")]
    assert lines
    for line in lines:
        assert ("(not reached)" in line) == line.startswith("power line, x0,")
        assert line.count("(not reached)") <= 1


def test_the_resample_check_sets_the_resamples_of_every_table(tmp_path: Path) -> None:
    inputs = make_inputs(
        tmp_path,
        simulation={"resamples": 20},
        resample_check={"studies": 30, "resamples": 400, "max_changed": 0.0},
    )

    def stop_after_the_checks(module: ModuleType, patch: pytest.MonkeyPatch) -> None:
        def stop(*_: Any) -> None:
            raise StepError("stopped after the checks")

        patch.setattr(module, "run_fits", stop)

    ended = run_power(inputs, prepare=stop_after_the_checks)
    assert "stopped after the checks" in failure(ended)
    check = ended.info["results"]["resample_check"]
    assert (check["passed"], check["resamples_used"]) == (False, 60)
    assert check["resamples"] == [20, 400]
    assert (
        check["compared"] == "the decision on the first resamples of each study against all of them"
    )
    assert set(check["changed"]) == {"0", "0.06"}
    assert set(check["changed"]["0.06"]) == {"0.01667", "0.025", "0.05"}
    assert "differ between the first 20 of each study's 400 resamples and all of them" in ended.out
    assert "fail; every table uses 60 resamples" in ended.out


def test_two_runs_write_byte_identical_tables(finished: tuple[Inputs, Ended]) -> None:
    inputs, first = finished
    second = run_power(inputs, job="993")
    assert second.ended == "done", second.out + second.err
    for name in ("power.csv", "power_fit.csv"):
        assert (first.folder / name).read_bytes() == (second.folder / name).read_bytes()


def test_only_aggregates_leave_the_job(finished: tuple[Inputs, Ended], tmp_path: Path) -> None:
    _, ended = finished
    assert_only_aggregates_leave(ended.folder / "run.json", ended.streams)
    refused = make_inputs(tmp_path)
    refused.summary.write_text("number,view\n", encoding="utf-8")
    stopped = run_power(refused, job="996")
    assert stopped.ended == "failed"
    assert_only_aggregates_leave(stopped.folder / "run.json", stopped.streams)


def test_the_leak_check_fails_on_a_planted_id(
    finished: tuple[Inputs, Ended], tmp_path: Path
) -> None:
    _, ended = finished
    with pytest.raises(AssertionError):
        assert_only_aggregates_leave(ended.folder / "run.json", ended.streams + "session 305\n")
    info = ended.info
    info["results"]["session"] = 305
    planted = tmp_path / "run.json"
    planted.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_only_aggregates_leave(planted, ended.streams)


def test_the_job_file_asks_for_16_cpus_on_short_for_at_most_2_5_hours() -> None:
    text = (REPO / "scripts" / "slurm" / "simulate_power.sbatch").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    assert [line for line in lines if line.startswith("#SBATCH")] == [
        "#SBATCH --job-name=lbw-simulate-power",
        "#SBATCH --partition=short",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=16",
        "#SBATCH --mem=8G",
        "#SBATCH --time=02:30:00",
    ]
    assert lines[-3:] == [
        "source scripts/slurm/common.sh",
        "",
        'python scripts/simulate_power.py "$@"',
    ]
    assert "scripts/slurm/submit.sh simulate_power" in text
