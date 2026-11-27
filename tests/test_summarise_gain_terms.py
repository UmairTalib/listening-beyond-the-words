"""The gain-term summary (scripts/summarise_gain_terms.py), on the analysis job's workspace.

It runs on the light synthetic workspace of evaluate_fakes, whose 30 training runs have
records from training.rule_measures, in a clean checkout at the newest release tag and
outside any Slurm job, as on a login node. The table must be diagnostics.gain_term_table of
the pinned records; every guard and pin refusal, the files it opens and what may leave OMNI
are checked one by one (docs/pilot_diagnostics.md, section "Gain-term summary").
"""

import contextlib
import copy
import hashlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import evaluate_fakes as fakes
from evaluate_fakes import CHECKOUT, EvaluateWorkspace, make_evaluate_workspace, run_name
from listening_beyond_words.diagnostics import gain_term_table
from listening_beyond_words.reports import write_table
from wordcheck_fakes import assert_only_aggregates_leave
from x1_fakes import load_script

STAMP = "20261128T090000Z"
KEY = ("b1", 2, 4)
ORDER = '(pilot diagnostics, section "Order")'


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> EvaluateWorkspace:
    """The light workspace, with all 30 runs done and pinned."""
    return make_evaluate_workspace(tmp_path_factory.mktemp("built"))


@pytest.fixture
def workspace(built: EvaluateWorkspace, tmp_path: Path) -> EvaluateWorkspace:
    return built.copy_to(tmp_path)


@dataclass
class Ended:
    """A run of the summary that ended, and what it wrote to its two streams."""

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


def run_summary(
    workspace: EvaluateWorkspace,
    *args: str,
    git: dict[str, Any] | None = None,
    installed: str = fakes.INSTALLED,
    stamp: str = STAMP,
) -> Ended:
    """Run the summary on the workspace, as on a login node at v0.1.1; it may fail."""
    out, err = io.StringIO(), io.StringIO()
    ended = "done"
    with pytest.MonkeyPatch.context() as monkeypatch:
        module = load_script("summarise_gain_terms")
        monkeypatch.setattr(sys, "unraisablehook", sys.unraisablehook)  # main() replaces it
        monkeypatch.setattr(module, "CONFIG", workspace.config_path)
        monkeypatch.setattr(module, "EVALUATE", workspace.evaluate_path)
        monkeypatch.setattr(module, "git_state", lambda: copy.deepcopy(git or CHECKOUT))
        monkeypatch.setattr(module, "installed_version", lambda: installed)
        monkeypatch.setattr(module, "stop_on_sigterm", lambda: None)
        monkeypatch.setattr(module, "utc_stamp", lambda: stamp)
        monkeypatch.setenv("LBW_WORKSPACE", str(workspace.root))
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)  # a login node: no Slurm job
        monkeypatch.setattr(sys, "argv", ["summarise_gain_terms.py", *args])
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                module.main()
            except SystemExit:
                ended = "failed"
    folder = workspace.root / "derived" / "gain_terms" / f"run-{stamp}"
    return Ended(out.getvalue(), err.getvalue(), ended, folder)


def failure(ended: Ended) -> str:
    """The summary's FAILED line."""
    lines = [line for line in ended.out.splitlines() if line.startswith("FAILED: ")]
    assert ended.ended == "failed", ended.out
    assert len(lines) == 1, ended.out
    return lines[0]


def pinned_records(workspace: EvaluateWorkspace) -> dict[tuple[str, int, int], dict[str, Any]]:
    """The pinned runs' records, read here without the job."""
    return {key: workspace.record(key) for key in workspace.pinned()}


def test_the_summary_reads_the_pinned_records_and_writes_its_table_and_record(
    workspace: EvaluateWorkspace, tmp_path: Path
) -> None:
    ended = run_summary(workspace)
    assert ended.ended == "done", ended.out
    expected = tmp_path / "expected.csv"
    write_table(gain_term_table(pinned_records(workspace)), expected)
    written = ended.folder / "gain_terms.csv"
    assert written.read_bytes() == expected.read_bytes()
    assert sorted(path.name for path in ended.folder.iterdir()) == ["gain_terms.csv", "run.json"]
    info = ended.info
    assert (info["status"], info["step"], info["run_id"]) == (
        "done",
        "summarise_gain_terms",
        f"run-{STAMP}",
    )
    assert (info["release"], info["installed_version"], info["git"]) == (
        "v0.1.1",
        "0.1.1",
        CHECKOUT,
    )
    for name, path in (("config", workspace.evaluate_path), ("data_config", workspace.config_path)):
        assert info[name] == {
            "file": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    runs = info["inputs"]["runs"]
    for arm, round_number, fold in workspace.pinned():
        pin = workspace.evaluate["runs"][arm][f"r{round_number}"][f"f{fold}"]
        entry = runs[arm][f"r{round_number}"][f"f{fold}"]
        assert (entry["run"], entry["record_sha256"]) == (pin["run"], pin["record_sha256"])
        first = fakes.first_run(arm, round_number, fold)
        assert entry["counts_by"] == (
            "the first runs' commit 271cce6"
            if first
            else "the newest release tag of its clone, v0.1.0"
        )
    assert info["results"]["runs"] == {"b0": 10, "b1": 10, "cpg": 10}
    assert info["results"]["epochs"] == [1, 2, 3]
    assert info["results"]["rows"] == 3 * (4 + 29 + 31)
    # Every other step's bags are active in the fake logs: a pooled share of one half.
    halves = {str(epoch): 0.5 for epoch in (1, 2, 3)}
    assert info["results"]["pooled_active_share"] == {"b1": halves, "cpg": halves}
    assert info["outputs"] == {"gain_terms.csv": hashlib.sha256(written.read_bytes()).hexdigest()}
    assert (
        "pins: 30 kept runs match their pins and records (10 of B0, 10 of B1, 10 of CPG); only"
        " their run.json files are read\n" in ended.out
    )
    assert "pooled active share, CPG: epoch 1 0.5000, epoch 2 0.5000, epoch 3 0.5000\n" in ended.out
    assert f"table: gain_terms.csv (192 rows) in derived/gain_terms/run-{STAMP}\n" in ended.out


def test_the_summary_opens_no_file_of_the_workspace_but_the_pinned_records(
    workspace: EvaluateWorkspace,
) -> None:
    # Scores, labels, partitions, caches and every other file of the workspace cannot be
    # opened at all; the summary still runs.
    records = {workspace.folder(key) / "run.json" for key in workspace.pinned()}
    closed = [path for path in workspace.root.rglob("*") if path.is_file() and path not in records]
    assert any(path.name == "scores.parquet" for path in closed)
    assert any(path.name == "labels.parquet" for path in closed)
    for path in closed:
        path.chmod(0)
    record = workspace.folder(KEY) / "run.json"
    try:
        ended = run_summary(workspace)
        # The control: a file closed so cannot be read here, so the summary stops.
        record.chmod(0)
        control = run_summary(workspace, stamp="20261128T090001Z")
    finally:
        for path in [*closed, record]:
            path.chmod(0o600)
    assert ended.ended == "done", ended.out
    assert failure(control).startswith("FAILED: PermissionError")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {"git": {**CHECKOUT, "dirty": True}},
            "the checkout has uncommitted changes: the gain-term summary runs only from a"
            f" release {ORDER}",
        ),
        (
            {"git": {**CHECKOUT, "tags": [], "release_tags": []}},
            "no v0.1.* tag exists in the clone: the gain-term summary runs only from the"
            f" release that holds its code {ORDER}",
        ),
        (
            {"git": {**CHECKOUT, "tags": ["v0.1.0"], "describe": "v0.1.0"}},
            "the checkout (v0.1.0) is not at the newest release tag v0.1.1: the gain-term"
            " summary runs only from it",
        ),
        (
            {"git": {**CHECKOUT, "release_tags": ["v0.1.0", "v0.1.1", "v0.1.x"]}},
            "the tag v0.1.x matches v0.1.* but is no release",
        ),
        ({"installed": "0.1.0"}, "the environment holds version 0.1.0 of the code, not 0.1.1"),
        ({"args": ("--trial",)}, "usage: summarise_gain_terms.py (no arguments)"),
    ],
)
def test_the_summary_refuses_anything_but_a_clean_checkout_at_the_newest_release(
    workspace: EvaluateWorkspace, change: dict[str, Any], message: str
) -> None:
    ended = run_summary(
        workspace,
        *change.get("args", ()),
        git=change.get("git"),
        installed=change.get("installed", fakes.INSTALLED),
    )
    assert message in failure(ended)
    assert not (workspace.root / "derived" / "gain_terms").exists()


def test_the_summary_refuses_while_no_run_is_pinned_or_a_pin_is_given_twice(
    workspace: EvaluateWorkspace,
) -> None:
    pins = workspace.evaluate["runs"]
    workspace.evaluate["runs"] = {}
    workspace.write_evaluate()
    assert "no training run is pinned in evaluate.yaml (runs)" in failure(run_summary(workspace))
    workspace.evaluate["runs"] = pins
    workspace.write_evaluate()
    text = workspace.evaluate_path.read_text(encoding="utf-8")
    twice = text.replace("  b1:\n", "  b0:\n    r9: {}\n  b1:\n", 1)
    workspace.write_evaluate(twice)
    assert "is given twice" in failure(run_summary(workspace))
    assert not (workspace.root / "derived" / "gain_terms").exists()


def test_a_record_that_does_not_match_its_pin_is_refused_before_any_table(
    workspace: EvaluateWorkspace,
) -> None:
    path = workspace.folder(KEY) / "run.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["results"]["rules"]["epochs"]["2"]["hinge"] = 0.125  # after the pins
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    ended = run_summary(workspace)
    assert f"{run_name(*KEY)}: its run.json does not match its pin (record_sha256)" in failure(
        ended
    )
    assert ended.info["status"] == "failed"
    assert ended.info["error"]["message"].endswith("does not match its pin (record_sha256)")
    assert not (ended.folder / "gain_terms.csv").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda record: record.update(allow_dirty=True), "was started with --allow-dirty"),
        (lambda record: record.update(status="failed"), "did not finish"),
        (
            lambda record: record["inputs"]["labels"].update(sha256="f" * 64),
            "was made from other labels than data.yaml pins now",
        ),
        (
            lambda record: record["results"]["rules"]["epochs"]["2"].pop("hinge"),
            "b1, round 2, fold 4, epoch 2: the record holds no hinge",
        ),
    ],
)
def test_a_pinned_run_that_does_not_count_or_lacks_a_measure_is_refused(
    workspace: EvaluateWorkspace, change: Any, message: str
) -> None:
    workspace.change_record(KEY, change)  # pinned again: only the change counts
    ended = run_summary(workspace)
    assert message in failure(ended)
    assert not (ended.folder / "gain_terms.csv").exists()


def test_two_runs_write_byte_identical_tables(workspace: EvaluateWorkspace) -> None:
    first = run_summary(workspace)
    second = run_summary(workspace, stamp="20261128T090001Z")
    assert first.ended == second.ended == "done", first.out + second.out
    tables = [ended.folder / "gain_terms.csv" for ended in (first, second)]
    assert tables[0].read_bytes() == tables[1].read_bytes()


@pytest.mark.parametrize("case", ["done", "a refusal"])
def test_only_aggregates_leave_the_summary(workspace: EvaluateWorkspace, case: str) -> None:
    # A record may hold anything outside its per-epoch measures; none of it leaves.
    def plant(record: dict[str, Any]) -> None:
        record["planted"] = {"session": fakes.PLANTED_SESSION, "segment": "305_007"}
        if case == "a refusal":
            record["results"]["rules"]["epochs"]["1"]["loss"] = "high"

    workspace.change_record(KEY, plant)
    ended = run_summary(workspace)
    assert ended.ended == ("failed" if case == "a refusal" else "done"), ended.out
    assert_only_aggregates_leave(ended.folder / "run.json", ended.streams)
    texts = [ended.streams] + [path.read_text("utf-8") for path in ended.folder.iterdir()]
    for text in texts:
        assert "305_007" not in text
        assert str(fakes.PLANTED_SESSION) not in text


def test_the_leak_check_fails_on_a_planted_id(workspace: EvaluateWorkspace) -> None:
    ended = run_summary(workspace)
    assert ended.ended == "done", ended.out
    assert_only_aggregates_leave(ended.folder / "run.json", ended.streams)
    with pytest.raises(AssertionError):
        assert_only_aggregates_leave(ended.folder / "run.json", ended.streams + "session 305\n")
    info = ended.info
    info["results"]["session"] = fakes.PLANTED_SESSION
    (ended.folder / "run.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_only_aggregates_leave(ended.folder / "run.json", ended.streams)
