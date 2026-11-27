"""The listing of the kept training runs (scripts/list_runs.py), on the analysis job's workspace.

It runs on the light synthetic workspace of evaluate_fakes, with the 30 training runs done:
the block it prints pins the earliest done run of each fold-run that counts, the analysis
job accepts it, and every run folder left out is named with the reason (docs/pilot_analysis.md,
section "Runs and pins").
"""

import contextlib
import io
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from evaluate_fakes import (
    EvaluateWorkspace,
    make_evaluate_workspace,
    run_job,
    run_name,
)
from listening_beyond_words.evaluation import VERSIONS, EvaluateSettings
from x1_fakes import load_script

SEGMENT_ID = re.compile(r"\b3\d\d_\d{3}\b")
KEY = ("b1", 2, 4)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> EvaluateWorkspace:
    """The light workspace, with all 30 runs done and pinned."""
    return make_evaluate_workspace(tmp_path_factory.mktemp("built"))


@pytest.fixture
def workspace(built: EvaluateWorkspace, tmp_path: Path) -> EvaluateWorkspace:
    return built.copy_to(tmp_path)


def run_listing(workspace: EvaluateWorkspace, *args: str, fails: bool = False) -> tuple[str, str]:
    """Run scripts/list_runs.py on the workspace; return its stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as monkeypatch:
        module = load_script("list_runs")
        monkeypatch.setattr(module, "CONFIG", workspace.config_path)
        monkeypatch.setattr(module, "EVALUATE", workspace.evaluate_path)
        monkeypatch.setenv("LBW_WORKSPACE", str(workspace.root))
        monkeypatch.setattr(sys, "argv", ["list_runs.py", *args])
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            if fails:
                with pytest.raises(SystemExit) as stopped:
                    module.main()
                assert stopped.value.code == 1
            else:
                module.main()
    return out.getvalue(), err.getvalue()


def sibling(workspace: EvaluateWorkspace, day: str, change: Any = None) -> Path:
    """A copy of KEY's run folder started on another day, with its record changed."""
    first = workspace.folder(KEY)
    folder = first.parent / first.name.replace("20261127", day)
    shutil.copytree(first, folder)
    record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    record["run_id"] = folder.name
    if change is not None:
        change(record)
    (folder / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return folder


def test_the_listing_prints_a_block_that_the_job_accepts(workspace: EvaluateWorkspace) -> None:
    pinned = workspace.evaluate["runs"]
    workspace.evaluate["runs"] = {}
    workspace.write_evaluate()
    out, err = run_listing(workspace)
    assert err == ""
    listed = yaml.safe_load(out)
    assert listed["runs"] == pinned
    assert "# version: all 30 runs (30 runs)\n" in out
    assert "# not pinned" not in out
    text = workspace.evaluate_path.read_text(encoding="utf-8").replace("runs: {}\n", out)
    workspace.write_evaluate(text)
    assert EvaluateSettings.from_config(yaml.safe_load(text)).version is VERSIONS[0]
    ended = run_job(workspace)
    assert ended.ended == "done", ended.out
    assert not SEGMENT_ID.search(out)


def test_the_listing_pins_the_earlier_of_two_runs_and_reports_the_others(
    workspace: EvaluateWorkspace,
) -> None:
    later = sibling(
        workspace, "20261128", lambda r: r.update(finished_utc="2026-11-28T11:00:00+00:00")
    )
    dirty = sibling(workspace, "20261129", lambda r: r.update(allow_dirty=True))
    failed = sibling(workspace, "20261130", lambda r: r.update(status="failed"))
    out, _ = run_listing(workspace)
    listed = yaml.safe_load(out)["runs"]
    assert listed["b1"]["r2"]["f4"]["run"] == run_name(*KEY)
    assert listed == workspace.evaluate["runs"]
    notes = [line for line in out.splitlines() if line.startswith("# not pinned: ")]
    assert len(notes) == 3
    relative = f"runs/pilot/{later.name}"
    assert f"# not pinned: {relative}: a later done run of b1, round 2, fold 4; the earlier" in out
    assert f"runs/pilot/{dirty.name} was started with --allow-dirty" in out
    assert f"runs/pilot/{failed.name} did not finish" in out
    assert "zeroshot" not in out


def test_of_two_runs_the_listing_pins_the_one_that_finished_first(
    workspace: EvaluateWorkspace,
) -> None:
    # Started later (its folder's name), but finished earlier: it is the earlier run.
    quicker = sibling(
        workspace, "20261128", lambda r: r.update(finished_utc="2026-11-27T10:30:00+00:00")
    )
    out, _ = run_listing(workspace)
    assert yaml.safe_load(out)["runs"]["b1"]["r2"]["f4"]["run"] == f"runs/pilot/{quicker.name}"
    assert f"# not pinned: {run_name(*KEY)}: a later done run of b1, round 2, fold 4" in out


def test_the_listing_reports_a_set_that_is_no_version(workspace: EvaluateWorkspace) -> None:
    shutil.rmtree(workspace.folder(("cpg", 2, 5)))
    out, _ = run_listing(workspace)
    assert "# version: none: the 29 pinned runs are none of the pilot's versions" in out


def lacks_finish(record: dict[str, Any]) -> None:
    del record["finished_utc"]


def lacks_outputs(record: dict[str, Any]) -> None:
    del record["outputs"]


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("no run.json", " has no run.json: it never started its record"),
        ("no JSON", ": its run.json cannot be read as JSON"),
        ("no JSON object", ": its run.json is not a JSON object"),
        ("no outputs", ": its run.json lists no scores.parquet with its SHA-256"),
        ("round 3", " names no fold-run of the pilot"),
    ],
)
def test_the_listing_leaves_out_a_run_that_it_cannot_pin_and_says_why(
    workspace: EvaluateWorkspace, case: str, reason: str
) -> None:
    if case == "round 3":
        first = workspace.folder(KEY)
        folder = first.parent / first.name.replace("-r2-", "-r3-")
        shutil.copytree(first, folder)
    else:
        change = {"no outputs": lacks_outputs}.get(case)
        folder = sibling(workspace, "20261128", change)
        if case == "no run.json":
            (folder / "run.json").unlink()
        elif case == "no JSON":
            (folder / "run.json").write_text("{not json", encoding="utf-8")
        elif case == "no JSON object":
            (folder / "run.json").write_text("[1, 2]", encoding="utf-8")
    out, err = run_listing(workspace)
    assert err == ""
    assert yaml.safe_load(out)["runs"] == workspace.evaluate["runs"]  # the 30 runs still
    notes = [line for line in out.splitlines() if line.startswith("# not pinned: ")]
    assert len(notes) == 1
    assert notes[0].startswith(f"# not pinned: runs/pilot/{folder.name}{reason}")


def test_a_run_that_counts_but_cannot_be_ordered_leaves_its_fold_run_unpinned(
    workspace: EvaluateWorkspace,
) -> None:
    # No run of the fold-run can be told to be the earlier one, and the analysis job would
    # refuse a pin of the other run (tests/test_evaluate.py), so neither is pinned.
    unordered = sibling(workspace, "20261128", lacks_finish)
    out, err = run_listing(workspace)
    assert err == ""
    listed = yaml.safe_load(out)["runs"]
    assert "f4" not in listed["b1"]["r2"]
    assert len([arm for arm in listed.values() for runs in arm.values() for _ in runs]) == 29
    assert "# version: none: the 29 pinned runs are none of the pilot's versions" in out
    notes = [line for line in out.splitlines() if line.startswith("# not pinned: ")]
    assert notes == [
        f"# not pinned: runs/pilot/{unordered.name}: its run.json does not say when it finished"
        " (finished_utc), so it cannot be told whether it is the earlier of two runs of its"
        " fold-run",
        f"# not pinned: {run_name(*KEY)}: another run of b1, round 2, fold 4 counts but cannot be"
        " ordered, so no run of it is pinned",
    ]


def test_an_earliest_run_that_cannot_be_pinned_leaves_its_fold_run_unpinned(
    workspace: EvaluateWorkspace,
) -> None:
    # It finished first and counts, so the analysis job would refuse a pin of the later run.
    def earlier_without_outputs(record: dict[str, Any]) -> None:
        record.update(finished_utc="2026-11-27T10:30:00+00:00")
        lacks_outputs(record)

    earliest = sibling(workspace, "20261128", earlier_without_outputs)
    out, err = run_listing(workspace)
    assert err == ""
    assert "f4" not in yaml.safe_load(out)["runs"]["b1"]["r2"]
    assert "# version: none: the 29 pinned runs are none of the pilot's versions" in out
    notes = [line for line in out.splitlines() if line.startswith("# not pinned: ")]
    assert notes == [
        f"# not pinned: runs/pilot/{earliest.name}: its run.json lists no scores.parquet with its"
        " SHA-256",
        f"# not pinned: {run_name(*KEY)}: the earliest run of b1, round 2, fold 4 that counts"
        " cannot be pinned, so no run of it is pinned",
    ]


@pytest.mark.parametrize(
    ("args", "line"),
    [
        (("--all",), "FAILED: StepError: usage: list_runs.py (no arguments)"),
        ((), "FAILED: FileNotFoundError"),  # a reader's error: its type, never its message
    ],
)
def test_a_refused_listing_prints_one_failed_line_and_no_block(
    workspace: EvaluateWorkspace, args: tuple[str, ...], line: str
) -> None:
    if not args:
        workspace.config_path.unlink()
    out, _ = run_listing(workspace, *args, fails=True)
    assert out.splitlines()[0].startswith(line)
    assert len(out.splitlines()) == 1
    assert "runs:" not in out
    assert str(workspace.config_path) not in out
