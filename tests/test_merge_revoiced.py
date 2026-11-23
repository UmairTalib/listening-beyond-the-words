"""The merge of an x1 build (scripts/merge_revoiced.py), on a synthetic workspace."""

import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import soundfile as sf
import yaml

from listening_beyond_words.clips import load_view_manifest
from listening_beyond_words.revoice import TASKS
from listening_beyond_words.runinfo import item_seed, sha256_file
from listening_beyond_words.settings import Workspace
from x1_fakes import (
    ARRAY_JOB,
    CHECK_JOB,
    SECRET,
    X1Workspace,
    build_all,
    load_script,
    make_x1_workspace,
    outside_text,
    patch_script,
)

RUN = f"views/x1/run-{ARRAY_JOB}"


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> X1Workspace:
    """One synthetic workspace with a passed check run and every task of a build done."""
    workspace = make_x1_workspace(tmp_path_factory.mktemp("built") / "workspace")
    with pytest.MonkeyPatch.context() as monkeypatch:
        build_all(workspace, monkeypatch)
    return workspace


@pytest.fixture
def merge(
    built: X1Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, X1Workspace]:
    """A copy of the built workspace, and the merge script pointed at it."""
    root = tmp_path / "workspace"
    shutil.copytree(built.root, root)
    shutil.copy(built.config_path, tmp_path / "data.yaml")  # byte for byte: tasks pin its hash
    workspace = X1Workspace(
        root,
        copy.deepcopy(built.config),
        tmp_path / "data.yaml",
        built.lines,
        built.segments,
        built.words,
    )
    module = load_script("merge_revoiced")
    patch_script(module, workspace, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["merge_revoiced.py", RUN])
    return module, workspace


def written(workspace: X1Workspace) -> list[str]:
    """What a merge writes into the run folder."""
    run = workspace.root / RUN
    return sorted(
        n for n in ("manifest.parquet", "parts.parquet", "run.json") if (run / n).exists()
    )


def task_file(workspace: X1Workspace, task: int, name: str) -> Path:
    return workspace.root / RUN / "tasks" / str(task) / name


def task_of(workspace: X1Workspace, session: int) -> int:
    for task in range(TASKS):
        manifest = pd.read_parquet(task_file(workspace, task, "manifest.parquet"))
        if session in set(manifest["session"]):
            return task
    raise AssertionError(f"no task holds session {session}")


def rewrite(workspace: X1Workspace, task: int, name: str, table: pd.DataFrame) -> None:
    """Replace a task's table and its SHA-256 in the task's run.json, as a task would write it."""
    table.to_parquet(task_file(workspace, task, name), index=False)
    info_path = task_file(workspace, task, "run.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["outputs"][name] = sha256_file(task_file(workspace, task, name))
    info_path.write_text(json.dumps(info), encoding="utf-8")


def test_the_merge_checks_every_task_and_file_and_writes_the_run_last(
    merge: tuple[Any, X1Workspace], capsys: pytest.CaptureFixture[str]
) -> None:
    module, workspace = merge
    module.main()
    out, err = capsys.readouterr()
    run = workspace.root / RUN
    info = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert info["status"] == "done"
    assert info["array_job_id"] == ARRAY_JOB
    assert info["output"]["sha256"] == sha256_file(run / "manifest.parquet")
    assert info["outputs"]["parts.parquet"] == sha256_file(run / "parts.parquet")
    assert f"run: {RUN}" in out
    assert f"sha256: {info['output']['sha256']}" in out
    manifest = pd.read_parquet(run / "manifest.parquet")
    assert manifest["segment"].tolist() == sorted(workspace.segments["segment"])
    parts = pd.read_parquet(run / "parts.parquet")
    assert len(parts) == sum(workspace.segments["parts"])
    assert set(info["tasks"]) == {str(task) for task in range(TASKS)}
    assert "test render (check job on" in out
    assert ": 4 of 4" in out  # every test-render clip equals the full render here
    assert "gender, voice, PHQ-8 class: participants, segments" in out
    voices = {(row["gender"], row["voice"]) for row in info["counts"]["voices"]}
    # Women 300 (negative), 304 (negative, no segments) and 302 (positive) take af_heart,
    # af_bella and af_heart; men 303 (negative) and 301 (positive) am_michael and am_fenrir.
    assert voices == {("female", "af_heart"), ("male", "am_michael"), ("male", "am_fenrir")}
    dry = json.loads((check_run(workspace) / "run.json").read_text(encoding="utf-8"))["dry_run"]
    assert info["counts"]["words_read"] == dry["words_read"] == parts["words_read"].sum()
    assert info["counts"]["numbers_written_out"] == dry["numbers_written_out"] > 0
    assert (
        f"text: {dry['words_read']:,} words read, {dry['numbers_written_out']:,} numbers"
        " written out in words: the counts of the check run's dry run"
    ) in out
    assert SECRET.lower() not in outside_text(
        run / "run.json", run / "manifest.parquet", streams=out + err
    )
    # The merged manifest is what gets pinned, and the pinned reader takes it.
    workspace.config["view_runs"]["x1"] = {"run": RUN, "sha256": info["output"]["sha256"]}
    pinned = load_view_manifest(Workspace(workspace.root), workspace.config, "x1")
    pd.testing.assert_frame_equal(pinned, manifest)


def check_run(workspace: X1Workspace) -> Path:
    return workspace.root / "derived" / "revoice_check" / f"run-{CHECK_JOB}"


def refused(module: Any, workspace: X1Workspace, capsys: pytest.CaptureFixture[str]) -> str:
    """Run a merge that must refuse; check that it wrote nothing and return what it printed."""
    with pytest.raises(SystemExit):
        module.main()
    assert written(workspace) == []
    return capsys.readouterr().out


def test_the_merge_refuses_a_missing_task(
    merge: tuple[Any, X1Workspace], capsys: pytest.CaptureFixture[str]
) -> None:
    module, workspace = merge
    shutil.rmtree(workspace.root / RUN / "tasks" / "7")
    assert "missing ['7']" in refused(module, workspace, capsys)


def test_the_merge_refuses_a_task_that_is_not_done(
    merge: tuple[Any, X1Workspace], capsys: pytest.CaptureFixture[str]
) -> None:
    module, workspace = merge
    path = task_file(workspace, 3, "run.json")
    path.write_text(
        json.dumps(json.loads(path.read_text()) | {"status": "failed"}), encoding="utf-8"
    )
    assert "tasks not done: [3]; run the whole array again" in refused(module, workspace, capsys)


def test_the_merge_refuses_tasks_with_mixed_commits(
    merge: tuple[Any, X1Workspace], capsys: pytest.CaptureFixture[str]
) -> None:
    module, workspace = merge
    path = task_file(workspace, 5, "run.json")
    info = json.loads(path.read_text())
    info["git"] = {"commit": "d" * 40, "dirty": False}
    path.write_text(json.dumps(info), encoding="utf-8")
    assert "the tasks differ in git" in refused(module, workspace, capsys)


def test_the_merge_refuses_a_session_in_two_tasks(
    merge: tuple[Any, X1Workspace], capsys: pytest.CaptureFixture[str]
) -> None:
    module, workspace = merge
    owner = task_of(workspace, 300)
    other = (owner + 1) % TASKS
    doubled = pd.read_parquet(task_file(workspace, owner, "manifest.parquet"))
    extra = pd.concat([pd.read_parquet(task_file(workspace, other, "manifest.parquet")), doubled])
    rewrite(workspace, other, "manifest.parquet", extra)
    assert "session 300 lies in tasks" in refused(module, workspace, capsys)


def test_the_merge_refuses_a_changed_file(
    merge: tuple[Any, X1Workspace], capsys: pytest.CaptureFixture[str]
) -> None:
    module, workspace = merge
    path = workspace.root / RUN / "301" / "301_000.flac"
    samples, rate = sf.read(path, dtype="int16")
    samples[100] += 1
    path.unlink()
    sf.write(path, samples, rate, subtype="PCM_16", format="FLAC")
    assert "segment 301_000: its file does not match" in refused(module, workspace, capsys)


def test_the_merge_refuses_a_file_rewritten_with_the_same_samples(
    merge: tuple[Any, X1Workspace], capsys: pytest.CaptureFixture[str]
) -> None:
    module, workspace = merge
    path = workspace.root / RUN / "301" / "301_000.flac"
    samples, rate = sf.read(path, dtype="int16")
    path.unlink()
    sf.write(path, samples, rate, subtype="PCM_16", format="FLAC", compression_level=1.0)
    assert sf.read(path, dtype="int16")[0].tolist() == samples.tolist()  # only the bytes differ
    assert "segment 301_000: its file does not match" in refused(module, workspace, capsys)


def test_the_merge_refuses_parts_numbered_out_of_clip_order(
    merge: tuple[Any, X1Workspace], capsys: pytest.CaptureFixture[str]
) -> None:
    module, workspace = merge
    task = task_of(workspace, 300)
    parts = pd.read_parquet(task_file(workspace, task, "parts.parquet"))
    first = parts.index[(parts["segment"] == "300_001") & (parts["position"] == 1)][0]
    parts.at[first, "position"] = 0  # positions 0, 0, ...: one part number twice
    parts.at[first, "seed"] = item_seed(106, "300_001", 0)
    rewrite(workspace, task, "parts.parquet", parts)
    assert "not numbered 0, 1, ... in clip order" in refused(module, workspace, capsys)


def test_the_merge_refuses_a_segment_with_other_parts_than_x0(
    merge: tuple[Any, X1Workspace], capsys: pytest.CaptureFixture[str]
) -> None:
    module, workspace = merge
    task = task_of(workspace, 302)
    manifest = pd.read_parquet(task_file(workspace, task, "manifest.parquet"))
    manifest.at[0, "part_starts"] = [*manifest.at[0, "part_starts"], 10**6]
    rewrite(workspace, task, "manifest.parquet", manifest)
    assert "another number of parts than in x0" in refused(module, workspace, capsys)


def test_the_merge_refuses_a_voice_or_seed_that_the_rules_do_not_give(
    merge: tuple[Any, X1Workspace], capsys: pytest.CaptureFixture[str]
) -> None:
    module, workspace = merge
    task = task_of(workspace, 303)
    original = pd.read_parquet(task_file(workspace, task, "parts.parquet"))
    changed = original.copy()
    changed.loc[changed["session"] == 303, "voice"] = "am_puck"
    rewrite(workspace, task, "parts.parquet", changed)
    assert "does not have the voice" in refused(module, workspace, capsys)
    changed = original.copy()
    changed.at[0, "seed"] = changed.at[0, "seed"] + 1
    rewrite(workspace, task, "parts.parquet", changed)
    assert "does not have the seed" in refused(module, workspace, capsys)


def test_the_merge_refuses_a_stray_file_and_inputs_that_changed(
    merge: tuple[Any, X1Workspace], capsys: pytest.CaptureFixture[str]
) -> None:
    module, workspace = merge
    for stray in (Path("300/300_099.flac"), Path("300/sub/300_000.flac"), Path("x.flac")):
        path = workspace.root / RUN / stray
        path.parent.mkdir(exist_ok=True)
        shutil.copy(workspace.root / RUN / "300" / "300_000.flac", path)
        assert "1 FLAC files in the run are in no task's manifest" in refused(
            module, workspace, capsys
        )
        path.unlink()
    labels = workspace.root / "derived" / "labels" / "run-1" / "labels.parquet"
    table = pd.read_parquet(labels)
    table.loc[304, "label"] = 1  # a participant without segments: still a new pin
    table.to_parquet(labels)
    workspace.config["labels"]["sha256"] = sha256_file(labels)
    workspace.write_config()
    assert "other inputs than the pinned ones" in refused(module, workspace, capsys)


def test_a_merged_run_is_never_merged_again(
    merge: tuple[Any, X1Workspace], capsys: pytest.CaptureFixture[str]
) -> None:
    module, workspace = merge
    module.main()
    before = (workspace.root / RUN / "run.json").read_text(encoding="utf-8")
    capsys.readouterr()
    with pytest.raises(SystemExit):
        module.main()
    assert "this run is merged already" in capsys.readouterr().out
    assert (workspace.root / RUN / "run.json").read_text(encoding="utf-8") == before


def test_an_unexpected_error_in_the_merge_prints_frames_but_no_message(
    merge: tuple[Any, X1Workspace],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module, workspace = merge

    def failing(*args: object, **kwargs: object) -> None:
        raise KeyError(SECRET)

    monkeypatch.setattr(module, "check_segment_views", failing)
    with pytest.raises(SystemExit):
        module.main()
    assert written(workspace) == []
    out, err = capsys.readouterr()
    assert out.strip() == "FAILED: KeyError"
    assert err.splitlines()[0] == "builtins.KeyError"  # the frames, without the message
    assert any(line.endswith(" in failing") for line in err.splitlines())
    assert SECRET.lower() not in (out + err).lower()


def test_the_merge_refuses_a_dirty_checkout(
    merge: tuple[Any, X1Workspace],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module, workspace = merge
    monkeypatch.setattr(module, "git_state", lambda: {"commit": "c" * 40, "dirty": True})
    assert "uncommitted changes" in refused(module, workspace, capsys)


def test_the_config_of_the_built_workspace_is_unchanged(built: X1Workspace) -> None:
    assert yaml.safe_load(built.config_path.read_text(encoding="utf-8")) == built.config


def edit_task_info(workspace: X1Workspace, task: int, change: Any) -> None:
    path = task_file(workspace, task, "run.json")
    info = json.loads(path.read_text(encoding="utf-8"))
    change(info)
    path.write_text(json.dumps(info), encoding="utf-8")


def test_the_merge_refuses_a_task_copied_in_from_another_array_job(
    merge: tuple[Any, X1Workspace], capsys: pytest.CaptureFixture[str]
) -> None:
    module, workspace = merge
    edit_task_info(workspace, 5, lambda info: info["slurm"].update(array_job_id="700"))
    out = refused(module, workspace, capsys)
    assert f"tasks [5] come from another array job than {ARRAY_JOB}" in out


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda info: info["model"]["voices"].update(af_heart="9" * 64), "differ in model"),
        (lambda info: info["model"].update(revision="e" * 40), "differ in model"),
        (
            lambda info: info["environment"]["packages"].update(numpy="0.0.0"),
            "differ in environment",
        ),
        (lambda info: info["settings"]["x1"].update(rms=0.06), "differ in settings"),
        (
            lambda info: info["check_run"].update(run="derived/revoice_check/run-1"),
            "differ in check_run",
        ),
    ],
)
def test_the_merge_refuses_tasks_that_differ_in_model_packages_or_settings(
    merge: tuple[Any, X1Workspace], capsys: pytest.CaptureFixture[str], change: Any, message: str
) -> None:
    module, workspace = merge
    edit_task_info(workspace, 5, change)
    assert f"the tasks {message}" in refused(module, workspace, capsys)


def test_the_merge_lists_tasks_on_other_cpus_and_merges_them(
    merge: tuple[Any, X1Workspace], capsys: pytest.CaptureFixture[str]
) -> None:
    module, workspace = merge

    def other_cpu(info: dict) -> None:
        info["model"]["cpu_capability"] = "AVX512"
        info["environment"]["cpu"] = "Another CPU"

    edit_task_info(workspace, 5, other_cpu)
    module.main()
    assert written(workspace) == ["manifest.parquet", "parts.parquet", "run.json"]
    out = capsys.readouterr().out
    assert "Another CPU: 1" in out
    info = json.loads((workspace.root / RUN / "run.json").read_text(encoding="utf-8"))
    assert info["tasks"]["5"]["cpu_capability"] == "AVX512"


def test_the_merge_refuses_a_session_moved_to_another_task(
    merge: tuple[Any, X1Workspace], capsys: pytest.CaptureFixture[str]
) -> None:
    module, workspace = merge
    owner = task_of(workspace, 300)
    empty = next(
        t
        for t in range(TASKS)
        if pd.read_parquet(task_file(workspace, t, "manifest.parquet")).empty
    )
    manifest = pd.read_parquet(task_file(workspace, owner, "manifest.parquet"))
    moved = manifest[manifest["session"] == 300]
    rewrite(workspace, owner, "manifest.parquet", manifest[manifest["session"] != 300])
    rewrite(workspace, empty, "manifest.parquet", moved)
    assert "do not lie in the tasks that assign_tasks gives them" in refused(
        module, workspace, capsys
    )


@pytest.mark.parametrize("change", ["a row dropped", "a row doubled", "a segment without a clip"])
def test_the_merge_refuses_parts_that_are_not_one_row_per_part(
    merge: tuple[Any, X1Workspace], capsys: pytest.CaptureFixture[str], change: str
) -> None:
    module, workspace = merge
    task = task_of(workspace, 300)
    parts = pd.read_parquet(task_file(workspace, task, "parts.parquet"))
    if change == "a row dropped":
        parts = parts.iloc[:-1]
    elif change == "a row doubled":
        parts = pd.concat([parts, parts.iloc[:1]], ignore_index=True)
    else:
        extra = parts.iloc[:1].copy()
        extra["segment"] = "300_099"
        extra["seed"] = item_seed(106, "300_099", int(extra["position"].iloc[0]))  # all else fits
        parts = pd.concat([parts, extra], ignore_index=True)
    rewrite(workspace, task, "parts.parquet", parts)
    assert "parts.parquet does not hold one row per part" in refused(module, workspace, capsys)


def test_the_merge_refuses_a_config_that_changed_since_the_build(
    merge: tuple[Any, X1Workspace], capsys: pytest.CaptureFixture[str]
) -> None:
    module, workspace = merge
    with workspace.config_path.open("a", encoding="utf-8") as config:
        config.write("# a comment\n")
    assert "the tasks were built with another config" in refused(module, workspace, capsys)


def test_a_merge_that_dies_while_writing_its_run_json_can_run_again(
    merge: tuple[Any, X1Workspace], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, workspace = merge
    run = workspace.root / RUN
    real = Path.write_text

    def dying(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        if self.parent == run and self.name.startswith("run.json"):
            real(self, data[:100], *args, **kwargs)
            raise KeyboardInterrupt
        return real(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", dying)
    with pytest.raises(SystemExit):
        module.main()
    assert not (run / "run.json").exists()  # not "merged": the merge may run again
    monkeypatch.setattr(Path, "write_text", real)
    module.main()
    assert json.loads((run / "run.json").read_text(encoding="utf-8"))["status"] == "done"
    assert not (run / "run.json.tmp").exists()


def test_the_merge_refuses_a_check_run_whose_run_json_changed_after_the_build(
    merge: tuple[Any, X1Workspace], capsys: pytest.CaptureFixture[str]
) -> None:
    module, workspace = merge
    path = check_run(workspace) / "run.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info["dry_run"]["words_read"] += 1
    path.write_text(json.dumps(info), encoding="utf-8")
    assert "the check run's run.json changed after the build" in refused(module, workspace, capsys)


@pytest.mark.parametrize("column", ["words_read", "numbers"])
def test_the_merge_refuses_tasks_whose_counts_differ_from_the_dry_run(
    merge: tuple[Any, X1Workspace], capsys: pytest.CaptureFixture[str], column: str
) -> None:
    module, workspace = merge
    parts = pd.read_parquet(task_file(workspace, 0, "parts.parquet"))
    parts.loc[0, column] += 1
    rewrite(workspace, 0, "parts.parquet", parts)
    out = refused(module, workspace, capsys)
    assert "the tasks read" in out
    assert "numbers written out; the check run's dry run read" in out
