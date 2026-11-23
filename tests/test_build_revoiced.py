"""The build of the re-voiced clip (scripts/build_revoiced.py), task by task, on a synthetic workspace."""

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import listening_beyond_words.revoice as revoice
from listening_beyond_words.clips import MANIFEST_COLUMNS, audio_sha256
from listening_beyond_words.revoice import TASKS, X1Settings, assign_tasks, assign_voices
from listening_beyond_words.runinfo import STOPPED, item_seed, sha256_file
from listening_beyond_words.views import check_segment_views
from x1_fakes import (
    ARRAY_JOB,
    CHECK_JOB,
    SECRET,
    X1Workspace,
    check_dir,
    load_script,
    outside_text,
    patch_script,
    run_check,
    run_task,
)


def run_of(workspace: X1Workspace, array: str = ARRAY_JOB) -> Path:
    return workspace.root / "views" / "x1" / f"run-{array}"


def info_of(path: Path) -> dict:
    return json.loads((path / "run.json").read_text(encoding="utf-8"))


@pytest.fixture
def build(x1_workspace: X1Workspace, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, X1Workspace]:
    """A passed check run, and the build script pointed at the same workspace."""
    run_check(x1_workspace, monkeypatch)
    module = load_script("build_revoiced")
    patch_script(module, x1_workspace, monkeypatch)
    return module, x1_workspace


def sessions_of(task: int) -> list[int]:
    counts = {300: 3, 301: 2, 302: 2, 303: 2}  # segments per session in the workspace
    return sorted(s for s, t in assign_tasks(counts).items() if t == task)


def test_every_task_renders_its_own_sessions_into_one_shared_run(
    build: tuple[Any, X1Workspace],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module, workspace = build
    for task in range(TASKS):
        run_task(module, monkeypatch, task)
    out, err = capsys.readouterr()
    run = run_of(workspace)
    settings = X1Settings.from_config(workspace.config)
    voices = assign_voices(
        pd.read_parquet(workspace.root / "derived" / "labels" / "run-1" / "labels.parquet"),
        settings.voices,
    )
    segments = workspace.segments.set_index("segment")
    seen = []
    for task in range(TASKS):
        folder = run / "tasks" / str(task)
        info = info_of(folder)
        assert info["status"] == "done"
        assert info["task"] == task
        assert "libsndfile" in info["environment"]
        assert info["slurm"]["array_job_id"] == ARRAY_JOB
        assert info["check_run"]["run"] == f"derived/revoice_check/run-{CHECK_JOB}"
        assert info["outputs"]["manifest.parquet"] == sha256_file(folder / "manifest.parquet")
        manifest = pd.read_parquet(folder / "manifest.parquet")
        parts = pd.read_parquet(folder / "parts.parquet")
        assert list(manifest.columns) == list(MANIFEST_COLUMNS)
        assert sorted(set(manifest["session"])) == sessions_of(task)
        seen += manifest["segment"].tolist()
        for row in manifest.itertuples():
            samples, rate = sf.read(run / row.file, dtype="int16")
            assert rate == 16_000
            assert audio_sha256(samples) == row.audio_sha256
            assert len(row.part_starts) == segments.loc[row.segment, "parts"]
            check_segment_views({"x1": samples}, row.part_starts, row.part_ends, gap=4800)
            own = parts[parts["segment"] == row.segment]
            assert own["position"].tolist() == list(range(len(own)))
            assert set(own["voice"]) == {voices[row.session]}
            assert own["seed"].tolist() == [
                item_seed(106, row.segment, int(p)) for p in own["position"]
            ]
            assert own["part_start"].tolist() == list(row.part_starts)
        assert SECRET.lower() not in outside_text(
            folder / "run.json", folder / "manifest.parquet", streams=out + err
        )
    assert sorted(seen) == sorted(workspace.segments["segment"])
    assert out.count("task run: views/x1/run-600/tasks/") == TASKS


def test_every_task_reads_the_text_with_numbers_written_out(
    x1_workspace: X1Workspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from listening_beyond_words.transcripts import normalise

    run_check(x1_workspace, monkeypatch)
    module = load_script("build_revoiced")
    pipeline = patch_script(module, x1_workspace, monkeypatch)
    for task in range(TASKS):
        run_task(module, monkeypatch, task)
    out, err = capsys.readouterr()
    run = run_of(x1_workspace)
    parts = pd.concat(
        [pd.read_parquet(run / "tasks" / str(t) / "parts.parquet") for t in range(TASKS)],
        ignore_index=True,
    )
    # Kokoro read exactly the texts that parts.parquet records, in the same order.
    assert [text for text, _, _ in pipeline.calls] == parts["text"].tolist()
    assert not any(word.isdigit() for text in parts["text"] for word in text.split())
    assert parts["written_text"].map(lambda t: len(normalise(t).split())).tolist() == (
        parts["n_words"].tolist()
    )
    assert (parts["words_read"] == parts["text"].str.split().str.len()).all()
    assert (parts["words_read"] >= parts["n_words"]).all()
    assert parts["numbers"].sum() > 0
    counts = [info_of(run / "tasks" / str(t))["counts"] for t in range(TASKS)]
    assert sum(c["words_read"] for c in counts) == parts["words_read"].sum()
    assert sum(c["numbers_written_out"] for c in counts) == parts["numbers"].sum()
    assert "numbers written out," in out
    for task in range(TASKS):
        folder = run / "tasks" / str(task)
        leaving = outside_text(folder / "run.json", streams=out + err)
        for word in (SECRET.lower(), "forty", "nineteen"):
            assert word not in leaving


def test_the_test_render_equals_the_full_render(
    build: tuple[Any, X1Workspace], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, workspace = build
    for task in range(TASKS):
        run_task(module, monkeypatch, task)
    test = pd.read_parquet(check_dir(workspace) / "test.parquet")
    manifests = pd.concat(
        pd.read_parquet(run_of(workspace) / "tasks" / str(t) / "manifest.parquet")
        for t in range(TASKS)
    ).set_index("segment")
    for segment, sha in zip(test["segment"], test["audio_sha256"], strict=True):
        assert manifests.loc[segment, "audio_sha256"] == sha


def test_a_second_attempt_of_a_task_never_writes_into_the_first(
    build: tuple[Any, X1Workspace],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module, workspace = build
    task = assign_tasks({300: 3, 301: 2, 302: 2, 303: 2})[300]
    run_task(module, monkeypatch, task)
    folder = run_of(workspace) / "tasks" / str(task)
    before = (folder / "run.json").read_text(encoding="utf-8")
    capsys.readouterr()
    with pytest.raises(SystemExit):
        run_task(module, monkeypatch, task)
    out, err = capsys.readouterr()
    assert out.strip() == "FAILED: FileExistsError"
    # No folder of its own holds the frames, so they go to stderr: types and frames only.
    assert err.splitlines()[0] == "builtins.FileExistsError"
    assert all(line.startswith("  ") or line.startswith("builtins.") for line in err.splitlines())
    assert (folder / "run.json").read_text(encoding="utf-8") == before
    assert not (folder / "errors.log").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"status": "failed"}, "did not pass"),
        ({"git": {"commit": "d" * 40, "dirty": False}}, "another commit"),
        ({"config": {"file": "configs/pilot/data.yaml", "sha256": "e" * 64}}, "another config"),
        ({"inputs": {}}, "other pinned inputs"),
    ],
)
def test_a_task_refuses_a_check_run_that_did_not_pass_or_differs(
    build: tuple[Any, X1Workspace],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    change: dict,
    message: str,
) -> None:
    module, workspace = build
    path = check_dir(workspace) / "run.json"
    path.write_text(json.dumps(info_of(check_dir(workspace)) | change), encoding="utf-8")
    with pytest.raises(SystemExit):
        run_task(module, monkeypatch, 0)
    assert message in capsys.readouterr().out
    folder = run_of(workspace) / "tasks" / "0"
    info = info_of(folder)
    assert info["status"] == "failed"
    assert message in info["error"]["message"]
    assert not list(run_of(workspace).glob("*/*.flac"))


def test_a_task_refuses_a_checkout_at_another_commit_than_its_check(
    build: tuple[Any, X1Workspace],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module, _ = build
    monkeypatch.setattr(module, "git_state", lambda: {"commit": "d" * 40, "dirty": False})
    with pytest.raises(SystemExit):
        run_task(module, monkeypatch, 0)
    assert "another commit" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("unset", "task", "message"),
    [
        ("SLURM_ARRAY_TASK_ID", "0", "submit this as an array job"),
        (None, "20", "not one of 0 to 19"),
        (None, "x", "not one of 0 to 19"),
    ],
)
def test_a_task_refuses_to_run_outside_an_array_job(
    build: tuple[Any, X1Workspace],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    unset: str | None,
    task: str,
    message: str,
) -> None:
    module, workspace = build
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", ARRAY_JOB)
    monkeypatch.setenv("SLURM_JOB_ID", "601")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", task)
    if unset:
        monkeypatch.delenv(unset)
    monkeypatch.setattr(
        sys, "argv", ["build_revoiced.py", f"derived/revoice_check/run-{CHECK_JOB}"]
    )
    with pytest.raises(SystemExit):
        module.main()
    assert message in capsys.readouterr().out
    assert not run_of(workspace).exists()


def test_a_task_that_slurm_stops_is_left_failed(
    build: tuple[Any, X1Workspace],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module, workspace = build
    calls = []
    write_view = module.write_view

    def stopped(*args: object, **kwargs: object) -> object:
        calls.append(1)
        if len(calls) == 2:
            raise SystemExit(STOPPED)
        return write_view(*args, **kwargs)

    monkeypatch.setattr(module, "write_view", stopped)
    task = assign_tasks({300: 3, 301: 2, 302: 2, 303: 2})[300]
    with pytest.raises(SystemExit):
        run_task(module, monkeypatch, task)
    assert f"FAILED: SystemExit in segment 300_001: {STOPPED}" in capsys.readouterr().out
    info = info_of(run_of(workspace) / "tasks" / str(task))
    assert info["status"] == "failed"
    assert info["error"]["message"] == STOPPED
    assert "finished_utc" in info
    assert (run_of(workspace) / "tasks" / str(task) / "errors.log").is_file()


def test_a_sound_in_a_gap_stops_the_task_before_its_file_is_written(
    build: tuple[Any, X1Workspace],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module, workspace = build
    join = revoice.join_revoiced

    def leaky(*args: Any, **kwargs: Any) -> revoice.Revoiced:
        built = join(*args, **kwargs)
        if len(built.part_starts) < 2:
            return built
        clip = built.clip.copy()
        clip[built.part_ends[0] + 10] = 0.01
        return dataclasses.replace(built, clip=clip)

    monkeypatch.setattr(revoice, "join_revoiced", leaky)
    task = assign_tasks({300: 3, 301: 2, 302: 2, 303: 2})[300]
    with pytest.raises(SystemExit):
        run_task(module, monkeypatch, task)
    out = capsys.readouterr().out
    own = workspace.segments[workspace.segments["session"] == 300]
    first = str(own[own["parts"] >= 2]["segment"].min())  # the first with a gap
    assert f"FAILED: ValueError in segment {first};" in out
    assert not (run_of(workspace) / "300" / f"{first}.flac").exists()
    assert info_of(run_of(workspace) / "tasks" / str(task))["status"] == "failed"


def test_an_error_while_writing_the_parts_never_shows_a_word(
    build: tuple[Any, X1Workspace],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module, workspace = build
    write_table = module.write_table

    def failing(table: pd.DataFrame, path: Path) -> str:
        if path.name == "parts.parquet":
            raise ValueError(f"cannot write {table['text'].iloc[0]} {SECRET}")
        return write_table(table, path)

    monkeypatch.setattr(module, "write_table", failing)
    task = assign_tasks({300: 3, 301: 2, 302: 2, 303: 2})[300]
    with pytest.raises(SystemExit):
        run_task(module, monkeypatch, task)
    out, err = capsys.readouterr()
    folder = run_of(workspace) / "tasks" / str(task)
    assert info_of(folder)["status"] == "failed"
    assert "FAILED: ValueError; traceback frames in" in out
    assert SECRET.lower() not in outside_text(
        folder / "run.json", folder / "errors.log", streams=out + err
    )


def test_a_task_refuses_a_run_that_was_merged_already(
    build: tuple[Any, X1Workspace],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module, workspace = build
    run_of(workspace).mkdir(parents=True)
    (run_of(workspace) / "run.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        run_task(module, monkeypatch, 0)
    assert "merged already" in capsys.readouterr().out
    assert not (run_of(workspace) / "tasks").exists()


def test_the_build_renders_the_same_samples_twice(
    build: tuple[Any, X1Workspace], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, workspace = build
    task = assign_tasks({300: 3, 301: 2, 302: 2, 303: 2})[300]
    run_task(module, monkeypatch, task, array="700")
    run_task(module, monkeypatch, task, array="701")
    first = pd.read_parquet(run_of(workspace, "700") / "tasks" / str(task) / "manifest.parquet")
    second = pd.read_parquet(run_of(workspace, "701") / "tasks" / str(task) / "manifest.parquet")
    assert first["audio_sha256"].tolist() == second["audio_sha256"].tolist()
    assert np.isfinite(first["gain_db"]).all()


def test_a_task_refuses_transcript_lines_that_no_longer_give_the_pinned_words(
    build: tuple[Any, X1Workspace],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import re

    module, workspace = build  # the check passed before the transcript changed
    path = workspace.root / "edaic" / "sessions" / "300_P" / "300_Transcript.csv"
    table = pd.read_csv(path)
    table.at[0, "Text"] = re.sub(r"\bwell\b", "swell", table.at[0, "Text"], count=1)
    table.to_csv(path, index=False)
    task = assign_tasks({300: 3, 301: 2, 302: 2, 303: 2})[300]
    with pytest.raises(SystemExit):
        run_task(module, monkeypatch, task)
    out, err = capsys.readouterr()
    assert "the transcript lines do not give its words" in out
    assert not list((run_of(workspace) / "300").glob("*.flac"))
    assert "swell" not in outside_text(
        run_of(workspace) / "tasks" / str(task) / "run.json", streams=out + err
    )


def test_a_task_refuses_a_check_run_whose_test_render_was_changed(
    build: tuple[Any, X1Workspace],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module, workspace = build
    path = check_dir(workspace) / "test.parquet"
    pd.read_parquet(path).iloc[:-1].to_parquet(path, index=False)
    with pytest.raises(SystemExit):
        run_task(module, monkeypatch, 0)
    assert "test.parquet does not match its run.json" in capsys.readouterr().out
