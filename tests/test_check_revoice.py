"""The check job of the re-voiced clip (scripts/check_revoice.py), on a synthetic workspace."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from listening_beyond_words.clips import audio_sha256
from listening_beyond_words.revoice import TASKS, X1Settings, render_segment
from listening_beyond_words.settings import Workspace
from x1_fakes import (
    CHECK_JOB,
    REPO,
    SECRET,
    FakePipeline,
    X1Workspace,
    check_dir,
    fake_kokoro,
    load_script,
    outside_text,
    patch_script,
    run_check,
)


def info_of(run: Path) -> dict:
    return json.loads((run / "run.json").read_text(encoding="utf-8"))


def test_the_check_passes_and_writes_what_is_heard(
    x1_workspace: X1Workspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = run_check(x1_workspace, monkeypatch)
    out, err = capsys.readouterr()
    info = info_of(run)
    assert info["status"] == "done"
    assert "stop rules: words without a sound 0, bracket tokens 0" in out
    assert ": passed" in out
    assert "repeats: 2 of 2 segments rendered twice are identical" in out
    assert f"run: derived/revoice_check/run-{CHECK_JOB}" in out
    dry = pd.read_parquet(run / "dry_run.parquet")
    parts = sum(len(row.starts) for row in x1_workspace.segments.itertuples())
    assert len(dry) == parts == info["dry_run"]["parts"]
    assert info["dry_run"]["words"] == 40 * len(x1_workspace.segments)
    assert info["dry_run"]["typographic_tokens"] > 0
    assert info["dry_run"]["split_edges"]  # x-ray and 9:30 are split at some part edges
    test = pd.read_parquet(run / "test.parquet")
    assert len(test) == 4
    assert not set(test["segment"]) & {"300_001", "301_000"}  # the word-check sample
    assert test["heard"].tolist() == [True, True, False, False]
    assert sorted(path.name for path in (run / "clips").iterdir()) == ["01.wav", "02.wav"]
    key = pd.read_csv(run / "key.tsv", sep="\t")
    assert key["clip"].tolist() == ["01.wav", "02.wav"]
    assert key["segment"].tolist() == test["segment"].tolist()[:2]
    assert (key["text"].str.len() > 0).all()  # the text read, kept on OMNI only
    for name, sha256 in info["outputs"].items():
        assert len(sha256) == 64, name
    assert info["test_render"]["repeats_identical"] == info["test_render"]["repeated"] == 2
    assert info["model"]["revision"] == "f" * 40
    assert set(info["environment"]) >= {"packages", "cpu", "uv_lock_sha256", "libsndfile"}
    assert info["inputs"]["cut_words"]["file"] == "derived/segments/run-1/words.parquet"
    # What may leave OMNI never holds a word: the printed lines, run.json, test.parquet.
    assert SECRET.lower() not in outside_text(run / "run.json", streams=out + err)
    assert not {"text", "phonemes"} & set(test.columns)


def test_the_test_render_is_the_builds_render_and_its_clips_are_listening_copies(
    x1_workspace: X1Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from listening_beyond_words.clips import listening_copy
    from listening_beyond_words.revoice import load_inputs

    run = run_check(x1_workspace, monkeypatch)
    test = pd.read_parquet(run / "test.parquet")
    settings = X1Settings.from_config(x1_workspace.config)
    inputs = load_inputs(Workspace(x1_workspace.root), x1_workspace.config, settings)
    kokoro = fake_kokoro(FakePipeline(), settings)
    for row in test.itertuples():
        parts = inputs.parts[inputs.parts["segment"] == row.segment]
        rendered = render_segment(
            kokoro, parts, x1_workspace.lines[row.session], row.voice, row.segment, settings
        )
        assert audio_sha256(rendered.samples) == row.audio_sha256
        assert row.voice == inputs.voices[row.session]
        if row.heard:
            heard, rate = sf.read(run / "clips" / f"{row.draw:02d}.wav", dtype="float32")
            copy, _ = listening_copy(rendered.samples.astype(np.float32) / 32768)
            assert rate == 16_000
            np.testing.assert_allclose(heard, copy, atol=1 / 32768)


def test_the_dry_run_stops_at_a_word_without_a_sound(
    x1_workspace: X1Workspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pipeline = FakePipeline(readings={SECRET: [("", 4, None)]})
    with pytest.raises(SystemExit):
        run_check(x1_workspace, monkeypatch, pipeline)
    out, err = capsys.readouterr()
    run = check_dir(x1_workspace)
    info = info_of(run)
    assert info["status"] == "failed"
    assert info["error"]["message"].startswith("the dry run stopped before any render")
    assert "words_without_a_sound" in info["error"]["message"]
    assert info["dry_run"]["stop_rules"]["words_without_a_sound"] > 0
    assert ": STOPPED" in out
    silent = info["dry_run"]["stop_rules"]["words_without_a_sound"]
    assert re.search(rf"words without a sound, by class: .*capitals {silent}\b", out)
    assert "FAILED: StepError: the dry run stopped before any render" in out
    assert not (run / "clips").exists()
    assert not (run / "test.parquet").exists()
    dry = pd.read_parquet(run / "dry_run.parquet")
    assert dry["silent"].sum() == info["dry_run"]["stop_rules"]["words_without_a_sound"]
    assert SECRET in {word for words in dry["silent_words"] for word in words}  # OMNI only
    assert SECRET.lower() not in outside_text(
        run / "run.json", run / "errors.log", streams=out + err
    )


def test_the_dry_run_stops_at_a_bracket_in_the_transcript(
    x1_workspace: X1Workspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = x1_workspace.root / "edaic" / "sessions" / "300_P" / "300_Transcript.csv"
    table = pd.read_csv(path)
    text = table.at[0, "Text"]
    assert re.search(r"\bwell\b", text)
    table.at[0, "Text"] = re.sub(r"\bwell\b", "[well]", text, count=1)  # same stored word
    table.to_csv(path, index=False)
    with pytest.raises(SystemExit):
        run_check(x1_workspace, monkeypatch)
    out = capsys.readouterr().out
    info = info_of(check_dir(x1_workspace))
    assert info["dry_run"]["stop_rules"]["bracket_tokens"] >= 1
    assert "bracket_tokens" in info["error"]["message"]
    assert "STOPPED" in out


@pytest.mark.parametrize(
    ("readings", "rule"),
    [
        ({SECRET: [("ˈsyn(fɹ)θ", 2, None)]}, "parts_with_brackets_in_phonemes"),
        ({SECRET: [("ˈsynʘ", 4, None)]}, "phoneme_chars_outside_the_vocabulary"),
        ({SECRET: [("ɑ" * 600, 4, None)]}, "parts_over_510_phoneme_chars"),
    ],
)
def test_the_dry_run_stops_at_what_kokoro_could_not_read_in_one_call(
    x1_workspace: X1Workspace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    readings: dict,
    rule: str,
) -> None:
    with pytest.raises(SystemExit):
        run_check(x1_workspace, monkeypatch, FakePipeline(readings=readings))
    out = capsys.readouterr().out
    info = info_of(check_dir(x1_workspace))
    assert info["dry_run"]["stop_rules"][rule] > 0
    assert rule in info["error"]["message"]
    if rule == "phoneme_chars_outside_the_vocabulary":
        assert "phoneme characters outside Kokoro's vocabulary: U+0298" in out
    if rule == "parts_with_brackets_in_phonemes":
        assert "went to espeak (a language flag)" in out


class Drifting(FakePipeline):
    """A pipeline whose audio changes from call to call, whatever the seed."""

    def __call__(self, text: str, voice: str, speed: float):
        for result in super().__call__(text, voice, speed):
            result.audio = result.audio + 1e-3 * len(self.calls)
            yield result


def test_the_check_fails_when_a_repeat_differs(
    x1_workspace: X1Workspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        run_check(x1_workspace, monkeypatch, Drifting())
    out = capsys.readouterr().out
    run = check_dir(x1_workspace)
    info = info_of(run)
    assert info["status"] == "failed"
    assert "repeat check failed: 0 of 2" in info["error"]["message"]
    assert "repeats: 0 of 2" in out
    assert (run / "test.parquet").exists()


def test_an_error_inside_kokoro_never_shows_a_word(
    x1_workspace: X1Workspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        run_check(x1_workspace, monkeypatch, FakePipeline(fail_on=SECRET))
    out, err = capsys.readouterr()
    run = check_dir(x1_workspace)
    info = info_of(run)
    assert info["status"] == "failed"
    assert info["error"]["type"] == "ValueError"
    assert "message" not in info["error"]
    assert re.search(
        r"FAILED: ValueError in segment \d{3}_\d{3}, part \d+; traceback frames in", out
    )
    record = (run / "errors.log").read_text(encoding="utf-8")
    assert "builtins.ValueError" in record
    assert "in g2p" in record
    assert SECRET.lower() not in outside_text(
        run / "run.json", run / "errors.log", streams=out + err
    )


def test_the_check_refuses_to_run_outside_a_job_or_from_a_dirty_checkout(
    x1_workspace: X1Workspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_script("check_revoice")
    patch_script(module, x1_workspace, monkeypatch)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit):
        module.main()
    assert "FAILED: StepError: submit this as a job" in capsys.readouterr().out
    monkeypatch.setenv("SLURM_JOB_ID", CHECK_JOB)
    monkeypatch.setattr(module, "git_state", lambda: {"commit": "c" * 40, "dirty": True})
    with pytest.raises(SystemExit):
        module.main()
    assert "uncommitted changes" in capsys.readouterr().out
    assert not check_dir(x1_workspace).exists()


def test_a_second_check_with_the_same_job_id_never_writes_into_the_first(
    x1_workspace: X1Workspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = run_check(x1_workspace, monkeypatch)
    before = (run / "run.json").read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        run_check(x1_workspace, monkeypatch)
    assert "FAILED: FileExistsError" in capsys.readouterr().out
    assert (run / "run.json").read_text(encoding="utf-8") == before


def sbatch(name: str) -> str:
    return (REPO / "scripts" / "slurm" / f"{name}.sbatch").read_text(encoding="utf-8")


def test_the_job_files_ask_for_the_threads_of_x1_and_one_task_per_share() -> None:
    settings = X1Settings.from_config(
        __import__("yaml").safe_load((REPO / "configs" / "pilot" / "data.yaml").read_text())
    )
    for name in ("check_revoice", "build_revoiced"):
        text = sbatch(name)
        assert f"#SBATCH --cpus-per-task={settings.threads}\n" in text
        assert "#SBATCH --partition=short\n" in text
        assert f"#SBATCH --job-name=lbw-{name.replace('_', '-')}\n" in text
    build = sbatch("build_revoiced")
    assert f"#SBATCH --array=0-{TASKS - 1}%{TASKS}\n" in build
    assert 'python scripts/build_revoiced.py "$@"\n' in build
    assert "SLURM_ARRAY_TASK_ID:?" in build


def change_a_word_of_session_300(workspace: X1Workspace, new: str) -> None:
    """Change one stored word of session 300's first transcript line."""
    path = workspace.root / "edaic" / "sessions" / "300_P" / "300_Transcript.csv"
    table = pd.read_csv(path)
    text = table.at[0, "Text"]
    assert re.search(r"\bwell\b", text)
    table.at[0, "Text"] = re.sub(r"\bwell\b", new, text, count=1)
    table.to_csv(path, index=False)


def test_the_check_refuses_transcript_lines_that_do_not_give_the_pinned_words(
    x1_workspace: X1Workspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    change_a_word_of_session_300(x1_workspace, "swell")
    with pytest.raises(SystemExit):
        run_check(x1_workspace, monkeypatch)
    out, err = capsys.readouterr()
    run = check_dir(x1_workspace)
    assert re.search(
        r"FAILED: StepError in segment 300_00\d: segment 300_00\d: the transcript", out
    )
    assert info_of(run)["status"] == "failed"
    assert not (run / "dry_run.parquet").exists()
    assert "swell" not in outside_text(run / "run.json", run / "errors.log", streams=out + err)


def test_the_repeats_are_the_first_test_segments_by_id(
    x1_workspace: X1Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script("check_revoice")
    patch_script(module, x1_workspace, monkeypatch)
    rendered: list[str] = []
    real = module.render_segment

    def recording(kokoro, parts, lines, voice, segment, settings):  # type: ignore[no-untyped-def]
        rendered.append(segment)
        return real(kokoro, parts, lines, voice, segment, settings)

    monkeypatch.setattr(module, "render_segment", recording)
    monkeypatch.setenv("SLURM_JOB_ID", CHECK_JOB)
    module.main()
    drawn = pd.read_parquet(check_dir(x1_workspace) / "test.parquet")["segment"].tolist()
    assert sorted(drawn)[:2] != drawn[:2]  # the draw tells the two orders apart
    assert rendered == drawn + sorted(drawn)[:2]


def test_a_failure_before_the_run_folder_exists_prints_frames_but_no_message(
    x1_workspace: X1Workspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_script("check_revoice")
    patch_script(module, x1_workspace, monkeypatch)

    def failing(config: dict) -> X1Settings:
        raise ValueError(f"cannot read {SECRET}")

    monkeypatch.setattr(module.X1Settings, "from_config", staticmethod(failing))
    monkeypatch.setenv("SLURM_JOB_ID", CHECK_JOB)
    with pytest.raises(SystemExit):
        module.main()
    out, err = capsys.readouterr()
    assert out.strip() == "FAILED: ValueError"
    assert err.splitlines()[0] == "builtins.ValueError"
    assert any(line.endswith(" in failing") for line in err.splitlines())
    assert SECRET.lower() not in (out + err).lower()
    assert not check_dir(x1_workspace).exists()


@pytest.mark.parametrize(
    ("script", "unset"),
    [
        ("check_revoice", "SLURM_JOB_ID"),
        ("build_revoiced", "SLURM_ARRAY_TASK_ID"),
        ("merge_revoiced", None),
    ],
)
def test_every_x1_script_reports_a_failure_without_chaining_it_even_if_printing_fails(
    x1_workspace: X1Workspace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    script: str,
    unset: str | None,
) -> None:
    import builtins
    import sys

    from listening_beyond_words.runinfo import quiet_unraisable

    module = load_script(script)
    patch_script(module, x1_workspace, monkeypatch)
    if unset:
        monkeypatch.delenv(unset, raising=False)
    monkeypatch.setattr(sys, "argv", [f"{script}.py"])  # the merge refuses: no run given
    real = builtins.print

    def broken(*args: object, **kwargs: object) -> None:
        if args and str(args[0]).startswith("FAILED"):
            raise BrokenPipeError(f"cannot print {SECRET}")
        real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "print", broken)
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 1
    # Reported outside the except clause: nothing that Python could print is chained to it.
    assert stopped.value.__context__ is None
    assert stopped.value.__cause__ is None
    assert sys.unraisablehook is quiet_unraisable
    out, err = capsys.readouterr()
    assert SECRET.lower() not in (out + err).lower()
