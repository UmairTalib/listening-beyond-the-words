"""The word check job (scripts/check_views.py), on a synthetic workspace with fake ASR systems."""

import contextlib
import io
import itertools
import json
import math
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

import wordcheck_fakes
from listening_beyond_words import wordcheck
from listening_beyond_words.models import load_pins
from listening_beyond_words.revoice import X1Settings, assign_voices
from listening_beyond_words.runinfo import STOPPED, sha256_file
from listening_beyond_words.segments import segment_parts
from listening_beyond_words.voicecheck import (
    FUNCTIONAL_NAMES,
    gate_values,
    level_copy,
    part_measures,
    participant_means,
    probe_aurocs,
    segment_measures,
    voice_iccs,
    voiced_seconds,
)
from listening_beyond_words.wordcheck import (
    RECOGNISERS,
    BootstrapSettings,
    LeakageSettings,
    bootstrap_interval,
    bootstrap_sets,
    chance_partners,
    fit_leakage,
    person_scores,
    wer_counts,
    words,
)
from wordcheck_fakes import (
    JOB,
    LABEL_WORDS,
    MODEL_FILES,
    PUBLIC_VOICES,
    RERUN_JOB,
    RESAMPLES,
    REV_TEXT,
    Ears,
    FakeSmile,
    WordcheckWorkspace,
    info_of,
    make_wordcheck_workspace,
    patch_job,
    run_job,
    write_pinned,
)
from x1_fakes import REPO, SECRET, load_script, outside_text

READING_ORDER = ("rev", "x2", "x0", "x1")
FIRST_RUN = f"derived/wordcheck/run-{JOB}"
# Participant text that must never leave the run folder: the secret word, the words of the
# labels and a number as the text rule writes it out ("1990" in the lines).
PRIVATE = (SECRET.lower(), *LABEL_WORDS, "nineteen ninety")
# What would show a number per segment or participant: a segment ID, or a participant ID
# as a key or paired with a value; and in run.json, a list as long as a fold of segments.
SEGMENT_ID = re.compile(r"\b3\d\d_\d{3}\b")
PARTICIPANT_PAIR = re.compile(r"[{\[,]\s*3\d\d\s*[:,]")
NUMBER_LIST = re.compile(r"[\[(](?:\s*-?[0-9][0-9.e+-]*\s*,){7,}")  # 8 numbers or more
ITEM_KEY = re.compile(r"3\d\d(_\d{3})?")
LONGEST_LIST = 7  # the public guard's voices; folds and intervals are shorter
LONG_LISTS = {("settings", "whisper_resolved", "suppress_tokens")}  # long with faster-whisper
UNVOICED_SESSION = 305  # the participant whose parts the level-aware opensmile hears unvoiced


@dataclass
class Finished:
    """A run of the job, what the fakes saw and what it printed."""

    workspace: WordcheckWorkspace
    ears: Ears
    run: Path
    out: str
    err: str

    @property
    def info(self) -> dict:
        return info_of(self.run)


def finish(
    workspace: WordcheckWorkspace,
    ears: Ears,
    *args: str,
    prepare: Callable[[ModuleType, pytest.MonkeyPatch], None] | None = None,
) -> Finished:
    """Run the job to its end in a patch context of its own, for fixtures shared by a module."""
    out, err = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as monkeypatch:
        module = load_script("check_views")
        if prepare is not None:
            prepare(module, monkeypatch)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            run = run_job(workspace, monkeypatch, ears, *args, module=module)
    return Finished(workspace, ears, run, out.getvalue(), err.getvalue())


WATCHED = (
    "read_view_clip",
    "fit_leakage",
    "bootstrap_sets",
    "control_aurocs",
    "voice_iccs",
    "probe_aurocs",
    "word_rows",
    "leakage_rows",
    "write_table",
)


def watch_stages(ears: Ears) -> Callable[[ModuleType, pytest.MonkeyPatch], None]:
    """Log each call of the job's stage functions in the fakes' events, in order."""

    def prepare(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in WATCHED:
            real = getattr(module, name)

            def watched(*args: Any, _real: Any = real, _name: str = name, **kwargs: Any) -> Any:
                table = Path(args[1]).name if _name == "write_table" else None
                ears.events.append(("stage", _name, table))
                return _real(*args, **kwargs)

            monkeypatch.setattr(module, name, watched)

    return prepare


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> WordcheckWorkspace:
    """The synthetic workspace, built once per module; tests change only their own copies."""
    return make_wordcheck_workspace(tmp_path_factory.mktemp("template") / "workspace")


@pytest.fixture
def workspace(
    template: WordcheckWorkspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> WordcheckWorkspace:
    copy = template.copy_to(tmp_path)
    monkeypatch.setenv("LBW_WORKSPACE", str(copy.root))
    return copy


@pytest.fixture(scope="module")
def sample_run(template: WordcheckWorkspace, tmp_path_factory: pytest.TempPathFactory) -> Finished:
    """The default mode, with every rule met, and the order of its stages logged."""
    workspace = template.copy_to(tmp_path_factory.mktemp("sample"))
    ears = Ears(workspace)
    return finish(workspace, ears, prepare=watch_stages(ears))


@pytest.fixture(scope="module")
def waiting_run(template: WordcheckWorkspace, tmp_path_factory: pytest.TempPathFactory) -> Finished:
    """A first run whose x0 control asks for the rerun: every x0 clip is heard as one text."""
    workspace = template.copy_to(tmp_path_factory.mktemp("waiting"))
    texts = {("x0", str(segment)): REV_TEXT for segment in workspace.sample["segment"]}
    return finish(workspace, Ears(workspace, texts=texts))


@pytest.fixture(scope="module")
def mixed_run(template: WordcheckWorkspace, tmp_path_factory: pytest.TempPathFactory) -> Finished:
    """x0 heard without the label words for a third of the participants: leakage below 1."""
    workspace = template.copy_to(tmp_path_factory.mktemp("mixed"))
    texts = {
        ("x0", segment): " ".join(
            word for word in workspace.text_of[segment].split() if word not in LABEL_WORDS
        )
        for segment in sampled(workspace)
        if int(segment[:3]) % 3 == 0
    }
    return finish(workspace, Ears(workspace, texts=texts))


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio))))


class LevelSmile(FakeSmile):
    """opensmile's stand-in that records the level of what it measures and, like opensmile's
    fixed voicing level, voices fewer frames in quieter audio: 70% at an RMS of 0.065 or
    more, 50% below."""

    def __init__(self) -> None:
        super().__init__()
        self.measured: list[tuple[int, float]] = []  # (samples, RMS) of each functionals call

    def functionals(self, part: np.ndarray) -> pd.Series:
        self.measured.append((len(part), rms(part)))
        return super().functionals(part)

    def voiced_share(self, part: np.ndarray) -> float:
        return 0.7 if rms(part) >= 0.065 else 0.5


class UnvoicedSmile(FakeSmile):
    """How opensmile hears a clip without any pitch: no voiced frame, no voiced stretch."""

    def stretches(self, part: np.ndarray) -> int:
        return 0

    def voiced_share(self, part: np.ndarray) -> float:
        return 0.0


@dataclass
class LevelRun:
    finished: Finished
    smile: LevelSmile


@pytest.fixture(scope="module")
def level_run(template: WordcheckWorkspace, tmp_path_factory: pytest.TempPathFactory) -> LevelRun:
    """opensmile hears the level, and one participant's clips without any pitch."""
    workspace = template.copy_to(tmp_path_factory.mktemp("level"))
    smile = LevelSmile()

    def prepare(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(wordcheck_fakes, "FakeSmile", lambda: smile)
        real = module.measure_segment

        def measure(clip: Any, settings: Any, extractor: Any, segment: str) -> Any:
            if int(segment[:3]) == UNVOICED_SESSION:
                extractor = UnvoicedSmile()
            return real(clip, settings, extractor, segment)

        monkeypatch.setattr(module, "measure_segment", measure)

    return LevelRun(finish(workspace, Ears(workspace), prepare=prepare), smile)


def sampled(workspace: WordcheckWorkspace) -> list[str]:
    return sorted(str(segment) for segment in workspace.sample["segment"])


def every_segment(workspace: WordcheckWorkspace) -> list[str]:
    return sorted(str(segment) for segment in workspace.segments["segment"])


def pinned_design(
    workspace: WordcheckWorkspace,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Labels, gender, the folds of round 1 and the x1 voices, from the workspace's pins."""
    config = workspace.config
    labels = pd.read_parquet(workspace.root / config["labels"]["run"] / "labels.parquet")
    labels.index = labels.index.astype(np.int64)
    partitions = pd.read_parquet(
        workspace.root / config["partitions"]["run"] / "partitions.parquet"
    )
    first = partitions[partitions["round"] == 1]
    folds = pd.Series(
        first["fold"].to_numpy(dtype=np.int64),
        index=pd.Index(first["session"].to_numpy(dtype=np.int64), name="session"),
        name="fold",
    ).sort_index()
    voices = pd.Series(assign_voices(labels, X1Settings.from_config(config).voices), name="voice")
    voices.index = pd.Index(voices.index.astype(np.int64), name="session")
    label = labels["label"].astype(np.int64)
    return label, labels["gender"].astype(str), folds, voices.sort_index()


def per_item_paths(node: Any, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Where run.json could hold a number per segment or participant: IDs as keys, long lists."""
    found: list[tuple[str, ...]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if ITEM_KEY.fullmatch(str(key)):
                found.append((*path, str(key)))
            found += per_item_paths(value, (*path, str(key)))
    elif isinstance(node, list):
        if len(node) > LONGEST_LIST and path not in LONG_LISTS:
            found.append(path)
        for value in node:
            found += per_item_paths(value, path)
    return found


def assert_only_aggregates_leave(run_json: Path, streams: str) -> None:
    """No ID on the streams (but on a failure line) and nothing per item in run.json."""
    shown = "\n".join(line for line in streams.splitlines() if not line.startswith("FAILED: "))
    assert not SEGMENT_ID.search(shown)
    assert not PARTICIPANT_PAIR.search(shown)
    assert not NUMBER_LIST.search(shown)
    assert per_item_paths(json.loads(run_json.read_text(encoding="utf-8"))) == []


# The default mode ----------------------------------------------------------------------------


def test_the_job_records_every_aggregate_and_the_four_verdicts(sample_run: Finished) -> None:
    info = sample_run.info
    out = sample_run.out
    assert info["status"] == "done"
    assert info["mode"] == {"name": "sample"}
    assert {name: verdict["verdict"] for name, verdict in info["verdicts"].items()} == {
        "f4": "PASS",
        "x0_control": "PASS",
        "x1_words": "PASS",
        "x1_voice": "PASS",
    }
    results = info["results"]
    assert {
        "inputs",
        "references",
        "bootstrap",
        "voice",
        "iccs",
        "probe",
        "words",
        "leakage",
        "repeats",
        "wer",
        "x1_short_parts",
        "error_classes",
        "rerun_estimate",
        "timings",
    } <= set(results)
    n = len(sample_run.workspace.sample)
    assert results["inputs"]["wer_segments"] == results["inputs"]["leakage_segments"] == n
    assert results["inputs"]["clips"] == 4 * n
    assert results["bootstrap"]["resamples"] == RESAMPLES
    for view in READING_ORDER:
        for name in RECOGNISERS:
            row = results["words"][view][name]
            assert row["clips"] == n
            assert row["hits"] + row["substitutions"] + row["deletions"] == row["reference_words"]
            leakage = results["leakage"][view][name]
            assert len(leakage["folds"]) == 5
            assert leakage["mean"] == pytest.approx(sum(leakage["folds"]) / 5)
            assert ("interval" in leakage) == (view in ("x0", "rev"))
            assert results["timings"]["reading"][name]["views"][view]["clips"] == n
    assert results["words"]["x0"]["whisper"]["wer"] == 0.0  # the fakes hear x0 word for word
    assert results["words"]["rev"]["whisper"]["heard"] == 0.0
    assert set(results["iccs"]["gate"]) == set(info["settings"]["resolved"]["voice"]["measures"])
    assert all(set(gate) == {"female", "male"} for gate in results["iccs"]["gate"].values())
    assert set(results["probe"]) >= {"x0", "x1", "control", "bar", "imputed"}
    assert set(results["error_classes"]) == {"x1", "x0"}
    assert results["repeats"] == {name: {"clips": 20, "identical": 20} for name in RECOGNISERS}
    for name in ("F4 (rev): PASS", "x0 control: PASS", "x1 words: PASS", "x1 voice: PASS"):
        assert re.search(rf"^{re.escape(name)}  ", out, flags=re.MULTILINE), name
    assert f"run: {FIRST_RUN}\n" in out
    assert f"transcripts.parquet sha256: {info['outputs']['transcripts.parquet']}\n" in out
    assert re.search(r"^done in \d+\.\d min$", out, flags=re.MULTILINE)


def test_the_outputs_hold_their_rows_and_match_run_json(sample_run: Finished) -> None:
    run, info = sample_run.run, sample_run.info
    assert set(info["outputs"]) == {
        "transcripts.parquet",
        "scores.parquet",
        "voice_parts.parquet",
        "voice_segments.parquet",
    }
    for name, sha256 in info["outputs"].items():
        assert sha256_file(run / name) == sha256, name
    assert not (run / "errors.log").exists()
    n = len(sample_run.workspace.sample)
    transcripts = pd.read_parquet(run / "transcripts.parquet")
    assert len(transcripts) == n * 4 * 2
    assert transcripts.groupby(["view", "recogniser"]).size().eq(n).all()
    # The text stays here, on OMNI only.
    assert transcripts["text"].str.lower().str.contains(SECRET.lower()).any()
    scores = pd.read_parquet(run / "scores.parquet")
    participants = 20
    assert (scores["kind"] == "leakage").sum() == participants * 4 * 2
    assert (scores["kind"] == "probe").sum() == participants * 2
    assert set(scores["fold"]) == {1, 2, 3, 4, 5}
    parts = pd.read_parquet(run / "voice_parts.parquet")
    assert set(parts["view"]) == {"x0", "x1"}
    voice = pd.read_parquet(run / "voice_segments.parquet")
    assert len(voice) == n * 2 * 2  # x0 and x1, weighted over the parts and as whole clips


def test_nothing_that_leaves_omni_holds_participant_text_or_a_number_per_segment(
    sample_run: Finished,
) -> None:
    leaving = outside_text(sample_run.run / "run.json", streams=sample_run.out + sample_run.err)
    for word in PRIVATE:
        assert word not in leaving, word
    text = (sample_run.run / "run.json").read_text(encoding="utf-8")
    assert not re.search(r"\b3\d\d_\d{3}\b", text)  # no segment ID: nothing per segment
    assert not re.search(r'"3\d\d":', text)  # no participant ID as a key: nothing per person
    assert_only_aggregates_leave(sample_run.run / "run.json", sample_run.out + sample_run.err)


def test_each_view_is_measured_by_its_own_part_bounds(sample_run: Finished) -> None:
    workspace = sample_run.workspace
    parts = pd.read_parquet(sample_run.run / "voice_parts.parquet")
    for view in ("x0", "x1"):
        folder = workspace.root / workspace.config["view_runs"][view]["run"]
        manifest = pd.read_parquet(folder / "manifest.parquet").set_index("segment")
        measured = parts[parts["view"] == view]
        assert set(measured["segment"]) == set(sampled(workspace))
        for segment, group in measured.groupby("segment"):
            row = manifest.loc[segment]
            bounds = zip(row["part_starts"], row["part_ends"], strict=True)
            lengths = [int(end) - int(start) for start, end in bounds]
            assert group.sort_values("part")["samples"].tolist() == lengths, (view, segment)


def test_the_voice_table_holds_each_views_own_clips_measured_on_copies(
    sample_run: Finished,
) -> None:
    workspace = sample_run.workspace
    table = pd.read_parquet(sample_run.run / "voice_segments.parquet")
    table = table.set_index(["segment", "view", "kind"])[list(FUNCTIONAL_NAMES)]
    smile = FakeSmile()  # gives the same values for the same samples
    for view in ("x0", "x1"):
        folder = workspace.root / workspace.config["view_runs"][view]["run"]
        manifest = pd.read_parquet(folder / "manifest.parquet").set_index("segment")
        for segment in sampled(workspace):
            row = manifest.loc[segment]
            samples, _ = sf.read(folder / row["file"], dtype="int16")
            copy, _ = level_copy(np.asarray(samples, dtype=np.float64) / 32768, 0.05, 0.99)
            parts = part_measures(copy, row["part_starts"], row["part_ends"], smile, 960)
            expected = {"parts": segment_measures(parts), "whole": smile.functionals(copy)}
            for kind, values in expected.items():
                np.testing.assert_allclose(
                    table.loc[(segment, view, kind)].to_numpy(dtype=np.float64),
                    values[list(FUNCTIONAL_NAMES)].to_numpy(dtype=np.float64),
                    rtol=0,
                    atol=1e-12,
                    equal_nan=True,
                    err_msg=f"{view} {kind}",
                )


def test_x1s_wer_is_split_by_the_segments_with_a_part_of_one_or_two_words(
    sample_run: Finished,
) -> None:
    workspace = sample_run.workspace
    cut = pd.read_parquet(workspace.root / workspace.config["segments"]["run"] / "words.parquet")
    smallest = segment_parts(cut).groupby("segment")["n_words"].min()
    segments = sampled(workspace)
    short = [segment for segment in segments if smallest[segment] <= 2]
    assert sorted({int(smallest[segment]) for segment in short}) == [1, 2]
    for name in RECOGNISERS:
        split = sample_run.info["results"]["x1_short_parts"][name]
        assert split["short_parts"]["segments"] == len(short)
        assert split["rest"]["segments"] == len(segments) - len(short)


def test_the_voice_numbers_follow_from_the_voice_table_by_the_fixed_rules(
    sample_run: Finished,
) -> None:
    info, workspace = sample_run.info, sample_run.workspace
    label, gender, folds, voices = pinned_design(workspace)
    voice = workspace.config["word_check"]["voice"]
    table = pd.read_parquet(sample_run.run / "voice_segments.parquet")
    weighted = table[table["kind"] == "parts"]
    views = {
        view: weighted[weighted["view"] == view].set_index("segment")[list(FUNCTIONAL_NAMES)]
        for view in ("x0", "x1")
    }
    owners = pd.Series([int(s.split("_")[0]) for s in views["x0"].index], index=views["x0"].index)
    described = pd.DataFrame({"gender": owners.map(gender), "voice": owners.map(voices)})
    iccs = voice_iccs(views["x0"], views["x1"], described, voice["measures"])
    results = info["results"]
    for measure in voice["measures"]:
        for name in ("female", "male"):
            recorded = results["iccs"]["gate"][measure][name]["icc"]
            assert recorded == pytest.approx(iccs["gate"][measure][name]["icc"], abs=1e-12)
    for view in ("x0", "x1"):
        features = participant_means(views[view], owners)
        aurocs = probe_aurocs(
            features,
            gender,
            voices,
            label,
            folds,
            voice["probe_logistic_c"],
            voice["probe_max_iter"],
        )
        assert results["probe"][view]["folds"] == pytest.approx(aurocs.tolist(), abs=1e-12)
    # The verdict reads the six gate values and each view's own probe.
    assert info["verdicts"]["x1_voice"]["iccs"] == pytest.approx(gate_values(results["iccs"]))
    probe = info["verdicts"]["x1_voice"]["probe"]
    assert probe["x0"] == results["probe"]["x0"]["mean"]
    assert probe["x1"] == results["probe"]["x1"]["mean"]
    assert probe["control"] == results["probe"]["control"]["mean"]
    assert results["probe"]["x0"]["mean"] != results["probe"]["x1"]["mean"]


def test_words_heard_by_chance_and_the_leakage_scores_follow_from_the_transcripts(
    sample_run: Finished,
) -> None:
    info, workspace = sample_run.info, sample_run.workspace
    label, _, folds, _ = pinned_design(workspace)
    segments = sampled(workspace)
    sample = pd.DataFrame({"segment": segments, "session": [int(s[:3]) for s in segments]})
    references = [words(workspace.text_of[segment]) for segment in segments]
    by_chance = [references[int(partner)] for partner in chance_partners(sample)]
    # The classifiers learn from the stored words of every segment, not only the sampled.
    models = fit_leakage(
        workspace.segments[["session", "text"]],
        label,
        folds,
        LeakageSettings.from_config(workspace.config),
    )
    transcripts = pd.read_parquet(sample_run.run / "transcripts.parquet")
    scores = pd.read_parquet(sample_run.run / "scores.parquet")
    for view in READING_ORDER:
        for name in RECOGNISERS:
            rows = transcripts[(transcripts["view"] == view) & (transcripts["recogniser"] == name)]
            texts = rows.set_index("segment").loc[segments, "text"].tolist()
            chance = wer_counts(by_chance, [words(text) for text in texts], segments).heard
            assert info["results"]["words"][view][name]["chance_heard"] == pytest.approx(chance)
            expected = person_scores(models, folds, sample[["session"]], texts).sort_index()
            mine = (scores["kind"] == "leakage") & (scores["view"] == view)
            recorded = scores[mine & (scores["recogniser"] == name)].set_index("session")["score"]
            np.testing.assert_allclose(recorded.sort_index(), expected, rtol=0, atol=1e-9)
    x0 = info["results"]["words"]["x0"]["whisper"]
    assert x0["chance_heard"] < x0["heard"] == 1.0


def clip_lengths(workspace: WordcheckWorkspace, view: str) -> dict[str, int]:
    folder = workspace.root / workspace.config["view_runs"][view]["run"]
    manifest = pd.read_parquet(folder / "manifest.parquet")
    return dict(zip(manifest["segment"].astype(str), manifest["samples"].astype(int), strict=True))


def test_the_voice_is_measured_on_copies_of_the_clips_at_x1s_level(level_run: LevelRun) -> None:
    workspace = level_run.finished.workspace
    measured = [s for s in sampled(workspace) if int(s[:3]) != UNVOICED_SESSION]
    wholes = {length for view in ("x0", "x1") for length in clip_lengths(workspace, view).values()}
    seen = [level for length, level in level_run.smile.measured if length in wholes]
    # Every whole clip of x0 and x1 at an RMS of 0.05 (no copy needed the peak guard here);
    # a clip of one part is measured twice at its length, as its part and as a whole.
    assert len(seen) >= 2 * len(measured)
    np.testing.assert_allclose(seen, 0.05, rtol=0, atol=1e-9)
    assert level_run.finished.info["results"]["voice"]["peak_guard"] == {"x0": 0, "x1": 0}


def test_x0s_voiced_time_at_its_stored_level_is_measured_on_the_stored_clip(
    level_run: LevelRun,
) -> None:
    finished = level_run.finished
    workspace = finished.workspace
    folder = workspace.root / workspace.config["view_runs"]["x0"]["run"]
    manifest = pd.read_parquet(folder / "manifest.parquet").set_index("segment")
    stored = 0.0
    for segment in sampled(workspace):
        row = manifest.loc[segment]
        samples, _ = sf.read(folder / row["file"], dtype="int16")
        audio = np.asarray(samples, dtype=np.float64) / 32768
        stored += voiced_seconds(audio, row["part_starts"], row["part_ends"], LevelSmile(), 960)
    parts = pd.read_parquet(finished.run / "voice_parts.parquet")
    copy = float(parts.loc[parts["view"] == "x0", "voiced_seconds"].sum())
    voice = finished.info["results"]["voice"]
    assert voice["x0_voiced_seconds"] == pytest.approx({"stored_level": stored, "copy": copy})
    assert voice["x0_voiced_share_at_stored_level"] == pytest.approx(stored / copy)
    assert voice["x0_voiced_share_at_stored_level"] > 1.05  # the fakes' x0 is louder stored


def test_a_participant_without_a_voiced_frame_is_imputed_and_counted(level_run: LevelRun) -> None:
    finished = level_run.finished
    results = finished.info["results"]
    measures = finished.workspace.config["word_check"]["voice"]["measures"]
    pitch, loudness, stretches = measures
    for view in ("x0", "x1"):
        # The 50 measures of voiced frames and the 2 of the voiced stretches' lengths.
        assert results["probe"]["imputed"][view] == {
            "values": 52,
            "participants": 1,
            "gate_measures": {pitch: 1, loudness: 0, stretches: 0},
        }
    unvoiced = sum(int(s[:3]) == UNVOICED_SESSION for s in sampled(finished.workspace))
    assert results["iccs"]["left_out"][pitch] == {"female": unvoiced, "male": 0}  # 305: a woman
    assert finished.info["status"] == "done"


def test_x0s_interval_is_drawn_from_its_own_stream_at_the_configured_level(
    mixed_run: Finished,
) -> None:
    info, workspace = mixed_run.info, mixed_run.workspace
    label, _, folds, _ = pinned_design(workspace)
    config = workspace.config
    sets, redraws = bootstrap_sets(
        label, folds, BootstrapSettings.from_config(config).rng(), RESAMPLES
    )
    assert info["results"]["bootstrap"]["redraws"] == redraws
    level = config["word_check"]["rules"]["x0_leakage_interval"]
    assert level == 0.95
    scores = pd.read_parquet(mixed_run.run / "scores.parquet")
    for name in RECOGNISERS:
        mine = (scores["kind"] == "leakage") & (scores["view"] == "x0")
        found = scores[mine & (scores["recogniser"] == name)].set_index("session")["score"]
        recorded = info["results"]["leakage"]["x0"][name]["interval"]
        assert recorded[0] < 0.9 < recorded[1]  # wide enough for the level and stream to show
        expected = bootstrap_interval(sets, found.sort_index(), label, folds, level)
        assert recorded == pytest.approx(list(expected), abs=1e-12)
        assert info["verdicts"]["x0_control"]["intervals"][name] == recorded


def test_the_stages_run_in_order_with_one_model_at_a_time(sample_run: Finished) -> None:
    events = sample_run.ears.events

    def where(test: Callable[[tuple[Any, ...]], bool]) -> list[int]:
        return [index for index, event in enumerate(events) if test(event)]

    guard_reads = where(lambda e: e[0] == "read" and e[2] == "guard")
    clips = where(lambda e: e[:2] == ("stage", "read_view_clip"))
    assert len(guard_reads) == 2 * len(PUBLIC_VOICES)
    assert max(guard_reads) < min(clips)  # the public guard before any participant clip
    loads = where(lambda e: e[0] == "load")
    assert [events[k] for k in loads] == [("load", "whisper", 1), ("load", "wav2vec2", 1)] * 2
    for before, after in itertools.pairwise(loads):
        assert ("released", events[before][1]) in events[before:after]
    own_stage = loads[2]  # Whisper loads for its own stage
    assert max(clips) < own_stage
    for name in ("fit_leakage", "bootstrap_sets", "control_aurocs", "voice_iccs", "probe_aurocs"):
        assert where(lambda e, name=name: e[:2] == ("stage", name)), name
        assert max(where(lambda e, name=name: e[:2] == ("stage", name))) < own_stage, name
    segments = sampled(sample_run.workspace)
    first = [(view, segment) for view in READING_ORDER for segment in segments]
    again = [(view, segment) for view in READING_ORDER for segment in segments[:5]]
    assert sample_run.ears.reads("whisper") == first + again
    assert sample_run.ears.reads("wav2vec2") == first + again
    (written,) = where(lambda e: e == ("stage", "write_table", "transcripts.parquet"))
    last_read = max(where(lambda e: e[0] == "read"))
    scoring = min(where(lambda e: e[:2] == ("stage", "word_rows")))
    assert last_read < written < scoring


def test_run_json_holds_whispers_options_from_the_public_guard_and_no_clip_duration(
    sample_run: Finished,
) -> None:
    info = sample_run.info
    assert info["settings"]["whisper_resolved"] == {
        "beam_size": 5,
        "temperatures": [0.0],
        "heard": "guard",  # resolved on a public file, never on a participant clip
    }
    assert "duration" not in (sample_run.run / "run.json").read_text(encoding="utf-8")


def test_run_json_records_every_pinned_input_and_the_model_snapshots(sample_run: Finished) -> None:
    info, workspace = sample_run.info, sample_run.workspace
    config = workspace.config
    assert set(info["inputs"]) == {
        "segments",
        "labels",
        "partitions",
        "wordcheck_sample",
        "cut_words",
        "transcripts",
        "views",
        "public_guard",
    }
    assert info["inputs"]["views"] == {
        view: {"file": pin["run"], "sha256": pin["sha256"]}
        for view, pin in config["view_runs"].items()
    }
    assert info["inputs"]["labels"] == {
        "file": config["labels"]["run"],
        "sha256": config["labels"]["sha256"],
    }
    assert info["inputs"]["public_guard"]["files"] == len(PUBLIC_VOICES)
    pins = load_pins(REPO / "configs" / "pilot" / "models.yaml")
    for name, pin in (("whisper", "faster_whisper"), ("wav2vec2", "wav2vec2_960h")):
        model = info["models"][name]
        assert (model["repo_id"], model["revision"]) == (pins[pin].repo_id, pins[pin].revision)
        assert set(model["blobs"]) == set(MODEL_FILES[pin])
        assert model["identity"]
    assert info["config"]["sha256"] == sha256_file(workspace.config_path)
    assert info["settings"]["word_check"] == json.loads(json.dumps(config["word_check"]))
    assert info["seeds"]["bootstrap"]["item_seed"] == 3402805958923752013
    assert info["seeds"]["partitions"] == 101
    assert info["environment"]["hf_hub_offline"] == "1"
    assert {"packages", "cpu", "uv_lock_sha256", "libsndfile"} <= set(info["environment"])


# Rules that fail, and faults ----------------------------------------------------------------


def test_a_failed_rule_is_a_result_and_the_job_still_ends_done(
    workspace: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    segments = sampled(workspace)
    texts = {}
    for segment in segments:
        stored = workspace.text_of[segment]
        texts[("rev", segment)] = stored  # rev heard word for word: F4 fails
        texts[("x1", segment)] = " ".join(stored.split()[20:])  # x1 loses half its words
    run = run_job(workspace, monkeypatch, Ears(workspace, texts=texts))
    out = capsys.readouterr().out
    info = info_of(run)
    assert info["status"] == "done"
    assert {name: verdict["verdict"] for name, verdict in info["verdicts"].items()} == {
        "f4": "FAIL",
        "x0_control": "PASS",
        "x1_words": "FAIL",
        "x1_voice": "PASS",
    }
    assert info["verdicts"]["f4"]["failed"] == [
        "whisper WER",
        "wav2vec2 WER",
        "whisper leakage",
        "wav2vec2 leakage",
    ]
    # Each output is compared with its own segment's reference.
    references = [words(workspace.text_of[segment]) for segment in segments]
    heard = [words(texts[("x1", segment)]) for segment in segments]
    expected = wer_counts(references, heard, segments)
    for name in RECOGNISERS:
        assert info["results"]["words"]["x1"][name]["wer"] == expected.wer
        assert info["results"]["wer"]["x1"][name] == expected.wer
        classes = info["results"]["error_classes"]["x1"][name]["all"]
        assert classes["deletions"] == expected.deletions
        assert classes["reference_words"] == expected.reference_words
    assert re.search(r"^F4 \(rev\): FAIL  ", out, flags=re.MULTILINE)
    assert re.search(r"^x1 words: FAIL  ", out, flags=re.MULTILINE)


def test_a_leakage_fit_that_does_not_converge_stops_the_job_before_the_asr_systems_read(
    workspace: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    workspace.config["word_check"]["leakage"]["max_iter"] = 1
    workspace.write_config()
    ears = Ears(workspace)
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, ears)
    out = capsys.readouterr().out
    info = info_of(workspace.run_dir())
    assert info["status"] == "failed"
    assert re.fullmatch(
        r"the leakage classifier of fold \d did not converge in 1 iterations",
        info["error"]["message"],
    )
    assert "FAILED: StepError: the leakage classifier of fold" in out
    assert [event[1] for event in ears.events if event[0] == "load"] == ["whisper", "wav2vec2"]
    assert ears.reads("whisper") == ears.reads("wav2vec2") == []


def fail_in_the_recogniser(
    workspace: WordcheckWorkspace, ears: Ears, module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    ears.fail_on = ("wav2vec2", "x1", sampled(workspace)[7])


def fail_in_the_fit(
    workspace: WordcheckWorkspace, ears: Ears, module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(self: Any, *args: Any, **kwargs: Any) -> None:
        raise ValueError(f"cannot fit {SECRET}")

    monkeypatch.setattr(wordcheck.LogisticRegression, "fit", failing)


def fail_in_the_scoring(
    workspace: WordcheckWorkspace, ears: Ears, module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(found: list[str]) -> float:
        raise KeyError(f"{SECRET} {' '.join(found)}")

    monkeypatch.setattr(module, "repetition_share", failing)


def fail_while_writing(
    workspace: WordcheckWorkspace, ears: Ears, module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = pd.DataFrame.to_parquet

    def failing(self: pd.DataFrame, path: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(path).name == "scores.parquet":
            raise OSError(f"cannot write {SECRET}")
        return real(self, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", failing)


@pytest.mark.parametrize(
    ("plant", "error", "transcribed"),
    [
        (fail_in_the_recogniser, "ValueError", False),
        (fail_in_the_fit, "ValueError", False),
        (fail_in_the_scoring, "KeyError", True),
        (fail_while_writing, "OSError", True),
    ],
)
def test_a_failed_job_never_shows_participant_text_outside_its_transcripts(
    workspace: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    plant: Callable[..., None],
    error: str,
    transcribed: bool,
) -> None:
    module = load_script("check_views")
    ears = Ears(workspace)
    plant(workspace, ears, module, monkeypatch)
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, ears, module=module)
    out, err = capsys.readouterr()
    run = workspace.run_dir()
    info = info_of(run)
    assert info["status"] == "failed"
    assert info["error"]["type"] == error
    assert "message" not in info["error"]
    assert f"FAILED: {error}" in out
    assert f"traceback frames in {run / 'errors.log'}" in out
    assert f"builtins.{error}" in (run / "errors.log").read_text(encoding="utf-8")
    if plant is fail_in_the_recogniser:
        assert info["error"]["segment"] == sampled(workspace)[7]
    leaving = outside_text(run / "run.json", run / "errors.log", streams=out + err)
    for word in PRIVATE:
        assert word not in leaving, word
    assert_only_aggregates_leave(run / "run.json", out + err)
    transcripts = run / "transcripts.parquet"
    assert transcripts.exists() == transcribed  # written before the scoring
    if transcribed:
        texts = pd.read_parquet(transcripts)["text"].str.lower()
        assert texts.str.contains(SECRET.lower()).any()
        assert sha256_file(transcripts) == info["outputs"]["transcripts.parquet"]


def test_a_public_guard_above_its_bound_stops_the_job_before_any_participant_clip_is_read(
    workspace: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("check_views")
    ears = Ears(workspace, guard_text=REV_TEXT)
    clips: list[Any] = []
    monkeypatch.setattr(module, "read_view_clip", lambda *args: clips.append(args))
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, ears, module=module)
    out = capsys.readouterr().out
    info = info_of(workspace.run_dir())
    assert info["status"] == "failed"
    assert info["error"]["message"].startswith("the public guard failed: whisper's WER 1.0000")
    assert info["error"]["message"].endswith("; no participant clip was read")
    assert "FAILED: StepError: the public guard failed" in out
    assert clips == []
    assert [event[:2] for event in ears.events if event[0] == "load"] == [("load", "whisper")]
    assert ears.reads("whisper") == []


def change_the_guard(workspace: WordcheckWorkspace, change: str) -> None:
    """Change the tool check's run that the public guard reads."""
    folder = workspace.root / "derived" / "check_view_tools" / "run-1"
    info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    voice = folder / "voices" / "af_heart.wav"
    if change == "failed":
        info["status"] = "failed"
    elif change == "verdict":
        info["verdict"] = "FAIL"
    elif change == "byte":
        voice.write_bytes(voice.read_bytes() + b"\0")
    elif change == "rate":  # the right SHA-256 in run.json, but 24 kHz
        samples, _ = sf.read(voice, dtype="int16")
        sf.write(voice, samples, 24_000, subtype="PCM_16")
        info["outputs"]["voices/af_heart.wav"] = sha256_file(voice)
    elif change == "none":
        info["settings"]["voices"] = []
    elif change == "ten words":
        info["settings"]["text"] = "one two three four five six seven eight nine ten"
    (folder / "run.json").write_text(json.dumps(info), encoding="utf-8")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("failed", "the public guard's run is not a tool check that passed"),
        ("verdict", "the public guard's run is not a tool check that passed"),
        ("byte", "the public guard's voices/af_heart.wav does not match its run.json"),
        ("rate", "the public guard's voices/af_heart.wav is not 16 kHz mono 16-bit PCM"),
        ("none", "the public guard's run lists no voice files"),
        (
            "wav2vec2",
            "the public guard failed: wav2vec2's WER 1.0000 on the tool check's public text is"
            " above 0.1; no participant clip was read",
        ),
    ],
)
def test_the_public_guard_refuses_files_or_a_reading_it_cannot_trust(
    workspace: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    change: str,
    message: str,
) -> None:
    module = load_script("check_views")
    change_the_guard(workspace, change)
    ears = Ears(workspace, guard_texts={"wav2vec2": REV_TEXT} if change == "wav2vec2" else {})
    clips: list[Any] = []
    monkeypatch.setattr(module, "read_view_clip", lambda *args: clips.append(args))
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, ears, module=module)
    assert f"FAILED: StepError: {message}" in capsys.readouterr().out
    info = info_of(workspace.run_dir())
    assert (info["status"], info["error"]["message"]) == ("failed", message)
    assert clips == []
    loads = [event[1] for event in ears.events if event[0] == "load"]
    assert loads == (["whisper", "wav2vec2"] if change == "wav2vec2" else [])


def test_a_public_guard_at_its_bound_passes(
    workspace: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    # Ten words, one of them heard wrongly in each of the seven files: a WER of 0.10 exactly.
    change_the_guard(workspace, "ten words")
    ears = Ears(workspace, guard_text="one two three four five six seven eight nine eleven")
    run = run_job(workspace, monkeypatch, ears)
    capsys.readouterr()
    info = info_of(run)
    assert info["status"] == "done"
    assert {name: info["public_guard"][name]["wer"] for name in RECOGNISERS} == dict.fromkeys(
        RECOGNISERS, 0.1
    )


@pytest.mark.parametrize("change", ["sample", "lines"])
def test_inputs_that_do_not_fit_together_stop_the_job_before_any_clip_is_read(
    workspace: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    change: str,
) -> None:
    if change == "sample":  # a sample without one participant, pinned as it is
        sample = workspace.sample[workspace.sample["session"] != 310]
        run = workspace.root / "derived" / "wordcheck_sample" / "run-2"
        sha256 = write_pinned(run, sample, "sample.parquet")
        workspace.config["wordcheck_sample"] = {
            "run": "derived/wordcheck_sample/run-2",
            "sha256": sha256,
        }
        workspace.write_config()
        message = (
            "the sample, the labels and round 1 of the partitions do not name the same"
            " participants (19, 20 and 20)"
        )
    else:  # one word of a transcript line no longer what the segments stored
        path = workspace.root / "edaic" / "sessions" / "300_P" / "300_Transcript.csv"
        lines = pd.read_csv(path)
        first, *rest = str(lines.at[0, "Text"]).split()
        lines.at[0, "Text"] = " ".join([first + "x", *rest])
        lines.to_csv(path, index=False)
        message = "segment 300_000: its transcript lines do not give its stored words"
    module = load_script("check_views")
    clips: list[Any] = []
    real = module.read_view_clip
    monkeypatch.setattr(module, "read_view_clip", lambda *args: clips.append(args) or real(*args))
    ears = Ears(workspace)
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, ears, module=module)
    out = capsys.readouterr().out
    info = info_of(workspace.run_dir())
    assert (info["status"], info["error"]["message"]) == ("failed", message)
    assert ears.reads("whisper") == ears.reads("wav2vec2") == []
    if change == "sample":
        assert clips == []
    for word in PRIVATE:
        assert word not in out + json.dumps(info), word


def test_a_clip_that_does_not_match_its_manifest_stops_the_job_before_any_transcription(
    workspace: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    segment = sampled(workspace)[3]
    session = segment.split("_")[0]
    path = workspace.root / "views" / "x2" / "run-1" / session / f"{segment}.flac"
    path.write_bytes(path.read_bytes() + b"\0")
    ears = Ears(workspace)
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, ears)
    out = capsys.readouterr().out
    info = info_of(workspace.run_dir())
    assert info["error"] == {
        "type": "StepError",
        "segment": segment,
        "part": None,
        "message": f"segment {segment}: its file does not match the SHA-256 of its manifest row",
    }
    assert f"FAILED: StepError in segment {segment}: segment {segment}: its file" in out
    assert ears.reads("whisper") == ears.reads("wav2vec2") == []


# Refusals -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("job", "submit this as a job"),
        ("offline", "HF_HUB_OFFLINE is not 1"),
        ("dirty", "the checkout has uncommitted changes"),
        ("gpu", "no CUDA GPU"),
    ],
)
def test_the_job_refuses_to_run_outside_a_clean_offline_gpu_job(
    workspace: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    change: str,
    message: str,
) -> None:
    module = load_script("check_views")
    patch_job(module, workspace, monkeypatch, Ears(workspace))
    monkeypatch.setenv("SLURM_JOB_ID", JOB)
    monkeypatch.setattr(sys, "argv", ["check_views.py"])
    if change == "job":
        monkeypatch.delenv("SLURM_JOB_ID")
    elif change == "offline":
        monkeypatch.delenv("HF_HUB_OFFLINE")
    elif change == "dirty":
        monkeypatch.setattr(module, "git_state", lambda: {"commit": "c" * 40, "dirty": True})
    else:
        monkeypatch.setattr(module, "gpu_facts", lambda: {"gpu": None, "cuda": None})
    with pytest.raises(SystemExit):
        module.main()
    assert f"FAILED: StepError: {message}" in capsys.readouterr().out
    assert not workspace.run_dir().exists()


@pytest.mark.parametrize(
    "args",
    [
        ["--views"],
        ["--views", "rev", "rev"],
        ["--views", "--rerun", FIRST_RUN],
        ["--rerun"],
        ["--rerun", FIRST_RUN, FIRST_RUN],
        ["--all-segments"],
        ["x0"],
    ],
)
def test_the_job_refuses_any_other_command_line(
    workspace: WordcheckWorkspace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    args: list[str],
) -> None:
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, Ears(workspace), *args)
    assert "FAILED: StepError: usage: check_views.py [--views" in capsys.readouterr().out
    assert not workspace.run_dir().exists()


def test_the_job_listens_for_slurms_stop_before_anything_else(
    workspace: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("check_views")
    patch_job(module, workspace, monkeypatch, Ears(workspace))
    calls: list[str] = []
    monkeypatch.setattr(module, "stop_on_sigterm", lambda: calls.append("listening"))
    monkeypatch.setattr(sys, "argv", ["check_views.py"])
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit):
        module.main()
    assert calls == ["listening"]  # even before the first refusal
    assert "FAILED: StepError: submit this as a job" in capsys.readouterr().out


def test_a_job_that_slurm_stops_is_left_failed_naming_its_segment(
    workspace: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = load_script("check_views")
    real = module.read_view_clip
    calls: list[int] = []

    def stopped(*args: Any) -> Any:  # what stop_on_sigterm's handler raises
        calls.append(1)
        if len(calls) == 3:
            raise SystemExit(STOPPED)
        return real(*args)

    monkeypatch.setattr(module, "read_view_clip", stopped)
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, Ears(workspace), module=module)
    segment = sampled(workspace)[2]
    assert f"FAILED: SystemExit in segment {segment}: {STOPPED}" in capsys.readouterr().out
    run = workspace.run_dir()
    info = info_of(run)
    assert (info["status"], info["error"]["message"]) == ("failed", STOPPED)
    assert info["error"]["segment"] == segment
    assert (run / "errors.log").is_file()
    assert not (run / "transcripts.parquet").exists()


def test_the_views_mode_refuses_a_view_without_a_pin(
    workspace: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, Ears(workspace), "--views", "rev_whole")
    out = capsys.readouterr().out
    assert "FAILED: StepError: --views names views without a pin in view_runs: ['rev_whole']" in out
    assert not workspace.run_dir().exists()


def test_a_second_run_with_the_same_job_id_never_writes_into_the_first(
    workspace: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    first = workspace.run_dir()
    first.mkdir(parents=True)
    (first / "run.json").write_text('{"status": "done"}\n', encoding="utf-8")
    ears = Ears(workspace)
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, ears)
    assert "FAILED: FileExistsError" in capsys.readouterr().out
    assert (first / "run.json").read_text(encoding="utf-8") == '{"status": "done"}\n'
    assert sorted(path.name for path in first.iterdir()) == ["run.json"]
    assert ears.events == []


# The leakage rerun ----------------------------------------------------------------------------


def test_a_control_that_asks_for_the_rerun_makes_f4_wait_and_prints_the_rerun(
    waiting_run: Finished,
) -> None:
    info, out = waiting_run.info, waiting_run.out
    assert info["status"] == "done"
    control = info["verdicts"]["x0_control"]
    assert control["verdict"] == "RERUN NEEDED"
    assert control["intervals"] == {name: [0.5, 0.5] for name in RECOGNISERS}
    f4 = info["verdicts"]["f4"]
    assert (f4["verdict"], f4["leakage_provisional"]) == ("WAITS FOR THE RERUN", True)
    assert re.search(
        r"^F4 \(rev\): WAITS FOR THE RERUN  .*\(leakage provisional while waiting\)$",
        out,
        flags=re.MULTILINE,
    )
    assert re.search(r"^x0 control: RERUN NEEDED  ", out, flags=re.MULTILINE)
    time = info["results"]["rerun_estimate"]["slurm_time"]
    assert re.fullmatch(r"\d\d:\d\d:00", time)
    # All segments at the sample's rates: Whisper per clip, wav2vec2 per second of audio.
    workspace = waiting_run.workspace
    readings = pd.read_parquet(waiting_run.run / "transcripts.parquet")
    rates = readings[readings["view"].isin(["x0", "rev"])]
    whisper = len(workspace.segments) * sum(
        rates[(rates["view"] == view) & (rates["recogniser"] == "whisper")]["seconds"].mean()
        for view in ("x0", "rev")
    )
    audio = sum(sum(clip_lengths(workspace, view).values()) for view in ("x0", "rev"))
    wav2vec2 = rates[rates["recogniser"] == "wav2vec2"]
    seconds = whisper + audio / wav2vec2["samples"].sum() * wav2vec2["seconds"].sum() + 15 * 60
    minutes = math.ceil(seconds * workspace.config["word_check"]["rerun_time_factor"] / 60)
    assert time == f"{minutes // 60:02d}:{minutes % 60:02d}:00"
    assert info["results"]["rerun_estimate"]["estimate_hours"] == pytest.approx(seconds / 3600)
    assert (
        "the x0 control calls for the leakage rerun: scripts/slurm/submit.sh"
        f" --time={time} check_views --rerun {FIRST_RUN}\n"
    ) in out


def test_the_rerun_reads_x0_and_rev_on_all_segments_and_replaces_only_the_leakage(
    waiting_run: Finished,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    workspace = waiting_run.workspace.copy_to(tmp_path)
    # x0 is now heard as it is; rev, outside the sample, as a long loop (a WER above 1).
    outside = sorted(set(every_segment(workspace)) - set(sampled(workspace)))
    ears = Ears(workspace, texts={("rev", s): " ".join([REV_TEXT] * 12) for s in outside})
    run = run_job(workspace, monkeypatch, ears, "--rerun", FIRST_RUN, job=RERUN_JOB)
    out = capsys.readouterr().out
    info, first = info_of(run), waiting_run.info
    assert info["status"] == "done"
    assert info["mode"] == {"name": "rerun", "first_run": FIRST_RUN}
    assert info["first_run"]["transcripts_sha256"] == first["outputs"]["transcripts.parquet"]
    segments = every_segment(workspace)
    assert len(segments) > len(waiting_run.workspace.sample)
    once = [(view, segment) for view in ("rev", "x0") for segment in segments]
    again = [(view, segment) for view in ("rev", "x0") for segment in segments[:5]]
    assert ears.reads("whisper") == ears.reads("wav2vec2") == once + again
    results = info["results"]
    assert results["wer"] == first["results"]["wer"]  # the sample's WER stands
    assert results["wer_all_segments"]["x0"] == dict.fromkeys(RECOGNISERS, 0.0)
    for view in ("x0", "rev"):
        for name in RECOGNISERS:
            assert results["leakage"][view][name]["segments"] == len(segments)
    assert results["leakage"]["x0"]["whisper"]["mean"] == 1.0  # was 0.5 on the sample
    control, f4 = info["verdicts"]["x0_control"], info["verdicts"]["f4"]
    assert set(info["verdicts"]) == {"f4", "x0_control"}
    assert (control["verdict"], control["after_rerun"]) == ("FAIL", True)
    assert control["failed"] == ["whisper WER", "wav2vec2 WER"]  # the first run's WER of x0
    assert control["wer"] == first["verdicts"]["x0_control"]["wer"]
    assert control["intervals"] == {name: [1.0, 1.0] for name in RECOGNISERS}
    assert (f4["verdict"], f4["leakage_provisional"]) == ("PASS", False)
    assert f4["wer"] == first["verdicts"]["f4"]["wer"]  # the sample's, not all segments'
    assert all(results["wer_all_segments"]["rev"][name] > 1.0 for name in RECOGNISERS)
    assert re.search(r"^F4 \(rev\): PASS  ", out, flags=re.MULTILINE)
    assert re.search(r"^x0 control: FAIL  ", out, flags=re.MULTILINE)
    assert "x1 words" not in out
    assert "calls for the leakage rerun" not in out
    assert not (run / "voice_parts.parquet").exists()


def change_the_first_run(workspace: WordcheckWorkspace, key: str, value: Any) -> None:
    path = workspace.run_dir(JOB) / "run.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info[key] = value
    path.write_text(json.dumps(info), encoding="utf-8")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing", "no word-check run in derived/wordcheck/run-999"),
        ("mode", "the first run is not a run of the word check on the sample"),
        ("not done", "the first run did not finish: its run.json does not say done"),
        ("commit", "the first run was made with another commit; rerun from its commit"),
        ("config", "the first run was made with another config"),
        ("models", "the first run was made with another config"),
        ("inputs", "the first run was made from other pinned inputs"),
        ("control", "the first run's x0 control said PASS, not RERUN NEEDED"),
        ("transcripts", "the first run's transcripts.parquet does not match its run.json"),
    ],
)
def test_the_rerun_refuses_a_first_run_that_it_cannot_follow(
    waiting_run: Finished,
    sample_run: Finished,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    change: str,
    message: str,
) -> None:
    source = sample_run if change == "control" else waiting_run
    workspace = source.workspace.copy_to(tmp_path)
    first, commit = FIRST_RUN, "c" * 40
    if change == "missing":
        first = "derived/wordcheck/run-999"
    elif change == "mode":
        change_the_first_run(workspace, "mode", {"name": "views", "views": ["rev"]})
    elif change == "not done":
        change_the_first_run(workspace, "status", "failed")
    elif change == "commit":
        commit = "d" * 40
    elif change == "config":
        workspace.config["word_check"]["rerun_time_factor"] = 2.0
        workspace.write_config()
    elif change == "models":
        change_the_first_run(workspace, "models_config", {"file": "x", "sha256": "0" * 64})
    elif change == "inputs":  # the transcripts' checksums, which no config pin covers
        (workspace.root / "edaic" / "manifest" / "sessions_sha256.txt").write_text("other\n")
    elif change == "transcripts":
        path = workspace.run_dir(JOB) / "transcripts.parquet"
        path.write_bytes(path.read_bytes() + b"\0")
    ears = Ears(workspace)
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, ears, "--rerun", first, job=RERUN_JOB, commit=commit)
    out = capsys.readouterr().out
    assert f"FAILED: StepError: {message}" in out
    info = info_of(workspace.run_dir(RERUN_JOB))
    assert (info["status"], info["error"]["message"]) == ("failed", message)
    assert ears.events == []  # refused before any model loads


# The views mode -----------------------------------------------------------------------------


VIEWS_JOB = "702"


def pin_the_backup(first: Finished, folder: Path) -> WordcheckWorkspace:
    """A copy of a finished first run's workspace with that run pinned (wordcheck_run) and a
    fake pin of the whole-part backup of rev: the clips of rev, under another name."""
    workspace = first.workspace.copy_to(folder)
    workspace.config["wordcheck_run"] = {
        "run": FIRST_RUN,
        "sha256": first.info["outputs"]["transcripts.parquet"],
    }
    workspace.config["view_runs"]["rev_whole"] = dict(workspace.config["view_runs"]["rev"])
    workspace.write_config()
    return workspace


def pin_a_rerun(workspace: WordcheckWorkspace, first_run: str = FIRST_RUN) -> dict[str, str]:
    """A fake leakage rerun of the first run, pinned (leakage_rerun)."""
    rerun = workspace.run_dir(RERUN_JOB)
    rerun.mkdir(parents=True)
    (rerun / "transcripts.parquet").write_bytes(b"a pinned leakage rerun")
    mode = {"name": "rerun", "first_run": first_run}
    info = {"step": "check_views", "mode": mode, "status": "done"}
    (rerun / "run.json").write_text(json.dumps(info), encoding="utf-8")
    pin = {
        "run": f"derived/wordcheck/run-{RERUN_JOB}",
        "sha256": sha256_file(rerun / "transcripts.parquet"),
    }
    workspace.config["leakage_rerun"] = pin
    workspace.write_config()
    return pin


def test_the_views_mode_reads_only_the_named_views_on_the_sample(
    sample_run: Finished,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    workspace = pin_the_backup(sample_run, tmp_path)
    ears = Ears(workspace)
    run = run_job(workspace, monkeypatch, ears, "--views", "rev_whole", job=VIEWS_JOB)
    out = capsys.readouterr().out
    info = info_of(run)
    assert info["status"] == "done"
    assert info["mode"] == {"name": "views", "views": ["rev_whole"]}
    segments = sampled(workspace)
    reads = [("rev", segment) for segment in segments] + [("rev", s) for s in segments[:5]]
    assert ears.reads("whisper") == ears.reads("wav2vec2") == reads  # the fakes know rev's clips
    assert set(info["inputs"]["views"]) == {"rev_whole"}
    assert info["inputs"]["wordcheck_run"] == {
        "file": FIRST_RUN,
        "sha256": sample_run.info["outputs"]["transcripts.parquet"],
    }
    assert "leakage_rerun" not in info["inputs"]
    results = info["results"]
    assert results["leakage"]["rev_whole"]["whisper"]["segments"] == len(segments)
    assert "interval" not in results["leakage"]["rev_whole"]["whisper"]
    assert results["bootstrap"]["resamples"] == 0
    assert set(info["verdicts"]) == {"rev_whole"}
    assert info["verdicts"]["rev_whole"]["view"] == "rev_whole"
    assert info["verdicts"]["rev_whole"]["verdict"] == "PASS"
    assert re.search(r"^F4 \(rev_whole\): PASS  ", out, flags=re.MULTILINE)
    assert "x0 control" not in out
    assert sorted(info["outputs"]) == ["scores.parquet", "transcripts.parquet"]


def test_the_views_mode_reads_leakage_on_all_segments_once_a_rerun_is_pinned(
    waiting_run: Finished,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    workspace = pin_the_backup(waiting_run, tmp_path)
    pin = pin_a_rerun(workspace)
    rerun = workspace.run_dir(RERUN_JOB)
    ears = Ears(workspace)
    run = run_job(workspace, monkeypatch, ears, "--views", "rev_whole", job=VIEWS_JOB)
    info = info_of(run)
    segments = every_segment(workspace)
    reads = [("rev", segment) for segment in segments] + [("rev", s) for s in segments[:5]]
    assert ears.reads("whisper") == reads
    assert info["inputs"]["leakage_rerun"] == {"file": pin["run"], "sha256": pin["sha256"]}
    results = info["results"]
    assert results["leakage"]["rev_whole"]["whisper"]["segments"] == len(segments)
    assert results["words"]["rev_whole"]["whisper"]["clips"] == len(workspace.sample)
    # A pin that does not match its file is refused.
    (rerun / "transcripts.parquet").write_bytes(b"another file")
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, Ears(workspace), "--views", "rev_whole", job="703")
    assert "the pinned leakage rerun's transcripts.parquet does not match its pin" in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("unpinned", "--views checks a backup of F4 after the word check: pin that run"),
        ("sha256", "the pinned word-check run's transcripts.parquet does not match its pin"),
        ("not done", "the pinned word-check run is not a finished run on the sample"),
        ("inputs", "the pinned word-check run was made from other pinned inputs"),
        ("called", "the pinned word-check run's x0 control called for the leakage rerun"),
        ("other rerun", "the pinned leakage rerun is not a finished rerun of the word-check run"),
    ],
)
def test_the_views_mode_refuses_a_word_check_run_that_it_cannot_follow(
    sample_run: Finished,
    waiting_run: Finished,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    change: str,
    message: str,
) -> None:
    workspace = pin_the_backup(
        waiting_run if change in ("called", "other rerun") else sample_run, tmp_path
    )
    if change == "unpinned":
        del workspace.config["wordcheck_run"]
        workspace.write_config()
    elif change == "sha256":
        workspace.config["wordcheck_run"]["sha256"] = "0" * 64
        workspace.write_config()
    elif change == "not done":
        change_the_first_run(workspace, "status", "failed")
    elif change == "inputs":
        inputs = json.loads((workspace.run_dir(JOB) / "run.json").read_text())["inputs"]
        inputs["labels"]["sha256"] = "0" * 64
        change_the_first_run(workspace, "inputs", inputs)
    elif change == "other rerun":
        pin_a_rerun(workspace, first_run="derived/wordcheck/run-999")
    ears = Ears(workspace)
    with pytest.raises(SystemExit):
        run_job(workspace, monkeypatch, ears, "--views", "rev_whole", job=VIEWS_JOB)
    assert f"FAILED: StepError: {message}" in capsys.readouterr().out
    info = info_of(workspace.run_dir(VIEWS_JOB))
    assert info["status"] == "failed"
    assert info["error"]["message"].startswith(message)
    assert ears.events == []  # refused before any model loads


# The job file, and the larger variant --------------------------------------------------------


def test_the_job_file_asks_for_one_gpu_for_twelve_hours_and_passes_its_arguments() -> None:
    text = (REPO / "scripts" / "slurm" / "check_views.sbatch").read_text(encoding="utf-8")
    for line in (
        "#SBATCH --job-name=lbw-check-views",
        "#SBATCH --partition=gpu",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=32G",
        "#SBATCH --time=12:00:00",
    ):
        assert f"{line}\n" in text, line
    assert "set -euo pipefail\nsource scripts/slurm/common.sh\n" in text
    assert "nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv\n" in (
        text
    )
    assert text.endswith('python scripts/check_views.py "$@"\n')


@pytest.mark.slow
def test_the_job_runs_with_the_full_bootstrap_of_the_config(
    workspace: WordcheckWorkspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    real = yaml.safe_load((REPO / "configs" / "pilot" / "data.yaml").read_text(encoding="utf-8"))
    workspace.config["word_check"]["bootstrap"] = real["word_check"]["bootstrap"]
    workspace.write_config()
    run = run_job(workspace, monkeypatch, Ears(workspace))
    capsys.readouterr()
    info = info_of(run)
    assert info["status"] == "done"
    assert info["results"]["bootstrap"]["resamples"] == 10_000
    assert info["verdicts"]["x0_control"]["verdict"] == "PASS"
