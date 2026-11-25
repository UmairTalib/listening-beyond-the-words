"""The scoring code that the zero-shot job and the trained runs share (scoring.py).

A tiny random Voxtral with the pass-through in place scores synthetic encoder outputs,
stored as the cache stores them (cache.to_bytes) for fake participants from 9001; one clip
has two chunks. The embargo reads a word-check run written as check_views writes it. The
job's own use of this code, end to end, is tested in tests/test_score_zeroshot.py.
"""

import hashlib
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from listening_beyond_words import scoring, voxtral
from listening_beyond_words.cache import CACHE_COLUMNS, CacheRun, PinnedCache, to_bytes
from listening_beyond_words.runinfo import Progress, StepError, sha256_file
from listening_beyond_words.settings import Workspace
from listening_beyond_words.wordcheck import FAIL, PASS
from listening_beyond_words.zeroshot import SCORE_COLUMNS
from voxtral_fakes import ANSWER_IDS, WIDTH, FakeProcessor, cached_encoder, tiny_model

SEGMENTS = {
    "x0": ("9001_000", "9001_001", "9002_000", "9002_001"),
    "rev": ("9001_000", "9001_001", "9002_000"),
}
LONG = "9002_001"  # two chunks


# A synthetic cache ---------------------------------------------------------------------------


def stored(view: str, segment: str) -> torch.Tensor:
    """A synthetic encoder output, from a seed per view and segment."""
    chunks = 2 if segment == LONG else 1
    generator = torch.Generator().manual_seed(
        int(hashlib.sha256(f"{view}{segment}".encode()).hexdigest()[:8], 16)
    )
    values = torch.randn(chunks, voxtral.FRAMES_PER_CHUNK, WIDTH, generator=generator) * 0.5
    return values.half()


def make_cache(folder: Path, view: str) -> PinnedCache:
    """A cache run of one view as cache.pinned_cache returns it: files and their manifest."""
    rows = []
    for segment in SEGMENTS[view]:
        data = to_bytes(stored(view, segment))
        file = Path(segment[:4]) / f"{segment}.npy"
        (folder / file).parent.mkdir(parents=True, exist_ok=True)
        (folder / file).write_bytes(data)
        chunks = 2 if segment == LONG else 1
        rows.append(
            {
                "segment": segment,
                "session": int(segment[:4]),
                "file": file.as_posix(),
                "chunks": chunks,
                "samples": chunks * voxtral.CHUNK_SAMPLES - 7,
                "sha256": hashlib.sha256(data).hexdigest(),
                "audio_sha256": "a" * 64,
                "features_sha256": "f" * 64,
                "max_abs": 1.0,
            }
        )
    manifest = pd.DataFrame(rows, columns=list(CACHE_COLUMNS))
    run = CacheRun(f"cache/{view}/run-9", folder, view, {}, manifest, "m" * 64, pd.DataFrame())
    return PinnedCache(view, {}, run, manifest, "derived/cache_check/run-9", ())


@pytest.fixture
def caches(tmp_path: Path) -> dict[str, PinnedCache]:
    found = {}
    for view in SEGMENTS:
        folder = tmp_path / view
        folder.mkdir()
        found[view] = make_cache(folder, view)
    return found


@pytest.fixture(scope="module")
def model() -> Iterator[Any]:
    tiny = tiny_model(half=True)
    with cached_encoder(tiny):
        yield tiny


@pytest.fixture(scope="module")
def prompts() -> dict[int, dict[str, torch.Tensor]]:
    processor = FakeProcessor()
    return {count: voxtral.prompt_inputs(processor, count) for count in (1, 2)}


def logged_scores(monkeypatch: pytest.MonkeyPatch, state: Progress | None = None) -> list[Any]:
    """Log each cached_score call of the scoring code: the clip's chunks and the named segment."""
    calls: list[Any] = []
    real = scoring.cached_score

    def score(model: Any, prompt: Any, output: Any, answer_ids: Any) -> float:
        calls.append((int(output.shape[0]), None if state is None else state.segment))
        return real(model, prompt, output, answer_ids)

    monkeypatch.setattr(scoring, "cached_score", score)
    return calls


# The scores ------------------------------------------------------------------------------------


def test_score_views_scores_every_segment_of_each_view_from_its_file_one_clip_per_call(
    model: Any, prompts: dict, caches: dict[str, PinnedCache], monkeypatch: pytest.MonkeyPatch
) -> None:
    state = Progress()
    calls = logged_scores(monkeypatch, state)
    done: list[int] = []
    table = scoring.score_views(
        model, caches, prompts, ANSWER_IDS, WIDTH, state=state, on_clip=done.append
    )
    assert list(table.columns) == list(SCORE_COLUMNS)
    assert (table["session"].dtype, table["z"].dtype) == (np.int64, np.float64)
    expected = [(view, segment) for view, segments in SEGMENTS.items() for segment in segments]
    assert list(zip(table["view"], table["segment"], strict=True)) == expected
    assert table["session"].tolist() == [int(segment[:4]) for _, segment in expected]
    for (view, segment), z in zip(expected, table["z"], strict=True):
        prompt = prompts[2 if segment == LONG else 1]
        direct = voxtral.cached_score(model, prompt, stored(view, segment), ANSWER_IDS)
        assert float(z).hex() == direct.hex()  # the stored file, scored as it is
    assert [segment for _, segment in calls] == [segment for _, segment in expected]
    assert [chunks for chunks, _ in calls] == [2 if s == LONG else 1 for _, s in expected]
    assert done == list(range(1, len(expected) + 1))
    assert state.segment is None  # named while scoring, cleared at the end
    assert table["z"].nunique() == len(expected)


def test_score_views_scores_the_given_segments_in_their_order(
    model: Any, prompts: dict, caches: dict[str, PinnedCache]
) -> None:
    chosen = {"x0": ("9002_001", "9001_000"), "rev": ("9001_001",)}
    table = scoring.score_views(model, caches, prompts, ANSWER_IDS, WIDTH, segments=chosen)
    pairs = list(zip(table["view"], table["segment"], strict=True))
    assert pairs == [("x0", "9002_001"), ("x0", "9001_000"), ("rev", "9001_001")]
    whole = scoring.score_views(model, caches, prompts, ANSWER_IDS, WIDTH).set_index(
        ["view", "segment"]
    )
    for view, segment, z in zip(table["view"], table["segment"], table["z"], strict=True):
        assert float(z).hex() == float(whole.loc[(view, segment), "z"]).hex()


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        ({"x0": ("9001_000",)}, r"segments are given of \['x0'\], but caches of \['rev', 'x0'\]"),
        ({"x0": (), "rev": (), "x1": ()}, "segments are given of"),
        ({"x0": ("9003_000",), "rev": ()}, "1 segments to score are not in the cache of x0"),
    ],
)
def test_score_views_refuses_segments_that_do_not_fit_the_caches_before_any_score(
    model: Any,
    prompts: dict,
    caches: dict[str, PinnedCache],
    monkeypatch: pytest.MonkeyPatch,
    segments: dict[str, tuple[str, ...]],
    message: str,
) -> None:
    calls = logged_scores(monkeypatch)
    with pytest.raises(StepError, match=message):
        scoring.score_views(model, caches, prompts, ANSWER_IDS, WIDTH, segments=segments)
    assert calls == []


def test_a_changed_cache_file_stops_scoring_naming_its_segment(
    model: Any, prompts: dict, caches: dict[str, PinnedCache]
) -> None:
    cache = caches["rev"]
    path = cache.run.folder / str(cache.manifest["file"].iloc[1])
    data = path.read_bytes()
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    state = Progress()
    with pytest.raises(
        StepError, match="segment 9001_001: its cache file does not match"
    ) as caught:
        scoring.score_views(model, caches, prompts, ANSWER_IDS, WIDTH, state=state)
    assert caught.value.segment == "9001_001"
    assert state.segment == "9001_001"  # a failure names the segment it stopped at


def test_a_z_that_is_not_finite_is_kept_for_the_caller_to_judge(
    model: Any, prompts: dict, caches: dict[str, PinnedCache], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = scoring.cached_score
    calls = [0]

    def planted(*args: Any) -> float:
        calls[0] += 1
        return math.nan if calls[0] == 2 else real(*args)

    monkeypatch.setattr(scoring, "cached_score", planted)
    table = scoring.score_views(model, caches, prompts, ANSWER_IDS, WIDTH)
    assert int(table["z"].isna().sum()) == 1  # the plant took; the caller writes no score
    assert math.isnan(table["z"].iloc[1])


def test_score_views_gives_an_empty_table_without_a_cache(model: Any, prompts: dict) -> None:
    table = scoring.score_views(model, {}, prompts, ANSWER_IDS, WIDTH)
    assert list(table.columns) == list(SCORE_COLUMNS)
    assert table.empty


def test_watch_summary_holds_the_parts_flags_largest_values_and_counts() -> None:
    layers = {f"layer{index}": torch.nn.Identity() for index in range(7)}
    with voxtral.watch_layers(layers) as watch:
        for index, layer in enumerate(layers.values()):
            layer(torch.tensor([float(index + 1)], dtype=torch.float16))
        layers["layer2"](torch.tensor([64_512.0], dtype=torch.float16))
    summary = scoring.watch_summary(watch)
    assert summary["parts"] == 7
    assert summary["flagged"] == ["layer2/output"]
    assert summary["largest"] == [
        ["layer2/output", 64_512.0],
        ["layer6/output", 7.0],
        ["layer5/output", 6.0],
        ["layer4/output", 5.0],
        ["layer3/output", 4.0],
    ]  # the five largest
    assert summary["layers"] == watch.record()
    json.dumps(summary, allow_nan=False)


def test_label_free_gives_only_numbers_that_need_no_label() -> None:
    z = np.array([-2.0, -0.0, 0.0, 0.5, 1.5, 4.0])
    numbers = scoring.label_free(z)
    assert numbers == {
        "count": 6,
        "all_finite": True,
        "mean": pytest.approx(z.mean()),
        "sd": pytest.approx(z.std()),
        "p5": pytest.approx(np.percentile(z, 5)),
        "p50": pytest.approx(0.25),
        "p95": pytest.approx(np.percentile(z, 95)),
        "share_above_0": 0.5,  # zeros of either sign are not above 0
    }
    assert scoring.label_free(np.array([1.0, math.nan]))["all_finite"] is False


# The embargo -----------------------------------------------------------------------------------


RUN = "derived/wordcheck/run-700"


def write_wordcheck_run(
    root: Path, verdicts: dict[str, str | None], **changes: Any
) -> dict[str, Any]:
    """A finished word-check run of the default mode, as check_views writes it; the data config."""
    folder = root / RUN
    folder.mkdir(parents=True)
    pd.DataFrame({"segment": ["300_000"], "words": [40]}).to_parquet(folder / "transcripts.parquet")
    sha256 = sha256_file(folder / "transcripts.parquet")
    pins = {view: {"run": f"views/{view}/run-1", "sha256": view * 32} for view in ("x1", "rev")}
    info = {
        "step": "check_views",
        "mode": {"name": "sample"},
        "status": "done",
        "inputs": {
            "views": {v: {"file": p["run"], "sha256": p["sha256"]} for v, p in pins.items()}
        },
        "verdicts": {key: {"verdict": value} for key, value in verdicts.items() if value},
        "outputs": {"transcripts.parquet": sha256},
    } | changes
    (folder / "run.json").write_text(json.dumps(info), encoding="utf-8")
    return {"wordcheck_run": {"run": RUN, "sha256": sha256}, "view_runs": pins}


LIFTED = {"f4": PASS, "x1_words": PASS, "x1_voice": FAIL}


def test_the_embargo_lifts_once_the_checks_of_x1_are_finished(tmp_path: Path) -> None:
    config = write_wordcheck_run(tmp_path, LIFTED)
    reading = scoring.embargo(Workspace(tmp_path), config)
    assert reading["run"] == RUN
    assert reading["sha256"] == config["wordcheck_run"]["sha256"]
    assert reading["run_json_sha256"] == sha256_file(tmp_path / RUN / "run.json")
    assert reading["verdicts"] == LIFTED
    assert reading["views"] == {
        view: {"file": pin["run"], "sha256": pin["sha256"]}
        for view, pin in config["view_runs"].items()
    }


@pytest.mark.parametrize(
    ("verdicts", "changes", "message"),
    [
        ({**LIFTED, "x1_words": FAIL}, {}, "the words of x1 say FAIL"),
        ({**LIFTED, "f4": FAIL}, {}, "check F4 says FAIL"),
        ({**LIFTED, "x1_voice": None}, {}, "holds no verdict on the voice of x1"),
        (LIFTED, {"status": "running"}, "did not finish"),
        (LIFTED, {"mode": {"name": "views"}}, "not a run of the word check in its default mode"),
        (LIFTED, {"inputs": {"views": {}}}, "checked other clips of x1 than the pinned ones"),
    ],
)
def test_the_embargo_refuses_unless_every_check_of_x1_is_finished(
    tmp_path: Path, verdicts: dict[str, str | None], changes: dict[str, Any], message: str
) -> None:
    config = write_wordcheck_run(tmp_path, verdicts, **changes)
    with pytest.raises(StepError, match=message):
        scoring.embargo(Workspace(tmp_path), config)


def test_the_embargo_refuses_without_a_pin_or_with_a_changed_file(tmp_path: Path) -> None:
    config = write_wordcheck_run(tmp_path, LIFTED)
    with pytest.raises(StepError, match="no word-check run is pinned"):
        scoring.embargo(Workspace(tmp_path), {**config, "wordcheck_run": None})
    other = {**config, "wordcheck_run": {"run": RUN, "sha256": "0" * 64}}
    with pytest.raises(StepError, match="does not match the pin"):
        scoring.embargo(Workspace(tmp_path), other)
