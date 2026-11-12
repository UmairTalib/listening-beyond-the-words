import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from listening_beyond_words import interviewer as iv
from listening_beyond_words.models import load_pins, local_snapshot

PINS = Path(__file__).resolve().parents[1] / "configs" / "pilot" / "models.yaml"


def test_merge_joins_overlapping_spans_and_drops_empty_ones() -> None:
    spans = [(5.0, 6.0), (0.0, 2.0), (1.5, 3.0), (4.0, 4.0), (6.0, 7.0)]
    assert iv.merge(spans) == [(0.0, 3.0), (5.0, 7.0)]


def test_gaps_are_the_parts_no_line_covers() -> None:
    lines = [(2.0, 4.0), (3.0, 5.0), (5.2, 8.0)]
    assert iv.gaps(lines, duration=10.0, min_length=0.5) == [(0.0, 2.0), (8.0, 10.0)]
    assert iv.gaps(lines, duration=10.0, min_length=0.1) == [(0.0, 2.0), (5.0, 5.2), (8.0, 10.0)]


def test_intersect_keeps_only_the_parts_inside_regions() -> None:
    assert iv.intersect([(0.0, 10.0)], [(1.0, 2.0), (4.0, 12.0)]) == [(1.0, 2.0), (4.0, 10.0)]
    assert iv.intersect([(0.0, 1.0)], [(2.0, 3.0)]) == []


def test_windows_cover_a_span_with_fixed_length_and_lose_no_tail() -> None:
    assert iv.windows((0.0, 3.2), length=1.5, hop=0.75) == [
        (0.0, 1.5),
        (0.75, 2.25),
        (1.5, 3.0),
        (pytest.approx(1.7), 3.2),
    ]
    assert iv.windows((0.0, 1.0), length=1.5, hop=0.75, min_length=0.75) == [(0.0, 1.0)]
    assert iv.windows((0.0, 0.5), length=1.5, hop=0.75, min_length=0.75) == []


NOISE = 0.6 / 8  # noise of length about 0.6 around unit-length voice directions in 64-d


def voices(seed: int) -> tuple[np.random.Generator, np.ndarray, np.ndarray]:
    """Two random voice directions, and the generator that samples around them."""
    rng = np.random.default_rng(seed)
    interviewer, participant = iv.unit(rng.standard_normal((2, 64)))
    return rng, interviewer, participant


def sample(rng: np.random.Generator, voice: np.ndarray, n: int) -> np.ndarray:
    """Synthetic embeddings: noisy unit-length vectors around a voice direction."""
    return iv.unit(voice + NOISE * rng.standard_normal((n, voice.size)))


def test_global_profile_ignores_a_minority_of_other_voices() -> None:
    rng, interviewer, participant = voices(9001)
    gap_windows = np.concatenate([sample(rng, interviewer, 300), sample(rng, participant, 100)])
    assert iv.global_profile(gap_windows) @ interviewer > 0.95


def test_scores_separate_the_two_voices_inside_lines() -> None:
    rng, interviewer, participant = voices(9002)
    gap_windows = np.concatenate([sample(rng, interviewer, 200), sample(rng, participant, 50)])
    line_windows = np.concatenate([sample(rng, interviewer, 20), sample(rng, participant, 400)])
    profiles = iv.session_profiles(line_windows, gap_windows, iv.global_profile(gap_windows))
    scores = iv.interviewer_scores(line_windows, *profiles)
    assert np.mean(scores[:20] > 0) > 0.95
    assert np.mean(scores[20:] <= 0) > 0.95


def test_session_profiles_fall_back_to_the_global_profile_without_enough_gaps() -> None:
    rng, interviewer, participant = voices(9003)
    line_windows = sample(rng, participant, 100)
    gap_windows = sample(rng, interviewer, 5)
    own, _ = iv.session_profiles(line_windows, gap_windows, interviewer, min_windows=10)
    assert np.allclose(own, interviewer)


def test_line_decisions_drop_interviewer_lines_and_phrase_lines() -> None:
    lines = pd.DataFrame(
        {
            "session": [9001] * 4,
            "line": [0, 1, 2, 3],
            "start": [0.0, 3.5, 6.0, 7.0],
            "end": [3.0, 5.0, 6.5, 7.6],
            "phrase": [False, False, True, False],
        }
    )
    windows = pd.DataFrame(
        {
            "session": [9001] * 7,
            "kind": ["line", "line", "line", "line", "line", "gap", "gap"],
            "line": [0, 0, 0, 1, 1, -1, -1],
            "score": [0.3, 0.1, -0.2, -0.4, -0.1, 0.5, 0.6],
        }
    )
    decided = iv.line_decisions(lines, windows).set_index("line")
    assert decided["drop"].to_dict() == {0: True, 1: False, 2: True, 3: False}
    assert decided.loc[0, "share_above_0"] == pytest.approx(2 / 3)
    assert np.isnan(decided.loc[2, "share_above_0"])


def test_line_decisions_keep_a_line_below_the_drop_share() -> None:
    lines = pd.DataFrame(
        {"session": [9001], "line": [0], "start": [0.0], "end": [3.0], "phrase": [False]}
    )
    windows = pd.DataFrame(
        {"session": [9001] * 3, "kind": ["line"] * 3, "line": [0] * 3, "score": [0.2, -0.1, -0.3]}
    )
    assert not iv.line_decisions(lines, windows)["drop"].iloc[0]


def test_line_decisions_drop_a_long_line_without_detected_speech() -> None:
    lines = pd.DataFrame(
        {
            "session": [9001] * 2,
            "line": [0, 1],
            "start": [0.0, 5.0],
            "end": [2.0, 5.6],
            "phrase": [False, False],
        }
    )
    windows = pd.DataFrame(
        {
            "session": pd.Series([], dtype="int64"),
            "kind": pd.Series([], dtype="object"),
            "line": pd.Series([], dtype="int64"),
            "score": pd.Series([], dtype="float64"),
        }
    )
    decided = iv.line_decisions(lines, windows)
    assert decided["unclear"].tolist() == [True, False]
    assert decided["drop"].tolist() == [True, False]
    assert not iv.line_decisions(lines, windows, unclear_min_seconds=None)["drop"].any()


@pytest.fixture(scope="module")
def embedder() -> Any:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        pytest.skip("HF_HOME is not set; run the GPU tests in scripts/slurm/smoke_test.sbatch")
    return iv.VoiceEmbedder(local_snapshot(load_pins(PINS)["wavlm_sv"], Path(hf_home)))


@pytest.mark.gpu
def test_voice_embedder_gives_one_unit_vector_per_clip(embedder: Any) -> None:
    rng = np.random.default_rng(9004)
    clips = [0.1 * rng.standard_normal(n).astype(np.float32) for n in (24_000, 12_000, 20_000)]
    embeddings = embedder.embed(clips, batch_size=2)
    assert embeddings.shape == (3, 512)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
