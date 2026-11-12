import math

import numpy as np
import pandas as pd
import pytest

from listening_beyond_words.segments import build_segments, segment_neighbours


def synthetic_words(lines: list[tuple[float, int]], session: int = 9001) -> pd.DataFrame:
    """Words at 0.25 s each, line after line; each line is (start time, word count)."""
    rows = []
    for line, (start, count) in enumerate(lines):
        for index in range(count):
            begin = start + 0.25 * index
            rows.append(
                {
                    "session": session,
                    "line": line,
                    "word_index": index,
                    "word": f"w{line}x{index}",
                    "start": begin,
                    "end": begin + 0.2,
                }
            )
    return pd.DataFrame(rows)


def test_segments_hold_exactly_n_words_and_drop_the_leftover() -> None:
    segments = build_segments(synthetic_words([(0.0, 30), (20.0, 30), (40.0, 25)]))
    assert list(segments["segment"]) == ["9001_000", "9001_001"]
    assert [len(text.split()) for text in segments["text"]] == [40, 40]


def test_segments_join_lines_with_fixed_gaps() -> None:
    segments = build_segments(synthetic_words([(0.0, 30), (20.0, 30), (40.0, 25)]))
    first = segments.iloc[0]
    assert first["parts"] == 2
    assert first["starts"] == [0.0, 20.0]
    assert first["ends"] == pytest.approx([29 * 0.25 + 0.2, 20.0 + 9 * 0.25 + 0.2])
    assert first["seconds"] == pytest.approx((29 * 0.25 + 0.2) + (9 * 0.25 + 0.2) + 0.3)


def test_segments_never_mix_sessions() -> None:
    words = pd.concat([synthetic_words([(0.0, 50)], 9001), synthetic_words([(0.0, 50)], 9002)])
    segments = build_segments(words)
    assert list(segments["session"]) == [9001, 9002]
    assert all(text.startswith("w0x0 ") for text in segments["text"])


def test_neighbours_are_the_words_just_outside_each_segment() -> None:
    neighbours = segment_neighbours(synthetic_words([(0.0, 30), (20.0, 30), (40.0, 25)]))
    assert list(neighbours["segment"]) == ["9001_000", "9001_001"]
    assert neighbours["before"].tolist() == pytest.approx([-math.inf, 20.0 + 9 * 0.25 + 0.2])
    assert neighbours["after"].tolist() == pytest.approx([20.0 + 10 * 0.25, 40.0 + 20 * 0.25])


def test_last_segment_has_no_word_after_it_when_nothing_is_left_over() -> None:
    neighbours = segment_neighbours(synthetic_words([(0.0, 40)]))
    assert neighbours["before"].tolist() == [-math.inf]
    assert neighbours["after"].tolist() == [math.inf]


def test_neighbours_match_the_segments_and_lie_outside_them() -> None:
    words = pd.concat(
        [
            synthetic_words([(0.0, 50), (30.0, 45)], 9001),
            synthetic_words([(5.0, 90)], 9002),
        ]
    )
    segments = build_segments(words)
    neighbours = segment_neighbours(words)
    assert list(neighbours["segment"]) == list(segments["segment"])
    for starts, ends, before, after in zip(
        segments["starts"], segments["ends"], neighbours["before"], neighbours["after"], strict=True
    ):
        assert before < starts[0]
        assert after > ends[-1]


def test_neighbours_do_not_depend_on_the_row_order_and_are_rounded() -> None:
    first = synthetic_words([(0.0, 50), (30.0, 45)], 9002)
    second = synthetic_words([(5.0, 30), (20.0, 60)], 9001)
    second[["start", "end"]] += 0.00037  # times that need rounding to 3 decimals
    words = pd.concat([first, second]).sample(frac=1.0, random_state=9001)
    neighbours = segment_neighbours(words)
    assert list(neighbours["segment"]) == list(build_segments(words)["segment"])
    for row in neighbours.itertuples():
        session, index = (int(part) for part in row.segment.split("_"))
        own = words[words["session"] == session].sort_values("start")
        inside = own.iloc[40 * index : 40 * index + 40]
        earlier = own[own["start"] < inside["start"].min()]
        later = own[own["start"] > inside["start"].max()]
        assert row.before == (np.round(earlier["end"].max(), 3) if len(earlier) else -math.inf)
        assert row.after == (np.round(later["start"].min(), 3) if len(later) else math.inf)
