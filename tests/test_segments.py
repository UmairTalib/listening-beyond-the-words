import math

import numpy as np
import pandas as pd
import pytest

from listening_beyond_words.segments import (
    build_segments,
    in_segment_order,
    segment_bounds,
    segment_neighbours,
    segment_walls,
)


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


def with_drops(words: pd.DataFrame, dropped: list[int]) -> pd.DataFrame:
    """Mark some words as dropped; a new part begins after every dropped word (one line only)."""
    words = words.copy()
    words["kept"] = ~words.index.isin(dropped)
    words["part"] = np.where(words["kept"], (~words["kept"]).cumsum(), -1)
    return words


def test_segments_use_only_kept_words_and_split_parts_by_the_part_key() -> None:
    words = with_drops(synthetic_words([(0.0, 50)]), [5, 6])
    segments = build_segments(words)
    assert len(segments) == 1
    first = segments.iloc[0]
    assert first["parts"] == 2
    assert first["starts"] == [0.0, 7 * 0.25]
    assert first["ends"] == pytest.approx([4 * 0.25 + 0.2, 41 * 0.25 + 0.2])
    assert "w0x5" not in first["text"].split()
    assert len(first["text"].split()) == 40


def test_kept_words_need_their_part_keys() -> None:
    words = synthetic_words([(0.0, 50)]).assign(kept=True)
    with pytest.raises(ValueError, match="part keys"):
        build_segments(words)


def test_neighbours_see_dropped_words_inside_and_just_outside_a_segment() -> None:
    words = with_drops(synthetic_words([(0.0, 100)]), [3, 40, 41, 83])
    neighbours = segment_neighbours(words)
    assert list(neighbours["segment"]) == list(build_segments(words)["segment"])
    first, second = neighbours.iloc[0], neighbours.iloc[1]
    # The first segment runs from word 0 to word 42 and skips the dropped words 3, 40 and 41.
    assert first["before"] == -math.inf
    assert first["after"] == pytest.approx(43 * 0.25)
    assert first["other_starts"] == pytest.approx([3 * 0.25, 40 * 0.25, 41 * 0.25])
    assert first["other_ends"] == pytest.approx([3 * 0.25 + 0.2, 40 * 0.25 + 0.2, 41 * 0.25 + 0.2])
    # The second runs from word 43 to word 82; the word just after it (83) was dropped.
    assert second["before"] == pytest.approx(42 * 0.25 + 0.2)
    assert second["after"] == pytest.approx(83 * 0.25)
    assert second["other_starts"] == []


def test_walls_are_the_zones_next_to_each_part() -> None:
    segments = pd.DataFrame(
        {"segment": ["9001_000"], "session": [9001], "starts": [[1.0, 5.0]], "ends": [[2.0, 6.0]]}
    )
    zones = pd.DataFrame(
        {
            "session": [9001, 9001, 9001, 9001, 9001, 9002],
            "start": [0.0, 0.5, 3.0, 7.0, 9.0, 2.5],
            "end": [0.2, 0.8, 4.0, 8.0, 9.5, 3.0],
        }
    )
    walls = segment_walls(segments, zones).iloc[0]
    assert walls["wall_starts"] == [0.5, 3.0, 7.0]
    assert walls["wall_ends"] == [0.8, 4.0, 8.0]
    no_zones = segment_walls(segments, zones[zones["session"] == 9002]).iloc[0]
    assert no_zones["wall_starts"] == []
    overlapping = pd.concat(
        [zones, pd.DataFrame({"session": [9001], "start": [1.5], "end": [1.6]})]
    )
    with pytest.raises(ValueError, match="overlaps a part"):
        segment_walls(segments, overlapping)


def test_a_dropped_first_word_is_the_first_segments_neighbour() -> None:
    words = with_drops(synthetic_words([(0.0, 50)]), [0])
    first = segment_neighbours(words).iloc[0]
    assert first["before"] == pytest.approx(0.2)
    assert first["after"] == pytest.approx(41 * 0.25)
    assert first["other_starts"] == []


def test_the_binding_wall_is_the_one_that_ends_last_or_starts_first() -> None:
    segments = pd.DataFrame(
        {"segment": ["9001_000"], "session": [9001], "starts": [[11.0]], "ends": [[12.0]]}
    )
    zones = pd.DataFrame(
        {"session": [9001] * 4, "start": [0.0, 2.0, 13.0, 12.5], "end": [10.9, 4.0, 13.5, 20.0]}
    )
    walls = segment_walls(segments, zones).iloc[0]
    assert walls["wall_starts"] == [0.0, 12.5]
    assert walls["wall_ends"] == [10.9, 20.0]


def test_words_that_start_together_are_ordered_by_position_then_line() -> None:
    rows = [(7, 0, 1.0), (3, 2, 1.0), (5, 0, 1.0), (1, 0, 0.5)]
    words = pd.DataFrame(
        [
            {"session": 9001, "line": line, "word_index": index, "start": start, "end": start}
            for line, index, start in rows
        ]
    )
    expected = [1, 5, 7, 3]
    assert in_segment_order(words)["line"].tolist() == expected
    assert in_segment_order(words.iloc[::-1])["line"].tolist() == expected


def test_bounds_follow_the_participants_turn_and_its_overhanging_words() -> None:
    segments = pd.DataFrame(
        {
            "segment": ["451_000", "300_000"],
            "session": [451, 300],
            "starts": [[1.0, 7.6, 9.2], [1.0]],
            "ends": [[3.0, 8.05, 11.0], [2.0]],
        }
    )
    turns = {451: np.array([[0.0, 8.0], [9.0, 12.0]])}
    bounds = segment_bounds(segments, turns).set_index("segment")
    assert bounds.loc["451_000", "bound_starts"] == [0.0, 0.0, 9.0]
    assert bounds.loc["451_000", "bound_ends"] == [8.0, 8.05, 12.0]
    assert bounds.loc["300_000", "bound_starts"] == [-math.inf]
    assert bounds.loc["300_000", "bound_ends"] == [math.inf]
    outside = segments.iloc[:1].assign(starts=[[8.2]], ends=[[8.6]])
    with pytest.raises(ValueError, match="no participant turn"):
        segment_bounds(outside, turns)
