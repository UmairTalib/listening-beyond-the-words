import math

import numpy as np
import pandas as pd
import pytest

from listening_beyond_words.segments import (
    build_segments,
    in_segment_order,
    segment_bounds,
    segment_neighbours,
    segment_parts,
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


def frozen_build_segments(words: pd.DataFrame, n_words: int = 40, gap: float = 0.3) -> pd.DataFrame:
    """``build_segments`` as it was when the pinned segments were built (commit 3ff97f4)."""
    if "kept" in words.columns and "part" not in words.columns:
        raise ValueError("words with a kept column need their part keys (cut.cut_words)")
    ordered = words.sort_values(["session", "start", "word_index", "line"], kind="stable")
    if "kept" in ordered.columns:
        ordered = ordered[ordered["kept"].to_numpy(dtype=bool)]
    key = "part" if "part" in ordered.columns else "line"
    rows = []
    for session_key, group in ordered.groupby("session", sort=True):
        session = int(session_key)
        usable = len(group) - len(group) % n_words
        for index, first in enumerate(range(0, usable, n_words)):
            chunk = group.iloc[first : first + n_words]
            parts = chunk.groupby(key, sort=False).agg(start=("start", "min"), end=("end", "max"))
            parts = parts.sort_values("start")
            seconds = float((parts["end"] - parts["start"]).sum()) + gap * (len(parts) - 1)
            rows.append(
                {
                    "segment": f"{session}_{index:03d}",
                    "session": session,
                    "index": index,
                    "starts": parts["start"].round(3).tolist(),
                    "ends": parts["end"].round(3).tolist(),
                    "parts": len(parts),
                    "seconds": round(seconds, 3),
                    "text": " ".join(chunk["word"]),
                }
            )
    return pd.DataFrame(rows)


def cut_like(words: pd.DataFrame, dropped_share: float, seed: int) -> pd.DataFrame:
    """Drop words at random and give the kept words part keys as cut.py does.

    In time order, a new part begins at a new line or after a dropped word; the part
    keys count on across the session, so a segment's first part rarely has key 0.
    """
    rng = np.random.default_rng(seed)
    ordered = in_segment_order(words).copy()
    ordered["kept"] = rng.random(len(ordered)) >= dropped_share
    ordered["score"] = rng.random(len(ordered))
    parts = []
    for _, group in ordered.groupby("session", sort=True):
        kept = group["kept"].to_numpy()
        lines = group["line"].to_numpy()
        part = np.full(len(group), -1)
        position = np.flatnonzero(kept)
        new = np.ones(len(position), dtype=bool)
        new[1:] = (lines[position[1:]] != lines[position[:-1]]) | (np.diff(position) > 1)
        part[position] = np.cumsum(new) - 1
        parts.append(pd.Series(part, index=group.index))
    ordered["part"] = pd.concat(parts)
    return ordered


def random_lines(seed: int, session: int, n_lines: int) -> pd.DataFrame:
    """Lines of 1-60 words at random times; some lines overlap, and some start together."""
    rng = np.random.default_rng(seed)
    rows = []
    start = 0.0
    for line in range(n_lines):
        start += float(rng.choice([0.0, 0.25, 3.0, 20.0], p=[0.1, 0.1, 0.4, 0.4]))
        for index in range(int(rng.integers(1, 61))):
            begin = round(start + 0.25 * index, 3)
            rows.append(
                {
                    "session": session,
                    "line": line,
                    "word_index": index,
                    "word": f"w{line}x{index}",
                    "start": begin,
                    "end": round(begin + 0.2, 3),
                }
            )
    return pd.DataFrame(rows)


def cut_sessions(seed: int) -> pd.DataFrame:
    words = pd.concat(
        [random_lines(seed, 9001, 40), random_lines(seed + 1, 9002, 25)], ignore_index=True
    )
    return cut_like(words, 0.05, seed)


@pytest.mark.parametrize("seed", [9001, 9002, 9003])
def test_build_segments_is_unchanged_bit_for_bit(seed: int) -> None:
    words = cut_sessions(seed).sample(frac=1.0, random_state=seed)
    pd.testing.assert_frame_equal(build_segments(words), frozen_build_segments(words))
    plain = words.drop(columns=["kept", "part"])  # the line fallback
    pd.testing.assert_frame_equal(build_segments(plain), frozen_build_segments(plain))
    with pytest.raises(ValueError, match="part keys"):
        frozen_build_segments(words.drop(columns=["part"]))


def test_the_frozen_copy_meets_tied_part_starts_and_parts_cut_by_segment_edges() -> None:
    words = cut_sessions(9001)
    segments = frozen_build_segments(words)
    tied = sum(len(set(starts)) < len(starts) for starts in segments["starts"])
    assert tied > 0
    chunks = in_segment_order(words[words["kept"]]).groupby("session")["part"]
    edges = 0
    for _, parts in chunks:
        usable = len(parts) - len(parts) % 40
        keys = parts.to_numpy()[:usable]
        edges += int(np.sum(keys[39:-1:40] == keys[40::40]))
    assert edges > 0


def words_of(parts: pd.DataFrame) -> dict[str, str]:
    """Each segment's words, read from the synthetic lines by line and word index."""
    return {
        segment: " ".join(
            f"w{row.line}x{index}"
            for row in group.itertuples()
            for index in range(row.first, row.last + 1)
        )
        for segment, group in parts.groupby("segment", sort=True)
    }


@pytest.mark.parametrize("seed", [9001, 9002, 9003])
def test_segment_parts_agree_with_the_segments_without_reading_the_words(seed: int) -> None:
    words = cut_sessions(seed)
    segments = build_segments(words)
    parts = segment_parts(words.drop(columns=["word", "score"]))
    assert list(parts.columns) == [
        "segment",
        "session",
        "position",
        "cut_part",
        "line",
        "first",
        "last",
        "n_words",
        "start",
        "end",
    ]
    assert parts["segment"].unique().tolist() == segments["segment"].tolist()
    by_segment = parts.groupby("segment", sort=False)
    for row in segments.itertuples():
        own = by_segment.get_group(row.segment)
        assert own["position"].tolist() == list(range(row.parts))
        assert own["session"].tolist() == [row.session] * row.parts
        assert own["start"].tolist() == row.starts
        assert own["end"].tolist() == row.ends
        assert int(own["n_words"].sum()) == 40
        assert (own["last"] - own["first"] + 1).tolist() == own["n_words"].tolist()
    # In time order the parts give back the segment's 40 words; with tied part starts the
    # words of the tied parts may come in another order.
    for segment, text in words_of(parts).items():
        expected = segments.set_index("segment").loc[segment, "text"]
        assert sorted(text.split()) == sorted(expected.split())
        starts = segments.set_index("segment").loc[segment, "starts"]
        if len(set(starts)) == len(starts):
            assert text == expected


def test_each_segment_counts_its_parts_from_0_whatever_their_cut_keys() -> None:
    words = cut_like(synthetic_words([(0.0, 30), (20.0, 30), (40.0, 25)]), 0.0, 9001)
    words.loc[words["line"] == 1, "part"] += 6  # cut keys count across the session
    parts = segment_parts(words)
    second = parts[parts["segment"] == "9001_001"]
    assert second["cut_part"].tolist() == [7, 2]
    assert second["position"].tolist() == [0, 1]
    assert second[["line", "first", "last", "n_words"]].values.tolist() == [
        [1, 10, 29, 20],
        [2, 0, 19, 20],
    ]


def test_segment_parts_refuse_a_part_that_is_not_one_unbroken_run_of_one_line() -> None:
    words = cut_like(synthetic_words([(0.0, 30), (20.0, 30)]), 0.0, 9001)
    across = words.assign(part=0)  # one part over two lines
    with pytest.raises(ValueError, match="not one unbroken run of one line"):
        segment_parts(across)
    gap = words[words["word_index"] != 5]  # a word missing from the middle of a part
    with pytest.raises(ValueError, match="not one unbroken run of one line"):
        segment_parts(gap)
    with pytest.raises(ValueError, match="part keys"):
        segment_parts(words.drop(columns=["part"]))


def test_segment_parts_without_part_keys_take_each_line_as_a_part() -> None:
    words = synthetic_words([(0.0, 30), (20.0, 30), (40.0, 25)])
    parts = segment_parts(words.drop(columns=["word"]))
    assert parts["cut_part"].tolist() == parts["line"].tolist() == [0, 1, 1, 2]
    assert parts["n_words"].tolist() == [30, 10, 20, 20]
