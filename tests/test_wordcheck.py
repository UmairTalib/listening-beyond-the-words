import numpy as np
import pandas as pd
import pytest

from listening_beyond_words.wordcheck import draw_sample


def segments(counts: dict[int, int]) -> pd.DataFrame:
    """Segments of fake participants: session ID -> number of segments."""
    rows = [
        {"segment": f"{session}_{index:03d}", "session": session}
        for session, count in counts.items()
        for index in range(count)
    ]
    return pd.DataFrame(rows)


COUNTS = {9001: 3, 9002: 4, 9003: 5, 9004: 6, 9005: 40}
# Computed 2026-11-22 with numpy 2.3.5. If this fails after a numpy upgrade, a pinned sample
# can be drawn again only with the numpy version named in its run.json.
PINNED_DRAW = [
    *("9001_000", "9001_001", "9001_002"),
    *("9002_000", "9002_001", "9002_002", "9002_003"),
    *("9003_000", "9003_001", "9003_002", "9003_003", "9003_004"),
    *("9004_000", "9004_001", "9004_003", "9004_004", "9004_005"),
    *("9005_007", "9005_009", "9005_014", "9005_026", "9005_035"),
]


def test_five_per_participant_or_all_if_fewer() -> None:
    drawn = draw_sample(segments(COUNTS), np.random.default_rng(105), 5)
    per = drawn.groupby("session").size().to_dict()
    assert per == {9001: 3, 9002: 4, 9003: 5, 9004: 5, 9005: 5}
    assert not drawn["segment"].duplicated().any()
    assert drawn["segment"].tolist() == sorted(drawn["segment"])
    assert list(drawn.columns) == ["segment", "session"]
    table = segments(COUNTS)
    assert set(drawn["segment"]) <= set(table["segment"])
    assert (drawn["segment"].str.split("_").str[0].astype(int) == drawn["session"]).all()


def test_the_same_seed_gives_the_same_draw_whatever_the_row_order() -> None:
    table = segments(COUNTS)
    first = draw_sample(table, np.random.default_rng(105), 5)
    shuffled = table.sample(frac=1.0, random_state=1).reset_index(drop=True)
    again = draw_sample(shuffled, np.random.default_rng(105), 5)
    pd.testing.assert_frame_equal(first, again)
    other = draw_sample(table, np.random.default_rng(106), 5)
    assert other["segment"].tolist() != first["segment"].tolist()


def test_the_draw_leaves_its_input_unchanged() -> None:
    table = segments(COUNTS)
    copy = table.copy()
    draw_sample(table, np.random.default_rng(105), 5)
    pd.testing.assert_frame_equal(table, copy)


def test_draw_refuses_a_count_below_one_and_repeated_segments() -> None:
    with pytest.raises(ValueError, match="at least one"):
        draw_sample(segments(COUNTS), np.random.default_rng(105), 0)
    twice = pd.concat([segments({9001: 2}), segments({9001: 1})])
    with pytest.raises(ValueError, match="appears twice"):
        draw_sample(twice, np.random.default_rng(105), 5)


def test_the_draw_for_seed_105_is_pinned() -> None:
    drawn = draw_sample(segments(COUNTS), np.random.default_rng(105), 5)
    assert drawn["segment"].tolist() == PINNED_DRAW
