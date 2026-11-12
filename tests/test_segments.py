import pandas as pd
import pytest

from listening_beyond_words.segments import build_segments


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
