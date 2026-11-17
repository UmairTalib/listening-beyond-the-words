import io
import zipfile

import pytest

from listening_beyond_words.daicwoz import find_member, member_name, turn_counts

HEADER = "start_time\tstop_time\tspeaker\tvalue"
LINES = ["1.0\t2.0\tEllie\thello", "2.5\t3.1\tParticipant\thi", "3.5\t4.0\tEllie\tbye"]


def transcript(lines: list[str], newline: str = "\n", bom: bool = False) -> bytes:
    text = newline.join([HEADER, *lines]) + newline
    return ("﻿" if bom else "").encode() + text.encode()


def archive_with(*names: str) -> zipfile.ZipFile:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, b"synthetic")
    return zipfile.ZipFile(io.BytesIO(buffer.getvalue()))


def test_counts_lines_per_speaker() -> None:
    counts = turn_counts(transcript(LINES))
    assert counts == {
        "parsed": 1,
        "ellie_lines": 2,
        "participant_lines": 1,
        "other_lines": 0,
        "bad_rows": 0,
    }


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_any_line_ending_gives_the_same_counts(newline: str) -> None:
    assert turn_counts(transcript(LINES, newline)) == turn_counts(transcript(LINES))


def test_a_byte_order_mark_is_ignored() -> None:
    assert turn_counts(transcript(LINES, bom=True)) == turn_counts(transcript(LINES))


def test_a_quote_cannot_swallow_the_following_lines() -> None:
    quoted = ['1.0\t2.0\tEllie\t"hello', *LINES[1:]]
    assert turn_counts(transcript(quoted))["participant_lines"] == 1
    assert turn_counts(transcript(quoted))["ellie_lines"] == 2


def test_bad_times_short_rows_and_other_speakers_are_counted() -> None:
    lines = [
        "2.0\t1.0\tEllie\tbackwards",
        "x\t1.0\tEllie\tnot a number",
        "1.0\t2.0",
        "1\t2\tAssistant\tx",
    ]
    counts = turn_counts(transcript(lines))
    assert counts["bad_rows"] == 3
    assert counts["other_lines"] == 1


def test_an_unknown_header_or_an_empty_file_is_not_parsed() -> None:
    assert turn_counts(b"a\tb\tc\n1\t2\t3\n")["parsed"] == 0
    assert turn_counts(b"")["parsed"] == 0


def test_member_name_drops_folders_with_either_separator() -> None:
    assert member_name("9001_P/9001_TRANSCRIPT.csv") == "9001_TRANSCRIPT.csv"
    assert member_name("9001_P\\9001_TRANSCRIPT.csv") == "9001_TRANSCRIPT.csv"
    assert member_name("9001_TRANSCRIPT.csv") == "9001_TRANSCRIPT.csv"


def test_find_member_ignores_folders_and_case() -> None:
    archive = archive_with("9001_P/9001_TRANSCRIPT.csv", "9001_AUDIO.wav")
    found = find_member(archive, "9001_transcript.CSV")
    assert found is not None
    assert found.filename == "9001_P/9001_TRANSCRIPT.csv"
    assert find_member(archive, "9001_FORMANT.csv") is None


def test_find_member_refuses_two_members_of_one_name() -> None:
    archive = archive_with("a/9001_TRANSCRIPT.csv", "b/9001_TRANSCRIPT.csv")
    with pytest.raises(ValueError, match="2 members"):
        find_member(archive, "9001_TRANSCRIPT.csv")
