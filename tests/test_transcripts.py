from pathlib import Path

import pytest

from listening_beyond_words.transcripts import (
    TYPOGRAPHIC_APOSTROPHE,
    normalise,
    read_transcript,
    summarise,
)

HEADER = "Start_Time,End_Time,Text,Confidence\n"


def write_transcript(folder: Path, body: str, header: str = HEADER) -> Path:
    path = folder / "9001_Transcript.csv"
    path.write_text(header + body, encoding="utf-8")
    return path


def test_normalise_keeps_lower_case_words_and_drops_apostrophes() -> None:
    text = "Hi, I" + TYPOGRAPHIC_APOSTROPHE + "m  Ellie! It's <laughter>"
    assert normalise(text) == "hi im ellie its laughter"


def test_read_transcript_renames_columns_and_parses_numbers(tmp_path: Path) -> None:
    frame = read_transcript(write_transcript(tmp_path, '3,4,,\n0.5,2.0,"Hello, there",0.9\n'))
    assert list(frame.columns) == ["start", "end", "text", "confidence"]
    assert frame["start"].tolist() == [0.5, 3.0]
    assert frame["text"].tolist() == ["Hello, there", ""]
    assert frame["confidence"].isna().tolist() == [False, True]


def test_read_transcript_rejects_other_columns(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unexpected columns"):
        read_transcript(write_transcript(tmp_path, "1,x\n", header="start,text\n"))


def test_summarise_counts_lines_words_timing_confidence_fillers_tags_and_phrases(
    tmp_path: Path,
) -> None:
    body = (
        '0.5,2.0,"Hi, thanks for coming in today.",0.9\n'
        "3.0,5.5,um I'm doing okay,0.8\n"
        '5.0,6.0,"<laughter> where are you from originally",0.4\n'
        "8.0,9.0,uh yeah,\n"
    )
    summary = summarise(read_transcript(write_transcript(tmp_path, body)))
    assert summary.lines == 4
    assert summary.words == 18
    assert summary.speech_seconds == pytest.approx(6.0)
    assert summary.first_start == 0.5
    assert summary.last_end == 9.0
    assert summary.overlaps == 1
    assert summary.gaps == pytest.approx((1.0, 0.0, 2.0))
    assert summary.confidences == pytest.approx((0.9, 0.8, 0.4))
    assert summary.missing_confidence == 1
    assert summary.fillers == 2
    assert summary.tags == 1
    assert summary.interviewer_phrases == {
        "thanks for coming in today",
        "where are you from originally",
    }


def test_summarise_finds_no_interviewer_phrase_in_a_participant_answer(tmp_path: Path) -> None:
    body = "1.0,3.0,I'm from Ohio originally and I came here for work,0.95\n"
    assert summarise(read_transcript(write_transcript(tmp_path, body))).interviewer_phrases == set()


def test_summarise_finds_ellies_follow_ups_and_comments(tmp_path: Path) -> None:
    body = '1.0,3.0,"That sounds really hard. Can you give me an example of that?",0.9\n'
    phrases = summarise(read_transcript(write_transcript(tmp_path, body))).interviewer_phrases
    assert phrases == {"that sounds really hard", "give me an example of that"}


def test_a_participant_talking_about_a_diagnosis_is_not_an_interviewer_phrase(
    tmp_path: Path,
) -> None:
    body = "1.0,3.0,yes I was diagnosed with PTSD after I served in the military,0.9\n"
    assert summarise(read_transcript(write_transcript(tmp_path, body))).interviewer_phrases == set()
