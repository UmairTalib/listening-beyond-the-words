import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from jiwer import AlignmentChunk
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.utils._param_validation import InvalidParameterError

from listening_beyond_words import wordcheck
from listening_beyond_words.revoice import part_text, spoken_number
from listening_beyond_words.runinfo import StepError, item_seed
from listening_beyond_words.transcripts import FILLERS, normalise
from listening_beyond_words.wordcheck import (
    ALIASES,
    FAIL,
    HESITATION,
    LINE_CLASSES,
    PASS,
    RERUN_NEEDED,
    TEXT_COUNTS,
    WAITS,
    WORD_CLASSES,
    BootstrapSettings,
    LeakageSettings,
    Results,
    Rules,
    WerCounts,
    bootstrap_aurocs,
    bootstrap_interval,
    bootstrap_sets,
    chance_partners,
    draw_sample,
    error_classes,
    f4_verdict,
    fit_leakage,
    fold_aurocs,
    join_letters,
    line_classes,
    person_scores,
    reference_classes,
    repetition_share,
    text_counts,
    verdict_lines,
    verdicts,
    wer_counts,
    words,
    words_with_source,
    x0_control_verdict,
    x1_voice_verdict,
    x1_words_verdict,
)

REPO = Path(__file__).parents[1]


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


# Text ------------------------------------------------------------------------------

X1_READS = {
    "1990": "nineteen ninety",
    "2008": "two thousand eight",
    "2010": "twenty ten",
    "1066": "ten sixty six",
    "1500": "fifteen hundred",
    "0123": "one twenty three",
    "007": "seven",
    "00": "zero",
    "125": "one hundred twenty five",
    "1000000": "one million",
}


@pytest.mark.parametrize(("digits", "spoken"), X1_READS.items())
def test_numbers_are_read_as_x1_reads_them(digits: str, spoken: str) -> None:
    assert spoken_number(digits) == spoken
    assert words(digits) == spoken.split()


def test_every_digit_string_of_one_to_five_characters_is_read_as_x1_reads_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # words(s) == spoken_number(s).split() for all 111,110 strings, checked with one call to
    # num2words per string (it takes 32 us): words must read each string by exactly one call
    # of spoken_number on the whole string, and give back that call's words unchanged.
    calls: list[tuple[str, str]] = []

    def recording(digits: str) -> str:
        spelled = spoken_number(digits)
        calls.append((digits, spelled))
        return spelled

    monkeypatch.setattr(wordcheck, "spoken_number", recording)
    wrong = []
    for length in range(1, 6):
        for value in range(10**length):
            digits = str(value).zfill(length)
            calls.clear()
            found = words(digits)
            if len(calls) != 1 or calls[0][0] != digits or found != calls[0][1].split():
                wrong.append(digits)
    assert wrong == []
    assert len(calls) == 1


SYNTHETIC_WORDS = [
    *("I", "was", "there", "don't", "I'm", "LA", "okay", "um", "yeah", "the", "x-ray"),
    *("PTSD", "y'all", "a", "A", "P.T.S.D.", "O.K.", "uh", "h", "M", "e", "r"),
]
MARKS = [",", ":", "/", ".", "%", "$", "&", "-", "'", "\u2019", "?", ""]


def synthetic_line(rng: random.Random) -> str:
    """A synthetic transcript line without words that mix digits and letters."""

    def number() -> str:
        return "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 7)))

    def token() -> str:
        if rng.random() < 0.3:
            return number() + (rng.choice(MARKS) + number() if rng.random() < 0.3 else "")
        before = rng.choice([*MARKS[:4], ""]) if rng.random() < 0.2 else ""
        after = rng.choice(MARKS) if rng.random() < 0.3 else ""
        return before + rng.choice(SYNTHETIC_WORDS) + after

    return " ".join(token() for _ in range(rng.randint(1, 12)))


def test_the_words_of_the_stored_words_equal_the_words_of_the_text_x1_read() -> None:
    rng = random.Random(20261124)
    compared = 0
    for _ in range(2000):
        line = synthetic_line(rng)
        stored = normalise(line).split()
        if not stored:
            continue
        first = rng.randint(0, len(stored) - 1)
        last = rng.randint(first, len(stored) - 1)
        read = part_text(line, first, last).text
        assert words(" ".join(stored[first : last + 1])) == words(read)
        compared += 1
    assert compared > 1500


def test_the_aligner_still_reads_numbers_its_own_way() -> None:
    from listening_beyond_words.alignment import spoken  # imports torch

    assert spoken("1990") == "one thousand nine hundred ninety"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1990s", "nineteen ninety s"),
        ("3rd", "three rd"),
        ("mp3", "mp three"),
        ("9:30", "nine thirty"),
        ("9.30", "nine thirty"),
        ("10,000", "ten zero"),
        ("50%", "fifty"),
        ("$5", "five"),
        ("2.5", "two five"),
        ("x-ray", "x ray"),
        ("24-7", "twenty four seven"),
        ("I'm", "im"),
        ("I\u2019M", "im"),
        ("DON'T", "dont"),
        ("'05", "five"),
    ],
)
def test_mixed_words_marks_apostrophes_and_case(text: str, expected: str) -> None:
    assert words(text) == expected.split()


@pytest.mark.parametrize("hesitation", sorted(FILLERS | {"m", "mmm", "uhh", "ah", "er"}))
def test_every_hesitation_is_dropped(hesitation: str) -> None:
    assert HESITATION.fullmatch(hesitation)
    assert words(hesitation) == []
    assert words(f"we {hesitation} were there") == ["we", "were", "there"]


@pytest.mark.parametrize(
    "word", ["am", "a", "i", "oh", "huh", "eh", "ehm", "e", "em", "hum", "him", "her"]
)
def test_words_that_only_look_like_hesitations_are_kept(word: str) -> None:
    assert words(word) == [word]
    assert words(f"tell {word} more") == ["tell", word, "more"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ok", "okay"),
        ("O.K.", "okay"),
        ("o k", "okay"),
        ("o ah k", "okay"),
        ("it's OK", "its okay"),
        ("alright", "all right"),
        ("gonna", "going to"),
        ("wanna", "want to"),
        ("gotta", "got to"),
        ("okay", "okay"),
    ],
)
def test_spellings_are_merged(text: str, expected: str) -> None:
    assert words(text) == expected.split()
    assert set(ALIASES) == {"ok", "alright", "gonna", "wanna", "gotta"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("P T S D", "ptsd"),
        ("p.t.s.d.", "ptsd"),
        ("PTSD", "ptsd"),
        ("A D H D", "adhd"),  # joined before "h" could be dropped as a hesitation
        ("M D", "md"),
        ("a uh i", "ai"),
        ("e r", ""),
        ("I was diagnosed with P T S D in 1990", "i was diagnosed with ptsd in nineteen ninety"),
    ],
)
def test_letters_spoken_one_by_one_are_joined(text: str, expected: str) -> None:
    assert words(text) == expected.split()


def test_join_letters_joins_runs_of_two_or_more_one_letter_words() -> None:
    assert join_letters(["p", "t", "s", "d", "is", "a", "b"]) == ["ptsd", "is", "ab"]
    assert join_letters(["i", "was", "a", "kid"]) == ["i", "was", "a", "kid"]
    assert join_letters([]) == []


def one_pass(text: str) -> list[str]:
    """The text rule with one pass of join, then drop, instead of repeating both."""
    found = [
        word
        for piece in normalise(text).split()
        for run in re.findall("[0-9]+|[a-z]+", piece)
        for word in (spoken_number(run).split() if run.isdigit() else [run])
    ]
    kept = [word for word in join_letters(found) if not HESITATION.fullmatch(word)]
    return [alias for word in kept for alias in ALIASES.get(word, (word,))]


def test_words_are_letters_only_and_idempotent_where_one_pass_is_not() -> None:
    rng = np.random.default_rng(20261124)
    alphabet = list("aehkmoru") + [" "] * 6 + list("01")
    not_idempotent = 0
    for _ in range(5000):
        text = "".join(rng.choice(alphabet, rng.integers(1, 30)))
        once = words(text)
        assert all(re.fullmatch("[a-z]+", word) for word in once)
        assert words(" ".join(once)) == once
        first = one_pass(text)
        not_idempotent += one_pass(" ".join(first)) != first
    assert not_idempotent > 0  # so the test catches a rule that joins and drops only once


def test_empty_texts_and_long_numbers() -> None:
    assert words("") == []
    assert words("Um.") == []
    assert words("...") == []
    assert words("1" * 400) == ["one"] * 400  # num2words refuses it: digit by digit
    long = ("1234567890" * 501)[:5001]  # more digits than Python turns into a number
    read = words(long)
    assert len(read) == 5001
    assert read[:3] == ["one", "two", "three"]


def test_words_with_source_gives_the_words_and_where_each_comes_from() -> None:
    stored = ["i", "was", "p", "t", "s", "d", "um", "in", "1990", "ok", "a", "uh", "i", "dont"]
    found, sources = words_with_source(stored)
    assert found == words(" ".join(stored))
    assert found == [
        *("i", "was", "ptsd", "in", "nineteen", "ninety"),
        *("okay", "ai", "dont"),
    ]
    assert sources == [0, 1, 2, 7, 8, 8, 9, 10, 13]
    rng = random.Random(5)
    for _ in range(300):
        line = synthetic_line(rng)
        stored = normalise(line).split()
        found, sources = words_with_source(stored)
        assert found == words(line)
        assert sources == sorted(sources)
        assert all(0 <= source < len(stored) for source in sources)
    with pytest.raises(TypeError):
        words_with_source("i was there")


# Word errors -------------------------------------------------------------------------


def test_the_wer_is_over_all_words_at_once() -> None:
    references = [["a", "b", "c", "d"], ["e"]]
    hypotheses = [["a", "x", "c", "d"], ["f", "g", "h"]]
    counts = wer_counts(references, hypotheses, ["9001_000", "9002_000"])
    table = counts.per_segment
    for name in ("hits", "substitutions", "deletions", "insertions", "reference_words"):
        assert getattr(counts, name) == table[name].sum()
    assert (table["hits"] + table["substitutions"] + table["deletions"]).tolist() == [4, 1]
    assert (counts.substitutions, counts.deletions, counts.insertions) == (2, 0, 2)
    assert counts.reference_words == 5
    assert counts.wer == pytest.approx(4 / 5)
    assert table["wer"].tolist() == pytest.approx([0.25, 3.0])
    assert table["wer"].mean() == pytest.approx(1.625)  # not the WER
    assert counts.heard == pytest.approx(3 / 5)
    assert counts.totals()["wer"] == counts.wer
    assert len(counts.alignments) == 2


def test_an_empty_output_counts_every_reference_word_as_deleted() -> None:
    counts = wer_counts([["i", "was", "there"]], [[]], ["9001_000"])
    assert (counts.hits, counts.substitutions, counts.deletions, counts.insertions) == (0, 0, 3, 0)
    assert counts.wer == 1.0
    assert counts.heard == 0.0


def test_an_empty_reference_is_refused_naming_only_its_segment() -> None:
    references = [["syntheticword", "here"], []]
    hypotheses = [["syntheticword"], ["syntheticword", "again"]]
    with pytest.raises(StepError) as raised:
        wer_counts(references, hypotheses, ["9001_000", "9001_001"])
    assert str(raised.value) == "segment 9001_001: the reference has no words"
    assert raised.value.segment == "9001_001"


def test_wer_counts_refuses_texts_and_mismatched_lengths() -> None:
    with pytest.raises(TypeError):
        wer_counts(["i was there"], [["i"]], ["9001_000"])
    with pytest.raises(TypeError):
        wer_counts([["i"]], ["i was"], ["9001_000"])
    with pytest.raises(ValueError, match="outputs for"):
        wer_counts([["i"]], [], ["9001_000"])
    with pytest.raises(ValueError, match="appears twice"):
        wer_counts([["i"], ["a"]], [["i"], ["a"]], ["9001_000", "9001_000"])


# Chance level and odd outputs ------------------------------------------------------


def sample_shape() -> pd.DataFrame:
    """The sample's shape: 188 participants with five segments and one with three."""
    counts = {9001 + k: 5 for k in range(189)}
    counts[9050] = 3
    return segments(counts)


def test_chance_partners_pair_each_participant_with_the_one_94_places_on() -> None:
    sample = sample_shape()
    partners = chance_partners(sample)
    sessions = sample["session"].to_numpy()
    assert (sessions[partners] != sessions).all()
    order = sorted(set(sessions))
    for k, session in enumerate(order):
        mine = np.flatnonzero(sessions == session)
        assert set(sessions[partners[mine]]) == {order[(k + 94) % 189]}
    used = np.bincount(partners, minlength=len(sample))
    assert (int((used == 1).sum()), int((used == 2).sum()), int((used == 0).sum())) == (939, 2, 2)


def test_chance_partners_do_not_depend_on_the_row_order() -> None:
    sample = sample_shape()
    names = sample["segment"].to_numpy()
    first = dict(zip(names, names[chance_partners(sample)], strict=True))
    shuffled = sample.sample(frac=1.0, random_state=3).reset_index(drop=True)
    names = shuffled["segment"].to_numpy()
    again = dict(zip(names, names[chance_partners(shuffled)], strict=True))
    assert first == again


def test_chance_partners_need_two_participants_and_unique_segments() -> None:
    with pytest.raises(ValueError, match="two participants"):
        chance_partners(segments({9001: 5}))
    with pytest.raises(ValueError, match="appears twice"):
        chance_partners(pd.concat([segments({9001: 2, 9002: 1}), segments({9002: 1})]))


def test_the_repetition_share() -> None:
    varied = [f"w{k}" for k in range(40)]
    assert repetition_share(varied) == 0.0
    # 38 triples, 2 of them new: 36 repeated.
    assert repetition_share(["thank", "you"] * 20) == pytest.approx(36 / 38, abs=1e-15)
    # 46 triples; "you know you" and "know you know" come back twice each.
    assert repetition_share([*varied, *(["you", "know"] * 4)]) == pytest.approx(4 / 46, abs=1e-15)
    assert repetition_share(["thank", "you"]) == 0.0
    assert repetition_share([]) == 0.0


def test_text_counts_count_what_the_text_rule_meets_and_changes() -> None:
    texts = [
        "I was diagnosed with P T S D in 1990, um, ok",
        "a uh i",
        "1990s 3rd 2010 10,000 uh-huh mm-hmm alright",
        "call 12345",
    ]
    assert text_counts(texts) == {
        "words": 27,
        "words_with_a_digit": 7,  # 1990, 1990s, 3rd, 2010, 10, 000 and 12345
        "four_digit_runs": 3,  # 1990 twice and 2010; not 12345
        "mixed_words": 2,  # 1990s and 3rd
        "one_letter_words": 8,  # i, p, t, s, d, a, i and the s of 1990s
        "letter_runs_joined": 2,  # p t s d, and a i
        "hesitations_dropped": 5,  # um, uh, the uh of uh-huh, and mm and hmm
        "aliases_merged": 2,  # ok and alright
    }
    assert tuple(text_counts([])) == TEXT_COUNTS
    with pytest.raises(TypeError, match="takes texts"):
        text_counts(["fine", 3])  # type: ignore[list-item]


# Word classes ------------------------------------------------------------------------


def chunk(kind: str, ref: tuple[int, int], hyp: tuple[int, int]) -> AlignmentChunk:
    return AlignmentChunk(
        type=kind,
        ref_start_idx=ref[0],
        ref_end_idx=ref[1],
        hyp_start_idx=hyp[0],
        hyp_end_idx=hyp[1],
    )


def fake_counts(segment: str, n_ref: int, chunks: list[AlignmentChunk]) -> WerCounts:
    table = pd.DataFrame([{"segment": segment, "reference_words": n_ref}])
    return WerCounts(0, 0, 0, 0, n_ref, 0, table, (tuple(chunks),))


def class_table(segment: str, flags: dict[str, list[int]], n_ref: int) -> pd.DataFrame:
    table = pd.DataFrame(
        {name: [k in flags.get(name, []) for k in range(n_ref)] for name in WORD_CLASSES}
    )
    table.insert(0, "source", np.arange(n_ref))
    table.insert(0, "position", np.arange(n_ref))
    table.insert(0, "segment", segment)
    return table


def test_errors_count_for_the_class_of_their_word_and_insertions_for_their_neighbours() -> None:
    # Reference words 0-4; word 1 is a year, word 2 a mixed word, words 3 and 4 one letter.
    flags = {"year": [1], "mixed": [2], "one_letter": [3, 4], "capitals": [0]}
    chunks = [
        chunk("insert", (0, 0), (0, 1)),  # before word 0: next to word 0 only
        chunk("equal", (0, 1), (1, 2)),
        chunk("substitute", (1, 2), (2, 3)),
        chunk("insert", (2, 2), (3, 5)),  # two words between words 1 and 2
        chunk("delete", (2, 3), (5, 5)),
        chunk("equal", (3, 4), (5, 6)),
        chunk("insert", (4, 4), (6, 7)),  # between words 3 and 4, both one letter: counts once
        chunk("equal", (4, 5), (7, 8)),
        chunk("insert", (5, 5), (8, 9)),  # after the last word
    ]
    table = error_classes(fake_counts("9001_000", 5, chunks), class_table("9001_000", flags, 5))
    assert table.loc["all"].tolist() == [5, 1, 1, 5]
    assert table.loc["capitals"].tolist() == [1, 0, 0, 1]
    assert table.loc["year"].tolist() == [1, 1, 0, 2]
    assert table.loc["mixed"].tolist() == [1, 0, 1, 2]
    assert table.loc["one_letter"].tolist() == [2, 0, 0, 2]
    assert table.loc["number"].tolist() == [0, 0, 0, 0]
    assert table.index.tolist() == ["all", *WORD_CLASSES]
    assert list(table.columns) == ["reference_words", "substitutions", "deletions", "insertions"]
    assert all(dtype.kind == "i" for dtype in table.dtypes)  # counts only


def test_error_classes_need_classes_for_every_reference_word() -> None:
    counts = fake_counts("9001_000", 3, [chunk("equal", (0, 3), (0, 3))])
    with pytest.raises(ValueError, match="do not cover"):
        error_classes(counts, class_table("9001_000", {}, 2))
    with pytest.raises(ValueError, match="do not cover"):
        error_classes(counts, class_table("9001_001", {}, 3))


def test_line_classes_come_from_the_transcript_line_through_part_text() -> None:
    kelvin = "\u212a"  # lower-cases to "k" in normalise, but has no written form of its own
    line = f"Well I'm fine PTSD and {kelvin} too"
    table = line_classes(line, 1, 5)  # im fine ptsd and k
    assert list(table.columns) == list(LINE_CLASSES)
    assert table["apostrophe"].tolist() == [True, False, False, False, False]
    assert table["capitals"].tolist() == [False, False, True, False, False]
    assert table["read_as_stored"].tolist() == [False, False, False, False, True]
    assert not table["split_edge"].any()
    split = line_classes("the P.T.S.D. thing", 3, 5)  # s d thing: the token runs on before
    assert split["split_edge"].tolist() == [True, True, False]
    assert not split["read_as_stored"].any()
    after = line_classes("the P.T.S.D. thing", 0, 2)  # the p t: the token runs on after
    assert after["split_edge"].tolist() == [False, True, True]
    assert line_classes("the P.T.S.D. thing", 1, 4)["capitals"].sum() == 0  # one letter each


def test_each_class_of_a_reference_word_is_found() -> None:
    # Parts: "i was born in" | "1990" | "and had" | "125 mp3 players a b c"
    stored = [
        "i",
        "was",
        "born",
        "in",
        "1990",
        "and",
        "had",
        "125",
        "mp3",
        "players",
        "a",
        "b",
        "c",
    ]
    lines = pd.DataFrame({name: [False] * len(stored) for name in LINE_CLASSES})
    lines.loc[9, "capitals"] = True
    lines.loc[1, "apostrophe"] = True
    lines.loc[2, "read_as_stored"] = True
    lines.loc[3, "split_edge"] = True
    table = reference_classes("9001_000", stored, [4, 1, 2, 6], lines)
    found, sources = words_with_source(stored)
    assert table["position"].tolist() == list(range(len(found)))
    assert table["source"].tolist() == sources
    assert (table["segment"] == "9001_000").all()
    assert list(table.columns) == ["segment", "position", "source", *WORD_CLASSES]
    # i was born in nineteen ninety and had one hundred twenty five mp three players abc
    expected = {
        "year": [4, 5],
        "number": [8, 9, 10, 11],
        "mixed": [12, 13],
        "one_letter": [0, 15],
        "part_edge": [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15],  # "abc" holds the part's last letter
        "short_part": [4, 5, 6, 7],
        "capitals": [14],
        "apostrophe": [1],
        "read_as_stored": [2],
        "split_edge": [3],
    }
    for name, positions in expected.items():
        assert np.flatnonzero(table[name].to_numpy()).tolist() == positions, name
    assert all(table[name].dtype == bool for name in WORD_CLASSES)


def test_only_four_digits_make_a_year() -> None:
    stored = ["in", "10000", "1990"]
    lines = pd.DataFrame({name: [False] * len(stored) for name in LINE_CLASSES})
    table = reference_classes("9001_000", stored, [3], lines)
    # in ten thousand nineteen ninety
    assert np.flatnonzero(table["year"]).tolist() == [3, 4]
    assert np.flatnonzero(table["number"]).tolist() == [1, 2]


def test_reference_classes_from_synthetic_transcript_lines() -> None:
    kelvin = "\u212a"
    lines = [f"Well I'm fine PTSD {kelvin} and too", "the P.T.S.D. thing ok"]
    parts = [(0, 1, 5), (1, 3, 5)]  # (line, first, last): im fine ptsd k and | s d thing
    stored = [
        word
        for line, first, last in parts
        for word in normalise(lines[line]).split()[first : last + 1]
    ]
    flags = pd.concat(
        [line_classes(lines[line], first, last) for line, first, last in parts], ignore_index=True
    )
    table = reference_classes("9001_000", stored, [5, 3], flags)
    # im fine ptsd k and | sd thing: "s d" is joined into one word, split at the part's start
    assert words(" ".join(stored)) == ["im", "fine", "ptsd", "k", "and", "sd", "thing"]
    assert np.flatnonzero(table["apostrophe"]).tolist() == [0]
    assert np.flatnonzero(table["capitals"]).tolist() == [2]
    assert np.flatnonzero(table["read_as_stored"]).tolist() == [3]
    assert np.flatnonzero(table["split_edge"]).tolist() == [5]
    assert np.flatnonzero(table["one_letter"]).tolist() == [3, 5]
    assert np.flatnonzero(table["part_edge"]).tolist() == [0, 4, 5, 6]


def test_reference_classes_refuse_parts_or_flags_that_do_not_cover_the_words() -> None:
    lines = pd.DataFrame({name: [False] * 3 for name in LINE_CLASSES})
    with pytest.raises(ValueError, match="parts of"):
        reference_classes("9001_000", ["a", "b", "c"], [2], lines)
    with pytest.raises(ValueError, match="line classes for"):
        reference_classes("9001_000", ["a", "b", "c"], [3], lines.iloc[:2])
    with pytest.raises(ValueError, match="lack"):
        reference_classes("9001_000", ["a", "b", "c"], [3], lines.drop(columns="capitals"))


# Leakage -----------------------------------------------------------------------------

COMMON = [
    *("am", "was", "the", "good", "day", "not", "sure", "really", "work", "home"),
    *("sleep", "well", "think", "so", "yes", "no", "maybe", "family", "friends", "time"),
]
LEAKAGE_BLOCK = {
    "ngram_range": [1, 2],
    "min_df": 2,
    "token_pattern": r"\S+",
    "logistic_c": 1.0,
    "max_iter": 5000,
}
RULES_BLOCK = {
    "rev_min_wer": 0.80,
    "rev_max_leakage_auroc": 0.60,
    "x0_max_wer": 0.40,
    "x0_leakage_interval": 0.95,
    "x1_max_wer": 0.15,
    "x1_max_icc": 0.30,
    "probe_margin_auroc": 0.05,
}
CONFIG = {
    "word_check": {
        "leakage": LEAKAGE_BLOCK,
        "bootstrap": {"resamples": 10000, "seed": 105},
        "rules": RULES_BLOCK,
    }
}
SETTINGS = LeakageSettings.from_config(CONFIG)


def design(per_fold: int = 8, n_folds: int = 5) -> tuple[pd.Series, pd.Series]:
    """Labels and folds of synthetic participants, both classes in every fold."""
    sessions = list(range(9001, 9001 + per_fold * n_folds))
    folds = pd.Series([k % n_folds + 1 for k in range(len(sessions))], index=sessions)
    labels = pd.Series([(k // n_folds) % 2 for k in range(len(sessions))], index=sessions)
    return labels, folds


def stored_texts(
    labels: pd.Series, rng: np.random.Generator, planted: str = "", each: int = 4
) -> pd.DataFrame:
    rows = []
    for session, label in labels.items():
        for _ in range(each):
            text = " ".join(rng.choice(COMMON, 8))
            rows.append({"session": session, "text": f"{text} {planted}" if label else text})
    return pd.DataFrame(rows)


def outputs(labels: pd.Series, rng: np.random.Generator, planted: str = "") -> pd.DataFrame:
    rows = []
    for session, label in labels.items():
        for _ in range(2):
            text = " ".join(rng.choice(COMMON, 6))
            rows.append({"session": session, "text": f"{planted} {text}" if label else text})
    return pd.DataFrame(rows)


def test_a_planted_word_gives_an_auroc_of_one_in_every_fold() -> None:
    labels, folds = design()
    rng = np.random.default_rng(1)
    models = fit_leakage(stored_texts(labels, rng, "zorblefax"), labels, folds, SETTINGS)
    assert sorted(models) == [1, 2, 3, 4, 5]
    test = outputs(labels, rng, "zorblefax")
    scores = person_scores(models, folds, test, test["text"].tolist())
    assert scores.index.tolist() == sorted(labels.index)
    assert fold_aurocs(scores, labels, folds).tolist() == [1.0] * 5


def test_the_stored_words_and_the_outputs_both_pass_through_the_text_rule() -> None:
    # Stored as the segments store it, written as an ASR system may write it: the same
    # words ("ptsd nineteen ninety") only once the text rule has read both sides.
    labels, folds = design()
    rng = np.random.default_rng(1)
    models = fit_leakage(stored_texts(labels, rng, "p t s d 1990"), labels, folds, SETTINGS)
    test = outputs(labels, rng, "P.T.S.D. 1990")
    scores = person_scores(models, folds, test, test["text"].tolist())
    assert fold_aurocs(scores, labels, folds).tolist() == [1.0] * 5


@pytest.mark.parametrize("text", ["", "qwertyuiop zxcvbnm", "Um."])
def test_empty_or_unknown_texts_score_the_intercept_and_give_one_half(text: str) -> None:
    labels, folds = design()
    rng = np.random.default_rng(2)
    models = fit_leakage(stored_texts(labels, rng, "zorblefax"), labels, folds, SETTINGS)
    test = outputs(labels, rng).assign(text=text)
    scores = person_scores(models, folds, test, test["text"].tolist())
    for session, score in scores.items():
        intercept = models[int(folds[session])].named_steps["classifier"].intercept_[0]
        assert score == intercept
    assert fold_aurocs(scores, labels, folds).tolist() == [0.5] * 5


def test_participants_whose_segments_score_alike_tie_whatever_their_number_and_order() -> None:
    labels, folds = design()
    rng = np.random.default_rng(4)
    models = fit_leakage(stored_texts(labels, rng, "zorblefax"), labels, folds, SETTINGS)
    text = "zorblefax " + " ".join(COMMON[:3])
    # Between 3 and 9 segments each, all heard alike: a floating-point mean of three or
    # more equal scores is often one rounding step away from the score.
    sessions = [session for k, session in enumerate(labels.index) for _ in range(3 + k % 7)]
    test = pd.DataFrame({"session": sessions})
    scores = person_scores(models, folds, test, [text] * len(test))
    for session, score in scores.items():
        model = models[int(folds[session])]
        assert score == model.decision_function([" ".join(words(text))])[0]
    assert fold_aurocs(scores, labels, folds).tolist() == [0.5] * 5

    class Fixed:
        def decision_function(self, texts: list[str]) -> np.ndarray:
            return np.array([{"alpha": 0.1, "beta": 0.2, "gamma": 0.3}[text] for text in texts])

    order = pd.DataFrame({"session": [9001] * 3 + [9002] * 3})
    texts = ["alpha", "beta", "gamma", "beta", "gamma", "alpha"]
    assert (0.1 + 0.2) + 0.3 != (0.2 + 0.3) + 0.1  # summed in order, the means would differ
    tied = person_scores({1: Fixed()}, pd.Series({9001: 1, 9002: 1}), order, texts)
    assert tied[9001] == tied[9002]


def test_the_vocabulary_comes_from_training_texts_in_two_segments_or_more() -> None:
    labels, folds = design()
    texts = stored_texts(labels, np.random.default_rng(3))
    own = texts.index[texts["session"] == 9001]  # fold 1
    texts.loc[own[0], "text"] += " foldword onceword i am here"
    texts.loc[own[1], "text"] += " foldword a cat"
    texts.loc[own[2], "text"] += " i am a"
    models = fit_leakage(texts, labels, folds, SETTINGS)
    vocabulary = {
        fold: set(model.named_steps["tfidf"].vocabulary_) for fold, model in models.items()
    }
    assert "foldword" not in vocabulary[1]
    assert all("foldword" in vocabulary[fold] for fold in (2, 3, 4, 5))
    assert all("onceword" not in vocabulary[fold] for fold in vocabulary)  # one training text
    assert {"i", "a", "i am"} <= vocabulary[2]  # one-letter words count


def test_the_person_score_is_the_mean_log_odds() -> None:
    class Fixed:
        """A classifier with fixed log-odds per text."""

        def __init__(self, values: dict[str, float]) -> None:
            self.values = values

        def decision_function(self, texts: list[str]) -> np.ndarray:
            return np.array([self.values[text] for text in texts])

    model = Fixed({"alpha": -10.0, "beta": 4.0, "gamma": -1.0})
    test = pd.DataFrame(
        {"session": [9001, 9001, 9002, 9002], "text": ["alpha", "beta", "gamma", "gamma"]}
    )
    folds = pd.Series({9001: 1, 9002: 1})
    labels = pd.Series({9001: 1, 9002: 0})
    scores = person_scores({1: model}, folds, test, test["text"].tolist())
    assert scores.tolist() == [-3.0, -1.0]
    assert fold_aurocs(scores, labels, folds).tolist() == [0.0]
    probabilities = pd.Series({9001: expit([-10.0, 4.0]).mean(), 9002: expit(-1.0)})
    assert fold_aurocs(probabilities, labels, folds).tolist() == [1.0]  # the other way round


def test_ties_count_one_half() -> None:
    sessions = [9001, 9002, 9003, 9004]
    labels = pd.Series([0, 0, 1, 1], index=sessions)
    folds = pd.Series([1, 1, 1, 1], index=sessions)
    scores = pd.Series([0.0, 1.0, 1.0, 2.0], index=sessions)
    assert fold_aurocs(scores, labels, folds).tolist() == [0.875]
    assert roc_auc_score(labels, scores) == 0.875


def test_a_fit_that_does_not_converge_stops_the_job() -> None:
    labels, folds = design()
    settings = LeakageSettings.from_config(
        {"word_check": {"leakage": {**LEAKAGE_BLOCK, "max_iter": 1}}}
    )
    texts = stored_texts(labels, np.random.default_rng(4), "zorblefax")
    with pytest.raises(StepError, match="did not converge in 1 iterations"):
        fit_leakage(texts, labels, folds, settings)


def test_the_n_gram_range_reaches_scikit_learn_as_a_tuple() -> None:
    assert SETTINGS.ngram_range == (1, 2)
    labels, folds = design()
    models = fit_leakage(stored_texts(labels, np.random.default_rng(5)), labels, folds, SETTINGS)
    for model in models.values():
        vectoriser = model.named_steps["tfidf"]
        assert type(vectoriser.ngram_range) is tuple
        assert vectoriser.ngram_range == (1, 2)
        assert vectoriser.token_pattern == r"\S+"
        assert vectoriser.min_df == 2
        classifier = model.named_steps["classifier"]
        assert (classifier.C, classifier.solver, classifier.max_iter) == (1.0, "lbfgs", 5000)
    with pytest.raises(InvalidParameterError):  # why: scikit-learn refuses a list
        TfidfVectorizer(ngram_range=[1, 2]).fit(["a b", "a c"])  # type: ignore[arg-type]


def test_each_participant_is_scored_only_by_its_own_folds_classifier() -> None:
    labels, folds = design()
    rng = np.random.default_rng(6)
    labels = pd.Series(rng.permutation(labels.to_numpy()), index=labels.index)  # random labels

    def unique(session: int) -> str:  # letters only: words() would read digits
        return "zz" + "".join(chr(97 + int(digit)) for digit in str(session))

    texts = pd.DataFrame(
        [
            {"session": session, "text": f"{unique(session)} {' '.join(rng.choice(COMMON, 6))}"}
            for session in labels.index
            for _ in range(3)
        ]
    )
    for fold in range(1, 6):  # random labels can leave a fold with one class: refit them
        members = folds.index[folds == fold]
        if labels[members].nunique() < 2:
            labels[members[0]] = 1 - labels[members[0]]
    models = fit_leakage(texts, labels, folds, SETTINGS)
    test = pd.DataFrame({"session": labels.index, "text": [unique(s) for s in labels.index]})
    own = person_scores(models, folds, test, test["text"].tolist())
    assert fold_aurocs(own, labels, folds).tolist() == [0.5] * 5
    wrong = folds % 5 + 1  # scored by a classifier that saw the participant's own words
    leaked = person_scores(models, wrong, test, test["text"].tolist())
    assert fold_aurocs(leaked, labels, folds).mean() > 0.95


def test_the_leakage_auroc_is_the_mean_of_the_folds_not_the_pooled_auroc() -> None:
    labels, folds = design()
    offset = folds.astype(float) * 10.0  # each fold its own scale
    scores = offset + labels * 0.5
    per_fold = fold_aurocs(scores, labels, folds)
    assert per_fold.tolist() == [1.0] * 5
    pooled = roc_auc_score(labels, scores)
    assert pooled < 0.7
    assert per_fold.mean() == 1.0


def test_fold_aurocs_refuse_other_participants_and_a_fold_of_one_class() -> None:
    labels, folds = design()
    scores = labels.astype(float)
    with pytest.raises(ValueError, match="same participants"):
        fold_aurocs(scores.iloc[1:], labels, folds)
    one_class = labels.copy()
    one_class[folds.index[folds == 2]] = 0
    with pytest.raises(ValueError, match="fold 2 holds one class only"):
        fold_aurocs(scores, one_class, folds)


def test_person_scores_need_a_fold_and_a_model() -> None:
    labels, folds = design()
    models = fit_leakage(stored_texts(labels, np.random.default_rng(7)), labels, folds, SETTINGS)
    test = pd.DataFrame({"session": [9999], "text": ["good day"]})
    with pytest.raises(ValueError, match="without a fold"):
        person_scores(models, folds, test, ["good day"])
    test = pd.DataFrame({"session": [9001], "text": ["good day"]})
    with pytest.raises(ValueError, match="no leakage classifier for fold 1"):
        person_scores({2: models[2]}, folds, test, ["good day"])


# The interval of the x0 control ----------------------------------------------------


def toy_design() -> tuple[pd.Series, pd.Series]:
    """Fold 1 holds one positive of three, so a resample often leaves it without one."""
    sessions = list(range(9001, 9009))
    labels = pd.Series([1, 0, 0, 1, 1, 0, 0, 0], index=sessions)
    folds = pd.Series([1, 1, 1, 2, 2, 2, 2, 2], index=sessions)
    return labels, folds


def test_a_fold_left_with_one_class_is_drawn_again_and_counted() -> None:
    labels, folds = toy_design()
    sets, redraws = bootstrap_sets(labels, folds, np.random.default_rng(1), 10000)
    assert sets.shape == (10000, 8)
    assert redraws > 1000  # about 3 in 10 resamples lose fold 1's positive
    positive = labels.reindex(sets.ravel()).to_numpy().reshape(sets.shape) == 1
    fold = folds.reindex(sets.ravel()).to_numpy().reshape(sets.shape)
    assert (positive.sum(axis=1) == 3).all()  # each class keeps its size
    for value in (1, 2):
        assert ((positive & (fold == value)).any(axis=1)).all()
        assert ((~positive & (fold == value)).any(axis=1)).all()
    assert set(np.unique(sets[:, :3])) <= {9001, 9004, 9005}  # positives first


def test_every_resample_keeps_each_class_size() -> None:
    labels, folds = design()
    sets, _ = bootstrap_sets(labels, folds, np.random.default_rng(2), 500)
    drawn_labels = labels.reindex(sets.ravel()).to_numpy().reshape(sets.shape)
    assert (drawn_labels.sum(axis=1) == labels.sum()).all()
    assert (drawn_labels[:, : labels.sum()] == 1).all()  # positives first


def test_perfect_scores_give_one_and_constant_scores_one_half() -> None:
    labels, folds = design()
    sets, _ = bootstrap_sets(labels, folds, np.random.default_rng(3), 1000)
    assert bootstrap_interval(sets, labels.astype(float), labels, folds, 0.95) == (1.0, 1.0)
    constant = pd.Series(0.3, index=labels.index)
    assert bootstrap_interval(sets, constant, labels, folds, 0.95) == (0.5, 0.5)


def random_scores(labels: pd.Series, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = np.round(rng.normal(size=len(labels)) + 0.5 * labels.to_numpy(), 1)  # ties
    return pd.Series(values, index=labels.index)


def test_the_same_seed_gives_the_same_interval() -> None:
    labels, folds = design()
    scores = random_scores(labels, 4)
    settings = BootstrapSettings.from_config(CONFIG)
    first = bootstrap_sets(labels, folds, settings.rng(), 300)
    again = bootstrap_sets(labels, folds, settings.rng(), 300)
    assert np.array_equal(first[0], again[0])
    assert bootstrap_interval(first[0], scores, labels, folds, 0.95) == bootstrap_interval(
        again[0], scores, labels, folds, 0.95
    )


def test_the_resampled_fold_aurocs_equal_those_of_the_repeated_rows() -> None:
    labels, folds = design()
    scores = random_scores(labels, 5)
    sets, _ = bootstrap_sets(labels, folds, np.random.default_rng(5), 200)
    fast = bootstrap_aurocs(sets, scores, labels, folds)
    assert fast.shape == (200, 5)
    for row, drawn in enumerate(sets):
        for column, fold in enumerate(range(1, 6)):
            members = drawn[folds.reindex(drawn).to_numpy() == fold]  # repeats stay
            expected = roc_auc_score(labels.reindex(members), scores.reindex(members))
            assert fast[row, column] == pytest.approx(expected, abs=1e-12)
    values = fast.mean(axis=1)
    low, high = np.quantile(values, [0.025, 0.975], method="linear")
    interval = bootstrap_interval(sets, scores, labels, folds, 0.95)
    assert interval == pytest.approx((low, high), abs=1e-12)


def test_the_interval_has_its_own_stream() -> None:
    settings = BootstrapSettings.from_config(CONFIG)
    assert item_seed(105, "wordcheck", "bootstrap") == 3402805958923752013
    own = settings.rng().integers(0, 2**32, size=5)
    sample = np.random.default_rng(105).integers(0, 2**32, size=5)
    assert not np.array_equal(own, sample)
    # The stream that run.json records is the one the draws come from.
    recorded = np.random.default_rng(3402805958923752013).integers(0, 2**32, size=5)
    np.testing.assert_array_equal(own, recorded)
    assert settings == BootstrapSettings(resamples=10000, seed=105)


def test_bootstrap_inputs_are_checked() -> None:
    labels, folds = design()
    with pytest.raises(ValueError, match="at least one resample"):
        bootstrap_sets(labels, folds, np.random.default_rng(1), 0)
    one_class = labels.copy()
    one_class[folds.index[folds == 3]] = 1
    with pytest.raises(ValueError, match="both classes"):
        bootstrap_sets(one_class, folds, np.random.default_rng(1), 10)
    sets, _ = bootstrap_sets(labels, folds, np.random.default_rng(1), 10)
    with pytest.raises(ValueError, match="between 0 and 1"):
        bootstrap_interval(sets, labels.astype(float), labels, folds, 1.0)
    other = sets.copy()
    other[0, 0] = 9999
    with pytest.raises(ValueError, match="without a score"):
        bootstrap_interval(other, labels.astype(float), labels, folds, 0.95)
    with pytest.raises(ValueError, match="one class"):
        bootstrap_aurocs(np.repeat(sets[:, :1], sets.shape[1], axis=1), labels, labels, folds)


# Settings ----------------------------------------------------------------------------


def test_the_settings_refuse_unknown_and_missing_keys_and_bad_values() -> None:
    def config(block: str, **changes: object) -> dict:
        values = {**CONFIG["word_check"][block], **changes}
        return {"word_check": {block: {k: v for k, v in values.items() if v is not None}}}

    with pytest.raises(ValueError, match="unknown keys \\['extra'\\]"):
        Rules.from_config(config("rules", extra=1.0))
    with pytest.raises(ValueError, match="missing keys \\['x1_max_icc'\\]"):
        Rules.from_config(config("rules", x1_max_icc=None))
    with pytest.raises(ValueError, match="finite number"):
        Rules.from_config(config("rules", x0_max_wer=True))
    with pytest.raises(ValueError, match="level"):
        Rules.from_config(config("rules", x0_leakage_interval=1.0))
    with pytest.raises(ValueError, match="whole number"):
        LeakageSettings.from_config(config("leakage", min_df=2.0))
    with pytest.raises(ValueError, match="two numbers"):
        LeakageSettings.from_config(config("leakage", ngram_range=[1]))
    with pytest.raises(ValueError, match="unknown keys"):
        LeakageSettings.from_config(config("leakage", lowercase=False))
    with pytest.raises(ValueError, match="at least one resample"):
        BootstrapSettings.from_config(config("bootstrap", resamples=0))
    assert Rules.from_config(CONFIG) == Rules(**RULES_BLOCK)


def test_the_word_check_settings_in_data_yaml_load() -> None:
    config = yaml.safe_load((REPO / "configs" / "pilot" / "data.yaml").read_text(encoding="utf-8"))
    if "rules" not in config["word_check"]:
        pytest.skip("data.yaml has no word_check rules yet")
    assert Rules.from_config(config) == Rules(**RULES_BLOCK)
    assert LeakageSettings.from_config(config) == SETTINGS
    assert BootstrapSettings.from_config(config) == BootstrapSettings(10000, 105)


# Verdicts ----------------------------------------------------------------------------

RULES = Rules.from_config(CONFIG)
ICCS = {
    f"{gender} {measure}": 0.1
    for gender in ("female", "male")
    for measure in ("pitch", "loudness", "voiced")
}


def results(**changes: object) -> Results:
    values: dict[str, object] = {
        "wer": {
            "rev": {"whisper": 0.95, "wav2vec2": 0.97},
            "x0": {"whisper": 0.20, "wav2vec2": 0.30},
            "x1": {"whisper": 0.05, "wav2vec2": 0.10},
        },
        "leakage": {
            "rev": {"whisper": 0.52, "wav2vec2": 0.50},
            "x0": {"whisper": 0.70, "wav2vec2": 0.72},
        },
        "x0_intervals": {"whisper": (0.60, 0.80), "wav2vec2": (0.62, 0.82)},
        "iccs": ICCS,
        "probe_x0": 0.70,
        "probe_x1": 0.58,
        "probe_control": 0.559861,
        "after_rerun": False,
    }
    values.update(changes)
    return Results(**values)  # type: ignore[arg-type]


def both(whisper: float, wav2vec2: float) -> dict[str, float]:
    return {"whisper": whisper, "wav2vec2": wav2vec2}


def test_everything_passes_on_good_numbers() -> None:
    judged = verdicts(results(), RULES)
    assert [judged[key]["verdict"] for key in ("f4", "x0_control", "x1_words", "x1_voice")] == [
        PASS
    ] * 4
    assert judged["x1_voice"]["probe"]["counts"]
    assert judged["x1_voice"]["probe"]["bar"] == pytest.approx(0.609861)


def test_each_bound_includes_its_value() -> None:
    at = verdicts(
        results(
            wer={"rev": both(0.80, 0.80), "x0": both(0.40, 0.40), "x1": both(0.15, 0.15)},
            leakage={"rev": both(0.60, 0.60), "x0": both(0.7, 0.7)},
            iccs=dict.fromkeys(ICCS, 0.30),
        ),
        RULES,
    )
    assert [at[key]["verdict"] for key in ("f4", "x0_control", "x1_words", "x1_voice")] == [
        PASS
    ] * 4
    up, down = math.inf, -math.inf
    assert (
        f4_verdict(both(np.nextafter(0.80, down), 0.9), both(0.5, 0.5), RULES, rerun_pending=False)[
            "verdict"
        ]
        == FAIL
    )
    assert (
        f4_verdict(both(0.9, 0.9), both(0.5, np.nextafter(0.60, up)), RULES, rerun_pending=False)[
            "verdict"
        ]
        == FAIL
    )
    control = x0_control_verdict(
        both(np.nextafter(0.40, up), 0.1),
        both(0.7, 0.7),
        {"whisper": (0.6, 0.8), "wav2vec2": (0.6, 0.8)},
        RULES,
        after_rerun=False,
    )
    assert control["verdict"] == FAIL
    assert x1_words_verdict(both(0.1, np.nextafter(0.15, up)), RULES)["verdict"] == FAIL
    worse = {**ICCS, "male voiced": float(np.nextafter(0.30, up))}
    voice = x1_voice_verdict(worse, 0.7, 0.5, 0.56, RULES)
    assert (voice["verdict"], voice["failed"]) == (FAIL, ["ICC male voiced"])
    negative = x1_voice_verdict(dict.fromkeys(ICCS, -0.4), 0.7, 0.5, 0.56, RULES)
    assert negative["verdict"] == PASS  # a negative ICC passes


def test_a_margin_of_exactly_005_neither_counts_nor_fails() -> None:
    assert 0.5 + 0.05 == 0.55  # so the bar lies exactly on the probe below
    at_bar = x1_voice_verdict(ICCS, 0.55, 0.9, 0.5, RULES)
    assert not at_bar["probe"]["counts"]
    assert at_bar["verdict"] == PASS
    counts = x1_voice_verdict(ICCS, 0.56, 0.55, 0.5, RULES)
    assert counts["probe"]["counts"]
    assert counts["verdict"] == PASS
    fails = x1_voice_verdict(ICCS, 0.56, 0.5501, 0.5, RULES)
    assert (fails["verdict"], fails["failed"]) == (FAIL, ["probe"])


def test_a_probe_that_does_not_count_leaves_the_iccs_to_decide_and_says_so() -> None:
    judged = x1_voice_verdict(ICCS, 0.58, 0.99, 0.559861, RULES)  # x0 below the bar
    assert judged["verdict"] == PASS  # x1's high probe does not count
    assert judged["decided_by"] == "ICCs alone: the probe does not count"
    line = verdict_lines({"x1_voice": judged})[0]
    assert line.endswith("(does not count: the ICCs decide)")
    worse = x1_voice_verdict({**ICCS, "female pitch": 0.31}, 0.58, 0.5, 0.559861, RULES)
    assert worse["verdict"] == FAIL


def test_an_interval_that_includes_one_half_calls_the_rerun() -> None:
    def control(whisper: tuple[float, float], *, after_rerun: bool = False) -> str:
        intervals = {"whisper": whisper, "wav2vec2": (0.6, 0.8)}
        return x0_control_verdict(
            both(0.2, 0.2), both(0.6, 0.7), intervals, RULES, after_rerun=after_rerun
        )["verdict"]

    assert control((0.5, 0.7)) == RERUN_NEEDED  # a lower end of exactly 0.5
    assert control((0.3, 0.5)) == RERUN_NEEDED  # an upper end of exactly 0.5
    assert control((np.nextafter(0.5, 1.0), 0.7)) == PASS
    assert control((0.30, 0.45)) == FAIL  # wholly below 0.5: no rerun
    assert control((0.5, 0.7), after_rerun=True) == FAIL  # no second rerun
    wer_only = x0_control_verdict(
        both(0.41, 0.2),
        both(0.7, 0.7),
        {"whisper": (0.6, 0.8), "wav2vec2": (0.6, 0.8)},
        RULES,
        after_rerun=False,
    )
    assert (wer_only["verdict"], wer_only["failed"]) == (FAIL, ["whisper WER"])  # never a rerun


def test_f4_waits_for_the_rerun_but_fails_at_once_on_a_wer_bound() -> None:
    needs_rerun = {"whisper": (0.45, 0.65), "wav2vec2": (0.6, 0.8)}
    waiting = verdicts(
        results(x0_intervals=needs_rerun, leakage={"rev": both(0.7, 0.5), "x0": both(0.55, 0.7)}),
        RULES,
    )
    assert waiting["x0_control"]["verdict"] == RERUN_NEEDED
    assert waiting["f4"]["verdict"] == WAITS
    assert waiting["f4"]["leakage_provisional"]
    assert waiting["f4"]["failed"] == []
    assert verdict_lines(waiting)[0].endswith("(leakage provisional while waiting)")
    low_wer = results(
        x0_intervals=needs_rerun,
        wer={"rev": both(0.95, 0.79), "x0": both(0.2, 0.2), "x1": both(0.1, 0.1)},
    )
    failed = verdicts(low_wer, RULES)["f4"]
    assert (failed["verdict"], failed["failed"]) == (FAIL, ["wav2vec2 WER"])
    after = verdicts(
        results(
            x0_intervals=needs_rerun,
            leakage={"rev": both(0.7, 0.5), "x0": both(0.55, 0.7)},
            after_rerun=True,
        ),
        RULES,
    )
    assert after["x0_control"]["verdict"] == FAIL
    assert (after["f4"]["verdict"], after["f4"]["failed"]) == (FAIL, ["whisper leakage"])
    assert not after["f4"]["leakage_provisional"]


@pytest.mark.parametrize(
    "changes",
    [
        {"wer": {"rev": both(math.nan, 0.9), "x0": both(0.2, 0.2), "x1": both(0.1, 0.1)}},
        {"leakage": {"rev": both(0.5, 0.5), "x0": both(0.7, math.nan)}},
        {"x0_intervals": {"whisper": (math.nan, 0.8), "wav2vec2": (0.6, 0.8)}},
        {"iccs": {**ICCS, "male pitch": math.nan}},
        {"probe_x0": math.nan},
        {"probe_control": math.inf},
    ],
)
def test_a_gate_value_that_is_not_a_number_stops_the_job(changes: dict[str, object]) -> None:
    with pytest.raises(StepError, match="is not a finite number"):
        verdicts(results(**changes), RULES)


def test_verdicts_need_both_recognisers_and_six_iccs() -> None:
    with pytest.raises(ValueError, match="one value for each"):
        x1_words_verdict({"whisper": 0.1}, RULES)
    with pytest.raises(ValueError, match="6 gate ICCs"):
        x1_voice_verdict(dict(list(ICCS.items())[:5]), 0.7, 0.5, 0.56, RULES)


def test_the_verdict_lines() -> None:
    lines = verdict_lines(verdicts(results(), RULES))
    assert lines[0] == (
        "F4 (rev): PASS  whisper WER 0.9500, leakage 0.5200; wav2vec2 WER 0.9700, leakage 0.5000"
    )
    assert lines[1] == (
        "x0 control: PASS  whisper WER 0.2000, leakage 0.7000 [0.6000, 0.8000];"
        " wav2vec2 WER 0.3000, leakage 0.7200 [0.6200, 0.8200]"
    )
    assert lines[2] == "x1 words: PASS  whisper 0.0500; wav2vec2 0.1000"
    assert lines[3].startswith("x1 voice: PASS  ICC female pitch 0.1000, female loudness 0.1000,")
    assert lines[3].endswith("probe x0 0.7000, x1 0.5800, control 0.5599, bar 0.6099 (counts)")
    backup = f4_verdict(
        both(0.9, 0.9), both(0.5, 0.5), RULES, rerun_pending=False, view="rev_whole"
    )
    assert verdict_lines({"f4": backup})[0].startswith("F4 (rev_whole): PASS")
