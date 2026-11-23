"""The re-voiced clip x1 (revoice.py): text, voices, tasks, settings, audio and the Kokoro wrapper."""

import dataclasses
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soxr
import yaml

import listening_beyond_words.revoice as revoice
from listening_beyond_words.cut import CutRule, all_turns, cut_words
from listening_beyond_words.revoice import (
    TASKS,
    FrontEnd,
    PartError,
    Render,
    X1Settings,
    assign_tasks,
    assign_voices,
    build_revoiced,
    check_eval,
    check_render,
    draw_test_segments,
    load_inputs,
    part_audio,
    part_text,
    render_segment,
    segment_words,
    spoken_number,
    trim_quiet,
    word_classes,
    word_has_sound,
    written_form,
)
from listening_beyond_words.runinfo import item_seed
from listening_beyond_words.segments import build_segments, segment_parts
from listening_beyond_words.transcripts import normalise
from listening_beyond_words.views import (
    check_segment_views,
    fade_part,
    join_parts,
    limit_peak,
    set_rms,
    to_pcm16,
)
from x1_fakes import (
    KOKORO_VOCABULARY,
    SECRET,
    FakePipeline,
    fake_kokoro,
    fake_token,
)

CONFIG = Path(__file__).parents[1] / "configs" / "pilot" / "data.yaml"
RATE = 16_000
KOKORO_RATE = 24_000
TYPOGRAPHIC = chr(0x2019)
DOTTED_I = chr(0x0130)
KELVIN = chr(0x212A)


def settings(**changes: object) -> X1Settings:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return dataclasses.replace(X1Settings.from_config(config), **changes)


# Text ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("I'm", "I'm"),
        (f"I{TYPOGRAPHIC}m", "I'm"),
        ("I'll", "I'll"),
        ("don't", "don't"),
        ("LA", "LA"),
        ("PTSD,", "PTSD"),
        ("x-ray", "x ray"),
        ("9:30", "9 30"),
        ("[laughter]", "laughter"),
        ("<sync>", "sync"),
        ("'", ""),
        ("''", ""),
        ("'cause", "'cause"),
        ("nothin'", "nothin'"),
        ("rock'n'roll", "rock'n'roll"),
        (chr(0x00E9).join(["caf", ""]), "caf"),
        (f"{DOTTED_I}stanbul", "stanbul"),
        (f"{KELVIN}elvin", "elvin"),
        (chr(0x2018) + "quoted" + chr(0x2018), "quoted"),
    ],
)
def test_written_form_keeps_capitals_and_apostrophes_and_drops_other_marks(
    token: str, expected: str
) -> None:
    assert written_form(token) == expected


@pytest.mark.parametrize(
    ("line", "first", "last", "written"),
    [
        ("Well I'm fine", 0, 2, "Well I'm fine"),
        (f"I{TYPOGRAPHIC}m here", 0, 1, "I'm here"),
        ("the x-ray was fine", 0, 4, "the x ray was fine"),
        ("the x-ray was fine", 0, 1, "the x"),  # x-ray split at the part's end
        ("the x-ray was fine", 2, 4, "ray was fine"),  # and at its start
        ("at 9:30 then", 1, 2, "9 30"),
        ("my PTSD and LA", 0, 3, "my PTSD and LA"),
        ("so [laughter] yes", 0, 2, "so laughter yes"),
        ("' lone ' marks '", 0, 1, "lone marks"),
        (f"{DOTTED_I}stanbul now", 0, 2, "i stanbul now"),  # read as its stored words
        (f"a {KELVIN}elvin", 0, 1, "a kelvin"),
    ],
)
def test_part_text_reads_the_words_as_written_where_they_lie_whole_in_the_part(
    line: str, first: int, last: int, written: str
) -> None:
    made = part_text(line, first, last)
    assert made.written_text == written
    assert normalise(made.written_text).split() == normalise(line).split()[first : last + 1]
    if not any(character.isdigit() for character in written):
        assert made.text == written
        assert made.written_out == (0,) * len(written.split())


@pytest.mark.parametrize(
    ("digits", "words"),
    [
        ("0", "zero"),
        ("00", "zero"),
        ("000", "zero"),
        ("0000", "zero"),
        ("007", "seven"),
        ("05", "five"),
        ("7", "seven"),
        ("10", "ten"),
        ("15", "fifteen"),
        ("42", "forty two"),
        ("45", "forty five"),
        ("100", "one hundred"),
        ("105", "one hundred five"),
        ("125", "one hundred twenty five"),
        ("1000", "one thousand"),
        ("1500", "fifteen hundred"),
        ("1900", "nineteen hundred"),
        ("1905", "nineteen oh five"),
        ("1990", "nineteen ninety"),
        ("2000", "two thousand"),
        ("2005", "two thousand five"),
        ("2010", "twenty ten"),
        ("0123", "one twenty three"),
        ("0800", "eight hundred"),
        ("12345", "twelve thousand three hundred forty five"),
        ("100000", "one hundred thousand"),
        ("1234567", "one million two hundred thirty four thousand five hundred sixty seven"),
    ],
)
def test_a_number_is_written_out_as_misaki_reads_it_by_its_value(digits: str, words: str) -> None:
    # Checked against misaki 0.9.4's Lexicon.get_number on OMNI (and by the slow test).
    assert spoken_number(digits) == words


@pytest.mark.parametrize(
    "refused", ["", "12a", "1 2", "-5", "1.5", " 7", chr(0x0661) + chr(0x0662), chr(0x00B2)]
)
def test_only_a_word_of_ascii_digits_is_written_out(refused: str) -> None:
    with pytest.raises(ValueError, match="only a word of ASCII digits") as caught:
        spoken_number(refused)
    if refused.strip():
        assert refused.strip() not in str(caught.value)


def test_an_error_inside_num2words_never_shows_the_number(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(number: int, to: str) -> str:
        raise OverflowError(f"abs({number}) must be less than 10**306")

    monkeypatch.setattr(revoice, "num2words", failing)
    with pytest.raises(ValueError, match="num2words could not write out a number") as caught:
        spoken_number("4242")
    assert "4242" not in str(caught.value)
    assert caught.value.__context__ is None  # raised outside the except clause
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    ("line", "first", "last", "written", "text"),
    [
        ("at 9:30 then", 1, 2, "9 30", "nine thirty"),
        ("at 9:30 then", 0, 1, "at 9", "at nine"),  # 9:30 split at the part's end
        ("at 9:30 then", 2, 3, "30 then", "thirty then"),  # and at its start
        ("in 1990 or 1990s", 0, 3, "in 1990 or 1990s", "in nineteen ninety or 1990s"),
        ("the 3rd one", 0, 2, "the 3rd one", "the 3rd one"),  # digits and letters: Kokoro's
        ("back in '90 yes", 0, 3, "back in '90 yes", "back in ninety yes"),  # stored: 90
        ("007 and 0123", 0, 2, "007 and 0123", "seven and one twenty three"),
        ("1,000 people", 0, 2, "1 000 people", "one zero people"),  # misaki's own reading
    ],
)
def test_part_text_writes_out_each_word_whose_stored_form_is_only_digits(
    line: str, first: int, last: int, written: str, text: str
) -> None:
    made = part_text(line, first, last)
    assert (made.written_text, made.text) == (written, text)
    assert normalise(made.written_text).split() == normalise(line).split()[first : last + 1]
    assert len(made.written_out) == len(written.split())
    assert len(made.in_number) == len(text.split())
    for word, count in zip(written.split(), made.written_out, strict=True):
        assert count == (
            len(spoken_number(normalise(word)).split()) if normalise(word).isdigit() else 0
        )


def test_part_text_counts_how_the_text_was_made() -> None:
    line = f"[so] I{TYPOGRAPHIC}m x-ray {DOTTED_I}t ok 42"
    whole = part_text(line, 0, 7)
    assert whole.written_text == "so I'm x ray i t ok 42"
    assert whole.text == "so I'm x ray i t ok forty two"
    assert (whole.written, whole.unmatched, whole.typographic, whole.brackets) == (5, 1, 1, 1)
    assert whole.written_out == (0, 0, 0, 0, 0, 0, 0, 2)
    assert whole.numbers == 1
    assert whole.in_number == (False,) * 7 + (True, True)
    assert not whole.split_start
    assert not whole.split_end
    middle = part_text(line, 3, 3)  # "ray" alone
    assert middle.text == "ray"
    assert (middle.split_start, middle.split_end, middle.written) == (True, False, 0)
    both = part_text("one x-y-z two", 2, 2)  # "y" of a token that runs on both ways
    assert (both.text, both.split_start, both.split_end) == ("y", True, True)


def test_part_text_refuses_words_outside_the_line() -> None:
    with pytest.raises(ValueError, match="has 3 words"):
        part_text("one two three", 1, 3)
    with pytest.raises(ValueError, match="cannot run"):
        part_text("one two three", 2, 1)


ALPHABET = [
    *"abcXYZ019'-:.,!? ",
    TYPOGRAPHIC,
    chr(0x2018),  # a left quote, which normalise turns into a space
    chr(0x00E9),  # e with an acute accent
    DOTTED_I,
    KELVIN,
    "\t",
    chr(0x00A0),  # no-break space
    chr(0x2003),  # em space
    chr(0x3000),  # ideographic space
    "[",
    ")",
    "<",
]


@pytest.mark.parametrize("seed", range(5))
def test_part_text_always_gives_back_the_stored_words_of_a_random_part(seed: int) -> None:
    rng = np.random.default_rng(seed)
    checked = numbers = 0
    for _ in range(2000):
        line = "".join(rng.choice(ALPHABET, size=int(rng.integers(1, 60))))
        stored = normalise(line).split()
        if not stored:
            continue
        first = int(rng.integers(len(stored)))
        last = int(rng.integers(first, len(stored)))
        made = part_text(line, first, last)
        # The words as written give back the stored words exactly; the text read gives
        # them back with each word of ASCII digits written out, and every word has a letter.
        assert normalise(made.written_text).split() == stored[first : last + 1]
        expected = [
            word
            for stored_word in stored[first : last + 1]
            for word in (
                spoken_number(stored_word).split()
                if stored_word.isascii() and stored_word.isdigit()
                else [stored_word]
            )
        ]
        assert normalise(made.text).split() == expected
        assert all(any(character.isalpha() for character in word) for word in made.text.split())
        assert made.text == " ".join(made.text.split())
        assert len(made.in_number) == len(made.text.split())
        assert sum(made.written_out) == sum(made.in_number)
        numbers += made.numbers
        checked += 1
    assert checked > 1500
    assert numbers > 100


# A part ends between the two stored words of "x-ray" for each reason a part can end.
HEADER = "start_time\tstop_time\tspeaker\tvalue"
NORMAL, WITHOUT = 9001, 9003
RULE = CutRule.from_config(
    {
        "not_turns": r"^\W*sync(h|ing|hing)?\W*$",
        "guard_ms": 200,
        "offset_guard_ms": 400,
        "interview_only": True,
        "no_interviewer_turns": [WITHOUT],
        "offsets_s": {},
        "interruptions_s": {},
        "interruption_pad_s": 10.0,
    }
)


def daicwoz(*rows: tuple[float, float, str]) -> bytes:
    lines = [HEADER, *(f"{a}\t{b}\t{speaker}\tsomething" for a, b, speaker in rows)]
    return ("\r\n".join(lines) + "\r\n").encode()


def timed(session: int, line: int, start: float, text: str, step: float = 0.25) -> list[dict]:
    return [
        {
            "session": session,
            "line": line,
            "word_index": index,
            "word": word,
            "start": round(start + step * index, 3),
            "end": round(start + step * index + 0.2, 3),
        }
        for index, word in enumerate(normalise(text).split())
    ]


def filler(count: int) -> str:
    return " ".join(f"f{k}" for k in range(count))


def test_a_word_split_at_a_part_edge_is_read_as_its_stored_words_for_every_reason() -> None:
    lines = {
        NORMAL: [
            filler(39) + " x-ray " + filler(10),  # 0: x is word 39, ray word 40: segment edge
            "one x-ray two",  # 1: ray falls outside the participant's turns: dropped
            "three x-ray four",  # 2: her turn between x and ray
            "five x-ray six",  # 3: a word of line 4 between x and ray
            "seven",  # 4
            filler(60),  # 5
        ],
        WITHOUT: ["eight x-ray nine " + filler(45)],  # x and ray in two participant turns
    }
    words = pd.DataFrame(
        timed(NORMAL, 0, 10.0, lines[NORMAL][0])
        + timed(NORMAL, 1, 30.0, "one x")
        + [
            {"session": NORMAL, "line": 1, "word_index": 2, "word": "ray"}
            | {"start": 30.9, "end": 31.1},
            {"session": NORMAL, "line": 1, "word_index": 3, "word": "two"}
            | {"start": 31.6, "end": 31.8},
        ]
        + timed(NORMAL, 2, 40.0, "three x")
        + [
            {"session": NORMAL, "line": 2, "word_index": 2, "word": "ray"}
            | {"start": 41.2, "end": 41.4},
            {"session": NORMAL, "line": 2, "word_index": 3, "word": "four"}
            | {"start": 41.45, "end": 41.6},
        ]
        + [
            {"session": NORMAL, "line": 3, "word_index": i, "word": w}
            | {"start": s, "end": s + 0.1}
            for i, (w, s) in enumerate([("five", 50.0), ("x", 50.2), ("ray", 50.6), ("six", 50.8)])
        ]
        + [
            {
                "session": NORMAL,
                "line": 4,
                "word_index": 0,
                "word": "seven",
                "start": 50.4,
                "end": 50.5,
            }
        ]
        + timed(NORMAL, 5, 60.0, lines[NORMAL][5])
        + timed(WITHOUT, 0, 1.0, lines[WITHOUT][0])
    )
    transcripts = {
        NORMAL: daicwoz(
            (0.0, 5.0, "Ellie"),
            (5.0, 30.75, "Participant"),  # ray of line 1 (30.9-31.1) lies outside her turns
            (31.5, 40.5, "Participant"),
            (40.65, 40.85, "Ellie"),  # between x (40.25-40.45) and ray (41.2-41.4) of line 2
            (41.1, 80.0, "Participant"),
            (80.0, 81.0, "Ellie"),
        ),
        WITHOUT: daicwoz((0.0, 1.52, "Participant"), (1.58, 20.0, "Participant")),
    }
    decided = cut_words(words, all_turns(transcripts, RULE))
    assert decided.loc[(decided["line"] == 1) & (decided["word_index"] == 2), "kept"].tolist() == [
        False
    ]
    parts = segment_parts(decided.drop(columns=["word"]))
    segments = build_segments(decided)
    split = Counter()
    causes = Counter()
    places = revoice.WordPlaces(decided)
    for row in parts.itertuples():
        made = part_text(lines[row.session][row.line], row.first, row.last)
        if made.split_start:
            causes[places.cause(row.session, row.segment, row.line, row.first, row.first - 1)] += 1
        if made.split_end:
            causes[places.cause(row.session, row.segment, row.line, row.last, row.last + 1)] += 1
        assert (
            normalise(made.text).split()
            == normalise(lines[row.session][row.line]).split()[row.first : row.last + 1]
        )
        split[(row.session, row.line, "start")] += made.split_start
        split[(row.session, row.line, "end")] += made.split_end
    # Each x-ray is split once at a part's end (x) and once at a part's start (ray),
    # except in line 1, where ray was dropped and only x is read.
    for session, line in [(NORMAL, 0), (NORMAL, 2), (NORMAL, 3), (WITHOUT, 0)]:
        assert split[(session, line, "end")] == 1
        assert split[(session, line, "start")] == 1
    assert split[(NORMAL, 1, "end")] == 1
    assert split[(NORMAL, 1, "start")] == 0
    assert causes == {
        revoice.WordPlaces.SEGMENT_EDGE: 2,  # line 0
        "a dropped word (outside participant turns)": 1,  # line 1
        revoice.WordPlaces.SAME_SEGMENT: 4,  # her turn in line 2, the turn edge in WITHOUT
        revoice.WordPlaces.OTHER_LINE: 2,  # line 3
    }
    for segment, group in parts.groupby("segment"):
        session = int(segment.split("_")[0])
        expected = segments.set_index("segment").loc[segment, "text"]
        assert " ".join(segment_words(group, lines[session])) == expected


def test_word_classes_describe_a_word_without_its_text() -> None:
    assert word_classes("PTSD", edge=False) == {
        "digit": False,
        "apostrophe": False,
        "capitals": True,
        "one_letter": False,
        "part_edge": False,
    }
    assert word_classes("I", edge=True)["one_letter"]
    assert not word_classes("I", edge=True)["capitals"]
    assert word_classes("90s", edge=False)["digit"]
    assert word_classes("don't", edge=False)["apostrophe"]


# Voices, tasks and the test render -------------------------------------------------

VOICES = {"female": ["af_heart", "af_bella"], "male": ["am_michael", "am_fenrir"]}


@pytest.mark.parametrize("seed", range(20))
def test_each_voice_gets_half_of_each_class_of_its_gender(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 120))
    labels = pd.DataFrame(
        {"label": rng.integers(0, 2, n), "gender": rng.choice(["female", "male"], n)},
        index=pd.Index(rng.permutation(np.arange(300, 300 + n)), name="session"),
    )
    voices = assign_voices(labels, VOICES)
    assert set(voices) == set(labels.index)
    for gender, names in VOICES.items():
        for label in (0, 1):
            group = labels[(labels["gender"] == gender) & (labels["label"] == label)]
            got = Counter(voices[s] for s in group.index)
            assert set(got) <= set(names)
            assert abs(got[names[0]] - got[names[1]]) <= 1
        own = labels[labels["gender"] == gender].sort_values(["label"]).index
        first = min(own, key=lambda s: (labels.loc[s, "label"], s), default=None)
        if first is not None:
            assert voices[first] == names[0]


def test_voices_alternate_along_class_then_id() -> None:
    labels = pd.DataFrame(
        {"label": [1, 0, 0, 1, 0], "gender": ["female"] * 5},
        index=pd.Index([305, 304, 301, 302, 303], name="session"),
    )
    voices = assign_voices(labels, VOICES)
    order = [301, 303, 304, 302, 305]  # negatives by ID, then positives by ID
    assert [voices[s] for s in order] == [
        "af_heart",
        "af_bella",
        "af_heart",
        "af_bella",
        "af_heart",
    ]


def test_voices_refuse_an_unknown_gender_or_label() -> None:
    labels = pd.DataFrame({"label": [0], "gender": ["other"]}, index=pd.Index([300]))
    with pytest.raises(ValueError, match="no voices"):
        assign_voices(labels, VOICES)
    with pytest.raises(ValueError, match="0 or 1"):
        assign_voices(pd.DataFrame({"label": [2], "gender": ["male"]}, index=[300]), VOICES)


def test_tasks_hold_every_session_once_and_about_equal_work() -> None:
    rng = np.random.default_rng(9001)
    counts = {300 + k: int(rng.integers(3, 104)) for k in range(189)}
    tasks = assign_tasks(counts)
    assert set(tasks) == set(counts)
    assert set(tasks.values()) == set(range(TASKS))
    assert assign_tasks(dict(reversed(list(counts.items())))) == tasks
    load = Counter()
    for session, task in tasks.items():
        load[task] += counts[session]
    mean = sum(counts.values()) / TASKS
    assert max(load.values()) <= 1.05 * mean


def test_tasks_take_the_largest_session_first_and_break_ties_by_id() -> None:
    assert assign_tasks({301: 5, 300: 5, 302: 9, 303: 1}, tasks=2) == {
        302: 0,
        300: 1,
        301: 1,
        303: 0,
    }
    # 300 and 301 tie: the lower ID goes first, to the lower task.
    assert assign_tasks({300: 5, 301: 5, 302: 3}, tasks=2) == {300: 0, 301: 1, 302: 0}
    with pytest.raises(ValueError, match="at least one task"):
        assign_tasks({300: 1}, tasks=0)


def test_the_test_render_draws_outside_the_sample_and_always_the_same() -> None:
    segments = [f"{s}_{i:03d}" for s in range(300, 320) for i in range(30)]
    sample = set(segments[::7])
    drawn = draw_test_segments(segments, sample, 20, 106)
    assert len(drawn) == len(set(drawn)) == 20
    assert not set(drawn) & sample
    assert draw_test_segments(list(reversed(segments)), sample, 20, 106) == drawn
    assert draw_test_segments(segments, sample, 20, 107) != drawn
    with pytest.raises(ValueError, match="cannot draw"):
        draw_test_segments(segments[:5], set(), 6, 106)


# Settings --------------------------------------------------------------------------


def test_x1_settings_come_from_the_data_config() -> None:
    made = settings()
    assert made.voices == {"female": ("af_heart", "af_bella"), "male": ("am_michael", "am_fenrir")}
    assert made.all_voices == ("af_heart", "af_bella", "am_michael", "am_fenrir")
    assert (made.language, made.speed, made.seed) == ("a", 1.0, 106)
    assert (made.kokoro_rate, made.soxr_quality, made.trim_db) == (24_000, "HQ", -40.0)
    assert (made.trim_margin, made.fade, made.fade_samples, made.gap_samples) == (
        160,
        0.01,
        160,
        4800,
    )
    assert (made.rms, made.peak_limit, made.threads) == (0.05, 0.99, 4)
    assert (made.test_segments, made.test_seed, made.test_repeats, made.test_heard) == (
        20,
        106,
        2,
        10,
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"language": "b"}, "US English"),
        (
            {"voices": {"female": ["af_heart"], "male": ["am_michael", "am_fenrir"]}},
            "two different",
        ),
        (
            {"voices": {"female": ["af_heart", "am_bella"], "male": ["am_michael", "am_fenrir"]}},
            "female voice",
        ),
        (
            {"voices": {"female": ["bf_emma", "af_bella"], "male": ["am_michael", "am_fenrir"]}},
            "female voice",
        ),
        ({"voices": {"female": ["af_heart", "af_bella"]}}, "female and male"),
        ({"kokoro_rate_hz": 22050}, "24000"),
        ({"soxr_quality": "best"}, "quality"),
        ({"speed": 0.0}, "speed"),
        ({"trim_db": 0}, "below the peak"),
        ({"trim_margin_ms": 10.03}, "whole number of samples"),
        ({"trim_margin_ms": 5}, "whole fade"),
        ({"rms": 1.5}, "RMS"),
        ({"threads": 0}, "thread"),
        ({"seed": 1.5}, "whole number"),
        ({"seed": True}, "whole number"),
        ({"test_render": {"segments": 20, "seed": 106, "repeats": 2, "heard": 21}}, "hear"),
        ({"test_render": {"segments": 20, "seed": 106, "repeats": 0, "heard": 10}}, "repeat"),
    ],
)
def test_x1_settings_refuse_values_out_of_range(change: dict, message: str) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["views"]["x1"] = {**config["views"]["x1"], **change}
    with pytest.raises(ValueError, match=message):
        X1Settings.from_config(config)


def test_x1_settings_refuse_files_that_are_not_16_bit() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["views"]["flac_bits"] = 24
    with pytest.raises(ValueError, match="16-bit"):
        X1Settings.from_config(config)


# Audio -----------------------------------------------------------------------------


def speech(seconds: float, rng: np.random.Generator, level: float = 0.1) -> np.ndarray:
    return (rng.normal(0.0, level, round(seconds * KOKORO_RATE))).astype(np.float32)


def kokoro_like(
    rng: np.random.Generator, body: float = 1.0, lead: float = 0.3, tail: float = 0.6
) -> np.ndarray:
    """A render as Kokoro makes one: quiet before and after the speech, never exactly 0."""
    quiet = lambda seconds: speech(seconds, rng, 1e-5)  # noqa: E731
    return np.concatenate([quiet(lead), speech(body, rng), quiet(tail)])


def test_a_part_is_resampled_then_trimmed_with_a_margin_then_faded() -> None:
    rng = np.random.default_rng(9001)
    render = kokoro_like(rng)
    made = settings()
    piece, record = part_audio(render, made)
    resampled = soxr.resample(render, KOKORO_RATE, RATE, quality="HQ")
    trimmed = trim_quiet(resampled, db=-40.0, margin=160)
    np.testing.assert_array_equal(piece, fade_part(trimmed.kept, fade=0.01))
    assert piece.dtype == np.float32
    assert piece[0] == 0.0
    assert piece[-1] == 0.0
    assert (record.rendered, record.resampled, record.kept) == (
        len(render),
        len(resampled),
        len(trimmed.kept),
    )
    # The resampler rings a few samples before and after a sharp edge of the speech.
    assert record.removed_start == pytest.approx(0.3 * RATE - 160, abs=20)
    assert record.removed_end == pytest.approx(0.6 * RATE - 160, abs=20)
    assert (record.margin_start, record.margin_end) == (160, 160)


def test_the_trim_keeps_quiet_stretches_inside_a_part() -> None:
    rng = np.random.default_rng(9002)
    quiet = np.zeros(8000, np.float32)
    part = np.concatenate([quiet, speech(0.2, rng), quiet, speech(0.2, rng), quiet])
    trimmed = trim_quiet(part, db=-40.0, margin=160)
    assert np.count_nonzero(trimmed.kept == 0.0) >= 8000
    assert trimmed.removed_start == 8000 - 160
    assert trimmed.removed_end == 8000 - 160


def test_the_trim_margin_stops_at_the_ends_of_the_render() -> None:
    part = np.array([0.5, 0.0, 0.0, 0.0, 0.5], np.float32)
    trimmed = trim_quiet(part, db=-40.0, margin=160)
    np.testing.assert_array_equal(trimmed.kept, part)
    assert (trimmed.removed_start, trimmed.margin_start, trimmed.margin_end) == (0, 0, 0)
    with pytest.raises(ValueError, match="silent"):
        trim_quiet(np.zeros(100, np.float32), db=-40.0, margin=160)


def test_a_burst_right_at_the_trim_point_keeps_its_energy() -> None:
    # A plosive: silence, then speech that starts at full strength. The fades fall in the
    # 10 ms margin before and after it, so none of its samples are faded.
    burst = np.zeros(KOKORO_RATE, np.float32)
    burst[7200:12000] = speech(0.2, np.random.default_rng(9003), 0.3)
    piece, record = part_audio(burst, settings())
    resampled = soxr.resample(burst, KOKORO_RATE, RATE, quality="HQ")
    loud = np.flatnonzero(np.abs(resampled) >= np.abs(resampled).max() / 100)
    inside = resampled[loud[0] : loud[-1] + 1]
    start = loud[0] - record.removed_start
    np.testing.assert_array_equal(piece[start : start + len(inside)], inside)
    assert record.margin_start == record.margin_end == 160


def test_a_clip_has_silent_gaps_parts_that_start_and_end_at_0_and_rms_0_05() -> None:
    rng = np.random.default_rng(9004)
    renders = [kokoro_like(rng, body) for body in (1.0, 0.4, 2.0)]
    built = build_revoiced(renders, settings())
    samples = to_pcm16(built.clip)
    check_segment_views({"x1": samples}, built.part_starts, built.part_ends, gap=4800)
    for end, start in zip(built.part_ends[:-1], built.part_starts[1:], strict=True):
        assert start - end == 4800
        assert not samples[end:start].any()
    assert np.sqrt(np.mean(built.clip**2)) == pytest.approx(0.05)
    assert not built.peak_limited
    assert built.clip.dtype == np.float64


def test_a_clip_with_a_high_peak_is_scaled_down_and_flagged() -> None:
    rng = np.random.default_rng(9005)
    render = kokoro_like(rng, 1.0) * 0.2  # speech at about 0.02, within 40 dB of ...
    render[12000] = 1.0  # ... one spike, which the RMS gain would push past full scale
    built = build_revoiced([render], settings())
    assert built.peak_limited
    assert np.max(np.abs(built.clip)) == pytest.approx(0.99)
    # The recorded gain is the RMS gain times the peak factor: it takes the part's peak to 0.99.
    assert built.gain * built.parts[0].peak == pytest.approx(0.99)
    assert np.sqrt(np.mean(built.clip**2)) < 0.05


@pytest.mark.parametrize(
    ("render", "message"),
    [
        (np.zeros(0, np.float32), "shape"),
        (np.array([0.1, np.nan, 0.2] * 100, np.float32), "NaN"),
        (np.array([0.1, np.inf, 0.2] * 100, np.float32), "NaN"),
        (np.full(4800, 1e-5, np.float32), "peak"),
        (np.full((2, 4800), 0.1, np.float32), "shape"),
    ],
)
def test_a_render_that_is_empty_not_finite_or_silent_is_refused(
    render: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        check_render(render)
    with pytest.raises(ValueError, match=message):
        build_revoiced([render], settings())


def test_a_part_shorter_than_one_fade_is_refused() -> None:
    short = speech(0.006, np.random.default_rng(9006))  # 144 samples at 24 kHz, 96 at 16 kHz
    with pytest.raises(ValueError, match="fewer than one fade"):
        part_audio(short, settings())
    with pytest.raises(ValueError, match="at least one part"):
        build_revoiced([], settings())


class SeededKokoro:
    """Renders that depend only on the seed and the text's length, from NumPy."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def render(self, text: str, voice: str, seed: int) -> Render:
        self.calls.append((text, voice, seed))
        rng = np.random.default_rng(seed)
        audio = kokoro_like(rng, 0.1 * len(text.split()))
        return Render(audio=audio, phonemes="ˈfake", first_frames=12, last_frames=30)


LINES = ["the x-ray was fine", f"I{TYPOGRAPHIC}m {SECRET} at 9:30 yes", "so it goes"]


def parts_of(segment: str, *rows: tuple[int, int, int, int]) -> pd.DataFrame:
    """Parts of one segment: (cut part, line, first, last) in clip order."""
    return pd.DataFrame(
        [
            {
                "segment": segment,
                "session": int(segment.split("_")[0]),
                "position": position,
                "cut_part": cut_part,
                "line": line,
                "first": first,
                "last": last,
                "n_words": last - first + 1,
                "start": 1.0 + position,
                "end": 1.5 + position,
            }
            for position, (cut_part, line, first, last) in enumerate(rows)
        ]
    )


def test_each_part_is_seeded_by_segment_and_position_from_0_whatever_its_cut_key() -> None:
    kokoro = SeededKokoro()
    parts = parts_of("300_004", (17, 0, 2, 4), (18, 1, 0, 5), (20, 2, 0, 2))
    rendered = render_segment(kokoro, parts, LINES, "af_bella", "300_004", settings())
    seeds = [item_seed(106, "300_004", position) for position in range(3)]
    assert [seed for _, _, seed in kokoro.calls] == seeds
    assert [record["seed"] for record in rendered.parts] == seeds
    assert len(set(seeds)) == 3
    assert [text for text, _, _ in kokoro.calls] == [
        "ray was fine",
        f"I'm {SECRET} at nine thirty yes",
        "so it goes",
    ]
    assert rendered.parts[1]["written_text"] == f"I'm {SECRET} at 9 30 yes"
    assert [record["numbers"] for record in rendered.parts] == [0, 2, 0]
    assert [record["words_read"] for record in rendered.parts] == [3, 6, 3]
    assert {voice for _, voice, _ in kokoro.calls} == {"af_bella"}
    again = render_segment(SeededKokoro(), parts, LINES, "af_bella", "300_004", settings())
    np.testing.assert_array_equal(rendered.samples, again.samples)


def test_render_segment_equals_the_part_functions_called_by_hand() -> None:
    parts = parts_of("301_002", (0, 0, 0, 4), (1, 1, 1, 2))
    rendered = render_segment(SeededKokoro(), parts, LINES, "am_fenrir", "301_002", settings())
    made = settings()
    renders = [
        SeededKokoro()
        .render(part_text(LINES[line], first, last).text, "am_fenrir", item_seed(106, "301_002", k))
        .audio
        for k, (line, first, last) in enumerate([(0, 0, 4), (1, 1, 2)])
    ]
    pieces = []
    for render in renders:
        resampled = soxr.resample(render, KOKORO_RATE, RATE, quality="HQ")
        pieces.append(fade_part(trim_quiet(resampled, db=-40.0, margin=160).kept, fade=0.01))
    joined, starts, ends = join_parts(pieces)
    clip, gain = set_rms(joined, 0.05)
    clip, factor, limited = limit_peak(clip, 0.99)
    np.testing.assert_array_equal(rendered.samples, to_pcm16(clip))
    assert rendered.part_starts == tuple(starts)
    assert rendered.part_ends == tuple(ends)
    assert rendered.gain == pytest.approx(gain * factor)
    assert rendered.peak_limited == limited
    assert [r.tolist() for r in rendered.renders] == [r.tolist() for r in renders]
    assert rendered.parts[1]["text"] == f"{SECRET} at"
    assert rendered.parts[1]["part_start"] == starts[1]
    built = build_revoiced(renders, made)
    np.testing.assert_array_equal(to_pcm16(built.clip), rendered.samples)


def test_render_segment_names_the_part_that_fails() -> None:
    class Failing(SeededKokoro):
        def render(self, text: str, voice: str, seed: int) -> Render:
            if SECRET in text:
                raise RuntimeError(f"cannot read {text}")
            return super().render(text, voice, seed)

    parts = parts_of("300_001", (0, 0, 0, 3), (1, 1, 0, 5))
    with pytest.raises(PartError, match="segment 300_001, part 1: RuntimeError") as caught:
        render_segment(Failing(), parts, LINES, "af_heart", "300_001", settings())
    assert SECRET not in str(caught.value)
    assert caught.value.__suppress_context__
    with pytest.raises(ValueError, match="clip order"):
        render_segment(SeededKokoro(), parts.iloc[::-1], LINES, "af_heart", "300_001", settings())


def test_a_sound_in_a_gap_stops_the_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    join = revoice.join_revoiced

    def leaky(*args: object, **kwargs: object) -> revoice.Revoiced:
        built = join(*args, **kwargs)
        clip = built.clip.copy()
        clip[built.part_ends[0]] = 0.01
        return dataclasses.replace(built, clip=clip)

    monkeypatch.setattr(revoice, "join_revoiced", leaky)
    parts = parts_of("300_001", (0, 0, 0, 3), (1, 1, 0, 5))
    with pytest.raises(ValueError, match="a gap is not silent"):
        render_segment(SeededKokoro(), parts, LINES, "af_heart", "300_001", settings())


# Kokoro ----------------------------------------------------------------------------


def test_the_stand_in_vocabulary_is_kokoros() -> None:
    assert len(KOKORO_VOCABULARY) == 114


@pytest.mark.parametrize(
    ("phonemes", "voiced"),
    [
        ("hˈɛloʊ", True),
        ("", False),
        ("“", False),  # read only as a quote mark
        ("ˈ", False),  # only a stress mark
        ("ːˌ", False),
        ("ʰ" + chr(0x0303), False),
        ("→↓", False),
        ("(fɹ)", True),  # an espeak language flag around sounds: caught by the bracket rule
        ("()", False),
        ("ꭧ", True),
        ("g", False),  # not in Kokoro's vocabulary (it has ɡ)
    ],
)
def test_a_word_has_a_sound_only_if_kokoro_can_voice_one_of_its_characters(
    phonemes: str, voiced: bool
) -> None:
    assert word_has_sound(phonemes, KOKORO_VOCABULARY) is voiced


def front_end_of(text: str, **pipeline: object) -> FrontEnd:
    return fake_kokoro(FakePipeline(**pipeline)).front_end(text)


def test_the_front_end_groups_tokens_into_written_words_and_reads_both_rating_fields() -> None:
    readings = {
        "I'm": [("ˈIm", None, 4)],  # merged: the rating sits in _
        "zorblefax": [("zˈɔɹbəlfæks", 2, None)],  # espeak, rating on the token
        "rock'n'roll": [("ɹˈɑk", None, 2), ("ən", None, 2), ("ɹˌOl", None, 2)],  # merged, espeak
    }
    front = front_end_of("I'm zorblefax rock'n'roll yes", readings=readings)
    assert [word.tokens for word in front.words] == [1, 1, 3, 1]
    assert [word.espeak for word in front.words] == [False, True, True, False]
    assert [word.rating for word in front.words] == [4, 2, 2, 4]
    assert front.silent == 0
    assert front.ok
    assert front.phonemes == "ˈIm zˈɔɹbəlfæks ɹˈɑkənɹˌOl ˈyes"
    assert front.word_phonemes == ("ˈIm", "zˈɔɹbəlfæks", "ɹˈɑkənɹˌOl", "ˈyes")
    assert not any(word.punctuation or word.lost_piece for word in front.words)


def test_the_front_end_marks_words_with_a_piece_that_misaki_read_as_punctuation() -> None:
    # (phonemes, own rating, merged rating, spaCy tag, token text), as misaki makes them.
    readings = {
        "90s": [("", None, 4, ",", "90"), ("ˈɛs", 4, None, ",", "s")],  # "90" lost, s kept
        "I'm": [  # an apostrophe read as punctuation: the word is split into letters
            ("ˈI", 4, None, "PRP", "I"),
            ("", None, 4, "NFP", "'"),
            ("ˈɛm", 4, None, "NFP", "m"),
        ],
        "uh": [("ˈʌ", 4, None, ".", "uh")],  # letters only: misaki's lexicon reads it
        "'cause": [("“", None, 4, "``", "'"), ("kˈʌz", 4, None, "IN", "cause")],  # a quote mark
        "don't": [("dˈO", 4, None, "VB", "do"), ("", None, 4, ".", "n't")],  # n't lost
        # Silent pieces that misaki did not read as punctuation: letters only, or no
        # punctuation tag.
        "uhm": [("ˈʌm", 4, None, "UH", "uhm"), ("", None, 4, ".", "uh")],
        "ok'": [("ˈOkA", 4, None, "UH", "ok"), ("", None, 4, "NN", "'")],
    }
    front = front_end_of("so 90s I'm uh 'cause don't uhm ok' yes", readings=readings)
    no, yes = False, True
    assert [word.punctuation for word in front.words] == [no, yes, yes, no, yes, yes, no, no, no]
    assert [word.lost_piece for word in front.words] == [no, yes, no, no, no, yes, no, no, no]
    assert front.silent == 0  # every word keeps a sound, so no stop rule sees them
    assert front.ok
    assert front.word_phonemes[1:3] == ("ˈɛs", "ˈIˈɛm")


def test_a_number_read_as_punctuation_is_silent_unless_it_is_written_out() -> None:
    as_written = front_end_of("maybe 00 yes", punctuation_digits=True)
    assert as_written.silent == 1
    assert as_written.words[1].punctuation
    assert (as_written.words[1].rating, as_written.words[1].tags) == (4, (".",))
    made = part_text("maybe 00 yes", 0, 2)
    assert made.text == "maybe zero yes"
    read = front_end_of(made.text, punctuation_digits=True)
    assert read.silent == 0
    assert read.ok


@pytest.mark.slow
def test_numbers_are_written_out_exactly_as_misaki_reads_them() -> None:
    """OMNI only, where misaki is installed: the readings of misaki 0.9.4 itself."""
    en = pytest.importorskip("misaki.en")
    lexicon = en.Lexicon(british=False)
    extra = ["00", "000", "0000", "007", "0123", "0800", "100000", "1234567", "18005551234"]
    for digits in [*map(str, range(20_001)), *extra]:
        theirs, _ = lexicon.get_number(digits, None, True, "")
        words = spoken_number(digits).split()
        ours = " ".join(lexicon.lookup(word, None, None, None)[0] for word in words)
        assert ours == theirs, f"{len(digits)} digits"
    assert en.PUNCT_TAGS == revoice.PUNCT_TAGS


@pytest.mark.parametrize(
    ("readings", "failure"),
    [
        ({"mid": [("", 4, None)]}, "silent"),  # a silent word in the middle of a part
        ({"mid": [(None, None, None)]}, "silent"),  # no phonemes at all
        ({"mid": [("“", 4, None)]}, "silent"),  # read only as a quote mark
        ({"mid": [("ˈ", 4, None)]}, "silent"),  # only a stress mark
        ({"mid": [("ˈmɪd" + chr(0x0294) * 0 + "ʘ", 4, None)]}, "outside"),  # ʘ is not Kokoro's
        ({"mid": [("(fɹ)mˈid", 2, None)]}, "brackets"),  # an espeak language flag
        ({"mid": [("(", 4, None), ("mˈid", 4, None)]}, "brackets"),  # a lone ( token
        ({"mid": [("ɑ" * 600, 4, None)]}, "too_long"),  # two Kokoro calls
    ],
)
def test_the_front_end_finds_what_kokoro_would_not_voice(readings: dict, failure: str) -> None:
    front = front_end_of("one mid two", readings=readings)
    assert not front.ok
    found = {
        "silent": front.silent > 0,
        "outside": bool(front.outside),
        "brackets": front.brackets,
        "too_long": front.too_long,
    }
    assert found[failure]
    if failure == "outside":
        assert front.outside == ("U+0298",)


def test_the_front_end_refuses_tokens_that_do_not_form_one_group_per_word() -> None:
    readings = {"two": [("tu", 4, None), ("", 4, None)]}
    pipeline = FakePipeline(readings=readings)
    kokoro = fake_kokoro(pipeline)
    tokens = [fake_token("one", " ", "wʌn"), fake_token("two", " ", "tu"), fake_token("", "", "")]
    pipeline.g2p = lambda text: ("", tokens)  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="3 word groups for 2 written words"):
        kokoro.front_end("one two")


def test_misaki_prints_are_caught_counted_and_never_shown(
    capsys: pytest.CaptureFixture[str],
) -> None:
    front = front_end_of(f"say {SECRET}", printing=True)
    assert front.prints == 1
    out, err = capsys.readouterr()
    assert SECRET not in out + err


def test_render_seeds_pytorch_right_before_kokoro_runs_and_returns_the_boundary_tokens() -> None:
    import torch

    kokoro = fake_kokoro()
    first = kokoro.render("one two three", "af_heart", 7)
    torch.manual_seed(0)
    torch.randn(1000)  # draws in between change nothing
    second = kokoro.render("one two three", "af_heart", 7)
    np.testing.assert_array_equal(first.audio, second.audio)
    assert first.audio.dtype == np.float32
    assert (first.first_frames, first.last_frames) == (12, 30)
    assert first.phonemes == "ˈone ˈtwo ˈthree"
    other = kokoro.render("one two three", "af_heart", 8)
    assert not np.array_equal(first.audio, other.audio)
    assert kokoro.pipeline.calls[0] == ("one two three", str(Path("voices/af_heart.pt")), 1.0)


@pytest.mark.parametrize(
    ("pipeline", "message"),
    [
        ({"results": 2}, "2 results"),  # two chunks
        ({"results": 0}, "0 results"),
        ({"phonemes": "ˈother"}, "other phonemes"),
        ({"readings": {"two": [("", 4, None)]}}, "1 words without a sound"),
    ],
)
def test_render_refuses_anything_but_one_result_of_the_front_ends_phonemes(
    pipeline: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fake_kokoro(FakePipeline(**pipeline)).render("one two", "af_heart", 1)


def test_a_model_left_in_training_mode_is_refused() -> None:
    import torch

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.2))
    with pytest.raises(ValueError, match="training mode"):
        check_eval(model)
    model.eval()
    check_eval(model)
    model[1].train()
    with pytest.raises(ValueError, match="training mode"):
        check_eval(model)


def test_kokoro_tokens_with_a_rating_in_either_place_count_as_espeak() -> None:
    own = SimpleNamespace(rating=2, _=SimpleNamespace(rating=None))
    merged = SimpleNamespace(_=SimpleNamespace(rating=2))
    neither = SimpleNamespace(_=None)
    assert revoice.token_ratings(own) == (2, None)
    assert revoice.token_ratings(merged) == (None, 2)
    assert revoice.token_ratings(neither) == (None, None)


# Inputs ----------------------------------------------------------------------------


def test_load_inputs_ties_the_parts_to_the_segments_labels_and_x0(x1_workspace) -> None:
    from listening_beyond_words.settings import Workspace

    workspace = Workspace(x1_workspace.root)
    made = X1Settings.from_config(x1_workspace.config)
    inputs = load_inputs(workspace, x1_workspace.config, made)
    assert inputs.segments["segment"].tolist() == sorted(x1_workspace.segments["segment"])
    assert inputs.parts["segment"].nunique() == len(inputs.segments)
    assert set(inputs.voices) == {300, 301, 302, 303, 304}
    assert set(inputs.tasks) == {300, 301, 302, 303}
    assert set(inputs.pins) == {
        "segments",
        "labels",
        "wordcheck_sample",
        "x0",
        "cut_words",
        "transcripts",
    }
    assert inputs.pins["cut_words"]["file"] == "derived/segments/run-1/words.parquet"
    for segment, group in inputs.parts.groupby("segment"):
        session = int(segment.split("_")[0])
        text = x1_workspace.segments.set_index("segment").loc[segment, "text"]
        assert " ".join(segment_words(group, x1_workspace.lines[session])) == text


def test_load_inputs_refuses_parts_that_differ_from_the_segments_or_x0(x1_workspace) -> None:
    from listening_beyond_words.runinfo import sha256_file
    from listening_beyond_words.settings import Workspace

    workspace = Workspace(x1_workspace.root)
    made = X1Settings.from_config(x1_workspace.config)
    x0_path = x1_workspace.root / "views" / "x0" / "run-1" / "manifest.parquet"
    x0 = pd.read_parquet(x0_path)
    x0.at[0, "part_starts"] = [0, 1, 2, 3, 4, 5, 6]
    x0.to_parquet(x0_path, index=False)
    x1_workspace.config["view_runs"]["x0"]["sha256"] = sha256_file(x0_path)
    with pytest.raises(ValueError, match="x0's pinned manifest"):
        load_inputs(workspace, x1_workspace.config, made)
    x0.at[0, "part_starts"] = list(range(len(x0.at[0, "part_ends"])))
    x0.to_parquet(x0_path, index=False)
    x1_workspace.config["view_runs"]["x0"]["sha256"] = sha256_file(x0_path)
    load_inputs(workspace, x1_workspace.config, made)  # x0 fits again
    labels_path = x1_workspace.root / "derived" / "labels" / "run-1" / "labels.parquet"
    pd.read_parquet(labels_path).drop(index=303).to_parquet(labels_path)
    x1_workspace.config["labels"]["sha256"] = sha256_file(labels_path)
    with pytest.raises(ValueError, match="has no label"):
        load_inputs(workspace, x1_workspace.config, made)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("a start moved by 1 ms", "starts differ"),
        ("an end moved by 1 ms", "ends differ"),
        ("a segment dropped", "do not give the pinned"),
    ],
)
def test_load_inputs_refuses_pinned_segments_that_the_cut_words_do_not_give(
    x1_workspace, change: str, message: str
) -> None:
    import json

    from listening_beyond_words.runinfo import sha256_file
    from listening_beyond_words.settings import Workspace

    run = x1_workspace.root / "derived" / "segments" / "run-1"
    segments = pd.read_parquet(run / "segments.parquet")
    if change.endswith("moved by 1 ms"):
        column = "starts" if change.startswith("a start") else "ends"
        times = [list(s) for s in segments[column]]
        times[0][-1] = round(times[0][-1] + 0.001, 3)
        segments[column] = times
    else:
        segments = segments.iloc[1:]
    segments.to_parquet(run / "segments.parquet", index=False)
    sha256 = sha256_file(run / "segments.parquet")
    built = json.loads((run / "run.json").read_text(encoding="utf-8"))
    built["outputs"]["segments.parquet"] = sha256  # pinned again, as a new segments run would be
    (run / "run.json").write_text(json.dumps(built), encoding="utf-8")
    x1_workspace.config["segments"]["sha256"] = sha256
    made = X1Settings.from_config(x1_workspace.config)
    with pytest.raises(ValueError, match=message):
        load_inputs(Workspace(x1_workspace.root), x1_workspace.config, made)
