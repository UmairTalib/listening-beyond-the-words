import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from listening_beyond_words.cut import (
    CAUSES,
    FLAGS,
    NO_PART,
    CutRule,
    all_turns,
    cut_words,
    session_turns,
    to_ms,
    zone_table,
)
from listening_beyond_words.daicwoz import read_turns
from listening_beyond_words.segments import build_segments, segment_neighbours, segment_walls
from listening_beyond_words.views import part_windows

HEADER = "start_time\tstop_time\tspeaker\tvalue"
NORMAL, OFFSET, WITHOUT, INTERRUPTED = 9001, 9002, 9003, 9004
SETTINGS = {
    "not_turns": r"^\W*sync(h|ing|hing)?\W*$",
    "guard_ms": 200,
    "offset_guard_ms": 400,
    "interview_only": True,
    "no_interviewer_turns": [WITHOUT],
    "listening": {"max_heard": 2, "seed": 102, "fallback_seed": 104, "fallback_guard_ms": 500},
    "offsets_s": {OFFSET: 2.5},
    "timing_check": {"run": "derived/daicwoz_timing/run-test", "sha256": "0" * 64},
    "interruptions_s": {INTERRUPTED: [50.0, 60.0]},
    "interruption_pad_s": 10.0,
}
RULE = CutRule.from_config(SETTINGS)


def transcript(*rows: tuple[float, float, str, str]) -> bytes:
    lines = [HEADER, *(f"{a}\t{b}\t{speaker}\t{text}" for a, b, speaker, text in rows)]
    return ("﻿" + "\r\n".join(lines) + "\r\n").encode()


TRANSCRIPTS = {
    NORMAL: transcript(
        (0.0, 0.9, "Participant", "hello"),
        (1.0, 2.0, "Ellie", "hi i'm ellie"),
        (2.0, 4.0, "Participant", "first"),
        (4.5, 10.0, "Participant", "second"),
        (5.0, 5.2, "Ellie", "mhm"),
        (10.0, 12.0, "Ellie", "how are you"),
        (12.0, 20.0, "Participant", "third"),
        (20.0, 21.0, "Participant", "[sync]"),
        (21.0, 29.0, "Participant", "fourth"),
        (25.0, 25.5, "Ellie", "  "),
        (30.0, 31.0, "Ellie", "goodbye"),
        (31.5, 40.0, "Participant", "bye"),
        (50.0, 50.5, "Ellie", "<SYNCH>"),
    ),
    OFFSET: transcript(
        (8.0, 9.0, "Ellie", "hi"),
        (9.0, 10.0, "Participant", "hi"),
        (10.0, 12.0, "Ellie", "how are you"),
        (12.0, 20.0, "Participant", "fine"),
        (25.0, 26.0, "Ellie", "bye"),
    ),
    WITHOUT: transcript(
        (0.0, 5.0, "Participant", "one"),
        (5.0, 8.0, "Participant", "two"),
        (9.0, 12.0, "Participant", "three"),
        (12.5, 15.0, "Participant", "four"),
    ),
    INTERRUPTED: transcript(
        (1.0, 2.0, "Ellie", "hi"),
        (2.0, 100.0, "Participant", "long story"),
        (100.0, 101.0, "Ellie", "bye"),
    ),
}
TURNS = all_turns(TRANSCRIPTS, RULE)


def words(
    session: int, *times: tuple[float, float], lines: list[int] | None = None
) -> pd.DataFrame:
    """One row per word; every word in line 0 unless ``lines`` says otherwise."""
    lines = lines or [0] * len(times)
    rows = []
    for k, ((start, end), line) in enumerate(zip(times, lines, strict=True)):
        rows.append(
            {
                "session": session,
                "line": line,
                "word_index": k,
                "word": f"w{k}",
                "start": start,
                "end": end,
            }
        )
    return pd.DataFrame(rows)


def decide(session: int, start: float, end: float) -> pd.Series:
    return cut_words(words(session, (start, end)), TURNS).iloc[0]


@pytest.mark.parametrize(
    ("start", "end", "kept", "cause"),
    [
        (12.2, 12.5, True, None),  # starts exactly 200 ms after her turn ends
        (12.199, 12.5, False, "interviewer turn"),  # 1 ms closer
        (9.3, 9.8, True, None),  # ends exactly 200 ms before her turn starts
        (9.3, 9.801, False, "interviewer turn"),
        (10.5, 11.0, False, "interviewer turn"),  # inside her turn, and outside the participant's
        (20.3, 20.6, False, "outside participant turns"),  # in a gap; a sync row is not a turn
        (19.7, 20.2, True, None),  # runs past the turn's end, but its middle lies inside
        (19.9, 20.1, True, None),  # its middle lies exactly on the turn's end
        (20.9, 21.1, True, None),  # its middle lies exactly on the next turn's start
        (20.899, 21.099, False, "outside participant turns"),  # 1 ms before that start
        (4.4, 4.6, True, None),  # middle on the start of a turn that follows a gap
        (19.9, 20.102, False, "outside participant turns"),  # middle 1 ms past the end
        (24.5, 24.9, True, None),  # a blank row is not one of her turns
        (0.2, 0.5, False, "interview limits"),  # before her first turn starts
        (32.0, 32.5, False, "interview limits"),  # after her last turn; a sync row is not a turn
    ],
)
def test_words_of_a_normal_session(start: float, end: float, kept: bool, cause: str | None) -> None:
    row = decide(NORMAL, start, end)
    assert row["kept"] == kept
    assert row["cause"] == cause
    assert (row["part"] == NO_PART) != kept


def test_a_word_records_every_condition_and_one_cause() -> None:
    row = decide(NORMAL, 10.5, 11.0)
    assert row["near_interviewer"]
    assert row["outside_participant"]
    assert not row["outside_interview"]
    assert not row["near_interruption"]
    assert row["cause"] == CAUSES["near_interviewer"]


@pytest.mark.parametrize(
    ("start", "end", "kept"),
    [
        (
            12.6,
            13.0,
            False,
        ),  # inside her turn once it is shifted by +2.5 s (10-12 s -> 12.5-14.5 s)
        (14.9, 15.2, True),  # exactly 400 ms after her shifted turn
        (14.899, 15.2, False),
        (14.8, 15.2, False),  # 300 ms away: kept with the normal guard, dropped with the wider one
        (9.5, 9.8, False),  # in the unshifted participant turn, before her shifted first turn
    ],
)
def test_words_of_a_session_with_an_offset(start: float, end: float, kept: bool) -> None:
    assert decide(OFFSET, start, end)["kept"] == kept


def test_offset_moves_every_turn_of_its_session_only() -> None:
    zones = zone_table(TURNS)
    shifted = zones[zones["session"] == OFFSET]
    assert shifted[["start", "end"]].to_numpy().tolist() == [
        [10.5, 11.5],
        [12.5, 14.5],
        [27.5, 28.5],
    ]
    assert TURNS[OFFSET].participant.tolist() == [[11500, 12500], [14500, 22500]]
    assert TURNS[OFFSET].interview == (10500, 28500)
    assert TURNS[NORMAL].interview == (1000, 31000)


def test_session_without_interviewer_turns_keeps_only_participant_turns() -> None:
    table = cut_words(
        words(WITHOUT, (1.0, 1.2), (4.9, 5.1), (6.0, 6.2), (9.5, 9.8), (12.0, 12.3), (13.0, 13.3)),
        TURNS,
    )
    assert table["kept"].tolist() == [True, True, True, True, False, True]
    assert table["cause"].tolist()[4] == "outside participant turns"
    # Touching turns merge (0-5 s and 5-8 s); a part ends where the next turn begins.
    assert table["part"].tolist() == [0, 0, 0, 1, NO_PART, 2]
    assert not table["outside_interview"].any()
    assert TURNS[WITHOUT].interview is None


def test_a_pause_between_two_participant_turns_stays_inside_a_part() -> None:
    table = cut_words(words(NORMAL, (19.0, 19.3), (21.2, 21.5)), TURNS)
    assert table["kept"].all()
    assert table["part"].tolist() == [0, 0]


@pytest.mark.parametrize(
    ("times", "lines", "parts"),
    [
        ([(9.0, 9.3), (12.5, 12.8)], [0, 0], [0, 1]),  # her turn lies between them
        ([(3.6, 3.9), (4.1, 4.4), (4.5, 4.75)], [0, 0, 0], [0, NO_PART, 1]),  # a dropped word
        ([(6.0, 6.3), (6.5, 6.8)], [3, 4], [0, 1]),  # a new transcript line
        ([(4.5, 4.7), (5.0, 5.1), (5.5, 5.8)], [0, 0, 0], [0, NO_PART, 1]),  # a short "mhm" of hers
        ([(6.0, 6.3), (6.5, 6.8)], [3, 3], [0, 0]),  # nothing between them
    ],
)
def test_parts_end_at_lines_dropped_words_and_her_turns(
    times: list[tuple[float, float]], lines: list[int], parts: list[int]
) -> None:
    assert cut_words(words(NORMAL, *times, lines=lines), TURNS)["part"].tolist() == parts


@pytest.mark.parametrize(
    ("start", "end", "kept"),
    [
        (39.7, 39.8, True),  # ends exactly 200 ms before the widened interruption (40-70 s)
        (39.7, 39.801, False),
        (55.0, 55.5, False),
        (70.2, 70.5, True),
        (70.199, 70.5, False),
    ],
)
def test_interruptions_are_widened_and_guarded_like_her_turns(
    start: float, end: float, kept: bool
) -> None:
    row = decide(INTERRUPTED, start, end)
    assert row["kept"] == kept
    assert row["near_interruption"] != kept
    assert row["cause"] == (None if kept else "interruption")


def test_an_interruption_between_two_words_ends_a_part() -> None:
    table = cut_words(words(INTERRUPTED, (39.0, 39.3), (70.3, 70.6)), TURNS)
    assert table["part"].tolist() == [0, 1]
    zones = zone_table(TURNS, [INTERRUPTED])
    assert zones["kind"].tolist() == ["interviewer", "interruption", "interviewer"]
    assert zones[["start", "end"]].to_numpy().tolist() == [[1.0, 2.0], [40.0, 70.0], [100.0, 101.0]]


def test_cut_words_returns_every_word_in_segment_order_with_the_input_index() -> None:
    table = words(NORMAL, (12.5, 12.8), (2.5, 2.8), (19.0, 19.3))
    table.index = [7, 3, 5]
    decided = cut_words(table, TURNS)
    assert decided.index.tolist() == [3, 7, 5]
    assert decided["start"].tolist() == [2.5, 12.5, 19.0]
    assert {"kept", "part", "cause", *FLAGS} <= set(decided.columns)
    assert decided["part"].tolist() == [0, 1, 1]  # her turn (10-12 s) lies between the first two


def test_kept_is_exactly_no_flag_set() -> None:
    rng = np.random.default_rng(9001)
    starts = np.round(rng.uniform(0.0, 45.0, 400), 3)
    table = words(
        NORMAL, *zip(starts, np.round(starts + rng.uniform(0.0, 0.6, 400), 3), strict=True)
    )
    decided = cut_words(table, TURNS)
    assert (decided["kept"] == ~decided[list(FLAGS)].any(axis=1)).all()
    assert (decided["cause"].isna() == decided["kept"]).all()
    kept = decided[decided["kept"]]
    assert kept["part"].is_monotonic_increasing
    assert (decided.loc[~decided["kept"], "part"] == NO_PART).all()


def test_settings_that_contradict_the_transcripts_are_refused() -> None:
    with pytest.raises(ValueError, match="has interviewer turns"):
        session_turns(TRANSCRIPTS[NORMAL], WITHOUT, RULE)
    with pytest.raises(ValueError, match="has no interviewer turns"):
        session_turns(TRANSCRIPTS[WITHOUT], NORMAL, RULE)
    with pytest.raises(ValueError, match="unexpected speakers"):
        session_turns(transcript((1.0, 2.0, "Ellie", "a"), (2.0, 3.0, "Nurse", "b")), NORMAL, RULE)
    with pytest.raises(ValueError, match="no participant turns"):
        session_turns(transcript((1.0, 2.0, "Ellie", "a")), NORMAL, RULE)
    with pytest.raises(ValueError, match="without a transcript"):
        all_turns({NORMAL: TRANSCRIPTS[NORMAL]}, RULE)
    with pytest.raises(ValueError, match="no DAIC-WOZ turns"):
        cut_words(words(1234, (1.0, 2.0)), TURNS)
    with pytest.raises(ValueError, match="in order"):
        cut_words(words(NORMAL, (2.0, 1.0)), TURNS)


def test_rule_reads_the_pilot_settings() -> None:
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "pilot" / "data.yaml").read_text(encoding="utf-8")
    )
    rule = CutRule.from_config(config["daicwoz_cut"])
    assert (rule.guard_ms, rule.offset_guard_ms, rule.interruption_pad_ms) == (200, 400, 10_000)
    assert rule.offsets_ms == {318: 34_100, 321: 3_500, 341: 5_900, 362: 16_700}
    assert rule.interruptions_ms == {373: (352_000, 420_000), 444: (286_000, 387_000)}
    assert rule.no_interviewer_turns == {451, 458, 480}
    assert rule.interview_only
    assert rule.guard_for(318) == 400
    assert rule.guard_for(300) == 200
    for marker in ("sync", "<sync>", "[SYNCH]", "synching", " syncing. "):
        assert rule.not_turns.search(marker)
    for text in ("sync it", "in sync", "synchronise"):
        assert not rule.not_turns.search(text)


def test_read_turns_can_leave_out_blank_rows_and_markers() -> None:
    data = TRANSCRIPTS[NORMAL]
    assert len(read_turns(data)) == 13
    turns = read_turns(data, re.compile(SETTINGS["not_turns"], re.IGNORECASE))
    assert len(turns) == 10
    assert list(turns.columns) == ["start", "stop", "speaker"]
    with pytest.raises(ValueError, match="no value column"):
        read_turns(b"start_time\tstop_time\tspeaker\n1\t2\tEllie\n", re.compile("x"))
    no_text = b"start_time\tstop_time\tspeaker\tvalue\n1\t2\tEllie\n3\t4\tParticipant\thello\n"
    assert read_turns(no_text, re.compile("x"))["speaker"].tolist() == ["participant"]


def test_times_are_compared_in_whole_milliseconds() -> None:
    assert to_ms([0.1, 34.1, 12.345, 1e-4]).tolist() == [100, 34_100, 12_345, 0]
    assert decide(NORMAL, 12.2 + 1e-9, 12.5)["kept"]
    assert decide(NORMAL, 12.2 - 1e-9, 12.5)["kept"]  # rounds to 12.200 s


def random_session(seed: int) -> tuple[bytes, pd.DataFrame]:
    """Turns that alternate, her turns sometimes overlapping the participant's, and words throughout."""
    rng = np.random.default_rng(seed)
    rows = [(0.5, 1.0, "Ellie", "hi")]
    t = 1.2
    while t < 290.0:
        stop = t + rng.uniform(2.0, 8.0)
        rows.append((round(t, 3), round(stop, 3), "Participant", "words"))
        her_start = stop + rng.uniform(-0.3, 0.5)
        her_stop = her_start + rng.uniform(0.2, 3.0)
        rows.append((round(her_start, 3), round(her_stop, 3), "Ellie", "question"))
        t = her_stop + rng.uniform(0.0, 1.0)
    rows.append((299.0, 300.0, "Ellie", "bye"))
    word_rows = []
    t, index = 0.6, 0
    while t < 305.0:
        end = t + rng.uniform(0.1, 0.3)
        line = int(t // 5)
        word_rows.append(
            {
                "session": NORMAL,
                "line": line,
                "word_index": index,
                "word": f"w{index}",
                "start": round(t, 3),
                "end": round(end, 3),
            }
        )
        index += 1
        t = end + rng.uniform(0.02, 0.3)
    return transcript(*rows), pd.DataFrame(word_rows)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_windows_stop_halfway_to_every_other_word_and_short_of_her_turns(seed: int) -> None:
    data, table = random_session(seed)
    turns = {NORMAL: session_turns(data, NORMAL, RULE)}
    decided = cut_words(table, turns)
    zones = zone_table(turns)
    n_words, margin, wall_guard, duration = 10, 0.1, 0.02, 310.0
    segments = build_segments(decided, n_words=n_words)
    segments = segments.merge(segment_neighbours(decided, n_words=n_words), on="segment")
    segments = segments.merge(segment_walls(segments, zones), on="segment")
    assert 0 < decided["kept"].mean() < 1
    assert segments["parts"].max() > 1
    starts, ends = decided["start"].to_numpy(), decided["end"].to_numpy()
    all_windows = []
    for row in segments.itertuples():
        windows = part_windows(
            row.starts,
            row.ends,
            duration,
            margin=margin,
            before=row.before,
            after=row.after,
            others=list(zip(row.other_starts, row.other_ends, strict=True)),
            walls=list(zip(row.wall_starts, row.wall_ends, strict=True)),
            wall_guard=wall_guard,
        )
        for (low, high), start, end in zip(windows, row.starts, row.ends, strict=True):
            for zone_start, zone_end in zones[["start", "end"]].to_numpy():
                assert high <= zone_start - wall_guard + 1e-9 or low >= zone_end + wall_guard - 1e-9
            earlier, later = ends[ends <= start], starts[starts >= end]
            if len(earlier):
                assert low >= (earlier.max() + start) / 2 - 1e-9
            if len(later):
                assert high <= (end + later.min()) / 2 + 1e-9
            inside = (starts >= start) & (ends <= end)
            assert decided.loc[inside, "kept"].all()  # no dropped word inside a part
            assert decided.loc[inside, "part"].nunique() == 1
        all_windows += windows
    flat = np.array(sorted(all_windows))
    assert np.all(flat[1:, 0] >= flat[:-1, 1])  # no two windows share audio


def test_each_session_is_cut_with_its_own_turns() -> None:
    both = pd.concat(
        [
            words(OFFSET, (12.6, 13.0), (14.9, 15.2)),
            words(NORMAL, (12.6, 13.0), (14.9, 15.2)),
        ],
        ignore_index=True,
    )
    decided = cut_words(both, TURNS)
    assert decided.groupby("session")["kept"].apply(list).to_dict() == {
        NORMAL: [True, True],
        OFFSET: [False, True],
    }
    assert decided.groupby("session")["part"].apply(list).to_dict() == {
        NORMAL: [0, 0],
        OFFSET: [NO_PART, 0],
    }


def test_interview_limits_can_be_switched_off() -> None:
    rule = CutRule.from_config({**SETTINGS, "interview_only": False})
    turns = {NORMAL: session_turns(TRANSCRIPTS[NORMAL], NORMAL, rule)}
    assert turns[NORMAL].interview is None
    decided = cut_words(words(NORMAL, (0.2, 0.5), (32.0, 32.5)), turns)
    assert decided["kept"].tolist() == [True, True]


def test_bad_settings_and_bad_words_are_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        CutRule.from_config({**SETTINGS, "guard_ms": -1})
    with pytest.raises(ValueError, match="narrower"):
        CutRule.from_config({**SETTINGS, "guard_ms": 500})  # a fallback that widened one guard only
    with pytest.raises(ValueError, match="before it starts"):
        CutRule.from_config({**SETTINGS, "interruptions_s": {INTERRUPTED: [60.0, 50.0]}})
    table = words(NORMAL, (2.5, 2.8), (3.0, 3.2))
    with pytest.raises(ValueError, match="unique index"):
        cut_words(table.set_axis([0, 0]), TURNS)
    with pytest.raises(ValueError, match="in order"):
        cut_words(table.assign(end=[2.8, np.nan]), TURNS)
    fallback = CutRule.from_config({**SETTINGS, "guard_ms": 500, "offset_guard_ms": 500})
    assert fallback.guard_for(OFFSET) == fallback.guard_for(NORMAL) == 500
