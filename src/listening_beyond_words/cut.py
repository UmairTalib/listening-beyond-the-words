"""The cut rule: which aligned words stay in the speech, and where a part of a segment ends.

Pilot protocol, section "Interviewer detector", cut rule fixed on 2026-11-18; its settings
are in ``configs/pilot/data.yaml`` (``daicwoz_cut``). The DAIC-WOZ transcripts give the
times of the interviewer's and the participant's turns. A turn is a transcript row with
valid times whose text is neither blank nor only a sync marker.

A word is kept only if all of these hold:

- its middle lies inside one of the participant's turns (ends included);
- it does not come within the guard of an interviewer turn or of a widened interruption
  (a word from s to e comes within g of a stretch from a to b if s < b + g and e > a - g);
- it lies within the interview: it does not begin before her first turn starts or end
  after her last turn ends (not in the sessions without interviewer turns).

The four sessions whose transcript times are offset from the recording are shifted by
their offset first, and the guard there is wider. All times are compared in whole
milliseconds.

A part is a run of kept words, in time order, from one transcript line, with no dropped
word and no interviewer turn or interruption between them. In the sessions without
interviewer turns, a part also ends where one participant turn ends and the next begins.
"""

import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from listening_beyond_words.daicwoz import merge_intervals, read_turns
from listening_beyond_words.segments import in_segment_order

INTERVIEWER: Final = "ellie"
PARTICIPANT: Final = "participant"
FLAGS: Final = ("near_interviewer", "outside_participant", "outside_interview", "near_interruption")
# Why a word was dropped, one cause per word: the first of FLAGS that holds, in this order.
CAUSES: Final = dict(
    zip(
        FLAGS,
        ("interviewer turn", "outside participant turns", "interview limits", "interruption"),
        strict=True,
    )
)
NO_PART: Final = -1


def to_ms(seconds: Any) -> np.ndarray:
    """Round times in seconds to whole milliseconds.

    Args:
        seconds: A time or an array of times in seconds.

    Returns:
        The times as int64 milliseconds (halves round to even, as numpy does).
    """
    return np.rint(np.asarray(seconds, dtype=float) * 1000.0).astype(np.int64)


@dataclass(frozen=True, slots=True)
class CutRule:
    """Settings of the cut rule, from the ``daicwoz_cut`` entry of ``configs/pilot/data.yaml``.

    Attributes:
        not_turns: Rows whose text matches this (any case), or is blank, are not turns.
        guard_ms: Distance in ms a kept word keeps from interviewer turns and interruptions.
        offset_guard_ms: The guard in the sessions with an offset.
        interview_only: Whether words outside the interview are dropped.
        no_interviewer_turns: Sessions without interviewer turns in the transcripts.
        offsets_ms: Offset in ms added to every transcript time of these sessions.
        interruptions_ms: Documented interruptions, start and end in ms of the recording.
        interruption_pad_ms: Widening in ms on each side of an interruption.
    """

    not_turns: re.Pattern[str]
    guard_ms: int
    offset_guard_ms: int
    interview_only: bool
    no_interviewer_turns: frozenset[int]
    offsets_ms: Mapping[int, int]
    interruptions_ms: Mapping[int, tuple[int, int]]
    interruption_pad_ms: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CutRule":
        """Read the rule from the ``daicwoz_cut`` entry of the data config.

        Args:
            config: The ``daicwoz_cut`` mapping as loaded from YAML.

        Returns:
            The rule.

        Raises:
            ValueError: If a guard or the widening is negative, the guard of the offset
                sessions is narrower than the guard elsewhere (the fallback widens both),
                or an interruption ends before it starts.
        """
        interruptions = {
            int(session): (int(to_ms(window[0])), int(to_ms(window[1])))
            for session, window in (config.get("interruptions_s") or {}).items()
        }
        rule = cls(
            not_turns=re.compile(str(config["not_turns"]), re.IGNORECASE),
            guard_ms=int(config["guard_ms"]),
            offset_guard_ms=int(config["offset_guard_ms"]),
            interview_only=bool(config["interview_only"]),
            no_interviewer_turns=frozenset(int(s) for s in config["no_interviewer_turns"]),
            offsets_ms={
                int(session): int(to_ms(offset))
                for session, offset in (config.get("offsets_s") or {}).items()
            },
            interruptions_ms=interruptions,
            interruption_pad_ms=int(to_ms(config["interruption_pad_s"])),
        )
        if min(rule.guard_ms, rule.offset_guard_ms, rule.interruption_pad_ms) < 0:
            raise ValueError("the guards and the widening of interruptions must not be negative")
        if rule.offset_guard_ms < rule.guard_ms:
            raise ValueError("the guard of the offset sessions must not be narrower than the guard")
        if any(start > end for start, end in interruptions.values()):
            raise ValueError(f"an interruption ends before it starts: {interruptions}")
        return rule

    def guard_for(self, session: int) -> int:
        """Return the guard in ms that applies in one session."""
        return self.offset_guard_ms if session in self.offsets_ms else self.guard_ms


@dataclass(frozen=True, slots=True)
class Turns:
    """The turns of one session, in whole ms of the recording, after its offset.

    Attributes:
        participant: Participant turns, merged where they touch or overlap: shape (n, 2).
        interviewer: Interviewer turns as they are in the transcript: shape (m, 2).
        interruptions: Interruptions, widened on each side: shape (k, 2).
        guard_ms: The guard that applies in this session.
        interview: Start of her first turn and end of her last turn; None where the
            interview-only condition does not apply.
        split_turns: Whether a part ends between two participant turns (the sessions
            without interviewer turns).
    """

    participant: np.ndarray
    interviewer: np.ndarray
    interruptions: np.ndarray
    guard_ms: int
    interview: tuple[int, int] | None
    split_turns: bool

    @property
    def zones(self) -> np.ndarray:
        """Interviewer turns and widened interruptions together: shape (m + k, 2)."""
        return np.concatenate([self.interviewer, self.interruptions]).reshape(-1, 2)


def _pairs(starts: Iterable[int], stops: Iterable[int]) -> np.ndarray:
    """Pairs of times as an int64 array of shape (n, 2), sorted by start then end."""
    pairs = np.column_stack([np.fromiter(starts, np.int64), np.fromiter(stops, np.int64)])
    pairs = pairs.reshape(-1, 2)
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


def session_turns(data: bytes, session: int, rule: CutRule) -> Turns:
    """Read one session's turns from its DAIC-WOZ transcript and apply offset and widening.

    Args:
        data: The bytes of the session's DAIC-WOZ transcript.
        session: The session ID.
        rule: The cut rule.

    Returns:
        The session's turns.

    Raises:
        ValueError: If a row names another speaker, the session has no participant turns,
            or it has interviewer turns exactly when the rule says it has none.
    """
    table = read_turns(data, rule.not_turns)
    others = sorted(set(table["speaker"]) - {INTERVIEWER, PARTICIPANT})
    if others:
        raise ValueError(f"session {session}: unexpected speakers {others}")
    offset = rule.offsets_ms.get(session, 0)
    starts = to_ms(table["start"].to_numpy()) + offset
    stops = to_ms(table["stop"].to_numpy()) + offset
    is_participant = (table["speaker"] == PARTICIPANT).to_numpy()
    if not is_participant.any():
        raise ValueError(f"session {session}: no participant turns")
    merged_starts, merged_stops = merge_intervals(starts[is_participant], stops[is_participant])
    participant = _pairs(merged_starts.astype(np.int64), merged_stops.astype(np.int64))
    interviewer = _pairs(starts[~is_participant], stops[~is_participant])
    without = session in rule.no_interviewer_turns
    if without == bool(len(interviewer)):
        state = "has" if len(interviewer) else "has no"
        raise ValueError(f"session {session} {state} interviewer turns, against the cut settings")
    pad = rule.interruption_pad_ms
    window = rule.interruptions_ms.get(session)
    interruptions = _pairs([window[0] - pad] if window else [], [window[1] + pad] if window else [])
    interview = None
    if rule.interview_only and not without:
        interview = (int(interviewer[:, 0].min()), int(interviewer[:, 1].max()))
    return Turns(
        participant=participant,
        interviewer=interviewer,
        interruptions=interruptions,
        guard_ms=rule.guard_for(session),
        interview=interview,
        split_turns=without,
    )


def all_turns(transcripts: Mapping[int, bytes], rule: CutRule) -> dict[int, Turns]:
    """Read the turns of every session.

    Args:
        transcripts: Each session's DAIC-WOZ transcript bytes.
        rule: The cut rule.

    Returns:
        Each session's turns.

    Raises:
        ValueError: If a session of the rule's offsets, interruptions or sessions without
            interviewer turns has no transcript.
    """
    named = set(rule.offsets_ms) | set(rule.interruptions_ms) | set(rule.no_interviewer_turns)
    missing = sorted(named - set(transcripts))
    if missing:
        raise ValueError(f"the cut settings name sessions without a transcript: {missing}")
    return {session: session_turns(data, session, rule) for session, data in transcripts.items()}


def _near(starts: np.ndarray, ends: np.ndarray, stretches: np.ndarray, guard: int) -> np.ndarray:
    """Whether each word comes within ``guard`` ms of any stretch: s < b + g and e > a - g."""
    if len(stretches) == 0:
        return np.zeros(len(starts), dtype=bool)
    near = (starts[:, None] < stretches[None, :, 1] + guard) & (
        ends[:, None] > stretches[None, :, 0] - guard
    )
    return np.asarray(near.any(axis=1), dtype=bool)


def _turn_of(starts: np.ndarray, ends: np.ndarray, turns: np.ndarray) -> np.ndarray:
    """Index of the merged turn that holds each word's middle (ends included), or -1."""
    doubled = starts + ends  # twice the middle, so that it stays a whole number
    index = np.searchsorted(2 * turns[:, 0], doubled, side="right") - 1
    found = index >= 0
    found[found] = doubled[found] <= 2 * turns[index[found], 1]
    return np.where(found, index, -1)


def decide_session(words: pd.DataFrame, turns: Turns) -> pd.DataFrame:
    """Decide for every word of one session whether it is kept, and give each kept word its part.

    Args:
        words: The session's words in time order (by start, then word index), with the
            columns ``line``, ``start`` and ``end`` (seconds in the recording).
        turns: The session's turns.

    Returns:
        A table with the index of ``words`` and the columns ``kept``, ``part`` (a counter
        of the session's parts, NO_PART for dropped words), ``cause`` (None for kept words)
        and one boolean column per entry of FLAGS.
    """
    starts, ends = to_ms(words["start"].to_numpy()), to_ms(words["end"].to_numpy())
    guard = turns.guard_ms
    turn = _turn_of(starts, ends, turns.participant)
    flags = {
        "near_interviewer": _near(starts, ends, turns.interviewer, guard),
        "outside_participant": turn < 0,
        "outside_interview": (
            (starts < turns.interview[0]) | (ends > turns.interview[1])
            if turns.interview is not None
            else np.zeros(len(words), dtype=bool)
        ),
        "near_interruption": _near(starts, ends, turns.interruptions, guard),
    }
    dropped = np.zeros(len(words), dtype=bool)
    cause = np.full(len(words), None, dtype=object)
    for name in FLAGS:
        cause[flags[name] & ~dropped] = CAUSES[name]
        dropped |= flags[name]
    kept = ~dropped

    part = np.full(len(words), NO_PART, dtype=np.int64)
    position = np.flatnonzero(kept)
    if len(position):
        new = np.ones(len(position), dtype=bool)
        previous, current = position[:-1], position[1:]
        lines = words["line"].to_numpy()
        zones = turns.zones
        zone_between = (
            (
                (zones[None, :, 0] < starts[current][:, None])
                & (zones[None, :, 1] > ends[previous][:, None])
            ).any(axis=1)
            if len(zones)
            else np.zeros(len(current), dtype=bool)
        )
        new[1:] = (
            (lines[current] != lines[previous])
            | (current - previous > 1)  # a dropped word lies between them
            | zone_between
            | (turns.split_turns & (turn[current] != turn[previous]))
        )
        part[position] = np.cumsum(new) - 1
    return pd.DataFrame({"kept": kept, "part": part, "cause": cause, **flags}, index=words.index)


def cut_words(words: pd.DataFrame, turns: Mapping[int, Turns]) -> pd.DataFrame:
    """Apply the cut rule to every aligned word.

    Args:
        words: One row per aligned word, with the columns ``session``, ``line``,
            ``word_index``, ``start`` and ``end`` (seconds in the recording).
        turns: Each session's turns, from ``all_turns``.

    Returns:
        The words in segment order (by session, then time), with the columns of
        ``decide_session`` added.

    Raises:
        ValueError: If a session of ``words`` has no turns, a word ends before it starts,
            or the index of ``words`` has repeats.
    """
    if not words.index.is_unique:
        raise ValueError("the words need a unique index")
    missing = sorted(set(words["session"].astype(int)) - set(turns))
    if missing:
        raise ValueError(f"no DAIC-WOZ turns for sessions {missing}")
    if (words["end"] < words["start"]).any() or words[["start", "end"]].isna().any().any():
        raise ValueError("every word needs a start and an end, in order")
    ordered = in_segment_order(words)
    decided = [
        decide_session(group, turns[int(cast(int, session))])
        for session, group in ordered.groupby("session", sort=True)
    ]
    columns = ["kept", "part", "cause", *FLAGS]
    if not decided:
        return ordered.assign(**{c: pd.Series(dtype=object) for c in columns})
    return ordered.join(pd.concat(decided))


def zone_table(turns: Mapping[int, Turns], sessions: Collection[int] | None = None) -> pd.DataFrame:
    """List the interviewer turns and widened interruptions of each session.

    Args:
        turns: Each session's turns, from ``all_turns``.
        sessions: The sessions to list; None lists every session.

    Returns:
        One row per zone, sorted by session and time, with the columns ``session``,
        ``start`` and ``end`` (seconds in the recording) and ``kind`` (``interviewer``
        or ``interruption``).
    """
    rows = []
    for session in sorted(turns if sessions is None else sessions):
        for kind, stretches in (
            ("interviewer", turns[session].interviewer),
            ("interruption", turns[session].interruptions),
        ):
            rows += [
                {"session": session, "start": a / 1000, "end": b / 1000, "kind": kind}
                for a, b in stretches.tolist()
            ]
    table = pd.DataFrame(rows, columns=["session", "start", "end", "kind"])
    return table.sort_values(["session", "start", "end"], ignore_index=True)
