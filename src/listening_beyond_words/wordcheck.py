"""The word check: check F4 and the checks of the re-voiced clip x1, apart from the voice and the recognisers.

Pilot protocol, check F4 and section "Clip versions", bullet "Word check"; the settings
are in ``configs/pilot/data.yaml`` (``word_check``). This module holds:

- the sample (``draw_sample``): five segments per participant, drawn at random, or all
  of a participant's segments if there are fewer; it is drawn and pinned before any clip
  is transcribed;
- the text rule (``words``), the one rule on both sides of every WER and for the leakage
  classifier, and counts of what it changes (``text_counts``); the word errors
  (``wer_counts``) and their text-free word classes (``reference_classes``,
  ``error_classes``); the chance pairing of the words heard (``chance_partners``) and the
  share of repeated word triples (``repetition_share``);
- the leakage classifiers (``fit_leakage``, ``person_scores``, ``fold_aurocs``) and the
  interval of the x0 control (``bootstrap_sets``, ``bootstrap_interval``);
- the verdicts of F4, the x0 control and the two rules of x1 (``Rules``, ``verdicts``).

The voice measures, the ICC and the probe are in ``voicecheck.py``, the two recognisers in
``recognisers.py``. The pair wins, the AUROCs weighted by draw counts and the percentile
interval come from ``evaluation.py``, which the analysis of the test scores shares.

Nothing here prints or stores participant text: errors name rules, counts and segment IDs
only, and the classes and counts hold no words. The fitted classifiers hold participant
words in their vocabulary and weights, so they are never printed or saved.
"""

import math
import re
import statistics
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, Final, NamedTuple

import jiwer
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from listening_beyond_words.evaluation import interval, pair_wins, weighted_aurocs
from listening_beyond_words.revoice import part_text, spoken_number, word_classes
from listening_beyond_words.runinfo import StepError, item_seed
from listening_beyond_words.transcripts import normalise

# Text: one rule on both sides of every WER and for the leakage classifier (pilot protocol,
# bullet "Word check"). A hesitation, however it is spelled: "um", "uhh", "mm", "hmm",
# "ah", "er", "erm"; never "am", "a", "oh", "huh", "eh", "hum", "him" or "her".
HESITATION: Final = re.compile(r"u*[hm]+|e+r+m*|a+h+")
ALIASES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "ok": ("okay",),
        "alright": ("all", "right"),
        "gonna": ("going", "to"),
        "wanna": ("want", "to"),
        "gotta": ("got", "to"),
    }
)
TEXT_COUNTS: Final = (  # what text_counts reports, never the words themselves
    "words",
    "words_with_a_digit",
    "four_digit_runs",
    "mixed_words",
    "one_letter_words",
    "letter_runs_joined",
    "hesitations_dropped",
    "aliases_merged",
)
VERY_REPETITIVE: Final = 0.5  # an output is very repetitive above this repetition_share
CHANCE: Final = 0.5  # the AUROC of a classifier that knows nothing
RECOGNISERS: Final = ("whisper", "wav2vec2")
BOOTSTRAP_KEYS: Final = ("wordcheck", "bootstrap")  # the interval's stream: item_seed(seed, *keys)

# Text-free classes of a reference word (pilot protocol, bullet "Word check"): from its
# stored word and its part, and from its transcript line (LINE_CLASSES, ``line_classes``).
LINE_CLASSES: Final = ("capitals", "apostrophe", "read_as_stored", "split_edge")
WORD_CLASSES: Final = (
    "year",  # four digits, which x1 reads as a year (rule B)
    "number",  # any other run of digits only, which x1 reads as a whole number (rule B)
    "mixed",  # digits and letters in one stored word, such as "1990s"
    "one_letter",
    "part_edge",  # the first or last word of its part
    "short_part",  # a word in a part of 1-2 words
    *LINE_CLASSES,
)
SHORT_PART_WORDS: Final = 2

PASS: Final = "PASS"
FAIL: Final = "FAIL"
WAITS: Final = "WAITS FOR THE RERUN"
RERUN_NEEDED: Final = "RERUN NEEDED"
GATE_ICCS: Final = 6  # the three voice measures within each of the two genders

_RUNS: Final = re.compile(r"[0-9]+|[a-z]+")


# The sample --------------------------------------------------------------------------


def draw_sample(
    segments: pd.DataFrame, rng: np.random.Generator, per_participant: int
) -> pd.DataFrame:
    """Draw the segments that the word check transcribes.

    Participants are taken in ascending ID, and each participant's segments in ascending
    ID, so the draw depends only on the seed, never on the order of the rows.

    Args:
        segments: One row per segment with the columns ``segment`` and ``session``.
        rng: The generator, seeded from the config.
        per_participant: Segments drawn per participant, without replacement; all of a
            participant's segments if there are this many or fewer.

    Returns:
        The drawn rows with the columns ``segment`` and ``session``, sorted by segment.

    Raises:
        ValueError: If the count is not positive or a segment appears twice.
    """
    if per_participant < 1:
        raise ValueError(f"draw at least one segment per participant, not {per_participant}")
    if segments["segment"].duplicated().any():
        raise ValueError("a segment appears twice")
    table = segments[["segment", "session"]].sort_values(["session", "segment"], ignore_index=True)
    drawn: list[pd.DataFrame] = []
    for _, group in table.groupby("session", sort=True):
        if len(group) <= per_participant:
            drawn.append(group)
        else:
            chosen = rng.choice(len(group), size=per_participant, replace=False)
            drawn.append(group.iloc[np.sort(chosen)])
    return pd.concat(drawn).sort_values("segment", ignore_index=True)


# Text --------------------------------------------------------------------------------


class _Word(NamedTuple):
    """A word of ``words`` with the first and last stored word it comes from (positions)."""

    word: str
    first: int
    last: int


def _number_words(digits: str) -> list[str]:
    """A run of digits as x1 reads it (``revoice.spoken_number``), digit by digit if refused."""
    try:
        spelled = spoken_number(digits)
    except ValueError:  # num2words refuses very long numbers, such as 307 or more digits
        spelled = " ".join(spoken_number(digit) for digit in digits)
    return spelled.split()


def _spoken(stored: Sequence[str]) -> list[_Word]:
    """Normalise each word, split it into runs of letters and of digits, and read the digits."""
    if isinstance(stored, str):
        raise TypeError("expected a list of words, not a text")
    found: list[_Word] = []
    for position, word in enumerate(stored):
        for piece in normalise(word).split():
            for run in _RUNS.findall(piece):
                spoken = _number_words(run) if run.isdigit() else [run]
                found.extend(_Word(item, position, position) for item in spoken)
    return found


def _joined(found: Sequence[_Word], counts: dict[str, int] | None = None) -> list[_Word]:
    """Join each run of two or more one-letter words into one word; count the runs joined."""
    result: list[_Word] = []
    letters: list[_Word] = []
    for item in [*found, None]:
        if item is not None and len(item.word) == 1:
            letters.append(item)
            continue
        if len(letters) >= 2:
            result.append(
                _Word(
                    "".join(letter.word for letter in letters), letters[0].first, letters[-1].last
                )
            )
            if counts is not None:
                counts["letter_runs_joined"] += 1
        else:
            result.extend(letters)
        letters = []
        if item is not None:
            result.append(item)
    return result


def _settled(found: list[_Word], counts: dict[str, int] | None = None) -> list[_Word]:
    """Join letters and drop hesitations until nothing changes; then merge the spellings.

    Joining comes first, so the "h" of "A D H D" is kept. Repeating both steps keeps
    ``words`` idempotent, which one pass in either order is not: one pass of join, then
    drop, leaves "o k" for "o ah k". With ``counts``, the runs joined, the hesitations
    dropped and the spellings merged are added to it (``text_counts``).
    """
    while True:
        joined = _joined(found, counts)
        kept = [item for item in joined if not HESITATION.fullmatch(item.word)]
        if counts is not None:
            counts["hesitations_dropped"] += len(joined) - len(kept)
        if kept == found:  # every change shortens the list, so this ends
            break
        found = kept
    if counts is not None:
        counts["aliases_merged"] += sum(item.word in ALIASES for item in found)
    return [
        _Word(alias, item.first, item.last)
        for item in found
        for alias in ALIASES.get(item.word, (item.word,))
    ]


def join_letters(words: Sequence[str]) -> list[str]:
    """Join each run of two or more one-letter words into one word ("p t s d" as "ptsd").

    Args:
        words: Words of letters a-z.

    Returns:
        The words, each run of one-letter words joined.
    """
    return [item.word for item in _joined([_Word(word, 0, 0) for word in words])]


def words(text: str) -> list[str]:
    """Return the words of a text as the word check compares them.

    The one rule for references, recogniser outputs and the leakage classifier (pilot
    protocol, bullet "Word check"): ``transcripts.normalise``; each word split into runs
    of letters and of digits; each run of digits as x1 reads it (``revoice.spoken_number``:
    four digits as a year, other numbers whole; digit by digit if num2words refuses the
    run); then, until nothing changes, each run of two or more one-letter words joined
    into one word and every hesitation dropped (``HESITATION``); then the spellings of
    ``ALIASES`` merged ("ok" as "okay"). The result holds only words of letters a-z, and
    the rule is idempotent: ``words(" ".join(words(text))) == words(text)``.

    Args:
        text: Any text, such as a segment's stored words or a recogniser's output.

    Returns:
        The words, possibly none.
    """
    return [item.word for item in _settled(_spoken(normalise(text).split()))]


def words_with_source(stored: Sequence[str]) -> tuple[list[str], list[int]]:
    """Return the words of stored words, each with the position of the stored word it comes from.

    The words equal ``words(" ".join(stored))``. A word joined from one-letter words
    comes from its first letter; a number's words and a merged spelling's words come from
    their stored word.

    Args:
        stored: A segment's stored words, in clip order.

    Returns:
        The words, and for each the position in ``stored`` of the word it comes from.

    Raises:
        TypeError: If ``stored`` is one text rather than a list of words.
    """
    found = _settled(_spoken(stored))
    return [item.word for item in found], [item.first for item in found]


def text_counts(texts: Iterable[str]) -> dict[str, int]:
    """Count what the text rule meets and changes in some texts, without their words.

    Reported only (pilot protocol, bullet "Word check"): for the references, how many
    numbers and letters the rule reads; for each view and recogniser, how much of the
    output it changes. The counts, in the order of ``TEXT_COUNTS``:

    - ``words``: the words after ``transcripts.normalise``;
    - ``words_with_a_digit``, ``four_digit_runs`` (runs of exactly four digits, which x1
      reads as years) and ``mixed_words`` (digits and letters in one word, such as
      "1990s"), among those words;
    - ``one_letter_words``: one-letter words once the digits are written out, before any
      letters are joined;
    - ``letter_runs_joined``, ``hesitations_dropped`` and ``aliases_merged``: what
      ``words`` joins, drops and merges.

    Args:
        texts: Texts, such as the segments' stored words or a recogniser's outputs.

    Returns:
        Each count, by name.

    Raises:
        TypeError: If an item is not a text.
    """
    counts = dict.fromkeys(TEXT_COUNTS, 0)
    for text in texts:
        if not isinstance(text, str):
            raise TypeError("text_counts takes texts")
        tokens = normalise(text).split()
        counts["words"] += len(tokens)
        for token in tokens:
            runs = _RUNS.findall(token)
            digits = [run for run in runs if run.isdigit()]
            counts["words_with_a_digit"] += bool(digits)
            counts["four_digit_runs"] += sum(len(run) == 4 for run in digits)
            counts["mixed_words"] += bool(digits) and len(digits) < len(runs)
        found = _spoken(tokens)
        counts["one_letter_words"] += sum(len(item.word) == 1 for item in found)
        _settled(found, counts)
    return counts


def repetition_share(words: Sequence[str]) -> float:
    """Return the share of an output's word triples that already occurred in it.

    A recogniser that loops ("thank you thank you ...") repeats most of its triples; an
    output above ``VERY_REPETITIVE`` counts as very repetitive (reported only).

    Args:
        words: The output's words, from ``words``.

    Returns:
        Repeated triples over all triples; 0 for fewer than three words.
    """
    if len(words) < 3:
        return 0.0
    seen: set[tuple[str, str, str]] = set()
    repeated = 0
    for start in range(len(words) - 2):
        triple = (words[start], words[start + 1], words[start + 2])
        if triple in seen:
            repeated += 1
        else:
            seen.add(triple)
    return repeated / (len(words) - 2)


def chance_partners(sample: pd.DataFrame) -> np.ndarray:
    """Pair every sampled segment with another participant's segment, for the chance level of words heard.

    Each output is also scored against the words of the paired segment, which it cannot
    have heard. Participants are taken in ascending ID; participant i is paired with the
    one ⌊P/2⌋ places on, wrapping round (94 of 189); its k-th sampled segment, in ID
    order, with that participant's segment k mod n, where n is its number of sampled
    segments. The pairing is fixed and the same for every view and recogniser.

    Args:
        sample: One row per sampled segment, with the columns ``segment`` and ``session``.

    Returns:
        For each row, the position of its partner's row, of shape (len(sample),).

    Raises:
        ValueError: If a segment appears twice or the sample has fewer than two participants.
    """
    if sample["segment"].duplicated().any():
        raise ValueError("a segment appears twice")
    table = pd.DataFrame(
        {
            "segment": sample["segment"].astype(str).to_numpy(),
            "session": sample["session"].to_numpy(dtype=np.int64),
            "row": np.arange(len(sample)),
        }
    ).sort_values(["session", "segment"])
    rows: dict[int, list[int]] = {}
    for session, row in zip(
        table["session"].to_numpy().tolist(), table["row"].to_numpy().tolist(), strict=True
    ):
        rows.setdefault(session, []).append(row)
    sessions = sorted(rows)
    if len(sessions) < 2:
        raise ValueError("the chance level needs at least two participants")
    offset = len(sessions) // 2
    partners = np.empty(len(sample), dtype=np.int64)
    for index, session in enumerate(sessions):
        theirs = rows[sessions[(index + offset) % len(sessions)]]
        for k, row in enumerate(rows[session]):
            partners[row] = theirs[k % len(theirs)]
    return partners


# Word errors -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WerCounts:
    """The word errors of one view and recogniser, over all its segments at once (jiwer 4.0.0).

    Attributes:
        hits: Reference words heard correctly.
        substitutions: Reference words heard as another word.
        deletions: Reference words not heard.
        insertions: Output words with no reference word.
        reference_words: All reference words (N).
        output_words: All output words.
        per_segment: One row per segment, in the order given: ``segment``,
            ``reference_words``, ``output_words``, ``hits``, ``substitutions``,
            ``deletions``, ``insertions`` and ``wer``.
        alignments: jiwer's alignment of each segment, in the same order: positions
            only, no words.
    """

    hits: int
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    output_words: int
    per_segment: pd.DataFrame
    alignments: tuple[tuple[jiwer.AlignmentChunk, ...], ...]

    @property
    def errors(self) -> int:
        """Substitutions, deletions and insertions."""
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        """(S + D + I) / N over all words at once; inserted words can push it above 1."""
        return self.errors / self.reference_words

    @property
    def heard(self) -> float:
        """Words heard: hits / N."""
        return self.hits / self.reference_words

    def totals(self) -> dict[str, float]:
        """The totals as plain numbers, for ``run.json``."""
        return {
            "wer": self.wer,
            "heard": self.heard,
            "hits": self.hits,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "reference_words": self.reference_words,
            "output_words": self.output_words,
        }


def check_references(references: Sequence[Sequence[str]], segments: Sequence[str]) -> None:
    """Refuse a reference without words, before any clip is transcribed.

    jiwer 4.0.0 does not refuse one, although its docstring says so: it counts the
    output's words as insertions.

    Args:
        references: Each segment's reference words, from ``words``.
        segments: The segment IDs, in the same order.

    Raises:
        TypeError: If a reference is one text rather than a list of words.
        ValueError: If the lengths differ.
        StepError: If a reference has no words; the message names only the segment.
    """
    if len(references) != len(segments):
        raise ValueError(f"{len(references)} references for {len(segments)} segments")
    for segment, reference in zip(segments, references, strict=True):
        if isinstance(reference, str):
            raise TypeError("a reference must be a list of words, not a text")
        if len(reference) == 0:
            raise StepError(f"segment {segment}: the reference has no words", segment=str(segment))


def wer_counts(
    references: Sequence[Sequence[str]],
    hypotheses: Sequence[Sequence[str]],
    segments: Sequence[str],
) -> WerCounts:
    """Count the word errors of one view and recogniser over all its segments, in one jiwer call.

    The WER is (substitutions + deletions + insertions) / reference words over all words
    at once (pilot protocol, bullet "Word check"), not the mean of the segments' WERs.

    Args:
        references: Each segment's reference words, from ``words``.
        hypotheses: Each segment's output words, from ``words``, in the same order.
        segments: The segment IDs, in the same order.

    Returns:
        The totals, the counts per segment and the alignments.

    Raises:
        TypeError: If a reference or output is one text rather than a list of words.
        ValueError: If there are no segments, the lengths differ, a segment appears twice,
            or an alignment does not cover its words.
        StepError: If a reference has no words; the message names only the segment.
    """
    names = [str(segment) for segment in segments]
    if not names:
        raise ValueError("a WER needs at least one segment")
    if len(set(names)) != len(names):
        raise ValueError("a segment appears twice")
    if len(hypotheses) != len(names):
        raise ValueError(f"{len(hypotheses)} outputs for {len(names)} segments")
    if any(isinstance(hypothesis, str) for hypothesis in hypotheses):
        raise TypeError("an output must be a list of words, not a text")
    check_references(references, names)
    output = jiwer.process_words(
        [" ".join(reference) for reference in references],
        [" ".join(hypothesis) for hypothesis in hypotheses],
    )
    rows = []
    for segment, reference, hypothesis, chunks in zip(
        names, references, hypotheses, output.alignments, strict=True
    ):
        counts = dict.fromkeys(("equal", "substitute", "delete", "insert"), 0)
        for chunk in chunks:
            span = chunk.hyp_end_idx - chunk.hyp_start_idx
            if chunk.type != "insert":
                span = chunk.ref_end_idx - chunk.ref_start_idx
            counts[chunk.type] += span
        hits, subs, dels, ins = (counts[key] for key in ("equal", "substitute", "delete", "insert"))
        if hits + subs + dels != len(reference) or hits + subs + ins != len(hypothesis):
            raise ValueError(f"segment {segment}: the alignment does not cover its words")
        rows.append(
            {
                "segment": segment,
                "reference_words": len(reference),
                "output_words": len(hypothesis),
                "hits": hits,
                "substitutions": subs,
                "deletions": dels,
                "insertions": ins,
                "wer": (subs + dels + ins) / len(reference),
            }
        )
    table = pd.DataFrame(rows)
    return WerCounts(
        hits=int(table["hits"].sum()),
        substitutions=int(table["substitutions"].sum()),
        deletions=int(table["deletions"].sum()),
        insertions=int(table["insertions"].sum()),
        reference_words=int(table["reference_words"].sum()),
        output_words=int(table["output_words"].sum()),
        per_segment=table,
        alignments=tuple(tuple(chunks) for chunks in output.alignments),
    )


# Word classes ------------------------------------------------------------------------


def line_classes(line: str, first: int, last: int) -> pd.DataFrame:
    """Describe a part's stored words by their transcript line, without their text.

    As x1 reads them (``revoice.part_text``): whether a word is written in capitals
    (two letters or more) or with an apostrophe (``revoice.word_classes``), whether its
    token was read as its stored words because its written form does not turn back into
    them, and whether its token runs over a part edge, so that only its stored words in
    the part were read.

    Args:
        line: The text of one transcript line (``clips.read_lines``).
        first: Index of the part's first word in the line's stored words.
        last: Index of the part's last word.

    Returns:
        One row per stored word of the part, in order, with the boolean columns of
        ``LINE_CLASSES``.

    Raises:
        ValueError: If the indices do not lie in the line (``revoice.part_text``).
    """
    text = part_text(line, first, last)
    written = text.written_text.split()
    n_words = last - first + 1
    if len(written) != n_words:
        raise ValueError(f"a part of {n_words} stored words has {len(written)} written words")
    described = [word_classes(word, edge=False) for word in written]
    read_as_stored = [False] * n_words
    split_edge = [False] * n_words
    position = 0
    for token in line.split():  # the tokens of part_text, with their stored words
        start = position
        position += len(normalise(token).split())
        if position == start or position <= first or start > last:
            continue
        inside = range(max(first, start) - first, min(last + 1, position) - first)
        if start < first or position > last + 1:
            for k in inside:
                split_edge[k] = True
        elif part_text(line, start, position - 1).unmatched:
            for k in inside:
                read_as_stored[k] = True
    return pd.DataFrame(
        {
            "capitals": [bool(classes["capitals"]) for classes in described],
            "apostrophe": [bool(classes["apostrophe"]) for classes in described],
            "read_as_stored": read_as_stored,
            "split_edge": split_edge,
        },
        columns=list(LINE_CLASSES),
    )


def reference_classes(
    segment: str,
    stored: Sequence[str],
    part_words: Sequence[int],
    lines: pd.DataFrame,
) -> pd.DataFrame:
    """Describe each reference word of a segment by text-free classes.

    The reference words are ``words_with_source(stored)``, so they match the positions
    of ``wer_counts``. Each gets the classes of the stored words it comes from (all the
    letters of a joined run): from the stored word, ``year`` and ``number`` (digits only,
    four of them or another count, written out by rule B), ``mixed`` (digits and
    letters) and ``one_letter``; from its part, ``part_edge`` (first or last word) and
    ``short_part`` (a part of 1-2 words); from its transcript line, the columns of
    ``lines`` (``line_classes``, part after part).

    Args:
        segment: The segment ID.
        stored: The segment's stored words, in clip order.
        part_words: The number of stored words of each part, in clip order.
        lines: One row per stored word with the boolean columns of ``LINE_CLASSES``.

    Returns:
        One row per reference word: ``segment``, ``position`` (in the reference),
        ``source`` (the stored word it comes from) and a boolean column per class of
        ``WORD_CLASSES``.

    Raises:
        ValueError: If the parts or the line classes do not cover the stored words, or a
            column of ``LINE_CLASSES`` is missing.
    """
    n_words = len(stored)
    sizes = [int(size) for size in part_words]
    if sum(sizes) != n_words or any(size < 1 for size in sizes):
        raise ValueError(f"segment {segment}: parts of {sizes} words for {n_words} stored words")
    if len(lines) != n_words:
        raise ValueError(f"segment {segment}: line classes for {len(lines)} of {n_words} words")
    missing = [name for name in LINE_CLASSES if name not in lines.columns]
    if missing:
        raise ValueError(f"segment {segment}: the line classes lack {missing}")
    flags = np.zeros((n_words, len(WORD_CLASSES)), dtype=bool)
    column = {name: k for k, name in enumerate(WORD_CLASSES)}
    for k, word in enumerate(stored):
        plain = normalise(word)
        digits = plain.isascii() and plain.isdigit()
        flags[k, column["year"]] = digits and len(plain) == 4
        flags[k, column["number"]] = digits and len(plain) != 4
        flags[k, column["mixed"]] = any(c.isdigit() for c in plain) and any(
            c.isalpha() for c in plain
        )
        flags[k, column["one_letter"]] = len(plain) == 1 and plain.isalpha()
    start = 0
    for size in sizes:
        flags[start, column["part_edge"]] = flags[start + size - 1, column["part_edge"]] = True
        flags[start : start + size, column["short_part"]] = size <= SHORT_PART_WORDS
        start += size
    for name in LINE_CLASSES:
        flags[:, column[name]] = lines[name].to_numpy(dtype=bool)
    found = _settled(_spoken(stored))
    rows = np.array([flags[item.first : item.last + 1].any(axis=0) for item in found]).reshape(
        len(found), len(WORD_CLASSES)
    )
    table = pd.DataFrame(rows, columns=list(WORD_CLASSES))
    table.insert(0, "source", np.array([item.first for item in found], dtype=np.int64))
    table.insert(0, "position", np.arange(len(found), dtype=np.int64))
    table.insert(0, "segment", str(segment))
    return table


def error_classes(counts: WerCounts, classes: pd.DataFrame) -> pd.DataFrame:
    """Count the word errors by text-free class of the reference words.

    For each class: its reference words, the substitutions and deletions of its words,
    and the inserted words next to one of its words (on either side; an insertion next
    to two of its words counts once). The row ``all`` counts every reference word.

    Args:
        counts: The word errors of one view and recogniser (``wer_counts``).
        classes: The classes of the reference words of every segment in ``counts``
            (``reference_classes``, one table for all segments).

    Returns:
        Counts only, indexed by ``all`` and the classes of ``WORD_CLASSES``, with the
        columns ``reference_words``, ``substitutions``, ``deletions`` and ``insertions``.

    Raises:
        ValueError: If a segment's classes do not cover its reference words.
    """
    names = ["all", *WORD_CLASSES]
    totals = np.zeros((len(names), 4), dtype=np.int64)
    by_segment = {str(key): group for key, group in classes.groupby("segment", sort=False)}
    segments = counts.per_segment["segment"].astype(str).tolist()
    n_refs = counts.per_segment["reference_words"].astype(int).tolist()
    for segment, n_ref, chunks in zip(segments, n_refs, counts.alignments, strict=True):
        group = by_segment.get(segment)
        if group is None or len(group) != n_ref:
            raise ValueError(f"segment {segment}: the classes do not cover its reference words")
        ordered = group.sort_values("position")
        if not np.array_equal(ordered["position"].to_numpy(), np.arange(n_ref)):
            raise ValueError(f"segment {segment}: the classes do not cover its reference words")
        member = np.ones((n_ref, len(names)), dtype=bool)
        member[:, 1:] = ordered[list(WORD_CLASSES)].to_numpy(dtype=bool)
        substituted = np.zeros(n_ref, dtype=bool)
        deleted = np.zeros(n_ref, dtype=bool)
        inserted = np.zeros(n_ref + 1, dtype=np.int64)  # inserted words in each gap
        for chunk in chunks:
            if chunk.type == "substitute":
                substituted[chunk.ref_start_idx : chunk.ref_end_idx] = True
            elif chunk.type == "delete":
                deleted[chunk.ref_start_idx : chunk.ref_end_idx] = True
            elif chunk.type == "insert":
                inserted[chunk.ref_start_idx] += chunk.hyp_end_idx - chunk.hyp_start_idx
        near = np.zeros((n_ref + 1, len(names)), dtype=bool)  # a word of the class beside a gap
        near[1:] |= member
        near[:-1] |= member
        totals[:, 0] += member.sum(axis=0)
        totals[:, 1] += member[substituted].sum(axis=0)
        totals[:, 2] += member[deleted].sum(axis=0)
        totals[:, 3] += inserted @ near
    return pd.DataFrame(
        totals,
        index=pd.Index(names, name="class"),
        columns=["reference_words", "substitutions", "deletions", "insertions"],
    )


# Leakage -----------------------------------------------------------------------------


def _keys(block: Mapping[str, Any], expected: Sequence[str], name: str) -> None:
    """Refuse a config block with a missing or unknown key."""
    missing = sorted(set(expected) - set(block))
    unknown = sorted(set(block) - set(expected))
    if missing or unknown:
        raise ValueError(f"{name}: missing keys {missing}, unknown keys {unknown}")


def _whole(value: Any, name: str) -> int:
    """A whole number from the config; True, False and fractions are refused."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    return value


def _number(value: Any, name: str) -> float:
    """A finite number from the config; True and False are refused."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


@dataclass(frozen=True, slots=True)
class LeakageSettings:
    r"""Settings of the leakage classifier, from ``configs/pilot/data.yaml`` (``word_check.leakage``).

    Attributes:
        ngram_range: The shortest and longest word sequences counted (1, 2: words and
            word pairs), passed to scikit-learn as a tuple, which it requires.
        min_df: A sequence counts only if it occurs in at least this many training segments.
        token_pattern: What a word is; ``\S+`` keeps one-letter words such as "i".
        logistic_c: The inverse strength of the L2 penalty of the logistic regression.
        max_iter: The most iterations of lbfgs; a fit that needs more stops the job.
    """

    ngram_range: tuple[int, int]
    min_df: int
    token_pattern: str
    logistic_c: float
    max_iter: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LeakageSettings":
        """Read the settings from the whole data config.

        Args:
            config: The data config as loaded from YAML, with ``word_check.leakage``.

        Returns:
            The settings.

        Raises:
            ValueError: If a key is missing or unknown, or a value has the wrong type or
                lies out of range.
        """
        block = config["word_check"]["leakage"]
        _keys(
            block, ("ngram_range", "min_df", "token_pattern", "logistic_c", "max_iter"), "leakage"
        )
        ngram = block["ngram_range"]
        if not isinstance(ngram, list | tuple) or len(ngram) != 2:
            raise ValueError(f"the leakage n-gram range must be two numbers, got {ngram!r}")
        low, high = (_whole(value, "an n-gram length") for value in ngram)
        settings = cls(
            ngram_range=(low, high),
            min_df=_whole(block["min_df"], "the leakage min_df"),
            token_pattern=str(block["token_pattern"]),
            logistic_c=_number(block["logistic_c"], "the leakage C"),
            max_iter=_whole(block["max_iter"], "the leakage max_iter"),
        )
        if not 1 <= low <= high:
            raise ValueError(f"the n-gram range must run from 1 or more upwards, got {ngram!r}")
        if settings.min_df < 1 or settings.max_iter < 1:
            raise ValueError("the leakage min_df and max_iter must be at least 1")
        if not settings.logistic_c > 0.0 or not settings.token_pattern:
            raise ValueError("the leakage C must be positive and the token pattern not empty")
        return settings


def _participants(*series: pd.Series) -> list[int]:
    """The participants that every series names, sorted; refuse series that name others."""
    sets = [{int(session) for session in values.index} for values in series]
    if any(other != sets[0] for other in sets[1:]) or any(
        values.index.duplicated().any() for values in series
    ):
        raise ValueError("the scores, labels and folds must name the same participants once each")
    return sorted(sets[0])


def _labels(labels: pd.Series) -> pd.Series:
    """The labels as whole numbers 0 and 1, by participant."""
    values = labels.to_numpy()
    if not np.isin(values, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    return pd.Series(values.astype(np.int64), index=labels.index.astype(np.int64))


def fit_leakage(
    texts: pd.DataFrame, labels: pd.Series, folds: pd.Series, settings: LeakageSettings
) -> dict[int, Pipeline]:
    """Fit one leakage classifier per fold, on the stored words of the other folds' participants.

    Pilot protocol, check F4: TF-IDF of the words and word pairs that occur in at least
    two training segments, one-letter words included, then a logistic regression (L2,
    lbfgs). Every segment of every training participant is one training text, labelled
    with its participant's label; the texts pass through ``words``. A fit that does not
    converge stops the job: scikit-learn's ``ConvergenceWarning`` becomes an error.

    Args:
        texts: One row per segment with the columns ``session`` and ``text`` (its stored
            words).
        labels: Each participant's label (1 if the PHQ-8 total is 10 or more), by session ID.
        folds: Each participant's fold, by session ID.
        settings: The classifier's settings.

    Returns:
        One fitted pipeline (``tfidf``, then ``classifier``) per fold, by fold. It holds
        participant words: never print or save it.

    Raises:
        ValueError: If labels and folds do not name the same participants, a segment's
            participant has no fold, or a fold's training texts hold one class only.
        StepError: If a fit does not converge.
    """
    _participants(labels, folds)
    label = _labels(labels)
    fold_of = folds.astype(np.int64)
    sessions = texts["session"].to_numpy(dtype=np.int64)
    unknown = int((~np.isin(sessions, fold_of.index.to_numpy())).sum())
    if unknown:
        raise ValueError(f"{unknown} segments belong to participants without a fold")
    segment_folds = fold_of.reindex(sessions).to_numpy()
    segment_labels = label.reindex(sessions).to_numpy()
    documents = np.array([" ".join(words(str(text))) for text in texts["text"]], dtype=object)
    models: dict[int, Pipeline] = {}
    for fold in sorted({int(value) for value in fold_of}):
        train = segment_folds != fold
        if len(np.unique(segment_labels[train])) != 2:
            raise ValueError(f"the training texts of fold {fold} hold one class only")
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=settings.ngram_range,
                        min_df=settings.min_df,
                        token_pattern=settings.token_pattern,
                        lowercase=False,  # the words are lower case already
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(  # L2 is the default; penalty is deprecated since 1.8
                        C=settings.logistic_c, solver="lbfgs", max_iter=settings.max_iter
                    ),
                ),
            ]
        )
        converged = True
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                model.fit(list(documents[train]), segment_labels[train])
            except ConvergenceWarning:
                converged = False
        if not converged:  # raised outside the except clause, so that nothing chains to it
            raise StepError(
                f"the leakage classifier of fold {fold} did not converge in"
                f" {settings.max_iter} iterations"
            )
        models[fold] = model
    return models


def person_scores(
    models: Mapping[int, Pipeline],
    folds: pd.Series,
    segments: pd.DataFrame,
    hypotheses: Sequence[str],
) -> pd.Series:
    """Score each participant by the leakage classifier of its own fold.

    Each output passes through ``words`` and gets the classifier's log-odds (its
    ``decision_function``); a participant's score is the mean over its segments. A text
    that is empty or holds no word the classifier knows scores the intercept. The mean is
    computed exactly and rounded once (``statistics.mean``), so that participants whose
    segments score alike, in any number and order, get the same score and tie in the fold
    AUROC: a floating-point sum would turn the mean of three equal scores into a value
    one rounding step away, and the rounding would rank such participants.

    Args:
        models: The fitted classifiers by fold (``fit_leakage``).
        folds: Each participant's fold, by session ID.
        segments: One row per scored segment with the column ``session``.
        hypotheses: The recogniser's output of each segment, in the same order.

    Returns:
        The mean log-odds of each participant, indexed by session ID in ascending order.

    Raises:
        TypeError: If an output is not a text.
        ValueError: If the lengths differ, a participant has no fold, or a fold no model.
    """
    if len(segments) != len(hypotheses):
        raise ValueError(f"{len(hypotheses)} outputs for {len(segments)} segments")
    if any(not isinstance(text, str) for text in hypotheses):
        raise TypeError("an output must be a text")
    sessions = segments["session"].to_numpy(dtype=np.int64)
    fold_of = folds.astype(np.int64)
    unknown = int((~np.isin(sessions, fold_of.index.to_numpy())).sum())
    if unknown:
        raise ValueError(f"{unknown} segments belong to participants without a fold")
    segment_folds = fold_of.reindex(sessions).to_numpy()
    documents = [" ".join(words(text)) for text in hypotheses]
    scores = np.empty(len(documents), dtype=np.float64)
    for fold in np.unique(segment_folds):
        if int(fold) not in models:
            raise ValueError(f"no leakage classifier for fold {int(fold)}")
        rows = np.flatnonzero(segment_folds == fold)
        scores[rows] = models[int(fold)].decision_function([documents[row] for row in rows])
    by_session: dict[int, list[float]] = {}
    for session, score in zip(sessions.tolist(), scores.tolist(), strict=True):
        by_session.setdefault(int(session), []).append(score)
    participants = sorted(by_session)
    return pd.Series(
        [statistics.mean(by_session[session]) for session in participants],
        index=pd.Index(participants, name="session", dtype=np.int64),
        dtype=np.float64,
        name="score",
    )


def _fold_members(
    scores: pd.Series, labels: pd.Series, folds: pd.Series
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Each fold with the positions of its positive and its negative participants, sorted by ID."""
    participants = _participants(scores, labels, folds)
    label = _labels(labels).reindex(participants).to_numpy()
    fold_of = folds.astype(np.int64).reindex(participants).to_numpy()
    members = []
    for fold in sorted({int(value) for value in fold_of}):
        positives = np.flatnonzero((fold_of == fold) & (label == 1))
        negatives = np.flatnonzero((fold_of == fold) & (label == 0))
        if not len(positives) or not len(negatives):
            raise ValueError(f"fold {fold} holds one class only")
        members.append((fold, positives, negatives))
    return members


def fold_aurocs(scores: pd.Series, labels: pd.Series, folds: pd.Series) -> np.ndarray:
    """Return the AUROC of the participants' scores in each fold; ties count one half.

    The leakage AUROC is the mean of these, never the AUROC of all folds pooled (pilot
    protocol, check F4).

    Args:
        scores: Each participant's score (``person_scores``), by session ID.
        labels: Each participant's label, 0 or 1, by session ID.
        folds: Each participant's fold, by session ID.

    Returns:
        One AUROC per fold, in ascending fold order, of shape (F,).

    Raises:
        ValueError: If the three do not name the same participants, or a fold holds one
            class only.
    """
    participants = _participants(scores, labels, folds)
    values = scores.astype(np.float64).reindex(participants).to_numpy()
    return np.array(
        [
            float(pair_wins(values[positives], values[negatives]).mean())
            for _, positives, negatives in _fold_members(scores, labels, folds)
        ]
    )


# The interval of the x0 control ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BootstrapSettings:
    """Settings of the x0 control's interval, from ``configs/pilot/data.yaml`` (``word_check.bootstrap``).

    Attributes:
        resamples: Resamples of the participants.
        seed: The run seed of the interval's own stream (``rng``).
    """

    resamples: int
    seed: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "BootstrapSettings":
        """Read the settings from the whole data config.

        Args:
            config: The data config as loaded from YAML, with ``word_check.bootstrap``.

        Returns:
            The settings.

        Raises:
            ValueError: If a key is missing or unknown, or a value is not a whole number
                in range.
        """
        block = config["word_check"]["bootstrap"]
        _keys(block, ("resamples", "seed"), "bootstrap")
        settings = cls(
            resamples=_whole(block["resamples"], "the bootstrap resamples"),
            seed=_whole(block["seed"], "the bootstrap seed"),
        )
        if settings.resamples < 1 or settings.seed < 0:
            raise ValueError("the bootstrap needs at least one resample and a seed of 0 or more")
        return settings

    def rng(self) -> np.random.Generator:
        """The interval's own stream: ``item_seed(seed, "wordcheck", "bootstrap")``.

        Not a second ``default_rng(seed)``, whose first draws are those of the sample.
        """
        return np.random.default_rng(item_seed(self.seed, *BOOTSTRAP_KEYS))


def bootstrap_sets(
    labels: pd.Series, folds: pd.Series, rng: np.random.Generator, resamples: int
) -> tuple[np.ndarray, int]:
    """Draw the participants of every resample of the x0 control's interval.

    Within each class, as many participants as the class has are drawn with replacement
    from all participants of the class: the positives first, then the negatives, each in
    ascending ID. A resample that leaves a fold without a positive or without a negative
    is drawn again and counted (scikit-learn's ``roc_auc_score`` would return NaN for it,
    without raising). The draws depend only on the labels and folds, so they are drawn
    once, before any transcription, and serve every view and recogniser.

    Args:
        labels: Each participant's label, 0 or 1, by session ID.
        folds: Each participant's fold, by session ID.
        rng: The interval's own stream (``BootstrapSettings.rng``).
        resamples: Resamples to keep.

    Returns:
        The session IDs drawn in each kept resample, of shape (resamples, participants),
        and the number of resamples drawn again.

    Raises:
        ValueError: If labels and folds do not name the same participants, the count is
            not positive, or a fold holds one class only.
    """
    if resamples < 1:
        raise ValueError(f"draw at least one resample, not {resamples}")
    participants = _participants(labels, folds)
    label = _labels(labels).reindex(participants).to_numpy()
    fold_of = folds.astype(np.int64).reindex(participants).to_numpy()
    fold_ids = sorted({int(value) for value in fold_of})
    index = {fold: k for k, fold in enumerate(fold_ids)}
    sessions = np.array(participants, dtype=np.int64)
    classes = []
    for value in (1, 0):
        members = label == value
        folds_present = {int(fold) for fold in fold_of[members]}
        if folds_present != set(fold_ids):
            raise ValueError("every fold must hold both classes")
        classes.append((sessions[members], np.array([index[int(f)] for f in fold_of[members]])))
    sets = np.empty((resamples, len(participants)), dtype=np.int64)
    redraws = kept = 0
    while kept < resamples:
        drawn = [
            rng.choice(len(members), size=len(members), replace=True) for members, _ in classes
        ]
        if all(
            np.bincount(position[choice], minlength=len(fold_ids)).all()
            for (_, position), choice in zip(classes, drawn, strict=True)
        ):
            sets[kept] = np.concatenate(
                [members[choice] for (members, _), choice in zip(classes, drawn, strict=True)]
            )
            kept += 1
        else:
            redraws += 1
    return sets, redraws


def bootstrap_aurocs(
    sets: np.ndarray, scores: pd.Series, labels: pd.Series, folds: pd.Series
) -> np.ndarray:
    """Return each resample's fold AUROCs, with the fold models and person scores fixed.

    A participant drawn k times counts k times: each fold's AUROC weights the pairs of a
    positive and a negative by their draw counts, which equals the AUROC of the repeated
    rows, ties one half.

    Args:
        sets: The drawn session IDs of each resample (``bootstrap_sets``), of shape (R, P).
        scores: Each participant's score, by session ID.
        labels: Each participant's label, 0 or 1, by session ID.
        folds: Each participant's fold, by session ID.

    Returns:
        The AUROCs of shape (R, F), folds in ascending order.

    Raises:
        ValueError: If the three do not name the same participants, a resample names
            another participant or leaves a fold with one class.
    """
    participants = _participants(scores, labels, folds)
    values = scores.astype(np.float64).reindex(participants).to_numpy()
    sorted_ids = np.array(participants, dtype=np.int64)
    drawn = np.asarray(sets, dtype=np.int64)
    if drawn.ndim != 2:
        raise ValueError("the resamples must form a table of shape (resamples, participants)")
    positions = np.searchsorted(sorted_ids, drawn)
    if (positions >= len(sorted_ids)).any() or (
        sorted_ids[np.minimum(positions, len(sorted_ids) - 1)] != drawn
    ).any():
        raise ValueError("a resample names a participant without a score")
    rows = np.arange(len(drawn))[:, None]
    flat = (rows * len(sorted_ids) + positions).ravel()
    counts = np.bincount(flat, minlength=len(drawn) * len(sorted_ids)).reshape(
        len(drawn), len(sorted_ids)
    )
    aurocs = []
    for fold, positives, negatives in _fold_members(scores, labels, folds):
        drawn_pos, drawn_neg = counts[:, positives], counts[:, negatives]
        if not (drawn_pos.sum(axis=1) * drawn_neg.sum(axis=1)).all():
            raise ValueError(f"a resample leaves fold {fold} with one class")
        wins = pair_wins(values[positives], values[negatives])
        aurocs.append(weighted_aurocs(wins, drawn_pos, drawn_neg))
    return np.column_stack(aurocs)


def bootstrap_interval(
    sets: np.ndarray, scores: pd.Series, labels: pd.Series, folds: pd.Series, level: float
) -> tuple[float, float]:
    """Return the percentile interval of the leakage AUROC, the mean of the fold AUROCs.

    Pilot protocol, section "Intervals", and check F4 (the x0 control): the fold models
    and person scores stay fixed; each resample's fold AUROCs (``bootstrap_aurocs``) are
    averaged; the ends are NumPy's linear percentiles.

    Args:
        sets: The drawn session IDs of each resample (``bootstrap_sets``), of shape (R, P).
        scores: Each participant's score, by session ID.
        labels: Each participant's label, 0 or 1, by session ID.
        folds: Each participant's fold, by session ID.
        level: The interval's coverage, such as 0.95.

    Returns:
        The lower and upper end.

    Raises:
        ValueError: If the level does not lie strictly between 0 and 1, or as
            ``bootstrap_aurocs``.
    """
    if not 0.0 < level < 1.0:  # before any AUROC is computed
        raise ValueError(f"the interval's level must lie between 0 and 1, got {level}")
    return interval(bootstrap_aurocs(sets, scores, labels, folds).mean(axis=1), level)


# Verdicts ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Rules:
    """The bounds of the word check, from ``configs/pilot/data.yaml`` (``word_check.rules``).

    Read on the point estimates, except the x0 control's interval. "At least" and "at
    most" include the bound; "more than" and "above" do not; values are compared
    unrounded.

    Attributes:
        rev_min_wer: F4: each recogniser's WER on rev must be at least this.
        rev_max_leakage_auroc: F4: each recogniser's leakage AUROC on rev must be at most this.
        x0_max_wer: x0 control: each recogniser's WER on x0 must be at most this.
        x0_leakage_interval: x0 control: the level of the leakage AUROC's interval, whose
            lower end must lie above 0.5.
        x1_max_wer: x1 words: each recogniser's WER on x1 must be at most this.
        x1_max_icc: x1 voice: each gate ICC must be at most this.
        probe_margin_auroc: x1 voice: x0's probe counts if it is more than this above the
            gender-only control; then x1 fails if its own probe is more than this above it.
    """

    rev_min_wer: float
    rev_max_leakage_auroc: float
    x0_max_wer: float
    x0_leakage_interval: float
    x1_max_wer: float
    x1_max_icc: float
    probe_margin_auroc: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Rules":
        """Read the bounds from the whole data config.

        Args:
            config: The data config as loaded from YAML, with ``word_check.rules``.

        Returns:
            The bounds.

        Raises:
            ValueError: If a key is missing or unknown, or a bound is not a number in range.
        """
        block = config["word_check"]["rules"]
        names = [field.name for field in fields(cls)]
        _keys(block, names, "rules")
        rules = cls(**{name: _number(block[name], name) for name in names})
        wers = (rules.rev_min_wer, rules.x0_max_wer, rules.x1_max_wer)
        if min(wers) < 0.0:
            raise ValueError("a WER bound must not be negative")
        if not 0.0 <= rules.rev_max_leakage_auroc <= 1.0:
            raise ValueError("the leakage bound must lie between 0 and 1")
        if not 0.0 < rules.x0_leakage_interval < 1.0:
            raise ValueError("the interval's level must lie between 0 and 1")
        if not -1.0 <= rules.x1_max_icc <= 1.0:
            raise ValueError("the ICC bound must lie between -1 and 1")
        if not 0.0 <= rules.probe_margin_auroc < 1.0:
            raise ValueError("the probe margin must lie between 0 and 1")
        return rules


@dataclass(frozen=True, slots=True)
class Results:
    """The numbers that the verdicts read, as the word check computed them.

    Attributes:
        wer: The WER by view (``rev``, ``x0``, ``x1``) and recogniser (``RECOGNISERS``).
        leakage: The leakage AUROC (the mean of the fold AUROCs) by view (``rev``, ``x0``)
            and recogniser.
        x0_intervals: x0's leakage interval (lower end, upper end) by recogniser.
        iccs: The six gate ICCs (three measures within each gender, after the centring
            by voice), by a text-free name such as ``female loudness_sma3_stddevNorm``.
        probe_x0: x0's probe AUROC (the mean of the fold AUROCs).
        probe_x1: x1's probe AUROC.
        probe_control: The gender-only control's AUROC.
        after_rerun: Whether the leakage numbers come from the rerun on all segments, so
            that no further rerun can follow.
    """

    wer: Mapping[str, Mapping[str, float]]
    leakage: Mapping[str, Mapping[str, float]]
    x0_intervals: Mapping[str, tuple[float, float]]
    iccs: Mapping[str, float]
    probe_x0: float
    probe_x1: float
    probe_control: float
    after_rerun: bool = False


def _gate(value: Any, name: str) -> float:
    """A gate value as a float; one that is not a finite number stops the job, never compared."""
    number = float(value)
    if not math.isfinite(number):
        raise StepError(f"the gate value {name} is not a finite number")
    return number


def _per_recogniser(values: Mapping[str, Any], name: str) -> dict[str, float]:
    """One gate value per recogniser."""
    if set(values) != set(RECOGNISERS):
        raise ValueError(f"{name} needs one value for each of {list(RECOGNISERS)}")
    return {
        recogniser: _gate(values[recogniser], f"{name} ({recogniser})")
        for recogniser in RECOGNISERS
    }


def f4_verdict(
    wer: Mapping[str, float],
    leakage: Mapping[str, float],
    rules: Rules,
    *,
    rerun_pending: bool,
    view: str = "rev",
) -> dict[str, Any]:
    """Judge check F4: each recogniser's WER at least the bound and leakage AUROC at most the bound.

    A WER below its bound fails F4 at once, because the rerun keeps the sample's WER.
    Otherwise, while the x0 control waits for the rerun, F4 waits too: its leakage is
    provisional and is read only after the rerun.

    Args:
        wer: The view's WER by recogniser.
        leakage: The view's leakage AUROC by recogniser.
        rules: The bounds.
        rerun_pending: Whether the x0 control said "RERUN NEEDED".
        view: The view judged: ``rev``, or its backup.

    Returns:
        ``verdict`` (``PASS``, ``FAIL`` or ``WAITS``), the numbers and bounds behind it,
        ``leakage_provisional`` and the rules that ``failed``.

    Raises:
        ValueError: If a recogniser's value is missing or another is given.
        StepError: If a value is not a finite number.
    """
    wers = _per_recogniser(wer, f"{view} WER")
    leaks = _per_recogniser(leakage, f"{view} leakage AUROC")
    failed = [f"{name} WER" for name in RECOGNISERS if wers[name] < rules.rev_min_wer]
    if not rerun_pending:
        failed += [
            f"{name} leakage" for name in RECOGNISERS if leaks[name] > rules.rev_max_leakage_auroc
        ]
    verdict = FAIL if failed else WAITS if rerun_pending else PASS
    return {
        "view": view,
        "verdict": verdict,
        "wer": wers,
        "leakage": leaks,
        "min_wer": rules.rev_min_wer,
        "max_leakage_auroc": rules.rev_max_leakage_auroc,
        "leakage_provisional": rerun_pending,
        "failed": failed,
    }


def x0_control_verdict(
    wer: Mapping[str, float],
    leakage: Mapping[str, float],
    intervals: Mapping[str, tuple[float, float]],
    rules: Rules,
    *,
    after_rerun: bool,
) -> dict[str, Any]:
    """Judge the x0 control: each recogniser's WER at most the bound and its interval above 0.5.

    An interval that includes 0.5 (lower end at most 0.5, upper end at least 0.5) calls
    the rerun on all segments, once; after the rerun it fails the control. An interval
    wholly below 0.5 fails the control without a rerun. A WER above its bound fails the
    control but never calls the rerun. A failed control never changes F4.

    Args:
        wer: x0's WER by recogniser.
        leakage: x0's leakage AUROC by recogniser (reported beside the interval).
        intervals: x0's leakage interval (lower end, upper end) by recogniser.
        rules: The bounds.
        after_rerun: Whether the leakage numbers come from the rerun.

    Returns:
        ``verdict`` (``PASS``, ``FAIL`` or ``RERUN_NEEDED``), the numbers and bounds
        behind it and the rules that ``failed``.

    Raises:
        ValueError: If a recogniser's value is missing or another is given, or an
            interval's lower end lies above its upper end.
        StepError: If a value is not a finite number.
    """
    wers = _per_recogniser(wer, "x0 WER")
    leaks = _per_recogniser(leakage, "x0 leakage AUROC")
    if set(intervals) != set(RECOGNISERS):
        raise ValueError(f"the x0 intervals need one for each of {list(RECOGNISERS)}")
    bounds: dict[str, tuple[float, float]] = {}
    for name in RECOGNISERS:
        low, high = intervals[name]
        bounds[name] = (
            _gate(low, f"x0 interval lower end ({name})"),
            _gate(high, f"x0 interval upper end ({name})"),
        )
        if bounds[name][0] > bounds[name][1]:
            raise ValueError(f"the x0 interval of {name} has its lower end above its upper end")
    includes = [name for name in RECOGNISERS if bounds[name][0] <= CHANCE <= bounds[name][1]]
    failed = [f"{name} WER" for name in RECOGNISERS if wers[name] > rules.x0_max_wer]
    failed += [f"{name} interval below 0.5" for name in RECOGNISERS if bounds[name][1] < CHANCE]
    if after_rerun:
        failed += [f"{name} interval includes 0.5" for name in includes]
    rerun_needed = bool(includes) and not after_rerun
    verdict = RERUN_NEEDED if rerun_needed else FAIL if failed else PASS
    return {
        "verdict": verdict,
        "wer": wers,
        "leakage": leaks,
        "intervals": {name: list(bounds[name]) for name in RECOGNISERS},
        "max_wer": rules.x0_max_wer,
        "interval_level": rules.x0_leakage_interval,
        "after_rerun": after_rerun,
        "failed": failed,
    }


def x1_words_verdict(wer: Mapping[str, float], rules: Rules) -> dict[str, Any]:
    """Judge x1's word rule: each recogniser's WER on x1 at most the bound.

    Args:
        wer: x1's WER by recogniser.
        rules: The bounds.

    Returns:
        ``verdict`` (``PASS`` or ``FAIL``), the numbers and bound behind it and the rules
        that ``failed``.

    Raises:
        ValueError: If a recogniser's value is missing or another is given.
        StepError: If a value is not a finite number.
    """
    wers = _per_recogniser(wer, "x1 WER")
    failed = [f"{name} WER" for name in RECOGNISERS if wers[name] > rules.x1_max_wer]
    return {
        "verdict": FAIL if failed else PASS,
        "wer": wers,
        "max_wer": rules.x1_max_wer,
        "failed": failed,
    }


def x1_voice_verdict(
    iccs: Mapping[str, float],
    probe_x0: float,
    probe_x1: float,
    control: float,
    rules: Rules,
) -> dict[str, Any]:
    """Judge x1's voice rule: the six gate ICCs and, if it counts, the probe.

    x1 fails if any gate ICC is above the bound, or if the probe counts (x0's probe more
    than the margin above the gender-only control) and x1's probe is more than the margin
    above the control too. If the probe does not count, the ICCs decide alone, and the
    result says so. A negative ICC passes.

    Args:
        iccs: The six gate ICCs, by text-free name.
        probe_x0: x0's probe AUROC.
        probe_x1: x1's probe AUROC.
        control: The gender-only control's AUROC.
        rules: The bounds.

    Returns:
        ``verdict`` (``PASS`` or ``FAIL``), the numbers and bounds behind it, the probe's
        ``bar`` and whether it ``counts``, ``decided_by`` and the rules that ``failed``.

    Raises:
        ValueError: If there are not six ICCs.
        StepError: If a value is not a finite number.
    """
    if len(iccs) != GATE_ICCS:
        raise ValueError(f"the voice rule reads {GATE_ICCS} gate ICCs, got {len(iccs)}")
    values = {str(name): _gate(value, f"ICC {name}") for name, value in iccs.items()}
    x0 = _gate(probe_x0, "x0 probe")
    x1 = _gate(probe_x1, "x1 probe")
    baseline = _gate(control, "probe control")
    bar = baseline + rules.probe_margin_auroc
    counts = x0 > bar
    failed = [f"ICC {name}" for name, value in values.items() if value > rules.x1_max_icc]
    if counts and x1 > bar:
        failed.append("probe")
    return {
        "verdict": FAIL if failed else PASS,
        "iccs": values,
        "max_icc": rules.x1_max_icc,
        "probe": {
            "x0": x0,
            "x1": x1,
            "control": baseline,
            "margin": rules.probe_margin_auroc,
            "bar": bar,
            "counts": counts,
        },
        "decided_by": "ICCs and probe" if counts else "ICCs alone: the probe does not count",
        "failed": failed,
    }


def verdicts(results: Results, rules: Rules) -> dict[str, dict[str, Any]]:
    """Judge F4, the x0 control and the two rules of x1.

    Args:
        results: The numbers of the word check.
        rules: The bounds.

    Returns:
        ``f4``, ``x0_control``, ``x1_words`` and ``x1_voice``, each a verdict with the
        numbers behind it.

    Raises:
        ValueError: If a number is missing.
        StepError: If a gate value is not a finite number.
    """
    control = x0_control_verdict(
        results.wer["x0"],
        results.leakage["x0"],
        results.x0_intervals,
        rules,
        after_rerun=results.after_rerun,
    )
    return {
        "f4": f4_verdict(
            results.wer["rev"],
            results.leakage["rev"],
            rules,
            rerun_pending=control["verdict"] == RERUN_NEEDED,
        ),
        "x0_control": control,
        "x1_words": x1_words_verdict(results.wer["x1"], rules),
        "x1_voice": x1_voice_verdict(
            results.iccs, results.probe_x0, results.probe_x1, results.probe_control, rules
        ),
    }


def verdict_lines(judged: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Write the verdicts as the job prints them, values with four decimals.

    Args:
        judged: Any of ``f4``, ``x0_control``, ``x1_words`` and ``x1_voice`` (``verdicts``).

    Returns:
        One line per verdict given, in that order.
    """
    lines = []
    if "f4" in judged:
        f4 = judged["f4"]
        parts = [
            f"{name} WER {f4['wer'][name]:.4f}, leakage {f4['leakage'][name]:.4f}"
            for name in RECOGNISERS
        ]
        note = " (leakage provisional while waiting)" if f4["leakage_provisional"] else ""
        lines.append(f"F4 ({f4['view']}): {f4['verdict']}  {'; '.join(parts)}{note}")
    if "x0_control" in judged:
        x0 = judged["x0_control"]
        parts = [
            f"{name} WER {x0['wer'][name]:.4f}, leakage {x0['leakage'][name]:.4f}"
            f" [{x0['intervals'][name][0]:.4f}, {x0['intervals'][name][1]:.4f}]"
            for name in RECOGNISERS
        ]
        lines.append(f"x0 control: {x0['verdict']}  {'; '.join(parts)}")
    if "x1_words" in judged:
        x1 = judged["x1_words"]
        parts = [f"{name} {x1['wer'][name]:.4f}" for name in RECOGNISERS]
        lines.append(f"x1 words: {x1['verdict']}  {'; '.join(parts)}")
    if "x1_voice" in judged:
        voice = judged["x1_voice"]
        probe = voice["probe"]
        iccs = ", ".join(f"{name} {value:.4f}" for name, value in voice["iccs"].items())
        how = "counts" if probe["counts"] else "does not count: the ICCs decide"
        lines.append(
            f"x1 voice: {voice['verdict']}  ICC {iccs}; probe x0 {probe['x0']:.4f},"
            f" x1 {probe['x1']:.4f}, control {probe['control']:.4f}, bar {probe['bar']:.4f}"
            f" ({how})"
        )
    return lines
