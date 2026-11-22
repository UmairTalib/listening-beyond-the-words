"""The re-voiced clip x1: Kokoro-82M reads each part's words in a voice of the participant's gender.

Pilot protocol, section "Clip versions", bullet "Re-voiced clip" (fixed on 2026-11-22);
the settings are in ``configs/pilot/data.yaml`` (``views.x1``). x1 has the parts of x0
(``segments.segment_parts``), one Kokoro call per part:

- Text: each part's words as the E-DAIC transcript writes them (``part_text``), so
  that Kokoro pronounces them correctly, and turning the text back into stored words
  gives exactly the part's words.
- Voices: within each gender, participants alternate between two voices along their
  order by PHQ-8 class and ID (``assign_voices``), so that the voice is balanced
  against the label.
- Audio: each render is resampled to 16 kHz, trimmed where it is 40 dB below its peak
  with a 10 ms margin, faded like x0 and joined by 300 ms of silence; the clip is set
  to an RMS of 0.05 before the peak guard (``build_revoiced``).

``render_segment`` is the one code path of the test render and the build. ``Kokoro``
loads the pinned model and its text front end and checks them; Kokoro, misaki and torch
are imported inside its methods, so that everything else runs without them.
"""

import contextlib
import io
import json
import math
import os
import re
import warnings
from collections import Counter
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import pandas as pd
import soxr

from listening_beyond_words.clips import (
    load_cut_words,
    load_pinned_labels,
    load_pinned_sample,
    load_segments,
    load_view_manifest,
)
from listening_beyond_words.espeak import use_workspace_espeak
from listening_beyond_words.models import ModelPin, check_weights, local_snapshot
from listening_beyond_words.runinfo import StepError, item_seed, sha256_file
from listening_beyond_words.segments import GAP_SECONDS, segment_parts
from listening_beyond_words.settings import Workspace
from listening_beyond_words.transcripts import TYPOGRAPHIC_APOSTROPHE, normalise
from listening_beyond_words.views import (
    SAMPLING_RATE,
    check_segment_views,
    fade_part,
    join_parts,
    limit_peak,
    set_rms,
    to_pcm16,
)

TASKS: Final = 20  # array tasks of the build; scripts/slurm/build_revoiced.sbatch runs 0-19
KOKORO_RATE: Final = 24_000  # Kokoro's output rate
SOXR_QUALITIES: Final = ("QQ", "LQ", "MQ", "HQ", "VHQ")
RENDER_FLOOR: Final = 1e-3  # a render whose peak is at most this is refused, as in the tool check
MAX_PHONEMES: Final = 510  # above this, Kokoro splits a text into two calls
BRACKETS: Final = frozenset("<>[](){}")
# Phoneme characters that carry no sound of their own: punctuation and quotes, stress and
# length marks, modifiers and intonation arrows. A word needs another one to be heard.
NOT_SOUNDS: Final = frozenset(';:,.!?—…"“”()' + "ˈˌː" + "ʰʲ\u0303" + "↓→↗↘")
ESPEAK_RATING: Final = 2  # misaki's rating of a word whose sounds come from espeak
INVENTED_WORD: Final = "zorblefax"  # public, in no lexicon: only espeak can give it sounds
WEIGHTS_FILE: Final = "kokoro-v1_0.pth"
SESSIONS_CHECKSUMS: Final = "manifest/sessions_sha256.txt"  # in edaic/, by unpack_edaic_pilot.sh

_NOT_WRITTEN: Final = re.compile(r"[^A-Za-z0-9']+")
_SPOKEN: Final = re.compile(r"[A-Za-z0-9]")
_VOICE_NAME: Final = re.compile(r"[a-z]{2}_[a-z]+")


# Text ------------------------------------------------------------------------------


def written_form(token: str) -> str:
    """Return one transcript token as Kokoro reads it.

    A typographic apostrophe becomes a plain one; every other character that is not an
    ASCII letter, digit or apostrophe becomes a space; and pieces without a letter or
    digit are dropped. Capitals stay, and Kokoro reads digits itself.

    Args:
        token: A run of characters between white space in a transcript line.

    Returns:
        The words to read, joined by single spaces; empty if the token has none.
    """
    pieces = _NOT_WRITTEN.sub(" ", token.replace(TYPOGRAPHIC_APOSTROPHE, "'")).split()
    return " ".join(piece for piece in pieces if _SPOKEN.search(piece))


@dataclass(frozen=True, slots=True)
class PartText:
    """The text that Kokoro reads for one part, and how it was made.

    Attributes:
        text: The words to read, joined by single spaces.
        written: Transcript tokens read as written.
        unmatched: Tokens read as their stored words, because their written form does
            not turn back into exactly those words.
        split_start: Whether the part's first token runs on before the part, so that
            only its stored words in the part are read.
        split_end: Whether the part's last token runs on after the part.
        typographic: Tokens with a typographic apostrophe.
        brackets: Tokens with any of ``<>[](){}``.
    """

    text: str
    written: int
    unmatched: int
    split_start: bool
    split_end: bool
    typographic: int
    brackets: int


def part_text(line: str, first: int, last: int) -> PartText:
    """Return the text that Kokoro reads for the words ``first`` to ``last`` of one line.

    The line is split at white space into tokens, and each token's stored words are
    ``normalise(token).split()``; in order they are exactly the line's stored words,
    which the word indices count. A token whose stored words touch the part gives its
    written form (``written_form``) if all its stored words lie in the part and its
    written form turns back into exactly them; otherwise it gives its stored words that
    lie in the part.

    Args:
        line: The text of one transcript line.
        first: Index of the part's first word in the line's stored words.
        last: Index of the part's last word.

    Returns:
        The text and how it was made.

    Raises:
        ValueError: If the indices do not lie in the line, or the text does not turn back
            into exactly the part's stored words.
    """
    if not 0 <= first <= last:
        raise ValueError(f"a part cannot run from word {first} to word {last}")
    words: list[str] = []
    written = unmatched = typographic = brackets = 0
    split_start = split_end = False
    position = 0
    for token in line.split():
        stored = normalise(token).split()
        start, stop = position, position + len(stored)
        position = stop
        if not stored or stop <= first or start > last:
            continue
        typographic += TYPOGRAPHIC_APOSTROPHE in token
        brackets += any(character in BRACKETS for character in token)
        if first <= start and stop <= last + 1:
            form = written_form(token)
            if normalise(form).split() == stored:
                words.append(form)
                written += 1
            else:
                words.extend(stored)
                unmatched += 1
        else:
            split_start |= start < first
            split_end |= stop > last + 1
            words.extend(stored[max(first, start) - start : min(last + 1, stop) - start])
    if last >= position:
        raise ValueError(f"a part ends at word {last}, but its line has {position} words")
    text = " ".join(words)
    if normalise(text).split() != normalise(line).split()[first : last + 1]:
        raise ValueError("the text of a part does not give back its stored words")
    return PartText(
        text=text,
        written=written,
        unmatched=unmatched,
        split_start=split_start,
        split_end=split_end,
        typographic=typographic,
        brackets=brackets,
    )


PART_KEYS: Final = ("session", "position", "cut_part", "line", "first", "last", "n_words")


def part_rows(parts: pd.DataFrame) -> list[dict[str, int]]:
    """The whole-number columns of parts from ``segments.segment_parts``, row by row.

    Args:
        parts: Rows of ``segments.segment_parts``.

    Returns:
        One dictionary per row, with the columns of ``PART_KEYS`` as ints.
    """
    columns = {key: parts[key].to_numpy(dtype=np.int64).tolist() for key in PART_KEYS}
    return [{key: int(columns[key][i]) for key in PART_KEYS} for i in range(len(parts))]


def segment_words(parts: pd.DataFrame, lines: Sequence[str]) -> list[str]:
    """Return a segment's stored words, part after part, read from its transcript lines.

    Args:
        parts: The segment's rows of ``segments.segment_parts``, in clip order.
        lines: The session's transcript lines (``clips.read_lines``).

    Returns:
        The stored words of every part, in clip order.

    Raises:
        ValueError: If a part's words do not lie in its line.
    """
    words: list[str] = []
    for row in part_rows(parts):
        stored = normalise(lines[row["line"]]).split()
        if row["last"] >= len(stored):
            raise ValueError(f"a part ends at word {row['last']}, but its line has {len(stored)}")
        words.extend(stored[row["first"] : row["last"] + 1])
    return words


def word_classes(word: str, *, edge: bool) -> dict[str, bool]:
    """Describe one written word without its text, for counts that may be printed.

    Args:
        word: A word of a part's text.
        edge: Whether it is the first or last word of its part.

    Returns:
        Whether it has a digit, has an apostrophe, is in capitals (two letters or more),
        is one letter, and lies at a part edge.
    """
    letters = sum(character.isalpha() for character in word)
    return {
        "digit": any(character.isdigit() for character in word),
        "apostrophe": "'" in word,
        "capitals": word.isupper() and letters > 1,
        "one_letter": letters == 1 and len(word) == 1,
        "part_edge": edge,
    }


# Voices, tasks and the test render -------------------------------------------------


def assign_voices(labels: pd.DataFrame, voices: Mapping[str, Sequence[str]]) -> dict[int, str]:
    """Give every participant a voice of their gender, balanced against the label.

    Within each gender, participants are ordered by PHQ-8 class (negative first) and then
    by ID, and take the gender's voices in turn along that order, starting with the
    first. With two voices, each voice gets half of each class of its gender, give or
    take one (pilot protocol, bullet "Re-voiced clip").

    Args:
        labels: One row per participant, indexed by session ID, with the columns
            ``label`` (0 or 1) and ``gender``.
        voices: The voices of each gender, in the order in which they are taken.

    Returns:
        Each participant's voice, by session ID.

    Raises:
        ValueError: If a gender has no voices or a label is not 0 or 1.
    """
    missing = sorted(set(map(str, labels["gender"])) - set(voices))
    if missing:
        raise ValueError(f"no voices for the gender(s) {missing}")
    if not set(labels["label"].tolist()) <= {0, 1}:
        raise ValueError("every label must be 0 or 1")
    chosen: dict[int, str] = {}
    for gender, group in labels.groupby("gender", sort=True):
        names = list(voices[str(gender)])
        if not names:
            raise ValueError(f"no voices for the gender {gender}")
        order = sorted(
            (int(label), int(session))
            for session, label in zip(group.index, group["label"], strict=True)
        )
        for turn, (_, session) in enumerate(order):
            chosen[session] = names[turn % len(names)]
    return chosen


def assign_tasks(segment_counts: Mapping[int, int], tasks: int = TASKS) -> dict[int, int]:
    """Spread the sessions over the build's array tasks, largest first.

    Sessions in descending order of their number of segments (ties by ID) go one by one
    to the task with the fewest segments so far (ties to the lowest task), so that the
    tasks take about equally long. Every session belongs to exactly one task.

    Args:
        segment_counts: Each session's number of segments.
        tasks: The number of tasks.

    Returns:
        Each session's task, from 0 to ``tasks - 1``.

    Raises:
        ValueError: If there are no tasks.
    """
    if tasks < 1:
        raise ValueError(f"the build needs at least one task, not {tasks}")
    load = [0] * tasks
    chosen: dict[int, int] = {}
    for session in sorted(segment_counts, key=lambda s: (-int(segment_counts[s]), int(s))):
        task = min(range(tasks), key=lambda t: (load[t], t))
        chosen[int(session)] = task
        load[task] += int(segment_counts[session])
    return chosen


def draw_test_segments(
    segments: Collection[str], sample: Collection[str], count: int, seed: int
) -> list[str]:
    """Draw the segments of the test render, outside the word-check sample.

    Args:
        segments: Every segment ID.
        sample: The segments of the word-check sample, which the test render never uses.
        count: How many to draw.
        seed: The seed of the draw's own generator.

    Returns:
        The drawn segments in draw order, without repeats. The remaining segments are
        taken in ascending ID, so the draw depends only on the seed.

    Raises:
        ValueError: If fewer segments remain than ``count``.
    """
    candidates = sorted(set(segments) - set(sample))
    if not 0 < count <= len(candidates):
        raise ValueError(f"cannot draw {count} test segments from {len(candidates)}")
    chosen = np.random.default_rng(seed).choice(len(candidates), size=count, replace=False)
    return [candidates[int(i)] for i in chosen]


# Settings --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class X1Settings:
    """Settings of x1, from ``configs/pilot/data.yaml`` (``views.x1``, ``views`` and ``x0_cut``).

    Attributes:
        voices: The two voices of each gender, in the order in which they are taken.
        language: Kokoro's language code (``a``, US English).
        speed: Kokoro's speed.
        seed: The run seed of the parts' seeds (``runinfo.item_seed``).
        kokoro_rate: Kokoro's output rate in Hz.
        soxr_quality: soxr's quality setting of the resampling to 16 kHz.
        trim_db: A part is trimmed where it is this many dB below its peak (negative).
        trim_margin: Samples kept beyond the first and last louder sample, at 16 kHz.
        fade: Seconds of x0's fades at both ends of each part.
        rms: The RMS of a whole clip, gaps included, before the peak guard.
        peak_limit: Highest peak of a clip after its RMS is set.
        threads: PyTorch CPU threads per render, equal to the job's CPUs.
        test_segments: Segments of the test render.
        test_seed: Seed of the test render's draw.
        test_repeats: Test segments rendered a second time, the first by ID.
        test_heard: Test segments heard by the researcher, the first in draw order.
        rate: Sampling rate of the stored clips in Hz.
        gap: Seconds of silence between two parts.
    """

    voices: dict[str, tuple[str, ...]]
    language: str
    speed: float
    seed: int
    kokoro_rate: int
    soxr_quality: str
    trim_db: float
    trim_margin: int
    fade: float
    rms: float
    peak_limit: float
    threads: int
    test_segments: int
    test_seed: int
    test_repeats: int
    test_heard: int
    rate: int = SAMPLING_RATE
    gap: float = GAP_SECONDS

    @property
    def fade_samples(self) -> int:
        """Samples of one fade."""
        return round(self.fade * self.rate)

    @property
    def gap_samples(self) -> int:
        """Samples of silence between two parts."""
        return round(self.gap * self.rate)

    @property
    def all_voices(self) -> tuple[str, ...]:
        """Every voice of x1, gender by gender."""
        return tuple(name for gender in sorted(self.voices) for name in self.voices[gender])

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "X1Settings":
        """Read the settings from the whole data config.

        Args:
            config: The data config as loaded from YAML, with ``views`` and ``x0_cut``.

        Returns:
            The settings.

        Raises:
            ValueError: If the files are not 16-bit, a voice is not a US English voice of
                its gender, the margin is not a whole number of samples or is shorter than
                a fade, or a setting is out of range.
        """
        views = config["views"]
        x1 = views["x1"]
        test = x1["test_render"]
        if views["flac_bits"] != 16:
            raise ValueError(f"the views are stored as 16-bit FLAC, not {views['flac_bits']}-bit")
        language = str(x1["language"])
        if language != "a":
            raise ValueError(f"x1 is read in US English (language a), not {language}")
        voices = {str(gender): tuple(map(str, names)) for gender, names in x1["voices"].items()}
        if set(voices) != {"female", "male"}:
            raise ValueError(f"x1 needs voices for female and male, got {sorted(voices)}")
        for gender, names in voices.items():
            if len(names) != 2 or len(set(names)) != 2:
                raise ValueError(f"x1 needs two different {gender} voices, got {list(names)}")
            for name in names:
                if not (_VOICE_NAME.fullmatch(name) and name[:2] == language + gender[0]):
                    raise ValueError(f"{name} is not a US English {gender} voice of Kokoro")
        margin_ms = x1["trim_margin_ms"]
        margin = round(margin_ms * SAMPLING_RATE / 1000)
        if margin * 1000 != margin_ms * SAMPLING_RATE:
            raise ValueError(f"x1's trim margin of {margin_ms} ms is not a whole number of samples")
        settings = cls(
            voices=voices,
            language=language,
            speed=float(x1["speed"]),
            seed=_whole(x1["seed"], "x1's seed"),
            kokoro_rate=_whole(x1["kokoro_rate_hz"], "Kokoro's rate"),
            soxr_quality=str(x1["soxr_quality"]),
            trim_db=float(x1["trim_db"]),
            trim_margin=margin,
            fade=config["x0_cut"]["fade_ms"] / 1000,
            rms=float(x1["rms"]),
            peak_limit=float(views["peak_limit"]),
            threads=_whole(x1["threads"], "x1's threads"),
            test_segments=_whole(test["segments"], "the test render's segments"),
            test_seed=_whole(test["seed"], "the test render's seed"),
            test_repeats=_whole(test["repeats"], "the test render's repeats"),
            test_heard=_whole(test["heard"], "the test render's heard clips"),
        )
        settings._check()
        return settings

    def _check(self) -> None:
        """Refuse values out of range."""
        if self.kokoro_rate != KOKORO_RATE:
            raise ValueError(f"Kokoro renders at {KOKORO_RATE} Hz, not {self.kokoro_rate}")
        if self.soxr_quality not in SOXR_QUALITIES:
            raise ValueError(f"soxr has no quality {self.soxr_quality}")
        if not (math.isfinite(self.speed) and self.speed > 0.0):
            raise ValueError(f"Kokoro's speed must be a positive number, got {self.speed}")
        if not (math.isfinite(self.trim_db) and self.trim_db < 0.0):
            raise ValueError(f"the trim must lie below the peak, got {self.trim_db} dB")
        if not self.fade > 0.0:
            raise ValueError(f"x1's parts need x0's fades, got {self.fade} s")
        if self.trim_margin < self.fade_samples:
            raise ValueError("x1's trim margin must hold a whole fade")
        if not 0.0 < self.peak_limit < 1.0:
            raise ValueError(f"the peak limit must lie between 0 and 1, got {self.peak_limit}")
        if not 0.0 < self.rms < self.peak_limit:
            raise ValueError(f"x1's RMS must lie between 0 and the peak limit, got {self.rms}")
        if self.threads < 1:
            raise ValueError(f"x1 needs at least one thread, got {self.threads}")
        if not 1 <= self.test_heard <= self.test_segments:
            raise ValueError("the test render must hear between 1 and all of its segments")
        if not 1 <= self.test_repeats <= self.test_segments:
            raise ValueError("the test render must repeat between 1 and all of its segments")
        if self.seed < 0 or self.test_seed < 0:
            raise ValueError("seeds must not be negative")


def _whole(value: Any, name: str) -> int:
    """A whole number from the config; True, False and fractions are refused."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    return value


# Audio -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Trimmed:
    """One part after its quiet ends are cut.

    Attributes:
        kept: The kept samples, a copy.
        removed_start: Samples cut at the start.
        removed_end: Samples cut at the end.
        margin_start: Quiet samples kept before the first louder sample (at most the margin).
        margin_end: Quiet samples kept after the last louder sample.
    """

    kept: np.ndarray
    removed_start: int
    removed_end: int
    margin_start: int
    margin_end: int


def trim_quiet(part: np.ndarray, *, db: float, margin: int) -> Trimmed:
    """Cut the quiet stretches at both ends of one part, keeping a margin.

    The first and last sample whose size is at least the part's peak times 10^(db/20)
    are found sample by sample (in float64). The part keeps from ``margin`` samples
    before the first to ``margin`` samples after the last, as far as it reaches. Quiet
    stretches inside the part stay.

    Args:
        part: One part at 16 kHz, shape (samples,).
        db: The threshold below the peak in dB, negative.
        margin: Samples kept beyond the first and last louder sample.

    Returns:
        The kept samples and what was cut.

    Raises:
        ValueError: If the part is not one-dimensional, or it is empty or silent.
    """
    if part.ndim != 1:
        raise ValueError(f"expected a part of shape (samples,), got shape {part.shape}")
    size = np.abs(part.astype(np.float64))
    peak = float(size.max()) if size.size else 0.0
    if not peak > 0.0:
        raise ValueError("an empty or silent part has nothing to trim to")
    loud = np.flatnonzero(size >= peak * 10.0 ** (db / 20.0))
    first, last = int(loud[0]), int(loud[-1])
    low, high = max(0, first - margin), min(len(part), last + 1 + margin)
    return Trimmed(
        kept=part[low:high].copy(),
        removed_start=low,
        removed_end=len(part) - high,
        margin_start=first - low,
        margin_end=high - 1 - last,
    )


@dataclass(frozen=True, slots=True)
class PartAudio:
    """How one part's render became the stored part; lengths in samples.

    Attributes:
        rendered: The render's length at 24 kHz.
        resampled: Its length at 16 kHz.
        kept: The length kept by the trim, which the fades then shape.
        removed_start: Samples cut at the start, at 16 kHz.
        removed_end: Samples cut at the end.
        margin_start: Quiet samples kept before the first louder sample.
        margin_end: Quiet samples kept after the last louder sample.
        peak: The kept part's peak, before the clip's level is set.
    """

    rendered: int
    resampled: int
    kept: int
    removed_start: int
    removed_end: int
    margin_start: int
    margin_end: int
    peak: float


@dataclass(frozen=True, slots=True)
class Revoiced:
    """One re-voiced clip before it is stored.

    Attributes:
        clip: The clip as float64, at the RMS of x1 (lower if its peak was limited).
        part_starts: The first sample of each part.
        part_ends: The sample just after each part.
        gain: The gain applied to the joined parts: the RMS gain times the peak factor.
        peak_limited: Whether the clip was scaled down to the peak limit.
        parts: How each part was made.
    """

    clip: np.ndarray
    part_starts: tuple[int, ...]
    part_ends: tuple[int, ...]
    gain: float
    peak_limited: bool
    parts: tuple[PartAudio, ...]


def check_render(audio: np.ndarray) -> None:
    """Refuse a render that is empty, not finite or silent, as the tool check did.

    Args:
        audio: One render at 24 kHz, shape (samples,).

    Raises:
        ValueError: If it is not one-dimensional, is empty, holds NaN or inf, or its peak
            is at most ``RENDER_FLOOR``.
    """
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"a render must hold samples of shape (samples,), got {audio.shape}")
    if not np.all(np.isfinite(audio)):
        raise ValueError("a render holds NaN or inf")
    if not float(np.max(np.abs(audio))) > RENDER_FLOOR:
        raise ValueError(f"a render's peak is at most {RENDER_FLOOR}")


def part_audio(render: np.ndarray, settings: X1Settings) -> tuple[np.ndarray, PartAudio]:
    """Turn one part's render into the part that x1 stores, before the clip's level is set.

    The render is checked (``check_render``), resampled to 16 kHz with soxr in float32,
    trimmed (``trim_quiet``) and given x0's fades (``views.fade_part``), which fall in the
    trim's margin. Resampling comes before the fades, so that the part starts and ends
    at exactly 0.

    Args:
        render: Kokoro's render of the part at 24 kHz as float32, shape (samples,).
        settings: x1's settings.

    Returns:
        The faded part as float32 at 16 kHz, and how it was made.

    Raises:
        ValueError: If the render is refused, or the part keeps fewer samples than one fade.
    """
    check_render(render)
    resampled = np.asarray(
        soxr.resample(
            np.asarray(render, dtype=np.float32),
            settings.kokoro_rate,
            settings.rate,
            quality=settings.soxr_quality,
        ),
        dtype=np.float32,
    )
    trimmed = trim_quiet(resampled, db=settings.trim_db, margin=settings.trim_margin)
    if len(trimmed.kept) < settings.fade_samples:
        raise ValueError(
            f"a part keeps {len(trimmed.kept)} samples, fewer than one fade"
            f" ({settings.fade_samples})"
        )
    record = PartAudio(
        rendered=len(render),
        resampled=len(resampled),
        kept=len(trimmed.kept),
        removed_start=trimmed.removed_start,
        removed_end=trimmed.removed_end,
        margin_start=trimmed.margin_start,
        margin_end=trimmed.margin_end,
        peak=float(np.max(np.abs(trimmed.kept.astype(np.float64)))),
    )
    return fade_part(trimmed.kept, fade=settings.fade, rate=settings.rate), record


def join_revoiced(
    pieces: Sequence[np.ndarray], records: Sequence[PartAudio], settings: X1Settings
) -> Revoiced:
    """Join the parts of one re-voiced clip and set its level.

    The parts are joined by the gap (``views.join_parts``); then the clip's RMS, gaps
    included, is set (``views.set_rms``) and the peak guard applied (``views.limit_peak``).

    Args:
        pieces: The parts from ``part_audio``, in clip order.
        records: How each part was made, from ``part_audio``.
        settings: x1's settings.

    Returns:
        The clip and how it was made.

    Raises:
        ValueError: If there are no parts, or a record is missing.
    """
    if len(pieces) != len(records):
        raise ValueError("expected one record per part")
    joined, starts, ends = join_parts(pieces, rate=settings.rate, gap=settings.gap)
    levelled, gain = set_rms(joined, settings.rms)
    clip, factor, limited = limit_peak(levelled, settings.peak_limit)
    return Revoiced(
        clip=clip,
        part_starts=tuple(starts),
        part_ends=tuple(ends),
        gain=gain * factor,
        peak_limited=limited,
        parts=tuple(records),
    )


def build_revoiced(renders: Sequence[np.ndarray], settings: X1Settings) -> Revoiced:
    """Build one re-voiced clip from the renders of its parts.

    Each render becomes a part (``part_audio``: resampled, trimmed, faded), and the parts
    are joined and levelled (``join_revoiced``).

    Args:
        renders: Kokoro's render of each part at 24 kHz as float32, in clip order.
        settings: x1's settings.

    Returns:
        The clip and how it was made.

    Raises:
        ValueError: If there are no renders, a render is refused, or a part keeps fewer
            samples than one fade.
    """
    if len(renders) == 0:
        raise ValueError("a clip needs at least one part")
    made = [part_audio(render, settings) for render in renders]
    return join_revoiced([piece for piece, _ in made], [record for _, record in made], settings)


@dataclass(frozen=True, slots=True)
class Render:
    """One Kokoro call: the audio and what was read.

    Attributes:
        audio: The audio at 24 kHz as float32, shape (samples,).
        phonemes: The phonemes Kokoro read.
        first_frames: Length of the boundary token before the speech, in frames of 25 ms.
        last_frames: Length of the boundary token after it.
    """

    audio: np.ndarray
    phonemes: str
    first_frames: int
    last_frames: int


class Renderer(Protocol):
    """What ``render_segment`` needs of Kokoro; the tests pass a stand-in."""

    def render(self, text: str, voice: str, seed: int) -> Render:
        """Read one part's text in one voice, seeded."""
        ...


class PartError(StepError):
    """A part that could not be rendered or built; the error that stopped it is its context."""

    def __init__(self, segment: str, position: int, error: BaseException) -> None:
        super().__init__(f"segment {segment}, part {position}: {type(error).__name__}")
        self.segment = segment
        self.position = position


@dataclass(frozen=True, slots=True)
class RenderedSegment:
    """One segment's x1 as it is stored, and how it was made.

    Attributes:
        samples: The clip as 16-bit samples, shape (samples,).
        part_starts: The first sample of each part.
        part_ends: The sample just after each part.
        gain: The gain applied to the joined parts.
        peak_limited: Whether the peak guard scaled the clip down.
        renders: Kokoro's render of each part at 24 kHz, for the repeat check.
        parts: One record per part, for ``parts.parquet``; it holds the text and the
            phonemes, so it stays on OMNI.
    """

    samples: np.ndarray
    part_starts: tuple[int, ...]
    part_ends: tuple[int, ...]
    gain: float
    peak_limited: bool
    renders: tuple[np.ndarray, ...]
    parts: tuple[dict[str, Any], ...]


def render_segment(
    kokoro: Renderer,
    parts: pd.DataFrame,
    lines: Sequence[str],
    voice: str,
    segment: str,
    settings: X1Settings,
) -> RenderedSegment:
    """Render the re-voiced clip of one segment, as both the test render and the build do.

    Each part's text comes from ``part_text``, and its call is seeded with
    ``item_seed(settings.seed, segment, position)``, the position counted from 0 in clip
    order. The renders become one clip (``build_revoiced``) in 16-bit samples, which
    must pass ``views.check_segment_views`` with x1's own part bounds.

    Args:
        kokoro: Kokoro, or a stand-in with the same ``render``.
        parts: The segment's rows of ``segments.segment_parts``, in clip order.
        lines: The session's transcript lines (``clips.read_lines``).
        voice: The participant's voice (``assign_voices``).
        segment: The segment ID.
        settings: x1's settings.

    Returns:
        The stored clip and how it was made.

    Raises:
        ValueError: If the rows are not the segment's parts in clip order, or the clip
            fails the check of its parts and gaps.
        PartError: If a part cannot be read, rendered or built; it names the part.
    """
    rows = part_rows(parts)
    if [row["position"] for row in rows] != list(range(len(rows))) or set(parts["segment"]) != {
        segment
    }:
        raise ValueError(f"segment {segment}: expected its parts in clip order")
    texts: list[PartText] = []
    seeds: list[int] = []
    renders: list[Render] = []
    pieces: list[np.ndarray] = []
    made: list[PartAudio] = []
    for row in rows:
        position = row["position"]
        try:
            text = part_text(lines[row["line"]], row["first"], row["last"])
            seed = item_seed(settings.seed, segment, position)
            render = kokoro.render(text.text, voice, seed)
            piece, record = part_audio(render.audio, settings)
        except Exception as error:
            raise PartError(segment, position, error) from None
        texts.append(text)
        seeds.append(seed)
        renders.append(render)
        pieces.append(piece)
        made.append(record)
    built = join_revoiced(pieces, made, settings)
    samples = to_pcm16(built.clip)
    check_segment_views(
        {"x1": samples}, built.part_starts, built.part_ends, gap=settings.gap_samples
    )
    records = []
    for row, text, seed, render, audio, start, end in zip(
        rows,
        texts,
        seeds,
        renders,
        built.parts,
        built.part_starts,
        built.part_ends,
        strict=True,
    ):
        records.append(
            {
                "segment": segment,
                **row,
                "voice": voice,
                "seed": seed,
                "text": text.text,
                "phonemes": render.phonemes,
                "words_read": len(text.text.split()),
                "phoneme_chars": len(render.phonemes),
                "written": text.written,
                "unmatched": text.unmatched,
                "split_start": text.split_start,
                "split_end": text.split_end,
                "rendered": audio.rendered,
                "resampled": audio.resampled,
                "kept": audio.kept,
                "removed_start": audio.removed_start,
                "removed_end": audio.removed_end,
                "margin_start": audio.margin_start,
                "margin_end": audio.margin_end,
                "part_peak": audio.peak,
                "first_frames": render.first_frames,
                "last_frames": render.last_frames,
                "part_start": start,
                "part_end": end,
            }
        )
    return RenderedSegment(
        samples=samples,
        part_starts=built.part_starts,
        part_ends=built.part_ends,
        gain=built.gain,
        peak_limited=built.peak_limited,
        renders=tuple(render.audio for render in renders),
        parts=tuple(records),
    )


# Kokoro ----------------------------------------------------------------------------


def word_has_sound(phonemes: str, vocabulary: Collection[str]) -> bool:
    """Return whether a word's phonemes give Kokoro anything to voice.

    A word has a sound only if its phonemes hold at least one character of Kokoro's
    vocabulary that is not white space and not in ``NOT_SOUNDS`` (punctuation, quotes,
    stress and length marks, modifiers, intonation arrows). Kokoro drops every other
    character without a word.

    Args:
        phonemes: The joined phonemes of a written word's tokens.
        vocabulary: The phoneme characters of Kokoro's ``config.json``.

    Returns:
        True if the word has a sound.
    """
    return any(
        character in vocabulary and character not in NOT_SOUNDS and not character.isspace()
        for character in phonemes
    )


def token_ratings(token: Any) -> tuple[int | None, int | None]:
    """Return the rating misaki gave one token, from both places where misaki 0.9.4 keeps it.

    misaki sets ``token.rating`` on a word it looks up as a single token and
    ``token._.rating`` on a token it merged from pieces.

    Args:
        token: A token of misaki's ``G2P``.

    Returns:
        The rating in ``token.rating`` and the one in ``token._.rating``; None where unset.
    """
    underscore = getattr(token, "_", None)
    own = getattr(token, "rating", None)
    merged = getattr(underscore, "rating", None) if underscore is not None else None
    return (None if own is None else int(own), None if merged is None else int(merged))


@dataclass(frozen=True, slots=True)
class WordSound:
    """How Kokoro's front end read one written word; it holds no text.

    Attributes:
        voiced: Whether the word has a sound (``word_has_sound``).
        espeak: Whether its sounds came from espeak (misaki's rating 2 in either place).
        brackets: Whether its phonemes hold ``(`` or ``)``.
        tokens: How many of misaki's tokens it became.
        rating: misaki's lowest rating among its tokens; None if it gave none.
        tags: The spaCy tags of its tokens.
    """

    voiced: bool
    espeak: bool
    brackets: bool
    tokens: int
    rating: int | None
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FrontEnd:
    """What Kokoro's text front end makes of one part's text; only ``phonemes`` holds speech.

    Attributes:
        phonemes: The phonemes Kokoro reads, as its pipeline joins them.
        words: How each written word was read, in order.
        outside: Phoneme characters outside Kokoro's vocabulary, as code points (U+XXXX).
        prints: Lines that misaki printed, caught and never shown.
        warnings: Warnings caught while it ran.
    """

    phonemes: str
    words: tuple[WordSound, ...]
    outside: tuple[str, ...]
    prints: int
    warnings: int

    @property
    def silent(self) -> int:
        """Written words without a sound."""
        return sum(not word.voiced for word in self.words)

    @property
    def brackets(self) -> bool:
        """Whether the phonemes hold ``(`` or ``)``: an espeak language flag or a bracket."""
        return "(" in self.phonemes or ")" in self.phonemes

    @property
    def too_long(self) -> bool:
        """Whether Kokoro would split the text into two calls."""
        return len(self.phonemes) > MAX_PHONEMES

    @property
    def ok(self) -> bool:
        """Whether one Kokoro call voices every word with sounds it knows."""
        return not (self.silent or self.outside or self.brackets or self.too_long)


@dataclass
class Caught:
    """Output caught around a call of misaki or Kokoro: counted, never shown."""

    prints: int = 0
    warnings: int = 0


@contextlib.contextmanager
def caught_output() -> Iterator[Caught]:
    """Catch what misaki and Kokoro print or warn, which could hold a word, and count it.

    This covers ``print`` (misaki's "TODO:NUM" line holds the word) and Python warnings.
    Kokoro's own loguru logger is not caught here: importing kokoro 0.9.4 switches it off,
    and nothing in this project switches it on again.

    Yields:
        The counts, filled in when the block ends: printed lines (stdout and stderr) and
        warnings.
    """
    caught = Caught()
    out, err = io.StringIO(), io.StringIO()
    with (
        warnings.catch_warnings(record=True) as records,
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        warnings.simplefilter("always")
        try:
            yield caught
        finally:
            text = out.getvalue() + err.getvalue()
            caught.prints = sum(1 for line in text.splitlines() if line.strip())
            caught.warnings = len(records)


def group_words(tokens: Sequence[Any]) -> list[list[Any]]:
    """Group misaki's tokens into written words: a word ends at a token with white space after it.

    Args:
        tokens: The tokens of one text, in order.

    Returns:
        The tokens of each written word, in order.
    """
    groups: list[list[Any]] = []
    current: list[Any] = []
    for token in tokens:
        current.append(token)
        if token.whitespace:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def check_eval(model: Any) -> None:
    """Stop unless the model and every part of it are in evaluation mode.

    Kokoro never sets evaluation mode on a model it is given. With dropout on, the audio
    changes but still repeats with the same seed, so no repeat test would notice.

    Args:
        model: A PyTorch module.

    Raises:
        ValueError: If any part of it is in training mode.
    """
    if any(module.training for module in model.modules()):
        raise ValueError("Kokoro is in training mode, so its dropout would be on")


class Kokoro:
    """Kokoro-82M and its text front end, loaded and checked as x1 needs them.

    Build it with ``Kokoro.load`` on OMNI. The constructor takes the parts directly, so
    that the tests can pass stand-ins.

    Args:
        pipeline: Kokoro's ``KPipeline`` for US English, with the model.
        vocabulary: The phoneme characters of Kokoro's ``config.json``.
        voices: The local file of each voice.
        speed: Kokoro's speed.
        identity: What ``run.json`` records about the model, voices and espeak.
    """

    def __init__(
        self,
        pipeline: Any,
        vocabulary: Collection[str],
        voices: Mapping[str, Path],
        speed: float,
        identity: Mapping[str, Any] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.vocabulary = frozenset(vocabulary)
        self.voices = dict(voices)
        self.speed = speed
        self.identity = dict(identity or {})

    @classmethod
    def load(cls, workspace: Workspace, pin: ModelPin, settings: X1Settings) -> "Kokoro":
        """Load Kokoro offline from the pinned snapshot and check it (OMNI only).

        - The model loads from the snapshot's local files, on the CPU, in evaluation
          mode, and ``models.check_weights`` must find every weight of its file in place.
        - PyTorch uses ``settings.threads`` threads, which must equal the job's CPUs.
        - The espeak fallback uses the workspace build (``espeak.use_workspace_espeak``),
          and an invented word must get sounds from it.
        - The voices load from the snapshot's local files, and each file's SHA-256 is kept.

        Args:
            workspace: The workspace, with the snapshot and the espeak-ng build.
            pin: The Kokoro pin from ``configs/pilot/models.yaml``.
            settings: x1's settings.

        Returns:
            Kokoro, with ``identity`` filled in.

        Raises:
            StepError: If the job's CPUs differ from the threads, a weight is not in place,
                the espeak fallback is off or does not voice the invented word, or a voice
                file is missing.
        """
        import torch
        from kokoro import KModel, KPipeline
        from misaki.espeak import EspeakFallback
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if cpus != str(settings.threads):
            raise StepError(
                f"x1 renders with {settings.threads} threads, but the job has {cpus} CPUs"
            )
        torch.set_num_threads(settings.threads)
        snapshot = local_snapshot(pin, workspace.hf_home)
        weights, config_file = snapshot / WEIGHTS_FILE, snapshot / "config.json"
        # Local paths: given a repo only, the package would download without a revision.
        with caught_output():
            model = KModel(repo_id=pin.repo_id, config=str(config_file), model=str(weights))
        model = model.to("cpu").eval()
        check_eval(model)
        parts = torch.load(weights, map_location="cpu", weights_only=True)
        state = {
            f"{part}.{name.removeprefix('module.')}": tensor
            for part, tensors in parts.items()
            for name, tensor in tensors.items()
        }
        weight_check = check_weights(model, state)
        if not weight_check.ok:
            raise StepError(
                f"Kokoro's weights are not in place: {len(weight_check.missing)} missing,"
                f" {len(weight_check.unused)} unused, {len(weight_check.changed)} changed"
            )
        check_eval(model)
        use_workspace_espeak(workspace)
        with caught_output():
            pipeline = KPipeline(lang_code=settings.language, repo_id=pin.repo_id, model=model)
        fallback = pipeline.g2p.fallback
        if not isinstance(fallback, EspeakFallback):
            raise StepError("Kokoro's espeak fallback is off, so unknown words would be silent")
        voices = {name: snapshot / "voices" / f"{name}.pt" for name in settings.all_voices}
        for name, path in voices.items():
            if not path.is_file():
                raise StepError(f"the voice file of {name} is missing from the snapshot")
        vocabulary = json.loads(config_file.read_text(encoding="utf-8"))["vocab"]
        espeak = EspeakWrapper()
        library = Path(str(espeak.library_path))
        identity = {
            "repo_id": pin.repo_id,
            "revision": pin.revision,
            "weights_sha256": sha256_file(weights),
            "config_sha256": sha256_file(config_file),
            "voices": {name: sha256_file(path) for name, path in voices.items()},
            "weights_check": {
                "tensors": weight_check.tensors,
                "identity_norms": len(weight_check.identity_norms),
                "fixed_buffers": list(weight_check.fixed_buffers),
            },
            "espeak_version": ".".join(str(part) for part in espeak.version),
            "espeak_library": (
                str(library.relative_to(workspace.root))
                if library.is_relative_to(workspace.root)
                else str(library)
            ),
            "fallback": type(fallback).__name__,
            "vocabulary": len(vocabulary),
            "torch_threads": torch.get_num_threads(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "training": model.training,
        }
        kokoro = cls(pipeline, vocabulary, voices, settings.speed, identity)
        invented = kokoro.front_end(INVENTED_WORD)
        if not (len(invented.words) == 1 and invented.words[0].voiced and invented.words[0].espeak):
            raise StepError(f"the espeak fallback does not voice the invented word {INVENTED_WORD}")
        return kokoro

    def front_end(self, text: str) -> FrontEnd:
        """Run Kokoro's text front end on one part's text, without audio.

        Args:
            text: A part's text from ``part_text``: words joined by single spaces.

        Returns:
            What the front end made of it. Prints and warnings are caught and counted.

        Raises:
            ValueError: If the front end's tokens do not form exactly one group per
                written word.
        """
        written = text.split()
        with caught_output() as caught:
            _, tokens = self.pipeline.g2p(text)
        groups = group_words(tokens)
        if len(groups) != len(written):
            raise ValueError(
                f"the front end gave {len(groups)} word groups for {len(written)} written words"
            )
        # As Kokoro's pipeline joins them (KPipeline.tokens_to_ps), missing phonemes as ''.
        phonemes = "".join(
            (token.phonemes or "") + (" " if token.whitespace else "") for token in tokens
        ).strip()
        words = []
        for group in groups:
            sounds = "".join(token.phonemes or "" for token in group)
            ratings = [rating for token in group for rating in token_ratings(token)]
            known = [rating for rating in ratings if rating is not None]
            words.append(
                WordSound(
                    voiced=word_has_sound(sounds, self.vocabulary),
                    espeak=ESPEAK_RATING in ratings,
                    brackets="(" in sounds or ")" in sounds,
                    tokens=len(group),
                    rating=min(known) if known else None,
                    tags=tuple(str(token.tag) for token in group),
                )
            )
        outside = sorted(
            f"U+{ord(character):04X}"
            for character in set(phonemes)
            if character not in self.vocabulary
        )
        return FrontEnd(
            phonemes=phonemes,
            words=tuple(words),
            outside=tuple(outside),
            prints=caught.prints,
            warnings=caught.warnings,
        )

    def render(self, text: str, voice: str, seed: int) -> Render:
        """Read one part's text in one voice: one Kokoro call, seeded right before it runs.

        Args:
            text: A part's text from ``part_text``.
            voice: One of x1's voices.
            seed: The part's seed; PyTorch's global generator is seeded with it right
                before the pipeline runs, because all of Kokoro's random draws come from it.

        Returns:
            The render.

        Raises:
            ValueError: If the text fails the front end (a word without a sound, a sound
                Kokoro does not know, a bracket, or two calls' worth of phonemes), or Kokoro
                does not give exactly one result with audio from the front end's phonemes.
        """
        import torch

        front = self.front_end(text)
        if not front.ok:
            raise ValueError(
                f"a part fails the front end: {front.silent} words without a sound,"
                f" {len(front.outside)} unknown phoneme characters, brackets {front.brackets},"
                f" {len(front.phonemes)} phoneme characters"
            )
        path = self.voices[voice]
        torch.manual_seed(seed)
        with caught_output():
            results = list(self.pipeline(text, voice=str(path), speed=self.speed))
        if len(results) != 1 or results[0].audio is None:
            raise ValueError(f"Kokoro gave {len(results)} results for one part, not one with audio")
        result = results[0]
        if result.phonemes != front.phonemes:
            raise ValueError("Kokoro read other phonemes than its front end gave")
        durations = result.pred_dur
        return Render(
            audio=np.asarray(result.audio.numpy(), dtype=np.float32),
            phonemes=str(result.phonemes),
            first_frames=int(durations[0]),
            last_frames=int(durations[-1]),
        )


# Inputs ----------------------------------------------------------------------------


@dataclass(frozen=True)
class X1Inputs:
    """Everything x1 is made from, checked against the pins and against each other.

    Attributes:
        segments: The pinned segments, sorted by segment.
        parts: The words of each part of every segment (``segments.segment_parts``).
        voices: Each participant's voice (``assign_voices`` over all pinned labels).
        labels: The pinned labels: ``label`` and ``gender`` by session.
        tasks: Each session's task of the build (``assign_tasks``).
        sample: The pinned word-check sample.
        pins: What ``run.json`` records under ``inputs``; a build compares it with its
            check run's.
    """

    segments: pd.DataFrame
    parts: pd.DataFrame
    voices: dict[int, str]
    labels: pd.DataFrame
    tasks: dict[int, int]
    sample: pd.DataFrame
    pins: dict[str, Any]


def load_inputs(workspace: Workspace, config: dict, settings: X1Settings) -> X1Inputs:
    """Load x1's inputs and check that they fit together.

    The parts come from the cut words of the pinned segments run; they must have the
    segments' part start and end times, and x0's pinned manifest the same segments with
    the same number of parts.

    Args:
        workspace: The workspace.
        config: The data config.
        settings: x1's settings.

    Returns:
        The inputs.

    Raises:
        ValueError: If a pinned file does not match, the parts differ from the segments
            or from x0, or a participant with segments has no label.
    """
    segments = load_segments(workspace, config)
    parts = segment_parts(load_cut_words(workspace, config))
    labels = load_pinned_labels(workspace, config)
    sample = load_pinned_sample(workspace, config)
    x0 = load_view_manifest(workspace, config, "x0")
    by_segment = parts.groupby("segment", sort=True).agg(
        starts=("start", list), ends=("end", list), count=("position", "size")
    )
    table = segments.set_index("segment")
    if by_segment.index.tolist() != table.index.tolist():
        raise ValueError("the cut words do not give the pinned segments")
    for name in ("starts", "ends"):
        if any(
            list(map(float, ours)) != list(map(float, pinned))
            for ours, pinned in zip(by_segment[name], table[name], strict=True)
        ):
            raise ValueError(f"the parts' {name} differ from the pinned segments'")
    counts = x0.set_index("segment")["part_starts"].map(len)
    if counts.sort_index().to_dict() != by_segment["count"].to_dict():
        raise ValueError("x0's pinned manifest does not have the segments' parts")
    if not set(segments["session"].astype(int)) <= set(labels.index.astype(int)):
        raise ValueError("a participant with segments has no label")
    words_path = workspace.resolve(config["segments"]["run"]) / "words.parquet"
    checksums = workspace.edaic / SESSIONS_CHECKSUMS
    pins = {
        name: {"file": config[key]["run"], "sha256": config[key]["sha256"]}
        for name, key in (
            ("segments", "segments"),
            ("labels", "labels"),
            ("wordcheck_sample", "wordcheck_sample"),
        )
    }
    pins["x0"] = {
        "file": config["view_runs"]["x0"]["run"],
        "sha256": config["view_runs"]["x0"]["sha256"],
    }
    pins["cut_words"] = {
        "file": str(Path(config["segments"]["run"]) / "words.parquet"),
        "sha256": sha256_file(words_path),
    }
    pins["transcripts"] = {
        "file": str(Path("edaic") / SESSIONS_CHECKSUMS),
        "sha256": sha256_file(checksums),
    }
    return X1Inputs(
        segments=segments,
        parts=parts,
        voices=assign_voices(labels, settings.voices),
        labels=labels,
        tasks=assign_tasks(Counter(segments["session"].to_numpy(dtype=np.int64).tolist())),
        sample=sample,
        pins=pins,
    )
