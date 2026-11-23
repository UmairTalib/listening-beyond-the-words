"""Check the re-voiced clip x1 before it is built: a dry run and a test render (CPU job).

Submit from an OMNI login node:
    scripts/slurm/submit.sh check_revoice

Reads the pinned segments, cut words, labels, word-check sample and x0 manifest, every
pilot session's transcript lines and x1's settings (``views.x1`` in
configs/pilot/data.yaml), and loads Kokoro offline from the pinned snapshot (pilot
protocol, section "Clip versions", bullet "Re-voiced clip").

1. Inputs: the parts of the cut words have the pinned segments' parts and times, and the
   transcript lines give back every segment's 40 words.
2. Dry run: Kokoro's text front end reads every part, without audio. Every word must get
   a sound, and no part may hold a bracket, need two Kokoro calls (over 510 phoneme
   characters), use a sound outside Kokoro's vocabulary or have ( or ) in its phonemes.
   If a rule fails, the job stops before rendering and prints counts by text-free classes
   of the words concerned (digit, apostrophe, capitals, one letter, part edge, number,
   tokens, misaki's rating, spaCy tags), from which a general fix of the text mapping is
   chosen. The counts from the transcript (words, capitals, apostrophes, digits) are
   taken on the words as written. Printed, never stop rules: the numbers written out in
   words; in each part with one, a word-by-word comparison with the words as written,
   numbers not written out (rule of 2026-11-23); and the words that keep a sound
   although misaki read a piece of them as punctuation.
3. Test render: 20 segments outside the word-check sample, drawn with seed 106, are
   rendered exactly as the build renders them (``revoice.render_segment``), and the first
   2 by ID once more in the same process; both renders must be identical. Prints the
   timing, the estimate for the build and how the trim cut the parts.

Writes $LBW_WORKSPACE/derived/revoice_check/run-<job id>/ with
  dry_run.parquet  one row per part: counts, and its words without a sound, if any
  clips/NN.wav     the first 10 test segments in draw order at a common loudness, to hear
  key.tsv          clip, segment, voice and the text read
  test.parquet     each test segment's voice and the SHA-256 of its 16-bit samples
  errors.log       if the job fails: traceback frames and error types, never a message (a
                   job that fails before this folder exists prints them to stderr)
  run.json         commit, config, settings, inputs, model, environment, counts and
                   timings; status "done" only if every rule passed and the repeats matched
dry_run.parquet and key.tsv hold participant text: only the researcher opens them, on
OMNI. The job prints aggregate numbers only: never text, never one segment's values.
"""

import contextlib
import os
import re
import socket
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from listening_beyond_words.clips import audio_sha256, listening_copy, load_cut_words, read_lines
from listening_beyond_words.models import load_pins
from listening_beyond_words.revoice import (
    MAX_PHONEMES,
    TASKS,
    FrontEnd,
    Kokoro,
    PartText,
    RenderedSegment,
    WordPlaces,
    X1Inputs,
    X1Settings,
    draw_test_segments,
    load_inputs,
    part_text,
    render_segment,
    segment_words,
    word_classes,
)
from listening_beyond_words.runinfo import (
    Progress,
    StepError,
    environment,
    git_state,
    quiet_unraisable,
    record_failure,
    sha256_file,
    stop_on_sigterm,
    utc_now,
    write_json,
)
from listening_beyond_words.settings import get_workspace
from listening_beyond_words.transcripts import normalise
from listening_beyond_words.views import PCM16_SCALE

CONFIG = Path("configs/pilot/data.yaml")
MODELS = Path("configs/pilot/models.yaml")
PACKAGES = (
    "kokoro",
    "misaki",
    "spacy",
    "en-core-web-sm",
    "thinc",
    "phonemizer-fork",
    "espeakng-loader",
    "num2words",
    "torch",
    "transformers",
    "soxr",
    "soundfile",
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "pyyaml",
)
YEAR = re.compile(r"(?<!\d)\d{4}(?!\d)")
# Counts of the comparison with the words as written, per part (without_the_rule).
CHANGED = (
    "numbers_without_a_sound",
    "numbers_with_a_sound_read_otherwise",
    "other_words_read_otherwise",
    "other_words_read_otherwise_same_tags",
    "compared_prints",
    "compared_warnings",
)


def classes_line(found: list[dict[str, Any]]) -> str:
    """Counts of text-free word classes, as one printed line."""
    counts: Counter[str] = Counter()
    for word in found:
        for name in ("digit", "apostrophe", "capitals", "one_letter", "part_edge", "number"):
            counts[name] += bool(word[name])
        counts[f"{word['tokens']} token(s)"] += 1
        counts[f"rating {word['rating']}"] += 1
        counts[f"tags {word['tags']}"] += 1
    return ", ".join(f"{name} {n}" for name, n in sorted(counts.items()))


def check_words(inputs: X1Inputs, lines: dict[int, list[str]], state: Progress) -> None:
    """Every segment's parts, read from its transcript lines, give back its 40 words."""
    texts = dict(zip(inputs.segments["segment"], inputs.segments["text"], strict=True))
    for segment, group in inputs.parts.groupby("segment", sort=True):
        state.segment = str(segment)
        session = int(group["session"].iloc[0])
        if " ".join(segment_words(group, lines[session])) != texts[segment]:
            raise StepError(f"segment {segment}: the transcript lines do not give its words")
    state.segment = None
    print(
        f"inputs: {len(inputs.parts):,} parts of {len(inputs.segments):,} segments; their"
        " times are the pinned segments', and the transcript lines give back every"
        " segment's words"
    )


def without_the_rule(kokoro: Kokoro, text: PartText, front: FrontEnd) -> dict[str, int]:
    """Compare a part's words read with those as written, numbers not written out.

    Counts only, word by word, against the rule of 2026-11-23. Numbers: those that would
    have had no sound without the rule, and, apart from them, those that would have had a
    sound but other phonemes (after an apostrophe, as in "'05", misaki reads digits by
    another rule, or reads the apostrophe as a quotation mark). Other words whose phonemes the rule changed, and how many of them kept
    their spaCy tags: spaCy may tag a word differently next to a number written out, and
    misaki reads some words, such as "the" or "to", by the sound that follows them, which
    a number without a sound did not give. Also the prints and warnings caught while the
    front end read the words as written.
    """
    before = kokoro.front_end(text.written_text)
    counts = dict.fromkeys(CHANGED, 0)
    counts["compared_prints"], counts["compared_warnings"] = before.prints, before.warnings
    k = 0
    for i, count in enumerate(text.written_out):
        if count:
            silent = not before.words[i].voiced
            spoken = " ".join(front.word_phonemes[k : k + count])
            counts["numbers_without_a_sound"] += silent
            counts["numbers_with_a_sound_read_otherwise"] += not silent and (
                before.word_phonemes[i] != spoken
            )
            k += count
        else:
            changed = before.word_phonemes[i] != front.word_phonemes[k]
            counts["other_words_read_otherwise"] += changed
            counts["other_words_read_otherwise_same_tags"] += (
                changed and before.words[i].tags == front.words[k].tags
            )
            k += 1
    return counts


def dry_run(
    kokoro: Kokoro,
    inputs: X1Inputs,
    lines: dict[int, list[str]],
    places: WordPlaces,
    state: Progress,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Run Kokoro's text front end on every part; return the rows, counts and failed rules."""
    rows = []
    silent_classes: list[dict[str, Any]] = []
    bracket_classes: list[dict[str, Any]] = []
    punctuation_classes: list[dict[str, Any]] = []
    outside: Counter[str] = Counter()
    edges: Counter[str] = Counter()
    for row in inputs.parts.itertuples(index=False):
        session, segment = int(row.session), str(row.segment)
        line, first, last = int(row.line), int(row.first), int(row.last)
        state.segment, state.part = segment, int(row.position)
        text = part_text(lines[session][line], first, last)
        front = kokoro.front_end(text.text)
        written, read = text.written_text.split(), text.text.split()
        described = [
            word_classes(word, edge=k in (0, len(read) - 1))
            | {
                "number": number,
                "tokens": sound.tokens,
                "rating": sound.rating,
                "tags": "+".join(sound.tags),
            }
            for k, (word, number, sound) in enumerate(
                zip(read, text.in_number, front.words, strict=True)
            )
        ]
        as_written = [word_classes(word, edge=False) for word in written]
        numbers = [w for w, count in zip(written, text.written_out, strict=True) if count]
        silent_classes += [d for d, s in zip(described, front.words, strict=True) if not s.voiced]
        bracket_classes += [
            d | {"espeak": s.espeak}
            for d, s in zip(described, front.words, strict=True)
            if s.brackets
        ]
        punctuation_classes += [
            d for d, s in zip(described, front.words, strict=True) if s.voiced and s.punctuation
        ]
        changed = (
            without_the_rule(kokoro, text, front) if text.numbers else dict.fromkeys(CHANGED, 0)
        )
        outside.update(front.outside)
        if text.split_start:
            edges[places.cause(session, segment, line, first, first - 1)] += 1
        if text.split_end:
            edges[places.cause(session, segment, line, last, last + 1)] += 1
        rows.append(
            {
                "segment": segment,
                "session": session,
                "position": int(row.position),
                "words": len(written),
                "words_read": len(read),
                "numbers": text.numbers,
                "number_words": sum(text.written_out),
                "numbers_as_years": sum(len(normalise(word)) == 4 for word in numbers),
                "numbers_with_an_apostrophe": sum("'" in word for word in numbers),
                "phoneme_chars": len(front.phonemes),
                "silent": front.silent,
                "espeak": sum(word.espeak for word in front.words),
                "punctuation": sum(s.voiced and s.punctuation for s in front.words),
                "lost_pieces": sum(s.voiced and s.lost_piece for s in front.words),
                "bracket_phonemes": front.brackets,
                "outside": list(front.outside),
                "written": text.written,
                "unmatched": text.unmatched,
                "split_start": text.split_start,
                "split_end": text.split_end,
                "typographic": text.typographic,
                "bracket_tokens": text.brackets,
                "capitals": sum(c["capitals"] for c in as_written),
                "apostrophes": sum(c["apostrophe"] for c in as_written),
                "digits": sum(c["digit"] for c in as_written),
                "years": sum(bool(YEAR.search(word)) for word in written),
                "read_digits": sum(d["digit"] for d in described),
                **changed,
                "prints": front.prints,
                "warnings": front.warnings,
                "silent_words": [w for w, s in zip(read, front.words, strict=True) if not s.voiced],
            }
        )
    state.segment = state.part = None
    table = pd.DataFrame(rows)
    sizes = table["words"].clip(upper=4).value_counts()
    counts: dict[str, Any] = {
        "parts": len(table),
        "words": int(table["words"].sum()),
        "parts_by_words": {
            f"{n}{'+' if n == 4 else ''}": int(sizes.get(n, 0)) for n in range(1, 5)
        },
        "most_phoneme_chars": int(table["phoneme_chars"].max()),
        "espeak_words": int(table["espeak"].sum()),
        "capitals": int(table["capitals"].sum()),
        "apostrophes": int(table["apostrophes"].sum()),
        "typographic_tokens": int(table["typographic"].sum()),
        "digit_words": int(table["digits"].sum()),
        "four_digit_numbers": int(table["years"].sum()),
        "written_tokens": int(table["written"].sum()),
        "unmatched_tokens": int(table["unmatched"].sum()),
        "split_edges": dict(sorted(edges.items())),
        "prints_caught": int(table["prints"].sum()),
        "warnings_caught": int(table["warnings"].sum()),
        "words_read": int(table["words_read"].sum()),
        "numbers_written_out": int(table["numbers"].sum()),
        "numbers_as_years": int(table["numbers_as_years"].sum()),
        "numbers_with_an_apostrophe": int(table["numbers_with_an_apostrophe"].sum()),
        "number_words": int(table["number_words"].sum()),
        "words_read_with_a_digit": int(table["read_digits"].sum()),
        "words_with_a_piece_read_as_punctuation": int(table["punctuation"].sum()),
        "words_that_lost_a_piece": int(table["lost_pieces"].sum()),
        "without_the_rule": {
            "parts": int((table["numbers"] > 0).sum()),
            "numbers_without_a_sound": int(table["numbers_without_a_sound"].sum()),
            "numbers_with_a_sound_read_otherwise": int(
                table["numbers_with_a_sound_read_otherwise"].sum()
            ),
            "other_words_read_otherwise": int(table["other_words_read_otherwise"].sum()),
            "other_words_read_otherwise_same_tags": int(
                table["other_words_read_otherwise_same_tags"].sum()
            ),
            "parts_with_other_words_read_otherwise": int(
                (table["other_words_read_otherwise"] > 0).sum()
            ),
            "prints_caught": int(table["compared_prints"].sum()),
            "warnings_caught": int(table["compared_warnings"].sum()),
        },
    }
    stops = {
        "words_without_a_sound": int(table["silent"].sum()),
        "bracket_tokens": int(table["bracket_tokens"].sum()),
        "parts_over_510_phoneme_chars": int((table["phoneme_chars"] > MAX_PHONEMES).sum()),
        "phoneme_chars_outside_the_vocabulary": int(sum(outside.values())),
        "parts_with_brackets_in_phonemes": int(table["bracket_phonemes"].sum()),
    }
    counts["stop_rules"] = stops
    by_words = counts["parts_by_words"]
    print(
        f"dry run: {counts['parts']:,} parts, {counts['words']:,} words; parts with 1, 2, 3"
        f" and 4+ words: {by_words['1']:,}, {by_words['2']:,}, {by_words['3']:,},"
        f" {by_words['4+']:,}; most phoneme characters in a part {counts['most_phoneme_chars']}"
        f" (Kokoro splits above {MAX_PHONEMES})"
    )
    print(
        f"words sent to espeak {counts['espeak_words']:,}; in capitals {counts['capitals']:,};"
        f" with an apostrophe {counts['apostrophes']:,}; with a digit {counts['digit_words']:,};"
        f" 4-digit numbers (read as years) {counts['four_digit_numbers']:,}; tokens with a"
        f" typographic apostrophe {counts['typographic_tokens']:,}"
    )
    print(
        f"tokens read as written {counts['written_tokens']:,}; read as stored words because"
        f" the written form did not turn back into them {counts['unmatched_tokens']:,};"
        f" misaki prints caught {counts['prints_caught']:,}, warnings caught"
        f" {counts['warnings_caught']:,}"
    )
    print(
        f"numbers written out in words {counts['numbers_written_out']:,} (as years"
        f" {counts['numbers_as_years']:,}; with an apostrophe"
        f" {counts['numbers_with_an_apostrophe']:,}), read as {counts['number_words']:,} words;"
        f" words read {counts['words_read']:,}, of which with a digit"
        f" {counts['words_read_with_a_digit']:,}"
    )
    without = counts["without_the_rule"]
    print(
        f"without that rule, in the {without['parts']:,} parts with a number: numbers without a"
        f" sound {without['numbers_without_a_sound']:,}; other numbers read otherwise"
        f" {without['numbers_with_a_sound_read_otherwise']:,}; other words read otherwise"
        f" {without['other_words_read_otherwise']:,} (of which with the same spaCy tags"
        f" {without['other_words_read_otherwise_same_tags']:,}), in"
        f" {without['parts_with_other_words_read_otherwise']:,} parts; misaki prints caught"
        f" {without['prints_caught']:,}, warnings caught {without['warnings_caught']:,}"
        " (counted, not a stop rule)"
    )
    print(
        "words with a sound but a piece read as punctuation"
        f" {counts['words_with_a_piece_read_as_punctuation']:,}, of which a letter or digit is"
        f" not spoken {counts['words_that_lost_a_piece']:,} (counted, not a stop rule)"
    )
    split = sum(edges.values())
    print(
        f"tokens split at a part edge: {split:,}"
        + (
            " (" + "; ".join(f"{cause} {n:,}" for cause, n in sorted(edges.items())) + ")"
            if split
            else ""
        )
    )
    failed = [name for name, n in stops.items() if n]
    print(
        "stop rules: "
        + ", ".join(f"{name.replace('_', ' ')} {n:,}" for name, n in stops.items())
        + (": passed" if not failed else ": STOPPED")
    )
    if silent_classes:
        print(f"words without a sound, by class: {classes_line(silent_classes)}")
    if punctuation_classes:
        print(
            "words with a sound but a piece read as punctuation, by class:"
            f" {classes_line(punctuation_classes)}"
        )
    if bracket_classes:
        espeak = sum(word["espeak"] for word in bracket_classes)
        print(
            f"words with ( or ) in their phonemes: {len(bracket_classes)}, of which {espeak}"
            f" went to espeak (a language flag); by class: {classes_line(bracket_classes)}"
        )
    if outside:
        print(
            "phoneme characters outside Kokoro's vocabulary: "
            + ", ".join(f"{code} {n}" for code, n in sorted(outside.items()))
        )
    return table, counts, failed


def test_render(
    kokoro: Kokoro,
    inputs: X1Inputs,
    lines: dict[int, list[str]],
    settings: X1Settings,
    run_dir: Path,
    state: Progress,
    load_seconds: float,
) -> dict[str, Any]:
    """Render the test segments, repeat the first by ID, write what is heard; return counts."""
    drawn = draw_test_segments(
        inputs.segments["segment"].tolist(),
        set(inputs.sample["segment"]),
        settings.test_segments,
        settings.test_seed,
    )
    parts = {str(s): group for s, group in inputs.parts.groupby("segment", sort=False)}
    session_of = {
        str(s): int(n)
        for s, n in zip(inputs.segments["segment"], inputs.segments["session"], strict=True)
    }

    def render(segment: str) -> RenderedSegment:
        session = session_of[segment]
        return render_segment(
            kokoro, parts[segment], lines[session], inputs.voices[session], segment, settings
        )

    rendered: dict[str, RenderedSegment] = {}
    seconds: dict[str, float] = {}
    for segment in drawn:
        state.segment = segment
        start = time.perf_counter()
        rendered[segment] = render(segment)
        seconds[segment] = time.perf_counter() - start
    repeated = sorted(drawn)[: settings.test_repeats]
    identical = 0
    for segment in repeated:
        state.segment = segment
        again, first = render(segment), rendered[segment]
        identical += all(
            np.array_equal(a, b) for a, b in zip(first.renders, again.renders, strict=True)
        ) and np.array_equal(first.samples, again.samples)
    state.segment = None

    clips = run_dir / "clips"
    clips.mkdir()
    key, test = [], []
    for number, segment in enumerate(drawn, start=1):
        result, session = rendered[segment], session_of[segment]
        heard = number <= settings.test_heard
        if heard:
            name = f"{number:02d}.wav"
            copy, _ = listening_copy(result.samples.astype(np.float32) / PCM16_SCALE)
            sf.write(clips / name, copy, settings.rate, subtype="PCM_16")
            key.append(
                {
                    "clip": name,
                    "segment": segment,
                    "voice": inputs.voices[session],
                    "text": " | ".join(str(p["text"]) for p in result.parts),
                }
            )
        test.append(
            {
                "draw": number,
                "segment": segment,
                "session": session,
                "voice": inputs.voices[session],
                "heard": heard,
                "samples": len(result.samples),
                "audio_sha256": audio_sha256(result.samples),
                "render_seconds": seconds[segment],
            }
        )
    pd.DataFrame(key, columns=["clip", "segment", "voice", "text"]).to_csv(
        run_dir / "key.tsv", sep="\t", index=False
    )
    pd.DataFrame(test).to_parquet(run_dir / "test.parquet", index=False)

    rate = settings.rate
    total = sum(seconds.values())
    words = sum(int(p["n_words"]) for s in drawn for p in rendered[s].parts)
    audio = sum(len(rendered[s].samples) for s in drawn) / rate
    per_word = total / words
    all_words = int(inputs.parts["n_words"].sum())
    per_task: Counter[int] = Counter()
    for session in session_of.values():
        per_task[inputs.tasks[session]] += 1
    heaviest = max(per_task.values())
    heaviest_words = heaviest * all_words / len(session_of)  # its segments times the mean words
    heaviest_minutes = (per_word * heaviest_words + load_seconds) / 60
    records = [p for s in drawn for p in rendered[s].parts]
    shortest_ms = min(int(p["kept"]) for p in records) / rate * 1000
    lowest = min(
        min(float(p["part_peak"]) for p in rendered[s].parts)
        / max(float(p["part_peak"]) for p in rendered[s].parts)
        for s in drawn
    )
    ms = lambda samples: np.asarray(samples, dtype=float) / rate * 1000  # noqa: E731
    removed_start = ms([p["removed_start"] for p in records])
    removed_end = ms([p["removed_end"] for p in records])
    first_token = np.asarray([p["first_frames"] for p in records], dtype=float) * 25.0
    last_token = np.asarray([p["last_frames"] for p in records], dtype=float) * 25.0
    short_margins = sum(
        int(p["margin_start"]) < settings.trim_margin or int(p["margin_end"]) < settings.trim_margin
        for p in records
    )
    counts = {
        "segments": len(drawn),
        "parts": len(records),
        "render_seconds": round(total, 2),
        "audio_seconds": round(audio, 2),
        "real_time_factor": round(total / audio, 4),
        "seconds_per_word": per_word,
        "load_seconds": round(load_seconds, 1),
        "estimated_cpu_hours": round(per_word * all_words * settings.threads / 3600, 2),
        "heaviest_task_segments": heaviest,
        "heaviest_task_minutes": round(heaviest_minutes, 1),
        "repeated": len(repeated),
        "repeats_identical": identical,
        "heard": min(settings.test_heard, len(drawn)),
        "shortest_part_ms": round(shortest_ms, 1),
        "lowest_part_peak_ratio": round(lowest, 4),
        "removed_start_ms": {
            "median": float(np.median(removed_start)),
            "max": float(removed_start.max()),
        },
        "removed_end_ms": {
            "median": float(np.median(removed_end)),
            "max": float(removed_end.max()),
        },
        "first_token_ms": {
            "median": float(np.median(first_token)),
            "max": float(first_token.max()),
        },
        "last_token_ms": {"median": float(np.median(last_token)), "max": float(last_token.max())},
        "parts_with_a_short_margin": short_margins,
    }
    print(
        f"test render: {len(drawn)} segments outside the word-check sample (seed"
        f" {settings.test_seed}), {len(records)} parts, {total:.1f} s for {audio:.1f} s of"
        f" audio (real-time factor {total / audio:.3f}, {per_word * 1000:.0f} ms per word,"
        f" {settings.threads} threads); Kokoro loaded in {load_seconds:.0f} s"
    )
    print(
        f"estimate for all {len(session_of):,} segments: {counts['estimated_cpu_hours']:.1f} CPU"
        f" hours; the heaviest of {TASKS} tasks holds {heaviest} segments, about"
        f" {heaviest_minutes:.0f} min with loading (ask for 1.5 times that, at most 6 h)"
    )
    print(
        f"repeats: {identical} of {len(repeated)} segments rendered twice are identical, at"
        f" 24 kHz and as 16-bit samples"
    )
    print(
        f"trim: shortest kept part {shortest_ms:.0f} ms; lowest part peak / loudest part of"
        f" its clip {lowest:.3f}; removed at the start median {np.median(removed_start):.0f} ms"
        f" (max {removed_start.max():.0f}), boundary token before the speech median"
        f" {np.median(first_token):.0f} ms (max {first_token.max():.0f}); removed at the end"
        f" median {np.median(removed_end):.0f} ms (max {removed_end.max():.0f}), boundary"
        f" token after it median {np.median(last_token):.0f} ms (max {last_token.max():.0f});"
        f" parts whose margin was cut short by the render's edge {short_margins}"
    )
    return counts


def check(state: Progress) -> None:
    """The job: inputs, dry run, test render."""
    stop_on_sigterm()
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise StepError("submit this as a job: scripts/slurm/submit.sh check_revoice")
    git = git_state()
    if git["dirty"]:
        raise StepError("the checkout has uncommitted changes; commit them on the Mac and pull")
    started = time.monotonic()
    workspace = get_workspace()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    settings = X1Settings.from_config(config)
    run_dir = workspace.derived / "revoice_check" / f"run-{job}"
    run_dir.mkdir(parents=True, exist_ok=False)
    state.run_dir = run_dir
    info: dict[str, Any] = {
        "step": "check_revoice",
        "status": "running",
        "started_utc": utc_now(),
        "git": git,
        "config": {"file": str(CONFIG), "sha256": sha256_file(CONFIG)},
        "settings": {
            "x1": config["views"]["x1"],
            "peak_limit": config["views"]["peak_limit"],
            "fade_ms": config["x0_cut"]["fade_ms"],
            "tasks": TASKS,
            "resolved": asdict(settings),
        },
        "slurm": {
            "job_id": job,
            "node": socket.gethostname(),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
    }
    state.info = info
    write_json(run_dir / "run.json", info)
    inputs = load_inputs(workspace, config, settings)
    info["inputs"] = inputs.pins
    loading = time.monotonic()
    kokoro = Kokoro.load(workspace, load_pins(MODELS)["kokoro"], settings)
    load_seconds = time.monotonic() - loading
    info.update(model=kokoro.identity, environment=environment(PACKAGES))
    info["environment"]["libsndfile"] = sf.__libsndfile_version__
    write_json(run_dir / "run.json", info)
    sessions = sorted(set(inputs.segments["session"].astype(int)))
    lines = {session: read_lines(workspace.edaic / "sessions", session) for session in sessions}
    check_words(inputs, lines, state)

    places = WordPlaces(load_cut_words(workspace, config))
    table, counts, failed = dry_run(kokoro, inputs, lines, places, state)
    table.to_parquet(run_dir / "dry_run.parquet", index=False)
    info["dry_run"] = counts
    outputs = {"dry_run.parquet": sha256_file(run_dir / "dry_run.parquet")}
    info["outputs"] = outputs
    if failed:
        raise StepError(f"the dry run stopped before any render: {', '.join(failed)}")

    info["test_render"] = test_render(kokoro, inputs, lines, settings, run_dir, state, load_seconds)
    for name in (
        "test.parquet",
        "key.tsv",
        *(f"clips/{p.name}" for p in sorted((run_dir / "clips").iterdir())),
    ):
        outputs[name] = sha256_file(run_dir / name)
    repeats = info["test_render"]
    if repeats["repeats_identical"] != repeats["repeated"]:
        raise StepError(
            f"the repeat check failed: {repeats['repeats_identical']} of {repeats['repeated']}"
            " segments rendered twice are identical"
        )
    minutes = (time.monotonic() - started) / 60
    info.update(status="done", finished_utc=utc_now(), minutes=round(minutes, 2))
    write_json(run_dir / "run.json", info)
    print(f"clips to hear: {(run_dir / 'clips').relative_to(workspace.root)}")
    print(f"run: {run_dir.relative_to(workspace.root)}")
    print(f"test.parquet sha256: {outputs['test.parquet']}")
    print(f"done in {minutes:.1f} min")


def main() -> None:
    os.umask(0o077)
    sys.unraisablehook = quiet_unraisable  # a finaliser's message could hold a word
    progress = Progress()
    try:
        check(progress)
        return
    except BaseException as caught:
        error = caught  # reported below, outside this clause, so that nothing chains to it
    with contextlib.suppress(BaseException):  # nothing here may print the error's message
        failure = record_failure(error, progress)
        print(failure.line)
        if failure.frames is not None:  # no errors.log holds them: frames and types only
            print(failure.frames, end="", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
