"""Check the clip-version tools on a GPU node before any clip is built (GPU job).

Submit from an OMNI login node, after fetching the models:
    scripts/slurm/submit.sh check_view_tools

Uses one fixed public text, never participant data: the first five Harvard sentences
(IEEE recommended practice for speech quality measurements, 1969). Kokoro reads them as
x1 will read a part: capitals and apostrophes kept, every other mark turned into a
space. Each check runs even when another one fails:
  kokoro_weights   Kokoro loads offline from the pinned snapshot, its own loader put every
                   tensor of the weights file in place, and the file names every model
                   weight (Kokoro 0.9.4 skips mismatched weights without a word). The
                   only exception: the scale and shift weights that Kokoro adds to its
                   AdaIN norms (istftnet.py) are not in the file and must hold 1 and 0
  espeak_fallback  the espeak fallback for words outside Kokoro's lexicon is on, and an
                   invented word gets sounds (without it such words are dropped silently);
                   if it is off, the job logs the real reason, which Kokoro would hide
  kokoro_render   every candidate voice renders the text, and every word gets sounds
  faster_whisper   Whisper large-v3 runs offline in float16 on the GPU (beam 5,
                   temperature 0, no conditioning on earlier text, no voice-activity
                   filter) and reads the renders back with a WER of at most 0.10
  wav2vec2         the pinned wav2vec2 runs offline in float16 on the GPU with greedy
                   decoding and reads the renders back with a WER of at most 0.10
  opensmile        opensmile gives all 88 eGeMAPS values, all finite, on every render
A WER counts the errors over the renders of all voices at once, after the repository's
normalise on both sides; each voice's WER is reported too.

Writes $LBW_WORKSPACE/derived/check_view_tools/run-<job id>/ with
  voices/<voice>.wav  the text in each candidate voice, resampled to 16 kHz as x1 will
                      be and scaled to a common loudness; public text in synthetic
                      voices, so these files may be played on the Mac
  run.json            commit, model revisions, package versions, settings and results
Prints one line per check and the verdict, and exits with status 1 if a check fails.
"""

import os
import re
import socket
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import soundfile as sf
import soxr
import torch  # imported before faster_whisper, so that CTranslate2 finds PyTorch's cuBLAS

from listening_beyond_words.alignment import WordAligner
from listening_beyond_words.models import check_weights, load_pins, local_snapshot
from listening_beyond_words.runinfo import git_state, sha256_file, utc_now, write_json
from listening_beyond_words.settings import get_workspace
from listening_beyond_words.transcripts import normalise

PINS = Path("configs/pilot/models.yaml")
TEXT = (
    "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background. "
    "It's easy to tell the depth of a well. These days a chicken leg is a rare dish. "
    "Rice is often served in round bowls."
)
VOICES = ("af_heart", "af_bella", "af_sarah", "af_kore", "am_michael", "am_fenrir", "am_puck")
INVENTED_WORD = "zorblefax"  # in no lexicon, so only espeak can give it sounds
SEED = 0  # Kokoro's decoder adds random noise; seeding each render makes it repeat
SPEED = 1.0
KOKORO_RATE = 24_000
RATE = 16_000  # every view is stored at 16 kHz
MAX_WER = 0.10
EGEMAPS_VALUES = 88
NAMED_MEASURES = (
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "loudness_sma3_stddevNorm",
    "VoicedSegmentsPerSec",
)
WHISPER = {
    "language": "en",
    "beam_size": 5,
    "temperature": 0.0,  # one temperature: no fallback to random sampling
    "condition_on_previous_text": False,
    "vad_filter": False,
}
LISTENING_RMS = 0.1  # common loudness of the voice files (-20 dBFS) ...
LISTENING_PEAK = 0.99  # ... unless that would clip
PACKAGES = (
    "kokoro",
    "misaki",
    "espeakng-loader",
    "spacy",
    "faster-whisper",
    "ctranslate2",
    "transformers",
    "torch",
    "opensmile",
    "jiwer",
    "soxr",
)


@dataclass
class Check:
    """The outcome of one check: passed (True), failed (False) or not run (None)."""

    name: str
    ok: bool | None = None
    detail: str = "not run"
    numbers: dict[str, Any] = field(default_factory=dict)


def run(check: Check, step: Callable[[Check], None]) -> None:
    """Run one check; an error fails this check only, and its traceback goes to the log."""
    try:
        step(check)
    except Exception as error:
        traceback.print_exc()
        check.ok = False
        check.detail = f"{type(error).__name__}: {error}"


def x1_text(text: str) -> str:
    """The text as x1 gives it to Kokoro: capitals and apostrophes kept, other marks as spaces."""
    return " ".join(re.sub(r"[^A-Za-z0-9']+", " ", text).split())


def silent_words(tokens: list[Any]) -> tuple[int, list[str]]:
    """Count the written words of Kokoro's tokens and list those that got no sounds."""
    words: list[tuple[str, str]] = []
    written, sounds = "", ""
    for token in tokens:
        written += token.text
        sounds += token.phonemes or ""
        if token.whitespace:
            words.append((written, sounds))
            written, sounds = "", ""
    if written:
        words.append((written, sounds))
    spoken = [(w, s) for w, s in words if any(c.isalnum() for c in w)]
    return len(spoken), [w for w, s in spoken if not s.strip()]


def render(pipeline: Any, voice: Path, text: str) -> np.ndarray:
    """Kokoro's reading of the text in one voice, at 24 kHz, with the fixed seed."""
    torch.manual_seed(SEED)
    pieces = [
        result.audio.numpy()
        for result in pipeline(text, voice=str(voice), speed=SPEED)
        if result.audio is not None
    ]
    return np.concatenate(pieces).astype(np.float32) if pieces else np.zeros(0, np.float32)


def listening_copy(audio: np.ndarray) -> np.ndarray:
    """The audio at the common loudness, scaled down further if it would clip."""
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    gain = min(LISTENING_RMS / rms, LISTENING_PEAK / float(np.max(np.abs(audio))))
    return (audio * gain).astype(np.float32)


def score(check: Check, reference: str, heard: dict[str, str]) -> None:
    """Pass the check if the recogniser's WER over all renders is at most MAX_WER."""
    truth = normalise(reference)
    voices = list(heard)
    pooled = jiwer.process_words([truth] * len(voices), [normalise(heard[v]) for v in voices])
    per_voice = {v: round(jiwer.process_words(truth, normalise(heard[v])).wer, 4) for v in voices}
    check.ok = pooled.wer <= MAX_WER
    check.numbers.update(
        wer=round(pooled.wer, 4),
        substitutions=pooled.substitutions,
        deletions=pooled.deletions,
        insertions=pooled.insertions,
        hits=pooled.hits,
        wer_by_voice=per_voice,
        heard=heard,  # public text only
    )
    worst = max(per_voice, key=per_voice.__getitem__)
    check.detail = (
        f"WER {pooled.wer:.3f} over {len(voices)} renders ({pooled.substitutions} substituted,"
        f" {pooled.deletions} deleted, {pooled.insertions} inserted); highest {worst} {per_voice[worst]:.3f}"
    )


def greedy_read(aligner: WordAligner, audio: np.ndarray) -> str:
    """wav2vec2's reading of 16 kHz audio: the most likely letter in every frame."""
    inputs = aligner.processor(audio, sampling_rate=RATE, return_tensors="pt")
    with torch.inference_mode():
        logits = aligner.model(inputs.input_values.to(aligner.device).half()).logits
    return str(aligner.processor.batch_decode(torch.argmax(logits, dim=-1))[0])


def environment() -> dict[str, object]:
    """Python, package versions, CUDA, GPU and CPU threads of this job."""
    packages: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "packages": packages,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_threads": torch.get_num_threads(),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
    }


def main() -> None:
    os.umask(0o077)
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        sys.exit(
            "run this as a job (scripts/slurm/submit.sh check_view_tools): it must run offline"
        )
    workspace = get_workspace()
    hf_home = Path(os.environ["HF_HOME"])
    pins = load_pins(PINS)
    text = x1_text(TEXT)
    run_dir = (
        workspace.derived / "check_view_tools" / f"run-{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    voices_dir = run_dir / "voices"
    voices_dir.mkdir()
    info: dict[str, object] = {
        "step": "check_view_tools",
        "status": "running",
        "started_utc": utc_now(),
        "git": git_state(),
        "models": {
            name: {"repo_id": pins[name].repo_id, "revision": pins[name].revision}
            for name in ("kokoro", "faster_whisper", "wav2vec2_960h")
        },
        "environment": environment(),
        "settings": {
            "text": TEXT,
            "kokoro_text": text,
            "voices": list(VOICES),
            "invented_word": INVENTED_WORD,
            "seed": SEED,
            "speed": SPEED,
            "rate_hz": RATE,
            "whisper": WHISPER,
            "max_wer": MAX_WER,
            "listening_rms": LISTENING_RMS,
        },
        "slurm": {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname()},
    }
    write_json(run_dir / "run.json", info)
    names = (
        "kokoro_weights",
        "espeak_fallback",
        "kokoro_render",
        "faster_whisper",
        "wav2vec2",
        "opensmile",
    )
    checks = {name: Check(name) for name in names}
    kokoro: dict[str, Any] = {}
    renders: dict[str, np.ndarray] = {}  # voice -> the render at 16 kHz

    def kokoro_weights(check: Check) -> None:
        from kokoro import KModel

        pin = pins["kokoro"]
        snapshot = local_snapshot(pin, hf_home)
        weights = snapshot / "kokoro-v1_0.pth"
        # Local paths: given a repo only, the package would download without a revision.
        model = KModel(
            repo_id=pin.repo_id, config=str(snapshot / "config.json"), model=str(weights)
        )
        model = model.to("cpu").eval()  # x1 renders on CPU nodes; eval mode turns dropout off
        kokoro.update(model=model, snapshot=snapshot)
        parts = torch.load(weights, map_location="cpu", weights_only=True)
        state = {
            f"{part}.{name.removeprefix('module.')}": tensor
            for part, tensors in parts.items()
            for name, tensor in tensors.items()
        }
        result = check_weights(model, state)
        check.ok = result.ok
        check.numbers = {
            "tensors": result.tensors,
            "parts": sorted(parts),
            "missing": list(result.missing),
            "unused": list(result.unused),
            "changed": list(result.changed),
            "fixed_buffers": list(result.fixed_buffers),
            "identity_norms": list(result.identity_norms),
        }
        check.detail = (
            f"{result.tensors} tensors in {len(parts)} parts; missing {len(result.missing)},"
            f" unused {len(result.unused)}, changed {len(result.changed)};"
            f" AdaIN norm weights not in the file, at 1 and 0: {len(result.identity_norms)};"
            f" computed buffers equal to the file: {', '.join(result.fixed_buffers) or 'none'}"
        )
        for kind in ("missing", "unused", "changed"):
            for name in getattr(result, kind)[:20]:
                print(f"  kokoro weight {kind}: {name}")

    def espeak_fallback(check: Check) -> None:
        from kokoro import KPipeline
        from misaki.espeak import EspeakFallback

        # Kokoro's logger is switched off, so its "EspeakFallback not Enabled" warning
        # never shows: check the fallback object itself.
        pipeline = KPipeline(
            lang_code="a", repo_id=pins["kokoro"].repo_id, model=kokoro.get("model", False)
        )
        kokoro["pipeline"] = pipeline
        fallback = pipeline.g2p.fallback
        if fallback is None:
            # Kokoro swallowed the error, and phonemizer would replace it with "espeak not
            # installed": load the library directly so that its own error ends the check.
            from phonemizer.backend.espeak.wrapper import EspeakWrapper

            EspeakWrapper()  # misaki.espeak set the bundled library and data paths on import
            EspeakFallback(british=False)
        _, tokens = pipeline.g2p(INVENTED_WORD)
        sounds = "".join(token.phonemes or "" for token in tokens)
        check.ok = isinstance(fallback, EspeakFallback) and bool(sounds.strip())
        check.numbers = {"fallback": type(fallback).__name__, "invented_word_sounds": sounds}
        check.detail = f"fallback {type(fallback).__name__}; '{INVENTED_WORD}' gets '{sounds}'"

    def kokoro_render(check: Check) -> None:
        if "model" not in kokoro or "pipeline" not in kokoro:
            check.detail = "not run: Kokoro did not load"
            return
        pipeline, voices = kokoro["pipeline"], kokoro["snapshot"] / "voices"
        _, tokens = pipeline.g2p(text)
        words, silent = silent_words(tokens)
        seconds: dict[str, float] = {}
        failed: list[str] = []
        first = np.zeros(0, np.float32)
        for voice in VOICES:
            audio = render(pipeline, voices / f"{voice}.pt", text)
            if not (audio.size and np.isfinite(audio).all() and np.max(np.abs(audio)) > 1e-3):
                failed.append(voice)
                continue
            if not first.size:
                first = audio
            seconds[voice] = round(audio.size / KOKORO_RATE, 2)
            renders[voice] = soxr.resample(audio, KOKORO_RATE, RATE).astype(np.float32)
            sf.write(voices_dir / f"{voice}.wav", listening_copy(renders[voice]), RATE, "PCM_16")
        repeat = render(pipeline, voices / f"{next(iter(seconds))}.pt", text) if seconds else first
        identical = bool(first.size) and np.array_equal(first, repeat)
        check.ok = not silent and not failed
        check.numbers = {
            "words": words,
            "silent_words": silent,
            "failed_voices": failed,
            "seconds": seconds,
            "same_seed_identical": identical,
        }
        check.detail = (
            f"{len(seconds)} of {len(VOICES)} voices rendered, {words} words,"
            f" {len(silent)} without sounds{': ' + ', '.join(silent) if silent else ''};"
            f" same seed twice: {'identical' if identical else 'different'}"
        )

    def faster_whisper(check: Check) -> None:
        import ctranslate2
        from faster_whisper import WhisperModel

        snapshot = local_snapshot(pins["faster_whisper"], hf_home)
        model = WhisperModel(str(snapshot), device="cuda", compute_type="float16")
        check.numbers["cuda_compute_types"] = sorted(
            ctranslate2.get_supported_compute_types("cuda")
        )
        if not renders:
            check.detail = "loaded in float16 on the GPU; not run: no renders to read"
            return
        heard = {
            voice: " ".join(
                segment.text.strip() for segment in model.transcribe(audio, **WHISPER)[0]
            )
            for voice, audio in renders.items()
        }
        score(check, text, heard)

    def wav2vec2(check: Check) -> None:
        aligner = WordAligner(local_snapshot(pins["wav2vec2_960h"], hf_home))  # float16, GPU
        weight = next(aligner.model.parameters())
        check.numbers.update(dtype=str(weight.dtype), device=str(weight.device))
        if not renders:
            check.detail = (
                f"loaded as {weight.dtype} on {weight.device}; not run: no renders to read"
            )
            return
        score(check, text, {voice: greedy_read(aligner, audio) for voice, audio in renders.items()})
        if weight.dtype != torch.float16 or weight.device.type != "cuda":
            check.ok = False
            check.detail += f"; ran as {weight.dtype} on {weight.device}"

    def opensmile_values(check: Check) -> None:
        import opensmile

        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        if not renders:
            check.detail = "loaded; not run: no renders to measure"
            return
        frames = {voice: smile.process_signal(audio, RATE) for voice, audio in renders.items()}
        bad = [
            voice
            for voice, frame in frames.items()
            if frame.shape != (1, EGEMAPS_VALUES) or not np.isfinite(frame.to_numpy()).all()
        ]
        columns = next(iter(frames.values())).columns
        absent = [measure for measure in NAMED_MEASURES if measure not in columns]
        check.ok = not bad and not absent
        check.numbers = {
            "values": {voice: frame.shape[1] for voice, frame in frames.items()},
            "renders_with_missing_values": bad,
            "named_measures_absent": absent,
            "named_measures": {
                voice: {
                    m: round(float(frame[m].iloc[0]), 4) for m in NAMED_MEASURES if m in columns
                }
                for voice, frame in frames.items()
            },
        }
        check.detail = (
            f"{len(frames) - len(bad)} of {len(frames)} renders with {EGEMAPS_VALUES} finite values;"
            f" named measures {'present' if not absent else 'absent: ' + ', '.join(absent)}"
        )

    steps: dict[str, Callable[[Check], None]] = {
        "kokoro_weights": kokoro_weights,
        "espeak_fallback": espeak_fallback,
        "kokoro_render": kokoro_render,
        "faster_whisper": faster_whisper,
        "wav2vec2": wav2vec2,
        "opensmile": opensmile_values,
    }
    try:
        for name, step in steps.items():
            run(checks[name], step)
            check = checks[name]
            label = {True: "PASS", False: "FAIL", None: "NOT RUN"}[check.ok]
            print(f"{name:<16} {label:<7} {check.detail}")
        whisper, w2v = checks["faster_whisper"].numbers, checks["wav2vec2"].numbers
        measures = checks["opensmile"].numbers.get("named_measures", {})
        seconds = checks["kokoro_render"].numbers.get("seconds", {})
        print("voice        seconds  WER whisper  WER wav2vec2  F0 CV   loudness CV  voiced/s")
        for voice in seconds:
            m = measures.get(voice, {})
            print(
                f"{voice:<12} {seconds[voice]:>7.2f}"
                f"  {whisper.get('wer_by_voice', {}).get(voice, float('nan')):>11.3f}"
                f"  {w2v.get('wer_by_voice', {}).get(voice, float('nan')):>12.3f}"
                f"  {m.get(NAMED_MEASURES[0], float('nan')):>6.3f}"
                f"  {m.get(NAMED_MEASURES[1], float('nan')):>11.3f}"
                f"  {m.get(NAMED_MEASURES[2], float('nan')):>8.2f}"
            )
        if seconds:
            print(
                "CV: coefficient of variation, standard deviation / mean (eGeMAPS stddevNorm);"
                " F0 in semitones from 27.5 Hz"
            )
        passed = sum(check.ok is True for check in checks.values())
        verdict = "PASS" if passed == len(checks) else "FAIL"
        info.update(
            status="done",
            finished_utc=utc_now(),
            checks={
                name: {"ok": c.ok, "detail": c.detail, **c.numbers} for name, c in checks.items()
            },
            outputs={
                f"voices/{path.name}": sha256_file(path)
                for path in sorted(voices_dir.glob("*.wav"))
            },
            verdict=verdict,
        )
        print(f"voices: {voices_dir.relative_to(workspace.root)}")
        print(f"verdict: {verdict} ({passed} of {len(checks)} checks passed)")
    except BaseException:
        info.update(status="failed", finished_utc=utc_now())
        raise
    finally:
        write_json(run_dir / "run.json", info)
    if verdict != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
