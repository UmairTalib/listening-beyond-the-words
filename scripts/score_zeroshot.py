"""Score every segment of x0, x1 and rev with the untrained model, from the encoder caches (GPU job).

Submit from an OMNI login node, once the rules of the training code are dated in the pilot
protocol (configs/pilot/train.yaml) and the commit that records the GPU tests and the
training timing run has set their repeatability mode (precision.deterministic; the job
refuses to run while it says pending):
    scripts/slurm/submit.sh score_zeroshot

Pilot protocol, design "Arms" and section "Encoder caches and zero-shot scores"; the
settings are in configs/pilot/zeroshot.yaml (the views and the check) and
configs/pilot/data.yaml (encoder_cache and the pins), the pinned model in
configs/pilot/models.yaml. The job takes no argument: everything it reads is pinned.

Stages, in one process:
  1. Guards: a Slurm job, offline, a GPU, a clean checkout, and the rules of the training
     code dated (configs/pilot/train.yaml exists) with their repeatability mode set
     (precision.deterministic is strict or warn_only, no longer pending), so that every
     outcome of the GPU tests and the timing run is settled, and no setting is written,
     before any test score exists (pilot protocol, section "Training", bullet "Order,
     release tag and fallbacks"). zeroshot.yaml is read with the train.yaml that it
     extends (config.resolve_config), whose every key and value is checked
     (config.TrainSettings). The numeric settings are set first (voxtral.set_numerics,
     with encoder_cache.cpu_threads threads) and recorded; then the run folder.
  2. The embargo (scoring.embargo; pilot protocol, section "Clip versions", bullet "If a
     check fails"): no test-fold score is computed until the checks of x1 are finished.
     The pinned word-check run (wordcheck_run) must match its pin, have finished in its
     default mode, have read the pinned clips of x1 and rev, which the caches were built
     from, and hold x1 words PASS, a verdict on x1's voice (PASS or FAIL) and F4 PASS.
     Anything else refuses: a later fallback brings its own code.
  3. The pinned caches of all four clip versions with their checks (cache.pinned_cache).
     Each check watched the language model on the longest clips of its version, x2's too,
     before any score exists; a flag there stops the job before the model loads, since no
     score comes from the language model until the protocol's fallback (the flagged layers
     in fp32) is in place. The caches of the scored views must hold this job's numeric
     settings. Then the model, and every step from the adapter to the logits watched for
     fp16 overflow (voxtral.watch_layers: the adapter, the 30 layers of the language
     model, its final norm and its output layer), until the last score.
  4. The check: check.segments_per_view segments of each view drawn with
     item_seed(seed, "zeroshot", view), plus every segment of two chunks
     (zeroshot.check_sample), scored from the audio (voxtral.yes_no_score), then, with the
     encoder swapped for the pass-through, from the cache (scoring.score_views, which calls
     voxtral.cached_score). Every z must be finite and the two must agree bit for bit
     (float.hex); otherwise the job stops with counts only and scores nothing else. A
     flagged part stops it here too.
  5. Every segment of each view scored from the cache, one clip per call
     (scoring.score_views, the scoring code that the trained runs share). A flagged part (a
     value at the clamp, |x| >= 64,512, or a value that is not finite, the logits
     included) or a z that is not finite ends the run failed with counts only, and no score
     is written: the job runs again under the protocol's fallback (the flagged layers in
     fp32), into a new folder.
  6. The rows of the check sample must hold their z from the audio; then scores.parquet is
     written and must read back bit for bit; run.json (done).
No label is read: the job prints label-free numbers per view (the count, all finite, the
mean, the SD, the 5th, 50th and 95th percentiles and the share above 0), and the analysis
code computes the AUROC later.

Writes $LBW_WORKSPACE/runs/pilot/pilot-zeroshot-<YYYYmmdd-HHMMSS>/ (UTC) with
  scores.parquet  one row per segment and view: segment, session, view and z
                  (zeroshot.SCORE_COLUMNS); pinned by its SHA-256 (zeroshot_scores)
  run.json        commit, configs (zeroshot.yaml with the train.yaml that it extends,
                  their SHA-256s and the SHA-256 of the resolved config), inputs (with
                  the four cache checks and what they flagged), model, settings with the
                  resolved config, the numeric ones and the prompt, seeds,
                  environment, timings, the embargo's reading, the check's counts, the
                  watch's counts keyed by layer name and the label-free numbers per view
A job that fails also leaves errors.log there: traceback frames and error types, never a
message. scores.parquet holds a test-fold score per segment: only the researcher opens it,
on OMNI. The job prints aggregate numbers only; a failure may name its segment on its one
FAILED line and in run.json's error.
"""

import contextlib
import hashlib
import math
import os
import socket
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from listening_beyond_words.cache import CacheSettings, PinnedCache, features_sha256, pinned_cache
from listening_beyond_words.clips import read_view_clip
from listening_beyond_words.config import STRICT, WARN_ONLY, TrainSettings, resolve_config
from listening_beyond_words.models import ModelPin, load_pins, local_snapshot, snapshot_record
from listening_beyond_words.runinfo import (
    Progress,
    StepError,
    environment,
    git_state,
    gpu_facts,
    item_seed,
    peak_memory_gb,
    quiet_unraisable,
    record_failure,
    sha256_file,
    stop_on_sigterm,
    utc_now,
    write_json,
)
from listening_beyond_words.scoring import (
    EMBARGO_VIEWS,
    embargo,
    label_free,
    score_views,
    watch_summary,
)
from listening_beyond_words.settings import Workspace, get_workspace
from listening_beyond_words.voxtral import (
    ANSWERS,
    CLAMP_ABS,
    PROMPT,
    LayerWatch,
    LoadedModel,
    answer_token_ids,
    clip_inputs,
    language_model_layers,
    load_model,
    prompt_inputs,
    restore_encoder,
    set_numerics,
    use_cached_encoder,
    watch_layers,
    yes_no_score,
)
from listening_beyond_words.zeroshot import (
    ARM,
    SAMPLE_STREAM,
    SCORES,
    STEP,
    ZeroshotSettings,
    check_sample,
)

CONFIG = Path("configs/pilot/data.yaml")
MODELS = Path("configs/pilot/models.yaml")
ZEROSHOT = Path("configs/pilot/zeroshot.yaml")
# The shared training settings, which date the rules of the training code: the zero-shot
# scores wait for them (pilot protocol, section "Encoder caches and zero-shot scores"),
# and for the repeatability mode that the commit recording the training timing run sets
# there (section "Training"): strict or warn_only, until then pending.
TRAINING = Path("configs/pilot/train.yaml")
SET_MODES = (STRICT, WARN_ONLY)
USAGE = "usage: score_zeroshot.py (no arguments: everything the job reads is pinned)"
STUDY = "pilot"
PROGRESS_EVERY = 1000  # clips between two progress lines
# Every cached clip version: the check of each one watched the language model on its
# longest clips before any score exists, x2's too, though x2 is never scored.
CHECKED_VIEWS = ("x0", "x1", "x2", "rev")
PACKAGES = (
    "torch",
    "transformers",
    "mistral-common",
    "safetensors",
    "tokenizers",
    "nvidia-cudnn-cu12",
    "nvidia-cublas-cu12",
    "numpy",
    "pandas",
    "pyarrow",
    "soundfile",
    "pyyaml",
)


# The machine -------------------------------------------------------------------------------


def utc_in(seconds: float) -> str:
    """The UTC time this many seconds from now, to the minute."""
    return (datetime.now(UTC) + timedelta(seconds=seconds)).strftime("%H:%M UTC")


def run_name() -> str:
    """The run's name: study, arm and the UTC time (engineering conventions, section 4)."""
    return f"{STUDY}-{ARM}-{datetime.now(UTC):%Y%m%d-%H%M%S}"


def ids_sha256(ids: Any) -> str:
    """The SHA-256 of token IDs as little-endian 64-bit integers."""
    return hashlib.sha256(ids.detach().cpu().numpy().astype("<i8").tobytes()).hexdigest()


def same_hex(first: float, second: float) -> bool:
    """True if two scores are finite and have the same bits (float.hex)."""
    return math.isfinite(first) and math.isfinite(second) and first.hex() == second.hex()


# Stage 1: the run ---------------------------------------------------------------------------


@dataclass
class Run:
    """The run being made: where it writes, what it records and what it read so far."""

    job: str
    workspace: Workspace
    config: dict[str, Any]
    settings: ZeroshotSettings
    voxtral: ModelPin
    numerics: dict[str, object]
    run_dir: Path
    info: dict[str, Any]
    started: float

    def save(self) -> None:
        """Write run.json as it stands."""
        write_json(self.run_dir / "run.json", self.info)

    @property
    def relative(self) -> Path:
        """The run folder, relative to the workspace."""
        return self.run_dir.relative_to(self.workspace.root)


def start(state: Progress) -> Run:
    """Stage 1: refuse to run outside a clean offline GPU job; set the numerics; the folder."""
    stop_on_sigterm()
    # Before transformers loads, which reads it for its own progress bars too.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    if sys.argv[1:]:
        raise StepError(USAGE)
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise StepError("submit this as a job: scripts/slurm/submit.sh score_zeroshot")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise StepError(
            "HF_HUB_OFFLINE is not 1: the job must run offline (scripts/slurm/common.sh)"
        )
    git = git_state()
    if git["dirty"]:
        raise StepError("the checkout has uncommitted changes; commit them on the Mac and pull")
    gpu = gpu_facts()
    if not gpu.get("gpu"):
        raise StepError(
            "no CUDA GPU: submit this as a GPU job (scripts/slurm/score_zeroshot.sbatch)"
        )
    if not TRAINING.is_file():
        raise StepError(
            f"the rules of the training code are not dated yet ({TRAINING} is missing): the"
            ' zero-shot scores wait for them (pilot protocol, section "Encoder caches and'
            ' zero-shot scores")'
        )
    zeroshot = resolve_config(ZEROSHOT)
    mode = TrainSettings.from_config(zeroshot.shared).precision.deterministic
    if mode not in SET_MODES:
        raise StepError(
            f"the repeatability mode of training is not set yet ({zeroshot.shared_file}:"
            f" precision.deterministic is {mode!r}, not one of {list(SET_MODES)}): the"
            " zero-shot scores wait for the commit that records the GPU tests and the"
            ' training timing run (pilot protocol, section "Training", bullet "Order,'
            ' release tag and fallbacks")'
        )
    started = time.monotonic()
    workspace = get_workspace()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    settings = ZeroshotSettings.from_config(zeroshot.own)
    threads = CacheSettings.from_config(config).cpu_threads
    voxtral = load_pins(MODELS)["voxtral"]
    # First, before any model or clip: z from the cache equals z from the audio only under
    # the settings of the caches (pilot protocol, section "Encoder caches and zero-shot
    # scores").
    numerics = set_numerics(threads)
    run_dir = workspace.runs / STUDY / run_name()
    run_dir.mkdir(parents=True, exist_ok=False)
    state.run_dir = run_dir
    machine = environment(PACKAGES)
    info: dict[str, Any] = {
        "run_id": run_dir.name,
        "step": STEP,
        "arm": ARM,
        "status": "running",
        "started_utc": utc_now(),
        "git": git,
        "config": {"file": str(CONFIG), "sha256": sha256_file(CONFIG)},
        "models_config": {"file": str(MODELS), "sha256": sha256_file(MODELS)},
        "zeroshot_config": {
            "file": str(ZEROSHOT),
            "sha256": zeroshot.file_sha256,
            "extends": {"file": str(zeroshot.shared_file), "sha256": zeroshot.shared_sha256},
            "resolved_sha256": zeroshot.sha256,
        },
        "slurm": {
            "job_id": job,
            "node": socket.gethostname(),
            "cpu": machine["cpu"],
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
        "settings": {
            "zeroshot": zeroshot.resolved,
            "encoder_cache": config["encoder_cache"],
            "numerics": numerics,
        },
        "seeds": {
            "check_sample": {
                "seed": settings.seed,
                "streams": {
                    view: f"numpy.random.default_rng(runinfo.item_seed({settings.seed},"
                    f' "{SAMPLE_STREAM}", "{view}"))'
                    for view in settings.views
                },
                "item_seeds": {
                    view: item_seed(settings.seed, SAMPLE_STREAM, view) for view in settings.views
                },
            }
        },
        "inputs": {},
        "environment": {
            **machine,
            **gpu,
            "libsndfile": sf.__libsndfile_version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        },
        "results": {},
    }
    state.info = info
    run = Run(job, workspace, config, settings, voxtral, numerics, run_dir, info, started)
    run.save()
    print(
        f"zero-shot scores of {', '.join(settings.views)}; job {job} on {socket.gethostname()},"
        f" {gpu['gpu']}; CPU {machine['cpu']}"
    )
    print("numerics: " + ", ".join(f"{key} {value}" for key, value in numerics.items()))
    return run


# Stage 3: the caches and the model ------------------------------------------------------------


def load_caches(run: Run) -> dict[str, PinnedCache]:
    """Stage 3: the pinned caches of the four versions with their checks; those to score.

    Every check must have watched the language model on the longest clips of its version
    without a flag: a flag leaves the caches standing, but no score comes from the
    language model until the protocol's fallback is in place. The caches of the scored
    views must hold this job's numeric settings.

    Returns:
        The pinned caches of the scored views, in the order they are scored.
    """
    pinned = {
        view: pinned_cache(run.workspace, run.config, view, run.voxtral) for view in CHECKED_VIEWS
    }
    run.info["inputs"]["cache_checks"] = {
        view: {"run": cache.check, "lm_flagged": list(cache.lm_flagged)}
        for view, cache in pinned.items()
    }
    run.save()
    checks = sorted({cache.check for cache in pinned.values()})
    flagged = {view: cache.lm_flagged for view, cache in pinned.items() if cache.lm_flagged}
    print(
        f"cache checks: {', '.join(checks)} watched the language model on the longest clips of"
        f" {', '.join(CHECKED_VIEWS)}: {sum(map(len, flagged.values()))} parts flagged"
    )
    for view, parts in flagged.items():
        print(f"cache check of {view}, flagged: {', '.join(parts)}")
    if flagged:
        raise StepError(
            "the checks of the caches flagged the language model on the longest clips of"
            f" {', '.join(flagged)}: no score is computed until the fallback of the pilot"
            " protocol (the flagged layers in fp32) is in place"
        )
    caches = {}
    for view in run.settings.views:
        cache = pinned[view]
        if cache.run.info["settings"]["numerics"] != run.numerics:
            raise StepError(
                f"{cache.run.run} was built with other numeric settings than this job's; score"
                " it with the job file, whose CPUs are those of the cache jobs"
                " (scripts/slurm/score_zeroshot.sbatch)"
            )
        caches[view] = cache
    run.info["inputs"]["caches"] = {view: cache.pin for view, cache in caches.items()}
    run.info["inputs"]["views"] = {
        view: dict(cache.run.info["inputs"]["view"]) for view, cache in caches.items()
    }
    run.save()
    for view, cache in caches.items():
        two = int((cache.manifest["chunks"] > 1).sum())
        print(
            f"cache {view}: {cache.run.run}, {len(cache.manifest):,} clips ({two} of two"
            f" chunks), checked by {cache.check}"
        )
    return caches


def load(run: Run, caches: Mapping[str, PinnedCache]) -> tuple[LoadedModel, tuple[int, int]]:
    """The pinned Voxtral, offline, in fp16, with the token IDs of "Yes" and "No"."""
    snapshot = local_snapshot(run.voxtral, run.workspace.hf_home)
    run.info["inputs"]["voxtral"] = snapshot_record(run.voxtral, snapshot)
    clock = time.monotonic()
    loaded = load_model(snapshot)
    seconds = time.monotonic() - clock
    width = int(loaded.model.config.audio_config.hidden_size)
    run.info["settings"]["width"] = width
    for cache in caches.values():
        if cache.run.info["settings"].get("width") != width:
            raise StepError(
                f"{cache.run.run} holds outputs of width {cache.run.info['settings'].get('width')},"
                f" not {width}"
            )
    answer_ids = answer_token_ids(loaded.processor.tokenizer)
    run.info["results"].update(
        load_seconds=round(seconds, 1), largest_weight=float(loaded.largest_weight)
    )
    run.save()
    print(
        f"model: {run.voxtral.repo_id} at {run.voxtral.revision[:8]}, loaded in {seconds:.0f} s;"
        f" largest |weight| {loaded.largest_weight:.6g}; encoder width {width};"
        f" {ANSWERS[0]} {answer_ids[0]}, {ANSWERS[1]} {answer_ids[1]}"
    )
    return loaded, answer_ids


def prompts_of(
    run: Run, caches: Mapping[str, PinnedCache], processor: Any, answer_ids: tuple[int, int]
) -> dict[int, dict[str, Any]]:
    """The token IDs and mask of a clip of each number of chunks that the caches hold."""
    counts = sorted({int(count) for cache in caches.values() for count in cache.manifest["chunks"]})
    prompts = {count: prompt_inputs(processor, count) for count in counts}
    run.info["settings"]["prompt"] = {
        "text": PROMPT,
        "answers": list(ANSWERS),
        "answer_ids": list(answer_ids),
        "token_ids": {
            str(count): {
                "tokens": int(prompt["input_ids"].shape[1]),
                "sha256": ids_sha256(prompt["input_ids"]),
            }
            for count, prompt in prompts.items()
        },
    }
    run.save()
    print(
        "prompt: "
        + "; ".join(
            f"{count} chunk(s): {int(prompt['input_ids'].shape[1])} token IDs, SHA-256"
            f" {ids_sha256(prompt['input_ids'])[:12]}..."
            for count, prompt in prompts.items()
        )
    )
    return prompts


# The overflow watch ---------------------------------------------------------------------------


def judge_watch(run: Run, watch: LayerWatch, non_finite: int, after: str) -> None:
    """Record the watch's counts; stop, writing no score, on a flagged part or a non-finite z."""
    summary = watch_summary(watch)
    run.info["results"]["lm_watch"] = summary
    run.info["results"]["non_finite_z"] = non_finite
    run.save()
    largest = "; ".join(f"{name} {value:.6g}" for name, value in summary["largest"])
    print(
        f"language-model watch after {after}: {len(summary['flagged'])} of {summary['parts']}"
        f" parts flagged (a value at the clamp, |x| >= {CLAMP_ABS:,}, or not finite; the"
        f" logits included); largest |value|: {largest or 'none'}"
    )
    if summary["flagged"]:
        print("language-model watch, flagged: " + ", ".join(summary["flagged"]))
    if summary["flagged"] or non_finite:
        raise StepError(
            f"the language-model watch flagged {len(summary['flagged'])} parts and {non_finite}"
            " scores are not finite: no score is written; the job runs again under the"
            " fallback of the pilot protocol (the flagged layers in fp32), into a new folder"
        )


# Stages 4 and 5: the check and the scores ---------------------------------------------------


def from_audio(
    run: Run,
    caches: Mapping[str, PinnedCache],
    samples: Mapping[str, tuple[str, ...]],
    loaded: LoadedModel,
    prompts: Mapping[int, Mapping[str, Any]],
    answer_ids: tuple[int, int],
    state: Progress,
) -> tuple[dict[tuple[str, str], float], dict[str, dict[str, int]]]:
    """The check sample scored from the audio, with the model's own encoder in place.

    Returns:
        z by (view, segment), and per view how many clips gave the stored log-mel
        fingerprint and the token IDs of the prompt of their number of chunks.
    """
    import torch

    model, processor = loaded.model, loaded.processor
    scores: dict[tuple[str, str], float] = {}
    same: dict[str, dict[str, int]] = {}
    for view, cache in caches.items():
        folder = run.workspace.resolve(run.config["view_runs"][view]["run"])
        clips = {str(row.segment): row for row in cache.run.clips.itertuples(index=False)}
        stored = {str(row.segment): row for row in cache.manifest.itertuples(index=False)}
        same[view] = {"features_equal": 0, "token_ids_equal": 0}
        for segment in samples[view]:
            state.segment = segment
            row = stored[segment]
            inputs = clip_inputs(processor, read_view_clip(folder, clips[segment]).audio)
            prompt = prompts[int(row.chunks)]
            same[view]["features_equal"] += (
                features_sha256(inputs["input_features"]) == row.features_sha256
            )
            same[view]["token_ids_equal"] += bool(
                torch.equal(inputs["input_ids"], prompt["input_ids"])
                and torch.equal(inputs["attention_mask"], prompt["attention_mask"])
            )
            scores[(view, segment)] = yes_no_score(model, inputs, answer_ids)
    state.segment = None
    return scores, same


def judge_check(
    run: Run,
    caches: Mapping[str, PinnedCache],
    samples: Mapping[str, tuple[str, ...]],
    audio: Mapping[tuple[str, str], float],
    cached: Mapping[tuple[str, str], float],
    same: Mapping[str, Mapping[str, int]],
    seconds: float,
) -> None:
    """Record and print the check's counts; stop unless every z agrees bit for bit."""
    results = {}
    for view, sample in samples.items():
        pairs = [(audio[(view, segment)], cached[(view, segment)]) for segment in sample]
        results[view] = {
            "segments": len(sample),
            "drawn": min(run.settings.segments_per_view, len(caches[view].manifest)),
            "two_chunk": int((caches[view].manifest["chunks"] > 1).sum()),
            "z_equal": sum(same_hex(first, second) for first, second in pairs),
            "non_finite": sum(
                not (math.isfinite(first) and math.isfinite(second)) for first, second in pairs
            ),
            **same[view],
        }
    run.info["results"]["check"] = {"views": results, "seconds": round(seconds, 1)}
    run.save()
    for view, result in results.items():
        n = result["segments"]
        print(
            f"check {view}: z from the cache equals z from the audio in {result['z_equal']} of"
            f" {n} segments ({result['drawn']} drawn with seed {run.settings.seed},"
            f" {result['two_chunk']} of two chunks); {result['non_finite']} with a z that is"
            f" not finite; log-mel features equal {result['features_equal']} of {n}, token IDs"
            f" equal {result['token_ids_equal']} of {n}"
        )
    differ = sum(result["segments"] - result["z_equal"] for result in results.values())
    if differ:
        total = sum(result["segments"] for result in results.values())
        raise StepError(
            f"the check found z from the cache unequal to z from the audio, or not finite, in"
            f" {differ} of {total} segments: a fault to find and fix; no score is written, and"
            " the check runs again under the same rule"
        )


def score(run: Run, state: Progress) -> None:
    """Stages 2 to 6, after stage 1."""
    reading = embargo(run.workspace, run.config)  # 2
    run.info["inputs"]["wordcheck_run"] = {
        key: reading[key] for key in ("run", "sha256", "run_json_sha256")
    }
    run.info["embargo"] = {"views": reading["views"], "verdicts": reading["verdicts"]}
    run.save()
    verdicts = reading["verdicts"]
    print(
        f"embargo: the checks of x1 are finished ({reading['run']}: F4 {verdicts['f4']}, x1"
        f" words {verdicts['x1_words']}, x1 voice {verdicts['x1_voice']}), on the pinned clips"
        f" of {' and '.join(EMBARGO_VIEWS)}"
    )

    caches = load_caches(run)  # 3
    loaded, answer_ids = load(run, caches)
    prompts = prompts_of(run, caches, loaded.processor, answer_ids)
    samples = {
        view: check_sample(cache.manifest, view, run.settings.segments_per_view, run.settings.seed)
        for view, cache in caches.items()
    }
    results = run.info["results"]
    width = int(run.info["settings"]["width"])
    with watch_layers(language_model_layers(loaded.model, logits=True)) as watch:
        clock = time.monotonic()  # 4
        audio, same = from_audio(run, caches, samples, loaded, prompts, answer_ids, state)
        encoder = use_cached_encoder(loaded.model)
        try:
            checked = score_views(
                loaded.model, caches, prompts, answer_ids, width, segments=samples, state=state
            )
            cached = {
                (str(view), str(segment)): float(z)
                for view, segment, z in zip(
                    checked["view"], checked["segment"], checked["z"], strict=True
                )
            }
            judge_check(run, caches, samples, audio, cached, same, time.monotonic() - clock)
            judge_watch(run, watch, 0, "the check")

            clock = time.monotonic()  # 5
            total = sum(len(cache.manifest) for cache in caches.values())

            def progress(done: int) -> None:
                if done % PROGRESS_EVERY == 0 and done < total:
                    elapsed = time.monotonic() - clock
                    print(
                        f"scores: {done:,} of {total:,}, {done / elapsed:.1f} per second;"
                        f" projected end {utc_in(elapsed / done * (total - done))}"
                    )

            table = score_views(
                loaded.model, caches, prompts, answer_ids, width, state=state, on_clip=progress
            )
        finally:
            restore_encoder(loaded.model, encoder)
    seconds = time.monotonic() - clock
    non_finite = int((~np.isfinite(table["z"].to_numpy())).sum())
    results["scoring"] = {
        "clips": len(table),
        "seconds": round(seconds, 1),
        "seconds_per_clip": round(seconds / len(table), 4),
    }
    results["peak_gpu_memory_gb"] = peak_memory_gb()
    print(
        f"scores: {len(table):,} clips from the cache in {seconds / 60:.1f} min,"
        f" {seconds / len(table):.3f} s per clip; peak GPU memory {results['peak_gpu_memory_gb']} GB"
    )
    judge_watch(run, watch, non_finite, "every score")

    # 6: each row of the check sample must hold the z that the audio gave, before any score is
    # written; the file must then read back as this table, bit for bit.
    rows = {
        (str(view), str(segment)): float(z)
        for view, segment, z in zip(table["view"], table["segment"], table["z"], strict=True)
    }
    differ = sum(not same_hex(rows[key], z) for key, z in audio.items())
    if differ:
        raise StepError(
            f"{differ} of the check sample's {len(audio)} rows differ from their z from the audio:"
            " no score is written"
        )
    path = run.run_dir / SCORES
    table.to_parquet(path, index=False)
    sha256 = sha256_file(path)
    back = pd.read_parquet(path)
    if not back.equals(table) or [z.hex() for z in back["z"]] != [z.hex() for z in table["z"]]:
        raise StepError(f"{SCORES} does not read back as written")
    results["z"] = {
        view: label_free(table.loc[table["view"] == view, "z"].to_numpy())
        for view in run.settings.views
    }
    run.info["outputs"] = {SCORES: sha256}
    minutes = (time.monotonic() - run.started) / 60
    run.info.update(status="done", finished_utc=utc_now(), minutes=round(minutes, 2))
    run.save()
    print(
        f"{SCORES}: {len(back):,} rows read back as written; the check sample's {len(audio)}"
        " rows equal their z from the audio"
    )
    for view, numbers in results["z"].items():
        print(
            f"z of {view}: {numbers['count']:,} scores, all finite; mean {numbers['mean']:.4f},"
            f" SD {numbers['sd']:.4f}; percentiles 5, 50 and 95: {numbers['p5']:.4f},"
            f" {numbers['p50']:.4f}, {numbers['p95']:.4f}; share above 0"
            f" {numbers['share_above_0']:.4f}"
        )
    print(f"run: {run.relative}")
    print(f"{SCORES} sha256: {sha256}")
    print(f"done in {minutes:.1f} min")


# The job -------------------------------------------------------------------------------------


def score_job(state: Progress) -> None:
    """The job: stage 1, then the rest."""
    run = start(state)
    score(run, state)


def main() -> None:
    os.umask(0o077)
    sys.unraisablehook = quiet_unraisable  # a finaliser's message could hold anything
    progress = Progress()
    try:
        score_job(progress)
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
