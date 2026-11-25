"""Build the encoder caches of one clip version, or check cache runs against the audio (GPU job).

Submit from an OMNI login node:
    scripts/slurm/submit.sh build_caches --view <view>
    scripts/slurm/submit.sh --time=01:00:00 build_caches --verify cache/<view>/run-<job id> ...

Pilot protocol, section "Encoder caches and zero-shot scores"; the settings are in
configs/pilot/data.yaml (encoder_cache), the pinned model in configs/pilot/models.yaml.
Two modes, whose options name a pinned view or earlier runs, never a setting:
  --view V      Voxtral's frozen audio encoder reads every clip of the pinned clip version
                V, one clip per call, and its output before the adapter is stored: every
                vector of every 30 s chunk, in fp16, one file per segment (cache.py). A
                clip over 30 s is read in two chunks.
  --verify RUN  for each named cache run, the log-mel features and the encoder outputs of
                a sample are computed again from the audio and compared with the stored
                fingerprints, features first, byte for byte: encoder_cache.verify
                .segments_per_view segments drawn with item_seed(seed, "cache", view), plus
                every segment of two chunks (cache.check_sample). The language model then
                runs without its output layer, from the cache, on the lm_watch_clips
                longest clips of each run, with the overflow watch on the adapter and the
                language model: no logit and no score is computed.

Stages, in one process:
  1. Guards: a Slurm job, offline, a GPU and a clean checkout. The numeric settings are set
     first (voxtral.set_numerics) and recorded; then the run folder.
  2. --view: the pinned clips of the view; the model; every step of the encoder watched
     for fp16 overflow (voxtral.watch_layers: its two convolutions, its 32 layers and its
     final norm). Each clip in manifest order: read and checked against its manifest row,
     its log-mel features and their SHA-256, the encoder's output, its bytes and their
     SHA-256, the file written whole.
  3. --view: the watch's counts go into run.json; a value at the clamp (|x| >= 64,512) or a
     non-finite value in any watched part, or a non-finite output, then ends the run
     failed: such a cache is never used, and it is built again under the protocol's
     fallback.
  4. --view: every file read back against its manifest row; manifest.parquet; run.json
     (done).
  --verify, after stage 1: each run checked against the current pins and settings and
     against this job's own numeric settings (cache.read_cache_run); the sample of every
     run computed again and compared, and the run ends failed once all are compared if
     any feature or output differs; then the adapter, the language model's 30 layers and
     its final norm watched on the longest clips of every run. A flag there is recorded
     and printed: the caches stand, and no score comes from the language model until the
     protocol's fallback is in place (cache.pinned_cache gives the flags to the zero-shot
     job and to training).

Writes, with --view, $LBW_WORKSPACE/cache/<view>/run-<job id>/ with
  <session>/<segment>.npy  the encoder output of each clip, (C, 1500, W), fp16
  manifest.parquet         one row per clip (cache.CACHE_COLUMNS); pinned by its SHA-256
  run.json                 mode, commit, configs, inputs, model, settings with the numeric
                           ones, environment, timings and aggregate results, with the
                           watch's counts keyed by layer name
and, with --verify, $LBW_WORKSPACE/derived/cache_check/run-<job id>/run.json only. A job
that fails also leaves errors.log there: traceback frames and error types, never a
message. The job prints aggregate numbers only; a failure may name its segment on its one
FAILED line and in run.json's error.
"""

import contextlib
import hashlib
import math
import os
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from listening_beyond_words.cache import (
    CACHE_COLUMNS,
    SAMPLE_STREAM,
    STEP,
    SUFFIX,
    VERIFY_MODE,
    VIEW_MODE,
    CacheRun,
    CacheSettings,
    check_sample,
    features_sha256,
    pinned_clips,
    read_cache_run,
    read_file,
    relative_run,
    to_bytes,
    write_file,
)
from listening_beyond_words.clips import read_view_clip
from listening_beyond_words.models import ModelPin, load_pins, local_snapshot, snapshot_record
from listening_beyond_words.runinfo import (
    Progress,
    StepError,
    environment,
    git_state,
    item_seed,
    quiet_unraisable,
    record_failure,
    sha256_file,
    stop_on_sigterm,
    utc_now,
    write_json,
)
from listening_beyond_words.settings import Workspace, get_workspace
from listening_beyond_words.views import SAMPLING_RATE
from listening_beyond_words.voxtral import (
    CLAMP_ABS,
    LoadedModel,
    chunks,
    clip_inputs,
    encoder_layers,
    encoder_output,
    language_model_layers,
    lm_pass,
    load_model,
    prompt_inputs,
    restore_encoder,
    set_numerics,
    use_cached_encoder,
    watch_layers,
)

CONFIG = Path("configs/pilot/data.yaml")
MODELS = Path("configs/pilot/models.yaml")
USAGE = "usage: build_caches.py --view <view> | --verify cache/<view>/run-<job id> ..."
PROGRESS_EVERY = 500  # clips between two progress lines
LARGEST = 5  # the layer maxima that are printed
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


# The mode and the machine ------------------------------------------------------------------


@dataclass(frozen=True)
class Mode:
    """What the job does: cache one view, or check earlier cache runs."""

    name: str
    view: str = ""
    runs: tuple[str, ...] = ()

    def record(self) -> dict[str, Any]:
        """The mode as run.json records it."""
        if self.name == VIEW_MODE:
            return {"name": VIEW_MODE, "view": self.view}
        return {"name": VERIFY_MODE, "runs": list(self.runs)}


def parse_mode(argv: Sequence[str]) -> Mode:
    """The mode from the command line; anything else is refused with the usage."""
    if len(argv) == 2 and argv[0] == "--view" and not argv[1].startswith("-"):
        return Mode(VIEW_MODE, view=argv[1])
    runs = list(argv[1:])
    if (
        argv
        and argv[0] == "--verify"
        and runs
        and not any(run.startswith("-") for run in runs)
        and len(set(runs)) == len(runs)
    ):
        return Mode(VERIFY_MODE, runs=tuple(runs))
    raise StepError(USAGE)


def gpu_facts() -> dict[str, Any]:
    """The GPU of this job, its driver, CUDA and cuDNN. PyTorch is imported here."""
    import torch

    if not torch.cuda.is_available():
        return {"gpu": None, "cuda": torch.version.cuda, "cudnn": None, "driver": None}
    properties = torch.cuda.get_device_properties(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": f"{properties.major}.{properties.minor}",
        "gpu_memory_gb": round(properties.total_memory / 1e9, 2),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "driver": driver_version(),
    }


def driver_version() -> str | None:
    """The NVIDIA driver's version, as nvidia-smi gives it; None if it cannot say."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.strip().splitlines()
    return lines[0].strip() if lines else None


def peak_memory_gb() -> float | None:
    """The most GPU memory that PyTorch held so far, in GB; None without a GPU."""
    import torch

    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1e9, 2)


def utc_in(seconds: float) -> str:
    """The UTC time this many seconds from now, to the minute."""
    return (datetime.now(UTC) + timedelta(seconds=seconds)).strftime("%H:%M UTC")


def synchronise() -> None:
    """Wait until the GPU has run every kernel queued so far: a time taken next includes them."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def write_table(table: pd.DataFrame, path: Path) -> str:
    """Write a table as Parquet and return its SHA-256."""
    table.to_parquet(path, index=False)
    return sha256_file(path)


# Stage 1: the run ---------------------------------------------------------------------------


@dataclass
class Run:
    """The run being made: where it writes, what it records and what it read so far."""

    mode: Mode
    job: str
    workspace: Workspace
    config: dict[str, Any]
    settings: CacheSettings
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
    mode = parse_mode(sys.argv[1:])
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise StepError("submit this as a job: scripts/slurm/submit.sh build_caches")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise StepError(
            "HF_HUB_OFFLINE is not 1: the job must run offline (scripts/slurm/common.sh)"
        )
    git = git_state()
    if git["dirty"]:
        raise StepError("the checkout has uncommitted changes; commit them on the Mac and pull")
    gpu = gpu_facts()
    if not gpu.get("gpu"):
        raise StepError("no CUDA GPU: submit this as a GPU job (scripts/slurm/build_caches.sbatch)")
    started = time.monotonic()
    workspace = get_workspace()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    settings = CacheSettings.from_config(config)
    voxtral = load_pins(MODELS)["voxtral"]
    if mode.name == VIEW_MODE:
        if mode.view not in config["view_runs"]:
            raise StepError(f"--view names a view without a pin in view_runs: {mode.view!r}")
        folder = workspace.cache / mode.view
    else:
        runs = tuple(relative_run(workspace, run) for run in mode.runs)
        if len(set(runs)) != len(runs):
            raise StepError("--verify names one cache run twice")
        mode = Mode(VERIFY_MODE, runs=runs)
        folder = workspace.derived / "cache_check"
    # First, before any model or clip: the same kernels give the same bits only under the
    # same settings (pilot protocol, section "Encoder caches and zero-shot scores").
    numerics = set_numerics(settings.cpu_threads)
    run_dir = folder / f"run-{job}"
    run_dir.mkdir(parents=True, exist_ok=False)
    state.run_dir = run_dir
    machine = environment(PACKAGES)
    info: dict[str, Any] = {
        "step": STEP,
        "mode": mode.record(),
        "status": "running",
        "started_utc": utc_now(),
        "git": git,
        "config": {"file": str(CONFIG), "sha256": sha256_file(CONFIG)},
        "models_config": {"file": str(MODELS), "sha256": sha256_file(MODELS)},
        "slurm": {
            "job_id": job,
            "node": socket.gethostname(),
            "cpu": machine["cpu"],
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
        "settings": {"encoder_cache": config["encoder_cache"], "numerics": numerics},
        "seeds": {},
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
    run = Run(mode, job, workspace, config, settings, voxtral, numerics, run_dir, info, started)
    run.save()
    what = (
        f"encoder caches of {mode.view} ({config['view_runs'][mode.view]['run']})"
        if mode.name == VIEW_MODE
        else f"check of {len(mode.runs)} cache runs ({', '.join(mode.runs)})"
    )
    print(f"{what}; job {job} on {socket.gethostname()}, {gpu['gpu']}; CPU {machine['cpu']}")
    print("numerics: " + ", ".join(f"{key} {value}" for key, value in numerics.items()))
    return run


def load(run: Run) -> LoadedModel:
    """The pinned Voxtral, offline, in fp16; its files and load time go into run.json."""
    snapshot = local_snapshot(run.voxtral, run.workspace.hf_home)
    run.info["inputs"]["voxtral"] = snapshot_record(run.voxtral, snapshot)
    clock = time.monotonic()
    loaded = load_model(snapshot)
    seconds = time.monotonic() - clock
    width = int(loaded.model.config.audio_config.hidden_size)
    run.info["settings"]["width"] = width
    run.info["results"].update(
        load_seconds=round(seconds, 1), largest_weight=float(loaded.largest_weight)
    )
    run.save()
    print(
        f"model: {run.voxtral.repo_id} at {run.voxtral.revision[:8]}, loaded in {seconds:.0f} s;"
        f" largest |weight| {loaded.largest_weight:.6g}; encoder width {width}"
    )
    return loaded


def watch_summary(watch: Any) -> dict[str, Any]:
    """What a layer watch saw: the flagged parts, the largest values and the counts by layer."""
    layers = watch.record()
    return {
        "parts": sum(len(parts) for parts in layers.values()),
        "flagged": watch.flagged(),
        "largest": [[name, value] for name, value in watch.largest(LARGEST)],
        "layers": layers,
    }


def watch_line(summary: Mapping[str, Any]) -> str:
    """The watch's counts and its largest values, for one printed line."""
    largest = "; ".join(f"{name} {value:.6g}" for name, value in summary["largest"])
    return (
        f"{len(summary['flagged'])} of {summary['parts']} parts flagged (a value at the clamp,"
        f" |x| >= {CLAMP_ABS:,}, or not finite); largest |value|: {largest or 'none'}"
    )


# --view: the cache of one clip version -------------------------------------------------------


def build(run: Run, state: Progress) -> None:
    """Stages 2 to 4: every clip of the view through the encoder, stored, then read back."""
    view = run.mode.view
    info, results = run.info, run.info["results"]
    pin = run.config["view_runs"][view]
    clips_dir = run.workspace.resolve(pin["run"])
    manifest = pinned_clips(run.workspace, run.config, view)
    if manifest.empty:
        raise StepError(f"the pinned clips of {view} hold no clip")
    info["inputs"]["view"] = {"run": pin["run"], "sha256": pin["sha256"]}
    loaded = load(run)
    model, processor = loaded.model, loaded.processor
    width = int(info["settings"]["width"])

    rows: list[dict[str, Any]] = []
    # The encoder's time is its own: the checks of its output and the .npy bytes after it
    # are timed apart (serialise), and so is the write.
    seconds = dict.fromkeys(("features", "encoder", "serialise", "write"), 0.0)
    non_finite = written = 0
    total = len(manifest)
    clock = time.monotonic()
    with watch_layers(encoder_layers(model)) as watch:  # 2
        for number, row in enumerate(manifest.itertuples(index=False), start=1):
            segment, session = str(row.segment), int(row.session)
            state.segment = segment
            began = time.monotonic()
            clip = read_view_clip(clips_dir, row)
            count = chunks(len(clip.samples))
            inputs = clip_inputs(processor, clip.audio)
            features = features_sha256(inputs["input_features"])
            featured = time.monotonic()
            output = encoder_output(model, inputs)
            synchronise()  # the encoder's kernels run until here
            encoded = time.monotonic()
            if output.shape[0] != count:
                raise StepError(
                    f"segment {segment}: the processor made {output.shape[0]} chunks of a clip"
                    f" of {len(clip.samples)} samples, not {count}",
                    segment=segment,
                )
            finite = bool(output.isfinite().all())
            largest = float(output.abs().max())
            data = to_bytes(output)  # on the CPU
            serialised = time.monotonic()
            file = f"{session}/{segment}{SUFFIX}"
            (run.run_dir / str(session)).mkdir(exist_ok=True)
            write_file(run.run_dir / file, data)
            seconds["features"] += featured - began
            seconds["encoder"] += encoded - featured
            seconds["serialise"] += serialised - encoded
            seconds["write"] += time.monotonic() - serialised
            written += len(data)
            rows.append(
                {
                    "segment": segment,
                    "session": session,
                    "file": file,
                    "chunks": count,
                    "samples": len(clip.samples),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "audio_sha256": str(row.audio_sha256),
                    "features_sha256": features,
                    "max_abs": largest,
                }
            )
            non_finite += not finite
            if number % PROGRESS_EVERY == 0 and number < total:
                elapsed = time.monotonic() - clock
                print(
                    f"clips: {number:,} of {total:,} encoded, {number / elapsed:.1f} per second;"
                    f" projected end {utc_in(elapsed / number * (total - number))}"
                )
        state.segment = None
    elapsed = time.monotonic() - clock

    table = pd.DataFrame(rows, columns=list(CACHE_COLUMNS))  # 3
    chunk_count = int(table["chunks"].sum())
    two = int((table["chunks"] > 1).sum())
    finite_maxima = [value for value in table["max_abs"].tolist() if math.isfinite(value)]
    summary = watch_summary(watch)
    results.update(
        clips=len(table),
        chunks=chunk_count,
        two_chunk_clips=two,
        audio_hours=round(float(table["samples"].sum()) / SAMPLING_RATE / 3600, 4),
        bytes=written,
        seconds={name: round(value, 1) for name, value in seconds.items()},
        seconds_per_clip=round(elapsed / len(table), 4),
        encoder_seconds_per_chunk=round(seconds["encoder"] / chunk_count, 4),
        peak_gpu_memory_gb=peak_memory_gb(),
        max_abs=max(finite_maxima) if finite_maxima else None,
        non_finite_outputs=non_finite,
        encoder_watch=summary,
    )
    run.save()
    print(
        f"clips: {len(table):,} encoded ({two} of two chunks), {chunk_count:,} chunks,"
        f" {written / 1e9:.2f} GB written in {elapsed / 60:.1f} min: {elapsed / len(table):.3f} s per clip,"
        f" {seconds['encoder'] / chunk_count:.3f} s per chunk in the encoder; peak GPU memory"
        f" {results['peak_gpu_memory_gb']} GB"
    )
    print(
        f"outputs: largest |value| {results['max_abs']}; {non_finite} with a value that is not"
        " finite"
    )
    print("encoder watch: " + watch_line(summary))
    if summary["flagged"]:
        print("encoder watch, flagged: " + ", ".join(summary["flagged"]))
    if summary["flagged"] or non_finite:
        raise StepError(
            f"the encoder watch flagged {len(summary['flagged'])} parts and {non_finite} outputs"
            " hold a value that is not finite: this cache is never used; it is built again"
            " under the fallback of the pilot protocol (the flagged layers in fp32)"
        )

    clock = time.monotonic()  # 4
    for row in table.itertuples(index=False):
        state.segment = str(row.segment)
        read_file(run.run_dir / str(row.file), row, width)
    state.segment = None
    read_back = time.monotonic() - clock
    sha256 = write_table(table, run.run_dir / "manifest.parquet")
    info["outputs"] = {"manifest.parquet": sha256}
    results["seconds"]["read_back"] = round(read_back, 1)
    minutes = (time.monotonic() - run.started) / 60
    info.update(status="done", finished_utc=utc_now(), minutes=round(minutes, 2))
    run.save()
    print(
        f"read back: {len(table):,} files match their manifest rows (SHA-256, shape, chunks,"
        f" all finite) in {read_back:.0f} s"
    )
    print(f"run: {run.relative}")
    print(f"manifest.parquet sha256: {sha256}")
    print(f"done in {minutes:.1f} min")


# --verify: the cache runs against the audio ------------------------------------------------


def fp16_steps(first: Any, second: Any) -> int | None:
    """The largest distance between two fp16 tensors, in fp16 steps; None if their shapes differ.

    An fp16 step is the gap between two neighbouring fp16 numbers: the bits are read as
    whole numbers in the order of the values, so +0 and -0 lie 0 steps apart.
    """
    keys = []
    for tensor in (first, second):
        bits = tensor.detach().cpu().contiguous().numpy().view(np.int16).astype(np.int32)
        keys.append(np.where(bits < 0, -(bits & 0x7FFF), bits))
    if keys[0].shape != keys[1].shape:
        return None
    return int(np.abs(keys[0] - keys[1]).max()) if keys[0].size else 0


def longest_clips(manifest: pd.DataFrame, count: int) -> tuple[str, ...]:
    """The segments of the longest clips, longest first; among equal lengths, by ID."""
    ordered = manifest.sort_values(["samples", "segment"], ascending=[False, True])
    return tuple(str(segment) for segment in ordered["segment"].head(count))


def compare(
    run: Run, cache: CacheRun, sample: Sequence[str], loaded: LoadedModel, state: Progress
) -> dict[str, Any]:
    """The sample of one cache run computed again from the audio and compared, features first."""
    model, processor = loaded.model, loaded.processor
    width = int(run.info["settings"]["width"])
    clips_dir = run.workspace.resolve(run.config["view_runs"][cache.view]["run"])
    clips = {str(row.segment): row for row in cache.clips.itertuples(index=False)}
    stored = {str(row.segment): row for row in cache.manifest.itertuples(index=False)}
    features_equal = outputs_equal = differ = 0
    steps: list[int | None] = []
    for segment in sample:
        state.segment = segment
        row = stored[segment]
        clip = read_view_clip(clips_dir, clips[segment])
        inputs = clip_inputs(processor, clip.audio)
        same_features = features_sha256(inputs["input_features"]) == row.features_sha256
        output = encoder_output(model, inputs)
        same_output = hashlib.sha256(to_bytes(output)).hexdigest() == row.sha256
        if not same_output:
            steps.append(fp16_steps(output, read_file(cache.folder / str(row.file), row, width)))
        features_equal += same_features
        outputs_equal += same_output
        differ += not (same_features and same_output)
    state.segment = None
    built = cache.info.get("environment") or {}
    mine = run.info["environment"]
    known = [step for step in steps if step is not None]
    result = {
        "view": cache.view,
        "manifest_sha256": cache.manifest_sha256,
        "segments": len(sample),
        "drawn": min(run.settings.segments_per_view, len(cache.manifest)),
        "two_chunk": int((cache.manifest["chunks"] > 1).sum()),
        "features_equal": features_equal,
        "outputs_equal": outputs_equal,
        "differ": differ,
        "largest_difference_fp16_steps": max(known) if known else None,
        "other_shapes": len(steps) - len(known),
        "same_cpu": built.get("cpu") == mine.get("cpu"),
        "same_gpu": built.get("gpu") == mine.get("gpu"),
    }
    mismatch = ""
    if steps:
        mismatch = f"; largest difference {result['largest_difference_fp16_steps']} fp16 steps"
        if result["other_shapes"]:
            mismatch += f", {result['other_shapes']} outputs of another shape"
    machine = (
        "the same CPU model and GPU as this job"
        if result["same_cpu"] and result["same_gpu"]
        else f"CPU {built.get('cpu')}, GPU {built.get('gpu')}, unlike this job"
    )
    print(
        f"{cache.run} ({cache.view}): features equal {features_equal} of {len(sample)};"
        f" encoder outputs equal {outputs_equal} of {len(sample)}{mismatch} ({result['drawn']}"
        f" drawn with seed {run.settings.seed}, {result['two_chunk']} of two chunks); built with"
        f" {machine}"
    )
    return result


def watch_language_model(
    run: Run, caches: Sequence[CacheRun], loaded: LoadedModel, state: Progress
) -> None:
    """The language model without its output layer, from the cache, on the longest clips."""
    model, processor = loaded.model, loaded.processor
    width = int(run.info["settings"]["width"])
    prompts: dict[int, dict[str, Any]] = {}
    encoder = use_cached_encoder(model)
    try:
        for cache in caches:
            stored = {str(row.segment): row for row in cache.manifest.itertuples(index=False)}
            longest = longest_clips(cache.manifest, run.settings.lm_watch_clips)
            clock = time.monotonic()
            with watch_layers(language_model_layers(model)) as watch:
                for segment in longest:
                    state.segment = segment
                    row = stored[segment]
                    count = int(row.chunks)
                    if count not in prompts:
                        prompts[count] = prompt_inputs(processor, count)
                    output = read_file(cache.folder / str(row.file), row, width)
                    lm_pass(model, prompts[count], output)
            state.segment = None
            summary = watch_summary(watch)
            seconds = [int(stored[segment].samples) / SAMPLING_RATE for segment in longest]
            two = sum(int(stored[segment].chunks) > 1 for segment in longest)
            run.info["results"]["runs"][cache.run]["lm_watch"] = {
                "clips": len(longest),
                "two_chunk_clips": two,
                "shortest_seconds": round(min(seconds), 3),
                "seconds": round(time.monotonic() - clock, 1),
                **summary,
            }
            run.save()
            print(
                f"{cache.run} ({cache.view}): the language model without its output layer on the"
                f" {len(longest)} longest clips ({two} of two chunks, the shortest"
                f" {min(seconds):.1f} s): " + watch_line(summary)
            )
            if summary["flagged"]:
                print(
                    f"{cache.run}: flagged: {', '.join(summary['flagged'])}; the caches stand,"
                    " and step 8 and the zero-shot job use the fallback of the pilot protocol"
                    " (the flagged layers in fp32)"
                )
    finally:
        restore_encoder(model, encoder)


def verify(run: Run, state: Progress) -> None:
    """--verify: each cache run checked, its sample computed again, then the language model."""
    info, settings = run.info, run.settings
    caches = [
        read_cache_run(run.workspace, run.config, name, run.voxtral) for name in run.mode.runs
    ]
    for cache in caches:
        if cache.info["settings"]["numerics"] != run.numerics:
            raise StepError(
                f"{cache.run} was built with other numeric settings than this job's; check it"
                " with the job file it was built with (scripts/slurm/build_caches.sbatch)"
            )
    info["inputs"]["runs"] = {
        cache.run: {
            "view": cache.view,
            "manifest_sha256": cache.manifest_sha256,
            "run_json_sha256": sha256_file(cache.folder / "run.json"),
            "view_run": cache.info["inputs"]["view"],
        }
        for cache in caches
    }
    views = sorted({cache.view for cache in caches})
    info["seeds"] = {
        "check_sample": {
            "seed": settings.seed,
            "streams": {
                view: f'numpy.random.default_rng(runinfo.item_seed({settings.seed}, "{SAMPLE_STREAM}",'
                f' "{view}"))'
                for view in views
            },
            "item_seeds": {view: item_seed(settings.seed, SAMPLE_STREAM, view) for view in views},
        }
    }
    samples = {
        cache.run: check_sample(
            cache.manifest, cache.view, settings.segments_per_view, settings.seed
        )
        for cache in caches
    }
    run.save()
    loaded = load(run)
    width = int(info["settings"]["width"])
    for cache in caches:
        if cache.info["settings"].get("width") != width:
            raise StepError(
                f"{cache.run} holds outputs of width {cache.info['settings'].get('width')}, not {width}"
            )

    results = info["results"]
    results["runs"] = {}
    clock = time.monotonic()
    for cache in caches:
        results["runs"][cache.run] = compare(run, cache, samples[cache.run], loaded, state)
        run.save()
    results["compare_seconds"] = round(time.monotonic() - clock, 1)
    differ = sum(result["differ"] for result in results["runs"].values())
    compared = sum(result["segments"] for result in results["runs"].values())
    run.save()
    if differ:
        raise StepError(
            f"{differ} of {compared} sampled segments differ from their caches in their features"
            " or encoder outputs: a fault to find and fix before any cache is pinned; the check"
            " runs again under the same rule"
        )

    clock = time.monotonic()
    watch_language_model(run, caches, loaded, state)
    results["lm_watch_seconds"] = round(time.monotonic() - clock, 1)
    results["peak_gpu_memory_gb"] = peak_memory_gb()
    minutes = (time.monotonic() - run.started) / 60
    info.update(status="done", finished_utc=utc_now(), minutes=round(minutes, 2))
    run.save()
    print(f"run: {run.relative}")
    print(f"done in {minutes:.1f} min")


# The job -------------------------------------------------------------------------------------


def cache_job(state: Progress) -> None:
    """The job: stage 1, then the stages of its mode."""
    run = start(state)
    if run.mode.name == VIEW_MODE:
        build(run, state)
    else:
        verify(run, state)


def main() -> None:
    os.umask(0o077)
    sys.unraisablehook = quiet_unraisable  # a finaliser's message could hold anything
    progress = Progress()
    try:
        cache_job(progress)
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
