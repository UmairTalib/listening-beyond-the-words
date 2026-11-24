"""Run the word check: check F4, the x0 control and the checks of the re-voiced clip x1 (GPU job).

Submit from an OMNI login node:
    scripts/slurm/submit.sh check_views
    scripts/slurm/submit.sh check_views --views <view> ...
    scripts/slurm/submit.sh --time=<limit> check_views --rerun derived/wordcheck/run-<job id>

Pilot protocol, check F4 and section "Clip versions", bullet "Word check"; the settings are
in configs/pilot/data.yaml (word_check), the pinned models in configs/pilot/models.yaml.
Three modes, whose options name a pinned view or an earlier run, never a setting:
  (default)    the pinned word-check sample in the views of word_check.transcribed_views,
               under every rule: check F4, the x0 control and the word and voice rules of x1
  --views V    only the named pinned views, on the sample, under the rules of check F4 (the
               backup of F4). It needs the word check's own run pinned (wordcheck_run),
               made from the same pinned inputs; once a leakage rerun of that run is
               pinned (leakage_rerun), their leakage is read on all segments, as the rerun
               read that of rev, and it refuses to run while a rerun that the run's x0
               control called for is not pinned
  --rerun RUN  the leakage part again, on all segments of x0 and rev, as the x0 control of
               the first run RUN asked. It refuses a first run that is not done, that comes
               from another commit, config or inputs, whose x0 control did not say RERUN
               NEEDED, or whose transcripts.parquet no longer matches its run.json. It keeps
               the first run's WER and prints the final F4 and x0 control.
The re-check of x1 on a new sample (seed 107) is fixed in the protocol and coded only if
x1's word rule fails.

Stages, in one process, one model at a time on the GPU:
  1. Guards: a Slurm job, offline, a GPU and a clean checkout; then the run folder. A rerun
     checks its first run here.
  2. Public guard, before any participant clip is read: Whisper, then wav2vec2, reads the
     public voice files of the tool check (word_check.public_guard) against that run's own
     text; a WER above the bound stops the job.
  3. Inputs: every pin checked; the sample, the labels and the partitions' round name the
     same participants; every clip read and checked against its manifest row; every
     reference built by the text rule (wordcheck.words), an empty one refused; and, in the
     default mode, the text-free class of every reference word.
  4. Work that needs no transcript: the five leakage classifiers, the bootstrap draws of the
     x0 control's interval and the gender-only control of the voice probe.
  5. Voice (default mode): x0 and x1 measured part by part and as whole clips, on copies at
     x1's level, and x0's voiced time at its stored level; the ICC gate and both probes.
  6. Whisper reads every clip once, views rev, x2, x0, x1, in segment order, then the first
     5 segments of each view again; it is then released.
  7. wav2vec2 the same; then transcripts.parquet is written, before any scoring.
  8. Scoring: WER, words heard and their chance level, odd outputs, leakage and its
     interval, x1's WER by part size, the errors of x1 and x0 by word class, the verdicts
     and the time to ask for a leakage rerun.
  9. The other outputs, then run.json ("done"). A failed rule is a result: the job still
     ends done. It fails only on its own faults.

Writes $LBW_WORKSPACE/derived/wordcheck/run-<job id>/ with
  transcripts.parquet     one row per segment, view and ASR system: the text, its words by
                          the text rule and the reading's diagnostics; pinned by its SHA-256
  voice_parts.parquet     the 88 voice measures of every part of x0 and x1 (default mode)
  voice_segments.parquet  the same per segment and per whole clip (default mode)
  scores.parquet          each participant's leakage and voice-probe scores
  errors.log              if the job fails: traceback frames and error types, never a
                          message (a job that fails before this folder exists prints them
                          to stderr)
  run.json                mode, commit, configs, inputs, models, settings (with Whisper's
                          options as faster-whisper resolved them in the public guard),
                          seeds, environment, timings, aggregate results and the verdicts
transcripts.parquet holds the ASR systems' text, and the other tables hold numbers per
segment and per participant: only the researcher opens them, on OMNI. The job prints
aggregate numbers only: never text, and never one segment's or participant's values.
"""

import contextlib
import gc
import json
import math
import os
import socket
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from listening_beyond_words.clips import (
    ViewClip,
    load_cut_words,
    load_pinned_labels,
    load_pinned_partitions,
    load_pinned_sample,
    load_segments,
    load_view_manifest,
    read_lines,
    read_view_clip,
)
from listening_beyond_words.models import ModelPin, load_pins, local_snapshot
from listening_beyond_words.partitions import SEEDS
from listening_beyond_words.recognisers import (
    Reading,
    Recogniser,
    Wav2vec2,
    Whisper,
    WhisperSettings,
)
from listening_beyond_words.revoice import (
    SESSIONS_CHECKSUMS,
    X1Settings,
    assign_voices,
    segment_words,
)
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
from listening_beyond_words.segments import segment_parts
from listening_beyond_words.settings import Workspace, get_workspace
from listening_beyond_words.views import PCM16_SCALE, SAMPLING_RATE
from listening_beyond_words.voicecheck import (
    FUNCTIONAL_NAMES,
    Smile,
    control_aurocs,
    gate_values,
    level_copy,
    part_measures,
    part_weights,
    participant_means,
    probe_aurocs,
    probe_scores,
    segment_measures,
    voice_iccs,
    voiced_seconds,
)
from listening_beyond_words.wordcheck import (
    BOOTSTRAP_KEYS,
    RECOGNISERS,
    RERUN_NEEDED,
    SHORT_PART_WORDS,
    VERY_REPETITIVE,
    BootstrapSettings,
    LeakageSettings,
    Results,
    Rules,
    WerCounts,
    bootstrap_interval,
    bootstrap_sets,
    chance_partners,
    check_references,
    error_classes,
    f4_verdict,
    fit_leakage,
    fold_aurocs,
    line_classes,
    person_scores,
    reference_classes,
    repetition_share,
    text_counts,
    verdict_lines,
    verdicts,
    wer_counts,
    words,
    x0_control_verdict,
)

CONFIG = Path("configs/pilot/data.yaml")
MODELS = Path("configs/pilot/models.yaml")
STEP = "check_views"
SAMPLE, VIEWS, RERUN = "sample", "views", "rerun"
USAGE = "usage: check_views.py [--views <view> ... | --rerun derived/wordcheck/run-<job id>]"
READING_ORDER = ("rev", "x2", "x0", "x1")  # the least predictable views first
RULE_VIEWS = ("rev", "x0", "x1")  # the views that the rules read; x2 is reported only
RERUN_VIEWS = ("x0", "rev")
INTERVAL_VIEWS = ("x0", "rev")  # x0's interval is a rule, rev's is reported
VOICE_VIEWS = ("x0", "x1")
REPEATS = 5  # the first segments of each view, read a second time
PROGRESS_EVERY = 100  # clips (or segments) between two progress lines
LONG_CLIP_SECONDS = 30.0  # Whisper reads 30 s of audio at once
RERUN_OVERHEAD_SECONDS = 15 * 60  # loads, the public guard and the clip checks of a rerun
GPU_LIMIT_HOURS = 24.0  # the time limit of the gpu partition
MODEL_PINS = {"whisper": "faster_whisper", "wav2vec2": "wav2vec2_960h"}
# The pinned inputs of every mode, but the views and the public guard (input_pins).
SHARED_INPUTS = ("segments", "labels", "partitions", "wordcheck_sample", "cut_words", "transcripts")
GUARD_STEP = "check_view_tools"
WORD_CHECK_KEYS = (
    "sample",
    "transcribed_views",
    "partition_round",
    "whisper",
    "wav2vec2",
    "rules",
    "leakage",
    "bootstrap",
    "voice",
    "recheck_seed",
    "rerun_time_factor",
    "public_guard",
)
VOICE_KEYS = ("measures", "level_rms", "min_part_samples", "probe_logistic_c", "probe_max_iter")
WAV2VEC2_SETTINGS = {"dtype": "float32", "decoding": "greedy"}  # what recognisers.Wav2vec2 does
DIAGNOSTIC_KEYS = (  # recognisers.WHISPER_DIAGNOSTICS and WAV2VEC2_DIAGNOSTICS, one column each
    "windows",
    "lowest_mean_log_prob",
    "highest_no_speech_prob",
    "default_dropped_windows",
    "default_dropped_words",
    "frames",
    "special_symbols",
)
PACKAGES = (
    "faster-whisper",
    "ctranslate2",
    "transformers",
    "tokenizers",
    "torch",
    "nvidia-cublas-cu12",
    "opensmile",
    "jiwer",
    "scikit-learn",
    "num2words",
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "soundfile",
    "pyyaml",
)


# Modes and settings ------------------------------------------------------------------------


@dataclass(frozen=True)
class Mode:
    """What the job reads: the default sample, named views, or the rerun of a first run."""

    name: str
    views: tuple[str, ...] = ()
    first_run: str = ""

    def record(self) -> dict[str, Any]:
        """The mode as run.json records it."""
        record: dict[str, Any] = {"name": self.name}
        if self.name == VIEWS:
            record["views"] = list(self.views)
        if self.name == RERUN:
            record["first_run"] = self.first_run
        return record


def parse_mode(argv: Sequence[str]) -> Mode:
    """The mode from the command line; anything else is refused with the usage."""
    if not argv:
        return Mode(SAMPLE)
    flag, *rest = argv
    if (
        flag == "--views"
        and rest
        and not any(view.startswith("-") for view in rest)
        and len(set(rest)) == len(rest)
    ):
        return Mode(VIEWS, views=tuple(rest))
    if flag == "--rerun" and len(rest) == 1 and not rest[0].startswith("-"):
        return Mode(RERUN, first_run=rest[0])
    raise StepError(USAGE)


@dataclass(frozen=True)
class JobSettings:
    """The word check's settings, from configs/pilot/data.yaml (word_check), checked."""

    views: tuple[str, ...]
    partition_round: int
    rules: Rules
    leakage: LeakageSettings
    bootstrap: BootstrapSettings
    whisper: WhisperSettings
    measures: tuple[str, ...]
    level_rms: float
    min_part_samples: int
    probe_c: float
    probe_max_iter: int
    peak_limit: float
    voices: Mapping[str, Sequence[str]]
    rerun_time_factor: float
    guard_run: str
    guard_max_wer: float

    def resolved(self) -> dict[str, Any]:
        """The settings as the job reads them, for run.json."""
        return {
            "views": list(self.views),
            "partition_round": self.partition_round,
            "rules": asdict(self.rules),
            "leakage": asdict(self.leakage),
            "bootstrap": asdict(self.bootstrap),
            "whisper": {
                "compute_type": self.whisper.compute_type,
                "options": dict(self.whisper.options),
            },
            "wav2vec2": dict(WAV2VEC2_SETTINGS),
            "voice": {
                "measures": list(self.measures),
                "level_rms": self.level_rms,
                "peak_limit": self.peak_limit,
                "min_part_samples": self.min_part_samples,
                "probe_logistic_c": self.probe_c,
                "probe_max_iter": self.probe_max_iter,
            },
            "x1_voices": {gender: list(names) for gender, names in self.voices.items()},
            "rerun_time_factor": self.rerun_time_factor,
            "public_guard": {"run": self.guard_run, "max_wer": self.guard_max_wer},
        }


def _keys(block: Any, expected: Sequence[str], name: str) -> None:
    """Refuse a config block that is not a mapping, or that has a missing or unknown key."""
    if not isinstance(block, Mapping):
        raise StepError(f"{name} must be a mapping")
    missing = sorted(set(expected) - set(block))
    unknown = sorted(str(key) for key in set(block) - set(expected))
    if missing or unknown:
        raise StepError(f"{name}: missing keys {missing}, unknown keys {unknown}")


def _positive(value: Any, name: str) -> float:
    """A positive finite number from the config."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value < math.inf:
        raise StepError(f"{name} must be a positive finite number")
    return float(value)


def _whole(value: Any, name: str) -> int:
    """A whole number from the config."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise StepError(f"{name} must be a whole number")
    return value


def read_settings(config: Mapping[str, Any]) -> JobSettings:
    """Read and check every setting of the word check that this job uses."""
    block = config["word_check"]
    _keys(block, WORD_CHECK_KEYS, "word_check")
    views = block["transcribed_views"]
    if (
        not isinstance(views, list)
        or not all(isinstance(view, str) for view in views)
        or len(set(views)) != len(views)
        or not set(RULE_VIEWS) <= set(views)
    ):
        raise StepError("word_check.transcribed_views must name x0, x1 and rev, each view once")
    unpinned = [view for view in views if view not in config["view_runs"]]
    if unpinned:
        raise StepError(f"word_check.transcribed_views names views without a pin: {unpinned}")
    partition_round = _whole(block["partition_round"], "word_check.partition_round")
    if not 1 <= partition_round <= len(SEEDS):
        raise StepError(f"word_check.partition_round must be one of 1 to {len(SEEDS)}")
    if block["wav2vec2"] != WAV2VEC2_SETTINGS:
        raise StepError("word_check.wav2vec2 must be {dtype: float32, decoding: greedy}")
    voice = block["voice"]
    _keys(voice, VOICE_KEYS, "word_check.voice")
    measures = voice["measures"]
    if (
        not isinstance(measures, list)
        or not measures
        or len(set(measures)) != len(measures)
        or not set(measures) <= set(FUNCTIONAL_NAMES)
    ):
        raise StepError("word_check.voice.measures must name eGeMAPSv02 functionals, each once")
    guard = block["public_guard"]
    _keys(guard, ("run", "max_wer"), "word_check.public_guard")
    factor = _positive(block["rerun_time_factor"], "word_check.rerun_time_factor")
    if factor < 1.0:
        raise StepError("word_check.rerun_time_factor must be at least 1")
    x1 = X1Settings.from_config(config)
    return JobSettings(
        views=tuple(views),
        partition_round=partition_round,
        rules=Rules.from_config(config),
        leakage=LeakageSettings.from_config(config),
        bootstrap=BootstrapSettings.from_config(config),
        whisper=WhisperSettings.from_config(config),
        measures=tuple(measures),
        level_rms=_positive(voice["level_rms"], "word_check.voice.level_rms"),
        min_part_samples=_whole(voice["min_part_samples"], "word_check.voice.min_part_samples"),
        probe_c=_positive(voice["probe_logistic_c"], "word_check.voice.probe_logistic_c"),
        probe_max_iter=_whole(voice["probe_max_iter"], "word_check.voice.probe_max_iter"),
        peak_limit=x1.peak_limit,
        voices=x1.voices,
        rerun_time_factor=factor,
        guard_run=str(guard["run"]),
        guard_max_wer=_positive(guard["max_wer"], "word_check.public_guard.max_wer"),
    )


def in_reading_order(views: Sequence[str]) -> tuple[str, ...]:
    """The views in the order the ASR systems read them; a backup view of rev comes first."""
    return tuple(
        sorted(views, key=lambda view: READING_ORDER.index(view) if view in READING_ORDER else -1)
    )


# The machine and the models ----------------------------------------------------------------


def gpu_facts() -> dict[str, Any]:
    """The GPU, CUDA and threads of this job. PyTorch is imported here, before faster-whisper."""
    import torch  # first: CTranslate2 then finds the cuBLAS that PyTorch loaded

    available = torch.cuda.is_available()
    facts: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0) if available else None,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if available else None,
        "torch_threads": torch.get_num_threads(),
    }
    if available:
        import ctranslate2

        facts["ctranslate2_cuda_compute_types"] = sorted(
            ctranslate2.get_supported_compute_types("cuda")
        )
    return facts


def release() -> None:
    """Free the model just used before the next one loads: one model at a time on the GPU."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def snapshot_record(pin: ModelPin, snapshot: Path) -> dict[str, Any]:
    """A pinned snapshot for run.json: repo, revision and the blob that each file links to."""
    blobs = {
        path.name: path.readlink().name for path in sorted(snapshot.iterdir()) if path.is_symlink()
    }
    return {"repo_id": pin.repo_id, "revision": pin.revision, "blobs": blobs}


def load_whisper(snapshot: Path, settings: WhisperSettings) -> Whisper:
    """Whisper large-v3 from its pinned snapshot, offline, in the settings' compute type."""
    return Whisper.load(snapshot, settings)


def load_wav2vec2(snapshot: Path) -> Wav2vec2:
    """wav2vec2 from its pinned snapshot, offline, in float32."""
    return Wav2vec2.load(snapshot)


def utc_in(seconds: float) -> str:
    """The UTC time this many seconds from now, to the minute."""
    return (datetime.now(UTC) + timedelta(seconds=seconds)).strftime("%H:%M UTC")


# Stage 1: the run ---------------------------------------------------------------------------


@dataclass
class Run:
    """The run being made: where it writes, what it records and what it read so far."""

    mode: Mode
    job: str
    workspace: Workspace
    config: dict[str, Any]
    settings: JobSettings
    pins: dict[str, ModelPin]
    run_dir: Path
    info: dict[str, Any]
    started: float
    stages: dict[str, float] = field(default_factory=dict)

    def save(self) -> None:
        """Write run.json as it stands."""
        write_json(self.run_dir / "run.json", self.info)

    @property
    def relative(self) -> Path:
        """The run folder, relative to the workspace."""
        return self.run_dir.relative_to(self.workspace.root)


def start(state: Progress) -> Run:
    """Stage 1: refuse to run outside a clean offline GPU job; make the run folder."""
    stop_on_sigterm()
    # Before transformers loads, which reads it for its own progress bars too.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    mode = parse_mode(sys.argv[1:])
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise StepError("submit this as a job: scripts/slurm/submit.sh check_views")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise StepError(
            "HF_HUB_OFFLINE is not 1: the job must run offline (scripts/slurm/common.sh)"
        )
    git = git_state()
    if git["dirty"]:
        raise StepError("the checkout has uncommitted changes; commit them on the Mac and pull")
    gpu = gpu_facts()
    if not gpu.get("gpu"):
        raise StepError("no CUDA GPU: submit this as a GPU job (scripts/slurm/check_views.sbatch)")
    started = time.monotonic()
    workspace = get_workspace()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    settings = read_settings(config)
    pins = load_pins(MODELS)
    if mode.name == VIEWS:
        unpinned = [view for view in mode.views if view not in config["view_runs"]]
        if unpinned:
            raise StepError(f"--views names views without a pin in view_runs: {unpinned}")
    run_dir = workspace.derived / "wordcheck" / f"run-{job}"
    run_dir.mkdir(parents=True, exist_ok=False)
    state.run_dir = run_dir
    seed = settings.bootstrap.seed
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
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
        "settings": {
            "word_check": config["word_check"],
            "peak_limit": config["views"]["peak_limit"],
            "x1_voices": config["views"]["x1"]["voices"],
            "resolved": settings.resolved(),
        },
        "seeds": {
            "partitions": SEEDS[settings.partition_round - 1],
            "bootstrap": {
                "seed": seed,
                "stream": "numpy.random.default_rng(runinfo.item_seed(seed, "
                + ", ".join(f'"{key}"' for key in BOOTSTRAP_KEYS)
                + "))",
                "item_seed": item_seed(seed, *BOOTSTRAP_KEYS),
            },
        },
        "environment": {
            **environment(PACKAGES),
            **gpu,
            "libsndfile": sf.__libsndfile_version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        },
    }
    state.info = info
    run = Run(mode, job, workspace, config, settings, pins, run_dir, info, started)
    run.save()
    print(
        f"word check, mode {mode.name}"
        + (f" ({', '.join(mode.views)})" if mode.name == VIEWS else "")
        + (f" of {mode.first_run}" if mode.name == RERUN else "")
        + f"; job {job} on {socket.gethostname()}, {gpu['gpu']}"
    )
    return run


def input_pins(run: Run, views: Sequence[str]) -> dict[str, Any]:
    """Every pinned input of the job, as run.json records it and a rerun compares it."""
    config, workspace = run.config, run.workspace
    pins: dict[str, Any] = {
        name: {"file": config[name]["run"], "sha256": config[name]["sha256"]}
        for name in ("segments", "labels", "partitions", "wordcheck_sample")
    }
    pins["cut_words"] = {
        "file": str(Path(config["segments"]["run"]) / "words.parquet"),
        "sha256": sha256_file(workspace.resolve(config["segments"]["run"]) / "words.parquet"),
    }
    pins["transcripts"] = {
        "file": str(Path("edaic") / SESSIONS_CHECKSUMS),
        "sha256": sha256_file(workspace.edaic / SESSIONS_CHECKSUMS),
    }
    pins["views"] = {
        view: {
            "file": config["view_runs"][view]["run"],
            "sha256": config["view_runs"][view]["sha256"],
        }
        for view in views
    }
    return pins


def pinned_run(run: Run, key: str, name: str) -> dict[str, Any]:
    """The run.json of a pinned word-check run, once its transcripts.parquet matches the pin."""
    pin = run.config[key]
    folder = run.workspace.resolve(pin["run"])
    transcripts = folder / "transcripts.parquet"
    if not transcripts.is_file() or sha256_file(transcripts) != pin["sha256"]:
        raise StepError(f"the pinned {name}'s transcripts.parquet does not match its pin")
    path = folder / "run.json"
    info = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    run.info["inputs"][key] = {"file": pin["run"], "sha256": pin["sha256"]}
    return info


def leakage_rerun_pin(run: Run) -> bool:
    """Whether the --views mode reads its leakage on all segments: a leakage rerun is pinned.

    The backup of F4 is checked after the word check, on its sample, so its run must be
    pinned (wordcheck_run) and made from the same pinned inputs. If that run's x0 control
    called for the leakage rerun, the rerun must be pinned too (leakage_rerun), as a
    finished rerun of that run: the backup's leakage is then read on all segments.
    """
    if not run.config.get("wordcheck_run"):
        raise StepError(
            "--views checks a backup of F4 after the word check: pin that run (wordcheck_run)"
        )
    first = pinned_run(run, "wordcheck_run", "word-check run")
    if (
        first.get("step") != STEP
        or (first.get("mode") or {}).get("name") != SAMPLE
        or first.get("status") != "done"
    ):
        raise StepError("the pinned word-check run is not a finished run on the sample")
    mine = json.loads(json.dumps({key: run.info["inputs"][key] for key in SHARED_INPUTS}))
    if {key: first["inputs"].get(key) for key in SHARED_INPUTS} != mine:
        raise StepError("the pinned word-check run was made from other pinned inputs")
    called = first["verdicts"]["x0_control"]["verdict"] == RERUN_NEEDED
    if not run.config.get("leakage_rerun"):
        if called:
            raise StepError(
                "the pinned word-check run's x0 control called for the leakage rerun: pin the"
                " rerun (leakage_rerun) first, so that the backup's leakage is read on all"
                " segments"
            )
        return False
    rerun = pinned_run(run, "leakage_rerun", "leakage rerun")
    if (
        rerun.get("step") != STEP
        or rerun.get("mode") != {"name": RERUN, "first_run": run.config["wordcheck_run"]["run"]}
        or rerun.get("status") != "done"
    ):
        raise StepError("the pinned leakage rerun is not a finished rerun of the word-check run")
    return True


def first_run(run: Run, folder: str) -> dict[str, Any]:
    """The first run that a rerun follows: done, the same code and inputs, and a rerun asked."""
    path = run.workspace.resolve(folder) / "run.json"
    if not path.is_file():
        raise StepError(f"no word-check run in {folder}")
    first = json.loads(path.read_text(encoding="utf-8"))
    if first.get("step") != STEP or (first.get("mode") or {}).get("name") != SAMPLE:
        raise StepError("the first run is not a run of the word check on the sample")
    if first.get("status") != "done":
        raise StepError("the first run did not finish: its run.json does not say done")
    if first["git"]["commit"] != run.info["git"]["commit"] or first["git"]["dirty"]:
        raise StepError("the first run was made with another commit; rerun from its commit")
    if (
        first["config"]["sha256"] != run.info["config"]["sha256"]
        or first["models_config"]["sha256"] != run.info["models_config"]["sha256"]
    ):
        raise StepError("the first run was made with another config")
    # The public guard is checked again in this run's own stage 2; every other pin must match.
    theirs = {key: value for key, value in first["inputs"].items() if key != "public_guard"}
    if theirs != json.loads(json.dumps(run.info["inputs"])):
        raise StepError("the first run was made from other pinned inputs")
    verdict = first["verdicts"]["x0_control"]["verdict"]
    if verdict != RERUN_NEEDED:
        raise StepError(f"the first run's x0 control said {verdict}, not {RERUN_NEEDED}")
    transcripts = path.parent / "transcripts.parquet"
    if (
        not transcripts.is_file()
        or sha256_file(transcripts) != first["outputs"]["transcripts.parquet"]
    ):
        raise StepError("the first run's transcripts.parquet does not match its run.json")
    wer = first["results"]["wer"]
    return {
        "run": folder,
        "run_json_sha256": sha256_file(path),
        "transcripts_sha256": first["outputs"]["transcripts.parquet"],
        "wer": {
            view: {name: float(wer[view][name]) for name in RECOGNISERS} for view in RULE_VIEWS
        },
    }


# Stage 2: the public guard -------------------------------------------------------------------


@dataclass
class Guard:
    """The tool check's public voice files and their text."""

    record: dict[str, Any]
    text: str
    files: list[tuple[str, np.ndarray]]


def guard_files(workspace: Workspace, settings: JobSettings) -> Guard:
    """The public voice files of the tool check, each checked against that run's run.json."""
    folder = workspace.resolve(settings.guard_run)
    path = folder / "run.json"
    if not path.is_file():
        raise StepError(f"the public guard's run {settings.guard_run} has no run.json")
    info = json.loads(path.read_text(encoding="utf-8"))
    if (
        info.get("step") != GUARD_STEP
        or info.get("status") != "done"
        or info.get("verdict") != "PASS"
    ):
        raise StepError("the public guard's run is not a tool check that passed")
    outputs = info.get("outputs") or {}
    files: list[tuple[str, np.ndarray]] = []
    for voice in info["settings"]["voices"]:
        name = f"voices/{voice}.wav"
        file = folder / name
        if name not in outputs or not file.is_file() or sha256_file(file) != outputs[name]:
            raise StepError(f"the public guard's {name} does not match its run.json")
        sound = sf.info(file)
        if (sound.samplerate, sound.channels, sound.subtype) != (SAMPLING_RATE, 1, "PCM_16"):
            raise StepError(f"the public guard's {name} is not 16 kHz mono 16-bit PCM")
        samples, _ = sf.read(file, dtype="int16")
        files.append((str(voice), (np.asarray(samples) / PCM16_SCALE).astype(np.float32)))
    if not files:
        raise StepError("the public guard's run lists no voice files")
    record = {"run": settings.guard_run, "run_json_sha256": sha256_file(path), "files": len(files)}
    return Guard(record=record, text=str(info["settings"]["text"]), files=files)


def guard_read(
    load: Callable[[], Recogniser], guard: Guard
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Load one ASR system and score its reading of the public files; it is dropped on return."""
    started = time.monotonic()
    recogniser = load()
    loaded = time.monotonic() - started
    readings = [recogniser.read(audio) for _, audio in guard.files]
    names = [voice for voice, _ in guard.files]
    reference = words(guard.text)
    counts = wer_counts(
        [reference] * len(readings), [words(reading.text) for reading in readings], names
    )
    by_voice = counts.per_segment.set_index("segment")["wer"]
    numbers = {
        **counts.totals(),
        "wer_by_voice": {name: float(by_voice[name]) for name in names},
        "load_seconds": round(loaded, 1),
        "read_seconds": round(sum(reading.seconds for reading in readings), 2),
    }
    options = getattr(recogniser, "resolved_options", None)
    return numbers, (dict(options) if options is not None else None)


def public_guard(run: Run) -> dict[str, Path]:
    """Stage 2: each ASR system reads the public files, before any participant clip is read."""
    clock = time.monotonic()
    settings, workspace = run.settings, run.workspace
    guard = guard_files(workspace, settings)
    snapshots = {
        name: local_snapshot(run.pins[pin], workspace.hf_home) for name, pin in MODEL_PINS.items()
    }
    run.info["models"] = {
        name: snapshot_record(run.pins[pin], snapshots[name]) for name, pin in MODEL_PINS.items()
    }
    run.info["inputs"]["public_guard"] = guard.record
    loaders: dict[str, Callable[[], Recogniser]] = {
        "whisper": lambda: load_whisper(snapshots["whisper"], settings.whisper),
        "wav2vec2": lambda: load_wav2vec2(snapshots["wav2vec2"]),
    }
    results: dict[str, Any] = {}
    run.info["public_guard"] = results
    for name in RECOGNISERS:
        numbers, options = guard_read(loaders[name], guard)
        release()
        results[name] = numbers
        if name == "whisper":
            if options is None:
                raise StepError("Whisper gave no resolved options in the public guard")
            run.info["settings"]["whisper_resolved"] = options
        print(
            f"public guard, {name}: WER {numbers['wer']:.4f} on {len(guard.files)} public voice"
            f" files ({numbers['substitutions']} substituted, {numbers['deletions']} deleted,"
            f" {numbers['insertions']} inserted, of {numbers['reference_words']} words);"
            f" loaded in {numbers['load_seconds']:.0f} s"
        )
        if numbers["wer"] > settings.guard_max_wer:
            raise StepError(
                f"the public guard failed: {name}'s WER {numbers['wer']:.4f} on the tool check's"
                f" public text is above {settings.guard_max_wer}; no participant clip was read"
            )
    run.stages["public_guard"] = time.monotonic() - clock
    run.save()
    return snapshots


# Stage 3: the inputs ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """What one run reads: views in reading order, and the segments of WER and of leakage."""

    views: tuple[str, ...]
    wer_segments: tuple[str, ...]
    leakage_segments: tuple[str, ...]
    voice: bool
    read_segments: tuple[str, ...]


def make_plan(mode: Mode, settings: JobSettings, inputs: "Inputs", leakage_rerun: bool) -> Plan:
    """What this mode reads: segments in ID order, views in reading order."""
    sampled = tuple(str(segment) for segment in inputs.sample["segment"])
    everything = tuple(sorted(inputs.session_of))
    if mode.name == SAMPLE:
        views, wer, leakage = in_reading_order(settings.views), sampled, sampled
    elif mode.name == VIEWS:
        views, wer = in_reading_order(mode.views), sampled
        leakage = everything if leakage_rerun else sampled
    else:
        views, wer, leakage = in_reading_order(RERUN_VIEWS), everything, everything
    read = tuple(sorted(set(wer) | set(leakage)))
    return Plan(views, wer, leakage, voice=mode.name == SAMPLE, read_segments=read)


@dataclass
class Inputs:
    """The checked inputs of the word check."""

    segments: pd.DataFrame
    parts: pd.DataFrame
    labels: pd.DataFrame
    folds: pd.Series
    voices: pd.Series
    sample: pd.DataFrame
    session_of: dict[str, int]
    manifests: dict[str, pd.DataFrame]
    clips: dict[str, dict[str, ViewClip]] = field(default_factory=dict)
    references: dict[str, list[str]] = field(default_factory=dict)
    classes: pd.DataFrame | None = None

    @property
    def label(self) -> pd.Series:
        """Each participant's label, 0 or 1, by session ID."""
        return self.labels["label"].astype(np.int64)

    @property
    def gender(self) -> pd.Series:
        """Each participant's gender, by session ID."""
        return self.labels["gender"].astype(str)


def load_tables(run: Run, views: Sequence[str]) -> Inputs:
    """The pinned tables, checked against each other; no clip is read here."""
    workspace, config, settings = run.workspace, run.config, run.settings
    segments = load_segments(workspace, config)
    parts = segment_parts(load_cut_words(workspace, config))
    labels = load_pinned_labels(workspace, config)
    labels.index = labels.index.astype(np.int64)
    sample = load_pinned_sample(workspace, config).sort_values("segment", ignore_index=True)
    partitions = load_pinned_partitions(workspace, config)
    chosen = partitions[partitions["round"] == settings.partition_round]
    folds = pd.Series(
        chosen["fold"].to_numpy(dtype=np.int64),
        index=pd.Index(chosen["session"].to_numpy(dtype=np.int64), name="session"),
        name="fold",
    ).sort_index()
    session_of = {
        str(segment): int(session)
        for segment, session in zip(segments["segment"], segments["session"], strict=True)
    }
    part_counts = parts.groupby("segment").size()
    if set(part_counts.index) != set(session_of) or any(
        int(part_counts[segment]) != int(count)
        for segment, count in zip(segments["segment"], segments["parts"], strict=True)
    ):
        raise StepError("the cut words do not give the pinned segments and their parts")
    if sample["segment"].duplicated().any() or any(
        session_of.get(str(segment)) != int(session)
        for segment, session in zip(sample["segment"], sample["session"], strict=True)
    ):
        raise StepError("the word-check sample holds segments that are not pinned segments")
    sampled = {int(session) for session in sample["session"]}
    labelled = {int(session) for session in labels.index}
    folded = {int(session) for session in folds.index}
    if not sampled == labelled == folded:
        raise StepError(
            f"the sample, the labels and round {settings.partition_round} of the partitions do"
            f" not name the same participants ({len(sampled)}, {len(labelled)} and"
            f" {len(folded)})"
        )
    if not set(session_of.values()) <= labelled:
        raise StepError("a participant with segments has no label")
    manifests = {}
    for view in views:
        manifest = load_view_manifest(workspace, config, view)
        if manifest["segment"].duplicated().any() or set(manifest["segment"]) != set(session_of):
            raise StepError(f"the manifest of {view} does not hold every pinned segment once")
        indexed = manifest.set_index("segment", drop=False)
        if any(
            int(indexed.at[segment, "session"]) != session
            or len(indexed.at[segment, "part_starts"]) != int(part_counts[segment])
            for segment, session in session_of.items()
        ):
            raise StepError(f"the manifest of {view} does not have the segments' parts")
        manifests[view] = indexed
    voices = pd.Series(assign_voices(labels, settings.voices), name="voice")
    voices.index = pd.Index(voices.index.astype(np.int64), name="session")
    return Inputs(
        segments=segments,
        parts=parts,
        labels=labels,
        folds=folds,
        voices=voices.sort_index(),
        sample=sample,
        session_of=session_of,
        manifests=manifests,
    )


def read_clips(run: Run, inputs: Inputs, plan: Plan, state: Progress) -> dict[str, Any]:
    """Read and check every clip of the plan against its manifest row; return the counts."""
    started = time.monotonic()
    wanted = set(plan.read_segments)
    long_clips: dict[str, int] = {}
    hours: dict[str, float] = {}
    for view in plan.views:
        folder = run.workspace.resolve(run.config["view_runs"][view]["run"])
        rows = {
            str(row.segment): row
            for row in inputs.manifests[view].itertuples(index=False)
            if str(row.segment) in wanted
        }
        clips: dict[str, ViewClip] = {}
        for segment in plan.read_segments:
            state.segment = segment
            clips[segment] = read_view_clip(folder, rows[segment])
        state.segment = None
        inputs.clips[view] = clips
        seconds = np.array([len(clip.samples) for clip in clips.values()]) / SAMPLING_RATE
        long_clips[view] = int((seconds > LONG_CLIP_SECONDS).sum())
        hours[view] = float(seconds.sum()) / 3600
    count = len(plan.views) * len(plan.read_segments)
    print(
        f"clips: {count:,} read and checked against their manifests in"
        f" {time.monotonic() - started:.0f} s ({sum(hours.values()):.2f} h of audio); over"
        f" {LONG_CLIP_SECONDS:.0f} s: "
        + ", ".join(f"{view} {long_clips[view]}" for view in plan.views)
    )
    return {"clips": count, "clips_over_30_s": long_clips, "audio_hours": hours}


def build_references(inputs: Inputs, plan: Plan, state: Progress) -> dict[str, int]:
    """Each WER segment's reference words by the text rule; an empty one is refused."""
    texts = dict(zip(inputs.segments["segment"], inputs.segments["text"], strict=True))
    segments = list(plan.wer_segments)
    for segment in segments:
        state.segment = segment
        inputs.references[segment] = words(str(texts[segment]))
    state.segment = None
    check_references([inputs.references[segment] for segment in segments], segments)
    counts = text_counts(str(texts[segment]) for segment in segments)
    counts["reference_words"] = sum(len(inputs.references[segment]) for segment in segments)
    print(
        f"references: {len(segments):,} segments, {counts['words']:,} stored words; four-digit"
        f" runs {counts['four_digit_runs']}, words mixing digits and letters"
        f" {counts['mixed_words']}, words with a digit {counts['words_with_a_digit']},"
        f" one-letter words {counts['one_letter_words']:,}, letter runs joined"
        f" {counts['letter_runs_joined']}; {counts['reference_words']:,} words after the text rule"
    )
    return counts


def build_classes(run: Run, inputs: Inputs, plan: Plan, state: Progress) -> pd.DataFrame:
    """The text-free classes of every reference word, from the stored words and transcript lines."""
    texts = dict(zip(inputs.segments["segment"], inputs.segments["text"], strict=True))
    by_segment = {str(segment): group for segment, group in inputs.parts.groupby("segment")}
    sessions = sorted({inputs.session_of[segment] for segment in plan.wer_segments})
    lines = {session: read_lines(run.workspace.edaic / "sessions", session) for session in sessions}
    tables = []
    for segment in plan.wer_segments:
        state.segment = segment
        group = by_segment[segment].sort_values("position")
        session = inputs.session_of[segment]
        stored = str(texts[segment]).split()
        if segment_words(group, lines[session]) != stored:
            raise StepError(
                f"segment {segment}: its transcript lines do not give its stored words",
                segment=segment,
            )
        flags = pd.concat(
            [
                line_classes(lines[session][int(row.line)], int(row.first), int(row.last))
                for row in group.itertuples(index=False)
            ],
            ignore_index=True,
        )
        table = reference_classes(segment, stored, group["n_words"].tolist(), flags)
        if len(table) != len(inputs.references[segment]):
            raise StepError(
                f"segment {segment}: its word classes do not cover its reference", segment=segment
            )
        tables.append(table)
    state.segment = None
    return pd.concat(tables, ignore_index=True)


def gather(run: Run, state: Progress, leakage_rerun: bool) -> tuple[Inputs, Plan]:
    """Stage 3: the inputs, every clip and every reference, checked."""
    clock = time.monotonic()
    mode = run.mode
    inputs = load_tables(run, mode.views if mode.name == VIEWS else run.settings.views)
    plan = make_plan(mode, run.settings, inputs, leakage_rerun)
    print(
        f"inputs: {len(inputs.labels)} participants, {len(inputs.sample):,} sampled segments of"
        f" {len(inputs.segments):,}; views {', '.join(plan.views)}; WER on"
        f" {len(plan.wer_segments):,} segments, leakage on {len(plan.leakage_segments):,}"
    )
    results = run.info.setdefault("results", {})
    results["inputs"] = {
        "participants": len(inputs.labels),
        "segments": len(inputs.segments),
        "sampled_segments": len(inputs.sample),
        "wer_segments": len(plan.wer_segments),
        "leakage_segments": len(plan.leakage_segments),
        **read_clips(run, inputs, plan, state),
    }
    results["references"] = build_references(inputs, plan, state)
    if mode.name == SAMPLE:
        inputs.classes = build_classes(run, inputs, plan, state)
    run.stages["inputs"] = time.monotonic() - clock
    run.save()
    return inputs, plan


# Stage 4: work that needs no transcript -------------------------------------------------------


@dataclass
class Prepared:
    """The leakage classifiers, the bootstrap draws and the probe's control."""

    models: dict[int, Any]
    sets: np.ndarray | None = None
    redraws: int = 0
    control: np.ndarray | None = None


def prepare(run: Run, inputs: Inputs, plan: Plan) -> Prepared:
    """Stage 4: fit the five classifiers, draw the bootstrap sets and fit the probe's control."""
    clock = time.monotonic()
    settings = run.settings
    texts = inputs.segments[["session", "text"]].copy()
    prepared = Prepared(models=fit_leakage(texts, inputs.label, inputs.folds, settings.leakage))
    print(
        f"leakage classifiers: {len(prepared.models)} folds of round {settings.partition_round},"
        f" each fitted on the stored words of every segment of its training participants"
        f" ({len(texts):,} segments in all)"
    )
    if run.mode.name in (SAMPLE, RERUN):
        prepared.sets, prepared.redraws = bootstrap_sets(
            inputs.label, inputs.folds, settings.bootstrap.rng(), settings.bootstrap.resamples
        )
        print(
            f"bootstrap: {len(prepared.sets):,} resamples of the participants within each class,"
            f" from their own stream (seed {settings.bootstrap.seed}); {prepared.redraws} drawn"
            " again because a fold lost a class"
        )
    if plan.voice:
        prepared.control = control_aurocs(
            inputs.gender, inputs.label, inputs.folds, settings.probe_c, settings.probe_max_iter
        )
        bar = float(prepared.control.mean()) + settings.rules.probe_margin_auroc
        print(f"voice probe control (gender only): {prepared.control.mean():.4f}; bar {bar:.4f}")
    run.info["results"]["bootstrap"] = {
        "resamples": 0 if prepared.sets is None else len(prepared.sets),
        "redraws": prepared.redraws,
    }
    run.stages["prepare"] = time.monotonic() - clock
    return prepared


# Stage 5: voice ---------------------------------------------------------------------------------


@dataclass
class Voice:
    """The voice measures and what the voice rule reads."""

    parts: pd.DataFrame
    segments: pd.DataFrame
    iccs: dict[str, Any]
    gate: dict[str, float]
    probe: dict[str, Any]
    scores: pd.DataFrame
    counts: dict[str, Any]


def measure_segment(
    clip: ViewClip, settings: JobSettings, extractor: Smile, segment: str
) -> tuple[pd.DataFrame, pd.Series, pd.Series, bool, np.ndarray]:
    """One clip on its level copy: its parts, its segment values, its whole-clip values."""
    audio = clip.samples.astype(np.float64) / PCM16_SCALE
    copy, limited = level_copy(audio, settings.level_rms, settings.peak_limit)
    parts = part_measures(
        copy,
        clip.part_starts,
        clip.part_ends,
        extractor,
        settings.min_part_samples,
        segment=segment,
    )
    return parts, segment_measures(parts), extractor.functionals(copy), limited, audio


def measure_voice(
    run: Run, inputs: Inputs, plan: Plan, control: np.ndarray, state: Progress
) -> Voice:
    """Stage 5: measure x0 and x1 at x1's level, then the ICC gate and both voice probes."""
    clock = time.monotonic()
    settings = run.settings
    extractor = Smile.load()
    segments = list(plan.wer_segments)
    part_tables: list[pd.DataFrame] = []
    values: dict[str, dict[str, pd.Series]] = {view: {} for view in VOICE_VIEWS}
    whole: dict[str, dict[str, pd.Series]] = {view: {} for view in VOICE_VIEWS}
    limited: dict[str, dict[str, bool]] = {view: {} for view in VOICE_VIEWS}
    stored_voiced: dict[str, float] = {}
    copy_voiced: dict[str, float] = {}
    for number, segment in enumerate(segments, start=1):
        state.segment = segment
        for view in VOICE_VIEWS:
            clip = inputs.clips[view][segment]
            parts, values[view][segment], whole[view][segment], scaled, audio = measure_segment(
                clip, settings, extractor, segment
            )
            limited[view][segment] = bool(scaled)
            if view == "x0":
                copy_voiced[segment] = float(parts["voiced_seconds"].sum(skipna=True))
                stored_voiced[segment] = voiced_seconds(
                    audio,
                    clip.part_starts,
                    clip.part_ends,
                    extractor,
                    settings.min_part_samples,
                    segment=segment,
                )
            parts.insert(0, "view", view)
            parts.insert(0, "segment", segment)
            part_tables.append(parts)
        if number % PROGRESS_EVERY == 0 and number < len(segments):
            elapsed = time.monotonic() - clock
            print(
                f"voice: {number:,} of {len(segments):,} segments measured; projected end"
                f" {utc_in(elapsed / number * (len(segments) - number))}"
            )
    state.segment = None
    del extractor
    parts_table = pd.concat(part_tables, ignore_index=True)
    tables = {
        view: pd.DataFrame.from_dict(values[view], orient="index")[list(FUNCTIONAL_NAMES)]
        for view in VOICE_VIEWS
    }
    wholes = {
        view: pd.DataFrame.from_dict(whole[view], orient="index")[list(FUNCTIONAL_NAMES)]
        for view in VOICE_VIEWS
    }
    owner = [inputs.session_of[segment] for segment in segments]
    gender_of, voice_of = inputs.gender.to_dict(), inputs.voices.to_dict()
    described = pd.DataFrame(
        {
            "gender": [gender_of[session] for session in owner],
            "voice": [voice_of[session] for session in owner],
        },
        index=pd.Index(segments),
    )
    iccs = voice_iccs(
        tables["x0"],
        tables["x1"],
        described,
        settings.measures,
        whole_x0=wholes["x0"],
        whole_x1=wholes["x1"],
    )
    sessions = pd.Series(owner, index=pd.Index(segments))
    features = {view: participant_means(tables[view], sessions) for view in VOICE_VIEWS}
    probe: dict[str, Any] = {}
    score_rows = []
    for view in VOICE_VIEWS:
        given = (
            features[view],
            inputs.gender,
            inputs.voices,
            inputs.label,
            inputs.folds,
            settings.probe_c,
            settings.probe_max_iter,
        )
        aurocs = probe_aurocs(*given)
        scores = probe_scores(*given)
        probe[view] = {"folds": [float(a) for a in aurocs], "mean": float(aurocs.mean())}
        score_rows.append(
            pd.DataFrame(
                {
                    "session": scores.index.to_numpy(dtype=np.int64),
                    "kind": "probe",
                    "view": view,
                    "recogniser": None,
                    "score": scores.to_numpy(dtype=np.float64),
                }
            )
        )
    probe["control"] = {"folds": [float(a) for a in control], "mean": float(control.mean())}
    probe["bar"] = probe["control"]["mean"] + settings.rules.probe_margin_auroc
    probe["imputed"] = {
        view: {
            "values": int(features[view].isna().to_numpy().sum()),
            "participants": int(features[view].isna().any(axis=1).sum()),
            "gate_measures": {m: int(features[view][m].isna().sum()) for m in settings.measures},
        }
        for view in VOICE_VIEWS
    }
    by_view = {
        view: parts_table[parts_table["view"] == view].reset_index(drop=True)
        for view in VOICE_VIEWS
    }
    weights = {view: part_weights(by_view[view]) for view in VOICE_VIEWS}
    stored, on_copy = sum(stored_voiced.values()), sum(copy_voiced.values())
    counts = {
        "segments": len(segments),
        "parts_measured": {view: int(by_view[view]["measured"].sum()) for view in VOICE_VIEWS},
        "parts_left_out": {view: int((~by_view[view]["measured"]).sum()) for view in VOICE_VIEWS},
        "parts_without_a_value": {
            view: {name: int((weights[view][name] == 0).sum()) for name in FUNCTIONAL_NAMES}
            for view in VOICE_VIEWS
        },
        "peak_guard": {view: sum(limited[view].values()) for view in VOICE_VIEWS},
        "x0_voiced_seconds": {"stored_level": stored, "copy": on_copy},
        "x0_voiced_share_at_stored_level": stored / on_copy if on_copy > 0 else math.nan,
    }
    guarded = counts["peak_guard"]
    print(
        f"voice: {len(segments):,} segments of x0 and x1 measured on copies at RMS"
        f" {settings.level_rms}; parts measured x0 {counts['parts_measured']['x0']:,}, x1"
        f" {counts['parts_measured']['x1']:,}; left out (under {settings.min_part_samples}"
        f" samples) x0 {counts['parts_left_out']['x0']}, x1 {counts['parts_left_out']['x1']};"
        f" copies scaled by the peak guard x0 {guarded['x0']}, x1 {guarded['x1']}; x0's voiced"
        f" time at its stored level is {counts['x0_voiced_share_at_stored_level']:.4f} of that"
        " on its copy"
    )
    print(
        "voice: parts without a value, gate measures: "
        + "; ".join(
            f"{name} x0 {counts['parts_without_a_value']['x0'][name]},"
            f" x1 {counts['parts_without_a_value']['x1'][name]}"
            for name in settings.measures
        )
    )
    frames = []
    for kind, source in (("parts", tables), ("whole", wholes)):
        for view, table in source.items():
            frame = table.reset_index(names="segment")
            frame.insert(1, "view", view)
            frame.insert(2, "kind", kind)  # weighted over the parts, or the whole clip at once
            frame.insert(3, "peak_limited", frame["segment"].map(limited[view]).astype(bool))
            # x0's voiced time over its parts at its stored level (NaN elsewhere)
            stored_or_none = stored_voiced if (view, kind) == ("x0", "parts") else {}
            frame.insert(4, "voiced_seconds_stored", frame["segment"].map(stored_or_none))
            frames.append(frame)
    run.stages["voice"] = time.monotonic() - clock
    return Voice(
        parts=parts_table,
        segments=pd.concat(frames, ignore_index=True),
        iccs=iccs,
        gate=gate_values(iccs),
        probe=probe,
        scores=pd.concat(score_rows, ignore_index=True),
        counts=counts,
    )


def print_voice(voice: Voice, settings: JobSettings) -> None:
    """The ICCs and the probe, as aggregates."""
    iccs = voice.iccs
    for measure in settings.measures:
        gate, raw = iccs["gate"][measure], iccs["within_gender"][measure]
        pooled = iccs["pooled"][measure]
        within = "; ".join(
            f"{gender} gate {gate[gender]['icc']:.4f} (r {gate[gender]['pearson']:.4f},"
            f" {gate[gender]['segments']} segments), without centring {raw[gender]['icc']:.4f}"
            f" (r {raw[gender]['pearson']:.4f})"
            for gender in gate
        )
        whole = iccs.get("whole_clip_pooled", {}).get(measure)
        clips = (
            f"; whole clips, pooled {whole['icc']:.4f} (r {whole['pearson']:.4f}), "
            + ", ".join(
                f"{gender} {numbers['icc']:.4f}"
                for gender, numbers in iccs["whole_clip"][measure].items()
            )
            if whole
            else ""
        )
        print(
            f"ICC {measure}: {within}; pooled {pooled['icc']:.4f} (r {pooled['pearson']:.4f})"
            f"{clips}"
        )
        print(
            "  voice offsets: "
            + ", ".join(
                f"{name} x0 {numbers['x0']:+.4f}, x1 {numbers['x1']:+.4f}"
                for name, numbers in iccs["offsets"][measure].items()
            )
            + "; segments without a value in a view: "
            + ", ".join(f"{gender} {n}" for gender, n in iccs["left_out"][measure].items())
        )
    probe = voice.probe
    print(
        f"voice probe: x0 {probe['x0']['mean']:.4f} (folds "
        + ", ".join(f"{a:.4f}" for a in probe["x0"]["folds"])
        + f"), x1 {probe['x1']['mean']:.4f} (folds "
        + ", ".join(f"{a:.4f}" for a in probe["x1"]["folds"])
        + f"), control {probe['control']['mean']:.4f}, bar {probe['bar']:.4f}; values imputed"
        f" x0 {probe['imputed']['x0']['values']}, x1 {probe['imputed']['x1']['values']}"
    )


# Stages 6 and 7: the ASR systems -------------------------------------------------------------


@dataclass
class Readings:
    """What one ASR system read: each clip once, and whether the repeats gave the same text."""

    texts: dict[str, dict[str, Reading]]
    repeats: dict[str, dict[str, bool]]
    load_seconds: float
    read_seconds: float
    identity: dict[str, Any]


def transcribe(
    name: str,
    load: Callable[[], Recogniser],
    plan: Plan,
    clips: Mapping[str, Mapping[str, ViewClip]],
    state: Progress,
) -> Readings:
    """Load one ASR system, read every clip once in segment order, then the repeats."""
    started = time.monotonic()
    recogniser = load()
    loaded = time.monotonic() - started
    total = len(plan.views) * len(plan.read_segments)
    done = 0
    windows: list[int] = []
    reading = time.monotonic()
    texts: dict[str, dict[str, Reading]] = {}
    for view in plan.views:
        texts[view] = {}
        for segment in plan.read_segments:
            state.segment = segment
            texts[view][segment] = recogniser.read(clips[view][segment].audio)
            if "windows" in texts[view][segment].diagnostics:
                windows.append(int(texts[view][segment].diagnostics["windows"]))
            done += 1
            if done % PROGRESS_EVERY == 0 and done < total:
                elapsed = time.monotonic() - reading
                seen = (
                    f"; windows per clip mean {np.mean(windows):.2f}, most {max(windows)}"
                    if windows
                    else ""
                )
                print(
                    f"{name}: {done:,} of {total:,} clips read (now {view}){seen}; projected end"
                    f" {utc_in(elapsed / done * (total - done))}"
                )
    read_seconds = time.monotonic() - reading
    repeats: dict[str, dict[str, bool]] = {}
    for view in plan.views:
        repeats[view] = {}
        for segment in plan.read_segments[:REPEATS]:
            state.segment = segment
            again = recogniser.read(clips[view][segment].audio)
            repeats[view][segment] = again.text == texts[view][segment].text
    state.segment = None
    identity = dict(getattr(recogniser, "identity", None) or {})
    identical = sum(sum(flags.values()) for flags in repeats.values())
    repeated = sum(len(flags) for flags in repeats.values())
    print(
        f"{name}: {total:,} clips read in {read_seconds / 60:.1f} min, loaded in {loaded:.0f} s;"
        f" {identical} of {repeated} clips read twice gave the same text"
    )
    return Readings(texts, repeats, loaded, read_seconds, identity)


def read_everything(
    run: Run, inputs: Inputs, plan: Plan, snapshots: Mapping[str, Path], state: Progress
) -> dict[str, Readings]:
    """Stages 6 and 7: Whisper, then wav2vec2, each released before the next loads."""
    settings = run.settings
    loaders: dict[str, Callable[[], Recogniser]] = {
        "whisper": lambda: load_whisper(snapshots["whisper"], settings.whisper),
        "wav2vec2": lambda: load_wav2vec2(snapshots["wav2vec2"]),
    }
    readings: dict[str, Readings] = {}
    for name in RECOGNISERS:
        clock = time.monotonic()
        readings[name] = transcribe(name, loaders[name], plan, inputs.clips, state)
        release()
        run.info["models"][name]["identity"] = readings[name].identity
        run.stages[name] = time.monotonic() - clock
    return readings


def transcripts_table(readings: Mapping[str, Readings], plan: Plan, inputs: Inputs) -> pd.DataFrame:
    """One row per segment, view and ASR system: the text, its words and the diagnostics."""
    rows = []
    for name in RECOGNISERS:
        for view in plan.views:
            for segment in plan.read_segments:
                reading = readings[name].texts[view][segment]
                found = words(reading.text)
                repeat = readings[name].repeats[view].get(segment)
                ratios = reading.diagnostics.get("compression_ratios")
                rows.append(
                    {
                        "segment": segment,
                        "session": inputs.session_of[segment],
                        "view": view,
                        "recogniser": name,
                        "text": reading.text,
                        "words": " ".join(found),
                        "output_words": len(found),
                        "seconds": float(reading.seconds),
                        "samples": len(inputs.clips[view][segment].samples),
                        **{key: reading.diagnostics.get(key) for key in DIAGNOSTIC_KEYS},
                        "compression_ratios": None
                        if ratios is None
                        else [float(r) for r in ratios],
                        "repeated": repeat is not None,
                        "repeat_identical": repeat,
                    }
                )
    return pd.DataFrame(rows)


# Stage 8: scoring --------------------------------------------------------------------------------


def word_rows(
    plan: Plan, inputs: Inputs, readings: Mapping[str, Readings]
) -> tuple[dict[str, dict[str, WerCounts]], dict[str, dict[str, dict[str, Any]]]]:
    """The WER of every view and ASR system, with words heard, their chance level and odd outputs."""
    segments = list(plan.wer_segments)
    references = [inputs.references[segment] for segment in segments]
    partners = chance_partners(
        pd.DataFrame({"segment": segments, "session": [inputs.session_of[s] for s in segments]})
    )
    chance_references = [references[int(partner)] for partner in partners]
    counts: dict[str, dict[str, WerCounts]] = {}
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    for view in plan.views:
        counts[view], rows[view] = {}, {}
        for name in RECOGNISERS:
            read = [readings[name].texts[view][segment] for segment in segments]
            texts = [reading.text for reading in read]
            outputs = [words(text) for text in texts]
            found = wer_counts(references, outputs, segments)
            chance = wer_counts(chance_references, outputs, segments)
            table = found.per_segment
            row: dict[str, Any] = {
                "clips": len(segments),
                **found.totals(),
                "chance_heard": chance.heard,
                "clips_with_wer_above_1": int((table["wer"] > 1).sum()),
                "very_repetitive_outputs": sum(
                    repetition_share(output) > VERY_REPETITIVE for output in outputs
                ),
                "median_output_to_reference_words": float(
                    (table["output_words"] / table["reference_words"]).median()
                ),
                "empty_outputs": int((table["output_words"] == 0).sum()),
                "outputs_text": text_counts(texts),
            }
            diagnostics = [reading.diagnostics for reading in read]
            if name == "whisper":
                windows = [int(d.get("windows", 0)) for d in diagnostics]
                row["windows"] = {
                    "total": sum(windows),
                    "clips_with_2_or_more": sum(w >= 2 for w in windows),
                    "most": max(windows, default=0),
                    "default_dropped_windows": sum(
                        int(d.get("default_dropped_windows", 0)) for d in diagnostics
                    ),
                    "default_dropped_words": sum(
                        int(d.get("default_dropped_words", 0)) for d in diagnostics
                    ),
                }
            else:
                row["special_symbols"] = sum(int(d.get("special_symbols", 0)) for d in diagnostics)
            counts[view][name], rows[view][name] = found, row
    return counts, rows


def print_words(
    rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    plan: Plan,
    timings: Mapping[str, Mapping[str, Any]],
) -> None:
    """The word table, one line per view and ASR system, then each one's reading and outputs."""
    print(
        "words: view, ASR system, WER, substituted, deleted, inserted, hits, reference words,"
        " words heard, chance level, clips with WER above 1, very repetitive outputs, median"
        " output / reference words, empty outputs"
    )
    for view in plan.views:
        for name in RECOGNISERS:
            row = rows[view][name]
            print(
                f"  {view:<8} {name:<8} {row['wer']:.4f} {row['substitutions']:>6}"
                f" {row['deletions']:>6} {row['insertions']:>6} {row['hits']:>7}"
                f" {row['reference_words']:>7} {row['heard']:.4f} {row['chance_heard']:.4f}"
                f" {row['clips_with_wer_above_1']:>5} {row['very_repetitive_outputs']:>5}"
                f" {row['median_output_to_reference_words']:.4f} {row['empty_outputs']:>5}"
            )
    for view in plan.views:
        for name in RECOGNISERS:
            row = rows[view][name]
            text = row["outputs_text"]
            if "windows" in row:
                windows = row["windows"]
                extra = (
                    f"; windows {windows['total']:,}, clips with 2 or more"
                    f" {windows['clips_with_2_or_more']}, most {windows['most']}; windows (words)"
                    " that the library's default silence rule would have dropped"
                    f" {windows['default_dropped_windows']} ({windows['default_dropped_words']})"
                )
            else:
                extra = f"; special symbols dropped {row['special_symbols']}"
            read = timings[name]["views"][view]
            print(
                f"outputs {view} {name}: {read['clips']:,} clips ({read['audio_seconds']:,.0f} s"
                f" of audio) read in {read['read_seconds']:,.0f} s; hesitations dropped"
                f" {text['hesitations_dropped']},"
                f" spellings merged {text['aliases_merged']}, one-letter words"
                f" {text['one_letter_words']}, letter runs joined {text['letter_runs_joined']}"
                + extra
            )


def leakage_rows(
    run: Run, plan: Plan, inputs: Inputs, readings: Mapping[str, Readings], prepared: Prepared
) -> tuple[dict[str, dict[str, dict[str, Any]]], pd.DataFrame]:
    """The leakage AUROC of every view and ASR system, with x0's and rev's intervals."""
    segments = list(plan.leakage_segments)
    frame = pd.DataFrame({"session": [inputs.session_of[segment] for segment in segments]})
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    tables = []
    for view in plan.views:
        rows[view] = {}
        for name in RECOGNISERS:
            texts = [readings[name].texts[view][segment].text for segment in segments]
            scores = person_scores(prepared.models, inputs.folds, frame, texts)
            aurocs = fold_aurocs(scores, inputs.label, inputs.folds)
            row: dict[str, Any] = {
                "segments": len(segments),
                "folds": [float(a) for a in aurocs],
                "mean": float(aurocs.mean()),
            }
            if prepared.sets is not None and view in INTERVAL_VIEWS:
                row["interval"] = list(
                    bootstrap_interval(
                        prepared.sets,
                        scores,
                        inputs.label,
                        inputs.folds,
                        run.settings.rules.x0_leakage_interval,
                    )
                )
            rows[view][name] = row
            tables.append(
                pd.DataFrame(
                    {
                        "session": scores.index.to_numpy(dtype=np.int64),
                        "kind": "leakage",
                        "view": view,
                        "recogniser": name,
                        "score": scores.to_numpy(dtype=np.float64),
                    }
                )
            )
    level = run.settings.rules.x0_leakage_interval
    for view in plan.views:
        for name in RECOGNISERS:
            row = rows[view][name]
            interval = (
                f"; {level:.0%} interval [{row['interval'][0]:.4f}, {row['interval'][1]:.4f}]"
                if "interval" in row
                else ""
            )
            print(
                f"leakage {view} {name}: {row['mean']:.4f} (folds "
                + ", ".join(f"{a:.4f}" for a in row["folds"])
                + f"; {row['segments']:,} segments){interval}"
            )
    return rows, pd.concat(tables, ignore_index=True)


def short_part_split(counts: Mapping[str, WerCounts], inputs: Inputs) -> dict[str, Any]:
    """x1's WER on the segments with a part of 1-2 words, against the rest."""
    short = {
        str(segment)
        for segment, group in inputs.parts.groupby("segment")
        if (group["n_words"] <= SHORT_PART_WORDS).any()
    }
    result: dict[str, Any] = {}
    for name, found in counts.items():
        table = found.per_segment
        flags = table["segment"].astype(str).isin(short).to_numpy()
        result[name] = {}
        for group, rows in (("short_parts", table[flags]), ("rest", table[~flags])):
            errors = int(rows[["substitutions", "deletions", "insertions"]].to_numpy().sum())
            reference = int(rows["reference_words"].sum())
            result[name][group] = {
                "segments": len(rows),
                "reference_words": reference,
                "wer": errors / reference if reference else math.nan,
            }
        short_part, rest = result[name]["short_parts"], result[name]["rest"]
        print(
            f"x1 {name}: WER {found.wer:.4f}; segments with a part of 1-2 words"
            f" {short_part['wer']:.4f} ({short_part['segments']} segments), the rest"
            f" {rest['wer']:.4f} ({rest['segments']})"
        )
    return result


def class_rows(
    counts: Mapping[str, Mapping[str, WerCounts]], classes: pd.DataFrame
) -> dict[str, Any]:
    """The errors of x1 and of x0 (the baseline) by text-free word class; counts only."""
    result: dict[str, Any] = {}
    for view in ("x1", "x0"):
        result[view] = {}
        for name in RECOGNISERS:
            table = error_classes(counts[view][name], classes)
            result[view][name] = {
                str(group): {column: int(table.at[group, column]) for column in table.columns}
                for group in table.index
            }
    for name in RECOGNISERS:
        print(
            f"errors by word class, {name}: class, then reference words, substituted, deleted"
            " and inserted beside them, in x1 | in x0"
        )
        for group, x1 in result["x1"][name].items():
            x0 = result["x0"][name][group]
            print(
                f"  {group:<15}"
                + "".join(f" {x1[key]:>7}" for key in x1)
                + " |"
                + "".join(f" {x0[key]:>7}" for key in x0)
            )
    return result


def rerun_estimate(
    run: Run, plan: Plan, inputs: Inputs, readings: Mapping[str, Readings]
) -> dict[str, Any]:
    """The time to ask for a leakage rerun on all segments of x0 and rev, from this run's rates."""
    everything = len(inputs.segments)
    whisper = everything * sum(
        float(np.mean([r.seconds for r in readings["whisper"].texts[view].values()]))
        for view in RERUN_VIEWS
    )
    sampled_audio = sum(
        len(inputs.clips[view][segment].samples)
        for view in RERUN_VIEWS
        for segment in plan.read_segments
    )
    all_audio = sum(int(inputs.manifests[view]["samples"].sum()) for view in RERUN_VIEWS)
    wav2vec2 = (all_audio / sampled_audio) * sum(
        float(sum(r.seconds for r in readings["wav2vec2"].texts[view].values()))
        for view in RERUN_VIEWS
    )
    seconds = whisper + wav2vec2 + RERUN_OVERHEAD_SECONDS
    limit = seconds * run.settings.rerun_time_factor
    minutes = math.ceil(limit / 60)
    estimate = {
        "segments": everything,
        "whisper_seconds": whisper,
        "wav2vec2_seconds": wav2vec2,
        "overhead_seconds": RERUN_OVERHEAD_SECONDS,
        "estimate_hours": seconds / 3600,
        "limit_hours": limit / 3600,
        "slurm_time": f"{minutes // 60:02d}:{minutes % 60:02d}:00",
        "above_gpu_limit": limit / 3600 > GPU_LIMIT_HOURS,
    }
    print(
        f"leakage rerun, if called: about {estimate['estimate_hours']:.1f} h for"
        f" {everything:,} segments of x0 and rev; ask for --time={estimate['slurm_time']}"
        f" (x{run.settings.rerun_time_factor})"
        + (
            f"; above the gpu partition's {GPU_LIMIT_HOURS:.0f} h: stop and decide"
            if estimate["above_gpu_limit"]
            else ""
        )
    )
    return estimate


def mean_leakage(
    leakage: Mapping[str, Mapping[str, Mapping[str, Any]]], view: str
) -> dict[str, float]:
    """A view's leakage AUROC by ASR system."""
    return {name: float(leakage[view][name]["mean"]) for name in RECOGNISERS}


def intervals_of(
    leakage: Mapping[str, Mapping[str, Mapping[str, Any]]], view: str
) -> dict[str, tuple[float, float]]:
    """A view's leakage interval by ASR system."""
    return {
        name: (float(leakage[view][name]["interval"][0]), float(leakage[view][name]["interval"][1]))
        for name in RECOGNISERS
    }


def wer_of(rows: Mapping[str, Mapping[str, Mapping[str, Any]]], view: str) -> dict[str, float]:
    """A view's WER by ASR system."""
    return {name: float(rows[view][name]["wer"]) for name in RECOGNISERS}


def judge_sample(
    run: Run,
    plan: Plan,
    inputs: Inputs,
    readings: Mapping[str, Readings],
    counts: Mapping[str, Mapping[str, WerCounts]],
    rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    leakage: Mapping[str, Mapping[str, Mapping[str, Any]]],
    voice: Voice,
) -> dict[str, Any]:
    """The default mode: x1 by part size and word class, then the four verdicts."""
    results = run.info["results"]
    results["wer"] = {view: wer_of(rows, view) for view in RULE_VIEWS}
    results["x1_short_parts"] = short_part_split(counts["x1"], inputs)
    if inputs.classes is None:
        raise StepError("the word classes were not built")
    results["error_classes"] = class_rows(counts, inputs.classes)
    results["rerun_estimate"] = rerun_estimate(run, plan, inputs, readings)
    numbers = Results(
        wer=results["wer"],
        leakage={view: mean_leakage(leakage, view) for view in ("rev", "x0")},
        x0_intervals=intervals_of(leakage, "x0"),
        iccs=voice.gate,
        probe_x0=voice.probe["x0"]["mean"],
        probe_x1=voice.probe["x1"]["mean"],
        probe_control=voice.probe["control"]["mean"],
    )
    return verdicts(numbers, run.settings.rules)


def judge_views(
    run: Run, plan: Plan, rows: Mapping[str, Mapping[str, Mapping[str, Any]]], leakage: Any
) -> dict[str, Any]:
    """The --views mode: check F4 on each named view."""
    run.info["results"]["wer"] = {view: wer_of(rows, view) for view in plan.views}
    return {
        view: f4_verdict(
            wer_of(rows, view),
            mean_leakage(leakage, view),
            run.settings.rules,
            rerun_pending=False,
            view=view,
        )
        for view in plan.views
    }


def judge_rerun(
    run: Run,
    first: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    leakage: Any,
) -> dict[str, Any]:
    """The rerun: the first run's WER, this run's leakage and intervals; the final F4 and x0 control."""
    results = run.info["results"]
    results["wer"] = first["wer"]
    results["wer_all_segments"] = {view: wer_of(rows, view) for view in RERUN_VIEWS}
    rules = run.settings.rules
    control = x0_control_verdict(
        first["wer"]["x0"],
        mean_leakage(leakage, "x0"),
        intervals_of(leakage, "x0"),
        rules,
        after_rerun=True,
    )
    f4 = f4_verdict(first["wer"]["rev"], mean_leakage(leakage, "rev"), rules, rerun_pending=False)
    return {"f4": f4, "x0_control": control}


def repeat_counts(readings: Mapping[str, Readings]) -> dict[str, dict[str, int]]:
    """Per ASR system: the clips read twice and how many gave the same text."""
    return {
        name: {
            "clips": sum(len(flags) for flags in readings[name].repeats.values()),
            "identical": sum(sum(flags.values()) for flags in readings[name].repeats.values()),
        }
        for name in RECOGNISERS
    }


def reading_timings(
    readings: Mapping[str, Readings], plan: Plan, inputs: Inputs
) -> dict[str, dict[str, Any]]:
    """Clips, audio and reading time per ASR system and view."""
    result: dict[str, dict[str, Any]] = {}
    for name in RECOGNISERS:
        result[name] = {
            "load_seconds": round(readings[name].load_seconds, 1),
            "read_seconds": round(readings[name].read_seconds, 1),
            "views": {},
        }
        for view in plan.views:
            seconds = float(sum(r.seconds for r in readings[name].texts[view].values()))
            audio = sum(len(inputs.clips[view][s].samples) for s in plan.read_segments)
            clips = len(plan.read_segments)
            result[name]["views"][view] = {
                "clips": clips,
                "audio_seconds": round(audio / SAMPLING_RATE, 1),
                "read_seconds": round(seconds, 1),
                "clips_per_second": clips / seconds if seconds > 0 else math.nan,
            }
    return result


# The job ----------------------------------------------------------------------------------------


def write_table(table: pd.DataFrame, path: Path) -> str:
    """Write a table as Parquet and return its SHA-256."""
    table.to_parquet(path, index=False)
    return sha256_file(path)


def check(state: Progress) -> None:
    """The job: the nine stages of the module docstring."""
    run = start(state)
    mode, settings, info = run.mode, run.settings, run.info
    info["inputs"] = input_pins(run, mode.views if mode.name == VIEWS else settings.views)
    leakage_rerun = mode.name == VIEWS and leakage_rerun_pin(run)
    first = first_run(run, mode.first_run) if mode.name == RERUN else None
    if first is not None:
        info["first_run"] = {key: value for key, value in first.items() if key != "wer"}
    run.save()

    snapshots = public_guard(run)  # 2
    inputs, plan = gather(run, state, leakage_rerun)  # 3
    prepared = prepare(run, inputs, plan)  # 4
    results = info["results"]
    voice = None
    if plan.voice and prepared.control is not None:  # 5
        voice = measure_voice(run, inputs, plan, prepared.control, state)
        print_voice(voice, settings)
        results.update(voice=voice.counts, iccs=voice.iccs, probe=voice.probe)
    run.save()

    readings = read_everything(run, inputs, plan, snapshots, state)  # 6 and 7
    outputs = {
        "transcripts.parquet": write_table(
            transcripts_table(readings, plan, inputs), run.run_dir / "transcripts.parquet"
        )
    }
    info["outputs"] = outputs
    run.save()

    clock = time.monotonic()  # 8
    timings = reading_timings(readings, plan, inputs)
    counts, rows = word_rows(plan, inputs, readings)
    results["words"] = rows
    print_words(rows, plan, timings)
    leakage, scores = leakage_rows(run, plan, inputs, readings, prepared)
    results["leakage"] = leakage
    results["repeats"] = repeat_counts(readings)
    if mode.name == SAMPLE and voice is not None:
        judged = judge_sample(run, plan, inputs, readings, counts, rows, leakage, voice)
    elif mode.name == VIEWS:
        judged = judge_views(run, plan, rows, leakage)
    elif first is not None:
        judged = judge_rerun(run, first, rows, leakage)
    else:
        raise StepError(f"mode {mode.name} has nothing to judge")
    info["verdicts"] = judged
    run.stages["scoring"] = time.monotonic() - clock

    scores = pd.concat([scores, *([voice.scores] if voice is not None else [])], ignore_index=True)
    scores.insert(1, "fold", scores["session"].map(inputs.folds).astype(np.int64))  # 9
    outputs["scores.parquet"] = write_table(scores, run.run_dir / "scores.parquet")
    if voice is not None:
        outputs["voice_parts.parquet"] = write_table(
            voice.parts, run.run_dir / "voice_parts.parquet"
        )
        outputs["voice_segments.parquet"] = write_table(
            voice.segments, run.run_dir / "voice_segments.parquet"
        )
    minutes = (time.monotonic() - run.started) / 60
    results["timings"] = {
        "stages_seconds": {name: round(seconds, 1) for name, seconds in run.stages.items()},
        "reading": timings,
    }
    info.update(status="done", finished_utc=utc_now(), minutes=round(minutes, 2))
    run.save()

    repeats = results["repeats"]
    print(
        "repeats: "
        + "; ".join(
            f"{name} {repeats[name]['identical']} of {repeats[name]['clips']} clips read twice"
            " gave the same text"
            for name in RECOGNISERS
        )
    )
    groups = [{"f4": verdict} for verdict in judged.values()] if mode.name == VIEWS else [judged]
    for group in groups:
        for line in verdict_lines(group):
            print(line)
    if mode.name == SAMPLE and judged["x0_control"]["verdict"] == RERUN_NEEDED:
        print(
            "the x0 control calls for the leakage rerun: scripts/slurm/submit.sh"
            f" --time={results['rerun_estimate']['slurm_time']} check_views --rerun {run.relative}"
        )
    print(f"run: {run.relative}")
    print(f"transcripts.parquet sha256: {outputs['transcripts.parquet']}")
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
