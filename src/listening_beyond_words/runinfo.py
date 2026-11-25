"""What every step records about itself: time, commit, seeds, file hashes, run.json and errors.

It also asks PyTorch for repeatable kernels (``deterministic_torch``), seeds every
random number generator of a process (``seed_everything``) and says what GPU a job runs
on (``gpu_facts``, ``peak_memory_gb``); torch is imported inside those functions only.
"""

import hashlib
import json
import numbers
import platform
import random
import signal
import subprocess
import sys
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import FrameType
from typing import Any, Final

import numpy as np

LOCK_FILE = Path("uv.lock")
STOPPED: Final = "stopped by Slurm (SIGTERM)"


def utc_now() -> str:
    """Return the current UTC time, to the second, in ISO 8601 form."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def utc_stamp() -> str:
    """Return the current UTC time for folder names, for example ``20261112T101500Z``."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


RELEASE_TAGS: Final = "v0.1.*"  # the pilot's releases: v0.1.0 freezes its settings


def git_state() -> dict[str, object]:
    """Return the checked-out commit, whether the checkout has uncommitted changes, and its tags.

    Engineering conventions, section 8: a run records the commit, the dirty flag and the
    output of ``git describe --tags --always --dirty`` with the tags at the commit. The
    pilot's release tags are listed too, so that the training job can tell whether the
    settings are frozen yet, and tests patch this one function.

    Returns:
        ``commit``; ``dirty``; ``describe``, the nearest tag with the commits since and the
        commit, or the short commit alone where no tag exists, with ``-dirty`` for
        uncommitted changes; ``tags``, the tags at the commit (``git tag --points-at
        HEAD``), sorted; and ``release_tags``, every ``v0.1.*`` tag of the clone (``git tag
        --list``), sorted, which a checkout of an older commit still sees.
    """

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout

    def names(text: str) -> list[str]:
        return sorted(line.strip() for line in text.splitlines() if line.strip())

    return {
        "commit": git("rev-parse", "HEAD").strip(),
        "dirty": bool(git("status", "--porcelain")),
        "describe": git("describe", "--tags", "--always", "--dirty").strip(),
        "tags": names(git("tag", "--points-at", "HEAD")),
        "release_tags": names(git("tag", "--list", RELEASE_TAGS)),
    }


def item_seed(seed: int, *keys: int | str) -> int:
    """Return the seed of one item, derived from the run seed and the item's keys with SHA-256.

    The same seed and keys give the same value in every process and on every machine,
    unlike Python's ``hash()``, which changes between processes. For example, a part of
    a re-voiced clip gets ``item_seed(106, "300_012", 2)``. A key's type counts: ``"2"``
    and ``2`` give different seeds.

    Args:
        seed: The run seed from the config.
        *keys: What identifies the item, as whole numbers or text, in a fixed order.

    Returns:
        A whole number from 0 to 2**63 - 1, usable as a NumPy or PyTorch seed.

    Raises:
        TypeError: If the seed or a key is neither a whole number nor text. True and
            False are refused too.
    """
    values: list[int | str] = []
    for value in (seed, *keys):
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, numbers.Integral) and not isinstance(value, bool):
            values.append(int(value))
        else:
            raise TypeError(f"seeds and keys must be whole numbers or text, got {value!r}")
    text = json.dumps(values, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def deterministic_torch() -> None:
    """Ask PyTorch for repeatable results: deterministic kernels, no cuDNN autotuning.

    An operation without a deterministic kernel warns instead of stopping the step
    (``warn_only``). The settings hold for the whole process.
    """
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_everything(seed: int) -> None:
    """Seed Python's, NumPy's global and PyTorch's random number generators from one seed.

    Engineering conventions, section 8. The package's own NumPy code still takes a seeded
    ``numpy.random.Generator`` and never draws from the global state; seeding it fixes only
    what another package might draw there. NumPy's global state takes 32-bit words, so the
    seed goes in as its low and its high 32 bits. ``torch.manual_seed`` seeds the CPU and
    every GPU.

    Args:
        seed: A whole number from 0 to 2**64 - 1, such as an ``item_seed``.

    Raises:
        TypeError: If the seed is not a whole number; True and False are refused too.
        ValueError: If the seed is negative or 2**64 or more.
    """
    import torch

    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError(f"a seed is a whole number, not {seed!r}")
    value = int(seed)
    if not 0 <= value < 2**64:
        raise ValueError(f"a seed lies from 0 to 2**64 - 1, not {value}")
    random.seed(value)
    # The global state, which the conventions ban for the package's own draws (ruff NPY002),
    # is seeded on purpose here.
    np.random.seed([value & 0xFFFF_FFFF, value >> 32])  # noqa: NPY002
    torch.manual_seed(value)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file as 64 hex characters.

    Args:
        path: The file.

    Returns:
        The hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_sha256(path: Path, expected: str) -> None:
    """Stop if a file does not have the pinned SHA-256.

    Args:
        path: The file.
        expected: The pinned hex digest.

    Raises:
        ValueError: If the file's digest differs.
    """
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{path.name}: SHA-256 {actual[:12]}... does not match the pinned {expected[:12]}..."
        )


def environment(packages: Sequence[str]) -> dict[str, object]:
    """Return the Python version, package versions, CPU and lock file of this process.

    Args:
        packages: Names of the installed packages whose versions to record.

    Returns:
        ``python``, ``packages`` (None for a package that is not installed), ``cpu`` (the
        CPU model where Linux names it) and ``uv_lock_sha256`` (of ``uv.lock`` in the
        current folder, the clone's root in every step; None if it is missing).
    """
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    cpu = platform.processor() or None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "python": platform.python_version(),
        "packages": versions,
        "cpu": cpu,
        "uv_lock_sha256": sha256_file(LOCK_FILE) if LOCK_FILE.is_file() else None,
    }


def gpu_facts() -> dict[str, Any]:
    """Return the GPU of this process, its driver, CUDA and cuDNN, for a job's ``run.json``.

    Returns:
        ``gpu`` (its name, or None without a GPU), ``cuda`` and, with a GPU,
        ``gpu_capability``, ``gpu_memory_gb``, ``cudnn`` and ``driver``
        (``driver_version``); without one ``cudnn`` and ``driver`` are None.
    """
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
    """Return the NVIDIA driver's version, as nvidia-smi gives it; None if it cannot say."""
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
    """Return the most GPU memory that PyTorch held so far, in GB; None without a GPU."""
    import torch

    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1e9, 2)


class StepError(Exception):
    """An error raised by a step's own rules, such as a refused input or a stop rule.

    Its message is written by the step and names rules, counts and IDs, never participant
    text, so it may be printed and stored in ``run.json``. Every other error is reported
    by its type only, because its message could hold a word.

    Args:
        message: What went wrong, without participant text.
        segment: The segment concerned, if any.
        part: The part of the segment concerned (its position from 0), if any.
    """

    def __init__(self, message: str, *, segment: str | None = None, part: int | None = None):
        super().__init__(message)
        self.segment = segment
        self.part = part


@dataclass
class Progress:
    """What a step knows so far, for the record of a failure.

    Attributes:
        run_dir: The step's own output folder, once it exists; it gets ``errors.log``.
        info: The step's ``run.json`` content, once it is written there.
        segment: The segment it works on, if any.
        part: The part of that segment it works on, if any.
    """

    run_dir: Path | None = None
    info: dict[str, Any] | None = None
    segment: str | None = None
    part: int | None = None


def stop_on_sigterm() -> None:
    """End a job through its error path when Slurm stops it, so that its run.json says failed."""

    def handler(_signum: int, _frame: FrameType | None) -> None:
        raise SystemExit(STOPPED)

    signal.signal(signal.SIGTERM, handler)


@dataclass(frozen=True, slots=True)
class Failure:
    """How a failed step reports itself, without any participant text.

    Attributes:
        line: For stdout: the error's type, the segment and part, the step's own message
            if it has one, and where ``errors.log`` is if it was written.
        frames: For stderr: the error's traceback frames and type names
            (``error_record``) when no ``errors.log`` holds them and the error is not the
            step's own; otherwise None. Never a message.
    """

    line: str
    frames: str | None


def record_failure(error: BaseException, progress: Progress) -> Failure:
    """Record a failed step without any participant text and say what to print.

    ``errors.log`` in the step's folder gets ``error_record(error)``, and ``run.json``
    status ``failed`` with the error's type and the segment and part concerned. Only a
    ``StepError`` or Slurm's stop keeps its message, which the step wrote; any other
    message could hold a word; Slurm's stop is recorded by the fixed text ``STOPPED``. The
    two files are written in separate guards, so one failed write never stops the other,
    and nothing raised while recording is passed on. The caller reports the failure
    outside its ``except`` clause, so nothing it raises can chain to the error.

    Args:
        error: The error that stopped the step.
        progress: What the step knew when it stopped.

    Returns:
        The line for stdout and, when no ``errors.log`` holds them, the frames for stderr.
    """
    stopped = isinstance(error, SystemExit) and type(error.code) is str and error.code == STOPPED
    message = STOPPED if stopped else str(error) if isinstance(error, StepError) else None
    # Only the step's own errors may say where they happened; any other error's fields could
    # hold anything.
    placed = isinstance(error, StepError) and error.segment is not None
    segment = error.segment if isinstance(error, StepError) and placed else progress.segment
    part = error.part if isinstance(error, StepError) and placed else progress.part
    log = None
    record = None
    try:
        record = error_record(error)
        if progress.run_dir is not None:
            path = progress.run_dir / "errors.log"
            path.write_text(record, encoding="utf-8")
            log = path  # named in the line only once it holds the frames
    except BaseException:  # the failure is being reported already; the rest must follow
        pass
    try:
        if progress.run_dir is not None and progress.info is not None:
            failure: dict[str, Any] = {
                "type": type(error).__name__,
                "segment": segment,
                "part": part,
            }
            if message is not None:
                failure["message"] = message
            progress.info.update(status="failed", finished_utc=utc_now(), error=failure)
            write_json(progress.run_dir / "run.json", progress.info)
    except BaseException:  # as above
        pass
    where = f" in segment {segment}" if segment else ""
    where += f", part {part}" if segment and part is not None else ""
    detail = f": {message}" if message else ""
    named = f"; traceback frames in {log}" if log is not None else ""
    shown = record if log is None and not isinstance(error, StepError) else None
    return Failure(line=f"FAILED: {type(error).__name__}{where}{detail}{named}", frames=shown)


def error_record(error: BaseException) -> str:
    """Describe an error for ``errors.log``: traceback frames and type names, never a message.

    The error and every error in its chain (its cause or context) are listed, from the
    last raised to the first, each with its frames as file, line and function. No
    message and no source line is written, so the record holds no participant text even
    when a message would.

    Args:
        error: The error that stopped a step.

    Returns:
        The record, one line per error type and one indented line per frame.
    """
    lines: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        lines.append(f"{type(current).__module__}.{type(current).__qualname__}")
        for frame in traceback.extract_tb(current.__traceback__):
            lines.append(f"  {frame.filename}:{frame.lineno} in {frame.name}")
        # "is not None": an exception's own truth value could raise.
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return "\n".join(lines) + "\n"


def quiet_unraisable(unraisable: Any) -> None:
    """Report an error that Python could not raise, such as one in a finaliser, by its type only.

    Python's own hook prints the error's message and the object concerned, either of
    which could hold a word. The steps that read participant text install this one as
    ``sys.unraisablehook``.

    Args:
        unraisable: What Python passes to ``sys.unraisablehook``.
    """
    sys.stderr.write(f"Exception ignored: {type(unraisable.exc_value).__name__}\n")


def _plain(value: Any) -> Any:
    """Turn the NumPy numbers and arrays that JSON cannot write into Python values."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot write {type(value).__name__} as JSON")


def write_json(path: Path, data: dict[str, object]) -> None:
    """Write a dictionary as indented JSON, all or nothing.

    NumPy numbers and arrays are written as plain numbers and lists, so that a count such
    as a NumPy sum never stops a step at its end. The JSON goes to a temporary file next to
    ``path`` that is then renamed onto it, so a step that dies while writing leaves the
    earlier file or none, never half of one: the merge of x1 counts a run folder with a
    ``run.json`` as merged.

    Args:
        path: The output file.
        data: What to write.
    """
    text = json.dumps(data, indent=2, default=_plain) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
