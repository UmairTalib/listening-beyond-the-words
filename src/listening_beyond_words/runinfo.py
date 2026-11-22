"""What every step records about itself: time, commit, seeds, file hashes and run.json."""

import hashlib
import json
import numbers
import platform
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

LOCK_FILE = Path("uv.lock")


def utc_now() -> str:
    """Return the current UTC time, to the second, in ISO 8601 form."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def utc_stamp() -> str:
    """Return the current UTC time for folder names, for example ``20261112T101500Z``."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def git_state() -> dict[str, object]:
    """Return the checked-out commit and whether the checkout has uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout

    return {"commit": git("rev-parse", "HEAD").strip(), "dirty": bool(git("status", "--porcelain"))}


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
    """Write a dictionary as indented JSON.

    NumPy numbers and arrays are written as plain numbers and lists, so that a count such
    as a NumPy sum never stops a step at its end.

    Args:
        path: The output file.
        data: What to write.
    """
    path.write_text(json.dumps(data, indent=2, default=_plain) + "\n", encoding="utf-8")
