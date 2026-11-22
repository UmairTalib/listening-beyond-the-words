"""What every step records about itself: time, commit, seeds, file hashes and run.json."""

import hashlib
import json
import numbers
import subprocess
from datetime import UTC, datetime
from pathlib import Path


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


def write_json(path: Path, data: dict[str, object]) -> None:
    """Write a dictionary as indented JSON.

    Args:
        path: The output file.
        data: What to write.
    """
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
