"""What every step records about itself: time, commit, file hashes and run.json."""

import hashlib
import json
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
