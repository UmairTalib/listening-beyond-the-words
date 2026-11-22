import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from listening_beyond_words.runinfo import check_sha256, item_seed, sha256_file, utc_stamp

PINNED_SEED = 7691467536369641371  # item_seed(106, "9001_012", 2), computed 2026-11-22


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"synthetic bytes" * 1000)
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_check_sha256_stops_on_a_changed_file(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"one")
    pinned = sha256_file(path)
    path.write_bytes(b"two")
    with pytest.raises(ValueError, match="does not match"):
        check_sha256(path, pinned)


def test_utc_stamp_suits_folder_names() -> None:
    stamp = utc_stamp()
    assert len(stamp) == 16
    assert stamp.endswith("Z")


def test_item_seed_is_pinned() -> None:
    # SHA-256 of '[106,"9001_012",2]', first 8 bytes, shifted right by one bit.
    digest = hashlib.sha256(b'[106,"9001_012",2]').digest()
    assert item_seed(106, "9001_012", 2) == int.from_bytes(digest[:8], "big") >> 1
    assert item_seed(106, "9001_012", 2) == PINNED_SEED


def test_item_seed_is_the_same_in_another_process() -> None:
    code = (
        "from listening_beyond_words.runinfo import item_seed; print(item_seed(106, '9001_012', 2))"
    )
    for hash_seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        assert int(result.stdout) == item_seed(106, "9001_012", 2)


def test_item_seed_differs_by_seed_key_order_and_key_type() -> None:
    seeds = {
        item_seed(106, "9001_012", 2),
        item_seed(107, "9001_012", 2),
        item_seed(106, "9001_012", 3),
        item_seed(106, 2, "9001_012"),
        item_seed(106, "9001_012", "2"),
        item_seed(106, "9001_012"),
    }
    assert len(seeds) == 6
    assert all(0 <= seed < 2**63 for seed in seeds)


def test_item_seed_takes_numpy_integers_like_python_integers() -> None:
    assert item_seed(np.int64(106), "9001_012", np.int32(2)) == item_seed(106, "9001_012", 2)


@pytest.mark.parametrize("key", [2.0, True, None, b"9001"])
def test_item_seed_refuses_other_keys(key: object) -> None:
    with pytest.raises(TypeError, match="whole numbers or text"):
        item_seed(106, key)  # type: ignore[arg-type]
