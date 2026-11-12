import hashlib
from pathlib import Path

import pytest

from listening_beyond_words.runinfo import check_sha256, sha256_file, utc_stamp


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
