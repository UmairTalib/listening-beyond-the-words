from pathlib import Path

import pytest

from listening_beyond_words.models import ModelPin, load_pins, local_snapshot

PINS = Path(__file__).resolve().parents[1] / "configs" / "pilot" / "models.yaml"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def write_pins(folder: Path, text: str) -> Path:
    path = folder / "models.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_pins_reads_repo_revision_and_ignore_patterns(tmp_path: Path) -> None:
    text = f"voxtral:\n  repo_id: org/model\n  revision: '{COMMIT}'\n  ignore: [big.bin]\n"
    pins = load_pins(write_pins(tmp_path, text))
    assert pins == {"voxtral": ModelPin("voxtral", "org/model", COMMIT, ("big.bin",))}


@pytest.mark.parametrize("revision", ["main", COMMIT[:12], COMMIT.upper()])
def test_load_pins_needs_a_full_commit_hash(tmp_path: Path, revision: str) -> None:
    path = write_pins(tmp_path, f"m:\n  repo_id: org/model\n  revision: '{revision}'\n")
    with pytest.raises(ValueError, match="40-character commit hash"):
        load_pins(path)


def test_load_pins_needs_repo_id_and_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs a repo_id and a revision"):
        load_pins(write_pins(tmp_path, "m:\n  repo_id: org/model\n"))


def test_pilot_pins_voxtral_mini_at_a_full_commit() -> None:
    pin = load_pins(PINS)["voxtral"]
    assert pin.repo_id == "mistralai/Voxtral-Mini-3B-2507"
    assert len(pin.revision) == 40


def test_local_snapshot_finds_a_fetched_snapshot(tmp_path: Path) -> None:
    folder = tmp_path / "hub" / "models--org--model" / "snapshots" / COMMIT
    folder.mkdir(parents=True)
    assert local_snapshot(ModelPin("m", "org/model", COMMIT), tmp_path) == folder


def test_local_snapshot_says_how_to_fetch_a_missing_snapshot(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"scripts/fetch_models\.py"):
        local_snapshot(ModelPin("m", "org/model", COMMIT), tmp_path)
