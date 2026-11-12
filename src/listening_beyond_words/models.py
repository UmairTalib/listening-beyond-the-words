"""Pinned model snapshots: which Hugging Face models the project uses, at which commit.

``configs/pilot/models.yaml`` lists the pins. ``scripts/fetch_models.py`` downloads
them on an OMNI login node into the workspace's Hugging Face folder, and jobs load
them from there without network access.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

_COMMIT: Final = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class ModelPin:
    """One Hugging Face model, pinned to a commit.

    Attributes:
        name: Short name used in the code, for example ``voxtral``.
        repo_id: Hugging Face repository, for example ``mistralai/Voxtral-Mini-3B-2507``.
        revision: Full 40-character commit hash.
        ignore: File patterns that are not downloaded.
    """

    name: str
    repo_id: str
    revision: str
    ignore: tuple[str, ...] = ()


def load_pins(path: Path) -> dict[str, ModelPin]:
    """Read the model pins from a YAML file.

    Args:
        path: The pins file, for example ``configs/pilot/models.yaml``.

    Returns:
        The pins by name.

    Raises:
        ValueError: If the file is not a mapping, an entry lacks ``repo_id`` or
            ``revision``, or a revision is not a full commit hash.
    """
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: expected a mapping from model names to pins")
    pins: dict[str, ModelPin] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict) or not {"repo_id", "revision"} <= entry.keys():
            raise ValueError(f"{path}: {name} needs a repo_id and a revision")
        revision = str(entry["revision"])
        if not _COMMIT.fullmatch(revision):
            raise ValueError(
                f"{path}: the revision of {name} must be a 40-character commit hash, "
                f"not {revision!r}"
            )
        pins[str(name)] = ModelPin(
            name=str(name),
            repo_id=str(entry["repo_id"]),
            revision=revision,
            ignore=tuple(str(pattern) for pattern in entry.get("ignore") or ()),
        )
    return pins


def local_snapshot(pin: ModelPin, hf_home: Path) -> Path:
    """Return the folder of a snapshot that was fetched before, without network access.

    Args:
        pin: The model pin.
        hf_home: The Hugging Face home folder (``HF_HOME``); on OMNI ``$LBW_WORKSPACE/hf``.

    Returns:
        The snapshot folder, in the standard Hugging Face cache layout.

    Raises:
        FileNotFoundError: If the snapshot has not been fetched.
    """
    repo_folder = "models--" + pin.repo_id.replace("/", "--")
    folder = hf_home / "hub" / repo_folder / "snapshots" / pin.revision
    if not folder.is_dir():
        raise FileNotFoundError(
            f"{pin.repo_id} at {pin.revision[:8]} is not in {hf_home}; "
            "fetch it on a login node with: python scripts/fetch_models.py"
        )
    return folder
