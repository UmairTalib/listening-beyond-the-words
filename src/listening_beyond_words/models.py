"""Pinned model snapshots: which Hugging Face models the project uses, at which commit.

``configs/pilot/models.yaml`` lists the pins. ``scripts/fetch_models.py`` downloads
them on an OMNI login node into the workspace's Hugging Face folder, and jobs load
them from there without network access and record which files they loaded
(``snapshot_record``). ``check_weights`` tells whether a loaded model holds every
weight of its file.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import yaml

if TYPE_CHECKING:
    import torch

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


def snapshot_record(pin: ModelPin, snapshot: Path) -> dict[str, Any]:
    """Describe a fetched snapshot for ``run.json``: its repo, revision and file blobs.

    In the Hugging Face cache each file of a snapshot is a link to a blob named by the
    file's hash, so the blobs say exactly which files a step loaded.

    Args:
        pin: The model pin.
        snapshot: The snapshot folder (``local_snapshot``).

    Returns:
        ``repo_id``, ``revision`` and ``blobs``: for each linked file of the folder, by
        name, the name of the blob it links to.
    """
    blobs = {
        path.name: path.readlink().name for path in sorted(snapshot.iterdir()) if path.is_symlink()
    }
    return {"repo_id": pin.repo_id, "revision": pin.revision, "blobs": blobs}


@dataclass(frozen=True, slots=True)
class WeightCheck:
    """How a loaded model compares with its weights file.

    ``ok`` is True when no weight is missing, unused or changed: the file names every
    weight that matters, each file tensor has a place in the model, and loading the
    file again changed nothing, so the model's own loader had put every tensor there.

    Attributes:
        tensors: Number of tensors in the file.
        missing: Model weights that the file does not name; they keep their initial values.
        unused: File tensors that the model has no place for.
        changed: Model weights that loading the file again changed, so the model's own
            loader had not put the file's values there.
        fixed_buffers: File tensors for buffers that the model computes itself
            (non-persistent buffers, such as position indices) and that equal the
            model's own values; they are not loaded.
        identity_norms: Scale and shift weights of instance norms that the file does not
            name and that still hold exactly 1 and 0, so that each norm computes the same
            as it would without them. They do not count as missing.
    """

    tensors: int
    missing: tuple[str, ...]
    unused: tuple[str, ...]
    changed: tuple[str, ...]
    fixed_buffers: tuple[str, ...]
    identity_norms: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True if no weight is missing, unused or changed."""
        return not (self.missing or self.unused or self.changed)


def check_weights(module: "torch.nn.Module", state: Mapping[str, "torch.Tensor"]) -> WeightCheck:
    """Check that a model's own loader put every tensor of its file in place.

    Some loaders skip tensors whose names do not match without saying so; Kokoro 0.9.4
    falls back to ``strict=False``. This loads the file's tensors into the model again
    and compares every weight before and after, so a weight that the model's own loader
    left out shows up as changed, missing or unused. Run it right after loading, before
    the model is used; the model then holds the file's values. It does not see dtype
    casts made while loading, nor weights tied to one another.

    Instance-norm scale and shift weights that the file does not name are accepted while
    they hold exactly 1 and 0. Kokoro 0.9.4 gives every AdaIN norm these weights
    (``affine=True``, a workaround in its ``istftnet.py``), while its file was trained
    without them; at 1 and 0 the norm computes exactly what it would without them.

    Args:
        module: The model, after its own loader ran.
        state: The file's tensors, named as in ``module.state_dict()``.

    Returns:
        The comparison.

    Raises:
        RuntimeError: If a file tensor's shape does not match the model's weight.
    """
    import torch

    def same(a: torch.Tensor, b: torch.Tensor) -> bool:
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)

    norms = (torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d)
    modules = dict(module.named_modules())

    def at_identity(name: str) -> bool:
        owner, _, leaf = name.rpartition(".")
        norm = modules.get(owner)
        if not isinstance(norm, norms) or not norm.affine or leaf not in ("weight", "bias"):
            return False
        return bool(torch.all(getattr(norm, leaf) == (1.0 if leaf == "weight" else 0.0)))

    before = {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}
    result = module.load_state_dict(dict(state), strict=False)
    after = module.state_dict()
    buffers = dict(module.named_buffers())
    fixed = [
        name
        for name in result.unexpected_keys
        if name in buffers and same(buffers[name], state[name].to(buffers[name].device))
    ]
    identity = [name for name in result.missing_keys if at_identity(name)]
    return WeightCheck(
        tensors=len(state),
        missing=tuple(sorted(set(result.missing_keys) - set(identity))),
        unused=tuple(sorted(set(result.unexpected_keys) - set(fixed))),
        changed=tuple(
            sorted(name for name, tensor in after.items() if not same(before[name], tensor))
        ),
        fixed_buffers=tuple(sorted(fixed)),
        identity_norms=tuple(sorted(identity)),
    )
