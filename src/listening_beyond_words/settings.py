"""Locations of data, caches and runs on the current machine.

Every path in the code comes from this module. The workspace root is read from
the environment variable ``LBW_WORKSPACE``; no other code hard-codes a location.
On OMNI, set it once in ``~/.bashrc``::

    export LBW_WORKSPACE="$(ws_find cpg-thesis)"

Configs store paths relative to the workspace root, so the same config works
after a move to a new workspace.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

WORKSPACE_ENV: Final = "LBW_WORKSPACE"


@dataclass(frozen=True, slots=True)
class Workspace:
    """Standard folders inside the workspace root.

    Attributes:
        root: Absolute path of the workspace root.
    """

    root: Path

    @property
    def edaic(self) -> Path:
        """E-DAIC download, checksums and the unpacked pilot sessions."""
        return self.root / "edaic"

    @property
    def derived(self) -> Path:
        """Labels, partitions, word times and segment manifests."""
        return self.root / "derived"

    @property
    def views(self) -> Path:
        """Clip versions: original, re-voiced, muffled and reversed."""
        return self.root / "views"

    @property
    def cache(self) -> Path:
        """Frozen-encoder outputs, one folder per clip version."""
        return self.root / "cache"

    @property
    def runs(self) -> Path:
        """Training and scoring runs, one folder per run."""
        return self.root / "runs"

    @property
    def hf_home(self) -> Path:
        """Pinned model snapshots; jobs use it as ``HF_HOME``."""
        return self.root / "hf"

    @property
    def logs(self) -> Path:
        """Slurm job logs."""
        return self.root / "logs"

    def resolve(self, relative: str | Path) -> Path:
        """Turn a workspace-relative path from a config into an absolute path.

        Args:
            relative: Path relative to the workspace root.

        Returns:
            The absolute path inside the workspace.

        Raises:
            ValueError: If the path is absolute or points outside the workspace.
        """
        candidate = Path(relative)
        if candidate.is_absolute():
            raise ValueError(f"config paths must be relative to the workspace: {relative}")
        resolved = (self.root / candidate).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"path leaves the workspace: {relative}")
        return resolved


def get_workspace() -> Workspace:
    """Return the workspace named by the ``LBW_WORKSPACE`` environment variable.

    Raises:
        RuntimeError: If the variable is not set.
        FileNotFoundError: If the folder does not exist.
    """
    value = os.environ.get(WORKSPACE_ENV, "").strip()
    if not value:
        raise RuntimeError(
            f"{WORKSPACE_ENV} is not set. Point it to the workspace root, "
            f'for example: export {WORKSPACE_ENV}="$(ws_find cpg-thesis)"'
        )
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{WORKSPACE_ENV} points to a missing folder: {root}")
    return Workspace(root)
