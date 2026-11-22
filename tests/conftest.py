"""Shared test setup: synthetic data only, no network and no real workspace."""

import os
from pathlib import Path

import pytest

from x1_fakes import X1Workspace, make_x1_workspace

# Tests never download models or data. Set before any test imports transformers.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


@pytest.fixture(autouse=True)
def isolated_workspace(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point LBW_WORKSPACE at an empty temporary folder in every test."""
    monkeypatch.setenv("LBW_WORKSPACE", str(tmp_path_factory.mktemp("workspace")))


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip tests marked ``gpu`` when no CUDA device is present."""
    gpu_items = [item for item in items if item.get_closest_marker("gpu")]
    if not gpu_items:
        return
    import torch  # imported here so that CPU-only runs start fast

    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in gpu_items:
        item.add_marker(skip)


@pytest.fixture
def x1_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> X1Workspace:
    """A synthetic x1 workspace, with LBW_WORKSPACE pointing at it."""
    workspace = make_x1_workspace(tmp_path / "workspace")
    monkeypatch.setenv("LBW_WORKSPACE", str(workspace.root))
    return workspace
