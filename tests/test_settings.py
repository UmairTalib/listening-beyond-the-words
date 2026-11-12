from pathlib import Path

import pytest

from listening_beyond_words.settings import WORKSPACE_ENV, get_workspace


def test_workspace_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(WORKSPACE_ENV, str(tmp_path))
    workspace = get_workspace()
    assert workspace.root == tmp_path.resolve()
    assert workspace.runs == tmp_path.resolve() / "runs"


def test_missing_variable_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(WORKSPACE_ENV)
    with pytest.raises(RuntimeError, match=WORKSPACE_ENV):
        get_workspace()


def test_missing_folder_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(WORKSPACE_ENV, str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        get_workspace()


@pytest.mark.parametrize("bad", ["/etc/passwd", "../outside", "derived/../../outside"])
def test_config_paths_stay_inside_the_workspace(bad: str) -> None:
    with pytest.raises(ValueError, match="workspace"):
        get_workspace().resolve(bad)


def test_relative_config_path_resolves_inside_the_workspace() -> None:
    workspace = get_workspace()
    assert workspace.resolve("derived/labels.parquet") == workspace.derived / "labels.parquet"
