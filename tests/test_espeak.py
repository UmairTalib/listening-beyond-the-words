import sys
import types
from pathlib import Path

import pytest

from listening_beyond_words.espeak import VERSION, use_workspace_espeak
from listening_beyond_words.settings import Workspace


class FakeEspeakWrapper:
    """Records the paths it is given, like phonemizer's class-level settings."""

    library: str | None = None
    data_path: str | None = None
    version: tuple[int, int, int] = VERSION

    @classmethod
    def set_library(cls, library: str) -> None:
        cls.library = library

    @classmethod
    def set_data_path(cls, data_path: str) -> None:
        cls.data_path = data_path


@pytest.fixture
def wrapper(monkeypatch: pytest.MonkeyPatch) -> type[FakeEspeakWrapper]:
    """Stand-ins for misaki and phonemizer, which only OMNI has installed."""
    monkeypatch.setattr(FakeEspeakWrapper, "library", None)
    monkeypatch.setattr(FakeEspeakWrapper, "data_path", None)
    monkeypatch.setattr(FakeEspeakWrapper, "version", VERSION)
    wrapper_module = types.ModuleType("phonemizer.backend.espeak.wrapper")
    wrapper_module.EspeakWrapper = FakeEspeakWrapper
    for name in ("misaki", "misaki.espeak", "phonemizer", "phonemizer.backend"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(
        sys.modules, "phonemizer.backend.espeak", types.ModuleType("phonemizer.backend.espeak")
    )
    monkeypatch.setitem(sys.modules, "phonemizer.backend.espeak.wrapper", wrapper_module)
    return FakeEspeakWrapper


def built(root: Path) -> Workspace:
    """A workspace that holds an espeak-ng build."""
    workspace = Workspace(root)
    (workspace.espeak / "lib").mkdir(parents=True)
    (workspace.espeak / "lib" / "libespeak-ng.so").write_bytes(b"")
    (workspace.espeak / "share" / "espeak-ng-data").mkdir(parents=True)
    return workspace


def test_workspace_build_lives_in_the_environment_folder(tmp_path: Path) -> None:
    assert Workspace(tmp_path).espeak == tmp_path / "env" / "espeak-ng"


def test_use_workspace_espeak_points_phonemizer_at_the_build(
    tmp_path: Path, wrapper: type[FakeEspeakWrapper]
) -> None:
    workspace = built(tmp_path)
    use_workspace_espeak(workspace)
    assert wrapper.library == str(workspace.espeak / "lib" / "libespeak-ng.so")
    assert wrapper.data_path == str(workspace.espeak / "share" / "espeak-ng-data")


def test_use_workspace_espeak_says_how_to_build_a_missing_library(
    tmp_path: Path, wrapper: type[FakeEspeakWrapper]
) -> None:
    with pytest.raises(FileNotFoundError, match=r"scripts/setup_omni_env\.sh"):
        use_workspace_espeak(Workspace(tmp_path))
    assert wrapper.library is None


def test_use_workspace_espeak_refuses_another_version(
    tmp_path: Path, wrapper: type[FakeEspeakWrapper], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wrapper, "version", (1, 51, 1))
    with pytest.raises(RuntimeError, match="not"):
        use_workspace_espeak(built(tmp_path))
