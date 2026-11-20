"""espeak-ng for Kokoro's text front end: the copy built in the workspace.

Kokoro's text front end (misaki) gives words outside its lexicon to espeak-ng; without
it, such words get no sound. The library bundled with espeakng-loader 0.2.4 needs glibc
2.29, but OMNI runs Rocky Linux 8 with glibc 2.28, so ``scripts/build_espeak_ng.sh``
builds the same espeak-ng commit (1.52.0) into the workspace, and this module points
misaki at that build.
"""

from typing import Final

from listening_beyond_words.settings import Workspace

VERSION: Final = (1, 52, 0)


def use_workspace_espeak(workspace: Workspace) -> None:
    """Make misaki's espeak fallback use the espeak-ng built in the workspace.

    Importing ``misaki.espeak`` points phonemizer at the bundled library. This replaces
    that choice, so call it after that import (Kokoro's pipeline module makes it) and
    before Kokoro's pipeline is built, because the pipeline builds the fallback.

    Args:
        workspace: The workspace that holds the build, in ``env/espeak-ng``.

    Raises:
        FileNotFoundError: If the build is missing.
        RuntimeError: If the library found is not espeak-ng 1.52.0.
    """
    library = workspace.espeak / "lib" / "libespeak-ng.so"
    data = workspace.espeak / "share" / "espeak-ng-data"
    for path in (library, data):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; build it on a login node with: bash scripts/setup_omni_env.sh"
            )
    import misaki.espeak  # noqa: F401  (on import it sets the bundled library, replaced below)
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(str(library))
    EspeakWrapper.set_data_path(str(data))
    found = tuple(EspeakWrapper().version)
    if found != VERSION:
        raise RuntimeError(f"{library} is espeak-ng {found}, not {VERSION}")
