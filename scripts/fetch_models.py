"""Download the pinned model snapshots into the workspace, on an OMNI login node.

Usage, inside the clone, with the environment from setup_omni_env.sh:
    "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/fetch_models.py [configs/pilot/models.yaml]

Each snapshot goes to $LBW_WORKSPACE/hf at its pinned revision. Files that are
already there are not downloaded again, so the script is safe to re-run. Jobs
never download: they load these snapshots offline.
"""

import os
import sys
from pathlib import Path

from listening_beyond_words.models import load_pins
from listening_beyond_words.settings import get_workspace


def main() -> None:
    os.umask(0o077)
    pins_file = Path(sys.argv[1] if len(sys.argv) > 1 else "configs/pilot/models.yaml")
    # Set before importing huggingface_hub, so that downloads and its caches stay
    # in the workspace instead of the small home folder.
    os.environ["HF_HOME"] = str(get_workspace().hf_home)
    from huggingface_hub import snapshot_download

    for pin in load_pins(pins_file).values():
        folder = snapshot_download(
            pin.repo_id,
            revision=pin.revision,
            ignore_patterns=list(pin.ignore) or None,
        )
        print(f"{pin.name}: {pin.repo_id} at {pin.revision[:8]} in {folder}")


if __name__ == "__main__":
    main()
