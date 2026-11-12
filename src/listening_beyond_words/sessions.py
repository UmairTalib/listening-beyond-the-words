"""The pilot's sessions: which E-DAIC sessions the pilot may read, and where their files are.

The pilot uses only sessions 300-492, the 189 DAIC-WOZ participants (342, 394, 398
and 460 never existed). Sessions 600-718 are the external test set of the final
thesis, and no code path of the pilot opens them.
"""

from pathlib import Path
from typing import Final

NEVER_EXISTED: Final = frozenset({342, 394, 398, 460})
PILOT_IDS: Final = tuple(i for i in range(300, 493) if i not in NEVER_EXISTED)


def check_pilot_id(session_id: int) -> int:
    """Return the session ID if the pilot may use it.

    Args:
        session_id: An E-DAIC session ID.

    Returns:
        The same ID.

    Raises:
        ValueError: If the ID is not one of the 189 pilot sessions.
    """
    if session_id not in PILOT_IDS:
        raise ValueError(f"session {session_id} is not a pilot session (300-492 only)")
    return session_id


def session_files(sessions_dir: Path, session_id: int) -> tuple[Path, Path]:
    """Return the audio and transcript paths of one pilot session.

    Args:
        sessions_dir: Folder with the unpacked sessions, ``<workspace>/edaic/sessions``.
        session_id: A pilot session ID.

    Returns:
        The paths of ``NNN_AUDIO.wav`` and ``NNN_Transcript.csv``.

    Raises:
        ValueError: If the ID is not one of the 189 pilot sessions.
    """
    check_pilot_id(session_id)
    folder = sessions_dir / f"{session_id}_P"
    return folder / f"{session_id}_AUDIO.wav", folder / f"{session_id}_Transcript.csv"
