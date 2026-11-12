from pathlib import Path

import pytest

from listening_beyond_words.sessions import PILOT_IDS, check_pilot_id, session_files


def test_pilot_ids_are_the_189_sessions_from_300_to_492() -> None:
    assert len(PILOT_IDS) == 189
    assert PILOT_IDS[0] == 300
    assert PILOT_IDS[-1] == 492
    assert not {342, 394, 398, 460} & set(PILOT_IDS)


@pytest.mark.parametrize("session_id", [299, 342, 493, 600, 718])
def test_check_pilot_id_refuses_every_other_session(session_id: int) -> None:
    with pytest.raises(ValueError, match="not a pilot session"):
        check_pilot_id(session_id)


def test_session_files_point_into_the_session_folder(tmp_path: Path) -> None:
    audio, transcript = session_files(tmp_path, 300)
    assert audio == tmp_path / "300_P" / "300_AUDIO.wav"
    assert transcript == tmp_path / "300_P" / "300_Transcript.csv"


def test_session_files_refuse_a_test_set_session(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a pilot session"):
        session_files(tmp_path, 600)
