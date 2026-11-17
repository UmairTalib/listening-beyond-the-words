import io
import random
import struct
import zipfile

import pytest

from listening_beyond_words.remote_zip import RangeFile, find_member, open_remote_zip

TRANSCRIPT = (
    b"start_time\tstop_time\tspeaker\tvalue\n1.0\t2.0\tEllie\thello\n2.5\t3.1\tParticipant\thi\n"
)


def make_archive(audio_bytes: int = 1 << 20) -> bytes:
    """A synthetic session archive: a large incompressible member, then a small stored one."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("999_AUDIO.wav", random.Random(0).randbytes(audio_bytes))
        archive.writestr("999_TRANSCRIPT.csv", TRANSCRIPT, compress_type=zipfile.ZIP_STORED)
        archive.writestr("999_COVAREP.csv", b"0,1,2\n" * 1000)
    return buffer.getvalue()


class Server:
    """A fake web server that answers range requests and counts them."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.requests = 0
        self.sent = 0

    def fetch(self, first: int, last: int) -> bytes:
        self.requests += 1
        chunk = self.data[first : last + 1]
        self.sent += len(chunk)
        return chunk


def test_reads_one_member_without_fetching_the_archive() -> None:
    data = make_archive()
    server = Server(data)
    archive, raw = open_remote_zip(len(data), server.fetch)
    with archive:
        member = find_member(archive, "999_transcript.CSV")
        assert member is not None
        assert archive.read(member) == TRANSCRIPT
    assert server.sent < len(data) // 4
    assert raw.fetched == server.sent
    assert raw.requests == server.requests


def test_lists_every_member_with_its_checksum() -> None:
    data = make_archive(audio_bytes=1000)
    archive, _ = open_remote_zip(len(data), Server(data).fetch)
    with archive, zipfile.ZipFile(io.BytesIO(data)) as local:
        assert [i.filename for i in archive.infolist()] == local.namelist()
        assert [i.CRC for i in archive.infolist()] == [i.CRC for i in local.infolist()]


def test_find_member_ignores_folders_and_reports_a_missing_name() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("999_P/999_TRANSCRIPT.csv", TRANSCRIPT)
    data = buffer.getvalue()
    archive, _ = open_remote_zip(len(data), Server(data).fetch)
    with archive:
        found = find_member(archive, "999_TRANSCRIPT.csv")
        assert found is not None
        assert found.filename == "999_P/999_TRANSCRIPT.csv"
        assert find_member(archive, "999_AUDIO.wav") is None


def test_a_changed_member_fails_its_checksum() -> None:
    data = bytearray(make_archive(audio_bytes=1000))
    with zipfile.ZipFile(io.BytesIO(bytes(data))) as local:
        offset = local.getinfo("999_TRANSCRIPT.csv").header_offset
    name_length, extra_length = struct.unpack("<HH", data[offset + 26 : offset + 30])
    data[offset + 30 + name_length + extra_length + 5] ^= 0xFF  # one byte of the stored text
    archive, _ = open_remote_zip(len(data), Server(bytes(data)).fetch)
    with archive, pytest.raises(zipfile.BadZipFile, match="CRC"):
        archive.read("999_TRANSCRIPT.csv")


def test_an_archive_cut_off_during_upload_cannot_be_opened() -> None:
    data = make_archive()
    cut = data[: len(data) // (1 << 16) * (1 << 16)]  # cut at a 64 KiB boundary
    with pytest.raises(zipfile.BadZipFile):
        open_remote_zip(len(cut), Server(cut).fetch)


def test_a_short_answer_from_the_server_is_an_error() -> None:
    data = make_archive(audio_bytes=1000)
    raw = RangeFile(len(data), lambda first, last: data[first:last])  # one byte short
    with pytest.raises(OSError, match="received"):
        raw.read(10)


def test_seek_tell_and_end_of_file() -> None:
    raw = RangeFile(100, lambda first, last: bytes(range(first, last + 1)))
    assert raw.seek(-10, io.SEEK_END) == 90
    assert raw.read(5) == bytes(range(90, 95))
    assert raw.tell() == 95
    assert raw.read(100) == bytes(range(95, 100))
    assert raw.read(1) == b""
    assert raw.requests == 2
    with pytest.raises(OSError, match="before the start"):
        raw.seek(-1)
