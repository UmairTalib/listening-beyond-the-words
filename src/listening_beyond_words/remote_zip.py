"""Read single members of a zip archive on a web server without downloading the archive.

A zip archive ends with its central directory, which gives the name, size, position and
CRC-32 of every member. Reading that directory and then one member takes only a few small
range requests, so a transcript of a few kilobytes can be taken out of a session archive of
several hundred megabytes. ``zipfile`` checks the CRC-32 of every member it reads in full,
and that the member's own header names the same file as the directory.
"""

import io
import zipfile
from collections.abc import Callable
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from _typeshed import WriteableBuffer

Fetch = Callable[[int, int], bytes]
"""Returns the bytes from ``first`` to ``last`` of the remote file, both included."""

BLOCK_BYTES: Final = 1 << 16  # least number of bytes asked for in one range request


class RangeFile(io.RawIOBase):
    """A read-only, seekable remote file whose bytes come from range requests.

    Args:
        size: Size of the remote file in bytes.
        fetch: Returns the bytes from ``first`` to ``last`` (both included).

    Attributes:
        size: Size of the remote file in bytes.
        requests: Number of range requests made so far.
        fetched: Number of bytes received so far.
    """

    def __init__(self, size: int, fetch: Fetch) -> None:
        super().__init__()
        if size < 0:
            raise ValueError(f"negative file size: {size}")
        self.size = size
        self.requests = 0
        self.fetched = 0
        self._fetch = fetch
        self._position = 0

    def readable(self) -> bool:
        """The file can be read."""
        return True

    def seekable(self) -> bool:
        """The file supports seeking."""
        return True

    def tell(self) -> int:
        """Return the current position in bytes."""
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move to a new position, as ``io.IOBase.seek``; nothing is fetched yet."""
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"unknown whence: {whence}")
        if position < 0:
            raise OSError(f"seek to {position}, before the start of the file")
        self._position = position
        return position

    def readinto(self, buffer: "WriteableBuffer") -> int:
        """Fill ``buffer`` from the current position with one range request.

        Returns:
            The number of bytes read; 0 at the end of the file.

        Raises:
            OSError: If the server sends fewer or more bytes than asked for.
        """
        view = memoryview(buffer).cast("B")
        wanted = min(len(view), self.size - self._position)
        if wanted <= 0:
            return 0
        first = self._position
        data = self._fetch(first, first + wanted - 1)
        if len(data) != wanted:
            raise OSError(f"asked for {wanted} bytes at offset {first}, received {len(data)}")
        view[:wanted] = data
        self._position += wanted
        self.requests += 1
        self.fetched += wanted
        return wanted


def open_remote_zip(
    size: int, fetch: Fetch, block: int = BLOCK_BYTES
) -> tuple[zipfile.ZipFile, RangeFile]:
    """Open a zip archive on a web server for reading.

    Args:
        size: Size of the archive in bytes.
        fetch: Range requests on the archive, as for ``RangeFile``.
        block: Read-ahead of the buffer in bytes; small reads are served from it.

    Returns:
        The open archive, and its range file, whose counters show how much was fetched.

    Raises:
        zipfile.BadZipFile: If the end of the archive or its central directory is missing
            or damaged, for example because the file was cut off during an upload.
    """
    raw = RangeFile(size, fetch)
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=block)), raw


def find_member(archive: zipfile.ZipFile, name: str) -> zipfile.ZipInfo | None:
    """Return the member whose file name, without folders, is ``name`` in any case.

    Args:
        archive: An open archive.
        name: The file name to look for, for example ``300_TRANSCRIPT.csv``.

    Returns:
        The member, or None if the archive has none of that name.

    Raises:
        ValueError: If more than one member has that name.
    """
    wanted = name.lower()
    matches = [
        info for info in archive.infolist() if info.filename.rsplit("/", 1)[-1].lower() == wanted
    ]
    if len(matches) > 1:
        raise ValueError(f"{len(matches)} members are named {name}")
    return matches[0] if matches else None
