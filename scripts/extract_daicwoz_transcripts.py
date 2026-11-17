"""Take the DAIC-WOZ transcripts out of the downloaded official archives, on an OMNI login node.

Usage, inside the clone, after scripts/download_daicwoz.sh has finished:
    "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/extract_daicwoz_transcripts.py

Reads $LBW_WORKSPACE/daicwoz/raw/NNN_P.zip for each of the 189 pilot sessions; zipfile checks
each transcript against the CRC-32 in its archive. Writes $LBW_WORKSPACE/daicwoz/run-<UTC time>/
with transcripts/NNN_TRANSCRIPT.csv (read-only), members.tsv (the index of every archive),
sessions.tsv and run.json. A session that fails is recorded with its error, and the run goes on.

It also compares each archive's NNN_AUDIO.wav (size and CRC-32 from the index) with the
E-DAIC recording of the same session. "identical" means the same recording, byte for byte.
Whether a transcript's times fit its own recording is a separate question: the DAIC-WOZ
documentation lists sessions with offsets and interruptions. Prints aggregate numbers and
session IDs, never transcript text.
"""

import hashlib
import os
import platform
import socket
import zipfile
import zlib
from pathlib import Path

import pandas as pd

from listening_beyond_words.daicwoz import COUNT_KEYS, find_member, turn_counts
from listening_beyond_words.runinfo import git_state, sha256_file, utc_now, utc_stamp, write_json
from listening_beyond_words.sessions import PILOT_IDS, session_files
from listening_beyond_words.settings import get_workspace

ERRORS = (OSError, EOFError, ValueError, zipfile.BadZipFile, zlib.error, NotImplementedError)


def crc32_file(path: Path) -> int:
    """CRC-32 of a file, as zip archives record it."""
    crc = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            crc = zlib.crc32(block, crc)
    return crc


def write_read_only(path: Path, data: bytes) -> None:
    """Write a new file that nobody can change afterwards; never overwrite one."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def read_session(
    raw_dir: Path, session_id: int, out_dir: Path, sessions_dir: Path
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Take one session's transcript out of its archive and compare the archive's audio."""
    path = raw_dir / f"{session_id}_P.zip"
    row: dict[str, object] = {"session": session_id, "zip_bytes": path.stat().st_size}
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as error:
        row.update(status="archive damaged", detail=str(error)[:120])
        return row, []
    with archive:
        infos = sorted(archive.infolist(), key=lambda info: info.header_offset)
        listing: list[dict[str, object]] = [
            {
                "session": session_id,
                "member": info.filename,
                "offset": info.header_offset,
                "bytes": info.file_size,
                "packed_bytes": info.compress_size,
                "crc32": f"{info.CRC:08x}",
                "modified": "{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}".format(*info.date_time),
                "method": info.compress_type,
            }
            for info in infos
        ]
        transcript = find_member(archive, f"{session_id}_TRANSCRIPT.csv")
        if transcript is None:
            row.update(status="no transcript in the archive", members=len(infos))
            return row, listing
        data = archive.read(transcript)  # zipfile checks the CRC-32 here
        counts = turn_counts(data)
        audio = find_member(archive, f"{session_id}_AUDIO.wav")
        position = infos.index(transcript) + 1
    write_read_only(out_dir / f"{session_id}_TRANSCRIPT.csv", data)
    row.update(
        status="ok",
        members=len(infos),
        transcript_position=position,
        transcript_member=transcript.filename,
        transcript_bytes=len(data),
        transcript_crc32=f"{transcript.CRC:08x}",
        transcript_sha256=hashlib.sha256(data).hexdigest(),
        transcript_modified=listing[position - 1]["modified"],
        **counts,
    )
    if audio is None:
        row.update(audio="no audio in the archive")
        return row, listing
    edaic_audio, _ = session_files(sessions_dir, session_id)
    edaic_bytes = edaic_audio.stat().st_size
    same = edaic_bytes == audio.file_size and crc32_file(edaic_audio) == audio.CRC
    row.update(
        audio="identical" if same else "different",
        audio_bytes_daicwoz=audio.file_size,
        audio_bytes_edaic=edaic_bytes,
    )
    return row, listing


def ids(table: pd.DataFrame) -> str:
    return ", ".join(str(s) for s in table["session"]) or "none"


def summary(sessions: pd.DataFrame, members: pd.DataFrame) -> None:
    ok = sessions[sessions["status"] == "ok"]
    print(f"archives read: {len(sessions)}; transcripts taken: {len(ok)}")
    for status, group in sessions[sessions["status"] != "ok"].groupby("status"):
        details = sorted({str(d) for d in group.get("detail", pd.Series(dtype=str)).dropna()})
        print(f"  {status}: {ids(group)}" + (f" ({'; '.join(details)})" if details else ""))
    if not members.empty:
        per_archive = members.groupby("session").size()
        print(
            f"members per archive: {per_archive.min()} to {per_archive.max()};"
            f" member dates {members['modified'].min()} to {members['modified'].max()}"
        )
    if ok.empty:
        print("no transcript was taken")
        return
    last = int((ok["transcript_position"] == ok["members"]).sum())
    print(f"the transcript is the last member in {last} of {len(ok)} archives")
    print(
        f"transcripts: {int(ok['transcript_bytes'].sum()):,} bytes in total;"
        f" dated {ok['transcript_modified'].min()} to {ok['transcript_modified'].max()}"
    )
    print(
        f"lines: Ellie {int(ok['ellie_lines'].sum()):,}, participant"
        f" {int(ok['participant_lines'].sum()):,}, other speakers {int(ok['other_lines'].sum())},"
        f" rows too short or with bad times {int(ok['bad_rows'].sum())};"
        f" unreadable header: {ids(ok[ok['parsed'] == 0])}"
    )
    print(f"sessions without Ellie lines: {ids(ok[ok['ellie_lines'] == 0])}")
    print(f"archive audio against the E-DAIC recording: {ok['audio'].value_counts().to_dict()}")
    print(f"  not identical: {ids(ok[ok['audio'] != 'identical'])}")


def main() -> None:
    os.umask(0o077)
    workspace = get_workspace()
    raw_dir = workspace.daicwoz / "raw"
    manifest = workspace.daicwoz / "manifest"
    status = (manifest / "download_status.txt").read_text(encoding="utf-8").strip()
    if not status.startswith("DONE"):
        raise SystemExit(f"the DAIC-WOZ download is not complete: {status}")
    run_dir = workspace.daicwoz / f"run-{utc_stamp()}"
    out_dir = run_dir / "transcripts"
    out_dir.mkdir(parents=True, exist_ok=False)
    edaic_sums = workspace.edaic / "manifest" / "sessions_sha256.txt"
    info: dict[str, object] = {
        "step": "extract_daicwoz_transcripts",
        "status": "running",
        "started_utc": utc_now(),
        "git": git_state(),
        "node": socket.gethostname(),
        "environment": {"python": platform.python_version(), "pandas": pd.__version__},
        "inputs": {
            "daicwoz_raw": {
                "folder": str(raw_dir.relative_to(workspace.root)),
                "sha256sums_sha256": sha256_file(manifest / "sha256sums.txt"),
                "download_status": status,
            },
            "edaic_sessions": {
                "folder": str((workspace.edaic / "sessions").relative_to(workspace.root)),
                "sessions_sha256_sha256": sha256_file(edaic_sums) if edaic_sums.exists() else None,
            },
        },
    }
    write_json(run_dir / "run.json", info)
    rows: list[dict[str, object]] = []
    members: list[dict[str, object]] = []
    try:
        for session_id in PILOT_IDS:
            try:
                row, listing = read_session(
                    raw_dir, session_id, out_dir, workspace.edaic / "sessions"
                )
            except ERRORS as error:
                row = {
                    "session": session_id,
                    "status": "error",
                    "detail": f"{type(error).__name__}: {str(error)[:120]}",
                }
                listing = []
            rows.append(row)
            members += listing
        info.update(status="done")
    except BaseException as error:
        info.update(status="failed", error=type(error).__name__)
        raise
    finally:
        sessions = pd.DataFrame(rows, columns=None if rows else ["session", "status"])
        for key in COUNT_KEYS:
            if key not in sessions:
                sessions[key] = pd.NA
        sessions.to_csv(run_dir / "sessions.tsv", sep="\t", index=False)
        table = pd.DataFrame(members)
        table.to_csv(run_dir / "members.tsv", sep="\t", index=False)
        out_dir.chmod(0o500)
        info.update(finished_utc=utc_now(), sessions_read=len(rows))
        write_json(run_dir / "run.json", info)
    summary(sessions, table)
    print(f"run: {run_dir.relative_to(workspace.root)}")


if __name__ == "__main__":
    main()
