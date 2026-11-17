"""Take the DAIC-WOZ transcripts out of the official session archives, on an OMNI login node.

Usage, inside the clone:
    "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/fetch_daicwoz_transcripts.py "$DAICWOZ_URL"

DAICWOZ_URL is the DAIC-WOZ download folder that USC ICT gives after the licence agreement.
It is passed when the script runs and never written into a file in git. For each of the 189
pilot sessions, the script reads only the zip index and NNN_TRANSCRIPT.csv of NNN_P.zip,
with HTTP range requests made by curl, and zipfile checks the transcript against the CRC-32
in the archive. It writes $LBW_WORKSPACE/daicwoz/run-<UTC time>/ with
transcripts/NNN_TRANSCRIPT.csv (read-only), members.tsv (the index of every archive),
sessions.tsv and run.json.

It also compares the NNN_AUDIO.wav of each archive (size and CRC-32 from the index) with the
E-DAIC recording of the same session. Equal files share their time zero, so the transcript
times apply to the E-DAIC audio as they are. Prints aggregate numbers only, never transcript
text.
"""

import csv
import hashlib
import io
import os
import subprocess
import sys
import zipfile
import zlib
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from listening_beyond_words.remote_zip import Fetch, find_member, open_remote_zip
from listening_beyond_words.runinfo import git_state, utc_now, utc_stamp, write_json
from listening_beyond_words.sessions import PILOT_IDS, session_files
from listening_beyond_words.settings import get_workspace

CURL = (
    "curl",
    "--silent",
    "--show-error",
    "--fail",
    "--retry",
    "5",
    "--retry-delay",
    "3",
    "--connect-timeout",
    "30",
    "--max-time",
    "300",
)
# No answer may be larger than this: a server that ignored the range would send a whole archive.
MAX_RANGE_BYTES = 16 << 20


def curl_head(url: str) -> dict[str, str]:
    """Headers of the final answer to a HEAD request, with lower-case names."""
    out = subprocess.run([*CURL, "--head", url], capture_output=True, text=True, check=True).stdout
    headers: dict[str, str] = {}
    for line in out.splitlines():
        if line.startswith("HTTP/"):
            headers = {"status": line.split()[1]}
        elif ":" in line:
            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()
    return headers


def curl_range(url: str) -> Fetch:
    """Range requests on one remote file, made by curl."""

    def fetch(first: int, last: int) -> bytes:
        if last - first + 1 > MAX_RANGE_BYTES:
            raise ValueError(f"range of {last - first + 1} bytes is larger than allowed")
        command = [*CURL, "--max-filesize", str(MAX_RANGE_BYTES), "--range", f"{first}-{last}"]
        return subprocess.run(
            [*command, "--output", "-", url], capture_output=True, check=True
        ).stdout

    return fetch


def turn_counts(data: bytes) -> dict[str, int]:
    """Lines per speaker and lines whose times are not two numbers in order."""
    rows = list(csv.reader(io.StringIO(data.decode("utf-8", errors="replace")), delimiter="\t"))
    header = [cell.strip().lower() for cell in rows[0]] if rows else []
    if not {"start_time", "stop_time", "speaker"} <= set(header):
        return {
            "parsed": 0,
            "ellie_lines": 0,
            "participant_lines": 0,
            "other_lines": 0,
            "bad_rows": 0,
        }
    i_start, i_stop, i_speaker = (header.index(c) for c in ("start_time", "stop_time", "speaker"))
    speakers: Counter[str] = Counter()
    bad = 0
    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue
        if len(row) <= max(i_start, i_stop, i_speaker):
            bad += 1
            continue
        speakers[row[i_speaker].strip().lower()] += 1
        try:
            in_order = 0.0 <= float(row[i_start]) <= float(row[i_stop])
        except ValueError:
            in_order = False
        bad += not in_order
    return {
        "parsed": 1,
        "ellie_lines": speakers["ellie"],
        "participant_lines": speakers["participant"],
        "other_lines": sum(n for s, n in speakers.items() if s not in ("ellie", "participant")),
        "bad_rows": bad,
    }


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


def fetch_session(
    base: str, session_id: int, out_dir: Path, sessions_dir: Path
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Take one session's transcript out of its archive and compare the archive's audio."""
    url = f"{base}{session_id}_P.zip"
    head = curl_head(url)
    size = int(head["content-length"])
    row: dict[str, object] = {
        "session": session_id,
        "zip_bytes": size,
        "zip_last_modified": head.get("last-modified", ""),
        "zip_etag": head.get("etag", ""),
    }
    try:
        archive, raw = open_remote_zip(size, curl_range(url))
    except zipfile.BadZipFile as error:
        row.update(status="archive damaged", detail=str(error)[:120])
        return row, []
    with archive:
        infos = archive.infolist()
        listing: list[dict[str, object]] = [
            {
                "session": session_id,
                "member": info.filename,
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
            row.update(status="no transcript in the archive")
            return row, listing
        data = archive.read(transcript)  # zipfile checks the CRC-32 here
        write_read_only(out_dir / f"{session_id}_TRANSCRIPT.csv", data)
        row.update(
            status="ok",
            transcript_member=transcript.filename,
            transcript_bytes=len(data),
            transcript_crc32=f"{transcript.CRC:08x}",
            transcript_sha256=hashlib.sha256(data).hexdigest(),
            transcript_modified=listing[infos.index(transcript)]["modified"],
            **turn_counts(data),
        )
        audio = find_member(archive, f"{session_id}_AUDIO.wav")
        row.update(requests=raw.requests, fetched_bytes=raw.fetched)
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


def main() -> None:
    os.umask(0o077)
    if len(sys.argv) != 2:
        raise SystemExit("usage: fetch_daicwoz_transcripts.py <DAIC-WOZ download folder URL>")
    base = sys.argv[1].rstrip("/") + "/"
    workspace = get_workspace()
    run_dir = workspace.daicwoz / f"run-{utc_stamp()}"
    out_dir = run_dir / "transcripts"
    out_dir.mkdir(parents=True, exist_ok=False)
    info: dict[str, object] = {
        "step": "fetch_daicwoz_transcripts",
        "status": "running",
        "started_utc": utc_now(),
        "git": git_state(),
        "source": {"host": urlparse(base).hostname, "files": "NNN_P.zip, transcript member only"},
        "settings": {"max_range_bytes": MAX_RANGE_BYTES},
    }
    write_json(run_dir / "run.json", info)
    rows: list[dict[str, object]] = []
    members: list[dict[str, object]] = []
    try:
        for number, session_id in enumerate(PILOT_IDS, start=1):
            row, listing = fetch_session(base, session_id, out_dir, workspace.edaic / "sessions")
            rows.append(row)
            members += listing
            if number % 25 == 0:
                print(f"{number} of {len(PILOT_IDS)} archives read", flush=True)
        info.update(status="done")
    except BaseException:
        info.update(status="failed")
        raise
    finally:
        sessions = pd.DataFrame(rows)
        sessions.to_csv(run_dir / "sessions.tsv", sep="\t", index=False)
        pd.DataFrame(members).to_csv(run_dir / "members.tsv", sep="\t", index=False)
        info.update(finished_utc=utc_now(), sessions_read=len(rows))
        write_json(run_dir / "run.json", info)

    out_dir.chmod(0o500)
    ok = sessions[sessions["status"] == "ok"]
    print(f"archives read: {len(sessions)}; transcripts taken: {len(ok)}")
    for status, group in sessions[sessions["status"] != "ok"].groupby("status"):
        print(f"  {status}: {ids(group)}")
    table = pd.DataFrame(members)
    print(
        f"members per archive: min {table.groupby('session').size().min()},"
        f" max {table.groupby('session').size().max()};"
        f" member dates {table['modified'].min()} to {table['modified'].max()}"
    )
    print(
        f"transcripts: {int(ok['transcript_bytes'].sum()):,} bytes in total;"
        f" dated {ok['transcript_modified'].min()} to {ok['transcript_modified'].max()}"
    )
    print(
        f"lines: Ellie {int(ok['ellie_lines'].sum()):,}, participant"
        f" {int(ok['participant_lines'].sum()):,}, other speakers {int(ok['other_lines'].sum())},"
        f" rows with bad times {int(ok['bad_rows'].sum())}; unreadable header: {ids(ok[ok['parsed'] == 0])}"
    )
    print(f"sessions without Ellie lines: {ids(ok[ok['ellie_lines'] == 0])}")
    audio = ok["audio"].value_counts().to_dict()
    print(f"archive audio against the E-DAIC recording: {audio}")
    print(f"  different: {ids(ok[ok['audio'] == 'different'])}")
    print(
        f"range requests: {int(ok['requests'].sum()):,},"
        f" {ok['fetched_bytes'].sum() / 2**20:.1f} MiB fetched"
        f" of {sessions['zip_bytes'].sum() / 2**30:.1f} GiB of archives"
    )
    print(f"run: {run_dir.relative_to(workspace.root)}")


if __name__ == "__main__":
    main()
