#!/usr/bin/env bash
# unpack_edaic_pilot.sh: unpack the audio and transcript of the 189 pilot sessions.
#
# Usage:   bash unpack_edaic_pilot.sh <edaic_dir> [parallel_jobs]
# Example: bash unpack_edaic_pilot.sh "$(ws_find cpg-thesis)/edaic" 8
#
# Reads   <edaic_dir>/raw/data/NNN_P.tar.gz             packed originals (read-only)
#         <edaic_dir>/manifest/sha256sums.pinned.txt    checksums recorded at download: a
#                                                       read-only copy of sha256sums.txt, made
#                                                       once after the download reports DONE
#                                                       (cp -p sha256sums.txt sha256sums.pinned.txt)
# Writes  <edaic_dir>/sessions/NNN_P/NNN_AUDIO.wav      read-only
#         <edaic_dir>/sessions/NNN_P/NNN_Transcript.csv read-only
#         <edaic_dir>/manifest/sessions_sha256.txt      checksums of the unpacked files
#         <edaic_dir>/manifest/sessions_inventory.tsv   session, audio bytes, transcript bytes
#         <edaic_dir>/manifest/unpack_status.txt        RUNNING / DONE / FAILED
#
# Rules it follows
#   - Uses an explicit list of the 189 pilot IDs: 300-492 without 342, 394, 398 and 460,
#     which never existed. It never lists, opens or matches sessions 600-718.
#   - Checks every archive against its download checksum before unpacking it.
#   - Unpacks only the two files the pilot needs; the feature files stay packed.
#   - Unpacks each session into a temporary folder and moves it into place only when it
#     is complete, so an interrupted run never leaves a half-unpacked session.
#   - Prints session IDs, sizes and status only, never file contents.
#   - Safe to re-run: complete sessions are skipped.
set -uo pipefail
umask 077

usage="usage: unpack_edaic_pilot.sh <edaic_dir> [parallel_jobs]"
EDAIC="${1:?$usage}"
JOBS="${2:-8}"
EDAIC=$(cd "$EDAIC" && pwd) || exit 1
RAW="$EDAIC/raw/data"
SUMS="$EDAIC/manifest/sha256sums.pinned.txt"
OUT="$EDAIC/sessions"
MANIFEST="$EDAIC/manifest"
STATUS="$MANIFEST/unpack_status.txt"

log() { echo "$(date '+%F %T') $*"; }
fail() { log "FAILED: $*"; echo "FAILED $(date '+%F %T'): $*" > "$STATUS"; exit 1; }
sha256() { if command -v sha256sum > /dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
nbytes() { wc -c < "$1" | tr -d ' '; }
rmtree() {   # remove a temporary folder of this script, even if its files are read-only
  [ -e "$1" ] || return 0
  chmod -R u+w -- "$1" 2> /dev/null
  rm -rf -- "$1"
}

[ -r "$SUMS" ] || fail "cannot read $SUMS"
mkdir -p "$OUT" || fail "cannot create $OUT"

# Only one run at a time (mkdir is atomic, also on the shared cluster file system).
LOCK="$MANIFEST/.unpack.lock"
if ! mkdir "$LOCK" 2> /dev/null; then
  log "another unpack seems to be running ($LOCK exists)."
  log "If no unpack job is running, remove that empty folder and start again."
  exit 1
fi
trap 'rmdir "$LOCK" 2> /dev/null' EXIT
echo "RUNNING since $(date '+%F %T') on $(hostname)" > "$STATUS"

# The 189 pilot IDs: 300-492 without the four IDs that never existed.
IDS=()
for id in $(seq 300 492); do
  case "$id" in 342 | 394 | 398 | 460) continue ;; esac
  IDS+=("$id")
done
[ "${#IDS[@]}" -eq 189 ] || fail "expected 189 pilot IDs, got ${#IDS[@]}"

unpack_one() {   # $1 = pilot session ID
  local id="$1" arc dst tmp want got f
  if [ "$id" -lt 300 ] || [ "$id" -gt 492 ]; then echo "REFUSE $id (not a pilot session)"; return 1; fi
  arc="$RAW/${id}_P.tar.gz"
  dst="$OUT/${id}_P"
  tmp="$OUT/.partial_${id}_P"
  rmtree "$tmp"   # leftover of an interrupted run (safe: only one run holds the lock)
  if [ -e "$dst" ]; then
    if [ -s "$dst/${id}_AUDIO.wav" ] && [ -s "$dst/${id}_Transcript.csv" ] &&
       [ "$(ls -A "$dst" | wc -l | tr -d ' ')" -eq 2 ]; then
      echo "SKIP  $id (already unpacked)"
      return 0
    fi
    echo "FAIL  $id (an incomplete $dst exists; left untouched)"
    return 1
  fi
  [ -r "$arc" ] || { echo "FAIL  $id (archive missing or unreadable)"; return 1; }
  want=$(awk -v p="./data/${id}_P.tar.gz" '$2 == p { print $1 }' "$SUMS")
  [ -n "$want" ] || { echo "FAIL  $id (no checksum recorded)"; return 1; }
  got=$(sha256 "$arc" | awk '{ print $1 }')
  [ "$got" = "$want" ] || { echo "FAIL  $id (archive checksum does not match the download)"; return 1; }
  mkdir -p "$tmp" || { echo "FAIL  $id (cannot create $tmp)"; return 1; }
  if ! tar -xzf "$arc" -C "$tmp" --no-same-owner \
      "${id}_P/${id}_AUDIO.wav" "${id}_P/${id}_Transcript.csv"; then
    rmtree "$tmp"
    echo "FAIL  $id (tar reported an error)"
    return 1
  fi
  for f in "${id}_AUDIO.wav" "${id}_Transcript.csv"; do
    if [ ! -s "$tmp/${id}_P/$f" ]; then
      rmtree "$tmp"
      echo "FAIL  $id ($f missing or empty)"
      return 1
    fi
  done
  chmod 0400 "$tmp/${id}_P/${id}_AUDIO.wav" "$tmp/${id}_P/${id}_Transcript.csv"
  # Move first, then make the folder read-only: a read-only folder cannot be moved.
  if ! mv -- "$tmp/${id}_P" "$dst"; then
    rmtree "$tmp"
    echo "FAIL  $id (could not move the session into place)"
    return 1
  fi
  rmdir -- "$tmp"
  chmod 0500 "$dst"
  echo "OK    $id (audio $(nbytes "$dst/${id}_AUDIO.wav") bytes, transcript $(nbytes "$dst/${id}_Transcript.csv") bytes)"
}
export -f unpack_one sha256 nbytes rmtree
export RAW OUT SUMS

log "unpacking ${#IDS[@]} pilot sessions with $JOBS parallel jobs into $OUT ..."
printf '%s\0' "${IDS[@]}" | xargs -0 -n 1 -P "$JOBS" bash -c 'unpack_one "$1"' _

log "checking the result ..."
expected=$(printf '%s_P\n' "${IDS[@]}" | sort)
found=$(cd "$OUT" && find . -mindepth 1 -maxdepth 1 -print | sed 's|^\./||' | sort)
if [ "$found" != "$expected" ]; then
  echo "Entries in $OUT that should not be there:"
  comm -13 <(echo "$expected") <(echo "$found") | sed 's/^/  /'
  echo "Pilot sessions missing:"
  comm -23 <(echo "$expected") <(echo "$found") | sed 's/^/  /'
  fail "the sessions folder does not hold exactly the 189 pilot sessions"
fi
bad=0
for id in "${IDS[@]}"; do
  d="$OUT/${id}_P"
  if [ ! -s "$d/${id}_AUDIO.wav" ] || [ ! -s "$d/${id}_Transcript.csv" ] ||
     [ "$(ls -A "$d" | wc -l | tr -d ' ')" -ne 2 ]; then
    echo "INCOMPLETE $id"
    bad=$((bad + 1))
  fi
done
[ "$bad" -eq 0 ] || fail "$bad session(s) incomplete"

log "writing checksums and the inventory ..."
(cd "$OUT" && find . -type f -print0 | sort -z | xargs -0 -n 32 -P 4 bash -c 'sha256 "$@"' _) |
  sort -k 2 > "$MANIFEST/sessions_sha256.txt"
nsums=$(wc -l < "$MANIFEST/sessions_sha256.txt" | tr -d ' ')
[ "$nsums" -eq $((2 * ${#IDS[@]})) ] || fail "expected $((2 * ${#IDS[@]})) checksums, wrote $nsums"
{
  printf 'session\taudio_bytes\ttranscript_bytes\n'
  for id in "${IDS[@]}"; do
    printf '%s\t%s\t%s\n' "$id" "$(nbytes "$OUT/${id}_P/${id}_AUDIO.wav")" "$(nbytes "$OUT/${id}_P/${id}_Transcript.csv")"
  done
} > "$MANIFEST/sessions_inventory.tsv"
total=$(awk -F'\t' 'NR > 1 { s += $2 + $3 } END { printf "%.2f", s / 1e9 }' "$MANIFEST/sessions_inventory.tsv")
echo "DONE $(date '+%F %T'): ${#IDS[@]} sessions, $nsums files ($total GB), checksums in sessions_sha256.txt, files read-only" > "$STATUS"
log "$(cat "$STATUS")"
