#!/usr/bin/env bash
# download_daicwoz.sh: download DAIC-WOZ from the USC server straight into an OMNI workspace.
#
# Usage:   bash scripts/download_daicwoz.sh <base_url> <dest_dir> [parallel_streams]
# Example: bash scripts/download_daicwoz.sh "$DAICWOZ_URL" "$LBW_WORKSPACE/daicwoz/raw" 6
#
# What it does
#   1. Reads the server's file index and asks the server for the exact size of every
#      file -> <dest_dir>/../manifest/server_listing.tsv
#   2. Downloads every file: the documents first, then the 189 session archives
#      (NNN_P.zip, sessions 300-492). Complete files are skipped and partial files
#      resumed, so it is safe to re-run after an interruption.
#   3. Checks every local size against the server, writes ../manifest/sha256sums.txt
#      and makes the downloaded files read-only.
#   Progress: ../manifest/download_status.txt (RUNNING / DONE / INCOMPLETE / FAILED).
#
# Rules it follows (project data rules)
#   - Downloads only. It never unpacks anything.
#   - Every session archive must be one of the 189 pilot sessions (300-492 without 342,
#     394, 398 and 460). Any other session stops the script before the first download.
#   - Prints file names, sizes and status only, never file contents.
#   - The USC link is passed in when it runs; it is not stored in git.
#   - Runs on OMNI (GNU coreutils), never on the Mac.
set -uo pipefail
umask 077

usage="usage: download_daicwoz.sh <base_url> <dest_dir> [parallel_streams]"
BASE="${1:?$usage}"
DEST="${2:?$usage}"
STREAMS="${3:-4}"
EXPECT_SESSIONS=189
[[ "$BASE" == */ ]] || BASE="$BASE/"
mkdir -p "$DEST" && DEST=$(cd "$DEST" && pwd) || exit 1
MANIFEST="$(dirname "$DEST")/manifest"
STATUS="$MANIFEST/download_status.txt"
mkdir -p "$MANIFEST" || exit 1

log() { echo "$(date '+%F %T') $*"; }
urldecode() { printf '%b' "${1//%/\\x}"; }

# Only one copy may run at a time (the lock file lives on the shared workspace).
exec 9>"$MANIFEST/.download.lock"
if ! flock -n 9; then
  log "another download is already running: $(cat "$STATUS" 2>/dev/null)"
  exit 1
fi
echo "RUNNING since $(date '+%F %T') on $(hostname), pid $$" > "$STATUS"
fail() { log "FAILED: $*"; echo "FAILED $(date '+%F %T'): $*" > "$STATUS"; exit 1; }

# ---- 1. File list with exact sizes ------------------------------------------------
list_files() {   # $1 = folder ('' = top level); prints paths relative to BASE
  local q=""
  [ -n "$1" ] && q="?dir=$1"
  curl -fsS --max-time 120 "$BASE$q" | grep -o 'href="[^"]*"' | sed 's/^href="//; s/"$//' | sort -u |
  while read -r h; do
    case "$h" in
      '?dir='*)
        sub="${h#\?dir=}"
        if [ -n "$sub" ] && { [ -z "$1" ] || [[ "$sub" == "$1"/* ]]; }; then list_files "$sub"; fi ;;
      '?'*|/*|http:*|https:*|mailto:*|index.php|robots.txt) ;;
      *) echo "$h" ;;
    esac
  done
}

log "reading the file list from the server ..."
list_files "" | sort -u > "$MANIFEST/.files.tmp"
[ -s "$MANIFEST/.files.tmp" ] || fail "the server page listed no files"

outside=$(grep -E '^[0-9]+_P\.zip$' "$MANIFEST/.files.tmp" | sed 's/_P\.zip$//' |
  awk '$1 < 300 || $1 > 492 || $1 == 342 || $1 == 394 || $1 == 398 || $1 == 460' | tr '\n' ' ')
[ -z "$outside" ] || fail "session archives outside the 189 pilot sessions: $outside"

head_size() {
  curl -fsSI --max-time 60 "$BASE$1" | tr -d '\r' |
    awk 'tolower($1)=="content-length:" {v=$2} END {print v}'
}
export -f head_size
export BASE
xargs -d '\n' -P 8 -n 1 bash -c 's=$(head_size "$1"); printf "%s\t%s\n" "$1" "${s:-?}"' _ \
  < "$MANIFEST/.files.tmp" > "$MANIFEST/.sizes.tmp"
unknown=$(awk -F'\t' '$2 !~ /^[0-9]+$/' "$MANIFEST/.sizes.tmp" | wc -l)
[ "$unknown" -eq 0 ] || fail "$unknown file(s) without a size from the server"
sessions=$(grep -c -E '^[0-9]+_P\.zip'$'\t' "$MANIFEST/.sizes.tmp")
[ "$sessions" -eq "$EXPECT_SESSIONS" ] ||
  fail "found $sessions session archives, expected $EXPECT_SESSIONS (listing incomplete?)"

# Order: documents first, then the sessions by number.
awk -F'\t' '{
    g = 0; id = 0
    if ($1 ~ /^[0-9]+_P\.zip$/) { g = 1; id = $1 + 0 }
    printf "%d\t%06d\t%s\t%s\n", g, id, $1, $2
  }' "$MANIFEST/.sizes.tmp" | sort -t$'\t' -k1,1n -k2,2n -k3,3 | cut -f3,4 \
  > "$MANIFEST/server_listing.tsv"
rm -f "$MANIFEST/.files.tmp" "$MANIFEST/.sizes.tmp"
nfiles=$(wc -l < "$MANIFEST/server_listing.tsv")
total=$(awk -F'\t' '{s += $2} END {printf "%.1f", s / 1e9}' "$MANIFEST/server_listing.tsv")
log "$nfiles files ($sessions session archives), $total GB in total"

# ---- 2. Download -------------------------------------------------------------------
fetch_one() {   # $1 = path relative to BASE, $2 = expected size in bytes
  local rel="$1" expect="$2" f have rc attempt
  f="$DEST/$(urldecode "$rel")"
  mkdir -p "$(dirname "$f")"
  have=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ "$have" = "$expect" ]; then echo "SKIP  $rel (already complete)"; return 0; fi
  for attempt in $(seq 1 30); do
    if [ "$have" -gt "$expect" ]; then rm -f -- "$f"; fi   # bigger than on the server: restart
    curl -fsS -L -C - --connect-timeout 60 --speed-limit 10240 --speed-time 180 \
      -o "$f" "$BASE$rel"
    rc=$?
    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$have" = "$expect" ]; then echo "OK    $rel ($expect bytes)"; return 0; fi
    if [ "$rc" = 33 ]; then rm -f -- "$f"; have=0; fi       # server cannot resume: restart
    echo "RETRY $rel (try $attempt/30, curl exit $rc, $have of $expect bytes)"
    sleep 30
  done
  echo "FAIL  $rel"
  return 1
}
export -f fetch_one urldecode
export DEST
log "downloading with $STREAMS parallel streams ..."
tr '\t' '\n' < "$MANIFEST/server_listing.tsv" |
  xargs -d '\n' -n 2 -P "$STREAMS" bash -c 'fetch_one "$1" "$2"' _

# ---- 3. Check sizes, write checksums, make read-only --------------------------------
log "checking every file against the server size ..."
problems=0
while IFS=$'\t' read -r rel size; do
  have=$(stat -c %s "$DEST/$(urldecode "$rel")" 2>/dev/null || echo missing)
  if [ "$have" != "$size" ]; then
    echo "MISMATCH $rel: local $have, server $size bytes"
    problems=$((problems + 1))
  fi
done < "$MANIFEST/server_listing.tsv"
if [ "$problems" -ne 0 ]; then
  echo "INCOMPLETE $(date '+%F %T'): $problems of $nfiles files missing or wrong size; re-run to resume" > "$STATUS"
  log "$(cat "$STATUS")"
  exit 1
fi
log "all $nfiles files complete; computing sha256 checksums ..."
(cd "$DEST" && find . -type f -print0 | sort -z | xargs -0 -P 4 -n 16 sha256sum) |
  sort -k 2 > "$MANIFEST/sha256sums.txt"
nsums=$(wc -l < "$MANIFEST/sha256sums.txt")
[ "$nsums" -eq "$nfiles" ] || log "WARNING: $nsums checksums for $nfiles files (extra files in $DEST?)"
find "$DEST" -type f -exec chmod a-w {} +
echo "DONE $(date '+%F %T'): $nfiles files ($total GB) complete, checksums in sha256sums.txt, files read-only" > "$STATUS"
log "$(cat "$STATUS")"
