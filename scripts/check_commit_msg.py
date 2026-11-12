"""Check that a commit message follows the project format.

Git runs this as the commit-msg hook (installed with ``make hooks``). The rules
are written down in docs/engineering_conventions.md, section "Commits".

Usage: python scripts/check_commit_msg.py <file-with-commit-message>
"""

import re
import sys
from pathlib import Path

# fmt: off
TYPES = (
    "build", "chore", "ci", "docs", "exp", "feat",
    "fix", "perf", "refactor", "revert", "style", "test",
)
SCOPES = (
    "cache", "cli", "config", "data", "deps", "eval", "loss",
    "model", "pilot", "readme", "release", "repo", "slurm", "train", "views",
)
NOT_IMPERATIVE = frozenset({
    "added", "adds", "adding", "changed", "changes", "changing", "created", "creates",
    "fixed", "fixes", "fixing", "improved", "improves", "moved", "moves", "removed",
    "removes", "renamed", "renames", "updated", "updates", "updating", "wrote", "writes",
})
# fmt: on
MAX_HEADER = 72
HEADER = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[a-z0-9-]+)\))?(?P<breaking>!)?: (?P<subject>\S.*)$"
)
AUTHOR_TRAILER = re.compile(r"^[A-Za-z][A-Za-z-]*-by\s*:", re.IGNORECASE)
SCISSORS = "# ------------------------ >8 ------------------------"
EXEMPT = ("Merge ", 'Revert "', "fixup! ", "squash! ", "amend! ")


def problems(message: str) -> list[str]:
    """Return the rule violations in a commit message; an empty list means it is fine."""
    lines = []
    for line in message.splitlines():
        if line.startswith(SCISSORS):
            break
        if not line.startswith("#"):
            lines.append(line.rstrip())
    while lines and not lines[-1]:
        lines.pop()
    if not lines or not lines[0]:
        return ["the message is empty or starts with a blank line"]

    header = lines[0]
    if header.startswith(EXEMPT):
        return []

    found = []
    match = HEADER.match(header)
    if match is None:
        found.append(f"the header must look like 'type(scope): subject', got: {header!r}")
    else:
        kind, scope, subject = match["type"], match["scope"], match["subject"]
        if kind not in TYPES:
            found.append(f"unknown type {kind!r}; use one of: {', '.join(TYPES)}")
        if scope is not None and scope not in SCOPES:
            found.append(f"unknown scope {scope!r}; use one of: {', '.join(SCOPES)}")
        if subject[0].isupper():
            found.append("start the subject with a lowercase verb, for example 'add'")
        if subject.split()[0].lower() in NOT_IMPERATIVE:
            found.append("write the subject in the imperative: 'add', not 'added' or 'adds'")
        if subject.endswith("."):
            found.append("do not end the subject with a period")
    if len(header) > MAX_HEADER:
        found.append(f"the header has {len(header)} characters; the limit is {MAX_HEADER}")
    if len(lines) > 1 and lines[1]:
        found.append("leave one blank line between the header and the body")
    for line in lines[1:]:
        if AUTHOR_TRAILER.match(line):
            name = line.split(":")[0]
            found.append(f"remove the trailer {name!r}; the commit author says who wrote it")
    return found


def main(argv: list[str]) -> int:
    """Check the message file named in ``argv[1]`` and report problems on stderr."""
    if len(argv) != 2:
        sys.stderr.write("usage: check_commit_msg.py <commit-message-file>\n")
        return 2
    found = problems(Path(argv[1]).read_text(encoding="utf-8"))
    for item in found:
        sys.stderr.write(f"commit message: {item}\n")
    if found:
        sys.stderr.write("See docs/engineering_conventions.md, section 'Commits'.\n")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
