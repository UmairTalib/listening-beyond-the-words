"""Print the runs: block of configs/pilot/evaluate.yaml for the done training runs (login node).

Run on an OMNI login node, in the clone, once the training jobs have ended (squeue shows no
lbw-train job waiting or running):
    "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/list_runs.py
It takes no option. It reads only the run.json files of the training run folders
$LBW_WORKSPACE/runs/pilot/pilot-<arm>-r<round>-f<fold>-<YYYYmmdd-HHMMSS>/, with
configs/pilot/data.yaml (the label and partition pins that a kept run must have used) and
configs/pilot/evaluate.yaml (the first runs' commit), and opens no score.

Pilot analysis, section "Runs and pins", bullet "Kept runs": a run counts if its record says
done, it was not started with --allow-dirty, it passed check F1, it used the pinned labels
and partitions, and it came from the first runs' commit (B0's round 1 and CPG's round 1,
fold 1 only) or from the newest v0.1.x tag of its clone (evaluation.kept_run). If two runs
that count share a fold-run, the earlier one counts: the one that finished first
(evaluation.run_order: finished_utc, then the folder's name, as the training job orders the
runs of CPG's round 1, fold 1), and the other is reported. If the earliest run that counts
cannot be pinned (its record does not say when it finished, so no run of the fold-run can be
ordered, or it lists no scores with their SHA-256), no run of its fold-run is pinned, since
the analysis job would refuse any other: it applies the same rules. Each kept run is pinned
by its folder, the SHA-256 of its scores.parquet as its run.json lists it, and the SHA-256
of its run.json; the analysis job checks both against the files.

It prints the block, ready to replace "runs: {}" in configs/pilot/evaluate.yaml in the pin
commit, then, as comments, the version of the protocol that the pinned set makes (all 30
runs, 25, 20 or 10) and every run folder that is not pinned, with the reason. It prints run
folders, hashes, rules and counts only: no participant ID and no score.
"""

import contextlib
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from listening_beyond_words.config import read_settings
from listening_beyond_words.evaluation import (
    ARMS,
    ROUNDS,
    RUN_NAME,
    SHA256,
    EvaluateSettings,
    RunPin,
    kept_run,
    pinned_version,
    run_order,
)
from listening_beyond_words.partitions import N_FOLDS
from listening_beyond_words.runinfo import Progress, StepError, record_failure
from listening_beyond_words.settings import Workspace, get_workspace
from listening_beyond_words.zeroshot import SCORES

CONFIG = Path("configs/pilot/data.yaml")
EVALUATE = Path("configs/pilot/evaluate.yaml")
STUDY = "pilot"
USAGE = "usage: list_runs.py (no arguments)"


class UnorderedRunError(StepError):
    """A run that counts but whose record does not say when it finished.

    No run of its fold-run can then be told to be the earlier one, so none is pinned.

    Attributes:
        key: The run's fold-run, as (arm, round, fold).
    """

    def __init__(self, key: tuple[str, int, int], message: str) -> None:
        super().__init__(message)
        self.key = key


def candidate(
    workspace: Workspace,
    folder: Path,
    config: Mapping[str, Any],
    first_runs_commit: str,
) -> tuple[tuple[str, int, int], tuple[str, str], RunPin | str]:
    """A training run folder that counts: its fold-run, its order and its pin.

    Args:
        workspace: The workspace.
        folder: The run folder.
        config: The data config, with the label and partition pins.
        first_runs_commit: The first runs' commit (evaluate.yaml).

    Returns:
        Its fold-run as (arm, round, fold); its order (evaluation.run_order); and its pin,
        or, if its run.json lists no scores with their SHA-256, why it cannot be pinned.

    Raises:
        UnorderedRunError: If the run counts but its record does not say when it finished.
        StepError: If the run does not count, with the reason.
    """
    name = folder.relative_to(workspace.root).as_posix()
    found = RUN_NAME.fullmatch(folder.name)
    if found is None:
        raise StepError(f"{name} is not named as a training run of the pilot")
    key = (found.group(1), int(found.group(2)), int(found.group(3)))
    if key[1] not in ROUNDS or not 1 <= key[2] <= N_FOLDS:
        raise StepError(f"{name} names no fold-run of the pilot")
    path = folder / "run.json"
    if not path.is_file():
        raise StepError(f"{name} has no run.json: it never started its record")
    data = path.read_bytes()
    try:
        record = json.loads(data.decode("utf-8"))
    except ValueError:
        raise StepError(f"{name}: its run.json cannot be read as JSON") from None
    if not isinstance(record, dict):
        raise StepError(f"{name}: its run.json is not a JSON object")
    kept_run(record, name, *key, config, first_runs_commit)
    try:
        order = run_order(record, name)
    except StepError as error:
        raise UnorderedRunError(key, str(error)) from None
    scores = (record.get("outputs") or {}).get(SCORES)
    if not isinstance(scores, str) or not SHA256.fullmatch(scores):
        return key, order, f"{name}: its run.json lists no {SCORES} with its SHA-256"
    return key, order, RunPin(name, scores, hashlib.sha256(data).hexdigest())


def listing(
    workspace: Workspace, config: Mapping[str, Any], first_runs_commit: str
) -> tuple[dict[tuple[str, int, int], RunPin], list[str]]:
    """The kept run of each fold-run, and every training run folder left out, with why.

    Args:
        workspace: The workspace.
        config: The data config, with the label and partition pins.
        first_runs_commit: The first runs' commit (evaluate.yaml).

    Returns:
        The pin of each fold-run's kept run; and one line per run folder left out. A fold-run
        whose earliest run that counts cannot be pinned, or that has a run that counts but
        cannot be ordered, gets no pin: each of its runs is left out, with why.
    """
    found: dict[tuple[str, int, int], list[tuple[tuple[str, str], RunPin | str]]] = {}
    unordered: dict[tuple[str, int, int], list[str]] = {}
    left_out: list[str] = []
    for folder in sorted((workspace.runs / STUDY).glob(f"{STUDY}-*")):
        if not folder.is_dir() or not RUN_NAME.fullmatch(folder.name):
            continue  # the zero-shot run and anything else that is no training run
        try:
            key, order, pin = candidate(workspace, folder, config, first_runs_commit)
        except UnorderedRunError as error:
            unordered.setdefault(error.key, []).append(str(error))
            continue
        except StepError as error:
            left_out.append(str(error))
            continue
        found.setdefault(key, []).append((order, pin))
    pins: dict[tuple[str, int, int], RunPin] = {}
    keys = sorted(set(found) | set(unordered), key=lambda item: (ARMS.index(item[0]), *item[1:]))
    for key in keys:
        where = f"{key[0]}, round {key[1]}, fold {key[2]}"
        ordered = sorted(found.get(key, []), key=lambda run: run[0])
        if key in unordered:
            left_out += unordered[key]
            left_out += [
                f"{order[1]}: another run of {where} counts but cannot be ordered, so no run of"
                " it is pinned"
                for order, _ in ordered
            ]
            continue
        (_, first), *later = ordered
        if isinstance(first, str):  # the earliest run that counts cannot be pinned
            left_out.append(first)
            left_out += [
                f"{order[1]}: the earliest run of {where} that counts cannot be pinned, so no"
                " run of it is pinned"
                for order, _ in later
            ]
            continue
        pins[key] = first
        left_out += [
            pin
            if isinstance(pin, str)
            else f"{order[1]}: a later done run of {where}; the earlier {first.run} counts"
            for order, pin in later
        ]
    return pins, left_out


def block(pins: Mapping[tuple[str, int, int], RunPin]) -> str:
    """The runs: block of evaluate.yaml for these pins, as YAML."""
    runs: dict[str, Any] = {}
    for (arm, round_number, fold), pin in pins.items():
        runs.setdefault(arm, {}).setdefault(f"r{round_number}", {})[f"f{fold}"] = {
            "run": pin.run,
            "scores_sha256": pin.scores_sha256,
            "record_sha256": pin.record_sha256,
        }
    return yaml.safe_dump({"runs": runs}, sort_keys=False, default_flow_style=False, width=1000)


def list_runs() -> None:
    """Print the block, the version it makes and the run folders left out."""
    if sys.argv[1:]:
        raise StepError(USAGE)
    workspace = get_workspace()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    content, _ = read_settings(EVALUATE)
    # Only the first runs' commit is needed, but every other setting is checked as well; the
    # pins already there are left out, so that a block can always be printed again.
    settings = EvaluateSettings.from_config({**content, "runs": {}})
    pins, left_out = listing(workspace, config, settings.first_runs_commit)
    print(block(pins), end="")
    try:
        version = pinned_version(tuple(pins))
        made = "no run" if version is None else f"{version.name} ({version.runs} runs)"
        print(f"# version: {made}")
    except StepError as error:
        print(f"# version: none: {error}")
    for line in left_out:
        print(f"# not pinned: {line}")


def main() -> None:
    os.umask(0o077)
    progress = Progress()
    try:
        list_runs()
        return
    except BaseException as caught:
        error = caught  # reported below, outside this clause, so that nothing chains to it
    with contextlib.suppress(BaseException):  # nothing here may print the error's message
        failure = record_failure(error, progress)
        print(failure.line)
        if failure.frames is not None:  # frames and types only, never a message
            print(failure.frames, end="", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
