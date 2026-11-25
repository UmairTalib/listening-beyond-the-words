"""The zero-shot scores: Voxtral's z of every segment of x0, x1 and rev, from the encoder caches.

The zero-shot arm (pilot protocol, design "Arms" and section "Encoder caches and zero-shot
scores") scores every segment of the two test views, ``x0`` and ``rev``, and of ``x1``,
whose scores start the running statistics of CPG, with the untrained model and the one
fixed prompt. ``scripts/score_zeroshot.py`` scores them from the pinned caches
(``cache.pinned_cache``), one clip per call (``voxtral.cached_score``), after a
check: the segments of ``check_sample`` are scored from the audio and from the cache, and
the two z must agree bit for bit. It writes ``scores.parquet`` (``SCORE_COLUMNS``) into
``runs/pilot/pilot-zeroshot-<YYYYmmdd-HHMMSS>/``, which ``configs/pilot/data.yaml`` pins as
``zeroshot_scores``. ``load_zeroshot_scores`` reads the pinned scores once they fit the
current cache pins. The settings are in ``configs/pilot/zeroshot.yaml``
(``ZeroshotSettings``), which extends the shared training settings,
``configs/pilot/train.yaml``.

No label is read here: the AUROC of the zero-shot arm is computed later by the analysis
code.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from listening_beyond_words.cache import relative_run
from listening_beyond_words.runinfo import StepError, item_seed, sha256_file
from listening_beyond_words.settings import Workspace

STEP: Final = "score_zeroshot"  # the step that writes the zero-shot scores
ARM: Final = "zeroshot"
# The scored views, as configs/pilot/zeroshot.yaml lists them: the two test views and x1,
# whose zero-shot scores start the running statistics. x2 is watched in the cache check,
# never scored.
VIEWS: Final = ("x0", "x1", "rev")
# One row per segment and view, in scores.parquet. ``z`` = logit("Yes") - logit("No"),
# computed in float32 and stored as float64, which holds every float32 value exactly.
SCORE_COLUMNS: Final = ("segment", "session", "view", "z")
SCORES: Final = "scores.parquet"
SAMPLE_STREAM: Final = "zeroshot"  # the check's draw: item_seed(seed, SAMPLE_STREAM, view)
# The shared training settings, which zeroshot.yaml extends like every arm file
# (engineering conventions, section 5): the zero-shot scores wait for the rules of the
# training code (pilot protocol, section "Training").
EXTENDS: Final = "train.yaml"
_CONFIG_KEYS: Final = ("extends", "arm", "views", "check")
_CHECK_KEYS: Final = ("segments_per_view", "seed")
_PIN_KEYS: Final = frozenset({"run", "sha256"})


@dataclass(frozen=True, slots=True)
class ZeroshotSettings:
    """The settings of the zero-shot arm (``configs/pilot/zeroshot.yaml``).

    The file extends ``train.yaml`` (``EXTENDS``), whose settings the zero-shot job reads
    apart; these are the zero-shot arm's own.

    Attributes:
        views: The scored views, in the order they are scored: ``VIEWS``.
        segments_per_view: Segments of each view that the check draws, besides every
            segment of two chunks.
        seed: The seed of that draw (``check_sample``).
    """

    views: tuple[str, ...]
    segments_per_view: int
    seed: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ZeroshotSettings":
        """Read the settings and check every key and value.

        Args:
            config: The content of ``configs/pilot/zeroshot.yaml``, ``extends`` included.

        Returns:
            The settings.

        Raises:
            StepError: If a key is missing or unknown, the file does not extend exactly
                ``train.yaml``, the arm is not ``zeroshot``, the views are not ``VIEWS``
                in this order, or a value of the check is not a whole number of at least 1.
        """
        block = _keys(config, _CONFIG_KEYS, "zeroshot.yaml")
        if block["extends"] != EXTENDS:
            raise StepError(
                f"zeroshot.yaml must extend {EXTENDS!r}, the shared training settings in its"
                f" folder (engineering conventions, section 5), not {block['extends']!r}"
            )
        if block["arm"] != ARM:
            raise StepError(f"zeroshot.yaml: the arm must be {ARM!r}, not {block['arm']!r}")
        views = block["views"]
        if not isinstance(views, list) or tuple(views) != VIEWS:
            raise StepError(
                f"zeroshot.yaml: the views must be {list(VIEWS)} (pilot protocol, section"
                f' "Encoder caches and zero-shot scores"), not {views!r}'
            )
        check = _keys(block["check"], _CHECK_KEYS, "zeroshot.yaml check")
        return cls(
            views=VIEWS,
            segments_per_view=_count(check["segments_per_view"], "check.segments_per_view"),
            seed=_count(check["seed"], "check.seed"),
        )


def _keys(block: Any, expected: tuple[str, ...], name: str) -> Mapping[str, Any]:
    """A config block, refused unless it is a mapping with exactly the expected keys."""
    if not isinstance(block, Mapping):
        raise StepError(f"{name} must be a mapping")
    missing = sorted(set(expected) - set(block))
    unknown = sorted(str(key) for key in set(block) - set(expected))
    if missing or unknown:
        raise StepError(f"{name}: missing keys {missing}, unknown keys {unknown}")
    return block


def _count(value: Any, name: str) -> int:
    """A whole number of at least 1 from the config; True and False are refused."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise StepError(f"zeroshot.yaml: {name} must be a whole number of at least 1")
    return value


def check_sample(manifest: pd.DataFrame, view: str, size: int, seed: int) -> tuple[str, ...]:
    """Return the segments that the check scores from the audio and from the cache.

    ``size`` segments, or all of them if there are fewer, are drawn without replacement
    from the segments in ascending ID, by ``numpy.random.default_rng(item_seed(seed,
    "zeroshot", view))``, so each view has a stream of its own, apart from the stream of
    the cache check (``cache.check_sample``); every segment of more than one chunk is
    added.

    Args:
        manifest: A cache manifest (``cache.CACHE_COLUMNS``); ``segment`` and ``chunks``
            are read.
        view: The clip version.
        size: Segments to draw (``check.segments_per_view`` of ``zeroshot.yaml``).
        seed: The seed (``check.seed``).

    Returns:
        The segments, in ascending ID.
    """
    segments = sorted(str(segment) for segment in manifest["segment"])
    rng = np.random.default_rng(item_seed(seed, SAMPLE_STREAM, view))
    drawn = rng.choice(len(segments), size=min(size, len(segments)), replace=False)
    long = {
        str(segment)
        for segment, count in zip(manifest["segment"], manifest["chunks"], strict=True)
        if int(count) > 1
    }
    return tuple(sorted({segments[int(index)] for index in drawn} | long))


def load_zeroshot_scores(workspace: Workspace, config: Mapping[str, Any]) -> pd.DataFrame:
    """The pinned zero-shot scores (``zeroshot_scores``), checked against the current pins.

    The pin names the run and the SHA-256 of its ``scores.parquet``. The file must match
    the pin, and the run's ``run.json`` must come from ``scripts/score_zeroshot.py``, say
    ``done``, list that file, have scored the views ``VIEWS`` and record as its caches
    exactly the current pins of ``cache_runs``. The table must hold exactly one finite z
    per segment of the pinned cache run of each view (its ``manifest.parquet``, checked
    against its pin) and per view, with the segment's session.

    Args:
        workspace: The workspace.
        config: The data config, with ``zeroshot_scores`` and ``cache_runs``.

    Returns:
        One row per segment and view (``SCORE_COLUMNS``), by view in the order of
        ``VIEWS`` and by segment within each view.

    Raises:
        StepError: If any of this does not hold. The message names rules, runs, views and
            counts, never a score.
    """
    pin = config.get("zeroshot_scores")
    if not isinstance(pin, Mapping) or set(pin) != _PIN_KEYS:
        raise StepError(
            "no zero-shot scores are pinned: zeroshot_scores needs exactly run and sha256"
        )
    name = relative_run(workspace, pin["run"])
    path = workspace.root / name / SCORES
    if not path.is_file() or sha256_file(path) != pin["sha256"]:
        raise StepError(f"the pinned zero-shot scores: {name}/{SCORES} does not match the pin")
    record = workspace.root / name / "run.json"
    info = json.loads(record.read_text(encoding="utf-8")) if record.is_file() else {}
    if info.get("step") != STEP:
        raise StepError(f"{name} is not a run of scripts/score_zeroshot.py")
    if info.get("status") != "done":
        raise StepError(f"{name} did not finish: its run.json does not say done")
    if (info.get("outputs") or {}).get(SCORES) != pin["sha256"]:
        raise StepError(f"{name}: its run.json lists another {SCORES} than the pin")
    views = ((info.get("settings") or {}).get("zeroshot") or {}).get("views")
    if views != list(VIEWS):
        raise StepError(f"{name} scored the views {views!r}, not {list(VIEWS)}")
    recorded = (info.get("inputs") or {}).get("caches") or {}
    current = config.get("cache_runs") or {}
    outdated = [
        view
        for view in VIEWS
        if view not in current or recorded.get(view) != json.loads(json.dumps(current[view]))
    ]
    if outdated or set(recorded) != set(VIEWS):
        raise StepError(
            f"{name} was scored from other cache runs than cache_runs pins now"
            f" ({', '.join(outdated) or 'other views'})"
        )
    table = pd.read_parquet(path)
    if tuple(table.columns) != SCORE_COLUMNS:
        raise StepError(f"{name}: its {SCORES} does not have the columns of SCORE_COLUMNS")
    if not pd.api.types.is_float_dtype(table["z"]):
        raise StepError(f"{name}: the z of its {SCORES} are not floating-point numbers")
    others = sorted(set(map(str, table["view"])) - set(VIEWS))
    if others:
        raise StepError(f"{name}: its {SCORES} holds rows of other views: {others}")
    parts = []
    for view in VIEWS:
        rows = table[table["view"] == view]
        _same_segments(name, view, rows, _cached_segments(workspace, current[view], view))
        parts.append(rows.sort_values("segment"))
    return pd.concat(parts, ignore_index=True)


def _cached_segments(workspace: Workspace, pin: Mapping[str, Any], view: str) -> dict[str, int]:
    """The session of each segment of a pinned cache run, from its manifest, checked by its pin."""
    path = workspace.root / relative_run(workspace, pin["run"]) / "manifest.parquet"
    if not path.is_file() or sha256_file(path) != pin["sha256"]:
        raise StepError(f"the pinned cache run of {view}: its manifest.parquet does not match")
    manifest = pd.read_parquet(path, columns=["segment", "session"])
    return {
        str(segment): int(session)
        for segment, session in zip(manifest["segment"], manifest["session"], strict=True)
    }


def _same_segments(name: str, view: str, rows: pd.DataFrame, cached: Mapping[str, int]) -> None:
    """Stop unless a view's rows hold one finite z per cached segment, with its session."""
    segments = [str(segment) for segment in rows["segment"]]
    twice = len(segments) - len(set(segments))
    if twice:
        raise StepError(f"{name}: {SCORES} holds {twice} segments of {view} twice")
    missing = len(set(cached) - set(segments))
    extra = len(set(segments) - set(cached))
    if missing or extra:
        raise StepError(
            f"{name}: {SCORES} lacks {missing} segments of the cache of {view} and holds"
            f" {extra} others"
        )
    sessions = sum(
        int(session) != cached[segment]
        for segment, session in zip(segments, rows["session"], strict=True)
    )
    if sessions:
        raise StepError(f"{name}: {sessions} rows of {view} name another session than the cache")
    non_finite = int((~np.isfinite(rows["z"].to_numpy(dtype=np.float64))).sum())
    if non_finite:
        raise StepError(f"{name}: {non_finite} scores of {view} are not finite")
