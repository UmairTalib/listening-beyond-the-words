"""Scoring clips from the pinned encoder caches, and the embargo that comes before any score.

Every test-fold score of the pilot comes from here: the zero-shot job's
(``scripts/score_zeroshot.py``) and those of the trained runs, which score their own test
fold (pilot protocol, section "Training", bullet "Scoring"). ``embargo`` refuses until the
pinned word-check run shows the checks of ``x1`` finished (section "Clip versions", bullet
"If a check fails"). ``score_views`` reads each segment's stored encoder output
(``cache.read_file``) and scores it, one clip per call (``voxtral.cached_score``), while
the caller's overflow watch counts; it writes nothing, so that the caller writes no score
after a flag or a z that is not finite. ``watch_summary`` is what a watch saw, and
``label_free`` gives the numbers of one view's scores that need no label.

Library code never prints: the functions return counts and readings, and the jobs print
them. A refusal is a ``StepError`` that names rules and runs; an unreadable cache file
names its segment.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from listening_beyond_words.cache import PinnedCache, read_file, relative_run
from listening_beyond_words.runinfo import Progress, StepError, sha256_file
from listening_beyond_words.settings import Workspace
from listening_beyond_words.voxtral import LayerWatch, cached_score
from listening_beyond_words.wordcheck import FAIL, PASS
from listening_beyond_words.zeroshot import SCORE_COLUMNS

WORDCHECK_STEP: Final = "check_views"  # scripts/check_views.py writes the word-check runs ...
WORDCHECK_MODE: Final = "sample"  # ... and this is its default mode
# The views whose checks lift the embargo: the word and voice rules of x1, and F4 on rev.
EMBARGO_VIEWS: Final = ("x1", "rev")
EMBARGO_VERDICTS: Final = ("f4", "x1_words", "x1_voice")
LARGEST: Final = 5  # the layer maxima that a watch's summary keeps


# The embargo ----------------------------------------------------------------------------------


def embargo(workspace: Workspace, config: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse unless the pinned word-check run shows the checks of x1 finished.

    No test-fold score is computed until then (pilot protocol, section "Clip versions",
    bullet "If a check fails"). The run must match its pin (``wordcheck_run``, by its
    ``transcripts.parquet``), come from the word check's default mode, say done, list the
    pinned file, have read the pinned clips of ``x1`` and ``rev`` (``view_runs``;
    ``cache.pinned_cache`` then refuses a cache not built from them), and hold x1 words
    PASS, a verdict on x1's voice and F4 PASS. Anything else refuses: a later fallback
    brings its own code.

    Args:
        workspace: The workspace.
        config: The data config, with ``wordcheck_run`` and ``view_runs``.

    Returns:
        The reading for ``run.json``: the run, its SHA-256s, the view pins compared and the
        three verdicts.

    Raises:
        StepError: If any of this does not hold.
    """
    pin = config.get("wordcheck_run")
    if not isinstance(pin, Mapping) or not pin.get("run") or not pin.get("sha256"):
        raise StepError(
            "no word-check run is pinned (wordcheck_run): no test-fold score is computed"
            " until the checks of x1 are finished"
        )
    name = relative_run(workspace, pin["run"])
    folder = workspace.root / name
    transcripts = folder / "transcripts.parquet"
    if not transcripts.is_file() or sha256_file(transcripts) != pin["sha256"]:
        raise StepError(
            f"the pinned word-check run {name}: its transcripts.parquet does not match the pin"
            " (wordcheck_run)"
        )
    path = folder / "run.json"
    info = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    mode = (info.get("mode") or {}).get("name")
    if info.get("step") != WORDCHECK_STEP or mode != WORDCHECK_MODE:
        raise StepError(f"{name} is not a run of the word check in its default mode")
    if info.get("status") != "done":
        raise StepError(f"{name} did not finish: its run.json does not say done")
    if (info.get("outputs") or {}).get("transcripts.parquet") != pin["sha256"]:
        raise StepError(f"{name}: its run.json lists another transcripts.parquet than the pin")
    read = (info.get("inputs") or {}).get("views") or {}
    views = {}
    for view in EMBARGO_VIEWS:
        pinned = (config.get("view_runs") or {}).get(view)
        if not isinstance(pinned, Mapping):
            raise StepError(f"no clips of {view} are pinned (view_runs)")
        views[view] = {"file": pinned["run"], "sha256": pinned["sha256"]}
        if read.get(view) != views[view]:
            raise StepError(
                f"{name} checked other clips of {view} than the pinned ones (view_runs.{view}),"
                " from which the caches are built"
            )
    verdicts = info.get("verdicts") or {}
    found = {key: (verdicts.get(key) or {}).get("verdict") for key in EMBARGO_VERDICTS}
    if found["x1_words"] != PASS:
        raise StepError(
            f"{name}: the words of x1 say {found['x1_words']}, not {PASS}; the checks of x1 are"
            " finished only after the re-check that the protocol sets, which brings its own code"
        )
    if found["x1_voice"] not in (PASS, FAIL):
        raise StepError(f"{name} holds no verdict on the voice of x1 ({found['x1_voice']})")
    if found["f4"] != PASS:
        raise StepError(
            f"{name}: check F4 says {found['f4']}, not {PASS}; rev is scored only once it has"
            " passed, and a backup or a leakage rerun brings its own code"
        )
    return {
        "run": name,
        "sha256": pin["sha256"],
        "run_json_sha256": sha256_file(path),
        "views": views,
        "verdicts": found,
    }


# The scores ------------------------------------------------------------------------------------


def score_views(
    model: Any,
    caches: Mapping[str, PinnedCache],
    prompts: Mapping[int, Mapping[str, Any]],
    answer_ids: tuple[int, int],
    width: int,
    segments: Mapping[str, Sequence[str]] | None = None,
    state: Progress | None = None,
    on_clip: Callable[[int], None] | None = None,
) -> pd.DataFrame:
    """Score segments of each view from their stored encoder outputs, one clip per call.

    View by view in the order of ``caches``, and within a view in the order of
    ``segments`` (every segment of its manifest, in the manifest's order, if None), each
    file is read and checked against its manifest row (``cache.read_file``) and scored
    (``voxtral.cached_score``) with the prompt of its number of chunks. A score that is not
    finite is kept as it is, and nothing is written: the caller reads its overflow watch and
    the scores before it writes any (pilot protocol, sections "Encoder caches and zero-shot
    scores" and "Training", bullet "Scoring").

    Args:
        model: The scoring model with the pass-through in place (``voxtral.use_cached_encoder``),
            or a PEFT model around it, its LoRA and adapter in fp16; in evaluation mode.
        caches: The pinned cache of each view to score, in the order to score them.
        prompts: The token IDs and mask of a clip of each chunk count, by chunk count.
        answer_ids: The token IDs of "Yes" and "No".
        width: The encoder's width W.
        segments: The segments to score of each view; every segment if None.
        state: The job's progress, if any: it names the segment being scored, so that a
            failure can name it; None again at the end.
        on_clip: Called after each clip with the number of clips scored so far, for the
            caller's progress lines.

    Returns:
        One row per segment and view (``zeroshot.SCORE_COLUMNS``): segment, session, view
        and z (float64).

    Raises:
        StepError: If ``segments`` does not name the views of ``caches``, or a segment is not
            in its view's manifest, before any score; or as ``cache.read_file`` and
            ``voxtral.cached_score``.
    """
    if segments is not None and set(segments) != set(caches):
        raise StepError(
            f"segments are given of {sorted(segments)}, but caches of {sorted(caches)}: one"
            " list of segments per cache"
        )
    chosen: dict[str, list[str]] = {}
    for view, cache in caches.items():
        listed = cache.manifest["segment"] if segments is None else segments[view]
        chosen[view] = [str(segment) for segment in listed]
        missing = len(set(chosen[view]) - {str(segment) for segment in cache.manifest["segment"]})
        if missing:
            raise StepError(f"{missing} segments to score are not in the cache of {view}")
    parts = []
    done = 0
    for view, cache in caches.items():
        # Any: the manifest's rows are named tuples of CACHE_COLUMNS.
        rows: dict[str, Any] = {
            str(row.segment): row for row in cache.manifest.itertuples(index=False)
        }
        scores = []
        for segment in chosen[view]:
            if state is not None:
                state.segment = segment
            row = rows[segment]
            output = read_file(Path(cache.run.folder) / str(row.file), row, width)
            scores.append(cached_score(model, prompts[int(row.chunks)], output, answer_ids))
            done += 1
            if on_clip is not None:
                on_clip(done)
        parts.append(
            pd.DataFrame(
                {
                    "segment": chosen[view],
                    "session": np.asarray(
                        [int(rows[segment].session) for segment in chosen[view]], dtype=np.int64
                    ),
                    "view": view,
                    "z": np.asarray(scores, dtype=np.float64),
                },
                columns=list(SCORE_COLUMNS),
            )
        )
    if state is not None:
        state.segment = None
    if not parts:
        return pd.DataFrame({name: [] for name in SCORE_COLUMNS})
    return pd.concat(parts, ignore_index=True)


def watch_summary(watch: LayerWatch) -> dict[str, Any]:
    """Return what a layer watch saw: its flagged parts, its largest values and its counts.

    Args:
        watch: The overflow watch (``voxtral.watch_layers``).

    Returns:
        ``parts`` (how many it watched), ``flagged``, ``largest`` (the ``LARGEST`` largest
        finite |values| as [part, value]) and ``layers`` (the counts by layer and part).
    """
    layers = watch.record()
    return {
        "parts": sum(len(parts) for parts in layers.values()),
        "flagged": watch.flagged(),
        "largest": [[name, value] for name, value in watch.largest(LARGEST)],
        "layers": layers,
    }


def label_free(z: np.ndarray) -> dict[str, Any]:
    """Return the numbers of one view's scores that need no label (no AUROC here).

    Pilot protocol, section "Encoder caches and zero-shot scores": the jobs print and record
    only these.

    Args:
        z: One view's scores, shape (N,), N of at least 1.

    Returns:
        ``count``, ``all_finite``, ``mean``, ``sd`` (the population SD), the 5th, 50th and
        95th percentiles (``p5``, ``p50``, ``p95``) and ``share_above_0`` (a z of exactly 0
        is not above 0).
    """
    return {
        "count": int(z.size),
        "all_finite": bool(np.isfinite(z).all()),
        "mean": float(z.mean()),
        "sd": float(z.std()),  # the population SD
        "p5": float(np.percentile(z, 5)),
        "p50": float(np.percentile(z, 50)),
        "p95": float(np.percentile(z, 95)),
        "share_above_0": float((z > 0).mean()),
    }
