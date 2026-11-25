"""The frozen encoder's output, stored once per segment and clip version.

Voxtral's audio encoder is frozen (pilot protocol, section "Encoder caches and zero-shot
scores"), so ``scripts/build_caches.py`` runs it once on every clip of a clip version
and stores its output before the adapter: every vector of every 30 s chunk, the padding
included, shape (C, 1500, W) in fp16, where C is the clip's number of chunks
(``voxtral.chunks``) and W the encoder's width (1,280). Each output is one file in
NumPy's ``.npy`` format, written without pickle (``to_bytes``). Its bytes depend only on
the values, so the same output gives the same file, and the same SHA-256, in every
process. A cache run lists its files in ``manifest.parquet`` (``CACHE_COLUMNS``) with
that SHA-256, the clip's SHA-256 and length, and the SHA-256 of the clip's log-mel
features (``features_sha256``). A check that computes an output again from the audio
(``build_caches.py --verify``, on the segments of ``check_sample``) can then tell a
difference in the features, which the CPU builds, from one in the encoder, which runs on
the GPU.

``read_cache_run`` reads a finished cache run once it fits the current pins and
settings; ``pinned_cache`` reads the pinned one (``cache_runs`` in
``configs/pilot/data.yaml``) once its check found every sampled output equal, with what
the check's overflow watch found in the language model (``PinnedCache.lm_flagged``), and
``load_cache_manifest`` gives its manifest; ``read_file`` reads one file back once it
matches its manifest row. Their refusals are ``StepError``s that name rules, runs, counts
and at most one segment.

torch is imported inside the functions that use it.
"""

import contextlib
import hashlib
import io
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from listening_beyond_words.clips import load_view_manifest
from listening_beyond_words.models import ModelPin
from listening_beyond_words.runinfo import StepError, item_seed, sha256_file
from listening_beyond_words.settings import Workspace
from listening_beyond_words.voxtral import FRAMES_PER_CHUNK, chunks

if TYPE_CHECKING:
    import torch

# One row per stored output, in manifest.parquet. ``file`` is relative to the run folder;
# ``samples`` is the clip's length, so that a clip's own frames can be told from padding;
# ``max_abs`` is the output's largest |value|.
CACHE_COLUMNS: Final = (
    "segment",
    "session",
    "file",
    "chunks",
    "samples",
    "sha256",
    "audio_sha256",
    "features_sha256",
    "max_abs",
)
STEP: Final = "build_caches"  # the step that writes cache runs and their checks
VIEW_MODE: Final = "view"  # build_caches.py --view: a cache run
VERIFY_MODE: Final = "verify"  # build_caches.py --verify: a check of cache runs
SAMPLE_STREAM: Final = "cache"  # the check's draw: item_seed(seed, SAMPLE_STREAM, view)
SUFFIX: Final = ".npy"
# What voxtral.set_numerics sets, as it records them; the thread count comes from the
# config (encoder_cache.cpu_threads).
NUMERICS: Final[Mapping[str, bool]] = MappingProxyType(
    {
        "deterministic_algorithms": True,
        "deterministic_warn_only": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "fp16_reduced_precision_reduction": False,
        "fp16_accumulation": False,
        "matmul_tf32": False,
        "cudnn_tf32": False,
    }
)
_SETTINGS_KEYS: Final = ("cpu_threads", "verify")
_VERIFY_KEYS: Final = ("segments_per_view", "seed", "lm_watch_clips")
_FP16: Final = np.dtype("<f2")


@dataclass(frozen=True, slots=True)
class CacheSettings:
    """The settings of the caches and of their check (``encoder_cache`` in ``data.yaml``).

    Attributes:
        cpu_threads: PyTorch's CPU threads in every job of this step
            (``voxtral.set_numerics``); they build the log-mel features.
        segments_per_view: Segments that the check draws from each cache run, besides
            every segment of two chunks.
        seed: The seed of that draw (``check_sample``).
        lm_watch_clips: The longest clips of each cache run that the check sends through
            the language model without its output layer, with the overflow watch on.
    """

    cpu_threads: int
    segments_per_view: int
    seed: int
    lm_watch_clips: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CacheSettings":
        """Read the settings and check every key and value.

        Args:
            config: The data config, with the ``encoder_cache`` block.

        Returns:
            The settings.

        Raises:
            StepError: If the block or its ``verify`` block lacks a key or has an unknown
                one, or a value is not a whole number of at least 1.
        """
        block = _keys(config.get("encoder_cache"), _SETTINGS_KEYS, "encoder_cache")
        verify = _keys(block["verify"], _VERIFY_KEYS, "encoder_cache.verify")
        return cls(
            cpu_threads=_count(block["cpu_threads"], "encoder_cache.cpu_threads"),
            segments_per_view=_count(
                verify["segments_per_view"], "encoder_cache.verify.segments_per_view"
            ),
            seed=_count(verify["seed"], "encoder_cache.verify.seed"),
            lm_watch_clips=_count(verify["lm_watch_clips"], "encoder_cache.verify.lm_watch_clips"),
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
        raise StepError(f"{name} must be a whole number of at least 1")
    return value


# Files ---------------------------------------------------------------------------------


def features_sha256(features: "torch.Tensor") -> str:
    """Return the SHA-256 of a clip's log-mel features: their float32 bytes in C order.

    Args:
        features: The features of one clip (``input_features`` of
            ``voxtral.clip_inputs``), shape (C, 128, 3000), float32.

    Returns:
        The hex digest.

    Raises:
        ValueError: If the features are not float32.
    """
    import torch

    if features.dtype != torch.float32:
        raise ValueError(f"log-mel features are float32, not {features.dtype}")
    array = features.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.astype("<f4", copy=False).tobytes()).hexdigest()


def to_bytes(output: "torch.Tensor") -> bytes:
    """Write one encoder output as the bytes of a ``.npy`` file, without pickle.

    The bytes depend only on the output's values and shape, so the same output gives the
    same bytes, and the same SHA-256, in every process.

    Args:
        output: The encoder output of one clip (``voxtral.encoder_output``), shape
            (C, 1500, W), fp16, on any device.

    Returns:
        The file's bytes.

    Raises:
        ValueError: If the output is not fp16 or not of shape (C, 1500, W) with C of at
            least 1.
    """
    import torch

    if output.dtype != torch.float16:
        raise ValueError(f"an encoder output is stored in fp16, not {output.dtype}")
    shape = tuple(output.shape)
    if len(shape) != 3 or shape[0] < 1 or shape[1] != FRAMES_PER_CHUNK:
        raise ValueError(f"an encoder output has shape (C, {FRAMES_PER_CHUNK}, W), not {shape}")
    buffer = io.BytesIO()
    np.save(buffer, output.detach().cpu().contiguous().numpy(), allow_pickle=False)
    return buffer.getvalue()


def from_bytes(data: bytes, width: int) -> "torch.Tensor":
    """Read one encoder output from the bytes of its ``.npy`` file, without pickle.

    Args:
        data: The file's bytes (``to_bytes``).
        width: The encoder's width W.

    Returns:
        The output, shape (C, 1500, W), fp16, on the CPU.

    Raises:
        ValueError: If the bytes are not exactly one ``.npy`` array in fp16 and C order of
            shape (C, 1500, W) with C of at least 1.
    """
    import torch

    stream = io.BytesIO(data)
    array = np.lib.format.read_array(stream, allow_pickle=False)
    if stream.tell() != len(data):
        raise ValueError(f"{len(data) - stream.tell()} bytes follow the array")
    if array.dtype != _FP16:
        raise ValueError(f"the array is {array.dtype.str}, not fp16 ({_FP16.str})")
    shape = tuple(array.shape)
    if len(shape) != 3 or shape[0] < 1 or shape[1:] != (FRAMES_PER_CHUNK, width):
        raise ValueError(f"the array has shape {shape}, not (C, {FRAMES_PER_CHUNK}, {width})")
    if not array.flags.c_contiguous:
        raise ValueError("the array is not in C order")
    return torch.from_numpy(array)


def write_file(path: Path, data: bytes) -> None:
    """Write a file whole: to a temporary name next to it, flushed to disk, then renamed.

    A job that dies while writing leaves no file under the final name, and no
    temporary file after an error that Python can raise: the partial temporary file that
    this call made is the one file it removes (engineering conventions, section 7).

    Args:
        path: The new file; its folder must exist.
        data: Its bytes.

    Raises:
        FileExistsError: If the file exists already: a cache file is never overwritten.
    """
    if path.exists():
        raise FileExistsError(f"{path.name} exists already; a cache file is never overwritten")
    temporary = path.with_name(path.name + ".tmp")
    created = False
    try:
        with temporary.open("xb") as handle:
            created = True
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.rename(path)
    finally:
        if created:  # never a temporary file that this call did not make
            temporary.unlink(missing_ok=True)


def read_file(path: Path, row: Any, width: int) -> "torch.Tensor":
    """Read one stored encoder output, after checking it against its manifest row.

    Args:
        path: The file, the row's ``file`` in its run folder.
        row: Its manifest row (``CACHE_COLUMNS``), such as from
            ``manifest.itertuples(index=False)``; ``segment``, ``sha256`` and ``chunks``
            are read.
        width: The encoder's width W.

    Returns:
        The output, shape (chunks, 1500, W), fp16, on the CPU, every value finite.

    Raises:
        StepError: If the file is missing, does not match the row's SHA-256, is not an
            fp16 output of that shape, or holds a value that is not finite. The message
            names the segment and the check, never a value.
    """
    import torch

    segment = str(row.segment)

    def refused(what: str) -> StepError:
        return StepError(f"segment {segment}: {what}", segment=segment)

    if not path.is_file():
        raise refused("its cache file is missing")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != row.sha256:
        raise refused("its cache file does not match the SHA-256 of its manifest row")
    output = None
    reason = ""
    try:
        output = from_bytes(data, width)
    except ValueError as error:  # its messages name the format and the shape only
        reason = str(error)
    if output is None:  # raised here, so that nothing chains to the reader's error
        raise refused(f"its cache file is not an encoder output of width {width} ({reason})")
    if output.shape[0] != int(row.chunks):
        raise refused(
            f"its cache file holds {output.shape[0]} chunks, its manifest row {int(row.chunks)}"
        )
    if not bool(torch.isfinite(output).all()):
        raise refused("its cache file holds a value that is not finite")
    return output


# The check's sample --------------------------------------------------------------------


def check_sample(manifest: pd.DataFrame, view: str, size: int, seed: int) -> tuple[str, ...]:
    """Return the segments whose outputs the check computes again: a draw and the long clips.

    ``size`` segments, or all of them if there are fewer, are drawn without replacement
    from the segments in ascending ID, by ``numpy.random.default_rng(item_seed(seed,
    "cache", view))``, so each view has a stream of its own; every segment of more than
    one chunk is added.

    Args:
        manifest: A cache run's manifest (``CACHE_COLUMNS``); ``segment`` and ``chunks``
            are read.
        view: The clip version.
        size: Segments to draw (``encoder_cache.verify.segments_per_view``).
        seed: The seed (``encoder_cache.verify.seed``).

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


# Cache runs ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CacheRun:
    """A finished cache run of one clip version, checked against the current pins and settings.

    Attributes:
        run: The run folder relative to the workspace, such as ``cache/x0/run-12345``.
        folder: The run folder.
        view: The clip version it caches.
        info: Its ``run.json``.
        manifest: Its manifest (``CACHE_COLUMNS``), in the order of its rows.
        manifest_sha256: The SHA-256 of its ``manifest.parquet``.
        clips: The manifest of the pinned clip version it was built from
            (``clips.MANIFEST_COLUMNS``).
    """

    run: str
    folder: Path
    view: str
    info: dict[str, Any]
    manifest: pd.DataFrame
    manifest_sha256: str
    clips: pd.DataFrame


def relative_run(workspace: Workspace, run: str | Path) -> str:
    """Return a run folder as a workspace-relative path in one fixed spelling.

    Args:
        workspace: The workspace.
        run: The folder, relative to the workspace, such as ``cache/x0/run-12345/``.

    Returns:
        The folder relative to the workspace root, such as ``cache/x0/run-12345``.

    Raises:
        StepError: If the path is absolute or leaves the workspace.
    """
    folder = None
    with contextlib.suppress(ValueError):  # its message names the path, as the one below
        folder = workspace.resolve(run)
    if folder is None or folder == workspace.root:
        raise StepError(f"{run} is not a folder inside the workspace")
    return folder.relative_to(workspace.root).as_posix()


def read_cache_run(
    workspace: Workspace, config: Mapping[str, Any], run: str, voxtral: ModelPin
) -> CacheRun:
    """Read a finished cache run, after checking that it fits the current pins and settings.

    Its ``run.json`` must say ``done`` and come from ``build_caches.py --view`` on a view
    of ``view_runs``, built from that view's pinned clips, with the Voxtral of ``voxtral``,
    the current ``encoder_cache`` block and the numeric settings that
    ``voxtral.set_numerics`` sets with ``encoder_cache.cpu_threads`` threads (``NUMERICS``).
    Its manifest must match the SHA-256 that ``run.json`` lists and hold one row per clip
    of the pinned view, with the clip's session, length and ``audio_sha256``, and the
    number of chunks of that length.

    Args:
        workspace: The workspace.
        config: The data config, with ``view_runs`` and ``encoder_cache``.
        run: The run folder, relative to the workspace.
        voxtral: The Voxtral pin of ``configs/pilot/models.yaml``.

    Returns:
        The run, its record and its manifest.

    Raises:
        StepError: If any of this does not hold.
    """
    name = relative_run(workspace, run)
    folder = workspace.root / name
    path = folder / "run.json"
    if not path.is_file():
        raise StepError(f"no cache run in {name}: it has no run.json")
    info = json.loads(path.read_text(encoding="utf-8"))
    mode = info.get("mode") or {}
    if info.get("step") != STEP or mode.get("name") != VIEW_MODE:
        raise StepError(f"{name} is not a run of build_caches.py --view")
    if info.get("status") != "done":
        raise StepError(f"{name} did not finish: its run.json does not say done")
    view = str(mode.get("view"))
    pins = config["view_runs"]
    if view not in pins:
        raise StepError(f"{name} caches {view}, a view without a pin in view_runs")
    inputs = info.get("inputs") or {}
    if inputs.get("view") != {"run": pins[view]["run"], "sha256": pins[view]["sha256"]}:
        raise StepError(f"{name} was not built from the pinned clips of {view} (view_runs)")
    model = inputs.get("voxtral") or {}
    if (model.get("repo_id"), model.get("revision")) != (voxtral.repo_id, voxtral.revision):
        raise StepError(f"{name} was built with another Voxtral than models.yaml pins")
    settings = info.get("settings") or {}
    if settings.get("encoder_cache") != json.loads(json.dumps(config["encoder_cache"])):
        raise StepError(f"{name} was built with other encoder_cache settings")
    threads = CacheSettings.from_config(config).cpu_threads
    expected = {**NUMERICS, "torch_threads": threads}
    numerics = settings.get("numerics")
    if not isinstance(numerics, dict) or any(numerics.get(k) != v for k, v in expected.items()):
        raise StepError(
            f"{name} was built with other numeric settings than voxtral.set_numerics sets"
            f" with {threads} threads"
        )
    manifest_path = folder / "manifest.parquet"
    listed = (info.get("outputs") or {}).get("manifest.parquet")
    if not listed or not manifest_path.is_file() or sha256_file(manifest_path) != listed:
        raise StepError(f"{name}: its manifest.parquet does not match its run.json")
    manifest = pd.read_parquet(manifest_path)
    if tuple(manifest.columns) != CACHE_COLUMNS:
        raise StepError(f"{name}: its manifest does not have the columns of CACHE_COLUMNS")
    clips = pinned_clips(workspace, config, view)
    _same_clips(name, view, manifest, clips)
    return CacheRun(name, folder, view, info, manifest, listed, clips)


def pinned_clips(workspace: Workspace, config: Mapping[str, Any], view: str) -> pd.DataFrame:
    """The manifest of a pinned clip version, as ``clips.load_view_manifest`` reads it.

    Args:
        workspace: The workspace.
        config: The data config, with the ``view_runs`` pins.
        view: The clip version, such as ``x0``.

    Returns:
        One row per clip (``clips.MANIFEST_COLUMNS``).

    Raises:
        StepError: If the manifest is missing or does not match its pin.
    """
    pin = config["view_runs"][view]
    path = workspace.resolve(pin["run"]) / "manifest.parquet"
    if not path.is_file() or sha256_file(path) != pin["sha256"]:
        raise StepError(f"the manifest of the pinned clips of {view} does not match its pin")
    return load_view_manifest(workspace, dict(config), view)


def _clip_values(table: pd.DataFrame) -> dict[str, tuple[int, int, str]]:
    """Session, length and ``audio_sha256`` of each clip of a manifest, by segment."""
    columns = (table[name].tolist() for name in ("segment", "session", "samples", "audio_sha256"))
    return {
        str(segment): (int(session), int(samples), str(audio))
        for segment, session, samples, audio in zip(*columns, strict=True)
    }


def _same_clips(name: str, view: str, manifest: pd.DataFrame, clips: pd.DataFrame) -> None:
    """Stop unless a cache manifest holds one row per clip of its view, with the clip's values."""
    rows = len(manifest)
    mine = _clip_values(manifest)
    if len(mine) != rows:
        raise StepError(f"{name}: its manifest holds {rows - len(mine)} segments twice")
    theirs = _clip_values(clips)
    missing = len(set(theirs) - set(mine))
    extra = len(set(mine) - set(theirs))
    if missing or extra:
        raise StepError(
            f"{name}: its manifest lacks {missing} segments of the clips of {view} and holds"
            f" {extra} others"
        )
    differ = sum(values != theirs[segment] for segment, values in mine.items())
    if differ:
        raise StepError(
            f"{name}: {differ} rows differ from the clips of {view} in session, length or"
            " audio_sha256"
        )
    counts = zip(manifest["chunks"].tolist(), manifest["samples"].tolist(), strict=True)
    wrong = sum(int(count) != chunks(int(samples)) for count, samples in counts)
    if wrong:
        raise StepError(f"{name}: {wrong} rows give another number of chunks than their length")


@dataclass(frozen=True, slots=True)
class PinnedCache:
    """The pinned cache run of one clip version, read together with its check.

    Attributes:
        view: The clip version.
        pin: Its pin (``cache_runs.<view>``): ``run``, ``sha256`` and ``verified_by``.
        run: The cache run, checked (``read_cache_run``); ``file`` in its manifest is
            relative to ``run.folder``.
        manifest: Its manifest (``CACHE_COLUMNS``), sorted by segment.
        check: The check run that verified it, relative to the workspace.
        lm_flagged: The parts of the language model that the check's overflow watch
            flagged on this run's longest clips, as ``<layer>/<part>``; empty if none. A
            flag leaves the cache standing, but no score may come from the language model
            until the pilot protocol's fallback (the flagged layers in fp32) is in place:
            the zero-shot job and training read this.
    """

    view: str
    pin: dict[str, Any]
    run: CacheRun
    manifest: pd.DataFrame
    check: str
    lm_flagged: tuple[str, ...]


def load_cache_manifest(
    workspace: Workspace, config: Mapping[str, Any], view: str, voxtral: ModelPin
) -> pd.DataFrame:
    """The manifest of the pinned cache run of a view (``cache_runs``), checked with its check.

    The checks are those of ``pinned_cache``.

    Args:
        workspace: The workspace.
        config: The data config, with ``cache_runs``, ``view_runs`` and ``encoder_cache``.
        view: The clip version, such as ``x0``.
        voxtral: The Voxtral pin of ``configs/pilot/models.yaml``.

    Returns:
        One row per clip (``CACHE_COLUMNS``), sorted by segment. ``file`` is relative to
        the pinned run's folder.

    Raises:
        StepError: If any check of ``pinned_cache`` fails.
    """
    return pinned_cache(workspace, config, view, voxtral).manifest


def pinned_cache(
    workspace: Workspace, config: Mapping[str, Any], view: str, voxtral: ModelPin
) -> PinnedCache:
    """The pinned cache run of a view (``cache_runs``), checked with its check.

    The pin names the run, the SHA-256 of its manifest and the check run that verified it
    (``verified_by``). The manifest must match the pin, and the run must pass
    ``read_cache_run``. The check run must say ``done``, come from
    ``build_caches.py --verify``, name this run and its manifest's SHA-256, have run with
    the current ``encoder_cache`` block and the cache run's own numeric settings, have
    found the log-mel features and the encoder output of every segment of its sample
    (``check_sample``) equal, and record its watch of the language model on this run.

    Args:
        workspace: The workspace.
        config: The data config, with ``cache_runs``, ``view_runs`` and ``encoder_cache``.
        view: The clip version, such as ``x0``.
        voxtral: The Voxtral pin of ``configs/pilot/models.yaml``.

    Returns:
        The run, its manifest sorted by segment, its check and the language-model parts
        that the check flagged.

    Raises:
        StepError: If any of this does not hold.
    """
    pins = config.get("cache_runs") or {}
    if view not in pins:
        raise StepError(f"no cache run of {view} is pinned (cache_runs)")
    pin = pins[view]
    if not isinstance(pin, Mapping) or set(pin) != {"run", "sha256", "verified_by"}:
        raise StepError(f"cache_runs.{view} needs exactly run, sha256 and verified_by")
    name = relative_run(workspace, pin["run"])
    path = workspace.root / name / "manifest.parquet"
    if not path.is_file() or sha256_file(path) != pin["sha256"]:
        raise StepError(f"the pinned cache run of {view}: its manifest.parquet does not match")
    cache = read_cache_run(workspace, config, name, voxtral)
    if cache.view != view:
        raise StepError(f"the cache run pinned for {view} caches {cache.view}")
    check = relative_run(workspace, pin["verified_by"])
    record = workspace.root / check / "run.json"
    if not record.is_file():
        raise StepError(f"{check}, the check of the pinned cache run of {view}, has no run.json")
    info = json.loads(record.read_text(encoding="utf-8"))
    mode = info.get("mode") or {}
    if info.get("step") != STEP or mode.get("name") != VERIFY_MODE:
        raise StepError(f"{check} is not a run of build_caches.py --verify")
    if info.get("status") != "done":
        raise StepError(f"{check} did not finish: its run.json does not say done")
    if name not in (mode.get("runs") or []):
        raise StepError(f"{check} did not check {name}")
    result = ((info.get("results") or {}).get("runs") or {}).get(name) or {}
    if result.get("manifest_sha256") != pin["sha256"]:
        raise StepError(f"{check} checked another manifest of {name}")
    settings = info.get("settings") or {}
    if settings.get("encoder_cache") != json.loads(json.dumps(config["encoder_cache"])):
        raise StepError(f"{check} ran with other encoder_cache settings")
    if settings.get("numerics") != cache.info["settings"]["numerics"]:
        raise StepError(f"{check} ran with other numeric settings than {name}")
    settings_now = CacheSettings.from_config(config)
    expected = len(
        check_sample(cache.manifest, view, settings_now.segments_per_view, settings_now.seed)
    )
    found = (result.get("segments"), result.get("features_equal"), result.get("outputs_equal"))
    if found != (expected, expected, expected):
        raise StepError(
            f"{check} found the features of {found[1]} and the encoder outputs of {found[2]}"
            f" of {found[0]} segments equal, not those of all {expected} of its sample"
        )
    watched = result.get("lm_watch")
    flagged = watched.get("flagged") if isinstance(watched, Mapping) else None
    if not isinstance(flagged, list) or not all(isinstance(part, str) for part in flagged):
        raise StepError(f"{check} holds no record of its language-model watch on {name}")
    return PinnedCache(
        view=view,
        pin=json.loads(json.dumps(dict(pin))),
        run=cache,
        manifest=cache.manifest.sort_values("segment", ignore_index=True),
        check=check,
        lm_flagged=tuple(flagged),
    )
