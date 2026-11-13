"""Finding the interviewer by her fixed recordings (pilot protocol, "Interviewer detector").

The interviewer's lines are pre-recorded clips played in every session, so a sound
that recurs almost identically in other sessions is hers; a participant's speech
never recurs. This module holds the pure parts of the detector: the spectral
features, the short sound events, their fingerprints (descriptors), the search for
an event's best match in every other session, and the templates built from the
clips that recur.

Time runs in frames of 10 ms. Frame ``k`` is centred on ``k * hop`` samples, so
frame ``k`` is at ``k / 100`` seconds at 16 kHz.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

SAMPLING_RATE: Final = 16_000
SILENCE_DB: Final = -99.0  # frame energy of digital silence (10 * log10(1e-10) = -100 dB)
_EPSILON: Final = 1e-10
_CHUNK_FRAMES: Final = 20_000


@dataclass(frozen=True, slots=True)
class Spectrum:
    """Settings of the short-time spectrum.

    Attributes:
        n_fft: Frame length in samples (Hann window).
        hop: Step between frames in samples.
        low_bin: First kept FFT bin.
        high_bin: Last kept FFT bin (inclusive).
    """

    n_fft: int = 512
    hop: int = 160
    low_bin: int = 3
    high_bin: int = 128

    @property
    def bins(self) -> int:
        """Number of kept bins."""
        return self.high_bin - self.low_bin + 1

    @property
    def frames_per_second(self) -> float:
        """Frames per second at 16 kHz."""
        return SAMPLING_RATE / self.hop


def power_spectrum(audio: np.ndarray, spectrum: Spectrum) -> np.ndarray:
    """Return the power of centred Hann frames in the kept bins.

    Args:
        audio: Mono 16 kHz samples in [-1, 1], shape (samples,).
        spectrum: The spectrum settings.

    Returns:
        Power of shape (frames, bins), float32; frame ``k`` is centred on sample ``k * hop``.
    """
    half = spectrum.n_fft // 2
    padded = np.pad(np.asarray(audio, dtype=np.float32), (half, half))
    n_frames = 1 + (len(padded) - spectrum.n_fft) // spectrum.hop
    window = np.hanning(spectrum.n_fft + 1)[:-1].astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(padded, spectrum.n_fft)[:: spectrum.hop]
    out = np.empty((n_frames, spectrum.bins), dtype=np.float32)
    for first in range(0, n_frames, _CHUNK_FRAMES):
        block = np.fft.rfft(frames[first : first + _CHUNK_FRAMES] * window, axis=1)
        block = block[:, spectrum.low_bin : spectrum.high_bin + 1]
        out[first : first + len(block)] = block.real**2 + block.imag**2
    return out


def to_db(power: np.ndarray) -> np.ndarray:
    """Return power in dB.

    Args:
        power: Power of shape (frames, bins).

    Returns:
        ``10 * log10(power + 1e-10)``, float32.
    """
    return (10.0 * np.log10(power + _EPSILON)).astype(np.float32)


def energy_db(power: np.ndarray) -> np.ndarray:
    """Return each frame's energy over the kept bins, in dB.

    Args:
        power: Power of shape (frames, bins).

    Returns:
        Shape (frames,).
    """
    return (10.0 * np.log10(power.sum(axis=1, dtype=np.float64) + _EPSILON)).astype(np.float32)


def whiten(db: np.ndarray) -> np.ndarray:
    """Subtract each bin's median over the session, which removes the room and microphone.

    Args:
        db: Spectrum in dB, shape (frames, bins).

    Returns:
        The whitened spectrum W, same shape.
    """
    return (db - np.median(db, axis=0, keepdims=True)).astype(np.float32)


def spectral_shape(white: np.ndarray) -> np.ndarray:
    """Subtract each frame's mean over the bins, which removes its loudness.

    Args:
        white: Whitened spectrum W, shape (frames, bins).

    Returns:
        The spectral shape S, same shape.
    """
    return (white - white.mean(axis=1, keepdims=True)).astype(np.float32)


def fine_structure(db: np.ndarray, width: int = 5) -> np.ndarray:
    """Subtract a moving average over ``width`` bins: the spectrum's fine structure.

    Args:
        db: Spectrum in dB, shape (frames, bins).
        width: Bins in the moving average; the edge bins are repeated.

    Returns:
        Same shape, float32.
    """
    from scipy.ndimage import uniform_filter1d

    smooth = uniform_filter1d(db.astype(np.float64), size=width, axis=1, mode="nearest")
    return (db - smooth).astype(np.float32)


def noise_floor(energy: np.ndarray, percentile: float = 10.0) -> float:
    """Return the session's noise floor F: a low percentile of frame energy.

    Args:
        energy: Frame energy in dB, shape (frames,).
        percentile: Which percentile.

    Returns:
        The floor in dB; frames of digital silence are ignored.
    """
    live = energy[energy > SILENCE_DB]
    return float(np.percentile(live if len(live) else energy, percentile))


def find_events(
    energy: np.ndarray,
    floor: float,
    allowed: np.ndarray,
    rise_db: float,
    merge_frames: int,
    min_frames: int,
    max_frames: int,
) -> list[tuple[int, int]]:
    """Find short sound events: runs of loud frames inside the allowed time.

    Args:
        energy: Frame energy in dB, shape (frames,).
        floor: The session's noise floor in dB.
        allowed: Frames where events may lie, boolean of shape (frames,).
        rise_db: An event frame lies at least this far above the floor.
        merge_frames: Runs separated by fewer silent frames are joined.
        min_frames: Shortest event.
        max_frames: Longest event; longer runs are dropped.

    Returns:
        Events as (first frame, end frame), end exclusive, in time order.
    """
    loud = (energy >= floor + rise_db) & allowed
    if not loud.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], loud.astype(np.int8), [0]])))
    runs = list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))
    merged: list[list[int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] < merge_frames and allowed[merged[-1][1] : start].all():
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged if min_frames <= e - s <= max_frames]


def descriptor(
    shape: np.ndarray,
    white: np.ndarray,
    onset: int,
    frames: int,
    lead: int,
    mask_db: float,
) -> np.ndarray | None:
    """Return an event's fingerprint: its masked, double-centred spectral shape.

    Cells of the whitened spectrum below ``mask_db`` (background) are set to 0;
    the kept cells are centred by row and by column and scaled to unit length.

    Args:
        shape: Spectral shape S of the session, shape (frames, bins).
        white: Whitened spectrum W of the session, same shape.
        onset: The event's first frame.
        frames: Frames in the fingerprint.
        lead: Frames taken before the onset.
        mask_db: Cells of W below this are background.

    Returns:
        A unit vector of length ``frames * bins``, or None if the window leaves the
        session or holds fewer than two kept cells.
    """
    first = onset - lead
    if first < 0 or first + frames > len(shape):
        return None
    x = shape[first : first + frames].astype(np.float64)
    keep = white[first : first + frames] >= mask_db
    if keep.sum() < 2:
        return None
    counts_rows = np.maximum(keep.sum(axis=1, keepdims=True), 1)
    counts_cols = np.maximum(keep.sum(axis=0, keepdims=True), 1)
    masked = np.where(keep, x, 0.0)
    grand = masked.sum() / keep.sum()
    centred = masked - masked.sum(axis=1, keepdims=True) / counts_rows
    centred -= masked.sum(axis=0, keepdims=True) / counts_cols
    centred = np.where(keep, centred + grand, 0.0)
    norm = np.linalg.norm(centred)
    if norm < 1e-12:
        return None
    return (centred / norm).astype(np.float32).ravel()


def lagged_descriptors(
    shape: np.ndarray,
    white: np.ndarray,
    onset: int,
    frames: int,
    lead: int,
    mask_db: float,
    max_lag: int,
) -> np.ndarray | None:
    """Return an event's fingerprints at onsets shifted by ``-max_lag .. +max_lag`` frames.

    Args:
        shape: Spectral shape S of the session.
        white: Whitened spectrum W of the session.
        onset: The event's first frame.
        frames: Frames in a fingerprint.
        lead: Frames taken before the onset.
        mask_db: Cells of W below this are background.
        max_lag: Largest shift in frames.

    Returns:
        Shape (2 * max_lag + 1, frames * bins); a shift that leaves the session is a
        zero row. None if no shift gives a fingerprint.
    """
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        d = descriptor(shape, white, onset + lag, frames, lead, mask_db)
        rows.append(d if d is not None else np.zeros(frames * shape.shape[1], dtype=np.float32))
    stacked = np.stack(rows)
    return stacked if np.abs(stacked).sum() > 0 else None


def best_by_session(
    base: np.ndarray,
    base_session: np.ndarray,
    lagged: Sequence[np.ndarray],
    device: str = "cpu",
    block: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every event, find its best match in every other session.

    Args:
        base: Fingerprints at the true onset, shape (n, d), unit rows.
        base_session: Index of each event's session in ``lagged``, shape (n,).
        lagged: Per session, the shifted fingerprints of its events, shape (m_s, lags, d);
            float16 is accepted and computed in float32.
        device: Where the products run ("cuda" on OMNI).
        block: Events per product, to bound memory.

    Returns:
        ``best`` (n, sessions) the best cosine (-inf for the event's own session or a
        session without events), ``event`` (n, sessions) the matched event's index in
        that session (-1 if none) and ``lag`` (n, sessions) the matched shift index.
    """
    import torch

    n, sessions = len(base), len(lagged)
    best = np.full((n, sessions), -np.inf, dtype=np.float32)
    which = np.full((n, sessions), -1, dtype=np.int32)
    lag_of = np.full((n, sessions), -1, dtype=np.int16)
    if n == 0:
        return best, which, lag_of
    with torch.inference_mode():
        for s, copies in enumerate(lagged):
            if len(copies) == 0:
                continue
            m, lags, d = copies.shape
            flat = np.ascontiguousarray(copies.reshape(m * lags, d), dtype=np.float32)
            other = torch.from_numpy(flat).to(device)
            for first in range(0, n, block):
                chunk = np.ascontiguousarray(base[first : first + block], dtype=np.float32)
                part = torch.from_numpy(chunk).to(device)
                values, index = (part @ other.T).max(dim=1)
                best[first : first + len(part), s] = values.float().cpu().numpy()
                index_np = index.cpu().numpy()
                which[first : first + len(part), s] = index_np // lags
                lag_of[first : first + len(part), s] = index_np % lags
    own = base_session.astype(np.int64)
    best[np.arange(n), own] = -np.inf
    which[np.arange(n), own] = -1
    return best, which, lag_of


def duplicate_groups(hits: np.ndarray, session_of: np.ndarray, share: float) -> np.ndarray:
    """Group sessions that look like two copies of one recording, so that they count once.

    The interviewer's clips recur in many sessions, so sharing them says nothing. Two
    copies of one recording also share the participant's own sounds, which recur
    nowhere else. Two sessions are therefore grouped when more than ``share`` of one
    session's events match the other session and no third session.

    Args:
        hits: Boolean (n events, sessions): the event's best match in that session
            reaches the recurrence threshold.
        session_of: Each event's session index, shape (n,).
        share: Share of a session's events, matching only one other session, above
            which the two sessions are grouped.

    Returns:
        A group label per session, shape (sessions,).
    """
    sessions = hits.shape[1]
    parent = list(range(sessions))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    only_one = hits.sum(axis=1) == 1
    partner = np.where(only_one, hits.argmax(axis=1), -1)
    for s in range(sessions):
        mine = session_of == s
        total = int(mine.sum())
        if total == 0:
            continue
        counts = np.bincount(partner[mine & only_one], minlength=sessions)
        for t in np.flatnonzero(counts > share * total):
            if t != s:
                parent[find(int(t))] = find(s)
    return np.array([find(s) for s in range(sessions)])


def recurring_sessions(
    best: np.ndarray, tau: float, groups: np.ndarray, session_of: np.ndarray
) -> np.ndarray:
    """Count, for each event, the distinct session groups other than its own where it recurs.

    Args:
        best: Best cosine per event and session, shape (n, sessions).
        tau: Recurrence threshold.
        groups: Group label per session, from ``duplicate_groups``.
        session_of: Each event's session index, shape (n,).

    Returns:
        Shape (n,), integer counts.
    """
    hit = best >= tau
    counts = np.zeros(len(best), dtype=np.int32)
    own = groups[session_of]
    for g in np.unique(groups):
        counts += hit[:, groups == g].any(axis=1) & (own != g)
    return counts


@dataclass(frozen=True, slots=True)
class Template:
    """One of the interviewer's clips, as a spectral template.

    Attributes:
        values: Level-normalised whitened spectrum W of the clip, shape (frames, bins).
        active: Cells at least ``active_db`` above the template's median, same shape.
        sessions: Number of distinct sessions in which the clip was found.
        members: Number of events averaged into it.
    """

    values: np.ndarray
    active: np.ndarray
    sessions: int
    members: int

    @property
    def frames(self) -> int:
        """Length in frames."""
        return int(self.values.shape[0])


def build_template(
    windows: Sequence[np.ndarray],
    sessions: int,
    active_db: float,
    min_active: int,
) -> Template | None:
    """Average aligned copies of a clip into a template.

    Args:
        windows: Whitened spectra W of the copies, aligned and of equal shape (frames, bins).
        sessions: Number of distinct sessions the copies come from.
        active_db: Cells at least this far above the template's median are active.
        min_active: Fewest active cells for a usable template.

    Returns:
        The template, or None if it has too few active cells.
    """
    stack = np.stack([w - np.median(w) for w in windows]).astype(np.float32)
    values = np.median(stack, axis=0)
    active = values >= np.median(values) + active_db
    if int(active.sum()) < min_active:
        return None
    return Template(values=values, active=active, sessions=sessions, members=len(windows))


def as_arrays(templates: Sequence[Template]) -> dict[str, Any]:
    """Pack templates for ``numpy.savez``: one flat array per field plus lengths.

    Args:
        templates: The library.

    Returns:
        Arrays ``values``, ``active``, ``lengths``, ``sessions`` and ``members``.
    """
    if not templates:
        return {
            "values": np.zeros((0, 0), np.float32),
            "active": np.zeros((0, 0), bool),
            "lengths": np.zeros(0, np.int32),
            "sessions": np.zeros(0, np.int32),
            "members": np.zeros(0, np.int32),
        }
    return {
        "values": np.concatenate([t.values for t in templates]),
        "active": np.concatenate([t.active for t in templates]),
        "lengths": np.array([t.frames for t in templates], dtype=np.int32),
        "sessions": np.array([t.sessions for t in templates], dtype=np.int32),
        "members": np.array([t.members for t in templates], dtype=np.int32),
    }


def from_arrays(arrays: Any) -> list[Template]:
    """Unpack templates packed by ``as_arrays``.

    Args:
        arrays: A mapping with the arrays of ``as_arrays`` (for example a loaded .npz).

    Returns:
        The library.
    """
    templates, first = [], 0
    for length, sessions, members in zip(
        arrays["lengths"], arrays["sessions"], arrays["members"], strict=True
    ):
        end = first + int(length)
        templates.append(
            Template(
                values=arrays["values"][first:end],
                active=arrays["active"][first:end].astype(bool),
                sessions=int(sessions),
                members=int(members),
            )
        )
        first = end
    return templates


def pitch_shift(
    window: np.ndarray, semitones: float, spectrum: Spectrum, fill: float
) -> np.ndarray:
    """Scale a spectrum window's frequency axis by ``2 ** (semitones / 12)``: a decoy.

    The time course stays the same. Bins whose source frequency falls outside the
    kept band get ``fill``.

    Args:
        window: Spectrum values of shape (frames, bins), in dB or a dB-like scale.
        semitones: Shift; +3 moves every component up by 3 semitones.
        spectrum: The spectrum settings, for the bins' frequencies.
        fill: Value for bins with no source.

    Returns:
        The shifted window, same shape, float32.
    """
    ratio = 2.0 ** (semitones / 12.0)
    bins = np.arange(spectrum.low_bin, spectrum.high_bin + 1, dtype=np.float64)
    source = bins / ratio - spectrum.low_bin  # fractional index of each target bin's source
    inside = (source >= 0) & (source <= spectrum.bins - 1)
    out = np.full(window.shape, fill, dtype=np.float32)
    lower = np.floor(source[inside]).astype(np.int64)
    upper = np.minimum(lower + 1, spectrum.bins - 1)
    weight = (source[inside] - lower).astype(np.float32)
    out[:, inside] = (1 - weight) * window[:, lower] + weight * window[:, upper]
    return out


@dataclass(frozen=True, slots=True)
class Cluster:
    """Events that are copies of one clip.

    Attributes:
        leader: Global index of the event the cluster started from.
        members: Global indices of all its events, the leader first.
        shifts: For each member, the frames to add to its onset to line it up with
            the leader's onset (0 for the leader).
    """

    leader: int
    members: tuple[int, ...]
    shifts: tuple[int, ...]


def leader_clusters(
    best: np.ndarray,
    which: np.ndarray,
    lag: np.ndarray,
    first_event: np.ndarray,
    session_of: np.ndarray,
    groups: np.ndarray,
    tau: float,
    min_sessions: int,
    max_lag: int,
    allowed_sessions: np.ndarray | None = None,
) -> list[Cluster]:
    """Group recurring events into clips, greedily, starting with the most recurring event.

    A leader takes, in every other session where its best match reaches ``tau``,
    that session's matching event, if no earlier cluster took it. A cluster is kept
    when its events come from at least ``min_sessions`` distinct session groups.

    Args:
        best: Best cosine per event and session, shape (n, sessions).
        which: Index of the matching event within that session, shape (n, sessions).
        lag: Index of the matching shift (0 .. 2 * max_lag), shape (n, sessions).
        first_event: Global index of each session's first event, shape (sessions,).
        session_of: Each event's session index, shape (n,).
        groups: Group label per session, from ``duplicate_groups``.
        tau: Recurrence threshold.
        min_sessions: Fewest distinct session groups in a kept cluster.
        max_lag: Largest shift in frames.
        allowed_sessions: If given, only events and matches in these sessions count
            (boolean per session), for libraries built from part of the sessions.

    Returns:
        The kept clusters, in the order they were formed.
    """
    n, sessions = best.shape
    usable = np.ones(sessions, dtype=bool) if allowed_sessions is None else allowed_sessions
    hits = (best >= tau) & usable[None, :]
    counts = np.zeros(n, dtype=np.int32)
    own = groups[session_of]
    for g in np.unique(groups[usable]):
        counts += hits[:, (groups == g) & usable].any(axis=1) & (own != g)
    order = np.lexsort((-np.where(np.isfinite(best), best, -1).max(axis=1), -counts))
    taken = np.zeros(n, dtype=bool)
    clusters: list[Cluster] = []
    for leader in order:
        if taken[leader] or counts[leader] < 1 or not usable[session_of[leader]]:
            continue
        members, shifts = [int(leader)], [0]
        for s in np.flatnonzero(hits[leader]):
            other = int(first_event[s] + which[leader, s])
            if not taken[other]:
                members.append(other)
                # The member's fingerprint at onset + (lag - max_lag) matched the leader's onset.
                shifts.append(int(lag[leader, s]) - max_lag)
        if len({int(groups[session_of[m]]) for m in members}) >= min_sessions:
            taken[members] = True
            clusters.append(Cluster(int(leader), tuple(members), tuple(shifts)))
    return clusters
