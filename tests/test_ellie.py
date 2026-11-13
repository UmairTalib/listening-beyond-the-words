import numpy as np
import pytest

from listening_beyond_words import ellie as el

RATE = 16_000
SPEC = el.Spectrum()
FRAMES, LEAD, MASK_DB, MAX_LAG = 40, 5, 4.0, 8


def clip(f0_start: float, f0_end: float, seconds: float = 0.6) -> np.ndarray:
    """A synthetic voiced sound: 7 harmonics of a gliding pitch, a 15 ms attack, a 60 ms release."""
    t = np.arange(int(seconds * RATE)) / RATE
    phase = 2 * np.pi * np.cumsum(np.linspace(f0_start, f0_end, len(t))) / RATE
    wave = sum(np.sin(k * phase) / k for k in range(1, 8))
    envelope = np.minimum(1.0, np.minimum(t / 0.015, (seconds - t) / 0.06))
    return (0.3 * wave * envelope).astype(np.float32)


def session(
    inserts: list[tuple[float, np.ndarray, float]],
    tilt: float = 0.0,
    seed: int = 9001,
    seconds: float = 6.0,
) -> np.ndarray:
    """Background noise with clips added at (start s, wave, gain dB), then a channel tilt."""
    rng = np.random.default_rng(seed)
    audio = rng.normal(0.0, 10 ** (-50 / 20), int(seconds * RATE))
    for start, wave, gain_db in inserts:
        first = int(start * RATE)
        audio[first : first + len(wave)] += wave * 10 ** (gain_db / 20)
    if tilt:
        audio = np.concatenate([audio[:1], audio[1:] - tilt * audio[:-1]])
    return audio.astype(np.float32)


def features(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    power = el.power_spectrum(audio, SPEC)
    white = el.whiten(el.to_db(power))
    return el.energy_db(power), white, el.spectral_shape(white)


def first_event(audio: np.ndarray) -> int:
    energy, _, _ = features(audio)
    allowed = np.ones(len(energy), dtype=bool)
    events = el.find_events(energy, el.noise_floor(energy), allowed, 6.0, 15, 25, 400)
    assert len(events) == 1
    return events[0][0]


def test_a_tone_lands_in_its_bin_and_frames_are_10_ms_apart() -> None:
    t = np.arange(RATE) / RATE
    power = el.power_spectrum(np.sin(2 * np.pi * 1000.0 * t).astype(np.float32), SPEC)
    assert power.shape == (1 + RATE // SPEC.hop, SPEC.bins)
    assert int(np.argmax(power[50])) == round(1000.0 / (RATE / SPEC.n_fft)) - SPEC.low_bin


def test_whitening_removes_the_channel_and_shape_removes_the_loudness() -> None:
    rng = np.random.default_rng(9001)
    db = rng.normal(-40.0, 5.0, (300, SPEC.bins)).astype(np.float32)
    channel = np.linspace(-6.0, 6.0, SPEC.bins, dtype=np.float32)
    np.testing.assert_allclose(el.whiten(db + channel), el.whiten(db), atol=1e-4)
    white = el.whiten(db)
    np.testing.assert_allclose(el.spectral_shape(white + 9.0), el.spectral_shape(white), atol=1e-4)


def test_events_merge_short_gaps_and_keep_their_length_and_place() -> None:
    energy = np.full(200, -60.0, dtype=np.float32)
    energy[10:20] = energy[24:40] = -40.0  # gap of 4 frames: merged
    energy[60:65] = -40.0  # too short
    energy[100:150] = -40.0  # split by a frame that is not allowed
    allowed = np.ones(200, dtype=bool)
    allowed[120] = False
    events = el.find_events(energy, -60.0, allowed, 6.0, 5, 10, 40)
    assert events == [(10, 40), (100, 120), (121, 150)]
    assert el.find_events(energy, -60.0, allowed, 6.0, 5, 10, 25) == [(100, 120)]


def test_fine_structure_ignores_a_smooth_tilt() -> None:
    db = np.tile(np.linspace(-50.0, -20.0, SPEC.bins, dtype=np.float32), (4, 1))
    np.testing.assert_allclose(el.fine_structure(db)[:, 2:-2], 0.0, atol=1e-4)


def test_the_same_clip_in_other_sessions_matches_and_another_clip_does_not() -> None:
    a, b = clip(150.0, 230.0), clip(260.0, 170.0)
    sessions = [
        session([(2.00, a, 0.0)], tilt=0.0, seed=9001),
        session([(3.03, a, -10.0)], tilt=0.5, seed=9002),
        session([(1.47, a, -12.0)], tilt=0.9, seed=9003),
        session([(2.50, b, 0.0)], tilt=0.3, seed=9004),
    ]
    base, lagged = [], []
    for audio in sessions:
        _, white, shape = features(audio)
        onset = first_event(audio)
        d = el.descriptor(shape, white, onset, FRAMES, LEAD, MASK_DB)
        copies = el.lagged_descriptors(shape, white, onset, FRAMES, LEAD, MASK_DB, MAX_LAG)
        assert d is not None
        assert copies is not None
        base.append(d)
        lagged.append(copies[None])
    best, event, lag = el.best_by_session(np.stack(base), np.arange(4), lagged)
    assert np.all(np.isneginf(np.diag(best)))
    same = [best[0, 1], best[0, 2], best[1, 2], best[1, 0], best[2, 0], best[2, 1]]
    assert min(same) > 0.6
    assert max(best[0:3, 3].max(), best[3, :3].max()) < 0.3
    assert event[0, 1] == 0
    assert 0 <= lag[0, 1] <= 2 * MAX_LAG


def test_a_fingerprint_needs_room_in_the_session() -> None:
    _, white, shape = features(session([(0.2, clip(150.0, 230.0), 0.0)]))
    assert el.descriptor(shape, white, 2, FRAMES, LEAD, MASK_DB) is None
    assert el.descriptor(shape, white, len(shape) - 10, FRAMES, LEAD, MASK_DB) is None


def test_only_sessions_that_share_their_own_sounds_count_once() -> None:
    # Sessions 0 and 1 are copies: their sounds match each other and nowhere else.
    # Events 6-8 are interviewer clips that match every session, which says nothing.
    session_of = np.array([0, 0, 0, 1, 1, 2, 0, 1, 2])
    hits = np.zeros((9, 3), dtype=bool)
    hits[0:3, 1] = True
    hits[3:5, 0] = True
    hits[6] = [False, True, True]
    hits[7] = [True, False, True]
    hits[8] = [True, True, False]
    groups = el.duplicate_groups(hits, session_of, 0.2)
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]
    best = np.where(hits, 0.9, 0.1).astype(np.float32)
    counts = el.recurring_sessions(best, 0.5, groups, session_of)
    # The copies' own sounds recur only inside their group; the clips recur in the other group.
    assert counts.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1]


def test_interviewer_clips_shared_by_every_session_merge_no_sessions() -> None:
    session_of = np.repeat(np.arange(4), 3)
    hits = np.ones((12, 4), dtype=bool)
    hits[np.arange(12), session_of] = False
    assert len(set(el.duplicate_groups(hits, session_of, 0.2).tolist())) == 4


def test_templates_round_trip_and_need_active_cells() -> None:
    rng = np.random.default_rng(9001)
    windows = [rng.normal(0.0, 1.0, (30, SPEC.bins)).astype(np.float32) for _ in range(3)]
    windows[0][5:15, 10:20] += 20.0
    windows[1][5:15, 10:20] += 20.0
    windows[2][5:15, 10:20] += 20.0
    template = el.build_template(windows, sessions=3, active_db=8.0, min_active=20)
    assert template is not None
    assert template.active[5:15, 10:20].all()
    assert template.active.sum() == pytest.approx(100, abs=5)
    back = el.from_arrays(el.as_arrays([template, template]))
    assert len(back) == 2
    np.testing.assert_array_equal(back[1].values, template.values)
    assert el.build_template(windows[:1], sessions=1, active_db=50.0, min_active=20) is None


def test_a_pitch_decoy_moves_every_component_up_by_three_semitones() -> None:
    t = np.arange(RATE) / RATE
    db = el.to_db(el.power_spectrum(np.sin(2 * np.pi * 1000.0 * t).astype(np.float32), SPEC))
    shifted = el.pitch_shift(db, 3.0, SPEC, fill=-100.0)
    peak = int(np.argmax(db[50])) + SPEC.low_bin
    moved = int(np.argmax(shifted[50])) + SPEC.low_bin
    assert moved == round(peak * 2 ** (3 / 12))
    np.testing.assert_allclose(el.pitch_shift(db, 0.0, SPEC, fill=-100.0), db, atol=1e-4)
    assert np.all(el.pitch_shift(db, -3.0, SPEC, fill=-100.0)[:, -10:] == -100.0)


def test_clusters_take_each_sessions_best_match_once() -> None:
    # Events: session 0 has events 0, 1; session 1 has 2; session 2 has 3; session 3 has 4, 5.
    session_of = np.array([0, 0, 1, 2, 3, 3])
    first_event = np.array([0, 2, 3, 4])
    best = np.full((6, 4), 0.1, dtype=np.float32)
    which = np.zeros((6, 4), dtype=np.int32)
    lag = np.full((6, 4), 8, dtype=np.int16)
    best[0, 1], best[0, 2], best[0, 3] = 0.9, 0.9, 0.9  # event 0 recurs in sessions 1, 2, 3
    which[0, 3], lag[0, 3] = 1, 10  # its match in session 3 is that session's second event
    best[1, 2] = 0.9  # event 1 recurs only in session 2, whose event is then taken
    best[np.arange(6), session_of] = -np.inf
    groups = np.arange(4)
    clusters = el.leader_clusters(best, which, lag, first_event, session_of, groups, 0.5, 3, 8)
    assert len(clusters) == 1
    assert clusters[0].leader == 0
    assert clusters[0].members == (0, 2, 3, 5)
    assert clusters[0].shifts == (0, 0, 0, 2)
    only_two = np.array([True, True, False, False])
    assert (
        el.leader_clusters(best, which, lag, first_event, session_of, groups, 0.5, 3, 8, only_two)
        == []
    )


def direct_ncc(x: np.ndarray, valid: np.ndarray, t: el.Template, start: int) -> float:
    cells = t.active & (valid[start : start + t.frames] > 0)[:, None]
    a, b = t.values[cells].astype(np.float64), x[start : start + t.frames][cells].astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def test_the_scan_equals_a_direct_correlation_over_the_valid_active_cells() -> None:
    rng = np.random.default_rng(9001)
    x = rng.normal(0.0, 1.0, (120, 16)).astype(np.float32)
    valid = (rng.random(120) > 0.2).astype(np.float32)
    templates = [
        el.Template(
            rng.normal(0, 1, (length, 16)).astype(np.float32), rng.random((length, 16)) > 0.5, 3, 3
        )
        for length in (7, 12)
    ]
    ncc, count = el.ncc_scan(x, valid, templates)
    for i, t in enumerate(templates):
        for start in (0, 17, 120 - t.frames):
            assert ncc[i, start] == pytest.approx(direct_ncc(x, valid, t, start), abs=1e-4)
            cells = t.active & (valid[start : start + t.frames] > 0)[:, None]
            assert count[i, start] == cells.sum()
        assert np.isnan(ncc[i, 120 - t.frames + 1 :]).all()


def reverb(audio: np.ndarray, seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    tail = rng.normal(0.0, 1.0, n) * np.exp(-np.arange(n) / (0.05 * RATE))
    response = np.concatenate([[1.0], 0.3 * tail / np.abs(tail).sum()])
    return np.convolve(audio, response)[: len(audio)].astype(np.float32)


def test_a_planted_clip_is_found_where_it_is_and_its_decoys_and_noise_are_not() -> None:
    a = clip(150.0, 230.0, seconds=0.8)
    copies = []
    for k, (gain, tilt) in enumerate([(0.0, 0.0), (-6.0, 0.5), (-10.0, 0.8)]):
        audio = session([(1.0, a, gain)], tilt=tilt, seed=9100 + k)
        _, white, _ = features(audio)
        onset = first_event(audio)
        copies.append(white[onset - 5 : onset + 85])
    template = el.build_template(copies, sessions=3, active_db=8.0, min_active=20)
    assert template is not None
    target = reverb(session([(2.37, a, -12.0)], tilt=0.3, seed=9200), 0.3, seed=9201)
    _, white, shape = features(target)
    valid = np.ones(len(shape), dtype=np.float32)
    real = [template, el.decoy(template, 3.0, SPEC), el.decoy(template, -3.0, SPEC)]
    ncc, _ = el.ncc_scan(shape, valid, real)
    true_start = round(2.37 * 100) - 5
    assert abs(int(np.nanargmax(ncc[0])) - true_start) <= 2
    assert np.nanmax(ncc[0]) > 0.6
    assert np.nanmax(ncc[1:, true_start - 3 : true_start + 4]) < 0.5 * np.nanmax(ncc[0])
    t = np.arange(6 * RATE) / RATE
    tone = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    noise = np.random.default_rng(9300).normal(0.0, 0.05, 6 * RATE).astype(np.float32)
    for other in (tone, noise):
        _, _, other_shape = features(other)
        other_ncc, _ = el.ncc_scan(other_shape, np.ones(len(other_shape), np.float32), [template])
        assert np.nanmax(other_ncc) < 0.3


def test_peaks_are_local_maxima_and_zones_merge_when_close() -> None:
    scores = np.array([0, 5, 9, 5, 0, 0, 7, 7, 0, 0, 0, 0, 12, 0], dtype=float)
    assert el.local_peaks(scores, 2, 6.0).tolist() == [2, 6, 12]
    assert el.local_peaks(scores, 5, 6.0).tolist() == [2, 12]
    assert el.merge_zones([(50, 60), (10, 20), (22, 30)], 5) == [(10, 30), (50, 60)]
    assert el.merge_zones([(10, 20), (26, 30)], 5) == [(10, 20), (26, 30)]
