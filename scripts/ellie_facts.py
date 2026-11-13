"""Facts and template libraries of the interviewer detector (GPU job).

Submit from an OMNI login node:
    scripts/slurm/submit.sh ellie_facts

Pilot protocol, section "Interviewer detector", gates P1 and P2. The interviewer's
lines are fixed recordings, so a sound that recurs almost identically in other
sessions is hers. For every pilot session this job
  A. aligns the lines that hold one of her scripted phrases and fingerprints each
     phrase, and fingerprints common words said by participants; the same phrase in
     two sessions must stand out from the same word said by two participants (P1),
     which also sets the recurrence threshold tau_rep;
  B. finds short sound events in the gaps between lines and in the pauses inside
     kept lines, fingerprints them and finds each event's best match in every other
     session;
  C. groups the recurring events into clips and averages each clip into a template,
     for all sessions and for even and odd session IDs (P2).

Writes $LBW_WORKSPACE/derived/ellie_detector/facts-<job id>/ with library_all.npz,
library_even.npz, library_odd.npz, events.parquet, pairs.parquet and run.json.
Prints aggregate numbers only: never text, IDs or one session's values.
"""

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from sklearn.metrics import roc_auc_score

from listening_beyond_words import ellie as el
from listening_beyond_words.alignment import WordAligner
from listening_beyond_words.models import load_pins, local_snapshot
from listening_beyond_words.runinfo import check_sha256, git_state, utc_now, write_json
from listening_beyond_words.sessions import PILOT_IDS, session_files
from listening_beyond_words.settings import get_workspace
from listening_beyond_words.transcripts import INTERVIEWER_PHRASES, normalise, read_transcript

CONFIG = Path("configs/pilot/data.yaml")
ALIGN_PAD_SECONDS = 0.2
PAIRS_PER_PHRASE = 200
VARIANTS = ("shape", "fine")


@dataclass
class Features:
    energy: np.ndarray
    white: np.ndarray
    shape: np.ndarray
    fine: np.ndarray
    floor: float

    def x(self, variant: str) -> np.ndarray:
        return self.shape if variant == "shape" else self.fine


def features(audio: np.ndarray, spectrum: el.Spectrum) -> Features:
    power = el.power_spectrum(audio, spectrum)
    db = el.to_db(power)
    white = el.whiten(db)
    energy = el.energy_db(power)
    return Features(
        energy, white, el.spectral_shape(white), el.fine_structure(db), el.noise_floor(energy)
    )


def frame(seconds: float) -> int:
    return round(seconds * 100)


def span_mask(n: int, spans: list[tuple[float, float]], grow: float = 0.0) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for start, end in spans:
        mask[max(0, frame(start - grow)) : max(0, min(n, frame(end + grow)))] = True
    return mask


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def phrase_positions(words: list[str]) -> list[tuple[str, int]]:
    joined = " " + " ".join(words) + " "
    found = []
    for phrase in INTERVIEWER_PHRASES:
        start = 0
        while (i := joined.find(" " + phrase + " ", start)) >= 0:
            found.append((phrase, joined[: i + 1].count(" ") - 1))
            start = i + 1
    return found


def decoy_descriptor(
    f: Features,
    variant: str,
    onset: int,
    d: dict[str, Any],
    semitones: float,
    spectrum: el.Spectrum,
) -> np.ndarray | None:
    first = onset - d["onset_lead_frames"]
    if first < 0 or first + d["frames"] > len(f.white):
        return None
    window = slice(first, first + d["frames"])
    white = el.pitch_shift(f.white[window], semitones, spectrum, fill=-30.0)
    x = el.pitch_shift(f.x(variant)[window], semitones, spectrum, fill=0.0)
    return el.descriptor(
        x, white, d["onset_lead_frames"], d["frames"], d["onset_lead_frames"], d["mask_db"]
    )


def quantiles(values: np.ndarray, qs: tuple[float, ...] = (0.1, 0.5, 0.9)) -> str:
    if len(values) == 0:
        return "none"
    return ", ".join(f"{q:.0%} {v:.3f}" for q, v in zip(qs, np.quantile(values, qs), strict=True))


def main() -> None:
    os.umask(0o077)
    workspace = get_workspace()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg = config["ellie_detector"]
    spectrum = el.Spectrum(**cfg["spectrum"])
    dsc, ev, disc, rec, tpl = (
        cfg["descriptor"],
        cfg["events"],
        cfg["discovery"],
        cfg["recurrence"],
        cfg["template"],
    )
    lines_path = workspace.resolve(config["participant_lines"]["run"]) / "lines.parquet"
    words_path = workspace.resolve(config["word_times"]["run"]) / "words.parquet"
    check_sha256(lines_path, config["participant_lines"]["sha256"])
    check_sha256(words_path, config["word_times"]["sha256"])
    pin = load_pins(Path("configs/pilot/models.yaml"))["wav2vec2_960h"]
    info: dict[str, Any] = {
        "step": "ellie_facts",
        "status": "running",
        "started_utc": utc_now(),
        "git": git_state(),
        "inputs": {
            "participant_lines": config["participant_lines"],
            "word_times": config["word_times"],
            "aligner": {"repo_id": pin.repo_id, "revision": pin.revision},
        },
        "settings": cfg,
        "slurm": {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname()},
    }
    run_dir = (
        workspace.derived / "ellie_detector" / f"facts-{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "run.json", info)
    rng = np.random.default_rng(cfg["seed"])
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        lines = pd.read_parquet(lines_path, columns=["session", "line", "start", "end", "drop"])
        words = pd.read_parquet(
            words_path, columns=["session", "line", "word", "start", "end", "score"]
        )
        aligner = WordAligner(local_snapshot(pin, Path(os.environ["HF_HOME"])))
        sessions_dir = workspace.edaic / "sessions"
        n_sessions = len(PILOT_IDS)

        def load(session_id: int) -> np.ndarray:
            audio_path, _ = session_files(sessions_dir, session_id)
            audio, rate = sf.read(audio_path, dtype="float32")
            if rate != el.SAMPLING_RATE or audio.ndim != 1:
                raise ValueError(f"session {session_id}: expected 16 kHz mono audio")
            return audio

        def masks(
            session_id: int, n: int, transcript: pd.DataFrame, phrase_line: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            spans = list(
                zip(transcript["start"].astype(float), transcript["end"].astype(float), strict=True)
            )
            phrase_spans = [s for s, p in zip(spans, phrase_line, strict=True) if p]
            gap = ~span_mask(n, spans, disc["line_guard_ms"] / 1000)
            gap &= ~span_mask(n, phrase_spans, disc["phrase_guard_ms"] / 1000)
            kept = lines[(lines["session"] == session_id) & ~lines["drop"]]
            kept = kept[~phrase_line[kept["line"].to_numpy()]]
            inside = span_mask(n, list(zip(kept["start"], kept["end"], strict=True)))
            w = words[words["session"] == session_id]
            spoken = span_mask(
                n, list(zip(w["start"], w["end"], strict=True)), disc["word_guard_ms"] / 1000
            )
            pause = inside & ~spoken
            for a, b in runs(pause):
                if b - a < disc["min_pause_ms"] / 10:
                    pause[a:b] = False
            return gap, pause

        # ---------------- Pass A: phrases, common words, levels ----------------
        phrase_rows: list[dict[str, Any]] = []
        word_rows: list[dict[str, Any]] = []
        phrase_desc = {v: [] for v in VARIANTS}  # base fingerprints of phrase occurrences
        phrase_lag = {v: [] for v in VARIANTS}
        phrase_decoy = {v: [] for v in VARIANTS}
        word_desc = {v: [] for v in VARIANTS}
        word_lag = {v: [] for v in VARIANTS}
        level_rows, align_failures = [], 0
        null_words = set(cfg["null_words"])
        nw = cfg["null_word"]
        for s_index, session_id in enumerate(PILOT_IDS):
            audio = load(session_id)
            f = features(audio, spectrum)
            _, tpath = session_files(sessions_dir, session_id)
            transcript = read_transcript(tpath)
            texts = [normalise(str(t)).split() for t in transcript["text"]]
            # Levels: the participant's median word level L_P.
            w = words[words["session"] == session_id]
            interior = span_mask(
                len(f.energy),
                [
                    (a + 0.02, b - 0.02)
                    for a, b, sc in zip(w["start"], w["end"], w["score"], strict=True)
                    if sc >= 0.2 and b - a > 0.04
                ],
            )
            level_p = float(np.median(f.energy[interior])) if interior.any() else float("nan")
            level_rows.append({"session": s_index, "level_p": level_p, "floor": f.floor})
            # Phrase occurrences, aligned.
            for line_index, tokens in enumerate(texts):
                found = phrase_positions(tokens)
                if not found:
                    continue
                start = max(0.0, float(transcript["start"].iloc[line_index]) - ALIGN_PAD_SECONDS)
                end = min(
                    len(audio) / el.SAMPLING_RATE,
                    float(transcript["end"].iloc[line_index]) + ALIGN_PAD_SECONDS,
                )
                placed = aligner.align(
                    audio[int(start * el.SAMPLING_RATE) : int(end * el.SAMPLING_RATE)], tokens
                )
                if placed is None:
                    align_failures += 1
                    continue
                for phrase, position in found:
                    onset = frame(start + placed[position].start)
                    descs = {}
                    for v in VARIANTS:
                        base = el.descriptor(
                            f.x(v),
                            f.white,
                            onset,
                            dsc["frames"],
                            dsc["onset_lead_frames"],
                            dsc["mask_db"],
                        )
                        lagged = el.lagged_descriptors(
                            f.x(v),
                            f.white,
                            onset,
                            dsc["frames"],
                            dsc["onset_lead_frames"],
                            dsc["mask_db"],
                            dsc["max_lag_frames"],
                        )
                        decoys = [
                            decoy_descriptor(f, v, onset, dsc, st, spectrum)
                            for st in (cfg["decoy_semitones"], -cfg["decoy_semitones"])
                        ]
                        descs[v] = (base, lagged, decoys)
                    if any(
                        b is None or lg is None or any(x is None for x in dc)
                        for b, lg, dc in descs.values()
                    ):
                        continue
                    phrase_rows.append({"session": s_index, "phrase": phrase})
                    for v in VARIANTS:
                        base, lagged, decoys = descs[v]
                        phrase_desc[v].append(base)
                        phrase_lag[v].append(lagged.astype(np.float16))
                        phrase_decoy[v].append(np.stack(decoys))
            # Common words said by the participant: the null.
            per_line = w.groupby("line")["word"].transform("size")
            pick = w[
                (per_line >= nw["min_line_words"])
                & w["word"].isin(null_words)
                & (w["score"] >= nw["min_score"])
            ]
            for word, a, b in zip(pick["word"], pick["start"], pick["end"], strict=True):
                span = slice(frame(a + 0.02), max(frame(a + 0.02) + 1, frame(b - 0.02)))
                if np.median(f.energy[span]) < level_p - nw["level_below_db"]:
                    continue
                onset = frame(a)
                got = {}
                for v in VARIANTS:
                    base = el.descriptor(
                        f.x(v),
                        f.white,
                        onset,
                        dsc["frames"],
                        dsc["onset_lead_frames"],
                        dsc["mask_db"],
                    )
                    lagged = el.lagged_descriptors(
                        f.x(v),
                        f.white,
                        onset,
                        dsc["frames"],
                        dsc["onset_lead_frames"],
                        dsc["mask_db"],
                        dsc["max_lag_frames"],
                    )
                    got[v] = (base, lagged)
                if any(b_ is None or lg is None for b_, lg in got.values()):
                    continue
                word_rows.append({"session": s_index, "word": word})
                for v in VARIANTS:
                    word_desc[v].append(got[v][0])
                    word_lag[v].append(got[v][1].astype(np.float16))
        phrases = pd.DataFrame(phrase_rows)
        null = pd.DataFrame(word_rows)
        levels = pd.DataFrame(level_rows)

        # ---------------- P1: the premise ----------------
        def pair_sims(a_idx: np.ndarray, b_idx: np.ndarray, base: list, lagged: list) -> np.ndarray:
            """Best cosine over the shifts of b, for each pair (a, b), in chunks."""
            out = np.zeros(len(a_idx), dtype=np.float32)
            for first in range(0, len(a_idx), 500):
                ia, ib = a_idx[first : first + 500], b_idx[first : first + 500]
                b = np.stack([lagged[j].astype(np.float32) for j in ib])  # (pairs, lags, d)
                a = np.stack([base[i] for i in ia])[:, None, :]
                out[first : first + len(ia)] = (a * b).sum(axis=2).max(axis=1)
            return out

        same_a, same_b = [], []
        for _, group in phrases.groupby("phrase"):
            idx = group.index.to_numpy()
            pairs = [
                (i, j)
                for i in idx
                for j in idx
                if i < j and phrases.at[i, "session"] != phrases.at[j, "session"]
            ]
            if len(pairs) > PAIRS_PER_PHRASE:
                pairs = [pairs[k] for k in rng.choice(len(pairs), PAIRS_PER_PHRASE, replace=False)]
            same_a += [p[0] for p in pairs]
            same_b += [p[1] for p in pairs]
        same_a_np, same_b_np = np.array(same_a, dtype=int), np.array(same_b, dtype=int)
        diff_a, diff_b = [], []
        while len(diff_a) < len(same_a) and len(phrases) > 1:
            i, j = rng.choice(len(phrases), 2, replace=False)
            if (
                phrases.at[i, "phrase"] != phrases.at[j, "phrase"]
                and phrases.at[i, "session"] != phrases.at[j, "session"]
            ):
                diff_a.append(i)
                diff_b.append(j)
        null_a, null_b = [], []
        by_word = {w_: g.index.to_numpy() for w_, g in null.groupby("word")}
        weights = np.array([len(v) for v in by_word.values()], dtype=float)
        keys = list(by_word)
        tries = 0
        while len(null_a) < cfg["null_pairs"] and tries < 20 * cfg["null_pairs"]:
            tries += 1
            members = by_word[keys[rng.choice(len(keys), p=weights / weights.sum())]]
            i, j = (
                rng.choice(members, 2, replace=False)
                if len(members) > 1
                else (members[0], members[0])
            )
            if null.at[i, "session"] != null.at[j, "session"]:
                null_a.append(i)
                null_b.append(j)
        p1: dict[str, Any] = {
            "phrase_occurrences": len(phrases),
            "align_failures": align_failures,
            "same_pairs": len(same_a),
            "null_pairs": len(null_a),
            "variants": {},
        }
        pair_rows = []
        for v in VARIANTS:
            same = pair_sims(same_a_np, same_b_np, phrase_desc[v], phrase_lag[v])
            diff = pair_sims(
                np.array(diff_a, int), np.array(diff_b, int), phrase_desc[v], phrase_lag[v]
            )
            nul = pair_sims(np.array(null_a, int), np.array(null_b, int), word_desc[v], word_lag[v])
            decoy = (
                np.concatenate(
                    [
                        pair_sims(
                            same_a_np, same_b_np, [d[k] for d in phrase_decoy[v]], phrase_lag[v]
                        )
                        for k in (0, 1)
                    ]
                )
                if len(same_a)
                else np.zeros(0)
            )
            tau = float(np.quantile(nul, rec["quantile"])) if len(nul) else float("nan")
            auroc = (
                float(
                    roc_auc_score(np.r_[np.ones(len(same)), np.zeros(len(nul))], np.r_[same, nul])
                )
                if len(same) and len(nul)
                else float("nan")
            )
            p1["variants"][v] = {
                "auroc": auroc,
                "share_above": float(np.mean(same > tau)) if len(same) else float("nan"),
                "tau": tau,
                "same_median": float(np.median(same)) if len(same) else float("nan"),
                "different_median": float(np.median(diff)) if len(diff) else float("nan"),
                "decoy_median": float(np.median(decoy)) if len(decoy) else float("nan"),
                "null_median": float(np.median(nul)) if len(nul) else float("nan"),
            }
            pair_rows += [
                {"variant": v, "kind": k, "similarity": float(x)}
                for k, arr in (("same", same), ("different", diff), ("null", nul), ("decoy", decoy))
                for x in arr
            ]
        pd.DataFrame(pair_rows).to_parquet(run_dir / "pairs.parquet", index=False)
        a_, b_ = p1["variants"]["shape"], p1["variants"]["fine"]
        variant = "fine" if b_["auroc"] >= a_["auroc"] + 0.01 else "shape"
        chosen = p1["variants"][variant]
        tau_rep = chosen["tau"]
        gate = cfg["gates"]["p1"]
        p1.update(
            variant=variant,
            tau_rep=tau_rep,
            passed=bool(
                chosen["auroc"] >= gate["min_auroc"]
                and chosen["share_above"] >= gate["min_share_above"]
            ),
        )
        print(
            f"phrase occurrences {len(phrases)} in {phrases['session'].nunique() if len(phrases) else 0} sessions"
            f" (alignment failures {align_failures}); common-word null occurrences {len(null)}"
        )
        print(
            f"pairs: same phrase {len(same_a)}, different phrase {len(diff_a)}, null {len(null_a)}"
        )
        for v in VARIANTS:
            r = p1["variants"][v]
            print(
                f"  {v:<5}: AUROC {r['auroc']:.3f}; same-phrase pairs above the null's 99.9% ({r['tau']:.3f}): {r['share_above']:.3f};"
                f" medians same {r['same_median']:.3f}, different {r['different_median']:.3f}, decoy {r['decoy_median']:.3f}, null {r['null_median']:.3f}"
            )
        print(
            f"P1 premise: variant {variant}, tau_rep {tau_rep:.3f}: {'PASSED' if p1['passed'] else 'FAILED'}"
        )
        info["p1"] = p1
        write_json(run_dir / "run.json", info)

        # ---------------- Pass B, discovery and pass C, in up to two rounds ----------------
        def discover(rise_db: float) -> dict[str, Any]:
            event_rows: list[dict[str, Any]] = []
            base: list[np.ndarray] = []
            lagged: list[np.ndarray] = []
            for s_index, session_id in enumerate(PILOT_IDS):
                f = features(load(session_id), spectrum)
                _, tpath = session_files(sessions_dir, session_id)
                transcript = read_transcript(tpath)
                phrase_line = np.array(
                    [bool(phrase_positions(normalise(str(t)).split())) for t in transcript["text"]]
                )
                gap, pause = masks(session_id, len(f.energy), transcript, phrase_line)
                found = el.find_events(
                    f.energy,
                    f.floor,
                    gap | pause,
                    rise_db,
                    ev["merge_ms"] // 10,
                    ev["min_ms"] // 10,
                    ev["max_ms"] // 10,
                )
                session_lagged = []
                for a, b in found:
                    d0 = el.descriptor(
                        f.x(variant),
                        f.white,
                        a,
                        dsc["frames"],
                        dsc["onset_lead_frames"],
                        dsc["mask_db"],
                    )
                    dl = el.lagged_descriptors(
                        f.x(variant),
                        f.white,
                        a,
                        dsc["frames"],
                        dsc["onset_lead_frames"],
                        dsc["mask_db"],
                        dsc["max_lag_frames"],
                    )
                    if d0 is None or dl is None:
                        continue
                    event_rows.append(
                        {
                            "session": s_index,
                            "start": a,
                            "end": b,
                            "kind": "gap" if gap[a:b].mean() >= 0.5 else "pause",
                        }
                    )
                    base.append(d0)
                    session_lagged.append(dl.astype(np.float16))
                dim = dsc["frames"] * spectrum.bins
                lagged.append(
                    np.stack(session_lagged)
                    if session_lagged
                    else np.zeros((0, 2 * dsc["max_lag_frames"] + 1, dim), np.float16)
                )
            events = pd.DataFrame(event_rows)
            session_of = events["session"].to_numpy()
            best, which, lag = el.best_by_session(np.stack(base), session_of, lagged, device=device)
            hits = best >= tau_rep
            groups = el.duplicate_groups(hits, session_of, rec["duplicate_share"])
            counts = el.recurring_sessions(best, tau_rep, groups, session_of)
            first_event = np.searchsorted(session_of, np.arange(n_sessions))
            parity = np.array([sid % 2 == 0 for sid in PILOT_IDS])
            clusters = {
                name: el.leader_clusters(
                    best,
                    which,
                    lag,
                    first_event,
                    session_of,
                    groups,
                    tau_rep,
                    rec["cluster_min_sessions"],
                    dsc["max_lag_frames"],
                    allowed,
                )
                for name, allowed in (("all", None), ("even", parity), ("odd", ~parity))
            }
            return {
                "events": events,
                "best": best,
                "groups": groups,
                "counts": counts,
                "clusters": clusters,
            }

        def templates_of(found: dict[str, Any]) -> dict[str, list[el.Template]]:
            events, groups = found["events"], found["groups"]
            pad = tpl["pad_ms"] // 10
            plans = {}
            needed: dict[int, list[tuple[str, int, int, int, int]]] = {}
            for name, clusters in found["clusters"].items():
                for c_index, c in enumerate(clusters):
                    points = [
                        int(events.at[m, "start"]) + sh
                        for m, sh in zip(c.members, c.shifts, strict=True)
                    ]
                    rel_end = [
                        int(events.at[m, "end"]) - p for m, p in zip(c.members, points, strict=True)
                    ]
                    rel_start = [-sh for sh in c.shifts]
                    first, last = int(np.median(rel_start)) - pad, int(np.median(rel_end)) + pad
                    plans[(name, c_index)] = (first, last)
                    for k, (m, p) in enumerate(zip(c.members, points, strict=True)):
                        needed.setdefault(int(events.at[m, "session"]), []).append(
                            (name, c_index, k, p + first, p + last)
                        )
            windows: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
            for s_index, asks in needed.items():
                f = features(load(PILOT_IDS[s_index]), spectrum)
                for name, c_index, k, a, b in asks:
                    if a >= 0 and b <= len(f.white):
                        windows[(name, c_index, k)] = (f.white[a:b], f.x(variant)[a:b])
            out: dict[str, list[el.Template]] = {}
            for name, clusters in found["clusters"].items():
                out[name] = []
                for c_index, c in enumerate(clusters):
                    lead = windows.get((name, c_index, 0))
                    if lead is None:
                        continue
                    length = lead[0].shape[0]
                    ref = el.descriptor(lead[1], lead[0], 0, length, 0, dsc["mask_db"])
                    keep_w, keep_groups = [], set()
                    for k, m in enumerate(c.members):
                        got = windows.get((name, c_index, k))
                        if got is None or ref is None:
                            continue
                        d = el.descriptor(got[1], got[0], 0, length, 0, dsc["mask_db"])
                        if k == 0 or (d is not None and float(d @ ref) >= tau_rep):
                            keep_w.append(got[0])
                            keep_groups.add(int(groups[int(events.at[m, "session"])]))
                    if len(keep_groups) >= rec["cluster_min_sessions"]:
                        t = el.build_template(
                            keep_w, len(keep_groups), tpl["active_db"], tpl["min_active_cells"]
                        )
                        if t is not None:
                            out[name].append(t)
            return out

        rounds = []
        rise = ev["rise_db"]
        while True:
            found = discover(rise)
            libraries = templates_of(found)
            events, counts = found["events"], found["counts"]
            recurring = counts >= rec["min_other_sessions"]
            gap_time = (events["end"] - events["start"])[events["kind"] == "gap"].sum()
            members_all = {m for c in found["clusters"]["all"] for m in c.members}
            explained = (events["end"] - events["start"])[
                (events["kind"] == "gap") & events.index.isin(list(members_all))
            ].sum()
            per_session = (
                events.groupby("session").size().reindex(range(n_sessions), fill_value=0).to_numpy()
            )
            lengths = np.array([t.frames for t in libraries["all"]]) / 100
            summary = {
                "rise_db": rise,
                "events": len(events),
                "events_per_session": [float(x) for x in np.quantile(per_session, [0.1, 0.5, 0.9])],
                "gap_events": int((events["kind"] == "gap").sum()),
                "recurring_share": float(recurring.mean()) if len(events) else 0.0,
                "session_groups": len(np.unique(found["groups"])),
                "clusters": {k: len(v) for k, v in found["clusters"].items()},
                "templates": {k: len(v) for k, v in libraries.items()},
                "templates_by_length_s": {
                    lab: int(((lengths > lo) & (lengths <= hi)).sum())
                    for lab, lo, hi in (
                        ("<=0.5", 0, 0.5),
                        ("<=1", 0.5, 1),
                        ("<=2", 1, 2),
                        ("<=4.3", 2, 4.3),
                    )
                },
                "gap_event_time_explained": float(explained / gap_time) if gap_time else 0.0,
            }
            rounds.append(summary)
            print(
                f"discovery at rise {rise} dB: events {len(events)} (gap {summary['gap_events']}),"
                f" per session 10/50/90% {summary['events_per_session']}; recurring share {summary['recurring_share']:.3f};"
                f" distinct session groups {summary['session_groups']} of {n_sessions}"
            )
            print(
                f"  clusters {summary['clusters']}; templates {summary['templates']}; by length {summary['templates_by_length_s']};"
                f" gap-event time explained by the library {summary['gap_event_time_explained']:.3f}"
            )
            if len(libraries["all"]) >= tpl["min_templates"] or rise == ev["rerun_rise_db"]:
                break
            rise = ev["rerun_rise_db"]
        p2 = {"passed": len(libraries["all"]) >= tpl["min_templates"], "rounds": rounds}
        print(
            f"P2 library: {len(libraries['all'])} templates (need {tpl['min_templates']}): {'PASSED' if p2['passed'] else 'FAILED'}"
        )
        for name, lib in libraries.items():
            np.savez_compressed(run_dir / f"library_{name}.npz", **el.as_arrays(lib))
        events.assign(recurring_sessions=counts).to_parquet(run_dir / "events.parquet", index=False)
        gaps_db = (levels["level_p"] - levels["floor"]).to_numpy()
        print(
            f"participant word level above the noise floor (dB): {quantiles(gaps_db[np.isfinite(gaps_db)])}"
        )
        info.update(
            status="done",
            finished_utc=utc_now(),
            p2=p2,
            tau_rep=tau_rep,
            variant=variant,
            levels={
                "level_p_minus_floor": [float(x) for x in np.nanquantile(gaps_db, [0.1, 0.5, 0.9])]
            },
        )
    except BaseException:
        info.update(status="failed", finished_utc=utc_now())
        raise
    finally:
        write_json(run_dir / "run.json", info)


if __name__ == "__main__":
    main()
