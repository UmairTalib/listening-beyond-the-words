"""Scan every pilot session for the interviewer's clips and check gates D1-D4 (GPU job).

Submit from an OMNI login node:
    scripts/slurm/submit.sh detect_ellie

Pilot protocol, section "Interviewer detector". Reads the template libraries of the
pinned facts run and the settings in configs/pilot/data.yaml (ellie_detector).
  1. Decoys (every template shifted by +-3 semitones) are scanned over all sessions;
     their mean and SD over the kept lines turn each template's score into a z-score.
  2. The clearest hits in the gaps between lines give the interviewer's level in each
     session; quieter frames than that (plus a margin) are where score B looks.
  3. Every session is scanned with every template (score A: all cells; score B: the
     quiet frames only). Candidates with z >= 3 are kept.
  4. The threshold is the smallest one in the grid at which the decoys' zones would drop
     at most 0.5% of the kept words on average and 0.75% per decoy family (gate D2).
  5. Zones are the hits' full template extents, merged when closer than 0.3 s.
  6. Gates D1 (known interviewer lines), D3 (planted copies) and D4 (cost), and facts:
     where the zones lie, what they would remove, and by class and gender.

Writes $LBW_WORKSPACE/derived/ellie_detector/run-<job id>/ with candidates.parquet,
zones.parquet, injections.parquet and run.json. Prints aggregate numbers only:
never text, IDs or one session's values.
"""

import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from scipy.stats import mannwhitneyu

from listening_beyond_words import ellie as el
from listening_beyond_words.runinfo import check_sha256, git_state, sha256_file, utc_now, write_json
from listening_beyond_words.sessions import PILOT_IDS, session_files
from listening_beyond_words.settings import get_workspace
from listening_beyond_words.transcripts import INTERVIEWER_PHRASES, normalise, read_transcript
from listening_beyond_words.views import part_windows

CONFIG = Path("configs/pilot/data.yaml")
BATCH = 64  # templates per scan
LOCAL_SECONDS = 3.0  # audio kept on each side of a planted copy
PLACEMENTS = ("pause", "edge", "partial", "full")


@dataclass
class Context:
    """One session's features and the masks the detector needs."""

    session: int  # index in PILOT_IDS
    audio: np.ndarray
    median: np.ndarray  # per-bin median of the dB spectrum, for whitening local windows
    energy: np.ndarray
    white: np.ndarray
    shape: np.ndarray
    floor: float
    level_p: float
    kept: np.ndarray  # frames inside kept lines
    gap: np.ndarray  # frames outside every transcript line
    lines: pd.DataFrame  # all transcript lines: start, end, text, phrase share
    words: pd.DataFrame  # aligned words of kept lines


def frame(seconds: float) -> int:
    return round(seconds * 100)


def span_mask(n: int, spans: Any, grow: float = 0.0) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for start, end in spans:
        mask[max(0, frame(start - grow)) : max(0, min(n, frame(end + grow)))] = True
    return mask


def phrase_share(words: list[str]) -> float:
    """Share of a line's words that belong to one of the interviewer's phrases."""
    if not words:
        return 0.0
    joined = " " + " ".join(words) + " "
    covered = np.zeros(len(words), dtype=bool)
    for phrase in INTERVIEWER_PHRASES:
        start = 0
        while (i := joined.find(" " + phrase + " ", start)) >= 0:
            first = joined[: i + 1].count(" ") - 1
            covered[first : first + len(phrase.split())] = True
            start = i + 1
    return float(covered.mean())


def quantiles(values: Any, qs: tuple[float, ...] = (0.1, 0.5, 0.9)) -> str:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "none"
    return ", ".join(f"{q:.0%} {x:.2f}" for q, x in zip(qs, np.quantile(v, qs), strict=True))


def main() -> None:
    os.umask(0o077)
    started = time.time()
    workspace = get_workspace()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg = config["ellie_detector"]
    spectrum = el.Spectrum(**cfg["spectrum"])
    scan, tpl, gates = cfg["scan"], cfg["template"], cfg["gates"]
    facts_dir = workspace.resolve(cfg["facts"]["run"])
    check_sha256(facts_dir / "library_all.npz", cfg["facts"]["sha256"])
    lines_path = workspace.resolve(config["participant_lines"]["run"]) / "lines.parquet"
    words_path = workspace.resolve(config["word_times"]["run"]) / "words.parquet"
    segments_path = workspace.resolve(config["segments"]["run"]) / "segments.parquet"
    labels_path = workspace.resolve(config["labels"]["run"]) / "labels.parquet"
    for path, name in (
        (lines_path, "participant_lines"),
        (words_path, "word_times"),
        (segments_path, "segments"),
        (labels_path, "labels"),
    ):
        check_sha256(path, config[name]["sha256"])
    info: dict[str, Any] = {
        "step": "detect_ellie",
        "status": "running",
        "started_utc": utc_now(),
        "git": git_state(),
        "inputs": {k: config[k] for k in ("participant_lines", "word_times", "segments", "labels")}
        | {"facts": cfg["facts"]},
        "settings": cfg,
        "slurm": {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname()},
    }
    run_dir = (
        workspace.derived / "ellie_detector" / f"run-{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "run.json", info)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(cfg["seed"])
    try:
        all_lines = pd.read_parquet(lines_path, columns=["session", "line", "start", "end", "drop"])
        all_words = pd.read_parquet(
            words_path, columns=["session", "line", "word", "start", "end", "score"]
        )
        segments = pd.read_parquet(segments_path, columns=["segment", "session", "starts", "ends"])
        labels = pd.read_parquet(labels_path)
        libraries = {
            name: el.from_arrays(np.load(facts_dir / f"library_{name}.npz"))
            for name in ("all", "even", "odd")
        }
        semis = (cfg["decoy_semitones"], -cfg["decoy_semitones"])
        decoys = {
            name: [[el.decoy(t, s, spectrum) for t in lib] for s in semis]
            for name, lib in libraries.items()
        }
        sessions_dir = workspace.edaic / "sessions"

        def context(s_index: int) -> Context:
            session_id = PILOT_IDS[s_index]
            audio_path, tpath = session_files(sessions_dir, session_id)
            audio, rate = sf.read(audio_path, dtype="float32")
            if rate != el.SAMPLING_RATE or audio.ndim != 1:
                raise ValueError(f"session {session_id}: expected 16 kHz mono audio")
            power = el.power_spectrum(audio, spectrum)
            db = el.to_db(power)
            median = np.median(db, axis=0)
            white = (db - median).astype(np.float32)
            energy = el.energy_db(power)
            transcript = read_transcript(tpath)
            share = [phrase_share(normalise(str(t)).split()) for t in transcript["text"]]
            lines = pd.DataFrame(
                {
                    "start": transcript["start"].astype(float),
                    "end": transcript["end"].astype(float),
                    "phrase_share": share,
                }
            )
            kept_lines = all_lines[(all_lines["session"] == session_id) & ~all_lines["drop"]]
            words = all_words[all_words["session"] == session_id]
            n = len(energy)
            interior = span_mask(
                n,
                [
                    (a + 0.02, b - 0.02)
                    for a, b, sc in zip(words["start"], words["end"], words["score"], strict=True)
                    if sc >= 0.2 and b - a > 0.04
                ],
            )
            return Context(
                session=s_index,
                audio=audio,
                median=median,
                energy=energy,
                white=white,
                shape=el.spectral_shape(white),
                floor=el.noise_floor(energy),
                level_p=float(np.median(energy[interior])) if interior.any() else float("nan"),
                kept=span_mask(n, zip(kept_lines["start"], kept_lines["end"], strict=True)),
                gap=~span_mask(n, zip(lines["start"], lines["end"], strict=True)),
                lines=lines,
                words=words,
            )

        def scan_bank(
            shape: np.ndarray, valid: np.ndarray, bank: list[el.Template]
        ) -> tuple[np.ndarray, np.ndarray]:
            outs, counts = [], []
            order = np.argsort([t.frames for t in bank])  # batch similar lengths together
            ncc = np.full((len(bank), len(shape)), np.nan, dtype=np.float32)
            cnt = np.zeros((len(bank), len(shape)), dtype=np.float32)
            for first in range(0, len(bank), BATCH):
                idx = order[first : first + BATCH]
                a, b = el.ncc_scan(shape, valid, [bank[i] for i in idx], device=device)
                ncc[idx], cnt[idx] = a, b
            del outs, counts
            return ncc, cnt

        active_cells = {
            name: np.array([t.active.sum() for t in lib], dtype=np.float32)
            for name, lib in libraries.items()
        }

        # ---------------- Pass 1: decoy statistics for score A ----------------
        stats_a = {
            name: np.zeros((len(lib), 3)) for name, lib in libraries.items()
        }  # count, sum, sum of squares
        for s_index in range(len(PILOT_IDS)):
            ctx = context(s_index)
            ones = np.ones(len(ctx.shape), dtype=np.float32)
            for name in libraries:
                for family in decoys[name]:
                    ncc, _ = scan_bank(ctx.shape, ones, family)
                    sel = ncc[:, ctx.kept]
                    ok = np.isfinite(sel)
                    stats_a[name][:, 0] += ok.sum(axis=1)
                    stats_a[name][:, 1] += np.where(ok, sel, 0).sum(axis=1)
                    stats_a[name][:, 2] += np.where(ok, sel * sel, 0).sum(axis=1)

        def mean_sd(stats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            count = np.maximum(stats[:, 0], 1)
            mean = stats[:, 1] / count
            return mean, np.sqrt(np.maximum(stats[:, 2] / count - mean**2, 1e-12))

        mu_a = {name: mean_sd(s) for name, s in stats_a.items()}
        print(f"pass 1 done: decoy statistics for score A ({time.time() - started:.0f} s)")

        # ---------------- Pass 2: the interviewer's level in each session ----------------
        level_rows, confident_rows = [], []
        for s_index in range(len(PILOT_IDS)):
            ctx = context(s_index)
            lib = libraries["all"]
            ncc, _ = scan_bank(ctx.shape, np.ones(len(ctx.shape), np.float32), lib)
            mean, sd = mu_a["all"]
            z = (ncc - mean[:, None]) / sd[:, None]
            frames_e, gains = [], []
            for i, t in enumerate(lib):
                for p in el.local_peaks(z[i], t.frames // 2, scan["confident_z"]):
                    if ctx.gap[p : p + t.frames].mean() < 0.5:
                        continue
                    active_frames = p + np.flatnonzero(t.active.any(axis=1))
                    frames_e.append(ctx.energy[active_frames[active_frames < len(ctx.energy)]])
                    gains.append(el.fitted_gain(ctx.white, t, int(p)))
                    confident_rows.append(
                        {
                            "session": s_index,
                            "template": i,
                            "start": int(p),
                            "end": int(p + t.frames),
                            "level": float(np.median(frames_e[-1])),
                        }
                    )
            level_rows.append(
                {
                    "session": s_index,
                    "hits": len(gains),
                    "floor": ctx.floor,
                    "level_p": ctx.level_p,
                    "level_e": float(np.median(np.concatenate(frames_e))) if gains else np.nan,
                    "gain_e": float(np.median(gains)) if gains else np.nan,
                }
            )
        levels = pd.DataFrame(level_rows)
        confident = pd.DataFrame(confident_rows)
        enough = levels["hits"] >= scan["min_confident_hits"]
        pooled_offset = (
            float(np.median((levels["level_e"] - levels["floor"])[enough]))
            if enough.any()
            else 10.0
        )
        pooled_gain = float(np.median(levels["gain_e"][enough])) if enough.any() else 0.0
        levels.loc[~enough, "level_e"] = levels.loc[~enough, "floor"] + pooled_offset
        levels.loc[~enough, "gain_e"] = pooled_gain
        levels["delta"] = np.clip(
            (levels["level_p"] - levels["level_e"]) / 2, scan["delta_db_min"], scan["delta_db_max"]
        )
        print(
            f"pass 2 done: confident hits {len(confident)}; sessions with at least {scan['min_confident_hits']}:"
            f" {int(enough.sum())} of {len(levels)} ({time.time() - started:.0f} s)"
        )

        def quiet(ctx: Context) -> np.ndarray:
            row = levels.iloc[ctx.session]
            return (ctx.energy <= row["level_e"] + row["delta"]).astype(np.float32)

        # ---------------- Pass 3: decoy statistics for score B ----------------
        stats_b = {name: np.zeros((len(lib), 3)) for name, lib in libraries.items()}
        for s_index in range(len(PILOT_IDS)):
            ctx = context(s_index)
            v = quiet(ctx)
            for name in libraries:
                for family in decoys[name]:
                    ncc, cnt = scan_bank(ctx.shape, v, family)
                    usable = cnt >= scan["valid_share"] * active_cells[name][:, None]
                    sel = np.where(usable, ncc, np.nan)[:, ctx.kept]
                    ok = np.isfinite(sel)
                    stats_b[name][:, 0] += ok.sum(axis=1)
                    stats_b[name][:, 1] += np.where(ok, sel, 0).sum(axis=1)
                    stats_b[name][:, 2] += np.where(ok, sel * sel, 0).sum(axis=1)
        mu_b = {name: mean_sd(s) for name, s in stats_b.items()}
        print(f"pass 3 done: decoy statistics for score B ({time.time() - started:.0f} s)")

        def candidates_of(
            ctx: Context,
            shape: np.ndarray,
            white: np.ndarray,
            v_b: np.ndarray,
            name: str,
            family: str,
            bank: list[el.Template],
            offset: int = 0,
        ) -> list[dict[str, Any]]:
            """Candidates (z >= candidate_z) of one bank in one session, for both scores."""
            row = levels.iloc[ctx.session]
            out = []
            for kind, valid, (mean, sd) in (
                ("A", np.ones(len(shape), np.float32), mu_a[name]),
                ("B", v_b, mu_b[name]),
            ):
                ncc, cnt = scan_bank(shape, valid, bank)
                if kind == "B":
                    ncc = np.where(
                        cnt >= scan["valid_share"] * active_cells[name][:, None], ncc, np.nan
                    )
                z = (ncc - mean[:, None]) / sd[:, None]
                for i, t in enumerate(bank):
                    for p in el.local_peaks(z[i], t.frames // 2, scan["candidate_z"]):
                        gain = el.fitted_gain(white, t, int(p)) if kind == "A" else np.nan
                        gain_ok = bool(kind == "B" or gain <= row["gain_e"] + row["delta"])
                        out.append(
                            {
                                "session": ctx.session,
                                "library": name,
                                "family": family,
                                "template": i,
                                "kind": kind,
                                "start": int(p) + offset,
                                "end": int(p + t.frames) + offset,
                                "z": float(z[i, p]),
                                "gain": float(gain),
                                "gain_ok": gain_ok,
                            }
                        )
            return out

        # ---------------- Pass 4: candidates ----------------
        rows: list[dict[str, Any]] = []
        kept_frames = np.zeros(len(PILOT_IDS))
        for s_index in range(len(PILOT_IDS)):
            ctx = context(s_index)
            kept_frames[s_index] = ctx.kept.sum()
            v_b = quiet(ctx)
            for name, lib in libraries.items():
                rows += candidates_of(ctx, ctx.shape, ctx.white, v_b, name, "real", lib)
            for family, bank in zip(("up", "down"), decoys["all"], strict=True):
                rows += candidates_of(ctx, ctx.shape, ctx.white, v_b, "all", family, bank)
        cand = pd.DataFrame(rows)
        cand.to_parquet(run_dir / "candidates.parquet", index=False)
        print(f"pass 4 done: candidates {len(cand)} ({time.time() - started:.0f} s)")

        # ---------------- Stationary-noise templates ----------------
        def coverage(sel: pd.DataFrame) -> pd.Series:
            """Share of kept-line time that each template's hits cover."""
            out: dict[tuple[str, int], float] = {}
            for (name, template), group in sel.groupby(["library", "template"]):
                covered = 0
                for s_index, g in group.groupby("session"):
                    lines = all_lines[
                        (all_lines["session"] == PILOT_IDS[int(s_index)]) & ~all_lines["drop"]
                    ]
                    n = int(max(g["end"].max(), frame(lines["end"].max()) if len(lines) else 0)) + 1
                    kept = span_mask(n, zip(lines["start"], lines["end"], strict=True))
                    hit = np.zeros(n, dtype=bool)
                    for a, b in zip(g["start"], g["end"], strict=True):
                        hit[a:b] = True
                    covered += int((hit & kept).sum())
                out[(str(name), int(template))] = covered / kept_frames.sum()
            return pd.Series(out, dtype=float)

        real = cand[cand["family"] == "real"]
        noisy = coverage(real[(real["kind"] == "A") & (real["z"] >= 4) & real["gain_ok"]])
        noisy = noisy[noisy > tpl["noise_share"]]
        removed = {
            name: sorted(int(t) for (lib, t) in noisy.index if lib == name) for name in libraries
        }
        drop_mask = (
            cand.apply(lambda r: r["template"] in removed[r["library"]], axis=1)
            if len(cand)
            else pd.Series(dtype=bool)
        )
        cand = cand[~drop_mask] if len(cand) else cand
        print(f"stationary-noise templates removed: { {k: len(v) for k, v in removed.items()} }")

        # ---------------- Zones, word costs and the threshold ----------------
        word_spans = {
            s: (g["start"].to_numpy(), g["end"].to_numpy()) for s, g in all_words.groupby("session")
        }
        n_kept_words = len(all_words)

        def zones_at(sel: pd.DataFrame, theta: float) -> dict[int, list[tuple[int, int]]]:
            hit = sel[(sel["z"] >= theta) & sel["gain_ok"]]
            return {
                int(s): el.merge_zones(
                    list(zip(g["start"], g["end"], strict=True)), scan["zone_merge_ms"] // 10
                )
                for s, g in hit.groupby("session")
            }

        def removed_words(
            zones: dict[int, list[tuple[int, int]]], guard: float
        ) -> dict[int, np.ndarray]:
            out = {}
            for s_index, zs in zones.items():
                starts, ends = word_spans.get(PILOT_IDS[s_index], (np.zeros(0), np.zeros(0)))
                hit = np.zeros(len(starts), dtype=bool)
                for a, b in zs:
                    hit |= (starts - guard < b / 100) & (ends + guard > a / 100)
                out[s_index] = hit
            return out

        def word_cost(zones: dict[int, list[tuple[int, int]]]) -> float:
            return (
                sum(
                    int(h.sum())
                    for h in removed_words(zones, cfg["words"]["guard_ms"] / 1000).values()
                )
                / n_kept_words
            )

        grid = []
        for theta in cfg["theta_grid"]:
            costs = {
                fam: word_cost(
                    zones_at(cand[(cand["library"] == "all") & (cand["family"] == fam)], theta)
                )
                for fam in ("up", "down")
            }
            grid.append(
                {
                    "theta": theta,
                    "up": costs["up"],
                    "down": costs["down"],
                    "mean": (costs["up"] + costs["down"]) / 2,
                }
            )
        grid_table = pd.DataFrame(grid)
        cap = cfg["threshold_cost"]
        meets = grid_table[
            (grid_table["mean"] <= cap["mean"])
            & (grid_table[["up", "down"]].max(axis=1) <= cap["family"])
        ]
        d2 = bool(len(meets))
        theta_z = float(meets["theta"].min()) if d2 else float(max(cfg["theta_grid"]))
        print("threshold grid (decoy cost as a share of kept words):")
        print(grid_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"D2 decoys: theta_z {theta_z}: {'PASSED' if d2 else 'FAILED'}")
        real_all = cand[(cand["library"] == "all") & (cand["family"] == "real")]
        zones = zones_at(real_all, theta_z)
        zone_rows = [
            {"session": s, "session_id": PILOT_IDS[s], "start": a / 100, "end": b / 100}
            for s, zs in zones.items()
            for a, b in zs
        ]
        zone_table = pd.DataFrame(zone_rows, columns=["session", "session_id", "start", "end"])
        zone_table.to_parquet(run_dir / "zones.parquet", index=False)

        # ---------------- D1: known interviewer lines ----------------
        d1_cov = []
        for s_index in range(len(PILOT_IDS)):
            _, tpath = session_files(sessions_dir, PILOT_IDS[s_index])
            transcript = read_transcript(tpath)
            for start, end, text in zip(
                transcript["start"], transcript["end"], transcript["text"], strict=True
            ):
                if phrase_share(normalise(str(text)).split()) >= 0.6 and end > start:
                    covered = sum(
                        max(0.0, min(end, b / 100) - max(start, a / 100))
                        for a, b in zones.get(s_index, [])
                    )
                    d1_cov.append(covered / (end - start))
        d1_cov_np = np.array(d1_cov)
        d1_share = (
            float(np.mean(d1_cov_np >= gates["d1"]["min_covered"])) if len(d1_cov_np) else 0.0
        )
        d1 = d1_share >= gates["d1"]["min_line_share"]
        print(
            f"D1 known lines: {len(d1_cov_np)} mostly-interviewer lines; at least half covered: {d1_share:.3f}:"
            f" {'PASSED' if d1 else 'FAILED'}"
        )

        # ---------------- D4: cost ----------------
        nw = cfg["null_word"]
        windows_hit, windows_total = 0, 0
        null_words = set(cfg["null_words"])
        for s, g in all_words.groupby("session"):
            s_index = PILOT_IDS.index(int(s))
            per_line = g.groupby("line")["word"].transform("size")
            pick = g[
                (per_line >= nw["min_line_words"])
                & g["word"].isin(null_words)
                & (g["score"] >= nw["min_score"])
            ]
            for a in pick["start"]:
                windows_total += 1
                b = a + nw["window_ms"] / 1000
                windows_hit += any(
                    zb / 100 > a and za / 100 < b for za, zb in zones.get(s_index, [])
                )
        word_windows = windows_hit / max(windows_total, 1)
        kept_cost = word_cost(zones)
        d4 = (
            word_windows <= gates["d4"]["max_word_windows"]
            and kept_cost <= gates["d4"]["max_kept_words"]
        )
        print(
            f"D4 cost: common-word windows touching a zone {word_windows:.4f} (of {windows_total});"
            f" kept words removed {kept_cost:.4f}: {'PASSED' if d4 else 'FAILED'}"
        )

        # ---------------- D3: planted copies ----------------
        def plant(
            target: Context, copy: np.ndarray, copy_level: float, onset_s: float, level: float
        ) -> tuple[np.ndarray, int]:
            first = max(0, int((onset_s - LOCAL_SECONDS) * el.SAMPLING_RATE))
            last = min(
                len(target.audio),
                int((onset_s + len(copy) / el.SAMPLING_RATE + LOCAL_SECONDS) * el.SAMPLING_RATE),
            )
            piece = target.audio[first:last].copy()
            at = int(onset_s * el.SAMPLING_RATE) - first
            piece[at : at + len(copy)] += copy[: max(0, len(piece) - at)] * 10 ** (
                (level - copy_level) / 20
            )
            return piece, first

        copies_by_parity: dict[int, list[tuple[np.ndarray, float, np.ndarray]]] = {0: [], 1: []}
        conf = confident.sample(frac=1.0, random_state=cfg["seed"]) if len(confident) else confident
        for s_index, g in conf.groupby("session"):
            ctx = context(int(s_index))
            for r in g.head(10).itertuples():
                a, b = int(r.start), int(r.end)
                wave = ctx.audio[a * spectrum.hop : b * spectrum.hop]
                audible = ctx.energy[a:b] >= ctx.floor + cfg["events"]["rise_db"]
                copies_by_parity[PILOT_IDS[int(s_index)] % 2].append(
                    (wave, float(r.level), audible)
                )
        inj_rows = []
        n_copies = gates["d3"]["copies"]
        for parity, name in ((0, "even"), (1, "odd")):
            sources = copies_by_parity[1 - parity]
            targets = [i for i, sid in enumerate(PILOT_IDS) if sid % 2 == parity]
            if not sources:
                continue
            plan = {pl: [] for pl in PLACEMENTS}
            for k in range(n_copies * len(PLACEMENTS)):
                plan[PLACEMENTS[k % len(PLACEMENTS)]].append(
                    (int(rng.choice(targets)), int(rng.integers(len(sources))))
                )
            by_target: dict[int, list[tuple[str, int]]] = {}
            for placement, picks in plan.items():
                for t_index, c_index in picks:
                    by_target.setdefault(t_index, []).append((placement, c_index))
            for t_index, asks in by_target.items():
                ctx = context(t_index)
                row = levels.iloc[t_index]
                w = ctx.words.sort_values("start")
                starts, ends, line_ids = (
                    w["start"].to_numpy(),
                    w["end"].to_numpy(),
                    w["line"].to_numpy(),
                )
                for placement, c_index in asks:
                    wave, copy_level, audible = sources[c_index]
                    dur = len(wave) / el.SAMPLING_RATE
                    if len(starts) < 2:
                        continue
                    same = line_ids[1:] == line_ids[:-1]
                    gaps_in = starts[1:] - ends[:-1]
                    if placement == "pause":
                        ok = np.flatnonzero(same & (gaps_in >= dur + 0.1))
                        onset = (
                            None if len(ok) == 0 else float(ends[:-1][int(rng.choice(ok))] + 0.05)
                        )
                    elif placement == "edge":
                        last = np.flatnonzero(np.r_[line_ids[1:] != line_ids[:-1], True])
                        onset = float(ends[int(rng.choice(last))] + rng.uniform(0.0, 0.3))
                    elif placement == "partial":
                        ok = np.flatnonzero(same & (gaps_in >= 0.4 * dur))
                        onset = (
                            None
                            if len(ok) == 0
                            else float(starts[1:][int(rng.choice(ok))] - 0.4 * dur)
                        )
                    else:
                        onset = float(starts[int(rng.integers(len(starts)))])
                    if (
                        onset is None
                        or onset + dur + LOCAL_SECONDS > len(ctx.audio) / el.SAMPLING_RATE
                    ):
                        continue
                    for level_name, level in (
                        ("at_level", row["level_e"]),
                        ("minus_6_db", row["level_e"] - 6),
                    ):
                        piece, first = plant(ctx, wave, copy_level, onset, level)
                        power = el.power_spectrum(piece, spectrum)
                        white = (el.to_db(power) - ctx.median).astype(np.float32)
                        shape_l = el.spectral_shape(white)
                        energy_l = el.energy_db(power)
                        v_b = (energy_l <= row["level_e"] + row["delta"]).astype(np.float32)
                        local = Context(
                            session=t_index,
                            audio=piece,
                            median=ctx.median,
                            energy=energy_l,
                            white=white,
                            shape=shape_l,
                            floor=ctx.floor,
                            level_p=ctx.level_p,
                            kept=ctx.kept,
                            gap=ctx.gap,
                            lines=ctx.lines,
                            words=ctx.words,
                        )
                        c = pd.DataFrame(
                            candidates_of(local, shape_l, white, v_b, name, "real", libraries[name])
                        )
                        if len(c):
                            c = c[~c["template"].isin(removed[name])]
                        hit = c[(c["z"] >= theta_z) & c["gain_ok"]] if len(c) else c
                        covered = np.zeros(len(energy_l), dtype=bool)
                        for a, b in el.merge_zones(
                            list(zip(hit.get("start", []), hit.get("end", []), strict=True)),
                            scan["zone_merge_ms"] // 10,
                        ):
                            covered[a:b] = True
                        at = frame(onset - first / el.SAMPLING_RATE)
                        frames_active = at + np.flatnonzero(audible)
                        frames_active = frames_active[frames_active < len(covered)]
                        found = bool(
                            len(frames_active)
                            and covered[frames_active].mean() >= gates["d3"]["found_share"]
                        )
                        inj_rows.append(
                            {
                                "parity": name,
                                "placement": placement,
                                "level": level_name,
                                "found": found,
                            }
                        )
        injections = pd.DataFrame(inj_rows, columns=["parity", "placement", "level", "found"])
        injections.to_parquet(run_dir / "injections.parquet", index=False)
        table = (
            injections.pivot_table(
                index="placement", columns="level", values="found", aggfunc="mean"
            )
            if len(injections)
            else pd.DataFrame()
        )
        print("D3 planted copies, share found (both parities):")
        print(table.to_string(float_format=lambda x: f"{x:.3f}") if len(table) else "  none")
        at_level = (
            injections[injections["level"] == "at_level"].groupby("placement")["found"].mean()
            if len(injections)
            else pd.Series(dtype=float)
        )
        d3 = bool(
            at_level.get("pause", 0) >= gates["d3"]["pause"]
            and at_level.get("edge", 0) >= gates["d3"]["edge"]
            and at_level.get("partial", 0) >= gates["d3"]["partial"]
        )
        print(f"D3: {'PASSED' if d3 else 'FAILED'}")

        # ---------------- Facts F4-F7 ----------------
        loc = {"gap": 0, "edge": 0, "pause": 0, "over_words": 0, "dropped_line": 0}
        seg_hits, seg_total = {0: [0, 0], 1: [0, 0]}, 0
        per_person = []
        for s_index in range(len(PILOT_IDS)):
            session_id = PILOT_IDS[s_index]
            zs = zones.get(s_index, [])
            lines_s = all_lines[all_lines["session"] == session_id]
            words_s = all_words[all_words["session"] == session_id].sort_values("start")
            n = max([b for _, b in zs] + [frame(lines_s["end"].max()) + 1 if len(lines_s) else 1])
            inline = span_mask(n, zip(lines_s["start"], lines_s["end"], strict=True))
            kept_m = span_mask(
                n,
                zip(
                    lines_s.loc[~lines_s["drop"], "start"],
                    lines_s.loc[~lines_s["drop"], "end"],
                    strict=True,
                ),
            )
            edge = span_mask(n, [(e, e + 0.3) for e in lines_s["end"]]) & ~inline
            over = span_mask(n, zip(words_s["start"], words_s["end"], strict=True))
            zmask = np.zeros(n, dtype=bool)
            for a, b in zs:
                zmask[a:b] = True
            loc["over_words"] += int((zmask & over).sum())
            loc["pause"] += int((zmask & kept_m & ~over).sum())
            loc["dropped_line"] += int((zmask & inline & ~kept_m).sum())
            loc["edge"] += int((zmask & edge).sum())
            loc["gap"] += int((zmask & ~inline & ~edge).sum())
            label = int(labels.loc[session_id, "label"])
            starts, ends = words_s["start"].to_numpy(), words_s["end"].to_numpy()
            gone = np.zeros(len(starts), dtype=bool)
            for a, b in zs:
                gone |= (starts - cfg["words"]["guard_ms"] / 1000 < b / 100) & (
                    ends + cfg["words"]["guard_ms"] / 1000 > a / 100
                )
            same_line = words_s["line"].to_numpy()[1:] == words_s["line"].to_numpy()[:-1]
            pauses = starts[1:] - ends[:-1]
            long_pause = same_line & (pauses >= 0.7)
            pause_cut = (
                np.array(
                    [
                        any(zb / 100 > e and za / 100 < s_ for za, zb in zs)
                        for e, s_ in zip(ends[:-1], starts[1:], strict=True)
                    ],
                    dtype=bool,
                )
                if len(starts) > 1
                else np.zeros(0, bool)
            )
            splits = (
                int((same_line & pause_cut & ~gone[:-1] & ~gone[1:]).sum())
                if len(starts) > 1
                else 0
            )
            per_person.append(
                {
                    "session": session_id,
                    "label": label,
                    "gender": labels.loc[session_id, "gender"],
                    "words": len(starts),
                    "removed": int(gone.sum()),
                    "splits": splits,
                    "long_pause_s": float(pauses[long_pause].sum()) if len(starts) > 1 else 0.0,
                    "long_pause_cut_s": float(pauses[long_pause & pause_cut].sum())
                    if len(starts) > 1
                    else 0.0,
                    "segments_before": len(starts) // 40,
                    "segments_after": int(len(starts) - gone.sum()) // 40,
                }
            )
            segs = segments[segments["session"] == session_id]
            for st, en in zip(segs["starts"], segs["ends"], strict=True):
                seg_total += 1
                windows = part_windows(
                    list(st), list(en), n / 100 + 10, margin=config["x0_cut"]["margin_ms"] / 1000
                )
                touched = any(
                    zb / 100 > lo and za / 100 < hi for lo, hi in windows for za, zb in zs
                )
                seg_hits[label][0] += touched
                seg_hits[label][1] += 1
        total_zone = sum(loc.values())
        people = pd.DataFrame(per_person)
        people["share"] = people["removed"] / people["words"].clip(lower=1)
        print(
            "zone time by place: "
            + ", ".join(f"{k} {v / max(total_zone, 1):.3f}" for k, v in loc.items())
            + f"; total {total_zone / 6000:.1f} min"
        )
        both = seg_hits[0][0] + seg_hits[1][0]
        print(
            f"segments with a zone inside their x0 windows: {both / max(seg_total, 1):.3f} (the listening check heard 4 of 20);"
            f" by class: negative {seg_hits[0][0] / max(seg_hits[0][1], 1):.3f}, positive {seg_hits[1][0] / max(seg_hits[1][1], 1):.3f}"
        )
        p_class = float(
            mannwhitneyu(
                people.loc[people["label"] == 1, "share"], people.loc[people["label"] == 0, "share"]
            ).pvalue
        )
        p_gender = (
            float(mannwhitneyu(*[g["share"] for _, g in people.groupby("gender")]).pvalue)
            if people["gender"].nunique() == 2
            else float("nan")
        )
        print(
            f"cost preview: words removed {people['removed'].sum() / max(people['words'].sum(), 1):.4f};"
            f" splits per 100 words {100 * people['splits'].sum() / max(people['words'].sum(), 1):.2f};"
            f" long within-line pause time removed {people['long_pause_cut_s'].sum() / max(people['long_pause_s'].sum(), 1e-9):.3f}"
        )
        print(
            f"  per participant removed share {quantiles(people['share'])}; by class Mann-Whitney p = {p_class:.3f};"
            f" by gender p = {p_gender:.3f}; participants below 16 segments: before {int((people['segments_before'] < 16).sum())},"
            f" after {int((people['segments_after'] < 16).sum())}; with no segment after {int((people['segments_after'] == 0).sum())}"
        )
        pre_words = 0
        for s_index in range(len(PILOT_IDS)):
            long_first = [
                (a, b)
                for a, b in zones.get(s_index, [])
                if b - a >= cfg["pre_interview"]["min_zone_ms"] // 10
            ]
            if long_first and long_first[0][0] <= cfg["pre_interview"]["within_ms"] // 10:
                w = all_words[all_words["session"] == PILOT_IDS[s_index]]
                pre_words += int((w["end"] < long_first[0][0] / 100).sum())
        pre_share = pre_words / n_kept_words
        pre_on = pre_share <= cfg["pre_interview"]["max_word_share"]
        print(
            f"pre-interview rule: would drop {pre_share:.4f} of the kept words: {'on' if pre_on else 'off'}"
        )
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        print(
            f"run time {time.time() - started:.0f} s; peak GPU memory {peak_gb:.1f} GB;"
            f" candidates file {(run_dir / 'candidates.parquet').stat().st_size / 1e6:.1f} MB"
        )
        info.update(
            status="done",
            finished_utc=utc_now(),
            theta_z=theta_z,
            grid=grid,
            removed_templates={k: len(v) for k, v in removed.items()},
            gates={
                "d1": {"passed": bool(d1), "share": d1_share, "lines": len(d1_cov_np)},
                "d2": {"passed": d2, "theta_z": theta_z},
                "d3": {"passed": d3, "found": json.loads(table.to_json()) if len(table) else {}},
                "d4": {"passed": bool(d4), "word_windows": word_windows, "kept_words": kept_cost},
            },
            facts={
                "zone_minutes": total_zone / 6000,
                "zone_place": {k: v / max(total_zone, 1) for k, v in loc.items()},
                "segments_touched": both / max(seg_total, 1),
                "words_removed_p_class": p_class,
                "pre_interview_share": pre_share,
                "pre_interview_on": bool(pre_on),
            },
            zones_sha256=sha256_file(run_dir / "zones.parquet"),
        )
    except BaseException:
        info.update(status="failed", finished_utc=utc_now())
        raise
    finally:
        write_json(run_dir / "run.json", info)


if __name__ == "__main__":
    main()
