"""Tempo map estimation: BPM sections and bar lines (meter changes).

The beat tracker is unreliable on complex songs (half/double time, odd meters),
so tempo is measured from drum onsets (~3 ms precise) by phase coherence on a
16th-note (tatum) grid. The 16th pulse survives odd meters like 15/16 where the
quarter-note phase jumps. The beat tracker only contributes downbeat evidence
for choosing bar lines.
"""
from __future__ import annotations

from dataclasses import dataclass

from fractions import Fraction

import numpy as np

# Bar lengths in 16ths and their prior cost (4/4 is free; odd meters must earn it).
# Whole-beat changes (12, 20, 8, 24) only move the downbeat, which the beat tracker often
# guesses wrong; 8th shifts (14, 18) are backed by the beat-position evidence.
BAR_COST_BASE = {16: 0.0, 12: 2.5, 8: 3.0, 20: 3.0, 24: 3.5, 14: 1.6, 18: 1.6, 15: 1.8, 17: 1.8}
REPEATABLE = {12, 14}   # 3/4 and 7/8 runs are common real meters
BAR_COST = dict(BAR_COST_BASE)
CHANGE_COST = 1.5       # switching bar length from the previous bar
REPEAT_DISCOUNT = 0.7   # an odd bar repeating the previous bar's length (7/8 7/8 ...) is a real meter
W_DOWN = 3.0            # weight of (accent-backed) downbeat evidence at a bar start
W_BEAT = 0.6            # weight per beat of "beats on the quarter positions, not the 8th offbeats"
UNSUPPORTED = 0.5       # share of a downbeat guess that counts without kick/crash support
SILENT_EXTRA = 50.0     # odd bar without any drum hit in it
TRANSITION_COST = 0.5   # any-length bar where the tempo changes


def coherence(times: np.ndarray, period: float, w: np.ndarray | None = None) -> complex:
    """Weighted mean phase vector of `times` on a grid of `period` (|z| = 1 means perfect fit)."""
    z = np.exp(2j * np.pi * times / period)
    return complex(np.average(z, weights=w))


def comb_score(times: np.ndarray, w: np.ndarray, bpm: float) -> float:
    """How well onsets fit a 16th grid of this BPM (a small quarter term breaks ties)."""
    if len(times) < 4:
        return 0.0
    p = 60.0 / bpm
    return abs(coherence(times, p / 4, w)) + 0.15 * abs(coherence(times, p, w))


def local_tempo(onsets: np.ndarray, w: np.ndarray, ref_bpm: float, win: float = 8.0,
                hop: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windowed tempo curve. Candidates span ref_bpm * [0.8, 1.25]: a 16th grid at 2x BPM
    fits the same onsets, so the range must stay inside one octave.

    Returns (window centers, candidate bpms, score matrix [windows x candidates]).
    """
    cands = np.arange(ref_bpm * 0.8, ref_bpm * 1.25, 0.1)
    centers = np.arange(win / 2, onsets[-1] - win / 2 + hop, hop)
    S = np.zeros((len(centers), len(cands)))
    for i, c in enumerate(centers):
        m = (onsets >= c - win / 2) & (onsets < c + win / 2)
        for j, b in enumerate(cands):
            S[i, j] = comb_score(onsets[m], w[m], b)
    return centers, cands, S


def viterbi_tempo(S: np.ndarray, cands: np.ndarray, change_penalty: float = 0.6) -> np.ndarray:
    """Smooth path through the tempo score matrix; staying is free, jumping costs a penalty
    (small drifts within 1% are cheap)."""
    n, k = S.shape
    rel = np.abs(np.log(cands[:, None] / cands[None, :]))
    trans = np.where(rel < 0.01, rel * 10, change_penalty)
    score = S[0].copy()
    back = np.zeros((n, k), int)
    for i in range(1, n):
        tot = score[:, None] - trans          # from j to k
        back[i] = tot.argmax(axis=0)
        score = tot.max(axis=0) + S[i]
    path = np.zeros(n, int)
    path[-1] = score.argmax()
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return cands[path]


# ---- sections -------------------------------------------------------------------
@dataclass
class Section:
    start: float      # audio seconds (first section starts at -inf conceptually)
    end: float
    bpm: float
    phase: float = 0.0  # a 16th grid point: phase + k * period/4
    ramp: bool = True   # allow a tempo ramp into this section (off when the user fixed the switch)

    @property
    def tatum(self) -> float:
        return 15.0 / self.bpm


def clean_bpm(onsets: np.ndarray, w: np.ndarray, bpm: float) -> float:
    """Snap to integer / .5 / .1 BPM when that fits (almost) as well."""
    best = abs(coherence(onsets, 15.0 / bpm, w))
    for cand in (round(bpm), round(bpm * 2) / 2, round(bpm, 1), round(bpm, 2)):
        if abs(coherence(onsets, 15.0 / cand, w)) >= 0.99 * best:
            return float(cand)
    return round(bpm, 2)


def fit_section(s: Section, onsets: np.ndarray, w: np.ndarray, margin: float, search: float) -> None:
    """Exact BPM (+-search) and 16th phase from the onsets inside the section."""
    lo = s.start + margin if np.isfinite(s.start) else -np.inf
    hi = s.end - margin if np.isfinite(s.end) else np.inf
    m = (onsets >= lo) & (onsets < hi)
    if m.sum() < 8:
        m = (onsets >= s.start) & (onsets < s.end)
    on, ww = onsets[m], w[m]
    if len(on) < 8:
        return
    fine = np.arange(s.bpm - search, s.bpm + search, 0.01)
    s.bpm = float(max(fine, key=lambda b: abs(coherence(on, 15.0 / b, ww))))
    s.bpm = clean_bpm(on, ww, s.bpm)
    z = coherence(on, s.tatum, ww)
    s.phase = (np.angle(z) % (2 * np.pi)) / (2 * np.pi) * s.tatum


def refine_boundary(a: Section, b: Section, onsets: np.ndarray, w: np.ndarray,
                    span: float = 6.0) -> float:
    """Switch time near the coarse boundary where onsets before fit grid A and after fit grid B."""
    t0 = a.end
    m = (onsets > t0 - span - 2) & (onsets < t0 + span + 2)
    on, ww = onsets[m], w[m]

    def on_grid(s: Section) -> np.ndarray:
        # strict fit: two tempos ~10% apart have 16ths only ~10 ms apart, a loose
        # (cosine) fit cannot tell them apart over a short fill
        d = ((on - s.phase) / s.tatum + 0.5) % 1 - 0.5
        return ww * np.exp(-(d * s.tatum / 0.012) ** 2)
    fa, fb = on_grid(a), on_grid(b)
    best_t, best = t0, -np.inf
    for t in np.arange(t0 - span, t0 + span, 0.01):
        sc = fa[on < t].sum() + fb[on >= t].sum()
        if sc > best:
            best_t, best = t, sc
    # switch on the B beat at/before the first B onset (tempo often changes in a quiet gap)
    after = on[on >= best_t]
    first_b = after[0] if len(after) else best_t
    p = 4 * b.tatum
    return float(b.phase + np.floor((first_b - 0.01 - b.phase) / p) * p)


MIN_SECTION = 2.0   # s; a tempo section shorter than this after refining its switches is dropped


def find_sections(onsets: np.ndarray, w: np.ndarray, ref_bpm: float,
                  min_len: float = 6.0) -> list[Section]:
    centers, cands, S = local_tempo(onsets, w, ref_bpm)
    path = viterbi_tempo(S, cands)
    # group windows with the same tempo (within 1.5%)
    groups: list[list[int]] = [[0]]
    for i in range(1, len(path)):
        ref = np.median(path[groups[-1]])
        if abs(np.log(path[i] / ref)) < 0.015:
            groups[-1].append(i)
        else:
            groups.append([i])
    # merge groups shorter than min_len into the closer-tempo neighbour
    hop = centers[1] - centers[0] if len(centers) > 1 else 1.0
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for gi, g in enumerate(groups):
            if len(g) * hop < min_len:
                nb = [j for j in (gi - 1, gi + 1) if 0 <= j < len(groups)]
                tgt = min(nb, key=lambda j: abs(np.log(np.median(path[groups[j]]) / np.median(path[g]))))
                groups[tgt] = sorted(groups[tgt] + g)
                del groups[gi]
                changed = True
                break
    sections = []
    for gi, g in enumerate(groups):
        start = -np.inf if gi == 0 else (centers[groups[gi - 1][-1]] + centers[g[0]]) / 2
        end = np.inf if gi == len(groups) - 1 else (centers[g[-1]] + centers[groups[gi + 1][0]]) / 2
        sections.append(Section(float(start), float(end), float(np.median(path[g]))))
    # exact BPM + phase per section from its own onsets (keep away from the coarse
    # boundaries, which can be several seconds off)
    for s in sections:
        fit_section(s, onsets, w, margin=3.0, search=1.5)
    # exact switch points, then refit on the corrected ranges; two refined switches can cross
    # (a short section between them comes out with a negative length) - drop that section and
    # refine its neighbours' switch again, searching the whole stretch it covered
    coarse = [s.end for s in sections[:-1]]   # coarse switch k: sections k -> k+1
    spans = [6.0] * len(coarse)
    while True:
        for k, (a, b) in enumerate(zip(sections, sections[1:])):
            a.end = coarse[k]
            a.end = b.start = refine_boundary(a, b, onsets, w, spans[k])
        short = [i for i, s in enumerate(sections) if s.end - s.start < MIN_SECTION]
        if not short or len(sections) < 3:
            break
        i = min(short, key=lambda j: sections[j].end - sections[j].start)
        lo, hi = coarse[i - 1], coarse[i]   # never the first or last section (infinite)
        coarse[i - 1], spans[i - 1] = (lo + hi) / 2, (hi - lo) / 2 + 6.0
        del sections[i], coarse[i], spans[i]
    for s in sections:
        fit_section(s, onsets, w, margin=0.0, search=0.3)
    # merge neighbours that ended up at the same BPM
    merged = [sections[0]]
    for s in sections[1:]:
        if s.bpm == merged[-1].bpm and abs(((s.phase - merged[-1].phase) / s.tatum + 0.5) % 1 - 0.5) < 0.1:
            merged[-1].end = s.end
        else:
            merged.append(s)
    return merged


def force_section(sections: list[Section], t: float, bpm: float, onsets: np.ndarray,
                  w: np.ndarray) -> list[Section]:
    """User hint --tempo-from: a new tempo starts at audio time t (e.g. the detector merged
    a 170 part into a 166 intro, < 4% apart). The grid before t is kept exactly as it was; the new section gets its own exact BPM + phase."""
    out = []
    for s in sections:
        if s.start < t < s.end:
            new = Section(t, s.end, bpm, ramp=False)
            fit_section(new, onsets, w, margin=0.0, search=1.0)
            out += [Section(s.start, t, s.bpm, s.phase, s.ramp), new]
        else:
            out.append(s)
    return out


def switch_at(sections: list[Section], prev_bar: float) -> float:
    """User hint: the bar starting at prev_bar is a plain 4/4 bar at the old tempo and the tempo
    changes (clean jump) exactly on the next bar line. That bar line goes on the new section's
    grid point nearest prev_bar + one old-tempo bar; the old grid is re-anchored there (the new
    grid, fitted to the drums, stays put). Returns the switch time."""
    i = min(range(1, len(sections)), key=lambda j: abs(sections[j].start - prev_bar))
    a, b = sections[i - 1], sections[i]
    t = prev_bar + 16 * a.tatum
    x = b.phase + round((t - b.phase) / b.tatum) * b.tatum
    a.phase = x
    a.end = b.start = x
    b.ramp = False
    return x


# ---- slow drift (live takes) ---------------------------------------------------
CLOSE_SECTIONS = 0.04  # neighbouring sections closer than 4% in BPM are one drifting take
STEP_MIN = 0.02        # a jump of at least 2% in BPM (a live band speeding up creeps by less) ...
STEP_LEN = 20.0        # s; between sections both at least this long ...
STEP_LOSS = 0.75       # ... where one grid keeps under 75% of a section's on-grid score: a real tempo step


def section_fit(s: Section, onsets: np.ndarray, w: np.ndarray, lo: float, hi: float) -> float:
    """Share of the onset strength in lo..hi that sits within ~12 ms of the section's 16th grid."""
    m = (onsets >= lo) & (onsets < hi)
    if m.sum() < 8:
        return 0.0
    on, ww = onsets[m], w[m]
    d = ((on - s.phase) / s.tatum + 0.5) % 1 - 0.5
    return float((ww * np.exp(-(d * s.tatum / 0.012) ** 2)).sum() / ww.sum())


def real_step(a: Section, b: Section, onsets: np.ndarray, w: np.ndarray) -> bool:
    """Two long sections a few % apart that each fit their own steady grid far better than one
    shared grid (e.g. 194.7 -> 199.9 for 30 s each) are a real tempo step, not a drifting take."""
    lo = a.start if np.isfinite(a.start) else float(onsets[0])
    hi = b.end if np.isfinite(b.end) else float(onsets[-1])
    if abs(np.log(b.bpm / a.bpm)) < STEP_MIN or a.end - lo < STEP_LEN or hi - b.start < STEP_LEN:
        return False
    one = Section(lo, hi, a.bpm)
    fit_section(one, onsets, w, margin=0.0, search=4.0)
    return any(section_fit(one, onsets, w, x, y) < STEP_LOSS * section_fit(s, onsets, w, x, y)
               for s, x, y in ((a, lo, a.end), (b, b.start, hi)))


def merge_close_sections(sections: list[Section], onsets: np.ndarray, w: np.ndarray) -> list[Section]:
    """A live take can come out as 128 / 124 / 121.8 / 122.7 / 124.8 BPM with odd bars between:
    a live take that speeds up, not tempo changes. Merge neighbours within CLOSE_SECTIONS and refit;
    the drift follower then charts the wander. Real changes (e.g. 146 -> 168,
    129.5 -> 111.7) are far apart and stay."""
    out = list(sections)
    while True:   # refitting can bring merged neighbours within range again
        merged = [out[0]]
        for prev, s in zip(out, out[1:]):   # the step test needs prev's own fitted grid
            if abs(np.log(s.bpm / merged[-1].bpm)) < CLOSE_SECTIONS and not real_step(prev, s, onsets, w):
                merged[-1] = Section(merged[-1].start, s.end, merged[-1].bpm)
            else:
                merged.append(s)
        if len(merged) == len(out):
            break
        for s in merged:
            fit_section(s, onsets, w, margin=0.0, search=4.0)
        out = merged
    if len(out) == len(sections):
        return sections
    print(f"      {len(sections)} tempo sections within {CLOSE_SECTIONS:.0%} -> {len(out)} (drifting take)")
    return out
DRIFT_MAX = 0.015      # follow the drift past ±15 ms (20-26 ms of wander stays as it is)
DRIFT_TOL = 0.005      # a constant-BPM stretch may miss the drift curve by this much
DRIFT_MIN_BARS = 8     # BPM changes no more often than every 8 bars (a phrase)


def drift_curve(T: np.ndarray, onsets: np.ndarray, w: np.ndarray, win: float = 8.0,
                hop: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """How far the drums sit from grid T over time (seconds, + = drums later), unwrapped across
    tatums so a slow drift past half a 16th is followed instead of snapping to the next 16th.
    Unwrapping can slip on sparse or noisy windows: callers check the result with grid_fit."""
    tat = float(np.median(np.diff(T)))
    centers, ang = [], []
    for c in np.arange(onsets[0] + win / 2, onsets[-1] - win / 2 + hop, hop):
        m = (onsets >= c - win / 2) & (onsets < c + win / 2)
        if m.sum() < 12:
            continue
        j = np.clip(np.searchsorted(T, onsets[m]), 1, len(T) - 1)
        r = onsets[m] - T[j - 1]
        z = np.average(np.exp(2j * np.pi * r / tat), weights=w[m])
        if abs(z) < 0.3:   # no clear pulse in this window
            continue
        centers.append(c)
        ang.append(np.angle(z))
    if len(centers) < 3:
        return np.array([0.0]), np.array([0.0])
    phi = np.unwrap(np.array(ang)) / (2 * np.pi) * tat
    phi = np.array([np.median(phi[max(0, i - 2):i + 3]) for i in range(len(phi))])  # de-spike
    return np.array(centers), phi


def grid_fit(T: np.ndarray, hits: np.ndarray, win: float = 15.0) -> float:
    """Mean over `win`-second stretches of |median offset of hits from the nearest grid point|
    (ms): how well a grid follows the drums, ignoring a constant lag."""
    j = np.clip(np.searchsorted(T, hits), 1, len(T) - 1)
    r = hits - np.where(np.abs(T[j - 1] - hits) < np.abs(T[j] - hits), T[j - 1], T[j])
    r = r - np.median(r)
    meds = [np.median(r[(hits >= a) & (hits < a + win)]) for a in np.arange(hits[0], hits[-1], win)
            if ((hits >= a) & (hits < a + win)).sum() >= 8]
    return float(np.mean(np.abs(meds)) * 1000) if meds else 0.0


STEP_PENALTY = 12.0   # BPM-bars: cost of one more tempo stretch in step_grid
STEP_SNAP = 0.012     # s: a whole BPM is used if the stretch still ends this close to its bar line


def step_grid(Tw: np.ndarray, starts: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Programmed song with tempo changes (e.g. 164 / 185 / 180). Per-bar tempo of the
    beat-locked grid Tw is noisy, so cut it into plateaus (least absolute deviation + a penalty
    per stretch), give each plateau one BPM that ends it on its last bar line (the chart's
    running time carried over), a whole BPM when that still lands within STEP_SNAP."""
    bs = list(starts)
    tau = Tw[bs]
    x = 15.0 * np.diff(bs) / np.diff(tau)            # BPM of each bar
    n = len(x)
    f = np.full(n + 1, np.inf)
    f[0] = 0.0
    back = np.zeros(n + 1, int)
    for j in range(1, n + 1):
        for i in range(max(0, j - 400), j):
            seg = x[i:j]
            c = f[i] + np.abs(seg - np.median(seg)).sum() + STEP_PENALTY
            if c < f[j]:
                f[j], back[j] = c, i
    cuts, j = [n], n
    while j > 0:
        j = back[j]
        cuts.append(j)
    cuts = cuts[::-1]
    seg_bpm = np.zeros(len(Tw) - 1)
    t_a = float(tau[0])
    for a, b in zip(cuts[:-1], cuts[1:]):
        n16 = bs[b] - bs[a]
        bpm = round(15.0 * n16 / (tau[b] - t_a), 2)
        whole = float(round(bpm))
        if abs(t_a + n16 * 15.0 / whole - tau[b]) <= STEP_SNAP:
            bpm = whole
        seg_bpm[bs[a]:bs[b]] = bpm
        t_a += n16 * 15.0 / bpm
    seg_bpm[:bs[0]] = seg_bpm[bs[0]]
    seg_bpm[bs[-1]:] = seg_bpm[bs[-1] - 1]
    T = np.empty(len(Tw))
    T[bs[0]] = tau[0]
    for i in range(bs[0], len(Tw) - 1):
        T[i + 1] = T[i] + 15.0 / seg_bpm[i]
    for i in range(bs[0] - 1, -1, -1):
        T[i] = T[i + 1] - 15.0 / seg_bpm[i]
    return T, np.append(seg_bpm, seg_bpm[-1])


def drift_grid(Tw: np.ndarray, starts: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Constant BPM (2 decimals, as DTX writes it) over stretches of whole bars, changing only on
    a bar line, so the grid follows the warped grid Tw within DRIFT_TOL. Same tatum count as Tw."""
    bs = [b for b in starts if b < len(Tw)]   # a bar line past the grid's end (ほしふりの果て: 2352 of 2349)
    tau = Tw[bs]
    seg_bpm = np.zeros(len(Tw) - 1)
    t_a, a = float(tau[0]), 0
    while a < len(bs) - 1:
        best = None
        for b in range(a + 1, len(bs)):
            n = bs[b] - bs[a]
            bpm = round(15.0 / ((tau[b] - t_a) / n), 2)
            p = 15.0 / bpm
            err = max(abs(t_a + (bs[k] - bs[a]) * p - tau[k]) for k in range(a, b + 1))
            if b - a > DRIFT_MIN_BARS and err > DRIFT_TOL:
                break
            best = (b, bpm)
        b, bpm = best
        seg_bpm[bs[a]:bs[b]] = bpm
        t_a += (bs[b] - bs[a]) * 15.0 / bpm
        a = b
    seg_bpm[:bs[0]] = seg_bpm[bs[0]]
    seg_bpm[bs[-1]:] = seg_bpm[bs[-1] - 1]
    T = np.empty(len(Tw))
    T[bs[0]] = tau[0]
    for i in range(bs[0], len(Tw) - 1):
        T[i + 1] = T[i] + 15.0 / seg_bpm[i]
    for i in range(bs[0] - 1, -1, -1):
        T[i] = T[i + 1] - 15.0 / seg_bpm[i]
    return T, np.append(seg_bpm, seg_bpm[-1])


def track_beats(beat: np.ndarray, hits: np.ndarray, fps: int = 50) -> np.ndarray:
    """Beat times from the beat tracker, cleaned up: sub-frame peaks, missing beats filled and
    double beats dropped, aimed at the drum hits (rolling median offset) and smoothed with a
    local line over 9 beats so one late hit doesn't bend the grid."""
    from scipy.signal import find_peaks
    pk, _ = find_peaks(beat, height=0.3, distance=int(fps * 0.25))
    a, b, c = beat[pk - 1], beat[pk], beat[np.minimum(pk + 1, len(beat) - 1)]
    t = (pk + 0.5 * (a - c) / np.maximum(a - 2 * b + c, -1e9) * (a - 2 * b + c < 0)) / fps
    for _ in range(3):
        d = np.diff(t)
        med = np.array([np.median(d[max(i - 8, 0):i + 8]) for i in range(len(d))])
        keep = np.ones(len(t), bool)
        keep[1:][d < 0.6 * med] = False
        t = t[keep]
        d = np.diff(t)
        med = np.array([np.median(d[max(i - 8, 0):i + 8]) for i in range(len(d))])
        add = [np.linspace(t[i], t[i + 1], int(round(d[i] / med[i])) + 1)[1:-1]
               for i in range(len(d)) if d[i] > 1.5 * med[i]]
        if not add:
            break
        t = np.sort(np.concatenate([t] + add))
    # the tracker sometimes counts half time for a stretch (e.g. ~82 in a
    # 163 BPM medley): keep one beat level, the song's usual one
    d = np.diff(t)
    g = np.median(d)
    loc = np.array([np.median(d[max(i - 4, 0):i + 5]) for i in range(len(d))])
    half = (loc > 1.7 * g) & (loc < 2.3 * g) & (d > 1.5 * g)
    if half.any():
        at = t[:-1][half]
        runs = np.split(at, np.flatnonzero(np.diff(at) > 3 * g) + 1)
        print("      beat tracker counted half time at " + ", ".join(f"{r[0]:.0f}-{r[-1]:.0f}s" for r in runs)
              + " -> beats doubled")
        t = np.sort(np.concatenate([t, (t[:-1][half] + t[1:][half]) / 2]))
    j = np.clip(np.searchsorted(hits, t), 1, len(hits) - 1)
    near = np.where(np.abs(hits[j - 1] - t) < np.abs(hits[j] - t), hits[j - 1], hits[j]) - t
    ok = np.abs(near) < 0.06
    off = np.array([np.median(near[max(i - 8, 0):i + 9][ok[max(i - 8, 0):i + 9]])
                    if ok[max(i - 8, 0):i + 9].sum() >= 3 else 0.0 for i in range(len(t))])
    tt = t + np.where(ok, near, off)            # each beat on its own hit where there is one
    out = np.empty(len(t))
    for i in range(len(t)):
        a0, a1 = max(i - 4, 0), min(i + 5, len(t))
        k = np.arange(a0, a1)
        out[i] = np.polyval(np.polyfit(k, tt[a0:a1], 1), i)
    return out


BEAT_ODD_BAR = 4.0   # cost of a bar that is not 4 beats long (beat grid)


def beat_bars(down: np.ndarray, crash: np.ndarray, kick: np.ndarray, snare: np.ndarray) -> list[int]:
    """Bar starts (beat indices) for a beat-locked chart: 4/4 unless the backbeat and the downbeat
    cues move (e.g. at a key change). Per-beat cues are 0..1. A bar earns its beat 1
    (tracker downbeat, crash, kick) and its backbeat (snare on 2 and 4, not on 1 and 3)."""
    n = len(down)
    sn = lambda i: snare[i] if i < n else 0.0

    def gain(i, L):
        g = 1.5 * down[i] + crash[i] + 0.5 * kick[i] - 0.5 * sn(i)
        if L >= 4:
            g += 0.5 * (sn(i + 1) + sn(i + 3) - sn(i + 2))
        return g - (0 if L == 4 else BEAT_ODD_BAR)
    f = np.full(n + 1, -1e18)
    back = np.full(n + 1, -1)
    f[:4] = 0.0                         # the first bar may start on any of the first 4 beats
    for j in range(1, n + 1):
        for L in (2, 3, 4, 5, 6):
            i = j - L
            if i >= 0 and f[i] > -1e17 and f[i] + gain(i, L) > f[j]:
                f[j], back[j] = f[i] + gain(i, L), i
    j = int(np.argmax(f[n - 4:n])) + n - 4   # the last bar line on one of the last 4 beats
    out = [j]
    while back[j] >= 0:
        j = int(back[j])
        out.append(j)
    return out[::-1]


def beat_grid(beats: np.ndarray, bar_beats: list[int], duration: float, per_beat: bool = False
              ) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """A 16th grid locked to the tracked beats with bars starting on `bar_beats`. One BPM chip per
    bar (2 decimals, the rounding carried into the next bar), or per beat. Rule for a
    drifting take: every bar line on the real beat 1, the chart stays tidy."""
    marks = list(range(bar_beats[0], bar_beats[-1] + 1)) if per_beat else list(bar_beats)
    T, bpm, starts = [float(beats[marks[0]])], [], [0]
    for a, b in zip(marks[:-1], marks[1:]):
        x = round(60.0 * (b - a) / (beats[b] - T[-1]), 2)
        for _ in range(4 * (b - a)):
            bpm.append(x)
            T.append(T[-1] + 15.0 / x)
    starts = [4 * (k - bar_beats[0]) for k in bar_beats]
    b0, b1 = bpm[0], bpm[-1]
    lead: list[float] = []
    while not lead or lead[0] > -8 * 60.0 / b0 or len(lead) % 16:
        lead.insert(0, (lead[0] if lead else T[0]) - 15.0 / b0)
    end = starts[-1]
    while T[-1] < duration + 60.0 / b1:
        T.append(T[-1] + 15.0 / b1)
        bpm.append(b1)
    n0 = len(lead)
    starts = list(range(0, n0, 16)) + [s + n0 for s in starts]
    k = end + n0
    while k + 16 < n0 + len(T):
        k += 16
        starts.append(k)
    T = np.array(lead + T)
    bpm = np.array([b0] * n0 + bpm + [b1])
    return T, bpm[:len(T)], starts


# ---- tatum grid --------------------------------------------------------------
def tatum_grid(sections: list[Section], duration: float, onsets: np.ndarray | None = None,
               w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Continuous 16th grid across sections. Returns (times, bpm of each tatum step).

    At a section change, the switch happens between the pair of grid points (last of A,
    first of B) whose gap is closest to one tatum, so no partial tatum is needed.
    """
    times: list[float] = []
    bpms: list[float] = []
    lead = 2 * 4 * sections[0].tatum * 4   # 2 bars before audio start for pickups
    cur = sections[0].phase - np.ceil((sections[0].phase + lead) / sections[0].tatum) * sections[0].tatum
    for si, s in enumerate(sections):
        nxt = sections[si + 1] if si + 1 < len(sections) else None
        stop = (nxt.start if nxt else duration) + 4 * s.tatum
        t = cur
        a_pts = []
        while t < stop:
            a_pts.append(t)
            t += s.tatum
        if nxt is None:
            times += a_pts
            bpms += [s.bpm] * len(a_pts)
            break
        if not a_pts:   # a short section already passed by the previous one's switch: skip it
            continue
        # candidate switch pairs near the boundary
        best = None
        for ai in range(max(0, len(a_pts) - 12), len(a_pts)):
            a = a_pts[ai]
            k = np.ceil((a - nxt.phase) / nxt.tatum)
            for dk in range(0, 3):
                b = nxt.phase + (k + dk) * nxt.tatum
                gap = b - a
                if gap <= 0.3 * s.tatum:
                    continue
                cost = abs(gap - s.tatum) + 0.002 * abs(a - nxt.start) / s.tatum
                if best is None or cost < best[0]:
                    best = (cost, ai, b)
        if best is None:
            best = (0.0, len(a_pts) - 1, a_pts[-1] + s.tatum)
        _, ai, b = best
        ramp = best_ramp(a_pts[:ai + 1], b, s, nxt, onsets, w) if onsets is not None and nxt.ramp else None
        if ramp is None:
            times += a_pts[:ai + 1]
            bpms += [s.bpm] * (ai + 1)
        else:
            ra_i, gaps = ramp
            times += a_pts[:ra_i + 1]
            bpms += [s.bpm] * ra_i
            t = a_pts[ra_i]
            for g in gaps[:-1]:
                bpms.append(round(15.0 / g, 2))
                t += g
                times.append(t)
            bpms.append(round(15.0 / gaps[-1], 2))
        cur = b
    return np.array(times), np.array(bpms)


def _fit(points: np.ndarray, on: np.ndarray, w: np.ndarray, sigma: float = 0.015) -> float:
    """Weighted count of onsets that sit on one of the grid points."""
    if len(on) == 0 or len(points) == 0:
        return 0.0
    j = np.clip(np.searchsorted(points, on), 1, len(points) - 1)
    d = np.minimum(np.abs(on - points[j - 1]), np.abs(on - points[j]))
    return float(np.sum(w * np.exp(-(d / sigma) ** 2)))


def best_ramp(a_pts: list[float], b: float, A: "Section", B: "Section", onsets: np.ndarray,
              w: np.ndarray, max_tatums: int = 48, margin: float = 0.08):
    """Try a linear tempo ramp (accelerando/ritardando) of n 16ths ending at b, starting on the A grid.

    Returns (index into a_pts where the ramp starts, list of ramp gap durations) when it fits the
    onsets clearly better than an instant tempo jump, else None.
    """
    dA, dB = A.tatum, B.tatum
    lo, hi = b - max_tatums * dA - 1.0, b + 2.0
    m = (onsets >= lo) & (onsets < hi)
    on, ww = onsets[m], w[m]
    if len(on) < 6:
        return None
    b_pts = b + dB * np.arange(int(2.0 / dB) + 2)
    a_arr = np.array(a_pts)
    step = _fit(np.concatenate([a_arr, b_pts]), on, ww)
    best = None
    for n in range(4, max_tatums + 1):
        gaps = dA + (dB - dA) * np.arange(1, n + 1) / (n + 1)
        ra_i = int(round((b - gaps.sum() - a_arr[0]) / dA))
        if ra_i < 1 or ra_i >= len(a_arr):
            continue
        scale = (b - a_arr[ra_i]) / gaps.sum()
        if abs(scale - 1) > 0.12:
            continue
        g = gaps * scale
        pts = np.concatenate([a_arr[:ra_i + 1], a_arr[ra_i] + np.cumsum(g)[:-1], b_pts])
        sc = _fit(pts, on, ww)
        if best is None or sc > best[0]:
            best = (sc, ra_i, list(g))
    if best and best[0] > step * (1 + margin) + 0.5:
        return best[1], best[2]
    return None


# ---- bar lines ------------------------------------------------------------------
NEG = -1e18


def choose_bars(T: np.ndarray, down: np.ndarray, beat: np.ndarray, accent: np.ndarray,
                audio_start: float = 0.0, active: np.ndarray | None = None,
                switches: list[int] | None = None) -> list[int]:
    """Semi-Markov DP over bar start tatums.

    down/beat/accent are per-tatum evidence in [0, 1]: downbeat activation, beat activation,
    and kick/crash accents. Returns bar start indices into T.
    switches: tatums where the BPM changes. The bar containing one is the transition bar and may
    take any length (a charter fixes the new phrase's first bar, then fits the bar before it).
    active: per-tatum drum activity; a bar without any drum hit stays 4/4 (nothing to justify
    an odd bar, and the beat tracker alone is unreliable there).
    """
    n = len(T)
    lengths = list(range(8, 25))
    sw = np.zeros(n + 1, int)
    for x in switches or []:
        sw[x] = 1
    sw_c = np.concatenate([[0], np.cumsum(sw)])
    Ls = {L: i for i, L in enumerate(lengths)}
    # A downbeat guess counts fully only when a kick/crash backs it up (within 1 tatum):
    # the beat tracker alone mislabels syncopated or half-time passages.
    acc_near = np.maximum(accent, np.maximum(np.roll(accent, 1), np.roll(accent, -1)))
    ev = down * (UNSUPPORTED + (1 - UNSUPPORTED) * acc_near)
    active_c = np.concatenate([[0], np.cumsum(active)]) if active is not None else None

    def bar_score(s: int, L: int) -> float:
        # downbeat at start; penalise downbeat evidence inside
        inner_down = max((ev[s + j] for j in range(2, L - 1) if s + j < n), default=0.0)
        # beat phase: the bar's quarter positions should carry the beats, not the 8th offbeats
        # (the strongest cue when a riff shifts the bar line by an 8th)
        on = sum(beat[s + j] for j in range(0, L, 4) if s + j < n)
        off = sum(beat[s + j] for j in range(2, L, 4) if s + j < n)
        if sw_c[min(s + L, n)] - sw_c[s + 1] > 0:   # tempo change strictly inside this bar
            cost = 0.0 if L == 16 else TRANSITION_COST
        elif L in BAR_COST:
            cost = BAR_COST[L]
        else:
            return NEG
        if active is not None and L != 16 and not active_c[min(s + L, n)] - active_c[s]:
            cost += SILENT_EXTRA
        return W_DOWN * ev[s] + 0.8 * accent[s] + W_BEAT * (on - off) - 1.0 * inner_down - cost

    best = np.full((n + 1, len(lengths)), NEG)
    back = np.zeros((n + 1, len(lengths), 2), int)
    # the first bar must start before the audio starts (pickup lives in the lead-in)
    first = np.searchsorted(T, audio_start)
    for s in range(max(0, first - 24), first + 1):
        for L in lengths:
            if s + L <= n:
                v = bar_score(s, L)
                if v > best[s + L, Ls[L]]:
                    best[s + L, Ls[L]] = v
                    back[s + L, Ls[L]] = (-1, s)
    for s in range(n):
        row = best[s]
        if row.max() <= NEG / 2:
            continue
        for L in lengths:
            if s + L > n:
                continue
            sc = bar_score(s, L)
            for pi, Lp in enumerate(lengths):
                if row[pi] <= NEG / 2:
                    continue
                v = row[pi] + sc - (CHANGE_COST if Lp != L else 0.0)
                if Lp == L and L in REPEATABLE:
                    v += REPEAT_DISCOUNT * BAR_COST[L]
                if v > best[s + L, Ls[L]]:
                    best[s + L, Ls[L]] = v
                    back[s + L, Ls[L]] = (pi, s)
    # end: best state reaching the last few tatums
    end_s = max(range(max(0, n - 24), n + 1), key=lambda s: best[s].max())
    pi = int(best[end_s].argmax())
    starts = []
    s = end_s
    while True:
        prev_pi, start = back[s, pi]
        starts.append(int(start))
        if prev_pi == -1:
            break
        s, pi = start, prev_pi
    starts.reverse()
    starts.append(end_s)  # closing boundary
    starts = normalize_bars(starts)
    if active is not None and active.any():
        hit = np.flatnonzero(active)
        starts = edge_shifts(starts, int(hit[0]), int(hit[-1]), switches)
    return starts


def parse_meter(tok: str) -> int:
    """Meter as the user writes it -> bar length in 16ths: '44' 4/4, '78' 7/8, '34' 3/4,
    '1516' 15/16, '7/8'."""
    if "/" in tok:
        num, den = tok.split("/")
    elif tok.endswith("16") and len(tok) > 2:
        num, den = tok[:-2], "16"
    else:
        num, den = tok[:-1], tok[-1]
    L = Fraction(int(num) * 16, int(den))
    if L.denominator != 1 or not 1 <= L <= 64:
        raise ValueError(f"bad meter {tok!r}")
    return int(L)


def parse_bar_hints(spec: str) -> list[tuple[int, list[int]]]:
    """'76=78 78 78 44 x4; 17=44' -> [(76, [14, 14, 14, 16] * 4), (17, [16])]."""
    out = []
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        m, body = part.split("=")
        toks = body.split()
        reps = 1
        if toks and toks[-1].lower().startswith("x"):
            reps = int(toks.pop()[1:])
        out.append((int(m), [parse_meter(t) for t in toks] * reps))
    return sorted(out)


def apply_bar_hints(starts: list[int], hints: list[tuple[int, list[int]]], n: int) -> list[int]:
    """Force the user's bar lengths. Measure m (DTX numbering, 001 = first bar) gets the listed
    lengths; after them the chart stays 4/4 until the next hint or the end."""
    for m, lens in hints:
        new = list(starts[:m])
        for L in lens:
            new.append(new[-1] + L)
        while new[-1] + 16 <= n:
            new.append(new[-1] + 16)
        starts = new
    return starts


def normalize_bars(starts: list[int]) -> list[int]:
    """A run of non-4/4 bars that adds up to whole 4/4 bars (e.g. 12+12+12+12 = 3x16) ends
    on the original bar phase: that is syncopation/hemiola, which charts write as 4/4."""
    lens = list(np.diff(starts))
    out = [starts[0]]
    i = 0
    while i < len(lens):
        if lens[i] == 16:
            out.append(out[-1] + 16)
            i += 1
            continue
        j = i
        while j < len(lens) and lens[j] != 16:
            j += 1
        run = lens[i:j]
        total = sum(run)
        if total % 16 == 0:
            out += [out[-1] + 16 * (k + 1) for k in range(total // 16)]
        else:
            for L in run:
                out.append(out[-1] + L)
        i = j
    return cancel_shifts(out)


def merge_bars(starts: list[int], L: int) -> list[int]:
    """Write two neighbouring bars that add up to L as one bar, when at least one is odd
    (7/8 7/8 or 4/4 3/4 -> one 7/4 bar)."""
    lens = list(np.diff(starts))
    out, i = [starts[0]], 0
    while i < len(lens):
        if i + 1 < len(lens) and lens[i] + lens[i + 1] == L and (lens[i] != 16 or lens[i + 1] != 16):
            out.append(out[-1] + L)
            i += 2
        else:
            out.append(out[-1] + lens[i])
            i += 1
    return out


def edge_shifts(starts: list[int], first: int, last: int, switches: list[int] | None = None,
                edge: int = 4) -> list[int]:
    """A lone odd bar within `edge` bars of the first/last drum hit (tatums first/last) only
    re-phases the intro/ending to a crash that anticipates the downbeat (usually by an 8th).
    Keep the main groove's bar phase through the intro and ending instead: move the bar lines
    before (or after) that odd bar."""
    bs = np.array(starts)
    lens = list(np.diff(starts))
    b_first = int(np.searchsorted(bs, first, side="right")) - 1
    b_last = int(np.searchsorted(bs, last, side="right")) - 1
    # a bar holding a tempo change is a transition bar, not a re-phase
    sw = [int(np.searchsorted(bs, x, side="right")) - 1 for x in switches or []]
    odd = [i for i, L in enumerate(lens) if L != 16 and i not in sw]
    head = [i for i in odd if i < b_first + edge]
    if len(head) == 1 and len(odd) > 1:
        k = head[0]
        d = lens[k] - 16
        starts = [x + d if i <= k else x for i, x in enumerate(starts)]
        lens = list(np.diff(starts))
        odd = [i for i, L in enumerate(lens) if L != 16 and i not in sw]
    tail = [i for i in odd if i > b_last - edge]
    if len(tail) == 1:
        k = tail[0]
        d = 16 - lens[k]
        starts = [x + d if i > k else x for i, x in enumerate(starts)]
    return starts


def cancel_shifts(starts: list[int], max_span: int = 12) -> list[int]:
    """A short detour of odd bars whose deviations from 4/4 add up to zero (e.g. 12 ... 18 18, or
    18 ... 14) ends on the original bar phase: the bar lines were only temporarily pulled off
    (beat tracker's downbeat guess, crashes anticipating the downbeat). Within max_span bars,
    rewrite such a stretch as plain 4/4. Longer real shifts (a riff moved by an 8th for a whole
    section) span more bars and are kept."""
    lens = list(np.diff(starts))
    odd = [i for i, L in enumerate(lens) if L != 16]
    used: set[int] = set()
    for a_i, a in enumerate(odd):
        if a in used:
            continue
        dev = 0
        for b in odd[a_i:]:
            if b - a >= max_span or b in used:
                break
            dev += lens[b] - 16
            if dev == 0:
                for k in range(a, b + 1):
                    lens[k] = 16
                    used.add(k)
                break
    out = [starts[0]]
    for L in lens:
        out.append(out[-1] + L)
    return out
