"""Chip sounds: which sample (and #VOLUME) each note plays.

Default: one sample per lane from kit/ (KIT in autochart.py), full volume.
Options:
  * kit file (JSON, lane -> [file, volume]) - e.g. your own samples.
  * sample match: pick the library snare / kick closest to the recorded one (ring pitch,
    brightness, decay), per stretch of the song where it changes (e.g. a long medley).
  * chip volume: a base #VOLUME for every drum chip, and with dynamics three volume steps per lane
    from how hard each hit is on the LarsNet stems (ghost notes quieter).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import KIT_DIR, SAMPLE_DIRS
from drumkit import kit_file

HERE = Path(__file__).resolve().parent
SR = 22050
STEM_OF = {"11": "hihat", "18": "hihat", "1B": "hihat", "12": "snare", "13": "kick", "1C": "kick",
           "14": "toms", "15": "toms", "17": "toms", "16": "cymbals", "1A": "cymbals", "19": "cymbals"}
TIERS = ((-6.0, 1.0), (-12.0, 0.75), (-99.0, 0.5))   # dB under the lane's strong hits -> volume share
STRETCH = 20.0   # s: a medley keeps one snare sound at least this long


# kind -> (name has one of, name has none of, default kit file, ring band Hz, lanes, LarsNet stem)
KINDS = {
    "snare": (("snare", "_sd", "sd"), ("rim", "march", "bassdrum", "kick", "_bd"), kit_file("snare"), (120, 400), ("12",), "snare"),
    "kick": (("_bd", "bd_", "bassdrum", "kick"), (), kit_file("kick"), (40, 120), ("13", "1C"), "kick"),
}


def library(kind: str) -> list[Path]:
    """Samples of this kind in the kit and in the sample folders from settings."""
    yes, no, default = KINDS[kind][:3]
    out = [KIT_DIR / default]      # first, so a copy of it elsewhere counts as the kit's own
    for d in SAMPLE_DIRS:
        for p in sorted(Path(d).rglob("*")):
            n = p.name.lower()
            if p.suffix.lower() in (".wav", ".ogg") and any(n.startswith(k) or k in n for k in yes)                     and not any(k in n for k in no):
                out.append(p)
    seen, uniq = set(), []
    for p in out:                      # same file copied into several song folders
        k = (p.name.lower(), p.stat().st_size)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def _feat(y: np.ndarray, i: int, band_hz=(120, 400)) -> np.ndarray:
    """(ring Hz, brightness Hz, decay dB) of a hit starting at sample i."""
    seg = y[i:i + int(0.2 * SR)]
    if len(seg) < int(0.2 * SR):
        seg = np.pad(seg, (0, int(0.2 * SR) - len(seg)))
    win = seg[int(0.02 * SR):int(0.12 * SR)] * np.hanning(int(0.1 * SR))
    S = np.abs(np.fft.rfft(win, 8192))
    f = np.fft.rfftfreq(8192, 1 / SR)
    band = (f >= band_hz[0]) & (f <= band_hz[1])
    ring = float(f[band][np.argmax(S[band])])
    head = np.abs(np.fft.rfft(seg[:int(0.05 * SR)] * np.hanning(int(0.05 * SR)), 4096))
    fh = np.fft.rfftfreq(4096, 1 / SR)
    bright = float((head * fh).sum() / (head.sum() + 1e-9))
    e0 = np.sqrt(np.mean(seg[:int(0.02 * SR)] ** 2)) + 1e-9
    e1 = np.sqrt(np.mean(seg[int(0.06 * SR):int(0.2 * SR)] ** 2)) + 1e-9
    return np.array([ring, bright, 20 * np.log10(e1 / e0)])


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(12 * np.log2(a[0] / b[0]) / 3.0,          # 3 semitones
                          np.hypot(12 * np.log2(a[1] / b[1]) / 4.0,  # 4 semitones of brightness
                                   (a[2] - b[2]) / 6.0)))            # 6 dB of decay


def sample_features(p: Path, band_hz=(120, 400)) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(p), sr=SR, mono=True)
    i = int(np.argmax(np.abs(y) > 0.1 * np.abs(y).max()))
    return _feat(y, max(i - int(0.002 * SR), 0), band_hz)


def match_kind(notes, kind: str, parts: dict[str, Path]) -> dict[int, Path]:
    """note id -> library sample, one choice per stretch of the song whose hits sound alike."""
    import librosa
    band, lanes, stem = KINDS[kind][3:]
    y, _ = librosa.load(str(parts[stem]), sr=SR, mono=True)
    hs = sorted((n for n in notes if n.ch in lanes), key=lambda n: n.t)
    if not hs:
        return {}
    F = np.array([_feat(y, int(n.t * SR), band) for n in hs])
    L = [(p, sample_features(p, band)) for p in library(kind)]
    # stretches: running median over 8 hits, cut where the sound moves > 1 (feature units)
    med = np.array([np.median(F[max(i - 4, 0):i + 4], axis=0) for i in range(len(hs))])
    cuts, a = [0], 0
    for i in range(1, len(hs)):
        if (hs[i].t - hs[a].t > STRETCH and _dist(med[i], np.median(F[a:i], axis=0)) > 1.0
                and i + 8 < len(hs)) or hs[i].t - hs[i - 1].t > 15.0:
            cuts.append(i)
            a = i
    cuts.append(len(hs))
    out, prev = {}, None
    for a, b in zip(cuts[:-1], cuts[1:]):
        f = np.median(F[a:b], axis=0)
        p, _ = min(L, key=lambda x: _dist(f, x[1]))
        for n in hs[a:b]:
            out[id(n)] = p
        if p != prev:
            print(f"      {kind} {hs[a].t:6.1f}s: ring {f[0]:.0f} Hz, bright {f[1]:.0f} Hz, decay {f[2]:+.0f} dB"
                  f" -> {p.parent.name}/{p.name}")
            prev = p
    return out


def loudness(notes, parts: dict[str, Path]) -> dict[int, float]:
    """note id -> dB under the 90th percentile of its lane's hits (LarsNet stem, first 30 ms)."""
    import librosa
    out = {}
    for stem in set(STEM_OF.values()):
        ns = [n for n in notes if STEM_OF.get(n.ch) == stem]
        if not ns:
            continue
        y, _ = librosa.load(str(parts[stem]), sr=SR, mono=True)
        lv = np.array([20 * np.log10(np.sqrt(np.mean(y[int(n.t * SR):int((n.t + 0.03) * SR) + 1] ** 2)) + 1e-9)
                       for n in ns])
        top = np.percentile(lv, 90)
        for n, v in zip(ns, lv):
            out[id(n)] = float(v - top)
    return out


def dress(notes, kit: dict, parts: dict[str, Path] | None, kit_file: Path | None = None,
          snare_match: bool = False, volume: int = 100, dynamics: bool = False,
          log: list[str] | None = None, default_dir: Path = KIT_DIR, kit_volume: dict | None = None):
    """Returns (wav id per note id, {wav id: (source file, volume)}). Without any option this is
    exactly the default kit at full volume (ids from KIT)."""
    # (file, volume, slot): the default kit keeps one id per lane as before (1C has its own id)
    kv = kit_volume or {}
    lane: dict[str, tuple[Path, int, str]] = {ch: (default_dir / f, volume * kv.get(ch, 100) // 100, wid)
                                              for ch, (wid, f) in kit.items()}
    if kit_file:
        spec = json.loads(Path(kit_file).read_text(encoding="utf-8"))
        for ch, (f, v) in spec.items():
            p = Path(f)
            lane[ch] = (p if p.is_absolute() else Path(kit_file).parent / p, int(v * volume / 100), "")
    per_note: dict[int, tuple[Path, int, str]] = {id(n): lane[n.ch] for n in notes}
    if snare_match and parts:
        for kind in KINDS:
            vol = lane[KINDS[kind][4][0]][1]
            for k, p in match_kind(notes, kind, parts).items():
                per_note[k] = (p, vol, "")
    if dynamics and parts:
        db = loudness(notes, parts)
        for n in notes:
            p, v, _ = per_note[id(n)]
            share = next(s for lim, s in TIERS if db.get(id(n), 0.0) >= lim)
            per_note[id(n)] = (p, max(int(round(v * share)), 1), "")
    # wav ids: the default kit keeps its KIT ids, anything else gets new ones
    ids: dict[tuple[Path, int, str], str] = {}
    for ch, (wid, f) in kit.items():
        ids.setdefault((default_dir / f, kv.get(ch, 100), wid), wid)
    used = set(ids.values()) | {"01"}
    nxt = 2

    def new_id():
        nonlocal nxt
        from dtx import to_b36
        while True:
            s = to_b36(nxt).rjust(2, "0")
            nxt += 1
            if s not in used:
                used.add(s)
                return s
    wav_of = {}
    for n in notes:
        key = per_note[id(n)]
        if key not in ids:
            ids[key] = new_id()
        wav_of[id(n)] = ids[key]
    table = {wid: key[:2] for key, wid in ids.items() if wid in set(wav_of.values()) or key[2] == wid}
    return wav_of, table
