"""Song audio -> playable DTX drum chart (prototype).

Pipeline:
  1. Demucs: split drums / no_drums stems
  2. Tempo map: BPM sections from drum-onset coherence on a 16th grid,
     a continuous 16th (tatum) grid across BPM changes
  3. Bar lines: DP over the tatum grid using beat_this downbeat activations
     and kick/crash accents (allows 3/4, 15/16, 17/16 ... when the audio says so)
  4. ADTOF: drum hits (BD / SD / TT / HH / CY) on the full mix
  5. Quantize hits to the bars and map to DTX lanes
  6. Write a song folder: mstr.dtx (+ #BPMxx / ch08 / ch02) + BGM + drum samples

Usage: python autochart.py SONG_AUDIO OUT_DIR [--title T] [--artist A] [--bgm full|drumless]
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

import config
import add_mv
import dtx
import preview as preview_sound
import sounds
import drumkit
import genres
import tempo
from drumkit import KIT, KIT_VOLUME

HERE = Path(__file__).resolve().parent
KIT_DIR, CACHE, VENDOR = config.KIT_DIR, config.CACHE, config.VENDOR

# ADTOF MIDI pitch -> DTX channel
ADTOF_TO_CH = {35: "13", 38: "12", 47: "14", 42: "11", 49: "16"}
# DTX channel -> kit sample (WAV id, file) and per-lane volume: drumkit.py
# Charters place notes slightly after the detected attack: on a hand-made reference chart notes sit
# +20 ms after the ADTOF kick/snare hits (very consistent). The grid's own lag behind the hits
# varies per song (soft ballad attacks make the onset detector fire later), so the BGM shift is
# set per song to land notes at this lag.
TARGET_LAG = 0.020


def grid_lag(hits: list[tuple[float, int]], T: np.ndarray) -> float:
    """Median (nearest grid point - hit) over kick/snare hits, in seconds."""
    h = np.array([t for t, p in hits if p in (35, 38) and T[0] < t < T[-1]])
    if len(h) < 8:
        return TARGET_LAG / 2
    j = np.clip(np.searchsorted(T, h), 1, len(T) - 1)
    g = np.where(np.abs(T[j - 1] - h) < np.abs(T[j] - h), T[j - 1], T[j]) - h
    return float(np.median(g[np.abs(g) < 0.05]))


def chart_lag(path: Path, hits: list[tuple[float, int]]) -> float | None:
    """Median (charted kick/snare - nearest kick/snare hit), in seconds."""
    placed = np.array([t for t, lane in dtx.read_dtx(path).drum_notes() if lane in ("BD", "LBD", "SD")])
    h = np.array(sorted(t for t, p in hits if p in (35, 38)))
    if len(placed) < 8 or len(h) < 8:
        return None
    j = np.clip(np.searchsorted(h, placed), 1, len(h) - 1)
    d = np.where(np.abs(h[j - 1] - placed) < np.abs(h[j] - placed), placed - h[j - 1], placed - h[j])
    return float(np.median(d[np.abs(d) < 0.05]))


def cache_key(audio: Path) -> str:
    return f"{audio.stem}_{hashlib.md5(str(audio).encode()).hexdigest()[:8]}"


# ---- 1. source separation ---------------------------------------------------
def separate(audio: Path) -> tuple[Path, Path]:
    out = CACHE / "demucs" / cache_key(audio)   # by full path: two folders can hold the same file name
    old = out / "htdemucs" / audio.stem           # where separations before the fixed folder name went
    stem_dir = old if (old / "drums.wav").exists() else out / "htdemucs" / "track"
    drums, rest = stem_dir / "drums.wav", stem_dir / "no_drums.wav"
    if not drums.exists():
        # a fixed folder name: Windows drops a trailing space or dot from a folder ("Song (Live) .wav"),
        # so a folder named after the file could not be found again
        subprocess.run([sys.executable, "-m", "demucs", "-n", "htdemucs", "--two-stems", "drums",
                        "--filename", "track/{stem}.{ext}", "-o", str(out), str(audio)], check=True,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})  # CJK paths in its log
    return drums, rest


LARSNET = VENDOR / "larsnet"   # github.com/polimi-ispl/larsnet, weights CC BY-NC 4.0


def drum_parts(drums: Path) -> dict[str, Path]:
    """LarsNet: split the Demucs drum stem into kick / snare / toms / hihat / cymbals (Wiener
    filtered, alpha 1, so the hi-hat stem carries little ride/crash wash). Cached next to it."""
    out = {s: drums.parent / f"larsnet_{s}.wav" for s in ("kick", "snare", "toms", "hihat", "cymbals")}
    if all(p.exists() for p in out.values()):
        return out
    import soundfile as sf
    import torch
    import yaml
    sys.path.insert(0, str(LARSNET))
    from unet import UNet, UNetUtils
    cfg = yaml.safe_load((LARSNET / "config.yaml").read_text())
    dev = config.device()
    utils = UNetUtils(device=dev)
    x, sr = sf.read(str(drums), dtype="float32", always_2d=True)
    if sr != cfg["global"]["sr"]:
        import librosa
        x = librosa.resample(x.T, orig_sr=sr, target_sr=cfg["global"]["sr"]).T
    x = torch.from_numpy(x.T.copy()).unsqueeze(0).to(dev)
    with torch.no_grad():
        mag, phase = utils.batch_stft(x)
        preds = {}
        for stem in out:
            model = UNet(input_size=(2, cfg[stem]["F"], cfg[stem]["T"]), device=dev)
            ck = torch.load(str(LARSNET / cfg["inference_models"][stem]), map_location=dev, weights_only=True)
            model.load_state_dict(ck["model_state_dict"])
            model.eval()
            _, mask = model(mag)
            preds[stem] = mask * mag
            del model
        total = sum(preds.values()) + 1e-7
        for stem, p in preds.items():
            y = utils.batch_istft(mag * p / total, phase, trim_length=x.size(-1))
            sf.write(str(out[stem]), y.squeeze(0).cpu().numpy().T, cfg["global"]["sr"])
    return out


# ---- 2. tempo evidence ---------------------------------------------------------
BEAT_FPS = 50


def beat_activations(audio: Path) -> tuple[np.ndarray, np.ndarray]:
    """beat_this frame activations (sigmoid, 50 fps): (beat, downbeat)."""
    cache = CACHE / f"{cache_key(audio)}.beatact.npz"
    if cache.exists():
        d = np.load(cache)
        return d["beat"], d["down"]
    import torch
    from beat_this.inference import Audio2Frames
    from beat_this.preprocessing import load_audio
    signal, sr = load_audio(str(audio))
    a2f = Audio2Frames(checkpoint_path="final0", device=config.device())
    b, d = a2f(signal, sr)
    beat, down = torch.sigmoid(b).cpu().numpy(), torch.sigmoid(d).cpu().numpy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, beat=beat, down=down)
    return beat, down


def drum_onsets(drums: Path) -> tuple[np.ndarray, np.ndarray]:
    """Fine-resolution onset times (~3 ms) and strengths from the drum stem."""
    import librosa
    y, sr = librosa.load(str(drums), sr=44100, mono=True)
    hop = 128
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop)
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop), env[frames]


def rough_bpm(beat_act: np.ndarray, onsets: np.ndarray, w: np.ndarray,
              hi: float = 220.0) -> tuple[float, float]:
    """Dominant tempo, folded into (hi/2.2, hi] BPM (default 100-220); used only to pick the
    search octave. A 16th grid at 2x BPM fits the same onsets, so the tempo limit
    (e.g. a ballad never over 100) decides the octave."""
    frames = np.flatnonzero((beat_act[1:-1] > 0.5) & (beat_act[1:-1] >= beat_act[:-2])
                            & (beat_act[1:-1] >= beat_act[2:])) + 1
    ibi = np.diff(frames) / BEAT_FPS
    ibi = ibi[(ibi > 0.2) & (ibi < 1.5)]
    bpm = 60.0 / float(np.median(ibi)) if len(ibi) else 120.0
    raw = bpm
    while bpm < hi / 2.2:
        bpm *= 2
    while bpm > hi:
        bpm /= 2
    # refine on drum onsets (16th grid) within +-8%
    cands = np.arange(bpm * 0.92, bpm * 1.08, 0.1)
    return float(max(cands, key=lambda b: tempo.comb_score(onsets, w, b))), raw


def sample_at(act: np.ndarray, times: np.ndarray, fps: int = BEAT_FPS) -> np.ndarray:
    """Max activation within +-1 frame of each time."""
    idx = np.round(times * fps).astype(int)
    out = np.zeros(len(times))
    for k in (-1, 0, 1):
        j = np.clip(idx + k, 0, len(act) - 1)
        v = np.where((idx + k >= 0) & (idx + k < len(act)), act[j], 0.0)
        out = np.maximum(out, v)
    return out


# ---- 4. drum transcription ---------------------------------------------------
CHUNK, OVERLAP = 300.0, 10.0   # s: ADTOF on long audio


def transcribe(audio: Path) -> list[tuple[float, int]]:
    midi_path = CACHE / f"{cache_key(audio)}.adtof.mid"
    import pretty_midi
    import soundfile as sf
    if not midi_path.exists():
        sys.path.insert(0, str(VENDOR / "ADTOF-pytorch" / "src"))
        from adtof_pytorch import transcribe_to_midi
        dur = sf.info(str(audio)).duration
        if dur <= CHUNK:
            transcribe_to_midi(audio, midi_path, device=config.device())
        else:
            # cuDNN's GRU gives up on very long input (a 15 min medley): overlapping chunks,
            # each keeping the notes of its middle part
            import librosa
            y, sr = librosa.load(str(audio), sr=44100, mono=False)
            y = np.atleast_2d(y)
            out = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(0, is_drum=True)
            a = 0.0
            while a < dur:
                tmp = midi_path.with_suffix(".chunk.wav")
                c0 = max(a - OVERLAP, 0.0)
                sf.write(str(tmp), y[:, int(c0 * sr):int((a + CHUNK + OVERLAP) * sr)].T, sr)
                part = midi_path.with_suffix(".chunk.mid")
                transcribe_to_midi(tmp, part, device=config.device())
                for n in (n for i in pretty_midi.PrettyMIDI(str(part)).instruments for n in i.notes):
                    if a <= n.start + c0 < a + CHUNK:
                        inst.notes.append(pretty_midi.Note(n.velocity, n.pitch, n.start + c0, n.end + c0))
                a += CHUNK
            tmp.unlink()
            part.unlink()
            out.instruments.append(inst)
            out.write(str(midi_path))
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    hits = [(n.start, n.pitch) for inst in pm.instruments for n in inst.notes]
    return sorted(hits + roll_toms(tom_activation(audio), [t for t, p in hits if p == 47]))


GHOST_TH = 0.0       # ADTOF snare threshold of a second, softer pass for ghost notes (0 = off; its own is 0.24)
GHOST_LEVEL = -30.0  # dB under the song's strong snares on the snare stem: quieter is noise, not a ghost note
GHOST_VOL = 0.5      # a ghost note plays at this share of the snare volume


def ghost_snares(audio: Path, hits: list[tuple[float, int]], drums: Path) -> list[float]:
    """Soft snare hits ADTOF's own threshold leaves out (ghost notes): a second pass with the snare
    threshold at GHOST_TH, kept where the LarsNet snare stem has a hit no more than GHOST_LEVEL under
    the song's strong snares and no snare is already within 40 ms."""
    import librosa
    import pretty_midi
    from scipy.ndimage import maximum_filter1d
    path = CACHE / f"{cache_key(audio)}.adtof-sd{GHOST_TH:.2f}.mid"
    if not path.exists():
        sys.path.insert(0, str(VENDOR / "ADTOF-pytorch" / "src"))
        from adtof_pytorch import transcribe_to_midi
        from adtof_pytorch.post_processing import FRAME_RNN_THRESHOLDS
        th = list(FRAME_RNN_THRESHOLDS)
        th[1] = GHOST_TH
        try:
            transcribe_to_midi(audio, path, thresholds=th, device=config.device())
        except RuntimeError as e:     # a very long song can be too much for one pass: no ghost notes then
            print(f"      ghost notes: skipped ({e})")
            return []
    sd = np.array(sorted(t for t, p in hits if p == 38))
    soft = [n.start for i in pretty_midi.PrettyMIDI(str(path)).instruments for n in i.notes if n.pitch == 38]
    soft = [t for t in soft if not len(sd) or np.min(np.abs(sd - t)) > 0.04]
    if not soft or len(sd) < 8:
        return []
    y, sr = librosa.load(str(drum_parts(drums)["snare"]), sr=22050, mono=True)
    env = maximum_filter1d(np.abs(y), int(0.03 * sr))
    lv = lambda t: 20 * np.log10(env[min(int(t * sr), len(env) - 1)] + 1e-9)
    top = np.percentile([lv(t) for t in sd], 90)
    return [t for t in soft if lv(t) - top >= GHOST_LEVEL]


ACT_FPS = 100
TOM_TH = 0.32       # ADTOF's own tom threshold
ROLL_BUSY = 0.25    # tom activation averaging this over +-ROLL_WIN is a roll
ROLL_WIN = 0.15     # s
ROLL_PROM = 0.15    # a roll hit rises this much over the lowest point of the 50 ms before it


def tom_activation(audio: Path) -> np.ndarray:
    """ADTOF's frame-wise tom activation (100 fps), cached; long audio in overlapping chunks."""
    path = CACHE / f"{cache_key(audio)}.tomact.npy"
    if path.exists():
        return np.load(path)
    import soundfile as sf
    sys.path.insert(0, str(VENDOR / "ADTOF-pytorch" / "src"))
    from adtof_pytorch import transcribe_to_midi

    def act(p):
        return transcribe_to_midi(p, path.with_suffix(".mid"), device=config.device(), return_activations=True)[0][:, 2]
    dur = sf.info(str(audio)).duration
    if dur <= CHUNK:
        a = act(audio)
    else:
        import librosa
        y, sr = librosa.load(str(audio), sr=44100, mono=False)
        y = np.atleast_2d(y)
        a = np.zeros(int(dur * ACT_FPS) + 1, dtype=np.float32)
        tmp = path.with_suffix(".chunk.wav")
        s = 0.0
        while s < dur:
            c0 = max(s - OVERLAP, 0.0)
            sf.write(str(tmp), y[:, int(c0 * sr):int((s + CHUNK + OVERLAP) * sr)].T, sr)
            part = act(tmp)
            i0, i1 = int(s * ACT_FPS), min(int((s + CHUNK) * ACT_FPS), len(a))
            k0 = i0 - int(c0 * ACT_FPS)
            seg = part[k0:k0 + i1 - i0]
            a[i0:i0 + len(seg)] = seg
            s += CHUNK
        tmp.unlink()
    np.save(path, a)
    return a


def roll_toms(a: np.ndarray, toms: list[float]) -> list[tuple[float, int]]:
    """ADTOF's peak picker subtracts the last 100 ms' average activation, so in a fast tom roll
    (hits 60-90 ms apart) the hits cancel each other and most vanish, while the activation still
    peaks on each one. Inside rolls (activation averaging ROLL_BUSY over +-ROLL_WIN) take every
    local peak over TOM_TH that rises ROLL_PROM over the dip before it."""
    k = int(ROLL_WIN * ACT_FPS)
    avg = np.convolve(a, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    have = np.array(sorted(toms))
    out = []
    for i in range(5, len(a) - 2):
        if (a[i] >= TOM_TH and a[i] == a[i - 2:i + 3].max() and a[i] - a[i - 5:i].min() >= ROLL_PROM
                and avg[i] >= ROLL_BUSY and (not out or i - out[-1] > 2)):
            out.append(i)
    return [(i / ACT_FPS, 47) for i in out if not len(have) or np.abs(have - i / ACT_FPS).min() > 0.03]


# ---- 5. quantize + lane mapping ------------------------------------------------
@dataclass
class Note:
    q: Fraction       # position in tatums (16ths) from tatum 0 of the grid
    ch: str
    t: float          # original audio time
    wav: str = ""     # sample id when not the lane's default (sounds.dress)
    vol: float = 1.0  # share of the lane's volume (soft cymbal rolls)


STRAIGHT = [Fraction(i, 4) for i in range(4)]      # 16ths (in beats)
TRIPLET = [Fraction(i, 6) for i in range(6)]       # 16th triplets (incl. 8th triplets)
TRIPLET_BIAS = 0.6   # a beat goes on the triplet grid when its hits miss it by under this share of the 16ths' miss


def quantize(hits: list[tuple[float, int]], T: np.ndarray, bar_starts: list[int]) -> list[Note]:
    """Snap hits to the tatum grid; per beat choose straight 16ths or triplets."""
    # continuous tatum coordinate for each hit
    by_beat: dict[tuple[int, int], list[tuple[float, int, float]]] = {}
    bs = np.array(bar_starts)
    for t, pitch in hits:
        if pitch not in ADTOF_TO_CH or t < T[0] or t >= T[-1]:
            continue
        i = int(np.searchsorted(T, t, side="right") - 1)
        x = i + (t - T[i]) / (T[i + 1] - T[i])
        k = int(np.searchsorted(bs, x + 1 / 3, side="right") - 1)  # bar (lean right for late notes)
        if k < 0 or k >= len(bs) - 1:
            continue
        local = x - bs[k]
        beat = int(np.floor(local / 4 + 1 / 12))
        by_beat.setdefault((k, beat), []).append((local / 4 - beat, pitch, t))
    out: dict[tuple[Fraction, str], Note] = {}
    for (k, beat), items in by_beat.items():
        L = bs[k + 1] - bs[k]
        beat_len = Fraction(min(4, L - 4 * beat), 4)  # last beat of an odd bar is shorter
        def err(g):
            return sum(min(abs(f - float(x)) for x in g + [beat_len]) for f, _, _ in items)
        g = TRIPLET if beat_len == 1 and err(TRIPLET) < TRIPLET_BIAS * err(STRAIGHT) else \
            [x for x in STRAIGHT if x < beat_len]
        for f, pitch, t in items:
            frac = min(g + [beat_len], key=lambda x: abs(f - float(x)))
            q = Fraction(int(bs[k])) + 4 * (beat + frac)
            out.setdefault((q, ADTOF_TO_CH[pitch]), Note(q, ADTOF_TO_CH[pitch], t))
    return sorted(out.values(), key=lambda n: (n.q, n.ch))


def refine_cymbals(notes: list[Note]) -> list[Note]:
    """ADTOF has one class for crash+ride. Steady runs of cymbal hits -> ride;
    remaining crashes alternate left/right (a charting choice, not audible)."""
    cy = {n.q for n in notes if n.ch == "16"}
    for n in notes:
        if n.ch == "16" and any(n.q - d in cy for d in (2, 4)) and any(n.q + d in cy for d in (2, 4)):
            n.ch = "19"
    # an extra 16th between two ride hits is the ride too
    ride = {n.q for n in notes if n.ch == "19"}
    left = True
    for n in notes:
        if n.ch != "16":
            continue
        # still counts in the left/right alternation so the other crashes keep their side
        n.ch = "19" if n.q - 1 in ride and n.q + 1 in ride else "1A" if left else "16"
        left = not left
    return notes


HANDS = {"11", "18", "12", "14", "15", "17", "16", "19", "1A"}
# Which hand note to drop first when more than two land together
HAND_DROP_ORDER = ["11", "18", "19", "17", "15", "14", "1A", "16", "12"]
DOUBLE_KICK_BPM = 150.0  # a 156 BPM double-kick song needs two feet
DOUBLE_KICK_RUN = 3   # split between the feet from 3 in a row
DOUBLE_KICK_PAIR_BPM = 120.0   # above this, already 2 kicks a 16th apart are split


def limit_hands(notes: list[Note]) -> list[Note]:
    """Two hands: keep at most two hand notes at the same position (crash/snare win over hi-hat/ride)."""
    by_q: dict[Fraction, list[Note]] = {}
    for n in notes:
        if n.ch in HANDS:
            by_q.setdefault(n.q, []).append(n)
    drop = set()
    for group in by_q.values():
        group.sort(key=lambda n: HAND_DROP_ORDER.index(n.ch))
        for n in group[:max(0, len(group) - 2)]:
            drop.add(id(n))
    return [n for n in notes if id(n) not in drop]


KICK_CLICK_RISE = 6.0   # dB: beater click (1.5-6 kHz) rise on the LarsNet kick stem
KICK_MIN_LEVEL = -20.0  # dB under the song's strong kicks (90th pct)
BEAT_GRID_FIT = 0.7     # a drifting take switches to the beat-locked grid if it fits this much better
BEAT_GRID_BAD = 5.0     # ms: ... and only if its own grid misses the drums by more
KICK_BODY_RISE = 6.0    # dB: low-end (30-150 Hz) rise at a real kick; bass-guitar bleed ~3 dB


def verify_kick_runs(notes: list[Note], drums: Path, bpms: np.ndarray, bar_starts: list[int],
                     log: list[str], every: bool = False) -> list[Note]:
    """Bass guitar leaks into the drum stem and ADTOF calls it kick streams. Inside runs of 3+ kicks (16ths or closer), keep a kick only if the LarsNet kick stem
    shows a real one: a beater click and a body within 20 dB of the song's strong kicks. Only
    below DOUBLE_KICK_BPM: fast metal streams are real and their clicks blur.
    `every` (--kick-check, noisy sources like loud remasters): check every kick, and
    also drop off-beat ones whose low end barely rises - bass guitar on the 8ths. Not by default:
    clean charts would lose real kicks."""
    import librosa
    kicks = sorted((n for n in notes if n.ch == "13"), key=lambda n: n.q)
    runs, cur = [], []
    for n in kicks:
        if cur and not 0 < n.q - cur[-1].q <= 1:
            runs.append(cur)
            cur = []
        cur.append(n)
    runs.append(cur)
    bs = np.array(bar_starts)

    def off_beat(n):
        return (n.q - bs[max(int(np.searchsorted(bs, n.q, side="right")) - 1, 0)]) % 4 != 0
    suspects = [n for n in kicks if off_beat(n)] if every else [n for r in runs if len(r) >= DOUBLE_KICK_RUN
                                    and bpms[min(int(r[0].q), len(bpms) - 1)] <= DOUBLE_KICK_BPM for n in r]
    if not suspects:
        return notes
    y, sr = librosa.load(str(drum_parts(drums)["kick"]), sr=22050, mono=True)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=128))
    f = librosa.fft_frequencies(sr=sr, n_fft=1024)
    body, click = S[(f >= 30) & (f < 150)].mean(0) + 1e-9, S[(f >= 1500) & (f < 6000)].mean(0) + 1e-9
    fps = sr / 128

    def lv(x, t, a, b):
        i = max(int((t + a) * fps), 0)
        return float(20 * np.log10(x[i:max(int((t + b) * fps) + 1, i + 1)].mean()))
    top = np.percentile([lv(body, n.t, 0, 0.03) for n in kicks], 90)
    if every:
        drop = {id(n) for n in suspects
                if lv(body, n.t, 0, 0.03) - lv(body, n.t, -0.03, -0.005) < KICK_BODY_RISE
                or lv(body, n.t, 0, 0.03) - top < KICK_MIN_LEVEL}
    else:
        drop = {id(n) for n in suspects
                if lv(click, n.t, 0, 0.01) - lv(click, n.t, -0.03, -0.005) < KICK_CLICK_RISE
                or lv(body, n.t, 0, 0.03) - top < KICK_MIN_LEVEL}
    if drop:
        print(f"      kick check: {len(drop)} of {len(suspects)} kicks have no real kick on the kick stem -> dropped")
        ms = sorted({int(np.searchsorted(bs, n.q, side="right")) for n in suspects if id(n) in drop})
        log.append(f"kicks dropped as bass-guitar bleed in measures {', '.join(f'{m:03d}' for m in ms)}"
                   f" - check they were not real")
    return [n for n in notes if id(n) not in drop]


KICK_ROLL_FLOOR = -27.0   # dB: the kick stem's low end never drops under this (vs the strong kicks) through a roll beat
KICK_ROLL_PULSE = 0.18    # ... and pulses at the 16th or 32nd rate (share of its mean level); an 808 tail doesn't
KICK_ROLL_32_BPM = 150.0  # 32nd rolls only up to here; faster ones are written as 16ths


def kick_rolls(notes: list[Note], drums: Path, T: np.ndarray, bpms: np.ndarray, bar_starts: list[int],
               log: list[str]) -> list[Note]:
    """A fast double-bass roll under loud cymbals blurs into one long rumble: ADTOF keeps only the
    kicks that line up with the snare, and the kick stem shows no separate beater clicks. Its low
    end (30-150 Hz) gives the roll away: through the beat it never drops far under the song's
    strong kicks (ordinary beats fall 35 dB+ between kicks) and it pulses at the roll's rate.
    Such a beat gets the whole 16th or 32nd stream; double_kick then splits it between the feet."""
    import librosa
    kicks = [n for n in notes if n.ch == "13"]
    if len(kicks) < 8:
        return notes
    y, sr = librosa.load(str(drum_parts(drums)["kick"]), sr=22050, mono=True)
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=64))
    f = librosa.fft_frequencies(sr=sr, n_fft=512)
    body = S[(f >= 30) & (f < 150)].mean(0) + 1e-9
    fps = sr / 64
    top = np.percentile([20 * np.log10(body[int(n.t * fps):int((n.t + 0.03) * fps) + 1].mean()) for n in kicks
                         if int(n.t * fps) < len(body)], 90)
    have = {n.q for n in notes if n.ch in ("13", "1C")}
    triplets = {int(n.q) // 4 * 4 for n in notes if n.q.denominator not in (1, 2)}
    found: dict[int, tuple[int, int]] = {}   # beat -> (bar, 1 for 16ths / 2 for 32nds)
    for b in range(len(bar_starts) - 1):
        for beat in range(bar_starts[b], bar_starts[b + 1] - 3, 4):
            if beat + 4 >= len(T) or beat in triplets:
                continue
            env = body[max(int(T[beat] * fps), 0):min(int(T[beat + 4] * fps), len(body))]
            if len(env) < 16 or 20 * np.log10(np.percentile(env, 10)) - top < KICK_ROLL_FLOOR:
                continue
            tt = np.arange(len(env)) / fps
            step = (T[beat + 4] - T[beat]) / 4

            def pulse(per):
                return float(np.abs(np.mean((env - env.mean()) * np.exp(-2j * np.pi * tt / per))) / env.mean())
            p16, p32 = pulse(step), pulse(step / 2)
            if max(p16, p32) < KICK_ROLL_PULSE:
                continue
            found[beat] = (b, 2 if p32 > p16 and bpms[min(beat, len(bpms) - 1)] <= KICK_ROLL_32_BPM else 1)
    # a roll lasts 2+ beats; one rumbling beat alone is as often a big ringing hit, so it is only flagged
    roll = {q for q in found if q - 4 in found or q + 4 in found}
    out, bars = list(notes), set()
    for beat in sorted(roll):
        b, sub = found[beat]
        for k in range(4 * sub):
            q = Fraction(beat) + Fraction(k, sub)
            if q not in have:
                i = int(q)
                out.append(Note(q, "13", float(T[i] + (T[i + 1] - T[i]) * float(q - i))))
                have.add(q)
                bars.add(b + 1)
    if bars:
        print(f"      kick rolls: {len(out) - len(notes)} kicks filled in under a rumbling kick stem")
        log.append(f"fast kick rolls filled in (16ths / 32nds the transcriber blurred) in measures "
                   f"{', '.join(f'{m:03d}' for m in sorted(bars))} - check the roll's start and end")
    single = sorted({found[q][0] + 1 for q in found if q not in roll})
    if single:
        log.append(f"a single beat sounds like a short kick roll in measures {', '.join(f'{m:03d}' for m in single)}"
                   f" - not filled in; add it if it is one")
    return sorted(out, key=lambda n: (n.q, n.ch))


HAT_MIN_RISE = 4.0     # dB: a real hat hit rises this much on the LarsNet hi-hat stem (6-16 kHz)
HAT_MIN_LEVEL = -20.0  # dB under the song's strong hat hits (90th pct)


SILENT_DRUMS = -40.0   # dB under the song's typical drum level at its notes: no drums playing there


def drop_silent(notes: list[Note], drums: Path, bar_starts: list[int], log: list[str]) -> list[Note]:
    """ADTOF reads the full mix, so a piano or keyboard intro with no drums at all can come out as
    hats, ride and crashes. Any note where the separated drum stem is silent (40 dB or more under
    the drums elsewhere; quiet ghost notes sit 20-30 dB under) is dropped."""
    import soundfile as sf
    if not notes:
        return notes
    y, sr = sf.read(str(drums), always_2d=True)
    hop = sr // 100
    e = np.sqrt(np.add.reduceat(y.mean(1) ** 2, np.arange(0, len(y), hop)) / hop) + 1e-9

    def level(t):
        i = int(t * 100)
        return 20 * np.log10(e[max(i - 2, 0):max(i + 6, 1)].max()) if i < len(e) else -200.0
    lv = {id(n): level(n.t) for n in notes}
    ref = float(np.median(list(lv.values())))
    drop = {k for k, v in lv.items() if v - ref < SILENT_DRUMS}
    if drop:
        bs = np.array(bar_starts)
        ms = sorted({int(np.searchsorted(bs, n.q, side="right")) for n in notes if id(n) in drop})
        print(f"      silent drums: {len(drop)} notes where the drum stem is silent -> dropped")
        log.append(f"notes dropped where the drums are silent (piano / keys intro?) in measures "
                   f"{', '.join(f'{m:03d}' for m in ms)} - check they were not real")
    return [n for n in notes if id(n) not in drop]


def verify_hats(notes: list[Note], drums: Path, bar_starts: list[int], log: list[str]) -> list[Note]:
    """ADTOF hears sharp piano / keyboard notes in the full mix as hi-hats. A hat is dropped only
    when nothing says it is one: no attack and a level far under the song's strong hats on the
    LarsNet hi-hat stem (piano "hats": no rise, 30-60 dB under; real ones rise 7-24 dB), no attack
    on the whole drum stem either, and the rest of the band louder than the drums there
    (programmed pop hats the hi-hat stem misses still show on the drum stem)."""
    import librosa
    hh = [n for n in notes if n.ch in ("11", "18")]
    if len(hh) < 8:
        return notes

    def high_band(path):
        y, sr = librosa.load(str(path), sr=44100, mono=True)
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=128))
        f = librosa.fft_frequencies(sr=sr, n_fft=1024)
        return S[(f >= 6000) & (f < 16000)].mean(0) + 1e-9
    H, D, R = high_band(drum_parts(drums)["hihat"]), high_band(drums), high_band(drums.parent / "no_drums.wav")
    fps = 44100 / 128

    def level(E, t, a, b):
        i, j = int((t + a) * fps), int((t + b) * fps) + 1
        return float(20 * np.log10(E[max(i, 0):max(j, i + 1)].mean())) if i < len(E) else -200.0

    def rise(E, t):
        return level(E, t, 0, 0.03) - level(E, t, -0.035, -0.005)
    top = np.percentile([level(H, n.t, 0, 0.03) for n in hh], 90)
    drop = {id(n) for n in hh if rise(H, n.t) < HAT_MIN_RISE and level(H, n.t, 0, 0.03) - top < HAT_MIN_LEVEL
            and rise(D, n.t) < HAT_MIN_RISE and level(D, n.t, 0, 0.03) < level(R, n.t, 0, 0.03)}
    if drop:
        bs = np.array(bar_starts)
        ms = sorted({int(np.searchsorted(bs, n.q, side="right")) for n in hh if id(n) in drop})
        print(f"      hat check: {len(drop)} of {len(hh)} hi-hats have no hat on the hi-hat stem -> dropped")
        log.append(f"hi-hats dropped as other instruments (piano, keys) in measures {', '.join(f'{m:03d}' for m in ms)}"
                   f" - check they were not real")
    return [n for n in notes if id(n) not in drop]


def left_foot_busy(notes: list[Note]) -> list[Note]:
    """No hi-hat pedal (1B) within a 16th of a left-pedal kick, unless a hand hat is there."""
    lbd = np.array(sorted(float(n.q) for n in notes if n.ch == "1C"))
    hats = {n.q for n in notes if n.ch == "11"}
    return [n for n in notes if not (n.ch == "1B" and n.q not in hats and len(lbd)
                                     and np.abs(lbd - float(n.q)).min() <= 1)]


def double_kick(notes: list[Note], bar_starts: list[int], bpms: np.ndarray) -> list[Note]:
    """Rule: DOUBLE_KICK_RUN+ kicks in a row (16ths or closer) are split between the feet, at
    any tempo; above DOUBLE_KICK_PAIR_BPM already a pair is, so no foot plays two 16ths in a row.
    Above DOUBLE_KICK_BPM an 8th-triplet stream (3+ in a row) is split too.
    Straight 16ths: the in-between 16ths ("e"/"a") go to the left bass pedal (1C); triplet and
    32nd runs alternate R L R."""
    kicks = sorted((n for n in notes if n.ch == "13"), key=lambda n: n.q)
    bs = np.array(bar_starts)
    run: list[Note] = []
    bpm_at = lambda q: bpms[min(int(q), len(bpms) - 1)]

    def joins(n):
        gap = n.q - run[-1].q
        return 0 < gap <= 1 or (gap == Fraction(4, 3) and bpm_at(n.q) > DOUBLE_KICK_BPM)

    def flush():
        fast = bpm_at(run[0].q) > DOUBLE_KICK_PAIR_BPM if run else False
        slow_gap = any(b.q - a.q > 1 for a, b in zip(run, run[1:]))     # 8th triplets: a pair is fine
        if len(run) < (2 if fast and not slow_gap else DOUBLE_KICK_RUN):
            return
        if all(n.q.denominator == 1 for n in run):
            for n in run:
                k = int(np.searchsorted(bs, n.q, side="right") - 1)
                if (n.q - bs[max(k, 0)]) % 2 == 1:
                    n.ch = "1C"
        else:
            for n in run[1::2]:
                n.ch = "1C"
    for n in kicks:
        if run and not joins(n):
            flush()
            run = []
        run.append(n)
    flush()
    return notes


def refine_toms(notes: list[Note], drums: Path) -> list[Note]:
    """ADTOF has one tom class. Split by pitch: high -> HT(14), mid -> LT(15), low -> FT(17)."""
    import librosa
    toms = [n for n in notes if n.ch == "14"]
    if len(toms) < 2:
        return notes
    y, sr = librosa.load(str(drums), sr=22050, mono=True)
    pitch = {}
    for n in toms:
        a, b = int((n.t + 0.02) * sr), int((n.t + 0.12) * sr)
        seg = y[max(a, 0):max(b, 1)]
        if len(seg) < 256:
            continue
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=8192))
        f = np.fft.rfftfreq(8192, 1 / sr)
        band = (f >= 60) & (f <= 500)
        pitch[id(n)] = float(f[band][np.argmax(spec[band])])
    if not pitch:
        return notes
    logp = np.log2(np.array(list(pitch.values())))
    lo, hi = logp.min(), logp.max()
    if hi - lo < 0.25:  # under ~3 semitones apart: one tom
        return notes
    for n in toms:
        if id(n) in pitch:
            x = (np.log2(pitch[id(n)]) - lo) / (hi - lo)
            n.ch = "14" if x > 2 / 3 else "15" if x > 1 / 3 else "17"
    return notes


def spread_flat_fills(notes: list[Note], drums: Path) -> list[Note]:
    """refine_toms can put a whole fill on one lane (e.g. 9 LT in a row) while the
    drums clearly walk down the kit. For runs of 6+ toms (gaps <= an 8th) that sit 80%+ on one
    lane and whose ring pitch falls 2+ semitones from the first to the last three hits, walk the
    kit down in time order (thirds: HT, LT, FT), ending on the kick.
    Rising or steady tom grooves are left as they are."""
    import librosa
    toms = sorted((n for n in notes if n.ch in ("14", "15", "17")), key=lambda n: n.q)
    if len(toms) < 6:
        return notes
    y, sr = librosa.load(str(drums), sr=22050, mono=True)
    hop = 64
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=60, n_bins=36, bins_per_octave=12))
    fr = librosa.cqt_frequencies(36, fmin=60)
    runs, cur = [], [toms[0]]
    for n in toms[1:]:
        if n.q - cur[-1].q <= 2:
            cur.append(n)
        else:
            runs.append(cur)
            cur = [n]
    runs.append(cur)
    for run in runs:
        lanes = [n.ch for n in run]
        if len(run) < 6 or max(map(lanes.count, set(lanes))) < 0.8 * len(run):
            continue
        r = []
        for n in run:
            a, b = int((n.t + 0.03) * sr / hop), int((n.t + 0.09) * sr / hop)
            if b > C.shape[1]:
                break
            r.append(12 * np.log2(fr[int(np.argmax(C[:, a:b].mean(1)))]))
        if len(r) < len(run) or np.mean(r[:3]) - np.mean(r[-3:]) < 2 or max(r) - min(r) < 3:
            continue
        # the fill ends on the feet: a tom on the same 16th as a kick gives way to
        # it, and a last hit below every other tom after that kick is a kick too
        kick_q = {n.q for n in notes if n.ch == "13"}
        toms_left = []
        for i, (n, p) in enumerate(zip(run, r)):
            if n.q in kick_q:
                n.ch = "drop"
            elif i == len(run) - 1 and p < min(r[:-1]) - 1 and any(run[0].q <= k < n.q for k in kick_q):
                n.ch = "13"
            else:
                toms_left.append(n)
        # lanes walk down the kit in time order, in thirds (7 toms -> HT HT HT LT LT FT FT)
        for lane, part in zip(("14", "15", "17"), np.array_split(np.array(toms_left, dtype=object), 3)):
            for n in part:
                n.ch = lane
    return [n for n in notes if n.ch != "drop"]


def parse_measures(spec: str) -> set[int]:
    """'88-91, 120' -> {88, 89, 90, 91, 120}"""
    out = set()
    for part in filter(None, (p.strip() for p in spec.replace(";", ",").split(","))):
        lo, _, hi = part.partition("-")
        out |= set(range(int(lo), int(hi or lo) + 1))
    return out


def backbeat_only(notes: list[Note], bar_starts: list[int], measures: set[int]) -> list[Note]:
    """Hint --backbeat: in these measures the snare is on beats 2 and 4 only. Over a loud
    double-kick chorus ADTOF hears snares on most 16ths and the drum
    stem can't tell bleed from real hits there."""
    bs = np.array(bar_starts)

    def keep(n):
        k = int(np.searchsorted(bs, n.q, side="right") - 1)
        return n.ch != "12" or k + 1 not in measures or n.q - bs[k] in (4, 12)
    return [n for n in notes if keep(n)]


def snare_on_24(notes: list[Note], bar_starts: list[int]) -> list[Note]:
    """--snare-24: in a groove bar whose snares all sit on beats
    (ADTOF hears the crash/kick on 1 and 3 as snare too) only beats 2 and 4 keep the snare.
    Bars with any other snare are fills and stay."""
    bs = np.array(bar_starts)
    bar_of = lambda q: int(np.searchsorted(bs, q, side="right") - 1)
    sn: dict[int, set] = {}
    for n in notes:
        if n.ch == "12":
            k = bar_of(n.q)
            sn.setdefault(k, set()).add(n.q - bs[k])
    groove = {k for k, qs in sn.items() if 0 <= k < len(bs) - 1 and bs[k + 1] - bs[k] == 16
              and {4, 12} <= qs <= {0, 4, 8, 12}}
    out = [n for n in notes if not (n.ch == "12" and bar_of(n.q) in groove and n.q - bs[bar_of(n.q)] in (0, 8))]
    print(f"      snare on 2 and 4: {len(notes) - len(out)} beat-1/3 snares dropped in {len(groove)} groove bars")
    return out


def crash_hints(notes: list[Note], bar_starts: list[int], T: np.ndarray, spec: str) -> list[Note]:
    """--crash: "39" puts a crash on beat 1 of measure 39 (its hi-hat becomes the crash); "4*" turns
    every hi-hat of measure 4 into a crash, alternating CY / LC on hits a 16th apart (intro hits, a
    chorus start)."""
    lag = float(np.median([n.t - T[int(n.q)] for n in notes if n.q == int(n.q) and int(n.q) < len(T)]))
    out, done = list(notes), 0
    for tok in filter(None, spec.replace(" ", "").split(",")):
        m, every = int(tok.rstrip("*")), tok.endswith("*")
        b0, b1 = bar_starts[m - 1], bar_starts[m]
        hats = sorted((n for n in out if n.ch in ("11", "18") and b0 <= n.q < b1 and (every or n.q == b0)),
                      key=lambda n: n.q)
        last = None
        for n in hats:
            n.ch = "1A" if last is not None and last.ch == "16" and n.q - last.q < 2 else "16"
            last = n
            done += 1
        if not hats and not any(n.ch in ("16", "1A") and n.q == b0 for n in out):
            out.append(Note(Fraction(b0), "16", T[b0] + lag))
            done += 1
    print(f"      crash hints: {done} crashes")
    return sorted(out, key=lambda n: (n.q, n.ch))


ROLL_TOP = 0.6      # cymbal roll volume at the song's loud-cymbal level ...
ROLL_FLOOR = 0.1    # ... and its softest share; in between it follows the stem level (dB / 30)
ROLL_SEEN = -25.0   # check list: cymbal stem this loud (dB under the song's loud cymbals) ...
ROLL_BUMPS = 8.0    # ... with this many small stick bumps per beat and no hand note: a roll?


def cymbal_env(drums: Path) -> tuple[np.ndarray, int, float]:
    """5 ms RMS level (dB) of the LarsNet cymbal stem, its frame rate and its loud level."""
    import soundfile as sf
    x, sr = sf.read(str(drum_parts(drums)["cymbals"]))
    x = x.mean(1) if x.ndim > 1 else x
    hop = sr // 200
    n = len(x) // hop
    e = 20 * np.log10(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12))
    return e, 200, float(np.percentile(e, 95))


def parse_roll_spans(spec: str, bar_starts: list[int]) -> list[tuple[int, int]]:
    """"9.5; 63-66; 112; 114.5-115.5" -> 16th ranges: a measure, part of one from a quarter on
    (9.5 = its second half) or a run of measures; it runs to the end of its last measure, or to
    a quarter inside it (115.5 = the middle of measure 115)."""
    def at(x: float) -> int:
        m = int(x)
        b0, b1 = bar_starts[m - 1], bar_starts[m]
        return b0 + int(round((x - m) * (b1 - b0)))
    out = []
    for tok in filter(None, (p.strip() for p in spec.replace(",", ";").split(";"))):
        lo, _, hi = tok.partition("-")
        b = float(hi or int(float(lo)))
        out.append((at(float(lo)), bar_starts[int(b)] if b == int(b) else at(b)))
    return out


def cymbal_roll_hints(notes: list[Note], bar_starts: list[int], T: np.ndarray, spec: str,
                      drums: Path) -> list[Note]:
    """--cymbal-roll: a cymbal roll the transcriber can't see
    (a wash with no clear strokes) is written as 16ths, CY on the 8ths and LC on the off-16ths,
    both hands on it (other hand notes there go; kicks and pedal stay). Each half bar gets one
    volume, following how loud the cymbal stem is there: a swell from silence steps up."""
    e, fps, ref = cymbal_env(drums)
    lag = float(np.median([n.t - T[int(n.q)] for n in notes if n.q == int(n.q) and int(n.q) < len(T)]))
    bs = np.array(bar_starts)
    out, added = list(notes), 0
    for q0, q1 in parse_roll_spans(spec, bar_starts):
        out = [n for n in out if not (q0 <= n.q < q1 and n.ch not in ("13", "1C", "1B"))]
        q = q0
        while q < q1:
            k = int(np.searchsorted(bs, q, side="right") - 1)
            half = (bs[k + 1] - bs[k]) / 2
            h1 = min(q1, int(bs[k] + half * (1 + int((q - bs[k]) >= half))))   # end of this half bar
            seg = e[int(T[q] * fps):max(int(T[min(h1, len(T) - 1)] * fps), int(T[q] * fps) + 1)]
            lvl = float(np.median(seg)) - ref
            share = round(float(np.clip(ROLL_TOP * 10 ** (lvl / 30), ROLL_FLOOR, ROLL_TOP)) * 20) / 20
            for p in range(q, h1):
                out.append(Note(Fraction(p), "16" if (p - bs[k]) % 2 == 0 else "1A", T[p] + lag, vol=share))
                added += 1
            q = h1
    print(f"      cymbal rolls: {added} roll notes written")
    return sorted(out, key=lambda n: (n.q, n.ch))


def unseen_rolls(notes: list[Note], bar_starts: list[int], T: np.ndarray, drums: Path,
                 log: list[str]) -> None:
    """CHECK items: 2+ beats where the cymbal stem keeps sounding, loud and full of small stick
    bumps, with no hand note charted - maybe a cymbal roll (written with --cymbal-roll)."""
    from scipy.signal import find_peaks
    e, fps, ref = cymbal_env(drums)
    hand = {int(n.q) for n in notes if n.ch not in ("13", "1C", "1B")}
    found = []
    run = []
    for q in range(bar_starts[0], bar_starts[-1] - 3, 4):
        if q + 4 >= len(T):
            break
        seg = e[int(T[q] * fps):int(T[q + 4] * fps)]
        roll = (len(seg) > 5 and np.median(seg) - ref >= ROLL_SEEN and not hand & set(range(q, q + 4))
                and len(find_peaks(seg, prominence=1.5)[0]) >= ROLL_BUMPS)
        if roll:
            run.append(q)
        if run and (not roll or q + 4 >= bar_starts[-1] - 3):
            if len(run) >= 2:
                found.append(int(np.searchsorted(bar_starts, run[0], side="right")))
            run = []
    if found:
        log.append("cymbal keeps sounding with no notes (a cymbal roll?) in measures "
                   + ", ".join(f"{m:03d}" for m in sorted(set(found)))
                   + " - add it with --cymbal-roll if it is one")


CRASH_RUN = 3   # crashes in a row, a beat or less apart, from here on alternate right / left


def crash_runs(notes: list[Note]) -> list[Note]:
    """Rule: a run of crashes, each a beat
    or less after the last, is split over both cymbals R L R L: CY, LC, CY ... A crash on the same
    16th as a hand hi-hat or the ride stays on CY (cymbal_pairs)."""
    at: dict[Fraction, set[str]] = {}
    for n in notes:
        at.setdefault(n.q, set()).add(n.ch)
    cys = sorted((n for n in notes if n.ch == "16"), key=lambda n: n.q)
    runs: list[list[Note]] = []
    for n in cys:
        if runs and n.q - runs[-1][-1].q <= 4 and not any(
                "1A" in at[q] for q in at if runs[-1][-1].q < q < n.q):
            runs[-1].append(n)
        else:
            runs.append([n])
    moved = 0
    for r in runs:
        if len(r) >= CRASH_RUN:
            for n in r[1::2]:
                if not at[n.q] & {"1A", "11", "18", "19"}:
                    n.ch = "1A"
                    moved += 1
    print(f"      crash runs: {sum(len(r) >= CRASH_RUN for r in runs)} runs, {moved} crashes moved to LC (R L R L)")
    return notes


ROLL_HANDS = {"16", "1A", "19", "11", "18", "12", "14", "15", "17"}


def roll_into_cymbal(notes: list[Note], starts: list[int]) -> list[Note]:
    """Rule: a tom or snare roll that ends on a crash
    plus a drum can't be played when the last two roll notes are closer than a 16th (and under 80 ms:
    a 32nd at 132 BPM). Both hands can't leave the roll for the cymbal and the drum, so the drum
    on the crash goes, and the roll notes in the 16th before the crash move onto that drum so the
    fill still lands there (LT LT LT LT -> LT LT FT FT, CY)."""
    at: dict[Fraction, list[Note]] = {}
    for n in notes:
        at.setdefault(n.q, []).append(n)
    hand_q = sorted(q for q, ns in at.items() if any(n.ch in ROLL_HANDS for n in ns))
    drop, fixed = set(), []
    for i, q in enumerate(hand_q[1:], 1):
        ns = at[q]
        cym = [n for n in ns if n.ch in ("16", "1A", "19")]
        drums = [n for n in ns if n.ch in ("12", "14", "15", "17")]
        p = hand_q[i - 1]
        if not (cym and drums) or q - p >= 1 or cym[0].t - at[p][0].t >= 0.08:
            continue
        if i < 2 or p - hand_q[i - 2] >= 1:   # one quick note is not a roll (SD 23/6 -> CY+SD is fine)
            continue
        drop |= {id(n) for n in drums}
        if len(drums) == 1:
            for r in (n for qq in hand_q[:i] if q - 1 <= qq < q for n in at[qq] if n.ch in ("12", "14", "15", "17")):
                r.ch = drums[0].ch
        fixed.append(bisect.bisect_right(starts, q))
    print(f"      roll into cymbal: {len(fixed)} crash+drum after a 32nd roll, drum dropped"
          + (f" (measures {' '.join(f'{m:03d}' for m in fixed)})" if fixed else ""))
    return [n for n in notes if id(n) not in drop]


SCORE_LANE = {"HH": "11", "HO": "18", "LP": "1B", "SD": "12", "BD": "13", "HT": "14", "LT": "15",
              "FT": "17", "CY": "16", "RD": "19", "LC": "1A", "CN": "1A"}


def score_notes(path: Path, T: np.ndarray, bar_starts: list[int], first: int) -> list[Note]:
    """--score: notes read from a drum score (e.g. read off a drum-cover video). One bar per
    line, "12: HH 0 1/4 1/2; SD 1 3; BD 0 1 2 3+1/3" (beats from 0), "=5" repeats bar 5, an empty
    bar is a rest. Bar 1 goes into measure `first`; the audio only gives the grid."""
    bars: dict[int, list[tuple[str, Fraction]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        num, body = (x.strip() for x in line.split(":", 1))
        if body.startswith("="):
            bars[int(num)] = list(bars[int(body[1:])])
            continue
        bars[int(num)] = [(lane, sum((Fraction(x) for x in p.split("+")), Fraction(0)))
                          for part in body.split(";") if part.strip()
                          for lane, *pos in [part.split()] for p in pos]
    out = []
    for num, items in bars.items():
        k = first - 2 + num
        if k + 1 >= len(bar_starts):
            raise SystemExit(f"score bar {num} is past the end of the song's grid")
        b0, L = bar_starts[k], bar_starts[k + 1] - bar_starts[k]
        for lane, b in items:
            q = b0 + b * 4
            if not 0 <= b * 4 < L:
                raise SystemExit(f"score bar {num}: {lane} at beat {b} is outside the bar")
            out.append(Note(q, SCORE_LANE[lane], float(np.interp(float(q), np.arange(len(T)), T))))
    print(f"      score {Path(path).name}: {len(bars)} bars, {len(out)} notes from measure {first:03d}")
    return sorted(out, key=lambda n: (n.q, n.ch))


CRLF = chr(13) + chr(10)


def add_video(path: Path, video: Path, out_dir: Path) -> None:
    """--video: the source video as BGA (#AVI01, channel 54 in the BGM chip's slot, so picture and
    music start together). DTXManiaNX plays clips through DirectShow: anything but .wmv/.avi is
    converted to WMV2 without sound."""
    name = video.name
    if video.suffix.lower() not in (".wmv", ".avi"):
        name = video.stem + ".wmv"
        if not (out_dir / name).exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-an", "-vf",
                            add_mv.FRAME, "-c:v", "wmv2", "-b:v", "3500k", str(out_dir / name)], check=True)
    elif video.resolve() != (out_dir / name).resolve():
        shutil.copy2(video, out_dir / name)
    lines = path.read_text(encoding="cp932").splitlines()
    bgm = next(l for l in lines if re.match(r"#\d{3}01:", l))
    lines.insert(next(i for i, l in enumerate(lines) if l.startswith("#BGMWAV")) + 1, f"#AVI01: {name}")
    lines.append(bgm[:4] + "54" + bgm[6:])
    path.write_bytes((CRLF.join(lines) + CRLF).encode("cp932", errors="replace"))
    print(f"      BGA: {name} starts with the BGM")


GROOVE_MATCH = 0.75   # F1 of a bar's kick/snare/hat against the groove before it is written as the groove
GROOVE_LANES = {"11": "hat", "18": "hat", "12": "12", "13": "13"}


def snap_grooves(notes: list[Note], T: np.ndarray, bar_starts: list[int], groove: dict[str, str]) -> tuple[list[Note], list[int]]:
    """Note design (--groove): a 4/4 bar whose kick/snare/hat
    are close to the song's groove is charted exactly as the groove (stray 16ths, ghost notes and
    open hats go). Crashes and the bar's other lanes stay; a beat with toms or extra snares is a
    fill and stays.
    Runs after the open hi-hat rule, so the groove's hats stay as drawn."""
    want = {(GROOVE_LANES.get(ch, ch), i) for ch, s in groove.items() for i, c in enumerate(s) if c != "."
            and ch in GROOVE_LANES}
    bs = np.array(bar_starts)
    per: dict[int, list[Note]] = {}
    for n in notes:
        per.setdefault(int(np.searchsorted(bs, n.q, side="right") - 1), []).append(n)
    out, snapped = [], []
    for k, ns in sorted(per.items()):
        if not 0 <= k < len(bs) - 1 or bs[k + 1] - bs[k] != 16:
            out += ns
            continue
        got = {(GROOVE_LANES[n.ch], int(n.q - bs[k])) for n in ns
               if n.ch in GROOVE_LANES and (n.q - bs[k]).denominator == 1}
        f1 = 2 * len(got & want) / (len(got) + len(want))
        # a beat with toms, or snares off the groove (a pickup like bar 28's last beat), is a fill
        fill = {int((n.q - bs[k]) // 4) for n in ns if n.ch in ("14", "15", "17")
                or (n.ch == "12" and groove.get("12", "")[int(n.q - bs[k]):][:1] in ("", "."))}
        if f1 < GROOVE_MATCH:
            out += ns
            continue
        snapped.append(k + 1)
        for n in ns:
            if int((n.q - bs[k]) // 4) in fill or n.ch not in tuple(GROOVE_LANES) + ("1B", "1C"):
                out.append(n)
        for ch, s in groove.items():
            for i, c in enumerate(s):
                if c != "." and i // 4 not in fill:
                    out.append(Note(Fraction(int(bs[k]) + i), ch, float(T[min(bs[k] + i, len(T) - 1)])))
    return sorted(out, key=lambda n: (n.q, n.ch)), snapped


def cymbal_pairs(notes: list[Note]) -> list[Note]:
    """Design rules:
    * CY + RD together is not allowed: the crash goes to LC (LC + RD).
    * HH + LC together should be rare: the crash goes to CY (unless RD is there too)."""
    at: dict[Fraction, set[str]] = {}
    for n in notes:
        at.setdefault(n.q, set()).add(n.ch)
    out = []
    for n in notes:
        chs = at[n.q]
        if n.ch == "16" and "19" in chs and "1A" not in chs:
            n = Note(n.q, "1A", n.t, n.wav)
        elif n.ch == "1A" and chs & {"11", "18"} and not chs & {"16", "19"}:
            n = Note(n.q, "16", n.t, n.wav)
        out.append(n)
    return out


HAT_CRASH_BPM = 130.0   # from here on a drum a 16th before hi-hat + crash leaves no hand for the hat


def hat_under_crash(notes: list[Note], bpms: np.ndarray) -> list[Note]:
    """Snare (or tom) on a 16th, then hi-hat + crash together: at speed the hand coming off the
    drum can't reach the hat, so the crash stays and the hat (and its pedal close) goes."""
    at: dict[Fraction, set[str]] = {}
    for n in notes:
        at.setdefault(n.q, set()).add(n.ch)
    drop = {q for q, chs in at.items()
            if chs & {"11", "18"} and chs & {"16", "1A"} and at.get(q - 1, set()) & {"12", "14", "15", "17"}
            and bpms[min(int(q), len(bpms) - 1)] >= HAT_CRASH_BPM}
    return [n for n in notes if not (n.q in drop and n.ch in ("11", "18", "1B"))]


def ride_on_cymbal(notes: list[Note]) -> list[Note]:
    """Game-chart style (as official GITADORA / drummania charts write it): no ride lane - the ride
    is played on the right cymbal (CY); a crash on the same 16th moves to the left cymbal (LC)."""
    ride = {n.q for n in notes if n.ch == "19"}
    out = []
    for n in notes:
        if n.ch == "19":
            n = Note(n.q, "16", n.t, n.wav)
        elif n.ch == "16" and n.q in ride:
            n = Note(n.q, "1A", n.t, n.wav)
        out.append(n)
    seen, uniq = set(), []
    for n in out:
        if (n.q, n.ch) not in seen:
            seen.add((n.q, n.ch)); uniq.append(n)
    return uniq


SNARE_ROLL = 4   # 16th snares in a row from here on are a two-handed roll


def snare_rolls(notes: list[Note], bar_starts: list[int]) -> list[str]:
    """Rules for a run of 16th snares:
    * 4+ in a row is a roll with both hands: no other hand note from its first snare to its last.
    * 3 in a row against the 8th-note hats (x-x-x, hat on the middle): the snare under the hat
      goes. Hats off the 8ths inside the run follow the snares (bleed) and go instead.
    Removed hats take their pedal close (1B) with them. Edits notes in place; returns the log."""
    bs = np.array(bar_starts)
    at: dict[int, list[Note]] = {}
    for n in notes:
        if n.q == int(n.q):
            at.setdefault(int(n.q), []).append(n)
    snares = sorted(q for q, g in at.items() if any(n.ch == "12" for n in g))
    runs: list[list[int]] = []
    for q in snares:
        if runs and q == runs[-1][-1] + 1:
            runs[-1].append(q)
        else:
            runs.append([q])
    drop, log, rolls, cut = set(), [], 0, 0
    for r in runs:
        if len(r) >= SNARE_ROLL:
            rolls += 1
            for q in r:
                for n in at[q]:
                    if n.ch in HANDS and n.ch != "12":
                        drop.add(id(n))
                        if n.ch in ("11", "18"):
                            drop |= {id(x) for x in at[q] if x.ch == "1B"}
        elif len(r) == 3:
            hats = [q for q in r if any(n.ch in ("11", "18", "19") for n in at[q])]
            if not hats:
                continue
            k = int(np.searchsorted(bs, r[0], side="right") - 1)
            on8 = all((q - bs[max(k, 0)]) % 2 == 0 for q in hats)
            for q in hats:
                for n in at[q]:
                    if (n.ch == "12") if on8 else (n.ch in ("11", "18", "19", "1B")):
                        drop.add(id(n))
            cut += 1
            log.append(f"{k + 1:03d}:{'snare' if on8 else 'hat'}")
    notes[:] = [n for n in notes if id(n) not in drop]
    print(f"      snare runs: {rolls} rolls cleared of other hand notes, {cut} 3-runs fixed "
          f"({' '.join(log[:12])}{' ...' if len(log) > 12 else ''})")
    return log


HAT_ROLL = 5          # 16th hi-hats in a row from here on are played with both hands ...
HAT_ROLL_BPM = 0.0    # ... at any tempo: the snare in the run stays and the hat under it goes
ROLL_BAR_SNARES = 8   # a bar with this many snares, some a 16th apart, is a snare roll bar


def hat_rolls(notes: list[Note], bar_starts: list[int], T: np.ndarray, bpms: np.ndarray) -> list[Note]:
    """Rules:
    * 5+ hi-hats in a row on 16ths are both hands rolling the hat: where a snare lands in the run
      the snare stays and the hi-hat goes (bar 45).
    * A bar with 8+ snares, some a 16th apart, has both hands on the snare: no hand hi-hat; the
      left foot keeps the time with a pedal chip on every beat (bar 46)."""
    bs = np.array(bar_starts)
    bar_of = lambda q: int(np.searchsorted(bs, q, side="right") - 1)
    at: dict[int, list[Note]] = {}
    for n in notes:
        if n.q == int(n.q):
            at.setdefault(int(n.q), []).append(n)
    has = lambda q, chs: any(n.ch in chs for n in at.get(q, []))
    drop = set()
    hats = sorted(q for q in at if has(q, ("11", "18")))
    runs: list[list[int]] = []
    for q in hats:
        if runs and q == runs[-1][-1] + 1:
            runs[-1].append(q)
        else:
            runs.append([q])
    cut = 0
    for r in runs:
        if len(r) >= HAT_ROLL and bpms[min(r[0], len(bpms) - 1)] >= HAT_ROLL_BPM:
            for q in r:
                if has(q, ("12",)):
                    drop |= {id(n) for n in at[q] if n.ch in ("11", "18", "1B")}
                    cut += 1
    lag = float(np.median([n.t - T[int(n.q)] for n in notes if n.q == int(n.q) and int(n.q) < len(T)]))
    add, roll_bars = [], []
    for k in range(len(bs) - 1):
        sq = [q for q in range(int(bs[k]), int(bs[k + 1])) if has(q, ("12",))]
        if len(sq) < ROLL_BAR_SNARES or not any(b - a == 1 for a, b in zip(sq, sq[1:])):
            continue
        hq = [q for q in range(int(bs[k]), int(bs[k + 1])) if has(q, ("11", "18"))]
        if not hq:
            continue
        roll_bars.append(k + 1)
        for q in range(int(bs[k]), int(bs[k + 1])):
            drop |= {id(n) for n in at.get(q, []) if n.ch in ("11", "18", "1B")}
        for b in range(int(bs[k]), int(bs[k + 1]), 4):
            if b < len(T):
                add.append(Note(Fraction(b), "1B", T[b] + lag))
    out = sorted([n for n in notes if id(n) not in drop] + add, key=lambda n: (n.q, n.ch))
    print(f"      hat rolls: {cut} hi-hats under a snare dropped; snare roll bars (hands off the hat, "
          f"pedal on the beats): {' '.join(f'{m:03d}' for m in roll_bars[:15]) or 'none'}")
    return out


def foot_hat_spacing(notes: list[Note], bar_starts: list[int], T: np.ndarray) -> list[Note]:
    """Left-foot hi-hat (1B) can kick 4ths and 8ths, never faster
    (no 4th-16th-8th). A bar with two or more pedal gaps under an 8th is sorted
    beat by beat: where the ride plays (2+ ride hits, or one against a single pedal chip) the
    hands are on the ride and the foot keeps a plain quarter on the beat; elsewhere the
    pedal set was really the hand hi-hat and moves to HH, taking over a lone ride hit on the
    same 16th (185 beat 2). A single tight pair loses its later (off-8th) chip."""
    bs = np.array(bar_starts)
    bar_of = lambda q: int(np.searchsorted(bs, q, side="right") - 1)
    lp = sorted((n for n in notes if n.ch == "1B"), key=lambda n: n.q)
    hands: dict[Fraction, int] = {}
    hat_at, ride = set(), {}
    for n in notes:
        if n.ch in HANDS:
            hands[n.q] = hands.get(n.q, 0) + 1
        if n.ch in ("11", "18"):
            hat_at.add(n.q)
        if n.ch == "19":
            ride[n.q] = n
    tight: dict[int, list[Note]] = {}
    for a, b in zip(lp, lp[1:]):
        if b.q - a.q < 2:
            tight.setdefault(bar_of(b.q), []).append(b)
    lag = float(np.median([n.t - T[int(n.q)] for n in notes if n.q == int(n.q) and int(n.q) < len(T)]))
    drop, add, moved, quarters, cut = set(), [], 0, 0, 0
    for k, late in tight.items():
        if len(late) < 2:
            drop.add(id(late[0]))
            cut += 1
            continue
        for b0 in range(int(bs[k]), int(bs[k + 1]), 4):
            beat = [n for n in lp if b0 <= n.q < b0 + 4]
            rides = [q for q in ride if b0 <= q < b0 + 4]
            if not beat:
                continue
            if len(rides) >= 2 or (rides and len(beat) == 1):
                drop |= {id(n) for n in beat if n.q != b0}
                if all(n.q != b0 for n in beat) and b0 < len(T):
                    add.append(Note(Fraction(b0), "1B", T[b0] + lag))
                quarters += 1
                continue
            for n in beat:
                if n.q in hat_at or hands.get(n.q, 0) - (n.q in ride) >= 2:
                    drop.add(id(n))
                    continue
                if n.q in ride and len(rides) == 1:
                    drop.add(id(ride[n.q]))
                else:
                    hands[n.q] = hands.get(n.q, 0) + 1
                n.ch = "11"
                moved += 1
    out = sorted([n for n in notes if id(n) not in drop] + add, key=lambda n: (n.q, n.ch))
    print(f"      foot hi-hat: {len(tight)} bars with pedal chips under an 8th apart; {quarters} ride beats "
          f"-> foot on the quarter, {moved} chips moved to HH, {cut} single tight chips dropped "
          f"(bars {' '.join(f'{k + 1:03d}' for k in sorted(tight)[:15])})")
    return out


SWING_SHARE = 0.25   # a hi-hat stem onset this strong (vs the song's median "&" hat) is a hat


def swing_hats(notes: list[Note], drums: Path, T: np.ndarray, bar_starts: list[int], grid: int) -> list[Note]:
    """--swing-hats: the hats are 8ths plus a late pickup hat
    before some beats, swung to ~0.78 of the beat. ADTOF also hears the percussion on the
    16ths as hats. Keep the 8th hats; an "e"/"a" hat stays only where the LarsNet hi-hat stem
    has an onset there; the late hat goes on the "a" 16th (grid 16) or the last 16th-triplet
    (grid 24, "24 beat")."""
    import librosa
    y, sr = librosa.load(str(drum_parts(drums)["hihat"]), sr=22050, mono=True)
    hop = 64
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    fps = sr / hop

    def peak(t0, t1):
        return float(env[int(t0 * fps):max(int(t1 * fps), int(t0 * fps) + 1)].max())
    bs = np.array(bar_starts)
    beats = [b + 4 * j for k, b in enumerate(bs[:-1]) if bs[k + 1] - b == 16 for j in range(4)]
    beats = [b for b in beats if b + 4 < len(T) and T[b] >= 0 and T[b + 4] * fps < len(env) - 1]
    ref = np.median([peak(T[b] + 0.4 * (T[b + 4] - T[b]), T[b] + 0.6 * (T[b + 4] - T[b])) for b in beats])
    ba = np.array(beats)

    def off_8th(n):     # a hat between the 8ths of a checked beat (bars need not start on a multiple of 4)
        i = int(np.searchsorted(ba, n.q, side="right") - 1)
        return n.ch == "11" and i >= 0 and n.q - ba[i] < 4 and (n.q - ba[i]) not in (0, 2)
    out = [n for n in notes if not off_8th(n)]
    added = 0
    for b in beats:
        t0, p = T[b], T[b + 4] - T[b]
        for lo, hi, pos in ((0.18, 0.32, Fraction(1)), (0.68, 0.9, Fraction(3) if grid == 16 else Fraction(10, 3))):
            if peak(t0 + lo * p, t0 + hi * p) >= SWING_SHARE * ref:
                out.append(Note(b + pos, "11", float(t0 + float(pos) / 4 * p)))
                added += 1
    print(f"      swing hats: {sum(n.ch == '11' for n in notes)} -> {sum(n.ch == '11' for n in out)} hats "
          f"({added} off-8th hats confirmed by the hi-hat stem, grid {grid})")
    return sorted(out, key=lambda n: (n.q, n.ch))


def busy_stream_snares(notes: list[Note], bar_starts: list[int], bpms: np.ndarray,
                       fixed: set[int]) -> list[str]:
    """CHECK items: double-kick bars with 5+ snares (bleed or a real blast? the charter decides)."""
    bs = np.array(bar_starts)
    per: dict[int, list[str]] = {}
    for n in notes:
        per.setdefault(int(np.searchsorted(bs, n.q, side="right") - 1), []).append(n.ch)
    busy = [k + 1 for k, chs in sorted(per.items()) if 0 <= k < len(bs) - 1 and k + 1 not in fixed
            and chs.count("13") + chs.count("1C") >= 12 and chs.count("12") >= 5
            and bpms[min(bs[k], len(bpms) - 1)] > DOUBLE_KICK_BPM]
    runs: list[list[int]] = []
    for m in busy:
        if runs and m == runs[-1][-1] + 1:
            runs[-1].append(m)
        else:
            runs.append([m])
    return [f"measures {r[0]:03d}-{r[-1]:03d}: many snares over the double kick - real, or backbeat "
            f"only? (--backbeat {r[0]}-{r[-1]})" for r in runs]


HH_OPEN_LEVEL = -13.0   # dB vs the song's strong hat attacks (90th pct): still ringing = open
HAT_STREAM = 5          # hand hats on this many of a 4/4 bar's 8 eighths make it an 8th hat-groove bar
HAT_ALT_GAP = 8.0       # dB between the beats' and the "&"s' median ring before a phrase is open-close-open-close
HAT_SHUT = -22.0        # ... and the closed one must really be shut (a foot-closed hat dies 22-40 dB under)


def hat_phrases(hh: list[Note], ring: dict[int, float], notes: list[Note], bar_starts: list[int],
                log: list[str] | None = None) -> dict[int, bool]:
    """Rule: through an 8th hi-hat groove a drummer keeps one sound - all closed, all open, or one
    open one closed - never a random mix. Each hit's own ring is noisy (a half-open chorus rings
    right around the open / closed split), so consecutive groove bars are one phrase and the
    phrase's median ring decides: in 2+ bars, the beats and the "&"s apart by HAT_ALT_GAP, the louder
    one open and the other really shut (HAT_SHUT) -> open on the louder one; else all open or all closed
    (a half-damped "&" in an open groove is still the open groove). In a closed phrase an
    open hat on a bar's last 8th before a crash stays open. Open-close phrases are listed in the
    log: all open is as likely what a charter wants there. Returns open / closed per hat."""
    bs = list(bar_starts)
    crash_at = {n.q for n in notes if n.ch in ("16", "1A")}
    by_bar: dict[int, list[Note]] = {}
    for n in hh:
        b = bisect.bisect_right(bs, int(n.q)) - 1
        if 0 <= b < len(bs) - 1 and bs[b + 1] - bs[b] == 16 and n.q.denominator == 1 and (n.q - bs[b]) % 2 == 0:
            by_bar.setdefault(b, []).append(n)
    groove = sorted(b for b, ns in by_bar.items() if len(ns) >= HAT_STREAM)
    phrases: list[list[int]] = []
    for b in groove:
        if phrases and phrases[-1][-1] == b - 1:
            phrases[-1].append(b)
        else:
            phrases.append([b])
    decided: dict[int, bool] = {}
    for ph in phrases:
        ns = [n for b in ph for n in by_bar[b]]
        on = [ring[id(n)] for n in ns if id(n) in ring and (n.q - bs[ph[0]]) % 4 == 0]
        off = [ring[id(n)] for n in ns if id(n) in ring and (n.q - bs[ph[0]]) % 4 == 2]
        if len(on) + len(off) < 4:
            continue
        m_on, m_off = (float(np.median(x)) if x else -99.0 for x in (on, off))
        if len(ph) >= 2 and abs(m_on - m_off) >= HAT_ALT_GAP and min(m_on, m_off) <= HAT_SHUT and max(m_on, m_off) >= HH_OPEN_LEVEL:
            pick = (lambda n, beat_open=m_on > m_off: ((n.q - bs[ph[0]]) % 4 == 0) == beat_open)
            if log is not None:
                log.append(f"hi-hats written {'open on the beat, closed on the &' if m_on > m_off else 'closed on the beat, open on the &'}"
                           f" in measures {ph[0] + 1:03d}-{ph[-1] + 1:03d} (the & and the beat ring differently)"
                           f" - or all open / all closed?")
        else:
            all_open = float(np.median(on + off)) >= HH_OPEN_LEVEL
            pick = (lambda n, o=all_open: o)
        for b in ph:
            for n in by_bar[b]:
                keep = (n.q - bs[b] == 14 and bs[b + 1] in crash_at and ring.get(id(n), -99.0) >= HH_OPEN_LEVEL)
                decided[id(n)] = pick(n) or keep
    return decided


def open_hihats(notes: list[Note], drums: Path, bar_starts: list[int] | None = None,
                log: list[str] | None = None) -> list[Note]:
    """Open / closed hi-hat (an open hat, then the closing hit = closed HH + left-foot pedal
    together). On the LarsNet hi-hat stem, just before the next hat hit a closed hat has died
    away (22-40 dB under a strong attack) while an open one still rings (within ~10 dB) -
    a clean recording splits at about -13 dB. Open hats -> 18; the first closed hat after
    open ones closes them: 11 + pedal 1B. With the bars, 8th hat grooves get one pattern per
    phrase (hat_phrases)."""
    import librosa
    hh = sorted((n for n in notes if n.ch == "11"), key=lambda n: n.q)
    if len(hh) < 8:
        return notes
    y, sr = librosa.load(str(drum_parts(drums)["hihat"]), sr=44100, mono=True)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=128))
    f = librosa.fft_frequencies(sr=sr, n_fft=1024)
    E = S[(f >= 6000) & (f < 16000)].mean(0) + 1e-9
    fps = sr / 128

    def level(t, a, b):
        i, j = int((t + a) * fps), int((t + b) * fps) + 1
        return float(20 * np.log10(E[max(i, 0):max(j, i + 1)].mean())) if i < len(E) else -200.0
    top = np.percentile([level(n.t, 0, 0.015) for n in hh], 90)
    ring = {id(n): level(nxt.t, -0.04, -0.01) - top for n, nxt in zip(hh, hh[1:]) if 0.1 < nxt.t - n.t < 0.5}
    decided = hat_phrases(hh, ring, notes, bar_starts, log) if bar_starts else {}
    out = list(notes)
    was_open = False
    for n in hh:
        is_open = decided.get(id(n), ring.get(id(n), -99.0) >= HH_OPEN_LEVEL)
        if is_open:
            n.ch = "18"
        elif was_open:
            out.append(Note(n.q, "1B", n.t))
        was_open = is_open
    # left foot busy on the double kick within a beat: no pedal, the hat just stays open
    lbd = np.array(sorted(float(n.q) for n in out if n.ch == "1C"))
    busy = {n.q for n in out if n.ch == "1B" and len(lbd) and np.abs(lbd - float(n.q)).min() <= 4}
    for n in out:
        if n.ch == "11" and n.q in busy:
            n.ch = "18"
    return [n for n in out if not (n.ch == "1B" and n.q in busy)]


def close_open_hats(notes: list[Note]) -> list[Note]:
    """Score mode: open hats come from the score, the foot close doesn't. The first closed hat
    after an open one gets the left-foot pedal (11 + 1B) unless a pedal chip closed it already;
    left foot on the double kick within a beat -> that hat stays open instead (as open_hihats)."""
    hats = sorted((n for n in notes if n.ch in ("11", "18", "1B")), key=lambda n: (n.q, n.ch != "1B"))
    lbd = np.array(sorted(float(n.q) for n in notes if n.ch == "1C"))
    pedal = {n.q for n in notes if n.ch == "1B"}
    out, was_open = list(notes), False
    for n in hats:
        if n.ch == "18":
            was_open = True
        elif n.ch == "1B":
            was_open = False
        elif was_open:
            was_open = False
            if n.q in pedal:
                continue
            if len(lbd) and np.abs(lbd - float(n.q)).min() <= 4:
                n.ch, was_open = "18", True
            else:
                out.append(Note(n.q, "1B", n.t))
                pedal.add(n.q)
    return out


FOOT_PATTERNS = {"8ths": range(0, 16, 2), "off 8ths": range(2, 16, 4), "2 and 4": (4, 12), "beats": range(0, 16, 4)}
FOOT_FIT = 0.6   # F1 of the heard foot chicks against a steady pattern before the bar gets the whole pattern


def steady_foot_hats(notes: list[Note], bar_starts: list[int]) -> list[Note]:
    """Under a ride (hands on the cymbals, no hand hi-hat) the left foot keeps a steady hi-hat:
    every 8th, the off-8ths, 2 and 4, or the beats. The stems only catch some of those chicks, so a
    4/4 bar whose heard foot chips fit one of these patterns gets the whole pattern."""
    by_bar: dict[int, list[Note]] = {}
    for n in notes:
        by_bar.setdefault(bisect.bisect_right(bar_starts, int(n.q)) - 1, []).append(n)
    out, added = list(notes), 0
    for b, ns in by_bar.items():
        if b < 0 or b + 1 >= len(bar_starts) or bar_starts[b + 1] - bar_starts[b] != 16:
            continue
        s0 = bar_starts[b]
        if any(n.ch in ("11", "18") for n in ns) or sum(n.ch in ("16", "19") for n in ns) < 4:
            continue
        heard = {int(n.q) - s0 for n in ns if n.ch == "1B" and n.q == int(n.q)}
        if len(heard) < 2:
            continue
        f1 = {k: 2 * len(heard & set(v)) / (len(heard) + len(v)) for k, v in FOOT_PATTERNS.items()}
        best = max(f1, key=f1.get)
        if f1[best] < FOOT_FIT:
            continue
        t_of = {int(n.q): n.t for n in ns}
        for k in FOOT_PATTERNS[best]:
            if k not in heard:
                out.append(Note(Fraction(s0 + k), "1B", t_of.get(s0 + k, 0.0)))
                added += 1
    print(f"      steady foot hi-hat: {added} pedal chips filled in under the ride")
    return sorted(out, key=lambda n: (n.q, n.ch))


RIDE_OVER_HAT = 0.0   # dB: cymbal stem over hi-hat stem at a "hi-hat" hit -> it was the ride
CHICK_RISE = 8.0   # dB: a foot chick on the LarsNet hi-hat stem (6-16 kHz) over the moment before


RIDE_BAR = 6   # ride hits in a bar that make the ride the time-keeper


def ride_bar_foot_hats(notes: list[Note], bar_starts: list[int], log: list[str]) -> list[Note]:
    """Rock-and-roll ride groove: right hand on the ride, left on the snare, so the hi-hat heard in
    between is the left foot (LP), not a hand. In a bar with RIDE_BAR+ ride hits every closed hand
    hi-hat becomes a pedal chip (1B); one already on that spot is kept once. Open hats stay."""
    bs = np.array(bar_starts)
    bar_of = lambda q: int(np.searchsorted(bs, q, side="right") - 1)
    rides: dict[int, int] = {}
    for n in notes:
        if n.ch == "19":
            rides[bar_of(n.q)] = rides.get(bar_of(n.q), 0) + 1
    bars = {b for b, c in rides.items() if c >= RIDE_BAR}
    pedal = {n.q for n in notes if n.ch == "1B"}
    out, moved = [], set()
    for n in notes:
        if n.ch == "11" and bar_of(n.q) in bars:
            moved.add(bar_of(n.q) + 1)
            if n.q in pedal:
                continue
            n.ch = "1B"
            pedal.add(n.q)
        out.append(n)
    if moved:
        log.append(f"hand hi-hats under a ride groove moved to the hi-hat pedal (LP) in measures "
                   f"{', '.join(f'{m:03d}' for m in sorted(moved))}")
    return out


def ride_foot_hats(notes: list[Note], drums: Path, T: np.ndarray, bar_starts: list[int]) -> list[Note]:
    """Riding on the ride cymbal, the hi-hat is the left foot (ride + hi-hat
    kicked on the 8ths). In bars with 4+ ride hits a closed hat becomes a single pedal chip (1B),
    and 8th positions where the hi-hat stem chicks get one too (dropped later where the left
    foot is on the double kick)."""
    import librosa
    bs = np.array(bar_starts)
    bar_of = lambda q: int(np.searchsorted(bs, q, side="right") - 1)
    # ADTOF has no ride class: a ride often comes out as hi-hat. A hat that is louder on the
    # LarsNet cymbal stem than on the hi-hat stem (2-10 kHz, first 20 ms) is the ride.
    parts = drum_parts(drums)

    def band_env(stem):
        y, sr = librosa.load(str(parts[stem]), sr=22050, mono=True)
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=128))
        f = librosa.fft_frequencies(sr=sr, n_fft=1024)
        return S[(f >= 2000) & (f < 10000)].mean(0) + 1e-9, sr / 128
    (Ec, fps2), (Eh, _) = band_env("cymbals"), band_env("hihat")
    lv = lambda E, t: float(20 * np.log10(E[int(t * fps2):int(t * fps2) + 4].mean() + 1e-12))
    n_ride = 0
    for n in notes:
        if n.ch == "11" and int(n.t * fps2) + 4 < len(Ec) and lv(Ec, n.t) - lv(Eh, n.t) > RIDE_OVER_HAT:
            n.ch = "19"
            n_ride += 1
    rides: dict[int, int] = {}
    for n in notes:
        if n.ch == "19":
            rides[bar_of(n.q)] = rides.get(bar_of(n.q), 0) + 1
    bars = {k for k, c in rides.items() if c >= 4 and 0 <= k < len(bs) - 1}
    print(f"      ride: {n_ride} hi-hat notes ring on the cymbal stem -> ride")
    if not bars:
        return notes
    y, sr = librosa.load(str(drum_parts(drums)["hihat"]), sr=44100, mono=True)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=128))
    f = librosa.fft_frequencies(sr=sr, n_fft=1024)
    E = S[(f >= 6000) & (f < 16000)].mean(0) + 1e-9
    fps = sr / 128

    def level(t, a, b):
        i, j = int((t + a) * fps), int((t + b) * fps) + 1
        return float(20 * np.log10(E[max(i, 0):max(j, i + 1)].mean())) if 0 <= i < len(E) else -200.0
    lag = float(np.median([n.t - T[int(n.q)] for n in notes if n.q == int(n.q) and int(n.q) < len(T)]))
    out, have, moved, added = [], set(), 0, 0
    for n in notes:
        if n.ch in ("16", "1A") and bar_of(n.q) in bars and n.q != bs[bar_of(n.q)]:
            n = Note(n.q, "19", n.t)        # ADTOF calls the ride crash too; crashes stay on beat 1
        if n.ch == "11" and bar_of(n.q) in bars:
            n = Note(n.q, "1B", n.t)
            moved += 1
        if n.ch == "1B":
            have.add(n.q)
        out.append(n)
    rq = {n.q for n in out if n.ch == "19"}
    for k in sorted(bars):
        for q in range(int(bs[k]), int(bs[k + 1]), 2):
            if q in have or q >= len(T):
                continue
            if q in rq:     # a hi-hat chip under every 8th-note ride hit
                out.append(Note(Fraction(q), "1B", T[q] + lag))
                added += 1
                continue
            t = T[q] + lag
            if level(t, -0.005, 0.015) - level(t, -0.05, -0.02) >= CHICK_RISE:
                out.append(Note(Fraction(q), "1B", t))
                added += 1
    print(f"      ride bars: {len(bars)}; hi-hat foot chips: {moved} hats moved to the pedal, {added} added")
    return sorted(out, key=lambda n: (n.q, n.ch))


TOM_HIGH_RING = (180.0, 260.0)  # Hz: electronic toms ring here (kick < 160)
TOM_MIN_PROM = 13.0             # ring peak / median of the 35-260 Hz CQT band
TOM_SWEEP_SEMIS = 5.0           # a roll moving this far in pitch is toms, not a kick


def find_tom_fills(notes: list[Note], drums: Path, bar_starts: list[int],
                   log: list[str] | None = None) -> list[Note]:
    """ADTOF often calls electronic toms kick+snare+hi-hat. A tom rings at a steady pitch for
    ~60 ms after the hit. Mark as toms (a) 3+ hits close together ringing at 180-260 Hz, away
    from the song's snare ring, and
    (b) runs of 4+ hits (5+ when the ring is faint) whose ring pitch moves steadily up or
    down by 5+ semitones (a roll).
    Inside a fill, lanes follow pitch rank (high -> HT, mid -> LT, low -> FT). Sweeps are less
    sure than the clear ring, so they go to `log` for the CHECK list."""
    import librosa
    by_q: dict[Fraction, list[Note]] = {}
    for n in notes:
        by_q.setdefault(n.q, []).append(n)
    cand = sorted(q for q, g in by_q.items()
                  if {"12", "13"} & {n.ch for n in g} and not {"14", "15", "17"} & {n.ch for n in g})
    if len(cand) < 3:
        return notes
    y, sr = librosa.load(str(drums), sr=22050, mono=True)
    hop = 64
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=35, n_bins=54, bins_per_octave=12))
    fr = librosa.cqt_frequencies(54, fmin=35)
    ring: dict[Fraction, tuple[float, float, float]] = {}  # q -> (Hz, prominence, low share)
    for q in cand:
        t = min(n.t for n in by_q[q])
        a, b = int((t + 0.03) * sr / hop), int((t + 0.09) * sr / hop)
        if b > C.shape[1]:
            continue
        m = C[:, a:b].mean(1)
        k = int(np.argmax(m))
        low = float(C[fr < 100, a - 4:b].sum() / (C[:, a - 4:b].sum() + 1e-9))
        ring[q] = (float(fr[k]), float(m[k] / (np.median(m) + 1e-9)), low)
    def chains(min_prom, ok, link):
        """Runs of candidate hits at most an 8th apart; `link` decides if q extends the run.
        A broken run restarts from its last hit so a turn in pitch can begin a new sweep."""
        out, cur = [], []
        for q in cand:
            if q not in ring or ring[q][1] < min_prom or not ok(q):
                continue
            if cur and not (q - cur[-1] <= 2 and link(cur, q)):
                out.append(cur)
                cur = [cur[-1]] if q - cur[-1] <= 2 and link(cur[-1:], q) else []
            cur.append(q)
        return out + ([cur] if cur else [])
    # an acoustic snare also rings near 200 Hz: a high ring is a tom only away from the
    # song's usual snare ring (some snares sit on the kick, ~120-150 Hz)
    sd = [ring[q][0] for q in cand if q in ring and ring[q][1] >= TOM_MIN_PROM
          and "12" in {n.ch for n in by_q[q]} and ring[q][0] >= 120]  # skip bass bleed
    sd_ring = float(np.median(sd)) if sd else 1.0
    lo, hi = TOM_HIGH_RING

    def high_tom(q):
        f = ring[q][0]
        return lo <= f <= hi and abs(12 * np.log2(f / sd_ring)) >= 2.5
    fills = [(c, False) for c in chains(TOM_MIN_PROM, high_tom, lambda c, q: True) if len(c) >= 3]

    def semis(a, b):
        return 12 * np.log2(ring[b][0] / ring[a][0])

    def monotone(c, q):  # every step moves at least a semitone, all the same way
        s = semis(c[-1], q)
        return abs(s) >= 0.9 and (len(c) < 2 or np.sign(s) == np.sign(semis(c[-2], c[-1])))
    # a clear ring needs only 4 hits; a kick under the roll blurs the ring, then ask for 5
    # below ~60 Hz a falling pitch is a synth kick drop, not a tom
    print(f"      snare ring {sd_ring:.0f} Hz")
    fills += [(c, True) for c in chains(TOM_MIN_PROM, lambda q: ring[q][0] >= 60, monotone)
              if len(c) >= 4 and abs(semis(c[0], c[-1])) >= TOM_SWEEP_SEMIS]
    fills += [(c, True) for c in chains(TOM_MIN_PROM / 2, lambda q: ring[q][0] >= 60, monotone)
              if len(c) >= 5 and abs(semis(c[0], c[-1])) >= TOM_SWEEP_SEMIS]
    bs = np.array(bar_starts)
    done: set[Fraction] = set()
    for c, sweep in fills:
        c = [q for q in c if q not in done]
        if len(c) < 3:
            continue
        # pitch rank inside the fill; on equal pitch the earlier hit is the higher drum
        order = sorted(range(len(c)), key=lambda i: (-ring[c[i]][0], i))
        for r, i in enumerate(order):
            q = c[i]
            lane = "14" if r < len(c) / 3 else "15" if r < 2 * len(c) / 3 else "17"
            k = int(np.searchsorted(bs, q, side="right") - 1)
            on_beat = (q - bs[max(k, 0)]) % 4 == 0
            t = min(n.t for n in by_q[q])
            keep = [n for n in by_q[q] if n.ch not in ("11", "18", "12", "13")
                    or (n.ch == "13" and on_beat and ring[q][2] >= 0.3)]
            by_q[q] = keep + [Note(q, lane, t)]
            done.add(q)
        k = int(np.searchsorted(bs, c[0], side="right") - 1)
        msg = (f"measure {k + 1:03d}: tom roll from 16th {c[0] - bs[max(k, 0)] + 1} ({len(c)} hits, "
               f"{ring[c[0]][0]:.0f} -> {ring[c[-1]][0]:.0f} Hz)")
        print(f"      tom fill: {msg}")
        if sweep and log is not None:
            log.append(msg + " - heard as a pitch sweep, check it is toms")
    return sorted((n for g in by_q.values() for n in g), key=lambda n: (n.q, n.ch))


# ---- 6. chart layout + writing ------------------------------------------------
def grid_from_chart(ref_path: Path, hits: list[tuple[float, int]], duration: float,
                    max_bpm: float) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """--grid-from: tempo and bar lines taken from another chart of the same song (an official
    chart, or one whose bar lines the user fixed by hand), lined up with this audio by its kick and
    snare notes. A tempo over max_bpm (a game chart's 272 for 136) is counted at half, so its
    bars keep 16ths of the felt tempo. Past the reference's end the last bar repeats."""
    ref = dtx.read_dtx(ref_path)
    off = ref.bgm_offset()
    ev = ref._tempo_events()

    def bpm_at(m: int, p: Fraction) -> float:
        b = ref.base_bpm
        for em, ep, eb in ev:
            if (em, ep) <= (m, p):
                b = eb
        return b
    mine = np.array(sorted(t for t, p in hits if ADTOF_TO_CH.get(p) in ("12", "13")))
    theirs = np.array(sorted(ref.time_of(c.measure, c.pos) - off for c in ref.chips
                             if c.channel.upper() in ("12", "13", "1C")))
    d = (mine[None, ::2] - theirs[::2, None]).ravel()
    d = d[np.abs(d) < 60]
    h, e = np.histogram(d, bins=np.arange(-60, 60, 0.01))
    shift = float(np.median(d[np.abs(d - (e[h.argmax()] + 0.005)) < 0.03]))
    T, bpms, starts = [], [], []
    m, last = 0, max(c.measure for c in ref.chips)
    while True:
        L = ref.measure_len.get(m, 1.0) if m <= last else ref.measure_len.get(last, 1.0)
        mm = min(m, last)
        b = bpm_at(mm, Fraction(0))
        half = b > max_bpm
        n = int(round(L * 16 / (2 if half else 1)))
        t0 = ref.time_of(m, Fraction(0)) - off + shift if m <= last else T[-1] + 15 / bpms[-1]
        if m > last:
            pts = [t0 + k * 15 / (b / (2 if half else 1)) for k in range(n)]
        else:
            pts = [ref.time_of(m, Fraction(k, n)) - off + shift for k in range(n)]
        if pts[0] >= -0.05:
            starts.append(len(T))
        for k, t in enumerate(pts):
            T.append(t)
            bk = bpm_at(mm, Fraction(k, n)) if m <= last else b
            bpms.append(round(bk / 2, 3) if bk > max_bpm else bk)
        if pts[-1] > duration + 1:
            starts.append(len(T))
            break
        m += 1
    print(f"      grid from {ref_path.name}: this audio = reference + {shift:+.3f} s, {len(starts) - 1} bars")
    return np.array(T), np.array(bpms), starts


def layout(notes: list[Note], T: np.ndarray, bpms: np.ndarray, bar_starts: list[int],
           note_offset: float = TARGET_LAG / 2, bar: int = 16):
    """Turn tatum positions into DTX measures. Measure 000 is an empty 4/4 lead-in at the first
    BPM; measure 001 starts at the first bar. Returns (chips, measure_len, bpm_defs, base_bpm)."""
    bs = bar_starts
    base_bpm = float(bpms[bs[0]])
    chips: list[dtx.Chip] = []
    measure_len: dict[int, float] = {}
    bpm_ids: dict[float, str] = {}

    def bpm_id(b: float) -> str:
        if b not in bpm_ids:
            bpm_ids[b] = dtx.to_b36(len(bpm_ids) + 1)
        return bpm_ids[b]

    cur_bpm = base_bpm
    for k in range(len(bs) - 1):
        m = k + 1
        L = bs[k + 1] - bs[k]
        if L != bar:
            measure_len[m] = L / bar
        for i in range(bs[k], bs[k + 1]):
            b = float(bpms[min(i, len(bpms) - 1)])
            if b != cur_bpm:
                chips.append(dtx.Chip(m, Fraction(i - bs[k], L), "08", bpm_id(b)))
                cur_bpm = b
    bar_of = np.array(bs)
    for n in notes:
        k = int(np.searchsorted(bar_of, n.q, side="right") - 1)
        if k < 0 or k >= len(bs) - 1:
            continue
        L = bs[k + 1] - bs[k]
        chips.append(dtx.Chip(k + 1, (n.q - bs[k]) / L, n.ch, n.wav or KIT[n.ch][0]))
    # BGM: audio time 0 sits -T[bs0] after measure 001 starts (T[bs0] <= 0)
    lead = 4 * 60.0 / base_bpm                        # measure 000 duration
    c = lead - T[bs[0]] - note_offset                  # chart time of audio start
    if c < lead:
        chips.append(dtx.Chip(0, Fraction(round(c / lead * 1536), 1536) % 1, "01", "01"))
    else:
        # inside measure 001 (first bar starts before the audio); walk its tatums
        i = int(np.searchsorted(T, -note_offset, side="right") - 1)
        a0 = -note_offset                              # grid time where the BGM chip goes
        x = i + (a0 - T[i]) / (T[i + 1] - T[i]) - bs[0]
        L = bs[1] - bs[0]
        chips.append(dtx.Chip(1, Fraction(round(x / L * 1536), 1536), "01", "01"))
    return chips, measure_len, {v: k for k, v in bpm_ids.items()}, base_bpm


def accent_at(hits: list[tuple[float, int]], T: np.ndarray) -> np.ndarray:
    """Per-tatum kick (0.5) / crash (1.0) accents."""
    accent = np.zeros(len(T))
    for t, p in hits:
        if p in (35, 49):
            i = int(np.argmin(np.abs(T - t)))
            if abs(T[i] - t) < 0.035:
                accent[i] = max(accent[i], 1.0 if p == 49 else 0.5)
    return accent


def review(T: np.ndarray, bpms: np.ndarray, starts: list[int], hits: list[tuple[float, int]],
           onsets: np.ndarray, w: np.ndarray,
           max_bpm: float, beat_bpm: float, fixed: set[int] = frozenset(), bar: int = 16) -> list[str]:
    """Spots to check, by DTX measure number (001 = first bar). Measures in `fixed`
    were set by hints and are not questioned."""
    out = []
    bs = np.array([b for b in starts if b < len(T)])

    def measure_at(t: float) -> int:
        return int(np.searchsorted(T[bs], t, side="right"))

    # tempo octave: the half (or double) tempo fits the drums about as well, and the beat
    # tracker's own beat rate doesn't settle it
    b0 = float(np.median(bpms))
    alt = b0 * 2 if b0 * 2 <= max_bpm * 1.0001 else b0 / 2
    if 60 <= alt <= 260:
        c0 = abs(tempo.coherence(onsets, 15.0 / b0, w))
        c1 = abs(tempo.coherence(onsets, 15.0 / alt, w))
        if c1 > 0.9 * c0 and abs(np.log(beat_bpm / b0)) > 0.1:
            out.append(f"tempo octave: {b0:.2f} or {alt:.2f} BPM both fit - say which (use --max-bpm)")
    # tempo changes and odd bars
    for i in np.flatnonzero(np.diff(bpms) != 0) + 1:
        k = measure_at(T[i] + 1e-6)
        if k in fixed or abs(np.log(bpms[i] / bpms[i - 1])) < 0.01:  # drift steps aren't changes
            continue
        out.append(f"measure {k:03d}: tempo {bpms[i - 1]:.2f} -> {bpms[i]:.2f} - check where the change starts")
    for k, L in enumerate(np.diff(bs)):
        if L != bar and k + 1 not in fixed:
            out.append(f"measure {k + 1:03d}: {L}/{bar} bar - check the meter")
    # timing drift: grid vs kick/snare hits, 8-bar windows
    h = np.array([t for t, p in hits if p in (35, 38) and T[0] < t < T[-1]])
    if len(h) > 16:
        j = np.clip(np.searchsorted(T, h), 1, len(T) - 1)
        g = np.where(np.abs(T[j - 1] - h) < np.abs(T[j] - h), T[j - 1], T[j]) - h
        ok = np.abs(g) < 0.05
        ref = np.median(g[ok])
        m = np.searchsorted(T[bs], h, side="right")
        for k0 in range(1, len(bs), 8):
            sel = ok & (m >= k0) & (m < k0 + 8)
            if sel.sum() >= 8:
                d = (np.median(g[sel]) - ref) * 1000
                if abs(d) > 10:
                    out.append(f"measures {k0:03d}-{k0 + 7:03d}: notes {abs(d):.0f} ms "
                               f"{'early' if d < 0 else 'late'} vs the drums - tempo may wander here")
    return out


def audio_notes(hits, T, starts, bpms, drums, backbeat, snare24, crash, kick_check, ride_foot, groove, swing):
    """The audio pipeline's notes: transcription hits on the grid, then the per-song hint rules
    and the playability rules (score mode skips all of this). Returns the notes and the tom log."""
    notes = spread_flat_fills(refine_toms(refine_cymbals(quantize(hits, T, starts)), drums), drums)
    n_toms = sum(n.ch in ("14", "15", "17") for n in notes)
    tom_log: list[str] = []
    notes = find_tom_fills(notes, drums, starts, tom_log)
    print(f"tom fills: {sum(n.ch in ('14', '15', '17') for n in notes) - n_toms} hits moved to toms")
    if backbeat:
        notes = backbeat_only(notes, starts, parse_measures(backbeat))
    if snare24:
        notes = snare_on_24(notes, starts)
    if crash:
        notes = crash_hints(notes, starts, T, crash)
    notes = verify_kick_runs(notes, drums, bpms, starts, tom_log, kick_check)
    notes = drop_silent(notes, drums, starts, tom_log)
    notes = kick_rolls(notes, drums, T, bpms, starts, tom_log)
    notes = verify_hats(notes, drums, starts, tom_log)
    if ride_foot:
        notes = steady_foot_hats(ride_foot_hats(notes, drums, T, starts), starts)   # before the hand limit: those hats are feet
    notes = ride_bar_foot_hats(notes, starts, tom_log)
    n_before = sum(n.ch != "1B" for n in notes)
    notes = limit_hands(notes)
    notes = double_kick(notes, starts, bpms)
    notes = open_hihats(notes, drums, starts, tom_log)
    notes = left_foot_busy(notes)
    print(f"      playability: dropped {n_before - sum(n.ch != '1B' for n in notes)} 3rd-hand notes, "
          f"{sum(n.ch == '1C' for n in notes)} kicks moved to left pedal; "
          f"{sum(n.ch == '18' for n in notes)} open hi-hats, {sum(n.ch == '1B' for n in notes)} foot closes")
    if groove:
        spec = json.loads(Path(groove).read_text(encoding="utf-8"))
        notes, snapped = snap_grooves(notes, T, starts, spec)
        print(f"      groove {Path(groove).stem}: {len(snapped)} of {len(starts) - 1} bars written as the groove")
    if swing:
        notes = swing_hats(notes, drums, T, starts, swing)
    return notes, tom_log


def apply_genre(names: str) -> dict:
    """Sets the tuning of the chosen genre(s) (genres.py) and returns their build options."""
    try:
        chosen = genres.parse(names)
    except ValueError as e:
        raise SystemExit(str(e))
    tune, flags = genres.blend(chosen)
    for g in chosen:
        print(f"      genre {genres.GENRES[g]['name']}: {genres.GENRES[g]['text']}")
    for key, v in tune.items():
        if key == "odd_bars":
            for L in ODD_BARS:
                tempo.BAR_COST[L] = tempo.BAR_COST_BASE[L] * v
        elif key.startswith("bar_cost_"):
            tempo.BAR_COST[int(key[9:])] = v
        else:
            mod, _, attr = key.rpartition(".")
            setattr(tempo if mod == "tempo" else sys.modules[__name__], attr, v)
    return flags


ODD_BARS = (12, 8, 20, 24, 14, 18, 15, 17)   # bar lengths (16ths) other than 4/4


def build(audio: Path, out_dir: Path, title: str, artist: str, bgm_mode: str, input_mode: str,
          bar_hints: str = "", tempo_at: list[int] | None = None, max_bpm: float = 220.0,
          backbeat: str = "", half: bool = False, kick_check: bool = False,
          beat_grid: str = "", kit_file: Path | None = None, snare_match: bool = False,
          volume: int = 100, dynamics: bool = False, preview: Path | None = None,
          preimage: Path | None = None, dlevel: int = 50, ride_foot: bool = False,
          merge_bars: int = 0, groove: Path | None = None, swing: int = 0,
          tempo_from: str = "", crash: str = "", snare24: bool = False,
          score: Path | None = None, score_first: int = 2, video: Path | None = None,
          fixed_bpm: float = 0.0, game_style: bool = False, grid_from: Path | None = None,
          cymbal_roll: str = "", genre: str = "") -> Path:
    import soundfile as sf
    out_dir.mkdir(parents=True, exist_ok=True)
    g = apply_genre(genre)
    if max_bpm == 220.0:
        max_bpm = g.get("max_bpm", max_bpm)
    snare24, ride_foot = snare24 or g.get("snare24", False), ride_foot or g.get("ride_foot", False)
    kick_check, snare_match = kick_check or g.get("kick_check", False), snare_match or g.get("snare_match", False)
    swing = swing or g.get("swing", 0)
    if g.get("kit") == "digi" and not kit_file:
        kit_file = drumkit.digital_kit()
    print("[1/5] separating stems ...")
    drums, rest = separate(audio)
    duration = sf.info(str(rest)).duration

    print("[2/5] tempo map ...")
    onsets, strength = drum_onsets(drums)
    beat_act, down_act = beat_activations(audio)
    ref, beat_bpm = rough_bpm(beat_act, onsets, strength, max_bpm)
    sections = tempo.find_sections(onsets, strength, ref)
    if not tempo_at:
        sections = tempo.merge_close_sections(sections, onsets, strength)
    if fixed_bpm:   # a known tempo (e.g. a game song's published BPM): one section, phase fitted
        sections = [tempo.Section(sections[0].start, sections[-1].end, fixed_bpm, ramp=False)]
        tempo.fit_section(sections[0], onsets, strength, margin=0.0, search=1.0)
    for tok in filter(None, tempo_from.split(";")):
        t0, b = (float(x) for x in tok.split("="))
        sections = tempo.force_section(sections, t0, b, onsets, strength)
    for s in sections:
        print(f"      {max(s.start, 0):7.1f}s - {min(s.end, duration):7.1f}s  BPM {s.bpm:.2f}")
    T, bpms = tempo.tatum_grid(sections, duration, onsets, strength)

    print("[3/5] transcribing drums ...")
    hits = transcribe(drums if input_mode == "drums" else audio)
    ghosts = []
    if GHOST_TH and not score:
        ghosts = ghost_snares(drums if input_mode == "drums" else audio, hits, drums)
        hits = sorted(hits + [(t, 38) for t in ghosts])
    print(f"      {len(hits)} hits")
    if len(hits) < 50:   # no drum kit in the song: nothing to chart
        sys.exit(f"NO DRUMS: only {len(hits)} drum hits found, not charted")

    print("[4/5] bar lines + quantizing ...")

    def bars_on(T):
        active = np.zeros(len(T))
        for t, p in hits:
            i = int(np.searchsorted(T, t))
            if 0 < i < len(T):
                active[i] = 1
        switches = [int(i) + 1 for i in np.flatnonzero(np.diff(bpms) != 0)]
        return tempo.choose_bars(T, sample_at(down_act, T), sample_at(beat_act, T),
                                 accent_at(hits, T), active=active, switches=switches)
    beats = tempo.track_beats(beat_act, np.array(sorted(t for t, p in hits)))

    def near(pitch):
        h = np.array(sorted(t for t, p in hits if p == pitch))
        if not len(h):
            return np.zeros(len(beats))
        j = np.clip(np.searchsorted(h, beats), 1, len(h) - 1)
        return (np.minimum(np.abs(h[j - 1] - beats), np.abs(h[j] - beats)) < 0.04).astype(float)

    def lock_to_beats(quiet=False):
        # rule for a strongly drifting take (e.g. 121.5 -> 125 BPM):
        # lock the grid to the tracked beats, a BPM chip on every bar (or beat), 4/4 unless the
        # backbeat moves
        bar_beats = tempo.beat_bars(sample_at(down_act, beats), near(49), near(35), near(38))
        T, bpms, starts = tempo.beat_grid(beats, bar_beats, duration, beat_grid == "beat")
        starts = [b for b in starts if b >= max(x for x in starts if T[x] <= 0)]
        if beat_grid == "steps":
            # a programmed song with tempo changes (e.g. 164 / 185 / 180): constant BPM
            # stretches along the beat-locked grid, whole BPMs where they fit
            T, bpms = tempo.step_grid(T, starts)
        span = bpms[starts[0]:starts[-1]]
        if not quiet:
            print(f"      beat grid: {len(beats)} beats, {int((np.diff(span) != 0).sum())} BPM chips "
                  f"({span.min():.2f}-{span.max():.2f})")
        return T, bpms, starts

    if beat_grid:
        T, bpms, starts = lock_to_beats()
    else:
        starts = bars_on(T)
    if len(sections) == 1 and not tempo_at and not beat_grid:
        # live take: one tempo that wanders. Bend the grid along the drums, then
        # chart it as small BPM steps on bar lines (115.02 / 114.95 style)
        centers, phi = tempo.drift_curve(T, onsets, strength)
        kd = np.array(sorted(t for t, p in hits if p in (35, 38)))   # judge the result on kick/snare
        print(f"      drift vs one fixed tempo: {np.ptp(phi) * 1000:.0f} ms")
        Tw = T + np.interp(T, centers, phi - np.median(phi))
        fit0, fitw = tempo.grid_fit(T, kd), tempo.grid_fit(Tw, kd)
        if np.ptp(phi) > 2 * tempo.DRIFT_MAX and fitw > 0.8 * fit0:
            # the unwrapped curve slipped (±24 ms on the fixed grid, worse bent)
            print(f"      ... but the bent grid fits the drums no better ({fitw:.0f} vs {fit0:.0f} ms): kept fixed")
        elif np.ptp(phi) > 2 * tempo.DRIFT_MAX:
            starts = bars_on(Tw)
            if bar_hints:   # BPM steps go on the hinted bar lines
                starts = tempo.apply_bar_hints(starts, tempo.parse_bar_hints(bar_hints), len(Tw) - 1)
            T, bpms = tempo.drift_grid(Tw, starts)
            steps = np.unique(bpms[starts[0]:starts[-1]])
            print(f"      tempo drifts {np.ptp(phi) * 1000:.0f} ms: following it with "
                  f"{int((np.diff(bpms[starts[0]:starts[-1]]) != 0).sum())} BPM steps "
                  f"({steps.min():.2f}-{steps.max():.2f})")
        # the bar picker can hide drift with many 17/16 bars while the bar lines slide an
        # 8th off. If the beat-locked grid needs fewer odd bars and fits the drums
        # better, take it (rule: BPM chip on every bar, bars on the real beat 1)
        try:
            Tb, bpmb, sb = lock_to_beats(quiet=True)
        except ValueError:      # too few tracked beats to lock: keep this grid
            Tb, bpmb, sb = T, bpms, starts
        odd_now, odd_b = int((np.diff(starts) != 16).sum()), int((np.diff(sb) != 16).sum())
        fit_now, fit_b = tempo.grid_fit(T, kd), tempo.grid_fit(Tb, kd)
        print(f"      beat-locked grid: {odd_b} odd bars, fit {fit_b:.1f} ms "
              f"(this grid: {odd_now} odd bars, fit {fit_now:.1f} ms)")
        if (odd_b < odd_now and fit_now > BEAT_GRID_BAD and fit_b < BEAT_GRID_FIT * fit_now
                and not bar_hints):    # hinted odd meters stay
            beat_grid = "bar"
            T, bpms, starts = lock_to_beats()
    if tempo_at:
        # hint: the tempo changes exactly on these measures' first beat; the measure
        # before stays a plain 4/4 bar at the old tempo
        old = T[starts]
        for m in sorted(tempo_at):
            x = tempo.switch_at(sections, float(old[m - 2]))
            old[m - 1] = x
            print(f"      tempo change fixed at measure {m:03d} ({x:.3f}s)")
        T, bpms = tempo.tatum_grid(sections, duration, onsets, strength)
        starts = [int(np.argmin(np.abs(T - t))) for t in old[:max(tempo_at)]]
    hints = tempo.parse_bar_hints(bar_hints)
    fixed = {m + i for m, lens in hints for i in range(len(lens))} | set(tempo_at or [])
    if tempo_at:
        hints = sorted(hints + [(max(tempo_at), [])])   # bars after a fixed switch: 4/4 unless hinted
    if hints:
        starts = tempo.apply_bar_hints(starts, hints, len(T) - 1)
    if merge_bars:
        starts = tempo.merge_bars(starts, merge_bars)
    if grid_from:
        T, bpms, starts = grid_from_chart(Path(grid_from), hits, duration, max_bpm)
    lens = np.diff(starts)
    odd = {int(L): int((lens == L).sum()) for L in np.unique(lens) if L != 16}
    print(f"      {len(lens)} bars; non-4/4 bars (in 16ths): {odd or 'none'}")
    if score:
        notes, tom_log = close_open_hats(left_foot_busy(double_kick(score_notes(score, T, starts, score_first), starts, bpms))), []
    else:
        notes, tom_log = audio_notes(hits, T, starts, bpms, drums, backbeat, snare24, crash, kick_check,
                                     ride_foot, groove, swing)
    if ghosts:
        g = np.array(ghosts)
        soft = [n for n in notes if n.ch == "12" and np.min(np.abs(g - n.t)) < 0.002]
        for n in soft:
            n.vol = GHOST_VOL
        print(f"      ghost notes: {len(soft)} soft snares added (snare stem confirms them), at {GHOST_VOL:.0%} volume")
    notes = foot_hat_spacing(notes, starts, T)
    snare_rolls(notes, starts)
    notes = hat_rolls(notes, starts, T, bpms)
    notes = hat_under_crash(cymbal_pairs(notes), bpms)
    notes = crash_runs(notes)
    notes = roll_into_cymbal(notes, starts)
    if cymbal_roll:
        notes = cymbal_roll_hints(notes, starts, T, cymbal_roll, drums)
    if not score:
        unseen_rolls(notes, starts, T, drums, tom_log)
    if game_style:
        notes = ride_on_cymbal(notes)

    late = tom_log + busy_stream_snares(notes, starts, bpms, fixed | parse_measures(backbeat))
    bar, fast_bar = 16, (lambda m: m)
    if half:
        # the song is counted at half the tempo (e.g. 67.62, not 135.25). Charting at 67.62
        # from the start merges the fast 16ths, so keep the fast grid and write two bars as one
        # 4/4 bar at half the BPM. Pairs start where more crashes land.
        crash = accent_at(hits, T) >= 1.0
        p = int(crash[starts[1::2]].sum() > crash[starts[0::2]].sum())
        if p and starts[0] < 16:
            p = 0
        w = ([starts[0] - 16] if p else []) + starts[p::2]
        starts = w + ([starts[-1]] if w[-1] != starts[-1] else [])     # odd count: last bar 2/4
        bpms, bar = bpms / 2, 32
        fast_bar = lambda m: (m - 1 + p) // 2 + 1
        print(f"      half tempo: {len(starts) - 1} bars of 4/4 at {np.median(bpms):.2f} BPM")
    print("[5/5] writing chart ...")
    lag = grid_lag(hits, T)
    note_offset = TARGET_LAG - lag
    print(f"      grid lag vs hits {lag * 1000:+.1f} ms -> BGM shift {note_offset * 1000:+.1f} ms")
    bgm_src = rest if bgm_mode == "drumless" else audio
    bgm_name = "bgm" + bgm_src.suffix
    shutil.copy2(bgm_src, out_dir / bgm_name)
    wavs = {"01": bgm_name}
    volumes: dict[str, int] = {}
    wav_of, table = sounds.dress(notes, KIT, drum_parts(drums) if (snare_match or dynamics) else None,
                                 kit_file, snare_match, volume, dynamics, kit_volume=KIT_VOLUME)
    for n in notes:
        n.wav = wav_of[id(n)]
    names: dict[Path, str] = {}
    for wid, (src, vol) in sorted(table.items()):
        if src not in names:   # two library files may share a name (Snare.ogg)
            nm = src.name if src.name not in names.values() else f"{src.parent.name}_{src.name}"
            names[src] = nm
            shutil.copy2(src, out_dir / nm)
        wavs[wid] = names[src]
        if vol != 100:
            volumes[wid] = vol
    headers = {"TITLE": title, "ARTIST": artist, "BPM": "", "DLEVEL": str(dlevel), "BGMWAV": "01"}
    for key, f in (("PREVIEW", preview), ("PREIMAGE", preimage)):
        if f:
            shutil.copy2(f, out_dir / Path(f).name)
            headers[key] = Path(f).name
    chips, measure_len, bpm_defs, base_bpm = layout(notes, T, bpms, starts, note_offset, bar)
    headers["BPM"] = f"{base_bpm:.2f}"
    path = out_dir / "mstr.dtx"
    dtx.write_dtx(path, headers, wavs, chips, measure_len, bpm_defs, volumes)
    # A tempo switch without a ramp leaves a part-tatum the chart can't show, so notes after it
    # sit off by that much (e.g. +10 ms after a drumless intro). Re-aim the BGM at the
    # notes as written; charts within 3 ms are left alone.
    for _ in range(2):
        d = chart_lag(path, hits)
        if d is None or abs(d - TARGET_LAG) <= 0.003:
            break
        note_offset += TARGET_LAG - d
        print(f"      notes land {d * 1000:+.1f} ms after the hits -> BGM shift {note_offset * 1000:+.1f} ms")
        chips, measure_len, bpm_defs, base_bpm = layout(notes, T, bpms, starts, note_offset, bar)
        dtx.write_dtx(path, headers, wavs, chips, measure_len, bpm_defs, volumes)
    prev = None
    if not preview:   # first build: a clip of the original song (never the drumless BGM)
        prev = preview_sound.write(path, audio)
        headers["PREVIEW"] = "preview.ogg"
        dtx.write_dtx(path, headers, wavs, chips, measure_len, bpm_defs, volumes)
        print(f"      preview: measure {prev['bar']:03d}, {prev['bars']} bars of the original song")
    if video:
        add_video(path, Path(video), out_dir)
    # (a bar-phase check from the beat tracker was tried: too many false alarms on metal riffs)
    if half:   # those were counted in fast bars
        late = [re.sub(r"(?<=measures? )[\d, -]+",
                       lambda mm: re.sub(r"\d+", lambda d: f"{fast_bar(int(d.group())):03d}", mm.group()), c)
                for c in late]
    checks = [c for c in review(T, bpms, starts, hits, onsets, strength, max_bpm, beat_bpm, fixed, bar)
              if not (half and c.startswith("tempo octave"))
              and not (beat_grid and ": tempo " in c)] + late     # every bar has its own BPM there
    if checks:
        print("      CHECK (please confirm or correct):")
        for c in checks:
            print("        - " + c)
    info = {"sections": [s.__dict__ for s in sections], "bars": len(lens), "odd_bars": odd,
            "checks": checks}
    if prev:
        info["preview"] = prev
    (out_dir / "autochart.json").write_text(json.dumps(info, indent=2, default=float))

    # self-check: how far each charted note lands from the hit it came from
    chart = dtx.read_dtx(path)
    rolls = {n.wav for n in notes if n.vol != 1.0}   # cymbal rolls have no transcribed hits
    off = chart.bgm_offset()
    placed = np.array(sorted(chart.time_of(c.measure, c.pos) - off for c in chart.chips
                             if c.channel in dtx.DRUM_CHANNELS and c.value not in rolls))
    src = np.array(sorted(t for t, p in hits if p in ADTOF_TO_CH))
    if len(placed) and len(src):
        j = np.clip(np.searchsorted(src, placed), 1, len(src) - 1)
        err = np.minimum(np.abs(src[j - 1] - placed), np.abs(src[j] - placed)) * 1000
        print(f"      chart vs audio hits: median {np.median(err):.1f} ms, p95 {np.percentile(err, 95):.1f} ms")
    print(f"done: {path}  ({len(notes)} notes)")
    return path


def main():
    import library_check
    library_check.require(config.SETTINGS)       # only the owner's signed Library charts (see library_check.py)
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--title", default=None)
    ap.add_argument("--artist", default="Unknown")
    ap.add_argument("--bgm", choices=["full", "drumless"], default="full",
                    help="BGM in the folder: the original song (default) or the drums removed")
    ap.add_argument("--input", choices=["drums", "mix"], default="mix",
                    help="audio fed to the drum transcriber")
    ap.add_argument("--bars", default="",
                    help='meter hints, e.g. "76=78 78 78 44 x4; 92=44" (4/4 after each hint)')
    ap.add_argument("--tempo-at", type=int, nargs="*", default=None,
                    help="measures whose first beat carries a tempo change")
    ap.add_argument("--max-bpm", type=float, default=220.0,
                    help="tempo ceiling that picks the octave (ballad: 100)")
    ap.add_argument("--beat-grid", nargs="?", const="bar", default="", choices=["bar", "beat", "steps"],
                    help="drifting take: lock bar lines to the tracked beats, BPM chip per bar (or beat)")
    ap.add_argument("--kit", type=Path, default=None,
                    help="JSON: lane -> [sample file, volume] (your own samples)")
    ap.add_argument("--snare-match", action="store_true",
                    help="medley: per stretch pick the library snare closest to the recorded one")
    ap.add_argument("--volume", type=int, default=100, help="#VOLUME of every drum chip")
    ap.add_argument("--dynamics", action="store_true", help="3 volume steps per lane by hit strength")
    ap.add_argument("--preview", type=Path, default=None, help="preview sound (#PREVIEW)")
    ap.add_argument("--preimage", type=Path, default=None, help="preview picture (#PREIMAGE)")
    ap.add_argument("--dlevel", type=int, default=50)
    ap.add_argument("--ride-foot", action="store_true",
                    help="riding the ride: hi-hat is the left foot, add pedal chips on the 8ths")
    ap.add_argument("--kick-check", action="store_true",
                    help="noisy source: check every kick against the kick stem (drops bass-guitar bleed)")
    ap.add_argument("--half", action="store_true",
                    help="chart at half the detected tempo (two bars -> one 4/4 bar, 32nd grid)")
    ap.add_argument("--merge", type=tempo.parse_meter, default=0,
                    help='write neighbouring bars adding up to this meter as one bar, e.g. "74"')
    ap.add_argument("--groove", type=Path, default=None,
                    help="JSON lane -> 16 slots ('x'/'.'): bars close to it are charted as exactly this groove")
    ap.add_argument("--swing-hats", type=int, default=0, choices=[16, 24],
                    help="8th hats + stem-confirmed off-8th hats; late hat on the 16th or the 16th-triplet")
    ap.add_argument("--grid-from", type=Path, default=None,
                    help="take tempo and bar lines from another chart of this song (official or hand-fixed)")
    ap.add_argument("--game-style", action="store_true",
                    help="write the ride on the right cymbal lane, as official game charts do")
    ap.add_argument("--bpm", type=float, default=0.0,
                    help="the song's known tempo, kept for the whole song (bar lines and odd bars still found)")
    ap.add_argument("--tempo-from", default="",
                    help='new tempo section from this audio second, e.g. "58.22=170" (fit nearby)')
    ap.add_argument("--crash", default="",
                    help='crash hints: "39" = crash on beat 1 of measure 39, "4*" = every hi-hat of measure 4')
    ap.add_argument("--cymbal-roll", default="",
                    help='cymbal rolls the transcriber can\'t see: "9.5; 63-66; 112" (9.5 = second half '
                         'of measure 9); 16ths CY/LC, volume following the cymbal stem')
    ap.add_argument("--snare-24", action="store_true",
                    help="groove bars with snares on all four beats keep only beats 2 and 4")
    ap.add_argument("--backbeat", default="",
                    help='measures whose snare is beats 2 and 4 only, e.g. "88-91"')
    ap.add_argument("--score", type=Path, default=None,
                    help="score mode: notes from this drum-score transcription (score.txt), audio gives the grid")
    ap.add_argument("--score-first", type=int, default=2, help="measure that score bar 1 goes into")
    ap.add_argument("--genre", default="auto",
                    help="the song's genre, one or several ('metal,progressive'): " + ", ".join(genres.GENRES))
    ap.add_argument("--video", type=Path, default=None,
                    help="source video as BGA, started with the BGM (non-WMV is converted to WMV2)")
    a = ap.parse_args()
    build(a.audio.resolve(), a.out_dir, a.title or a.audio.stem, a.artist, a.bgm, a.input,
          a.bars, a.tempo_at, a.max_bpm, a.backbeat, a.half, a.kick_check, a.beat_grid,
          a.kit, a.snare_match, a.volume, a.dynamics, a.preview, a.preimage, a.dlevel, a.ride_foot,
          a.merge, a.groove, a.swing_hats, a.tempo_from, a.crash, a.snare_24,
          a.score, a.score_first, a.video, a.bpm, a.game_style, a.grid_from,
          a.cymbal_roll, a.genre)
    # the options this chart was made with, so a rebuild (e.g. from the app) can repeat them
    meta = a.out_dir / "autochart.json"
    info = json.loads(meta.read_text())
    info["build"] = sys.argv[3:]
    info["library"] = (HERE / "LIBRARY").read_text().strip()
    info["audio"] = str(a.audio.resolve())
    meta.write_text(json.dumps(info, indent=2, default=float))


if __name__ == "__main__":
    main()
