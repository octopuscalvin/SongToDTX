"""Write a simple synthesized drum kit (public domain, made here from scratch) into the kit folder,
one file per name in drumkit.py. Samples already there are kept, so your own sounds win.

usage: python make_kit.py [--force]"""
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from config import KIT_DIR
from drumkit import SAMPLE_NAMES

SR = 44100
rng = np.random.default_rng(7)


def t(sec):
    return np.arange(int(SR * sec)) / SR


def env(sec, decay, attack=0.002):
    x = t(sec)
    return np.minimum(x / attack, 1.0) * np.exp(-x / decay)


def band(x, lo=None, hi=None):
    if lo and hi:
        sos = butter(4, [lo, hi], "bandpass", fs=SR, output="sos")
    elif lo:
        sos = butter(4, lo, "highpass", fs=SR, output="sos")
    else:
        sos = butter(4, hi, "lowpass", fs=SR, output="sos")
    return sosfilt(sos, x)


def noise(sec):
    return rng.standard_normal(int(SR * sec))


def sweep(f0, f1, sec, k=25.0):
    x = t(sec)
    f = f1 + (f0 - f1) * np.exp(-x * k)
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def metal(freqs, sec, decay):
    x = t(sec)
    s = sum(np.sign(np.sin(2 * np.pi * f * x + rng.uniform(0, 6))) for f in freqs)
    return band(s, 3000, 16000) * env(sec, decay)


def make(name):
    if name == "kick":
        return sweep(150, 48, 0.5, 30) * env(0.5, 0.18) + 0.3 * band(noise(0.5), 1500, 6000) * env(0.5, 0.004)
    if name == "snare":
        return 0.6 * sweep(260, 185, 0.35, 40) * env(0.35, 0.06) + band(noise(0.35), 1200, 9000) * env(0.35, 0.09)
    if name.startswith("tom_"):
        f = {"tom_high": 210, "tom_mid": 155, "tom_floor": 105}[name]
        return sweep(f * 1.35, f, 0.7, 12) * env(0.7, 0.22) + 0.15 * band(noise(0.7), 800, 5000) * env(0.7, 0.01)
    if name == "hh_closed":
        return band(noise(0.2), 7000, 16000) * env(0.2, 0.025) + 0.5 * metal([540, 810, 1150, 1560], 0.2, 0.02)
    if name == "hh_open":
        return band(noise(1.0), 6000, 16000) * env(1.0, 0.28) + 0.5 * metal([540, 810, 1150, 1560], 1.0, 0.25)
    if name == "hh_pedal":
        return band(noise(0.15), 3000, 9000) * env(0.15, 0.018, 0.004)
    if name in ("crash_right", "crash_left"):
        base = [420, 610, 890, 1330, 1810, 2470] if name == "crash_right" else [380, 560, 830, 1240, 1700, 2310]
        return band(noise(2.5), 3500, 16000) * env(2.5, 0.75) + 0.7 * metal(base, 2.5, 0.9)
    if name == "ride":
        x = t(2.0)
        bell = sum(a * np.sin(2 * np.pi * f * x) for f, a in ((620, 1.0), (1490, 0.6), (3270, 0.35)))
        return 0.5 * bell * env(2.0, 0.6) + 0.35 * band(noise(2.0), 5000, 15000) * env(2.0, 0.5)
    raise KeyError(name)


# old-school digital drum machine sounds (the classic 80s machines' recipes, made here from scratch):
# boomy pitched kick, gated snare, pitch-dropping electronic toms, square-wave metallic cymbals
MACHINE_METAL = [205.3, 304.4, 369.6, 522.7, 540.0, 800.0]


def gate(y, sec):
    n = int(SR * sec)
    y = y.copy()
    y[n:] *= np.exp(-np.arange(len(y) - n) / (0.004 * SR))
    return y


def make_digi(name):
    if name == "kick":
        return 1.2 * sweep(220, 46, 0.6, 22) * env(0.6, 0.30) + 0.25 * band(noise(0.6), 2000, 8000) * env(0.6, 0.003)
    if name == "snare":
        tone = sweep(330, 180, 0.4, 30) + 0.6 * np.sin(2 * np.pi * 476 * t(0.4))
        return gate(0.5 * tone * env(0.4, 0.05) + band(noise(0.4), 1500, 11000) * env(0.4, 0.2), 0.16)
    if name.startswith("tom_"):
        f = {"tom_high": 240, "tom_mid": 170, "tom_floor": 115}[name]
        return sweep(f * 2.2, f, 0.8, 7) * env(0.8, 0.3) + 0.2 * band(noise(0.8), 1000, 6000) * env(0.8, 0.008)
    if name == "hh_closed":
        return metal(MACHINE_METAL, 0.15, 0.018) + 0.3 * band(noise(0.15), 8000, 16000) * env(0.15, 0.012)
    if name == "hh_open":
        return metal(MACHINE_METAL, 0.8, 0.22) + 0.3 * band(noise(0.8), 8000, 16000) * env(0.8, 0.18)
    if name == "hh_pedal":
        return metal(MACHINE_METAL, 0.1, 0.01)
    if name in ("crash_right", "crash_left"):
        k = 1.0 if name == "crash_right" else 0.94
        return metal([f * 1.9 * k for f in MACHINE_METAL], 2.0, 0.7) + 0.6 * band(noise(2.0), 4000, 16000) * env(2.0, 0.6)
    if name == "ride":
        return 0.8 * metal([f * 2.6 for f in MACHINE_METAL], 1.5, 0.5) + 0.3 * np.sin(2 * np.pi * 760 * t(1.5)) * env(1.5, 0.4)
    raise KeyError(name)


def write_digi(folder):
    """The digital kit into folder (kept if already there)."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in SAMPLE_NAMES:
        p = folder / f"{name}.wav"
        if not p.exists():
            y = make_digi(name)
            sf.write(str(p), (0.89 * y / np.max(np.abs(y))).astype(np.float32), SR, subtype="PCM_16")


def main():
    force = "--force" in sys.argv
    KIT_DIR.mkdir(parents=True, exist_ok=True)
    for name in SAMPLE_NAMES:
        if not force and any((KIT_DIR / (name + e)).exists() for e in (".ogg", ".wav", ".flac")):
            continue
        y = make(name)
        y = 0.89 * y / np.max(np.abs(y))
        sf.write(str(KIT_DIR / f"{name}.wav"), y.astype(np.float32), SR, subtype="PCM_16")
        print("wrote", KIT_DIR / f"{name}.wav")


if __name__ == "__main__":
    main()
