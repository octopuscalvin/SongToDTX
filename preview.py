"""Preview sound (#PREVIEW): a stretch of whole bars of the original song with a volume shape,
like an arcade song-select clip. Used by autochart.py (first build) and the app (re-cut, reshape)."""
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

import dtx

BARS = (5, 17, 8)          # shortest, longest, default clip in bars
FADE_IN = (0.8, 1.0)       # volume at the start, reached full at 1 s
FADE_OUT_MAX = 5.0         # seconds; the fade-out takes the last 30% of the clip, at most this


def default_shape(length: float) -> list[list[float]]:
    out = min(FADE_OUT_MAX, length * 0.3)
    return [[0.0, FADE_IN[0]], [1.0, FADE_IN[1]], [round(max(1.2, length - out), 3), 1.0], [round(length, 3), 0.0]]


def bar_seconds(ch: dtx.Chart, bar: float) -> float:
    """BGM seconds at a bar number with a fraction (64.25 = a quarter into measure 064)."""
    m = int(bar)
    return ch.time_of(m, Fraction(bar - m).limit_denominator(64)) - ch.bgm_offset()


def span(ch: dtx.Chart, bar: float, bars: float) -> tuple[float, float]:
    bars = min(max(bars, BARS[0]), BARS[1])
    a = max(0.0, bar_seconds(ch, bar))
    return a, bar_seconds(ch, bar + bars) - a


def gain(shape: list, t: np.ndarray) -> np.ndarray:
    """Volume at clip seconds t: straight lines between the shape's (seconds, volume) points."""
    pts = sorted((float(x), float(g)) for x, g in shape) or [(0.0, 1.0)]
    return np.interp(t, [p[0] for p in pts], [p[1] for p in pts])


def cut(src: Path, start: float, length: float, out: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(src),
                    "-vn", "-ac", "2", "-ar", "44100", str(out)], check=True)


def render(src: Path, start: float, length: float, shape: list, out_ogg: Path) -> None:
    import soundfile as sf
    raw = out_ogg.with_name(out_ogg.stem + "_raw.wav")
    cut(src, start, length, raw)
    x, sr = sf.read(str(raw))
    if shape:
        x = x * gain(shape, np.arange(len(x)) / sr)[:, None]
    sf.write(str(raw), np.clip(x, -1, 1), sr)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-c:a", "libvorbis", "-q:a", "5", str(out_ogg)],
                   check=True)
    raw.unlink()


def pick_bar(ch: dtx.Chart, src: Path, bars: int = BARS[2]) -> int:
    """Start bar of the loudest stretch of the song (usually the first big chorus): the loudest
    `bars`-bar window in the first 70% of the song, ties going to the earlier one."""
    import soundfile as sf
    x, sr = sf.read(str(src), always_2d=True)
    e = (x.mean(1) ** 2)
    hop = sr // 10
    rms = np.sqrt(e[:len(e) // hop * hop].reshape(-1, hop).mean(1))          # 0.1 s frames
    last = max(c.measure for c in ch.chips) if ch.chips else 0
    best, best_m = -1.0, 1
    for m in range(1, last + 1):
        a, n = span(ch, m, bars)
        if a + n > 0.7 * len(rms) / 10:
            break
        v = rms[int(a * 10):int((a + n) * 10)]
        if len(v) and v.mean() > best * 1.03:   # a later window must be clearly louder
            best, best_m = float(v.mean()), m
    return best_m


def write(chart_path: Path, src: Path, bar: float | None = None, bars: float = BARS[2],
          shape: list | None = None) -> dict:
    """preview.ogg next to the chart, from the original song `src`; returns the choice
    (bar, bars, shape) for autochart.json."""
    ch = dtx.read_dtx(chart_path)
    if bar is None:
        bar = pick_bar(ch, src, int(bars))
    a, n = span(ch, bar, bars)
    shape = shape if shape else default_shape(n)
    render(src, a, n, shape, chart_path.parent / "preview.ogg")
    return {"bar": bar, "bars": bars, "shape": shape}
