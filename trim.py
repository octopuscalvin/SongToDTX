"""Short version of a chart: the notes of bars START..END, then FADE_BARS more bars of music fading
out, and nothing after that. The kept part becomes bars 001.. (000 stays an empty lead-in bar) and
the BGM (and a BGA video, if any) are cut to match.

usage: trim.py <chart folder> <new folder> <start bar> <end bar>"""
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf

import dtx
import preview

FADE_BARS = 3               # bars of music after the last notes, fading to silence
FADE_IN = 0.03              # seconds: a cut in the middle of the song would click
END_WAV, END_ID = "trim_end.wav", "ZZ"   # a silent chip after the fade: the song ends there, not earlier
SKIP = {"mstr.dtx", "mstr.auto.dtx", "job.json", "preview.ogg", "autochart.json"}


def bpm_at(ch: dtx.Chart, m: int) -> float:
    """The tempo in effect at the start of measure m."""
    bpm = ch.base_bpm
    for em, ep, b in ch._tempo_events():
        if (em, ep) <= (m, Fraction(0)):
            bpm = b
    return bpm


def cut_audio(src: Path, out: Path, start: float, length: float, fade_at: float) -> None:
    """start..start+length seconds of src, full volume until fade_at, then down to silence."""
    tmp = out.with_name(out.stem + "_cut.wav")
    preview.cut(src, start, length, tmp)
    x, sr = sf.read(str(tmp))
    shape = [[0.0, 0.0 if start > 0 else 1.0], [FADE_IN, 1.0], [fade_at, 1.0], [length, 0.0]]
    x = x * preview.gain(shape, np.arange(len(x)) / sr)[:, None]
    sf.write(str(out), np.clip(x, -1, 1), sr)
    tmp.unlink()


def cut_video(src: Path, out: Path, start: float, length: float, fade_at: float) -> None:
    tmp = out.with_name("trim_" + out.name)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{length:.3f}", "-an",
                    "-vf", f"fade=t=out:st={fade_at:.3f}:d={max(0.1, length - fade_at):.3f}",
                    "-c:v", "wmv2", "-b:v", "3500k", str(tmp)], check=True)
    tmp.replace(out)


def trim(src: Path, dst: Path, start: int, end: int, title: str | None = None) -> dict:
    ch = dtx.read_dtx(src / "mstr.dtx")
    last = max((c.measure for c in ch.chips if c.channel in dtx.DRUM_CHANNELS), default=0)
    if not 1 <= start <= end <= last:
        raise ValueError(f"bars must be within 001-{last:03d}, start before end")
    stop = end + FADE_BARS                      # last bar kept
    shift = start - 1
    off = ch.bgm_offset()
    t0 = ch.time_of(start, Fraction(0)) - off   # in BGM seconds
    t_fade = ch.time_of(end + 1, Fraction(0)) - off
    t1 = ch.time_of(stop + 1, Fraction(0)) - off

    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name not in SKIP:
            shutil.copy2(f, dst / f.name)

    chips = []
    for c in ch.chips:
        if c.channel in ("01", "54"):
            continue
        if c.channel in dtx.DRUM_CHANNELS and not start <= c.measure <= end:
            continue
        if c.channel in ("03", "08") and (c.measure, c.pos) <= (start, Fraction(0)):
            continue                                # folded into #BPM
        if start <= c.measure <= stop:
            chips.append(dtx.Chip(c.measure - shift, c.pos, c.channel, c.value))

    # BGM and BGA: cut so they line up with the new bar 001
    files = {"01": "WAV", "54": "AVI"}
    for c in [c for c in ch.chips if c.channel in files]:
        name = ch.headers.get(files[c.channel] + c.value)
        tc = ch.time_of(c.measure, c.pos) - off
        if not name or tc >= t1 or not (src / name).exists():
            continue
        if tc <= t0:                                # playing when the short version starts
            at, pos, cut_from = 1, Fraction(0), t0 - tc
        else:                                       # starts later (the start bar is before it)
            at, pos, cut_from = c.measure - shift, c.pos, 0.0
        length, fade_at = t1 - (tc + cut_from), t_fade - (tc + cut_from)
        if c.channel == "01":
            cut_audio(src / name, dst / name, cut_from, length, fade_at)
        else:
            cut_video(src / name, dst / name, cut_from, length, fade_at)
        chips.append(dtx.Chip(at, pos, c.channel, c.value))

    # silence at the end of the fade, so DTXMania plays the fade before the result screen
    sf.write(str(dst / END_WAV), np.zeros((4410, 2)), 44100)
    chips.append(dtx.Chip(stop - shift + 1, Fraction(0), "61", END_ID))

    heads = {k: v for k, v in ch.headers.items()
             if not (k.startswith(("WAV", "VOLUME", "PAN")) or k == "PREVIEW")}
    heads["BPM"] = f"{bpm_at(ch, start):.2f}"
    if title:
        heads["TITLE"] = title
    wavs = {k[3:]: v for k, v in ch.headers.items() if k.startswith("WAV") and len(k) == 5}
    wavs[END_ID] = END_WAV
    vols = {k[6:]: int(v) for k, v in ch.headers.items() if k.startswith("VOLUME") and len(k) == 8 and v.isdigit()}
    pans = {k[3:]: int(v) for k, v in ch.headers.items() if k.startswith("PAN") and len(k) == 5
            and v.lstrip("-").isdigit()}
    mlen = {m - shift: v for m, v in ch.measure_len.items() if start <= m <= stop}
    dtx.write_dtx(dst / "mstr.dtx", heads, wavs, chips, mlen, ch.bpm_defs, vols, pans)
    return {"bars": stop - shift, "start": start, "end": end, "seconds": round(t1 - t0, 1)}


def main():
    a = sys.argv[1:]
    print(trim(Path(a[0]), Path(a[1]), int(a[2]), int(a[3])))


if __name__ == "__main__":
    main()
