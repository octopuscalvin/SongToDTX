"""Adds a music video to a chart you already have, as its BGA:
  * download the video (yt-dlp) when given a URL,
  * find where the chart's BGM sits in the video's sound (onset cross-correlation, 10 s windows
    over the whole song, so a different edit or an extra scene is noticed and reported by measure),
  * cut the video's extra intro (or put black in front when the video starts later than the BGM),
    convert it to WMV2 without sound and start it with the BGM chip (#AVI01, channel 54).
The chart is backed up once to <name>.before_mv.dtx; earlier #AVI lines and channel 54 are replaced.

usage: add_mv.py <chart folder or .dtx> <video URL|file> <work dir>"""
import argparse, re, shutil, subprocess, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import dtx

SR, HOP = 22050, 55                     # 2.5 ms onset frames
WIN = 10                                # seconds per matching window
NAME = "bga_mv.wmv"
CRLF = chr(13) + chr(10)


def fetch(src: str, work: Path) -> Path:
    if Path(src).exists():
        return Path(src)
    out = work / "mv.mp4"
    if not out.exists():                # H.264 first: DTXMania-friendly and quick to decode
        subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-playlist", "-f",
                        "bv*[height<=720][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=720][ext=mp4]+ba[ext=m4a]/b",
                        "-o", str(out), src], check=True)
    return out


def mono(src: Path, out: Path) -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ar", str(SR), "-ac", "1", str(out)],
                   check=True)
    return out


def onsets(f: Path) -> np.ndarray:
    import soundfile as sf
    x, _ = sf.read(str(f))
    n = len(x) // HOP
    e = np.sqrt((x[:n * HOP].reshape(n, HOP) ** 2).mean(1))
    o = np.maximum(np.diff(np.log(e + 1e-6)), 0)
    return o - o.mean()


def windows(video: np.ndarray, bgm: np.ndarray) -> list[tuple[float, float, np.ndarray, int]]:
    """(BGM second, video second - BGM second of the best match, correlation, window start) for every
    10 s window of the BGM that has a clear match anywhere in the video; quiet windows are left out."""
    from scipy.signal import fftconvolve
    dt, w, out = HOP / SR, int(WIN / HOP * SR), []
    for i0 in range(0, len(bgm) - w, w):
        seg = bgm[i0:i0 + w]
        if seg.std() < 1e-3:
            continue
        c = fftconvolve(video, seg[::-1], "valid")
        k = int(np.argmax(c))
        if c[k] > 12 * (np.abs(c).mean() + 1e-9):
            out.append((i0 * dt, (k - i0) * dt, c, i0))
    return out


def out_of_sync(win, d: float) -> list[float]:
    """BGM seconds whose window matches much worse at offset d than at its best spot. A repeated
    chorus matches equally well in two places, so only a clearly weaker match counts."""
    k, m = round(d * SR / HOP), int(0.05 * SR / HOP)
    bad = []
    for t, _, c, i0 in win:
        near = c[max(0, i0 + k - m):max(0, i0 + k + m + 1)]
        if len(near) == 0 or near.max() < 0.6 * c.max():
            bad.append(t)
    return bad


def bgm_of(path: Path) -> tuple[Path, str, int, int, int]:
    """BGM file, and where its first chip sits: (file, line, measure, slot, slots in the line)."""
    lines = path.read_text(encoding="shift_jis", errors="replace").splitlines()
    first = None
    for l in lines:
        m = re.match(r"#(\d{3})01:\s*(\S+)", l)
        if m:
            d = m[2]
            for s in range(len(d) // 2):
                if d[2 * s:2 * s + 2] != "00":
                    key = (int(m[1]), s / (len(d) // 2))
                    if first is None or key < first[0]:
                        first = (key, d[2 * s:2 * s + 2], int(m[1]), s, len(d) // 2)
                    break
    if not first:
        raise SystemExit(f"{path.name} has no BGM chip (channel 01)")
    wav = next((re.split(r":\s*|\s+", l, 1)[1].strip() for l in lines
                if l.upper().startswith(f"#WAV{first[1]}")), None)
    if not wav or not (path.parent / wav).exists():
        raise SystemExit(f"BGM file #WAV{first[1]} not found next to {path.name}")
    return path.parent / wav, *first[2:]


def attach(path: Path, name: str, measure: int, slot: int, slots: int) -> None:
    """#AVI01 after #BGMWAV and a channel 54 chip in the BGM chip's slot; replaces earlier BGA."""
    lines = [l for l in path.read_text(encoding="shift_jis", errors="replace").splitlines()
             if not re.match(r"#AVI[0-9A-Z]{2}[:\s]", l, re.I) and not re.match(r"#\d{3}54:", l)]
    at = next((i + 1 for i, l in enumerate(lines) if l.upper().startswith("#BGMWAV")), 0)
    lines.insert(at, f"#AVI01: {name}")
    lines.append(f"#{measure:03d}54: " + "".join("01" if s == slot else "00" for s in range(slots)))
    path.write_text(CRLF.join(lines) + CRLF, encoding="shift_jis", errors="replace")


def measure_at(ch: dtx.Chart, t: float) -> int:
    m = 0
    while ch.time_of(m + 1, 0) <= t and m < 999:
        m += 1
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chart", type=Path)
    ap.add_argument("video")
    ap.add_argument("work", type=Path)
    a = ap.parse_args()
    path = a.chart if a.chart.suffix.lower() == ".dtx" else (
        a.chart / "mstr.dtx" if (a.chart / "mstr.dtx").exists() else next(a.chart.glob("*.dtx"), None))
    if not path or not path.exists():
        raise SystemExit(f"No .dtx in {a.chart}")
    a.work.mkdir(parents=True, exist_ok=True)

    print("[1/3] getting the video ...", flush=True)
    video = fetch(a.video, a.work)
    bgm, measure, slot, slots = bgm_of(path)

    print("[2/3] matching the video's sound to the chart's BGM ...", flush=True)
    win = windows(onsets(mono(video, a.work / "mv_m.wav")), onsets(mono(bgm, a.work / "bgm_m.wav")))
    if len(win) < 3:
        raise SystemExit("The video's sound doesn't match this chart's music. Is it the same song?")
    d = float(np.median([w[1] for w in win]))
    ch = dtx.read_dtx(path)
    start = ch.bgm_offset()
    off = sorted({measure_at(ch, start + t) for t in out_of_sync(win, d)})
    print(f"      {len(win)} parts of the song matched; video = BGM {d:+.3f} s")
    if off:
        spans = [[off[0], off[0]]]
        for m in off[1:]:                       # neighbouring windows are ~8 measures apart
            if m - spans[-1][1] <= 20:
                spans[-1][1] = m
            else:
                spans.append([m, m])
        print("OUT OF SYNC: the video is a different edit at measure "
              + ", ".join(f"{a:03d}" if a == b else f"{a:03d}-{b:03d}" for a, b in spans)
              + " (extra scene, cut or other mix): the picture is out of sync there")

    print("[3/3] cutting and converting the video ...", flush=True)
    vf = "scale=1280:-2" if d >= 0 else f"tpad=start_duration={-d:.3f}:color=black,scale=1280:-2"
    cut = ["-ss", f"{d:.3f}"] if d > 0 else []
    tmp = a.work / NAME
    subprocess.run(["ffmpeg", "-v", "error", "-y", *cut, "-i", str(video), "-an", "-vf", vf,
                    "-c:v", "wmv2", "-b:v", "3500k", str(tmp)], check=True)
    shutil.copy2(tmp, path.parent / NAME)
    backup = path.with_suffix(".before_mv.dtx")
    if not backup.exists():
        shutil.copy2(path, backup)
    attach(path, NAME, measure, slot, slots)
    what = (f"cut the first {d:.2f} s of the video" if d > 0.0005 else
            f"put {-d:.2f} s of black before the video" if d < -0.0005 else "no cut needed")
    print(f"MV added to {path}: {what}; it starts with the BGM at measure {measure:03d}")


if __name__ == "__main__":
    main()
