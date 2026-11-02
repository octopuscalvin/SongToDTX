"""Score mode, step 1 : from a drum-cover video with a
scrolling drum score and a blue cursor, and the song's source video:
  * download both (yt-dlp) when given URLs,
  * find how far the cover's audio sits after the source audio (onset cross-correlation),
  * find the score pages (the cursor jumps from the right edge back to the left at a page turn)
    and save each page mid-interval (the cursor jumps ~1 s before the page redraws) plus a zoomed
    two-half view to read it from.
Then the pages are read into score.txt (see autochart.score_notes) and charted with
  autochart.py source.wav <out> --score score.txt --score-first 2 --video source.mp4

usage: video_score.py <chart video URL|file> <source video URL|file> <work dir> [--strip 0.83]"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np

FPS = 4


def fetch(src: str, work: Path, name: str, height: int) -> Path:
    if Path(src).exists():
        return Path(src)
    out = work / f"{name}.mp4"
    if not out.exists():
        subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-playlist", "-f",
                        f"bv*[height<={height}][ext=mp4]+ba[ext=m4a]/b", "-o", str(out), src], check=True)
    return out


def wav(video: Path, out: Path, sr: int, mono: bool) -> Path:
    if not out.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-ar", str(sr)]
                       + (["-ac", "1"] if mono else []) + [str(out)], check=True)
    return out


def offset(a: Path, b: Path) -> float:
    """Seconds the chart video's audio runs behind the source's (median over 20 s windows)."""
    import soundfile as sf
    from scipy.signal import fftconvolve
    def env(f, hop):
        x, _ = sf.read(str(f))
        n = len(x) // hop
        e = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1))
        o = np.maximum(np.diff(np.log(e + 1e-6)), 0)
        return o - o.mean()
    ca, cb = env(a, 220), env(b, 220)
    c = fftconvolve(ca, cb[::-1])
    k = np.argmax(c) - (len(cb) - 1)                    # rough, 10 ms
    fa, fb, dt = env(a, 55), env(b, 55), 55 / 22050
    base, m, offs = int(k * 0.01 / dt), int(0.08 / dt), []
    for s in range(10, int(len(fb) * dt) - 20, 20):
        i0 = int(s / dt)
        seg = fb[i0:i0 + int(15 / dt)]
        w = fa[i0 + base - m:i0 + base + len(seg) + m]
        if len(w) >= len(seg) + 2 * m and i0 + base - m >= 0:
            offs.append((np.argmax(np.correlate(w, seg, "valid")) - m + base) * dt)
    return float(np.median(offs))


def pages(video: Path, work: Path, strip: float) -> list[tuple[float, float]]:
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                      "stream=width,height", "-of", "json", str(video)],
                                     capture_output=True, text=True).stdout)["streams"][0]
    W, H = info["width"], info["height"]
    y0 = int(H * strip)
    h = H - y0
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={FPS},crop={W}:{h}:0:{y0}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    cur, i = [], 0
    while True:
        b = p.stdout.read(W * h * 3)
        if len(b) < W * h * 3:
            break
        f = np.frombuffer(b, np.uint8).reshape(h, W, 3).astype(int)
        cols = np.flatnonzero(((f[..., 2] - f[..., 0]) > 40).sum(0) > h // 6)
        if len(cols):
            cur.append((i / FPS, float(np.median(cols))))
        i += 1
    turns = [cur[0][0]] + [t1 for (t0, x0), (t1, x1) in zip(cur, cur[1:]) if x0 - x1 > W * 0.36]
    ends = turns[1:] + [cur[-1][0] + 1 / FPS]
    out = [(a, b - a) for a, b in zip(turns, ends) if b - a > 1.0]
    (work / "pages").mkdir(exist_ok=True)
    for k, (a, d) in enumerate(out, 1):
        crop = f"crop={W}:{h}:0:{y0}"
        half = W * 52 // 100
        view = (f"[0]{crop},split[l][r];[l]crop={half}:{h}:0:0,scale=1500:-2[a];"
                f"[r]crop={half}:{h}:{W - half}:0,scale=1500:-2[b];[a][b]vstack")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a + d / 2:.2f}", "-i", str(video), "-frames:v", "1",
                        "-filter_complex", view, str(work / "pages" / f"v{k:02d}.png")], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a + d / 2:.2f}", "-i", str(video), "-frames:v", "1",
                        "-vf", crop, str(work / "pages" / f"{k:02d}.png")], check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chart")
    ap.add_argument("source")
    ap.add_argument("work", type=Path)
    ap.add_argument("--strip", type=float, default=0.83, help="score strip starts at this fraction of the height")
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    chart = fetch(a.chart, a.work, "chart", 1080)
    source = fetch(a.source, a.work, "source", 720)
    wav(source, a.work / "source.wav", 44100, False)
    off = offset(wav(chart, a.work / "chart_m.wav", 22050, True), wav(source, a.work / "source_m.wav", 22050, True))
    pg = pages(chart, a.work, a.strip)
    json.dump({"offset": off, "pages": pg}, open(a.work / "pages.json", "w"), indent=1)
    print(f"chart video = source + {off:.3f} s; {len(pg)} score pages in {a.work / 'pages'}")
    for k, (t, d) in enumerate(pg, 1):
        print(f"  page {k:02d}: {t:7.2f} s (source {t - off:7.2f} s), {d:5.2f} s")


if __name__ == "__main__":
    main()
