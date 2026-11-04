"""Adds a music video to a chart you already have, as its BGA:
  * download the video (yt-dlp) when given a URL,
  * find where the chart's BGM sits in the video's sound (onset cross-correlation, 10 s windows
    over the whole song, so a different edit or an extra scene is noticed and reported by measure),
  * cut the video's extra intro (or put black in front when the video starts later than the BGM),
    convert it to WMV2 without sound and start it with the BGM chip (#AVI01, channel 54).
With --live the video's music may run at another, changing tempo (a live take, a slower or
faster performance): the picture is fitted to the BGM moment by moment from both recordings'
drum hits (live_mv.py) instead of one cut and stretch.
The chart is backed up once to <name>.dtx.before_mv (not .dtx, or DTXMania lists it as a song); earlier #AVI lines and channel 54 are replaced.

usage: add_mv.py <chart folder or .dtx> <video URL|file> <work dir> [--live]"""
import argparse, re, shutil, subprocess, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import dtx

SR, HOP = 22050, 55                     # 2.5 ms onset frames
WIN = 10                                # seconds per matching window
NAME = "bga_mv.wmv"
# DTXMania places a clip by its pixel size: anything but 1280x720 with square pixels lands off frame,
# so other shapes are fitted inside it with black bars
FRAME = ("scale=1280:720:force_original_aspect_ratio=decrease,"
         "pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,setsar=1")
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


CLEAR, FAIR = 12, 6      # correlation peak over its mean: a sure match / good enough to measure drift
DRIFT_MIN = 0.02         # seconds over the song: a slower or faster video is stretched from here on


def windows(video: np.ndarray, bgm: np.ndarray) -> list[tuple[float, float, np.ndarray, int, float]]:
    """(BGM second, video second - BGM second of the best match, correlation, window start, peak
    over mean) for every 10 s window of the BGM that matches somewhere in the video; quiet windows
    are left out."""
    from scipy.signal import fftconvolve
    dt, w, out = HOP / SR, int(WIN / HOP * SR), []
    for i0 in range(0, len(bgm) - w, w):
        seg = bgm[i0:i0 + w]
        if seg.std() < 1e-3:
            continue
        c = fftconvolve(video, seg[::-1], "valid")
        k = int(np.argmax(c))
        ratio = c[k] / (np.abs(c).mean() + 1e-9)
        if ratio > FAIR:
            out.append((i0 * dt, (k - i0) * dt, c, i0, ratio))
    return out


def out_of_sync(win, d: float, s: float = 0.0) -> list[float]:
    """BGM seconds whose window matches much worse at offset d (+ drift s per second) than at its
    best spot. A repeated chorus matches equally well in two places, so only a clearly weaker match counts."""
    m = int(0.05 * SR / HOP)
    bad = []
    for t, _, c, i0, _ in win:
        k = round((d + s * t) * SR / HOP)
        near = c[max(0, i0 + k - m):max(0, i0 + k + m + 1)]
        if len(near) == 0 or near.max() < 0.6 * c.max():
            bad.append(t)
    return bad


def bgm_of(path: Path) -> tuple[Path, str, int, int, int]:
    """BGM file, and where its first chip sits: (file, line, measure, slot, slots in the line)."""
    lines = path.read_text(encoding="cp932", errors="replace").splitlines()
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
    lines = [l for l in path.read_text(encoding="cp932", errors="replace").splitlines()
             if not re.match(r"#AVI[0-9A-Z]{2}[:\s]", l, re.I) and not re.match(r"#\d{3}54:", l)]
    at = next((i + 1 for i, l in enumerate(lines) if l.upper().startswith("#BGMWAV")), 0)
    lines.insert(at, f"#AVI01: {name}")
    lines.append(f"#{measure:03d}54: " + "".join("01" if s == slot else "00" for s in range(slots)))
    path.write_bytes((CRLF.join(lines) + CRLF).encode("cp932", errors="replace"))


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
    ap.add_argument("--live", action="store_true",
                    help="the video's music runs at another, changing tempo: fit the picture moment by moment")
    a = ap.parse_args()
    path = a.chart if a.chart.suffix.lower() == ".dtx" else (
        a.chart / "mstr.dtx" if (a.chart / "mstr.dtx").exists() else next(a.chart.glob("*.dtx"), None))
    if not path or not path.exists():
        raise SystemExit(f"No .dtx in {a.chart}")
    a.work.mkdir(parents=True, exist_ok=True)

    print("[1/3] getting the video ...", flush=True)
    video = fetch(a.video, a.work)
    bgm, measure, slot, slots = bgm_of(path)
    if a.live:
        live(path, video, bgm, measure, slot, slots, a.work)
        return

    print("[2/3] matching the video's sound to the chart's BGM ...", flush=True)
    win = windows(onsets(mono(video, a.work / "mv_m.wav")), onsets(mono(bgm, a.work / "bgm_m.wav")))
    if len(win) < 3:
        raise SystemExit("The video's sound doesn't match this chart's music. Is it the same song?")
    # the offset most windows agree on (weighted by how clearly each matched): with a different edit
    # the windows scatter, and a plain median lands between them on no real offset
    best = max((w[1] for w in win), key=lambda o: sum(x[4] for x in win if abs(x[1] - o) < 0.1))
    d = float(np.median([w[1] for w in win if abs(w[1] - best) < 0.1]))
    # a video from another master can run a little fast or slow: fit offset = d0 + s * t
    inl = [w for w in win if abs(w[1] - d) < 0.1]
    s = 0.0
    if len(inl) >= 6:
        s, d0 = np.polyfit([w[0] for w in inl], [w[1] for w in inl], 1)
        if abs(s * inl[-1][0]) > DRIFT_MIN:
            d = float(d0)
            print(f"      the video runs {abs(s) * 100:.3f}% {'slow' if s > 0 else 'fast'} "
                  f"({s * inl[-1][0] * 1000:+.0f} ms by the end): stretched to fit")
        else:
            s = 0.0
    win = [w for w in win if w[4] > CLEAR] or win
    ch = dtx.read_dtx(path)
    start = ch.bgm_offset()
    off = sorted({measure_at(ch, start + t) for t in out_of_sync(win, d, s)})
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
    vf = ("" if d >= 0 else f"tpad=start_duration={-d:.3f}:color=black,") + \
        (f"setpts=PTS/{1 + s:.7f}," if s else "") + FRAME
    cut = ["-ss", f"{d:.3f}"] if d > 0 else []
    tmp = a.work / NAME
    subprocess.run(["ffmpeg", "-v", "error", "-y", *cut, "-i", str(video), "-an", "-vf", vf,
                    "-c:v", "wmv2", "-b:v", "3500k", str(tmp)], check=True)
    shutil.copy2(tmp, path.parent / NAME)
    backup = path.with_name(path.name + ".before_mv")
    if not backup.exists():
        shutil.copy2(path, backup)
    attach(path, NAME, measure, slot, slots)
    what = (f"cut the first {d:.2f} s of the video" if d > 0.0005 else
            f"put {-d:.2f} s of black before the video" if d < -0.0005 else "no cut needed")
    print(f"MV added to {path}: {what}; it starts with the BGM at measure {measure:03d}")


def live(path: Path, video: Path, bgm: Path, measure: int, slot: int, slots: int, work: Path) -> None:
    import live_mv
    print("[2/3] fitting the video to the music moment by moment ...", flush=True)
    audio = work / "mv_audio.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "2", "-ar", "44100", str(audio)],
                   check=True)
    tmp = work / NAME
    med, share, loose, speed = live_mv.fit(bgm, video, audio, tmp, FRAME)
    tempo = (1 - 1 / speed) * 100   # + = the video's music is slower
    print(f"      the video's music is {abs(tempo):.1f}% {'slower' if tempo > 0 else 'faster'} than the chart's "
          f"(and wanders): fitted; drum hits {med:.0f} ms apart (median), {share:.0%} within 40 ms")
    ch = dtx.read_dtx(path)
    start = ch.bgm_offset()
    if loose:
        print("OUT OF SYNC: no drums to line up at measure "
              + ", ".join(f"{measure_at(ch, start + a):03d}-{measure_at(ch, start + b):03d}" for a, b in loose)
              + " (the picture follows the music only there)")
    if share < 0.5:
        print("OUT OF SYNC: few drum hits line up - is this the same song and arrangement?")
    shutil.copy2(tmp, path.parent / NAME)
    backup = path.with_name(path.name + ".before_mv")
    if not backup.exists():
        shutil.copy2(path, backup)
    attach(path, NAME, measure, slot, slots)
    print(f"MV added to {path}: fitted to the music (live take); it starts with the BGM at measure {measure:03d}")


if __name__ == "__main__":
    main()
