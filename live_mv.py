"""Fits a video whose music runs at another, changing tempo (a live take, a slower or faster
performance) to a chart's BGM, moment by moment, for add_mv.py --live:
  * rough map: the two recordings' harmony (chroma) aligned by dynamic time warping,
  * fine map: drum hits on both (Demucs drum stem -> ADTOF), in 8 s windows the shift and speed
    that put the most BGM kick / snare / hi-hat hits within 30 ms of a video hit,
  * stretches without drums (an intro, a drumless part) follow the smoothed harmony map, bent to
    meet the drum map at both ends; the speed stays within 0.85-1.35 of the anchors' so the
    picture never jumps or freezes,
  * every output frame shows the video frame its BGM moment maps to."""
import subprocess
import sys
from pathlib import Path

import numpy as np

WIN = 8.0           # s per matching window ...
STEP = 1.0          # ... one every second
TOL = 0.03          # s: a BGM hit within this of a video hit counts as matched
MIN_SHARE = 0.35    # a window is used when this share of its hits match
SHIFT = 0.8         # s searched around the rough map
SPEED = 0.08        # +- this around the rough map's local speed
SPEED_LIMIT = (0.85, 1.35)   # of the anchors' speed: never faster or slower than this (no jumps)
KICK, SNARE, HAT = 35, 38, 42


def rough_map(bgm: Path, video_audio: Path) -> np.ndarray:
    """(BGM second, video second) pairs from chroma DTW, 93 ms steps."""
    import librosa
    sr, hop = 22050, 2048

    def chroma(p):
        y, _ = librosa.load(str(p), sr=sr, mono=True)
        return librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop) + 1e-4
    _, wp = librosa.sequence.dtw(X=chroma(bgm), Y=chroma(video_audio), metric="cosine")
    return wp[::-1] * hop / sr


def anchors(bgm_hits, video_hits, rough: np.ndarray, end: float) -> np.ndarray:
    """(BGM second, video second, matched share) for windows whose hits line up."""
    ro = lambda t: float(np.interp(t, rough[:, 0], rough[:, 1]))
    pitches = (KICK, SNARE, HAT)
    B = {p: np.array(sorted(t for t, q in bgm_hits if q == p)) for p in pitches}
    V = {p: np.array(sorted(t for t, q in video_hits if q == p)) for p in pitches}
    offs = np.arange(-SHIFT, SHIFT + 1e-9, 0.01)
    out = []
    for c in np.arange(WIN / 2, end - WIN / 2, STEP):
        a, b = c - WIN / 2, c + WIN / 2
        win = {p: B[p][(B[p] >= a) & (B[p] < b)] for p in pitches}
        n = sum(len(x) for x in win.values())
        if n < 8:
            continue
        sp0 = (ro(b) - ro(a)) / WIN
        best = (0.0, 0.0, sp0)
        for sp in np.arange(sp0 - SPEED, sp0 + SPEED + 1e-9, 0.01):
            score = np.zeros(len(offs))
            for p in pitches:
                if not len(win[p]) or len(V[p]) < 2:
                    continue
                m = ro(a) + offs[:, None] + (win[p] - a)[None, :] * sp      # offsets x hits
                j = np.clip(np.searchsorted(V[p], m), 1, len(V[p]) - 1)
                d = np.minimum(abs(V[p][j] - m), abs(V[p][j - 1] - m))
                score += (d < TOL).sum(1) * (0.5 if p == HAT else 1.0)
            k = int(score.argmax())
            if score[k] > best[0]:
                best = (float(score[k]), float(offs[k]), float(sp))
        share = best[0] / n
        out.append((c, ro(a) + best[1] + (c - a) * best[2], share))
    A = np.array(out) if out else np.zeros((0, 3))
    if len(A) < 3:
        return A
    from scipy.ndimage import median_filter
    off = A[:, 1] - A[:, 0]
    keep = (A[:, 2] >= MIN_SHARE) & (np.abs(off - median_filter(off, size=5, mode="nearest")) < 0.03)
    return A[keep]


def time_map(A: np.ndarray, rough: np.ndarray, end: float) -> tuple[np.ndarray, np.ndarray, list]:
    """Video second for every 10 ms of the BGM, and the BGM stretches (start, end) that follow
    the music only (no drum anchors for over 10 s)."""
    from scipy.ndimage import gaussian_filter1d
    t = np.arange(0, end, 0.01)
    tc = np.arange(0, max(end, rough[-1, 0]) + 1, 0.01)
    cs = gaussian_filter1d(np.interp(tc, rough[:, 0], rough[:, 1]), 300)    # harmony map, 3 s smooth
    harm = lambda x: np.interp(x, tc, cs)
    if len(A) < 3:
        return harm(t), t, [(0.0, end)]
    at, al = A[:, 0], A[:, 1]
    f = np.interp(t, at, al)
    gaps = [(0.0, at[0])] + [(a, b) for a, b in zip(at[:-1], at[1:]) if b - a > 3.0] + [(at[-1], end)]
    loose = []
    for a, b in gaps:
        m = (t >= a) & (t <= b)
        if b >= end - 1e-6:          # after the last drums: the last 20 s of anchors' speed
            k = at > at[-1] - 20
            sp = np.polyfit(at[k], al[k], 1)[0] if k.sum() >= 2 else 1.0
            f[m] = al[-1] + (t[m] - at[-1]) * sp
        elif a > 0 and b - a <= 10:  # short gap: steady speed between the anchors
            f[m] = np.interp(t[m], [a, b], [np.interp(a, at, al), np.interp(b, at, al)])
        else:                        # intro / drumless part: the harmony map, bent to meet the anchors
            da = np.interp(a, at, al) - harm(a) if a > 0 else al[0] - harm(at[0])
            db = np.interp(b, at, al) - harm(b)
            w = (t[m] - a) / max(b - a, 1e-6)
            f[m] = harm(t[m]) + da * (1 - w) + db * w
        if b - a > 10:
            loose.append((a, b))
    f = gaussian_filter1d(f, 30)
    base = float(np.median(np.diff(al) / np.diff(at)))
    sl = np.clip(np.gradient(f, 0.01), SPEED_LIMIT[0] * base, SPEED_LIMIT[1] * base)
    f2 = f[0] + np.concatenate([[0], np.cumsum((sl[1:] + sl[:-1]) / 2 * 0.01)])
    return f2 + gaussian_filter1d(f - f2, 200), t, loose


def fit_quality(t, f, bgm_hits, video_hits) -> tuple[float, float]:
    """Median distance (ms) from mapped BGM kicks / snares to the nearest video hit, share within 40 ms."""
    e = []
    for p in (KICK, SNARE):
        v = np.array(sorted(x for x, q in video_hits if q == p))
        b = np.array(sorted(x for x, q in bgm_hits if q == p))
        if len(v) < 2 or not len(b):
            continue
        m = np.interp(b, t, f)
        j = np.clip(np.searchsorted(v, m), 1, len(v) - 1)
        e += list(np.minimum(abs(v[j] - m), abs(v[j - 1] - m)) * 1000)
    e = np.array(e) if e else np.array([999.0])
    return float(np.median(e)), float(np.mean(e < 40))


def render(video: Path, t: np.ndarray, f: np.ndarray, out: Path, frame_filter: str) -> None:
    """Every output frame (the video's own frame rate) shows the video frame at f(its BGM time);
    before the video starts: black."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=r_frame_rate", "-of", "csv=p=0", str(video)],
                           capture_output=True, text=True).stdout.strip()
    num, _, den = probe.partition("/")
    fps = float(num) / float(den or 1) if num else 30.0
    fps = min(fps, 30.0)
    W, H = 1280, 720
    fb = W * H * 3
    n = int(t[-1] * fps)
    want = np.round(np.interp(np.arange(n) / fps, t, f) * fps).astype(int)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-an", "-vf", f"fps={fps:.5f},{frame_filter}",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", f"{fps:.5f}", "-i", "-", "-c:v", "wmv2", "-b:v", "3500k", str(out)],
                           stdin=subprocess.PIPE)
    black, cur, frame = bytes(fb), -1, None
    for k in want:
        if k < 0:
            enc.stdin.write(black)
            continue
        while cur < k:
            b = dec.stdout.read(fb)
            if len(b) < fb:
                break
            frame, cur = b, cur + 1
        enc.stdin.write(frame or black)
    enc.stdin.close()
    enc.wait()
    dec.kill()


def fit(bgm: Path, video: Path, video_audio: Path, out: Path, frame_filter: str):
    """Writes the fitted video to `out`; returns (median ms, share within 40 ms, loose stretches,
    the speed of the video against the BGM)."""
    import soundfile as sf
    sys.path.insert(0, str(Path(__file__).parent))
    import autochart
    print("      matching drum hits (separating both recordings) ...", flush=True)
    rough = rough_map(bgm, video_audio)
    hb = autochart.transcribe(autochart.separate(bgm)[0])
    hv = autochart.transcribe(autochart.separate(video_audio)[0])
    end = sf.info(str(bgm)).duration
    A = anchors(hb, hv, rough, end)
    f, t, loose = time_map(A, rough, end)
    med, share = fit_quality(t, f, hb, hv)
    speed = float(np.median(np.diff(A[:, 1]) / np.diff(A[:, 0]))) if len(A) > 2 else 1.0
    print("[3/3] cutting and converting the video ...", flush=True)
    render(video, t, f, out, frame_filter)
    return med, share, loose, speed
