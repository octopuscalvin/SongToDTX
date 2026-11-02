"""Song to DTX - local web app. Python standard library only (plus the engine next to it): serves
app/index.html and runs the existing tools (autochart.py, video_score.py, batch_albums.py) as
child processes, one job at a time (they share the GPU).

  .venv/Scripts/python app/server.py [--port 8680]      then open http://127.0.0.1:8680

Flow: 1 source (song file / YouTube / score video / batch folder) -> 2 generate -> 3 check the
flagged measures, rebuild with hints -> 4 song info -> 5 save into DTXFiles/AutoChart/<title> (auto vN)."""
import argparse, json, re, shutil, subprocess, sys, threading, time, uuid, webbrowser
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

APP = Path(__file__).resolve().parent
HERE = APP.parent                                  # the Song to DTX folder
sys.path.insert(0, str(HERE))
import config                                      # noqa: E402
import dtx                                         # noqa: E402
from drumkit import KIT                            # noqa: E402

VERSION = (HERE / "VERSION").read_text().strip()

PY = sys.executable
WORK = config.CACHE / "app"
DTXMANIA = config.DTXMANIA
DEST = DTXMANIA / "DTXFiles" / "AutoChart"
AUDIO = (".mp3", ".flac", ".wav", ".m4a", ".ogg", ".aif", ".aiff", ".wma")
safe = lambda s: re.sub(r'[\\/:*?"<>|]', "_", s).strip(" .") or "untitled"
JOBS: dict[str, dict] = {}
CRLF = chr(13) + chr(10)
GPU = threading.Lock()


# ---- jobs ---------------------------------------------------------------------------------------
def run(job: dict, cmd: list, cwd: Path = HERE) -> int:
    job["log"].append("> " + " ".join(str(c) for c in cmd[1:3]) + " ...")
    p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace",
                         env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    job["proc"] = p
    for line in p.stdout:
        line = line.rstrip()
        job["log"].append(line)
        m = re.match(r"\[(\d)/5\]\s*(.*)", line)
        if m:
            job["step"], job["stage"] = int(m[1]), m[2].rstrip(" .")
    return p.wait()


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], check=True)


def chart_cmd(job: dict, wav: Path) -> list:
    o = job["opts"]
    cmd = [PY, HERE / "autochart.py", wav, job["out"], "--title", o.get("title") or wav.stem,
           "--artist", o.get("artist") or "Unknown", "--dlevel", str(o.get("level") or 680)]
    if o.get("hints"):
        cmd += ["--bars", o["hints"]]
    if o.get("jacket"):
        cmd += ["--preimage", o["jacket"]]
    if o.get("video"):
        cmd += ["--video", o["video"]]
    if o.get("score"):
        cmd += ["--score", o["score"], "--score-first", str(o.get("score_first") or 2)]
    return cmd


def yt(url: str, out: Path, fmt: str, *extra) -> Path:
    if not out.exists():
        subprocess.run([PY, "-m", "yt_dlp", "-q", "--no-playlist", "-f", fmt, "-o", str(out), *extra, url], check=True)
    return out


def job_thread(job: dict):
    o, w = job["opts"], job["work"]
    try:
        with GPU:
            job["state"] = "running"
            mode = o["mode"]
            if mode == "batch":
                job["stage"] = "charting album folders"
                code = run(job, [PY, HERE / "batch_albums.py", o.get("dest") or DTXMANIA / "DTXFiles" / "AutoChart (batch)",
                                 *[f for f in o["folders"].splitlines() if f.strip()]])
                job["state"] = "done" if code == 0 else "failed"
                return
            if mode == "file":
                src = Path(o["path"])
                wav = w / (src.stem + ".wav")
                job["stage"] = "decoding the song"
                if not wav.exists():
                    ffmpeg("-i", src, "-vn", wav)        # malformed mp3 headers can fool libsndfile
                r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", str(src)],
                                   capture_output=True, text=True, encoding="utf-8", errors="replace")
                tags = {k.lower(): v for k, v in json.loads(r.stdout or "{}").get("format", {}).get("tags", {}).items()}
                o["title"] = o.get("title") or tags.get("title") or src.stem
                o["artist"] = o.get("artist") or tags.get("artist") or tags.get("album_artist") or "Unknown"
                if not o.get("jacket"):                   # cover art inside the song file, if any
                    pic = w / "jacket.jpg"
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-an", "-frames:v", "1", str(pic)])
                    if pic.exists() and pic.stat().st_size > 1000:
                        o["jacket"] = str(pic)
            elif mode == "youtube":
                job["stage"] = "downloading from YouTube"
                meta = subprocess.run([PY, "-m", "yt_dlp", "--no-playlist", "--print", "%(title)s\t%(uploader)s", o["url"]],
                                      capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.strip()
                t, _, up = meta.partition("\t")
                o.setdefault("title", t)
                o["title"] = o["title"] or t
                o["artist"] = o.get("artist") or up
                if o.get("bga"):
                    src = yt(o["url"], w / "source.mp4", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b")
                    o["video"] = str(src)
                else:
                    src = yt(o["url"], w / "source.m4a", "ba[ext=m4a]/ba")
                if not o.get("jacket"):
                    subprocess.run([PY, "-m", "yt_dlp", "-q", "--skip-download", "--write-thumbnail", "--convert-thumbnails",
                                    "jpg", "-o", str(w / "jacket"), o["url"]])
                    if (w / "jacket.jpg").exists():
                        o["jacket"] = str(w / "jacket.jpg")
                wav = w / "source.wav"
                if not wav.exists():
                    ffmpeg("-i", src, "-vn", wav)
            elif mode == "score":
                job["stage"] = "reading the score video"
                if run(job, [PY, HERE / "video_score.py", o["chart_url"], o["url"], w]) != 0:
                    raise RuntimeError("video_score.py failed")
                o["video"] = str(next(w.glob("source.*")))
                wav = w / "source.wav"
                score = w / "score.txt"
                if not score.exists():
                    job["state"] = "needs_score"
                    job["stage"] = f"score pages are in {w / 'pages'}; score.txt has to be written from them (ask Claude)"
                    return
                o["score"] = str(score)
                if not o.get("jacket"):
                    pic = w / "jacket.jpg"
                    ffmpeg("-ss", "60", "-i", o["video"], "-frames:v", "1", "-vf", "scale=640:-2", pic)
                    o["jacket"] = str(pic)
            job["wav"] = str(wav)
            code = run(job, chart_cmd(job, wav))
            if code != 0 or not (job["out"] / "mstr.dtx").exists():
                raise RuntimeError("NO DRUMS: this song has almost no drums" if any("NO DRUMS" in l for l in job["log"])
                                   else (job["log"] or ["autochart failed"])[-1])
            job["info"] = chart_info(job)
            job["state"], job["step"], job["stage"] = "done", 5, "chart ready"
            (job["out"] / "job.json").write_text(json.dumps({"opts": o, "log": job["log"]}, default=str), encoding="utf-8")
    except Exception as e:                                                   # shown in the page
        job["state"], job["error"] = "failed", str(e)
        job["log"].append(f"ERROR: {e}")


def start_job(opts: dict, rebuild_of: str | None = None) -> dict:
    jid = uuid.uuid4().hex[:8]
    if rebuild_of:                                   # same source and work dir, new hints
        old = JOBS[rebuild_of]
        opts = {**old["opts"], **opts}
        work = old["work"]
    else:
        key = opts.get("path") or opts.get("url") or opts.get("folders") or jid
        work = WORK / (time.strftime("%Y%m%d") + "_" + safe(Path(key).stem if opts.get("path") else
                                                              re.sub(r"\W+", "_", key)[-40:]))
    work.mkdir(parents=True, exist_ok=True)
    job = {"id": jid, "opts": opts, "work": work, "out": work / f"chart_{jid}", "state": "queued", "step": 0,
           "stage": "waiting for the GPU", "log": [], "info": None, "error": None}
    JOBS[jid] = job
    threading.Thread(target=job_thread, args=(job,), daemon=True).start()
    return job


def load_jobs():
    """Finished charts from earlier runs (cache/app/<work>/chart_<id>), oldest first, so the page
    can reopen the last one after a restart."""
    for out in sorted(WORK.glob("*/chart_*/mstr.dtx"), key=lambda p: p.stat().st_mtime):
        out = out.parent
        jid = out.name[6:]
        meta = json.loads((out / "job.json").read_text(encoding="utf-8")) if (out / "job.json").exists() else {}
        opts = meta.get("opts") or {"mode": "file", "path": str(next(out.parent.glob("*.wav"), ""))}
        jacket = out.parent / "jacket.jpg"
        if not opts.get("jacket") and jacket.exists():
            opts["jacket"] = str(jacket)
        job = {"id": jid, "opts": opts, "work": out.parent, "out": out, "state": "done", "step": 5,
               "stage": "chart ready", "log": meta.get("log", []), "info": None, "error": None}
        try:
            job["info"] = chart_info(job)
        except Exception as e:                                               # a broken old chart
            job["state"], job["error"] = "failed", str(e)
        JOBS[jid] = job


# ---- chart editor -------------------------------------------------------------------------------
KIT_FILES = {ch: f for ch, (_, f) in KIT.items()}


def chart_json(job: dict) -> dict:
    """Everything the page's editor and player need: measures, tempo events, BGM start, drum chips
    (measure, pos as n/d, lane, wav id) and the sound files with their #VOLUME."""
    d = job["out"]
    ch = dtx.read_dtx(d / "mstr.dtx")
    wavs = {k[3:]: v for k, v in ch.headers.items() if k.startswith("WAV") and len(k) == 5}
    vols = {k[6:]: int(v) for k, v in ch.headers.items() if k.startswith("VOLUME") and len(k) == 8 and v.isdigit()}
    drums = [c for c in ch.chips if c.channel in dtx.DRUM_CHANNELS]
    used = {c.value for c in drums}
    lane_wav: dict[str, str] = {}
    for c in drums:                                    # first wav a lane uses = its sound for new notes
        lane_wav.setdefault(c.channel, c.value)
    for lane, f in KIT_FILES.items():
        lane_wav.setdefault(lane, next((k for k, v in wavs.items() if v == f), lane_wav.get("11", "02")))
    bgm = min((c for c in ch.chips if c.channel == "01"), key=lambda c: (c.measure, c.pos), default=None)
    tempo = sorted([(c.measure, c.pos, float(int(c.value, 16)) if c.channel == "03" else ch.bpm_defs.get(c.value))
                    for c in ch.chips if c.channel in ("03", "08")], key=lambda e: (e[0], e[1]))
    last = max((c.measure for c in ch.chips), default=1)
    url = lambda f: "/api/file?path=" + __import__("urllib.parse").parse.quote(str(d / f))
    return {"base_bpm": ch.base_bpm, "last": last + 1,
            "measure_len": {m: L for m, L in ch.measure_len.items()},
            "tempo": [[m, p.numerator, p.denominator, b] for m, p, b in tempo if b],
            "bgm": [bgm.measure, bgm.pos.numerator, bgm.pos.denominator] if bgm else None,
            "bgm_url": url(wavs[bgm.value]) if bgm and bgm.value in wavs else None,
            "chips": [[c.measure, c.pos.numerator, c.pos.denominator, c.channel, c.value] for c in drums],
            "wavs": {k: {"url": url(v), "vol": vols.get(k, 100)} for k, v in wavs.items() if k in used | set(lane_wav.values())},
            "lane_wav": lane_wav, "edited": (d / "mstr.auto.dtx").exists()}


def save_chart(job: dict, chips: list) -> int:
    """Rewrite the drum chips of the job's chart from the editor (the tool's own version is kept
    once as mstr.auto.dtx). Returns the number of notes written."""
    p = job["out"] / "mstr.dtx"
    if not (job["out"] / "mstr.auto.dtx").exists():
        shutil.copy2(p, job["out"] / "mstr.auto.dtx")
    lines = [l for l in p.read_bytes().decode("shift_jis", errors="replace").splitlines()
             if not ((m := dtx.LINE_RE.match(l)) and m[2].upper() in dtx.DRUM_CHANNELS)]
    groups: dict[tuple[int, str], dict] = {}
    for m, n, dd, lane, wav in chips:
        pos = Fraction(int(n), int(dd))
        if lane in dtx.DRUM_CHANNELS and 0 <= pos < 1:
            groups.setdefault((int(m), lane), {})[pos] = wav
    while lines and not lines[-1].strip():
        lines.pop()
    for (m, lane), items in sorted(groups.items()):
        lines.append(f"#{m:03d}{lane}: {dtx._encode_measure(sorted(items.items()))}")
    p.write_bytes((CRLF.join(lines) + CRLF).encode("shift_jis", errors="replace"))
    job["info"] = chart_info(job)
    return sum(len(v) for v in groups.values())


# ---- chart facts --------------------------------------------------------------------------------
def bpm_spans(ch: dtx.Chart) -> dict[float, float]:
    """Seconds spent at each BPM (measure 001 on)."""
    ev = sorted(((c.measure, c.pos, ch.bpm_defs[c.value]) for c in ch.chips
                 if c.channel == "08" and c.value in ch.bpm_defs), key=lambda e: (e[0], e[1]))
    last = max((c.measure for c in ch.chips if c.channel in dtx.DRUM_CHANNELS), default=1)
    spans: dict[float, float] = {}
    bpm, cur = ch.base_bpm, (0, Fraction(0))
    for m, p, b in [e for e in ev if e[0] >= 1] + [(last + 1, Fraction(0), None)]:
        if (m, p) <= (1, Fraction(0)):
            bpm = b if b else bpm
            continue
        a = max(cur, (1, Fraction(0)))
        beats = sum(4 * ch.measure_len.get(k, 1.0) for k in range(a[0], m)) \
            - 4 * ch.measure_len.get(a[0], 1.0) * float(a[1]) + 4 * ch.measure_len.get(m, 1.0) * float(p)
        spans[round(bpm, 2)] = spans.get(round(bpm, 2), 0) + beats * 60 / bpm
        if b:
            bpm, cur = b, (m, p)
    return spans


def chart_info(job: dict) -> dict:
    d = job["out"]
    ch = dtx.read_dtx(d / "mstr.dtx")
    spans = bpm_spans(ch)
    main = max(spans, key=spans.get) if spans else ch.base_bpm
    meta = json.loads((d / "autochart.json").read_text()) if (d / "autochart.json").exists() else {}
    rules, on = [], False
    for l in job["log"]:                                   # what the hand rules changed (step 4 lines)
        if l.startswith("[4/5]"):
            on = True
        elif l.startswith("[5/5]"):
            on = False
        elif on and re.search(r"\b(bars?|measures?)\b.*\d{3}", l):
            rules.append(l.strip())
    o = job["opts"]
    return {"title": ch.headers.get("TITLE", ""), "artist": ch.headers.get("ARTIST", ""),
            "bpm_header": ch.base_bpm, "bpm_main": main, "bpm_min": min(spans, default=main),
            "bpm_max": max(spans, default=main), "bars": meta.get("bars"),
            "notes": sum(c.channel in dtx.DRUM_CHANNELS for c in ch.chips), "checks": meta.get("checks", []),
            "rules": rules, "jacket": o.get("jacket"), "bgm_seconds": audio_seconds(d / "bgm.wav")}


def audio_seconds(f: Path) -> float:
    if not f.exists():
        return 0.0
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


# ---- save ---------------------------------------------------------------------------------------
def next_version(title: str) -> int:
    pat = re.compile(re.escape(safe(title)) + r" \((?:auto|score) v(\d+)\)$")
    found = [int(m[1]) for d in (DEST, DEST / "Old versions") if d.exists() for p in d.iterdir() if (m := pat.match(p.name))]
    return max(found, default=0) + 1


def save(job: dict, f: dict) -> dict:
    """Copy the chart into DTXFiles/AutoChart/<title> (auto vN), rewrite the song info headers,
    cut the preview sound, move older versions to Old versions (skipped while DTXMania has them open)."""
    title = f.get("title") or job["info"]["title"]
    kind = "score" if job["opts"]["mode"] == "score" else "auto"
    v = next_version(title)
    dst = DEST / f"{safe(title)} ({kind} v{v})"
    shutil.copytree(job["out"], dst, ignore=shutil.ignore_patterns("mstr.auto.dtx", "job.json"))
    heads = {"TITLE": title, "ARTIST": f.get("artist", ""), "COMMENT": f.get("comment", ""),
             "DLEVEL": str(int(f.get("level") or 680))}
    jacket = f.get("jacket") or job["info"].get("jacket")
    if jacket and Path(jacket).exists():
        name = "jacket" + Path(jacket).suffix.lower()
        if Path(jacket).resolve() != (dst / name).resolve():
            shutil.copy2(jacket, dst / name)
        heads["PREIMAGE"] = name
    if f.get("preview_start") not in (None, ""):
        s = float(f["preview_start"])
        ffmpeg("-ss", f"{s:.2f}", "-t", "15", "-i", dst / "bgm.wav", "-af",
               "afade=t=in:d=0.5,afade=t=out:st=13.5:d=1.5", "-c:a", "libvorbis", "-q:a", "5", dst / "preview.ogg")
        heads["PREVIEW"] = "preview.ogg"
    set_headers(dst / "mstr.dtx", heads, float(f["bpm_display"]) if f.get("bpm_display") else None)
    moved = []
    (DEST / "Old versions").mkdir(exist_ok=True)
    for d in DEST.iterdir():
        if d != dst and d.is_dir() and re.fullmatch(re.escape(safe(title)) + r" \((auto|score) v\d+\)", d.name):
            try:
                shutil.move(str(d), str(DEST / "Old versions" / d.name))
                moved.append(d.name)
            except OSError:
                pass
    return {"saved": str(dst), "moved": moved}


def set_headers(path: Path, heads: dict, bpm_display: float | None):
    """Header lines are replaced or added. A shown BPM other than the real starting tempo gets a
    BPM chip at the very start (measure 000) so the timing does not change."""
    lines = path.read_text(encoding="shift_jis", errors="replace").splitlines()
    base = next(float(l.split(":", 1)[1]) for l in lines if re.match(r"#BPM\s*:", l))
    if bpm_display and abs(bpm_display - base) > 0.005 and not any(l.startswith("#00008") for l in lines):
        heads["BPM"] = f"{bpm_display:.2f}"
        used = {l[4:6] for l in lines if re.match(r"#BPM[0-9A-Z]{2}\s*:", l)}
        zid = next(dtx.to_b36(n) for n in range(36 * 36 - 1, 0, -1) if dtx.to_b36(n) not in used)
        lines.insert(next(i for i, l in enumerate(lines) if re.match(r"#BPM\s*:", l)) + 1, f"#BPM{zid}: {base:.2f}")
        lines.append(f"#00008: {zid}")
    for k, v in heads.items():
        if k == "COMMENT" and not v:
            continue
        val = dtx.sjis_safe(v) if hasattr(dtx, "sjis_safe") else v
        i = next((i for i, l in enumerate(lines) if re.match(rf"#{k}\s*:", l)), None)
        if i is None:
            i = next(i for i, l in enumerate(lines) if l.startswith("#DLEVEL") or l.startswith("#BPM")) + 1
            lines.insert(i, f"#{k}: {val}")
        else:
            lines[i] = f"#{k}: {val}"
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("shift_jis", errors="replace"))


# ---- http ---------------------------------------------------------------------------------------
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json", code=200):
        if not isinstance(body, bytes):
            body = json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self.send((APP / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path.startswith("/api/job/"):
            j = JOBS.get(u.path.rsplit("/", 1)[1])
            if not j:
                return self.send({"error": "no such job"}, code=404)
            return self.send({k: j[k] for k in ("id", "state", "step", "stage", "info", "error")} |
                             {"log": j["log"][-400:], "title": j["opts"].get("title"), "out": str(j["out"]),
                              "mode": j["opts"].get("mode"), "work": str(j["work"])})
        if u.path.startswith("/api/chart/"):
            j = JOBS.get(u.path.rsplit("/", 1)[1])
            if not j or not (j["out"] / "mstr.dtx").exists():
                return self.send({"error": "no chart"}, code=404)
            return self.send(chart_json(j))
        if u.path == "/api/file":                     # jacket / sounds; only files under the app's own folders
            p = Path(unquote(parse_qs(u.query).get("path", [""])[0])).resolve()
            if any(r.resolve() in p.parents for r in (HERE, config.CACHE, DTXMANIA)) and p.is_file():
                ct = {".png": "image/png", ".ogg": "audio/ogg", ".wav": "audio/wav"}.get(p.suffix.lower(), "image/jpeg")
                return self.send(p.read_bytes(), ct)
            return self.send({"error": "not found"}, code=404)
        if u.path == "/api/defaults":                 # last job too, so a reloaded page picks it up again
            return self.send({"dest_batch": str(DTXMANIA / "DTXFiles" / "AutoChart (batch)"), "level": 680,
                              "version": VERSION, "dtxmania": str(DTXMANIA),
                              "dtxmania_ok": (DTXMANIA / "DTXFiles").is_dir(),
                              "last_job": list(JOBS)[-1] if JOBS else None})
        self.send({"error": "not found"}, code=404)

    def do_POST(self):
        global DTXMANIA, DEST
        u = urlparse(self.path)
        try:
            if u.path == "/api/upload":               # raw file body; name in X-Name
                name = safe(unquote(self.headers.get("X-Name", "upload")))
                d = WORK / "uploads" / time.strftime("%Y%m%d_%H%M%S")
                d.mkdir(parents=True, exist_ok=True)
                (d / name).write_bytes(self.body())
                return self.send({"path": str(d / name)})
            data = json.loads(self.body() or b"{}")
            if u.path == "/api/start":
                return self.send({"id": start_job(data)["id"]})
            if u.path == "/api/rebuild":
                return self.send({"id": start_job(data.get("opts", {}), data["id"])["id"]})
            if u.path.startswith("/api/chart/"):
                n = save_chart(JOBS[u.path.rsplit("/", 1)[1]], data["chips"])
                return self.send({"notes": n})
            if u.path == "/api/revert":            # back to the tool's own chart
                j = JOBS[data["id"]]
                if (j["out"] / "mstr.auto.dtx").exists():
                    shutil.copy2(j["out"] / "mstr.auto.dtx", j["out"] / "mstr.dtx")
                    (j["out"] / "mstr.auto.dtx").unlink()
                    j["info"] = chart_info(j)
                return self.send({"ok": True})
            if u.path == "/api/save":
                return self.send(save(JOBS[data["id"]], data))
            if u.path == "/api/open":                 # DTXCreator with the saved chart
                p = Path(data["path"]) / "mstr.dtx"
                subprocess.Popen([str(DTXMANIA / "DTXCreator.exe"), str(p)], cwd=DTXMANIA)
                return self.send({"ok": True})
            if u.path == "/api/settings":           # DTXMania folder chosen in the page
                d = Path(data["dtxmania"])
                if not (d / "DTXFiles").is_dir():
                    return self.send({"error": f"No DTXFiles folder in {d}"}, code=400)
                f = HERE / "settings.json"
                cur = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
                cur["dtxmania_dir"] = str(d)
                f.write_text(json.dumps(cur, indent=2), encoding="utf-8")
                DTXMANIA, DEST = d, d / "DTXFiles" / "AutoChart"
                return self.send({"ok": True})
            if u.path == "/api/folder":
                subprocess.Popen(["explorer", str(Path(data["path"]))])
                return self.send({"ok": True})
            if u.path == "/api/cancel":
                p = JOBS[data["id"]].get("proc")
                if p and p.poll() is None:
                    p.kill()
                return self.send({"ok": True})
        except Exception as e:
            return self.send({"error": str(e)}, code=500)
        self.send({"error": "not found"}, code=404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8680)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    load_jobs()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    url = f"http://127.0.0.1:{a.port}"
    print(f"Song to DTX {VERSION} running at {url}  (close this window to stop)", flush=True)
    if not a.no_browser:
        webbrowser.open(url)
    srv.serve_forever()


if __name__ == "__main__":
    main()
