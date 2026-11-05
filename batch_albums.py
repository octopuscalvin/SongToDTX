"""Chart whole album folders : <album folder>\ -> <dest>\<album>\<song>\ with a
box.def per level. Jacket: the album folder's picture (cover/folder/front first, else any picture,
else Images\ or Scans\), else the art embedded in a song file, else none. Every song is decoded
by ffmpeg first (malformed mp3 headers can fool libsndfile). Only top-level audio is
used (some albums carry an mp3\ copy of the flacs). Title/artist from tags, else the file
name. A folder of song folders named "Title - Artist" (no audio at its top) is charted as one box,
title and artist from each folder's name, jacket from a picture in it or the song's own art.
Zip / 7z / rar archives are unpacked first. Finished albums inside an editing\ folder (or their
archive) move to editing\completed\. summary.tsv has the seconds each song took."""
import json, re, shutil, subprocess, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
import archive  # noqa: E402
ED = config.EDITING_DIR
GENRE = "auto"                     # --genre for every song in the batch (autochart.py --genre)
if "--genre" in sys.argv:
    k = sys.argv.index("--genre")
    GENRE = sys.argv[k + 1]
    del sys.argv[k:k + 2]
dest = Path(sys.argv[1])
AUDIO = (".mp3", ".flac", ".aif", ".aiff", ".wav", ".m4a", ".wma", ".ogg")
PICS = (".jpg", ".jpeg", ".png")
safe = lambda s: re.sub(r'[\\/:*?"<>|]', "_", s).strip(" .") or "untitled"
work = config.CACHE / "albums"
work.mkdir(parents=True, exist_ok=True)
summary = work / "summary.tsv"
if not summary.exists():
    summary.write_text("album\tsong\tseconds\tBPM\tbars\tp95 ms\tchecks\tstatus\n", encoding="utf-8-sig")
dest.mkdir(parents=True, exist_ok=True)
(dest / "box.def").write_text(f"#TITLE: {dest.name}\n", encoding="cp932", errors="replace")


def tags(f: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags:stream=codec_type,codec_name",
                        "-of", "json", str(f)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    j = json.loads(r.stdout or "{}")
    t = {k.lower(): v for k, v in j.get("format", {}).get("tags", {}).items()}
    t["_art"] = any(s.get("codec_type") == "video" for s in j.get("streams", []))
    return t


def art(f: Path, out: Path) -> Path | None:
    """Cover art inside a song file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(f), "-an", "-frames:v", "1", str(out)])
    return out if out.exists() and out.stat().st_size > 1000 else None


def song_folders(album: Path) -> tuple[list[Path], dict]:
    """One song per sub-folder; "Title - Artist" folder names give title and artist."""
    songs, named = [], {}
    for d in sorted(p for p in album.iterdir() if p.is_dir()):
        a = sorted(p for p in d.iterdir() if p.suffix.lower() in AUDIO)
        if a:
            songs.append(a[0])
            title, _, artist = d.name.rpartition(" - ")
            named[a[0]] = (title, artist) if title else (d.name, None)
    return songs, named


def jacket(album: Path, songs: list[Path], info: dict) -> Path | None:
    pics = [p for d in (album, album / "Images", album / "Scans") if d.is_dir()
            for p in sorted(d.iterdir()) if p.suffix.lower() in PICS]
    pref = [p for p in pics if re.search(r"cover|folder|front", p.stem, re.I)]
    if pref or pics:
        return (pref or pics)[0]
    for f in songs:
        if info[f]["_art"] and (out := art(f, work / safe(album.name) / "embedded_cover.jpg")):
            return out
    return None


t_all = time.time()
albums = [Path(a) for a in sys.argv[2:]] or sorted(p for p in ED.iterdir() if p.name != "completed" and
                                                   (p.is_dir() or archive.is_archive(p)))
for src in albums:
    album = archive.unpack(src, config.CACHE / "unpacked") if archive.is_archive(src) else src
    songs, named = sorted(p for p in album.iterdir() if p.suffix.lower() in AUDIO), {}
    if not songs:
        songs, named = song_folders(album)
    if not songs:
        continue
    info = {f: tags(f) for f in songs}
    cover = None if named else jacket(album, songs, info)
    box = dest / safe(album.name)
    box.mkdir(parents=True, exist_ok=True)
    (box / "box.def").write_text(f"#TITLE: {album.name}\n", encoding="cp932", errors="replace")
    print(f"== {album.name}: {len(songs)} songs, jacket: {cover.name if cover else 'none'}", flush=True)
    t_album, failed, seen = time.time(), 0, set()
    for f in songs:
        t = info[f]
        title = (named.get(f) or [None])[0] or t.get("title") or re.sub(r"^\s*\d+\s*[-.]?\s*", "", f.stem).strip() or f.stem
        if title in seen:   # disc 2 can repeat a title
            title += f" (Disc {t.get('disc', '2').split('/')[0]})"
        seen.add(title)
        name = safe(title)
        if (box / name).exists():
            continue
        t0 = time.time()
        wav = config.CACHE / "input" / safe(album.name) / f"{f.stem}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        if not wav.exists():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(f), "-vn", str(wav)])
        out = work / safe(album.name) / name
        artist = (named.get(f) or [None, None])[1] or t.get("artist") or t.get("album_artist") or "Unknown"
        cmd = [sys.executable, str(HERE / "autochart.py"), str(wav), str(out), "--title", title,
               "--artist", artist, "--dlevel", "680"]
        if GENRE != "auto":
            cmd += ["--genre", GENRE]
        pic = cover
        if f in named:      # a song folder's own picture, else the song's own art
            pic = next((p for p in sorted(f.parent.iterdir()) if p.suffix.lower() in PICS), None) or                 (info[f]["_art"] and art(f, work / safe(album.name) / f"{name}_cover.jpg")) or None
        if pic:
            cmd += ["--preimage", str(pic)]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=HERE)
        log = r.stdout + r.stderr
        out.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{out}.log").write_text(log, encoding="utf-8")
        ok = (out / "mstr.dtx").exists()
        if ok:
            shutil.copytree(out, box / name, dirs_exist_ok=True)
        elif "NO DRUMS" not in log:   # a drumless song is skipped, not a failure
            failed += 1
        row = [album.name, title, f"{time.time() - t0:.0f}", " / ".join(re.findall(r"BPM ([\d.]+)", log)[:8]),
               (re.findall(r"(\d+) bars;", log) or ["?"])[-1], (re.findall(r"p95 ([\d.]+) ms", log) or ["?"])[-1],
               str(log.count("\n        - ")), "ok" if ok else "skipped, no drums" if "NO DRUMS" in log else "FAILED: " + (log.strip().splitlines() or ["?"])[-1][:150]]
        with summary.open("a", encoding="utf-8") as fh:
            fh.write("\t".join(row) + "\n")
        print("   " + " | ".join(row[1:]), flush=True)
    print(f"   album time {time.time() - t_album:.0f} s", flush=True)
    if not failed and ED != Path() and src.parent == ED:
        shutil.move(str(src), str(ED / "completed" / src.name))
print(f"ALL DONE in {time.time() - t_all:.0f} s", flush=True)
