"""Song name and artist of a YouTube video, read from its title, description, channel and tags.

A video title is rarely just the song name: 【MV】Artist「Song」, 『Song』/ Artist, Artist - Song
(Official Video), 【カバーライブ映像】「Song（Original artist）」Artist / date ... This module takes the
video's own facts (yt-dlp -J) and returns the best guess with how sure it is and what to look at:

  guess(meta) -> {"title", "artist", "kind", "original_artist", "confidence", "warnings", "mv_url", "why"}

kind: mv, live, cover, documentary, lyric, audio, drum cover or video. A cover keeps the performer in
the video as the artist; the original artist goes into "original_artist" (and the chart comment).

  python songinfo.py URL [URL ...]       prints the guesses (reads the facts only, no download)"""
import json, re, subprocess, sys

QUOTES = [("「", "」"), ("『", "』"), ("“", "”"), ('"', '"'), ("«", "»")]
TAG = re.compile(r"[【\[［]([^】\]］]{1,40})[】\]］]")
KIND_WORDS = re.compile(r"(?i)初披露|shooting\s*live|(カバー)?ライブ映像|live\s*(ver\.?|version|映像)?|ライブ|cover|カバー|"
                        r"documentary(\s*film)?|ドキュメンタリー|lyric\s*video|歌ってみた|叩いてみた")
VERSION = re.compile(r"(?i)\b(ver\.?|version|remix|edit|mix)(\b|$)|バージョン")
KINDS = [  # (kind, words in a tag or the title)
    ("drum cover", r"叩いてみた|drum\s*cover|drum\s*play"),
    ("documentary", r"documentary|ドキュメンタリー|密着|behind the scenes|making"),
    ("cover", r"カバー|cover|歌ってみた|弾いてみた"),
    ("live", r"ライブ|live|初披露|ステージ|stage|concert|fes\b|フェス"),
    ("lyric", r"lyric|リリック|歌詞"),
    ("audio", r"official audio|audio|topic"),
    ("mv", r"\bmv\b|m/v|music video|ミュージックビデオ|official video|pv\b"),
]
NOISE = re.compile(r"(?i)[(（][^)）]*\b(official|video|mv|audio|lyrics?|live|cover|remaster(ed)?)\b[^)）]*[)）]|\b(official|music video|video|mv|m/v|pv|lyrics?|audio|full ver\.?|ver\.|hd|4k|"
                   r"高画質|超高画質)\b|【[^】]*】|\[[^\]]*\]")
DATE = re.compile(r"\d{4}[./年-]\d{1,2}[./月-]\d{1,2}日?|\(\d{4}/\d{1,2}/\d{1,2}\)|\d{4}\.\d{2}\.\d{2}")
CHANNEL_NOISE = re.compile(r"(?i)\s*(-\s*topic|official( channel)?|vevo|channel|公式チャンネル|オフィシャル)\s*$")


def _clean(s: str) -> str:
    s = DATE.sub(" ", s)
    s = re.sub(r"[@＠]\S+.*$", " ", s)                    # "＠Kアリーナ横浜 ..." venue tail
    s = re.sub(r"\s+", " ", s).strip(" /／|｜-–—:：・,")
    return s


def _quoted(s: str) -> list[tuple[int, str]]:
    out = []
    for a, b in QUOTES:
        for m in re.finditer(re.escape(a) + r"([^" + re.escape(b) + r"]{1,80})" + re.escape(b), s):
            out.append((m.start(), m[1].strip()))
    return sorted(out)


def _kind(title: str, desc: str) -> list[str]:
    outside = title                                       # the title without the song name itself
    for _, t in _quoted(title):
        outside = outside.replace(t, " ")
    tags = " ".join(TAG.findall(title)) + " " + TAG.sub(" ", outside)
    found = []
    for k, pat in KINDS:
        if re.search(pat, tags, re.I):
            found.append(k)
    if "cover" not in found and re.search(r"カバー|cover", title, re.I) and \
            re.search(r"[『「][^』」]+[』」]\s*を?カバー|をカバーしました|\bcover(ed)? of\b", desc, re.I):
        found.append("cover")
    return found


def _norm(s: str) -> str:
    return re.sub(r"[\s・.\-_！!？?、。,]+", "", s).lower()


def _artist_candidates(meta: dict) -> list[str]:
    c = []
    for k in ("artist", "creator"):
        if meta.get(k):
            c += [x.strip() for x in str(meta[k]).split(",")]
    c += meta.get("artists") or []
    c += meta.get("creators") or []
    ch = CHANNEL_NOISE.sub("", meta.get("channel") or meta.get("uploader") or "").strip()
    if ch:
        c.append(ch)
    seen, out = set(), []
    for x in c:
        if x and _norm(x) not in seen:
            seen.add(_norm(x)); out.append(x)
    return out


def _split_dash(s: str, known: list[str]) -> tuple[str, str] | None:
    """'Artist - Song' or 'Song / Artist': the side that is a known artist (channel, tags) is the
    artist; otherwise the usual order, artist first."""
    parts = re.split(r"\s+[-–—]\s+|\s*[／/]\s*", s, maxsplit=1)
    if len(parts) != 2 or not all(p.strip() for p in parts):
        return None
    a, b = (p.strip() for p in parts)
    kn = [_norm(k) for k in known]
    if _norm(b) in kn or any(k and k in _norm(b) for k in kn):
        return a, b                                       # song, artist
    if _norm(a) in kn or any(k and k in _norm(a) for k in kn):
        return b, a
    if VERSION.search(b) or VERSION.search(a):            # "Song - Live ver.": no artist in the title
        return (a if VERSION.search(b) else b), ""
    return b, a


def guess(meta: dict) -> dict:
    title = meta.get("title") or ""
    desc = meta.get("description") or ""
    known = _artist_candidates(meta)
    kinds = _kind(title, desc)
    why, warn = [], []
    song = artist = original = None
    conf = 0

    if meta.get("track") and (meta.get("artist") or meta.get("artists")):        # YouTube Music facts
        song = meta["track"]
        artist = meta.get("artist") or ", ".join(meta["artists"])
        artist = artist.split(",")[0].strip()
        conf += 2; why.append("YouTube Music track and artist")

    q = _quoted(TAG.sub(lambda m: " " * len(m[0]), title))
    if not song and q:
        pos, song = q[0]
        why.append("song name in quotes in the title")
        conf += 2
        m = re.match(r"(.*?)\s*[（(]([^）)]+)[）)]\s*$", song)            # 「Song（Original artist）」
        if m and m[1]:
            song, inner = m[1].strip(), m[2].strip()
            if "cover" in kinds or re.search(re.escape(inner) + r"\s*の", desc):
                original = inner
                if "cover" not in kinds:
                    kinds.append("cover")
                why.append(f"({inner}) after the song name is the original artist")
        rest = TAG.sub(" ", title)
        for _, t in q:
            rest = rest.replace(t, " ")
        for a, b in QUOTES:
            rest = rest.replace(a, " ").replace(b, " ")
        rest = _clean(KIND_WORDS.sub(" ", NOISE.sub(" ", rest)))
        if rest:
            artist = rest; conf += 1; why.append("artist next to the song name in the title")
    if not song:                                          # a name quoted in the description that is in the title
        bare = _clean(NOISE.sub(" ", TAG.sub(" ", title)))
        hits = [t for _, t in _quoted(desc) if len(_norm(t)) >= 2 and _norm(t) in _norm(bare)]
        if hits:
            song = max(hits, key=len); conf += 2
            why.append("song name quoted in the description and in the title")
            left = _clean(KIND_WORDS.sub(" ", bare.replace(song, " ")))
            if left and not VERSION.search(left) and len(left) < 40:
                artist = left; why.append("artist next to the song name in the title")
            elif left:
                warn.append(f'The title also says "{left}" (a version of the song?).')
    if not song:
        sp = _split_dash(bare, known)
        if sp:
            song, artist = sp; conf += 1; why.append("title split at ' - ' or ' / '")
            artist = artist or None
        else:
            song = bare; why.append("whole title (no song name marked)")
            warn.append("The title does not mark the song name: check it.")

    if not artist and known:
        artist = known[0]; why.append("artist from the channel")
        if len(known) > 1:
            warn.append(f"Artist taken from the channel ({known[0]}); the video also names {', '.join(known[1:])}.")
    if artist and known and _norm(artist) in [_norm(k) for k in known]:
        conf += 1; why.append("artist matches the channel / credits")

    if song and _norm(song) in _norm(desc):
        conf += 1; why.append("song name also in the description")
    if "cover" in kinds and not original:                 # 「...」の『Song』をカバー / Artistの『Song』
        m = re.search(r"((?:[A-Za-z0-9][\w.&'’-]*\s?){1,5}|[^\s「『、。！!をがはにで]{2,20})\s*の\s*[『「]"
                      + re.escape(song or "\0") + r"[』」]", desc)
        if m:
            original = m[1].strip(); why.append(f"description: {m[0]}")
    if original:
        warn.append(f"Cover: sung by {artist or 'the channel'}, original by {original}.")

    mv_url = None
    if "live" in kinds or "documentary" in kinds:
        m = re.search(r"(?i)(?:MV|music video)[^\n]{0,20}\n?\s*(https?://(?:youtu\.be/|www\.youtube\.com/watch\?v=)[\w-]{11})", desc)
        if m:
            mv_url = m[1]
            warn.append(f"{'Live take' if 'live' in kinds else 'Not the MV'}; the description links the MV ({mv_url}), its sound is cleaner.")
        elif "live" in kinds:
            warn.append("Live take: crowd and MC may be in the sound.")
    if "documentary" in kinds:
        warn.append("Documentary: talking and other sounds over the music.")
        conf -= 1
    if "drum cover" in kinds:
        warn.append("Drum cover video: the drums may not be the original drummer's.")
    if meta.get("duration") and meta["duration"] > 600:
        warn.append(f"{meta['duration'] // 60} minutes long: a medley or a full concert?")
        conf -= 1
    kind = next((k for k, _ in KINDS if k in kinds), "video")
    if kind == "video" and not kinds:
        kind = "mv" if re.search(r"(?i)\bmv\b|music video", title) else "video"
    return {"title": (song or "").strip(), "artist": (artist or "").strip(), "kind": kind,
            "original_artist": original, "confidence": "high" if conf >= 4 else "medium" if conf >= 2 else "low",
            "warnings": warn, "mv_url": mv_url, "why": why,
            "video_title": title, "channel": meta.get("channel") or meta.get("uploader") or "",
            "comment": " / ".join(x for x in (f"Cover (original: {original})" if original else "",
                                               "Live" if kind == "live" else "") if x)}


def read(url: str, py: str = sys.executable) -> dict:
    """The video's facts from yt-dlp (no download)."""
    r = subprocess.run([py, "-m", "yt_dlp", "-J", "--no-playlist", "--skip-download", "--no-warnings", url],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode or not r.stdout.strip():
        raise RuntimeError((r.stderr.strip().splitlines() or ["yt-dlp failed"])[-1])
    j = json.loads(r.stdout)
    keep = ("id", "title", "description", "channel", "uploader", "duration", "track", "artist", "artists",
            "creator", "creators", "album", "tags", "thumbnail", "webpage_url")
    return {k: j.get(k) for k in keep}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for u in sys.argv[1:]:
        try:
            g = guess(read(u))
        except Exception as e:
            print(u, "ERROR", e); continue
        print(json.dumps({"url": u, **{k: g[k] for k in ("title", "artist", "kind", "original_artist", "confidence",
                                                         "warnings", "mv_url", "why")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
