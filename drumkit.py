"""The default drum kit: one sample per DTX lane, looked up in the kit folder (config.KIT_DIR) by
name, as .ogg, .wav or .flac. Put your own samples there under these names to change the sound;
make_kit.py writes a simple synthesized kit if the folder is empty."""
from config import KIT_DIR, PERC_KIT, SNARE_KIT

# DTX channel -> (WAV id, sample name)
NAMES = {
    "11": ("02", "hh_closed"),
    "12": ("03", "snare"),
    "13": ("04", "kick"),
    "14": ("05", "tom_high"),
    "15": ("06", "tom_mid"),
    "16": ("07", "crash_right"),
    "17": ("08", "tom_floor"),
    "18": ("09", "hh_open"),
    "19": ("0A", "ride"),
    "1A": ("0B", "crash_left"),
    "1C": ("0C", "kick"),        # left bass pedal: same kick sound
    "1B": ("0D", "hh_pedal"),    # left-foot hi-hat close: its own pedal chick sound
}
SAMPLE_NAMES = sorted({n for _, n in NAMES.values()})


def kit_file(name: str) -> str:
    for ext in (".ogg", ".wav", ".flac"):
        if (KIT_DIR / (name + ext)).exists():
            return name + ext
    return name + ".wav"


# DTX channel -> (WAV id, file in the kit folder)
KIT = {ch: (wid, kit_file(n)) for ch, (wid, n) in NAMES.items()}
# per-lane volume of the default kit, %: the pedal chick sits under the hats (charters use 30-35)
KIT_VOLUME = {"1B": 35}


def kit_map(folder) -> dict[str, tuple[str, str]]:
    """Like KIT for another kit folder with the same sample names: lane -> (WAV id, file)."""
    pick = lambda n: next((f"{n}{e}" for e in (".ogg", ".flac") if (folder / f"{n}{e}").exists()), f"{n}.wav")
    return {ch: (wid, pick(n)) for ch, (wid, n) in NAMES.items()}


def digital_kit():
    """Folder of the old-school digital kit (kit/digi, same sample names as the default kit), made
    by make_kit.py the first time; your own samples under the same names there win."""
    folder = KIT_DIR / "digi"
    if not all((folder / f"{n}.wav").exists() or (folder / f"{n}.ogg").exists() for n in SAMPLE_NAMES):
        import make_kit
        make_kit.write_digi(folder)
    return folder


CROSS_WORDS = ("xstick", "x-stick", "cross", "sidestick", "side_stick", "side-stick", "rimclick", "rim_click")


def stroke_of(name: str) -> str:
    """Snare stroke a sample is, from its file name: "cross" (cross-stick / side stick), "rim" (rim
    shot) or "normal"."""
    n = name.lower()
    return "cross" if any(w in n for w in CROSS_WORDS) else "rim" if "rim" in n else "normal"


def snare_kit() -> dict[str, list]:
    """Snare samples by stroke from the snare kit folder (settings.json "snare_kit_dir"); a folder
    without a cross-stick gets the synthesized one from make_kit.py."""
    files = [p for p in sorted(SNARE_KIT.glob("*")) if p.suffix.lower() in (".wav", ".ogg", ".flac")]
    if not any(stroke_of(p.stem) == "cross" for p in files):
        import make_kit
        make_kit.write_strokes(SNARE_KIT)
        files = [p for p in sorted(SNARE_KIT.glob("*")) if p.suffix.lower() in (".wav", ".ogg", ".flac")]
    out: dict[str, list] = {"normal": [], "rim": [], "cross": []}
    for p in files:
        out[stroke_of(p.stem)].append(p)
    return out


WOOD_WORDS = ("clave", "crave", "block", "wood", "mokugyo", "木魚", "temple")


def perc_kit() -> dict[str, list]:
    """Hand percussion samples by kind ("wood": wood block / clave / 木魚, "cowbell") from the
    percussion kit folder (settings.json "perc_kit_dir"); synthesized ones the first time."""
    files = [p for p in sorted(PERC_KIT.glob("*")) if p.suffix.lower() in (".wav", ".ogg", ".flac")]
    if not files:
        import make_kit
        make_kit.write_perc(PERC_KIT)
        files = [p for p in sorted(PERC_KIT.glob("*")) if p.suffix.lower() in (".wav", ".ogg", ".flac")]
    out: dict[str, list] = {"wood": [], "cowbell": []}
    for p in files:
        n = p.stem.lower()
        if "cowbell" in n or "bell" in n:
            out["cowbell"].append(p)
        elif any(w in n for w in WOOD_WORDS):
            out["wood"].append(p)
    return out


def tom_files() -> list:
    """The kit's tom samples (what a tom sounds like, against hand percussion)."""
    return [KIT_DIR / KIT[ch][1] for ch in ("14", "15", "17") if (KIT_DIR / KIT[ch][1]).exists()]
