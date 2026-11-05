"""The default drum kit: one sample per DTX lane, looked up in the kit folder (config.KIT_DIR) by
name, as .ogg, .wav or .flac. Put your own samples there under these names to change the sound;
make_kit.py writes a simple synthesized kit if the folder is empty."""
from config import KIT_DIR

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


def digital_kit():
    """Kit file (JSON, lane -> [sample, volume]) of the old-school digital kit in kit/digi, made
    by make_kit.py the first time; your own samples under the same names there win."""
    import json
    folder = KIT_DIR / "digi"
    if not all((folder / f"{n}.wav").exists() or (folder / f"{n}.ogg").exists() for n in SAMPLE_NAMES):
        import make_kit
        make_kit.write_digi(folder)
    pick = lambda n: next((f"{n}{e}" for e in (".ogg", ".flac") if (folder / f"{n}{e}").exists()), f"{n}.wav")
    spec = {ch: [pick(n), KIT_VOLUME.get(ch, 100)] for ch, (_, n) in NAMES.items()}
    (folder / "kit.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    return folder / "kit.json"
