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
