"""Folders and options. Defaults work out of the box; settings.json next to this file (not in git,
see settings.example.json) can point any of them somewhere else. Every key is optional."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
try:
    SETTINGS = json.loads((HERE / "settings.json").read_text(encoding="utf-8"))
except FileNotFoundError:
    SETTINGS = {}


def _dir(key: str, default: Path) -> Path:
    v = SETTINGS.get(key)
    return Path(v) if v else default


CACHE = _dir("cache_dir", HERE / "cache")              # stems, beat/drum detections, app work folders
VENDOR = _dir("vendor_dir", HERE / "vendor")           # LarsNet + ADTOF-pytorch (setup.bat downloads them)
KIT_DIR = _dir("kit_dir", HERE / "kit")                # drum samples, see drumkit.py for the names
DTXMANIA = _dir("dtxmania_dir", HERE.parent / "DTXMania")   # DTXManiaNX.exe / DTXCreator.exe / DTXFiles
SAMPLE_DIRS = [Path(p) for p in SETTINGS.get("sample_dirs", [])]   # extra snare/kick samples for --snare-match
EDITING_DIR = _dir("editing_dir", Path())              # batch: finished albums from here move to completed\
# snare strokes (normal, rim shot, cross-stick): file names say which; make_kit.py fills it the first time
SNARE_KIT = _dir("snare_kit_dir", KIT_DIR / "snare")
# hand percussion heard on the tom lanes (wood block, clave, cowbell); file names say which
PERC_KIT = _dir("perc_kit_dir", KIT_DIR / "perc")


def device() -> str:
    """'cuda' on an NVIDIA GPU, else 'cpu' (much slower). settings.json "device" overrides."""
    if SETTINGS.get("device"):
        return SETTINGS["device"]
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
