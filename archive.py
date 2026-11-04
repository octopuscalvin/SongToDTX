"""Unpack zip / 7z / rar archives of songs. Uses 7-Zip when it is installed (all three formats),
else Python's zipfile for zip and Windows' tar for the rest."""
import shutil, subprocess, zipfile
from pathlib import Path

KINDS = (".zip", ".7z", ".rar")
SEVEN = [Path(r"C:\Program Files\7-Zip\7z.exe"), Path(r"C:\Program Files (x86)\7-Zip\7z.exe")]


def is_archive(p) -> bool:
    return Path(p).suffix.lower() in KINDS


def seven_zip() -> str | None:
    return shutil.which("7z") or shutil.which("7za") or next((str(p) for p in SEVEN if p.exists()), None)


def unpack(src, dest) -> Path:
    """Unpack src into dest\\<archive name>\\ (kept if already there) and return the folder holding the
    songs: a single top folder inside the archive is stepped into."""
    src = Path(src)
    out = Path(dest) / src.stem
    if not (out.exists() and any(out.iterdir())):
        out.mkdir(parents=True, exist_ok=True)
        if exe := seven_zip():
            r = subprocess.run([exe, "x", str(src), f"-o{out}", "-y"], capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            if r.returncode:
                raise RuntimeError(f"7-Zip could not unpack {src.name}: {(r.stderr or r.stdout).strip()[-200:]}")
        elif src.suffix.lower() == ".zip":
            with zipfile.ZipFile(src) as z:
                z.extractall(out)
        else:
            r = subprocess.run(["tar", "-xf", str(src), "-C", str(out)], capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            if r.returncode:
                raise RuntimeError(f"Can't unpack {src.suffix} files here. Install 7-Zip (7-zip.org) and try again.")
    inner = [p for p in out.iterdir() if not p.name.startswith(".") and p.name != "__MACOSX"]
    return inner[0] if len(inner) == 1 and inner[0].is_dir() else out
