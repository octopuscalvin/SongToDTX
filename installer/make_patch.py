"""Builds an update patch: the app and its charting rules only (a few MB), for a PC that already has
Song to DTX installed (either edition). Python, the packages, the models and ffmpeg stay as they are.

usage: make_patch.py <patch dir> --out <output dir> [--iscc <ISCC.exe>]

The patch refuses to install over a version whose requirements.txt differs from its own: a new
package needs the full installer.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEV_ONLY = (".gitignore", "setup.bat", "Song to DTX.bat")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iscc", type=Path, default=None, help="Inno Setup compiler; without it only the folder is made")
    a = ap.parse_args()
    a.stage, a.out = a.stage.resolve(), a.out.resolve()   # ISCC reads relative paths from the .iss folder
    sys.path.insert(0, str(REPO))
    import library_check
    ok, msg = library_check.check()
    if not ok:                                 # a patch carries only the owner's signed Library
        raise SystemExit(f"not building: {msg}. Sign the Library first (owner's private tools).")
    if a.stage.exists():
        shutil.rmtree(a.stage)
    a.stage.mkdir(parents=True)

    # the same app files the full installer carries (the files the user changes are not tracked)
    files = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
    for f in files:
        if f.startswith("installer/") or f in DEV_ONLY:
            continue
        (a.stage / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / f, a.stage / f)
    import json
    signed = json.loads((REPO / "LIBRARY.sig").read_text(encoding="utf-8"))["manifest"]["files"]
    for f in [*signed, "LIBRARY.sig", "download_parts.py"]:    # every signed file, also before its first commit
        (a.stage / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / f, a.stage / f)

    version = (REPO / "VERSION").read_text().strip()
    library = (REPO / "LIBRARY").read_text().strip()
    print(f"patch {version} / library {library}: {sum(1 for _ in a.stage.rglob('*') if _.is_file())} files, "
          f"{sum(p.stat().st_size for p in a.stage.rglob('*') if p.is_file()) / 1e6:.1f} MB")
    if a.iscc:
        a.out.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(a.iscc), f"/DStage={a.stage}", f"/DOut={a.out}", str(HERE / "SongToDTXPatch.iss")], check=True)


if __name__ == "__main__":
    main()
