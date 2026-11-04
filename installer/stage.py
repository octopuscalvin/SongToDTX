"""Assembles one install folder for the Windows installer (installer\SongToDTX.iss).

usage: stage.py <stage dir> --python <Python 3.11 folder> --models <folder> --ffmpeg <folder>

<stage dir>\python\Lib\site-packages must already hold the packages, e.g.
  pip install --target <stage dir>\python\Lib\site-packages torch==2.6.0+cu124 torchaudio==2.6.0+cu124
      -r requirements.txt --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
<models> holds pretrained_larsnet_models\, HTDemucs (Hugging Face cache folder models--adefossez--HTDemucs)
and beat_this-final0.ckpt. The ADTOF-pytorch and LarsNet code are not copied (no license to
redistribute): download_parts.py fetches them during setup."""
import argparse
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CRLF = b"\r\n"
SKIP_LIB = {"site-packages", "test", "idlelib", "tkinter", "turtledemo", "ensurepip", "__pycache__"}
SKIP_DLL = ("_d.pyd", ".pdb", "_tkinter.pyd", "tcl86t.dll", "tk86t.dll")
PY_FILES = ("python.exe", "pythonw.exe", "python3.dll", "python311.dll", "vcruntime140.dll",
            "vcruntime140_1.dll", "LICENSE.txt")


def crlf(src: Path, dst: Path, extra: bytes = b"") -> None:
    text = src.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", CRLF)
    dst.write_bytes(text + extra.replace(b"\r\n", b"\n").replace(b"\n", CRLF))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", type=Path)
    ap.add_argument("--python", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--ffmpeg", type=Path, required=True)
    a = ap.parse_args()
    st, py = a.stage, a.stage / "python"

    # Python runtime (site-packages is installed separately)
    for f in PY_FILES:
        shutil.copy2(a.python / f, py / f)
    shutil.copytree(a.python / "DLLs", py / "DLLs", dirs_exist_ok=True,
                    ignore=lambda d, names: [n for n in names if n.endswith(SKIP_DLL)])
    for p in (a.python / "Lib").iterdir():
        if p.name in SKIP_LIB:
            continue
        if p.is_dir():
            shutil.copytree(p, py / "Lib" / p.name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(p, py / "Lib" / p.name)
    # build-only files pip brings along
    sp = py / "Lib" / "site-packages"
    for junk in list(sp.rglob("*.lib")) + [sp / "torch" / "include", sp / "bin"]:
        if junk.is_dir():
            shutil.rmtree(junk)
        elif junk.exists():
            junk.unlink()

    # the app: tracked files, minus the developer setup
    files = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
    for f in files:
        if f.startswith("installer/") or f in (".gitignore", "setup.bat", "Song to DTX.bat"):
            continue
        (st / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / f, st / f)
    shutil.copy2(REPO / "download_parts.py", st / "download_parts.py")
    crlf(HERE / "Song to DTX.bat", st / "Song to DTX.bat")
    crlf(HERE / "THIRD_PARTY_NOTICES.txt", st / "THIRD_PARTY_NOTICES.txt", (REPO / "LICENSE").read_bytes())

    # models
    shutil.copytree(a.models / "pretrained_larsnet_models", st / "vendor" / "larsnet" / "pretrained_larsnet_models",
                    dirs_exist_ok=True)
    shutil.copytree(a.models / "models--adefossez--HTDemucs",
                    st / "models" / "huggingface" / "hub" / "models--adefossez--HTDemucs", dirs_exist_ok=True)
    (st / "models" / "torch" / "hub" / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.models / "beat_this-final0.ckpt", st / "models" / "torch" / "hub" / "checkpoints")

    # ffmpeg
    (st / "ffmpeg").mkdir(exist_ok=True)
    for f in ("ffmpeg.exe", "ffprobe.exe"):
        shutil.copy2(a.ffmpeg / f, st / "ffmpeg" / f)

    # starter drum kit, made with the staged Python (also proves it runs)
    subprocess.run([str(py / "python.exe"), "-s", "make_kit.py"], cwd=st, check=True)
    print("staged", st)


if __name__ == "__main__":
    main()
