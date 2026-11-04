"""Downloads the two code parts that can't be shipped inside the installer (their repositories carry
no license): ADTOF-pytorch (drum transcription, weights CC BY-NC-SA 4.0) and the LarsNet code
(the LarsNet weights come with the installer). Pinned to the same commits as setup.bat.
Run again any time to repair; parts already in place are skipped."""
import io
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
PARTS = {
    "ADTOF-pytorch": ("https://github.com/xavriley/ADTOF-pytorch/archive/"
                      "85c192e78f716ea0b111cc8a5ee4a8f6a3a4f8a9.zip", "src/adtof_pytorch/__init__.py"),
    "larsnet": ("https://github.com/polimi-ispl/larsnet/archive/"
                "17d631fd18e77ee2f1d23ee7b3fc0fb46ae629e2.zip", "larsnet.py"),
}


def fetch(url: str) -> bytes:
    try:                                # Python's own SSL doesn't read the Windows certificate store
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=60, context=ctx) as r:
        size, data = int(r.headers.get("Content-Length") or 0), bytearray()
        while chunk := r.read(1 << 16):
            data += chunk
            if size:
                print(f"\r  {len(data) * 100 // size:3d}%", end="", flush=True)
    print()
    return bytes(data)


def main() -> int:
    for name, (url, marker) in PARTS.items():
        dest = VENDOR / name
        if (dest / marker).exists():
            continue
        print(f"Downloading {name} ...", flush=True)
        try:
            z = zipfile.ZipFile(io.BytesIO(fetch(url)))
        except OSError as e:
            print(f"Could not download {name}: {e}\nCheck the internet connection and start Song to DTX again.")
            return 1
        for info in z.infolist():
            rel = info.filename.split("/", 1)[1] if "/" in info.filename else ""
            if not rel or info.is_dir():
                continue
            out = dest / rel
            if out.exists():                    # keep what the installer brought (LarsNet weights)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(info))
    print("All parts are in place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
