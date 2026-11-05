"""Library signature: the tool's files as released by its owner. LIBRARY.sig lists every file of the
tool with its SHA-256, signed with the owner's private key; this checks it with the owner's public
key below (plain Python, no packages). On a PC without the owner's key, a changed, missing or
unsigned file stops charting: only the owner can update the tool. On the owner's PC (settings.json
"owner_key": their public key file) it is only a warning, so they can work on the rules.

Signing (owner only, with the private key): their private tools write LIBRARY.sig."""
import base64
import hashlib
import json
import os
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIG = HERE / "LIBRARY.sig"
OWNER_E = 65537
OWNER_N = int("0x89ea7643846a2259f41003e3240cd5c9eff0759f17b9b9c337bc835544a9bfecdf86dd43ae81d157f507f07357d8d597c3fcc345627e47eef83a96955142a189012d66a513cf8215b031641ac9f555aa09d9294ebc85aa96a5458a212301205109139314f6391488378444b61ac07a1a7177b656f134ef8f716cded0fd810d15eda3381e0aab04660d8f935472903bd21bdcd9b88493ec68605b2bf990146fedee413c54b7e796f4befa270045a7e365eb784e8386620cbcc69e0996d954f19867b9f0ae7bc59f68688f748fb89b4e8a30b16d4d4d88c1a0cdd9e7d0e7a11f6fb73c75fbc7a93a5592e0e4d20cc6094f76ad211b6d4c521752b823e523f53dfd", 16)
SIGNED = (".py", ".pyd", ".html")               # with VERSION, LIBRARY and requirements.txt
# not the tool's own files: developer and installed runtime folders
SKIP = ("installer/", ".venv/", "vendor/", "cache/", "kit/", "python/", "models/", "ffmpeg/")
EXTRA = ("VERSION", "LIBRARY", "requirements.txt")
# PKCS#1 v1.5 DigestInfo prefix for SHA-256
SHA256_INFO = bytes.fromhex("3031300d060960864801650304020105000420")


def tool_files(root: Path = HERE) -> list[str]:
    """The files a release signs, relative to the tool folder (a release has compiled .pyd modules)."""
    out = [f for f in EXTRA if (root / f).exists()]
    for d, dirs, names in os.walk(root):
        rel_d = Path(d).relative_to(root).as_posix()
        rel_d = "" if rel_d == "." else rel_d + "/"
        dirs[:] = sorted(x for x in dirs if x != "__pycache__" and not (rel_d + x + "/").startswith(SKIP))
        out += [rel_d + n for n in sorted(names) if Path(n).suffix in SIGNED]
    return sorted(set(out), key=out.index)


def digest(rel: str, root: Path = HERE) -> str:
    return hashlib.sha256((root / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def manifest_bytes(m: dict) -> bytes:
    return json.dumps(m, sort_keys=True, separators=(",", ":")).encode()


def _verify(data: bytes, sig: bytes, n: int = OWNER_N, e: int = OWNER_E) -> bool:
    k = (n.bit_length() + 7) // 8
    if len(sig) != k:
        return False
    em = pow(int.from_bytes(sig, "big"), e, n).to_bytes(k, "big")
    t = SHA256_INFO + hashlib.sha256(data).digest()
    return em == b"\x00\x01" + b"\xff" * (k - len(t) - 3) + b"\x00" + t


def owner_pc(settings: dict) -> bool:
    """settings.json "owner_key" names the owner's public key file (SSH2 / PuTTY public key) and it is theirs."""
    try:
        text = Path(settings["owner_key"]).read_text(encoding="ascii")
        b64 = "".join(l for l in text.splitlines() if l and not l.startswith("----") and ":" not in l)
        blob, f, i = base64.b64decode(b64), [], 0
        while i < len(blob):
            n = struct.unpack(">I", blob[i:i + 4])[0]
            f.append(blob[i + 4:i + 4 + n])
            i += 4 + n
        return f[0] == b"ssh-rsa" and int.from_bytes(f[2], "big") == OWNER_N
    except (KeyError, OSError, ValueError, IndexError, struct.error):
        return False


def check() -> tuple[bool, str]:
    """(official, message): official when LIBRARY.sig is the owner's and every file matches it."""
    if not SIG.exists():
        return False, "no Library signature (LIBRARY.sig)"
    try:
        s = json.loads(SIG.read_text(encoding="utf-8"))
        m, sig = s["manifest"], base64.b64decode(s["signature"])
    except (ValueError, KeyError):
        return False, "the Library signature is damaged"
    if not _verify(manifest_bytes(m), sig):
        return False, "the Library signature is not the owner's"
    files = m["files"]
    changed = [f for f, h in files.items() if not (HERE / f).exists() or digest(f) != h]
    extra = [f for f in tool_files() if f not in files]
    if changed or extra:
        return False, "changed outside an official update: " + ", ".join((changed + extra)[:8]) +             (" ..." if len(changed + extra) > 8 else "")
    return True, f"Library {m.get('library', '?')} signed by the owner"


def require(settings: dict) -> None:
    """Stops charting with a changed Library, except on the owner's PC."""
    ok, msg = check()
    if ok:
        return
    if owner_pc(settings):
        print(f"      Library: {msg} (owner's PC: allowed; sign it before releasing)")
        return
    raise SystemExit(f"LIBRARY NOT OFFICIAL: {msg}. Only the owner can update Song to DTX's Library; "
                     "reinstall it or apply the official update.")
