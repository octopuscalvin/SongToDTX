"""Read and write DTX drum charts.

Timing model: a chart is a list of measures. Each measure has a length
multiplier (channel 02, 1.0 = 4/4) and starts at the BPM in effect at its start.
BPM changes (channel 03 = hex integer, channel 08 = #BPMxx reference) take effect
at the chip's position inside the measure.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

# DTX drum lanes
DRUM_CHANNELS = {
    "11": "HH",   # hi-hat closed
    "12": "SD",   # snare
    "13": "BD",   # bass drum
    "14": "HT",   # high tom
    "15": "LT",   # low tom
    "16": "CY",   # right cymbal
    "17": "FT",   # floor tom
    "18": "HHO",  # hi-hat open
    "19": "RD",   # ride
    "1A": "LC",   # left cymbal
    "1B": "LP",   # left pedal (hi-hat foot)
    "1C": "LBD",  # left bass drum
}

LINE_RE = re.compile(r"^#(\d{3})([0-9A-Za-z]{2})\s*:\s*(\S+)")


def b36(s: str) -> int:
    return int(s, 36)


def to_b36(n: int) -> str:
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return digits[n // 36] + digits[n % 36]


@dataclass
class Chip:
    measure: int
    pos: Fraction       # 0 <= pos < 1, fraction of the measure
    channel: str
    value: str          # two-char base-36 id


@dataclass
class Chart:
    headers: dict[str, str] = field(default_factory=dict)
    chips: list[Chip] = field(default_factory=list)
    measure_len: dict[int, float] = field(default_factory=dict)
    bpm_defs: dict[str, float] = field(default_factory=dict)

    @property
    def base_bpm(self) -> float:
        return float(self.headers.get("BPM", "120"))

    # ---- timing -----------------------------------------------------------
    def _tempo_events(self) -> list[tuple[int, Fraction, float]]:
        ev = []
        for c in self.chips:
            if c.channel == "03":
                ev.append((c.measure, c.pos, float(int(c.value, 16))))
            elif c.channel == "08":
                bpm = self.bpm_defs.get(c.value)
                if bpm:
                    ev.append((c.measure, c.pos, bpm))
        ev.sort(key=lambda e: (e[0], e[1]))
        return ev

    def time_of(self, measure: int, pos: Fraction) -> float:
        """Seconds from chart start (measure 000, pos 0)."""
        events = self._tempo_events()
        bpm = self.base_bpm
        t = 0.0
        ei = 0
        for m in range(measure + 1):
            mlen = self.measure_len.get(m, 1.0)
            end = pos if m == measure else Fraction(1)
            cur = Fraction(0)
            while ei < len(events) and events[ei][0] == m and events[ei][1] <= end:
                _, p, new_bpm = events[ei]
                t += float(p - cur) * 4 * mlen * 60 / bpm
                cur, bpm = p, new_bpm
                ei += 1
            t += float(end - cur) * 4 * mlen * 60 / bpm
            # skip events in this measure past `end` only when this is the target measure
        return t

    def bgm_offset(self) -> float:
        """Chart time at which the BGM (channel 01) starts."""
        bgm = [c for c in self.chips if c.channel == "01"]
        if not bgm:
            return 0.0
        c = min(bgm, key=lambda c: (c.measure, c.pos))
        return self.time_of(c.measure, c.pos)

    def drum_notes(self) -> list[tuple[float, str]]:
        """(seconds relative to BGM start, lane) for every drum chip."""
        off = self.bgm_offset()
        out = [
            (self.time_of(c.measure, c.pos) - off, DRUM_CHANNELS[c.channel])
            for c in self.chips
            if c.channel in DRUM_CHANNELS
        ]
        out.sort()
        return out


def read_dtx(path: str | Path) -> Chart:
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp932", errors="replace")
    chart = Chart()
    for line in text.splitlines():
        line = line.strip()
        m = LINE_RE.match(line)
        if m:
            measure, ch, data = int(m.group(1)), m.group(2).upper(), m.group(3)
            if ch == "02":
                chart.measure_len[measure] = float(data)
                continue
            n = len(data) // 2
            for i in range(n):
                v = data[2 * i: 2 * i + 2].upper()
                if v != "00":
                    chart.chips.append(Chip(measure, Fraction(i, n), ch, v))
            continue
        hm = re.match(r"^#([A-Za-z0-9_]+)\s*:?\s*(.*)$", line)
        if hm:
            key, val = hm.group(1).upper(), hm.group(2).strip()
            if key.startswith("BPM") and len(key) == 5:
                chart.bpm_defs[key[3:]] = float(val)
            else:
                chart.headers[key] = val
    # a bar length holds until the next ch02 (DTXMania behaviour): make it explicit per measure
    last = max([c.measure for c in chart.chips] + list(chart.measure_len), default=0)
    cur = 1.0
    for m in range(last + 1):
        cur = chart.measure_len.get(m, cur)
        if cur != 1.0:
            chart.measure_len[m] = cur
        else:
            chart.measure_len.pop(m, None)
    return chart


# ---- writing --------------------------------------------------------------
def _encode_measure(chips: list[tuple[Fraction, str]]) -> str:
    denom = 1
    for p, _ in chips:
        denom = denom * p.denominator // math.gcd(denom, p.denominator)
    slots = ["00"] * denom
    for p, v in chips:
        slots[int(p * denom)] = v
    return "".join(slots)


def sjis_safe(text: str) -> str:
    """Latin letters Shift-JIS can't hold: Å -> the Ångström sign (same look), others lose the
    accent (ä -> a)."""
    import unicodedata
    out = []
    for c in text:
        try:
            c.encode("cp932")
        except UnicodeEncodeError:
            if c == "Å":
                c = "Å"
            elif unicodedata.normalize("NFKD", c)[0].isascii():
                c = unicodedata.normalize("NFKD", c)[0]
        out.append(c)
    return "".join(out)


def write_dtx(path: str | Path, headers: dict[str, str], wavs: dict[str, str],
              chips: list[Chip], measure_len: dict[int, float] | None = None,
              bpm_defs: dict[str, float] | None = None, volumes: dict[str, int] | None = None) -> None:
    lines = ["; Created by Song to DTX", ""]
    for k, v in headers.items():
        v = sjis_safe(v)
        lost = [c for c in v if c.encode("cp932", "replace") == b"?" and c != "?"]
        if lost:  # DTXMania reads Shift-JIS: e.g. 你 is not in it
            print(f"      WARNING: #{k} has characters Shift-JIS can't hold: {ascii(''.join(lost))}")
        lines.append(f"#{k}: {v}")
    lines.append("")
    for k, v in (bpm_defs or {}).items():
        lines.append(f"#BPM{k}: {v:.2f}")
    for k, v in wavs.items():
        lines.append(f"#WAV{k}: {v}")
        if k in (volumes or {}):
            lines.append(f"#VOLUME{k}: {volumes[k]}")
    lines.append("")
    # DTXMania keeps a bar length (ch02) until the next ch02, so write every change, including
    # the return to 1.0 after an odd bar. measure_len lists only the non-4/4 measures.
    measure_len = measure_len or {}
    last = max([c.measure for c in chips] + list(measure_len), default=0)
    cur = 1.0
    for m in range(last + 1):
        length = measure_len.get(m, 1.0)
        if length != cur:
            lines.append(f"#{m:03d}02: {length}")
            cur = length
    groups: dict[tuple[int, str], list[tuple[Fraction, str]]] = {}
    for c in chips:
        groups.setdefault((c.measure, c.channel), []).append((c.pos, c.value))
    for (m, ch), items in sorted(groups.items()):
        lines.append(f"#{m:03d}{ch}: {_encode_measure(items)}")
    # bytes: text mode on Windows would turn each \r\n into \r\r\n
    Path(path).write_bytes(("\r\n".join(lines) + "\r\n").encode("cp932", errors="replace"))
