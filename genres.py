"""Genres: the tool's starting point for a song (autochart.py --genre, the app's genre boxes).

Each genre sets the tuning its drumming usually needs; what the audio clearly shows still decides.
"tune" holds module settings (autochart's, or tempo's with "tempo."), plus:
  odd_bars       cost of non-4/4 bars, times the normal (>1 fewer odd bars, <1 more)
  bar_cost_16    cost of a 4/4 bar / bar_cost_12 of a 3/4 bar (a waltz swaps them)
"flags" are build options: max_bpm, snare24, ride_foot, kick_check, snare_match, swing, kit.
Several genres blend: a setting one of them changes is used; where several change the same one
they meet halfway (the mean), the highest max_bpm counts, a flag any of them sets is on."""
from __future__ import annotations

import numpy as np

# (group shown in the app, genres in it); the first group has no title
GROUPS = [
    ("", ["auto"]),
    ("Rock", ["rock", "visual", "punk", "funk"]),
    ("Metal", ["metal", "speedmetal"]),
    ("Pop", ["pop", "8090pop", "70pop"]),
    ("Jazz, Latin & Ballad", ["jazz", "bossa", "waltz", "tango", "ballad", "swing", "fusion"]),
    ("Progressive", ["progressive"]),
    ("Digital", ["edm", "citypop", "digirock"]),
]

LIVE_TEMPO = {"tempo.DRIFT_MAX": 0.010, "tempo.STEP_MIN": 0.015, "tempo.CLOSE_SECTIONS": 0.03}

GENRES: dict[str, dict] = {
    "auto": {"name": "I don't know WTF is it", "text": "The tool decides everything from the audio."},
    "rock": {"name": "Rock", "text": "4/4, 8th hi-hat grooves, crashes on downbeats (the default tuning)."},
    "visual": {"name": "Visual", "text": "Fast, often double bass; kicks off the beat around phrase starts and ends "
               "are kept without turning the bar into an odd bar; tempos up to 240.",
               "tune": {"DOUBLE_KICK_BPM": 140.0, "KICK_ROLL_FLOOR": -29.0, "odd_bars": 1.4},
               "flags": {"max_bpm": 240.0}},
    "punk": {"name": "Punk", "text": "Fast and steady 4/4 (tempos up to 260), double bass split right / left; "
             "off-beat snares (skank beat) are kept.",
             "tune": {"odd_bars": 1.5, "tempo.DRIFT_MAX": 0.012}, "flags": {"max_bpm": 260.0}},
    "funk": {"name": "Funk", "text": "Ghost notes on the snare are looked for and written soft; 16th hi-hats "
             "with quieter hits kept.",
             "tune": {"GHOST_TH": 0.12, "HAT_MIN_RISE": 3.5, "odd_bars": 1.3}},
    "metal": {"name": "Metal", "text": "Double kick from 120 BPM, fast kick rolls found more easily, tempos up to 260.",
              "tune": {"DOUBLE_KICK_BPM": 120.0, "DOUBLE_KICK_PAIR_BPM": 100.0, "KICK_ROLL_FLOOR": -30.0},
              "flags": {"max_bpm": 260.0}},
    "speedmetal": {"name": "Speed Metal", "text": "Usually over 180 BPM (up to 300): double kick streams, "
                   "time kept on the ride and hi-hat more than crashes.",
                   "tune": {"DOUBLE_KICK_BPM": 120.0, "DOUBLE_KICK_PAIR_BPM": 100.0, "KICK_ROLL_FLOOR": -30.0,
                            "RIDE_OVER_HAT": -3.0, "odd_bars": 1.3},
                   "flags": {"max_bpm": 300.0}},
    "pop": {"name": "Pop", "text": "Steady 4/4, odd bars only when clear, groove snare on 2 and 4, fewer ghost hats.",
            "tune": {"HAT_MIN_RISE": 5.0, "odd_bars": 1.5}, "flags": {"snare24": True}},
    "8090pop": {"name": "80's 90's pop", "text": "Kit sounds matched to the recording; tempo changes and a "
                "drummer's drift followed.",
                "tune": {**LIVE_TEMPO, "odd_bars": 1.3}, "flags": {"snare_match": True}},
    "70pop": {"name": "70's pop", "text": "Kit sounds matched to the recording; no click track, so the "
              "tempo drift and changes are followed closely.",
              "tune": {**LIVE_TEMPO, "tempo.DRIFT_MAX": 0.008, "tempo.STEP_MIN": 0.012,
                       "tempo.STEP_LOSS": 0.85, "odd_bars": 1.3},
              "flags": {"snare_match": True}},
    "jazz": {"name": "Jazz", "text": "Ride keeps time, hi-hat pedal on 2 and 4, ghost notes on the snare, soft "
             "(feathered) kicks kept, swung triplet feel preferred.",
             "tune": {"GHOST_TH": 0.12, "RIDE_OVER_HAT": -3.0, "KICK_MIN_LEVEL": -26.0, "TRIPLET_BIAS": 1.2,
                      "tempo.DRIFT_MAX": 0.010},
             "flags": {"ride_foot": True}},
    "bossa": {"name": "Bossa", "text": "Soft kicks kept (two per half bar, surdo style); a kick pair stays on the "
              "right foot unless it comes faster than a 16th apart; cross-stick clave on the snare.",
              "tune": {"KICK_MIN_LEVEL": -26.0, "DOUBLE_KICK_PAIR_BPM": 999.0, "odd_bars": 1.5}},
    "waltz": {"name": "Waltz", "text": "3/4 bars expected all through.",
              "tune": {"bar_cost_16": 2.5, "bar_cost_12": 0.0}},
    "tango": {"name": "Tango", "text": "Strong beats (marcato) and habanera figures, sudden stops and pushed / "
              "pulled tempo followed.",
              "tune": {**LIVE_TEMPO, "tempo.STEP_LOSS": 0.85, "odd_bars": 1.2}},
    "ballad": {"name": "Ballade", "text": "Slow tempo (up to 110): a fast reading is counted at half; 6/8 and "
               "12/8 feels kept as triplets; a live drift followed.",
               "tune": {"TRIPLET_BIAS": 0.8, "tempo.DRIFT_MAX": 0.010}, "flags": {"max_bpm": 110.0}},
    "swing": {"name": "Swing", "text": "12-beat (shuffle / 12/8) feel: triplets preferred, the late hi-hat on "
              "the last triplet.",
              "tune": {"TRIPLET_BIAS": 1.5}, "flags": {"swing": 24}},
    "fusion": {"name": "Fusion", "text": "Loud bass and guitar: kicks checked against the kick drum (no bass "
               "bleed); ghost notes, 16th detail and odd bars expected.",
               "tune": {"GHOST_TH": 0.15, "odd_bars": 0.8, "tempo.CHANGE_COST": 1.2},
               "flags": {"kick_check": True, "max_bpm": 240.0}},
    "progressive": {"name": "Progressive", "text": "Odd meters and tempo changes expected: odd bars cheaper, real "
                    "tempo steps kept, tempos up to 240. Tick Rock or Fusion too for how it sounds.",
                    "tune": {"odd_bars": 0.6, "tempo.REPEAT_DISCOUNT": 0.85, "tempo.CHANGE_COST": 1.0,
                             "tempo.STEP_MIN": 0.015, "tempo.STEP_LOSS": 0.85},
                    "flags": {"max_bpm": 240.0}},
    "edm": {"name": "EDM", "text": "One fixed tempo (programmed, no drift), steady 4/4, tempos up to 180, "
            "digital drum kit.",
            "tune": {"tempo.CLOSE_SECTIONS": 0.08, "tempo.DRIFT_MAX": 1.0, "odd_bars": 2.5},
            "flags": {"max_bpm": 180.0, "kit": "digi"}},
    "citypop": {"name": "City Pop", "text": "Old-school digital drum kit, 16th hi-hats, steady 4/4.",
                "tune": {"HAT_MIN_RISE": 3.5, "tempo.CLOSE_SECTIONS": 0.06, "odd_bars": 1.5},
                "flags": {"kit": "digi"}},
    "digirock": {"name": "DIGI Rock", "text": "Rock on a programmed beat: old-school digital drum kit, one "
                 "steady tempo.",
                 "tune": {"tempo.CLOSE_SECTIONS": 0.06, "tempo.DRIFT_MAX": 1.0, "odd_bars": 1.3},
                 "flags": {"kit": "digi"}},
}
assert set(GENRES) == {g for _, gs in GROUPS for g in gs}


def parse(names: str | list) -> list[str]:
    """ "metal,progressive" or a list -> known genre keys; "auto" (don't know) -> none."""
    if isinstance(names, str):
        names = names.replace(" ", "").split(",")
    chosen = [g for g in names if g and g != "auto"]
    bad = [g for g in chosen if g not in GENRES]
    if bad:
        raise ValueError(f"unknown genre {', '.join(bad)} (known: {', '.join(GENRES)})")
    return chosen


def blend(chosen: list[str]) -> tuple[dict, dict]:
    """(tune, flags) of one genre or several."""
    tune: dict[str, list[float]] = {}
    flags: dict = {}
    for g in chosen:
        for k, v in GENRES[g].get("tune", {}).items():
            tune.setdefault(k, []).append(v)
        for k, v in GENRES[g].get("flags", {}).items():
            if k in ("max_bpm", "swing"):
                flags[k] = max(flags.get(k, 0), v)
            else:
                flags.setdefault(k, v)
    return {k: float(np.mean(v)) for k, v in tune.items()}, flags


def for_page() -> list[dict]:
    """The groups and genres for the app's genre boxes."""
    return [{"group": t, "genres": [{"key": g, "name": GENRES[g]["name"], "text": GENRES[g]["text"]} for g in gs]}
            for t, gs in GROUPS]
