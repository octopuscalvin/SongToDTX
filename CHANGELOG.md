# Changelog

Versions go 0.7.0, 0.7.1 … 0.9.9, 0.10.0, 0.10.1 … — 1.0.0 only when it is announced.

## 0.7.3

- Preview clip by bars: start at any bar or quarter bar (64.25) and run 5-17 bars (8 by default)
  instead of a fixed 15 s. The Play button now plays exactly that clip (it used to start at the
  song's beginning).
- Volume shape for the preview, edited on a picture of the clip with bar lines: drag the dots,
  double-click to add or remove one. It plays with the shape in the page and is saved the same way;
  the default rises from 0.8 over the first second and fades out over the last 30% (5 s at most).
  The choice is kept in autochart.json and shown again when the chart is opened.
- Hand rule: snare or tom on a 16th, then hi-hat + crash together (130 BPM and up): the hat and its
  pedal close are dropped, the crash stays.

## 0.7.2

- Double kick: above 120 BPM, already two kicks a 16th apart are split between the feet (right, left),
  so no foot plays two 16ths in a row; below that, runs of 3+ as before. Score-mode charts now get the
  same foot split.
- BGM: every chart folder now gets the original song as its BGM (`--bgm drumless` for the old way).
- Add MV: the chart backup is now `mstr.dtx.before_mv`, so DTXMania doesn't list it as a second song.
- Score mode: after an open hi-hat the closing closed hat gets the left-foot pedal (as charts from audio do).
- **Existing chart** tab: open any chart folder, check and move notes, play it, and change the song name,
  artist, comment, level, displayed BPM, jacket and preview start. "Save changes" writes into the same
  folder; the chart as it was is kept once as `mstr.dtx.bak`.
- The editor's backup of the tool's chart is now `mstr.dtx.auto` (was `mstr.auto.dtx`, which DTXMania listed).

## 0.7.1

- YouTube (beta): **Add a music video to a chart you already have** (`add_mv.py`). The app finds where the
  chart's music sits in the video's sound, cuts the video's longer intro (or puts black in front when it
  starts later), converts it to WMV and starts it with the BGM. The chart is backed up as
  `mstr.dtx.before_mv`. If the video is a different edit, it lists the measures that will be out of sync.

## 0.7.0

First public version.

- Song file, YouTube link or folder of albums → DTX drum chart for DTXManiaNX.
- Score-video mode (beta): follow the drum score shown in a drum-cover video, with the original video as BGA.
- Local web app: generate, check the flagged measures with explanations, edit notes in a DTXCreator-style
  editor, play the chart with the music, set song info / level / jacket / preview, save into DTXFiles.
- Playability rules: two hands, double kick, left-foot hi-hat spacing, hi-hat and snare rolls,
  crashes alternating left and right, rolls into a crash.
- Synthesized starter drum kit (`make_kit.py`); use your own samples by putting them in `kit\`.
