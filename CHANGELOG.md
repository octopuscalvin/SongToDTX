# Changelog

Versions go 0.7.0, 0.7.1 … 0.9.9, 0.10.0, 0.10.1 … — 1.0.0 only when it is announced.

## 0.7.1

- YouTube (beta): **Add a music video to a chart you already have** (`add_mv.py`). The app finds where the
  chart's music sits in the video's sound, cuts the video's longer intro (or puts black in front when it
  starts later), converts it to WMV and starts it with the BGM. The chart is backed up as
  `mstr.before_mv.dtx`. If the video is a different edit, it lists the measures that will be out of sync.

## 0.7.0

First public version.

- Song file, YouTube link or folder of albums → DTX drum chart for DTXManiaNX.
- Score-video mode (beta): follow the drum score shown in a drum-cover video, with the original video as BGA.
- Local web app: generate, check the flagged measures with explanations, edit notes in a DTXCreator-style
  editor, play the chart with the music, set song info / level / jacket / preview, save into DTXFiles.
- Playability rules: two hands, double kick, left-foot hi-hat spacing, hi-hat and snare rolls,
  crashes alternating left and right, rolls into a crash.
- Synthesized starter drum kit (`make_kit.py`); use your own samples by putting them in `kit\`.
