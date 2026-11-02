# Song to DTX

Turns a song into a drum chart (`.dtx`) for [DTXManiaNX](https://github.com/limyz/DTXmaniaNX), with a local
web app to check, fix and play the chart before saving it. A hobby project, free, never sold.

Version: see [VERSION](VERSION) and [CHANGELOG.md](CHANGELOG.md).

## What it does

1. **Source**: a song file (mp3, flac, wav, m4a, ogg, aiff), a YouTube link, a drum-score video, or whole album folders.
   A music video can also be added to a chart you already have (YouTube tab): its intro is cut to fit.
   **Existing chart** opens a chart you already have to fix notes, song info, jacket or preview in place.
2. **Generate**: separates the drums, finds the tempo and bar lines, transcribes the hits and writes a chart that
   two hands and two feet can play.
3. **Check and edit**: measures the tool is unsure about are explained in plain words. A DTXCreator-style editor
   (BPM lane + 11 drum lanes) lets you move, add and delete notes, and plays the chart with the music.
4. **Song info**: title, artist, comment, level (000-999, default 680), displayed BPM, jacket, preview clip (start bar, 5-17 bars, volume shape you can draw and hear).
5. **Output**: saves into `DTXFiles\AutoChart\<song> (auto vN)`, older versions move to `Old versions`.

## Requirements

- Windows 10/11, Python **3.11** ([python.org](https://www.python.org/downloads/)), [ffmpeg](https://ffmpeg.org/)
  (`winget install Gyan.FFmpeg`), DTXManiaNX.
- About **7 GB** of disk: PyTorch 4.4 GB (GPU build), other packages 0.7 GB, models 0.75 GB.
- An **NVIDIA graphics card** is strongly recommended. Without one it runs on the processor, several times slower
  (PyTorch CPU build, about 2.5 GB in total).

## Install

1. Download this repository (Code → Download ZIP) and unzip it, e.g. next to your DTXMania folder.
2. Double-click **`setup.bat`**. It installs the packages, downloads the two drum models and writes a starter
   drum kit. It asks before downloading the LarsNet weights (see Licenses); if the download fails it opens the
   page so you can download and unzip them yourself.
3. Double-click **`Song to DTX.bat`**. The app opens in your browser at http://127.0.0.1:8680.
   If your DTXMania folder isn't next to this one, the Output card asks where it is.

Optional settings go in `settings.json` (copy `settings.example.json`): other folders for the cache, models,
drum kit and DTXMania, extra sample folders, or `"device": "cpu"`.

## Drum sounds

The chart plays the samples in `kit\`: `kick`, `snare`, `hh_closed`, `hh_open`, `hh_pedal`, `tom_high`,
`tom_mid`, `tom_floor`, `crash_right`, `crash_left`, `ride` (each `.ogg`, `.wav` or `.flac`). `setup.bat`
fills missing ones with simple synthesized sounds; replace any of them with samples you're allowed to use.

## Command line

```
.venv\Scripts\python autochart.py song.wav out\folder --title "Title" --artist "Artist"
.venv\Scripts\python autochart.py -h          (all options: meter hints, tempo hints, kit, score mode ...)
.venv\Scripts\python batch_albums.py <dest> <album folder> [...]
.venv\Scripts\python video_score.py <score video> <source video> <work folder>
.venv\Scripts\python add_mv.py <chart folder> <music video URL|file> <work folder>
```

Useful hints when you know more than the audio tells: `--bpm 213` (the song's published tempo),
`--grid-from other.dtx` (take tempo and bar lines from another chart of the same song),
`--game-style` (ride written on the cymbal lane, as game charts do), `--ride-foot` (left-foot hi-hat under a ride).

## Licenses and credits

Song to DTX itself is under the [MIT License](LICENSE). The models it uses have their own licenses; they are
downloaded by `setup.bat` and not included here:

- [Demucs](https://github.com/facebookresearch/demucs) (MIT) — separates the drums from the song.
- [LarsNet](https://github.com/polimi-ispl/larsnet) — splits the drums into kick / snare / toms / hi-hat / cymbals.
  Model weights © 2023 A. I. Mezza, Image and Sound Processing Lab, Politecnico di Milano,
  **CC BY-NC 4.0 (non-commercial use only)**.
- [ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch) — drum transcription (ADTOF by M. Zehren et al.).
- [beat_this](https://github.com/CPJKU/beat_this) (MIT) — beat and downbeat tracking.
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube downloads. Only download what you have the right to use.

Songs, charts and samples you make or use stay yours and are never uploaded anywhere.
