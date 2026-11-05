# Changelog

Versions go 0.7.0, 0.7.1 … 0.9.9, 0.10.0, 0.10.1 … — 1.0.0 only when it is announced.

## 0.7.5

- Windows installers (Inno Setup, `installer\`): an NVIDIA and a CPU edition with Python, packages,
  models, ffmpeg and a starter kit inside; `stage.py` assembles the folder, `SongToDTX.iss` builds it.
  The ADTOF-pytorch and LarsNet code (no license to redistribute) are fetched during setup by
  `download_parts.py`.
- App: a floating octopus at the bottom right says a random line from `app/declare.txt` (5 s on,
  2 s off; click it for another one).
- Separated drum tracks are cached by the song's full path, not its folder name: two folders with
  the same name (e.g. two `completed` folders) no longer share each other's drums.
- App: a new song no longer keeps the previous song's name, artist, jacket and preview bar; only
  what was typed for it is kept. Step 3 shows a loading animation while the chart is made
  (`app\loading.gif`, `.webp`, `.png` or `.svg` replaces it), and the time taken is shown when done.
- Short version (`trim.py`, step 3 "Trim song into short cut"): keeps the notes of bars From..To,
  then 3 bars of music fading out, and ends there. BGM and BGA video are cut to match and the
  title gets "(Short ver.)"; the full version stays and unticking goes back to it.
- Double kick: above 150 BPM an 8th-triplet kick stream (3+ in a row) alternates right / left
  too, so no foot plays 3 fast kicks in a row.
- Hi-hat check: hats that are really piano or keyboard notes (no hat on the hi-hat stem or the
  drum stem, the band louder than the drums) are dropped and their measures listed for checking.
- Fast tom rolls: the hits ADTOF's peak picking cancels inside a roll are picked from its tom
  activation (cached per song); tom recall against official charts 70% -> 78%.
- App: Rebuild works on an opened chart: a new chart from its own BGM with the options it was made
  with (now saved in `autochart.json` "build"; older charts: their BPM if they have one), the new
  meter hints and its music video. The opened chart stays as it is.
- Archives: a zip, 7z or rar of songs can be dropped on the app (`archive.py`; 7-Zip when it is
  installed, else Python's zip reader and Windows' tar). One song inside goes to "Song file"; more
  go to the Batch folders list, which also takes archive paths and has its own drop zone.
- Batch: a folder of song folders named "Title - Artist" is charted as one box, with title and
  artist from each folder's name and the jacket from a picture in that folder or the song's own art.
- Tempo grid: a very short tempo section that the previous section's switch has already passed no
  longer crashes the build ("cannot unpack non-iterable NoneType").
- Kick rolls: a fast double-bass roll that blurs into one rumble (the kick stem's low end never
  drops through the beat and pulses at the 16th or 32nd rate) gets the whole 16th / 32nd stream,
  split between the feet. Runs of 2+ beats only; a single such beat is listed instead.
- Hi-hat phrases: consecutive 8th hi-hat groove bars keep one sound - all open, all closed, or open
  and closed in turn (only when the closed ones are really shut; listed) - decided from the phrase's median ring
  instead of hit by hit.
- Editor: double-click a hi-hat note, or select hi-hats and press O, to switch closed / open.
- Ride groove: in a bar with 6+ ride hits (right hand on the ride, left on the snare) a closed
  hi-hat is the foot: it becomes a hi-hat pedal chip (LP) and the measures are listed.
- Silent drums: any note where the separated drum stem is silent (40 dB or more under the song's
  usual drum level) is dropped and listed by measure; a piano intro read from the full mix no
  longer comes out as hats, ride and crashes. (The hi-hat check could miss these: over silence its
  attack test compares noise.)
- Add MV: the video offset is the one most matching windows agree on; with a different edit a
  median of all windows could land on no real offset.
- Add MV, live take (`add_mv.py --live`, app: "Live take or another tempo"): a video whose music
  runs at another, changing tempo (a live performance, a slower or faster version) is sped up and
  slowed down moment by moment so its drummer's hits meet the chart's music. Both recordings' drum
  hits are matched in 8 s windows; parts without drums follow the harmony and are listed by measure.
  The result reports how close the hits land.
- Cymbal rolls (`--cymbal-roll "9.5; 63-66; 114.5-115.5"`): a roll or soft swell on the cymbals that
  has no strokes the transcriber can hear is written as 16ths, right cymbal on the 8ths and left
  cymbal between, with a volume per half bar that follows how loud the cymbals are (a swell from
  silence steps up). Stretches where the cymbals keep sounding with no notes are listed for checking.
- Tempo map: two tempo switches that crossed while being placed no longer leave a section running
  backwards in time (the short section between them is dropped and its neighbours' switch searched
  again). A real tempo step of 2-4% between two long sections (each fits its own steady tempo far
  better than one shared tempo) is no longer merged as a "drifting take"; slow live drifts still are.
- Genre (`--genre metal,progressive`, `genres.py`; app: step 2, picked before Generate): the song's
  genre is the tool's starting point. 21 genres in groups - Rock (Rock, Visual, Punk, Funk), Metal
  (Metal, Speed Metal), Pop (Pop, 80's 90's pop, 70's pop), Jazz, Latin & Ballad (Jazz, Bossa, Waltz,
  Tango, Ballade, Swing, Fusion), Progressive, Digital (EDM, City Pop, DIGI Rock) - each setting the
  tuning its drumming usually needs: double kick, tempo ceiling, odd-bar cost, following a drifting
  tempo, triplet feel, ride, soft kicks, kick check, kit sound. Several genres can be ticked: what one
  changes is used, and where two change the same setting they meet halfway. "I don't know WTF is it"
  (no genre) decides everything from the audio, as before. The app shows the groups as buttons that
  open their genres. The chart remembers its genre; change it and Rebuild when the first chart looks
  wrong. Batch folders use it too.
- Ghost notes (Funk, Jazz, Fusion): a second, softer snare pass of the transcriber; a soft snare is
  kept where the separated snare drum has a hit and is written at half volume.
- Digital drum kit (EDM, City Pop, DIGI Rock): old-school drum-machine sounds made by `make_kit.py`
  into `kit\digi` the first time (your own samples under the same names there win).
- App: a YouTube song with "Use the video as BGA" no longer fails at the end with "expected str,
  bytes or os.PathLike object, not bool".
- Library version (`LIBRARY`, shown beside the app version): the charting rules' own version,
  starting at 1.0, also saved in each chart's `autochart.json`.
- Update patch (`installer\make_patch.py`, `SongToDTXPatch.iss`): the app and its charting rules
  only (about 2.5 MB), installed over either edition. It needs an existing install, keeps settings,
  kit, cache and charts, and asks for the full installer when the Python packages changed.
- Octopus app icon (`app\icon.png`, `app\icon.ico`): browser tab, installer, Start menu and desktop
  shortcuts.

## 0.7.4

- First build writes the preview sound itself (`preview.py`): 8 bars of the ORIGINAL song (never the
  drumless track), from the loudest stretch in the first 70% of the song, with the arcade-style volume
  shape. The app re-cuts and reshapes it with the same code.
- Add MV: a video from another master that runs slightly fast or slow (e.g. 0.04%, 0.1 s by the end)
  is measured over the whole song and stretched to fit.
- `--bpm`: one known tempo for the whole song (e.g. a game song's published BPM); bar lines and odd
  bars are still found.
- `--grid-from <chart>`: tempo and bar lines from another chart of the song (official, or one whose
  bar lines were fixed by hand), lined up automatically by its kick and snare notes; a tempo over
  `--max-bpm` is counted at half.
- `--game-style`: no ride lane - the ride goes on the right cymbal and a crash at the same moment
  on the left, as official game charts write it.
- With `--ride-foot`: under a ride, bars whose heard foot chicks fit a steady pattern (8ths, off-8ths,
  2 and 4, beats) get the whole pattern.
- Charts are written in cp932 (Windows Japanese), so titles like "～" survive.
- Charts end their lines with a plain CRLF (they had an extra CR, and adding a video doubled the
  blank lines each time).
- Add MV: the video is always 1280x720 with square pixels, other shapes get black bars; an odd-sized
  video (e.g. 1906x1080) was drawn off frame in DTXMania.
- Hi-hat rolls: the hat under a snare in a run of 16th hats goes at any tempo (was 130 BPM and up).

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
