# singfix roadmap

## Done
- Learn a song: Demucs vocals → CREPE pitch → key → note bars (`learn_song.py`)
- Play + record on one clock, latency calibration (`audio/calibrate.py`, `record_take.py`)
- Onkey-style scoring with per-line report and practice-loop suggestion (`score.py`)
- Transpose with formant-preserving pitch shift, `--transpose N|auto`
- Live view (`live.py`)
- Synced lyrics from LRCLIB under the live view: version picker, L toggle, P full-text side panel with Follow, [ ] nudge, T tap-sync (`learn/lyrics.py`)

## Next: find songs that fit my voice
- **Range test** (when in the mood): ~30 s guided exercise — slide from lowest to highest note,
  then hold a few comfortable notes — to build a voice profile without needing a song.
- **Voice profile** from takes, using only on-note frames (current estimate from Fields of Gold: ~F2–G3).
  Also makes `--transpose auto` smarter (it picked -7; -4 fit best).
- **Batch learning**: `learn_song.py <folder>` to learn a whole library (~25 s/song on GPU).
- **`fit.py`**: rank learned songs by fit — melody range and span vs my range, best shift,
  and flag songs that fit with 0 to ±2 semitones (singable with the original recording).

## Ideas
- Remember the chosen transpose per song so `--transpose` isn't needed every time.
- Slow-down practice (`pedalboard.time_stretch` stretch_factor) for tricky lines.
- Note names with flats in flat keys (B♭ instead of A#).
- Lyrics: plain-only LRCLIB entries could be made usable via tap-sync; word-level (enhanced LRC) highlighting.
