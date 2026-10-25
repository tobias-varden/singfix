"""Learn a song: separate vocals, track pitch, detect key, build note bars, cache it all.

    uv run learn_song.py path/to/song.wav
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from learn.key import detect_key
from learn.notes import to_notes
from learn.pitch import HOP, SR, track
from learn.plot import plot_melody
from learn.separate import separate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("--model", default="htdemucs", help="demucs model (htdemucs_ft is slower, cleaner)")
    ap.add_argument("--out", default="songs")
    args = ap.parse_args()

    song = Path(args.song)
    out = Path(args.out) / song.stem
    out.mkdir(parents=True, exist_ok=True)
    clock = time.perf_counter()

    print("Separating vocals...")
    vocals, harmonic, sr = separate(song, args.model)
    sf.write(out / "vocals.wav", vocals, sr)
    print(f"  {time.perf_counter() - clock:.1f}s")

    print("Detecting key...")
    key = detect_key(harmonic, sr)
    print(f"  {key['name']} (r={key['confidence']}), runners-up: {key['candidates'][1:]}")

    print("Tracking pitch...")
    times, f0 = track(vocals, sr)
    print(f"  {np.mean(~np.isnan(f0)):.0%} of frames voiced, {time.perf_counter() - clock:.1f}s")

    notes = to_notes(times, f0, key["scale"])
    print(f"  {len(notes)} notes")

    analysis = {
        "song": str(song.resolve()),
        "key": key,
        "pitch": {"sr": SR, "hop": HOP, "f0": [None if np.isnan(v) else round(float(v), 2) for v in f0]},
        "notes": notes,
    }
    (out / "analysis.json").write_text(json.dumps(analysis))
    plot_melody(out / "melody.png", times, f0, notes, key)
    print(f"Saved to {out}/ in {time.perf_counter() - clock:.1f}s")


if __name__ == "__main__":
    main()
