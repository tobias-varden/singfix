"""Sing along to a learned song and record a take.

    uv run record_take.py Fieldsofgold --start 0 --end 35 --guide 0.3 --transpose auto

--guide sets how loud the original singer is (1 = full mix, 0 = karaoke).
--transpose shifts the song by semitones; `auto` fits it to your range from past takes.
Use headphones. Run `uv run python -m audio.calibrate` once first.
Re-plot a saved take:  uv run record_take.py Fieldsofgold --replot 20261024-123717 --latency 0.078
"""
import argparse
import json
import time

import librosa
import numpy as np
import soundfile as sf

from audio.engine import Session, load_config
from learn.notes import hz_to_midi
from learn.pitch import track
from learn.plot import plot_melody
from learn.store import load_song, voice_ranges
from learn.transpose import stems, suggest
from score import print_report, score_take


def build_track(song, sr, guide):
    mix, vocals, song_sr = stems(song, song["transpose"])
    if guide < 1:
        mix = mix - (1 - guide) * vocals
    if song_sr != sr:
        mix = librosa.resample(mix, orig_sr=song_sr, target_sr=sr)
    return np.ascontiguousarray(np.clip(mix.T, -1, 1), dtype=np.float32)


def pick_transpose(name, value):
    if value != "auto":
        return int(value)
    ranges = voice_ranges(load_song(name)["folder"])
    if not ranges:
        print("No past takes to learn your range from; not transposing.")
        return 0
    semitones = suggest(load_song(name), ranges)
    print(f"Auto transpose: {semitones:+d} semitones to centre the melody in your range.")
    return semitones


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song", help="learned song name, e.g. Fieldsofgold")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--guide", type=float, default=1.0, help="original vocal level, 0..1")
    ap.add_argument("--transpose", default="0", help="semitones (e.g. -7) or 'auto'")
    ap.add_argument("--volume", type=float, default=1.0)
    ap.add_argument("--floor", type=float, default=-38, help="mic level (dBFS) below which you count as silent")
    ap.add_argument("--replot", help="re-analyze a saved take (e.g. 20261024-123717) instead of recording")
    ap.add_argument("--latency", type=float, help="with --replot: override latency in seconds")
    args = ap.parse_args()

    if args.replot:
        take = load_song(args.song)["folder"] / "takes" / args.replot
        analyze(take, args.floor, args.latency)
        return

    cfg = load_config()
    if cfg.get("latency_s") is None:
        print("No latency calibration yet; run `uv run python -m audio.calibrate`. Using 0 for now.")
    latency = cfg.get("latency_s") or 0.0
    sr = cfg["samplerate"]
    song = load_song(args.song, transpose=pick_transpose(args.song, args.transpose))
    audio = build_track(song, sr, args.guide)

    start = int(args.start * sr)
    end = None if args.end is None else int(args.end * sr)
    session = Session(audio, sr, cfg, start=start, end=end, volume=args.volume)
    print(f"Singing in {song['key']['name']} from {args.start:.0f}s. Ctrl+C to stop early.")
    mic = session.run()
    if session.status_flags:
        print(f"  warning: {session.status_flags} audio glitches (over/underruns)")

    finish_take(song, mic, sr, args.start, latency, args.guide, args.floor)


def finish_take(song, mic, sr, start_s, latency, guide, floor):
    """Save a recorded take, plot it, score it, and print the report."""
    takes = song["folder"] / "takes"
    takes.mkdir(exist_ok=True)
    take = takes / time.strftime("%Y%m%d-%H%M%S")
    sf.write(take.with_suffix(".wav"), mic, sr)
    meta = {"start_s": start_s, "sr": sr, "latency_s": latency, "guide": guide,
            "transpose": song["transpose"], "duration_s": len(mic) / sr}
    take.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    analyze(take, floor)
    result = score_take(song, take, floor)
    take.with_suffix(".score.json").write_text(json.dumps(result, indent=2))
    print_report(result, song["folder"].name, guide, song["transpose"])
    return take, result


def analyze(take, floor, latency=None):
    """Track a saved take's pitch, note your range, and plot it over the song as it was transposed."""
    mic, sr = sf.read(take.with_suffix(".wav"), dtype="float32")
    meta = json.loads(take.with_suffix(".json").read_text())
    if latency is not None:
        meta["latency_s"] = latency
    song = load_song(take.parent.parent, transpose=meta.get("transpose", 0))

    # Mic sample k heard song time start + k/sr - latency.
    vt, vf0 = track(mic, sr, floor_dbfs=floor)
    vt = vt + meta["start_s"] - meta["latency_s"]
    vm = hz_to_midi(vf0[~np.isnan(vf0)])
    if len(vm) > 100:  # at least a second of singing
        meta["voice_range"] = [round(float(p), 2) for p in np.percentile(vm, [5, 95])]
    take.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    span = (meta["start_s"], meta["start_s"] + len(mic) / sr)
    plot_melody(take.with_suffix(".png"), song["times"], song["f0"], song["notes"], song["key"], voice=(vt, vf0), span=span)
    print(f"Saved {take}.wav/.png ({np.mean(~np.isnan(vf0)):.0%} voiced, latency {meta['latency_s'] * 1000:.0f} ms)")


if __name__ == "__main__":
    main()
