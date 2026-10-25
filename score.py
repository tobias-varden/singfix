"""Score a take against a learned song, using onkey's rules.

    uv run score.py Fieldsofgold                    # latest take
    uv run score.py Fieldsofgold 20261024-123717
"""
import argparse
import json

import numpy as np
import soundfile as sf

from learn.key import NOTE_NAMES
from learn.notes import hz_to_midi
from learn.pitch import track
from learn.store import load_song

FULL, ZERO = 50, 150         # cents: full credit within FULL, fading to none at ZERO
WINDOW, SLACK = 0.5, 0.08    # each 0.5 s window may slide ±80 ms to line up with the singer
EARLY, LATE = 0.3, 1.5       # search range for the singer's overall delay
HIT_CENTS, SCOOP, MIN_NOTE = 100, 0.08, 0.15
FULL_COVERAGE = 0.4          # singing 40% of the vocal earns full coverage
LINE_GAP = 0.6               # a pause this long between notes starts a new line


def fold(cents):
    """Fold to ±600 cents so an octave away counts as the same note."""
    return (cents + 600) % 1200 - 600


def credit(err):
    return np.clip((ZERO - np.abs(err)) / (ZERO - FULL), 0, 1)


def note_name(m):
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def tendency(cents):
    if np.isnan(cents):
        return "-"
    if abs(cents) < 20:
        return "in tune"
    return f"{'flat' if cents < 0 else 'sharp'} {abs(cents):.0f}c"


def score_take(song, take, floor=-38):
    mic, sr = sf.read(take.with_suffix(".wav"), dtype="float32")
    meta = json.loads(take.with_suffix(".json").read_text())
    if meta.get("transpose", 0) != song.get("transpose", 0):
        song = load_song(song["folder"], transpose=meta.get("transpose", 0))
    vt, vf = track(mic, sr, floor_dbfs=floor)
    vt = vt + meta["start_s"] - meta["latency_s"]
    voice = hz_to_midi(vf)

    t = song["times"]
    hop = t[1] - t[0]
    span = (t >= vt[0]) & (t <= vt[-1])
    t = t[span]
    line = hz_to_midi(song["f0"][span])
    bars = np.full(len(t), np.nan)
    for n in song["notes"]:
        bars[(t >= n["start"]) & (t < n["end"])] = n["midi"]

    def voice_at(delay):
        """Your pitch for each song moment, assuming you sang `delay` seconds late."""
        return np.interp(t + delay, vt, voice, left=np.nan, right=np.nan)

    def credits(u):
        err = np.fmin(np.abs(fold((u - line) * 100)), np.abs(fold((u - bars) * 100)))
        return credit(err)

    # Overall delay first, then let each window wiggle around it.
    delays = np.arange(-round(EARLY / hop), round(LATE / hop) + 1) * hop
    delay = delays[np.argmax([np.nansum(credits(voice_at(d))) for d in delays])]

    k = round(SLACK / hop)
    shifts = np.arange(-k, k + 1)
    U = np.stack([voice_at(delay + s * hop) for s in shifts])
    C = credits(U)
    win = round(WINDOW / hop)
    best = np.zeros(len(t), int)
    for w0 in range(0, len(t), win):
        sums = np.nansum(C[:, w0 : w0 + win], axis=1) - 1e-6 * np.abs(shifts)  # ties go to no shift
        best[w0 : w0 + win] = np.argmax(sums)
    idx = np.arange(len(t))
    u, cr = U[best, idx], C[best, idx]

    def summarize(mask):
        sung = mask & ~np.isnan(line)
        both = sung & ~np.isnan(u)
        raw = (u[both] - line[both]) * 100
        # Tuning only counts frames where you're on the right note; a wrong note isn't "sharp".
        on_note = np.abs(fold(raw)) < HIT_CENTS
        return {
            "accuracy": float(np.mean(cr[both])) if both.any() else 0.0,
            "coverage": float(both.sum() / sung.sum()) if sung.any() else 0.0,
            "on_note": float(np.mean(on_note)) if both.any() else 0.0,
            "tuning_cents": float(np.median(fold(raw)[on_note])) if on_note.any() else float("nan"),
            "octave": int(np.median(np.round(raw / 1200))) if both.any() else 0,
        }

    # Group the whole song into lines first, then keep only lines the take fully covers,
    # so a loop's tail catching the next line's first notes doesn't count against you.
    song_lines = []
    for n in song["notes"]:
        if not song_lines or n["start"] - song_lines[-1][-1]["end"] > LINE_GAP:
            song_lines.append([])
        song_lines[-1].append(n)
    song_lines = [ln for ln in song_lines if ln[0]["start"] >= t[0] and ln[-1]["end"] <= t[-1]]

    lines, notes = [], []
    for ln in song_lines:
        scored = []
        for n in ln:
            if n["end"] - n["start"] < MIN_NOTE:
                continue
            m = (t >= n["start"] + SCOOP) & (t < n["end"])
            err = fold((u[m] - n["midi"]) * 100)
            voiced = ~np.isnan(err)
            hit = bool(np.sum(np.abs(err[voiced]) < HIT_CENTS) >= 0.5 * m.sum())
            scored.append({**n, "hit": hit, "cents": float(np.median(err[voiced])) if voiced.any() else None})
        if scored:
            lines.append(scored)
            notes.extend(scored)

    report_lines = []
    for ln in lines:
        start, end = ln[0]["start"], ln[-1]["end"]
        stats = summarize((t >= start) & (t < end))
        misses = [n for n in ln if not n["hit"]]
        report_lines.append({
            "start": start, "end": end, **stats,
            "hit": len(ln) - len(misses), "notes": len(ln),
            "melody": " ".join(note_name(n["midi"]) for n in ln),
            "misses": [{"note": note_name(n["midi"]), "at": n["start"], "cents": n["cents"]} for n in misses],
        })

    in_lines = np.zeros(len(t), bool)
    for ln in lines:
        in_lines |= (t >= ln[0]["start"]) & (t < ln[-1]["end"])
    overall = summarize(in_lines)
    final = overall["accuracy"] * min(1.0, overall["coverage"] / FULL_COVERAGE)
    return {
        "score": round(100 * final),
        "delay_s": round(float(delay), 3),
        **overall,
        "notes_hit": sum(n["hit"] for n in notes),
        "notes_total": len(notes),
        "lines": report_lines,
    }


def octave_text(o):
    return {0: "in the same octave as", -1: "an octave below", 1: "an octave above"}.get(o, f"{o:+d} octaves from")


def print_report(result, song_name, guide=0.3, transpose=0):
    r = result
    if r["coverage"] == 0:
        print("\nNo singing detected during the vocal parts. Mic too quiet? Try a lower --floor (e.g. -45).")
        return
    print(f"\nScore {r['score']}/100   accuracy {r['accuracy']:.0%} | coverage {r['coverage']:.0%} | "
          f"notes hit {r['notes_hit']}/{r['notes_total']} | timing {r['delay_s'] * 1000:+.0f} ms")
    print(f"Mostly {octave_text(r['octave'])} the original. On the right note {r['on_note']:.0%} of the time, "
          f"and when you are: {tendency(r['tuning_cents'])}\n")
    print(f"{'#':>2}  {'time':<12} {'hit':>5} {'acc':>5} {'on note':>8}  {'tuning':<10} melody")
    for i, ln in enumerate(r["lines"], 1):
        melody = ln["melody"] if len(ln["melody"]) <= 40 else ln["melody"][:37] + "..."
        print(f"{i:>2}  {ln['start']:5.1f}-{ln['end']:<5.1f} {ln['hit']:>2}/{ln['notes']:<2} {ln['accuracy']:>4.0%} {ln['on_note']:>8.0%}  "
              f"{tendency(ln['tuning_cents']):<10} {melody}")

    sung = [ln for ln in r["lines"] if ln["coverage"] > 0.2]
    if sung:
        worst = min(sung, key=lambda ln: ln["accuracy"])
        i = r["lines"].index(worst) + 1
        missed = ", ".join(f"{m['note']}@{m['at']:.1f}s ({tendency(m['cents']) if m['cents'] is not None else 'silent'})"
                           for m in worst["misses"][:4])
        print(f"\nPractice line {i} next: {worst['accuracy']:.0%} accurate, on the note {worst['on_note']:.0%}" +
              (f"; missed {missed}" if missed else ""))
        print(f"  uv run record_take.py {song_name} --start {max(0, worst['start'] - 1.5):.1f} "
              f"--end {worst['end'] + 0.5:.1f} --guide {guide}" + (f" --transpose {transpose}" if transpose else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("take", nargs="?", help="take name; defaults to the latest")
    ap.add_argument("--floor", type=float, default=-38)
    args = ap.parse_args()

    song = load_song(args.song)
    takes = song["folder"] / "takes"
    take = takes / args.take if args.take else sorted(takes.glob("*.wav"))[-1].with_suffix("")
    result = score_take(song, take, args.floor)
    take.with_suffix(".score.json").write_text(json.dumps(result, indent=2))
    meta = json.loads(take.with_suffix(".json").read_text())
    print_report(result, song["folder"].name, meta.get("guide", 0.3), meta.get("transpose", 0))


if __name__ == "__main__":
    main()
