"""Turn a pitch line into note bars, one per held pitch."""
import numpy as np


def hz_to_midi(f0):
    return 69 + 12 * np.log2(f0 / 440.0)


def to_notes(times, f0, scale, hysteresis=0.7, min_len=0.1, merge_gap=0.05, snap=0.35):
    """Segment pitch into notes; a note changes only when pitch strays > `hysteresis` semitones.

    Off-scale notes get pulled to the nearest scale note when the sung pitch is within `snap` of the boundary.
    """
    midi = hz_to_midi(f0)
    hop = times[1] - times[0]
    runs, current, start = [], None, 0
    for i, m in enumerate(np.append(midi, np.nan)):
        if np.isnan(m):
            if current is not None:
                runs.append((start, i))
            current = None
        elif current is None:
            current, start = round(m), i
        elif abs(m - current) > hysteresis:
            runs.append((start, i))
            current, start = round(m), i

    notes = []
    for s, e in runs:
        if (e - s) * hop < min_len:
            continue
        pitch = float(np.median(midi[s:e]))
        note = _nudge_to_scale(pitch, scale, snap)
        start, end = float(times[s]), float(times[e - 1] + hop)
        if notes and notes[-1]["midi"] == note and start - notes[-1]["end"] <= merge_gap:
            notes[-1]["end"] = end
        else:
            notes.append({"start": round(start, 3), "end": round(end, 3), "midi": note})
    for n in notes:
        n["end"] = round(n["end"], 3)
    return notes


def _nudge_to_scale(pitch, scale, snap):
    nearest = round(pitch)
    if nearest % 12 in scale:
        return nearest
    for candidate in sorted((nearest - 1, nearest + 1), key=lambda c: abs(pitch - c)):
        if candidate % 12 in scale and abs(pitch - candidate) < 0.5 + snap:
            return candidate
    return nearest
