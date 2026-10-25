"""Estimate a song's key with chroma + Krumhansl-Schmuckler profiles."""
import librosa
import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]
MINOR_STEPS = [0, 2, 3, 5, 7, 8, 10]


def detect_key(audio, sr):
    """Return dict with tonic, mode, name, scale pitch classes, and ranked candidates."""
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr).sum(axis=1)
    scores = []
    for tonic in range(12):
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            r = np.corrcoef(chroma, np.roll(profile, tonic))[0, 1]
            scores.append((float(r), tonic, mode))
    scores.sort(reverse=True)
    r, tonic, mode = scores[0]
    steps = MAJOR_STEPS if mode == "major" else MINOR_STEPS
    return {
        "tonic": tonic,
        "mode": mode,
        "name": f"{NOTE_NAMES[tonic]} {mode}",
        "confidence": round(r, 3),
        "scale": sorted((tonic + s) % 12 for s in steps),
        "candidates": [(f"{NOTE_NAMES[t]} {m}", round(s, 3)) for s, t, m in scores[:4]],
    }


def transpose_key(key, semitones):
    tonic = (key["tonic"] + semitones) % 12
    return {**key, "tonic": tonic, "name": f"{NOTE_NAMES[tonic]} {key['mode']}",
            "scale": sorted((p + semitones) % 12 for p in key["scale"])}
