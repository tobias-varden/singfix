"""Transpose a learned song: shift its audio (cached per semitone) and its reference melody."""
import librosa
import numpy as np
import pedalboard
import soundfile as sf

from .separate import load_audio


def shift(audio, sr, semitones):
    """Pitch-shift (channels, samples) audio with Rubber Band, keeping timing and vocal formants."""
    if semitones == 0:
        return audio
    return pedalboard.time_stretch(
        np.ascontiguousarray(audio, dtype=np.float32), sr,
        pitch_shift_in_semitones=semitones, preserve_formants=True,
    )


def stems(song, semitones):
    """Return (mix (2, n), vocals (n,), sr) at the song's own rate, shifted by `semitones`."""
    cache = song["folder"] / "transposed"
    mix_path, vocals_path = cache / f"{semitones:+d}_mix.wav", cache / f"{semitones:+d}_vocals.wav"
    if semitones and mix_path.exists() and vocals_path.exists():
        mix, sr = load_audio(mix_path)
        vocals, _ = sf.read(vocals_path, dtype="float32")
        return mix, vocals, sr

    mix, sr = load_audio(song["song"])
    if mix.shape[0] == 1:
        mix = np.repeat(mix, 2, axis=0)
    vocals, vsr = sf.read(song["folder"] / "vocals.wav", dtype="float32")
    vocals = librosa.resample(vocals, orig_sr=vsr, target_sr=sr)
    vocals = np.pad(vocals, (0, max(0, mix.shape[1] - len(vocals))))[: mix.shape[1]]
    if semitones:
        print(f"Transposing {semitones:+d} semitones (first time only)...")
        mix = shift(mix, sr, semitones)
        vocals = shift(vocals[None], sr, semitones)[0]
        cache.mkdir(exist_ok=True)
        sf.write(mix_path, mix.T, sr)
        sf.write(vocals_path, vocals, sr)
    return mix, vocals, sr


def suggest(song, voice_ranges):
    """Pick a shift that centres the song's melody in the singer's range.

    `voice_ranges` are (low, high) MIDI percentiles from past takes, measured in absolute pitch.
    """
    f0 = song["f0"][~np.isnan(song["f0"])]
    lo, hi = np.percentile(69 + 12 * np.log2(f0 / 440), [5, 95])
    you_lo, you_hi = np.median(np.array(voice_ranges), axis=0)
    return int(np.clip(round((you_lo + you_hi) / 2 - (lo + hi) / 2), -12, 12))
