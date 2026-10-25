"""Split a song into vocals and accompaniment with Demucs."""
import librosa
import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model


def load_audio(path):
    """Load any audio file as float32 (channels, samples) plus its sample rate."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return audio.T, sr


def separate(path, model_name="htdemucs", device="cuda"):
    """Return (vocals, harmonic, sr): mono vocal stem and mono drum-free accompaniment."""
    model = get_model(model_name).to(device).eval()
    audio, sr = load_audio(path)
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    if sr != model.samplerate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=model.samplerate)
        sr = model.samplerate

    wav = torch.from_numpy(audio)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    with torch.inference_mode():
        stems = apply_model(model, wav[None], device=device, split=True, overlap=0.25, progress=True)[0]
    stems = (stems * ref.std() + ref.mean()).cpu().numpy()

    by_name = dict(zip(model.sources, stems))
    vocals = by_name["vocals"].mean(0)
    harmonic = sum(s for name, s in by_name.items() if name not in ("drums",)).mean(0)
    return vocals, harmonic, sr
