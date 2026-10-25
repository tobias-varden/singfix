"""Track the singer's pitch with CREPE."""
import librosa
import numpy as np
import torch
import torchcrepe
from scipy.signal import medfilt

SR = 16000
HOP = 160  # 10 ms frames


def track(vocals, sr, device="cuda", model="full", confidence=0.45, gate_db=-40, floor_dbfs=None):
    """Return (times, f0_hz) with NaN where the singer is silent or unsure.

    `gate_db` is relative to the loudest frame; `floor_dbfs` is an absolute floor for mic takes,
    where room noise would otherwise pass a relative gate.
    """
    audio = librosa.resample(vocals, orig_sr=sr, target_sr=SR) if sr != SR else vocals
    x = torch.from_numpy(audio.astype(np.float32))[None]
    f0, periodicity = torchcrepe.predict(
        x, SR, HOP, fmin=50, fmax=1100, model=model,
        batch_size=2048, device=device, return_periodicity=True,
    )
    f0 = f0[0].cpu().numpy()
    periodicity = medfilt(periodicity[0].cpu().numpy(), 3)

    # Demucs leaves faint bleed and reverb tails; ignore frames that are much quieter than the vocal peak.
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=HOP, center=True)[0][: len(f0)]
    loud = librosa.amplitude_to_db(rms, ref=np.max) > gate_db
    if floor_dbfs is not None:
        loud &= librosa.amplitude_to_db(rms, ref=1.0) > floor_dbfs

    voiced = (periodicity > confidence) & loud[: len(f0)]
    f0 = np.where(voiced, medfilt(f0, 5), np.nan)
    times = np.arange(len(f0)) * HOP / SR
    return times, f0
