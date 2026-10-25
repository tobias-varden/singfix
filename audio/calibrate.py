"""Measure round-trip latency: play chirps, find them in the mic by cross-correlation.

With headphones, hold an ear cup right against the mic while it runs.
    uv run python -m audio.calibrate
"""
import numpy as np
from scipy.signal import chirp, correlate

from .engine import Session, load_config, save_config


# Irregular gaps: with even spacing, a shift of one gap still lines up all but one chirp,
# so a swallowed first chirp makes the wrong peak win (we measured 678 ms instead of 78).
ONSETS = np.cumsum([0.6, 0.43, 0.71, 0.52, 0.84, 0.47, 0.66, 0.58])


def chirp_track(sr, length=0.05):
    t = np.arange(int(length * sr)) / sr
    pulse = (chirp(t, 500, length, 4000) * np.hanning(len(t))).astype(np.float32)
    mono = np.zeros(int((ONSETS[-1] + 0.5) * sr), np.float32)
    for onset in ONSETS:
        s = int(onset * sr)
        mono[s : s + len(pulse)] = pulse
    return mono


def measure(cfg, volume=0.5):
    sr = cfg["samplerate"]
    mono = chirp_track(sr)
    session = Session(np.repeat(mono[:, None], 2, axis=1), sr, cfg, volume=volume)
    mic = session.run()
    corr = np.abs(correlate(mic, mono, mode="full", method="fft"))[len(mono) - 1 :]  # non-negative lags
    window = corr[: sr]  # latency is under 1 s
    lag = int(np.argmax(window))
    guard = int(0.02 * sr)
    runner_up = np.concatenate([window[: max(0, lag - guard)], window[lag + guard :]]).max()
    clarity = window[lag] / (runner_up + 1e-9)  # how much the best lag beats any other
    return lag / sr, clarity, session.stream_latency


def main():
    cfg = load_config()
    print("Playing 8 chirps... (headphones: hold an ear cup against the mic)")
    measured, clarity, reported = measure(cfg)
    print(f"  measured {measured * 1000:.1f} ms (clarity {clarity:.1f}x over next-best lag), WASAPI reports {reported * 1000:.1f} ms")
    if clarity >= 1.5:
        cfg.update(latency_s=round(measured, 4), latency_method="measured")
    else:
        print("  Couldn't hear the chirps clearly; using the reported latency instead.")
        cfg.update(latency_s=round(reported, 4), latency_method="reported")
    save_config(cfg)
    print(f"Saved latency {cfg['latency_s'] * 1000:.1f} ms ({cfg['latency_method']}) to config.json")


if __name__ == "__main__":
    main()
