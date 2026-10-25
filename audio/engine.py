"""Play a track and record the mic in one duplex stream, so both share a sample clock."""
import json
import queue
from pathlib import Path

import numpy as np
import sounddevice as sd

CONFIG = Path(__file__).resolve().parent.parent / "config.json"


def wasapi_defaults():
    api = next(i for i, h in enumerate(sd.query_hostapis()) if h["name"] == "Windows WASAPI")
    h = sd.query_hostapis(api)
    return h["default_input_device"], h["default_output_device"]


def load_config():
    if CONFIG.exists():
        return json.loads(CONFIG.read_text())
    inp, out = wasapi_defaults()
    return {"input": inp, "output": out, "samplerate": int(sd.query_devices(out)["default_samplerate"]), "latency_s": None}


def save_config(cfg):
    CONFIG.write_text(json.dumps(cfg, indent=2))


class Session:
    """Plays `track` (frames x 2, float32) from `start` and records mono mic aligned to it.

    Mic sample k was captured while song sample `start + k` was being sent out; subtract the
    round-trip latency to find which song moment the singer actually heard.
    """

    def __init__(self, track, sr, cfg, start=0, end=None, volume=1.0, live=False):
        self.track, self.sr, self.cfg = track, sr, cfg
        self.start, self.end = start, len(track) if end is None else min(end, len(track))
        self.volume = volume
        self.pos = start
        self.mic = np.zeros(self.end - self.start + sr, np.float32)  # +1 s tail to catch the latency
        self.recorded = 0
        self.live = queue.Queue() if live else None
        self.status_flags = 0
        self.stream_latency = None
        self.stop_requested = False  # set from another thread to end playback early

    def _callback(self, indata, outdata, frames, time, status):
        if status:
            self.status_flags += 1
        n = max(0, min(frames, self.end - self.pos))
        outdata[:n] = self.track[self.pos : self.pos + n] * self.volume
        outdata[n:] = 0
        k = min(frames, len(self.mic) - self.recorded)
        self.mic[self.recorded : self.recorded + k] = indata[:k, 0]
        if self.live is not None:
            self.live.put_nowait((self.pos, indata[:, 0].copy()))
        self.pos += frames
        self.recorded += k
        if k < frames or self.stop_requested:
            raise sd.CallbackStop

    def run(self):
        """Block until the section (plus tail) finishes or Ctrl+C. Returns the recorded mic."""
        with sd.Stream(
            samplerate=self.sr, channels=(1, 2), dtype="float32", latency="low",
            device=(self.cfg["input"], self.cfg["output"]), callback=self._callback,
        ) as stream:
            self.stream_latency = sum(stream.latency)
            try:
                while stream.active:
                    sd.sleep(100)
            except KeyboardInterrupt:
                print("\nStopped.")
        return self.mic[: self.recorded]
