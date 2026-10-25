"""Load a learned song from songs/<name>/, optionally transposed."""
import json
from pathlib import Path

import numpy as np

from .key import transpose_key

SONGS = Path(__file__).resolve().parent.parent / "songs"


def load_song(name, transpose=0):
    folder = Path(name) if Path(name).is_dir() else SONGS / Path(name).stem
    analysis = json.loads((folder / "analysis.json").read_text())
    p = analysis["pitch"]
    f0 = np.array(p["f0"], dtype=float) * 2 ** (transpose / 12)
    analysis["times"] = np.arange(len(f0)) * p["hop"] / p["sr"]
    analysis["f0"] = f0
    analysis["notes"] = [{**n, "midi": n["midi"] + transpose} for n in analysis["notes"]]
    analysis["key"] = transpose_key(analysis["key"], transpose)
    analysis["transpose"] = transpose
    analysis["folder"] = folder
    return analysis


def voice_ranges(folder):
    """(low, high) MIDI ranges recorded in past takes' metadata."""
    ranges = []
    for meta in sorted((Path(folder) / "takes").glob("*.json")):
        if meta.name.endswith(".score.json"):
            continue
        r = json.loads(meta.read_text()).get("voice_range")
        if r:
            ranges.append(r)
    return ranges
