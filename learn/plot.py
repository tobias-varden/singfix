"""Draw the learned melody: pitch line over note bars, with the song's scale shaded."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .key import NOTE_NAMES
from .notes import hz_to_midi


def plot_melody(path, times, f0, notes, key, seconds_per_row=30, voice=None, span=None):
    """`voice` is an optional (times, f0) line for the user's singing; `span` limits the plot to (t0, t1)."""
    midi = hz_to_midi(f0)
    lo, hi = np.nanmin(midi), np.nanmax(midi)
    if voice is not None and not np.all(np.isnan(voice[1])):
        vm = hz_to_midi(voice[1])
        lo, hi = min(lo, np.nanpercentile(vm, 1)), max(hi, np.nanpercentile(vm, 99))
    lo, hi = int(lo) - 2, int(hi) + 2
    first, last = span or (0, times[-1])
    first = seconds_per_row * int(first // seconds_per_row)
    rows = max(1, int(np.ceil((last - first) / seconds_per_row)))
    fig, axes = plt.subplots(rows, 1, figsize=(18, 2.6 * rows), squeeze=False)
    for r, ax in enumerate(axes[:, 0]):
        t0, t1 = first + r * seconds_per_row, first + (r + 1) * seconds_per_row
        for m in range(lo, hi + 1):
            if m % 12 in key["scale"]:
                ax.axhspan(m - 0.5, m + 0.5, color="#4a90d9" if m % 12 == key["tonic"] else "#eef2f7", alpha=0.25, lw=0)
        for n in notes:
            if n["end"] > t0 and n["start"] < t1:
                ax.add_patch(plt.Rectangle((n["start"], n["midi"] - 0.4), n["end"] - n["start"], 0.8, color="#f5a623", alpha=0.8))
        sel = (times >= t0) & (times < t1)
        ax.plot(times[sel], midi[sel], color="#222", lw=0.9)
        if voice is not None:
            vt, vm = voice[0], hz_to_midi(voice[1])
            vsel = (vt >= t0) & (vt < t1)
            ax.plot(vt[vsel], vm[vsel], color="#d0021b", lw=1.1)
        ax.set_xlim(t0, t1)
        ax.set_ylim(lo, hi)
        ticks = [m for m in range(lo, hi + 1) if m % 12 in key["scale"]]
        ax.set_yticks(ticks, [f"{NOTE_NAMES[m % 12]}{m // 12 - 1}" for m in ticks], fontsize=7)
    axes[0, 0].set_title(f"Key: {key['name']}  (confidence {key['confidence']})  ·  {len(notes)} notes")
    axes[-1, 0].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
