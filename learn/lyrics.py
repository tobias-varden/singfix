"""Synced lyrics from LRCLIB (https://lrclib.net), cached per song.

    uv run -m learn.lyrics Fieldsofgold "fields of gold sting"           # list versions, take closest duration
    uv run -m learn.lyrics Fieldsofgold "fields of gold sting" --pick 3  # take another one

Saves songs/<name>/lyrics.lrc and lyrics.json: which version, on/off, a global offset, and tapped line
times when the recording's timing differs from the LRC (tap them in the live view with T).
"""
import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

API = "https://lrclib.net/api/search"
CONFIG = Path(__file__).resolve().parent.parent / "config.json"
STAMP = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")
MAX_LINE = 8.0  # a line stays up at most this long, so instrumental gaps show as gaps


def user_agent():
    """LRCLIB asks clients to identify themselves with a homepage or email; it's set in config.json."""
    contact = json.loads(CONFIG.read_text()).get("lrclib_contact") if CONFIG.exists() else None
    if not contact:
        raise SystemExit('Set "lrclib_contact" (your email or project URL) in config.json so LRCLIB knows who is asking.')
    return f"singfix/0.1 ({contact})"


def search(query, duration):
    """Synced versions for a free-text query, closest in duration first."""
    url = f"{API}?{urllib.parse.urlencode({'q': query})}"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": user_agent()}), timeout=10) as r:
        hits = [h for h in json.load(r) if h.get("syncedLyrics")]
    return sorted(hits, key=lambda h: abs((h.get("duration") or 0) - duration))


def parse_lrc(text):
    """[(seconds, text)] sorted by time; blank lines are kept as '' to mark breaks."""
    lines = []
    for raw in text.splitlines():
        words = STAMP.sub("", raw).strip()
        lines += [(int(m) * 60 + float(s), words) for m, s in STAMP.findall(raw)]
    return sorted(lines)


def fetch(song, query, pick=None):
    """List LRCLIB versions, cache one (closest duration unless `pick`), and reset its timing. True if saved."""
    duration = song["times"][-1]
    hits = search(query, duration)
    if not hits:
        print(f"No synced lyrics on LRCLIB for {query!r}.")
        return False
    choice = 0 if pick is None else pick
    print(f"Lyrics versions for {query!r} (song is {duration:.0f}s):")
    for i, h in enumerate(hits):
        mark = "→" if i == choice else " "
        print(f" {mark}{i:2d}  {h.get('duration') or 0:5.0f}s  {h['artistName']} - {h['trackName']}  [{h.get('albumName') or '?'}]")
    if not 0 <= choice < len(hits):
        print(f"--pick {choice} is out of range.")
        return False
    hit = hits[choice]
    folder = song["folder"]
    (folder / "lyrics.lrc").write_text(hit["syncedLyrics"], encoding="utf-8")
    (folder / "lyrics.json").write_text(json.dumps({
        "lrclib_id": hit["id"], "track": f"{hit['artistName']} - {hit['trackName']}",
        "album": hit.get("albumName"), "duration": hit.get("duration"), "enabled": True, "offset": 0.0,
    }, indent=2))
    if pick is None and len(hits) > 1:
        print("Not the right one? Re-run with --pick N.")
    if abs((hit.get("duration") or 0) - duration) > 3:
        print("Durations differ, so this is probably another cut: press T in the live view to tap the line starts.")
    return True


class Lyrics:
    """A song's lyric lines on the song clock, plus tap-sync and nudging. Times are seconds."""

    def __init__(self, folder):
        self.folder = folder
        self.settings = json.loads((folder / "lyrics.json").read_text())
        lines = parse_lrc((folder / "lyrics.lrc").read_text(encoding="utf-8"))
        sung = [i for i, (_, w) in enumerate(lines) if w]
        self.words = [lines[i][1] for i in sung]
        self.breaks = [i > 0 and not lines[i - 1][1] for i in sung]  # a blank LRC line before it: new stanza
        lrc = np.array([lines[i][0] for i in sung])
        # How long each line stays up: until the next LRC entry (blank or not), capped.
        follow = np.array([lines[i + 1][0] if i + 1 < len(lines) else lines[i][0] + MAX_LINE for i in sung])
        self.lengths = np.minimum(follow - lrc, MAX_LINE)
        tapped = self.settings.get("times")
        self.base = np.array(tapped) if tapped and len(tapped) == len(sung) else lrc
        self.cursor = None   # next line a tap assigns; set on the first tap
        self.undo_stack = []
        self.dirty = False

    @property
    def enabled(self):
        return self.settings["enabled"]

    def toggle(self):
        self.settings["enabled"] = not self.enabled
        self.dirty = True

    @property
    def panel(self):
        """Show the whole text in a side panel instead of the line strip."""
        return self.settings.get("panel", False)

    def toggle_panel(self):
        self.settings["panel"] = not self.panel
        self.dirty = True

    @property
    def follow(self):
        """Highlight and scroll to the current line in the panel."""
        return self.settings.get("follow", True)

    def set_follow(self, on):
        self.settings["follow"] = on
        self.dirty = True

    def starts(self):
        return self.base + self.settings["offset"]

    def current(self, t):
        """Index of the line being sung at song time t, or None in a gap."""
        starts = self.starts()
        i = int(np.searchsorted(starts, t, "right")) - 1
        ends = np.minimum(starts + self.lengths, np.append(starts[1:], np.inf))
        return i if i >= 0 and t < ends[i] else None

    def at(self, t):
        """(current line or None, next line or None) at song time t."""
        i = self.current(t)
        n = i + 1 if i is not None else int(np.searchsorted(self.starts(), t, "right"))
        return (self.words[i] if i is not None else None), (self.words[n] if n < len(self.words) else None)

    def tapping(self, t):
        """(last tapped line, line the next tap assigns) for the tap-sync display."""
        k = self.cursor if self.cursor is not None else int(np.searchsorted(self.starts(), t))
        return (self.words[k - 1] if k > 0 else None), (self.words[k] if k < len(self.words) else None)

    def tap(self, t):
        """Line `cursor` starts now; later lines move by the same amount so the preview stays close."""
        if self.cursor is None:
            self.cursor = int(np.searchsorted(self.starts(), t))
        k = self.cursor
        if k >= len(self.words):
            return
        self.undo_stack.append((k, self.base.copy()))
        self.base[k:] += t - self.starts()[k]
        self.cursor += 1
        self.dirty = True

    def undo(self):
        """Take back the last tap; with none left, step the cursor back a line."""
        if self.undo_stack:
            self.cursor, self.base = self.undo_stack.pop()
        elif self.cursor:
            self.cursor -= 1

    def nudge(self, seconds):
        self.settings["offset"] = round(self.settings["offset"] + seconds, 2)
        self.dirty = True

    def save(self):
        if not self.dirty:
            return
        if self.undo_stack:
            self.settings["times"] = [round(float(t), 2) for t in self.base]
        (self.folder / "lyrics.json").write_text(json.dumps(self.settings, indent=2))
        print(f"Saved lyric timing (offset {self.settings['offset']:+.2f}s"
              + (f", {len(self.undo_stack)} lines tapped)" if self.undo_stack else ")"))


def load(folder):
    """The song's Lyrics, or None if none are cached."""
    return Lyrics(folder) if (folder / "lyrics.lrc").exists() and (folder / "lyrics.json").exists() else None


def main():
    from .store import load_song
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("query", nargs="+", help="search text, e.g. title and artist")
    ap.add_argument("--pick", type=int, default=None, help="version number from the list")
    args = ap.parse_args()
    fetch(load_song(args.song), " ".join(args.query), args.pick)


if __name__ == "__main__":
    main()
