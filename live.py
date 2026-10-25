"""Sing with a live view: note bars scroll toward the "now" line and your pitch is drawn over them.

    uv run live.py Fieldsofgold --transpose -4 --guide 0.3

Esc stops early. When it ends, the take is saved and scored like record_take.py.

Lyrics (from LRCLIB, see learn/lyrics.py) show under the notes once fetched with --lyrics "title artist".
L hides/shows them, P swaps the line strip for a side panel with the whole text (its Follow
button highlights and tracks the current line),
[ and ] shift them 0.1 s, T starts tap-sync: Space marks each line's start as it's
sung, Backspace takes a tap back. Timing and on/off are saved per song.
"""
import argparse
import queue
import threading
import time

import numpy as np
import pyqtgraph as pg
import torch
import torchcrepe
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.signal import firwin, lfilter

from audio.engine import Session, load_config
from learn import lyrics as lyrics_lib
from learn.key import NOTE_NAMES
from learn.notes import hz_to_midi
from learn.store import load_song
from record_take import build_track, finish_take, pick_transpose
from score import HIT_CENTS, MIN_NOTE, SCOOP, SLACK, credit, fold

CREPE_SR, HOP, WINDOW = 16000, 160, 1024  # CREPE wants 16 kHz, 1024-sample frames; we step 10 ms
PAST, FUTURE = 3.0, 6.0                   # seconds shown behind / ahead of now

GREEN, YELLOW, RED, GREY = "#3ddc84", "#ffd166", "#ef476f", "#7a8194"
BAR, BAR_HIT, BAR_MISS = "#f5a623", "#1f6e4a", "#3a4152"  # darker hit green so green dots stay visible on it


def note_name(m):
    m = int(round(m))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


class PitchTracker(threading.Thread):
    """Turns live mic blocks into (song time, MIDI pitch) points every 10 ms, with CREPE tiny on the GPU."""

    def __init__(self, session, sr, latency, floor_dbfs, confidence=0.4):
        super().__init__(daemon=True)
        if sr % CREPE_SR:
            raise ValueError(f"live view needs a device rate that's a multiple of {CREPE_SR} Hz (got {sr})")
        self.session, self.sr, self.latency = session, sr, latency
        self.floor, self.confidence = floor_dbfs, confidence
        self.factor = sr // CREPE_SR
        self.fir = firwin(63, 7000, fs=sr).astype(np.float32)  # anti-alias before keeping every 3rd sample
        self.zi = np.zeros(len(self.fir) - 1, np.float32)
        self.audio = np.zeros(len(session.mic) // self.factor + WINDOW, np.float32)
        self.filled = 0      # 16 kHz samples written to self.audio
        self.seen = 0        # device samples received so far
        self.pos0 = None     # song sample of the first mic sample
        self.next_frame = 0
        self.times, self.midi = [], []  # append-only; the UI reads them
        self.now = session.start / sr - latency
        self.now_wall = time.perf_counter()
        self.done = False

    def warm_up(self):
        """The first CREPE call loads the model (~1 s); do it before the music starts."""
        self._crepe(np.zeros(WINDOW + HOP, np.float32))

    def _crepe(self, audio):
        f0, periodicity = torchcrepe.predict(
            torch.from_numpy(audio)[None], CREPE_SR, HOP, 50, 1100, "tiny",
            decoder=torchcrepe.decode.weighted_argmax, return_periodicity=True, device="cuda", pad=False,
        )
        return f0[0].cpu().numpy(), periodicity[0].cpu().numpy()

    def run(self):
        while not self.done:
            try:
                blocks = [self.session.live.get(timeout=0.05)]
            except queue.Empty:
                continue
            while True:  # take everything queued so the GPU sees one batch
                try:
                    blocks.append(self.session.live.get_nowait())
                except queue.Empty:
                    break
            if self.pos0 is None:
                self.pos0 = blocks[0][0]
            x = np.concatenate([b for _, b in blocks])
            self.now = (blocks[-1][0] + len(blocks[-1][1])) / self.sr - self.latency
            self.now_wall = time.perf_counter()
            self._ingest(x)

    def _ingest(self, x):
        y, self.zi = lfilter(self.fir, 1, x, zi=self.zi)
        first = (-self.seen) % self.factor  # keep samples whose global index is a multiple of factor
        self.seen += len(x)
        down = y[first :: self.factor].astype(np.float32)
        n = min(len(down), len(self.audio) - self.filled)
        self.audio[self.filled : self.filled + n] = down[:n]
        self.filled += n

        frames = (self.filled - WINDOW) // HOP + 1 - self.next_frame
        if frames <= 0:
            return
        a = self.next_frame * HOP
        chunk = self.audio[a : a + WINDOW + (frames - 1) * HOP]
        f0, periodicity = self._crepe(chunk)
        for i in range(len(f0)):
            k = self.next_frame + i
            frame = self.audio[k * HOP : k * HOP + WINDOW]
            loud = 10 * np.log10(np.mean(frame**2) + 1e-12) > self.floor
            center = self.pos0 + (k * HOP + WINDOW // 2) * self.factor
            self.times.append(center / self.sr - self.latency)
            self.midi.append(float(hz_to_midi(f0[i])) if loud and periodicity[i] > self.confidence else np.nan)
        self.next_frame += len(f0)


class LiveView(QtWidgets.QWidget):
    def __init__(self, song, session, tracker, audio_thread, lyrics=None, show_lyrics=True):
        super().__init__()
        self.song, self.session, self.tracker, self.audio_thread = song, session, tracker, audio_thread
        self.lyrics, self.show_lyrics, self.tap_mode = lyrics, show_lyrics, False
        self.now = 0.0
        self.setWindowTitle(f"singfix · {song['folder'].name} · {song['key']['name']}")
        self.resize(1400, 650)
        self.setStyleSheet("background:#0f1420; color:#e8ecf3;")

        self.header = QtWidgets.QLabel("Get ready…")
        self.header.setTextFormat(QtCore.Qt.RichText)
        self.header.setFont(QtGui.QFont("Segoe UI", 18))
        self.plot = pg.PlotWidget(background="#0f1420")
        self.plot.setFocusPolicy(QtCore.Qt.NoFocus)  # keys go to the window
        self.line_now = QtWidgets.QLabel()
        self.line_now.setFont(QtGui.QFont("Segoe UI", 26, QtGui.QFont.DemiBold))
        self.line_next = QtWidgets.QLabel()
        self.line_next.setFont(QtGui.QFont("Segoe UI", 18))
        self.line_next.setStyleSheet(f"color:{GREY};")
        for label in (self.line_now, self.line_next):
            label.setAlignment(QtCore.Qt.AlignCenter)
        self.panel = QtWidgets.QListWidget()
        self.panel.setFixedWidth(420)
        self.panel.setFocusPolicy(QtCore.Qt.NoFocus)
        self.panel.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.panel.setWordWrap(True)
        self.panel.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.panel.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)  # the wheel still scrolls
        self.panel.setFont(QtGui.QFont("Segoe UI", 14))
        self.panel.setStyleSheet("QListWidget { border:none; background:#141a28; color:#aab2c5; padding:8px; }")
        self.panel_rows, self.panel_line = [], None  # list row of each lyric line; line highlighted now
        if lyrics is not None:
            for words, new_stanza in zip(lyrics.words, lyrics.breaks):
                if new_stanza:
                    self.panel.addItem("")
                self.panel_rows.append(self.panel.count())
                self.panel.addItem(words)
        self.follow = QtWidgets.QPushButton("Follow")
        self.follow.setCheckable(True)
        self.follow.setChecked(lyrics is not None and lyrics.follow)
        self.follow.setFocusPolicy(QtCore.Qt.NoFocus)  # Space is for tapping
        self.follow.setStyleSheet("QPushButton { background:#1c2436; border:1px solid #2c3650; padding:4px 12px; }"
                                  "QPushButton:checked { background:#2c3e66; color:#ffffff; }")
        self.follow.toggled.connect(self.set_follow)
        self.side = QtWidgets.QWidget()
        side = QtWidgets.QVBoxLayout(self.side)
        side.setContentsMargins(0, 0, 0, 0)
        side.addWidget(self.follow, 0, QtCore.Qt.AlignRight)
        side.addWidget(self.panel, 1)
        body = QtWidgets.QHBoxLayout()
        body.addWidget(self.plot, 1)
        body.addWidget(self.side)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.header)
        layout.addLayout(body, 1)
        layout.addWidget(self.line_now)
        layout.addWidget(self.line_next)
        self.update_lyrics_visible()

        self.ref_t = song["times"]
        self.ref_midi = hz_to_midi(song["f0"])
        notes = song["notes"]
        self.starts = np.array([n["start"] for n in notes])
        self.ends = np.array([n["end"] for n in notes])
        self.pitches = np.array([n["midi"] for n in notes], float)
        lo, hi = int(self.pitches.min()) - 2, int(self.pitches.max()) + 2

        p = self.plot
        p.setMouseEnabled(False, False)
        p.hideButtons()
        p.setYRange(lo, hi, padding=0)
        p.getAxis("bottom").setStyle(showValues=False)
        scale = [m for m in range(lo, hi + 1) if m % 12 in song["key"]["scale"]]
        p.getAxis("left").setTicks([[(m, note_name(m)) for m in scale]])
        tonic = [m for m in scale if m % 12 == song["key"]["tonic"]]
        bands = pg.BarGraphItem(
            x0=[-10] * len(scale), width=[1e5] * len(scale), y0=[m - 0.5 for m in scale], height=[1] * len(scale),
            brushes=[pg.mkBrush("#22314d" if m in tonic else "#18202f") for m in scale], pens=[pg.mkPen(None)] * len(scale),
        )
        p.addItem(bands)
        p.addItem(pg.PlotDataItem(self.ref_t, self.ref_midi, pen=pg.mkPen("#5c6680", width=1), connect="finite"))
        self.bar_colors = [BAR] * len(notes)
        self.bars = pg.BarGraphItem(x0=self.starts, width=self.ends - self.starts, y0=self.pitches - 0.4, height=0.8,
                                    brushes=[pg.mkBrush(c) for c in self.bar_colors], pens=[pg.mkPen(None)] * len(notes))
        p.addItem(self.bars)
        self.now_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#ffffff", width=2))
        p.addItem(self.now_line)
        self.voice = pg.ScatterPlotItem(size=7, pen=pg.mkPen("#0f1420", width=1))
        p.addItem(self.voice)
        self.brushes = {c: pg.mkBrush(c) for c in (GREEN, YELLOW, RED, GREY)}

        self.read = 0  # tracker points consumed
        self.vt, self.vm, self.vy, self.vc = [], [], [], []  # time, raw midi, folded midi, colour
        self.credit_sum, self.credit_n = 0.0, 0
        self.judged = np.zeros(len(notes), bool)
        self.started = False  # set once audio is running; until then the window waits
        self.hits = self.total = 0

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

    def keyPressEvent(self, event):
        key, ly = event.key(), self.lyrics
        if key == QtCore.Qt.Key_Escape:
            self.session.stop_requested = True
        elif ly is None:
            return
        elif key == QtCore.Qt.Key_L:
            ly.toggle()
            self.update_lyrics_visible()
        elif key == QtCore.Qt.Key_P:
            ly.toggle_panel()
            self.update_lyrics_visible()
        elif key == QtCore.Qt.Key_T:
            self.tap_mode = not self.tap_mode
            self.update_lyrics_visible()
        elif key == QtCore.Qt.Key_Space and self.tap_mode:
            ly.tap(self.now)
        elif key == QtCore.Qt.Key_Backspace and self.tap_mode:
            ly.undo()
        elif key in (QtCore.Qt.Key_BracketLeft, QtCore.Qt.Key_BracketRight):
            ly.nudge(-0.1 if key == QtCore.Qt.Key_BracketLeft else 0.1)

    def update_lyrics_visible(self):
        """Tap-sync always uses the strip; otherwise it's the strip or the panel, or neither when hidden."""
        ly = self.lyrics
        on = ly is not None and self.show_lyrics and ly.enabled
        strip = ly is not None and (self.tap_mode or (on and not ly.panel))
        self.line_now.setVisible(strip)
        self.line_next.setVisible(strip)
        self.side.setVisible(on and not self.tap_mode and ly.panel)

    def set_follow(self, on):
        self.lyrics.set_follow(on)
        if not on:
            self.highlight(self.panel_line, False)
            self.panel_line = None

    def highlight(self, line, on):
        if line is None:
            return
        item = self.panel.item(self.panel_rows[line])
        item.setForeground(QtGui.QColor("#ffffff" if on else "#aab2c5"))
        font = item.font()
        font.setBold(on)
        item.setFont(font)

    def update_panel(self, now):
        i = self.lyrics.current(now)
        if not self.lyrics.follow or i == self.panel_line:
            return
        self.highlight(self.panel_line, False)
        self.highlight(i, True)
        if i is not None:
            self.panel.scrollToItem(self.panel.item(self.panel_rows[i]), QtWidgets.QAbstractItemView.PositionAtCenter)
        self.panel_line = i

    def update_lyrics(self, now):
        if self.side.isVisible():
            self.update_panel(now)
        if not self.line_now.isVisible():
            return
        if self.tap_mode:
            done, up = self.lyrics.tapping(now)
            self.line_now.setText(done or "")
            self.line_next.setText(f"Space ▸ {up}" if up else "all lines tapped")
        else:
            current, upcoming = self.lyrics.at(now)
            self.line_now.setText(current or "♪")
            self.line_next.setText(upcoming or "")

    def closeEvent(self, event):
        self.session.stop_requested = True
        super().closeEvent(event)

    def target(self, t):
        """(line pitch, bar pitch) of the singer at song time t; NaN where there's none."""
        line = np.interp(t, self.ref_t, self.ref_midi)
        i = np.searchsorted(self.starts, t, "right") - 1
        bar = self.pitches[i] if i >= 0 and t < self.ends[i] else np.nan
        return line, bar

    def tick(self):
        tr = self.tracker
        now = tr.now + min(time.perf_counter() - tr.now_wall, 0.1)
        self.now = now
        self.plot.setXRange(now - PAST, now + FUTURE, padding=0)
        self.now_line.setPos(now)

        status = None
        n = len(tr.midi)
        for t, m in zip(tr.times[self.read : n], tr.midi[self.read : n]):
            if np.isnan(m):
                continue
            line, bar = self.target(t)
            goal = bar if not np.isnan(bar) else line
            if np.isnan(goal):
                folded, colour, err = m, GREY, np.nan
            else:
                folded = m + 12 * round((goal - m) / 12)  # show you in the singer's octave
                err = np.nanmin([abs(fold((m - line) * 100)), abs(fold((m - bar) * 100))])
                c = float(credit(err))
                colour = GREEN if c >= 0.99 else YELLOW if c > 0 else RED
                self.credit_sum += c
                self.credit_n += 1
            self.vt.append(t); self.vm.append(m); self.vy.append(folded); self.vc.append(colour)
            status = (m, goal, folded, err)
        self.read = n

        keep = next((i for i, t in enumerate(self.vt) if t >= now - PAST - 1), len(self.vt))
        del self.vt[:keep], self.vm[:keep], self.vy[:keep], self.vc[:keep]
        self.voice.setData(self.vt, self.vy, brush=[self.brushes[c] for c in self.vc])
        self.judge_notes(now)
        self.update_header(now, status)
        self.update_lyrics(now)

        if self.started and not self.audio_thread.is_alive():
            self.timer.stop()
            self.close()

    def judge_notes(self, now):
        """Once a note has passed, colour it: hit if you were within 100 cents for half of it."""
        changed = False
        times = np.array(self.tracker.times[: self.read])
        midi = np.array(self.tracker.midi[: self.read])
        start_s = self.session.start / self.session.sr
        for i in np.flatnonzero(~self.judged & (self.ends + SLACK < now)):
            self.judged[i] = True
            if self.ends[i] - self.starts[i] < MIN_NOTE or self.starts[i] < start_s:
                continue
            # Same ±80 ms timing slack as the offline scorer: count frames a little either side.
            m = midi[(times >= self.starts[i] + SCOOP - SLACK) & (times < self.ends[i] + SLACK)]
            expected = (self.ends[i] - self.starts[i] - SCOOP) / (HOP / CREPE_SR)
            hit = np.sum(np.abs(fold((m - self.pitches[i]) * 100)) < HIT_CENTS) >= 0.5 * expected
            self.bar_colors[i] = BAR_HIT if hit else BAR_MISS
            self.hits += hit
            self.total += 1
            changed = True
        if changed:
            self.bars.setOpts(brushes=[pg.mkBrush(c) for c in self.bar_colors])

    def update_header(self, now, status):
        acc = f"{self.credit_sum / self.credit_n:.0%}" if self.credit_n else "–"
        clock = f"{int(now // 60)}:{int(now % 60):02d}"
        if status is not None:
            m, goal, folded, err = status
            if np.isnan(goal):
                you = f"<span style='color:{GREY}'>you {note_name(m)}</span>"
            else:
                cents = (folded - goal) * 100
                colour = GREEN if abs(cents) <= 50 else YELLOW if abs(cents) <= 150 else RED
                arrow = "♯ sharp" if cents > 20 else "♭ flat" if cents < -20 else "on it"
                you = (f"you <b>{note_name(m)}</b> → target <b>{note_name(goal)}</b> "
                       f"<span style='color:{colour}'>{cents:+.0f}c {arrow}</span>")
            self.last_you = you
        you = getattr(self, "last_you", "sing along…")
        tap = f" &nbsp;·&nbsp; <span style='color:{YELLOW}'>TAP SYNC</span>" if self.tap_mode else ""
        self.header.setText(f"{you} &nbsp;&nbsp;|&nbsp;&nbsp; accuracy <b>{acc}</b> &nbsp;·&nbsp; "
                            f"notes {self.hits}/{self.total} &nbsp;·&nbsp; {clock}{tap}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--guide", type=float, default=0.3, help="original vocal level, 0..1")
    ap.add_argument("--transpose", default="0", help="semitones (e.g. -4) or 'auto'")
    ap.add_argument("--volume", type=float, default=1.0)
    ap.add_argument("--floor", type=float, default=-38)
    ap.add_argument("--lyrics", metavar="QUERY", help='fetch lyrics from LRCLIB, e.g. "fields of gold sting"')
    ap.add_argument("--lyrics-pick", type=int, default=None, help="which listed lyrics version to use")
    ap.add_argument("--no-lyrics", action="store_true", help="hide lyrics this run")
    args = ap.parse_args()

    cfg = load_config()
    latency = cfg.get("latency_s") or 0.0
    sr = cfg["samplerate"]
    song = load_song(args.song, transpose=pick_transpose(args.song, args.transpose))
    if args.lyrics:
        lyrics_lib.fetch(song, args.lyrics, args.lyrics_pick)
    lyrics = lyrics_lib.load(song["folder"])
    audio = build_track(song, sr, args.guide)
    start = int(args.start * sr)
    end = None if args.end is None else int(args.end * sr)
    session = Session(audio, sr, cfg, start=start, end=end, volume=args.volume, live=True)
    tracker = PitchTracker(session, sr, latency, args.floor)
    tracker.warm_up()

    app = QtWidgets.QApplication([])
    pg.setConfigOptions(antialias=True)
    audio_thread = threading.Thread(target=session.run, daemon=True)
    view = LiveView(song, session, tracker, audio_thread, lyrics, show_lyrics=not args.no_lyrics)
    view.show()

    def go():
        tracker.start()
        audio_thread.start()
        view.started = True
    QtCore.QTimer.singleShot(1000, go)  # a beat to look at the first notes
    app.exec()

    session.stop_requested = True
    audio_thread.join(timeout=2)
    tracker.done = True
    if lyrics is not None:
        lyrics.save()
    print(f"Live accuracy {view.credit_sum / max(view.credit_n, 1):.0%}, notes {view.hits}/{view.total}")
    finish_take(song, session.mic[: session.recorded], sr, args.start, latency, args.guide, args.floor)


if __name__ == "__main__":
    main()
