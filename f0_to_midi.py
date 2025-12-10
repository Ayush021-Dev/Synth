"""
f0_to_midi.py

Convert CREPE F0 (melody_f0.npy + melody_time.npy) into a monophonic MIDI file (melody.mid).

Default expects the files created by extract_f0.py:
  melody_f0.npy
  melody_time.npy

Usage examples:
  python f0_to_midi.py
  python f0_to_midi.py --f0 melody_f0.npy --time melody_time.npy --out melody.mid --min-freq 80 --min-len 0.06 --max-gap 0.06
  python f0_to_midi.py --smooth 5   # apply median smoothing kernel=5 before conversion
"""

import argparse
from pathlib import Path
import numpy as np
import pretty_midi

# Optional: median filter
try:
    from scipy.signal import medfilt
    _HAVE_MEDFILT = True
except Exception:
    _HAVE_MEDFILT = False

def forward_fill_nan(arr):
    """Replace NaNs with the last seen finite value (forward fill)."""
    out = arr.copy()
    last = np.nan
    for i in range(len(out)):
        if np.isfinite(out[i]):
            last = out[i]
        else:
            out[i] = last
    return out

def load_f0(f0_path: Path, time_path: Path, smooth: int = 1):
    f0 = np.load(f0_path)
    times = np.load(time_path)
    if smooth and smooth > 1:
        if not _HAVE_MEDFILT:
            raise SystemExit("scipy not available for median filtering. Install scipy or set --smooth 1.")
        # replace NaNs (temporary) then median filter, then re-mask where original was NaN
        original_nan = ~np.isfinite(f0)
        temp = forward_fill_nan(f0)
        temp = np.nan_to_num(temp, nan=0.0)
        temp_s = medfilt(temp, kernel_size=smooth)
        # put NaNs back where original was NaN
        temp_s[original_nan] = np.nan
        f0 = temp_s
    return f0, times

def f0_to_midi(f0_path: Path, time_path: Path, out_midi: Path,
               min_freq: float = 80.0, max_gap: float = 0.06, min_note_len: float = 0.06,
               program: int = 0, smooth: int = 1):
    f0, times = load_f0(f0_path, time_path, smooth=smooth)
    if len(times) < 2:
        raise SystemExit("Time array too short")

    dt = times[1] - times[0]
    voiced_mask = np.isfinite(f0) & (f0 >= min_freq)

    notes = []
    i = 0
    N = len(f0)
    while i < N:
        if not voiced_mask[i]:
            i += 1
            continue
        start = i
        end = i
        while end + 1 < N:
            if voiced_mask[end + 1]:
                end += 1
                continue
            # find next voiced frame index
            j = end + 1
            while j < N and not voiced_mask[j]:
                j += 1
            if j < N:
                gap = times[j] - times[end]
            else:
                gap = float('inf')
            if gap <= max_gap:
                end = j
                continue
            else:
                break
        t_start = times[start]
        t_end = times[end]
        dur = t_end - t_start + dt
        if dur >= min_note_len:
            pitch_hz = float(np.nanmedian(f0[start:end+1]))
            if not np.isfinite(pitch_hz) or pitch_hz <= 0:
                i = end + 1
                continue
            midi_note = int(round(69 + 12 * np.log2(pitch_hz / 440.0)))
            # clamp MIDI to 0..127
            midi_note = max(0, min(127, midi_note))
            notes.append((t_start, dur, midi_note))
        i = end + 1

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    for (start, dur, note) in notes:
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=int(note),
                                           start=float(start), end=float(start+dur)))
    pm.instruments.append(inst)
    pm.write(str(out_midi))
    print(f"Wrote {out_midi} notes: {len(notes)}")
    return out_midi, len(notes)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--f0", type=str, default="melody_f0.npy", help="Path to melody_f0.npy")
    p.add_argument("--time", type=str, default="melody_time.npy", help="Path to melody_time.npy")
    p.add_argument("--out", type=str, default="melody.mid", help="Output MIDI file")
    p.add_argument("--min-freq", type=float, default=80.0, help="Ignore frames below this freq (Hz)")
    p.add_argument("--max-gap", type=float, default=0.06, help="Bridge small unvoiced gaps <= seconds")
    p.add_argument("--min-note-len", type=float, default=0.06, help="Minimum note length to keep (sec)")
    p.add_argument("--program", type=int, default=0, help="MIDI program/instrument number (0 = piano)")
    p.add_argument("--smooth", type=int, default=1, help="Median smoothing kernel (odd integer >=1). 1 = no smoothing")
    args = p.parse_args()

    f0_to_midi(Path(args.f0), Path(args.time), Path(args.out),
               min_freq=args.min_freq, max_gap=args.max_gap, min_note_len=args.min_note_len,
               program=args.program, smooth=args.smooth)
