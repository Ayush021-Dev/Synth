# make_ds_quick.py
# Quick converter: diffsinger_input.txt + melody_f0.npy -> ds.json (minimal .ds)
# Usage (from project root): python make_ds_quick.py

import json
import numpy as np
from pathlib import Path

# paths (adjust if needed)
PROJECT = Path(".").resolve()
INPUT = PROJECT / "diffsinger_input.txt"
F0_NPY = PROJECT / "melody_f0.npy"
T_NPY = PROJECT / "melody_time.npy"
OUT_DS = PROJECT / "ds_quick.json"

# parameters
F0_TIMESTEP = 0.01  # seconds (CREPE step size default 10 ms)

def read_alignments(fn):
    segments = []
    with open(fn, "r", encoding="utf8") as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 4:
                continue
            start = float(parts[0])
            dur = float(parts[1])
            midi = int(parts[2])
            lyric = parts[3]
            segments.append((start, dur, midi, lyric))
    return segments

def build_f0_for_segment(f0_arr, t_arr, seg_start, seg_end, step=F0_TIMESTEP):
    # sample at step from seg_start to seg_end (inclusive start, exclusive end)
    times = np.arange(seg_start, seg_end, step)
    if len(times) == 0:
        return [], step
    # for each time, find nearest index in t_arr and take f0 (nan stays nan)
    idx = np.searchsorted(t_arr, times, side="right") - 1
    idx = np.clip(idx, 0, len(t_arr)-1)
    vals = f0_arr[idx]
    # replace nan by 0 (model will treat low f0 as unvoiced)
    vals = [float(v) if np.isfinite(v) else 0.0 for v in vals]
    return vals, step

def main():
    if not INPUT.exists():
        print("Missing", INPUT)
        return
    if not F0_NPY.exists() or not T_NPY.exists():
        print("Missing melody_f0.npy or melody_time.npy")
        return

    f0 = np.load(F0_NPY)
    t = np.load(T_NPY)

    segs = read_alignments(INPUT)
    ds = []
    for i, (start, dur, midi, lyric) in enumerate(segs):
        seg_end = start + dur
        f0_seq, f0_timestep = build_f0_for_segment(f0, t, start, seg_end)
        # minimal ph_seq: treat the lyric as the single token (replace spaces with _)
        ph_seq = lyric.replace(" ", "_")
        # ph_dur: one phoneme whose duration equals the note duration
        ph_dur = "{:.6f}".format(dur)
        entry = {
            "id": f"seg_{i:04d}",
            "offset": start,
            "ph_seq": ph_seq,
            "ph_dur": ph_dur,
            "f0_seq": " ".join([f"{v:.3f}" for v in f0_seq]) if f0_seq else "0.0",
            "f0_timestep": f"{f0_timestep:.6f}"
        }
        ds.append(entry)

    with open(OUT_DS, "w", encoding="utf8") as f:
        json.dump(ds, f, indent=2, ensure_ascii=False)
    print("Wrote", OUT_DS)
    print("Segments:", len(ds))

if __name__ == "__main__":
    main()
