"""
extract_f0.py

Runs CREPE on an audio file and saves:
 - melody_f0.npy     : F0 in Hz (NaN for unvoiced frames, after confidence thresholding)
 - melody_conf.npy   : CREPE confidence [0..1]
 - melody_time.npy   : timestamps (seconds)

Default input path points to the Demucs no_vocals stem you created:
  separated/htdemucs/test_instrumental/no_vocals.wav

Usage examples:
  python extract_f0.py                              # uses default no_vocals path
  python extract_f0.py separated/htdemucs/test_instrumental/vocals.wav
  python extract_f0.py --conf-thresh 0.5 --model tiny path/to/file.wav
"""

import argparse
from pathlib import Path
import numpy as np
import crepe
import librosa
import sys

DEFAULT_PATH = Path("separated/htdemucs/test_instrumental/no_vocals.wav")
F0_OUT = Path("melody_f0.npy")
CONF_OUT = Path("melody_conf.npy")
T_OUT = Path("melody_time.npy")

def extract_f0(path_in: Path,
               sr: int = 16000,
               step_size: int = 10,
               conf_thresh: float = 0.45,
               model_capacity: str = "full"):
    path_in = Path(path_in)
    if not path_in.exists():
        raise SystemExit(f"Input file not found: {path_in}")

    print(f"Loading audio: {path_in}")
    audio, sr_loaded = librosa.load(str(path_in), sr=sr, mono=True)
    print(f"Audio loaded — duration: {len(audio)/sr:.2f}s, sr: {sr_loaded}, frames: {audio.shape[0]}")

    audio = audio.astype(np.float32)

    print("Running CREPE (this can take a while for long files)...")
    time, frequency, confidence, activation = crepe.predict(
        audio,
        sr,
        viterbi=True,
        step_size=step_size,
        model_capacity=model_capacity
    )

    print("Masking low-confidence frames (confidence < {:.3f})".format(conf_thresh))
    frequency_masked = np.where(confidence >= conf_thresh, frequency, np.nan)

    print("Saving outputs:")
    np.save(F0_OUT, frequency_masked)
    np.save(CONF_OUT, confidence)
    np.save(T_OUT, time)

    print(f"Saved {F0_OUT} shape={frequency_masked.shape}")
    print(f"Saved {CONF_OUT} shape={confidence.shape}")
    print(f"Saved {T_OUT} shape={time.shape}")
    return F0_OUT, CONF_OUT, T_OUT

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", nargs="?", default=str(DEFAULT_PATH),
                   help="Input audio file (wav). Default: separated/htdemucs/test_instrumental/no_vocals.wav")
    p.add_argument("--sr", type=int, default=16000, help="Sample rate to load audio (Hz)")
    p.add_argument("--step-size", type=int, default=10, help="CREPE step size (ms)")
    p.add_argument("--conf-thresh", type=float, default=0.45, help="Confidence threshold (0..1)")
    p.add_argument("--model", default="full", choices=["tiny", "full"], help="CREPE model capacity")
    args = p.parse_args()

    extract_f0(args.input, sr=args.sr, step_size=args.step_size,
               conf_thresh=args.conf_thresh, model_capacity=args.model)
