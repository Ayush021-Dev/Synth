# mix_vocals.py
# Mix vocals_synth.wav with test_instrumental.wav into final_mix.wav
# Usage:
# python mix_vocals.py test_instrumental.wav vocals_synth.wav final_mix.wav --voc-vol 0.9 --inst-vol 1.0

import sys
import argparse
import numpy as np
import soundfile as sf
import librosa

def align_and_mix(inst_path, vox_path, out_path, sr=32000, inst_vol=1.0, vox_vol=0.9, offset=0.0):
    # load
    inst, sr1 = librosa.load(inst_path, sr=sr, mono=True)
    vox, sr2 = librosa.load(vox_path, sr=sr, mono=True)
    # apply offset to vocals (seconds)
    if offset > 0:
        pad = int(round(offset * sr))
        vox = np.concatenate([np.zeros(pad, dtype=vox.dtype), vox])
    # pad shorter to longer
    L = max(len(inst), len(vox))
    if len(inst) < L:
        inst = np.pad(inst, (0, L - len(inst)))
    if len(vox) < L:
        vox = np.pad(vox, (0, L - len(vox)))
    # mix with volumes
    mix = inst * inst_vol + vox * vox_vol
    # normalize to avoid clipping
    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix = mix / peak
    sf.write(out_path, mix, sr)
    print(f"Wrote {out_path} (sr={sr}) — inst_vol={inst_vol}, vox_vol={vox_vol}, offset={offset}s")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("inst", help="instrumental file (wav)")
    p.add_argument("vox", help="vocals file (wav)")
    p.add_argument("out", help="output mixed wav")
    p.add_argument("--sr", type=int, default=32000)
    p.add_argument("--inst-vol", type=float, default=1.0)
    p.add_argument("--vox-vol", type=float, default=0.9)
    p.add_argument("--offset", type=float, default=0.0, help="delay vocals by seconds before mixing")
    args = p.parse_args()
    align_and_mix(args.inst, args.vox, args.out, sr=args.sr, inst_vol=args.inst_vol, vox_vol=args.vox_vol, offset=args.offset)
