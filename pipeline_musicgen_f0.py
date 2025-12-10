import torch
import torchaudio
import librosa
import numpy as np
import crepe
from audiocraft.models import MusicGen

def generate_instrumental(prompt: str, duration: int, out_wav: str = "instrumental.wav"):
    print("Loading MusicGen-small...")
    model = MusicGen.get_pretrained("small")
    model.set_generation_params(duration=duration)

    print("Generating instrumental...")
    wav = model.generate([prompt])
    torchaudio.save(out_wav, wav[0].cpu(), 32000)
    print(f"Saved instrumental to {out_wav}")
    return out_wav

def extract_f0(audio_path: str, f0_out="melody_f0.npy", t_out="melody_time.npy"):
    print(f"Loading audio for F0: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio = audio.astype(np.float32)

    print("Running CREPE...")
    time, frequency, confidence, _ = crepe.predict(
        audio, sr, viterbi=True, step_size=10
    )

    np.save(f0_out, frequency)
    np.save(t_out, time)
    print(f"Saved F0 to {f0_out}, time to {t_out}")
    return f0_out, t_out

if __name__ == "__main__":
    prompt = "emotional industrial rock, electric guitar, drums, bass, evolving melody"
    instr = generate_instrumental(prompt, duration=10, out_wav="test_instrumental.wav")
    extract_f0(instr, "melody_f0.npy", "melody_time.npy")
