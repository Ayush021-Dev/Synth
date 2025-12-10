# generate_instrumental.py (fixed)
import torch
import torchaudio
from audiocraft.models import MusicGen

def generate_instrumental(duration=30, seed=0, output="test_instrumental.wav"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    torch.manual_seed(seed)

    print("Loading MusicGen-small...")
    # pass device here instead of calling .to()
    model = MusicGen.get_pretrained("small", device=device)

    # set generation params (duration in seconds)
    model.set_generation_params(duration=duration, use_sampling=True, temperature=1.0)

    prompt = "classical with synths and drums, energetic and uplifting"
    print(f"Generating audio for {duration}s...")
    wav = model.generate([prompt])  # returns (batch, channels, samples)

    waveform = wav[0].cpu()
    sr = 32000
    print("Saving to", output)
    torchaudio.save(output, waveform, sr)
    print("Saved", output)

if __name__ == "__main__":
    generate_instrumental(duration=30, seed=42)
