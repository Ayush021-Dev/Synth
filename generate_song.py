import subprocess
from pydub import AudioSegment

from pipeline_musicgen_f0 import generate_instrumental, extract_f0

def run_diffsinger(lyrics_txt: str, f0_path: str, out_vocals: str = "vocals.wav"):
    """
    Placeholder: adapt this to the actual DiffSinger inference command
    according to the repo you cloned.
    """
    # Example shape – you MUST change this to match the real script:
    cmd = [
        "python", "path/to/inference_script.py",
        "--lyrics", lyrics_txt,
        "--f0", f0_path,
        "--output", out_vocals,
        # plus whatever config / ckpt args DiffSinger needs
    ]
    print("Running DiffSinger:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_vocals

def mix_tracks(instr_path: str, vocal_path: str, out_path: str = "final_song.wav"):
    print("Mixing instrumental + vocals...")
    inst = AudioSegment.from_wav(instr_path)
    voc = AudioSegment.from_wav(vocal_path)

    voc = voc + 3  # boost vocals slightly
    mixed = inst.overlay(voc)
    mixed.export(out_path, format="wav")
    print("Saved final song to", out_path)
    return out_path

def generate_full_song(lyrics: str, genre_prompt: str, duration: int = 20):
    # 1. Save lyrics to file for the singing model
    lyrics_file = "lyrics.txt"
    with open(lyrics_file, "w", encoding="utf-8") as f:
        f.write(lyrics)

    # 2. Instrumental
    instrumental_path = generate_instrumental(genre_prompt, duration, out_wav="instrumental.wav")

    # 3. Melody (F0)
    f0_path, t_path = extract_f0(instrumental_path, "melody_f0.npy", "melody_time.npy")

    # 4. Vocals (you will wire this to DiffSinger)
    vocals_path = run_diffsinger(lyrics_file, f0_path, out_vocals="vocals.wav")

    # 5. Mix
    final_song = mix_tracks(instrumental_path, vocals_path, "final_song.wav")
    return final_song

if __name__ == "__main__":
    lyrics = """Your lyrics here..."""
    genre = "industrial rock, heavy guitars, big drums, 130 bpm, cinematic"
    generate_full_song(lyrics, genre, duration=20)
