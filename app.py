import torch
import torchaudio
import os
import io
import json
import traceback # Already in your code, good for debugging
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from audiocraft.models import MusicGen

# --- Initialization ---
app = Flask(_name_)
# IMPORTANT: Use resources="/" to allow CORS for all routes
CORS(app, resources={r"/": {"origins": ""}}) 

# Global variables for model and device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = None
SAMPLING_RATE = 32000
DEFAULT_DURATION = 30 # Max duration for 'small' model

def load_musicgen_model():
    """Load the model once when the Flask server starts."""
    global MODEL
    if MODEL is not None:
        return

    print("="*50)
    print("MUSICGEN LOADING...")
    print("Device:", DEVICE)
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    try:
        # Load the model globally, passing the device
        MODEL = MusicGen.get_pretrained("small", device=DEVICE)
        print("MusicGen model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load MusicGen model: {e}")
        MODEL = None
    print("="*50)

# Load the model immediately when the script runs
load_musicgen_model()

# --- Music Generation Function ---

def generate_music_track(genre_prompt: str, lyrics: str, duration: int = DEFAULT_DURATION, seed: int = 42):
    """
    Generates music using the loaded MusicGen model based on the combined prompt.
    Returns the audio data as a BytesIO object.
    """
    if MODEL is None:
        # Raise a specific error that can be caught as 503
        raise RuntimeError("MusicGen model is not loaded. Cannot generate audio.")

    # 1. Combine Genre and Lyrics into a single prompt
    if lyrics and lyrics.strip():
        # Combine them for maximum influence
        full_prompt = f"Instrumental in the style of {genre_prompt}. The atmosphere should match a song with these lyrics: {lyrics.strip()}"
    else:
        full_prompt = f"Instrumental in the style of {genre_prompt}"
        
    print(f"Full Prompt: {full_prompt}")

    # 2. Set generation parameters
    torch.manual_seed(seed)
    duration = min(duration, DEFAULT_DURATION) 

    MODEL.set_generation_params(duration=duration, use_sampling=True, temperature=1.0)

    # 3. Generate the audio
    print(f"Generating instrumental audio for {duration}s...")
    # wav is (batch=1, channels, samples)
    wav = MODEL.generate([full_prompt]) 
    waveform = wav[0].cpu()

    # 4. In-Memory WAV File Creation
    buffer = io.BytesIO()
    # Save the waveform directly to the in-memory buffer
    torchaudio.save(buffer, waveform, SAMPLING_RATE, format="wav")
    buffer.seek(0)
    
    print("Generation complete and saved to memory buffer.")
    return buffer

# --- Flask Routes ---

@app.route('/generate', methods=['POST'])
def generate_instrumental():
    try:
        # Log incoming request
        print("="*50)
        print("Received request to /generate")

        # 1. Check Model Status first
        if MODEL is None:
             print("ERROR: Model not loaded, returning 503")
             return jsonify({"error": "Music generation service is currently unavailable. Model failed to load."}), 503

        # 2. Get and validate data from request
        data = request.get_json()
        print(f"Request data: {data}")
        
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
        
        lyrics = data.get('lyrics', '')
        genre = data.get('genre', 'a simple melody') # Use a non-empty default for generation
        duration = data.get('duration', DEFAULT_DURATION) # Optionally support duration from frontend
        
        if not genre.strip():
             return jsonify({"error": "Genre prompt cannot be empty."}), 400

        # Generate a seed based on inputs for consistency/logging
        seed = hash(f"{genre}{lyrics}") % 10000 

        print(f"Genre: {genre}")
        print(f"Lyrics length: {len(lyrics)} characters")
        print(f"Lyrics preview: {lyrics[:50]}...")
        
        # 3. Call the generation function
        wav_buffer = generate_music_track(genre, lyrics, duration, seed)
        
        print(f"Sending audio buffer as WAV...")
        
        # 4. Return the WAV file from the in-memory buffer
        return send_file(
            wav_buffer,
            mimetype='audio/wav',
            as_attachment=False,
            download_name=f'song-instrumental-{seed}.wav'
        )
        
    except RuntimeError as e:
        # Catch errors specifically related to the generation process (e.g., CUDA issues)
        print("="*50)
        print("RUNTIME ERROR (Generation failed):")
        print(traceback.format_exc())
        print("="*50)
        return jsonify({"error": f"Generation failed: {str(e)}"}), 503
        
    except Exception as e:
        print("="*50)
        print("GENERAL ERROR occurred:")
        print(traceback.format_exc())
        print("="*50)
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/health', methods=['GET'])
def health_check():
    model_status = "ok" if MODEL is not None else "failed"
    return jsonify({
        "status": "ok", 
        "message": "Flask server is running",
        "model_status": model_status,
        "device": DEVICE
    })

if _name_ == '_main_':
    # NOTE: os.makedirs('output', exist_ok=True) is no longer strictly needed 
    # since we use in-memory buffers, but keeping it doesn't hurt.
    
    print("="*50)
    print("Flask server starting...")
    print("="*50)
    
    # IMPORTANT: Set debug=False and threaded=False for stability when using CUDA/ML models
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)