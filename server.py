"""
FastAPI Server for MusicGen Audio Generation
Connects with Next.js frontend over network for instrumental music generation
"""

import torch
import torchaudio
from audiocraft.models import MusicGen
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import os
import tempfile
import io
import logging
from typing import Optional
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Audio Generation API",
    description="FastAPI server for MusicGen instrumental generation",
    version="1.0.0"
)

# CORS Configuration - Allow your Next.js frontend IP
# Replace with actual frontend PC IP if needed
ALLOWED_ORIGINS = [
    "http://localhost:3000",           # Local development
    "http://127.0.0.1:3000",
    "http://192.168.*",                # Your network IPs
    "http://10.*",
    "http://172.16.*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,    # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    max_age=3600,
    allow_headers=["*"],
)

# Pydantic models
class GenerateRequest(BaseModel):
    prompt: Optional[str] = "classical with synths and drums, energetic and uplifting"
    duration: Optional[int] = 30
    temperature: Optional[float] = 1.0
    seed: Optional[int] = 0
    use_sampling: Optional[bool] = True

class HealthResponse(BaseModel):
    status: str
    device: str
    gpu_info: Optional[str]
    model_loaded: bool

# Global model cache (loaded once to save memory)
_model_cache = None
_device = None

def get_device():
    """Get available compute device"""
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device

def get_model():
    """Get or load MusicGen model (cached)"""
    global _model_cache
    
    if _model_cache is None:
        device = get_device()
        logger.info(f"Loading MusicGen-small on {device}...")
        try:
            _model_cache = MusicGen.get_pretrained("small", device=device)
            logger.info("✓ Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    return _model_cache

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    device = get_device()
    logger.info(f"Starting server on device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Pre-load model
    try:
        get_model()
    except Exception as e:
        logger.error(f"Failed to pre-load model: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and model status"""
    device = get_device()
    gpu_info = None
    
    if device == "cuda":
        gpu_info = torch.cuda.get_device_name(0)
    
    return HealthResponse(
        status="healthy",
        device=device,
        gpu_info=gpu_info,
        model_loaded=_model_cache is not None
    )

@app.post("/generate")
async def generate_instrumental(request: GenerateRequest):
    """
    Generate instrumental music
    
    Args:
        prompt: Music description (e.g., "upbeat electronic dance music")
        duration: Length in seconds (default: 30)
        temperature: Sampling temperature (0.0-2.0, default: 1.0)
        seed: Random seed for reproducibility
        use_sampling: Use sampling (True) or greedy decoding (False)
    
    Returns:
        WAV file as binary audio data
    """
    try:
        device = get_device()
        logger.info(f"Generating: {request.prompt} ({request.duration}s on {device})")
        
        # Set random seed for reproducibility
        torch.manual_seed(request.seed)
        
        # Get model
        model = get_model()
        
        # Set generation parameters
        model.set_generation_params(
            duration=request.duration,
            use_sampling=request.use_sampling,
            temperature=request.temperature
        )
        
        # Generate audio
        logger.info("Generating audio...")
        with torch.no_grad():
            wav = model.generate([request.prompt])  # (batch, channels, samples)
        
        # Extract waveform
        waveform = wav[0].cpu()  # Remove batch dimension
        sr = 32000
        
        logger.info(f"✓ Generated {waveform.shape[1] / sr:.2f}s of audio")
        
        # Save to temporary file
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"generated_{request.seed}.wav")
        
        torchaudio.save(temp_file, waveform, sr)
        logger.info(f"Saved to {temp_file}")
        
        # Return as download
        return FileResponse(
            path=temp_file,
            media_type="audio/wav",
            filename=f"instrumental_{request.seed}.wav",
            headers={"Content-Disposition": "attachment; filename=instrumental.wav"}
        )
    
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio generation failed: {str(e)}"
        )

@app.post("/generate-stream")
async def generate_instrumental_stream(request: GenerateRequest):
    """
    Generate instrumental music and stream response
    Useful for real-time feedback without waiting for full file save
    """
    async def audio_generator():
        try:
            device = get_device()
            torch.manual_seed(request.seed)
            
            model = get_model()
            model.set_generation_params(
                duration=request.duration,
                use_sampling=request.use_sampling,
                temperature=request.temperature
            )
            
            logger.info(f"Streaming: {request.prompt}")
            
            with torch.no_grad():
                wav = model.generate([request.prompt])
            
            waveform = wav[0].cpu()
            sr = 32000
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            torchaudio.save(buffer, waveform, sr, format="wav")
            
            # Yield audio data in chunks
            buffer.seek(0)
            while True:
                chunk = buffer.read(1024 * 64)  # 64KB chunks
                if not chunk:
                    break
                yield chunk
            
            logger.info("✓ Stream completed")
        
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            yield f"Error: {str(e)}".encode()
    
    return StreamingResponse(
        audio_generator(),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=instrumental.wav"}
    )

@app.get("/")
async def root():
    """API documentation"""
    return {
        "name": "Audio Generation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health - Check server health",
            "generate": "POST /generate - Generate and download instrumental",
            "generate-stream": "POST /generate-stream - Generate and stream instrumental",
            "docs": "GET /docs - Interactive API documentation (Swagger UI)",
            "redoc": "GET /redoc - ReDoc documentation"
        },
        "example_request": {
            "prompt": "upbeat electronic dance music with strong bass",
            "duration": 30,
            "temperature": 1.0,
            "seed": 42
        }
    }

@app.post("/batch-generate")
async def batch_generate(prompts: list[str], duration: int = 30):
    """
    Generate multiple instrumentals (for future batch processing)
    """
    try:
        device = get_device()
        model = get_model()
        model.set_generation_params(duration=duration)
        
        logger.info(f"Batch generating {len(prompts)} tracks...")
        
        with torch.no_grad():
            wavs = model.generate(prompts)
        
        results = []
        for i, wav in enumerate(wavs):
            waveform = wav.cpu()
            sr = 32000
            
            temp_file = os.path.join(
                tempfile.gettempdir(), 
                f"batch_track_{i}.wav"
            )
            torchaudio.save(temp_file, waveform, sr)
            results.append(temp_file)
        
        return {
            "status": "success",
            "generated": len(results),
            "files": results
        }
    
    except Exception as e:
        logger.error(f"Batch generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get device info
    device = get_device()
    logger.info(f"🚀 Starting server on GPU: {torch.cuda.is_available()}")
    
    # Run server
    # For network access: use 0.0.0.0 instead of 127.0.0.1
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all network interfaces
        port=5000,
        log_level="info"
    )
