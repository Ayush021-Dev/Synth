# 🎶 Synth: AI-Powered Instrumental Song Generator

## 💡 Project Overview

**Synth** is a full-stack web application designed to generate custom instrumental music tracks. It combines a user-friendly Next.js frontend with a powerful Python Flask backend that leverages the **MusicGen** model for state-of-the-art audio composition. 

### Key Features:

* **Text-to-Music Generation:** Generates original audio based on descriptive text prompts (genre, tempo, instrumentation).
* **Lyrics-as-Influence:** Uses user-provided lyrics to influence the mood and atmosphere of the instrumental composition.
* **Seamless Caching:** Features a robust, deceptive caching layer that serves pre-generated WAV files instantly for popular or predefined genre requests, providing a faster user experience while simulating real-time AI generation in the backend logs.
* **Audio Mixing API:** The Next.js Route Handler is structured to call the Flask API for the instrumental and then mix it with a separate vocal track before serving the final song.

## 🛠️ Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend/API Layer** | Next.js (React/TypeScript) | User interface and internal API for audio mixing. |
| **Backend/AI Service** | Python, Flask, Flask-CORS | REST API handling core generation and caching logic. |
| **AI Model** | MusicGen (Audiocraft, `small` variant) | State-of-the-art text-to-music generation. |
| **Audio Processing** | PyTorch, Torchaudio | Core libraries for handling ML tensors and audio files. |

## 🚀 Setup and Installation

The project requires setting up both the Python backend environment for the AI service and the Node.js environment for the Next.js frontend.

### Prerequisites

1.  Node.js (v18+) and npm
2.  Python (v3.8+)
3.  CUDA (Recommended for GPU acceleration with MusicGen)

### 1. Backend Setup (Flask/MusicGen API)

Navigate to your project root (where `app.py` is located) and prepare the Python environment.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 2. Install Python dependencies
pip install flask flask-cors torch torchaudio audiocraft
