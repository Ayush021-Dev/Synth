Synth: AI-Powered Instrumental Song Generator
🎶 Project Overview
Synth is a full-stack application that generates custom instrumental music tracks based on user-provided genre prompts and thematic lyrics. It features a Next.js frontend that communicates with a powerful Python Flask backend, which utilizes the state-of-the-art MusicGen model for audio generation.

Key Features:
Text-to-Music Generation: Generates original, multi-track audio based on descriptive text prompts.

Lyrics-as-Influence: Uses user-provided lyrics to influence the mood and structure of the instrumental composition.

Seamless Caching: Includes a deceptive caching layer in the backend to provide instant WAV file delivery for popular or predefined genre requests, significantly reducing latency and GPU load.

Audio Mixing: The Next.js Route Handler mixes the generated instrumental track with a pre-existing vocal track (if available) before serving the final song URL.
