# Install LiveKit agents with optional dependencies

pip install \
  "livekit-agents[deepgram,openai,cartesia,silero,turn-detector]~=1.0" \
  "livekit-plugins-noise-cancellation~=0.2" \
  python-dotenv

# Install Sarvam TTS support

pip install "livekit-agents[sarvam]~=1.0"

