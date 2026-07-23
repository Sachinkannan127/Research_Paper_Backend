import sys
import os
from pathlib import Path

# Add project root to sys.path to resolve 'app' module when running the script directly
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import asyncio
from app.routes.voice import speech_to_text, text_to_speech

async def main():
    # Attempt to locate audio2.mp3 relative to backend_llm
    audio_file = "recordings/audio2.mp3"
    if not os.path.exists(audio_file):
        # Fallback if run from backend_llm parent directory
        audio_file = "backend_llm/recordings/audio2.mp3"

    print(f"Using audio file: {audio_file}")
    transcript, response, retrieved_chunks = await speech_to_text(audio_file)

    print(f"Transcript: {transcript}")
    print(f"Response: {response}")
    print(f"Retrieved Chunks: {len(retrieved_chunks)} items matched from the paper.")

    output = await text_to_speech(response)

    print(f"Output saved to: {output}")

if __name__ == "__main__":
    asyncio.run(main())