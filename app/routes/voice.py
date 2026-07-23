import os
import litellm
import edge_tts
from litellm import completion
from app.core.config import settings

# FastAPI imports
from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import base64

router = APIRouter()

# -----------------------------------------
# Speech To Text
# -----------------------------------------
async def speech_to_text(audio_path: str) -> tuple[str, dict]:
    """
    Converts speech to text using Groq Whisper,
    queries the vector store RAG database for matching paper content,
    and generates an context-augmented answer using Gemini/Llama in a non-blocking thread.
    Returns (transcript, rag_result)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 1. Transcribe speech using Whisper
    with open(audio_path, "rb") as audio_file:
        transcription = await litellm.atranscription(
            model="groq/whisper-large-v3-turbo",
            file=audio_file,
            api_key=settings.GROQ_API_KEY
        )

    transcript = transcription.text

    # 2. Query RAG vector database and retrieve context + response
    import asyncio
    from app.routes.chat import ChatService
    
    # Run the synchronous RAG pipeline in thread pool to prevent blocking the async event loop
    loop = asyncio.get_running_loop()
    rag_result = await loop.run_in_executor(
        None,
        lambda: ChatService(question=transcript, model_name="smart", history=[])
    )
    
    return transcript, rag_result


# -----------------------------------------
# Text To Speech
# -----------------------------------------
async def text_to_speech(
    text: str,
    voice: str = "en-IN-NeerjaNeural"
) -> str:
    """
    Converts text to speech and saves it as:
    response_1.mp3
    response_2.mp3
    response_3.mp3
    ...
    """
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Find the next available filename
    for i in range(1, 100000):
        filename = f"response_{i}.mp3"
        output_path = os.path.join(output_dir, filename)
        if not os.path.exists(output_path):
            break

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice
    )

    await communicate.save(output_path)
    print(f"Audio saved successfully: {output_path}")

    return output_path


# -----------------------------------------
# FastAPI Route Handler
# -----------------------------------------
@router.post("/voice/process")
async def process_voice_audio(file: UploadFile = File(...)):
    """
    Accepts an audio file upload, performs Speech-to-Text transcription,
    queries vector search index for RAG context, generates Text-to-Speech audio,
    and returns transcript, answer info, and base64 audio response.
    """
    # Create necessary folders
    os.makedirs("recordings", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Save uploaded file
    file_path = os.path.join("recordings", file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload audio file: {str(e)}")

    try:
        # 1. Speech-to-Text and Answer (from Groq/Gemini RAG)
        transcript, rag_result = await speech_to_text(file_path)

        # 2. Text-to-Speech synthesis
        answer = rag_result.get("answer")
        output_path = await text_to_speech(answer)

        # 3. Read output audio and encode to base64 for playing in front-end
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output audio not found at: {output_path}")

        with open(output_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "status": "success",
            "transcription": transcript,
            "answer": answer,
            "audio_base64": audio_base64,
            "filename": os.path.basename(output_path),
            "retrieved_chunks": rag_result.get("retrieved_chunks"),
            "pipeline_steps": rag_result.get("pipeline_steps"),
            "latency_metrics": rag_result.get("latency_metrics"),
            "model_name": rag_result.get("model_name"),
            "retry_attempts": rag_result.get("retry_attempts"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))