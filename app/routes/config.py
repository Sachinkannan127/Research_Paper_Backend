import os
import shutil
import time
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.core.config import settings
from app.rag.vector_store import VectorStore
from app.rag.chunk import chunk_text
from app.rag.embeddings import Embeddings

router = APIRouter(prefix="/config", tags=["Configuration"])

class ConfigUpdate(BaseModel):
    system_prompt: str
    welcome_message: str

@router.get("")
def get_config():
    config = settings.load_rag_config()
    return {
        "active_pdf_name": config.get("active_pdf_name", "Research_paper.pdf"),
        "system_prompt": config.get("system_prompt", ""),
        "welcome_message": config.get("welcome_message", "")
    }

@router.post("")
def update_config(data: ConfigUpdate):
    config = settings.load_rag_config()
    config["system_prompt"] = data.system_prompt
    config["welcome_message"] = data.welcome_message
    settings.save_rag_config(config)
    return {"status": "success", "message": "Configuration updated successfully"}

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    upload_dir = os.path.join("app", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
    config = settings.load_rag_config()
    config["active_pdf_name"] = file.filename
    config["active_pdf_path"] = file_path
    settings.save_rag_config(config)
    
    return {
        "status": "success",
        "filename": file.filename,
        "path": file_path
    }

@router.post("/clear-database")
def clear_database():
    try:
        db = VectorStore()
        db.delete_all()
        return {"status": "success", "message": "Vector database cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")

@router.get("/ingest/stream")
def ingest_stream():
    def _stream_ingest():
        config = settings.load_rag_config()
        pdf_path = config.get("active_pdf_path")
        
        if not pdf_path or not os.path.exists(pdf_path):
            yield "__STEP__:text_extract:failed\n"
            yield f"Error: Active PDF file not found at {pdf_path}\n"
            return
            
        try:
            db = VectorStore()
            
            # Step 1: Text extraction
            yield "__STEP__:text_extract:active\n"
            time.sleep(0.3)
            # Text extraction runs as part of chunking, but we status check here
            yield "__STEP__:text_extract:done\n"
            
            # Step 2: Chunking
            yield "__STEP__:chunking:active\n"
            time.sleep(0.3)
            chunks = chunk_text(pdf_path)
            yield "__STEP__:chunking:done\n"
            
            # Step 3: Embeddings
            yield "__STEP__:embedding:active\n"
            embeddings_service = Embeddings()
            embeddings = embeddings_service.embed_texts(chunks)
            yield "__STEP__:embedding:done\n"
            
            # Step 4: Vector store
            yield "__STEP__:vector_store:active\n"
            time.sleep(0.3)
            
            # Clear database first so we don't mix documents
            db.delete_all()
            
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"source": pdf_path, "page": i + 1} for i in range(len(chunks))]
            
            db.add_documents(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )
            yield "__STEP__:vector_store:done\n"
            yield f"Successfully ingested {len(chunks)} chunks from {config.get('active_pdf_name')}\n"
            
        except Exception as e:
            yield f"Error: Ingestion failed due to {str(e)}\n"
            
    return StreamingResponse(_stream_ingest(), media_type="text/plain")
