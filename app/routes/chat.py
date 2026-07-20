import os
from litellm import completion
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from app.prompts.system_prompt import SYSTEM_PROMPT
from app.rag.chunk import chunk_text
from app.rag.embeddings import Embeddings
from app.rag.vector_store import VectorStore
from app.rag.retriever import Retriever

from app.core.config import settings

router = APIRouter()

class MessageParam(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    model_name: Optional[str] = "fast"
    history: Optional[List[MessageParam]] = []


def ensure_ingested():
    """Only ingest the PDF if the vector store is empty. Skips if data already exists."""
    db = VectorStore()

    if db.count() > 0:
        return

    config = settings.load_rag_config()
    pdf_path = config.get("active_pdf_path")
    pdf_name = config.get("active_pdf_name", "Research_paper.pdf")

    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    print(f"[Ingest] Vector store is empty — starting ingestion of {pdf_name}...")

    chunks = chunk_text(pdf_path)

    embeddings_service = Embeddings()
    embeddings = embeddings_service.embed_texts(chunks)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": pdf_path, "page": i + 1} for i in range(len(chunks))]

    db.add_documents(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"[Ingest] Done — {db.count()} chunks stored.")


def _run_model(model_name: str, question: str, context: str, history: List[MessageParam] = None):
    config = settings.load_rag_config()
    prompt_template = config.get("system_prompt")

    # Add placeholders to system prompt if omitted
    if "{context}" not in prompt_template:
        prompt_template += "\n\nContext:\n{context}"
    if "{question}" not in prompt_template:
        prompt_template += "\n\nUser Question:\n{question}"

    system_content = prompt_template.format(context=context, question=question)
    messages = [{"role": "system", "content": system_content}]
    if history:
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": question})
    
    return completion(
        model=model_name,
        messages=messages,
        temperature=0.3,
        max_tokens=400,
    )


def _is_rate_limit_error(error: Exception) -> bool:
    error_name = error.__class__.__name__.lower()
    error_message = str(error).lower()
    return (
        "rate" in error_name and "limit" in error_name
        or "rate limit" in error_message
        or "429" in error_message
    )


def _run_primary_model(primary_model: str, question: str, context: str, history: List[MessageParam] = None, max_attempts: int = 3):
    last_error = None
    attempts = 0

    for _ in range(max_attempts):
        attempts += 1
        try:
            return _run_model(primary_model, question, context, history), primary_model, attempts
        except Exception as error:
            last_error = error
            if not _is_rate_limit_error(error):
                break

    raise last_error


def ChatService(question: str, model_name: str, history: List[MessageParam] = None):
    db = VectorStore()
    config = settings.load_rag_config()
    pdf_path = config.get("active_pdf_path")
    pdf_name = config.get("active_pdf_name", "Research_paper.pdf")
    
    is_empty = db.count() == 0
    
    pipeline_steps = []
    if is_empty:
        # Step 1: Text extraction
        try:
            from app.rag.text_extract import PDFLoader
            if not pdf_path or not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found at {pdf_path}")
            loader = PDFLoader()
            text = loader.load_pdf(pdf_path)
            pipeline_steps.append({"name": "text_extract", "status": "done"})
        except Exception:
            pipeline_steps.append({"name": "text_extract", "status": "failed"})
            raise
            
        # Step 2: Chunking
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""],
            )
            chunks = splitter.split_text(text)
            pipeline_steps.append({"name": "chunking", "status": "done"})
        except Exception:
            pipeline_steps.append({"name": "chunking", "status": "failed"})
            raise
            
        # Step 3: Embed chunks
        try:
            embeddings_service = Embeddings()
            embeddings = embeddings_service.embed_texts(chunks)
            pipeline_steps.append({"name": "embedding", "status": "done"})
        except Exception:
            pipeline_steps.append({"name": "embedding", "status": "failed"})
            raise
            
        # Step 4: Vector store
        try:
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"source": pdf_path, "page": i + 1} for i in range(len(chunks))]
            db.add_documents(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            pipeline_steps.append({"name": "vector_store", "status": "done"})
        except Exception:
            pipeline_steps.append({"name": "vector_store", "status": "failed"})
            raise
    else:
        pipeline_steps.extend([
            {"name": "text_extract", "status": "cached"},
            {"name": "chunking", "status": "cached"},
            {"name": "embedding", "status": "cached"},
            {"name": "vector_store", "status": "cached"},
        ])
        
    # Step 5: Query embedding
    try:
        embeddings_service = Embeddings()
        query_embedding = embeddings_service.embed_query(question)
        pipeline_steps.append({"name": "query_embedding", "status": "done"})
    except Exception:
        pipeline_steps.append({"name": "query_embedding", "status": "failed"})
        raise
        
    # Step 6: Similarity search
    try:
        results = db.query(query_embedding, top_k=3)
        pipeline_steps.append({"name": "similarity_search", "status": "done"})
    except Exception:
        pipeline_steps.append({"name": "similarity_search", "status": "failed"})
        raise
        
    # Step 7: Top-k
    try:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        retrieved_chunks = []
        parts = []
        for doc, metadata, score in zip(documents, metadatas, distances):
            retrieved_chunks.append({
                "text": doc,
                "page": metadata.get("page"),
                "source": metadata.get("source"),
                "score": score,
            })
            parts.append(
                f"[Source: {metadata.get('source')}, Page: {metadata.get('page')}]\n{doc}"
            )
        context_str = "\n\n".join(parts) if parts else "No relevant context found."
        pipeline_steps.append({"name": "top_k", "status": "done"})
    except Exception:
        pipeline_steps.append({"name": "top_k", "status": "failed"})
        raise

    model_choice = model_name.lower().strip()
    if model_choice == "fast":
        primary_model = "groq/llama-3.1-8b-instant"
        fallback_model = "gemini/gemini-2.5-flash"
    else:
        primary_model = "gemini/gemini-2.5-flash"
        fallback_model = "groq/llama-3.1-8b-instant"

    try:
        response, used_model, retry_attempts = _run_primary_model(primary_model, question, context_str, history)
    except Exception:
        response = _run_model(fallback_model, question, context_str, history)
        used_model = fallback_model
        retry_attempts = 3

    return {
        "answer": response.choices[0].message.content,
        "model_name": used_model,
        "retry_attempts": retry_attempts,
        "retrieved_chunks": retrieved_chunks,
        "pipeline_steps": pipeline_steps,
    }


@router.post("/chat")
def chat(request: ChatRequest):
    return ChatService(request.question, request.model_name, request.history)