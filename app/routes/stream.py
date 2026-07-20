import os
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from litellm import completion
from typing import List

from app.prompts.system_prompt import SYSTEM_PROMPT
from app.rag.chunk import chunk_text
from app.rag.embeddings import Embeddings
from app.rag.vector_store import VectorStore
from app.rag.retriever import Retriever
from app.routes.chat import MessageParam, ChatRequest
from app.core.config import settings

router = APIRouter()


def ensure_ingested():
    """Only ingest the PDF if the vector store is empty. Skips if data already exists."""
    db = VectorStore()

    if db.count() > 0:
        return  # Already ingested — skip
        
    config = settings.load_rag_config()
    pdf_path = config.get("active_pdf_path")
    pdf_name = config.get("active_pdf_name", "Research_paper.pdf")
    
    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

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


def _build_context(question: str) -> str:
    """Retrieve top-k relevant chunks and format them into a context string."""
    retriever = Retriever()
    retrieved_chunks = retriever.retrieve(question, top_k=3)

    if not retrieved_chunks:
        return "No relevant context found."

    parts = []
    for chunk in retrieved_chunks:
        parts.append(
            f"[Source: {chunk['source']}, Page: {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def _run_model_stream(model_name: str, question: str, context: str, history: List[MessageParam] = None):
    """Call LiteLLM with a fully formatted RAG system prompt."""
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
        stream=True,
    )


def _is_rate_limit_error(error: Exception) -> bool:
    error_name = error.__class__.__name__.lower()
    error_message = str(error).lower()
    return (
        ("rate" in error_name and "limit" in error_name)
        or "rate limit" in error_message
        or "429" in error_message
    )


def _choose_models(model_name: str):
    model_choice = model_name.lower().strip()
    if model_choice == "fast":
        return "groq/llama-3.1-8b-instant", "gemini/gemini-2.5-flash"
    return "gemini/gemini-2.5-flash", "groq/llama-3.1-8b-instant"


def _stream_answer(model_name: str, question: str, history: List[MessageParam] = None):
    """Full RAG streaming pipeline: ingest (if needed) → retrieve → stream."""
    import time
    import json
    from app.rag.text_extract import PDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    db = VectorStore()
    config = settings.load_rag_config()
    pdf_path = config.get("active_pdf_path")
    pdf_name = config.get("active_pdf_name", "Research_paper.pdf")
    
    # 1. Check ingest status
    is_empty = db.count() == 0

    if is_empty:
        # Step 1: Text extraction
        yield "__STEP__:text_extract:active\n"
        if not pdf_path or not os.path.exists(pdf_path):
            yield f"Error: PDF file not found at {pdf_path}\n"
            return
        try:
            loader = PDFLoader()
            text = loader.load_pdf(pdf_path)
            yield "__STEP__:text_extract:done\n"
            time.sleep(0.15)
        except Exception as e:
            yield f"Error during text extraction: {str(e)}\n"
            return

        # Step 2: Chunking
        yield "__STEP__:chunking:active\n"
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""],
            )
            chunks = splitter.split_text(text)
            yield "__STEP__:chunking:done\n"
            time.sleep(0.15)
        except Exception as e:
            yield f"Error during text chunking: {str(e)}\n"
            return

        # Step 3: Embed chunks
        yield "__STEP__:embedding:active\n"
        try:
            embeddings_service = Embeddings()
            embeddings = embeddings_service.embed_texts(chunks)
            yield "__STEP__:embedding:done\n"
            time.sleep(0.15)
        except Exception as e:
            yield f"Error during embedding: {str(e)}\n"
            return

        # Step 4: Vector store
        yield "__STEP__:vector_store:active\n"
        try:
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"source": pdf_path, "page": i + 1} for i in range(len(chunks))]
            db.add_documents(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            yield "__STEP__:vector_store:done\n"
            time.sleep(0.15)
        except Exception as e:
            yield f"Error during vector store: {str(e)}\n"
            return
    else:
        # Already ingested - use cached status
        yield "__STEP__:text_extract:cached\n"
        time.sleep(0.15)
        yield "__STEP__:chunking:cached\n"
        time.sleep(0.15)
        yield "__STEP__:embedding:cached\n"
        time.sleep(0.15)
        yield "__STEP__:vector_store:cached\n"
        time.sleep(0.15)

    # Step 5: Query embedding
    yield "__STEP__:query_embedding:active\n"
    try:
        embeddings_service = Embeddings()
        query_embedding = embeddings_service.embed_query(question)
        yield "__STEP__:query_embedding:done\n"
        time.sleep(0.15)
    except Exception as e:
        yield f"Error generating query embedding: {str(e)}\n"
        return

    # Step 6: Similarity search
    yield "__STEP__:similarity_search:active\n"
    try:
        # Get query results from DB
        results = db.query(query_embedding, top_k=3)
        yield "__STEP__:similarity_search:done\n"
        time.sleep(0.15)
    except Exception as e:
        yield f"Error during similarity search: {str(e)}\n"
        return

    # Step 7: Top-k
    yield "__STEP__:top_k:active\n"
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
        context = "\n\n".join(parts) if parts else "No relevant context found."
        yield "__STEP__:top_k:done\n"
        time.sleep(0.1)
    except Exception as e:
        yield f"Error extracting top-k: {str(e)}\n"
        return

    # Send retrieved chunks to frontend as metadata statement
    try:
        chunks_json = json.dumps(retrieved_chunks)
        yield f"__RETRIEVED_CHUNKS__:{chunks_json}\n"
    except Exception as e:
        print("Error serializing chunks:", e)

    # Stream LLM response
    primary_model, fallback_model = _choose_models(model_name)
    attempts = 0

    while attempts < 3:
        attempts += 1
        try:
            yield f"Model: {primary_model}\nAttempts: {attempts}\n\n"
            for chunk in _run_model_stream(primary_model, question, context, history):
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    yield delta
            return
        except Exception as error:
            if not _is_rate_limit_error(error):
                break

    try:
        yield f"\n\nModel: {fallback_model}\nAttempts: 1\n\n"
        for chunk in _run_model_stream(fallback_model, question, context, history):
            delta = getattr(chunk.choices[0].delta, "content", None)
            if delta:
                yield delta
    except Exception as error:
        yield f"\n\nError streaming from fallback model ({fallback_model}): {str(error)}"


@router.post("/chat/stream")
def stream_chat(request: ChatRequest):
    return StreamingResponse(
        _stream_answer(request.model_name, request.question, request.history),
        media_type="text/plain"
    )