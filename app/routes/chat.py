import os
import time
import json
from litellm import completion
from fastapi import APIRouter, Depends
from app.servics.exa import search_exa
from pydantic import BaseModel
from typing import List, Optional
from app.prompts.system_prompt import SYSTEM_PROMPT
from app.rag.chunk import chunk_text
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore
from app.rag.retriever import Retriever
from app.core.config import settings
from app.core.security import get_current_user

def calculate_similarity_percentage(score: float, metric: str) -> float:
    # Compute similarity based on space metric configured
    if metric == "cosine":
        # Cosine distance ranges from [0, 2], so similarity is 1.0 - distance
        sim = 1.0 - score
    elif metric == "ip":
        # Inner Product Distance is 1.0 - inner product
        sim = 1.0 - score
    else:  # "l2"
        # Euclidean distance can exceed 1, normalize roughly
        sim = 1.0 - (score / 2.0)
    
    return max(0.0, min(100.0, sim * 100.0))

router = APIRouter()

class MessageParam(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    model_name: Optional[str] = "fast"
    history: Optional[List[MessageParam]] = None


async def ensure_ingested():
    """Only ingest the PDF if the vector store is empty. Skips if data already exists."""
    db = VectorStore()

    if await db.count() > 0:
        return

    config = settings.load_rag_config()
    pdf_path = config.get("active_pdf_path")
    pdf_name = config.get("active_pdf_name", "Research_paper.pdf")

    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    print(f"[Ingest] Vector store is empty — starting ingestion of {pdf_name}...")

    chunks = chunk_text(pdf_path)

    embeddings_service = EmbeddingModel()
    embeddings = await embeddings_service.embed_texts(chunks)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": pdf_path, "page": i + 1} for i in range(len(chunks))]

    await db.add_documents(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"[Ingest] Done — {await db.count()} chunks stored.")


async def _run_model(model_name: str, question: str, context: str, history: List[MessageParam] = None, clerk_id: Optional[str] = None):
    config = settings.load_rag_config()
    prompt_template = config.get("system_prompt")

    # Add placeholders to system prompt if omitted
    if "{context}" not in prompt_template:
        prompt_template += "\n\nContext:\n{context}"
    if "{question}" not in prompt_template:
        prompt_template += "\n\nUser Question:\n{question}"

    system_content = prompt_template.format(context=context, question=question)

    # Append active integrations context if available
    integration_info = []
    from app.mcp.client_manager import mcp_client_manager
    ctx = mcp_client_manager.get_user_context(clerk_id)
    if ctx.get("github_username"):
        integration_info.append(f"- Active GitHub User: {ctx['github_username']}")
    
    slack_team_id = os.getenv("SLACK_TEAM_ID") or config.get("slack_team_id")
    if slack_team_id:
        integration_info.append(f"- Active Slack Team ID: {slack_team_id}")

    if integration_info:
        system_content += "\n\nIntegrations & User Context:\n" + "\n".join(integration_info)

    messages = [{"role": "system", "content": system_content}]
    if history:
        for msg in history:
            role = msg.role.lower().strip() if msg.role else ""
            if role in {"user", "assistant", "system", "model"} and msg.content and msg.content != "string":
                messages.append({"role": role, "content": msg.content})
    messages.append({"role": "user", "content": question})

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web using Exa to find recent information, current events, or academic/external context not present in the provided local research papers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to send to the search engine. Be specific."
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    # Fetch tools from active external MCP servers for this user
    mcp_tools = await mcp_client_manager.get_all_tools(clerk_id)
    tools.extend(mcp_tools)
    
    response = completion(
        model=model_name,
        messages=messages,
        temperature=0.3,
        max_tokens=400,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message
    if hasattr(message, "tool_calls") and message.tool_calls:
        messages.append(message)
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except Exception:
                arguments = {}

            if tool_name == "search_web":
                search_query = arguments.get("query", question)
                search_results = search_exa(search_query)
                if "error" in search_results:
                    tool_content = f"Search failed: {search_results['error']}"
                else:
                    formatted_results = []
                    for res in search_results.get("results", []):
                        title = res.get("title", "No Title")
                        url = res.get("url", "No URL")
                        highlights = res.get("highlights", [])
                        highlight_text = " | ".join(highlights) if highlights else "No highlights"
                        formatted_results.append(f"Title: {title}\nURL: {url}\nExcerpt: {highlight_text}")
                    tool_content = "\n\n".join(formatted_results) if formatted_results else "No results found."
            else:
                # Route to external user-specific MCP servers
                tool_content = await mcp_client_manager.execute_tool(tool_name, arguments, clerk_id)

            messages.append({
                "role": "tool",
                "name": tool_name,
                "tool_call_id": tool_call.id,
                "content": tool_content
            })
        
        response = completion(
            model=model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=400
        )

    return response


def _is_rate_limit_error(error: Exception) -> bool:
    error_name = error.__class__.__name__.lower()
    error_message = str(error).lower()
    return (
        ("rate" in error_name and "limit" in error_name)
        or "rate limit" in error_message
        or "429" in error_message
    )


async def _run_primary_model(primary_model: str, question: str, context: str, history: List[MessageParam] = None, max_attempts: int = 3, clerk_id: Optional[str] = None):
    last_error = None
    attempts = 0

    for _ in range(max_attempts):
        attempts += 1
        try:
            return await _run_model(primary_model, question, context, history, clerk_id=clerk_id), primary_model, attempts
        except Exception as error:
            last_error = error
            if not _is_rate_limit_error(error):
                break

    raise last_error


async def ChatService(question: str, model_name: str, history: List[MessageParam] = None, current_user: Optional[dict] = None):
    total_start = time.time()
    db = VectorStore()
    config = settings.load_rag_config()
    pdf_path = config.get("active_pdf_path")
    pdf_name = config.get("active_pdf_name", "Research_paper.pdf")
    similarity_metric_type = config.get("similarity_metric", "cosine")
    
    # Initialize user MCP connection if logged in
    clerk_id = None
    if current_user:
        clerk_id = current_user.get("clerk_id")
        from app.mcp.client_manager import mcp_client_manager
        await mcp_client_manager.ensure_user_initialized(clerk_id, current_user)
        
    is_empty = (await db.count()) == 0
    
    pipeline_steps = []
    if is_empty:
        # Step 1: Text extraction
        try:
            step_start = time.time()
            from app.rag.text_extract import PDFLoader
            if not pdf_path or not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found at {pdf_path}")
            loader = PDFLoader()
            text = loader.load_pdf(pdf_path)
            pipeline_steps.append({
                "name": "text_extract", 
                "status": "done",
                "latency_ms": round((time.time() - step_start) * 1000, 2)
            })
        except Exception:
            pipeline_steps.append({"name": "text_extract", "status": "failed"})
            raise
            
        # Step 2: Chunking
        try:
            step_start = time.time()
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""],
            )
            chunks = splitter.split_text(text)
            pipeline_steps.append({
                "name": "chunking", 
                "status": "done",
                "latency_ms": round((time.time() - step_start) * 1000, 2)
            })
        except Exception:
            pipeline_steps.append({"name": "chunking", "status": "failed"})
            raise
            
        # Step 3: Embed chunks
        try:
            step_start = time.time()
            embeddings_service = EmbeddingModel()
            embeddings = await embeddings_service.embed_texts(chunks)
            pipeline_steps.append({
                "name": "embedding", 
                "status": "done",
                "latency_ms": round((time.time() - step_start) * 1000, 2)
            })
        except Exception:
            pipeline_steps.append({"name": "embedding", "status": "failed"})
            raise
            
        # Step 4: Vector store
        try:
            step_start = time.time()
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"source": pdf_path, "page": i + 1} for i in range(len(chunks))]
            await db.add_documents(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            pipeline_steps.append({
                "name": "vector_store", 
                "status": "done",
                "latency_ms": round((time.time() - step_start) * 1000, 2)
            })
        except Exception:
            pipeline_steps.append({"name": "vector_store", "status": "failed"})
            raise
    else:
        pipeline_steps.extend([
            {"name": "text_extract", "status": "cached", "latency_ms": 0.0},
            {"name": "chunking", "status": "cached", "latency_ms": 0.0},
            {"name": "embedding", "status": "cached", "latency_ms": 0.0},
            {"name": "vector_store", "status": "cached", "latency_ms": 0.0},
        ])
        
    # Step 5: Query embedding
    try:
        step_start = time.time()
        embeddings_service = EmbeddingModel()
        query_embedding = await embeddings_service.embed_query(question)
        pipeline_steps.append({
            "name": "query_embedding", 
            "status": "done",
            "latency_ms": round((time.time() - step_start) * 1000, 2)
        })
    except Exception:
        pipeline_steps.append({"name": "query_embedding", "status": "failed"})
        raise
        
    # Step 6: Similarity search
    try:
        step_start = time.time()
        results = await db.query(query_embedding=query_embedding, top_k=3)
        pipeline_steps.append({
            "name": "similarity_search", 
            "status": "done",
            "latency_ms": round((time.time() - step_start) * 1000, 2)
        })
    except Exception:
        pipeline_steps.append({"name": "similarity_search", "status": "failed"})
        raise
        
    # Step 7: Top-k
    try:
        step_start = time.time()
        retrieved_chunks = []
        parts = []
        for doc in results:
            score = doc.get("score", 0.0)
            sim_pct = calculate_similarity_percentage(score, similarity_metric_type)
            retrieved_chunks.append({
                "text": doc.get("text", ""),
                "page": doc.get("page"),
                "source": doc.get("source", "unknown"),
                "score": score,
                "similarity_percentage": sim_pct,
                "metric": similarity_metric_type.upper()
            })
            parts.append(
                f"[Source: {doc.get('source', 'unknown')}, Page: {doc.get('page')}]\n{doc.get('text', '')}"
            )
        context_str = "\n\n".join(parts) if parts else "No relevant context found."
        pipeline_steps.append({
            "name": "top_k", 
            "status": "done",
            "latency_ms": round((time.time() - step_start) * 1000, 2)
        })
    except Exception:
        pipeline_steps.append({"name": "top_k", "status": "failed"})
        raise

    rag_end = time.time()
    rag_latency = round((rag_end - total_start) * 1000, 2)

    model_choice = model_name.lower().strip()
    if model_choice == "fast":
        primary_model = "mistral/mistral-small-latest"
        fallback_model = "gemini/gemini-2.5-flash"
    else:
        primary_model = "gemini/gemini-2.5-flash"
        fallback_model = "mistral/mistral-small-latest"

    llm_start = time.time()
    try:
        response, used_model, retry_attempts = await _run_primary_model(primary_model, question, context_str, history, clerk_id=clerk_id)
    except Exception:
        response = await _run_model(fallback_model, question, context_str, history, clerk_id=clerk_id)
        used_model = fallback_model
        retry_attempts = 3

    model_end = time.time()
    llm_latency = round((model_end - llm_start) * 1000, 2)
    total_latency = round((model_end - total_start) * 1000, 2)

    return {
        "answer": response.choices[0].message.content,
        "model_name": used_model,
        "retry_attempts": retry_attempts,
        "retrieved_chunks": retrieved_chunks,
        "pipeline_steps": pipeline_steps,
        "latency_metrics": {
            "total_latency_ms": total_latency,
            "rag_latency_ms": rag_latency,
            "llm_latency_ms": llm_latency,
        }
    }


@router.post("/chat")
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    return await ChatService(request.question, request.model_name, request.history, current_user=current_user)