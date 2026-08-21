import os
import json
import time
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from litellm import completion
from typing import List, Optional
from app.servics.exa import search_exa

from app.prompts.system_prompt import SYSTEM_PROMPT
from app.rag.chunk import chunk_text
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore
from app.rag.retriever import Retriever
from app.routes.chat import MessageParam, ChatRequest, calculate_similarity_percentage
from app.core.config import settings
from app.core.security import get_current_user

router = APIRouter()


async def ensure_ingested():
    """Only ingest the PDF if the vector store is empty. Skips if data already exists."""
    db = VectorStore()

    if await db.count() > 0:
        return  # Already ingested — skip
        
    config = settings.load_rag_config()
    pdf_path = config.get("active_pdf_path")
    pdf_name = config.get("active_pdf_name", "Research_paper.pdf")
    
    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

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


async def _build_context(question: str) -> str:
    """Retrieve top-k relevant chunks and format them into a context string."""
    retriever = Retriever()
    retrieved_chunks = await retriever.retrieve(question, top_k=3)

    if not retrieved_chunks:
        return "No relevant context found."

    parts = []
    for chunk in retrieved_chunks:
        parts.append(
            f"[Source: {chunk['source']}, Page: {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def _run_model_stream(model_name: str, question: str, context: str, history: List[MessageParam] = None, clerk_id: Optional[str] = None):
    """Call LiteLLM with a fully formatted RAG system prompt."""
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
        return "mistral/mistral-small-latest", "gemini/gemini-2.5-flash"
    return "gemini/gemini-2.5-flash", "mistral/mistral-small-latest"


async def _stream_answer(model_name: str, question: str, history: List[MessageParam] = None, current_user: Optional[dict] = None):
    """Full RAG streaming pipeline: ingest (if needed) → retrieve → stream."""
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
        
    # 1. Check ingest status
    is_empty = (await db.count()) == 0

    if is_empty:
        # Step 1: Text extraction
        yield "__STEP__:text_extract:active\n"
        if not pdf_path or not os.path.exists(pdf_path):
            yield f"Error: PDF file not found at {pdf_path}\n"
            return
        try:
            step_start = time.time()
            from app.rag.text_extract import PDFLoader
            loader = PDFLoader()
            text = loader.load_pdf(pdf_path)
            lat = round((time.time() - step_start) * 1000, 2)
            yield f"__STEP__:text_extract:done:{lat}\n"
            time.sleep(0.05)
        except Exception as e:
            yield f"Error during text extraction: {str(e)}\n"
            return

        # Step 2: Chunking
        yield "__STEP__:chunking:active\n"
        try:
            step_start = time.time()
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""],
            )
            chunks = splitter.split_text(text)
            lat = round((time.time() - step_start) * 1000, 2)
            yield f"__STEP__:chunking:done:{lat}\n"
            time.sleep(0.05)
        except Exception as e:
            yield f"Error during text chunking: {str(e)}\n"
            return

        # Step 3: Embed chunks
        yield "__STEP__:embedding:active\n"
        try:
            step_start = time.time()
            embeddings_service = EmbeddingModel()
            embeddings = await embeddings_service.embed_texts(chunks)
            lat = round((time.time() - step_start) * 1000, 2)
            yield f"__STEP__:embedding:done:{lat}\n"
            time.sleep(0.05)
        except Exception as e:
            yield f"Error during embedding: {str(e)}\n"
            return

        # Step 4: Vector store
        yield "__STEP__:vector_store:active\n"
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
            lat = round((time.time() - step_start) * 1000, 2)
            yield f"__STEP__:vector_store:done:{lat}\n"
            time.sleep(0.05)
        except Exception as e:
            yield f"Error during vector store: {str(e)}\n"
            return
    else:
        # Already ingested - use cached status with 0.0 latency yield
        yield "__STEP__:text_extract:cached:0.0\n"
        yield "__STEP__:chunking:cached:0.0\n"
        yield "__STEP__:embedding:cached:0.0\n"
        yield "__STEP__:vector_store:cached:0.0\n"

    # Step 5: Query embedding
    yield "__STEP__:query_embedding:active\n"
    try:
        step_start = time.time()
        embeddings_service = EmbeddingModel()
        query_embedding = await embeddings_service.embed_query(question)
        lat = round((time.time() - step_start) * 1000, 2)
        yield f"__STEP__:query_embedding:done:{lat}\n"
        time.sleep(0.05)
    except Exception as e:
        yield f"Error generating query embedding: {str(e)}\n"
        return

    # Step 6: Similarity search
    yield "__STEP__:similarity_search:active\n"
    try:
        step_start = time.time()
        results = await db.query(query_embedding=query_embedding, top_k=3)
        lat = round((time.time() - step_start) * 1000, 2)
        yield f"__STEP__:similarity_search:done:{lat}\n"
        time.sleep(0.05)
    except Exception as e:
        yield f"Error during similarity search: {str(e)}\n"
        return

    # Step 7: Top-k
    yield "__STEP__:top_k:active\n"
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
        context = "\n\n".join(parts) if parts else "No relevant context found."
        lat = round((time.time() - step_start) * 1000, 2)
        yield f"__STEP__:top_k:done:{lat}\n"
        time.sleep(0.05)
    except Exception as e:
        yield f"Error extracting top-k: {str(e)}\n"
        return

    # Send retrieved chunks to frontend as metadata statement
    try:
        chunks_json = json.dumps(retrieved_chunks)
        yield f"__RETRIEVED_CHUNKS__:{chunks_json}\n"
    except Exception as e:
        print("Error serializing chunks:", e)

    rag_end = time.time()
    rag_latency = round((rag_end - total_start) * 1000, 2)

    # Stream LLM response
    primary_model, fallback_model = _choose_models(model_name)
    attempts = 0
    llm_start = time.time()
    success = False

    # Build messages list
    system_content = config.get("system_prompt")
    if "{context}" not in system_content:
        system_content += "\n\nContext:\n{context}"
    if "{question}" not in system_content:
        system_content += "\n\nUser Question:\n{question}"
    
    system_content_formatted = system_content.format(context=context, question=question)

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
        system_content_formatted += "\n\nIntegrations & User Context:\n" + "\n".join(integration_info)

    messages = [{"role": "system", "content": system_content_formatted}]
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

    # Fetch tools from active external user-specific MCP servers
    mcp_tools = await mcp_client_manager.get_all_tools(clerk_id)
    tools.extend(mcp_tools)

    while attempts < 3:
        attempts += 1
        try:
            yield f"Model: {primary_model}\nAttempts: {attempts}\n\n"
            
            # Start streaming the first response immediately
            stream_res = completion(
                model=primary_model,
                messages=messages,
                temperature=0.3,
                max_tokens=400,
                tools=tools,
                tool_choice="auto" if tools else None,
                stream=True
            )
            
            tool_calls_accumulator = {}
            has_tool_calls = False
            
            for chunk in stream_res:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue
                
                # Check for streaming tool calls
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    has_tool_calls = True
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_accumulator:
                            tool_calls_accumulator[idx] = {
                                "id": tc.id or "",
                                "name": "",
                                "arguments": ""
                            }
                        if tc.id:
                            tool_calls_accumulator[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_accumulator[idx]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls_accumulator[idx]["arguments"] += tc.function.arguments
                
                # If it's regular content and no tool calls have been detected yet, stream it
                elif not has_tool_calls and hasattr(delta, "content") and delta.content:
                    yield delta.content
            
            # If tool calls were accumulated, execute them and stream the final response
            if has_tool_calls:
                # Add assistant message with tool calls to history
                assistant_tool_msg = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc_val["id"],
                            "type": "function",
                            "function": {
                                "name": tc_val["name"],
                                "arguments": tc_val["arguments"]
                            }
                        }
                        for tc_val in tool_calls_accumulator.values()
                    ]
                }
                messages.append(assistant_tool_msg)
                
                # Execute tools
                for tc_val in tool_calls_accumulator.values():
                    tool_name = tc_val["name"]
                    try:
                        arguments = json.loads(tc_val["arguments"])
                    except Exception:
                        arguments = {}

                    if tool_name == "search_web":
                        search_query = arguments.get("query", question)
                        search_results = search_exa(search_query)
                        if "error" in search_results:
                            tool_content = f"Search failed: {search_results['error']}"
                        else:
                            formatted_results = []
                            for r in search_results.get("results", []):
                                title = r.get("title", "No Title")
                                url = r.get("url", "No URL")
                                highlights = r.get("highlights", [])
                                highlight_text = " | ".join(highlights) if highlights else "No highlights"
                                formatted_results.append(f"Title: {title}\nURL: {url}\nExcerpt: {highlight_text}")
                            tool_content = "\n\n".join(formatted_results) if formatted_results else "No results found."
                    else:
                        # Route to external user-specific MCP servers
                        tool_content = await mcp_client_manager.execute_tool(tool_name, arguments, clerk_id)
                    
                    messages.append({
                        "role": "tool",
                        "name": tool_name,
                        "tool_call_id": tc_val["id"],
                        "content": tool_content
                    })
                
                # Stream the final answer after tool completion
                final_stream = completion(
                    model=primary_model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=400,
                    stream=True
                )
                for chunk in final_stream:
                    delta = getattr(chunk.choices[0].delta, "content", None)
                    if delta:
                        yield delta
            
            success = True
            break
        except Exception as error:
            if not _is_rate_limit_error(error):
                break

    if not success:
        try:
            yield f"\n\nModel: {fallback_model}\nAttempts: 1\n\n"
            
            # Simple fallback completion without streaming first
            res = completion(
                model=fallback_model,
                messages=messages,
                temperature=0.3,
                max_tokens=400,
                tools=tools,
                tool_choice="auto" if tools else None
            )
            msg_obj = res.choices[0].message
            if hasattr(msg_obj, "tool_calls") and msg_obj.tool_calls:
                messages.append(msg_obj)
                for tool_call in msg_obj.tool_calls:
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
                            for r in search_results.get("results", []):
                                title = r.get("title", "No Title")
                                url = r.get("url", "No URL")
                                highlights = r.get("highlights", [])
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
                
                stream_res = completion(
                    model=fallback_model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=400,
                    stream=True
                )
                for chunk in stream_res:
                    delta = getattr(chunk.choices[0].delta, "content", None)
                    if delta:
                        yield delta
            else:
                content = getattr(msg_obj, "content", "")
                if content:
                    yield content
        except Exception as error:
            yield f"\n\nError streaming from fallback model ({fallback_model}): {str(error)}"



@router.post("/chat/stream")
async def stream_chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    return StreamingResponse(
        _stream_answer(request.model_name, request.question, request.history, current_user=current_user),
        media_type="text/plain"
    )