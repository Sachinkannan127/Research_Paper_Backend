"""
MCP (Model Context Protocol) Server for Research Paper Assistant.

Exposes the existing RAG pipeline, web search, and configuration management
as MCP tools and resources, authenticated via Clerk JWT (same auth as the
existing FastAPI backend).

Run from the backend_llm directory:
    venv\\Scripts\\activate
    python app/mcp/mcp_server.py        # HTTP on port 8001
    mcp dev app/mcp/mcp_server.py       # Dev mode with MCP Inspector
"""

import os
import sys
import json
import base64
import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

# ── Ensure the backend root (backend_llm/) is on sys.path ──
# This file lives at app/mcp/mcp_server.py, so we go up two levels.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mcp.server.mcpserver import MCPServer
from mcp.server.auth.provider import TokenVerifier, AccessToken

# ── Logging ──
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_server")

import jwt  # PyJWT (already in requirements)

CLERK_JWKS_URL = os.getenv(
    "CLERK_JWKS_URL",
    "https://choice-wolf-9655.clerk.accounts.dev/.well-known/jwks.json"
)


# ─────────────────────────────────────────────
# Clerk JWT Token Verifier
# ─────────────────────────────────────────────
class ClerkTokenVerifier:
    """
    Implements the MCP TokenVerifier protocol using Clerk JWTs.
    Verifies tokens against Clerk's JWKS endpoint — identical to the
    approach used in app/core/security.py (verify_clerk_token).

    When CLERK_JWKS_URL is not set, falls back to dev mode (all requests pass).
    """

    def __init__(self, jwks_url: str):
        self.jwks_url = jwks_url
        # PyJWKClient caches the JWKS and auto-refreshes on key rotation
        self._jwk_client = jwt.PyJWKClient(jwks_url) if jwks_url else None

    async def verify_token(self, token: str) -> AccessToken | None:
        """
        Verify a Clerk session JWT and return an MCP AccessToken if valid.
        Returns None if verification fails (MCP will respond with 401).
        """
        # Dev mode — no JWKS URL configured
        if not self._jwk_client:
            logger.warning("CLERK_JWKS_URL not set — allowing request (dev mode)")
            return AccessToken(
                token=token,
                client_id="dev",
                scopes=["openid"],
                subject="dev@localhost",
            )

        try:
            # Fetch the matching signing key from Clerk's JWKS (cached)
            signing_key = await asyncio.to_thread(
                self._jwk_client.get_signing_key_from_jwt, token
            )
            # Decode and verify the JWT (RS256, expiry enforced)
            data = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                options={"verify_exp": True},
            )

            clerk_id = data.get("sub", "")
            logger.info(f"Verified Clerk token for: {clerk_id}")

            return AccessToken(
                token=token,
                client_id="clerk",
                scopes=["openid"],
                subject=clerk_id,
                expires_at=data.get("exp"),
                claims=data,
            )

        except jwt.ExpiredSignatureError:
            logger.warning("Clerk token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Clerk token invalid: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error verifying Clerk token: {e}")
            return None


# ─────────────────────────────────────────────
# MongoDB Lifespan Management
# ─────────────────────────────────────────────
@asynccontextmanager
async def mcp_lifespan(server):
    """Connect to MongoDB on startup, disconnect on shutdown."""
    from app.db.mongodb import connect_to_mongo, close_mongo_connection
    logger.info("🚀 MCP Server starting up...")
    try:
        await connect_to_mongo()
        logger.info("✅ MongoDB connected for MCP server")
    except Exception as e:
        logger.error(f"❌ Failed to connect to MongoDB: {e}")
        raise
    yield {}
    await close_mongo_connection()
    logger.info("🔒 MCP Server shut down.")


# ─────────────────────────────────────────────
# MCPServer Instance
# ─────────────────────────────────────────────
from mcp.server.auth.settings import AuthSettings

MCP_PORT = int(os.getenv("MCP_SERVER_PORT", "8001"))

# Use Clerk as the issuer; enable auth only when a JWKS URL is available
if CLERK_JWKS_URL:
    # Extract the base issuer URL from the JWKS URL
    # e.g. https://choice-wolf-9655.clerk.accounts.dev/.well-known/jwks.json
    #   -> https://choice-wolf-9655.clerk.accounts.dev
    _clerk_issuer = CLERK_JWKS_URL.split("/.well-known/")[0]
    _auth_settings = AuthSettings(
        issuer_url=_clerk_issuer,
        resource_server_url=f"http://localhost:{MCP_PORT}",
        required_scopes=["openid"],
    )
    _token_verifier = ClerkTokenVerifier(CLERK_JWKS_URL)
else:
    _auth_settings = None
    _token_verifier = None

mcp = MCPServer(
    name="research-paper-assistant",
    title="Research Paper Assistant",
    instructions=(
        "You are connected to the Research Paper Assistant MCP server. "
        "Use the available tools to ask questions about uploaded research papers, "
        "search the web for additional context, upload new papers, and manage "
        "the RAG pipeline configuration."
    ),
    version="1.0.0",
    token_verifier=_token_verifier,
    auth=_auth_settings,
    lifespan=mcp_lifespan,
)



# ═════════════════════════════════════════════
#  TOOLS
# ═════════════════════════════════════════════

@mcp.tool()
async def ask_paper(
    question: str,
    model_name: str = "fast",
    history: Optional[list[dict]] = None,
) -> str:
    """
    Ask a question about the currently loaded research paper.
    Uses the full RAG pipeline: embed query → similarity search → LLM answer.

    Args:
        question: The question to ask about the paper.
        model_name: Model tier — "fast" (Mistral) or "quality" (Gemini). Defaults to "fast".
        history: Optional conversation history as a list of {role, content} dicts.
    """
    from app.routes.chat import ChatService, MessageParam

    # Convert raw dicts to MessageParam objects if history is provided
    msg_history = None
    if history:
        msg_history = [
            MessageParam(role=h.get("role", "user"), content=h.get("content", ""))
            for h in history
        ]

    try:
        result = await ChatService(question, model_name, msg_history)
        # Format a readable response
        answer = result.get("answer", "No answer generated.")
        model_used = result.get("model_name", "unknown")
        chunks = result.get("retrieved_chunks", [])
        latency = result.get("latency_metrics", {})

        response_parts = [
            f"**Answer:**\n{answer}",
            f"\n**Model:** {model_used}",
            f"**Total Latency:** {latency.get('total_latency_ms', 0):.0f}ms "
            f"(RAG: {latency.get('rag_latency_ms', 0):.0f}ms, LLM: {latency.get('llm_latency_ms', 0):.0f}ms)",
        ]

        if chunks:
            response_parts.append(f"\n**Retrieved Chunks ({len(chunks)}):**")
            for i, chunk in enumerate(chunks, 1):
                sim = chunk.get("similarity_percentage", 0)
                page = chunk.get("page", "?")
                text_preview = chunk.get("text", "")[:200]
                response_parts.append(
                    f"  {i}. [Page {page}, {sim:.1f}% similar] {text_preview}..."
                )

        return "\n".join(response_parts)

    except FileNotFoundError as e:
        return f"❌ No paper loaded: {e}. Use the `upload_paper` tool first."
    except Exception as e:
        return f"❌ Error querying paper: {e}"


@mcp.tool()
def search_web(query: str, num_results: int = 5) -> str:
    """
    Search the web using Exa for recent information, academic context,
    or anything not present in the uploaded research paper.

    Args:
        query: The search query. Be specific for better results.
        num_results: Number of results to return (default: 5, max: 10).
    """
    from app.servics.exa import search_exa

    num_results = min(max(num_results, 1), 10)
    results = search_exa(query, num_results)

    if "error" in results:
        return f"❌ Search failed: {results['error']}"

    formatted = []
    for r in results.get("results", []):
        title = r.get("title", "No Title")
        url = r.get("url", "")
        highlights = r.get("highlights", [])
        excerpt = " | ".join(highlights[:3]) if highlights else "No excerpt"
        formatted.append(f"**{title}**\n  URL: {url}\n  Excerpt: {excerpt}")

    if not formatted:
        return "No results found."

    return f"**Web Search Results for:** \"{query}\"\n\n" + "\n\n".join(formatted)


@mcp.tool()
async def upload_paper(filename: str, content_base64: str) -> str:
    """
    Upload a new research paper (PDF) to the system.
    This replaces the currently active paper and clears the vector store
    so the new paper will be ingested on the next question.

    Args:
        filename: Name for the PDF file (must end in .pdf).
        content_base64: The PDF file content encoded as a base64 string.
    """
    if not filename.endswith(".pdf"):
        return "❌ Only PDF files are supported. Filename must end in .pdf"

    try:
        pdf_bytes = base64.b64decode(content_base64)
    except Exception as e:
        return f"❌ Invalid base64 content: {e}"

    from app.core.config import settings
    from app.rag.vector_store import VectorStore

    upload_dir = os.path.join("app", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)

    try:
        with open(file_path, "wb") as f:
            f.write(pdf_bytes)
    except Exception as e:
        return f"❌ Failed to save file: {e}"

    # Update config to point to new paper
    config = settings.load_rag_config()
    config["active_pdf_name"] = filename
    config["active_pdf_path"] = file_path
    settings.save_rag_config(config)

    # Clear vector store so new paper gets ingested on next query
    try:
        db = VectorStore()
        await db.delete_all()
    except Exception as e:
        logger.warning(f"Could not clear vector store: {e}")

    size_kb = len(pdf_bytes) / 1024
    return (
        f"✅ Paper uploaded successfully!\n"
        f"  **File:** {filename}\n"
        f"  **Size:** {size_kb:.1f} KB\n"
        f"  **Status:** Vector store cleared — paper will be ingested on next question."
    )


@mcp.tool()
def get_config() -> str:
    """
    Get the current RAG pipeline configuration including
    active paper name, system prompt, welcome message, and similarity metric.
    """
    from app.core.config import settings

    config = settings.load_rag_config()
    return json.dumps({
        "active_pdf_name": config.get("active_pdf_name", "Research_paper.pdf"),
        "system_prompt": config.get("system_prompt", ""),
        "welcome_message": config.get("welcome_message", ""),
        "similarity_metric": config.get("similarity_metric", "cosine"),
    }, indent=2)


@mcp.tool()
def update_config(
    system_prompt: Optional[str] = None,
    welcome_message: Optional[str] = None,
    similarity_metric: Optional[str] = None,
) -> str:
    """
    Update the RAG pipeline configuration.

    Args:
        system_prompt: New system prompt template. Use {context} and {question} placeholders.
        welcome_message: New welcome message shown to users.
        similarity_metric: Similarity metric for vector search — "cosine", "ip", or "l2".
    """
    from app.core.config import settings

    config = settings.load_rag_config()
    changes = []

    if system_prompt is not None:
        config["system_prompt"] = system_prompt
        changes.append("system_prompt")
    if welcome_message is not None:
        config["welcome_message"] = welcome_message
        changes.append("welcome_message")
    if similarity_metric is not None:
        valid_metrics = {"cosine", "ip", "l2"}
        if similarity_metric not in valid_metrics:
            return f"❌ Invalid metric '{similarity_metric}'. Must be one of: {valid_metrics}"
        config["similarity_metric"] = similarity_metric
        changes.append("similarity_metric")

    if not changes:
        return "⚠️ No changes specified. Provide at least one parameter to update."

    settings.save_rag_config(config)
    return f"✅ Configuration updated: {', '.join(changes)}"


@mcp.tool()
async def get_stats() -> str:
    """
    Get vector store statistics including the number of ingested chunks
    and the currently active paper name.
    """
    from app.core.config import settings
    from app.rag.vector_store import VectorStore

    db = VectorStore()
    count = await db.count()
    config = settings.load_rag_config()

    return json.dumps({
        "chunks_count": count,
        "active_pdf_name": config.get("active_pdf_name", "None"),
        "active_pdf_path": config.get("active_pdf_path", "None"),
        "similarity_metric": config.get("similarity_metric", "cosine"),
    }, indent=2)


@mcp.tool()
async def clear_database() -> str:
    """
    Clear all ingested data from the vector store.
    The paper will need to be re-ingested on the next question.
    """
    from app.rag.vector_store import VectorStore

    try:
        db = VectorStore()
        await db.delete_all()
        return "✅ Vector database cleared successfully. Paper will be re-ingested on next question."
    except Exception as e:
        return f"❌ Failed to clear database: {e}"


# ═════════════════════════════════════════════
#  RESOURCES
# ═════════════════════════════════════════════

@mcp.resource("paper://config")
def resource_config() -> str:
    """Current RAG pipeline configuration as JSON."""
    from app.core.config import settings

    config = settings.load_rag_config()
    return json.dumps({
        "active_pdf_name": config.get("active_pdf_name", "Research_paper.pdf"),
        "system_prompt": config.get("system_prompt", ""),
        "welcome_message": config.get("welcome_message", ""),
        "similarity_metric": config.get("similarity_metric", "cosine"),
    }, indent=2)


@mcp.resource("paper://stats")
async def resource_stats() -> str:
    """Vector store statistics: chunk count and active paper info."""
    from app.core.config import settings
    from app.rag.vector_store import VectorStore

    db = VectorStore()
    count = await db.count()
    config = settings.load_rag_config()

    return json.dumps({
        "chunks_count": count,
        "active_pdf_name": config.get("active_pdf_name", "None"),
    }, indent=2)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"🚀 Starting MCP server on http://localhost:{MCP_PORT}")

    if CLERK_JWKS_URL:
        logger.info(f"Clerk auth enabled — JWKS: {CLERK_JWKS_URL}")
    else:
        logger.warning(
            "CLERK_JWKS_URL not set in .env — running without auth (dev mode)."
        )

    try:
        mcp.run(transport="streamable-http", host="0.0.0.0", port=MCP_PORT)
    except KeyboardInterrupt:
        logger.info("👋 MCP server stopped cleanly.")

