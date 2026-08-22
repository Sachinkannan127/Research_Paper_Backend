import os
import shutil
import time
import asyncio
import httpx
import jwt
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, RedirectResponse
from app.core.security import get_current_user
from typing import Optional
from pydantic import BaseModel
from app.core.config import settings
from app.rag.vector_store import VectorStore
from app.rag.chunk import chunk_text
from app.rag.embeddings import EmbeddingModel
from app.db.mongodb import get_user_collection

router = APIRouter(prefix="/config", tags=["Configuration"])

class ConfigUpdate(BaseModel):
    system_prompt: str
    welcome_message: str
    similarity_metric: str = "cosine"
    github_token: Optional[str] = None
    slack_token: Optional[str] = None
    slack_team_id: Optional[str] = None
    gmail_client_id: Optional[str] = None
    gmail_client_secret: Optional[str] = None
    gmail_refresh_token: Optional[str] = None
    apify_token: Optional[str] = None

def obfuscate_token(token: str | None) -> str:
    if not token:
        return ""
    if len(token) <= 8:
        return "*" * len(token)
    return token[:4] + "*****" + token[-4:]

def is_obfuscated(token: str | None) -> bool:
    if not token:
        return False
    return "*****" in token or token.startswith("****")

@router.get("/stats", dependencies=[Depends(get_current_user)])
async def get_stats():
    db = VectorStore()
    count = await db.count()
    return {
        "chunks_count": count
    }

@router.get("", dependencies=[Depends(get_current_user)])
def get_config(current_user: dict = Depends(get_current_user)):
    config = settings.load_rag_config()
    
    # Retrieve user-specific tokens from MongoDB user doc ONLY
    github_token = current_user.get("github_token")
    slack_token = current_user.get("slack_token")
    slack_team_id = current_user.get("slack_team_id")
    gmail_client_id = current_user.get("gmail_client_id")
    gmail_client_secret = current_user.get("gmail_client_secret")
    gmail_refresh_token = current_user.get("gmail_refresh_token")
    apify_token = current_user.get("apify_token")
    
    return {
        "active_pdf_name": config.get("active_pdf_name", "Research_paper.pdf"),
        "system_prompt": config.get("system_prompt", ""),
        "welcome_message": config.get("welcome_message", ""),
        "similarity_metric": config.get("similarity_metric", "cosine"),
        "github_token": obfuscate_token(github_token),
        "slack_token": obfuscate_token(slack_token),
        "slack_team_id": slack_team_id or "",
        "gmail_client_id": obfuscate_token(gmail_client_id),
        "gmail_client_secret": obfuscate_token(gmail_client_secret),
        "gmail_refresh_token": obfuscate_token(gmail_refresh_token),
        "apify_token": obfuscate_token(apify_token)
    }

@router.post("", dependencies=[Depends(get_current_user)])
async def update_config(data: ConfigUpdate, current_user: dict = Depends(get_current_user)):
    config = settings.load_rag_config()
    old_metric = config.get("similarity_metric", "cosine")
    
    config["system_prompt"] = data.system_prompt
    config["welcome_message"] = data.welcome_message
    if data.similarity_metric:
        config["similarity_metric"] = data.similarity_metric
        
    settings.save_rag_config(config)

    # Save user-specific fields to user doc in DB
    users_coll = get_user_collection()
    clerk_id = current_user.get("clerk_id")
    
    user_updates = {}
    if data.github_token is not None and not is_obfuscated(data.github_token):
        user_updates["github_token"] = data.github_token
    elif data.github_token == "":
        user_updates["github_token"] = None

    if data.slack_token is not None and not is_obfuscated(data.slack_token):
        user_updates["slack_token"] = data.slack_token
    elif data.slack_token == "":
        user_updates["slack_token"] = None
        
    if data.slack_team_id is not None:
        user_updates["slack_team_id"] = data.slack_team_id
        
    if data.gmail_client_id is not None and not is_obfuscated(data.gmail_client_id):
        user_updates["gmail_client_id"] = data.gmail_client_id
    elif data.gmail_client_id == "":
        user_updates["gmail_client_id"] = None
        
    if data.gmail_client_secret is not None and not is_obfuscated(data.gmail_client_secret):
        user_updates["gmail_client_secret"] = data.gmail_client_secret
    elif data.gmail_client_secret == "":
        user_updates["gmail_client_secret"] = None
        
    if data.gmail_refresh_token is not None and not is_obfuscated(data.gmail_refresh_token):
        user_updates["gmail_refresh_token"] = data.gmail_refresh_token
    elif data.gmail_refresh_token == "":
        user_updates["gmail_refresh_token"] = None
        
    if data.apify_token is not None and not is_obfuscated(data.apify_token):
        user_updates["apify_token"] = data.apify_token
    elif data.apify_token == "":
        user_updates["apify_token"] = None
        
    if user_updates:
        await users_coll.update_one({"clerk_id": clerk_id}, {"$set": user_updates})
        
        # Trigger reload of MCP Connectors for this user
        try:
            from app.mcp.client_manager import mcp_client_manager
            updated_user = await users_coll.find_one({"clerk_id": clerk_id})
            asyncio.create_task(mcp_client_manager.reload_for_user(clerk_id, updated_user))
        except Exception as reload_err:
            print(f"[Config] Failed to reload MCP Connectors for user: {reload_err}")
            
    if data.similarity_metric and data.similarity_metric != old_metric:
        try:
            db = VectorStore()
            await db.recreate_collection(data.similarity_metric)
            print(f"[Config] Recreated collection with new similarity space: {data.similarity_metric}")
        except Exception as e:
            print(f"[Config] Failed to recreate collection: {e}")
            
    return {"status": "success", "message": "Configuration updated successfully"}


@router.post("/upload", dependencies=[Depends(get_current_user)])
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

@router.post("/clear-database", dependencies=[Depends(get_current_user)])
async def clear_database():
    try:
        db = VectorStore()
        await db.delete_all()
        return {"status": "success", "message": "Vector database cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")

@router.get("/ingest/stream", dependencies=[Depends(get_current_user)])
async def ingest_stream():
    async def _stream_ingest():
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
            embeddings_service = EmbeddingModel()
            embeddings = await embeddings_service.embed_texts(chunks)
            yield "__STEP__:embedding:done\n"
            
            # Step 4: Vector store
            yield "__STEP__:vector_store:active\n"
            time.sleep(0.3)
            
            # Clear database first so we don't mix documents
            await db.delete_all()
            
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"source": pdf_path, "page": i + 1} for i in range(len(chunks))]
            
            await db.add_documents(
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


@router.post("/github/authorize")
def github_authorize(request: Request, current_user: dict = Depends(get_current_user)):
    client_id = os.getenv("GITHUB_CLIENT_ID")
    if not client_id:
        print("[Config] GitHub Client ID missing in environment")
        raise HTTPException(status_code=400, detail="GitHub OAuth Client ID is missing in environment")
    
    clerk_id = current_user.get("clerk_id")
    
    # Determine the redirect URI (override via env, or construct dynamically)
    redirect_uri = os.getenv("GITHUB_REDIRECT_URI")
    if not redirect_uri:
        backend_url = os.getenv("BACKEND_URL")
        if not backend_url:
            # Fallback to request's base URL, respecting proxy headers if possible
            proto = request.headers.get("x-forwarded-proto", "http")
            host = request.headers.get("x-forwarded-host") or request.url.netloc
            backend_url = f"{proto}://{host}"
        redirect_uri = f"{backend_url.rstrip('/')}/config/github/callback"
        
    print(f"[Config] GitHub Authorize: client_id={client_id}, redirect_uri={redirect_uri}")
    auth_url = f"https://github.com/login/oauth/authorize?client_id={client_id}&redirect_uri={redirect_uri}&scope=repo,user&prompt=select_account"
    if clerk_id:
        auth_url += f"&state={clerk_id}"
        
    return {"url": auth_url}


@router.get("/github/callback")
async def github_callback(code: str, state: Optional[str] = None):
    client_id = os.getenv("GITHUB_CLIENT_ID")
    client_secret = os.getenv("GITHUB_CLIENT_SECRET")
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    if not client_id or not client_secret:
        print("[Config] GitHub client credentials missing in environment")
        return RedirectResponse(url=f"{frontend_url}/workspace/connectors?error=oauth_not_configured")
        
    token_url = "https://github.com/login/oauth/access_token"
    headers = {"Accept": "application/json"}
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(token_url, json=payload, headers=headers)
            res_data = response.json()
            
            if "access_token" in res_data:
                access_token = res_data["access_token"]
                
                # Update user-specifically if state is present, otherwise fallback to global
                if state:
                    users_coll = get_user_collection()
                    await users_coll.update_one(
                        {"clerk_id": state},
                        {"$set": {"github_token": access_token}}
                    )
                    # Trigger reload of MCP Connectors for this user
                    try:
                        from app.mcp.client_manager import mcp_client_manager
                        updated_user = await users_coll.find_one({"clerk_id": state})
                        asyncio.create_task(mcp_client_manager.reload_for_user(state, updated_user))
                    except Exception as reload_err:
                        print(f"[Config] Failed to reload MCP Connectors for user: {reload_err}")
                else:
                    config = settings.load_rag_config()
                    config["github_token"] = access_token
                    settings.save_rag_config(config)
                    try:
                        from app.mcp.client_manager import mcp_client_manager
                        asyncio.create_task(mcp_client_manager.reload())
                    except Exception as reload_err:
                        print(f"[Config] Failed to reload MCP Connectors: {reload_err}")
                
                return RedirectResponse(url=f"{frontend_url}/workspace/connectors?success=github_connected")
            else:
                error_desc = res_data.get("error_description", "Failed to retrieve access token")
                print(f"[Config] GitHub OAuth token exchange failed: {res_data}")
                return RedirectResponse(url=f"{frontend_url}/workspace/connectors?error={error_desc}")
    except Exception as e:
        print(f"[Config] Exception during GitHub OAuth callback: {e}")
        return RedirectResponse(url=f"{frontend_url}/workspace/connectors?error={str(e)}")
