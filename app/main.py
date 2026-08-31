from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

from app.routes.chat import ChatService, router
from app.routes.stream import router as stream_router
from app.routes.config import router as config_router
# from app.routes.voice import router as voice_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    from app.db.mongodb import connect_to_mongo, close_mongo_connection
    from app.mcp.client_manager import mcp_client_manager
    try:
        await connect_to_mongo()
    except Exception as e:
        print(f"Failed to connect to MongoDB on startup: {e}")
    try:
        await mcp_client_manager.initialize()
    except Exception as e:
        print(f"Failed to initialize MCP client connectors: {e}")
    yield
    await mcp_client_manager.shutdown()
    await close_mongo_connection()
    print("Shutting down...")

app = FastAPI(title="Research Paper Assistant", version="1.0.0", lifespan=lifespan)

# Allow specific origins (Vercel production frontend, Render backend, and local dev server)
origins = [
    "https://research-paper-backend-1ub4.onrender.com",
    "https://research-paper-frontend-sable.vercel.app",
    "https://research-paper-frontend-sable.vercel.app/",
    "https://research-paper-frontend-theta.vercel.app",
    "https://research-paper-frontend-theta.vercel.app/",
    "https://research-paper-frontend-seven.vercel.app",
    "https://research-paper-assistant-ruddy.vercel.app",
    "https://research-paper-assistant-ruddy.vercel.app/",
    "http://localhost:3001",
    "https://localhost:3001",
    "http://127.0.0.1:3001",
    "https://127.0.0.1:3001",
    "http://localhost:3000",
    "https://localhost:3000",
    "http://127.0.0.1:3000",
    "https://127.0.0.1:3000",
    "http://localhost:5173",
    "https://localhost:5173", ]

# Allow any additional custom origins defined in environment variables
allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env:
    for origin in allowed_origins_env.split(","):
        trimmed = origin.strip()
        if trimmed and trimmed not in origins:
            origins.append(trimmed)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex="https://.*\\.vercel\\.app",  # Matches any Vercel deployment URL (previews, branches)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/")
def landing_page():
    return {
        "message": "I am your Research Paper Asssistant Backend Server"
    }

@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    return {
        "status": "Naa Nalla Irukken"
    }

from app.routes.auth import router as auth_router
from app.core.security import get_current_user
from fastapi import Depends

app.include_router(auth_router)
app.include_router(router, dependencies=[Depends(get_current_user)])
app.include_router(stream_router, dependencies=[Depends(get_current_user)])
app.include_router(config_router)
# app.include_router(voice_router)