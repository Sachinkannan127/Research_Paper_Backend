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
    try:
        await connect_to_mongo()
    except Exception as e:
        print(f"Failed to connect to MongoDB on startup: {e}")
    yield
    await close_mongo_connection()
    print("Shutting down...")

app = FastAPI(title="Research Paper Assistant", version="1.0.0", lifespan=lifespan)

# Allow specific origins (Vercel production frontend, Render backend, and local dev server)
origins = [
    "https://research-paper-assistant-ylic.onrender.com",
    "https://research-paper-frontend-sable.vercel.app",
    "https://research-paper-frontend-sable.vercel.app/",
    "http://localhost:3000",
    "https://localhost:3000",
    "http://127.0.0.1:3000",
    "https://127.0.0.1:3000",
]

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

app.include_router(router)
app.include_router(stream_router)
app.include_router(config_router)
# app.include_router(voice_router)