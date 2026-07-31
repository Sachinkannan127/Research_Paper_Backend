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

# Allow specific origins dynamically (Vercel subdomains, Render subdomains, and local dev servers)
allow_origin_regex = r"^(https://.*\.vercel\.app|https://.*\.onrender\.com|https?://localhost(:\d+)?|https?://127\.0\.0\.1(:\d+)?)$"

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=allow_origin_regex,
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