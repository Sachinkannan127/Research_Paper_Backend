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
from app.routes.voice import router as voice_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    yield
    print("Shutting down...")

app = FastAPI(title="Research Paper Assistant", version="1.0.0", lifespan=lifespan)

# Allow the React frontend to access this backend from any hostname or port dynamically
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="https?://.*",
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
app.include_router(voice_router)