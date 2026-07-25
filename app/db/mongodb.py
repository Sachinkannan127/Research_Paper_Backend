import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
import logging

import certifi

# Setup Logger (Critical for Cloud Debugging)
logger = logging.getLogger("uvicorn")

class Database:
    client: AsyncIOMotorClient = None

db_instance = Database()

def get_database_client():
    return db_instance.client


def get_vector_collection():
    db_name = getattr(settings, "DB_NAME", None) or "rag_db"
    return db_instance.client[db_name]["vector_documents"]


async def connect_to_mongo():
    try:
        logger.info("⏳ Connecting to MongoDB...")
        db_instance.client = AsyncIOMotorClient(
            settings.MONGO_DB_URL,
            tlsCAFile=certifi.where()
        )
        
        # THE PING TEST (Crucial for Cloud)
        await db_instance.client.admin.command('ping')
        logger.info("✅ MongoDB Connected Successfully!")
        
    except Exception as e:
        logger.error(f"❌ MongoDB Connection Failed: {e}")
        raise e


async def close_mongo_connection():
    if db_instance.client:
        db_instance.client.close()
        logger.info("🔒 MongoDB connection closed.")

if __name__ == "__main__":
    import asyncio
    
    async def test_connection():
        # Setup logging to console for testing
        logging.basicConfig(level=logging.INFO)
        # Load environment variables explicitly if running script directly
        from dotenv import load_dotenv
        load_dotenv()
        
        await connect_to_mongo()
        await close_mongo_connection()

    asyncio.run(test_connection())