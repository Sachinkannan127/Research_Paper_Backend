import asyncio
import os
import sys

# Ensure backend_llm workspace root and app path are in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from app.routes.stream import _stream_answer
from app.db.mongodb import connect_to_mongo, close_mongo_connection

async def async_main():
    print("Connecting to MongoDB...")
    await connect_to_mongo()
    try:
        print("Starting generator test...")
        generator = _stream_answer(model_name="fast", question="Hello")
        async for chunk in generator:
            print(f"YIELDED: {repr(chunk)}")
        print("Generator completed successfully.")
    finally:
        print("Closing MongoDB connection...")
        await close_mongo_connection()

if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except Exception as e:
        import traceback
        print("Generator raised an exception:")
        traceback.print_exc()
