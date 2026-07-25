import asyncio
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.rag.vector_store import VectorStore
from app.db.mongodb import connect_to_mongo, close_mongo_connection

async def test():
    print("Connecting to MongoDB...")
    await connect_to_mongo()
    
    db = VectorStore()

    count = await db.count()
    print("Total Chunks:", count)

    data = await db.get_all()

    print("\nIDs:")
    print(data["ids"])

    print("\nDocuments (Snippets):")
    for doc in data["documents"]:
        print(f"  - {doc[:100]}...")

    print("\nChunks (Snippets):")
    for chunk in data.get("chunks", []):
        print(f"  - {chunk[:100]}...")

    print("\nEmbeddings (Dimensions and Snippets):")
    for emb in data.get("embeddings", []):
        print(f"  - Length: {len(emb)}, Snippet: {emb[:5]}")

    print("\nMetadata:")
    for meta in data["metadatas"]:
        print(f"  - {meta}")
        
    print("Closing connection...")
    await close_mongo_connection()

if __name__ == "__main__":
    asyncio.run(test())