import asyncio
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.rag.chunk import chunk_text
from app.rag.embeddings import Embeddings
from app.rag.vector_store import VectorStore
from app.db.mongodb import connect_to_mongo, close_mongo_connection

async def main():
    pdf_path = os.path.join("app", "uploads", "Research_paper.pdf")
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print("Connecting to MongoDB...")
    await connect_to_mongo()

    print("Chunking PDF...")
    chunks = chunk_text(pdf_path)
    print(f"Total chunks: {len(chunks)}")

    print("Generating embeddings...")
    embeddings_service = Embeddings()
    embeddings = await embeddings_service.embed_texts(chunks)

    print("Adding to Vector Store...")
    db = VectorStore()
    
    # Clean old collection data to avoid duplicates/conflicts
    await db.delete_all()
    
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": pdf_path, "page": i + 1} for i in range(len(chunks))]

    await db.add_documents(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print("Ingestion complete!")
    print("Total chunks in DB:", await db.count())

    # Verify retrieval
    print("\n--- Testing Retrieval ---")
    from app.rag.retriever import Retriever
    retriever = Retriever()
    test_query = "What is the purpose of the paper?"
    print(f"Query: {test_query}\n")
    
    # top_k=3 to query instead of 0
    results = await retriever.retrieve(test_query, top_k=3)
    for index, chunk in enumerate(results, start=1):
        print(f"Result {index}:")
        print(f"  Source : {chunk.get('source')}")
        print(f"  Page   : {chunk.get('page')}")
        print(f"  Score  : {chunk.get('score')}")
        print("  Text snippet:")
        print(f"    {chunk.get('text')[:300]}...\n")

    print("Closing connection...")
    await close_mongo_connection()

if __name__ == "__main__":
    asyncio.run(main())
