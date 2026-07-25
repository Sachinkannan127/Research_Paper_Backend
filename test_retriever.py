import asyncio
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.rag.retriever import Retriever
from app.db.mongodb import connect_to_mongo, close_mongo_connection


async def async_main():
    print("Connecting to MongoDB...")
    await connect_to_mongo()

    # Create Retriever object
    retriever = Retriever()

    # User Question
    question = input("Enter your question: ")

    print("=" * 60)
    print("Question:")
    print(question)
    print("=" * 60)

    # Retrieve relevant chunks
    results = await retriever.retrieve(
        question=question,
        top_k=1
    )

    # Print retrieved chunks
    if not results:
        print("No relevant documents found.")
        await close_mongo_connection()
        return

    for index, chunk in enumerate(results, start=1):

        print(f"\nChunk {index}")
        print("-" * 60)

        print(f"Source : {chunk.get('source')}")
        print(f"Page   : {chunk.get('page')}")
        print(f"Score  : {chunk.get('score')}")

        print("\nText:")
        print(chunk.get("text"))

        print("-" * 60)

    print("Closing connection...")
    await close_mongo_connection()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()