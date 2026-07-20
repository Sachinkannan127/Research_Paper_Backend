import os
from app.rag.chunk import chunk_text
from app.rag.embeddings import Embeddings
from app.rag.vector_store import VectorStore

def main():
    pdf_path = os.path.join("app", "uploads", "Research_paper.pdf")
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print("Chunking PDF...")
    chunks = chunk_text(pdf_path)
    print(f"Total chunks: {len(chunks)}")

    print("Generating embeddings...")
    embeddings_service = Embeddings()
    embeddings = embeddings_service.embed_texts(chunks)

    print("Adding to Vector Store...")
    db = VectorStore()
    
    # Clean old collection data to avoid duplicates/conflicts
    db.delete_all()
    
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    # Page numbers can be dummy/fallback or actual pages. We default to page index.
    metadatas = [{"source": pdf_path, "page": i + 1} for i in range(len(chunks))]

    db.add_documents(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print("Ingestion complete!")
    print("Total chunks in DB:", db.count())

    # Verify retrieval
    print("\n--- Testing Retrieval ---")
    from app.rag.retriever import Retriever
    retriever = Retriever()
    test_query = "What is the purpose of the paper?"
    print(f"Query: {test_query}\n")
    
    results = retriever.retrieve(test_query, top_k=0)
    for index, chunk in enumerate(results, start=1):
        print(f"Result {index}:")
        print(f"  Source : {chunk.get('source')}")
        print(f"  Page   : {chunk.get('page')}")
        print(f"  Score  : {chunk.get('score')}")
        print("  Text snippet:")
        print(f"    {chunk.get('text')[:300]}...\n")

if __name__ == "__main__":
    main()
