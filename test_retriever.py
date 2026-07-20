from app.rag.retriever import Retriever


def main():
    # Create Retriever object
    retriever = Retriever()

    # User Question
    question = input("Enter your question: ")

    print("=" * 60)
    print("Question:")
    print(question)
    print("=" * 60)

    # Retrieve relevant chunks
    results = retriever.retrieve(
        question=question,
        top_k=1
    )

    # Print retrieved chunks
    if not results:
        print("No relevant documents found.")
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


if __name__ == "__main__":
    main()