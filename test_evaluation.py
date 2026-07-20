# from app.evaluation.evaluator import RagasEvaluator

# from app.rag.retriever import Retriever

# from app.routes.chat import ChatService


# question = "What is Retrieval Augmented Generation?"

# chat = ChatService(question=question, model_name="fast")

# evaluator = RagasEvaluator()

# answer = chat["answer"]
# retrieved_chunks = chat["retrieved_chunks"]

# contexts = []

# for chunk in retrieved_chunks:

#     contexts.append(chunk["text"])


# ground_truth = """
# Retrieval-Augmented Generation (RAG)
# combines vector retrieval with an LLM to
# generate answers from retrieved documents.
# """


# result = evaluator.evaluate(
#     question=question,
#     answer=answer,
#     contexts=contexts,
#     ground_truth=ground_truth,
# )

# print(result)