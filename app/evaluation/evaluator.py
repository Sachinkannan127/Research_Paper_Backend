import os
import sys

# Ensure the project root is in the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets import Dataset

from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.core.config import settings


class RagasEvaluator:

    def __init__(self):
        self.default_embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
        llm=None,
        embeddings=None,
    ):

        dataset = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth],
            }
        )

        if llm is None:
            # Use Groq's OpenAI-compatible endpoint
            llm = ChatOpenAI(
                model="llama-3.1-8b-instant",
                base_url="https://api.groq.com/openai/v1",
                api_key=settings.GROQ_API_KEY,
            )

        if embeddings is None:
            embeddings = self.default_embeddings

        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=llm,
            embeddings=embeddings,
        )

        return result