from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.text_extract import PDFLoader


PDF_PATH = "app\\uploads\\Research_paper.pdf"


def chunk_text(path: str):
    text = PDFLoader().load_pdf(path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


if __name__ == "__main__":
    for index, chunk in enumerate(chunk_text(PDF_PATH), start=1):
        print(f"\n--- Chunk {index} ---\n{chunk}") 