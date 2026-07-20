from pypdf import PdfReader
import os

#To Extract Text from PDF:
class PDFLoader:
    def load_pdf(self, path: str):
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)


if __name__ == "__main__":
    print(PDFLoader().load_pdf("app\\uploads\\Research_paper.pdf"))