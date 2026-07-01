import os
import re

import pymupdf
from bs4 import BeautifulSoup
from langchain_core.documents import Document


class multiloader:
    def __init__(self, path):
        self.path = path

    def _document_loader(self):
        documents = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                filepath = os.path.join(root, file)
                doc = self._process_file(filepath)
                if doc:
                    documents.append(doc)
        return documents

    def _process_file(self, filepath):
        ext = os.path.splitext(filepath)[1].lower()
        metadata = {
            "filepath": filepath,
            "extension": ext,
            "filename": os.path.basename(filepath),
        }
        try:
            if ext == ".pdf":
                content = self._load_pdf(filepath)
            elif ext in [".html", ".htm"]:
                content = self._load_html(filepath)
            elif ext in [".md", ".txt"]:
                content = self._load_text(filepath)
            else:
                return None

            if not content:
                return None
            return Document(page_content=content, metadata=metadata)
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")
            return None

    def _load_pdf(self, filepath):
        text = ""
        with pymupdf.open(filepath) as doc:
            for page in doc:
                text += str(page.get_text("text"))
        return self._normalize_text(text)

    def _load_html(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            text = soup.get_text(separator=" ")
        return self._normalize_text(text)

    def _load_text(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return self._normalize_text(text)

    def _normalize_text(self, text):
        text = re.sub(r"\s+", " ", text)
        return text.strip()
