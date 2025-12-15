# backend/retrieval/retriever.py

from pathlib import Path
from typing import List, Optional

# ---- FAISS import (LangChain 1.x safe) ----
from langchain_community.vectorstores import FAISS

# ---- Embeddings import ----
from langchain_huggingface import HuggingFaceEmbeddings


class Retriever:
    """
    FAISS Retriever for RAG (Render-safe, lazy loading).
    """

    def __init__(self, index_path: str = "data/faiss_index"):
        self.index_path = Path(index_path)
        self._embeddings = None  # ⛔ DO NOT LOAD AT STARTUP
        self.vectorstore: Optional[FAISS] = None

        if self.index_path.exists():
            self.load_index()
        else:
            print(f"[Retriever] FAISS index not found at {self.index_path}")

    # ✅ Lazy embeddings loader
    @property
    def embeddings(self):
        if self._embeddings is None:
            print("[Retriever] Loading HuggingFace embeddings...")
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        return self._embeddings

    def load_index(self):
        try:
            self.vectorstore = FAISS.load_local(
                str(self.index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print("[Retriever] FAISS index loaded")
        except Exception as e:
            print(f"[Retriever] Failed to load FAISS index: {e}")
            self.vectorstore = None

    def search(self, query: str, k: int = 5) -> List[str]:
        if not self.vectorstore:
            return []

        results = self.vectorstore.similarity_search(query, k=k)
        return [r.page_content for r in results]
