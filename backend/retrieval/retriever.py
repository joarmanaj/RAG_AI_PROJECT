# backend/retrieval/retriever.py

from pathlib import Path
from typing import List, Optional

# --- Robust imports for FAISS ---
try:
    from langchain_community.vectorstores import FAISS
except ModuleNotFoundError:
    from langchain.vectorstores import FAISS


class Retriever:
    """
    FAISS Retriever for your RAG setup.

    Embeddings are lazy-loaded to reduce startup memory usage.
    """

    def __init__(self, index_path: str = "data/faiss_index"):
        self.index_path = Path(index_path)
        self.embeddings = None
        self.vectorstore: Optional[FAISS] = None

        if self.index_path.exists():
            self.load_index()
        else:
            print(f"[Retriever] FAISS index not found at {self.index_path}, will need to create it.")

    def _init_embeddings(self):
        """Lazy-load HuggingFace embeddings only when needed."""
        if self.embeddings is None:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ModuleNotFoundError:
                raise ImportError(
                    "Please install langchain_huggingface via pip: pip install langchain-huggingface"
                )
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

    def load_index(self):
        """Load FAISS index from disk."""
        self._init_embeddings()
        try:
            self.vectorstore = FAISS.load_local(str(self.index_path), self.embeddings)
            print(f"[Retriever] Loaded FAISS index from {self.index_path}")
        except Exception as e:
            print(f"[Retriever] Failed to load FAISS index: {e}")
            self.vectorstore = None

    def search(self, query: str, k: int = 5) -> List[str]:
        """Search FAISS index for similar documents."""
        if self.vectorstore is None:
            print("[Retriever] Vectorstore not initialized. Returning empty results.")
            return []

        self._init_embeddings()

        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return [r.page_content for r in results]
        except Exception as e:
            print(f"[Retriever] Error during search: {e}")
            return []
