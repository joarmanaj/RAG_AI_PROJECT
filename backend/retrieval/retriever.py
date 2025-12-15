# backend/retrieval/retriever.py

from pathlib import Path
from typing import List, Optional

# --- Robust imports for FAISS ---
try:
    # Newer LangChain split packages
    from langchain_community.vectorstores import FAISS
except ModuleNotFoundError:
    # Older monolithic LangChain fallback
    from langchain.vectorstores import FAISS

try:
    # HuggingFace embeddings
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:
    # Fallback if package not installed
    raise ImportError(
        "Please install langchain_huggingface via pip: pip install langchain-huggingface"
    )


class Retriever:
    """
    FAISS Retriever for your RAG setup.

    Attributes:
        index_path (str): Path to your FAISS index directory.
        embeddings: HuggingFace embeddings object.
    """

    def __init__(self, index_path: str = "data/faiss_index"):
        self.index_path = Path(index_path)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore: Optional[FAISS] = None

        if self.index_path.exists():
            self.load_index()
        else:
            print(f"[Retriever] FAISS index not found at {self.index_path}, will need to create it.")

    def load_index(self):
        """
        Load FAISS index from disk.
        """
        try:
            self.vectorstore = FAISS.load_local(str(self.index_path), self.embeddings)
            print(f"[Retriever] Loaded FAISS index from {self.index_path}")
        except Exception as e:
            print(f"[Retriever] Failed to load FAISS index: {e}")
            self.vectorstore = None

    def search(self, query: str, k: int = 5) -> List[str]:
        """
        Search FAISS index for similar documents.

        Args:
            query (str): Query string.
            k (int): Number of results to return.

        Returns:
            List[str]: List of text results.
        """
        if self.vectorstore is None:
            print("[Retriever] Vectorstore not initialized. Returning empty results.")
            return []

        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return [r.page_content for r in results]
        except Exception as e:
            print(f"[Retriever] Error during search: {e}")
            return []
