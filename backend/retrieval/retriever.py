from pathlib import Path
from typing import List

# --- Robust imports with backward compatibility ---
try:
    # Newer LangChain split packages
    from langchain_community.vectorstores import FAISS
except ModuleNotFoundError:
    # Older / monolithic LangChain fallback
    from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "HuggingFace embeddings backend is not available. "
        "Install it with: pip install langchain-huggingface"
    ) from e


class Retriever:
    def __init__(self, index_path: str = "data/faiss_index"):
        """
        Retriever for FAISS vector store.

        :param index_path: Path to the FAISS index directory.
                           Defaults to data/faiss_index
        """
        self.index_path = Path(index_path)

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at: {self.index_path}. "
                "Run app/ingest.py first to create the index."
            )

        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        self.db = FAISS.load_local(
            str(self.index_path),
            self.embedder,
            allow_dangerous_deserialization=True
        )

    def retrieve(self, query: str, k: int = 3):
        """
        Retrieve top-k relevant documents for a query.
        """
        return self.db.similarity_search(query, k=k)


# --------------------
# Basic self-test
# --------------------
if __name__ == "__main__":
    retriever = Retriever()
    results = retriever.retrieve("test query", k=1)
    assert isinstance(results, list), "Retriever should return a list"
    print("Retriever self-test passed ✔")
