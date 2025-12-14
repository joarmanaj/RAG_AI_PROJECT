import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest():
    # Folder containing your documents
    folder = "data/docs"

    # Path where the FAISS index will be saved
    index_path = Path("data/faiss_index")
    index_path.mkdir(parents=True, exist_ok=True)  # ensure the folder exists

    # Initialize embedder and text splitter
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # Load documents
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                docs.append(f.read())

    # Split documents into chunks
    chunks = splitter.create_documents(docs)

    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embedder)

    # Save index
    vectorstore.save_local(str(index_path))
    print("Index created at:", index_path.resolve())

if __name__ == "__main__":
    ingest()
