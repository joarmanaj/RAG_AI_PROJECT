from backend.retrieval.retriever import Retriever

retriever = Retriever("data/index")
results = retriever.search("What is RAG?")
print(results)
