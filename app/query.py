from backend.retrieval.retriever import Retriever
from backend.llms.router import LLMRouter

def main():
    retriever = Retriever()
    llm = LLMRouter(use_ollama=True)  # Set True if your Ollama server is running

    print("🔹 Local RAG CLI (type 'exit' to quit)\n")

    while True:
        question = input("You: ")
        if question.lower() in {"exit", "quit"}:
            break

        docs = retriever.retrieve(question)
        context_docs = [d.page_content for d in docs]

        try:
            answer = llm.generate(question, context_docs)
            print(f"\n🤖 Answer:\n{answer}\n")
        except Exception as e:
            print(f"\n❌ Error generating answer: {e}\n")

if __name__ == "__main__":
    main()
