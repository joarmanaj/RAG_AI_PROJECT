import requests

class OllamaLLM:
    """
    Ollama LLM wrapper for local API (primary LLM)
    """

    def __init__(self, model: str = "phi:latest", base_url: str = "http://localhost:11435"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, context_docs):
        """
        Generate professional, human-like answers using Ollama.
        """
        # Combine retrieved documents
        context = "\n\n".join(context_docs)

        # Instruction-style prompt (matches HF LLM style)
        full_prompt = f"""
You are a professional digital marketing assistant for Joarmanaj Agency, answer the user's question using ONLY the context below.
Answer the user's question in a clear, concise, and professional manner (2-5 sentences).
Do NOT hallucinate; if the answer is not in the context, say "I don't know."

--- CONTEXT ---
{context}
--- CONTEXT ---

User Question:
{prompt}

Answer:
""".strip()

        # Call Ollama API
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "num_predict": 200,
                "temperature": 0.2
            },
            timeout=120
        )

        response.raise_for_status()
        return response.json()["response"].split("Answer:")[-1].strip()
