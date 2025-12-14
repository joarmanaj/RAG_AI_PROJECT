from backend.llms.hf_llm import HuggingFaceLLM

class LLMRouter:
    """
    CPU/GPU-safe LLM router (fallback only for Render)
    """

    def __init__(self):
        # Use HuggingFace GPT2-medium as default
        self.fallback = HuggingFaceLLM(max_new_tokens=150)

    def generate(self, prompt: str, context_docs):
        try:
            print("[LLM Router] Using HuggingFace (CPU/GPU)...")
            return self.fallback.generate(prompt, context_docs)
        except Exception as e:
            print(f"[LLM Router] HuggingFace failed\n{e}")
            return "I don't know."
