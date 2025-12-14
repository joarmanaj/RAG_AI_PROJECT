"""
HuggingFace Local LLM (Fallback)
CPU/GPU-friendly, instruction-tuned for professional responses.
Model: GPT2-medium (small, portable)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class HuggingFaceLLM:
    def __init__(self, model_name: str = "gpt2-medium", max_new_tokens: int = 150):
        """
        :param model_name: HuggingFace model name (small, portable)
        :param max_new_tokens: maximum tokens for generation
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load(self):
        if self.model is not None:
            return
        print(f"[HuggingFaceLLM] Loading model '{self.model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.model.to(self.device)

    def generate(self, question: str, context_docs):
        """
        Generate professional answers using retrieved context.
        """
        self._load()

        # Combine retrieved documents
        context = "\n\n".join(context_docs)

        # Instruction prompt
        full_prompt = f"""
You are a professional digital marketing assistant for Joarmanaj Agency.
Use ONLY the context below to answer the user's question in a clear, concise, and professional manner (2-5 sentences).
Do NOT hallucinate; if the answer is not in the context, say "I don't know."

--- CONTEXT ---
{context}
--- CONTEXT ---

User Question:
{question}

Answer:
""".strip()

        # Tokenize and move to device
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

        # Generate output
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract only the answer portion
        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()
        return response.strip()
