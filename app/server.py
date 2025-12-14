from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.retrieval.retriever import Retriever
from backend.llms.router import LLMRouter
import os
import uvicorn

app = FastAPI(title="Joarmanaj RAG API")

# Initialize retriever and LLM router
retriever = Retriever()
llm = LLMRouter()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    try:
        docs = retriever.retrieve(request.question)
        context_docs = [d.page_content for d in docs]
        answer = llm.generate(request.question, context_docs)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# Render / local entry
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render sets PORT automatically
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
