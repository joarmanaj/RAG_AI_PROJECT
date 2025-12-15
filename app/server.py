from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.retrieval.retriever import Retriever
from backend.llms.router import LLMRouter
import os
import uvicorn

app = FastAPI(title="Joarmanaj RAG API")

# -----------------------
# Initialize services
# -----------------------
retriever = Retriever()
llm = LLMRouter()

# -----------------------
# Request/Response Models
# -----------------------
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# -----------------------
# API Endpoints
# -----------------------
@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    try:
        docs = retriever.retrieve(request.question)
        context_docs = [d.page_content for d in docs]
        answer = llm.generate(request.question, context_docs)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """
    Simple health check for Render or monitoring.
    """
    return {
        "status": "ok",
        "python_version": os.sys.version
    }

# -----------------------
# Render / Local Entry
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render automatically sets PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
