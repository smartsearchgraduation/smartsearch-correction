"""
SmartSearch Correction API
Port: 5001
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from app.typo_corrector import TypoCorrector, CorrectionModel

# Initialize
app = FastAPI(
    title="SmartSearch Correction API",
    description="E-commerce search query spell correction",
    version="2.0"
)

# CORS - backend'in erişebilmesi için
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load corrector once at startup
corrector = TypoCorrector()


# Request/Response models
class CorrectionRequest(BaseModel):
    query: str
    model: Optional[str] = "symspell"  # symspell, keyboard, byt5


class CorrectionResponse(BaseModel):
    original_query: str
    corrected_query: str
    changed: bool
    model_used: str
    latency_ms: float


# Endpoints
@app.get("/")
def root():
    return {"service": "correction", "status": "running", "port": 5001}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/correct", response_model=CorrectionResponse)
def correct_query(request: CorrectionRequest):
    """
    Correct a search query for typos.
    
    Models:
    - symspell: Fast dictionary-based (~0.1ms)
    - keyboard: Keyboard proximity-aware (~5ms)
    - byt5: ByT5 fine-tuned model (~180ms)
    """
    try:
        result = corrector.correct(request.query, model=request.model)
        
        return CorrectionResponse(
            original_query=result["original_query"],
            corrected_query=result["normalized_query"],
            changed=result["changed"],
            model_used=request.model,
            latency_ms=result["latency"]["total_ms"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/correct", response_model=CorrectionResponse)
def correct_query_get(query: str, model: Optional[str] = "symspell"):
    """
    Correct a search query for typos (GET version for backend integration).
    
    Parameters:
    - query: The search query to correct
    - model: Correction model (symspell, keyboard, byt5)
    
    Example: GET /correct?query=rg%20hedseet&model=symspell
    """
    try:
        result = corrector.correct(query, model=model)
        
        return CorrectionResponse(
            original_query=result["original_query"],
            corrected_query=result["normalized_query"],
            changed=result["changed"],
            model_used=model,
            latency_ms=result["latency"]["total_ms"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
def list_models():
    """List available correction models."""
    return {
        "models": [
            {"name": "symspell", "latency": "~0.1ms", "accuracy": "96%", "description": "Fast dictionary-based"},
            {"name": "keyboard", "latency": "~5ms", "accuracy": "89%", "description": "Keyboard proximity-aware"},
            {"name": "byt5", "latency": "~180ms", "accuracy": "79%", "description": "ByT5 fine-tuned model"},
        ],
        "default": "symspell"
    }


if __name__ == "__main__":
    print("🚀 Starting Correction API on port 5001...")
    uvicorn.run(app, host="0.0.0.0", port=5001)
