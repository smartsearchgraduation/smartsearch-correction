"""
SmartSearch Correction API
Port: 5001
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import time

from app.typo_corrector import TypoCorrector, CorrectionModel

# Baseline corrector import
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))
from baseline_vs_offline import BaselineSymSpell

# =============================================================================
# ⚙️ DEFAULT MODEL CONFIG - Buradan değiştir!
# =============================================================================
# Seçenekler:
#   - "symspell_keyboard" : Hızlı (~0.1ms), Production için önerilen
#   - "byt5"              : Yavaş (~300-500ms), Maximum accuracy
# =============================================================================
DEFAULT_MODEL = "symspell_keyboard"
# =============================================================================

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
baseline_corrector = BaselineSymSpell()


# Request/Response models
class CorrectionRequest(BaseModel):
    query: str
    model: Optional[str] = None  # None = DEFAULT_MODEL kullanılır


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


# =============================================================================
# PRODUCTION MODELS (for backend integration)
# These are the only models backend should use
# =============================================================================
PRODUCTION_MODELS = ["symspell_keyboard", "byt5"]

# =============================================================================
# ALL MODELS (for testing/documentation)
# =============================================================================
ALL_MODELS = [
    "symspell", "baseline", "byt5", "offline",
    "keyboard", "symspell_keyboard", "e5_onnx", 
    "qwen", "qwen-1.5b", "llama", "smart_hybrid"
]


@app.post("/correct", response_model=CorrectionResponse)
def correct_query(request: CorrectionRequest):
    """
    Correct a search query for typos.
    
    🚀 PRODUCTION MODELS (for backend):
    - symspell_keyboard: SymSpell + Keyboard hybrid (~0.1ms, ~98%)
    - byt5: ByT5 fine-tuned (~300ms, 100%) - Maximum accuracy
    
    Default model is set in DEFAULT_MODEL config at top of file.
    
    🧪 Testing Models (for documentation/benchmarks):
    - baseline, keyboard, qwen, qwen-1.5b, llama, smart_hybrid
    
    Note: 'symspell' is automatically mapped to 'symspell_keyboard'
    """
    # Use DEFAULT_MODEL if not specified
    model = (request.model or DEFAULT_MODEL).lower()
    
    # Map legacy/shorthand names to actual models
    if model in ["offline", "symspell"]:
        model = "symspell_keyboard"  # symspell now always uses keyboard hybrid
    
    if model not in ALL_MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{request.model}' not available. Use one of: {ALL_MODELS}")
    
    try:
        # Baseline uses separate corrector
        if model == "baseline":
            start = time.time()
            corrected = baseline_corrector.correct_query(request.query)
            latency = (time.time() - start) * 1000
            return CorrectionResponse(
                original_query=request.query,
                corrected_query=corrected,
                changed=request.query.lower() != corrected.lower(),
                model_used="baseline",
                latency_ms=latency
            )
        
        # Other models use main corrector
        result = corrector.correct(request.query, model=model)
        
        # Handle different response formats from different models
        if "latency" in result and isinstance(result["latency"], dict):
            latency = result["latency"]["total_ms"]
        else:
            latency = result.get("latency_ms", 0.0)
        
        return CorrectionResponse(
            original_query=result["original_query"],
            corrected_query=result["normalized_query"],
            changed=result["changed"],
            model_used=model,
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/correct", response_model=CorrectionResponse)
def correct_query_get(query: str, model: Optional[str] = None):
    """
    Correct a search query for typos (GET version for backend integration).
    
    Parameters:
    - query: The search query to correct
    - model: Correction model (optional, uses DEFAULT_MODEL if not specified)
    
    Production Models: symspell_keyboard, byt5
    Testing Models: baseline, keyboard, qwen, qwen-1.5b, llama, smart_hybrid
    
    Example: GET /correct?query=samsugn%20galxy
    Example: GET /correct?query=samsugn%20galxy&model=byt5
    """
    # Use DEFAULT_MODEL if not specified
    model = (model or DEFAULT_MODEL).lower()
    
    # Map legacy/shorthand names to actual models
    if model in ["offline", "symspell"]:
        model = "symspell_keyboard"  # symspell now always uses keyboard hybrid
    
    if model not in ALL_MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{model}' not available. Use one of: {ALL_MODELS}")
    
    try:
        # Baseline uses separate corrector
        if model == "baseline":
            start = time.time()
            corrected = baseline_corrector.correct_query(query)
            latency = (time.time() - start) * 1000
            return CorrectionResponse(
                original_query=query,
                corrected_query=corrected,
                changed=query.lower() != corrected.lower(),
                model_used="baseline",
                latency_ms=latency
            )
        
        # Other models use main corrector
        result = corrector.correct(query, model=model)
        
        # Handle different response formats from different models
        if "latency" in result and isinstance(result["latency"], dict):
            latency = result["latency"]["total_ms"]
        else:
            latency = result.get("latency_ms", 0.0)
        
        return CorrectionResponse(
            original_query=result["original_query"],
            corrected_query=result["normalized_query"],
            changed=result["changed"],
            model_used=model,
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
def list_models():
    """List available correction models for backend integration."""
    return {
        "production_models": [
            {"name": "symspell_keyboard", "latency": "~0.1ms", "accuracy": "~98%", "description": "SymSpell + Keyboard hybrid (fast)"},
            {"name": "byt5", "latency": "~300ms", "accuracy": "100%", "description": "ByT5 fine-tuned (maximum accuracy)"},
        ],
        "testing_models": [
            {"name": "baseline", "latency": "~0.1ms", "accuracy": "93%", "description": "Basic SymSpell (legacy)"},
            {"name": "keyboard", "latency": "~1ms", "accuracy": "~90%", "description": "Keyboard proximity only"},
            {"name": "qwen", "latency": "~300ms", "accuracy": "~80%", "description": "Qwen2.5-0.5B fine-tuned"},
            {"name": "qwen-1.5b", "latency": "~2s", "accuracy": "81%", "description": "Qwen2.5-1.5B fine-tuned"},
            {"name": "llama", "latency": "~260ms", "accuracy": "84%", "description": "Llama-3.2-1B fine-tuned"},
            {"name": "smart_hybrid", "latency": "~300ms", "accuracy": "~95%", "description": "SymSpell + LLM fallback"},
        ],
        "current_default": DEFAULT_MODEL,
        "note": "Default model can be changed in api.py -> DEFAULT_MODEL"
    }


@app.get("/correct/baseline")
def correct_baseline(query: str):
    """
    Correct a search query using Baseline SymSpell.
    """
    try:
        start = time.time()
        corrected = baseline_corrector.correct_query(query)
        latency = (time.time() - start) * 1000
        
        return CorrectionResponse(
            original_query=query,
            corrected_query=corrected,
            changed=query.lower() != corrected.lower(),
            model_used="baseline",
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("🚀 Starting Correction API on port 5001...")
    uvicorn.run(app, host="0.0.0.0", port=5001)
