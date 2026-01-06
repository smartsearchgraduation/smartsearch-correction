"""
SmartSearch Correction API
Port: 5001
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import logging
import sys
import json
from datetime import datetime

from app.typo_corrector import TypoCorrector, CorrectionModel

# Configure logging with colors
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',      # Reset
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)

# Setup logger
logger = logging.getLogger("correction_api")
logger.setLevel(logging.DEBUG)

# Console handler with colors
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(ColoredFormatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
))
logger.addHandler(console_handler)

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
logger.info("🔧 Loading TypoCorrector...")
corrector = TypoCorrector()
logger.info("✅ TypoCorrector loaded successfully!")


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
    logger.debug("📍 GET / - Root endpoint called")
    return {"service": "correction", "status": "running", "port": 5001}


@app.get("/health")
def health():
    logger.debug("💓 GET /health - Health check")
    return {"status": "healthy"}


@app.post("/correct", response_model=CorrectionResponse)
def correct_query(request: CorrectionRequest):
    """
    Correct a search query for typos.
    Takes input as JSON body.
    
    Models:
    - symspell: Fast dictionary-based (~0.1ms)
    - keyboard: Keyboard proximity-aware (~5ms)
    - byt5: ByT5 fine-tuned model (~180ms)
    """
    logger.info("=" * 60)
    logger.info(f"📥 POST /correct - New correction request")
    
    # Log request as JSON
    request_json = {"query": request.query, "model": request.model}
    logger.info(f"📝 Request JSON:")
    logger.info(json.dumps(request_json, indent=2, ensure_ascii=False))
    
    try:
        result = corrector.correct(request.query, model=request.model)
        
        response = CorrectionResponse(
            original_query=result["original_query"],
            corrected_query=result["normalized_query"],
            changed=result["changed"],
            model_used=request.model,
            latency_ms=result["latency_ms"]
        )
        
        # Log response as JSON
        response_json = {
            "original_query": response.original_query,
            "corrected_query": response.corrected_query,
            "changed": response.changed,
            "model_used": response.model_used,
            "latency_ms": response.latency_ms
        }
        logger.info(f"📤 Response JSON:")
        logger.info(json.dumps(response_json, indent=2, ensure_ascii=False))
        
        if response.changed:
            logger.info(f"✅ '{response.original_query}' → '{response.corrected_query}' ({response.latency_ms:.2f}ms)")
        else:
            logger.info(f"ℹ️  No change needed for '{response.original_query}' ({response.latency_ms:.2f}ms)")
        
        logger.info("=" * 60)
        return response
        
    except Exception as e:
        logger.error(f"❌ Error processing request: {str(e)}")
        logger.exception("Full traceback:")
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
