"""
SmartSearch Correction API
Port: 5001
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import logging
import sys

from app.corrector import TypoCorrector

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
        "RESET": "\033[0m",
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


logger = logging.getLogger("correction_api")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(
    ColoredFormatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
)
logger.addHandler(console_handler)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SmartSearch Correction API",
    description="E-commerce search query spell correction (ML-based)",
    version="3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load corrector once at startup
logger.info("Loading TypoCorrector ...")
corrector = TypoCorrector()
logger.info("TypoCorrector ready.")

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CorrectionRequest(BaseModel):
    query: str
    model: Optional[str] = None


class CorrectionResponse(BaseModel):
    original_query: str
    corrected_query: str
    changed: bool
    model_used: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"service": "correction", "status": "running", "port": 5001}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/correct", response_model=CorrectionResponse)
def correct_query(request: CorrectionRequest):
    """Correct a search query for typos."""
    logger.info("=" * 50)
    logger.info("POST /correct  query=%s  model=%s", request.query, request.model)

    try:
        result = corrector.correct(request.query, model=request.model)

        response = CorrectionResponse(
            original_query=result["original_query"],
            corrected_query=result["corrected_query"],
            changed=result["changed"],
            model_used=result["model_used"],
            latency_ms=result["latency_ms"],
        )

        if response.changed:
            logger.info(
                "'%s' -> '%s'  (%.1fms)",
                response.original_query,
                response.corrected_query,
                response.latency_ms,
            )
        else:
            logger.info(
                "No change: '%s'  (%.1fms)",
                response.original_query,
                response.latency_ms,
            )

        logger.info("=" * 50)
        return response

    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
def list_models():
    """List available correction models.

    Response format aligned with the Retrieval service template so
    the Backend / Frontend can consume both endpoints consistently.
    """
    models = corrector.list_models()

    return {
        "status": "success",
        "data": {
            "correction_models": models,
            "defaults": {
                "correction": corrector.get_default_model(),
            },
        },
    }


if __name__ == "__main__":
    print("Starting Correction API on port 5001 ...")
    uvicorn.run(app, host="0.0.0.0", port=5001)
