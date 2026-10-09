import json
from fastapi import APIRouter, HTTPException

from backend.core.config import settings, REPORTS_DIR
from backend.models.schemas import ConfigResponse, ConfigUpdateRequest

router = APIRouter(prefix="/api", tags=["Configuration & Samples"])

@router.get("/config", response_model=ConfigResponse)
def get_system_configuration():
    """
    Returns non-sensitive system configuration, active OCR provider, and model info.
    Does NOT expose API keys to the frontend.
    """
    return ConfigResponse(
        app_name=settings.APP_NAME,
        app_version=settings.APP_VERSION,
        ocr_provider=settings.OCR_PROVIDER,
        gemini_configured=bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip()),
        gemini_model=settings.GEMINI_MODEL,
        qwen_base_url=settings.QWEN_BASE_URL,
        qwen_model=settings.QWEN_MODEL,
        active_semantic_model=settings.ACTIVE_SEMANTIC_MODEL,
        min_training_samples=settings.MIN_TRAINING_SAMPLES
    )

@router.post("/config")
def update_system_configuration(req: ConfigUpdateRequest):
    """
    Allows runtime switching of OCR provider (Local Qwen vs Gemini API),
    active model version, or training threshold.
    """
    if req.ocr_provider:
        provider = req.ocr_provider.lower().strip()
        if provider in ["gemini", "qwen"]:
            settings.OCR_PROVIDER = provider
        else:
            raise HTTPException(status_code=400, detail="OCR provider must be 'gemini' or 'qwen'.")

    if req.active_semantic_model:
        settings.ACTIVE_SEMANTIC_MODEL = req.active_semantic_model.strip()

    if req.min_training_samples is not None and req.min_training_samples > 0:
        settings.MIN_TRAINING_SAMPLES = req.min_training_samples

    if req.gemini_api_key is not None:
        settings.GEMINI_API_KEY = req.gemini_api_key.strip()

    if req.gemini_model:
        settings.GEMINI_MODEL = req.gemini_model.strip()

    if req.qwen_base_url:
        settings.QWEN_BASE_URL = req.qwen_base_url.strip()

    if req.qwen_model:
        settings.QWEN_MODEL = req.qwen_model.strip()

    return {
        "success": True,
        "message": "Configuration updated successfully.",
        "config": get_system_configuration()
    }

@router.get("/evaluation/{id}")
def get_evaluation_report(id: str):
    """Retrieves a saved evaluation report by ID."""
    report_file = REPORTS_DIR / f"evaluation_report_{id}.json"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail=f"Report with ID '{id}' not found.")

    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

