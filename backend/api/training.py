import logging
import re
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.config import MODELS_DIR, settings
from backend.services.hitl_service import hitl_service
from backend.training.train import run_fine_tuning
from backend.models.database_models import ModelRegistry
from backend.models.schemas import TrainingStartRequest, TrainingStatsResponse

router = APIRouter(prefix="/api/training", tags=["Training & Fine-Tuning"])
logger = logging.getLogger(__name__)

@router.get("/stats", response_model=TrainingStatsResponse)
def get_training_statistics(db: Session = Depends(get_db)):
    """
    Returns current count of verified training examples, threshold, and fine-tuning readiness.
    """
    stats = hitl_service.get_training_stats(db)
    return TrainingStatsResponse(**stats)

@router.post("/start")
def start_fine_tuning_job(
    force: bool = Query(False, description="Allow starting even if below threshold for demonstration purposes"),
    db: Session = Depends(get_db),
    request: TrainingStartRequest | None = None
):
    """
    Triggers the fine-tuning pipeline on verified human feedback.
    Splits into train/val/test, validates the result, and updates or registers a checkpoint.
    """
    stats = hitl_service.get_training_stats(db)
    if not stats["is_ready"] and not force:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Insufficient verified training data. Currently have {stats['verified_count']} / "
                f"{stats['min_threshold']} required samples. Complete more HITL reviews to reach threshold."
            )
        )

    model_version = None
    overwrite_existing = False
    if request and request.strategy == "update_active":
        model_version = settings.ACTIVE_SEMANTIC_MODEL
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,49}", model_version):
            raise HTTPException(status_code=400, detail="The active model name is not a valid fine-tuned checkpoint name.")
        model_dir = MODELS_DIR / model_version
        if (
            model_version == "codbert_base"
            or not (model_dir / "metadata.json").is_file()
            or not (model_dir / "weights.json").is_file()
        ):
            raise HTTPException(
                status_code=400,
                detail="The active model is not a fine-tuned checkpoint. Activate a fine-tuned model or create a new named model."
            )
        overwrite_existing = True
    elif request and request.strategy == "create_new":
        model_version = request.model_name
        if not model_version:
            raise HTTPException(status_code=400, detail="A model name is required when creating a new model.")
        registry_entry = db.query(ModelRegistry).filter(ModelRegistry.model_version == model_version).first()
        if (MODELS_DIR / model_version).exists() or registry_entry:
            raise HTTPException(status_code=409, detail=f"Model '{model_version}' already exists. Choose a different name.")

    try:
        metadata = run_fine_tuning(db, model_version=model_version, overwrite_existing=overwrite_existing)
        return {
            "success": True,
            "message": f"Successfully fine-tuned model {metadata['model_version']}",
            "metadata": metadata
        }
    except Exception as e:
        logger.exception("Fine-tuning execution failed")
        raise HTTPException(status_code=500, detail=f"Fine-tuning failed: {str(e)}")
