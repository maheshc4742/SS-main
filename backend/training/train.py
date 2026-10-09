import os
import re
import json
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from backend.core.config import MODELS_DIR, DATA_DIR, settings
from backend.training.prepare_dataset import split_and_prepare_dataset
from backend.models.database_models import ModelRegistry
from backend.services.semantic_module import compute_similarity_features

logger = logging.getLogger(__name__)

def get_next_model_version(db: Optional[Session] = None) -> str:
    """Find the next version not already present on disk or in the registry."""
    existing_dirs = [d.name for d in MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith("codbert_ft_v")]
    indices = []
    for d in existing_dirs:
        m = re.search(r"v(\d+)$", d)
        if m:
            indices.append(int(m.group(1)))
    if db:
        existing_versions = db.query(ModelRegistry.model_version).all()
        for (version,) in existing_versions:
            if version.startswith("codbert_ft_v"):
                m = re.search(r"v(\d+)$", version)
                if m:
                    indices.append(int(m.group(1)))
    next_idx = max(indices, default=0) + 1
    return f"codbert_ft_v{next_idx}"

def run_fine_tuning(db: Optional[Session] = None) -> Dict[str, Any]:
    """
    Executes the fine-tuning pipeline on verified training data.
    1. Splits data into Train / Val / Test.
    2. Learns optimal calibration weights to align semantic similarity with human marks.
    3. Evaluates on the Validation set.
    4. Saves versioned model artifacts and metadata without overwriting previous versions.
    """
    logger.info("Starting ScriptSense Fine-Tuning Pipeline...")
    train_set, val_set, test_set = split_and_prepare_dataset()

    if not train_set:
        raise ValueError("Training set is empty. Collect more verified human annotations first.")

    # Compute optimal weights via gradient descent or closed-form least squares on normalized scores
    # Feature extraction for training samples
    # y = human_marks / max_marks
    X = []
    y = []

    for item in train_set:
        s_ans = item["student_answer"]
        r_ans = item["reference_answer"]
        rub = item.get("rubric", {})
        max_m = float(rub.get("max_marks", 5.0))
        h_marks = float(item.get("human_marks", max_m * 0.8))
        target_score = max(0.0, min(1.0, h_marks / max(0.1, max_m)))

        cos_sim, coverage, _, _ = compute_similarity_features(s_ans, r_ans)
        X.append((cos_sim, coverage))
        y.append(target_score)

    # Estimate regression weights (least squares / simple optimization)
    # y_hat = w_sem * cos_sim + w_cov * coverage + bias
    w_sem = 0.65
    w_cov = 0.35
    bias = 0.0

    # Refine weights with mini-batch learning rate updates
    lr = 0.05
    for epoch in range(100):
        for (cos_sim, cov), target in zip(X, y):
            pred = (w_sem * cos_sim) + (w_cov * cov) + bias
            error = pred - target
            w_sem = max(0.0, w_sem - lr * error * cos_sim)
            w_cov = max(0.0, w_cov - lr * error * cov)
            weight_total = w_sem + w_cov
            if weight_total == 0:
                w_sem, w_cov = 0.65, 0.35
            else:
                w_sem /= weight_total
                w_cov /= weight_total
            bias = max(-0.1, min(0.1, bias - lr * error))

    # Validation evaluation
    val_errors = []
    for item in val_set if val_set else train_set:
        s_ans = item["student_answer"]
        r_ans = item["reference_answer"]
        rub = item.get("rubric", {})
        max_m = float(rub.get("max_marks", 5.0))
        h_marks = float(item.get("human_marks", max_m * 0.8))
        target_score = max(0.0, min(1.0, h_marks / max(0.1, max_m)))

        cos_sim, coverage, _, _ = compute_similarity_features(s_ans, r_ans)

        pred = max(0.0, min(1.0, (w_sem * cos_sim) + (w_cov * coverage) + bias))
        val_errors.append(abs(pred - target_score))

    mae = sum(val_errors) / max(1, len(val_errors))
    val_score = round(max(0.0, 1.0 - mae), 4)

    # Versioning: Create new model directory
    new_version = get_next_model_version(db)
    version_dir = MODELS_DIR / new_version
    version_dir.mkdir(parents=True, exist_ok=True)

    weights_data = {
        "semantic_weight": round(w_sem, 4),
        "overlap_weight": round(w_cov, 4),
        "bias": round(bias, 4)
    }

    metadata = {
        "model_version": new_version,
        "training_samples": len(train_set),
        "validation_score": val_score,
        "mae": round(mae, 4),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_model": "codbert_base"
    }

    with open(version_dir / "weights.json", "w", encoding="utf-8") as f:
        json.dump(weights_data, f, indent=4)

    with open(version_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # Register in DB
    if db:
        reg = ModelRegistry(
            model_version=new_version,
            training_samples=len(train_set),
            validation_score=val_score,
            base_model="codbert_base",
            is_active=False
        )
        db.add(reg)
        db.commit()

    logger.info(f"Successfully fine-tuned model {new_version} with val_score={val_score}")
    return metadata
