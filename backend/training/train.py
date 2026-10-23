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

def run_fine_tuning(
    db: Optional[Session] = None,
    model_version: Optional[str] = None,
    overwrite_existing: bool = False
) -> Dict[str, Any]:
    """
    Executes the fine-tuning pipeline on verified training data.
    1. Splits data into Train / Val / Test.
    2. Learns optimal calibration weights to align semantic similarity with human marks.
    3. Evaluates on the Validation set.
    4. Saves model artifacts under the requested name, or allocates a versioned name.
    """
    logger.info("Starting ScriptSense Fine-Tuning Pipeline...")
    new_version = model_version or get_next_model_version(db)
    windows_reserved_names = {"CON", "PRN", "AUX", "NUL"}
    windows_reserved_names.update(f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10))
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,49}", new_version)
        or new_version == "codbert_base"
        or new_version.upper() in windows_reserved_names
    ):
        raise ValueError("Model name must be a safe 1-50 character identifier and cannot be 'codbert_base'.")
    version_dir = MODELS_DIR / new_version
    if version_dir.exists() and not overwrite_existing:
        raise ValueError(f"Model '{new_version}' already exists. Choose a different model name.")
    if overwrite_existing and not version_dir.is_dir():
        raise ValueError(f"Cannot update model '{new_version}': checkpoint directory was not found.")

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

    # Continue from the selected checkpoint when updating an existing model.
    w_sem = 0.65
    w_cov = 0.35
    bias = 0.0
    if overwrite_existing:
        weights_file = version_dir / "weights.json"
        if not weights_file.is_file():
            raise ValueError(f"Cannot update model '{model_version}': checkpoint weights were not found.")
        with open(weights_file, "r", encoding="utf-8") as f:
            existing_weights = json.load(f)
        w_sem = max(0.0, float(existing_weights.get("semantic_weight", w_sem)))
        w_cov = max(0.0, float(existing_weights.get("overlap_weight", w_cov)))
        bias = max(-0.1, min(0.1, float(existing_weights.get("bias", bias))))
        weight_total = w_sem + w_cov
        if weight_total == 0:
            w_sem, w_cov = 0.65, 0.35
        else:
            w_sem /= weight_total
            w_cov /= weight_total

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
    metadata_file = version_dir / "metadata.json"
    if overwrite_existing and metadata_file.is_file():
        with open(metadata_file, "r", encoding="utf-8") as f:
            existing_metadata = json.load(f)
        metadata["base_model"] = existing_metadata.get("base_model", "codbert_base")

    with open(version_dir / "weights.json", "w", encoding="utf-8") as f:
        json.dump(weights_data, f, indent=4)

    with open(version_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # Register in DB
    if db:
        reg = db.query(ModelRegistry).filter(ModelRegistry.model_version == new_version).first()
        if reg is None:
            reg = ModelRegistry(model_version=new_version)
            db.add(reg)
        reg.training_samples = len(train_set)
        reg.validation_score = val_score
        reg.base_model = metadata["base_model"]
        reg.is_active = overwrite_existing
        db.commit()

    logger.info(f"Successfully fine-tuned model {new_version} with val_score={val_score}")
    return metadata
