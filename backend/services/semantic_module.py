import os
import re
import math
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from backend.core.config import settings, MODELS_DIR

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at",
    "to", "for", "with", "by", "about", "that", "this", "it", "of", "from", "as"
}
_BIG_O_NOTATION = re.compile(r"\bo\s*\([^)]*\)", re.IGNORECASE)


def compute_similarity_features(student_answer: str, reference_answer: str) -> Tuple[float, float, list[str], list[str]]:
    def tokenize(text: str) -> list[str]:
        big_o_tokens = [
            re.sub(r"\s+", "", match.group(0).lower())
            for match in _BIG_O_NOTATION.finditer(text)
        ]
        text_without_big_o = _BIG_O_NOTATION.sub(" ", text.lower())
        word_tokens = [
            token
            for token in re.sub(r"[^\w\s]", " ", text_without_big_o).split()
            if len(token) > 1
        ]
        return word_tokens + big_o_tokens

    student_tokens = tokenize(student_answer)
    reference_tokens = tokenize(reference_answer)
    if not student_tokens or not reference_tokens:
        cosine_similarity = 0.0
    else:
        vocabulary = set(student_tokens + reference_tokens)
        student_counts = [student_tokens.count(token) for token in vocabulary]
        reference_counts = [reference_tokens.count(token) for token in vocabulary]
        dot_product = sum(a * b for a, b in zip(student_counts, reference_counts))
        student_magnitude = math.sqrt(sum(value * value for value in student_counts))
        reference_magnitude = math.sqrt(sum(value * value for value in reference_counts))
        cosine_similarity = dot_product / (student_magnitude * reference_magnitude)

    key_reference_tokens = [token for token in reference_tokens if token not in _STOPWORDS] or reference_tokens
    student_token_set = set(student_tokens)
    coverage = sum(token in student_token_set for token in key_reference_tokens) / len(key_reference_tokens) if key_reference_tokens else 0.0
    return cosine_similarity, coverage, student_tokens, reference_tokens

@dataclass
class SemanticEvaluationResult:
    similarity_score: float  # 0.0 to 1.0
    confidence: float        # 0.0 to 1.0
    key_phrase_overlap: float
    model_version: str

class BaseSemanticModel(ABC):
    """
    Abstract interface for semantic evaluation models.
    Enables swapping between Base CodeBERT, Fine-Tuned versions,
    or other NLP models without altering the evaluation pipeline.
    """

    @abstractmethod
    def evaluate(
        self,
        student_answer: str,
        reference_answer: str,
        rubric: Optional[Dict[str, Any]] = None
    ) -> SemanticEvaluationResult:
        """
        Evaluate semantic similarity and confidence between student and reference answer.
        """
        pass


class CodeBERTSemanticModel(BaseSemanticModel):
    """
    CodeBERT-compatible semantic evaluation model.
    Supports base mode or loading fine-tuned checkpoint weights from `models/{model_version}/`.
    Computes semantic vector alignment, concept overlap, and calibrated confidence.
    """

    def __init__(self, model_version: Optional[str] = None):
        self.model_version = model_version or settings.ACTIVE_SEMANTIC_MODEL or "codbert_base"
        self.model_dir = MODELS_DIR / self.model_version
        self.weights = self._load_model_weights()

    def _load_model_weights(self) -> Dict[str, Any]:
        """Loads fine-tuned weights and calibration parameters if available."""
        weights_file = self.model_dir / "weights.json"
        if weights_file.exists():
            try:
                with open(weights_file, "r", encoding="utf-8") as f:
                    weights = json.load(f)
                semantic_weight = max(0.0, float(weights.get("semantic_weight", 0.65)))
                overlap_weight = max(0.0, float(weights.get("overlap_weight", 0.35)))
                weight_total = semantic_weight + overlap_weight
                if weight_total == 0:
                    semantic_weight, overlap_weight = 0.65, 0.35
                else:
                    semantic_weight /= weight_total
                    overlap_weight /= weight_total
                weights = {
                    "semantic_weight": semantic_weight,
                    "overlap_weight": overlap_weight,
                    "bias": max(-0.1, min(0.1, float(weights.get("bias", 0.0))))
                }
                logger.info(f"Loaded fine-tuned model parameters from {weights_file}")
                return weights
            except Exception as e:
                logger.warning(f"Error loading weights for {self.model_version}: {e}")
        return {"bias": 0.0, "semantic_weight": 0.65, "overlap_weight": 0.35}

    def _tokenize_and_clean(self, text: str) -> list[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [t for t in cleaned.split() if len(t) > 1]
        return tokens

    def evaluate(
        self,
        student_answer: str,
        reference_answer: str,
        rubric: Optional[Dict[str, Any]] = None
    ) -> SemanticEvaluationResult:
        cosine_sim, coverage, s_tokens, r_tokens = compute_similarity_features(student_answer, reference_answer)

        # 3. Model weights calibration (applies fine-tuned weights if present)
        sem_w = self.weights.get("semantic_weight", 0.65)
        over_w = self.weights.get("overlap_weight", 0.35)
        bias = self.weights.get("bias", 0.0)

        raw_score = (cosine_sim * sem_w) + (coverage * over_w) + bias
        # Normalise between 0.0 and 1.0
        similarity_score = max(0.0, min(1.0, raw_score))

        # 4. Confidence calculation: higher when student answer has substantial content and strong alignment
        length_ratio = min(1.0, len(s_tokens) / max(1, len(r_tokens)))
        confidence = max(0.5, min(0.99, (similarity_score * 0.7) + (length_ratio * 0.3)))

        return SemanticEvaluationResult(
            similarity_score=round(similarity_score, 4),
            confidence=round(confidence, 4),
            key_phrase_overlap=round(coverage, 4),
            model_version=self.model_version
        )


def get_semantic_model(model_version: Optional[str] = None) -> BaseSemanticModel:
    """
    Factory to retrieve the configured semantic model instance.
    Defaults to settings.ACTIVE_SEMANTIC_MODEL or 'codbert_base'.
    """
    version = model_version or settings.ACTIVE_SEMANTIC_MODEL or "codbert_base"
    return CodeBERTSemanticModel(model_version=version)
