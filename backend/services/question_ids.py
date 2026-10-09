import re
from typing import Any, Dict


_NUMERIC_SUBQUESTION_ID = re.compile(r"^(\d+)\s*\.?\s*([a-z])$", re.IGNORECASE)


def normalize_question_id(question_id: Any) -> str:
    """Normalize numeric subquestion IDs such as ``1.a`` and ``1a``."""
    value = str(question_id).strip()
    match = _NUMERIC_SUBQUESTION_ID.fullmatch(value)
    if match:
        return f"{match.group(1)}{match.group(2).lower()}"
    return value


def normalize_question_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    original_ids: Dict[str, str] = {}
    for question_id, value in mapping.items():
        normalized_id = normalize_question_id(question_id)
        if normalized_id in normalized:
            raise ValueError(
                f"Question IDs '{original_ids[normalized_id]}' and '{question_id}' "
                f"both normalize to '{normalized_id}'."
            )
        normalized[normalized_id] = value
        original_ids[normalized_id] = str(question_id)
    return normalized
