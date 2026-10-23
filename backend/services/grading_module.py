import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


_RUBRIC_STOPWORDS = {
    "a", "all", "an", "and", "are", "as", "by", "during", "for", "from",
    "if", "in", "into", "is", "of", "or", "the", "to", "with",
}
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


@dataclass
class CriterionEvaluation:
    criterion: str
    max_marks: float
    awarded_marks: float
    status: str  # "Fully Met", "Partially Met", "Not Met"
    notes: str

@dataclass
class GradingResult:
    marks: float
    max_marks: float
    feedback: str
    semantic_score: float
    confidence: float
    criteria_breakdown: List[Dict[str, Any]]

def quantize_half_or_whole(score: float, max_marks: float) -> float:
    """
    Enforces the rule: keep fraction .5 or whole number as per rubric.
    Clamps between 0.0 and max_marks.
    Examples:
      3.22 -> 3.0
      3.35 -> 3.5
      3.65 -> 3.5
      3.78 -> 4.0
    """
    quantized = round(score * 2.0) / 2.0
    clamped = max(0.0, min(float(max_marks), quantized))
    # Return as int if it's a whole number (e.g. 4.0 -> 4.0 for float consistency or displayed cleanly)
    return clamped


def _numbered_reference_items(reference_answer: str) -> List[str]:
    items = []
    for line in reference_answer.splitlines():
        item = re.sub(r"^\s*(?:[-*]\s*|\d+[.)]\s*|[ivxlcdm]+[.)]\s*)", "", line, flags=re.IGNORECASE).strip()
        if item != line.strip() and item:
            items.append(item)
    return items


def _required_component_count(criteria: str, item_count: int) -> Optional[int]:
    count_pattern = r"\b(?:any|all|at least)\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b"
    match = re.search(count_pattern, criteria, flags=re.IGNORECASE)
    if match:
        count = match.group(1).lower()
        required = _NUMBER_WORDS[count] if count in _NUMBER_WORDS else int(count)
        return required if 0 < required <= item_count else None

    if re.search(r"\beach missing\b", criteria, flags=re.IGNORECASE):
        return item_count
    return None


def _normalized_component_tokens(text: str) -> set[str]:
    text = re.sub(r"\([^)]*\)", " ", text.lower())
    tokens = re.findall(r"[a-z0-9]+", text)
    normalized = set()
    for token in tokens:
        if token in _RUBRIC_STOPWORDS:
            continue
        if token == "sectional":
            token = "section"
        elif token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        normalized.add(token)
    return normalized


def _component_is_present(item: str, student_answer: str) -> bool:
    student_tokens = _normalized_component_tokens(student_answer)
    alternatives = re.split(r"\bor\b", item, flags=re.IGNORECASE)
    for alternative in alternatives:
        expected_tokens = _normalized_component_tokens(alternative)
        if expected_tokens:
            overlap = len(expected_tokens & student_tokens) / len(expected_tokens)
            if overlap >= 0.5:
                return True
    return False


def _numeric_role(label: str) -> Optional[str]:
    normalized = label.lower()
    if re.search(r"\b(largest|maximum|max)\b", normalized):
        return "largest"
    if re.search(r"\b(smallest|minimum|min)\b", normalized):
        return "smallest"
    if re.search(r"\b(sum|total)\b", normalized):
        return "sum"
    if re.search(r"\b(average|avg|mean)\b", normalized):
        return "average"
    return None


def _grade_explicit_numeric_rubric(
    student_answer: str,
    criteria: Any,
    max_marks: float,
) -> Optional[tuple[float, List[Dict[str, Any]], List[str]]]:
    if not isinstance(criteria, str):
        return None

    award_pattern = re.compile(
        r"(?P<marks>\d+(?:\.\d+)?)\s+marks?\s+for\s+(?:the\s+)?correct\s+"
        r"(?P<items>.*?)(?=(?:,\s*(?:and\s+)?\d+(?:\.\d+)?\s+marks?\s+"
        r"for\s+(?:the\s+)?correct\b)|(?:\.\s|$))",
        flags=re.IGNORECASE,
    )
    value_pattern = re.compile(
        r"(?P<label>[a-z][a-z ]*?)\s*\(\s*(?P<value>[-+]?\d+(?:\.\d+)?)\s*\)",
        flags=re.IGNORECASE,
    )
    student_value_pattern = re.compile(
        r"\b(?P<label>largest(?:\s+(?:number|value))?|maximum|max|"
        r"smallest(?:\s+(?:number|value))?|minimum|min|sum|total|"
        r"average|avg|mean)\b"
        r"[^0-9+\-]{0,20}(?P<value>[-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
        flags=re.IGNORECASE,
    )

    awards = []
    for match in award_pattern.finditer(criteria):
        expected_values = []
        for value_match in value_pattern.finditer(match.group("items")):
            role = _numeric_role(value_match.group("label"))
            if role is None:
                return None
            expected_values.append((role, float(value_match.group("value"))))
        if not expected_values:
            return None
        awards.append((float(match.group("marks")), expected_values))

    if not awards:
        return None

    student_values = {}
    for match in student_value_pattern.finditer(student_answer):
        role = _numeric_role(match.group("label"))
        if role is not None:
            student_values[role] = float(match.group("value").replace(",", ""))

    breakdown = []
    feedback_lines = []
    total_awarded = 0.0
    for award_marks, expected_values in awards:
        is_correct = all(
            role in student_values
            and abs(student_values[role] - expected_value) <= 1e-9
            for role, expected_value in expected_values
        )
        awarded = award_marks if is_correct else 0.0
        total_awarded += awarded
        labels = " and ".join(role for role, _ in expected_values)
        status = "Fully Met" if is_correct else "Not Met"
        notes = (
            f"Correct {labels} value(s)."
            if is_correct
            else f"Expected {', '.join(f'{role} = {value:g}' for role, value in expected_values)}."
        )
        breakdown.append({
            "criterion": labels.title(),
            "max_marks": award_marks,
            "awarded_marks": awarded,
            "status": status,
            "notes": notes,
        })
        feedback_lines.append(
            f"{labels.title()} ({awarded:g}/{award_marks:g} marks): {notes}"
        )

    final_marks = quantize_half_or_whole(total_awarded, max_marks)
    return final_marks, breakdown, feedback_lines


def _grade_numbered_rubric(
    student_answer: str,
    reference_answer: str,
    criteria: Any,
    max_marks: float,
) -> Optional[tuple[float, int, int, List[str]]]:
    if not isinstance(criteria, str):
        return None

    items = _numbered_reference_items(reference_answer)
    if len(items) < 2:
        return None

    required_count = _required_component_count(criteria, len(items))
    if required_count is None:
        return None

    matched_items = [item for item in items if _component_is_present(item, student_answer)]
    awarded_marks = quantize_half_or_whole(
        max_marks * min(len(matched_items), required_count) / required_count,
        max_marks,
    )
    missing_items = [item for item in items if item not in matched_items]
    return awarded_marks, len(matched_items), required_count, missing_items


class GradingModule:
    """
    Rubric-based grading module.
    Evaluates student descriptive answers against reference answers, rubrics, and semantic similarity.
    Adheres strictly to the fraction .5 or whole number requirement.
    """

    @staticmethod
    def evaluate_criterion(
        criterion_name: str,
        crit_max: float,
        student_answer: str,
        reference_answer: str,
        semantic_score: float
    ) -> CriterionEvaluation:
        s_lower = student_answer.lower()
        r_lower = reference_answer.lower()
        crit_lower = criterion_name.lower()

        # Keywords specific to this criterion
        crit_keywords = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', crit_lower) if w not in {'the', 'and', 'for', 'with'}]

        # Check for criterion-related terms in student text
        matches = [kw for kw in crit_keywords if kw in s_lower]

        # Calculate criterion satisfaction score
        criterion_match_ratio = len(matches) / max(1, len(crit_keywords))
        
        # Combined criterion fulfillment using overall semantic alignment and specific keyword presence
        fulfillment = (semantic_score * 0.7) + (criterion_match_ratio * 0.3)

        raw_score = crit_max * fulfillment
        awarded = quantize_half_or_whole(raw_score, crit_max)

        if awarded >= crit_max:
            status = "Fully Met"
            notes = f"Criterion '{criterion_name}' thoroughly addressed."
        elif awarded > 0:
            status = "Partially Met"
            notes = f"Criterion '{criterion_name}' partially covered. Lacks full depth."
        else:
            status = "Not Met"
            notes = f"Criterion '{criterion_name}' missing or insufficient in the response."

        return CriterionEvaluation(
            criterion=criterion_name,
            max_marks=crit_max,
            awarded_marks=awarded,
            status=status,
            notes=notes
        )

    def grade(
        self,
        student_answer: str,
        reference_answer: str,
        rubric: Dict[str, Any],
        semantic_score: float,
        confidence: float
    ) -> GradingResult:
        max_marks = float(rubric.get("max_marks", 5.0))
        criteria_list = rubric.get("criteria", [])

        criteria_evals: List[CriterionEvaluation] = []
        feedback_lines: List[str] = []
        criteria_breakdown: List[Dict[str, Any]] = []

        if criteria_list and isinstance(criteria_list, list):
            # Sum up criteria marks
            for crit in criteria_list:
                crit_name = crit.get("criterion", "Criterion")
                crit_max = float(crit.get("marks", 1.0))
                c_eval = self.evaluate_criterion(
                    crit_name,
                    crit_max,
                    student_answer,
                    reference_answer,
                    semantic_score
                )
                criteria_evals.append(c_eval)
                feedback_lines.append(f"• {c_eval.criterion} ({c_eval.awarded_marks}/{c_eval.max_marks} marks): {c_eval.notes}")

            total_awarded = sum(c.awarded_marks for c in criteria_evals)
            # Bound to max_marks and quantize
            final_marks = quantize_half_or_whole(total_awarded, max_marks)
            criteria_breakdown = [
                {
                    "criterion": c.criterion,
                    "max_marks": c.max_marks,
                    "awarded_marks": c.awarded_marks,
                    "status": c.status,
                    "notes": c.notes
                }
                for c in criteria_evals
            ]
        else:
            numeric_result = _grade_explicit_numeric_rubric(
                student_answer,
                criteria_list,
                max_marks,
            )
            if numeric_result is not None:
                final_marks, criteria_breakdown, feedback_lines = numeric_result
            else:
                component_result = _grade_numbered_rubric(
                    student_answer,
                    reference_answer,
                    criteria_list,
                    max_marks,
                )
            if numeric_result is None and component_result is not None:
                final_marks, matched_count, required_count, missing_items = component_result
                feedback_lines.append(
                    f"Matched {min(matched_count, required_count)} of {required_count} required listed answer points."
                )
                if missing_items:
                    feedback_lines.append("Missing or unrecognized points: " + "; ".join(missing_items))
                criteria_breakdown = [{
                    "criterion": "Listed answer points",
                    "max_marks": max_marks,
                    "awarded_marks": final_marks,
                    "status": "Fully Met" if final_marks >= max_marks else (
                        "Partially Met" if final_marks > 0 else "Not Met"
                    ),
                    "notes": feedback_lines[0],
                }]
            elif numeric_result is None:
                # Direct semantic-based rubric grading
                raw_marks = max_marks * semantic_score
                final_marks = quantize_half_or_whole(raw_marks, max_marks)

                if final_marks >= max_marks * 0.9:
                    feedback_lines.append("Excellent answer covering all core points accurately.")
                elif final_marks >= max_marks * 0.7:
                    feedback_lines.append("Good answer demonstrating sound conceptual understanding with minor omissions.")
                elif final_marks >= max_marks * 0.4:
                    feedback_lines.append("Average response. Demonstrates basic awareness but lacks key descriptive elements.")
                else:
                    feedback_lines.append("Answer is insufficient or does not match reference concepts.")

        # Overall summary feedback
        percentage = (final_marks / max_marks * 100) if max_marks > 0 else 0
        overall_critique = (
            f"Awarded {final_marks:g} out of {max_marks:g} marks ({percentage:.1f}%). "
            f"Semantic similarity: {semantic_score:.0%}, Confidence: {confidence:.0%}.\n"
            + "\n".join(feedback_lines)
        )

        return GradingResult(
            marks=final_marks,
            max_marks=max_marks,
            feedback=overall_critique.strip(),
            semantic_score=semantic_score,
            confidence=confidence,
            criteria_breakdown=criteria_breakdown
        )

# Global singleton
grading_module = GradingModule()
