import os
import json
import asyncio
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
import pymupdf as fitz
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.services.grading_module import quantize_half_or_whole, grading_module
from backend.services.semantic_module import CodeBERTSemanticModel, compute_similarity_features, get_semantic_model
from backend.services.hitl_service import hitl_service
from backend.services.evaluator import ScriptSenseEvaluator
from backend.ocr.base_ocr import OCRProviderError
from backend.ocr.gemini_ocr import GeminiOCR
from backend.ocr.qwen_ocr import QwenOCR
from backend.core.config import settings
from backend.main import app
from backend.api.hitl import router as hitl_router
from backend.core.database import Base, SessionLocal, init_db
from backend.models.database_models import HITLItem, ModelRegistry
from backend.models.schemas import TrainingStartRequest
from backend.ocr.pdf_utils import render_pdf_to_images, is_pdf
from backend.training.prepare_dataset import split_and_prepare_dataset
from backend.training.train import run_fine_tuning, get_next_model_version
from backend.training.evaluate_model import evaluate_model_on_test_set


def test_next_model_version_accounts_for_database_registry(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "codbert_ft_v2").mkdir()
    monkeypatch.setattr("backend.training.train.MODELS_DIR", models_dir)

    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine, tables=[ModelRegistry.__table__])
    db = sessionmaker(bind=engine)()
    db.add(ModelRegistry(model_version="codbert_ft_v4"))
    db.commit()

    assert get_next_model_version(db) == "codbert_ft_v5"

    db.close()
    engine.dispose()


def test_training_request_validates_user_defined_model_names():
    request = TrainingStartRequest(strategy="create_new", model_name="answer_model-v1")
    assert request.model_name == "answer_model-v1"

    with pytest.raises(ValueError):
        TrainingStartRequest(strategy="create_new", model_name="../outside")
    with pytest.raises(ValueError):
        TrainingStartRequest(strategy="create_new", model_name="codbert_base")
    with pytest.raises(ValueError):
        TrainingStartRequest(strategy="create_new", model_name="CON")


def test_fine_tuning_updates_named_checkpoint_in_place(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr("backend.training.train.MODELS_DIR", models_dir)
    monkeypatch.setattr(
        "backend.training.train.split_and_prepare_dataset",
        lambda: (
            [{
                "student_answer": "Objects inherit behavior.",
                "reference_answer": "Objects inherit behavior.",
                "rubric": {"max_marks": 5},
                "human_marks": 4,
            }],
            [],
            [],
        ),
    )

    first_metadata = run_fine_tuning(model_version="answer_model")
    checkpoint = models_dir / "answer_model"
    initial_metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))

    updated_metadata = run_fine_tuning(model_version="answer_model", overwrite_existing=True)

    assert first_metadata["model_version"] == "answer_model"
    assert updated_metadata["model_version"] == "answer_model"
    assert (checkpoint / "metadata.json").is_file()
    assert json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))["model_version"] == initial_metadata["model_version"]
    assert [path.name for path in models_dir.iterdir()] == ["answer_model"]
    with pytest.raises(ValueError, match="already exists"):
        run_fine_tuning(model_version="answer_model")


def test_frontend_uploads_key_and_rubric_json_files():
    """Verify the evaluation UI sends uploaded key/rubric JSON files as multipart form data."""
    app_js = Path("frontend/js/app.js").read_text(encoding="utf-8")
    assert "fileInputKeys" in app_js
    assert "fileInputRubric" in app_js
    assert "key_answers_file" in app_js
    assert "rubrics_file" in app_js
    assert "state.keyAnswersFile" in app_js
    assert "state.rubricsFile" in app_js
    assert "const endpoint = trainingMode ? '/api/train/evaluate' : '/api/evaluate'" in app_js
    assert "formData.append('ocr_provider', state.config?.ocr_provider || 'gemini')" in app_js


def test_reject_all_button_is_wired_to_current_frontend_script_and_api():
    index_html = Path("frontend/index.html").read_text(encoding="utf-8")
    app_js = Path("frontend/js/app.js").read_text(encoding="utf-8")

    assert 'id="btn-reject-all-pending"' in index_html
    assert "/static/js/app.js?v=" in index_html
    assert "elements.btnRejectAllPending.addEventListener('click', rejectAllPendingItems)" in app_js
    assert "fetch('/api/hitl/reject-all'" in app_js
    assert any(
        route.path == "/api/hitl/reject-all" and "POST" in route.methods
        for route in hitl_router.routes
    )


def test_predefined_oop_exam_is_removed():
    config_api = Path("backend/api/config_api.py").read_text(encoding="utf-8")
    app_js = Path("frontend/js/app.js").read_text(encoding="utf-8")
    index_html = Path("frontend/index.html").read_text(encoding="utf-8")

    assert "/sample-data" not in config_api
    assert "sample_student_answers" not in app_js
    assert "Load OOP Exam" not in index_html
    assert not Path("generate_sample_assets.py").exists()
    assert not Path("samples/key_answers.json").exists()
    assert not Path("samples/rubric.json").exists()
    assert not Path("samples/student_script.json").exists()
    assert not Path("samples/sample_student_script.pdf").exists()
    assert not Path("samples/sample_student_script.png").exists()


def test_semantic_features_and_legacy_model_weights_are_bounded():
    same_answer = compute_similarity_features("Inheritance supports reuse", "Inheritance supports reuse")
    empty_reference = compute_similarity_features("Inheritance supports reuse", "")

    assert abs(same_answer[0] - 1.0) < 1e-9
    assert same_answer[1] == 1.0
    assert empty_reference[0] == 0.0
    assert empty_reference[1] == 0.0

    model = CodeBERTSemanticModel("codbert_ft_v5")
    assert abs(model.weights["semantic_weight"] + model.weights["overlap_weight"] - 1.0) < 1e-9
    assert abs(model.weights["bias"]) <= 0.1


def test_big_o_notation_matches_when_reference_has_choice_label():
    cosine_similarity, coverage, student_tokens, reference_tokens = compute_similarity_features(
        "O(n).",
        "b) O(n)",
    )
    result = CodeBERTSemanticModel().evaluate("O(n).", "b) O(n)")
    grade = grading_module.grade(
        student_answer="O(n).",
        reference_answer="b) O(n)",
        rubric={"max_marks": 5.0},
        semantic_score=result.similarity_score,
        confidence=result.confidence,
    )

    assert student_tokens == ["o(n)"]
    assert reference_tokens == ["o(n)"]
    assert cosine_similarity == 1.0
    assert coverage == 1.0
    assert result.similarity_score == 1.0
    assert grade.marks == 5.0


def test_evaluation_matches_numeric_subquestion_ids_with_or_without_period():
    result = asyncio.run(ScriptSenseEvaluator().evaluate_script(
        mode="evaluate",
        key_answers={"1.a": "Polymorphism allows objects to take many forms."},
        rubrics={"1.a": {"max_marks": 5.0, "criteria": []}},
        student_answers={"1a": "Polymorphism allows objects to take many forms."},
    ))

    assert len(result["questions"]) == 1
    assert result["questions"][0]["question"] == "1a"
    assert result["questions"][0]["reference_answer"] == (
        "Polymorphism allows objects to take many forms."
    )
    assert result["questions"][0]["student_answer"] == (
        "Polymorphism allows objects to take many forms."
    )
    assert result["questions"][0]["max_marks"] == 5.0


def test_extra_detected_answer_gets_zero_without_increasing_total_marks():
    result = asyncio.run(ScriptSenseEvaluator().evaluate_script(
        mode="evaluate",
        key_answers={"1a": "Polymorphism allows objects to take many forms."},
        rubrics={"1a": {"max_marks": 5.0, "criteria": []}},
        student_answers={
            "1a": "Polymorphism allows objects to take many forms.",
            "1b": "in a program",
        },
    ))

    extra_answer = next(question for question in result["questions"] if question["question"] == "1b")
    assert extra_answer["marks"] == 0.0
    assert extra_answer["max_marks"] == 0.0
    assert extra_answer["student_answer"] == "in a program"
    assert "not present in the answer key or rubric" in extra_answer["feedback"]
    assert result["total_marks"] == 5.0
    assert result["obtained_marks"] == 5.0


def test_ocr_without_credentials_does_not_return_predefined_answers(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    with pytest.raises(OCRProviderError, match="GEMINI_API_KEY"):
        asyncio.run(GeminiOCR().extract_answers(b"image", "script.png", ["1"]))


def test_unavailable_ocr_returns_service_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    with TestClient(app) as client:
        response = client.post(
            "/api/evaluate",
            data={
                "key_answers": json.dumps({"1": "Reference answer"}),
                "rubrics": json.dumps({"1": {"max_marks": 5}})
            },
            files={"file": ("script.png", b"image bytes", "image/png")}
        )

    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


@pytest.mark.parametrize(
    ("response_body", "expected_answers", "expected_error"),
    [
        (
            {"choices": [{"message": {"content": '{"1": "transcribed answer"}'}}]},
            {"1": "transcribed answer"},
            None,
        ),
        (
            {"error": "Unexpected endpoint or method."},
            None,
            "endpoint returned an error",
        ),
        (
            {"choices": [{"message": {"content": "not valid JSON"}}]},
            None,
            "valid JSON answer mapping",
        ),
    ],
)
def test_qwen_ocr_uses_versioned_endpoint_and_validates_response(
    monkeypatch, response_body, expected_answers, expected_error
):
    requests = []

    class FakeResponse:
        status_code = 200

        def json(self):
            return response_body

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return False

        async def post(self, url, **kwargs):
            requests.append((url, kwargs))
            return FakeResponse()

    monkeypatch.setattr("backend.ocr.qwen_ocr.httpx.AsyncClient", FakeClient)
    ocr = QwenOCR(base_url="http://127.0.0.1:1234", model="qwen-local")

    if expected_error:
        with pytest.raises(OCRProviderError, match=expected_error):
            asyncio.run(ocr.extract_answers(b"image bytes", "script.jpg", ["1"]))
    else:
        answers = asyncio.run(ocr.extract_answers(b"image bytes", "script.jpg", ["1"]))
        assert answers == expected_answers

    assert requests[0][0] == "http://127.0.0.1:1234/v1/chat/completions"
    prompt = requests[0][1]["json"]["messages"][1]["content"][0]["text"]
    assert "These are hints, not a limit" in prompt
    assert "Include all detected labels" in prompt
    assert "Include brief answers and fragments too" in prompt


@pytest.mark.parametrize(
    ("status_code", "response_body", "expected_answers", "expected_error"),
    [
        (
            200,
            {
                "steps": [{
                    "type": "model_output",
                    "content": [{"type": "text", "text": '{"1": "answer"}'}],
                }],
            },
            {"1": "answer"},
            None,
        ),
        (
            404,
            {"error": {"message": "Model not found for this API key"}},
            None,
            "Check GEMINI_MODEL",
        ),
    ],
)
def test_gemini_ocr_uses_api_key_header_and_reports_model_404(
    monkeypatch, status_code, response_body, expected_answers, expected_error
):
    requests = []
    api_key = "test-secret-key"

    class FakeResponse:
        def __init__(self):
            self.status_code = status_code

        def json(self):
            return response_body

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return False

        async def post(self, url, **kwargs):
            requests.append((url, kwargs))
            return FakeResponse()

    monkeypatch.setattr("backend.ocr.gemini_ocr.httpx.AsyncClient", FakeClient)
    ocr = GeminiOCR(api_key=api_key, model="gemini-test")

    if expected_error:
        with pytest.raises(OCRProviderError, match=expected_error) as exc_info:
            asyncio.run(ocr.extract_answers(b"image bytes", "script.jpg", ["1"]))
        if status_code == 404:
            assert "Google API detail: Model not found for this API key." in str(exc_info.value)
    else:
        answers = asyncio.run(ocr.extract_answers(b"image bytes", "script.jpg", ["1"]))
        assert answers == expected_answers

    request_url, request_kwargs = requests[0]
    assert request_url == (
        "https://generativelanguage.googleapis.com/v1beta/interactions"
    )
    assert api_key not in request_url
    assert request_kwargs["headers"]["x-goog-api-key"] == api_key
    assert request_kwargs["json"]["model"] == "gemini-test"
    assert request_kwargs["json"]["input"][0]["type"] == "text"
    assert request_kwargs["json"]["input"][1]["type"] == "image"
    assert request_kwargs["json"]["input"][1]["mime_type"] == "image/jpeg"


def test_gemini_ocr_retries_transient_503_and_returns_success(monkeypatch):
    statuses = iter([
        (503, {"error": {"message": "Temporarily unavailable"}}),
        (200, {
            "steps": [{
                "type": "model_output",
                "content": [{"type": "text", "text": '{"1": "answer"}'}],
            }],
        }),
    ])
    request_count = 0

    class FakeResponse:
        def __init__(self, status_code, body):
            self.status_code = status_code
            self.body = body

        def json(self):
            return self.body

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return False

        async def post(self, url, **kwargs):
            nonlocal request_count
            request_count += 1
            status_code, body = next(statuses)
            return FakeResponse(status_code, body)

    async def skip_delay(_seconds):
        return None

    monkeypatch.setattr("backend.ocr.gemini_ocr.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr("backend.ocr.gemini_ocr.asyncio.sleep", skip_delay)

    answers = asyncio.run(GeminiOCR(api_key="test-key", model="gemini-test").extract_answers(
        b"image bytes", "script.jpg", ["1"]
    ))

    assert answers == {"1": "answer"}
    assert request_count == 2


def test_gemini_ocr_reports_provider_detail_after_retry_exhaustion(monkeypatch):
    request_count = 0

    class FakeResponse:
        status_code = 503

        def json(self):
            return {"error": {"message": "Service temporarily unavailable"}}

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return False

        async def post(self, url, **kwargs):
            nonlocal request_count
            request_count += 1
            return FakeResponse()

    async def skip_delay(_seconds):
        return None

    monkeypatch.setattr("backend.ocr.gemini_ocr.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr("backend.ocr.gemini_ocr.asyncio.sleep", skip_delay)

    with pytest.raises(OCRProviderError, match="HTTP 503 after 3 attempts") as exc_info:
        asyncio.run(GeminiOCR(api_key="test-key", model="gemini-test").extract_answers(
            b"image bytes", "script.jpg", ["1"]
        ))

    assert "Service temporarily unavailable" in str(exc_info.value)
    assert request_count == 3


def test_mixed_numeric_question_ids_do_not_crash_sorting():
    """Verify mixed numeric/string question IDs are normalized before sorting and lookup."""
    keys = {"1": "Answer 1", 2: "Answer 2", "10": "Answer 10"}
    normalized = {str(k): v for k, v in keys.items()}
    extracted = {"3": "Student 3"}

    def sort_question_key(question_id):
        q_text = str(question_id)
        return (0, int(q_text)) if q_text.isdigit() else (1, q_text)

    all_q_keys = sorted(set(normalized.keys()) | set(extracted.keys()), key=sort_question_key)
    assert all_q_keys == ["1", "2", "3", "10"]
    assert normalized["2"] == "Answer 2"


def test_hitl_pending_item_keeps_student_answers_with_numeric_keys():
    """Verify HITL queue preserves student answers when JSON keys are numbers instead of strings."""
    init_db()
    db = SessionLocal()

    items = hitl_service.create_pending_items(
        evaluation_id="test-eval-456",
        question_results=[{"question": "1", "marks": 4.0, "max_marks": 5.0, "feedback": "Good", "confidence": 0.9, "semantic_score": 0.85}],
        key_answers={"1": "Reference answer one"},
        rubrics={"1": {"max_marks": 5}},
        student_answers={1: "Student answer one"},
        db=db
    )

    assert items[0].student_answer == "Student answer one"
    assert items[0].reference_answer == "Reference answer one"
    db.close()


def test_half_or_whole_quantization():
    """Verify strictly .5 or integer mark quantization."""
    assert quantize_half_or_whole(3.22, 5.0) == 3.0
    assert quantize_half_or_whole(3.35, 5.0) == 3.5
    assert quantize_half_or_whole(3.68, 5.0) == 3.5
    assert quantize_half_or_whole(3.85, 5.0) == 4.0
    assert quantize_half_or_whole(5.4, 5.0) == 5.0
    assert quantize_half_or_whole(-0.5, 5.0) == 0.0

def test_rubric_based_grading():
    """Verify rubric criteria breakdown and score bounding."""
    student_ans = "Polymorphism allows methods to have multiple forms through overloading and overriding."
    ref_ans = "Polymorphism means many forms. Compile time is overloading, runtime is overriding."
    rubric = {
        "max_marks": 5.0,
        "criteria": [
            {"criterion": "Definition", "marks": 1.0},
            {"criterion": "Overloading", "marks": 2.0},
            {"criterion": "Overriding", "marks": 2.0}
        ]
    }

    result = grading_module.grade(
        student_answer=student_ans,
        reference_answer=ref_ans,
        rubric=rubric,
        semantic_score=0.88,
        confidence=0.92
    )

    # Marks must be half or whole number
    assert result.marks in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    assert result.marks <= 5.0
    assert len(result.criteria_breakdown) == 3
    for crit in result.criteria_breakdown:
        # Each criterion must be half or whole number
        assert crit["awarded_marks"] in [0.0, 0.5, 1.0, 1.5, 2.0]


def test_explicit_numeric_rubric_awards_correct_checkpoints_only():
    reference_answer = (
        "Largest value = 20\n"
        "Smallest value = 3\n"
        "Sum = 48\n"
        "Average = 9.6"
    )
    rubric = {
        "max_marks": 3,
        "criteria": (
            "Award 1 mark for the correct largest value (20), "
            "1 mark for the correct smallest value (3), and "
            "1 mark for the correct sum (48) and average (9.6). "
            "If the method is correct but arithmetic contains a minor error, "
            "award appropriate partial credit."
        ),
    }

    incorrect_largest = grading_module.grade(
        student_answer="Largest number : 80\nSmallest number : 3\nSum : 48\nAvg : 9.6",
        reference_answer=reference_answer,
        rubric=rubric,
        semantic_score=0.42,
        confidence=0.5,
    )
    correct_answer = grading_module.grade(
        student_answer="Largest number : 20\nSmallest number : 3\nSum : 48\nAvg : 9.6",
        reference_answer=reference_answer,
        rubric=rubric,
        semantic_score=0.42,
        confidence=0.5,
    )

    assert incorrect_largest.marks == 2.0
    assert [item["awarded_marks"] for item in incorrect_largest.criteria_breakdown] == [
        0.0, 1.0, 1.0
    ]
    assert correct_answer.marks == 3.0


def test_finetuned_model_evaluation_uses_rubric_checkpoint_marks(tmp_path):
    sample = {
        "student_answer": (
            "Largest number : 80\nSmallest number : 3\nSum : 48\nAvg : 9.6"
        ),
        "reference_answer": (
            "Largest value = 20\nSmallest value = 3\nSum = 48\nAverage = 9.6"
        ),
        "rubric": {
            "max_marks": 3,
            "criteria": (
                "Award 1 mark for the correct largest value (20), "
                "1 mark for the correct smallest value (3), and "
                "1 mark for the correct sum (48) and average (9.6)."
            ),
        },
        "human_marks": 2,
    }
    test_file = tmp_path / "test_split.jsonl"
    test_file.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    report = evaluate_model_on_test_set("codbert_ft_v14", test_file=test_file)

    assert report["mean_absolute_error"] == 0.0
    assert report["accuracy_within_0_5_marks"] == 100.0


def test_prose_rubric_scores_numbered_answer_points():
    """Prose rubrics should award marks for correctly answered numbered points."""
    cases = [
        (
            "Temperature\nArea of cross section\nResistance\nMaterials",
            (
                "The resistance of a conductor depends on:\n"
                "1. Length of the conductor\n"
                "2. Cross-sectional area (diameter)\n"
                "3. Nature of the material (resistivity)\n"
                "4. Temperature"
            ),
            "Award full marks if the student mentions length, cross-sectional area, "
            "material/resistivity, and temperature. Deduct 0.5 marks for each missing factor.",
            1.5,
        ),
        (
            "Change in state\nChange in colour\nEvolution of a gas\nChange in temperature",
            (
                "Observations indicating a chemical reaction:\n"
                "1. Change in colour\n"
                "2. Evolution of gas\n"
                "3. Formation of precipitate\n"
                "4. Change in temperature\n"
                "5. Change in smell or state"
            ),
            "Award full marks if any four valid observations of a chemical reaction are "
            "mentioned. Award 0.5 marks per correct observation.",
            2.0,
        ),
        (
            "Absorption of light energy by chlorophyll\n"
            "Conversion of light energy to chemical energy and splitting of water molecule "
            "into hydrogen and oxygen\n"
            "Reduction of carbon dioxide to carbohydrate",
            (
                "Events during photosynthesis:\n"
                "1. Absorption of light energy by chlorophyll.\n"
                "2. Conversion of light energy into chemical energy and splitting of water "
                "into hydrogen and oxygen.\n"
                "3. Reduction of carbon dioxide to carbohydrates (glucose)."
            ),
            "Award full marks if the student mentions all three events.",
            2.0,
        ),
    ]

    for student_answer, reference_answer, criteria, expected_marks in cases:
        result = grading_module.grade(
            student_answer=student_answer,
            reference_answer=reference_answer,
            rubric={"max_marks": 2.0, "criteria": criteria},
            semantic_score=0.1,
            confidence=0.8,
        )
        assert result.marks == expected_marks
        assert result.criteria_breakdown[0]["criterion"] == "Listed answer points"


def test_pdf_rendering_without_tesseract():
    """Verify PyMuPDF renders PDF to images with zero Tesseract usage."""
    document = fitz.open()
    document.new_page()
    pdf_bytes = document.tobytes()
    document.close()
    assert is_pdf("sample.pdf")

    images = render_pdf_to_images(pdf_bytes, dpi=100)
    assert len(images) >= 1
    # Check PNG signature
    assert images[0].startswith(b'\x89PNG')

def test_hitl_approval_and_rejection():
    """Verify HITL workflow strictly stores only human_verified = True records."""
    init_db()
    db = SessionLocal()

    # Stage items
    items = hitl_service.create_pending_items(
        evaluation_id="test-eval-123",
        question_results=[
            {"question": "1", "marks": 4.0, "max_marks": 5.0, "feedback": "Good", "confidence": 0.9, "semantic_score": 0.85},
            {"question": "2", "marks": 2.0, "max_marks": 5.0, "feedback": "Partial", "confidence": 0.7, "semantic_score": 0.5}
        ],
        key_answers={"1": "Ref 1", "2": "Ref 2"},
        rubrics={"1": {"max_marks": 5}, "2": {"max_marks": 5}},
        student_answers={"1": "Student 1", "2": "Student 2"},
        db=db
    )
    assert len(items) == 2

    # Approve item 1 with adjusted half mark
    approved = hitl_service.approve_item(
        item_id=items[0].id,
        human_marks=4.5,
        human_feedback="Human verified: Excellent definition",
        db=db
    )
    assert approved.human_verified is True
    assert approved.human_marks == 4.5
    assert approved.status == "approved"

    # Reject item 2
    rejected = hitl_service.reject_item(
        item_id=items[1].id,
        reason="Illegible handwriting",
        db=db
    )
    assert rejected.human_verified is False
    assert rejected.status == "rejected"

    db.close()


def test_reject_all_pending_hitl_items_leaves_approved_items_unchanged():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine, tables=[HITLItem.__table__])
    db = sessionmaker(bind=engine)()
    db.add_all([
        HITLItem(
            evaluation_id="bulk-reject-test",
            question_id=str(index),
            student_answer="Answer",
            reference_answer="Reference",
            rubric_json="{}",
            ai_marks=0.0,
            status=status,
            human_verified=verified,
        )
        for index, status, verified in [
            (1, "pending", False),
            (2, "pending", False),
            (3, "approved", True),
        ]
    ])
    db.commit()

    assert hitl_service.reject_all_pending(db) == 2
    items = {item.question_id: item for item in db.query(HITLItem).all()}
    assert items["1"].status == "rejected"
    assert items["1"].human_verified is False
    assert items["2"].status == "rejected"
    assert items["3"].status == "approved"
    assert items["3"].human_verified is True

    db.close()
    engine.dispose()


def test_fine_tuning_pipeline():
    """Verify fine-tuning pipeline creates versioned model and metadata."""
    init_db()
    db = SessionLocal()

    # Ensure there are samples in verified_training.jsonl
    training_file = Path("data/verified_training.jsonl")
    training_file.parent.mkdir(parents=True, exist_ok=True)
    with open(training_file, "a", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({
                "question": str(i+1),
                "student_answer": f"Polymorphism allows overloading and overriding in question {i+1}.",
                "reference_answer": "Polymorphism means many forms including compile-time and runtime.",
                "rubric": {"max_marks": 5.0},
                "ai_marks": 3.5,
                "human_marks": 4.0,
                "ai_feedback": "Good answer",
                "human_feedback": "Verified 4.0 marks",
                "human_verified": True
            }) + "\n")

    # Run fine-tuning
    meta = run_fine_tuning(db)
    assert "model_version" in meta
    assert meta["model_version"].startswith("codbert_ft_v")
    assert meta["training_samples"] >= 1
    assert "validation_score" in meta

    # Check model directory artifacts
    model_dir = Path("models") / meta["model_version"]
    assert model_dir.exists()
    assert (model_dir / "weights.json").exists()
    assert (model_dir / "metadata.json").exists()
    with open(model_dir / "weights.json", encoding="utf-8") as f:
        weights = json.load(f)
    assert abs(weights["semantic_weight"] + weights["overlap_weight"] - 1.0) < 1e-6
    assert abs(weights["bias"]) <= 0.1

    db.close()
