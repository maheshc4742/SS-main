import json
import base64
import logging
import re
from typing import Dict, Optional, List
from urllib.parse import urlsplit, urlunsplit
import httpx

from backend.ocr.base_ocr import BaseOCR, OCRProviderError
from backend.ocr.pdf_utils import is_pdf, render_pdf_to_images
from backend.core.config import settings

logger = logging.getLogger(__name__)

class QwenOCR(BaseOCR):
    """
    OCR implementation connecting to a locally running Qwen vision-language model.
    Connects via OpenAI-compatible vision chat completions endpoint (vLLM, Ollama, LM Studio, etc.).
    Fully configurable via QWEN_BASE_URL and QWEN_MODEL.
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = (base_url or settings.QWEN_BASE_URL).rstrip("/")
        self.model = model or settings.QWEN_MODEL or "qwen2-vl-7b-instruct"
        self.api_key = api_key or settings.QWEN_API_KEY or "not-needed"

    async def extract_answers(
        self,
        file_bytes: bytes,
        filename: str,
        questions_hint: Optional[List[str]] = None
    ) -> Dict[str, str]:
        # 1. Render/prepare images
        image_content_parts = []
        if is_pdf(filename):
            page_images = render_pdf_to_images(file_bytes)
            for img_bytes in page_images:
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                image_content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}"
                    }
                })
        else:
            mime = "image/jpeg"
            if filename.lower().endswith(".png"):
                mime = "image/png"
            b64 = base64.b64encode(file_bytes).decode("utf-8")
            image_content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{b64}"
                }
            })

        hint_text = ""
        if questions_hint:
            hint_text = f"The questions to extract are: {', '.join(questions_hint)}."

        prompt_text = (
            "You are a careful OCR transcription assistant specializing in handwritten exam solutions.\n"
            "Examine every attached image and page of the student's answer sheet.\n"
            f"{hint_text}\n"
            "Transcribe each answer in full, from its first word or step through its final word or step. "
            "Do not summarize, paraphrase, correct, or omit any readable content, including explanations, "
            "working steps, equations, examples, and conclusions. Preserve the student's wording and the "
            "original question and subquestion identifiers (for example, 1a or 1.a). Keep paragraph and "
            "line breaks where practical. If a word is genuinely unreadable, mark it as [illegible] rather "
            "than guessing. Do not stop after the first part of a long answer.\n"
            "Organize the complete transcriptions strictly as a JSON object mapping each question or "
            "subquestion identifier to its full answer text.\n"
            "Example format:\n"
            '{\n  "1": "complete transcription of every readable step in the answer...",\n'
            '  "2": "complete transcription of the next answer..."\n}\n'
            "Output ONLY the JSON object. Do not include markdown or explanations."
        )

        content = [{"type": "text", "text": prompt_text}] + image_content_parts
        messages = [
            {"role": "system", "content": "You are a precise handwritten OCR transcription system. Transcribe complete answers without summarizing or omitting readable content, and respond solely with valid JSON."},
            {"role": "user", "content": content}
        ]

        base_url = urlsplit(self.base_url)
        api_path = base_url.path.rstrip("/")
        if not api_path:
            api_path = "/v1"
        endpoint = urlunsplit((
            base_url.scheme,
            base_url.netloc,
            f"{api_path}/chat/completions",
            base_url.query,
            "",
        ))

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                headers = {"Content-Type": "application/json"}
                if self.api_key and self.api_key != "not-needed":
                    headers["Authorization"] = f"Bearer {self.api_key}"

                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 2048
                }

                response = await client.post(endpoint, json=payload, headers=headers)
                if response.status_code == 200:
                    try:
                        data = response.json()
                    except ValueError as e:
                        raise OCRProviderError(
                            "Local Qwen OCR returned a non-JSON response. Verify the configured Qwen API URL."
                        ) from e

                    if not isinstance(data, dict) or "error" in data:
                        raise OCRProviderError(
                            "Local Qwen OCR endpoint returned an error. Verify the configured API URL and loaded model."
                        )

                    choices = data.get("choices")
                    if not isinstance(choices, list) or not choices:
                        raise OCRProviderError(
                            "Local Qwen OCR returned no completion choices. Verify the configured API URL and loaded model."
                        )

                    message = choices[0].get("message") if isinstance(choices[0], dict) else None
                    content_str = message.get("content") if isinstance(message, dict) else None
                    if isinstance(content_str, list):
                        content_str = "".join(
                            part.get("text", "")
                            for part in content_str
                            if isinstance(part, dict) and isinstance(part.get("text"), str)
                        )
                    if not isinstance(content_str, str) or not content_str.strip():
                        raise OCRProviderError("Local Qwen OCR returned empty or unsupported completion content.")
                    return self._parse_json_response(content_str, questions_hint)
                else:
                    raise OCRProviderError(f"Local Qwen OCR returned HTTP {response.status_code}.")

        except Exception as e:
            if isinstance(e, OCRProviderError):
                raise
            logger.warning(f"Could not reach Local Qwen endpoint at {self.base_url}: {e}")
            raise OCRProviderError("Local Qwen OCR request failed. Check that the configured endpoint is running.") from e

    def _parse_json_response(self, text: str, questions_hint: Optional[List[str]] = None) -> Dict[str, str]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback regex
        result = {}
        matches = re.findall(r'["\']?(\d+[a-zA-Z]?)["\']?\s*:\s*["\']([^"\']+)["\']', cleaned)
        for q, ans in matches:
            result[str(q)] = ans.strip()

        if not result:
            expected_questions = f" for questions {', '.join(questions_hint)}" if questions_hint else ""
            logger.warning("Qwen OCR response contained no parseable answer mapping%s.", expected_questions)
            raise OCRProviderError("Local Qwen OCR response did not contain a valid JSON answer mapping.")

        return result
