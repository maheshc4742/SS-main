import asyncio
import json
import base64
import logging
import re
from typing import Dict, Optional, List
import httpx

from backend.ocr.base_ocr import BaseOCR, OCRProviderError
from backend.ocr.pdf_utils import is_pdf, render_pdf_to_images
from backend.core.config import settings

logger = logging.getLogger(__name__)
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_MAX_REQUEST_ATTEMPTS = 3

class GeminiOCR(BaseOCR):
    """
    OCR implementation using Google Gemini API Vision capabilities.
    Supports single or multi-page image and PDF scripts without Tesseract.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL or "gemini-3.8-flash"

    async def extract_answers(
        self,
        file_bytes: bytes,
        filename: str,
        questions_hint: Optional[List[str]] = None
    ) -> Dict[str, str]:
        # 1. Prepare images from PDF or image files
        image_inputs = []
        if is_pdf(filename):
            page_images = render_pdf_to_images(file_bytes)
            for img_bytes in page_images:
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                image_inputs.append({
                    "type": "image",
                    "mime_type": "image/png",
                    "data": b64
                })
        else:
            mime = "image/jpeg"
            if filename.lower().endswith(".png"):
                mime = "image/png"
            b64 = base64.b64encode(file_bytes).decode("utf-8")
            image_inputs.append({
                "type": "image",
                "mime_type": mime,
                "data": b64
            })

        # Check API key
        if not self.api_key or self.api_key.strip() == "":
            raise OCRProviderError("Gemini OCR requires GEMINI_API_KEY. Configure a key or submit student answers as JSON.")

        hint_text = ""
        if questions_hint:
            hint_text = f"The exam contains the following question identifiers: {', '.join(questions_hint)}. Ensure you extract each available answer corresponding to these numbers."

        prompt= (
            "You are a high-accuracy OCR system for handwritten student examination answer sheets. "
            "Read every uploaded image and every page carefully, in normal reading order, and transcribe "
            "the student's handwritten answers exactly as written. This is transcription only, not "
            "answering, grading, correcting, or generating missing content.\n"

            f"{hint_text}\n"

            "First identify each handwritten question number and subquestion label, then extract the answer "
            "written under or beside it. Preserve the original question numbering and order. Include every "
            "answer, even if it consists of a single word, short phrase, incomplete sentence, code snippet, "
            "bullet point, calculation, or rough work. Never omit an answer because it appears incorrect, "
            "incomplete, or unrelated to the expected answer.\n"

            "Transcribe only the student's answers, not printed questions, marks, page numbers, or unrelated "
            "background text. Preserve original wording, spelling, capitalization, punctuation, numbers, "
            "symbols, and answer order. Do not summarize, paraphrase, correct, solve, or complete anything. "
            "If a word or character is unreadable, use [illegible] only for that portion. Do not guess.\n"

            "QUESTION IDENTIFICATION:\n"
            "Preserve question and subquestion identifiers as handwritten, including labels such as "
            "'1', '2', '11a', '11b', '11(a)', and '11(b)'. Do not force identifiers into a predefined "
            "sequence or change them to match expected-identifier hints. Keep each subquestion as a separate "
            "JSON entry. Never overwrite one answer with another. If an answer continues across lines or "
            "pages, combine it with the original answer when the continuation is clear. Do not invent "
            "question identifiers. If an answer is visible but its question number cannot be determined, "
            "use 'unknown_1', 'unknown_2', and so on.\n"

            "PROGRAMMING CODE:\n"
            "Transcribe code exactly as written, including keywords, variable names, capitalization, "
            "operators, brackets, punctuation, and indentation. Preserve the student's code even if it "
            "contains syntax errors. Do not execute, debug, or correct it.\n"

            "MATHEMATICS AND SCIENCE:\n"
            "Transcribe all mathematical expressions, scientific formulas, equations, and calculations "
            "faithfully. Use LaTeX notation for mathematical expressions when appropriate. Preserve "
            "fractions, exponents, subscripts, roots, matrices, operators, chemical formulas, units, "
            "reaction arrows, and intermediate calculations. Do not solve, simplify, balance, or correct "
            "expressions. If a symbol is genuinely unclear, use [unclear symbol]. Ensure LaTeX backslashes "
            "are escaped correctly in the JSON output.\n"

            "DIAGRAMS AND GRAPHS:\n"
            "Describe each visible student-drawn diagram, graph, circuit, flowchart, or scientific "
            "illustration within the corresponding answer string using the format '[Diagram: description]'. "
            "Include readable labels, arrows, directions, components, connections, measurements, axis labels, "
            "and important relationships. Preserve labels exactly as written. Do not invent labels or "
            "describe features that cannot be reliably observed.\n"

            "BULLET POINTS AND MULTILINE ANSWERS:\n"
            "Preserve all bullet points, numbered steps, and lines of the answer in their original order. "
            "Represent line breaks using newline characters inside the JSON strings. Keep all content "
            "belonging to the same question together in one string. Preserve readable tables using text "
            "or Markdown table syntax inside the answer string.\n"

            "OUTPUT FORMAT:\n"
            "Return exactly one valid JSON object mapping each question identifier to its complete "
            "transcribed answer as a string. Use unique string keys and string values only. Do not use "
            "nested objects, arrays, confidence scores, marks, or extra fields.\n"

            "Example structure only:\n"
            '{"1":"Python",'
            '"2":"A variable is like a containers used to store data in a program.",'
            '"11a":"Select name from student where marks > 80",'
            '"11b":"for i in range (5):\\n    Print (i)",'
            '"12":"Binary Search is a searching technique.\\nTime complexity\\nO (log n)"}\n'

            "The example illustrates the output structure only. Never copy example content into the "
            "actual output unless it is present in the uploaded image.\n"

            "Before responding, verify that every page, question, short answer, code snippet, calculation, "
            "diagram, and readable label has been considered. Ensure that no answer has been omitted, "
            "corrected, generated, or assigned to the wrong question. Escape double quotes, backslashes, "
            "and newlines correctly so the result can be parsed by a standard JSON parser. If no student "
            "answers are visible, return {}.\n"

            "Return JSON only, without Markdown code fences or explanations."
        )

        url = "https://generativelanguage.googleapis.com/v1beta/interactions"
        interaction_input = [{"type": "text", "text": prompt}] + image_inputs

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                for attempt in range(_MAX_REQUEST_ATTEMPTS):
                    try:
                        response = await client.post(
                            url,
                            json={"model": self.model, "input": interaction_input},
                            headers={
                                "Content-Type": "application/json",
                                "x-goog-api-key": self.api_key,
                            }
                        )
                    except (httpx.TimeoutException, httpx.NetworkError):
                        if attempt == _MAX_REQUEST_ATTEMPTS - 1:
                            raise
                        delay = 2 ** attempt
                        logger.warning(
                            "Gemini OCR request failed temporarily; retrying in %s second(s) "
                            "(attempt %s/%s).",
                            delay,
                            attempt + 1,
                            _MAX_REQUEST_ATTEMPTS,
                        )
                        await asyncio.sleep(delay)
                        continue

                    if (
                        response.status_code not in _RETRYABLE_STATUS_CODES
                        or attempt == _MAX_REQUEST_ATTEMPTS - 1
                    ):
                        break

                    delay = 2 ** attempt
                    logger.warning(
                        "Gemini OCR returned HTTP %s; retrying in %s second(s) "
                        "(attempt %s/%s).",
                        response.status_code,
                        delay,
                        attempt + 1,
                        _MAX_REQUEST_ATTEMPTS,
                    )
                    await asyncio.sleep(delay)

                if response.status_code != 200:
                    provider_message = self._get_provider_error_message(response)

                    if response.status_code == 404:
                        detail = f" Google API detail: {provider_message}." if provider_message else ""
                        raise OCRProviderError(
                            f"Gemini model '{self.model}' was not found or is unavailable to this API key "
                            f"(HTTP 404).{detail} Check GEMINI_MODEL and its API access."
                        )
                    detail = f" Google API detail: {provider_message}." if provider_message else ""
                    attempts = (
                        f" after {_MAX_REQUEST_ATTEMPTS} attempts"
                        if response.status_code in _RETRYABLE_STATUS_CODES
                        else ""
                    )
                    raise OCRProviderError(
                        f"Gemini OCR returned HTTP {response.status_code}{attempts}.{detail}"
                    )

                try:
                    data = response.json()
                    output_steps = [
                        step for step in data["steps"]
                        if step.get("type") == "model_output"
                    ]
                    text_parts = [
                        content.get("text", "")
                        for content in output_steps[-1]["content"]
                        if content.get("type") == "text"
                    ]
                    text_out = "".join(text_parts)
                    if not text_out:
                        raise ValueError("Gemini interaction contained no text output.")
                except (ValueError, KeyError, IndexError, TypeError) as e:
                    raise OCRProviderError(
                        "Gemini OCR returned an invalid Interactions API response. "
                        "Check the model's Interactions API access."
                    ) from e
                return self._parse_json_response(text_out, questions_hint)

        except Exception as e:
            logger.exception(f"Exception during Gemini OCR call: {e}")
            if isinstance(e, OCRProviderError):
                raise
            raise OCRProviderError("Gemini OCR request failed.") from e

    def _get_provider_error_message(self, response: httpx.Response) -> str:
        try:
            response_data = response.json()
        except (ValueError, TypeError):
            return ""
        error_data = response_data.get("error") if isinstance(response_data, dict) else None
        provider_message = error_data.get("message") if isinstance(error_data, dict) else None
        if not isinstance(provider_message, str):
            return ""
        return provider_message.replace(self.api_key, "[redacted]")[:500]

    def _parse_json_response(self, text: str, questions_hint: Optional[List[str]] = None) -> Dict[str, str]:
        cleaned = text.strip()
        # Remove markdown code block if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except Exception:
            logger.warning(f"Could not parse Gemini output as raw JSON: {cleaned[:100]}... Falling back to regex.")

        # Fallback regex extraction for question patterns
        result = {}
        matches = re.findall(r'["\']?(\d+[a-zA-Z]?)["\']?\s*:\s*["\']([^"\']+)["\']', cleaned)
        for q, ans in matches:
            result[str(q)] = ans.strip()

        if not result:
            logger.warning("Gemini OCR response contained no parseable answers.")
            raise OCRProviderError("Gemini OCR response did not contain a valid JSON answer mapping.")

        return result
