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
            hint_text = (
                f"Expected question identifiers include: {', '.join(questions_hint)}. "
                "These are hints, not a limit: also transcribe any other clearly labeled "
                "question or subquestion found on the pages.\n"
            )


        prompt = (
            "You are a high-accuracy OCR system for handwritten student examination answer sheets. "
            "Read every uploaded image and every page carefully, in normal reading order, and transcribe "
            "the student's handwritten answers exactly as written. This is transcription only, not "
            "answering, grading, correcting, or generating missing content.\n"

            f"{hint_text}\n"

            "First identify each handwritten question number and subquestion label, then extract the answer "
            "written under or beside it. Preserve the original question numbering and order. Include all "
            "detected labels. Include brief answers and fragments too, even if an answer consists of a "
            "single word, short phrase, incomplete sentence, code snippet, "
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


        content = [{"type": "text", "text": prompt}] + image_content_parts
        messages = [
            {"role": "system", "content": "Transcribe all visible student answers faithfully, including brief answers and unexpected question labels. Do not answer or grade. Return only valid JSON."},
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
