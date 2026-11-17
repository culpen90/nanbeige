"""Request a context-compression decision from the Gemini Developer API."""

import json
import math
import re

from nanbeige_watcher_api import ASSESSMENT_SCHEMA, AssessmentError, parse_assessment, post_json


DEFAULT_GEMINI_MODEL = "gemini-4-hydrogen"
_API = "https://generativelanguage.googleapis.com/v1beta/models"
_HTTP_ERRORS = {
    400: "Gemini rejected the request (HTTP 400). Check the AI Studio key and model.",
    401: "Gemini rejected the AI Studio key (HTTP 401).",
    403: "Gemini denied access (HTTP 403). Check the AI Studio key and project access.",
    404: "The selected model is unavailable to Gemini GenerateContent (HTTP 404).",
    429: "Gemini rate limit or quota reached (HTTP 429).",
}


class GeminiClient:
    provider_name = "Gemini"

    def __init__(self, api_key: str, model: str = DEFAULT_GEMINI_MODEL, timeout: float = 12.0):
        self._api_key = api_key.strip()
        self.model = model.strip().removeprefix("models/")
        self.timeout = timeout
        if not re.fullmatch(r"[!-~]+", self._api_key):
            raise ValueError("Supply a valid Google AI Studio API key.")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.model):
            raise ValueError("Supply a Gemini model ID such as gemini-4-hydrogen.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("The Gemini request timeout must be a positive finite number.")

    def assess(self, system: str, observation: dict) -> dict:
        """Send a stateless observation and validate the model's decision."""
        generation = {
            "maxOutputTokens": 4096,
            "responseFormat": {
                "text": {"mimeType": "APPLICATION_JSON", "schema": ASSESSMENT_SCHEMA},
            },
        }
        if self.model == DEFAULT_GEMINI_MODEL:
            generation["thinkingConfig"] = {"thinkingLevel": "LOW"}
        observed = json.dumps(observation, ensure_ascii=False, separators=(",", ":"))
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": observed}]}],
            "generationConfig": generation,
        }
        response = post_json(
            f"{_API}/{self.model}:generateContent", payload, auth_header=("x-goog-api-key", self._api_key),
            timeout=self.timeout, provider=self.provider_name, http_errors=_HTTP_ERRORS,
        )
        try:
            candidate = response["candidates"][0]
            if candidate.get("finishReason") not in (None, "STOP"):
                raise AssessmentError("Gemini stopped before completing the assessment.")
            parts = candidate["content"]["parts"]
            text = "".join(part["text"] for part in parts if "text" in part and not part.get("thought"))
            return parse_assessment(text)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RecursionError):
            raise AssessmentError("Gemini returned an invalid assessment.") from None
