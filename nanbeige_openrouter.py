"""Request a context-compression decision from OpenRouter."""

import json
import math
import re

from nanbeige_watcher_api import ASSESSMENT_SCHEMA, AssessmentError, parse_assessment, post_json


_API = "https://openrouter.ai/api/v1/chat/completions"
_MODEL_ID = re.compile(r"~?[\w.-]+/[\w.-]+(?::[\w.-]+)*", re.ASCII)
_HTTP_ERRORS = {
    400: "OpenRouter rejected the request (HTTP 400). Check the model ID.",
    401: "OpenRouter rejected the API key (HTTP 401).",
    402: "OpenRouter reports insufficient credits (HTTP 402).",
    403: "OpenRouter denied the request (HTTP 403). Check the key and account settings.",
    404: "OpenRouter found no compatible endpoint (HTTP 404). Check that the model supports structured outputs.",
    429: "OpenRouter rate limit reached (HTTP 429).",
    503: "OpenRouter has no available endpoint (HTTP 503).",
}


class OpenRouterClient:
    provider_name = "OpenRouter"

    def __init__(self, api_key: str, model: str, timeout: float = 12.0):
        self._api_key = api_key.strip()
        self.model = model.strip()
        self.timeout = timeout
        if not re.fullmatch(r"[!-~]+", self._api_key):
            raise ValueError("Supply a valid OpenRouter API key.")
        if not _MODEL_ID.fullmatch(self.model):
            raise ValueError("Supply an OpenRouter model ID such as openai/gpt-oss-20b.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("The OpenRouter request timeout must be a positive finite number.")

    def assess(self, system: str, observation: dict) -> dict:
        observed = json.dumps(observation, ensure_ascii=False, separators=(",", ":"))
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": observed},
            ],
            "stream": False,
            "max_tokens": 4096,
            "provider": {"require_parameters": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "context_compression_decision",
                    "strict": True,
                    "schema": ASSESSMENT_SCHEMA,
                },
            },
        }
        response = post_json(
            _API, payload, auth_header=("Authorization", f"Bearer {self._api_key}"),
            timeout=self.timeout, provider=self.provider_name, http_errors=_HTTP_ERRORS,
        )
        return _read_assessment(response)


def _read_assessment(response: dict) -> dict:
    try:
        if response.get("error") is not None:
            raise _provider_error(response["error"])
        choice = response["choices"][0]
        if choice.get("error") is not None:
            raise _provider_error(choice["error"])
        message = choice["message"]
        if choice.get("finish_reason") != "stop" or message.get("refusal"):
            raise AssessmentError("OpenRouter stopped before completing the assessment.")
        return parse_assessment(message["content"])
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RecursionError):
        raise AssessmentError("OpenRouter returned an invalid assessment.") from None


def _provider_error(error) -> AssessmentError:
    status = error.get("code") if isinstance(error, dict) else None
    metadata = error.get("metadata") if isinstance(error, dict) else None
    if isinstance(metadata, dict) and metadata.get("error_type") == "rate_limit_exceeded":
        status = 429
    if type(status) is not int or not 400 <= status <= 599:
        return AssessmentError("OpenRouter failed while generating the assessment.")
    return AssessmentError(
        _HTTP_ERRORS.get(status, f"OpenRouter failed while generating the assessment (code {status})."),
        status=status,
    )
