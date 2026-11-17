"""HTTP transport and decision validation shared by context watcher providers."""

from http.client import HTTPException
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_MAX_RESPONSE_BYTES = 128 * 1024
ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "compress": {"type": "boolean"},
        "summary": {"type": "string"},
    },
    "required": ["compress", "summary"],
    "additionalProperties": False,
}


class AssessmentError(RuntimeError):
    """A provider failure safe to report without exposing the request or reply."""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


def post_json(url, payload, *, auth_header, timeout, provider, http_errors):
    request = Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    request.add_unredirected_header(*auth_header)

    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read(_MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        status = error.code
        error.close()
        message = http_errors.get(status, f"{provider} request failed (HTTP {status}).")
        raise AssessmentError(message, status=status) from None
    except (OSError, HTTPException) as error:
        reason = error.reason if isinstance(error, URLError) else error
        if isinstance(reason, TimeoutError):
            raise AssessmentError(f"{provider} request timed out.") from None
        raise AssessmentError(f"{provider} network request failed.") from None

    if len(body) > _MAX_RESPONSE_BYTES:
        raise AssessmentError(f"{provider} returned an oversized response.")
    try:
        result = json.loads(body)
    except (ValueError, RecursionError):
        raise AssessmentError(f"{provider} returned an invalid response.") from None
    if not isinstance(result, dict):
        raise AssessmentError(f"{provider} returned an invalid response.")
    return result


def parse_assessment(text: str) -> dict:
    result = json.loads(text)
    if (not isinstance(result, dict) or result.keys() != {"compress", "summary"}
            or not isinstance(result["compress"], bool) or not isinstance(result["summary"], str)):
        raise ValueError("Invalid compression decision.")
    result["summary"].encode("utf-8")
    return result
