"""Exercise the Gemini REST contract without external credentials."""

from io import BytesIO
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from nanbeige_gemini import GeminiClient
from nanbeige_watcher_api import AssessmentError


def response(*parts, finish_reason="STOP"):
    candidate = {"content": {"parts": parts}, "finishReason": finish_reason}
    return BytesIO(json.dumps({"candidates": [candidate]}).encode())


class GeminiClientTests(unittest.TestCase):
    def test_assess_uses_structured_output_and_reads_visible_text(self):
        observation = {"turns": [{"role": "user", "content": "Remember café"}]}
        client = GeminiClient("test-key", timeout=4.0)
        reply = response(
            {"text": "Hidden scratchpad", "thought": True},
            {"text": '{"compress": true, '},
            {"text": '"summary": "User mentioned café."}'},
        )

        with patch("nanbeige_watcher_api.urlopen", return_value=reply) as open_url:
            result = client.assess("Assess the context.", observation)

        request = open_url.call_args.args[0]
        self.assertEqual(open_url.call_args.kwargs, {"timeout": 4.0})
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(
            request.full_url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-4-hydrogen:generateContent",
        )
        self.assertEqual(request.get_header("X-goog-api-key"), "test-key")
        self.assertNotIn("test-key", request.data.decode())
        body = json.loads(request.data)
        self.assertEqual(body["systemInstruction"]["parts"], [{"text": "Assess the context."}])
        text_format = body["generationConfig"]["responseFormat"]["text"]
        self.assertEqual(text_format["mimeType"], "APPLICATION_JSON")
        self.assertEqual(text_format["schema"]["required"], ["compress", "summary"])
        self.assertFalse(text_format["schema"]["additionalProperties"])
        self.assertEqual(body["generationConfig"]["thinkingConfig"]["thinkingLevel"], "LOW")
        self.assertEqual(json.loads(body["contents"][0]["parts"][0]["text"]), observation)
        self.assertEqual(result, {"compress": True, "summary": "User mentioned café."})

    def test_rejects_malformed_or_incomplete_replies(self):
        replies = [
            response({"text": '{"compress": "true", "summary": "Private text"}'}),
            response({"text": '{"compress": true, "summary": "Private text"}'}, finish_reason="MAX_TOKENS"),
            BytesIO(b'{"promptFeedback":{"blockReason":"SAFETY"}}'),
        ]
        for reply in replies:
            with self.subTest(reply=reply), patch("nanbeige_watcher_api.urlopen", return_value=reply):
                with self.assertRaises(AssessmentError) as raised:
                    GeminiClient("test-key").assess("system", {"secret": "Private text"})
                self.assertNotIn("Private text", str(raised.exception))

    def test_http_and_timeout_errors_exclude_provider_details(self):
        body = BytesIO(b"Private text and test-key")
        quota_error = HTTPError("https://example.com", 429, "test-key", {}, body)
        cases = [(quota_error, 429, "quota"), (URLError(TimeoutError("test-key")), None, "timed out")]
        for failure, status, message in cases:
            with self.subTest(status=status), patch("nanbeige_watcher_api.urlopen", side_effect=failure):
                with self.assertRaisesRegex(AssessmentError, message) as raised:
                    GeminiClient("test-key").assess("system", {"secret": "Private text"})
                self.assertEqual(raised.exception.status, status)
                self.assertNotIn("test-key", str(raised.exception))
                self.assertNotIn("Private text", str(raised.exception))
        self.assertTrue(body.closed)
