"""Verify OpenRouter request and response handling without external credentials."""

from io import BytesIO
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from nanbeige_openrouter import OpenRouterClient
from nanbeige_watcher_api import AssessmentError


def _response(choice):
    return BytesIO(json.dumps({"choices": [choice]}).encode())


class OpenRouterClientTests(unittest.TestCase):
    def test_requests_strict_json_from_selected_model(self):
        model = "liquid/lfm-3-1.7b:free"
        observation = {"older_messages": [{"role": "user", "content": "Meet at the café"}]}
        decision = {"compress": True, "summary": "Meet at the café."}
        reply = _response({
            "finish_reason": "stop",
            "message": {"content": json.dumps(decision), "reasoning": "Provider scratchpad"},
        })

        with patch("nanbeige_watcher_api.urlopen", return_value=reply) as send:
            result = OpenRouterClient("test-key", model, timeout=3).assess("Assess chat.", observation)

        self.assertEqual(result, decision)
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(send.call_args.kwargs["timeout"], 3)
        self.assertNotIn(b"test-key", request.data)
        body = json.loads(request.data)
        self.assertEqual(body["model"], model)
        self.assertFalse(body["stream"])
        self.assertEqual(body["provider"]["require_parameters"], True)
        self.assertEqual(body["messages"][0], {"role": "system", "content": "Assess chat."})
        self.assertEqual(body["messages"][1]["role"], "user")
        self.assertEqual(json.loads(body["messages"][1]["content"]), observation)
        format = body["response_format"]
        self.assertEqual(format["type"], "json_schema")
        self.assertTrue(format["json_schema"]["strict"])
        self.assertFalse(format["json_schema"]["schema"]["additionalProperties"])
        self.assertCountEqual(format["json_schema"]["schema"]["required"], ["compress", "summary"])

    def test_rejects_schema_violations_and_truncated_responses(self):
        client = OpenRouterClient("test-key", "openai/gpt-oss-20b")
        for reason, content in (
            ("stop", '{"compress": "yes", "summary": "private text"}'),
            ("length", '{"compress": true, "summary": "private text"}'),
        ):
            reply = _response({"finish_reason": reason, "message": {"content": content}})
            with self.subTest(reason=reason), patch("nanbeige_watcher_api.urlopen", return_value=reply):
                with self.assertRaises(AssessmentError) as caught:
                    client.assess("Assess chat.", {})
            self.assertNotIn("private text", str(caught.exception))

    def test_reports_http_and_in_band_failures_without_provider_details(self):
        client = OpenRouterClient("test-key", "openai/gpt-oss-20b")
        private_message = "provider details with private text and test-key"
        failure = HTTPError("https://openrouter.ai", 402, private_message, {}, BytesIO(b"private text"))
        with patch("nanbeige_watcher_api.urlopen", side_effect=failure):
            with self.assertRaisesRegex(AssessmentError, "insufficient credits") as caught:
                client.assess("Assess chat.", {})
        self.assertEqual(caught.exception.status, 402)
        self.assertNotIn(private_message, str(caught.exception))

        limited = {"error": {"code": 429, "message": private_message}}
        unavailable = {
            "choices": [{
                "finish_reason": "error",
                "error": {"code": 502, "message": private_message},
                "message": {"content": "private text"},
            }],
        }
        for response, status in ((limited, 429), (unavailable, 502)):
            reply = BytesIO(json.dumps(response).encode())
            with self.subTest(status=status), patch("nanbeige_watcher_api.urlopen", return_value=reply):
                with self.assertRaises(AssessmentError) as caught:
                    client.assess("Assess chat.", {})
            self.assertEqual(caught.exception.status, status)
            self.assertNotIn("private text", str(caught.exception))
            self.assertNotIn("test-key", str(caught.exception))
