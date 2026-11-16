import json
import threading
import unittest
from dataclasses import dataclass
from http.client import HTTPConnection
from queue import Queue

from nanbeige_server import _Server


@dataclass
class Stats:
    prompt_tokens: int = 40
    generated_tokens: int = 2
    cached_tokens: int = 20
    ttft_s: float = 0.1
    decode_tps: float = 31.2
    prefill_tps: float = 182.3


@dataclass
class Event:
    text: str
    finish_reason: str | None = None
    stats: Stats | None = None


class Engine:
    model_id = "test/nanbeige"

    def __init__(self):
        self.owner = threading.get_ident()
        self.calls = []

    def generate(self, messages, **options):
        assert threading.get_ident() == self.owner
        self.calls.append((messages, options))
        if messages[0]["content"] == "too long":
            raise ValueError("The prompt exceeds the model context limit.")
        yield Event("Mac")
        yield Event(" 日本", "stop", Stats())


class ServerTests(unittest.TestCase):
    def setUp(self):
        started = Queue()

        def run():
            with _Server("127.0.0.1", 0, Engine) as server:
                started.put(server)
                server.serve_forever(poll_interval=0.02)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.server = started.get(timeout=5)
        self.addCleanup(thread.join, 5)
        self.addCleanup(self.server.shutdown)

    def request(self, method, path, *, body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        self.addCleanup(connection.close)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type"), response.read()

    def complete(self, **params):
        return self.request("POST", "/v1/chat/completions", body=json.dumps({
            "model": Engine.model_id, "messages": [{"role": "user", "content": "Hello"}], **params,
        }), headers={"Content-Type": "application/json"})

    def test_json_and_sse_share_one_engine_and_report_matching_usage(self):
        status, media_type, body = self.complete(max_tokens=12, thinking=True, top_p=0.8)
        self.assertEqual((status, media_type), (200, "application/json"))
        completion = json.loads(body)
        choice = completion["choices"][0]
        self.assertEqual(choice["message"]["content"], "Mac 日本")

        status, media_type, body = self.complete(stream=True, stream_options={"include_usage": True})
        self.assertEqual((status, media_type), (200, "text/event-stream"))
        frames = body.splitlines()[::2]
        self.assertEqual(frames[-1], b"data: [DONE]")
        chunks = [json.loads(frame.removeprefix(b"data: ")) for frame in frames[:-1]]
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual("".join(chunk["choices"][0]["delta"].get("content", "")
                                 for chunk in chunks[:-1]), choice["message"]["content"])
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], choice["finish_reason"])
        self.assertEqual(chunks[-1]["usage"], completion["usage"])
        self.assertEqual(completion["usage"]["prompt_tokens_details"]["cached_tokens"], 20)
        self.assertEqual(self.server.engine.calls[0][1], {
            "max_tokens": 12, "temperature": 0.0, "thinking": True, "top_p": 0.8,
        })
        self.assertEqual(len(self.server.engine.calls), 2)

    def test_invalid_context_remains_a_json_error_for_streaming(self):
        status, media_type, body = self.complete(stream=True, messages=[{
            "role": "user", "content": "too long",
        }])
        self.assertEqual((status, media_type), (400, "application/json"))
        self.assertIn("context limit", json.loads(body)["error"]["message"])

    def test_loopback_host_and_body_limits(self):
        status, _, body = self.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["data"][0]["id"], Engine.model_id)
        status, _, _ = self.request("GET", "/health", headers={"Host": "external.example"})
        self.assertEqual(status, 403)
        status, _, _ = self.request("POST", "/v1/chat/completions", headers={
            "Content-Type": "application/json", "Content-Length": str(2 * 1024 * 1024),
        })
        self.assertEqual(status, 413)
        self.assertEqual(self.server.engine.calls, [])
