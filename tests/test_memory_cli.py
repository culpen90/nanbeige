"""Exercise memory commands and prompt injection together without loading weights."""

from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import nanbeige_cli
from nanbeige_compression import CompressionWatcher
from nanbeige_engine import Engine, Event


class StubEngine(Engine):
    def __init__(self, settings):
        self.settings = settings
        self.last_prompt = ""

    def encode(self, messages, thinking):
        rendered = "\n".join(f"{item['role']}: {item['content']}" for item in messages)
        return rendered.split()

    def generate_ids(self, ids, **options):
        self.last_prompt = " ".join(ids)
        if "INTERRUPT" in ids:
            raise KeyboardInterrupt
        yield Event("private scratch text</think>\n\nPublic answer.", finish_reason="stop")


class MemoryCLITests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.home = Path(self.directory.name)
        self.default_path = self.home / "nanbeige-memory.sqlite3"
        self.path = str(self.home / "notes.sqlite3")
        self.engines = []

    def invoke(self, *argv, stdin=""):
        def load(settings):
            engine = StubEngine(settings)
            self.engines.append(engine)
            return engine

        output, error = StringIO(), StringIO()
        with (patch("sys.argv", ["nanbeige", *argv]), patch("sys.stdin", StringIO(stdin)),
              patch.object(Path, "home", return_value=self.home),
              patch.object(nanbeige_cli, "load_engine", side_effect=load),
              redirect_stdout(output), redirect_stderr(error)):
            result = nanbeige_cli.main()
        self.assertEqual(result, 0, error.getvalue())
        return output.getvalue()

    def test_plain_chat_recalls_notes_and_completed_turns_across_invocations(self):
        self.invoke("chat", "--impl", "stock",
                    stdin="/remember My workshop project is called Cedar.\nMy vise is cobalt blue.\n/quit\n")
        self.assertTrue(self.default_path.is_file())

        self.invoke("chat", "--impl", "stock",
                    stdin="What is my workshop project called, and what color is my vise?\n/quit\n")
        self.assertIn("Cedar", self.engines[-1].last_prompt)
        self.assertIn("cobalt blue", self.engines[-1].last_prompt)

    def test_session_only_leaves_no_file_and_does_not_recall_other_sessions(self):
        self.invoke("chat", "--impl", "stock", "--session-only",
                    stdin="/remember My call sign is Maple.\nWhere did you hear the call sign?\n/quit\n")
        self.assertIn("Maple", self.engines[-1].last_prompt)
        self.assertFalse(self.default_path.exists())

        self.invoke("memory", str(self.default_path), "remember", "My call sign is Cedar.")
        self.invoke("chat", "--impl", "stock", "--session-only", stdin="What is my call sign?\n/quit\n")
        self.assertNotIn("Maple", self.engines[-1].last_prompt)
        self.assertNotIn("Cedar", self.engines[-1].last_prompt)
        self.assertIn("No saved turns.", self.invoke("memory", str(self.default_path), "history"))

    def test_session_only_and_explicit_database_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit) as error:
            self.invoke("chat", "--session-only", "--memory", self.path)
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(self.engines, [])
        self.assertFalse(Path(self.path).exists())

    def test_durable_memory_crosses_invocations_and_does_not_leak_reasoning(self):
        self.invoke("chat", "--impl", "stock", "--think", "--memory", self.path,
                    stdin="/remember Workshop project: Cedar\nI own a blue vise.\nINTERRUPT\n/quit\n")

        self.invoke("run", "--impl", "stock", "--memory", self.path, "What is my workshop project?")
        self.assertIn("Workshop project: Cedar", self.engines[-1].last_prompt)

        history = self.invoke("memory", self.path, "history")
        self.assertIn("I own a blue vise.", history)
        self.assertIn("Public answer.", history)
        self.assertNotIn("private scratch text", history)
        self.assertNotIn("INTERRUPT", history)
        self.assertEqual(len(self.engines), 2)

        self.invoke("run", "--impl", "stock", "What is my workshop project?")
        self.assertNotIn("Cedar", self.engines[-1].last_prompt)

    def test_reset_keeps_durable_memory_and_clear_removes_it(self):
        self.invoke("chat", "--impl", "stock", "--memory", self.path,
                    stdin="/remember My call sign is Maple.\n/reset\nWhat is my call sign?\n/quit\n")
        self.assertIn("Maple", self.engines[-1].last_prompt)

        self.invoke("chat", "--impl", "stock", "--memory", self.path,
                    stdin="I am in Oslo.\n/clear-memory\nWhat is my call sign?\n/quit\n")
        self.assertNotIn("Maple", self.engines[-1].last_prompt)
        self.assertNotIn("Oslo", self.engines[-1].last_prompt)
        self.assertIn("No saved notes.", self.invoke("memory", self.path, "list"))
        self.assertNotIn("Oslo", self.invoke("memory", self.path, "history"))

    def test_session_reset_wipes_memory_without_a_database_file(self):
        self.invoke("chat", "--impl", "stock", "--session-only",
                    stdin="/remember My call sign is Maple.\n/reset\nWhat is my call sign?\n/quit\n")
        self.assertNotIn("Maple", self.engines[-1].last_prompt)

    def test_gemini_watcher_applies_a_summary_in_chat(self):
        self.check_chat_compression(openrouter=False)

    def test_openrouter_watcher_applies_a_summary_in_chat(self):
        self.check_chat_compression(openrouter=True)

    def check_chat_compression(self, *, openrouter):
        prompts = iter(["Cedar " + "details " * 70, "Recent exchange", "Current request", "/quit"])
        request_bodies = []
        watchers = []

        def start_watcher(*args, **kwargs):
            watcher = CompressionWatcher(*args, **kwargs)
            watchers.append(watcher)
            return watcher

        def read_prompt(_):
            prompt = next(prompts)
            if prompt in ("Recent exchange", "Current request", "/quit"):
                watcher = watchers[0]
                with watcher._condition:
                    self.assertTrue(watcher._condition.wait_for(lambda: watcher._ready is not None, timeout=3))
            return prompt

        def request(url_request, *, timeout):
            payload = json.loads(url_request.data)
            if openrouter:
                self.assertEqual(url_request.full_url, "https://openrouter.ai/api/v1/chat/completions")
                self.assertEqual(url_request.get_header("Authorization"), "Bearer router-test-key")
                self.assertIsNone(url_request.get_header("X-goog-api-key"))
                self.assertEqual(payload["model"], "openai/gpt-oss-20b")
                observed = json.loads(payload["messages"][1]["content"])
            else:
                self.assertEqual(url_request.get_header("X-goog-api-key"), "studio-test-key")
                self.assertIsNone(url_request.get_header("Authorization"))
                self.assertTrue(url_request.full_url.endswith("/gemini-4-hydrogen:generateContent"))
                observed = json.loads(payload["contents"][0]["parts"][0]["text"])
            request_bodies.append(observed)
            compress = bool(observed["older_messages"])
            decision = json.dumps({"compress": compress, "summary": "Cedar is active." if compress else ""})
            if openrouter:
                result = {"choices": [{"message": {"content": decision}, "finish_reason": "stop"}]}
            else:
                result = {"candidates": [{"content": {"parts": [{"text": decision}]}, "finishReason": "STOP"}]}
            return BytesIO(json.dumps(result).encode())

        provider_args = ("--compression-provider", "openrouter", "--openrouter-model", "openai/gpt-oss-20b")
        with (patch.dict(os.environ, {"GEMINI_API_KEY": "studio-test-key", "OPENROUTER_API_KEY": "router-test-key"}),
              patch("nanbeige_compression.CompressionWatcher", side_effect=start_watcher),
              patch("nanbeige_watcher_api.urlopen", side_effect=request),
              patch("builtins.input", side_effect=read_prompt)):
            self.invoke("chat", "--impl", "stock", "--session-only", "--compress-context", "--think",
                        "--system", "Private system prompt", *(provider_args if openrouter else ()))

        self.assertIn("Cedar is active.", self.engines[-1].last_prompt)
        self.assertIn("Recent exchange", self.engines[-1].last_prompt)
        self.assertIn("Current request", self.engines[-1].last_prompt)
        self.assertNotIn("details details", self.engines[-1].last_prompt)
        self.assertEqual(request_bodies[0]["recent_exchange"][1]["content"], "Public answer.")
        self.assertNotIn("Private system prompt", str(request_bodies))
