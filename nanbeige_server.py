"""A local, text-only subset of the OpenAI Chat Completions API."""

import json
import socket
import time
import uuid
from contextlib import closing
from http.server import BaseHTTPRequestHandler, HTTPServer
from ipaddress import ip_address
from itertools import chain


_MAX_BODY = 1024 * 1024
_CHAT_PATH = "/v1/chat/completions"
_FIELDS = {
    "model", "messages", "max_tokens", "max_completion_tokens", "temperature",
    "top_p", "thinking", "stream", "stream_options", "n",
}


def _json(payload):
    return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _error(message, kind="invalid_request_error"):
    return {"error": {"message": message, "type": kind}}


def _parse_request(body, model_id):
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object.")
    unsupported = body.keys() - _FIELDS
    if unsupported:
        raise ValueError(f"Unsupported parameter: {min(unsupported)}.")
    if body.get("model", model_id) != model_id:
        raise ValueError(f"The loaded model is {model_id}.")
    if type(body.get("n", 1)) is not int or body.get("n", 1) != 1:
        raise ValueError("Only n=1 is supported.")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list.")
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object.")
        role = message.get("role")
        if role == "developer":
            role = "system"
        if role not in ("system", "user", "assistant") or (role == "system" and normalized):
            raise ValueError("Use an optional initial system message followed by user or assistant messages.")
        content = message.get("content")
        if isinstance(content, list) and all(
            isinstance(part, dict) and part.get("type") == "text"
            and isinstance(part.get("text"), str) for part in content
        ):
            content = "".join(part["text"] for part in content)
        if not isinstance(content, str):
            raise ValueError("Message content must be text.")
        item = {"role": role, "content": content}
        if role == "assistant" and isinstance(message.get("reasoning_content"), str):
            item["reasoning_content"] = message["reasoning_content"]
        normalized.append(item)

    max_tokens = body.get("max_completion_tokens", body.get("max_tokens", 256))
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_completion_tokens must be a positive integer.")
    temperature = body.get("temperature", 0.0)
    top_p = body.get("top_p", 1.0)
    if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2.")
    if type(top_p) not in (int, float) or not 0 < top_p <= 1:
        raise ValueError("top_p must be greater than 0 and at most 1.")
    stream = body.get("stream", False)
    thinking = body.get("thinking", False)
    if not isinstance(stream, bool) or not isinstance(thinking, bool):
        raise ValueError("stream and thinking must be booleans.")
    stream_options = body.get("stream_options")
    if stream_options is None:
        stream_options = {}
    if (not isinstance(stream_options, dict)
            or stream_options.keys() - {"include_usage"}
            or not isinstance(stream_options.get("include_usage", False), bool)):
        raise ValueError("Only the boolean stream_options.include_usage is supported.")
    return normalized, stream, stream_options.get("include_usage", False), {
        "max_tokens": max_tokens, "temperature": temperature, "thinking": thinking, "top_p": top_p,
    }


def _usage(stats):
    return {
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.generated_tokens,
        "total_tokens": stats.prompt_tokens + stats.generated_tokens,
        "prompt_tokens_details": {"cached_tokens": stats.cached_tokens},
    }


def _metrics(stats):
    return {
        "ttft_s": stats.ttft_s,
        "decode_tps": stats.decode_tps,
        "prefill_tps": stats.prefill_tps,
    }


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Nanbeige"
    sys_version = ""
    timeout = 15
    disable_nagle_algorithm = True

    def _headers(self, status, media_type, length=None):
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.end_headers()

    def _send_json(self, status, payload):
        data = _json(payload)
        self._headers(status, "application/json", len(data))
        self.wfile.write(data)

    def _valid_host(self):
        host = self.headers.get("Host")
        if host is None or host.lower() in self.server.allowed_hosts:
            return True
        self._send_json(403, _error("Use a loopback Host header."))
        return False

    def do_GET(self):
        if not self._valid_host():
            return
        engine = self.server.engine
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model": engine.model_id})
        elif self.path == "/v1/models":
            model = {"id": engine.model_id, "object": "model", "created": 0, "owned_by": "local"}
            self._send_json(200, {"object": "list", "data": [model]})
        else:
            self._send_json(404, _error("Unknown endpoint."))

    def do_POST(self):
        try:
            self._post()
        except (ConnectionError, TimeoutError):
            self.close_connection = True

    def _post(self):
        if not self._valid_host():
            return
        if self.path != _CHAT_PATH:
            self._send_json(404, _error("Unknown endpoint."))
            return
        if self.headers.get_content_type() != "application/json":
            self._send_json(415, _error("Expected application/json."))
            return
        lengths = self.headers.get_all("Content-Length", [])
        if (len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdecimal()
                or self.headers.get("Transfer-Encoding") is not None):
            self._send_json(400, _error("Expected one valid Content-Length header."))
            return
        digits = lengths[0].lstrip("0") or "0"
        if len(digits) > len(str(_MAX_BODY)) or int(digits) > _MAX_BODY:
            self._send_json(413, _error("Request exceeds 1 MiB."))
            return
        try:
            length = int(digits)
            data = self.rfile.read(length)
            if len(data) != length:
                raise ValueError("Incomplete request body")
            parsed = json.loads(data)
        except (ValueError, RecursionError):
            self._send_json(400, _error("Could not parse the request JSON."))
            return

        engine = self.server.engine
        streaming_headers = False
        try:
            messages, stream, include_usage, options = _parse_request(parsed, engine.model_id)
            response = {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()), "model": engine.model_id}
            with closing(engine.generate(messages, **options)) as events:
                # Advance before committing to HTTP 200 so context errors can return JSON.
                first = next(events)
                if stream:
                    streaming_headers = True
                    self._stream(response, chain((first,), events), include_usage)
                else:
                    self._complete(response, chain((first,), events))
        except (ConnectionError, TimeoutError):
            raise
        except Exception as exc:
            if isinstance(exc, ValueError) and not streaming_headers:
                self._send_json(400, _error(str(exc)))
                return
            self.log_error("Generation failed (%s)", type(exc).__name__)
            error = _error("Local generation failed.", "server_error")
            if streaming_headers:
                self._sse(error)
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self._send_json(500, error)

    def _complete(self, response, events):
        text = []
        for event in events:
            text.append(event.text)
        response.update(object="chat.completion", choices=[{
            "index": 0, "message": {"role": "assistant", "content": "".join(text)},
            "finish_reason": event.finish_reason, "logprobs": None,
        }])
        if event.stats is not None:
            response.update(usage=_usage(event.stats), nanbeige=_metrics(event.stats))
        self._send_json(200, response)

    def _sse(self, data):
        self.wfile.write(b"data: " + _json(data) + b"\n\n")

    def _stream(self, response, events, include_usage):
        response["object"] = "chat.completion.chunk"
        if include_usage:
            response["usage"] = None

        def chunk(delta, finish_reason=None):
            self._sse({**response, "choices": [{
                "index": 0, "delta": delta, "finish_reason": finish_reason, "logprobs": None,
            }]})

        self._headers(200, "text/event-stream")
        chunk({"role": "assistant", "content": ""})
        for event in events:
            if event.text:
                chunk({"content": event.text})
        chunk({}, event.finish_reason)
        if include_usage and event.stats is not None:
            self._sse({**response, "choices": [], "usage": _usage(event.stats), "nanbeige": _metrics(event.stats)})
        self.wfile.write(b"data: [DONE]\n\n")

    def log_request(self, code="-", size="-"):
        pass


class _Server(HTTPServer):
    def __init__(self, host, port, engine_factory):
        address = ip_address("127.0.0.1" if host.lower() == "localhost" else host)
        if not address.is_loopback or getattr(address, "scope_id", None):
            raise ValueError("The server binds only to a loopback address.")
        self.address_family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
        super().__init__((str(address), port), _Handler)
        self.url_host = f"[{address}]" if address.version == 6 else str(address)
        self.allowed_hosts = {"localhost", self.url_host, f"localhost:{self.server_port}",
                              f"{self.url_host}:{self.server_port}"}
        try:
            self.engine = engine_factory()
        except BaseException:
            self.server_close()
            raise


def serve(engine_factory, host="127.0.0.1", port=8080):
    with _Server(host, port, engine_factory) as server:
        print(f"{server.engine.model_id} ready at http://{server.url_host}:{server.server_port}/v1", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
