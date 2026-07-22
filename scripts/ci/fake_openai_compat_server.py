from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class FakeOpenAICompatHandler(BaseHTTPRequestHandler):
    server_version = "TurnkeyFakeOpenAI/1.0"

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json({"status": "ok"})
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:  # noqa: N802
        payload = self._read_json()
        if self.path == "/v1/chat/completions":
            self._write_json(_chat_response(payload))
            return
        if self.path == "/v1/completions":
            self._write_json(_completion_response(payload))
            return
        self.send_error(404, "not found")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _write_json(self, value: dict[str, Any], *, status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _chat_response(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages") or []
    user_text = ""
    if messages and isinstance(messages[-1], dict):
        user_text = str(messages[-1].get("content") or "")
    text = f"FAKE_OPENAI_OK: {user_text[:80]}"
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": payload.get("model", "fake-model"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": max(1, len(user_text.split())),
            "completion_tokens": max(1, len(text.split())),
            "total_tokens": max(2, len(user_text.split()) + len(text.split())),
        },
    }


def _completion_response(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = str(payload.get("prompt") or "")
    tokens = prompt.split()
    token_logprobs = [round(-0.01 * (index + 1), 6) for index, _ in enumerate(tokens)]
    return {
        "id": "cmpl-fake",
        "object": "text_completion",
        "model": payload.get("model", "fake-model"),
        "choices": [
            {
                "index": 0,
                "text": prompt if payload.get("echo") else "FAKE_OPENAI_COMPLETION",
                "finish_reason": "stop",
                "logprobs": {
                    "tokens": tokens,
                    "token_logprobs": token_logprobs,
                    "token_ids": list(range(1, len(tokens) + 1)),
                },
            }
        ],
        "usage": {
            "prompt_tokens": len(tokens),
            "completion_tokens": 0 if payload.get("echo") else 1,
            "total_tokens": len(tokens) + (0 if payload.get("echo") else 1),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a fake OpenAI-compatible server for CI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), FakeOpenAICompatHandler)
    host, port = server.server_address
    print(f"http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
