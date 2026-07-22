from __future__ import annotations

import threading
from pathlib import Path
from socketserver import ThreadingTCPServer
from types import ModuleType

import requests

from script_loader import load_script_module


def test_fake_openai_server_returns_chat_and_completion_logprobs() -> None:
    server_module: ModuleType = load_script_module(Path("scripts/ci/fake_openai_compat_server.py"))

    server = ThreadingTCPServer(("127.0.0.1", 0), server_module.FakeOpenAICompatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        chat = requests.post(
            f"{base_url}/v1/chat/completions",
            json={"model": "dummy", "messages": [{"role": "user", "content": "hello"}]},
            timeout=5,
        )
        assert chat.status_code == 200
        assert chat.json()["choices"][0]["message"]["content"].startswith("FAKE_OPENAI_OK")

        completion = requests.post(
            f"{base_url}/v1/completions",
            json={"model": "dummy", "prompt": "hello world", "echo": True, "logprobs": 1},
            timeout=5,
        )
        assert completion.status_code == 200
        logprobs = completion.json()["choices"][0]["logprobs"]
        assert logprobs["tokens"] == ["hello", "world"]
        assert logprobs["token_logprobs"] == [-0.01, -0.02]
    finally:
        server.shutdown()
        server.server_close()
