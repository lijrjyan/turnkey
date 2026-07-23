from __future__ import annotations

import pytest

from turnkey.components.backends.openai_compat import OpenAICompatBackend


@pytest.mark.parametrize(
    ("response", "error_detail"),
    [
        pytest.param({}, "choices", id="missing-choices"),
        pytest.param({"choices": []}, "choices", id="empty-choices"),
        pytest.param({"choices": [{}]}, "content", id="missing-content"),
        pytest.param(
            {"choices": [{"message": {"content": ""}}]},
            "content",
            id="empty-content",
        ),
        pytest.param(
            {"choices": [{"message": {"content": "   "}}]},
            "content",
            id="whitespace-content",
        ),
        pytest.param(
            {"choices": [{"message": {"content": 42}}]},
            "content",
            id="non-string-content",
        ),
    ],
)
def test_openai_compat_generate_rejects_malformed_response(
    monkeypatch,
    response: dict,
    error_detail: str,
) -> None:
    backend = OpenAICompatBackend(model_id="test-model", base_url="http://unused")
    monkeypatch.setattr(backend, "_post", lambda _path, _payload: response)

    with pytest.raises(RuntimeError, match=rf"openai_compat protocol error.*{error_detail}"):
        backend.generate(prompt="hello", max_new_tokens=8, temperature=0.0)


def test_openai_compat_generate_parses_chat_content(monkeypatch) -> None:
    backend = OpenAICompatBackend(model_id="test-model", base_url="http://unused")
    monkeypatch.setattr(
        backend,
        "_post",
        lambda _path, _payload: {
            "choices": [{"message": {"content": "response"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
    )

    output = backend.generate(prompt="hello", max_new_tokens=8, temperature=0.0)

    assert output.executed is True
    assert output.response_text == "response"
    assert output.total_tokens == 3
