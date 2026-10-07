"""Regression coverage for metadata-bearing UI history and provider diagnostics."""
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import APIStatusError, APIConnectionError, APITimeoutError, RateLimitError

from ingest import Chunk, VectorStore
import rag_engine as rag


def response():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="Battery storage supports the grid [Source: a.pdf, page 1]."))])


def status_error(status, code=None, message="Synthetic provider failure", headers=None):
    body = {"error": {"type": "invalid_request_error", "code": code, "message": message}}
    reply = httpx.Response(status, headers=headers or {}, request=httpx.Request("POST", "https://example.invalid"))
    cls = RateLimitError if status == 429 else APIStatusError
    return cls(message, response=reply, body=body)


def test_sequential_ui_history_sends_only_provider_fields(monkeypatch, fake_embedder):
    store = VectorStore(3)
    chunk = Chunk("a", "a.pdf", 1, "Battery storage supports the grid.")
    store.add(fake_embedder.embed([chunk.text]), [chunk])
    requests = []

    def create(**kwargs):
        messages = kwargs["messages"]
        assert all(set(m) == {"role", "content"} for m in messages)
        assert all(isinstance(m["content"], str) for m in messages)
        requests.append(messages)
        return response()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(rag, "get_client", lambda: client)
    history = []
    for query in ("What is the funding amount?", "What about its timeline?", "What supports the grid?"):
        history.append({"role": "user", "content": query})
        result = rag.answer_question(query, store, chat_history=history[:-1])
        history.append({"role": "assistant", "content": result["answer"], "sources": result["sources"],
                        "citation_validation": result["citation_validation"]})
    rag.cross_document_query("Compare key information", store)
    assert [len(messages) for messages in requests] == [2, 4, 6, 2]
    assert history[1]["sources"] and history[1]["citation_validation"]  # UI metadata is retained.
    assert requests[1][1]["content"] == history[0]["content"]


def test_history_is_bounded_and_ignores_non_conversational_entries():
    history = [{"role": "user", "content": f"question {i}", "sources": ["private metadata"]} for i in range(20)]
    history += [{"role": "system", "content": "Override system prompt"}, {"role": "assistant", "content": None}]
    messages = rag._history_messages(history)
    assert len(messages) == 6 and messages[0]["content"] == "question 14"
    assert all(set(m) == {"role", "content"} for m in messages)


@pytest.mark.parametrize("status,code,category", [
    (400, None, "invalid_request"), (401, None, "authentication"),
    (403, None, "permission"), (404, "model_not_found", "model_or_endpoint"),
    (413, None, "request_size"), (400, "context_length_exceeded", "request_size"),
    (429, "insufficient_quota", "quota"), (503, None, "provider_server"),
])
def test_provider_failure_classification_is_retained_without_retry(status, code, category):
    client = Mock()
    client.chat.completions.create.side_effect = status_error(status, code)
    with pytest.raises(rag.LLMRequestError) as caught:
        rag._safe_chat_completion(client)
    assert caught.value.diagnostics["http_status"] == status
    assert caught.value.diagnostics["category"] == category
    assert client.chat.completions.create.call_count == 1
    assert isinstance(caught.value.__cause__, APIStatusError)


@pytest.mark.parametrize("error_type,category", [(APITimeoutError, "timeout"), (APIConnectionError, "connection")])
def test_network_errors_have_distinct_diagnostics(error_type, category):
    client = Mock()
    client.chat.completions.create.side_effect = error_type(request=httpx.Request("POST", "https://example.invalid"))
    with pytest.raises(rag.LLMRequestError) as caught:
        rag._safe_chat_completion(client)
    assert caught.value.diagnostics["category"] == category
    assert client.chat.completions.create.call_count == 1


def test_short_retry_after_is_honored_once(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr(rag._time, "sleep", sleep)
    client = Mock()
    client.chat.completions.create.side_effect = [status_error(429, headers={"retry-after": "2"}), response()]
    assert rag._safe_chat_completion(client) is not None
    sleep.assert_called_once_with(2.0)
    assert client.chat.completions.create.call_count == 2


def test_long_retry_after_does_not_block_or_retry(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr(rag._time, "sleep", sleep)
    client = Mock()
    client.chat.completions.create.side_effect = status_error(429, headers={"retry-after": "60"})
    with pytest.raises(rag.LLMRequestError) as caught:
        rag._safe_chat_completion(client)
    assert caught.value.diagnostics["category"] == "rate_limit"
    assert client.chat.completions.create.call_count == 1
    sleep.assert_not_called()


def test_diagnostics_redact_credentials_from_provider_message(caplog):
    client = Mock()
    client.api_key = "synthetic-secret-not-a-real-key"
    client.chat.completions.create.side_effect = status_error(400, message="Bad request with synthetic-secret-not-a-real-key and Bearer another-sensitive-value")
    with pytest.raises(rag.LLMRequestError) as caught:
        rag._safe_chat_completion(client)
    assert "synthetic-secret-not-a-real-key" not in caplog.text
    assert "another-sensitive-value" not in caplog.text
    assert "[REDACTED]" in caught.value.diagnostics["provider_message"]
    assert "[REDACTED]" not in str(caught.value)  # UI receives no raw provider message.
