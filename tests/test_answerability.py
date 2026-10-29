import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from answerability import parse_response, decision_metrics
import rag_engine as rag
from ingest import Chunk

SOURCE = {"chunk_id": "a", "doc_name": "a.pdf", "page": 1,
          "text": "Aster stores 12 MWh. Its grant amount is not recorded."}


def payload(text="Aster stores 12 MWh.", quote="Aster stores 12 MWh.", chunk_id="a"):
    return {"decision": "answer", "reason": None, "claims": [
        {"text": text, "supports": [{"chunk_id": chunk_id, "quote": quote}]}]}


def test_answer_citations_are_built_from_metadata():
    result = parse_response(json.dumps(payload()), [SOURCE])
    assert result["decision"] == "answer"
    assert result["answer"].endswith("[Source: a.pdf, page 1]")


@pytest.mark.parametrize("raw", ["not json", "[]", '{}',
    '{"decision":"abstain","claims":[],"reason":"made_up"}',
    '{"decision":"answer","claims":[],"reason":null}',
    '{"decision":"abstain","decision":"answer","claims":[],"reason":null}',
    json.dumps(payload(quote="13 MWh")), json.dumps(payload(chunk_id="unknown")),
    json.dumps(payload(quote=" ")), json.dumps(payload(text="Claim [Source: forged.pdf, page 9]"))])
def test_invalid_outputs_are_errors_never_semantic_refusals(raw):
    result = parse_response(raw, [SOURCE])
    assert result["decision"] == "error" and result["claims"] == []


def test_span_matching_normalizes_only_whitespace():
    assert parse_response(json.dumps(payload(quote="Aster  stores\n12 MWh.")), [SOURCE])["decision"] == "answer"


def test_provenance_does_not_establish_entailment():
    # Deliberate wrong-attribute example: manual semantic review must catch it.
    result = parse_response(json.dumps(payload(text="Aster received 12 million dollars.")), [SOURCE])
    assert result["validation_status"] == "valid"


@pytest.mark.parametrize("reason", ["insufficient_evidence", "ambiguous_reference", "incomplete_comparison"])
def test_refusal_contract(reason):
    result = parse_response(json.dumps({"decision": "abstain", "claims": [], "reason": reason}), [SOURCE])
    assert result["decision"] == "abstain"


def test_empty_evidence_never_calls_provider(monkeypatch):
    provider = Mock(side_effect=AssertionError("must not call"))
    monkeypatch.setattr(rag, "get_client", provider)
    assert rag.answer_from_results("question", [])["reason"] == "no_evidence"
    provider.assert_not_called()


@pytest.mark.parametrize("entry", ["chat", "cross"])
def test_both_entry_points_keep_evidence_on_refusal(monkeypatch, entry):
    chunk = Chunk("a", "a.pdf", 1, SOURCE["text"])
    results = [{"chunk": chunk, "score": 0.03}]
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: results)
    monkeypatch.setattr(rag, "retrieve_multi_document", lambda *a, **k: results)
    monkeypatch.setattr(rag, "get_client", lambda: object())
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(
        {"decision": "abstain", "claims": [], "reason": "insufficient_evidence"})))])
    monkeypatch.setattr(rag, "_safe_chat_completion", lambda *a, **k: response)
    result = rag.answer_question("Grant amount?", object()) if entry == "chat" else rag.cross_document_query("Grant amount?", object())
    assert result["decision"] == "abstain" and result["sources"]
    assert result["citation_validation"]["status"] == "not_applicable"


def test_provider_failure_is_not_a_refusal(monkeypatch):
    monkeypatch.setattr(rag, "get_client", Mock(side_effect=RuntimeError("unavailable")))
    result = rag.answer_from_results("question", [{"chunk": Chunk("a", "a.pdf", 1, "text"), "score": 1}])
    assert result["decision"] == "error" and result["reason"] == "provider_error"


def test_metrics_keep_errors_in_denominators():
    result = decision_metrics([{"answerable": False, "decision": d} for d in ["abstain", "error", "answer"]])
    assert result["refusal_recall"] == 1 / 3 and result["coverage"] == 1 / 3
    assert result["errors"] == 1 and result["false_answers"] == 1
    assert decision_metrics([])["coverage"] is None
