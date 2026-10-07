from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np

from citations import validate_citations, normalize_citations
from ingest import Chunk, VectorStore
import rag_engine


def test_citations_map_to_chunk_ids():
    sources = [{"doc_name": "report.pdf", "page": 2, "chunk_id": "a"},
               {"doc_name": "report.pdf", "page": 2, "chunk_id": "b"}]
    result = validate_citations("Evidence [Source: report.pdf, page 2].", sources)
    assert result["status"] == "matched"
    assert result["valid"][0]["chunk_ids"] == ["a", "b"]


def test_unknown_citation_and_missing_citation_are_explicit():
    sources = [{"doc_name": "report.pdf", "page": 2, "chunk_id": "a"}]
    result = validate_citations("Claim [Source: other.pdf, page 99].", sources)
    assert result["status"] == "invalid" and result["reference_precision"] == 0
    assert validate_citations("Uncited answer", sources)["status"] == "missing"
    assert validate_citations("No evidence", [])["status"] == "no_evidence"


def test_unicode_citations_are_canonicalized_without_inventing_evidence():
    answer = "Funding【Source: report.pdf, page\u202f2】."
    assert normalize_citations(answer) == "Funding[Source: report.pdf, page 2]."
    assert validate_citations(answer, [{"doc_name": "report.pdf", "page": 2}])["status"] == "matched"
    assert validate_citations(answer, [{"doc_name": "other.pdf", "page": 2}])["status"] == "invalid"


def test_generation_preserves_exact_source_metadata(monkeypatch, fake_embedder):
    chunk = Chunk("report_p2_c1", "report.pdf", 2, "Battery storage supports the grid.", "ocr")
    store = VectorStore(3)
    store.add(fake_embedder.embed([chunk.text]), [chunk])
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Supports the grid [Source: report.pdf, page 2]."))])
    monkeypatch.setattr(rag_engine, "get_client", lambda: object())
    monkeypatch.setattr(rag_engine, "_safe_chat_completion", lambda *a, **kw: response)
    result = rag_engine.answer_question("What supports the grid?", store)
    assert result["sources"][0]["chunk_id"] == chunk.chunk_id
    assert result["sources"][0]["source"] == "ocr"
    assert result["citation_validation"]["status"] == "matched"
    assert result["retrieval_seconds"] >= 0


def test_summary_covers_tail_of_long_document(monkeypatch):
    store = VectorStore(3)
    chunks = [Chunk(str(i), "long.pdf", i + 1, f"SECTION_{i} content") for i in range(45)]
    store.add(np.ones((45, 3), dtype="float32"), chunks)
    calls = []
    monkeypatch.setattr(rag_engine, "get_client", lambda: object())

    def complete(*args, **kwargs):
        calls.append(kwargs["messages"][-1]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Section summary [Source: long.pdf, page 45]."))])

    monkeypatch.setattr(rag_engine, "_safe_chat_completion", complete)
    summary = rag_engine.summarize_document("long.pdf", store)
    assert summary
    assert any("SECTION_44" in text for text in calls)
    assert len(calls) >= 3  # At least two map batches and one reduction.
