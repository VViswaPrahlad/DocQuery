from unittest.mock import Mock
import numpy as np

from hybrid_search import BM25Index, _tokenize, reciprocal_rank_fusion
from ingest import Chunk, VectorStore
from rag_engine import retrieve, conversational_query, retrieve_multi_document


def test_scoped_retrieval_cannot_be_crowded_out(monkeypatch):
    from ingest import EmbeddingModel
    vec = np.array([[1, 0, 0]], dtype="float32")
    monkeypatch.setattr(EmbeddingModel, "get", classmethod(lambda cls: Mock(embed=lambda *a, **kw: vec)))
    chunks = [Chunk(str(i), "dominant.pdf", 1, "battery cost") for i in range(50)]
    chunks += [Chunk("target", "selected.pdf", 8, "battery cost target evidence")]
    store = VectorStore(3)
    store.add(np.vstack([vec] * 50 + [np.array([[0.8, 0.6, 0]], dtype="float32")]), chunks)
    for hybrid in (False, True):
        result = retrieve("battery cost", store, top_k=1, doc_filter=["selected.pdf"], use_hybrid=hybrid)
        assert result[0]["chunk"].chunk_id == "target"
    assert retrieve("battery", store, doc_filter=["absent.pdf"]) == []


def test_bm25_tokenization_and_zero_candidates():
    assert _tokenize("MF4, L–BFGS (GPS).") == ["mf4", "l", "bfgs", "gps"]
    chunks = [Chunk("a", "a.pdf", 1, "MF4, format"), Chunk("b", "b.pdf", 1, "unrelated battery")]
    index = BM25Index(chunks)
    assert [i for i, _ in index.search("MF4")] == [0]
    assert index.search("doesnotexist") == []
    assert index.search("MF4", allowed_indices=[1]) == []
    assert BM25Index([]).search("MF4") == []


def test_rrf_duplicate_ids_do_not_gain_credit():
    assert reciprocal_rank_fusion([[1, 1, 2]]) == reciprocal_rank_fusion([[1, 2]])


def test_followup_uses_user_context_only():
    history = [{"role": "user", "content": "Explain battery storage."},
               {"role": "assistant", "content": "Invented model claim"}]
    query = conversational_query("What about its limitations?", history)
    assert "battery storage" in query and "limitations" in query
    assert "Invented" not in query
    assert conversational_query("What is MF4?", history) == "What is MF4?"


def test_multi_document_evidence_includes_both_sources(fake_embedder):
    chunks = [Chunk(str(i), "a.pdf", 1, "battery storage costs") for i in range(30)]
    chunks += [Chunk("b", "b.pdf", 2, "battery storage costs and safety")]
    store = VectorStore(3)
    store.add(fake_embedder.embed([c.text for c in chunks]), chunks)
    results = retrieve_multi_document("battery storage costs", store, top_k=4)
    assert {r["chunk"].doc_name for r in results} == {"a.pdf", "b.pdf"}
    assert len({r["idx"] for r in results}) == len(results) == 4
