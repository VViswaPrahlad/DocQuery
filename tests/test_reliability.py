import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
from openai import APIConnectionError, RateLimitError

import ingest
import ocr_utils
import rag_engine
from evaluation import evaluate_answer_quality
from ingest import Chunk, VectorStore, build_index_from_pdfs, add_pdfs_to_existing_index
from session_utils import stage_uploads


def test_empty_store_does_not_load_model(monkeypatch):
    monkeypatch.setattr(ingest.EmbeddingModel, "get", Mock(side_effect=AssertionError("model loaded")))
    store = VectorStore(3)
    assert store.search(np.ones((1, 3), dtype="float32")) == []
    assert store.hybrid_search(np.ones((1, 3), dtype="float32"), "battery") == []
    assert rag_engine.retrieve("battery", store) == []
    assert rag_engine.answer_question("battery", store)["sources"] == []


def test_missing_tesseract_preserves_native_text(monkeypatch):
    page = SimpleNamespace(number=0, get_text=lambda _: "7")
    monkeypatch.setattr(ocr_utils, "ocr_page", Mock(side_effect=ocr_utils.pytesseract.TesseractNotFoundError()))
    with pytest.warns(RuntimeWarning, match="Image-only"):
        assert ocr_utils.extract_page_text(page) == ("7", "native")


def test_ocr_fallback_retains_source(monkeypatch):
    page = SimpleNamespace(number=0, get_text=lambda _: "")
    monkeypatch.setattr(ocr_utils, "ocr_page", lambda _: "Readable scanned content")
    assert ocr_utils.extract_page_text(page) == ("Readable scanned content", "ocr")


def test_empty_pdf_fails_before_model_loading(pdf_factory, monkeypatch):
    monkeypatch.setattr(ingest, "OCR_FALLBACK_ENABLED", False)
    monkeypatch.setattr(ingest, "extract_pages", lambda _: [])
    load = Mock(side_effect=AssertionError("model loaded"))
    monkeypatch.setattr(ingest.EmbeddingModel, "get", load)
    with pytest.raises(ValueError, match="No extractable text"):
        build_index_from_pdfs([pdf_factory(text="")], persist=False)
    load.assert_not_called()


def test_pdf_index_duplicate_and_replacement(pdf_factory, fake_embedder):
    path = pdf_factory()
    store = build_index_from_pdfs([path, path], persist=False)
    original_count = len(store.chunks)
    add_pdfs_to_existing_index([path], store, persist=False)
    assert len(store.chunks) == original_count
    replacement = pdf_factory(text="Telemetry records GPS signals in the MF4 format.")
    add_pdfs_to_existing_index([replacement], store, persist=False)
    assert len(store.chunks) == original_count
    assert all("Telemetry" in c.text for c in store.chunks)
    assert len({c.chunk_id for c in store.chunks}) == store.index.ntotal


def test_failed_replacement_leaves_original_index(pdf_factory, fake_embedder, monkeypatch):
    path = pdf_factory()
    store = build_index_from_pdfs([path], persist=False)
    old_chunks, old_hashes = list(store.chunks), dict(store.document_hashes)
    pdf_factory(text="Different replacement content for battery storage.")
    monkeypatch.setattr(fake_embedder, "embed", Mock(side_effect=RuntimeError("embedding failed")))
    with pytest.raises(RuntimeError):
        add_pdfs_to_existing_index([path], store, persist=False)
    assert store.chunks == old_chunks and store.document_hashes == old_hashes
    assert store.index.ntotal == len(old_chunks)


def test_remove_document_keeps_other_vectors(pdf_factory, fake_embedder):
    store = build_index_from_pdfs([pdf_factory("a.pdf"), pdf_factory("b.pdf")], persist=False)
    store.hybrid_search(fake_embedder.embed(["battery"]), "battery")
    store.remove_document("a.pdf")
    assert store.list_documents() == ["b.pdf"]
    assert "a.pdf" not in store.document_hashes
    assert store._bm25 is None
    store.remove_document("b.pdf")
    assert store.hybrid_search(fake_embedder.embed(["battery"]), "battery") == []


def test_persistence_round_trip_and_corruption(tmp_path, pdf_factory, fake_embedder):
    store = build_index_from_pdfs([pdf_factory()], persist=False)
    path = tmp_path / "index"
    store.save(path)
    loaded = VectorStore.load(path)
    assert loaded.chunks == store.chunks
    assert loaded.document_hashes == store.document_hashes
    assert loaded.index.ntotal == store.index.ntotal
    (path / "chunks.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        VectorStore.load(path)


def test_saved_model_mismatch(tmp_path, pdf_factory, fake_embedder):
    store = build_index_from_pdfs([pdf_factory()], persist=False)
    path = tmp_path / "index"
    store.save(path)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["embedding_model"] = "different-model"
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="model"):
        VectorStore.load(path)


def test_missing_index_is_friendly(tmp_path):
    with pytest.raises(FileNotFoundError, match="No saved FAISS index"):
        VectorStore.load(tmp_path)


def test_invalid_vectors_do_not_mutate_store():
    store = VectorStore(3)
    chunk = Chunk("a", "a.pdf", 1, "text")
    with pytest.raises(ValueError):
        store.add(np.array([[float("nan"), 0, 1]], dtype="float32"), [chunk])
    assert store.index.ntotal == 0 and store.chunks == []


def test_provider_errors_are_safe(monkeypatch):
    client = Mock()
    client.chat.completions.create.side_effect = APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
    with pytest.raises(RuntimeError, match="request failed"):
        rag_engine._safe_chat_completion(client)


def test_rate_limit_retries_are_bounded(monkeypatch):
    monkeypatch.setattr(rag_engine._time, "sleep", lambda _: None)
    client = Mock()
    client.chat.completions.create.side_effect = RateLimitError("rate limit", response=httpx.Response(429, request=httpx.Request("POST", "https://example.invalid")), body=None)
    with pytest.raises(RuntimeError, match="rate limit"):
        rag_engine._safe_chat_completion(client)
    assert client.chat.completions.create.call_count == 2


def test_missing_provider_key_is_friendly():
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        rag_engine.get_client()


def test_optional_judge_failure_is_isolated(monkeypatch):
    import evaluation
    monkeypatch.setattr(evaluation, "get_client", Mock(side_effect=RuntimeError("unavailable")))
    result = evaluate_answer_quality("question", [], "answer")
    assert result["groundedness"] is None and result["judge_status"] == "RuntimeError"


@pytest.mark.parametrize("output", ['[]', '{"groundedness": 8, "relevance": 4}', 'not json'])
def test_judge_schema_is_validated(monkeypatch, output):
    import evaluation
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=output))])
    monkeypatch.setattr(evaluation, "get_client", lambda: object())
    monkeypatch.setattr(evaluation, "_safe_chat_completion", lambda *a, **kw: response)
    assert evaluate_answer_quality("question", [], "answer")["groundedness"] is None


def test_upload_staging_sanitizes_and_cleans_files():
    uploaded = SimpleNamespace(name="../../safe.pdf", getbuffer=lambda: b"test")
    with stage_uploads([uploaded]) as paths:
        from pathlib import Path
        path = Path(paths[0])
        assert path.name == "safe.pdf" and path.read_bytes() == b"test"
    assert not path.exists()


def test_upload_cleanup_on_failure():
    uploaded = SimpleNamespace(name="safe.pdf", getbuffer=lambda: b"test")
    from pathlib import Path
    with pytest.raises(RuntimeError):
        with stage_uploads([uploaded]) as paths:
            path = Path(paths[0])
            raise RuntimeError("ingestion failed")
    assert not path.exists()
