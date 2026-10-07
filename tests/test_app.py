from pathlib import Path
from unittest.mock import Mock
import pytest
import streamlit as st

from streamlit.testing.v1 import AppTest
from ingest import Chunk, VectorStore
import rag_engine

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def app_with_store(fake_embedder):
    app = AppTest.from_file(APP, default_timeout=30).run()
    store = VectorStore(3)
    chunks = [Chunk("a", "a.pdf", 1, "Battery storage supports the grid.")]
    store.add(fake_embedder.embed([c.text for c in chunks]), chunks)
    app.session_state["store"] = store
    app.session_state["doc_names"] = ["a.pdf"]
    app.run()
    assert not app.exception
    return app


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_initial_render_needs_no_model_or_provider():
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert app.session_state["store"] is None


def test_cross_document_call_requires_click_and_persists(monkeypatch, fake_embedder):
    result = {"answer": "Stored comparison", "documents_used": ["a.pdf"], "sources": []}
    mock = Mock(return_value=result)
    monkeypatch.setattr(rag_engine, "cross_document_query", mock)
    app = app_with_store(fake_embedder)
    app.text_input(key="cross_box").set_value("Compare safety").run()
    assert mock.call_count == 0
    button(app, "Compare documents").click().run()
    assert mock.call_count == 1 and not app.exception
    app.run()
    assert mock.call_count == 1
    assert app.session_state["cross_result"]["answer"] == "Stored comparison"


def test_summary_persists_across_rerun(monkeypatch, fake_embedder):
    mock = Mock(return_value="Stored summary")
    monkeypatch.setattr(rag_engine, "summarize_document", mock)
    app = app_with_store(fake_embedder)
    button(app, "Generate summary").click().run()
    app.run()
    assert not app.exception and mock.call_count == 1
    assert app.session_state["summary_result"][1] == "Stored summary"


def test_removing_last_document_returns_to_upload_screen(fake_embedder):
    app = app_with_store(fake_embedder)
    app.checkbox(key="select_a.pdf").check().run()
    remove = next(b for b in app.button if "Remove selected" in b.label)
    remove.click().run()
    assert not app.exception and app.session_state["store"] is None
    assert app.session_state["doc_names"] == []


def test_synthetic_demo_requires_click_and_builds_library(fake_embedder):
    app = AppTest.from_file(APP, default_timeout=30).run()
    button(app, "Try Demo").click().run()
    assert not app.exception
    assert app.session_state["doc_names"] == ["battery_brief.pdf", "fleet_report.pdf"]
    assert app.session_state["store"].index.ntotal == 4


def test_chat_metrics_use_answer_sources_without_extra_retrieval(monkeypatch, fake_embedder):
    import evaluation
    sources = [{"doc_name": "a.pdf", "page": 1, "chunk_id": "a", "source": "native",
                "text": "Battery storage supports the grid.", "score": 0.025, "score_type": "rrf"}]
    result = {"answer": "Supports the grid [Source: a.pdf, page 1].", "sources": sources,
              "retrieval_seconds": 0.125, "retrieval_query": "battery",
              "citation_validation": {"status": "matched"}}
    monkeypatch.setattr(rag_engine, "answer_question", Mock(return_value=result))
    judge = Mock(side_effect=AssertionError("Judge should be opt-in"))
    monkeypatch.setattr(evaluation, "evaluate_answer_quality", judge)
    app = app_with_store(fake_embedder)
    app.chat_input[0].set_value("battery").run()
    assert not app.exception
    assert app.session_state["chat_history"][-1]["sources"] == sources
    log = app.session_state["eval_logs"][0]
    assert log["docs_hit"] == ["a.pdf"] and log["latency_sec"] == 0.125
    assert log["judge_status"] == "not_requested"
    judge.assert_not_called()


def watch_processing(monkeypatch):
    events = []
    native_status = st.status

    def watch(label, **kwargs):
        events.append(("start", label))
        status = native_status(label, **kwargs)
        native_update = status.update

        def update(**values):
            if "state" in values:
                events.append((values["state"], values.get("label")))
            return native_update(**values)

        status.update = update
        return status

    monkeypatch.setattr(st, "status", watch)
    return events


def test_landing_and_navigation_positioning(fake_embedder):
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert any("Multi-document RAG with evaluated retrieval" in item.value for item in app.markdown)
    assert any(item.value == "Upload PDFs" for item in app.subheader)
    assert button(app, "Try Demo").proto.type == "primary"
    assert button(app, "Clear documents").proto.type == "secondary"
    app = app_with_store(fake_embedder)
    assert [tab.label for tab in app.tabs] == ['Chat / Q&A', 'Evidence Search', 'Summarize', 'Cross-Document', 'Evaluation']


def test_chat_processing_precedes_call_and_clears_without_duplicate(monkeypatch, fake_embedder):
    events = watch_processing(monkeypatch)

    def answer(*args, **kwargs):
        assert events[-1] == ("start", "Searching your documents and generating an answer…")
        return {"answer": "A response", "sources": [], "citation_validation": {"status": "no_evidence"}}

    generate = Mock(side_effect=answer)
    monkeypatch.setattr(rag_engine, "answer_question", generate)
    app = app_with_store(fake_embedder)
    app.chat_input[0].set_value("question").run()
    assert not app.exception and len(app.status) == 0
    assert events[-1][0] == "complete"
    assert any(item.value.startswith("Answered in ") for item in app.caption)
    app.run()
    assert generate.call_count == 1 and len(events) == 2


def test_chat_failure_ends_loading_and_preserves_clean_error(monkeypatch, fake_embedder):
    events = watch_processing(monkeypatch)
    generate = Mock(side_effect=RuntimeError("Provider unavailable; please retry."))
    monkeypatch.setattr(rag_engine, "answer_question", generate)
    app = app_with_store(fake_embedder)
    app.chat_input[0].set_value("question").run()
    assert not app.exception and len(app.status) == 0
    assert events[-1][0] == "error"
    assert any("Provider unavailable" in error.value for error in app.error)
    app.run()
    assert generate.call_count == 1


@pytest.mark.parametrize("operation,fail", [("summary", False), ("summary", True), ("cross", False), ("cross", True)])
def test_generation_operation_loading_is_bounded(monkeypatch, fake_embedder, operation, fail):
    events = watch_processing(monkeypatch)
    label = "Generating document summary…" if operation == "summary" else "Comparing evidence across documents…"

    def generate(*args, **kwargs):
        assert events[-1] == ("start", label)
        if fail:
            raise RuntimeError("Controlled provider failure")
        return "A summary" if operation == "summary" else {"answer": "A comparison", "sources": [], "documents_used": ["a.pdf"]}

    mock = Mock(side_effect=generate)
    monkeypatch.setattr(rag_engine, "summarize_document" if operation == "summary" else "cross_document_query", mock)
    app = app_with_store(fake_embedder)
    if operation == "cross":
        app.text_input(key="cross_box").set_value("Compare documents").run()
        assert mock.call_count == 0
    button(app, "Generate summary" if operation == "summary" else "Compare documents").click().run()
    assert not app.exception and len(app.status) == 0
    assert events[-1][0] == ("error" if fail else "complete")
    if fail:
        assert any("Controlled provider failure" in error.value for error in app.error)
    app.run()
    assert mock.call_count == 1


def test_frozen_benchmark_reused_and_single_unjudged_query_has_no_charts(monkeypatch, fake_embedder):
    app = app_with_store(fake_embedder)
    app.session_state["eval_logs"] = [{"query": "question", "num_results": 1, "latency_sec": 0.1,
                                      "answer_latency_sec": 0.5, "citation_status": "matched", "avg_score": 0.025,
                                      "groundedness": None, "relevance": None}]
    judge = Mock(side_effect=AssertionError("Viewing evaluation must not invoke the provider"))
    import evaluation
    monkeypatch.setattr(evaluation, "evaluate_answer_quality", judge)
    app.run()
    assert not app.exception
    expected = {}
    artifact = Path(APP).parent / "docs" / "evaluation.md"
    for line in artifact.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        if len(cells) >= 4 and cells[0] in {'Dense / FAISS', 'BM25', 'Hybrid RRF'} and cells[0] not in expected:
            expected[cells[0]] = cells[1:4]
    table = app.table[0].value
    for method, values in expected.items():
        assert table.loc[method].tolist() == values
    assert any("36 questions · 6 fictional documents" in item.value for item in app.caption)
    assert any("has not produced scores" in item.value for item in app.caption)
    assert len(app.get("plotly_chart")) == 0
    judge.assert_not_called()


def test_search_runs_only_on_submit_and_keeps_evidence_metadata(monkeypatch, fake_embedder):
    retrieve = Mock(return_value=[])
    monkeypatch.setattr(rag_engine, "retrieve", retrieve)
    app = app_with_store(fake_embedder)
    app.text_input(key="search_box").set_value("battery").run()
    assert retrieve.call_count == 0
    button(app, "Search").click().run()
    assert not app.exception and retrieve.call_count == 1
    app.run()
    assert retrieve.call_count == 1
    assert any("Retrieval strategy: Hybrid RRF" in item.value for item in app.caption)


@pytest.mark.parametrize("fail", [False, True])
def test_demo_loading_precedes_indexing_and_clears(monkeypatch, fake_embedder, fail):
    import ingest
    events = watch_processing(monkeypatch)
    native_build = ingest.build_index_from_pdfs

    def build(*args, **kwargs):
        assert events[-1] == ("start", "Preparing demo documents…")
        if fail:
            raise RuntimeError("Controlled embedding failure")
        return native_build(*args, **kwargs)

    mock = Mock(side_effect=build)
    monkeypatch.setattr(ingest, "build_index_from_pdfs", mock)
    app = AppTest.from_file(APP, default_timeout=30).run()
    button(app, "Try Demo").click().run()
    assert not app.exception and len(app.status) == 0
    assert events[-1][0] == ("error" if fail else "complete")
    if fail:
        assert any("Could not load the demo" in error.value for error in app.error)
    app.run()
    assert mock.call_count == 1
